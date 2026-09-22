import contextlib
import functools
import http.client
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _serve
import survey_workflow

METADATA = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
            "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
            "single_pass": True, "video_duration_s": 600}
TELEMETRY = ("t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
             "0,28,77,100,1,2\n1,28.0001,77,100,1,2\n"
             "2,28.0001,77.0001,101,1,2\n3,28,77.0001,102,1,2\n")
# A drive-letter path in any response body means the server leaked its workspace.
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")


class SurveyApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_serve.H, directory=self.tmp.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None, server=None):
        server = server or self.server
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        self.addCleanup(connection.close)
        data = json.dumps(body).encode() if body is not None else None
        actual_headers = {"Content-Type": "application/json"}
        if method == "POST" and path.startswith("/api/survey/"):
            # The write guard demands a browser-shaped request: Origin present and
            # equal to Host. Every survey POST here starts from that, and the
            # guard tests below override or drop the header on purpose.
            actual_headers["Origin"] = "http://127.0.0.1:" + str(server.server_port)
        actual_headers.update(headers or {})
        actual_headers = {name: value for name, value in actual_headers.items() if value is not None}
        connection.request(method, path, data, actual_headers)
        result = connection.getresponse()
        return result.status, json.loads(result.read())

    def lan_peer_server(self, peer="192.0.2.50"):
        """The same handler, but every connection claims a non-loopback peer."""
        class LanPeer(_serve.H):
            def setup(self):
                super().setup()
                self.client_address = (peer, 54321)
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(LanPeer, directory=self.tmp.name))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def inputs_payload(self):
        return {"scene": "flight", "metadata": METADATA, "telemetry_csv": TELEMETRY}

    def assert_no_workspace_leak(self, body):
        text = json.dumps(body)
        self.assertNotIn(str(self.root), text)
        self.assertIsNone(ABSOLUTE_PATH.search(text), text)

    def test_missing_scene_returns_structured_error(self):
        status, body = self.request("GET", "/api/survey")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_unknown_scene_is_safe_unprepared_status(self):
        status, body = self.request("GET", "/api/survey?scene=flight")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "not_prepared")
        self.assertFalse(body["gpu_execution_enabled"])

    def test_survey_api_has_no_gpu_execution_route(self):
        status, body = self.request("POST", "/api/survey/reconstruct", {"scene": "flight", "allow_gpu": True})
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_cross_origin_write_is_rejected(self):
        status, body = self.request("POST", "/api/survey/prepare", {"scene": "flight"}, {"Origin": "https://untrusted.invalid"})
        self.assertEqual(status, 403)
        self.assertIn("error", body)

    def test_survey_write_without_origin_is_rejected_before_any_write(self):
        # A script, not a browser, sends no Origin: that must be a refusal, not a
        # bypass, and it must be refused before the workflow touches the disk.
        status, body = self.request("POST", "/api/survey/inputs", self.inputs_payload(), {"Origin": None})
        self.assertEqual(status, 403)
        self.assertIn("error", body)
        self.assertFalse((self.root / "videos").exists())
        self.assertFalse((self.root / "work").exists())

    def test_survey_write_guard_runs_before_route_dispatch(self):
        status, body = self.request("POST", "/api/survey/reconstruct", {"scene": "flight", "allow_gpu": True},
                                   {"Origin": None})
        self.assertEqual(status, 403)
        self.assertIn("error", body)

    def test_survey_write_from_a_non_loopback_peer_is_rejected(self):
        # Both headers agree, so only the loopback rule can refuse this one. Host
        # is caller-supplied, which is why peer address is checked separately.
        server = self.lan_peer_server()
        host = "localhost:" + str(server.server_port)
        status, body = self.request("POST", "/api/survey/prepare", {"scene": "flight"},
                                   {"Origin": "http://" + host, "Host": host}, server=server)
        self.assertEqual(status, 403)
        self.assertIn("error", body)
        self.assertFalse((self.root / "work").exists())

    def test_survey_status_still_reads_from_a_non_loopback_peer(self):
        # Deliberate: the dashboard iframe and the LAN viewer read status over HTTP.
        server = self.lan_peer_server()
        status, body = self.request("GET", "/api/survey?scene=flight", server=server)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "not_prepared")

    def test_invalid_payload_type_is_rejected(self):
        status, body = self.request("POST", "/api/survey/prepare", ["flight"])
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_error_body_never_leaks_the_workspace_path(self):
        leaked = str(self.root / "work" / "flight" / "survey" / "points3D.txt")
        with patch.object(survey_workflow, "prepare_scene",
                          side_effect=FileNotFoundError(2, "No such file or directory: " + repr(leaked))):
            status, body = self.request("POST", "/api/survey/prepare", {"scene": "flight"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error_class"], "FileNotFoundError")
        self.assertTrue(body["error"].strip())
        self.assert_no_workspace_leak(body)

    def test_error_detail_is_logged_instead_of_returned(self):
        detail = "Supply a JSON request below 2 MiB."
        with patch.object(survey_workflow, "prepare_scene", side_effect=ValueError(detail)), \
                contextlib.redirect_stderr(io.StringIO()) as log:
            status, body = self.request("POST", "/api/survey/prepare", {"scene": "flight"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error_class"], "ValueError")
        self.assertIn(detail, json.dumps(body))
        self.assertIn(detail, log.getvalue())

    def test_value_error_carrying_an_absolute_path_is_sanitized(self):
        detail = "cannot read " + str(self.root / "work" / "flight" / "georeference.json")
        with patch.object(survey_workflow, "prepare_scene", side_effect=ValueError(detail)), \
                contextlib.redirect_stderr(io.StringIO()) as log:
            status, body = self.request("POST", "/api/survey/prepare", {"scene": "flight"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "Survey request failed; the server log holds the detail.")
        self.assert_no_workspace_leak(body)
        self.assertIn(detail, log.getvalue())

    def test_align_before_prepare_names_the_missing_step(self):
        status, body = self.request("POST", "/api/survey/align", {"scene": "flight"})
        self.assertEqual(status, 400)
        self.assertIn("prepare", body["error"].lower())
        self.assertNotEqual(body["error"], "Survey request failed; the server log holds the detail.")
        self.assert_no_workspace_leak(body)

    def test_conflict_reports_only_a_class_and_a_fixed_message(self):
        status, body = self.request("POST", "/api/survey/inputs", self.inputs_payload())
        self.assertEqual(status, 200, body)
        status, body = self.request("POST", "/api/survey/inputs", self.inputs_payload())
        self.assertEqual(status, 409)
        self.assertEqual(body["error_class"], "FileExistsError")
        self.assert_no_workspace_leak(body)

    def test_path_traversal_is_rejected_not_sanitized(self):
        status, body = self.request("GET", "/api/survey?scene=../flight")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_cpu_prepare_evaluate_and_stale_input_flow(self):
        root = Path(self.tmp.name)
        source = root / "videos/flight"
        source.mkdir(parents=True)
        (source / "capture.mp4").write_bytes(b"not-decoded-during-preparation")
        metadata = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
                    "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
                    "single_pass": True, "video_duration_s": 600}
        telemetry = ("t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
                     "0,28,77,100,1,2\n1,28.0001,77,100,1,2\n"
                     "2,28.0001,77.0001,101,1,2\n3,28,77.0001,102,1,2\n")
        payload = {"scene": "flight", "metadata": metadata, "telemetry_csv": telemetry}
        status, _ = self.request("POST", "/api/survey/inputs", payload)
        self.assertEqual(status, 200)
        status, _ = self.request("POST", "/api/survey/inputs", payload)
        self.assertEqual(status, 409)
        status, result = self.request("POST", "/api/survey/prepare", {"scene": "flight"})
        self.assertEqual(status, 200, result)
        self.assertEqual(result["status"], "prepared")
        status, result = self.request("POST", "/api/survey/evaluate", {"scene": "flight"})
        self.assertEqual(status, 200, result)
        self.assertTrue(all(c["status"] == "not_evaluated" for c in result["evaluation"]["criteria"]))
        (source / "telemetry.csv").write_text(telemetry + "4,28,77,103,1,2\n")
        status, result = self.request("GET", "/api/survey?scene=flight")
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse((root / "work/flight/splat.ply").exists())

    def test_non_json_content_type_is_rejected(self):
        status, result = self.request("POST", "/api/survey/prepare", {"scene": "flight"}, {"Content-Type": "text/plain"})
        self.assertEqual(status, 400)
        self.assertEqual(result["error_class"], "ValueError")
        self.assertTrue(result["error"].strip())

    def test_scene_listing_includes_raw_videos_and_existing_work(self):
        root = Path(self.tmp.name)
        (root / "videos/new_flight").mkdir(parents=True)
        (root / "videos/new_flight/clip.mp4").write_bytes(b"video")
        (root / "work/old_scene").mkdir(parents=True)
        scenes = {s["name"]: s for s in _serve.scan_scenes(root)}
        self.assertTrue(scenes["new_flight"]["has_video"])
        self.assertFalse(scenes["new_flight"]["has_work"])
        self.assertTrue(scenes["old_scene"]["has_work"])


if __name__ == "__main__":
    unittest.main()
