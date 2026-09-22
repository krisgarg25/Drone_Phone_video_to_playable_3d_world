import functools
import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _serve


class SurveyApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_serve.H, directory=self.tmp.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(connection.close)
        data = json.dumps(body).encode() if body is not None else None
        actual_headers = {"Content-Type": "application/json"}
        actual_headers.update(headers or {})
        connection.request(method, path, data, actual_headers)
        result = connection.getresponse()
        return result.status, json.loads(result.read())

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

    def test_invalid_payload_type_is_rejected(self):
        status, body = self.request("POST", "/api/survey/prepare", ["flight"])
        self.assertEqual(status, 400)
        self.assertIn("error", body)

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
        self.assertIn("application/json", result["error"])

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
