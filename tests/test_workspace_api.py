"""Workspace HTTP contract against isolated files; never launches reconstruction."""
import functools
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _serve


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with _serve.process_lock:
            _serve.active_process = None
            _serve.active_job_info = {"status": "idle", "scene": "", "step": "", "logs": []}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_serve.H, directory=self.tmp.name))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def file(self, name, content=b"data"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        return path

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(conn.close)
        actual = {"Origin": "http://127.0.0.1:" + str(self.server.server_port)}
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            actual["Content-Type"] = "application/json"
        actual.update(headers or {})
        conn.request(method, path, body, {k: v for k, v in actual.items() if v is not None})
        response = conn.getresponse()
        payload = response.read()
        if response.getheader("Content-Type", "").startswith("application/json") and payload:
            payload = json.loads(payload)
        return response.status, payload, dict(response.getheaders())

    def detail(self, scene="flight"):
        status, body, _ = self.request("GET", "/api/workspace/project?scene=" + quote(scene))
        self.assertEqual(status, 200, body)
        return body

    def model(self):
        self.file("work/flight/viewer_assets/scene.ply", b"ply\nreal fixture geometry")
        self.file("work/flight/pc/collision.collision.glb", b"proxy geometry")

    def post(self, route, body, expected=200, headers=None):
        status, result, _ = self.request("POST", "/api/workspace/" + route, body, headers)
        self.assertEqual(status, expected, result)
        return result

    def test_rebound_hostname_cannot_write_to_local_service(self):
        authority = "untrusted.invalid:" + str(self.server.server_port)
        self.post("project", {"scene": "blocked", "name": "Blocked"}, 403,
                  headers={"Origin": "http://" + authority, "Host": authority})
        self.assertFalse((self.root / "work/blocked").exists())

    def test_malformed_legacy_launch_does_not_reserve_worker(self):
        with patch.object(_serve, "spawn_pipeline_job") as spawn:
            status, _, _ = self.request("POST", "/api/run", {"scene": "flight", "extra_args": None})
        self.assertEqual(status, 400)
        spawn.assert_not_called()
        self.assertEqual(_serve.active_job_info["status"], "idle")

    def test_completed_run_supersedes_an_old_failed_scan(self):
        self.model()
        old = self.file("work/flight/logs/00-workspace_scan.log", "[exit -1 cancelled] 1.0s\n")
        os.utime(old, (100, 100))
        self.file("work/flight/logs/00-workspace_run.log", "[exit 0] 2.0s\n")
        self.assertEqual(self.detail()["status"], "ready")

    def test_legacy_video_is_explicit_in_validated_launch(self):
        self.file("videos/flight.mp4", b"source")
        self.file("videos/flight/flight.jsonl", b"poses")
        with patch.object(_serve, "spawn_pipeline_job") as spawn:
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "smoke"})
        args = spawn.call_args.args[3]
        self.assertIn("--video", args)
        self.assertIn(str(self.root / "videos/flight.mp4"), args)

    def test_disconnected_client_does_not_trigger_a_second_response(self):
        from unittest.mock import MagicMock
        import workspace_api
        handler = MagicMock()
        handler.directory = str(self.root)
        handler.path = "/api/workspace/projects"
        handler.command = "GET"
        handler._survey_json.side_effect = ConnectionAbortedError("client disconnected")
        workspace_api.handle(handler, _serve)
        self.assertEqual(handler._survey_json.call_count, 1)

    def test_validated_survey_accuracy_and_crs_surface_in_detail(self):
        self.model()
        self.file("work/flight/survey/preparation.json", "{}")
        surveyed = {"status": "evaluated", "crs": "EPSG:4326 -> local ENU at 28.6, 77.2",
                    "vertical_datum": "ellipsoidal", "position_reference": "camera_center",
                    "alignment": {"fit_rmse_m": 0.31},
                    "evaluation": {"criteria": [{"id": "accuracy", "status": "measured",
                        "metrics": {"rmse_3d_m": 0.42, "horizontal_rmse_m": 0.3, "vertical_rmse_m": 0.29, "count": 6}}]},
                    "artifacts": [], "blockers": []}
        with patch("survey_workflow.scene_status", return_value=surveyed):
            body = self.detail()
        self.assertEqual(body["georeference"]["status"], "georeferenced")
        self.assertIn("EPSG:4326", body["georeference"]["crs"])
        self.assertEqual(body["georeference"]["viewer_frame"], "local Y-up; the interactive splat is not itself georeferenced")
        self.assertEqual(body["accuracy"], {"status": "verified", "rmse_m": 0.42, "horizontal_rmse_m": 0.3,
                                            "vertical_rmse_m": 0.29, "checkpoints": 6,
                                            "basis": "independent surveyed checkpoints"})

    def test_invalid_or_unmeasured_survey_stays_local_and_unverified(self):
        self.model()
        self.file("work/flight/survey/preparation.json", "{}")
        for surveyed in ({"status": "invalid", "alignment": None, "evaluation": {"criteria": []}, "artifacts": [], "blockers": ["stale"]},
                         {"status": "prepared", "alignment": None,
                          "evaluation": {"criteria": [{"id": "accuracy", "status": "not_evaluated", "metrics": {}}]}, "artifacts": [], "blockers": []}):
            with patch("survey_workflow.scene_status", return_value=surveyed):
                body = self.detail()
            self.assertEqual(body["georeference"], {"status": "local", "crs": None})
            self.assertEqual(body["accuracy"], {"status": "unverified", "rmse_m": None})

    def test_fit_rmse_is_never_reported_as_accuracy(self):
        self.model()
        self.file("work/flight/survey/preparation.json", "{}")
        surveyed = {"status": "aligned", "crs": "EPSG:4326 -> local ENU", "alignment": {"fit_rmse_m": 0.05},
                    "evaluation": {"criteria": [{"id": "accuracy", "status": "not_evaluated"}]}, "artifacts": [], "blockers": []}
        with patch("survey_workflow.scene_status", return_value=surveyed):
            body = self.detail()
        self.assertEqual(body["accuracy"]["status"], "unverified")
        self.assertEqual(body["survey"]["fit_rmse_m"], 0.05)

    def test_listing_without_telemetry_uses_handler_directory(self):
        self.file("videos/raw/clip.mp4")
        self.file("work/empty/project.json", '{"name":"Empty","workflow":"heritage"}')
        status, body, _ = self.request("GET", "/api/workspace/projects")
        self.assertEqual(status, 200, body)
        projects = {p["id"]: p for p in body["projects"]}
        self.assertEqual(set(projects), {"raw", "empty"})
        self.assertEqual(projects["raw"]["status"], "uploaded")
        self.assertIsNone(projects["raw"]["thumbnail_url"])
        self.assertEqual(projects["raw"]["scale"]["status"], "relative")
        self.assertEqual(projects["empty"]["workflow"], "heritage")
        self.assertIn("auto", body["presets"])
        self.assertIn("high", body["qualities"])
        self.assertEqual(body["job"]["status"], "idle")
        self.assertFalse((self.root / "work/raw").exists())

    def test_metadata_preserves_unspecified_values_and_geometry(self):
        self.model()
        self.post("project", {"scene": "flight", "name": "West facade", "workflow": "inspection", "capture": "drone", "notes": "Review crack"})
        body = self.post("project", {"scene": "flight", "name": "East facade"})
        self.assertEqual((body["workflow"], body["capture"], body["notes"]), ("inspection", "drone", "Review crack"))
        self.assertEqual(body["name"], "East facade")
        self.assertEqual((self.root / "work/flight/viewer_assets/scene.ply").read_bytes(), b"ply\nreal fixture geometry")
        self.assertEqual(json.loads((self.root / "work/flight/project.json").read_text())["name"], "East facade")
        self.assertEqual(self.post("project", {"scene": "new", "name": "New"})["status"], "empty")

    def test_invalid_scene_and_missing_detail(self):
        for scene in ("../secret", "bad\\name", "a/b", "a" * 65, "", ".", ".."):
            status, body, _ = self.request("GET", "/api/workspace/project?scene=" + quote(scene))
            self.assertEqual(status, 400, body)
            self.assertIn("error", body)
        self.assertEqual(self.request("GET", "/api/workspace/project?scene=absent")[0], 404)
        self.post("project", {"scene": "new.dot"}, 400)
        self.file("work/old.dot/project.json", "{}")
        self.assertEqual(self.detail("old.dot")["id"], "old.dot")
        for payload in ({"scene": "flight", "workflow": "invented"}, {"scene": "flight", "capture": "satellite"}, {"scene": "flight", "name": ""}):
            self.post("project", payload, 400)

    def test_real_frames_timestamps_and_latest_step_attempt(self):
        self.model()
        self.file("work/flight/frames_train/clip/00000.jpg", b"jpeg")
        self.file("work/flight/viewer_assets/cameras.json", json.dumps([
            {"name": "clip/00000.jpg", "t_sec": 1.25, "pos": [1, 2, 3]},
            {"name": "missing.jpg", "t_sec": 2.0, "pos": [2, 3, 4]}]))
        old = self.file("work/flight/logs/01-export.log", "[exit 1] 1.0s\n")
        os.utime(old, (100, 100))
        self.file("work/flight/logs/08-export.log", "[ok 2] 2.0s\n")
        body = self.detail()
        self.assertEqual(body["status"], "ready")
        self.assertEqual(len(body["steps"]), 1)
        self.assertEqual(body["steps"][0]["status"], "recovered")
        self.assertEqual(body["frame_count"], 1)
        self.assertEqual(body["frames"][0]["t_sec"], 1.25)
        self.assertEqual(body["frames"][0]["pos"], [1, 2, 3])
        self.assertEqual(body["thumbnail_url"], "/runtime/work/flight/frames_train/clip/00000.jpg")
        self.assertEqual(body["viewer_url"], "/runtime/viewer/pc.html?asset=/runtime/work/flight/viewer_assets&embed=1")

    def test_source_frames_reference_their_actual_viewer_camera(self):
        self.file("work/flight/frames_train/a.jpg", b"first")
        self.file("work/flight/frames_train/b.jpg", b"second")
        self.file("work/flight/frames_train/c.jpg", b"unregistered")
        self.file("work/flight/viewer_assets/cameras.json", json.dumps([
            {"name": "b.jpg", "pos": [1, 2, 3]},
            {"name": "a.jpg", "pos": [4, 5, 6]}]))
        frames = self.detail()["frames"]
        self.assertEqual([(frame["name"], frame.get("camera_index")) for frame in frames],
                         [("a.jpg", 1), ("b.jpg", 0), ("c.jpg", None)])

    def test_frame_count_does_not_double_count_undistorted_copies(self):
        self.file("work/flight/frames_train/data/00000.jpg", b"train")
        self.file("work/flight/frames_full/data/00000.jpg", b"full")
        self.file("work/flight/frames_undist/1210x908__data__00000.jpg", b"undistorted")
        self.file("work/flight/keyframes.jsonl", '{"file":"data/00000.jpg","t_sec":0.2}\n')
        body = self.detail()
        self.assertEqual(body["frame_count"], 1)
        self.assertEqual(body["frames"][0]["t_sec"], 0.2)

    def test_scale_is_not_independent_accuracy_or_georeference(self):
        self.model()
        for source, expected in (("AR pose-prior metric path", "metric"), ("flight speed x clip duration", "estimated"), ("camera height 1.6 m above ground", "estimated")):
            self.file("work/flight/frame.json", json.dumps({"scale_source": source, "scale_m_per_unit": 2}))
            body = self.detail()
            self.assertEqual(body["scale"], {"status": expected, "source": source, "unit": "m"})
            self.assertEqual(body["accuracy"], {"status": "unverified", "rmse_m": None})
            self.assertEqual(body["georeference"], {"status": "local", "crs": None})

    def test_measurements_compute_viewer_geometry_and_persist(self):
        self.model()
        revision = self.detail()["model_revision"]
        cases = [("point", [[1, 2, 3]], None, "units"),
                 ("distance", [[0, 0, 0], [3, 4, 0], [3, 4, 2]], 7, "units"),
                 ("height", [[0, -2, 0], [100, 5, 100]], 7, "units"),
                 ("area", [[0, 9, 0], [4, 1, 0], [4, 3, 3], [0, 5, 3]], 12, "units²")]
        for kind, points, expected, unit in cases:
            body = self.post("measurements", {"scene": "flight", "kind": kind, "label": kind, "points": points, "model_revision": revision})
            measurement = body["measurements"][-1]
            self.assertEqual(measurement["value"], expected)
            self.assertEqual(measurement["unit"], unit)
            self.assertEqual(measurement["points"], points)
            self.assertFalse(measurement["stale"])
            self.assertEqual(measurement["geometry"], "viewer-pick")
        self.assertEqual(len(self.detail()["measurements"]), 4)
        self.assertTrue(any("proxy" in w and "accuracy" in w for w in body["warnings"]))
        deleted = self.post("measurements/delete", {"scene": "flight", "id": measurement["id"]})
        self.assertEqual(len(deleted["measurements"]), 3)

    def test_cloud_backed_measurement_uses_engine_with_uncertainty(self):
        import numpy as np
        self.model()
        g = np.linspace(-4, 4, 25)
        cloud = [[float(x), 0.0, float(z)] for x in g for z in g]
        self.file("work/flight/viewer_assets/sparse_points.json",
                  json.dumps({"count": len(cloud), "points": cloud}))
        revision = self.detail()["model_revision"]
        body = self.post("measurements", {"scene": "flight", "kind": "distance", "label": "Span",
                                          "points": [[-2, 0, 0], [2, 0, 0]], "model_revision": revision})
        m = body["measurements"][-1]
        self.assertTrue(m["engine"])
        self.assertTrue(m["valid"], m.get("reason"))
        self.assertIsNotNone(m["uncertainty"])
        self.assertGreaterEqual(m["uncertainty"]["m"], 0.0)
        self.assertAlmostEqual(m["value"], 4.0, delta=0.5)
        self.assertEqual(m["points"], [[-2, 0, 0], [2, 0, 0]])  # raw picks still stored

    def test_height_uses_engine_vertical_between_two_points(self):
        import numpy as np
        self.model()
        g = np.linspace(-4, 4, 25)
        cloud = [[float(x), 0.0, float(z)] for x in g for z in g]
        cloud += [[1.0, float(h), 1.0] for h in np.linspace(0, 3, 12)]
        self.file("work/flight/viewer_assets/sparse_points.json",
                  json.dumps({"count": len(cloud), "points": cloud}))
        revision = self.detail()["model_revision"]
        m = self.post("measurements", {"scene": "flight", "kind": "height", "label": "Post",
                                       "points": [[1, 0, 1], [1, 3, 1]], "model_revision": revision})["measurements"][-1]
        self.assertTrue(m["engine"])
        self.assertAlmostEqual(m["value"], 3.0, delta=0.4)

    def test_volume_kind_needs_cloud_and_integrates_above_ground(self):
        import numpy as np
        self.model()
        g = np.linspace(-1, 1, 15)
        cloud = [[float(x), 0.0, float(z)] for x in g for z in g] + \
                [[float(x), 1.0, float(z)] for x in g for z in g]
        square = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
        revision = self.detail()["model_revision"]
        # No cloud yet → volume is refused, not fabricated.
        self.post("measurements", {"scene": "flight", "kind": "volume", "label": "Pile",
                                   "points": square, "model_revision": revision}, 409)
        self.file("work/flight/viewer_assets/sparse_points.json",
                  json.dumps({"count": len(cloud), "points": cloud}))
        m = self.post("measurements", {"scene": "flight", "kind": "volume", "label": "Pile",
                                       "points": square, "model_revision": revision})["measurements"][-1]
        self.assertTrue(m["engine"])
        self.assertEqual(m["unit"], "m³")
        self.assertAlmostEqual(m["value"], 4.0, delta=0.8)

    def test_semantics_detail_carries_per_class_summary(self):
        import numpy as np
        self.model()
        # A 20x20 m vegetation patch. class_summary's default occupancy cell is
        # sized for real clouds, so on a large footprint it converges near the true
        # 400 m^2; on a tiny patch it would over-count. (Detailed arithmetic lives
        # in tests/test_workspace_measure.py.)
        g = np.linspace(0, 20, 50)
        coords = [[float(x), 0.0, float(z)] for x in g for z in g]
        sem = {"classes": ["ground", "road", "building", "vegetation", "obstacle"],
               "counts": {"vegetation": len(coords)}, "coords": coords,
               "rgb": [[62, 168, 84] for _ in coords]}
        self.file("work/flight/viewer_assets/semantics.json", json.dumps(sem))
        summary = self.detail()["semantics"]["summary"]
        self.assertIn("vegetation", summary)
        self.assertEqual(summary["vegetation"]["count"], len(coords))
        self.assertAlmostEqual(summary["vegetation"]["area_m2"], 400.0, delta=40.0)

    def test_revision_changes_for_proxy_or_frame_not_metadata(self):
        self.model()
        revision = self.detail()["model_revision"]
        request = {"scene": "flight", "kind": "distance", "label": "Span", "points": [[0, 0, 0], [1, 0, 0]], "model_revision": revision}
        self.post("measurements", request)
        self.post("project", {"scene": "flight", "name": "Rename"})
        self.assertEqual(self.detail()["model_revision"], revision)
        self.file("work/flight/frame.json", '{"scale_source":"camera height 2 m","scale_m_per_unit":2}')
        body = self.detail()
        self.assertNotEqual(body["model_revision"], revision)
        self.assertTrue(body["measurements"][0]["stale"])
        self.assertEqual(body["measurements"][0]["value"], 1)
        self.assertEqual(body["measurements"][0]["unit"], "units")
        self.post("measurements", request, 409)
        revision = body["model_revision"]
        self.file("work/flight/pc/collision.collision.glb", b"different proxy")
        self.assertNotEqual(self.detail()["model_revision"], revision)

    def place_grid(self):
        """A supported floor patch with a ceiling, for placement snap + fit."""
        import numpy as np
        nx = nz = 60
        cell = 0.1
        ground = np.zeros((nz, nx), np.float32)
        top = np.full((nz, nx), 2.4, np.float32)
        cover = np.zeros((nz, nx), np.uint8)
        cover[8:52, 8:52] = 1
        self.file("work/flight/viewer_assets/collision.json", json.dumps(
            {"nx": nx, "nz": nz, "cell": cell, "origin_xz": [0.0, 0.0],
             "character_height": 1.75, "scale_m_per_unit": 1.0, "object_colliders": 0}))
        self.file("work/flight/viewer_assets/ground.f32", ground.tobytes())
        self.file("work/flight/viewer_assets/heights.f32", top.tobytes())
        self.file("work/flight/viewer_assets/coverage.u8", cover.tobytes())

    def test_furniture_library_is_offered_in_detail(self):
        self.model()
        lib = self.detail()["furniture"]
        keys = {i["item"] for i in lib}
        self.assertIn("sofa", keys)
        self.assertTrue(all(len(i["size"]) == 3 for i in lib))

    def test_placement_snaps_to_floor_and_reports_fit(self):
        self.model()
        self.place_grid()
        revision = self.detail()["model_revision"]
        body = self.post("placements", {"scene": "flight", "item": "sofa",
                                        "point": [2.5, 0.0, 2.5], "yaw_deg": 0.0, "model_revision": revision})
        p = body["placements"][-1]
        self.assertEqual(p["item"], "sofa")
        self.assertTrue(p["fit"]["supported"], p["fit"])
        self.assertTrue(p["fit"]["valid"], p["fit"])
        self.assertAlmostEqual(p["center_y"], p["size"][1] / 2, delta=0.1)
        self.assertEqual(p["model_revision"], revision)
        self.assertFalse(p["stale"])

    def test_placement_update_moves_and_refits_without_multiplying(self):
        self.model()
        self.place_grid()
        revision = self.detail()["model_revision"]
        created = self.post("placements", {"scene": "flight", "item": "desk",
                                           "point": [2.0, 0.0, 2.0], "model_revision": revision})
        pid = created["placements"][-1]["id"]
        moved = self.post("placements/update", {"scene": "flight", "id": pid, "item": "desk",
                                                "point": [3.0, 0.0, 3.0], "yaw_deg": 45.0, "model_revision": revision})
        self.assertEqual(len(moved["placements"]), 1)
        self.assertEqual(moved["placements"][0]["id"], pid)
        self.assertEqual(moved["placements"][0]["center_xz"], [3.0, 3.0])
        self.assertEqual(moved["placements"][0]["yaw_deg"], 45.0)

    def test_placement_delete_removes_only_that_item(self):
        self.model()
        self.place_grid()
        revision = self.detail()["model_revision"]
        a = self.post("placements", {"scene": "flight", "item": "dining_chair",
                                     "point": [2.0, 0.0, 2.0], "model_revision": revision})["placements"][-1]
        self.post("placements", {"scene": "flight", "item": "stool", "point": [2.5, 0.0, 2.5], "model_revision": revision})
        gone = self.post("placements/delete", {"scene": "flight", "id": a["id"], "model_revision": revision})
        self.assertEqual(len(gone["placements"]), 1)
        self.assertNotIn(a["id"], {p["id"] for p in gone["placements"]})

    def test_placement_rejects_unknown_item_and_bad_point(self):
        self.model()
        self.place_grid()
        revision = self.detail()["model_revision"]
        self.post("placements", {"scene": "flight", "item": "spaceship", "point": [2, 0, 2], "model_revision": revision}, 400)
        self.post("placements", {"scene": "flight", "item": "sofa", "point": [float("nan"), 0, 2], "model_revision": revision}, 400)
        self.post("placements", {"scene": "flight", "item": "sofa", "point": [2, 0, 2], "scale": 999, "model_revision": revision}, 400)

    def test_placement_requires_viewable_and_matching_revision(self):
        self.file("work/flight/project.json", "{}")
        self.place_grid()
        # No scene.ply yet → not viewable → placement refused, not fabricated.
        self.post("placements", {"scene": "flight", "item": "sofa", "point": [2, 0, 2], "model_revision": "x"}, 409)
        self.model()
        revision = self.detail()["model_revision"]
        self.post("placements", {"scene": "flight", "item": "sofa", "point": [2, 0, 2], "model_revision": "stale"}, 409)
        self.post("placements", {"scene": "flight", "item": "sofa", "point": [2, 0, 2], "model_revision": revision})

    def test_placements_go_stale_when_the_model_changes(self):
        self.model()
        self.place_grid()
        revision = self.detail()["model_revision"]
        self.post("placements", {"scene": "flight", "item": "sofa", "point": [2.5, 0.0, 2.5], "model_revision": revision})
        self.file("work/flight/pc/collision.collision.glb", b"different proxy geometry")
        body = self.detail()
        self.assertTrue(body["placements"][0]["stale"])

    def test_invalid_measurement_rejected(self):
        self.model()
        revision = self.detail()["model_revision"]
        base = {"scene": "flight", "kind": "distance", "label": "Span", "points": [[0, 0, 0], [1, 0, 0]], "model_revision": revision}
        for change in ({"kind": "volume"}, {"label": ""}, {"label": "x" * 201}, {"points": [[float("nan"), 0, 0], [0, 0, 0]]}, {"points": [[float("inf"), 0, 0], [0, 0, 0]]}, {"points": [[1e100, 0, 0], [0, 0, 0]]}, {"points": [[0, 0, 0]] * 1001}, {"kind": "height", "points": [[0, 0, 0]] * 3}, {"kind": "area", "points": [[0, 0, 0]] * 2}):
            self.post("measurements", dict(base, **change), 400)
        self.file("work/empty/project.json", "{}")
        self.post("measurements", dict(base, scene="empty"), 409)

    def test_binary_range_suffix_head_and_download(self):
        self.file("videos/flight/a clip.mp4", b"0123456789\x00\r\n")
        url = "/api/workspace/file?path=" + quote("videos/flight/a clip.mp4")
        for value, code, expected, content_range in (("bytes=2-5", 206, b"2345", "bytes 2-5/13"), ("bytes=-3", 206, b"\x00\r\n", "bytes 10-12/13"), ("bytes=10-", 206, b"\x00\r\n", "bytes 10-12/13")):
            status, body, headers = self.request("GET", url, headers={"Range": value})
            self.assertEqual(status, code, body)
            self.assertEqual(body, expected)
            self.assertEqual(headers["Content-Range"], content_range)
            self.assertEqual(int(headers["Content-Length"]), len(expected))
        status, body, headers = self.request("HEAD", url)
        self.assertEqual((status, body, headers["Content-Length"]), (200, b"", "13"))
        for value in ("bytes=99-", "bytes=5-2", "bytes=0-1,3-4", "bytes=-0"):
            status, _, headers = self.request("GET", url, headers={"Range": value})
            self.assertEqual(status, 416)
            self.assertEqual(headers["Content-Range"], "bytes */13")
        status, body, headers = self.request("GET", url + "&download=1")
        self.assertEqual(body, b"0123456789\x00\r\n")
        self.assertIn("attachment", headers["Content-Disposition"])

    def test_file_allowlist_blocks_secrets_html_and_traversal(self):
        self.file(".env", b"SECRET")
        self.file("scripts/private.py", b"SECRET")
        self.file("videos/flight/upload.html", b"<script>SECRET</script>")
        self.file("work/flight/viewer_assets/private.py", b"SECRET")
        for path in ("../.env", ".env", "scripts/private.py", "videos/flight/upload.html", "work/flight/viewer_assets/private.py", "work/flight", "/.env", "C:/secret", "videos\\flight\\a.mp4", "videos/flight/../flight/a.mp4"):
            status, body, _ = self.request("GET", "/api/workspace/file?path=" + quote(path))
            self.assertIn(status, (400, 403, 404), body)
            self.assertNotIn("SECRET", str(body))
            self.assertNotIn(str(self.root), str(body))
        for path in ("viewer/pc.html", "viewer/pc.js", "viewer/workspace.js", "viewer/pc/ammo/ammo.wasm.wasm", "viewer/assets/models/cesium_man.glb"):
            self.file(path, b"allowed")
            self.assertEqual(self.request("GET", "/api/workspace/file?path=" + quote(path))[1], b"allowed")

    def test_symlink_escapes_are_not_read_or_written(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "clip.mp4"
            secret.write_bytes(b"SECRET")
            (self.root / "videos").mkdir()
            try:
                (self.root / "videos/flight").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Creating symlinks requires Windows developer mode")
            status, body, _ = self.request("GET", "/api/workspace/file?path=videos/flight/clip.mp4")
            self.assertIn(status, (400, 403, 404))
            self.assertNotIn("SECRET", str(body))
            self.post("project", {"scene": "flight"}, 400)

    def test_all_workspace_writes_require_matching_origin_and_loopback(self):
        for route in ("project", "measurements", "measurements/delete", "run", "cancel", "upload", "unknown"):
            for origin in (None, "https://untrusted.invalid"):
                self.post(route, {"scene": "flight"}, 403, {"Origin": origin})
        with patch.object(_serve, "LOOPBACK_PEERS", ()):
            self.post("project", {"scene": "flight"}, 403)
        self.assertFalse((self.root / "work").exists())

    def test_run_fields_validated_before_start(self):
        self.file("videos/flight/clip.mp4")
        base = {"scene": "flight", "preset": "auto", "quality": "high"}
        for change in ({"preset": "bad"}, {"quality": "bad"}, {"action": "shell"}, {"engine": "other"}, {"dense_profile": "other"}, {"extra_args": ["--only", "train"]}, {"anchor": {"kind": "speed", "value": -1}}, {"anchor": {"kind": "height", "value": float("nan")}}, {"anchor": {"kind": "other", "value": 1}}):
            self.post("run", dict(base, **change), 400)
        self.post("run", dict(base, scene="missing"), 400)
        self.post("run", dict(base, engine="survey"), 409)
        self.assertEqual(self.request("GET", "/api/status")[1]["status"], "idle")

    def test_empty_video_cannot_launch_a_reconstruction(self):
        self.file("videos/flight/clip.mp4", b"")
        with patch.object(_serve, "run_pipeline_thread", return_value=None):
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "high"}, 400)

    def test_atomic_reservation_serializes_legacy_and_workspace(self):
        self.file("videos/flight/clip.mp4")
        # Hold only worker execution, not request handling or the reservation.
        with patch.object(_serve, "run_pipeline_thread", return_value=None):
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "high", "action": "scan"})
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "high"}, 409)
            self.assertEqual(self.request("POST", "/api/run", {"scene": "flight"})[0], 409)
        self.assertEqual(self.detail()["job"]["status"], "running")
        self.post("cancel", {"scene": "other"}, 409)
        result = self.post("cancel", {"scene": "flight"})
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.request("GET", "/api/status")[1]["status"], "cancelled")

    def test_worker_persists_scan_logs_and_uses_typed_argv(self):
        self.file("videos/flight/clip.mp4")
        observed = []

        class Process:
            stdout = io.StringIO("[01/02] diagnostics: RUN\nfixture output\n")
            returncode = 0
            def wait(self, **kwargs):
                return 0
            def poll(self):
                return 0

        def launch(argv, **options):
            observed.append((argv, options))
            return Process()

        with patch.object(_serve.subprocess, "Popen", side_effect=launch), patch.object(_serve, "spawn_pipeline_job", side_effect=_serve.run_pipeline_thread):
            self.post("run", {"scene": "flight", "preset": "room", "quality": "smoke", "action": "scan", "anchor": {"kind": "height", "value": 1.7}})
        argv, options = observed[0]
        self.assertEqual(argv[1:], [str(self.root / "pipeline.py"), "scan", "flight", "--preset", "room", "--quality", "smoke", "--height-anchor", "1.7", "--video", str(self.root / "videos/flight/clip.mp4")])
        self.assertEqual(options["cwd"], str(self.root))
        self.assertFalse(options.get("shell", False))
        self.assertEqual(self.request("GET", "/api/status")[1]["status"], "completed")
        logs = list((self.root / "work/flight/logs").glob("*.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("fixture output", logs[0].read_text())
        self.assertIn("[exit 0]", logs[0].read_text())

    def test_prepared_survey_launch_uses_current_readiness_and_persists_logs(self):
        self.file("videos/flight/clip.mp4")
        metadata = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
                    "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
                    "single_pass": True, "video_duration_s": 600}
        telemetry = "t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n0,28,77,100,1,2\n1,28.0001,77,100,1,2\n2,28.0001,77.0001,101,1,2\n3,28,77.0001,102,1,2\n"
        self.assertEqual(self.request("POST", "/api/survey/inputs", {"scene": "flight", "metadata": metadata, "telemetry_csv": telemetry})[0], 200)
        self.assertEqual(self.request("POST", "/api/survey/prepare", {"scene": "flight"})[0], 200)
        observed = []

        class Process:
            stdout = io.StringIO("[survey] dense\nfixture survey output\n")
            returncode = 0
            def wait(self, **kwargs):
                return 0
            def poll(self):
                return 0

        def launch(argv, **kwargs):
            observed.append(argv)
            return Process()

        with patch.object(_serve.subprocess, "Popen", side_effect=launch), patch.object(_serve, "spawn_pipeline_job", side_effect=_serve.run_pipeline_thread):
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "high", "engine": "survey", "dense_profile": "budget"})
        self.assertEqual(observed[0][1:], [str(self.root / "survey.py"), "reconstruct", "flight", "--allow-gpu", "--dense-profile", "budget"])
        self.assertIn("fixture survey output", next((self.root / "work/flight/logs").glob("*.log")).read_text())
        self.file("videos/flight/telemetry.csv", telemetry + "4,28,77,103,1,2\n")
        self.post("run", {"scene": "flight", "preset": "auto", "quality": "high", "engine": "survey"}, 409)

    def test_cancel_terminates_only_its_tree_and_worker_keeps_cancelled(self):
        self.file("videos/flight/clip.mp4")
        started, released, finished = threading.Event(), threading.Event(), threading.Event()
        commands = []

        class Output:
            def __iter__(self):
                yield "[01/02] train: RUN\n"
                released.wait(5)
            def close(self):
                pass

        class Process:
            pid = 876543
            stdout = Output()
            returncode = None
            def wait(self, **kwargs):
                released.wait(5)
                return self.returncode
            def poll(self):
                return self.returncode

        process = Process()
        def launch(*args, **kwargs):
            started.set()
            return process
        def terminate(argv, **kwargs):
            commands.append(argv)
            process.returncode = -15
            released.set()
            return _serve.subprocess.CompletedProcess(argv, 0)
        original = _serve.run_pipeline_thread
        def worker(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                finished.set()

        self.addCleanup(released.set)
        with patch.object(_serve.subprocess, "Popen", side_effect=launch), patch.object(_serve.subprocess, "run", side_effect=terminate), patch.object(_serve, "run_pipeline_thread", side_effect=worker):
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "high"})
            self.assertTrue(started.wait(5))
            self.post("cancel", {"scene": "different"}, 409)
            self.assertIsNone(process.returncode)
            if os.name == "nt":
                self.post("cancel", {"scene": "flight"})
            else:
                with patch.object(_serve.os, "killpg", side_effect=lambda *args: terminate(["killpg", *args])):
                    self.post("cancel", {"scene": "flight"})
            self.assertTrue(finished.wait(5))
        if os.name == "nt":
            self.assertEqual(commands, [["taskkill", "/PID", "876543", "/T", "/F"]])
        self.assertEqual(self.request("GET", "/api/status")[1]["status"], "cancelled")
        self.assertIn("cancelled]", next((self.root / "work/flight/logs").glob("*.log")).read_text())

    def test_upload_limit_has_explicit_json_error(self):
        status, result, _ = self.request("POST", "/api/workspace/upload", b"", {"Content-Type": "multipart/form-data; boundary=abc", "Content-Length": str((2 << 30) + 1)})
        self.assertEqual(status, 413)
        self.assertIn("2 GiB", result["error"])
        self.assertFalse((self.root / "videos").exists())

    def test_large_integer_coordinates_return_json_not_a_dropped_connection(self):
        self.model()
        revision = self.detail()["model_revision"]
        self.post("measurements", {"scene": "flight", "kind": "point", "label": "Huge", "points": [[10 ** 500, 0, 0]], "model_revision": revision}, 400)

    def test_multipart_binary_across_many_read_chunks(self):
        data = bytes(range(256)) * 3000 + b"\r\n\x00\xff"
        body, headers = self.multipart([("large.mp4", data)])
        self.post("upload", body, headers=headers)
        self.assertEqual((self.root / "videos/flight/large.mp4").read_bytes(), data)
        url = "/api/workspace/file?path=videos/flight/large.mp4"
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Media must stream")):
            status, payload, _ = self.request("GET", url)
        self.assertEqual(status, 200)
        self.assertEqual(payload, data)

    def test_scene_case_alias_cannot_modify_running_inputs(self):
        self.file("videos/flight/clip.mp4")
        with _serve.process_lock:
            _serve.active_job_info = {"status": "running", "scene": "flight", "step": "train", "logs": []}
        body, headers = self.multipart([("new.mp4", b"video")], scene="FLIGHT")
        self.post("upload", body, 409, headers)
        self.post("project", {"scene": "FLIGHT", "name": "Rename"}, 409)
        self.assertFalse((self.root / "videos/flight/new.mp4").exists())

    def test_legacy_reservation_prevents_workspace_job_and_scene_writes(self):
        self.file("videos/flight/clip.mp4")
        with patch.object(_serve, "run_pipeline_thread", return_value=None):
            self.assertEqual(self.request("POST", "/api/run", {"scene": "flight"})[0], 200)
            self.post("run", {"scene": "flight", "preset": "auto", "quality": "high"}, 409)
            self.post("project", {"scene": "flight", "name": "Busy"}, 409)
        self.post("cancel", {"scene": "flight"})

    def multipart(self, files, scene="flight", boundary="workspace-boundary"):
        chunks = [b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="scene"\r\n\r\n' + scene.encode() + b"\r\n"]
        for name, data in files:
            chunks.append(b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="files"; filename="' + name.encode() + b'"\r\nContent-Type: application/octet-stream\r\n\r\n' + data + b"\r\n")
        chunks.append(b"--" + boundary.encode() + b"--\r\n")
        return b"".join(chunks), {"Content-Type": "multipart/form-data; boundary=" + boundary}

    def test_upload_preserves_binary_and_refuses_collisions(self):
        data = b"\x00\xffbinary\r\n--workspace-boundaryNOT-A-DELIMITER\x00\r\n\r\n"
        body, headers = self.multipart([("capture.mp4", data), ("poses.jsonl", b'{"t":1}\r\n')])
        result = self.post("upload", body, headers=headers)
        self.assertEqual(result["scene"], "flight")
        self.assertEqual((self.root / "videos/flight/capture.mp4").read_bytes(), data)
        self.assertEqual((self.root / "videos/flight/poses.jsonl").read_bytes(), b'{"t":1}\r\n')
        self.post("upload", body, 409, headers)
        self.assertEqual((self.root / "videos/flight/capture.mp4").read_bytes(), data)
        self.assertEqual(self.detail()["video_count"], 1)

    def test_upload_validates_all_files_before_persistence(self):
        for bad in ("../evil.mp4", "evil\\clip.mp4", "evil.html", "bad.py"):
            body, headers = self.multipart([("good.mp4", b"valid"), (bad, b"bad")])
            self.post("upload", body, 400, headers)
            self.assertFalse((self.root / "videos/flight/good.mp4").exists())
        body, headers = self.multipart([("same.mp4", b"one"), ("same.mp4", b"two")])
        self.post("upload", body, 409, headers)
        with _serve.process_lock:
            _serve.active_job_info = {"status": "running", "scene": "flight", "step": "scan", "logs": []}
        body, headers = self.multipart([("good.mp4", b"valid")])
        self.post("upload", body, 409, headers)


if __name__ == "__main__":
    unittest.main()
