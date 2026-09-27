"""CPU-only tests for the proposal layer (Phase 2) and the scene frame registry.

A synthetic scene is written the way the pipeline writes one - collision.json +
ground.f32 + coverage.u8 + semantics.json + frame.json - so the planner reads the same
files a real scene gives it. Ground is a gentle slope with a 2 m hill; an existing
10 x 10 x 9 m building and a tree cluster sit on it.
"""
import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CELL, N, ORIGIN = 0.5, 200, -50.0


def ground_y(x, z):
    return 0.02 * x + 2.0 * np.exp(-((x - 0.0) ** 2 + (z - 20.0) ** 2) / (2 * 4.0 ** 2))


def make_scene(root, *, georeferenced=False):
    work = Path(root) / "work" / "town"
    va = work / "viewer_assets"
    va.mkdir(parents=True)
    centres = ORIGIN + (np.arange(N) + 0.5) * CELL
    xx, zz = np.meshgrid(centres, centres)            # rows = z, cols = x
    ground = ground_y(xx, zz).astype("<f4")
    (va / "collision.json").write_text(json.dumps({"nx": N, "nz": N, "cell": CELL,
                                                   "origin_xz": [ORIGIN, ORIGIN]}))
    ground.tofile(va / "ground.f32")
    ground.tofile(va / "heights.f32")
    np.ones((N, N), dtype=np.uint8).tofile(va / "coverage.u8")
    import label_semantics
    coords, rgb = [], []
    for x in np.arange(10.25, 20, 0.5):               # existing building roof, 9 m tall
        for z in np.arange(-4.75, 5, 0.5):
            coords.append([x, float(ground_y(15, 0)) + 9.0, z])
            rgb.append(list(label_semantics.CLASS_RGB["building"]))
    rng = np.random.default_rng(0)
    for _ in range(400):                              # a tree at (-20, 0)
        r, a = 3 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        x, z = -20 + r * math.cos(a), r * math.sin(a)
        coords.append([x, float(ground_y(x, z)) + rng.uniform(4, 8), z])
        rgb.append(list(label_semantics.CLASS_RGB["vegetation"]))
    (va / "semantics.json").write_text(json.dumps({"coords": coords, "rgb": rgb}))
    frame = {"rotation_rowmajor": np.eye(3).tolist(), "scale_m_per_unit": 1.0,
             "scale_source": "AR pose-prior metric path"}
    if georeferenced:
        # colmap == viewer here; ENU = R_y->up applied to viewer (x east, -z north, y up).
        rot = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        frame.update(scale_source="GPS telemetry similarity fit (ruler D)", scale_anchor_gps={
            "scale": 1.0, "alignment": {
                "schema_version": 1, "status": "aligned", "method": "test", "scale": 1.0,
                "rotation": rot, "translation": [0.0, 0.0, 0.0], "fit_rmse_m": 0.5,
                "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                                     "altitude_datum": "ellipsoidal",
                                     "origin": {"latitude_deg": 28.6, "longitude_deg": 77.2,
                                                "altitude_m": 210.0}}}})
    (work / "frame.json").write_text(json.dumps(frame))
    return work


def square(cx, cz, half):
    return [[cx - half, cz - half], [cx + half, cz - half], [cx + half, cz + half], [cx - half, cz + half]]


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = make_scene(self.tmp.name)
        self.wp = importlib.import_module("workspace_proposals")
        self.ground = self.wp.Ground(self.work)
        self.existing = self.wp.existing_inventory(self.work, self.ground)
        self.p = self.wp.create_proposal(self.work, "Scheme A")

    def add(self, feature):
        self.p = self.wp.upsert_feature(self.work, self.p["id"], feature, self.p["revision"])
        return self.p["features"][-1]["id"]

    def evaluate(self):
        return self.wp.evaluate(self.p, self.ground, self.existing)

    def test_existing_inventory_finds_the_building_and_the_tree(self):
        buildings, trees = self.existing["buildings"], self.existing["trees"]
        self.assertEqual(len(buildings), 1)
        b = buildings[0]
        self.assertAlmostEqual(b["height_m"], 9.0, delta=0.2)
        self.assertAlmostEqual(sorted(b["size"])[0], 9.5, delta=1.0)
        np.testing.assert_allclose(b["centre"], [15.0, 0.0], atol=0.6)
        self.assertEqual(b["floors_estimate"], 3)
        self.assertEqual(len(trees), 1)

    def test_draped_road_follows_the_ground(self):
        fid = self.add({"type": "road", "params": {"centerline": [[-40, 20], [40, 20]], "width_m": 7,
                                                   "footpath_m": 1.5}})
        ev = self.evaluate()
        m = ev["features"][fid]["metrics"]
        self.assertAlmostEqual(m["length_m"], 80.0, delta=0.1)
        self.assertAlmostEqual(m["width_total_m"], 10.0, delta=0.01)
        self.assertEqual((m["cut_m3"], m["fill_m3"]), (0.0, 0.0))
        carriageway = next(x for x in ev["features"][fid]["meshes"] if x["part"] == "carriageway")
        pos = np.asarray(carriageway["positions"]).reshape(-1, 3)
        np.testing.assert_allclose(pos[:, 1], ground_y(pos[:, 0], pos[:, 2]) + 0.06, atol=0.05)
        self.assertEqual(len([x for x in ev["features"][fid]["meshes"] if x["part"] == "footpath"]), 2)

    def test_unmeasured_ground_is_bridged_not_draped(self):
        # Knock the coverage out under the middle of the road: a cliff in the floor grid there.
        cov = np.ones((N, N), dtype=np.uint8)
        floor = np.fromfile(self.work / "viewer_assets/ground.f32", dtype="<f4").reshape(N, N)
        cols = (ORIGIN + (np.arange(N) + 0.5) * CELL)
        mid = (np.abs(cols) < 10)
        cov[:, mid] = 0
        floor[:, mid] = -40.0
        cov.tofile(self.work / "viewer_assets/coverage.u8")
        floor.astype("<f4").tofile(self.work / "viewer_assets/ground.f32")
        ground = self.wp.Ground(self.work)
        fid = self.add({"type": "road", "params": {"centerline": [[-40, -30], [40, -30]]}})
        m = self.wp.evaluate(self.p, ground, self.existing)["features"][fid]["metrics"]
        self.assertLess(m["max_grade_pct"], 5.0)                 # no plunge into the fake cliff
        self.assertAlmostEqual(m["profile_bridged_m"], 20.0, delta=1.5)
        self.assertLess(m["ground_supported_fraction"], 0.8)

    def test_graded_road_limits_grade_and_moves_earth(self):
        fid = self.add({"type": "road", "params": {"centerline": [[-40, 20], [40, 20]], "mode": "graded",
                                                   "max_grade_pct": 3.0}})
        m = self.evaluate()["features"][fid]["metrics"]
        self.assertLessEqual(m["max_grade_pct"], 3.0 + 1e-6)
        self.assertGreater(m["cut_m3"], 10.0)            # it cuts through the hill
        # Draped over the hill the same line is steeper than 3%.
        draped = self.add({"type": "road", "params": {"centerline": [[-40, 20], [40, 20]]}})
        self.assertGreater(self.evaluate()["features"][draped]["metrics"]["max_grade_pct"], 3.0)

    def test_building_metrics_and_pitched_roof(self):
        fid = self.add({"type": "building", "params": {"footprint": square(-10, -20, 5), "floors": 4,
                                                       "floor_height_m": 3.0, "roof": "gable",
                                                       "roof_pitch_deg": 30}})
        f = self.evaluate()["features"][fid]
        m = f["metrics"]
        self.assertAlmostEqual(m["footprint_area_m2"], 100.0, places=3)
        self.assertAlmostEqual(m["gfa_m2"], 400.0, places=3)
        self.assertAlmostEqual(m["eave_height_m"], 12.0, places=3)
        self.assertAlmostEqual(m["height_m"], 12.0 + 5 * math.tan(math.radians(30)), places=2)
        roof = next(x for x in f["meshes"] if x["part"] == "roof")
        self.assertEqual(len(roof["indices"]) // 3, 6)
        self.assertAlmostEqual(m["base_y"], float(ground_y(-15, -20)), delta=0.05)  # lowest corner

    def test_bad_geometry_is_refused(self):
        bow = [[0, 0], [10, 10], [10, 0], [0, 10]]
        with self.assertRaises(self.wp.ProposalError):
            self.add({"type": "building", "params": {"footprint": bow}})
        with self.assertRaises(self.wp.ProposalError):
            self.add({"type": "road", "params": {"centerline": [[0, 0]]}})
        with self.assertRaises(self.wp.ProposalError):
            self.add({"type": "object", "params": {"item": "spaceship", "position": [0, 0]}})

    def test_zone_rules_flag_and_clear_violations(self):
        self.add({"type": "zone", "params": {"polygon": square(-10, -20, 10), "rules": {
            "max_height_m": 15, "max_fsi": 1.5, "setback_m": 3}}})
        bid = self.add({"type": "building", "params": {"footprint": square(-10, -20, 8.5), "floors": 6}})
        ev = self.evaluate()
        rules = {v["rule"] for v in ev["violations"]}
        self.assertEqual(rules, {"max_height_m", "max_fsi", "setback_m"})
        self.assertTrue(all(mesh.get("violation") for mesh in ev["features"][bid]["meshes"]))
        # 12 x 12 at 3 floors: 432/400 = 1.08 FSI, 9.6 m tall, 4 m setback -> compliant.
        feature = next(f for f in self.p["features"] if f["id"] == bid)
        self.p = self.wp.upsert_feature(self.work, self.p["id"], {
            **feature, "params": {**feature["params"], "footprint": square(-10, -20, 6), "floors": 3}},
            self.p["revision"])
        self.assertEqual(self.evaluate()["violations"], [])

    def test_demolish_and_replace_changes_the_before_after_table(self):
        self.add({"type": "clip", "params": {"polygon": square(15, 0, 7)}})
        self.add({"type": "building", "params": {"footprint": square(15, 0, 5), "floors": 10}})
        ev = self.evaluate()
        self.assertEqual(ev["demolished"], [self.existing["buildings"][0]["id"]])
        table = {row["metric"]: row for row in ev["metrics"]}
        self.assertEqual(table["Buildings"]["existing"], 1)
        self.assertEqual(table["Buildings"]["proposal"], 1)
        self.assertAlmostEqual(table["Tallest building"]["proposal"], 32.0, delta=0.01)
        self.assertGreater(table["Residents (estimate)"]["proposal"], 0)

    def test_road_impacts_hit_the_building_and_the_tree(self):
        fid = self.add({"type": "road", "params": {"centerline": [[-45, 0], [45, 0]], "width_m": 8}})
        ev = self.evaluate()
        self.assertEqual(ev["impacts"][fid]["existing_buildings_hit"], [self.existing["buildings"][0]["id"]])
        self.assertEqual(len(ev["impacts"][fid]["trees_removed"]), 1)
        table = {row["metric"]: row for row in ev["metrics"]}
        self.assertEqual(table["Trees"]["change"], -1)

    def test_revision_conflict_and_undo_by_replacement(self):
        fid = self.add({"type": "object", "params": {"item": "bench", "position": [0, 0]}})
        stale = self.p["revision"] - 1
        with self.assertRaises(self.wp.ProposalError) as ctx:
            self.wp.upsert_feature(self.work, self.p["id"], {"type": "object", "params": {
                "item": "bench", "position": [1, 1]}}, stale)
        self.assertEqual(ctx.exception.status, 409)
        before = [f for f in self.p["features"] if f["id"] != fid]
        self.p = self.wp.replace_features(self.work, self.p["id"], before, self.p["revision"])
        self.assertEqual(self.p["features"], [])
        with self.assertRaises(self.wp.ProposalError):
            self.wp.replace_features(self.work, self.p["id"], [{"id": "../x", "type": "object",
                                                                "params": {"item": "bench", "position": [0, 0]}}],
                                     self.p["revision"])

    def test_array_places_objects_along_a_line(self):
        self.p = self.wp.array_objects(self.work, self.p["id"], "street_light", [[-30, 10], [30, 10]],
                                       30.0, self.p["revision"], offset_m=5)
        lights = [f for f in self.p["features"] if f["params"].get("item") == "street_light"]
        self.assertEqual(len(lights), 3)
        np.testing.assert_allclose([f["params"]["position"][1] for f in lights], 15.0, atol=0.01)

    def test_proposals_list_duplicate_rename_delete(self):
        self.add({"type": "object", "params": {"item": "bench", "position": [0, 0]}})
        copy = self.wp.create_proposal(self.work, "Scheme B", source=self.p["id"])
        self.assertEqual(len(copy["features"]), 1)
        self.wp.rename_proposal(self.work, copy["id"], "Scheme B2")
        index = self.wp.list_proposals(self.work)
        self.assertEqual([p["name"] for p in index["proposals"]], ["Scheme A", "Scheme B2"])
        index = self.wp.delete_proposal(self.work, copy["id"])
        self.assertEqual(len(index["proposals"]), 1)
        with self.assertRaises(self.wp.ProposalError):
            self.wp.load_proposal(self.work, "../../etc")

    def test_local_export_says_it_is_local(self):
        self.add({"type": "building", "params": {"footprint": square(-10, -20, 5)}})
        out = self.wp.export_geojson(self.work, self.p, self.evaluate())
        self.assertIn("LOCAL", out["crs_note"])
        self.assertEqual(out["features"][0]["properties"]["status"], "proposed, not measured")


class FrameRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sf = importlib.import_module("scene_frames")

    def test_local_scene_refuses_earth_coordinates(self):
        work = make_scene(self.tmp.name)
        reg = self.sf.load(work)
        self.assertEqual(reg["status"], "local_metric")
        with self.assertRaises(ValueError):
            self.sf.viewer_to_enu([0, 0, 0], reg)
        self.assertEqual(self.sf.describe([1, 2, 3], reg)["status"], "local_metric")

    def test_georeferenced_chain_round_trips_and_describes(self):
        work = make_scene(self.tmp.name, georeferenced=True)
        reg = self.sf.load(work)
        self.assertEqual(reg["status"], "georeferenced")
        pts = np.array([[10.0, 2.0, -30.0], [-5.0, 0.0, 12.0]])
        enu = self.sf.viewer_to_enu(pts, reg)
        np.testing.assert_allclose(enu[0], [10.0, 30.0, 2.0], atol=1e-9)   # -z is north
        np.testing.assert_allclose(self.sf.enu_to_viewer(enu, reg), pts, atol=1e-9)
        out = self.sf.describe([0, 0, -100], reg, sigma_h_m=1.0, sigma_v_m=2.0)
        self.assertAlmostEqual(out["lat_deg"], 28.6 + 100 / 110_850, delta=2e-5)
        self.assertTrue(out["mgrs"].startswith("43R"))
        self.assertIn("ce90_m", out)

    def test_georeferenced_export_is_wgs84(self):
        work = make_scene(self.tmp.name, georeferenced=True)
        wp = importlib.import_module("workspace_proposals")
        p = wp.create_proposal(work, "Geo")
        p = wp.upsert_feature(work, p["id"], {"type": "building", "params": {"footprint": square(0, 0, 5)}},
                              p["revision"])
        ground = wp.Ground(work)
        ev = wp.evaluate(p, ground, wp.existing_inventory(work, ground))
        out = wp.export_geojson(work, p, ev, self.sf.load(work))
        lon, lat = out["features"][0]["geometry"]["coordinates"][0][0]
        self.assertAlmostEqual(lat, 28.6, delta=0.001)
        self.assertAlmostEqual(lon, 77.2, delta=0.001)



class PlanHttpTests(unittest.TestCase):
    """The /api/workspace/plan/* contract through the real server handler."""

    def setUp(self):
        import functools
        import threading
        from http.server import ThreadingHTTPServer
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import _serve
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make_scene(self.tmp.name, georeferenced=True)
        with _serve.process_lock:
            _serve.active_process = None
            _serve.active_job_info = {"status": "idle", "scene": "", "step": "", "logs": []}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                          functools.partial(_serve.H, directory=self.tmp.name))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, origin=True):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        self.addCleanup(conn.close)
        headers = {"Origin": f"http://127.0.0.1:{self.server.server_port}"} if origin else {}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, payload, headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"null")

    def test_create_edit_evaluate_export_and_coordinates(self):
        status, created = self.request("POST", "/api/workspace/plan/proposals",
                                       {"scene": "town", "name": "Scheme A"})
        self.assertEqual(status, 200, created)
        pid, rev = created["proposal"]["id"], created["proposal"]["revision"]
        status, state = self.request("POST", "/api/workspace/plan/feature", {
            "scene": "town", "id": pid, "revision": rev,
            "feature": {"type": "building", "params": {"footprint": square(-10, -20, 5), "floors": 4}}})
        self.assertEqual(status, 200, state)
        fid = state["proposal"]["features"][0]["id"]
        self.assertEqual(state["evaluation"]["features"][fid]["metrics"]["gfa_m2"], 400.0)
        self.assertEqual(state["frame"]["status"], "georeferenced")
        self.assertEqual(len(state["existing"]["buildings"]), 1)
        status, conflict = self.request("POST", "/api/workspace/plan/feature", {
            "scene": "town", "id": pid, "revision": rev,
            "feature": {"type": "object", "params": {"item": "bench", "position": [0, 0]}}})
        self.assertEqual(status, 409, conflict)
        status, geo = self.request("GET", f"/api/workspace/plan/export?scene=town&id={pid}")
        self.assertEqual(status, 200)
        self.assertEqual(geo["type"], "FeatureCollection")
        status, where = self.request("GET", "/api/workspace/coords?scene=town&x=0&y=0&z=-100")
        self.assertEqual(status, 200, where)
        self.assertTrue(where["mgrs"].startswith("43R"))
        status, listing = self.request("GET", "/api/workspace/plan/proposals?scene=town")
        self.assertEqual([p["name"] for p in listing["index"]["proposals"]], ["Scheme A"])
        self.assertIn("street_light", listing["catalogue"])

    def test_writes_need_a_browser_origin_and_bad_input_is_a_clean_error(self):
        status, _ = self.request("POST", "/api/workspace/plan/proposals",
                                 {"scene": "town", "name": "X"}, origin=False)
        self.assertEqual(status, 403)
        status, created = self.request("POST", "/api/workspace/plan/proposals",
                                       {"scene": "town", "name": "Scheme A"})
        status, err = self.request("POST", "/api/workspace/plan/feature", {
            "scene": "town", "id": created["proposal"]["id"], "revision": 0,
            "feature": {"type": "building", "params": {"footprint": [[0, 0], [1, 1]]}}})
        self.assertEqual(status, 400)
        self.assertIn("footprint", err["error"])
        status, err = self.request("GET", "/api/workspace/plan/proposal?scene=town&id=../../x")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
