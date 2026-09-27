"""CPU tests for mission planning and rehearsal records (Phase 3).

A 60 x 60 m flat scene at 0.5 m with one 10 x 10 x 8 m block at the centre - enough to
check sight, exposure, cover, landing zones and the mission pack against hand numbers.
Viewer frame: x east, -z north (the synthetic GPS fit says so).
"""
import re
import importlib
import io
import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CELL, N, ORIGIN = 0.5, 120, -30.0
BLOCK = (-5.0, 5.0, -5.0, 5.0, 8.0)      # x0, x1, z0, z1, height


def make_scene(root, *, georeferenced=True):
    work = Path(root) / "work" / "range"
    va = work / "viewer_assets"
    va.mkdir(parents=True)
    centres = ORIGIN + (np.arange(N) + 0.5) * CELL
    xx, zz = np.meshgrid(centres, centres)
    floor = np.zeros((N, N), dtype="<f4")
    top = floor.copy()
    x0, x1, z0, z1, h = BLOCK
    top[(xx > x0) & (xx < x1) & (zz > z0) & (zz < z1)] = h
    (va / "collision.json").write_text(json.dumps({"nx": N, "nz": N, "cell": CELL, "origin_xz": [ORIGIN, ORIGIN]}))
    floor.tofile(va / "ground.f32")
    top.tofile(va / "heights.f32")
    np.ones((N, N), dtype=np.uint8).tofile(va / "coverage.u8")
    frame = {"rotation_rowmajor": np.eye(3).tolist(), "scale_m_per_unit": 1.0, "scale_source": "AR pose-prior metric path"}
    if georeferenced:
        frame.update(scale_source="GPS telemetry similarity fit (test)", scale_anchor_gps={"scale": 1.0, "alignment": {
            "schema_version": 1, "status": "aligned", "method": "test", "scale": 1.0,
            "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "translation": [0.0, 0.0, 0.0], "fit_rmse_m": 0.0,
            "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979", "altitude_datum": "ellipsoidal",
                                 "origin": {"latitude_deg": 30.7, "longitude_deg": 76.8, "altitude_m": 350.0}}}})
    (work / "frame.json").write_text(json.dumps(frame))
    return work


class Base(unittest.TestCase):
    georeferenced = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = make_scene(self.tmp.name, georeferenced=self.georeferenced)
        self.wp = importlib.import_module("workspace_proposals")
        self.ms = importlib.import_module("mission")
        self.an = importlib.import_module("mission_analysis")
        self.api = importlib.import_module("workspace_plan_api")
        self.p = self.wp.create_proposal(self.work, "Op Test", kind="mission")

    def add(self, feature):
        self.p = self.wp.upsert_feature(self.work, self.p["id"], feature, self.p["revision"])
        return self.p["features"][-1]["id"]

    def evaluate(self):
        evaluation, ground, registry, _ = self.api._evaluate(self.work, self.p)
        return evaluation, ground, registry

    def hostile(self, x, z, bearing, **extra):
        return self.add({"type": "symbol", "name": "Enemy", "params": {"position": [x, z], "role": "infantry",
                                                                       "bearing_deg": bearing, **extra}})


class FeatureTests(Base):
    def test_defaults_wrap_and_kinds(self):
        clean = self.wp.validate_feature({"type": "symbol", "params": {"position": [1, 2], "role": "sniper", "bearing_deg": 450}})
        self.assertEqual((clean["params"]["affiliation"], clean["params"]["bearing_deg"], clean["params"]["range_m"]), ("hostile", 90.0, 300.0))
        self.assertEqual(self.wp.validate_feature({"type": "symbol", "params": {"position": [0, 0], "role": "rally_point"}})["params"]["affiliation"], "friendly")
        for bad in ({"type": "symbol", "params": {"position": [0]}}, {"type": "route", "params": {"waypoints": [[0, 0], [0.2, 0]]}},
                    {"type": "symbol", "params": {"position": [0, 0], "behaviour": "berserk"}}):
            with self.assertRaises(self.wp.ProposalError):
                self.wp.validate_feature(bad)
        plan = self.wp.create_proposal(self.work, "Scheme", kind="plan")
        self.assertEqual([p["name"] for p in self.wp.list_proposals(self.work, "mission")["proposals"]], ["Op Test"])
        self.assertEqual([p["name"] for p in self.wp.list_proposals(self.work, "plan")["proposals"]], ["Scheme"])
        copy = self.wp.create_proposal(self.work, "Op Test copy", source=self.p["id"])
        self.assertEqual(copy["kind"], "mission")
        with self.assertRaises(self.wp.ProposalError):
            self.wp.create_proposal(self.work, "x", kind="war")
        self.assertTrue(plan)

    def test_a_long_route_stays_within_the_viewer_mesh_limit(self):
        wps = [[-25 + 2.5 * k, 25 - (k % 2)] for k in range(20)]
        fid = self.add({"type": "route", "params": {"waypoints": wps}})
        ev, _, _ = self.evaluate()
        self.assertLessEqual(len(ev["features"][fid]["meshes"]), 16)

    def test_symbols_sit_on_roofs_and_bearings_become_viewer_vectors(self):
        roof = self.hostile(0, 0, 90)
        ground = self.hostile(-20, 20, 0)
        ev, _, _ = self.evaluate()
        self.assertAlmostEqual(ev["features"][roof]["metrics"]["base_y"], 8.0, delta=0.01)
        self.assertAlmostEqual(ev["features"][roof]["metrics"]["eye_y"], 9.6, delta=0.01)
        np.testing.assert_allclose(ev["features"][roof]["facing_xz"], [1, 0], atol=1e-6)     # 90 deg = east = +x
        np.testing.assert_allclose(ev["features"][ground]["facing_xz"], [0, -1], atol=1e-6)  # 0 deg = north = -z
        parts = {m["part"] for m in ev["features"][roof]["meshes"]}
        self.assertTrue({"pole", "frame", "facing", "sector"} <= parts)


class SightTests(Base):
    def setUp(self):
        super().setUp()
        _, ground, _ = self.evaluate()
        self.terrain = self.an.Terrain(ground.grid)

    def test_los_is_blocked_by_the_block_and_clear_beside_it(self):
        visible, at = self.an.los(self.terrain, [-20, 1.6, 0], [20, 1.2, 0])
        self.assertFalse(visible)
        self.assertAlmostEqual(at[0], -5, delta=0.6)
        self.assertTrue(self.an.los(self.terrain, [-20, 1.6, 10], [20, 1.2, 10])[0])
        # From the roof's edge the ground below is in view; from its middle the edge hides it.
        self.assertTrue(self.an.los(self.terrain, [4.5, 9.6, 4.5], [20, 1.2, 20])[0])
        self.assertFalse(self.an.los(self.terrain, [0, 9.6, 0], [8, 1.2, 8])[0])

    def test_viewshed_has_a_shadow_behind_the_block_and_honours_the_sector(self):
        vs = self.an.viewshed(self.terrain, [-20, 1.6, 0], range_m=60)
        at = lambda x, z: bool(self.terrain.at(vs, [x, z])[0])
        self.assertTrue(at(-10, 0))
        self.assertFalse(at(15, 0))          # directly behind the block
        self.assertTrue(at(15, 15))          # off to the side of it
        east = self.an.viewshed(self.terrain, [-20, 1.6, 0], range_m=60, facing_xz=[1, 0], sector_deg=60)
        self.assertFalse(bool(self.terrain.at(east, [-25, 0])[0]))   # behind the observer
        self.assertTrue(bool(self.terrain.at(east, [-10, 2])[0]))


class RouteTests(Base):
    def test_tobler_eta_on_flat_ground(self):
        _, ground, _ = self.evaluate()
        rep = self.an.route_report(self.an.Terrain(ground.grid), [[-25, 25], [25, 25]], [])
        self.assertAlmostEqual(rep["length_m"], 50.0, delta=0.1)
        self.assertAlmostEqual(rep["eta_s"], 50 / (6 * math.exp(-3.5 * 0.05) / 3.6), delta=0.5)
        self.assertEqual((rep["exposure_s"], rep["dead_ground_pct"]), (0.0, 100.0))

    def test_exposure_depends_on_facing_and_cover(self):
        self.hostile(0, -20, 180, sector_deg=120, range_m=80)       # north of the block, facing south
        ev, ground, _ = self.evaluate()
        terrain = self.an.Terrain(ground.grid)
        hostiles = self.an.observers(self.p, ev)
        behind = self.an.route_report(terrain, [[-2, 10], [2, 10]], hostiles)       # in the block's shadow
        open_ = self.an.route_report(terrain, [[15, 10], [25, 10]], hostiles)       # beside it, in the sector
        self.assertEqual(behind["exposure_s"], 0.0)
        self.assertGreater(open_["exposure_s"], 5.0)
        self.assertEqual(open_["per_hostile"][0]["first_seen_m"], 0.0)
        self.hostile(0, 25, 0, sector_deg=60)                       # a second one facing away north... toward the block
        ev, _, _ = self.evaluate()
        self.assertEqual(len(self.an.observers(self.p, ev)), 2)

    def test_covered_route_trades_length_for_exposure(self):
        self.hostile(-25, -25, 135, sector_deg=90, range_m=80)      # NW corner looking south-east over the field
        ev, ground, _ = self.evaluate()
        terrain = self.an.Terrain(ground.grid)
        hostiles = self.an.observers(self.p, ev)
        planned = [[-25, 25], [25, -25]]
        straight = self.an.route_report(terrain, planned, hostiles)
        line = self.an.covered_route(terrain, np.array(planned[0], float), np.array(planned[1], float), hostiles)
        covered = self.an.route_report(terrain, line, hostiles)
        self.assertLess(covered["exposure_s"], straight["exposure_s"])
        self.assertGreaterEqual(covered["length_m"], straight["length_m"] - 0.5)
        np.testing.assert_allclose(line[0], planned[0])
        np.testing.assert_allclose(line[-1], planned[-1])

    def test_phase_line_crossings_and_waypoint_times(self):
        _, ground, _ = self.evaluate()
        rep = self.an.route_report(self.an.Terrain(ground.grid), [[-25, 25], [0, 25], [25, 25]], [],
                                   phase_lines=[{"label": "PL BLUE", "line": [[10, 20], [10, 28]]}], names=["SP", "", "OBJ"])
        self.assertEqual([w["name"] for w in rep["waypoints"]], ["SP", "WP2", "OBJ"])
        self.assertAlmostEqual(rep["waypoints"][1]["station_m"], 25.0, delta=0.01)
        self.assertEqual(rep["phase_lines"][0]["label"], "PL BLUE")
        self.assertAlmostEqual(rep["phase_lines"][0]["station_m"], 35.0, delta=0.6)


class HlzTests(Base):
    def test_threat_heatmap_prefers_the_roof_and_ignores_dead_ground(self):
        _, ground, _ = self.evaluate()
        terrain = self.an.Terrain(ground.grid)
        out = self.an.threat_heatmap(terrain, [[-25, 20], [25, 20]], range_m=200)
        best = out["candidates"][0]
        self.assertTrue(best["elevated"], best)
        self.assertTrue(abs(best["position"][0]) < 5 and abs(best["position"][1]) < 5)
        # Right behind the block (north of it) nobody sees the route: zero score.
        j, i = terrain.index([[0.0, -8.0]])
        self.assertEqual(float(out["score"][j[0], i[0]]), 0.0)
        # Nothing is suggested on the route itself.
        self.assertTrue(all(abs(c["position"][1] - 20) > 1.0 for c in out["candidates"]))

    def test_trafficability_classes(self):
        _, ground, _ = self.evaluate()
        terrain = self.an.Terrain(ground.grid)
        out = self.an.trafficability(terrain, mobility="wheeled")
        j, i = terrain.index([[0.0, 0.0]])
        self.assertEqual(int(out["class"][j[0], i[0]]), 2)          # the 8 m block is NO-GO
        j, i = terrain.index([[20.0, 20.0]])
        self.assertEqual(int(out["class"][j[0], i[0]]), 0)          # open flat ground is GO
        self.assertAlmostEqual(out["area_m2"]["no_go"], 100.0, delta=15)
        with self.assertRaises(self.an.AnalysisError):
            self.an.trafficability(terrain, mobility="hovercraft")

    def test_obstacles_list_the_block(self):
        _, ground, _ = self.evaluate()
        out = self.an.obstacles(self.an.Terrain(ground.grid), min_height_m=3.0)
        self.assertEqual(out["count"], 1)
        o = out["obstacles"][0]
        self.assertAlmostEqual(o["height_m"], 8.0)
        self.assertAlmostEqual(o["area_m2"], 100.0, delta=12.0)
        self.assertEqual(self.an.obstacles(self.an.Terrain(ground.grid), min_height_m=9.0)["count"], 0)

    def test_hlz_finds_clear_flat_circles_and_excludes_the_block(self):
        _, ground, _ = self.evaluate()
        out = self.an.hlz_candidates(self.an.Terrain(ground.grid), diameter_m=15)
        self.assertGreaterEqual(len(out["candidates"]), 2)
        for c in out["candidates"]:
            cx, cz = c["centre"]
            self.assertGreaterEqual(c["usable_diameter_m"], 15)
            nearest_block = math.hypot(max(abs(cx) - 5, 0), max(abs(cz) - 5, 0))
            self.assertGreaterEqual(nearest_block, 7.5 - 0.5)
            self.assertLessEqual(c["max_obstacle_m"], 0.5)
        self.assertEqual(self.an.hlz_candidates(self.an.Terrain(ground.grid), diameter_m=80)["candidates"], [])


class PackTests(Base):
    def test_kmz_and_gpx_carry_symbols_and_routes_in_wgs84(self):
        self.hostile(0, -20, 180)
        self.add({"type": "route", "name": "Route RED", "params": {"waypoints": [[-25, 25], [0, 25], [25, 25]], "names": ["SP", "", "OBJ"]}})
        ev, ground, registry = self.evaluate()
        data, kml, gpx = self.ms.mission_pack(self.p, ev, registry, ground)
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        self.assertEqual(sorted(names), ["Op-Test.gpx", "Op-Test.kmz"])
        self.assertIn("<Point><coordinates>76.8", kml)
        self.assertEqual(gpx.count("<rtept"), 3)
        self.assertIn("<name>OBJ</name>", gpx)
        self.assertIn("planned, not observed", kml)


class LocalPackTests(Base):
    georeferenced = False

    def test_local_scene_refuses_wgs84_pack(self):
        ev, ground, registry = self.evaluate()
        with self.assertRaises(self.wp.ProposalError) as caught:
            self.ms.mission_pack(self.p, ev, None, ground)
        self.assertEqual(caught.exception.status, 409)


class RunTests(Base):
    def frames(self):
        t = np.arange(0, 10.01, 0.1)
        player = np.column_stack([t, t * 2, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.full_like(t, 100)])
        sees = ((t >= 2) & (t < 5)).astype(float)
        bot = np.column_stack([t, np.full_like(t, 30), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), sees, (t < 9).astype(float)])
        return player, bot

    def test_run_summary_exposure_and_storage(self):
        player, bot = self.frames()
        run = {"hz": 10, "player": player.tolist(), "bots": [{"id": 0, "label": "Enemy", "frames": bot.tolist()}],
               "events": [{"t": 4.0, "type": "waypoint", "text": "WP2"}, {"t": 10.0, "type": "complete", "text": "OBJ"}],
               "conditions": {"light": "night", "fog_m": "80"}}
        rec = self.ms.save_run(self.work, self.p["id"], run)
        s = rec["summary"]
        self.assertAlmostEqual(s["duration_s"], 10.0, delta=0.01)
        self.assertAlmostEqual(s["distance_m"], 20.0, delta=0.01)
        jumped = player.copy()
        jumped[60:, 1] += 50                       # a respawn 50 m away mid-run
        # The jump frame (and the 0.2 m walked within it) is not counted.
        self.assertAlmostEqual(self.ms.run_summary({**rec, "player": jumped.tolist()})["distance_m"], 19.8, delta=0.01)
        self.assertAlmostEqual(s["exposure_s"], 3.0, delta=0.15)
        self.assertEqual((s["per_bot"][0]["first_seen_s"], s["neutralised"], s["completed"], s["waypoints_reached"]), (2.0, 1, True, 1))
        listed = self.ms.list_runs(self.work, self.p["id"])
        self.assertEqual(listed[0]["id"], rec["id"])
        self.assertEqual(self.ms.load_run(self.work, self.p["id"], rec["id"])["conditions"]["light"], "night")
        # The after-action review on paper: three pages, starts as a PDF, names the mission.
        import aar_report
        pdf = aar_report.aar_pdf(rec, mission_name="Op TEST", basemap_rgb=np.zeros((10, 10, 3), np.uint8),
                                 bounds=(-10, -10, 40, 40))
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(len(re.findall(rb"/Type\s*/Page(?!s)", pdf)), 3)
        self.assertEqual(self.ms.delete_run(self.work, self.p["id"], rec["id"]), [])

    def test_bad_runs_are_refused(self):
        player, _ = self.frames()
        for bad in ({"hz": 10, "player": [[0, 0]]}, {"hz": 99, "player": player.tolist()},
                    {"hz": 10, "player": [[0, 0, 0, float("nan"), 0, 0]] * 3}):
            with self.assertRaises(self.wp.ProposalError):
                self.ms.save_run(self.work, self.p["id"], bad)
        with self.assertRaises(self.wp.ProposalError):
            self.ms.load_run(self.work, self.p["id"], "../../etc")


class MissionHttpTests(unittest.TestCase):
    def setUp(self):
        import functools
        import threading
        from http.server import ThreadingHTTPServer
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _serve
        from test_workspace_proposals import PlanHttpTests
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make_scene(self.tmp.name)
        with _serve.process_lock:
            _serve.active_process = None
            _serve.active_job_info = {"status": "idle", "scene": "", "step": "", "logs": []}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_serve.H, directory=self.tmp.name))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.request = lambda *a, **k: PlanHttpTests.request(self, *a, **k)

    def test_plan_analyse_rehearse_review(self):
        status, created = self.request("POST", "/api/workspace/plan/proposals", {"scene": "range", "name": "Op A", "kind": "mission"})
        self.assertEqual(status, 200, created)
        pid, rev = created["proposal"]["id"], created["proposal"]["revision"]
        features = [{"type": "symbol", "name": "Sentry", "params": {"position": [-25, -25], "role": "infantry", "bearing_deg": 135, "sector_deg": 90, "range_m": 80}},
                    {"type": "route", "name": "Route RED", "params": {"waypoints": [[-25, 25], [25, -25]]}},
                    {"type": "symbol", "name": "OBJ", "params": {"position": [25, -25], "role": "objective"}}]
        for f in features:
            status, state = self.request("POST", "/api/workspace/plan/feature", {"scene": "range", "id": pid, "revision": rev, "feature": f})
            self.assertEqual(status, 200, state)
            rev = state["proposal"]["revision"]
        route_id = state["proposal"]["features"][1]["id"]
        status, listing = self.request("GET", "/api/workspace/plan/proposals?scene=range&kind=plan")
        self.assertEqual(listing["index"]["proposals"], [])
        status, rep = self.request("GET", f"/api/workspace/mission/analysis?scene=range&id={pid}")
        self.assertEqual(status, 200, rep)
        self.assertEqual(rep["hostiles"], 1)
        self.assertGreater(rep["routes"][0]["report"]["exposure_s"], 0)
        self.assertTrue(rep["overlay"])
        status, sc = self.request("GET", f"/api/workspace/mission/scenario?scene=range&id={pid}")
        self.assertEqual((len(sc["bots"]), len(sc["route"]["waypoints"]), sc["markers"][0]["role"]), (1, 2, "objective"))
        status, covered = self.request("POST", "/api/workspace/mission/covered-route", {"scene": "range", "id": pid, "revision": rev, "route": route_id})
        self.assertEqual(status, 200, covered)
        self.assertLess(covered["suggested"]["covered"]["exposure_s"], covered["suggested"]["planned"]["exposure_s"])
        status, los = self.request("GET", "/api/workspace/mission/los?scene=range&x1=-20&y1=0&z1=0&x2=20&y2=0&z2=0")
        self.assertEqual((status, los["visible"]), (200, False))
        status, hlz = self.request("GET", "/api/workspace/mission/hlz?scene=range&diameter=15")
        self.assertTrue(hlz["candidates"])
        status, pack = self.request("GET", f"/api/workspace/mission/pack?scene=range&id={pid}")
        self.assertEqual((status, pack["media_type"]), (200, "application/zip"))
        status, base = self.request("GET", "/api/workspace/mission/basemap?scene=range")
        self.assertTrue(base["png_base64"].startswith("iVBORw0KGgo"))
        t = [i / 10 for i in range(50)]
        run = {"hz": 10, "player": [[v, v, 0, 0, 0, 100] for v in t], "bots": [{"id": 0, "frames": [[v, 0, 0, 0, 0, 1, 1] for v in t]}], "events": []}
        status, saved = self.request("POST", "/api/workspace/mission/run", {"scene": "range", "id": pid, "run": run})
        self.assertEqual(status, 200, saved)
        status, runs = self.request("GET", f"/api/workspace/mission/runs?scene=range&id={pid}")
        self.assertEqual(runs["runs"][0]["id"], saved["id"])
        status, err = self.request("GET", f"/api/workspace/mission/scenario?scene=range&id=p-0000000000")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
