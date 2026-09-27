"""Operations API (Phase 4) end to end on the synthetic two-epoch site (make_ops_scene)."""
import base64
import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_ops_scene  # noqa: E402
import workspace_api as api  # noqa: E402
import workspace_ops_api as ops  # noqa: E402


class Server:
    process_lock = threading.Lock()


class OpsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_ops_scene.write_epoch(cls.root / "work" / "opsbefore", after=False)
        make_ops_scene.write_epoch(cls.root / "work" / "opsafter", after=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def post(self, route, **data):
        return ops.post(api, self.root, route, dict(scene="opsafter", **data), Server())

    def get(self, route, **query):
        q = {k: [v] for k, v in dict(scene="opsafter", **query).items()}
        return ops.get(api, self.root, route, q)

    def test_summary_lists_the_other_epoch(self):
        s = self.get("ops/summary")
        self.assertEqual(s["frame"]["status"], "georeferenced")
        self.assertEqual(s["epochs"][0]["scene"], "opsbefore")
        self.assertTrue(s["epochs"][0]["same_site"])
        self.assertIn("pad.xml", s["designs"])
        self.assertGreater(s["road_labels"], 1000)
        self.assertGreaterEqual(s["detections"]["raw"], 3)       # boxes inside at least one frame

    def test_change_finds_the_event(self):
        out = self.post("ops/change", before="opsbefore")
        regions = out["report"]["regions"]
        kinds = {r["kind"] for r in regions}
        self.assertEqual(kinds, {"gain", "loss"})
        # Stockpile (154 m3) and debris (6 x 7 x 2.5 = 105 m3) are gains; B2/B3 and the pad are losses.
        pile = min(regions, key=lambda r: np.hypot(r["centre_enu"][0] + 30, r["centre_enu"][1] + 45))
        self.assertEqual(pile["kind"], "gain")
        self.assertAlmostEqual(pile["volume_m3"], 154, delta=25)
        self.assertTrue(pile["mgrs"].startswith("43R"))
        self.assertTrue(out["overlay"])
        self.assertLess(abs(out["report"]["registration"]["dx_m"]), 0.3)

    def test_volume_of_the_stockpile(self):
        ring = [[-38, 37], [-22, 37], [-22, 53], [-38, 53]]            # viewer x, z
        out = self.post("ops/volume", before="opsbefore", polygon=ring)
        self.assertAlmostEqual(out["report"]["net_m3"], 154, delta=12)

    def test_damage_with_pre_event_footprints(self):
        out = self.post("ops/damage", before="opsbefore")
        grades = {}
        self.assertEqual(len(out["report"]["buildings"]), 3)      # the ridge is terrain, not graded
        for b in out["report"]["buildings"]:
            x, y = b["centre_enu"]
            name = min(make_ops_scene.BUILDINGS, key=lambda k: np.hypot(make_ops_scene.BUILDINGS[k][0] - x,
                                                                         -make_ops_scene.BUILDINGS[k][1] - y))
            grades[name] = b["grade"]
        self.assertEqual(grades, {"B1": "intact", "B2": "collapsed", "B3": "partial"}, out["report"]["buildings"])

    def test_access_blocked_and_route(self):
        out = self.post("ops/access", before="opsbefore", start=[-60, 0], end=[30, 0])
        rep = out["report"]
        # The debris pile and the car parked on the road since the last flight.
        self.assertEqual(sorted(round(b["centre"][0]) for b in rep["blocked"]), [-40, 8])
        route = rep["route"]
        self.assertGreater(route["offroad_m"], 5)              # it has to leave the road round the debris
        path = np.array(route["path"])
        self.assertFalse(np.any((path[:, 0] > 5) & (path[:, 0] < 11) & (np.abs(path[:, 1]) < 3.5)))

    def test_flood_from_the_river(self):
        out = self.post("ops/flood", seed=[-75, 0], rise_m=2.0)
        rep = out["report"]
        self.assertGreater(rep["flooded_m2"], 1000)
        self.assertTrue(out["overlay"])

    def test_detections_layer(self):
        out = self.get("ops/detections")
        objs = out["report"]["objects"]
        self.assertEqual(sorted(o["class"] for o in objs), ["person", "person", "vehicle"])
        for o in objs:
            truth = [p for p in make_ops_scene.PEOPLE + [make_ops_scene.VEHICLE]]
            d = min(np.hypot(o["position"][0] - x, o["position"][2] - z) for x, z in truth)
            self.assertLess(d, 1.0, o)
            self.assertIn("mgrs", o)

    def test_cutfill_and_progress(self):
        out = self.post("ops/cutfill", design="pad.xml", baseline="opsbefore")
        rep = out["report"]
        self.assertAlmostEqual(rep["cut_m3"], 200, delta=30)        # 20 x 20 x 0.5 m still to dig
        self.assertAlmostEqual(rep["zones"][0]["progress_pct"], 75, delta=8)
        with self.assertRaises(api.Error):
            self.post("ops/cutfill", design="missing.xml")

    def test_design_upload_refuses_junk(self):
        with self.assertRaises(api.Error):
            self.post("ops/design", filename="x.xml", content=base64.b64encode(b"not xml").decode())
        good = (self.root / "work/opsafter/ops/designs/pad.xml").read_bytes()
        out = self.post("ops/design", filename="pad copy.xml", content=base64.b64encode(good).decode())
        self.assertEqual(out["design"], "pad-copy.xml")

    def test_corridor_ridge_blind_side(self):
        out = self.post("ops/corridor", line=[[20, -70], [20, 70], [70, 70]],
                        posts=[{"at": [0, 0], "height_m": 3}])
        rep = out["report"]
        self.assertLess(rep["covered_fraction"], 0.9)
        self.assertTrue(rep["blind_stretches"])
        self.assertEqual(len(rep["profile"]["z"]), len(rep["profile"]["covered"]))

    def test_packs_and_tiles(self):
        self.post("ops/change", before="opsbefore")
        f = self.post("ops/pack", kind="change")
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(f["content"]))) as z:
            names = z.namelist()
        self.assertIn("report.pdf", names)
        self.assertIn("change.kmz", names)
        t = self.post("ops/tiles", tile_m=100)
        self.assertGreaterEqual(t["tiles"], 4)
        self.assertTrue(t["crs"])
        with self.assertRaises(api.Error):
            self.post("ops/pack", kind="nonsense")

    def test_detections_become_unknown_mission_candidates(self):
        import mission
        import workspace_proposals
        work = self.root / "work" / "opsafter"
        cands = mission.detection_candidates(work, workspace_proposals.Ground(work))
        self.assertEqual(sorted(c["role"] for c in cands), ["infantry", "infantry", "vehicle"])
        self.assertTrue(all(c["source"] == "detections" and "seen" in c["basis"] for c in cands))
        params = mission.validate_params("symbol", {"position": cands[0]["position"], "role": cands[0]["role"],
                                                    "affiliation": "unknown", "source": "detections"})
        self.assertEqual(params["source"], "detections")

    def test_firstmap_absent_is_explained(self):
        out = self.get("ops/firstmap")
        self.assertFalse(out["available"])


if __name__ == "__main__":
    unittest.main()
