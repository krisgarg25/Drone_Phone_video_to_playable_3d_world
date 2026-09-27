"""Inspection and archaeology tools (Phase 5) against known answers."""
import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class FlatGround:
    """Ground at y = 0 with one 10 m block (x, z in [-5, 5]) - the workspace Ground contract."""
    cell = 0.5

    def sample(self, xz):
        xz = np.atleast_2d(xz)
        return np.zeros(len(xz)), np.ones(len(xz), bool)

    def sample_top(self, xz):
        xz = np.atleast_2d(xz)
        inside = (np.abs(xz[:, 0]) <= 5) & (np.abs(xz[:, 1]) <= 5)
        return np.where(inside, 10.0, 0.0), np.ones(len(xz), bool)


def nadir_camera(name, x, z, h=40.0):
    return {"name": name, "pos": [x, h, z], "forward": [0, -1, 0], "up": [0, 0, -1], "right": [1, 0, 0],
            "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720, "t_sec": 0.0}


def side_camera(name, x, z, look):
    f = np.asarray(look, float) - np.array([x, 1.6, z])
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return {"name": name, "pos": [x, 1.6, z], "forward": f.tolist(), "up": u.tolist(), "right": r.tolist(),
            "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720, "t_sec": 1.0}


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.i = importlib.import_module("inspection")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / "viewer_assets").mkdir()
        cams = [nadir_camera("a/0.jpg", 20, 0), side_camera("a/1.jpg", 30, 0, [5.0, 3.0, 0.0]),
                side_camera("a/2.jpg", -30, 0, [5.0, 3.0, 0.0])]
        (self.work / "viewer_assets" / "cameras.json").write_text(json.dumps(cams))
        from PIL import Image
        (self.work / "frames_full" / "a").mkdir(parents=True)
        for k in range(3):
            Image.new("RGB", (2560, 1440), (120, 130, 140)).save(self.work / "frames_full" / "a" / f"{k}.jpg")

    def test_east_wall_point_is_seen_from_the_east_only(self):
        # A point on the block's east face (x = 5) at 3 m: the east camera sees it, the west one is
        # blocked by the block, the nadir camera 40 m up looking down sees only the roof.
        out = self.i.frames_that_saw(self.work, [5.05, 3.0, 0.0], ground=FlatGround())
        names = [f["name"] for f in out["frames"]]
        self.assertIn("a/1.jpg", names)
        self.assertNotIn("a/2.jpg", names)
        east = next(f for f in out["frames"] if f["name"] == "a/1.jpg")
        self.assertAlmostEqual(east["u"], 640, delta=1)
        self.assertAlmostEqual(east["distance_m"], math.dist([30, 1.6, 0], [5.05, 3, 0]), delta=0.05)

    def test_register_round_trip_with_photo(self):
        data, entry = self.i.add_annotation(self.work, {"position": [5.05, 3.0, 0.0], "type": "crack", "severity": 4,
                                                        "note": "diagonal crack", "measurements": {"length_mm": 420}},
                                            ground=FlatGround())
        self.assertEqual(entry["photo"]["frame"], "a/1.jpg")
        self.assertTrue((self.work / entry["photo"]["file"]).is_file())
        self.assertEqual(entry["photo"]["scale"], [2.0, 2.0])            # full-resolution frame
        data = self.i.update_annotation(self.work, entry["id"], {"status": "repair_planned"}, revision=data["revision"])
        self.assertEqual(data["items"][0]["status"], "repair_planned")
        with self.assertRaises(self.i.InspectionError):
            self.i.update_annotation(self.work, entry["id"], {"status": "closed"}, revision=0)   # stale
        with self.assertRaises(self.i.InspectionError):
            self.i.add_annotation(self.work, {"position": [0, 0, 0], "type": "nonsense"})
        pdf = self.i.report_pdf(self.work, title="Bridge 12 inspection", scene_name="test")
        self.assertTrue(pdf.startswith(b"%PDF"))
        data = self.i.delete_annotation(self.work, entry["id"])
        self.assertEqual(data["items"], [])

    def test_crack_candidates_find_a_thin_line(self):
        import cv2
        img = np.full((300, 300, 3), 170, np.uint8)
        cv2.line(img, (40, 50), (250, 240), (60, 60, 60), 2)
        cv2.circle(img, (80, 220), 18, (60, 60, 60), -1)                # a blob, not a crack
        out = self.i.crack_candidates(img, gsd_mm=0.5)
        self.assertEqual(len(out["candidates"]), 1)
        c = out["candidates"][0]
        self.assertAlmostEqual(c["length_px"], math.hypot(210, 190), delta=12)
        self.assertAlmostEqual(c["length_mm"], c["length_px"] * 0.5, delta=0.1)
        blank = np.full((200, 200, 3), 150, np.uint8)
        self.assertEqual(self.i.crack_candidates(blank)["candidates"], [])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.g = importlib.import_module("inspect_geometry")
        self.rng = np.random.default_rng(0)

    def pole(self, tilt_deg=3.0, bearing_deg=90.0, height=10.0, n=3000):
        t = math.radians(tilt_deg)
        b = math.radians(bearing_deg)
        axis = np.array([math.sin(t) * math.sin(b), math.cos(t), -math.sin(t) * math.cos(b)])
        h = self.rng.uniform(0, height, n)
        ang = self.rng.uniform(0, 2 * math.pi, n)
        perp1 = np.cross(axis, [1, 0, 0]); perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        pts = np.outer(h, axis) + 0.15 * (np.outer(np.cos(ang), perp1) + np.outer(np.sin(ang), perp2))
        ground = np.column_stack([self.rng.uniform(-5, 5, 4000), self.rng.normal(0, 0.02, 4000), self.rng.uniform(-5, 5, 4000)])
        return np.vstack([pts + self.rng.normal(0, 0.01, pts.shape), ground])

    def test_tilt_recovered(self):
        out = self.g.tilt(self.pole(3.0, 90.0), [0, 0, 0])
        self.assertAlmostEqual(out["tilt_deg"], 3.0, delta=0.3)
        self.assertAlmostEqual(out["lean_bearing_deg"], 90.0, delta=6)
        self.assertAlmostEqual(out["height_m"], 10.0, delta=0.6)
        self.assertAlmostEqual(out["lean_mm_per_m"], 1000 * math.tan(math.radians(3)), delta=6)
        with self.assertRaises(self.g.GeometryError):
            self.g.tilt(self.pole(), [20, 0, 20])

    def test_wire_sag_and_clearance(self):
        a_param, span = 60.0, 40.0
        s = self.rng.uniform(0, span, 800)
        y = 12.0 + a_param * (np.cosh((s - span / 2) / a_param) - 1) - a_param * (np.cosh((span / 2) / a_param) - 1)
        wire = np.column_stack([s, y + self.rng.normal(0, 0.02, 800), self.rng.normal(0, 0.03, 800)])
        tree = np.column_stack([self.rng.normal(22, 1.0, 600), self.rng.uniform(0, 8.0, 600), self.rng.normal(0, 1.0, 600)])
        ground = FlatGround()
        a = [0, 12.0, 0]; b = [span, 12.0, 0]
        # Ground block sits under x in [-5, 5]: keep the span clear of it.
        out = self.g.wire(np.vstack([wire, tree]), a, b, ground=None, clearance_limit_m=5.0)
        true_sag = a_param * (math.cosh(span / 2 / a_param) - 1)
        self.assertAlmostEqual(out["sag_m"], true_sag, delta=0.08)
        self.assertAlmostEqual(out["min_clearance_m"], 12.0 - true_sag * 0.99 - 8.0, delta=0.6)
        self.assertFalse(out["clearance_ok"])
        self.assertAlmostEqual(out["clearance_at_m"], 22, delta=4)

    def test_section_outline_ground_svg_dxf(self):
        block = np.column_stack([self.rng.uniform(-5, 5, 20000), self.rng.uniform(0, 10, 20000), self.rng.uniform(-5, 5, 20000)])
        block = block[(np.abs(block[:, 0]) > 4.9) | (block[:, 1] > 9.9) | (np.abs(block[:, 2]) > 4.9)]
        block[block[:, 1] > 9.9, 1] = 10.0
        roof = np.column_stack([self.rng.uniform(-5, 5, 8000), np.full(8000, 10.0), self.rng.uniform(-5, 5, 8000)])
        pts = np.vstack([block, roof])
        sec = self.g.section(pts, [-8, 0, 0], [8, 0, 0], ground=FlatGround(), half_width_m=0.3)
        self.assertAlmostEqual(sec["length_m"], 16.0)
        mid = np.nanmax(sec["outline_m"][(sec["station_m"] > 4) & (sec["station_m"] < 12)])
        self.assertAlmostEqual(mid, 10.0, delta=0.05)
        svg = self.g.section_svg(sec)
        self.assertTrue(svg.startswith("<svg") and "scale" not in svg[:5])
        dxf = self.g.section_dxf(sec)
        self.assertIn("OUTLINE", dxf)
        self.assertTrue(dxf.rstrip().endswith("EOF"))

    def test_hillshade_and_lrm(self):
        x = np.arange(100) * 0.5
        floor = np.tile(0.05 * x, (100, 1))
        floor[40:45, 40:45] -= 0.8                                          # a buried ditch
        out = self.g.terrain_rasters(floor, floor, np.ones_like(floor, bool), 0.5)
        self.assertLess(np.nanmin(out["lrm"][40:45, 40:45]), -0.5)
        self.assertLess(abs(float(np.nanmedian(out["lrm"][5:30, 5:30]))), 0.02)   # the regional slope is removed
        self.assertTrue(self.g.raster_png(out["hillshade_dtm"]).startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
