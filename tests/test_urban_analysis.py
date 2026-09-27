"""Tests for URB-01..04 and the URB-12 extra rows (scripts/urban_analysis.py).

Roofs are built point by point with known slopes and aspects; the road is a labelled
strip of known width laid into the synthetic town of test_workspace_proposals.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_workspace_proposals import ground_y, make_scene  # noqa: E402

ua = importlib.import_module("urban_analysis")
NORTH = np.array([0.0, -1.0])          # viewer -Z is north in a local scene


def gable(cx=0.0, cz=0.0, half_x=6.0, half_z=5.0, eave=6.0, pitch_deg=30.0, step=0.25):
    """Points on a gable roof with its ridge along x: one face looks north, one south."""
    rise = math.tan(math.radians(pitch_deg))
    pts = []
    for x in np.arange(cx - half_x + step / 2, cx + half_x, step):
        for z in np.arange(cz - half_z + step / 2, cz + half_z, step):
            pts.append([x, eave + rise * (half_z - abs(z - cz)), z])
    return np.array(pts)


def building(cx=0.0, cz=0.0, half_x=6.0, half_z=5.0, height=9.0):
    fp = [[cx - half_x, cz - half_z], [cx + half_x, cz - half_z], [cx + half_x, cz + half_z], [cx - half_x, cz + half_z]]
    return {"id": "b1", "footprint": fp, "base_y": 0.0, "height_m": height, "area_m2": 4 * half_x * half_z}


class RoofTests(unittest.TestCase):
    def test_gable_gives_two_faces_with_their_slope_and_aspect(self):
        roof = ua.roof_planes(building(height=6.0 + 5 * math.tan(math.radians(30))), gable(), NORTH)
        self.assertEqual(len(roof["planes"]), 2, roof)
        self.assertEqual(roof["kind"], "pitched")
        aspects = sorted(p["aspect"] for p in roof["planes"])
        self.assertEqual(aspects, ["N", "S"])
        for p in roof["planes"]:
            self.assertAlmostEqual(p["slope_deg"], 30.0, delta=1.0)
            # each face is 12 m x 5 m in plan; its true area is plan / cos(30)
            self.assertAlmostEqual(p["plan_area_m2"], 60.0, delta=6.0)
            self.assertAlmostEqual(p["area_m2"], p["plan_area_m2"] / math.cos(math.radians(p["slope_deg"])), delta=1.0)
        south = next(p for p in roof["planes"] if p["aspect"] == "S")
        self.assertAlmostEqual(south["aspect_deg"], 180.0, delta=3.0)

    def test_flat_roof_is_one_flat_plane_and_walls_are_ignored(self):
        flat = np.array([[x, 9.0, z] for x in np.arange(-5.8, 6, 0.25) for z in np.arange(-4.8, 5, 0.25)])
        wall = np.array([[6.0, y, z] for y in np.arange(0.5, 9, 0.25) for z in np.arange(-4.8, 5, 0.25)])
        roof = ua.roof_planes(building(), np.vstack([flat, wall]), NORTH)
        self.assertEqual(len(roof["planes"]), 1)
        p = roof["planes"][0]
        self.assertLess(p["slope_deg"], 1.0)
        self.assertEqual(p["aspect"], "flat")
        self.assertAlmostEqual(p["height_m"], 9.0, delta=0.05)

    def test_bare_footprint_says_so(self):
        roof = ua.roof_planes(building(), np.zeros((0, 3)), NORTH)
        self.assertEqual(roof["planes"], [])
        self.assertIn("too few", roof["note"])


class SolarTests(unittest.TestCase):
    def plane(self, slope, aspect_deg):
        # viewer frame: north = -z, east = +x; a face "looking" toward aspect tilts its normal there
        s, a = math.radians(slope), math.radians(aspect_deg)
        n = [math.sin(s) * math.sin(a), math.cos(s), -math.sin(s) * math.cos(a)]
        return {"normal": n, "area_m2": 50.0, "plan_area_m2": 50.0 * math.cos(s)}

    def test_south_face_beats_flat_beats_north_in_the_northern_hemisphere(self):
        planes, basis = ua.solar_potential([self.plane(30, 180), self.plane(0, 0), self.plane(30, 0)],
                                           None, lat=28.6, lon=77.2)
        south, flat, north = (p["irradiation_kwh_m2_yr"] for p in planes)
        self.assertGreater(south, flat)
        self.assertGreater(flat, north)
        # clear-sky horizontal at Delhi: an upper bound near 2,000-2,700 kWh/m2/yr
        self.assertGreater(flat, 1800)
        self.assertLess(flat, 2900)
        self.assertIn("upper bound", basis)
        self.assertIn("scene north", basis)
        self.assertAlmostEqual(planes[0]["pv_kwp"], 50 * 0.7 * 0.2, delta=0.05)

    def test_local_scene_without_a_latitude_is_refused(self):
        with self.assertRaises(ua.UrbanError) as ctx:
            ua.solar_potential([self.plane(0, 0)], {"status": "local_relative"})
        self.assertEqual(ctx.exception.status, 409)


class SiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = make_scene(self.tmp.name)
        import label_semantics
        path = self.work / "viewer_assets" / "semantics.json"
        data = json.loads(path.read_text())
        # a 7 m road running along x at z in [26.5, 33.5], open ground either side of it
        for x in np.arange(-30, 30, 0.4):
            for z in np.arange(20.2, 40, 0.4):
                label = "road" if 26.5 <= z <= 33.5 else "ground"
                data["coords"].append([float(x), float(ground_y(x, z)), float(z)])
                data["rgb"].append(list(label_semantics.CLASS_RGB[label]))
        path.write_text(json.dumps(data))
        self.wp = importlib.import_module("workspace_proposals")
        self.ground = self.wp.Ground(self.work)
        self.existing = self.wp.existing_inventory(self.work, self.ground)

    def test_road_width_across_the_strip(self):
        out = ua.road_width(self.work, [0, 0, 22], [0, 0, 38])
        self.assertAlmostEqual(out["width_m"], 7.0, delta=0.6)
        self.assertLess(out["uncertainty_m"], 1.2)
        self.assertEqual(len(out["edges"]), 2)
        # a skewed section is longer than the road is wide: the width is along the line drawn
        skew = ua.road_width(self.work, [-8, 0, 22], [8, 0, 38])
        self.assertAlmostEqual(skew["width_m"], 7.0 * math.sqrt(2), delta=0.8)

    def test_a_line_off_the_road_finds_none(self):
        out = ua.road_width(self.work, [0, 0, 36], [0, 0, 39])
        self.assertIsNone(out["width_m"])

    def test_surface_layers_count_road_roof_and_canopy(self):
        layers = ua.surface_layers(self.work, self.ground, self.existing)
        self.assertEqual(layers["status"], "measured")
        self.assertAlmostEqual(layers["road_surface_m2"], 60 * 7.2, delta=60)
        self.assertGreater(layers["impervious_m2"], layers["road_surface_m2"])
        self.assertEqual(layers["canopy"]["trees"], 1)
        self.assertGreater(layers["canopy"]["height_m"]["max"], 4)
        self.assertIsNotNone(layers["impervious_pct"])
        overlay = ua.road_overlay(layers, self.ground)
        self.assertEqual(len(overlay), 1)

    def test_parking_rates_and_override(self):
        ecs, rates = ua.parking_demand({"residential": 1000.0, "commercial": 500.0})
        self.assertAlmostEqual(ecs, 10 * 2.0 + 5 * 3.0)
        ecs, _ = ua.parking_demand({"residential": 1000.0}, {"parking_ecs_per_100m2": {"residential": 1.0}})
        self.assertAlmostEqual(ecs, 10.0)

    def test_study_runs_on_a_scene_without_a_splat_cloud(self):
        out = ua.study(self.work, self.ground, self.existing, {"status": "local_relative"})
        self.assertEqual(len(out["buildings"]), 1)
        self.assertTrue(any("splat" in n for n in out["notes"]))


if __name__ == "__main__":
    unittest.main()
