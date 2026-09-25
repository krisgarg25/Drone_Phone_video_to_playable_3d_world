import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import workspace_measure as wm


def synthetic_cloud():
    """Viewer Y-up: a dense flat ground plane at y=0 (grid over x,z) plus a 3 m post."""
    g = np.linspace(-4, 4, 25)
    ground = np.array([[x, 0.0, z] for x in g for z in g])
    post = np.array([[1.0, h, 1.0] for h in np.linspace(0.0, 3.0, 12)])
    return np.vstack([ground, post])


class WorkspaceMeasureTests(unittest.TestCase):
    def setUp(self):
        self.cloud = synthetic_cloud()

    def test_axis_remap_vertical_is_engine_z(self):
        # Click the post base and top (viewer Y-up): a purely vertical separation
        # must read as engine-Z height, not horizontal.
        rec = wm.measure("distance", self.cloud, [[1, 0, 1], [1, 3, 1]])
        self.assertAlmostEqual(rec["value"], 3.0, places=1)
        self.assertAlmostEqual(rec["components"]["horizontal_m"], 0.0, places=1)
        self.assertAlmostEqual(rec["components"]["vertical_m"], 3.0, places=1)

    def test_distance_between_two_ground_points(self):
        rec = wm.measure("distance", self.cloud, [[-2, 0, 0], [2, 0, 0]])
        self.assertTrue(rec["valid"], rec["reason"])
        self.assertAlmostEqual(rec["value"], 4.0, delta=0.5)
        self.assertEqual(rec["unit"], "m")

    def test_height_above_ground_of_post_top(self):
        rec = wm.measure("height", self.cloud, [[1, 3, 1]])
        self.assertTrue(rec["valid"], rec["reason"])
        self.assertAlmostEqual(rec["value"], 3.0, delta=0.4)

    def test_area_of_a_square_ring(self):
        ring = [[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]]
        rec = wm.measure("area", self.cloud, ring)
        self.assertTrue(rec["valid"], rec["reason"])
        self.assertAlmostEqual(rec["value"], 16.0, delta=1.0)
        self.assertEqual(rec["unit"], "m²")

    def test_uncertainty_is_reported_when_cloud_supports_it(self):
        rec = wm.measure("distance", self.cloud, [[-2, 0, 0], [2, 0, 0]])
        self.assertIsNotNone(rec["uncertainty"])
        self.assertGreaterEqual(rec["uncertainty"]["m"], 0.0)

    def test_click_off_the_cloud_is_invalid_not_fabricated(self):
        rec = wm.measure("distance", self.cloud, [[-2, 0, 0], [1000, 500, -900]], radius_m=0.5)
        self.assertFalse(rec["valid"])
        self.assertIn("reason", rec)

    def test_snapped_points_returned_in_viewer_frame(self):
        rec = wm.measure("distance", self.cloud, [[-2.1, 0.2, 0.1], [1.9, -0.1, 0.0]])
        for pt in rec["snapped"]:
            self.assertEqual(len(pt), 3)
        # Snapped to the ground plane, so viewer-Y should be near zero.
        self.assertTrue(all(abs(p[1]) < 0.5 for p in rec["snapped"]), rec["snapped"])

    def test_volume_above_ground_within_a_footprint(self):
        # A flat-topped 2x2x1 mound (viewer y=1) over a 2x2 square → ~4 m^3 above ground.
        g = np.linspace(-1, 1, 15)
        ground = np.array([[x, 0.0, z] for x in g for z in g])
        mound = np.array([[x, 1.0, z] for x in g for z in g])
        cloud = np.vstack([ground, mound])
        square = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
        rec = wm.measure("volume", cloud, square)
        self.assertTrue(rec["valid"], rec.get("reason"))
        self.assertAlmostEqual(rec["value"], 4.0, delta=0.6)
        self.assertEqual(rec["unit"], "m³")

    def test_volume_is_zero_when_nothing_rises_above_ground(self):
        g = np.linspace(-2, 2, 15)
        cloud = np.array([[x, 0.0, z] for x in g for z in g])
        square = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
        rec = wm.measure("volume", cloud, square)
        self.assertAlmostEqual(rec["value"] or 0.0, 0.0, delta=0.2)

    def test_class_summary_reports_area_and_count_per_class(self):
        import label_semantics as ls
        # A 20x20 m vegetation footprint. class_summary picks a coarse occupancy
        # cell (>=0.5 m) sized for real scene point clouds; on a large patch that
        # coarseness is a few percent, so the area should sit near the true 400 m^2.
        g = np.linspace(0, 20, 50)
        coords = [[float(x), 0.0, float(z)] for x in g for z in g]
        rgb = [list(ls.CLASS_RGB["vegetation"]) for _ in coords]
        sem = {"coords": coords, "rgb": rgb, "classes": list(ls.CLASSES)}
        summary = wm.class_summary(sem)
        self.assertIn("vegetation", summary)
        self.assertEqual(summary["vegetation"]["count"], len(coords))
        self.assertAlmostEqual(summary["vegetation"]["area_m2"], 400.0, delta=40.0)

    def test_class_summary_area_is_an_explicit_cell_occupancy(self):
        # Pin the grid-occupancy semantics: with a fixed 0.5 m cell a 2x2 m square
        # covers exactly 16 half-metre cells -> 4 m^2. This is what the "area" figure
        # means: an occupancy footprint, robust to floaters, not a convex hull.
        import label_semantics as ls
        g = np.arange(-1.0, 1.0, 0.25)
        coords = [[float(x), 0.0, float(z)] for x in g for z in g]
        rgb = [list(ls.CLASS_RGB["vegetation"]) for _ in coords]
        sem = {"coords": coords, "rgb": rgb, "classes": list(ls.CLASSES)}
        summary = wm.class_summary(sem, cell_m=0.5)
        self.assertAlmostEqual(summary["vegetation"]["area_m2"], 4.0, delta=0.6)

    def test_measurements_export_geojson_is_local_and_honest(self):
        measurements = [{"id": "a", "kind": "distance", "label": "Span", "value": 4.0,
                         "unit": "m", "points": [[0, 0, 0], [4, 0, 0]], "stale": False,
                         "valid": True, "uncertainty": {"m": 0.05, "unit": "m"}}]
        gj = wm.measurements_geojson(measurements, scene="rocks")
        self.assertEqual(gj["type"], "FeatureCollection")
        self.assertEqual(len(gj["features"]), 1)
        self.assertEqual(gj["features"][0]["geometry"]["type"], "LineString")
        self.assertEqual(gj["features"][0]["properties"]["uncertainty_m"], 0.05)
        self.assertIn("local", gj["coordinate_reference_system"].lower())

    def test_measurements_export_csv_has_uncertainty_and_reason(self):
        measurements = [{"id": "a", "kind": "area", "label": "Pad", "value": 16.0,
                         "unit": "m²", "points": [[0, 0, 0]], "stale": False, "valid": False,
                         "reason": "weak support", "uncertainty": {"m": 0.3, "unit": "m"}}]
        csv = wm.measurements_csv(measurements)
        self.assertIn("name,kind,value,unit,uncertainty_m,valid,reason", csv.splitlines()[0])
        self.assertIn("weak support", csv)
        self.assertIn("Pad", csv)


if __name__ == "__main__":
    unittest.main()
