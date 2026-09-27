"""CPU-only tests for the SMRF ground filter (M2) on scenes with known ground truth."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def scene(seed=0, *, ground_slope=0.05, n_ground=60_000):
    """Sloped terrain, a 20x30 m 10 m-tall building, six trees and a few low blunders."""
    rng = np.random.default_rng(seed)
    terrain = lambda x, y: ground_slope * x + 0.3 * np.sin(y / 15.0)
    xy = rng.uniform(0, 120, (n_ground, 2))
    in_building = (xy[:, 0] > 50) & (xy[:, 0] < 70) & (xy[:, 1] > 40) & (xy[:, 1] < 70)
    xy = xy[~in_building]                           # no ground is seen under the roof
    ground = np.column_stack([xy, terrain(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.05, len(xy))])
    roof_xy = np.column_stack([rng.uniform(50, 70, 6000), rng.uniform(40, 70, 6000)])
    roof = np.column_stack([roof_xy, np.full(len(roof_xy), terrain(60, 55) + 10.0)])
    wall_z = rng.uniform(0, 10, 3000)
    wall = np.column_stack([np.full(3000, 50.0), rng.uniform(40, 70, 3000),
                            terrain(50, 55) + wall_z])
    trees = []
    for cx, cy in ((15, 20), (25, 90), (95, 15), (100, 100), (35, 55), (85, 60)):
        r = np.sqrt(rng.uniform(0, 1, 1500)) * 3.0
        a = rng.uniform(0, 2 * np.pi, 1500)
        z = terrain(cx, cy) + 8.0 - r * 1.2 + rng.normal(0, 0.2, 1500)
        trees.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a), z]))
    blunders = np.column_stack([rng.uniform(5, 115, (12, 2)), np.full(12, -8.0)])
    points = np.vstack([ground, roof, wall, *trees, blunders])
    truth = np.r_[np.ones(len(ground), bool), np.zeros(len(points) - len(ground), bool)]
    return points, truth, terrain


class TerrainTests(unittest.TestCase):
    def setUp(self):
        self.t = importlib.import_module("survey_terrain")

    def test_ground_and_objects_are_separated(self):
        points, truth, _ = scene()
        out = self.t.classify_ground(points, cell_m=1.0)
        ground = out["classification"] == 2
        recall = ground[truth].mean()
        false_ground = ground[~truth].mean()
        self.assertGreater(recall, 0.97, recall)
        self.assertLess(false_ground, 0.03, false_ground)

    def test_dtm_under_the_building_is_interpolated_and_flagged(self):
        points, _, terrain = scene()
        out = self.t.classify_ground(points, cell_m=1.0)
        a, b, _, d, _, f = out["transform"]
        row, col = int((55 - d) / f), int((60 - a) / b)
        self.assertAlmostEqual(float(out["dtm"][row, col]), terrain(60, 55), delta=0.3)
        self.assertFalse(out["ground_observed"][row, col])
        self.assertAlmostEqual(float(out["ndsm"][row, col]), 10.0, delta=0.4)
        # North-up: the northern edge is row 0.
        north_row = int((115 - d) / f)
        self.assertLess(north_row, row)

    def test_low_blunders_do_not_drag_the_dtm_down(self):
        points, _, terrain = scene()
        out = self.t.classify_ground(points, cell_m=1.0)
        self.assertGreater(out["report"]["low_blunder_cells"], 0)
        blunders = points[-12:]
        self.assertTrue(np.all(out["classification"][-12:] == 1))
        a, b, _, d, _, f = out["transform"]
        for x, y, _ in blunders:
            dtm = out["dtm"][int((y - d) / f), int((x - a) / b)]
            self.assertAlmostEqual(float(dtm), terrain(x, y), delta=0.5)

    def test_steep_terrain_survives_with_a_matching_slope_parameter(self):
        points, truth, _ = scene(ground_slope=0.25)
        loose = self.t.classify_ground(points, cell_m=1.0, slope=0.35)
        self.assertGreater((loose["classification"] == 2)[truth].mean(), 0.97)

    def test_report_is_honest_about_interpolation(self):
        points, _, _ = scene()
        report = self.t.classify_ground(points, cell_m=2.0)["report"]
        self.assertGreater(report["cells_interpolated"], 0)
        self.assertEqual(report["cells"], report["cells_ground_observed"] + report["cells_interpolated"])
        self.assertIn("interpolated", report["notes"][0])

    def test_bad_input_refused(self):
        with self.assertRaises(ValueError):
            self.t.classify_ground(np.zeros((5, 3)))
        with self.assertRaises(ValueError):
            self.t.classify_ground(scene()[0], cell_m=0)


if __name__ == "__main__":
    unittest.main()
