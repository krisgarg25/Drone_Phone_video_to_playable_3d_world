import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import label_semantics as ls


def grid(u, v, fn):
    pts, cols = [], []
    for a in u:
        for b in v:
            pts.append(fn(a, b))
            cols.append([120, 120, 120])
    return np.array(pts, float), np.array(cols, float)


class LabelSemanticsTests(unittest.TestCase):
    def setUp(self):
        u = np.linspace(-3, 3, 12)
        v = np.linspace(-3, 3, 12)
        rng = np.random.default_rng(0)

        # Flat grey road: constant y=0, perfectly planar, up-facing, non-green.
        self.road_pts, self.road_cols = grid(u, v, lambda a, b: [a, 0.0, b])
        self.road_cols[:] = [90, 92, 95]

        # Rough brown ground: jittered height, up-facing, low greenness.
        g, gc = grid(u, v, lambda a, b: [a, 0.0, b])
        g[:, 1] += rng.uniform(0.02, 0.18, len(g))
        self.ground_pts, self.ground_cols = g, gc
        self.ground_cols[:] = [120, 100, 70]

        # Vertical building facade: constant x, tall in y, horizontal normal, grey.
        wall = np.array([[2.0, yy, b] for yy in np.linspace(0.5, 4.0, 12) for b in v])
        self.build_pts = wall
        self.build_cols = np.tile([150, 150, 152], (len(wall), 1)).astype(float)

        # Green rough canopy blob above ground: elevated, green, non-planar.
        veg = np.column_stack([rng.uniform(-2, 2, 200), rng.uniform(1.2, 2.6, 200), rng.uniform(-2, 2, 200)])
        self.veg_pts = veg
        self.veg_cols = np.tile([40, 140, 45], (200, 1)).astype(float)

    def dominant(self, pts, cols):
        labels = ls.classify_points(pts, cols, ground_height=0.0)
        return Counter(labels).most_common(1)[0][0]

    def test_flat_grey_planar_is_road(self):
        self.assertEqual(self.dominant(self.road_pts, self.road_cols), "road")

    def test_rough_brown_planar_is_ground(self):
        self.assertEqual(self.dominant(self.ground_pts, self.ground_cols), "ground")

    def test_vertical_facade_is_building(self):
        self.assertEqual(self.dominant(self.build_pts, self.build_cols), "building")

    def test_green_elevated_rough_is_vegetation(self):
        self.assertEqual(self.dominant(self.veg_pts, self.veg_cols), "vegetation")

    def test_labels_are_from_the_fixed_class_set(self):
        labels = set(ls.classify_points(self.veg_pts, self.veg_cols, ground_height=0.0))
        self.assertTrue(labels <= set(ls.CLASSES))

    def test_normals_are_unit_and_up_facing_for_a_plane(self):
        n = ls.point_normals(self.road_pts, k=8)
        self.assertTrue(np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.all(np.abs(n[:, 1]) > 0.9))

    def test_classifier_is_deterministic(self):
        a = ls.classify_points(self.ground_pts, self.ground_cols, ground_height=0.0)
        b = ls.classify_points(self.ground_pts, self.ground_cols, ground_height=0.0)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
