"""CPU tests for occlusion-aware visibility support using depth maps.

COLMAP convention: x_cam = R @ x_world + t, camera looks along +Z in its own
frame, so a point in front of the camera has z_cam > 0.
"""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import survey_visibility as vis


def _view(depth, *, fx=100.0, cx=4.0, cy=4.0, translation=(0, 0, 0)):
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], float)
    viewmat = np.eye(4)
    viewmat[:3, 3] = translation
    return {"K": K, "viewmat": viewmat, "depth": np.asarray(depth, np.float32)}


class SurveyVisibilityTests(unittest.TestCase):
    def test_point_on_the_recorded_surface_counts_as_visible(self):
        depth = np.full((9, 9), 5.0, np.float32)
        self.assertEqual(vis.visible_support(np.array([[0.0, 0.0, 5.0]]), [_view(depth)]).tolist(), [1])

    def test_point_behind_the_recorded_surface_is_occluded(self):
        depth = np.full((9, 9), 5.0, np.float32)
        self.assertEqual(vis.visible_support(np.array([[0.0, 0.0, 8.0]]), [_view(depth)]).tolist(), [0])

    def test_point_behind_the_camera_is_not_observed(self):
        depth = np.full((9, 9), 5.0, np.float32)
        self.assertEqual(vis.visible_support(np.array([[0.0, 0.0, -1.0]]), [_view(depth)]).tolist(), [0])

    def test_projection_uses_intrinsics_for_off_centre_points(self):
        depth = np.full((9, 9), 5.0, np.float32)
        view = _view(depth)
        # u = cx + fx * x / z = 4 + 100 * 0.25 / 5 = 9, outside the 9x9 image.
        self.assertEqual(vis.visible_support(np.array([[0.25, 0.0, 5.0]]), [view]).tolist(), [0])
        # u = 4 + 100 * 0.2 / 5 = 8, inside it.
        self.assertEqual(vis.visible_support(np.array([[0.2, 0.0, 5.0]]), [view]).tolist(), [1])

    def test_camera_rotation_is_applied_before_projection(self):
        depth = np.full((9, 9), 5.0, np.float32)
        view = _view(depth)
        # Columns are camera axes in world coordinates: world +X becomes camera +Z.
        view["viewmat"][:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        self.assertEqual(vis.visible_support(np.array([[5.0, 0.0, 0.0]]), [view]).tolist(), [1])
        self.assertEqual(vis.visible_support(np.array([[-5.0, 0.0, 0.0]]), [view]).tolist(), [0])

    def test_missing_depth_measurements_do_not_count_as_visibility(self):
        depth = np.full((9, 9), 5.0, np.float32)
        depth[4, 4] = 0.0
        self.assertEqual(vis.visible_support(np.array([[0.0, 0.0, 5.0]]), [_view(depth)]).tolist(), [0])

    def test_multiple_views_accumulate(self):
        depth = np.full((9, 9), 5.0, np.float32)
        points = np.array([[0.0, 0.0, 5.0]])
        self.assertEqual(vis.visible_support(points, [_view(depth), _view(depth)]).tolist(), [2])

    def test_summary_reports_retained_views_and_lost_points(self):
        summary = vis.visibility_summary(np.array([4, 3, 2, 1]), np.array([4, 0, 2, 0]), min_views=2)
        self.assertEqual(summary["point_count"], 4)
        self.assertEqual(summary["views_before"], 10)
        self.assertEqual(summary["views_after"], 6)
        self.assertAlmostEqual(summary["retained_view_fraction"], 0.6)
        # Of the three points that met min_views before (4, 3, 2), only the second
        # drops below it, so one quarter of all points lose the threshold.
        self.assertAlmostEqual(summary["points_losing_min_views"], 0.25)
        self.assertEqual(summary["min_views"], 2)

    def test_invalid_inputs_raise(self):
        depth = np.full((9, 9), 5.0, np.float32)
        bad_views = [
            {"K": np.zeros((3, 3)), "viewmat": np.eye(4), "depth": depth},      # zero focal
            {"K": np.eye(3) * 100, "viewmat": np.eye(3), "depth": depth},       # wrong shape
            {"K": np.eye(3) * 100, "viewmat": np.eye(4), "depth": np.zeros(9)}, # not 2-D
            {"K": np.eye(3) * 100, "viewmat": np.eye(4), "depth": np.full((9, 9), np.nan)},
        ]
        for view in bad_views:
            with self.subTest(view=view), self.assertRaises(ValueError):
                vis.visible_support(np.zeros((1, 3)), [view])
        with self.assertRaises(ValueError):
            vis.visible_support(np.zeros((1, 2)), [_view(depth)])
        with self.assertRaises(ValueError):
            vis.visible_support(np.zeros((1, 3)), [_view(depth)], relative_tolerance=0)
        with self.assertRaises(ValueError):
            vis.visibility_summary(np.array([1]), np.array([1, 2]))

    def test_depth_map_reader_accepts_colmap_binary_layout(self):
        rows, cols = 3, 4
        values = np.arange(rows * cols, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "depth_map_0000.bin"
            path.write_bytes(struct.pack("<ii", rows, cols) + values.tobytes())
            read = vis.read_depth_map(path)
            self.assertEqual(read.shape, (rows, cols))
            self.assertTrue(np.allclose(read, values.reshape(rows, cols)))
            short = Path(folder) / "depth_map_0001.bin"
            short.write_bytes(struct.pack("<ii", rows, cols) + values[:-1].tobytes())
            with self.assertRaises(ValueError):
                vis.read_depth_map(short)


if __name__ == "__main__":
    unittest.main()
