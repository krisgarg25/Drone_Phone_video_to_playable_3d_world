"""Tests for georeferenced evidence products (cloud + per-point support)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import survey_products as prod


POINTS = """#POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
1 0.0 0.0 0.0 10 20 30 0.2 1:0 2:1 3:2
2 1.0 2.0 3.0 40 50 60 2.5 1:3
3 2.0 1.0 0.5 1 2 3 0.1 1:4 2:5 3:6 4:7 5:8
"""

ALIGNMENT = {"schema_version": 1, "status": "aligned", "scale": 2.0,
             "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
             "translation": [100.0, 200.0, 300.0],
             "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                                  "altitude_datum": "ellipsoidal",
                                  "origin": {"latitude_deg": 28.0, "longitude_deg": 77.0,
                                             "altitude_m": 100.0}}}


class SurveyProductsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.points = self.root / "points3D.txt"
        self.points.write_text(POINTS, encoding="utf-8")

    def views(self, *, point2_depth=1.0):
        """One camera at (0,0,-5) looking along +Z, with a chosen surface per pixel."""
        K = [[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]]
        viewmat = np.eye(4)
        viewmat[2, 3] = 5.0  # x_cam = (x, y, z + 5)
        depth = np.zeros((128, 128), np.float32)
        depth[64, 64] = 5.0        # point 1 sits exactly on the recorded surface
        depth[89, 76] = point2_depth  # point 2 (ray 8.31) is behind this surface
        depth[82, 100] = 20.0      # point 3 is well in front of it
        return [{"K": K, "viewmat": viewmat, "depth": depth, "name": "view_0"}]

    def test_writes_georeferenced_cloud_with_support_and_confidence(self):
        ply, report = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "out")
        self.assertTrue(ply.is_file() and report.is_file())
        model = prod.read_evidence(ply)
        self.assertEqual(len(model.xyz), 3)
        # x=0,y=0 -> east = 100, north = 200 under the 90-degree rotation and scale 2.
        self.assertAlmostEqual(model.xyz[0, 0], 100.0, places=4)
        self.assertAlmostEqual(model.xyz[0, 1], 200.0, places=4)
        self.assertEqual(model.support.tolist(), [3, 1, 5])
        self.assertEqual(model.rgb[2].tolist(), [1, 2, 3])
        self.assertTrue(np.all(model.confidence >= 0) and np.all(model.confidence <= 1))
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(summary["point_count"], 3)
        self.assertEqual(summary["coordinate_frame"], ALIGNMENT["coordinate_frame"])
        self.assertAlmostEqual(summary["well_supported_fraction"], 2 / 3, places=6)
        self.assertFalse(list(ply.with_name("out").glob("*.tmp")))

    def test_alignment_is_applied_to_positions_not_to_colors_or_counts(self):
        model = prod.read_evidence(prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "o2")[0])
        self.assertEqual(model.rgb.dtype, np.uint8)
        self.assertEqual(model.support.dtype, np.int32)
        self.assertTrue(np.all(model.error >= 0))

    def test_bad_alignment_or_missing_points_fail_clearly(self):
        broken = dict(ALIGNMENT, scale=-1.0)
        with self.assertRaises(ValueError):
            prod.evidence_for_sparse(self.points, broken, self.root / "o3")
        empty = self.root / "empty.txt"
        empty.write_text("# nothing\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            prod.evidence_for_sparse(empty, ALIGNMENT, self.root / "o4")
        with self.assertRaises(ValueError):
            prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "o5", views=[])
        self.assertFalse((self.root / "o4").exists())
        self.assertFalse((self.root / "o5").exists())

    def test_views_add_a_visible_support_column_and_a_visibility_summary(self):
        ply, report = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "vis",
                                               views=self.views())
        header = ply.read_text(encoding="ascii", errors="replace").split("end_header")[0]
        self.assertIn("property int support", header)
        self.assertIn("property int visible_support", header)
        model = prod.read_evidence(ply)
        self.assertEqual(model.support.tolist(), [3, 1, 5])  # claimed COLMAP tracks
        self.assertEqual(model.visible_support.tolist(), [1, 0, 1])  # measured visibility
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertIs(summary["occlusion_checked"], True)
        self.assertEqual(summary["visibility"]["views_before"], 9)
        self.assertEqual(summary["visibility"]["views_after"], 2)
        self.assertEqual(summary["visibility"]["point_count"], 3)
        self.assertEqual(summary["visibility"]["min_views"], 2)
        self.assertEqual(summary["support_basis"], "colmap_track_length")
        self.assertEqual(summary["visibility_basis"], "depth_map_ray_consistency")
        self.assertEqual(summary["visibility_source_frames"], ["view_0"])
        self.assertIn("surveyed", summary["interpretation"])
        self.assertIn("visible_support", summary["interpretation"])
        self.assertIn("occlusion", summary["visibility_note"].lower())
        self.assertFalse(list((self.root / "vis").glob("*.tmp")))

    def test_visibility_is_measured_in_the_sparse_frame_not_the_transformed_cloud(self):
        """The scale 2 / 90 degree alignment must not move the geometry off the surface."""
        ply, report = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "vis2",
                                               views=self.views())
        self.assertEqual(prod.read_evidence(ply).visible_support.tolist(), [1, 0, 1])
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["visibility"]["views_after"], 2)

    def test_visibility_tolerance_reaches_the_depth_consistency_test(self):
        # Point 2 sits at ray distance 8.31 against a recorded surface at 8.0: a
        # hair behind it, so the tolerance decides whether it counts as visible.
        views = self.views(point2_depth=8.0)
        strict = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "strict",
                                          views=views)[1]
        loose = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "loose",
                                         views=views, relative_tolerance=1.0)[1]
        self.assertEqual(json.loads(strict.read_text(encoding="utf-8"))["visibility"]["views_after"], 2)
        self.assertEqual(json.loads(loose.read_text(encoding="utf-8"))["visibility"]["views_after"], 3)

    def test_no_views_means_no_claim_of_occlusion_checking(self):
        ply, report = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "plain")
        header = ply.read_text(encoding="ascii", errors="replace").split("end_header")[0]
        self.assertNotIn("visible_support", header)
        self.assertIsNone(prod.read_evidence(ply).visible_support)
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertIs(summary["occlusion_checked"], False)
        self.assertIsNone(summary["visibility"])
        self.assertEqual(summary["visibility_source_frames"], [])
        self.assertIn("not", summary["visibility_note"].lower())
        self.assertIn("not checked", summary["interpretation"])
        self.assertNotIn("occlusion checked", summary["interpretation"])
        self.assertEqual(summary["point_count"], 3)

    def test_min_views_applies_to_the_track_claim_and_min_visible_views_to_visibility(self):
        _, report = prod.evidence_for_sparse(self.points, ALIGNMENT, self.root / "mv",
                                             views=self.views(), min_views=2, min_visible_views=2)
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(summary["min_views"], 2)
        self.assertAlmostEqual(summary["well_supported_fraction"], 2 / 3, places=6)
        self.assertAlmostEqual(summary["visibility"]["points_losing_min_views"], 2 / 3, places=6)
        self.assertEqual(summary["visibility"]["min_views"], 2)


if __name__ == "__main__":
    unittest.main()
