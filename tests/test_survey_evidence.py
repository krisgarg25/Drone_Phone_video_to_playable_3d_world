"""CPU-only tests for per-point multi-view evidence (support/reprojection)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import survey_evidence as ev


POINTS = """#3D point list with one line of data per point:
#POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 0.0 0.0 0.0 10 20 30 0.4 1:0 2:1 3:2
2 1.0 2.0 3.0 40 50 60 1.9 1:3
3 -1.0 0.5 0.25 1 2 3 0.1 1:4 2:5 3:6 4:7 5:8
"""


class SurveyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "points3D.txt"
        self.path.write_text(POINTS, encoding="utf-8")

    def test_parses_positions_colors_reprojection_error_and_support(self):
        model = ev.parse_points3d(self.path)
        self.assertEqual(model.xyz.shape, (3, 3))
        self.assertEqual(model.rgb.dtype, np.uint8)
        self.assertEqual(model.rgb.tolist(), [[10, 20, 30], [40, 50, 60], [1, 2, 3]])
        self.assertEqual(model.error.tolist(), [0.4, 1.9, 0.1])
        self.assertEqual(model.support.tolist(), [3, 1, 5])

    def test_malformed_rows_are_reported_and_zero_support_is_kept(self):
        bad = self.path.with_name("bad.txt")
        bad.write_text(POINTS + "4 nan 0 0 0 0 0 0 1:0\n5 0 0 0 0 0 0 0.1\n", encoding="utf-8")
        model = ev.parse_points3d(bad)
        # The non-finite row is unusable; the track-less row is a real point that
        # no view supports, so it must stay visible as zero support.
        self.assertEqual(len(model.xyz), 4)
        self.assertEqual(model.rejected, 1)
        self.assertEqual(model.support.tolist(), [3, 1, 5, 0])

    def test_empty_model_raises_instead_of_producing_empty_evidence(self):
        empty = self.path.with_name("empty.txt")
        empty.write_text("# header only\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            ev.parse_points3d(empty)

    def test_summary_counts_support_fraction_and_spatial_cells(self):
        model = ev.parse_points3d(self.path)
        summary = ev.support_summary(model, min_views=3, cell_size_m=1.0)
        self.assertEqual(summary["point_count"], 3)
        self.assertEqual(summary["min_views"], 3)
        self.assertAlmostEqual(summary["well_supported_fraction"], 2 / 3, places=6)
        # Cells: (0,0,0), (1,2,3), (-1,0,0); the poorly supported middle point
        # leaves one of three cells without well-supported geometry.
        self.assertEqual(summary["cell_count"], 3)
        self.assertEqual(summary["well_supported_cell_count"], 2)
        self.assertAlmostEqual(summary["empty_cell_fraction"], 1 / 3, places=6)
        self.assertAlmostEqual(summary["mean_reprojection_error_px"], 0.8)

    def test_confidence_is_zero_for_unsupported_and_one_for_strong_support(self):
        model = ev.parse_points3d(self.path)
        confidence = ev.support_confidence(model, min_views=3, max_error_px=1.0)
        self.assertEqual(confidence[1], 0.0)
        self.assertGreater(confidence[2], confidence[0])
        self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)))

    def test_export_writes_readable_ply_with_evidence_properties(self):
        model = ev.parse_points3d(self.path)
        confidence = ev.support_confidence(model, min_views=3, max_error_px=1.0)
        out = Path(self.tmp.name) / "evidence.ply"
        ev.export_evidence_ply(model, confidence, out)
        header = out.read_text(encoding="ascii", errors="replace").split("end_header")[0]
        self.assertIn("element vertex 3", header)
        for declaration in ("property float x", "property float y", "property float z",
                            "property uchar red", "property uchar green", "property uchar blue",
                            "property int support", "property float reprojection_error_px",
                            "property float confidence"):
            self.assertIn(declaration, header)
        parsed = ev.read_ply(out)
        self.assertEqual(len(parsed.xyz), 3)
        self.assertEqual(parsed.support.tolist(), [3, 1, 5])
        self.assertAlmostEqual(float(parsed.confidence[2]), float(confidence[2]), places=5)

    def test_export_is_atomic_and_rejects_mismatched_lengths(self):
        model = ev.parse_points3d(self.path)
        out = Path(self.tmp.name) / "nope.ply"
        with self.assertRaises(ValueError):
            ev.export_evidence_ply(model, np.array([0.5, 0.5]), out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
