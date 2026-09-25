"""CPU-only tests for per-point multi-view evidence (support/reprojection)."""
from dataclasses import replace
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

PLAIN_HEADER = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""


def plain_cloud(path, rows=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, 0.5, 0.25))):
    """Write an XYZ/RGB cloud that carries no reprojection error or support."""
    body = "\n".join(f"{x} {y} {z} 10 20 30" for x, y, z in rows)
    path.write_text(PLAIN_HEADER + body + "\n", encoding="utf-8")
    return path


def evidence_model(xyz, error, support, *, rgb=None, confidence=None, visible_support=None):
    return ev.EvidenceModel(np.asarray(xyz, float),
                            np.zeros((len(xyz), 3), np.uint8) if rgb is None else rgb,
                            error, support, confidence=confidence,
                            visible_support=visible_support)


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

    def test_missing_evidence_columns_read_back_as_none_not_as_perfect_evidence(self):
        parsed = ev.read_ply(plain_cloud(Path(self.tmp.name) / "plain.ply"))
        self.assertEqual(len(parsed.xyz), 3)
        self.assertIsNone(parsed.error, "absent reprojection error must not become 0.0 px")
        self.assertIsNone(parsed.support, "absent support must not become one view each")
        self.assertIsNone(parsed.confidence)
        self.assertIsNone(parsed.visible_support)

    def test_summary_refuses_to_report_evidence_the_cloud_does_not_carry(self):
        parsed = ev.read_ply(plain_cloud(Path(self.tmp.name) / "plain.ply"))
        with self.assertRaisesRegex(ValueError, "reprojection_error_px"):
            ev.support_summary(parsed)
        with self.assertRaisesRegex(ValueError, "support"):
            ev.support_confidence(parsed)
        full = ev.parse_points3d(self.path)
        self.assertEqual(ev.support_summary(full)["mean_reprojection_error_px"], 0.8)

    def test_partial_evidence_names_the_missing_column(self):
        no_support = evidence_model([[0.0, 0.0, 0.0]], np.array([0.4]), None)
        for call in (lambda: ev.support_summary(no_support),
                     lambda: ev.support_confidence(no_support)):
            with self.assertRaisesRegex(ValueError, "support"):
                call()
        no_error = evidence_model([[0.0, 0.0, 0.0]], None, np.array([3], np.int32))
        with self.assertRaisesRegex(ValueError, "reprojection_error_px"):
            ev.support_summary(no_error)
        with self.assertRaisesRegex(ValueError, "reprojection_error_px"):
            ev.support_confidence(no_error)

    def test_export_refuses_to_write_invented_evidence_columns(self):
        out = Path(self.tmp.name) / "invented.ply"
        with self.assertRaisesRegex(ValueError, "support"):
            ev.export_evidence_ply(evidence_model([[0.0, 1.0, 2.0]], np.array([0.2]), None),
                                   np.array([0.5], np.float32), out)
        self.assertFalse(out.exists())

    def test_visible_support_column_round_trips_only_when_supplied(self):
        source = ev.parse_points3d(self.path)
        visible = np.array([2, 0, 4], np.int32)
        out = Path(self.tmp.name) / "visible.ply"
        ev.export_evidence_ply(replace(source, visible_support=visible),
                               ev.support_confidence(source), out)
        header = out.read_text(encoding="ascii", errors="replace").split("end_header")[0]
        self.assertIn("property int visible_support", header)
        parsed = ev.read_ply(out)
        self.assertEqual(parsed.visible_support.tolist(), [2, 0, 4])
        self.assertEqual(parsed.support.tolist(), [3, 1, 5])
        plain = Path(self.tmp.name) / "hidden-plain.ply"
        ev.export_evidence_ply(source, ev.support_confidence(source), plain)
        self.assertIsNone(ev.read_ply(plain).visible_support)

    def test_read_ply_hands_back_owned_arrays_not_a_lock_on_the_cloud(self):
        source = ev.parse_points3d(self.path)
        out = Path(self.tmp.name) / "locked.ply"
        ev.export_evidence_ply(source, ev.support_confidence(source), out)
        parsed = ev.read_ply(out)
        try:
            out.unlink()  # verifying an artefact must not lock it on Windows
        except PermissionError as exc:
            self.fail(f"read_ply kept {out.name} open: {exc}")
        self.assertEqual(parsed.support.tolist(), [3, 1, 5])
        self.assertTrue(np.allclose(parsed.error, [0.4, 1.9, 0.1]))

    def test_grid_cells_reject_wrapping_and_non_finite_coordinates(self):
        source = ev.parse_points3d(self.path)
        for cell_size in (1e-19, 1e-30):
            with self.subTest(cell_size=cell_size), self.assertRaisesRegex(ValueError, "cell"):
                ev.support_summary(source, cell_size_m=cell_size)
        for offset in (1e30, 2.0 ** 62, float("nan"), float("inf")):
            huge = evidence_model([[offset, 0.0, 0.0], [1.0, 2.0, 3.0]],
                                  np.array([0.4, 0.1]), np.array([3, 4], np.int32))
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "cell"):
                ev.support_summary(huge, cell_size_m=1.0)


if __name__ == "__main__":
    unittest.main()
