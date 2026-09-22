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
        self.assertFalse((self.root / "o4").exists())


if __name__ == "__main__":
    unittest.main()
