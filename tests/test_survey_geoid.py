"""Tests for scripts/survey_geoid.py: EGM96 undulation and the MSL product option (F3).

The interpolation is checked on a synthetic grid whose bilinear answer is exact, and -
where the real grid and pyproj are installed - against PROJ's own ``vgridshift`` on the
same file. Delivery is checked both ways: MSL heights with a compound CRS when the grid
exists, and a recorded refusal (heights left ellipsoidal) when it does not.
"""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from scripts import survey_deliver as deliver
from scripts import survey_geoid as geoid
from tests.test_survey_deliver import ALIGNMENT, cloud

REAL_GRID = geoid.GRID_PATH.is_file()


def synthetic_grid(path):
    """N = 0.5 * lat + 0.1 * lon (linear, so bilinear is exact), with the EGM96
    extremes planted at the north pole row so the range check accepts the file."""
    lat = 90.0 - np.arange(geoid.ROWS) * geoid.STEP_DEG
    lon = -180.0 + np.arange(geoid.COLUMNS) * geoid.STEP_DEG
    values = (0.5 * lat[:, None] + 0.1 * lon[None, :]).astype(np.float32)
    values[0, 0], values[0, 1] = -107.0, 85.39
    assert cv2.imwrite(str(path), values)
    geoid._grid.cache_clear()
    return path


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(geoid._grid.cache_clear)
        self.grid = synthetic_grid(Path(self.tmp.name) / "grid.tif")

    def test_bilinear_is_exact_on_a_linear_field(self):
        lat = np.array([12.3, -45.678, 60.01, 0.0])
        lon = np.array([77.2, -120.5, 8.51, 179.5])
        np.testing.assert_allclose(geoid.undulation(lat, lon, self.grid),
                                   0.5 * lat + 0.1 * lon, atol=1e-5)

    def test_heights_convert_both_ways(self):
        h = np.array([300.0, 250.0])
        msl = geoid.to_orthometric([10.0, 20.0], [30.0, 40.0], h, self.grid)
        np.testing.assert_allclose(msl, h - np.array([8.0, 14.0]), atol=1e-5)
        np.testing.assert_allclose(geoid.to_ellipsoidal([10.0, 20.0], [30.0, 40.0], msl,
                                                        self.grid), h, atol=1e-9)

    def test_longitude_wraps_at_the_antimeridian(self):
        # 180 E is the same meridian as 180 W; the cell east of 179.75 is column 0.
        east = geoid.undulation([5.0], [180.0], self.grid)
        west = geoid.undulation([5.0], [-180.0], self.grid)
        np.testing.assert_allclose(east, west, atol=1e-9)

    def test_a_missing_or_foreign_grid_is_refused_not_guessed(self):
        with self.assertRaises(geoid.GeoidUnavailable):
            geoid.undulation([0.0], [0.0], Path(self.tmp.name) / "absent.tif")
        wrong = Path(self.tmp.name) / "wrong.tif"
        cv2.imwrite(str(wrong), np.zeros((721, 1440), np.float32))
        with self.assertRaisesRegex(geoid.GeoidUnavailable, "range"):
            geoid.undulation([0.0], [0.0], wrong)
        self.assertFalse(geoid.describe(Path(self.tmp.name) / "absent.tif")["available"])

    def test_latitude_out_of_range_is_an_error(self):
        with self.assertRaises(ValueError):
            geoid.undulation([91.0], [0.0], self.grid)


@unittest.skipUnless(REAL_GRID, "EGM96 grid not installed in data/geoid")
class RealGridTests(unittest.TestCase):
    def test_published_value_at_the_origin(self):
        # EGM96 at 0 N 0 E is +17.16 m (NGA; GeographicLib's GeoidEval gives 17.1624).
        self.assertAlmostEqual(float(geoid.undulation([0.0], [0.0])[0]), 17.16, delta=0.02)

    @unittest.skipUnless(importlib.util.find_spec("pyproj"), "pyproj not installed")
    def test_matches_proj_vgridshift_on_the_same_file(self):
        from pyproj import Transformer, datadir
        datadir.append_data_dir(str(geoid.GRID_PATH.parent))
        shift = Transformer.from_pipeline("+proj=vgridshift +grids=us_nga_egm96_15.tif "
                                          "+multiplier=1")
        rng = np.random.default_rng(7)
        lat, lon = rng.uniform(-89.0, 89.0, 3000), rng.uniform(-180.0, 180.0, 3000)
        _, _, proj = shift.transform(lon, lat, np.zeros_like(lat))
        self.assertLess(float(np.max(np.abs(geoid.undulation(lat, lon) - proj))), 1e-6)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(geoid._grid.cache_clear)
        self.root = Path(self.tmp.name)

    @unittest.skipUnless(REAL_GRID, "EGM96 grid not installed in data/geoid")
    def test_egm96_products_carry_msl_heights_and_a_compound_crs(self):
        points, colors = cloud()
        ellipsoidal = deliver.deliver(points, colors, ALIGNMENT, self.root / "ellipsoidal")
        msl = deliver.deliver(points, colors, ALIGNMENT, self.root / "msl", vertical_datum="egm96")
        self.assertEqual(msl["georeferenced"]["heights"]["model"], "EGM96")
        self.assertEqual(msl["georeferenced"]["manifest"]["coordinate_frame"]["vertical_epsg"],
                         5773)
        self.assertIn("COMPD_CS", msl["georeferenced"]["manifest"]["crs_wkt"])
        read = lambda root: np.loadtxt(root / "georeferenced/cloud.xyz")  # noqa: E731
        # Delhi sits about 52.6 m BELOW the ellipsoid's geoid offset, so MSL = h + 52.6.
        n = geoid.undulation([28.6], [77.2])[0]
        self.assertAlmostEqual(n, -52.6, delta=0.2)
        gap = read(self.root / "msl")[:, 2] - read(self.root / "ellipsoidal")[:, 2]
        np.testing.assert_allclose(gap, -n, atol=0.05)
        self.assertEqual(ellipsoidal["georeferenced"]["heights"]["datum"], "ellipsoidal")
        with (self.root / "ellipsoidal/georeferenced/positions_wgs84.csv").open() as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[1], ["latitude_deg", "longitude_deg", "ellipsoidal_height_m",
                                   "egm96_height_m"])
        self.assertAlmostEqual(float(rows[2][2]) - float(rows[2][3]), n, delta=0.01)

    def test_without_the_grid_egm96_is_refused_and_heights_stay_ellipsoidal(self):
        points, colors = cloud()
        with mock.patch.object(geoid, "GRID_PATH", self.root / "absent.tif"):
            geoid._grid.cache_clear()
            result = deliver.deliver(points, colors, ALIGNMENT, self.root / "out",
                                     vertical_datum="egm96")
        self.assertEqual(result["georeferenced"]["heights"]["datum"], "ellipsoidal")
        self.assertNotIn("COMPD_CS", result["georeferenced"]["manifest"]["crs_wkt"])
        self.assertEqual([r["set"] for r in result["refusals"]], ["vertical_datum"])
        with (self.root / "out/georeferenced/positions_wgs84.csv").open() as stream:
            self.assertNotIn("egm96_height_m", stream.readlines()[1])

    def test_an_unknown_datum_is_an_error(self):
        points, colors = cloud()
        with self.assertRaises(ValueError):
            deliver.deliver(points, colors, ALIGNMENT, self.root / "x", vertical_datum="navd88")


if __name__ == "__main__":
    unittest.main()
