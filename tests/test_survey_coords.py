"""CPU-only tests for MGRS / DMS / CE90 formatting (M8)."""
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class CoordsTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module("survey_coords")

    def test_published_reference_at_null_island(self):
        self.assertEqual(self.c.to_mgrs(0.0, 0.0), "31NAA6602100000")

    def test_grid_letters_for_an_even_zone_north(self):
        # Washington Monument: zone 18 (even: row letters offset by 5), band S.
        self.assertTrue(self.c.to_mgrs(38.8895, -77.0352).startswith("18SUJ"))

    def test_exception_zones(self):
        self.assertTrue(self.c.to_mgrs(60.0, 8.0).startswith("32V"))
        self.assertTrue(self.c.to_mgrs(78.0, 14.0).startswith("33X"))
        for lat, lon in ((60.0, 4.0), (78.0, 10.0)):  # > 3.5 deg off the widened meridian
            with self.assertRaises(ValueError):
                self.c.to_mgrs(lat, lon)

    def test_round_trip_everywhere(self):
        rng = np.random.default_rng(0)
        lats = np.concatenate([rng.uniform(-79.5, 83.5, 400), [0.0, -0.0001, 8.0, 71.9, 72.1]])
        lons = np.concatenate([rng.uniform(-179.9, 179.9, 400), [0.0, 3.0, 77.2, 5.5, 20.0]])
        for lat, lon in zip(lats, lons):
            if (56 <= lat < 64 and 3 <= lon < 12) or lat >= 72:
                continue  # exception zones: covered separately above
            reference = self.c.to_mgrs(lat, lon, 5)
            back_lat, back_lon, cell = self.c.from_mgrs(reference)
            self.assertEqual(cell, 1.0)
            dy = (back_lat - lat) * 111_320
            dx = (back_lon - lon) * 111_320 * math.cos(math.radians(lat))
            self.assertLess(math.hypot(dx, dy), 1.5, (lat, lon, reference))

    def test_precision_and_spacing(self):
        self.assertEqual(self.c.to_mgrs(28.6139, 77.2090, 0), "43RGM")
        spaced = self.c.to_mgrs(28.6139, 77.2090, 3, spaced=True)
        self.assertRegex(spaced, r"^43R GM \d{3} \d{3}$")
        lat, lon, cell = self.c.from_mgrs(spaced)
        self.assertEqual(cell, 100.0)
        self.assertLess(abs(lat - 28.6139) * 111_320, 80)

    def test_southern_hemisphere(self):
        reference = self.c.to_mgrs(-33.8568, 151.2153)  # Sydney
        self.assertTrue(reference.startswith("56H"))
        lat, lon, _ = self.c.from_mgrs(reference)
        self.assertAlmostEqual(lat, -33.8568, places=4)

    def test_bad_input_is_refused(self):
        for bad in ("", "43RGM123", "99RGM", "43IGM", "43RGW11"):
            with self.assertRaises(ValueError):
                self.c.from_mgrs(bad)
        with self.assertRaises(ValueError):
            self.c.to_mgrs(85.0, 0.0)
        with self.assertRaises(ValueError):
            self.c.to_mgrs(10.0, 10.0, 6)

    def test_ce90_circular_matches_closed_form(self):
        self.assertAlmostEqual(self.c.ce90(1.0, 1.0), math.sqrt(-2 * math.log(0.1)), places=4)

    def test_ce90_elongated_matches_monte_carlo(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(size=(2_000_000, 2)) * [3.0, 0.5]
        empirical = float(np.quantile(np.hypot(samples[:, 0], samples[:, 1]), 0.9))
        self.assertAlmostEqual(self.c.ce90(3.0, 0.5), empirical, delta=0.01 * empirical)
        # The circular shortcut on the mean sigma is measurably wrong here.
        self.assertGreater(abs(2.146 * 1.75 - empirical), 0.3)

    def test_le90_and_describe(self):
        self.assertAlmostEqual(self.c.le90(2.0), 3.2897, places=3)
        out = self.c.describe_point(28.6139, 77.2090, 216.3, height_datum="ellipsoidal",
                                    sigma_east_m=1.0, sigma_north_m=1.0, sigma_up_m=2.0)
        self.assertEqual(out["utm"]["epsg"], 32643)
        self.assertAlmostEqual(out["ce90_m"], 2.146, places=3)
        self.assertIn("°", out["dms"])
        bare = self.c.describe_point(28.6, 77.2, 0, height_datum="egm96")
        self.assertNotIn("ce90_m", bare)
        self.assertIn("no uncertainty", bare["basis"])

    def test_dms_rollover(self):
        self.assertEqual(self.c.to_dms(10.999999999, -0.5), "11°00'00.00\"N 0°30'00.00\"W")


if __name__ == "__main__":
    unittest.main()
