"""Tests for scripts/survey_crs.py: the inverse of survey_georef, plus UTM forward.

No pyproj/GDAL is available in this repository, so the projection is verified
against properties that must hold for a transverse Mercator expansion rather than
against a table of published coordinates.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import survey_crs as crs
from scripts import survey_formats as formats
from scripts import survey_georef as georef

TELEMETRY = ("t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
             "0,28.612900,77.229500,231.4,1.2,2.4\n"
             "1,28.612990,77.229560,232.0,1.2,2.4\n"
             "2,28.613080,77.229620,232.6,1.4,2.8\n"
             "3,28.613170,77.229680,233.2,1.4,2.8\n")
METADATA = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
            "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
            "single_pass": True, "video_duration_s": 4}


class GeodeticRoundTripTests(unittest.TestCase):
    def test_ecef_round_trip_is_exact_to_float_precision(self):
        points = np.array([[28.6129, 77.2295, 231.4], [49.0, 3.0, 100.0],
                           [-33.8688, 151.2093, 58.0], [64.1, -21.9, 12.0],
                           [90.0, 0.0, 0.0], [80.0, 12.0, 500.0]])
        ecef = crs.ecef_from_geodetic(points[:, 0], points[:, 1], points[:, 2])
        back = crs.geodetic_from_ecef(ecef)
        np.testing.assert_allclose(back, points, atol=1e-8, rtol=0)

    def test_enu_inverse_reproduces_survey_georef_forward(self):
        """The point of this module: ENU produced by survey_georef must invert to its own input."""
        import csv
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            path.write_text(TELEMETRY)
            telemetry = georef.normalize_telemetry(path, METADATA)
        origin = telemetry["coordinate_frame"]["origin"]
        enu = np.array([sample["position"] for sample in telemetry["samples"]])
        recovered = crs.enu_to_geodetic(enu, origin["latitude_deg"], origin["longitude_deg"],
                                        origin["altitude_m"])
        truth = np.array([[float(row["latitude_deg"]), float(row["longitude_deg"]),
                           float(row["altitude_m"])]
                          for row in csv.DictReader(TELEMETRY.splitlines())])
        # 1e-7 deg is ~1 cm; the samples are the origin plus a few tens of metres,
        # so this is the transform inverting itself, not a tolerance being stretched.
        np.testing.assert_allclose(recovered, truth, atol=1e-7)

    def test_origin_maps_to_the_origin(self):
        recovered = crs.enu_to_geodetic([[0.0, 0.0, 0.0]], 28.6129, 77.2295, 231.4)
        np.testing.assert_allclose(recovered, [[28.6129, 77.2295, 231.4]], atol=1e-9)

    def test_local_enu_offsets_survive_the_geodetic_round_trip(self):
        """A 100 m north step must still be 100 m apart on the ellipsoid."""
        origin = (28.6129, 77.2295, 231.4)
        a = crs.enu_to_geodetic([[0.0, 0.0, 0.0]], *origin)
        b = crs.enu_to_geodetic([[0.0, 100.0, 0.0]], *origin)
        ea = crs.ecef_from_geodetic(a[:, 0], a[:, 1], a[:, 2])[0]
        eb = crs.ecef_from_geodetic(b[:, 0], b[:, 1], b[:, 2])[0]
        self.assertAlmostEqual(float(np.linalg.norm(eb - ea)), 100.0, places=6)

    def test_utm_and_local_enu_metres_disagree_by_the_map_scale_factor(self):
        """The fact a reviewer needs: UTM metres are not tangent-plane metres.

        ENU is true-scale at its own origin; the transverse Mercator grid carries
        k0 and grows away from the central meridian. Over a 100 m step that is a
        difference of millimetres here, and it is why an accuracy figure has to
        name the CRS it was measured in.
        """
        origin = (28.6129, 77.2295, 231.4)
        a = crs.enu_to_geodetic([[0.0, 0.0, 0.0]], *origin)
        b = crs.enu_to_geodetic([[0.0, 100.0, 0.0]], *origin)
        zone = crs.utm_zone_for(*origin[:2])[0]
        e1, n1 = crs.utm_forward(a[:, 0], a[:, 1], zone)
        e2, n2 = crs.utm_forward(b[:, 0], b[:, 1], zone)
        grid = math.hypot(float(e2[0] - e1[0]), float(n2[0] - n1[0]))
        self.assertLess(abs(grid - 100.0) / 100.0, 2e-3)
        self.assertNotAlmostEqual(grid, 100.0, places=4)


class UtmZoneTests(unittest.TestCase):
    def test_zone_boundaries_and_epsg(self):
        self.assertEqual(crs.utm_zone_for(28.6, 77.2), (43, "N", 32643))
        self.assertEqual(crs.utm_zone_for(-33.8, 151.2), (56, "S", 32756))
        self.assertEqual(crs.utm_zone_for(0.0, -3.0), (30, "N", 32630))
        self.assertEqual(crs.utm_zone_for(0.0, 3.0), (31, "N", 32631))

    def test_norway_and_svalbard_exceptions(self):
        self.assertEqual(crs.utm_zone_for(58.0, 8.0)[0], 32)
        self.assertEqual(crs.utm_zone_for(78.0, 4.0)[0], 31)
        self.assertEqual(crs.utm_zone_for(78.0, 15.0)[0], 33)
        self.assertEqual(crs.utm_zone_for(78.0, 26.0)[0], 35)
        self.assertEqual(crs.utm_zone_for(78.0, 36.0)[0], 37)

    def test_poles_are_refused_not_invented(self):
        with self.assertRaisesRegex(ValueError, "UPS"):
            crs.utm_zone_for(86.0, 0.0)


class UtmProjectionTests(unittest.TestCase):
    def test_central_meridian_is_exactly_false_easting(self):
        for lat in (-60.0, -3.0, 0.0, 28.6, 79.9):
            zone = crs.utm_zone_for(lat, 3.0)[0]
            easting, _ = crs.utm_forward([lat], [3.0], zone)
            self.assertAlmostEqual(float(easting[0]), 500000.0, places=6)

    def test_equator_on_the_central_meridian_is_zero_northing(self):
        _, northing = crs.utm_forward([0.0], [3.0], 31)
        self.assertAlmostEqual(float(northing[0]), 0.0, places=6)

    def test_southern_hemisphere_gets_the_false_northing(self):
        _, north_n = crs.utm_forward([30.0], [3.0], 31)
        _, north_s = crs.utm_forward([-30.0], [3.0], 31)
        self.assertAlmostEqual(float(north_s[0]) + float(north_n[0]), 10000000.0, places=6)

    def test_scale_on_the_central_meridian_is_the_nominal_k0(self):
        """dnorthing over the ellipsoid's meridional arc must equal the nominal k0."""
        zone, lat = 43, 28.6129
        step_deg, step = 1e-5, math.radians(1e-5)
        _, n_hi = crs.utm_forward([lat + step_deg], [75.0], zone)
        _, n_lo = crs.utm_forward([lat - step_deg], [75.0], zone)
        phi = math.radians(lat)
        arc = crs.WGS84_A * ((1 - crs.WGS84_E2 / 4 - 3 * crs.WGS84_E2 ** 2 / 64
                              - 5 * crs.WGS84_E2 ** 3 / 256) * (2 * step)
                             - (3 * crs.WGS84_E2 / 8 + 3 * crs.WGS84_E2 ** 2 / 32
                                + 45 * crs.WGS84_E2 ** 3 / 1024)
                             * (math.sin(2 * (phi + step)) - math.sin(2 * (phi - step)))
                             + (15 * crs.WGS84_E2 ** 2 / 256 + 45 * crs.WGS84_E2 ** 3 / 1024)
                             * (math.sin(4 * (phi + step)) - math.sin(4 * (phi - step)))
                             - (35 * crs.WGS84_E2 ** 3 / 3072)
                             * (math.sin(6 * (phi + step)) - math.sin(6 * (phi - step))))
        self.assertAlmostEqual(float(n_hi[0] - n_lo[0]) / arc, crs.K0, places=8)

    def test_forward_inverse_round_trip(self):
        for lat, lon in ((28.6129, 77.2), (49.0, 3.0), (-33.8, 151.2), (65.0, 10.0)):
            zone, hemisphere, _ = crs.utm_zone_for(lat, lon)
            e, n = crs.utm_forward([lat], [lon], zone)
            back_lat, back_lon = crs.utm_inverse(e, n, zone, hemisphere)
            # Both series are truncated at 6th order, so state the closure in
            # metres on the ground rather than in degrees of latitude.
            e2, n2 = crs.utm_forward(back_lat, back_lon, zone)
            self.assertLess(float(math.hypot(e2[0] - e[0], n2[0] - n[0])), 0.001)

    def test_a_point_outside_the_zone_is_refused(self):
        with self.assertRaisesRegex(ValueError, "validity of this series"):
            crs.utm_forward([28.6], [77.2 + 4.0], 43)


class CrsPayloadTests(unittest.TestCase):
    def test_wkt_is_what_the_format_writers_accept(self):
        info = crs.crs_from_origin(28.6129, 77.2295, 231.4)
        formats._check_wkt(info["wkt"])
        self.assertEqual(formats._crs_kind(info["wkt"]), "projected")
        self.assertEqual(formats._epsg_from_wkt(info["wkt"]), 32643)
        self.assertEqual(info["epsg"], 32643)

    def test_origin_projects_to_the_reported_coordinates(self):
        info = crs.crs_from_origin(28.6129, 77.2295, 231.4)
        alignment = {"coordinate_frame": {"geodetic_crs": "EPSG:4979",
                                          "altitude_datum": "ellipsoidal",
                                          "origin": {"latitude_deg": 28.6129,
                                                     "longitude_deg": 77.2295,
                                                     "altitude_m": 231.4}}}
        utm, geodetic, _ = crs.enu_to_crs([[0.0, 0.0, 0.0]], alignment, crs=info)
        self.assertAlmostEqual(float(utm[0, 0]), info["origin_easting_m"], places=6)
        self.assertAlmostEqual(float(utm[0, 1]), info["origin_northing_m"], places=6)
        self.assertAlmostEqual(float(utm[0, 2]), 231.4, places=6)
        np.testing.assert_allclose(geodetic[0], [28.6129, 77.2295, 231.4], atol=1e-9)

    def test_a_non_wgs84_or_non_ellipsoidal_frame_is_refused(self):
        for frame in ({"geodetic_crs": "EPSG:4326", "altitude_datum": "ellipsoidal",
                       "origin": {"latitude_deg": 1.0, "longitude_deg": 1.0}},
                      {"geodetic_crs": "EPSG:4979", "altitude_datum": "msl",
                       "origin": {"latitude_deg": 1.0, "longitude_deg": 1.0}},
                      {"geodetic_crs": "EPSG:4979", "altitude_datum": "ellipsoidal"}):
            with self.assertRaises(ValueError):
                crs.crs_from_alignment({"coordinate_frame": frame})

    def test_heights_stay_ellipsoidal_and_say_so(self):
        info = crs.crs_from_origin(28.6129, 77.2295, 231.4)
        self.assertIn("ellipsoidal", info["height_datum"])
        self.assertIn("orthometric", info["validity"])


if __name__ == "__main__":
    unittest.main()
