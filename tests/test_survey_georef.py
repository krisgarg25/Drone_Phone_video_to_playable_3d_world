"""CPU-only behavioral tests; input streams are in memory, never work artifacts."""
import copy
import csv
import importlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


FIELDS = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m",
          "horizontal_std_m", "vertical_std_m")
FRAME = {"type": "ENU", "units": "m", "origin": {
    "latitude_deg": 0.0, "longitude_deg": 0.0, "altitude_m": 0.0},
    "geodetic_crs": "EPSG:4979", "altitude_datum": "ellipsoidal"}
LOCAL = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [3, 2, 0],
                  [4, 5, 1], [7, 4, 2], [8, 8, 1], [12, 9, 4]], dtype=float)
ROTATION = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
TRANSLATION = np.array([40.0, -12.0, 7.0])


def metadata(**updates):
    value = dict(schema_version=1, time_reference="video", time_offset_s=0.0,
                 altitude_datum="ellipsoidal", position_reference="camera_center",
                 single_pass=True, video_duration_s=30.0)
    value.update(updates)
    return value


def gps_rows():
    return [dict(zip(FIELDS, row)) for row in (
        (0, 0, 0, 0, 0.3, 0.8), (1, 0, 0.001, 0, 0.3, 0.8),
        (2, 0.001, 0, 0, 0.3, 0.8), (3, 0, 0, 10, 0.3, 0.8))]


def cameras(points, times=None):
    if times is None:
        times = range(len(points))
    result = []
    for i, (point, time) in enumerate(zip(points, times)):
        # Deliberately nonidentity world->camera rotations catch t/center confusion.
        rotation = np.linalg.matrix_power(ROTATION, i % 3 + 1)
        result.append(dict(file=f"frame_{i}.jpg", t_sec=float(time), camera={
            "R_rowmajor": rotation.tolist(), "t": (-rotation @ point).tolist()}))
    return result


def telemetry(points, times=None):
    if times is None:
        times = range(len(points))
    return dict(schema_version=1, coordinate_frame=copy.deepcopy(FRAME), samples=[
        dict(t_sec=float(t), position=list(map(float, p)),
             horizontal_std_m=0.3, vertical_std_m=0.8)
        for t, p in zip(times, points)])


class SurveyGeorefTests(unittest.TestCase):
    def setUp(self):
        try:
            self.geo = importlib.import_module("scripts.survey_georef")
        except ModuleNotFoundError as exc:
            if exc.name != "scripts.survey_georef":
                raise
            self.fail("scripts.survey_georef has not been implemented")

    def normalize(self, rows=None, meta=None, suffix=".csv", text=None):
        rows = gps_rows() if rows is None else rows
        if text is None:
            if suffix == ".csv":
                stream = io.StringIO()
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                text = stream.getvalue()
            else:
                text = "\n".join(json.dumps(row) for row in rows)
        # Only replace filesystem reading; conversion and validation remain real.
        with patch.object(Path, "open", return_value=io.StringIO(text)):
            return self.geo.normalize_telemetry(
                Path(__file__).resolve().parent / ("in_memory" + suffix),
                metadata() if meta is None else meta)

    def known_fit(self, local=LOCAL, scale=2.5):
        target = scale * (local @ ROTATION.T) + TRANSLATION
        return cameras(local), telemetry(target), target

    def test_wgs84_origin_east_north_up_and_schema(self):
        result = self.normalize()
        self.assertEqual(set(result), {"schema_version", "coordinate_frame", "samples"})
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["coordinate_frame"], FRAME)
        np.testing.assert_allclose(result["samples"][0]["position"], [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(result["samples"][1]["position"],
                                   [111.319490788, 0, -0.000971446], atol=1e-6)
        np.testing.assert_allclose(result["samples"][2]["position"],
                                   [0, 110.574275816, -0.000964943], atol=1e-6)
        np.testing.assert_allclose(result["samples"][3]["position"], [0, 0, 10], atol=1e-9)
        self.assertEqual(set(result["samples"][0]), {
            "t_sec", "position", "horizontal_std_m", "vertical_std_m"})

    def test_jsonl_and_csv_are_equivalent(self):
        self.assertEqual(self.normalize(), self.normalize(suffix=".jsonl"))

    def test_wgs84_nonzero_origin_and_dateline(self):
        rows = gps_rows()[:2]
        rows[0].update(latitude_deg=45, longitude_deg=179.999, altitude_m=500)
        rows[1].update(latitude_deg=45, longitude_deg=-179.999, altitude_m=500)
        result = self.normalize(rows)
        self.assertEqual(result["coordinate_frame"]["origin"], {
            "latitude_deg": 45.0, "longitude_deg": 179.999, "altitude_m": 500.0})
        east, north, up = result["samples"][1]["position"]
        self.assertAlmostEqual(east, 157.706, delta=0.01)
        self.assertLess(abs(north), 0.01)
        self.assertLess(abs(up), 0.01)

    def test_metadata_missing_or_unsupported_declarations_rejected(self):
        for key in metadata():
            bad = metadata()
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.geo.validate_metadata(bad)
        for key, value in [("schema_version", 2), ("schema_version", True),
                           ("time_reference", "unix"), ("single_pass", False),
                           ("single_pass", 1), ("altitude_datum", "orthometric"),
                           ("position_reference", "antenna"),
                           ("position_reference", "camera")]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.normalize(meta=metadata(**{key: value}))

    def test_metadata_finite_positive_duration_and_offset(self):
        for key, values in [("video_duration_s", [0, -1, np.nan, np.inf, None, True]),
                            ("time_offset_s", [np.nan, np.inf, None, True])]:
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.geo.validate_metadata(metadata(**{key: value}))

    def test_offset_applied_once_and_sample_times_bounded_to_video(self):
        rows = gps_rows()
        for row in rows:
            row["t_sec"] -= 5
        result = self.normalize(rows, metadata(time_offset_s=5))
        self.assertEqual([s["t_sec"] for s in result["samples"]], [0, 1, 2, 3])
        for offset in [-1, 30]:
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.normalize(meta=metadata(time_offset_s=offset))

    def test_unordered_duplicate_and_nonfinite_telemetry_times_rejected(self):
        for value in [0, -1, np.nan, np.inf]:
            rows = gps_rows()
            rows[1]["t_sec"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.normalize(rows)

    def test_invalid_geodetic_ranges_and_uncertainties_rejected(self):
        for field, values in [("latitude_deg", [-91, 91, np.nan]),
                              ("longitude_deg", [-181, 181, np.inf]),
                              ("altitude_m", [np.nan, np.inf]),
                              ("horizontal_std_m", [0, -1, np.inf, 1e300, 1e-300]),
                              ("vertical_std_m", [0, -1, np.nan])]:
            for value in values:
                rows = gps_rows()
                rows[1][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.normalize(rows)

    def test_empty_malformed_or_wrong_fields_rejected(self):
        bad_inputs = [(".csv", ""), (".csv", ",".join(FIELDS) + "\n"),
                      (".jsonl", ""), (".jsonl", "not JSON"),
                      (".jsonl", "[]"), (".txt", "anything"),
                      (".csv", ",".join(FIELDS[:-1]) + "\n0,0,0,0,1\n"),
                      (".csv", ",".join(FIELDS) + ",t_sec\n0,0,0,0,1,1,9\n")]
        for suffix, text in bad_inputs:
            with self.subTest(suffix=suffix, text=text), self.assertRaises(ValueError):
                self.normalize(suffix=suffix, text=text)
        for row in [dict(gps_rows()[0], speed=4), dict(gps_rows()[0], t_sec=True)]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.normalize([row], suffix=".jsonl")

    def test_known_transform_positive_scales_and_axis_convention(self):
        for scale in [0.15, 2.5, 120.0]:
            rows, gps, target = self.known_fit(scale=scale)
            result = self.geo.align_camera_trajectory(rows, gps)
            self.assertAlmostEqual(result["scale"], scale, places=9)
            np.testing.assert_allclose(result["rotation"], ROTATION, atol=1e-10)
            np.testing.assert_allclose(result["translation"], TRANSLATION, atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(result["rotation"]), 1.0)
            np.testing.assert_allclose(self.geo.transform_points(LOCAL, result), target, atol=1e-9)

    def test_single_large_gps_outlier_excluded_deterministically(self):
        rows, gps, target = self.known_fit()
        gps["samples"][3]["position"] = (target[3] + [150, -60, 40]).tolist()
        first = self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=0.05)
        second = self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=0.05)
        self.assertEqual(first, second)
        self.assertEqual(first["matched_count"], 8)
        self.assertEqual(first["inlier_count"], 7)
        self.assertEqual(first["inlier_mask"], [True, True, True, False, True, True, True, True])
        self.assertGreater(first["residuals_m"][3], 160)
        self.assertLess(first["fit_rmse_m"], 1e-9)
        np.testing.assert_allclose(self.geo.transform_points(LOCAL, first), target, atol=1e-9)

    def test_million_metre_outlier_does_not_veto_valid_consensus(self):
        for side in ("source", "target"):
            rows, gps, target = self.known_fit()
            if side == "target":
                gps["samples"][3]["position"] = (target[3] + [1e6, 0, 0]).tolist()
            else:
                contaminated = LOCAL.copy()
                contaminated[3] += [1e6, 0, 0]
                rows = cameras(contaminated)
            with self.subTest(side=side):
                result = self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=.05)
                self.assertEqual(result["inlier_mask"], [True, True, True, False, True, True, True, True])
                self.assertEqual(result["inlier_count"], 7)
                np.testing.assert_allclose(self.geo.transform_points(LOCAL, result), target, atol=1e-9)
                json.dumps(result, allow_nan=False)

    def test_weighted_geometry_is_not_recentred_after_sqrt_weighting(self):
        points = np.array([[0, 0, 0], [1, .008, 0], [1, 0, 0], [1, -.008, 0]])
        weights = np.array([1, 1e-6, 1e-6, 1e-6])
        # Unweighted geometry passes, but effective second/first is only .00653.
        for source, target in ((points, LOCAL[:4]), (LOCAL[:4], points)):
            with self.subTest(source=source.tolist()), self.assertRaisesRegex(ValueError, "collinear"):
                self.geo._fit(source, target, weights)
        gps = telemetry(points)
        for sample in gps["samples"][1:]:
            sample.update(horizontal_std_m=300, vertical_std_m=800)
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(cameras(points), gps, inlier_threshold_m=.001)

    def test_report_schema_and_accuracy_honesty(self):
        rows, gps, _ = self.known_fit()
        result = self.geo.align_camera_trajectory(rows, gps)
        self.assertEqual(set(result), {"schema_version", "status", "method", "scale",
            "rotation", "translation", "coordinate_frame", "matched_count", "inlier_count",
            "matched_files", "inlier_files",
            "residuals_m", "inlier_mask", "fit_rmse_m", "warnings", "accuracy_validated"})
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "aligned")
        self.assertIs(result["accuracy_validated"], False)
        self.assertTrue(result["method"])
        self.assertEqual(result["coordinate_frame"], gps["coordinate_frame"])
        warnings = " ".join(result["warnings"]).lower()
        self.assertIn("independent", warnings)
        self.assertIn("anisotropic", warnings)
        self.assertTrue(all(type(x) is bool for x in result["inlier_mask"]))
        json.dumps(result, allow_nan=False)

    def test_independence_is_declared_by_the_caller_not_proven_here(self):
        rows, gps, _ = self.known_fit()
        warnings = " ".join(self.geo.align_camera_trajectory(rows, gps)["warnings"]).lower()
        self.assertIn("independen", warnings)
        self.assertIn("declared", warnings)
        self.assertIn("not proven", warnings)

    def test_fit_records_which_cameras_entered_and_which_stayed(self):
        rows, gps, target = self.known_fit()
        for i, row in enumerate(rows):
            row["file"] = f"cam_{i:04d}.jpg"
        gps["samples"][3]["position"] = (target[3] + [150, -60, 40]).tolist()
        result = self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=0.05)
        names = [f"cam_{i:04d}.jpg" for i in range(8)]
        self.assertEqual(result["matched_files"], names)
        self.assertEqual(result["inlier_files"], [n for i, n in enumerate(names) if i != 3])
        # The names must index the same rows as the residuals and the mask, in the
        # same order, or the record proves nothing about the fit.
        self.assertEqual(len(result["matched_files"]), result["matched_count"])
        self.assertEqual(len(result["inlier_files"]), result["inlier_count"])
        self.assertEqual(len(result["residuals_m"]), len(result["matched_files"]))
        self.assertEqual(len(result["inlier_mask"]), len(result["matched_files"]))
        self.assertEqual(result["inlier_files"],
                         [n for n, keep in zip(result["matched_files"], result["inlier_mask"]) if keep])
        json.dumps(result, allow_nan=False)

    def test_cameras_held_out_of_the_fit_are_absent_from_the_record(self):
        rows, gps, _ = self.known_fit(local=LOCAL[:4])
        for i, row in enumerate(rows):
            row["file"] = f"used_{i}.jpg"
        for row, sample in zip(rows, gps["samples"]):
            row["t_sec"] = sample["t_sec"] = row["t_sec"] * 5 + 1
        extras = cameras(np.ones((3, 3)) * 100, [0, 3, 17])
        for i, row in enumerate(extras):
            row["file"] = f"heldout_{i}.jpg"
        result = self.geo.align_camera_trajectory(rows + extras, gps)
        self.assertEqual(result["matched_files"],
                         ["used_0.jpg", "used_1.jpg", "used_2.jpg", "used_3.jpg"])
        self.assertEqual(result["inlier_files"], result["matched_files"])
        for name in ("heldout_0.jpg", "heldout_1.jpg", "heldout_2.jpg"):
            self.assertNotIn(name, " ".join(result["matched_files"]))
        self.assertEqual(result["matched_count"], 4)
        self.assertEqual(len(result["residuals_m"]), 4)

    def test_planar_noncollinear_trajectory_allowed_with_four_cameras(self):
        points = LOCAL[:4].copy()
        rows, gps, target = self.known_fit(local=points)
        result = self.geo.align_camera_trajectory(rows, gps)
        self.assertEqual(result["inlier_count"], 4)
        np.testing.assert_allclose(self.geo.transform_points(points, result), target, atol=1e-9)

    def test_source_and_target_collinear_or_near_collinear_rejected(self):
        line = np.column_stack([np.arange(8.0), np.zeros((8, 2))])
        near = line.copy()
        near[:, 1] = np.sin(np.arange(8)) * 1e-5
        for points in [line, near, np.zeros((8, 3))]:
            for side in ["source", "target"]:
                rows, gps, _ = self.known_fit()
                if side == "source":
                    rows = cameras(points)
                else:
                    gps = telemetry(points)
                with self.subTest(side=side), self.assertRaises(ValueError):
                    self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=100)

    def test_inlier_geometry_checked_after_outlier_removal(self):
        points = np.column_stack([np.arange(8.0), np.zeros((8, 2))])
        points[-1] = [1, 4, 3]
        rows, gps, target = self.known_fit(local=points)
        gps["samples"][-1]["position"] = (target[-1] + [100, 200, 300]).tolist()
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=0.001)

    def test_fewer_than_four_matches_and_insufficient_consensus_rejected(self):
        rows, gps, _ = self.known_fit()
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(rows[:3], gps)
        rng = np.random.default_rng(18)
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(rows, telemetry(rng.normal(size=(8, 3)) * 100),
                                             inlier_threshold_m=0.001)
        # Four agreeing cameras out of eight are not a conservative consensus.
        for sample in gps["samples"][4:]:
            sample["position"] = (rng.normal(size=3) * 100).tolist()
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=0.001)

    def test_reflection_not_silently_accepted_as_proper_rotation(self):
        reflected = LOCAL.copy()
        reflected[:, 0] *= -1
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(cameras(LOCAL), telemetry(reflected),
                                             inlier_threshold_m=1e-6)

    def test_interpolation_matches_bracketed_camera_centers(self):
        gps_points = LOCAL[:5]
        gps_times = np.arange(5) * 2.0
        midpoint_local = (gps_points[:-1] + gps_points[1:]) / 2
        target = 2.5 * (gps_points @ ROTATION.T) + TRANSLATION
        result = self.geo.align_camera_trajectory(
            cameras(midpoint_local, [1, 3, 5, 7]), telemetry(target, gps_times))
        self.assertEqual(result["matched_count"], 4)
        self.assertAlmostEqual(result["scale"], 2.5, places=9)
        np.testing.assert_allclose(result["translation"], TRANSLATION, atol=1e-9)

    def test_gap_limit_no_extrapolation_and_exact_times(self):
        rows, gps, _ = self.known_fit(local=LOCAL[:4])
        for row, sample in zip(rows, gps["samples"]):
            row["t_sec"] = sample["t_sec"] = row["t_sec"] * 5 + 1
        extras = cameras(np.ones((3, 3)) * 100, [0, 3, 17])
        result = self.geo.align_camera_trajectory(rows + extras, gps)
        self.assertEqual(result["matched_count"], 4)
        self.assertEqual(len(result["residuals_m"]), 4)
        self.assertIn("unmatched", " ".join(result["warnings"]).lower())
        mids = (LOCAL[:3] + LOCAL[1:4]) / 2
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(cameras(mids, [3.5, 8.5, 13.5]), gps)
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(extras, gps)

    def test_normalization_offset_flows_into_alignment(self):
        gps = self.normalize(meta=metadata(time_offset_s=5))
        points = np.array([s["position"] for s in gps["samples"]])
        result = self.geo.align_camera_trajectory(cameras(points, [5, 6, 7, 8]), gps)
        self.assertAlmostEqual(result["scale"], 1.0, places=9)
        np.testing.assert_allclose(result["translation"], [0, 0, 0], atol=1e-9)

    def test_high_variance_measurement_has_lower_scalar_weight(self):
        rows, gps, target = self.known_fit()
        gps["samples"][-1]["position"] = (target[-1] + [1, -0.5, 0.75]).tolist()
        equal = self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=5)
        gps["samples"][-1].update(horizontal_std_m=100, vertical_std_m=100)
        weighted = self.geo.align_camera_trajectory(rows, gps, inlier_threshold_m=5)
        error_equal = np.linalg.norm(self.geo.transform_points(LOCAL, equal) - target)
        error_weighted = np.linalg.norm(self.geo.transform_points(LOCAL, weighted) - target)
        self.assertLess(error_weighted, error_equal * 0.01)
        self.assertEqual(weighted["inlier_count"], 8)
        residuals = np.array(weighted["residuals_m"])
        self.assertAlmostEqual(weighted["fit_rmse_m"], np.sqrt(np.mean(residuals ** 2)))

    def test_invalid_alignment_parameters_rejected(self):
        rows, gps, _ = self.known_fit()
        for key in ["max_gap_s", "inlier_threshold_m"]:
            for value in [0, -1, np.inf, np.nan, True]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.geo.align_camera_trajectory(rows, gps, **{key: value})

    def test_invalid_camera_rotations_vectors_and_times_rejected(self):
        rows, gps, _ = self.known_fit()
        for matrix in [np.diag([-1, 1, 1]), np.eye(3) * 2,
                       np.ones((3, 3)), np.full((3, 3), np.nan), [1, 2]]:
            bad = copy.deepcopy(rows)
            bad[0]["camera"]["R_rowmajor"] = np.asarray(matrix).tolist()
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                self.geo.align_camera_trajectory(bad, gps)
        for vector in [[0, 0], [0, np.inf, 0], [np.nan, 0, 0], [[0], [0], [0]]]:
            bad = copy.deepcopy(rows)
            bad[0]["camera"]["t"] = vector
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.geo.align_camera_trajectory(bad, gps)
        for time in [None, np.nan, np.inf, -1, True]:
            bad = copy.deepcopy(rows)
            bad[0]["t_sec"] = time
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.geo.align_camera_trajectory(bad, gps)
        with self.assertRaises(ValueError):
            self.geo.align_camera_trajectory(rows + [rows[0]], gps)

    def test_malformed_normalized_telemetry_not_trusted(self):
        rows, gps, _ = self.known_fit()
        for field, value in [("position", [1, 2]), ("position", [np.inf, 0, 0]),
                             ("horizontal_std_m", 0), ("vertical_std_m", np.nan),
                             ("t_sec", -1), ("t_sec", 0)]:
            bad = copy.deepcopy(gps)
            bad["samples"][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.geo.align_camera_trajectory(rows, bad)
        for field, value in [("type", "ECEF"), ("units", "feet"),
                             ("altitude_datum", "orthometric"), ("geodetic_crs", "EPSG:4326")]:
            bad = copy.deepcopy(gps)
            bad["coordinate_frame"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.geo.align_camera_trajectory(rows, bad)
        for value in [2, True, None]:
            bad = copy.deepcopy(gps)
            bad["schema_version"] = value
            with self.subTest(version=value), self.assertRaises(ValueError):
                self.geo.align_camera_trajectory(rows, bad)

    def test_transform_points_shape_finite_and_alignment_guards(self):
        rows, gps, _ = self.known_fit()
        result = self.geo.align_camera_trajectory(rows, gps)
        self.assertEqual(self.geo.transform_points(np.empty((0, 3)), result).shape, (0, 3))
        for points in [[1, 2, 3], [[0, 0]], [[0, 0, np.nan]], [[0, np.inf, 0]]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.geo.transform_points(points, result)
        for field, value in [("scale", -1), ("scale", 0), ("scale", np.nan),
                             ("translation", [0, 0, np.inf]),
                             ("rotation", np.diag([-1, 1, 1]).tolist()),
                             ("status", "failed"), ("schema_version", 2)]:
            bad = dict(result, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.geo.transform_points(LOCAL, bad)
        bad = dict(result, scale=1e308)
        with self.assertRaises(ValueError):
            self.geo.transform_points(LOCAL, bad)

    def test_inputs_and_outputs_do_not_alias_or_mutate(self):
        rows, gps, _ = self.known_fit()
        before = copy.deepcopy((rows, gps))
        result = self.geo.align_camera_trajectory(rows, gps)
        self.assertEqual((rows, gps), before)
        points = LOCAL.copy()
        original_result = copy.deepcopy(result)
        transformed = self.geo.transform_points(points, result)
        np.testing.assert_array_equal(points, LOCAL)
        self.assertFalse(np.shares_memory(points, transformed))
        self.assertEqual(result, original_result)
        result["coordinate_frame"]["origin"]["altitude_m"] = 99
        self.assertEqual(gps, before[1])
        meta = metadata()
        original_meta = copy.deepcopy(meta)
        validated = self.geo.validate_metadata(meta)
        self.normalize(meta=meta)
        self.assertEqual(meta, original_meta)
        validated["time_offset_s"] = 22
        self.assertEqual(meta, original_meta)


if __name__ == "__main__":
    unittest.main()
