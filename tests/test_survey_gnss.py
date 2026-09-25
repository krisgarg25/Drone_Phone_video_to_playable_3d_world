"""CPU-only tests for survey_gnss: synthetic traces, analytic expectations, no claims.

Every input here is a hand-built synthetic trace. Nothing in this file measures a
real flight, and no assertion may be read as evidence of accuracy.
"""
import copy
import json
import math
import unittest

import numpy as np

from scripts import survey_gnss as gnss


def fix(t, e=0.0, n=0.0, u=0.0, h=0.4, v=0.8):
    """One normalised telemetry sample (scripts/survey_georef.py sample schema)."""
    return {"t_sec": float(t), "position": [float(e), float(n), float(u)],
            "horizontal_std_m": float(h), "vertical_std_m": float(v)}


def line(count=5, step=1.0, dt=1.0, **kwargs):
    return [fix(i * dt, e=i * step, **kwargs) for i in range(count)]


class QualityReportTests(unittest.TestCase):
    def test_displacement_speeds_and_gaps_are_reported_exactly(self):
        samples = [fix(t, e=t) for t in (0.0, 2.5, 5.0, 7.5, 10.0)]
        got = gnss.quality_report(samples, max_speed_m_s=5.0)
        self.assertEqual(got["count"], 5)
        self.assertEqual(got["duration_s"], 10.0)
        self.assertEqual(got["total_displacement_m"], 10.0)
        self.assertEqual(got["path_length_m"], 10.0)
        self.assertEqual(got["step_m"], [2.5, 2.5, 2.5, 2.5])
        self.assertEqual(got["median_step_m"], 2.5)
        self.assertEqual(got["max_step_m"], 2.5)
        self.assertEqual(got["max_step_index"], 1)
        self.assertEqual(got["implied_speed_m_s"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(got["median_implied_speed_m_s"], 1.0)
        self.assertEqual(got["max_implied_speed_m_s"], 1.0)
        self.assertEqual(got["time_gap_s"], [2.5, 2.5, 2.5, 2.5])
        self.assertEqual(got["median_gap_s"], 2.5)
        self.assertEqual(got["max_gap_s"], 2.5)
        self.assertEqual(got["mean_gap_s"], 2.5)
        self.assertEqual(got["gap_threshold_s"], 2.0)
        self.assertEqual(got["gaps_over_threshold_count"], 4)
        self.assertEqual(got["gaps_over_threshold_indices"], [1, 2, 3, 4])
        self.assertEqual(got["suspicious"], [])
        self.assertEqual(got["suspicious_count"], 0)
        self.assertEqual(got["median_horizontal_std_m"], 0.4)
        self.assertEqual(got["median_vertical_std_m"], 0.8)
        self.assertIs(got["displacement_below_noise"], False)
        self.assertIs(got["speeds_are_implied_not_measured"], True)
        self.assertIs(got["accuracy_validated"], False)
        json.dumps(got, allow_nan=False)

    def test_a_40_m_s_implied_speed_from_a_5_m_s_drone_is_flagged_not_motion(self):
        samples = line(5, step=1.0, dt=1.0)
        samples[4]["position"] = [83.0, 0.0, 0.0]
        got = gnss.quality_report(samples, max_speed_m_s=5.0)
        self.assertEqual(got["suspicious"], [4])
        self.assertEqual(got["suspicious_count"], 1)
        self.assertEqual(got["max_step_index"], 4)
        self.assertEqual(got["max_step_m"], 80.0)
        self.assertEqual(got["total_displacement_m"], 83.0)
        # The median stays honest about the 5 m/s flight even with a wild fix.
        self.assertEqual(got["median_step_m"], 1.0)
        self.assertAlmostEqual(got["mean_step_m"], 20.75)
        detail = got["suspicious_detail"][0]
        self.assertEqual(detail, {"index": 4, "interval": 3, "t_sec": 3.0, "dt_s": 1.0,
                                  "distance_m": 80.0, "implied_speed_m_s": 80.0})
        self.assertTrue(any("noise" in w.lower() for w in got["warnings"]))

    def test_every_index_names_the_later_fix_of_the_interval(self):
        samples = line(6, step=1.0, dt=1.0)
        samples[2]["position"] = [50.0, 0.0, 0.0]
        samples[3]["position"] = [51.0, 0.0, 0.0]
        got = gnss.quality_report(samples, max_speed_m_s=5.0)
        self.assertEqual(sorted(got["suspicious"]), [2, 4])
        self.assertEqual(len(got["implied_speed_m_s"]), 5)
        self.assertEqual(len(got["step_m"]), 5)

    def test_displacement_below_reported_noise_is_a_quality_signal(self):
        samples = [fix(i * 1.0, e=i * 0.05, h=3.0, v=6.0) for i in range(6)]
        got = gnss.quality_report(samples, max_speed_m_s=5.0)
        self.assertIs(got["displacement_below_noise"], True)
        self.assertAlmostEqual(got["reported_std_over_median_step"], 60.0, delta=1e-9)
        self.assertTrue(any("noise" in w.lower() for w in got["warnings"]))

    def test_declared_gap_threshold_and_physical_limit_are_caller_supplied(self):
        samples = line(4, step=10.0, dt=2.0)
        self.assertEqual(gnss.quality_report(samples, max_speed_m_s=5.0,
                                            gap_threshold_s=2.0)["gaps_over_threshold_count"], 0)
        self.assertEqual(gnss.quality_report(samples, max_speed_m_s=5.0,
                                            gap_threshold_s=1.0)["gaps_over_threshold_count"], 3)
        with self.assertRaises(TypeError):
            gnss.quality_report(samples)

    def test_single_fix_reports_no_motion_instead_of_inventing_speed(self):
        got = gnss.quality_report([fix(0.0, e=1.0)], max_speed_m_s=5.0)
        self.assertEqual(got["count"], 1)
        self.assertEqual(got["step_m"], [])
        self.assertIsNone(got["median_step_m"])
        self.assertIsNone(got["median_implied_speed_m_s"])
        self.assertIsNone(got["reported_std_over_median_step"])
        self.assertIs(got["displacement_below_noise"], False)
        self.assertTrue(any("single" in w.lower() for w in got["warnings"]))

    def test_inputs_are_never_modified(self):
        samples = line(4)
        before = copy.deepcopy(samples)
        gnss.quality_report(samples, max_speed_m_s=5.0)
        self.assertEqual(samples, before)

    def test_invalid_telemetry_and_parameters_raise(self):
        bad_traces = ([], [{}], [{"t_sec": 0, "position": [0, 0, 0]}],
                      [{"t_sec": 0, "position": [0, 0], "horizontal_std_m": 1, "vertical_std_m": 1}],
                      [fix(0.0), fix(0.0)], [fix(1.0), fix(0.5)],
                      [fix(0.0, e=np.nan)], [fix(0.0, h=0.0)], [fix(0.0, v=-1.0)],
                      [fix(0.0), dict(fix(1.0), extra=1)], "not a list",
                      [fix(0.0, h=np.inf)])
        for trace in bad_traces:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                gnss.quality_report(trace, max_speed_m_s=5.0)
        for limit in (0, -5, np.nan, np.inf, "fast", True, None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                gnss.quality_report(line(4), max_speed_m_s=limit)
        for gap in (0, -1, np.nan, True):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                gnss.quality_report(line(4), max_speed_m_s=5.0, gap_threshold_s=gap)


class VerticalReferenceTests(unittest.TestCase):
    @staticmethod
    def spread(count=6, extent=100.0, relief=6.0, h=None, v=None, u_base=100.0):
        """A trace with real vertical variation and per-fix varying uncertainties."""
        out = []
        for i in range(count):
            fraction = i / max(1, count - 1)
            out.append(fix(i * 2.0, e=fraction * extent,
                           u=u_base + relief * math.sin(math.pi * fraction),
                           h=0.4 + 0.02 * i, v=0.8 + 0.05 * i))
        return out

    def test_agl_or_barometric_source_in_an_ellipsoidal_slot_is_detected(self):
        declared = {"altitude_datum": "ellipsoidal", "height_reference": "above_ground_level",
                    "source": "barometric"}
        got = gnss.vertical_reference_check(self.spread(), declared)
        self.assertEqual(got["status"], "inconsistency_detected")
        self.assertIs(got["signals"]["declared_reference_mismatch"]["detected"], True)
        self.assertIn("declared_reference_mismatch", got["detected_signals"])
        self.assertIs(got["requires_confirmation"], True)
        self.assertTrue(got["confirmation_actions"])
        self.assertIs(got["vertical_datum_verified"], False)
        self.assertNotIn("ellipsoidal", " ".join(got["detected_signals"]).lower())

    def test_a_height_that_never_moves_while_covering_500_m_is_a_signal(self):
        samples = [fix(i * 2.0, e=i * 100.0, u=97.5, h=0.4 + 0.01 * i, v=0.8 + 0.01 * i)
                   for i in range(6)]
        declared = {"altitude_datum": "ellipsoidal", "height_reference": "unknown",
                    "source": "unknown", "expected_relief_m": 40.0}
        got = gnss.vertical_reference_check(samples, declared)
        self.assertIs(got["signals"]["vertical_series_invariant"]["detected"], True)
        self.assertIs(got["signals"]["vertical_relief_inconsistent_with_extent"]["detected"], True)
        # An undeclared source can never be cleared, only flagged.
        self.assertIs(got["status"], "inconsistency_detected")
        self.assertIs(got["requires_confirmation"], True)

    def test_a_second_vertical_source_offset_by_a_constant_sub_centimetre_amount_is_flagged(self):
        samples = self.spread()
        declared = {"altitude_datum": "ellipsoidal", "height_reference": "unknown",
                    "source": "unknown",
                    "alternative_altitudes_m": [s["position"][2] - 42.0 for s in samples]}
        got = gnss.vertical_reference_check(samples, declared)
        signal = got["signals"]["constant_offset_below_centimetre"]
        self.assertIs(signal["detected"], True)
        self.assertAlmostEqual(signal["evidence"]["mean_difference_m"], 42.0)
        self.assertLess(signal["evidence"]["difference_spread_m"], 1e-12)

    def test_a_second_source_that_tracks_terrain_is_not_a_constant_offset(self):
        samples = self.spread()
        declared = {"altitude_datum": "ellipsoidal", "height_reference": "unknown",
                    "source": "unknown",
                    "alternative_altitudes_m": [40.0 + 0.5 * s["position"][2] for s in samples]}
        got = gnss.vertical_reference_check(samples, declared)
        self.assertIs(got["signals"]["constant_offset_below_centimetre"]["detected"], False)

    def test_static_reported_uncertainties_are_flagged_as_placeholders(self):
        samples = [fix(i * 2.0, e=i * 50.0, u=100.0 + i * 1.3) for i in range(6)]
        got = gnss.vertical_reference_check(samples, {"altitude_datum": "ellipsoidal",
                                                      "height_reference": "unknown",
                                                      "source": "unknown"})
        self.assertIs(got["signals"]["static_reported_uncertainty"]["detected"], True)

    def test_a_consistent_trace_is_reported_without_being_declared_correct(self):
        declared = {"altitude_datum": "ellipsoidal", "source": "gnss",
                    "height_reference": "altitude_above_reference_ellipsoid",
                    "expected_relief_m": 6.0}
        got = gnss.vertical_reference_check(self.spread(relief=6.0), declared)
        self.assertEqual(got["status"], "no_inconsistency_signal")
        self.assertIs(got["requires_confirmation"], False)
        self.assertEqual(got["detected_signals"], [])
        self.assertIs(got["vertical_datum_verified"], False)
        self.assertIs(got["signals_absent_prove_consistency"], False)
        self.assertIs(got["guessed"], False)
        json.dumps(got, allow_nan=False)

    def test_the_function_never_rewrites_altitudes_and_never_guesses_the_truth(self):
        samples = [fix(0.0, e=0.0, u=0.0), fix(1.0, e=500.0, u=0.0)]
        declared = {"altitude_datum": "orthometric", "height_reference": "barometric",
                    "source": "barometric"}
        got = gnss.vertical_reference_check(samples, declared)
        self.assertEqual([s["position"][2] for s in samples], [0.0, 0.0])
        self.assertEqual(got["declared"]["altitude_datum"], "orthometric")
        self.assertNotIn("corrected_altitudes_m", got)
        self.assertNotIn("recommended_altitude_datum", got)
        self.assertIs(got["vertical_datum_verified"], False)

    def test_invalid_declarations_and_altitude_series_raise(self):
        samples = self.spread()
        for declared in (None, "ellipsoidal", {}, {"altitude_datum": ""},
                         {"altitude_datum": 42},
                         {"altitude_datum": "ellipsoidal", "expected_relief_m": -1},
                         {"altitude_datum": "ellipsoidal", "expected_relief_m": np.nan},
                         {"altitude_datum": "ellipsoidal", "source": 3},
                         {"altitude_datum": "ellipsoidal", "alternative_altitudes_m": [1.0]},
                         {"altitude_datum": "ellipsoidal", "alternative_altitudes_m": [[1.0]] * 6},
                         {"altitude_datum": "ellipsoidal", "alternative_altitudes_m": None}):
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                gnss.vertical_reference_check(samples, declared)
        with self.assertRaises(ValueError):
            gnss.vertical_reference_check([], {"altitude_datum": "ellipsoidal"})


class TimeOffsetBoundsTests(unittest.TestCase):
    def test_coverage_is_the_only_honest_bound_and_it_is_wide(self):
        got = gnss.time_offset_bounds([0.0, 1.0, 10.0], [0.0, 4.0, 20.0], max_speed_m_s=5.0)
        self.assertEqual(got["status"], "bounded")
        self.assertEqual(got["offset_bounds_s"], [-0.0, 10.0])
        self.assertEqual(got["offset_width_s"], 10.0)
        self.assertEqual(got["camera_span_s"], [0.0, 10.0])
        self.assertEqual(got["sample_span_s"], [0.0, 20.0])
        self.assertEqual(got["convention"], "t_gnss = t_camera + offset_s")
        self.assertEqual(got["worst_case_along_track_error_m"], 50.0)
        self.assertIs(got["estimates_offset"], False)
        self.assertIsNone(got["offset_estimate_s"])
        json.dumps(got, allow_nan=False)

    def test_a_centred_camera_span_gives_a_symmetric_interval(self):
        got = gnss.time_offset_bounds([5.0, 10.0, 15.0], [0.0, 10.0, 20.0], max_speed_m_s=2.0)
        self.assertEqual(got["offset_bounds_s"], [-5.0, 5.0])
        self.assertEqual(got["worst_case_along_track_error_m"], 20.0)

    def test_metric_consequence_scales_with_the_supplied_physical_limit(self):
        slow = gnss.time_offset_bounds([5.0, 15.0], [0.0, 20.0], max_speed_m_s=1.0)
        fast = gnss.time_offset_bounds([5.0, 15.0], [0.0, 20.0], max_speed_m_s=20.0)
        self.assertEqual(slow["offset_bounds_s"], fast["offset_bounds_s"])
        self.assertEqual(fast["worst_case_along_track_error_m"],
                         20.0 * slow["worst_case_along_track_error_m"])

    def test_cameras_spanning_longer_than_the_gnss_log_are_infeasible_not_silently_clipped(self):
        got = gnss.time_offset_bounds(list(np.arange(0.0, 31.0, 1.0)), [0.0, 10.0, 20.0],
                                     max_speed_m_s=5.0)
        self.assertEqual(got["status"], "infeasible")
        self.assertIsNone(got["offset_bounds_s"])
        self.assertEqual(got["infeasibility_s"], 10.0)
        self.assertIsNone(got["worst_case_along_track_error_m"])
        self.assertTrue(any("no offset" in w.lower() for w in got["warnings"]))

    def test_invalid_time_series_and_speeds_raise(self):
        good = [0.0, 1.0, 2.0]
        for camera, samples in (([], good), (good, []),
                                ([0.0, 2.0, 1.0], good), (good, [0.0, 0.0, 1.0]),
                                ([0.0, np.nan], good), (good, [np.inf]),
                                ([0.0, 1.0], "x"), ([True, 2.0], good)):
            with self.subTest(camera=camera, samples=samples), self.assertRaises(ValueError):
                gnss.time_offset_bounds(camera, samples, max_speed_m_s=5.0)
        for limit in (0, -1, np.nan, "5", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                gnss.time_offset_bounds(good, [0.0, 5.0], max_speed_m_s=limit)


class FixQualityWeightTests(unittest.TestCase):
    def test_documented_table_is_applied_per_code(self):
        codes = ["rtk_fixed", "rtk_float", "single", "no_fix"]
        got = gnss.fix_quality_weights(codes)
        table = gnss.FIX_QUALITY_TABLE
        self.assertEqual(got["codes"], ["rtk_fixed", "rtk_float", "single", "no_fix"])
        self.assertEqual(got["horizontal_std_m"], [table["rtk_fixed"]["horizontal_std_m"],
                                                   table["rtk_float"]["horizontal_std_m"],
                                                   table["single"]["horizontal_std_m"], None])
        self.assertEqual(got["vertical_std_m"], [table["rtk_fixed"]["vertical_std_m"],
                                                 table["rtk_float"]["vertical_std_m"],
                                                 table["single"]["vertical_std_m"], None])
        self.assertEqual(got["unusable_indices"], [3])
        self.assertEqual(got["unresolved_indices"], [])
        self.assertIs(got["all_rows_resolved"], True)
        self.assertIs(got["all_rows_usable"], False)
        self.assertEqual(got["status"], "mapped")

    def test_a_fixed_solution_is_never_better_than_the_conservative_floor(self):
        table = gnss.FIX_QUALITY_TABLE
        self.assertGreaterEqual(table["rtk_fixed"]["horizontal_std_m"], 0.02)
        self.assertGreater(table["rtk_fixed"]["vertical_std_m"],
                           table["rtk_fixed"]["horizontal_std_m"])
        self.assertGreater(table["rtk_float"]["horizontal_std_m"],
                           table["rtk_fixed"]["horizontal_std_m"])
        self.assertGreater(table["single"]["horizontal_std_m"],
                           table["rtk_float"]["horizontal_std_m"])
        self.assertIs(gnss.fix_quality_weights([4])["measured_on_this_hardware"], False)

    def test_gga_numeric_codes_and_spelling_variants_normalise(self):
        got = gnss.fix_quality_weights([0, 1, 2, 4, 5, "RTK Fixed", "rtk-float", " 3D_fix "])
        self.assertEqual(got["codes"], ["no_fix", "single", "dgps", "rtk_fixed", "rtk_float",
                                        "rtk_fixed", "rtk_float", "single"])

    def test_hdop_inflates_only_it_never_buys_back_precision(self):
        base = gnss.fix_quality_weights(["single"])["horizontal_std_m"][0]
        poor = gnss.fix_quality_weights(["single", "single"], hdop=[6.0, 0.4])
        self.assertEqual(poor["dop_factor"], [6.0, 1.0])
        self.assertEqual(poor["horizontal_std_m"], [base * 6.0, base])
        self.assertTrue(any("one-sided" in w.lower() or "never" in w.lower()
                            for w in poor["warnings"]))

    def test_hdop_absent_is_reported_as_absent_not_as_good_geometry(self):
        got = gnss.fix_quality_weights(["single", "single"], hdop=[2.0, None])
        self.assertEqual(got["dop_missing_indices"], [1])
        self.assertEqual(got["dop_factor"], [2.0, 1.0])
        self.assertTrue(any("hdop" in w.lower() for w in got["warnings"]))

    def test_no_quality_field_is_refused_rather_than_inventing_precision(self):
        got = gnss.fix_quality_weights([None, "", "unknown"])
        self.assertEqual(got["status"], "refused_no_quality_field")
        self.assertEqual(got["horizontal_std_m"], [None, None, None])
        self.assertEqual(got["vertical_std_m"], [None, None, None])
        self.assertEqual(got["unresolved_indices"], [0, 1, 2])
        self.assertIs(got["precision_invented"], False)
        self.assertIsNone(got["fallback"])
        self.assertTrue(got["warnings"])

    def test_mixed_codes_leave_the_unknown_ones_unresolved(self):
        got = gnss.fix_quality_weights(["single", "banana"])
        self.assertEqual(got["status"], "partially_unresolved")
        self.assertEqual(got["unresolved_indices"], [1])
        self.assertIs(got["all_rows_resolved"], False)
        self.assertIsNone(got["horizontal_std_m"][1])
        self.assertIsNotNone(got["horizontal_std_m"][0])

    def test_an_explicit_conservative_fallback_is_labelled_as_a_fallback(self):
        got = gnss.fix_quality_weights([None, "single"], allow_unknown_fallback=True)
        table = gnss.FIX_QUALITY_TABLE
        self.assertEqual(got["horizontal_std_m"], [table["single"]["horizontal_std_m"],
                                                   table["single"]["horizontal_std_m"]])
        self.assertEqual(got["fallback"], "single")
        self.assertIs(got["precision_invented"], False)
        self.assertTrue(any("fallback" in w.lower() for w in got["warnings"]))

    def test_invalid_code_and_dop_inputs_raise(self):
        for codes in ([], "single", [object()], [[1]], [True]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                gnss.fix_quality_weights(codes)
        for hdop in ([1.0], [0.0], [-1.0], [np.nan], ["2"], [True], 2.0):
            with self.subTest(hdop=hdop), self.assertRaises(ValueError):
                gnss.fix_quality_weights(["single", "single"], hdop=hdop)


class ObservabilityTests(unittest.TestCase):
    @staticmethod
    def straight():
        camera = [[i * 10.0, 0.0, 0.0] for i in range(8)]
        samples = [fix(i * 2.0, e=i * 10.0) for i in range(8)]
        return camera, samples

    def test_straight_constant_speed_aliasses_clock_offset_with_lever_arm(self):
        camera, samples = self.straight()
        got = gnss.observability(camera, samples)
        self.assertEqual(got["status"], "evaluated")
        self.assertEqual(got["geometry"], "collinear")
        self.assertIs(got["clock_offset_and_lever_arm_separable"], False)
        ident = got["identifiability"]
        self.assertIs(ident["clock_offset"], False)
        self.assertIs(ident["lever_arm_along_track"], False)
        self.assertIs(ident["lever_arm_cross_track"], False)
        self.assertIs(ident["rotation_about_trajectory_axis"], False)
        self.assertIs(got["georef_rejects_this_geometry"], True)
        self.assertLess(got["geometry_stats"]["second_to_first_ratio"],
                        gnss.MIN_SECOND_RATIO)
        joined = " ".join(got["never_claim"]).lower()
        self.assertIn("clock offset", joined)
        self.assertIn("lever arm", joined)
        self.assertIs(got["estimates_anything"], False)
        self.assertIs(got["accuracy_validated"], False)
        json.dumps(got, allow_nan=False)

    def test_curved_varying_speed_trajectory_supports_separation(self):
        times = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0]
        camera = [[t, 0.05 * t * t, 0.02 * t * t] for t in times]
        samples = [fix(t, e=p[0], n=p[1], u=p[2]) for t, p in zip(times, camera)]
        got = gnss.observability(camera, samples)
        self.assertNotEqual(got["geometry"], "collinear")
        self.assertIs(got["clock_offset_and_lever_arm_separable"], True)
        self.assertIs(got["identifiability"]["clock_offset"], True)
        self.assertIs(got["identifiability"]["lever_arm_cross_track"], True)
        self.assertIs(got["georef_rejects_this_geometry"], False)
        self.assertEqual(got["never_claim"], [])

    def test_noise_larger_than_inter_fix_motion_cannot_transfer_metric_scale(self):
        camera, _ = self.straight()
        samples = [fix(i * 2.0, e=i * 10.0, h=25.0, v=40.0) for i in range(8)]
        got = gnss.observability(camera, samples)
        self.assertEqual(got["metric_scale_support"]["status"], "not_supported")
        self.assertIs(got["identifiability"]["metric_scale"], False)

    def test_a_clean_spread_flight_supports_scale_without_proving_accuracy(self):
        camera, samples = self.straight()
        got = gnss.observability(camera, samples)
        self.assertEqual(got["metric_scale_support"]["status"], "supported")
        self.assertIs(got["identifiability"]["metric_scale"], True)
        self.assertIn("scale", " ".join(got["supported_statements"]).lower())

    def test_fewer_than_four_camera_centres_is_insufficient_not_degenerate(self):
        camera, samples = self.straight()
        got = gnss.observability(camera[:3], samples)
        self.assertEqual(got["status"], "insufficient_data")
        self.assertIsNone(got["geometry"])
        self.assertIs(got["clock_offset_and_lever_arm_separable"], False)
        self.assertTrue(all(v is False for v in got["identifiability"].values()))
        self.assertTrue(got["never_claim"])

    def test_planar_trajectory_is_reported_as_planar(self):
        times = [0.0, 1.0, 2.5, 4.0, 6.0, 8.5]
        camera = [[t, 0.2 * t * t, 0.0] for t in times]
        samples = [fix(t, e=p[0], n=p[1]) for t, p in zip(times, camera)]
        got = gnss.observability(camera, samples)
        self.assertEqual(got["geometry"], "planar")
        self.assertIs(got["identifiability"]["rotation_about_trajectory_axis"], True)
        self.assertLess(got["geometry_stats"]["third_to_first_ratio"], 1e-9)

    def test_invalid_geometry_inputs_raise(self):
        _, samples = self.straight()
        for camera in ([], [[0, 0]], [[0, 0, np.nan]], [[0, 0, 0]] * 2 + [[1, 1]],
                       "x", np.zeros((5, 4))):
            with self.subTest(camera=camera), self.assertRaises(ValueError):
                gnss.observability(camera, samples)
        for bad in ([], "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gnss.observability([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 0, 2]], bad)

    def test_one_sample_cannot_assess_motion_and_says_so(self):
        camera, _ = self.straight()
        got = gnss.observability(camera, [fix(0.0, e=0.0)])
        self.assertEqual(got["status"], "insufficient_data")
        self.assertIs(got["motion_stats"]["constant_speed"], None)


if __name__ == "__main__":
    unittest.main()
