"""Tests for scripts/survey_assess.py: the run's own honest self-reporting.

Fixtures, not evidence. These prove the wrappers call the underlying modules
correctly and preserve every "not measured" label; they say nothing about a real
scene's completeness, timing or accuracy.
"""
import unittest

import numpy as np

from scripts import survey_assess as assess
from scripts import survey_streaming as streaming


def arc(count=40, radius=60.0, height=40.0):
    t = np.linspace(0, np.pi, count)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t),
                            np.full(count, height)])


def cloud(count=800, seed=4):
    rng = np.random.default_rng(seed)
    return rng.uniform([-60, -60, 0], [60, 60, 12], (count, 3))


SAMPLES = [{"t_sec": float(i), "position": [i * 2.0, 0.0, 40.0],
            "horizontal_std_m": 1.5, "vertical_std_m": 3.0} for i in range(30)]


class CompletenessTests(unittest.TestCase):
    def test_region_observability_is_reported_with_its_limits_attached(self):
        result = assess.completeness(arc(), cloud(), cell_size_m=5.0)
        self.assertEqual(result["point_count_assessed"], 800)
        self.assertFalse(result["decimated"])
        self.assertIn(result["trajectory"]["verdict"],
                      ("observable", "weak", "degenerate"))
        self.assertGreater(result["regions"]["cell_count"], 0)
        self.assertTrue(result["not_measured"], "the not-measured list must survive wrapping")
        self.assertIn("ceiling", result["interpretation"])
        self.assertIsNone(result["layers"])

    def test_a_large_cloud_is_decimated_and_says_so(self):
        points = cloud(3000)
        result = assess.completeness(arc(), points, cell_size_m=10.0, max_points=1000)
        self.assertTrue(result["decimated"])
        self.assertEqual(result["point_count_total"], 3000)
        self.assertLessEqual(result["point_count_assessed"], 1000)

    def test_support_counts_turn_observability_into_layers(self):
        centers, points = arc(), cloud(300, seed=2)
        support = np.random.default_rng(1).integers(0, 9, len(points))
        visible = np.minimum(support, np.random.default_rng(2).integers(0, 7, len(points)))
        result = assess.completeness(centers, points, cell_size_m=10.0,
                                     support=support, visible_support=visible)
        self.assertEqual(sum(result["layers"]["counts"].values()), 300)
        self.assertAlmostEqual(sum(result["layers"]["fractions"].values()), 1.0, places=6)
        self.assertEqual(result["layers"]["visibility_basis"],
                         "depth-map distance along the unit ray (survey_visibility)")

    def test_a_straight_flight_path_is_called_degenerate_not_good(self):
        line = np.column_stack([np.linspace(0, 200, 40), np.zeros(40), np.full(40, 40.0)])
        result = assess.completeness(line, cloud(200), cell_size_m=10.0)
        self.assertEqual(result["trajectory"]["verdict"], "degenerate")

    def test_too_few_cameras_is_refused(self):
        with self.assertRaises(ValueError):
            assess.completeness(arc(2), cloud(10))


class ThroughputTests(unittest.TestCase):
    def test_prediction_keeps_both_honesty_labels(self):
        result = assess.throughput(frame_count=291, image_px=1000, video_duration_s=600)
        self.assertEqual(result["status_label"], streaming.PREDICTED_LABEL)
        self.assertIn(streaming.NOT_MEASURED_LABEL, " ".join(result["disclaimers"]))
        self.assertEqual(result["frame_count_used"], 291)
        self.assertEqual(result["budget"]["total_s"], result["budget"]["total_s"])
        self.assertGreater(result["budget"]["total_s"], 0)

    def test_a_budget_the_measured_rates_cannot_meet_says_so(self):
        result = assess.throughput(frame_count=291, image_px=1000, video_duration_s=600)
        self.assertFalse(result["budget"]["fits_deadline"])
        self.assertEqual(result["dominant_stage"], "dense")

    def test_rates_are_copied_never_mutated(self):
        before = {name: dict(spec) for name, spec in streaming.MEASURED_RATES.items()}
        assess.throughput(frame_count=10, image_px=700, video_duration_s=60)
        self.assertEqual(streaming.MEASURED_RATES, before)


class ControlRequirementTests(unittest.TestCase):
    def test_the_answer_is_bound_to_the_scenes_own_telemetry(self):
        result = assess.control_requirement(
            telemetry={"video_duration_s": 30.0}, camera_centers_enu=arc(30),
            samples=SAMPLES, max_speed_m_s=8.0)
        requirement = result["requirement"]
        self.assertEqual(requirement["kind"], "decision_support")
        self.assertFalse(requirement["claims_accuracy"])
        self.assertEqual([o["id"] for o in requirement["constraint_options"]],
                         ["known_baseline", "barometric_trend", "imu_attitude", "rtk_fix"])
        self.assertEqual(result["telemetry_quality"]["count"], len(SAMPLES))
        self.assertFalse(result["vertical_reference"]["vertical_datum_verified"])


class AccuracySplitTests(unittest.TestCase):
    def rows(self, count=20, error=0.4):
        rng = np.random.default_rng(7)
        rows = []
        for index in range(count):
            surveyed = rng.uniform([-50, -50, 0], [50, 50, 30])
            rows.append({"id": f"cp{index}", "surveyed": surveyed,
                         "model": surveyed + rng.normal(0, error, 3)})
        return rows

    def test_too_few_checkpoints_returns_none_rather_than_a_fake_split(self):
        self.assertIsNone(assess.accuracy_from_checkpoints(
            self.rows(7), crs="EPSG:4979 ENU"))

    def test_a_split_report_carries_all_three_tracks(self):
        import json
        report = assess.accuracy_from_checkpoints(self.rows(), crs="EPSG:4979 ENU, origin 1 1")
        # The report goes straight into evaluation.json, so numpy vectors in the
        # caller's checkpoint rows must not survive into it.
        json.dumps(report, allow_nan=False)
        self.assertEqual(sorted(report["tracks"]),
                         ["se3_aligned", "sim3_aligned", "unaligned"])
        self.assertEqual(report["primary_track"], "unaligned")
        self.assertGreater(report["counts"]["checkpoints"], 0)
        self.assertTrue(report["what_this_does_not_prove"])


if __name__ == "__main__":
    unittest.main()
