"""Tests for baseline-aware keyframe selection (CPU, camera geometry only)."""
import unittest

import numpy as np

from scripts import survey_selection as sel


def _arc(count=60, radius=10.0, sweep=np.pi):
    angles = np.linspace(0, sweep, count)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)])


class SurveySelectionTests(unittest.TestCase):
    def test_selects_endpoints_and_drops_redundant_near_duplicates(self):
        positions = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.1, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
        chosen = sel.select_keyframes(positions, budget=3, min_baseline_m=1.0)
        self.assertEqual(chosen.tolist(), [0, 3, 4])

    def test_budget_is_never_exceeded_and_order_is_preserved(self):
        positions = _arc()
        chosen = sel.select_keyframes(positions, budget=8, min_baseline_m=0.5)
        self.assertLessEqual(len(chosen), 8)
        self.assertEqual(list(chosen), sorted(chosen))
        self.assertIn(0, chosen)
        self.assertIn(len(positions) - 1, chosen)

    def test_min_baseline_is_respected_between_consecutive_selections(self):
        positions = _arc()
        chosen = sel.select_keyframes(positions, budget=20, min_baseline_m=2.0)
        gaps = np.linalg.norm(np.diff(positions[chosen], axis=0), axis=1)
        self.assertTrue(np.all(gaps >= 2.0 - 1e-9), gaps)
        self.assertLess(len(chosen), 20)

    def test_thinning_a_straight_flight_leaves_no_long_gaps(self):
        positions = np.stack([np.linspace(0, 100, 200), np.zeros(200), np.zeros(200)], axis=1)
        chosen = sel.select_keyframes(positions, budget=10, min_baseline_m=1.0)
        gaps = np.linalg.norm(np.diff(positions[chosen], axis=0), axis=1)
        self.assertLess(gaps.max(), 15.0)
        self.assertLess(float(gaps.max() - gaps.min()), 1.5)

    def test_coverage_reports_the_longest_gap_and_covered_path_fraction(self):
        positions = np.stack([np.linspace(0, 100, 101), np.zeros(101), np.zeros(101)], axis=1)
        chosen = sel.select_keyframes(positions, budget=3, min_baseline_m=1.0)
        coverage = sel.path_coverage(positions, chosen, min_baseline_m=10.0)
        # Selected at 0, 50 and 100 m: 10 m + 20 m + 10 m of the 100 m path is
        # within one baseline of a chosen frame.
        self.assertAlmostEqual(coverage["max_gap_m"], 50.0, places=6)
        self.assertAlmostEqual(coverage["covered_fraction"], 0.4, places=6)
        self.assertEqual(coverage["selected"], 3)

    def test_parallax_angle_is_measured_at_an_explicit_scene_target(self):
        positions = _arc(4, radius=10.0, sweep=np.pi)  # 0, 60, 120, 180 degrees
        target = np.array([0.0, 0.0, 0.0])
        self.assertAlmostEqual(sel.mean_parallax_deg(positions, [0, 1], target), 60.0, places=4)
        self.assertAlmostEqual(sel.mean_parallax_deg(positions, [0, 1, 2, 3], target), 60.0, places=4)

    def test_forward_target_on_the_flight_line_gives_almost_no_parallax(self):
        positions = np.stack([np.linspace(0, 50, 20), np.zeros(20), np.zeros(20)], axis=1)
        angle = sel.mean_parallax_deg(positions, [0, 10, 19], np.array([60.0, 0.0, 0.0]))
        self.assertAlmostEqual(angle, 0.0, places=6)

    def test_invalid_inputs_raise(self):
        positions = _arc(10)
        for kwargs in ({"budget": 0, "min_baseline_m": 1.0}, {"budget": 3, "min_baseline_m": -1.0},
                       {"budget": 3, "min_baseline_m": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sel.select_keyframes(positions, **kwargs)
        with self.assertRaises(ValueError):
            sel.select_keyframes(np.zeros((2, 3)), budget=3, min_baseline_m=1.0)
        with self.assertRaises(ValueError):
            sel.select_keyframes(np.array([[0.0, 0, np.nan]] * 5), budget=3, min_baseline_m=1.0)
        with self.assertRaises(ValueError):
            sel.mean_parallax_deg(positions, [0], np.zeros(3))
        with self.assertRaises(ValueError):
            sel.mean_parallax_deg(positions, [0, 1], np.zeros(2))
        with self.assertRaises(ValueError):
            sel.path_coverage(positions, [0, 1], min_baseline_m=0)


if __name__ == "__main__":
    unittest.main()
