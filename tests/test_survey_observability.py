"""CPU tests for viewing-geometry observability: what a single pass can measure.

These are measurement tests. Every assertion pins a number that can be derived
from the geometry by hand, and every threshold that changes a verdict has to be
reachable as a named parameter - a verdict that cannot be re-parameterised is a
promise, not a measurement.
"""
import math
import unittest

import numpy as np

from scripts import survey_observability as obs


def _line(count=51, length=50.0, *, y=0.0, z=0.0):
    return np.column_stack([np.linspace(0.0, length, count),
                            np.full(count, float(y)), np.full(count, float(z))])


def _circle(count=36, radius=10.0):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles),
                            np.zeros(count)])


class TrajectoryQualityTests(unittest.TestCase):
    def test_straight_flight_is_degenerate_by_the_georeferencing_convention(self):
        quality = obs.trajectory_quality(_line(51, 100.0))
        self.assertAlmostEqual(quality["path_length_m"], 100.0, places=6)
        self.assertAlmostEqual(quality["displacement_m"], 100.0, places=6)
        self.assertEqual(len(quality["singular_values_m"]), 3)
        self.assertTrue(np.all(np.diff(quality["singular_values_m"]) <= 1e-9))
        self.assertLess(quality["second_over_first"], 1e-9)
        self.assertEqual(quality["verdict"], "degenerate")
        # Same numeric convention as survey_georef._noncollinear: the centred
        # trajectory's second/first singular value against the ratio floor.
        self.assertIn("min_second_ratio", quality["reason"])
        self.assertAlmostEqual(quality["straightness"], 1.0, places=6)

    def test_a_full_orbit_is_observable_and_not_straight(self):
        quality = obs.trajectory_quality(_circle(72, radius=10.0))
        self.assertAlmostEqual(quality["second_over_first"], 1.0, places=3)
        self.assertAlmostEqual(quality["straightness"], 0.5, places=3)
        self.assertEqual(quality["verdict"], "observable")
        self.assertLess(quality["path_length_m"], 63.0)  # 71 chords of a 2*pi*R circle
        self.assertIn("straightness", quality["reason"])

    def test_lateral_wander_on_a_long_line_is_weak_not_observable(self):
        positions = _line(51, 100.0)
        positions[:, 1] = 1.5 * ((-1.0) ** np.arange(51))  # +-1.5 m GPS wobble
        quality = obs.trajectory_quality(positions)
        # std(y)/std(x) = 1.5 / (100/sqrt(12)) ~ 0.051: above the degeneracy
        # floor, below the weak floor.
        self.assertGreater(quality["second_over_first"], 0.01)
        self.assertLess(quality["second_over_first"], 0.10)
        self.assertEqual(quality["verdict"], "weak")
        self.assertIn("weak_second_ratio", quality["reason"])

    def test_both_verdict_thresholds_are_parameters_not_baked_in(self):
        positions = _line(51, 100.0)
        positions[:, 1] = 1.5 * ((-1.0) ** np.arange(51))
        self.assertEqual(obs.trajectory_quality(positions, weak_second_ratio=0.02)["verdict"],
                         "observable")
        # The floor is a strict "<", so sitting just below it still trips the verdict.
        self.assertEqual(obs.trajectory_quality(positions, min_second_ratio=0.049)["verdict"],
                         "weak")
        self.assertEqual(obs.trajectory_quality(positions, min_second_ratio=0.053)["verdict"],
                         "degenerate")
        semicircle = _circle(61)[:31]  # half an orbit: second/first ~ 0.44
        ratio = obs.trajectory_quality(semicircle)["second_over_first"]
        self.assertTrue(0.3 < ratio < 0.6, ratio)
        self.assertEqual(obs.trajectory_quality(semicircle)["verdict"], "observable")
        self.assertEqual(
            obs.trajectory_quality(semicircle, min_second_ratio=0.5,
                                   weak_second_ratio=0.6)["verdict"], "degenerate")
        # An inverted band is an inconsistent request, not a silent clamp.
        with self.assertRaises(ValueError):
            obs.trajectory_quality(semicircle, min_second_ratio=0.5)

    def test_zero_extent_reports_degenerate_instead_of_dividing_by_zero(self):
        quality = obs.trajectory_quality(np.zeros((5, 3)))
        self.assertEqual(quality["verdict"], "degenerate")
        self.assertEqual(quality["path_length_m"], 0.0)
        self.assertEqual(quality["straightness"], 0.0)
        self.assertIn("no extent", quality["reason"])

    def test_inputs_are_validated(self):
        for bad in (np.zeros((2, 3)), np.zeros((3, 2)), np.full((4, 3), np.nan),
                    np.zeros((4, 4)), [0.0, 1.0, 2.0], "nonsense"):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                obs.trajectory_quality(bad)
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                obs.trajectory_quality(_line(), min_second_ratio=bad)
        with self.assertRaises(ValueError):
            obs.trajectory_quality(_line(), weak_second_ratio=0.005)  # below the floor


class BaselineTests(unittest.TestCase):
    def test_evenly_spaced_ring_gives_the_expected_angle_and_relative_baseline(self):
        report = obs.baselines(_circle(36, radius=10.0), np.zeros(3))
        self.assertEqual(report["camera_count"], 36)
        self.assertTrue(np.allclose(report["range_m"], 10.0))
        self.assertTrue(np.allclose(report["triangulation_angle_deg"], 10.0, atol=1e-6))
        # Equidistant pair: perpendicular baseline / range = sin(intersection angle).
        self.assertTrue(np.allclose(report["baseline_range_ratio"], math.sin(math.radians(10.0)),
                                    atol=1e-6))
        self.assertTrue(np.allclose(report["effective_baseline_m"],
                                    10.0 * math.sin(math.radians(10.0)), atol=1e-6))
        self.assertTrue(np.allclose(report["separation_m"], 20.0 * math.sin(math.radians(5.0)),
                                    atol=1e-6))
        # 36 evenly spaced cameras: the ring closes, so each camera has two
        # equidistant neighbours and either is a correct answer.
        for index, partner in enumerate(report["partner_index"]):
            self.assertIn(partner, ((index - 1) % 36, (index + 1) % 36))

    def test_partner_is_the_geometrically_nearest_other_camera(self):
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0], [5.2, 0.0, 0.0]])
        report = obs.baselines(centers, np.array([2.6, 20.0, 0.0]))
        self.assertEqual(report["partner_index"], [1, 0, 3, 2])
        self.assertAlmostEqual(report["separation_m"][0], 1.0, places=6)
        self.assertAlmostEqual(report["separation_m"][3], 0.2, places=6)
        # Close cameras on one side of a distant target buy almost no intersection
        # angle, which is the failure mode a long straight pass produces.
        self.assertLess(max(report["triangulation_angle_deg"]), obs.MIN_PARALLAX_DEG)

    def test_cameras_on_the_target_line_have_no_usable_baseline(self):
        """Both cameras look along one line: depth is unconstrained there."""
        centers = np.array([[0.0, 0, 0], [50.0, 0, 0], [0.2, 0.0, 0.0]])
        report = obs.baselines(centers, np.array([25.0, 0.0, 0.0]))
        self.assertLess(max(report["triangulation_angle_deg"]), 1.0)
        self.assertTrue(np.all(np.asarray(report["baseline_range_ratio"]) < 0.02))

    def test_max_range_flags_cameras_that_cannot_reach_the_scene(self):
        centers = np.array([[10.0, 0, 0], [10.2, 0, 0], [500.0, 0, 0]])
        report = obs.baselines(centers, np.zeros(3), max_range_m=50.0)
        self.assertEqual(report["in_range"], [True, True, False])
        self.assertEqual(report["cameras_in_range"], 2)
        self.assertEqual(report["max_range_m"], 50.0)
        self.assertTrue(report["usable"][0])
        self.assertFalse(report["usable"][2])  # outside the declared working range
        unlimited = obs.baselines(centers, np.zeros(3))
        self.assertIsNone(unlimited["max_range_m"])
        self.assertEqual(unlimited["cameras_in_range"], 3)

    def test_a_lone_camera_reports_a_zero_baseline_instead_of_crashing(self):
        report = obs.baselines(np.array([[5.0, 0.0, 0.0]]), np.zeros(3))
        self.assertEqual(report["partner_index"], [-1])
        self.assertEqual(report["effective_baseline_m"], [0.0])
        self.assertEqual(report["triangulation_angle_deg"], [0.0])
        self.assertEqual(report["usable"], [False])

    def test_baseline_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            obs.baselines(np.zeros((4, 3)), np.zeros(3))          # cameras on the target
        with self.assertRaises(ValueError):
            obs.baselines(np.array([[1.0, 0, 0]]), np.array([1.0, 0, 0.0]))
        with self.assertRaises(ValueError):
            obs.baselines(_circle(6), np.zeros(2))                # target not a 3-vector
        with self.assertRaises(ValueError):
            obs.baselines(_circle(6), np.zeros(3), max_range_m=0.0)
        with self.assertRaises(ValueError):
            obs.baselines(np.zeros((0, 3)), np.zeros(3))


class RegionObservabilityTests(unittest.TestCase):
    def setUp(self):
        # 51 cameras along x at y=z=0. Ten points per cell: the row on the camera
        # line (all rays collinear -> nothing can be triangulated) and the row 20 m
        # to the side (the pass sweeps a wide angle across it).
        self.cameras = _line(51, 50.0)
        xs = np.linspace(0.25, 49.25, 50)
        self.points = np.array([[x, y, 0.0] for y in (0.0, 20.0) for x in xs])

    def test_cells_on_the_camera_line_are_weak_and_off_line_cells_are_not(self):
        result = obs.region_observability(self.cameras, self.points, cell_size_m=10.0)
        self.assertEqual(result["cell_size_m"], 10.0)
        self.assertEqual(result["cell_count"], 10)
        self.assertEqual(result["point_count"], 100)
        on_line = [c for c in result["cells"] if abs(c["center_m"][1]) < 1.0]
        off_line = [c for c in result["cells"] if c["center_m"][1] > 1.0]
        self.assertEqual(len(on_line), 5)
        self.assertEqual(len(off_line), 5)
        self.assertTrue(all(c["verdict"] == "weak" for c in on_line), on_line)
        self.assertTrue(all(c["verdict"] == "observable" for c in off_line), off_line)
        for cell in on_line:
            self.assertLess(cell["max_triangulation_angle_deg"], result["min_parallax_deg"])
            self.assertEqual(cell["cameras_in_range"], 51)
        for cell in off_line:
            self.assertGreaterEqual(cell["max_triangulation_angle_deg"],
                                    result["min_parallax_deg"])
        self.assertAlmostEqual(result["weak_fraction"], 0.5, places=6)
        self.assertEqual(result["unobservable_fraction"], 0.0)
        self.assertAlmostEqual(result["unmeasurable_fraction"], 0.5, places=6)

    def test_min_views_decides_unobservable_cells(self):
        tight = obs.region_observability(self.cameras, self.points, cell_size_m=10.0,
                                         max_range_m=22.0, min_views=40)
        self.assertGreater(tight["unobservable_fraction"], 0.0)
        self.assertEqual(tight["min_views"], 40)
        for cell in tight["cells"]:
            if cell["verdict"] == "unobservable":
                self.assertLess(cell["cameras_in_range"], 40)
            self.assertLessEqual(cell["cameras_in_range"], 51)
        lenient = obs.region_observability(self.cameras, self.points, cell_size_m=10.0,
                                           max_range_m=22.0, min_views=3)
        self.assertEqual(lenient["unobservable_fraction"], 0.0)
        self.assertLess(lenient["unobservable_fraction"], tight["unobservable_fraction"])

    def test_parallax_threshold_is_a_named_parameter(self):
        default = obs.region_observability(self.cameras, self.points, cell_size_m=10.0)
        strict = obs.region_observability(self.cameras, self.points, cell_size_m=10.0,
                                          min_parallax_deg=80.0)
        self.assertEqual(default["min_parallax_deg"], obs.MIN_PARALLAX_DEG)
        self.assertEqual(strict["min_parallax_deg"], 80.0)
        # Raising the required intersection angle can only remove cells from the
        # observable set, never add them.
        self.assertEqual(default["observable_cells"], 5)
        self.assertEqual(strict["observable_cells"], 4)
        self.assertLess(strict["observable_cells"], default["observable_cells"])
        self.assertAlmostEqual(strict["weak_fraction"], 0.6, places=6)
        for cell in strict["cells"]:
            if cell["verdict"] == "weak":
                self.assertLess(cell["max_triangulation_angle_deg"], 80.0)

    def test_median_baseline_range_ratio_is_reported_per_cell(self):
        result = obs.region_observability(self.cameras, self.points, cell_size_m=10.0)
        off_line = [c for c in result["cells"] if c["center_m"][1] > 1.0]
        on_line = [c for c in result["cells"] if abs(c["center_m"][1]) < 1.0]
        self.assertGreater(max(c["median_baseline_range_ratio"] for c in off_line), 0.5)
        self.assertLess(max(c["median_baseline_range_ratio"] for c in on_line), 0.02)
        for cell in result["cells"]:
            self.assertEqual(cell["point_count"], 10)
            self.assertTrue(0.0 <= cell["median_baseline_range_ratio"] <= 1.0)
            self.assertEqual(cell["local_stereo_weak"],
                             cell["median_baseline_range_ratio"] < math.sin(
                                 math.radians(result["min_parallax_deg"])))

    def test_camera_subsampling_is_disclosed_when_it_happens(self):
        dense_cameras = _line(501, 50.0)
        capped = obs.region_observability(dense_cameras, self.points, cell_size_m=10.0,
                                          max_cameras_per_cell=64)
        self.assertTrue(capped["subsampling_applied"])
        self.assertIn("subsampled", capped["method"])
        for cell in capped["cells"]:
            self.assertLessEqual(cell["cameras_evaluated"], 64)
        full = obs.region_observability(dense_cameras, self.points, cell_size_m=10.0,
                                        max_cameras_per_cell=1024)
        self.assertFalse(full["subsampling_applied"])
        # Order-preserving subsampling keeps the endpoints, so the angular span
        # must not collapse: the same verdicts either way.
        self.assertEqual([c["verdict"] for c in full["cells"]],
                         [c["verdict"] for c in capped["cells"]])

    def test_region_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            obs.region_observability(self.cameras, self.points, cell_size_m=0)
        with self.assertRaises(ValueError):
            obs.region_observability(self.cameras, self.points, cell_size_m=10.0,
                                     min_views=0)
        with self.assertRaises(ValueError):
            obs.region_observability(self.cameras, self.points, cell_size_m=10.0,
                                     min_parallax_deg=-1.0)
        with self.assertRaises(ValueError):
            obs.region_observability(self.cameras, np.zeros((0, 3)), cell_size_m=10.0)
        with self.assertRaises(ValueError):
            obs.region_observability(self.cameras, self.points, cell_size_m=10.0,
                                     plane_axes=(0, 0))


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.cameras = _line(51, 50.0)
        xs = np.linspace(0.25, 49.25, 50)
        self.points = np.array([[x, y, 0.0] for y in (0.0, 20.0) for x in xs])

    def test_summary_combines_all_three_measurements(self):
        summary = obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0)
        for key in ("trajectory", "baselines", "regions", "limits", "thresholds",
                    "not_measured"):
            self.assertIn(key, summary)
        self.assertEqual(summary["trajectory"]["verdict"], "degenerate")
        self.assertEqual(summary["regions"]["cell_count"],
                         summary["regions"]["observable_cells"]
                         + summary["regions"]["weak_cells"]
                         + summary["regions"]["unobservable_cells"])
        self.assertEqual(summary["thresholds"]["cell_size_m"], 10.0)
        self.assertEqual(summary["thresholds"]["min_parallax_deg"], obs.MIN_PARALLAX_DEG)
        self.assertEqual(summary["thresholds"]["min_second_ratio"], obs.MIN_SECOND_RATIO)

    def test_limits_state_the_unmeasurable_regions_with_numbers(self):
        summary = obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0)
        self.assertTrue(summary["limits"], "a straight pass must report its limits")
        text = "\n".join(summary["limits"])
        weak, cells = summary["regions"]["weak_cells"], summary["regions"]["cell_count"]
        self.assertEqual((weak, cells), (5, 10))
        self.assertIn(f"{weak} of {cells} cells", text)
        self.assertIn("5.0", text)  # the threshold itself is quoted, not hidden
        self.assertIn("cannot be reconstructed from this single pass", text)
        self.assertIn("degenerate", text)

    def test_summary_says_what_it_did_not_measure(self):
        summary = obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0)
        text = "\n".join(summary["not_measured"])
        self.assertIn("camera geometry", text)
        self.assertIn("surface completeness", text)
        self.assertNotIn("guarantee", text.lower())
        self.assertNotIn("demonstrates accuracy", text)

    def test_explicit_target_is_used_when_supplied(self):
        target = np.array([25.0, 20.0, 0.0])
        summary = obs.summarise(self.cameras, self.cameras, self.points,
                                cell_size_m=10.0, target=target)
        self.assertEqual(summary["target"]["source"], "supplied")
        self.assertEqual(summary["target"]["position_m"], [25.0, 20.0, 0.0])
        default = obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0)
        self.assertEqual(default["target"]["source"], "mean of points")

    def test_an_orbiting_pass_measures_where_a_straight_one_cannot(self):
        orbit = _circle(36, radius=25.0)
        points = np.array([[x, y, 0.0] for x in np.linspace(-20, 20, 9)
                           for y in np.linspace(-20, 20, 9)])
        round = obs.summarise(orbit, orbit, points, cell_size_m=10.0, max_range_m=60.0)
        straight = obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0)
        self.assertEqual(round["trajectory"]["verdict"], "observable")
        self.assertEqual(round["regions"]["cell_count"], 25)
        self.assertLess(round["regions"]["unmeasurable_fraction"],
                        straight["regions"]["unmeasurable_fraction"])
        self.assertGreater(round["baselines"]["triangulation_angle_deg"]["min"], 5.0)
        # The tool measures camera geometry only, so it still has to state a limit.
        self.assertTrue(any("surface completeness" in line for line in round["not_measured"]))

    def test_summary_thresholds_reach_through_to_the_regions(self):
        summary = obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0,
                                min_parallax_deg=30.0, min_views=40, max_range_m=12.0)
        self.assertEqual(summary["regions"]["min_parallax_deg"], 30.0)
        self.assertEqual(summary["regions"]["min_views"], 40)
        self.assertEqual(summary["regions"]["max_range_m"], 12.0)
        self.assertEqual(summary["thresholds"]["min_views"], 40)
        self.assertGreater(summary["regions"]["unobservable_fraction"], 0.0)

    def test_summary_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=-1.0)
        with self.assertRaises(ValueError):
            obs.summarise(self.cameras, self.cameras, np.zeros((0, 3)), cell_size_m=10.0)
        with self.assertRaises(ValueError):
            obs.summarise(self.cameras, self.cameras, self.points, cell_size_m=10.0,
                          target=np.zeros(4))


if __name__ == "__main__":
    unittest.main()
