"""CPU-only tests for the metric-accuracy protocol; synthetic surveys only.

The injected error in every case is known by construction, so these tests check
that the protocol reproduces ground truth on held-out points - they do not, and
must not be quoted as, evidence about a real outdoor scene.
"""
import copy
import json
import math
import unittest

import numpy as np

from scripts import survey_accuracy as acc
from scripts import survey_evaluation as ev
from scripts import survey_georef as georef
from scripts import survey_gnss as gnss


def yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def survey_table(count=40, *, scale=1.2, rotation_angle=0.0, translation=(0.0, 0.0, 0.0),
                 noise=0.0, seed=3, with_model=True):
    """Surveyed ENU points plus model coordinates carrying a KNOWN injected error.

    surveyed == scale * (model @ R.T) + translation, so the true alignment is known.
    """
    rng = np.random.default_rng(seed)
    surveyed = np.column_stack((rng.uniform(-90.0, 110.0, count),
                                rng.uniform(-70.0, 80.0, count),
                                rng.uniform(-6.0, 48.0, count)))
    rotation = yaw(rotation_angle)
    model = ((surveyed - np.array(translation, dtype=float)) / scale) @ rotation
    if noise:
        model = model + rng.normal(0.0, noise, model.shape)
    rows = []
    for index, (surveyed_point, model_point) in enumerate(zip(surveyed, model)):
        row = {"id": f"ctl{index:02d}", "surveyed": surveyed_point.tolist()}
        if with_model:
            row["model"] = model_point.tolist()
        rows.append(row)
    return rows


def ids_of(rows):
    return [row["id"] for row in rows]


class SplitReferenceTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_replayable(self):
        rows = survey_table(40)
        first = acc.split_reference(rows)
        second = acc.split_reference(survey_table(40))
        self.assertEqual(first["control_ids"], second["control_ids"])
        self.assertEqual(first["checkpoint_ids"], second["checkpoint_ids"])
        self.assertEqual(first["replay"], second["replay"])
        self.assertFalse(set(first["control_ids"]) & set(first["checkpoint_ids"]))
        self.assertEqual(sorted(first["control_ids"] + first["checkpoint_ids"]),
                         sorted(ids_of(rows)))
        self.assertEqual(first["counts"], {"rows": 40, "controls": 24, "checkpoints": 16})
        self.assertEqual(first["fit_fraction"], 0.6)
        self.assertEqual(first["method"], "seeded_stratified_farthest_point")
        self.assertIs(first["disjoint"], True)
        self.assertIs(first["independent_checkpoints_held_out"], True)
        json.dumps(first, allow_nan=False)

    def test_a_different_seed_replays_a_different_partition_of_the_same_rows(self):
        rows = survey_table(40)
        a = acc.split_reference(rows, seed=0)
        b = acc.split_reference(rows, seed=7)
        self.assertNotEqual(a["control_ids"], b["control_ids"])
        self.assertEqual(a["replay"]["input_digest"], b["replay"]["input_digest"])
        self.assertEqual(a["counts"], b["counts"])

    def test_inputs_are_never_modified_and_outputs_are_independent_copies(self):
        rows = survey_table(20)
        before = copy.deepcopy(rows)
        got = acc.split_reference(rows)
        got["controls"][0]["surveyed"][0] = 12345.0
        self.assertEqual(rows, before)
        self.assertNotIn(12345.0, [p for row in got["checkpoints"] for p in row["surveyed"]])

    def test_controls_and_checkpoints_are_spread_not_clustered(self):
        got = acc.split_reference(survey_table(40))
        for key in ("control", "checkpoint"):
            fraction = got["spread"][f"{key}_radius_fraction"]
            self.assertGreaterEqual(fraction, 0.6, msg=key)
        self.assertGreater(got["spread"]["controls"]["diameter_m"], 100.0)
        self.assertGreater(got["spread"]["checkpoints"]["radius_of_gyration_m"], 30.0)

    def test_height_stratification_spreads_both_sets_over_height(self):
        got = acc.split_reference(survey_table(40), stratify_by="height")
        self.assertEqual(got["stratify_by"], "height")
        for key in ("controls", "checkpoints"):
            self.assertGreaterEqual(got["spread"][key]["height_range_m"],
                                    0.6 * got["spread"]["all"]["height_range_m"] - 1e-9, msg=key)

    def test_a_survey_clustered_beside_the_takeoff_point_is_rejected(self):
        dense = [{"id": f"d{i}", "surveyed": [(i % 5) * 0.9, ((i // 5) % 5) * 0.9, (i % 4) * 0.3]}
                 for i in range(34)]
        far = [{"id": f"f{i}", "surveyed": [200.0 + 60.0 * i, 200.0 - 90.0 * i, 0.0]}
               for i in range(6)]
        with self.assertRaises(ValueError) as caught:
            acc.split_reference(dense + far, stratify_by=None)
        self.assertRegex(str(caught.exception), r"(?i)spread|cluster")

    def test_height_only_clustering_is_caught_by_the_vertical_spread_check(self):
        plane = [{"id": f"p{i}", "surveyed": [float((i % 6) * 400), float((i // 6) * 200), 0.0]}
                 for i in range(24)]
        rows = plane + [{"id": "HIGH", "surveyed": [2000.0, 1000.0, 300.0]}]
        for seed in range(4):
            with self.subTest(seed=seed), self.assertRaises(ValueError) as caught:
                acc.split_reference(rows, stratify_by=None, seed=seed)
            self.assertRegex(str(caught.exception), r"(?i)height")

    def test_near_collinear_controls_are_rejected_for_the_georeferencing_reason(self):
        rows = [{"id": f"l{i}", "surveyed": [i * 10.0, 2.0 * i, -i]} for i in range(1, 21)]
        with self.assertRaises(ValueError) as caught:
            acc.split_reference(rows)
        self.assertRegex(str(caught.exception), r"(?i)collinear")

    def test_a_declared_scene_extent_the_survey_does_not_cover_is_rejected(self):
        rows = [{"id": f"t{i}", "surveyed": [i % 3, (i // 3) % 3, (i % 5) * 0.5]}
                for i in range(24)]
        self.assertEqual(acc.split_reference(rows)["counts"]["rows"], 24)
        with self.assertRaises(ValueError) as caught:
            acc.split_reference(rows, scene_extent_m=300.0)
        self.assertRegex(str(caught.exception), r"(?i)scene")

    def test_a_flat_survey_is_split_but_warns_that_vertical_error_cannot_be_measured(self):
        flat = [{"id": f"g{i}", "surveyed": [float((i % 6) * 30), float((i // 6) * 30), 0.0]}
                for i in range(18)]
        got = acc.split_reference(flat)
        self.assertNotIn("checkpoint_height_fraction", got["spread"])
        self.assertTrue(any("height" in w.lower() and "vertical" in w.lower()
                            for w in got["warnings"]))

    def test_a_clustered_CONTROL_set_is_rejected_too(self):
        # With a strict 1.0 spread requirement the guard must name the controls, not
        # only the leftovers: both subsets have to cover the scene.
        rows = survey_table(12)
        with self.assertRaises(ValueError) as caught:
            acc.split_reference(rows, fit_fraction=0.34, min_spread_fraction=1.0)
        self.assertRegex(str(caught.exception), r"(?i)the control points are clustered")

    def test_a_well_spread_survey_satisfies_the_vertical_spread_guard(self):
        got = acc.split_reference(survey_table(40))
        self.assertEqual(got["spread"]["absolute"], "not_declared")
        for key in ("control", "checkpoint"):
            self.assertGreaterEqual(got["spread"][f"{key}_height_fraction"], 0.6, msg=key)
            self.assertGreaterEqual(got["spread"][f"{key}_radius_fraction"], 0.6, msg=key)
        self.assertEqual(sorted(got["strata"]["checkpoints_covered"]), [0, 1, 2, 3])

    def test_split_works_on_a_survey_with_no_model_coordinates_yet(self):
        rows = survey_table(20, with_model=False)
        got = acc.split_reference(rows)
        self.assertEqual(got["counts"]["checkpoints"], 8)
        self.assertNotIn("model", got["controls"][0])

    def test_split_parameters_and_tables_are_validated(self):
        rows = survey_table(20)
        for kwargs in ({"fit_fraction": 0.0}, {"fit_fraction": 1.0}, {"fit_fraction": 1.5},
                       {"fit_fraction": "0.6"}, {"seed": 0.5}, {"seed": "0"},
                       {"stratify_by": "class"}, {"stratify_by": "heights"},
                       {"min_controls": 0}, {"min_checkpoints": -1},
                       {"min_spread_fraction": 1.5}, {"min_spread_fraction": 0},
                       {"scene_extent_m": 0}, {"scene_extent_m": np.nan},
                       {"fit_fraction": 0.95}, {"fit_fraction": 0.05}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                acc.split_reference(rows, **kwargs)
        bad_tables = ([], [{"id": "a", "surveyed": [0, 0, 0]}] * 2,
                      [{"surveyed": [0, 0, 0]}] * 8,
                      [{"id": "a", "surveyed": [0, 0]}] * 8,
                      [{"id": f"x{i}", "surveyed": [np.nan, 0, 0]} for i in range(8)],
                      [{"id": f"x{i}", "surveyed": [i, 0, 0], "model": [i, 0]} for i in range(8)],
                      "x")
        for table in bad_tables:
            with self.subTest(table=table), self.assertRaises(ValueError):
                acc.split_reference(table)
        for count in (7, 8):
            with self.subTest(count=count), self.assertRaises(ValueError):
                acc.split_reference(survey_table(count))
        grid = [{"id": f"g{i}", "surveyed": [(i % 5) * 20.0, (i // 5) * 20.0, (i % 3) * 9.0]}
                for i in range(16)]
        self.assertEqual(acc.split_reference(grid)["counts"]["checkpoints"], 6)
        self.assertEqual(acc.split_reference(survey_table(10))["counts"]["checkpoints"], 4)


class AlignmentTests(unittest.TestCase):
    def test_sim3_fit_on_controls_recovers_the_injected_transform(self):
        rows = survey_table(40, scale=1.02, rotation_angle=0.35,
                            translation=(12.5, -7.25, 3.1), noise=0.05)
        split = acc.split_reference(rows)
        got = acc.alignment_trackpoints(split["controls"], allow_scale=True)
        self.assertEqual(got["kind"], "sim3")
        self.assertEqual(got["status"], "aligned")
        self.assertAlmostEqual(got["scale"], 1.02, delta=1e-3)
        self.assertTrue(np.allclose(got["rotation"], yaw(0.35), atol=1e-3))
        self.assertTrue(np.allclose(got["translation"], [12.5, -7.25, 3.1], atol=0.2))
        self.assertEqual(got["control_count"], len(split["controls"]))
        self.assertEqual(got["fitted_from"], "controls_only")

    def test_the_rigid_track_forbids_scale_so_a_scale_error_survives(self):
        rows = survey_table(40, scale=1.2, rotation_angle=0.2, translation=(5.0, 0.0, -2.0))
        split = acc.split_reference(rows)
        controls = split["controls"]
        rigid = acc.alignment_trackpoints(controls, allow_scale=False)
        self.assertEqual(rigid["kind"], "se3")
        self.assertEqual(rigid["scale"], 1.0)
        similarity = acc.alignment_trackpoints(controls, allow_scale=True)
        self.assertAlmostEqual(similarity["scale"], 1.2, delta=1e-9)
        self.assertGreater(rigid["fit_rmse_m"], 1.0)
        self.assertLess(similarity["fit_rmse_m"], 1e-9)

    def test_checkpoints_cannot_enter_the_fit_even_by_being_passed_in(self):
        rows = survey_table(30)
        split = acc.split_reference(rows)
        controls, checkpoints = split["controls"], split["checkpoints"]
        clean = acc.alignment_trackpoints(controls, allow_scale=True)
        with self.assertRaises(ValueError) as caught:
            acc.alignment_trackpoints(controls + checkpoints[:1], allow_scale=True,
                                      checkpoint_ids=ids_of(checkpoints))
        self.assertRegex(str(caught.exception), r"(?i)checkpoint")
        # Perturbing the held-out rows cannot move a controls-only fit.
        polluted = copy.deepcopy(rows)
        for row in polluted:
            if row["id"] in set(split["checkpoint_ids"]):
                row["model"] = [v + 500.0 for v in row["model"]]
        moved = acc.alignment_trackpoints(acc.split_reference(polluted)["controls"],
                                          allow_scale=True)
        self.assertTrue(np.allclose(clean["rotation"], moved["rotation"], atol=1e-9))
        self.assertAlmostEqual(clean["scale"], moved["scale"], delta=1e-9)

    def test_alignment_input_validation(self):
        rows = survey_table(20)
        split = acc.split_reference(rows)
        for kwargs in ({}, {"allow_scale": "yes"}, {"allow_scale": None}, {"allow_scale": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                acc.alignment_trackpoints(split["controls"], **kwargs)
        for size in (0, 1, 2):
            with self.subTest(size=size), self.assertRaises(ValueError):
                acc.alignment_trackpoints(split["controls"][:size], allow_scale=True)
        collinear = [{"id": f"c{i}", "surveyed": [i * 5.0, 0.0, 0.0],
                      "model": [i * 5.0, 0.0, 0.0]} for i in range(6)]
        with self.assertRaises(ValueError):
            acc.alignment_trackpoints(collinear, allow_scale=True)
        no_model = survey_table(20, with_model=False)
        with self.assertRaises(ValueError):
            acc.alignment_trackpoints(acc.split_reference(no_model)["controls"],
                                      allow_scale=True)
        duplicate = copy.deepcopy(split["controls"]) + [copy.deepcopy(split["controls"][0])]
        with self.assertRaises(ValueError):
            acc.alignment_trackpoints(duplicate, allow_scale=True)

    def test_the_transform_is_consumable_by_the_georeferencing_lane(self):
        rows = survey_table(20)
        split = acc.split_reference(rows)
        transform = acc.alignment_trackpoints(split["controls"], allow_scale=True)
        model = np.array([row["model"] for row in split["checkpoints"]])
        applied = georef.transform_points(model, transform)
        self.assertEqual(applied.shape, model.shape)
        self.assertTrue(np.all(applied[:, 2] > -1000))


class TrackEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = survey_table(40, scale=1.2)
        self.split = acc.split_reference(self.rows)
        factor = 1.0 / 1.2 - 1.0
        cp = np.array([row["surveyed"] for row in self.split["checkpoints"]])
        ct = np.array([row["surveyed"] for row in self.split["controls"]])
        self.expected_unaligned = abs(factor) * float(np.sqrt(np.mean(np.sum(cp ** 2, axis=1))))
        self.expected_rigid = abs(1.0 - 1.0 / 1.2) * float(
            np.sqrt(np.mean(np.sum((cp - ct.mean(axis=0)) ** 2, axis=1))))

    def test_a_sim3_fit_hides_a_20_percent_scale_error_that_the_other_tracks_show(self):
        unaligned = acc.evaluate_checkpoints(self.split["checkpoints"], None, "unaligned")
        rigid = acc.evaluate_checkpoints(self.split["checkpoints"],
                                         acc.alignment_trackpoints(self.split["controls"],
                                                                   allow_scale=False), "se3")
        similarity = acc.evaluate_checkpoints(self.split["checkpoints"],
                                              acc.alignment_trackpoints(self.split["controls"],
                                                                        allow_scale=True), "sim3")
        self.assertAlmostEqual(unaligned["metrics"]["rmse_3d_m"], self.expected_unaligned,
                               delta=1e-9)
        self.assertAlmostEqual(rigid["metrics"]["rmse_3d_m"], self.expected_rigid, delta=1e-9)
        self.assertGreater(unaligned["metrics"]["rmse_3d_m"], 10.0)
        self.assertGreater(rigid["metrics"]["rmse_3d_m"], 5.0)
        self.assertLess(similarity["metrics"]["rmse_3d_m"], 1e-9)
        self.assertEqual(unaligned["alignment"], "none")
        self.assertEqual(rigid["alignment"], "rigid_se3")
        self.assertEqual(similarity["alignment"], "similarity_sim3")
        self.assertIs(unaligned["usable_as_accuracy_evidence"], True)
        self.assertIs(rigid["usable_as_accuracy_evidence"], False)
        self.assertIs(similarity["usable_as_accuracy_evidence"], False)
        self.assertIs(similarity["scale_error_absorbed"], True)
        self.assertIs(rigid["scale_error_absorbed"], False)
        for label, track in (("unaligned", unaligned), ("se3", rigid), ("sim3", similarity)):
            self.assertIn(label, json.dumps({"t": track["label"]}, allow_nan=False))
        # A Sim(3) number must never be turned into an accuracy verdict.
        with self.assertRaises(ValueError) as caught:
            acc.accuracy_verdict(similarity, statistic="rmse_3d_m", target_m=1.0)
        self.assertRegex(str(caught.exception), r"(?i)scale")
        with self.assertRaises(ValueError):
            acc.accuracy_verdict(similarity)

    def test_the_unaligned_track_feeds_the_official_evaluation_unchanged(self):
        track = acc.evaluate_checkpoints(self.split["checkpoints"], None, "unaligned")
        built = ev.build_evaluation(checkpoints=track["metrics"])
        self.assertEqual(built["criteria"][0]["status"], "measured")
        self.assertEqual(built["criteria"][0]["metrics"]["rmse_3d_m"],
                         track["metrics"]["rmse_3d_m"])
        for mode, transform in (
                ("se3", acc.alignment_trackpoints(self.split["controls"], allow_scale=False)),
                ("sim3", acc.alignment_trackpoints(self.split["controls"], allow_scale=True))):
            aligned = acc.evaluate_checkpoints(self.split["checkpoints"], transform, mode)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ev.build_evaluation(checkpoints=aligned["metrics"])

    def test_verdicts_are_labelled_by_what_each_alignment_removes(self):
        unaligned = acc.evaluate_checkpoints(self.split["checkpoints"], None, "unaligned")
        rigid = acc.evaluate_checkpoints(self.split["checkpoints"],
                                         acc.alignment_trackpoints(self.split["controls"],
                                                                   allow_scale=False), "se3")
        absolute = acc.accuracy_verdict(unaligned, statistic="rmse_3d_m", target_m=1.0)
        self.assertEqual(absolute["kind"], "absolute_metric_accuracy")
        self.assertIs(absolute["meets_target"], False)
        secondary = acc.accuracy_verdict(rigid, statistic="rmse_3d_m", target_m=1.0)
        self.assertEqual(secondary["kind"], "relative_geometry_secondary")
        self.assertIsNone(secondary["absolute_accuracy"])
        undeclared = acc.accuracy_verdict(unaligned)
        self.assertIsNone(undeclared["meets_target"])
        self.assertTrue(undeclared["reason"])
        for bogus in ("p95_99_m", "count", "bias_xyz_m", 1.0):
            with self.subTest(bogus=bogus), self.assertRaises(ValueError):
                acc.accuracy_verdict(unaligned, statistic=bogus)

    def test_modes_and_transform_kinds_cannot_be_mixed(self):
        rigid = acc.alignment_trackpoints(self.split["controls"], allow_scale=False)
        similarity = acc.alignment_trackpoints(self.split["controls"], allow_scale=True)
        for mode, transform in (("se3", similarity), ("sim3", rigid), ("se3", None),
                                ("sim3", None), ("unaligned", rigid)):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                acc.evaluate_checkpoints(self.split["checkpoints"], transform, mode)
        for mode in ("SIM3", "", "similarity", None, "accuracy"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                acc.evaluate_checkpoints(self.split["checkpoints"], None, mode)
        with self.assertRaises(ValueError):
            acc.evaluate_checkpoints([], None, "unaligned")
        broken = copy.deepcopy(self.split["checkpoints"])
        broken[0]["model"] = [0.0, 0.0, np.nan]
        with self.assertRaises(ValueError):
            acc.evaluate_checkpoints(broken, None, "unaligned")
        without_model = survey_table(20, with_model=False)
        with self.assertRaises(ValueError):
            acc.evaluate_checkpoints(acc.split_reference(without_model)["checkpoints"],
                                     None, "unaligned")


class ReportTests(unittest.TestCase):
    def test_report_runs_the_whole_protocol_with_explicit_tracks(self):
        rows = survey_table(40, scale=1.2)
        got = acc.report(rows=rows, crs="EPSG:4979 ENU, origin 12.34N 100.98E",
                         vertical_datum="ellipsoidal", withheld_from_reconstruction=True)
        self.assertEqual(sorted(got["tracks"]), ["se3_aligned", "sim3_aligned", "unaligned"])
        self.assertEqual(got["primary_track"], "unaligned")
        self.assertIs(got["accuracy_validated"], True)
        self.assertEqual(got["crs"], "EPSG:4979 ENU, origin 12.34N 100.98E")
        self.assertEqual(got["vertical_datum"], "ellipsoidal")
        self.assertEqual(got["counts"]["checkpoints"], 16)
        self.assertGreater(got["checkpoint_spread"]["radius_of_gyration_m"], 0.0)
        for key in ("horizontal_rmse_m", "vertical_rmse_m", "rmse_3d_m", "median_3d_m",
                    "p95_3d_m", "max_3d_m", "bias_xyz_m"):
            self.assertIn(key, got["headline"])
            self.assertEqual(got["headline"][key],
                             got["tracks"]["unaligned"]["metrics"][key])
        self.assertIsNone(got["meets_accuracy_target"])
        self.assertTrue(got["what_this_does_not_prove"])
        self.assertRegex(" ".join(got["what_this_does_not_prove"]), r"(?i)scale")
        json.dumps(got, allow_nan=False)

    def test_a_sim3_primary_track_cannot_be_asked_for_an_accuracy_verdict(self):
        rows = survey_table(40, scale=1.2)
        with self.assertRaises(ValueError) as caught:
            acc.report(rows=rows, crs="EPSG:4979 ENU", vertical_datum="ellipsoidal",
                       withheld_from_reconstruction=True, primary_track="sim3_aligned",
                       target_statistic="rmse_3d_m")
        self.assertRegex(str(caught.exception), r"(?i)scale|metric")
        rigid = acc.report(rows=rows, crs="EPSG:4979 ENU", vertical_datum="ellipsoidal",
                           withheld_from_reconstruction=True, primary_track="se3_aligned",
                           target_statistic="rmse_3d_m")
        self.assertIs(rigid["accuracy_validated"], False)
        self.assertIsNone(rigid["meets_accuracy_target"])

    def test_accuracy_is_only_validated_when_the_checkpoints_are_independent(self):
        rows = survey_table(40)
        base = dict(crs="EPSG:4979 ENU", vertical_datum="ellipsoidal")
        withheld = acc.report(rows=rows, **base, withheld_from_reconstruction=False)
        self.assertIs(withheld["accuracy_validated"], False)
        reasons = " ".join(withheld["accuracy_validation_reasons"]).lower()
        self.assertTrue("independent" in reasons or "withheld" in reasons)
        unknown_datum = acc.report(rows=rows, crs="EPSG:4979 ENU", vertical_datum="unknown",
                                   withheld_from_reconstruction=True)
        self.assertIs(unknown_datum["accuracy_validated"], False)
        self.assertTrue(any("vertical" in r.lower() for r in
                            unknown_datum["accuracy_validation_reasons"]))
        for kwargs in ({"crs": ""}, {"crs": None}, {"vertical_datum": ""},
                       {"vertical_datum": "MSL"}, {"vertical_datum": None},
                       {"crs": 42}, {"withheld_from_reconstruction": "yes"},
                       {"target_statistic": "rmse_99_m"}, {"target_statistic": 3},
                       {"accuracy_target_m": 0}, {"accuracy_target_m": np.nan},
                       {"primary_track": "sim3"}, {"primary_track": "accuracy"},
                       {"seed": "0"}):
            case = dict(base, rows=rows)
            case.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                acc.report(**case)

    def test_perfect_table_meets_a_declared_statistic(self):
        perfect = [{"id": f"p{i}", "surveyed": [float(x), float(y), float(z)],
                    "model": [float(x), float(y), float(z)]}
                   for i, (x, y, z) in enumerate(
                       [(i * 9.0, (i % 7) * 11.0, (i % 11) * 5.0) for i in range(40)])]
        passed = acc.report(rows=perfect, crs="EPSG:4979 ENU", vertical_datum="ellipsoidal",
                            withheld_from_reconstruction=True,
                            target_statistic="p95_3d_m", accuracy_target_m=1.0)
        self.assertIs(passed["accuracy_validated"], True)
        self.assertIs(passed["meets_accuracy_target"], True)
        self.assertEqual(passed["target"]["statistic"], "p95_3d_m")
        self.assertEqual(passed["target"]["threshold_m"], 1.0)
        strict = acc.report(rows=survey_table(40, scale=1.2), crs="EPSG:4979 ENU",
                            vertical_datum="ellipsoidal", withheld_from_reconstruction=True,
                            target_statistic="rmse_3d_m")
        self.assertIs(strict["meets_accuracy_target"], False)

    def test_report_refuses_tables_that_cannot_be_split_safely(self):
        line = [{"id": f"l{i}", "surveyed": [i * 10.0, 3.0 * i, -2.0 * i],
                 "model": [i * 10.0, 3.0 * i, -2.0 * i]} for i in range(1, 21)]
        with self.assertRaises(ValueError) as caught:
            acc.report(rows=line, crs="EPSG:4979 ENU", vertical_datum="ellipsoidal")
        self.assertRegex(str(caught.exception), r"(?i)collinear")
        dense = [{"id": f"d{i}", "surveyed": [(i % 5) * 0.9, ((i // 5) % 5) * 0.9, 0.0],
                  "model": [(i % 5) * 0.9, ((i // 5) % 5) * 0.9, 0.0]} for i in range(34)]
        far = [{"id": f"f{i}", "surveyed": [200.0 + 60.0 * i, 200.0 - 90.0 * i, 0.0],
                "model": [200.0 + 60.0 * i, 200.0 - 90.0 * i, 0.0]} for i in range(6)]
        with self.assertRaises(ValueError):
            acc.report(rows=dense + far, crs="EPSG:4979 ENU", vertical_datum="ellipsoidal",
                       stratify_by=None)


class GcpRequirementTests(unittest.TestCase):
    @staticmethod
    def trace(count=8, step=10.0, dt=2.0, h=0.4, curve=0.0):
        samples = []
        for i in range(count):
            position = [i * step, curve * (i * dt) ** 2, 0.5 * i]
            samples.append({"t_sec": i * dt, "position": position,
                            "horizontal_std_m": h + 0.01 * i, "vertical_std_m": 2 * h})
        return samples

    def straight_case(self):
        samples = self.trace()
        camera = [s["position"] for s in samples]
        return gnss.quality_report(samples, max_speed_m_s=6.0), gnss.observability(camera, samples)

    def test_it_reads_the_gnss_reports_and_returns_decision_support_only(self):
        quality, observability = self.straight_case()
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability)
        self.assertEqual(got["kind"], "decision_support")
        self.assertIs(got["claims_accuracy"], False)
        self.assertIs(got["measured"], False)
        self.assertEqual(got["scale_identifiable_from_gps_alone"]["status"], "supported")
        self.assertIs(got["clock_offset_and_lever_arm"]["separable"], False)
        self.assertIn("imu_attitude", got["recommended_minimum"]["constraint_ids"])
        self.assertEqual([o["id"] for o in got["constraint_options"]],
                         ["known_baseline", "barometric_trend", "imu_attitude", "rtk_fix"])
        self.assertIsNone(got["ground_control_points"]["count_claimed"])
        self.assertTrue(got["requires_confirmation"])
        json.dumps(got, allow_nan=False)

    def test_a_noisier_than_the_motion_trace_says_scale_is_not_identifiable(self):
        samples = self.trace(h=25.0)
        quality = gnss.quality_report(samples, max_speed_m_s=6.0)
        observability = gnss.observability([s["position"] for s in samples], samples)
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability)
        self.assertEqual(got["scale_identifiable_from_gps_alone"]["status"], "not_supported")
        self.assertIn("known_baseline", got["recommended_minimum"]["constraint_ids"])
        self.assertIn("rtk_fix", got["recommended_minimum"]["constraint_ids"])
        self.assertTrue(got["scale_identifiable_from_gps_alone"]["reasons"])

    def test_a_fix_that_implies_an_impossible_speed_demotes_the_scale_estimate(self):
        samples = self.trace(curve=0.05)
        samples[5]["position"] = [900.0, 0.0, 0.0]
        quality = gnss.quality_report(samples, max_speed_m_s=6.0)
        observability = gnss.observability([s["position"] for s in samples], samples)
        self.assertGreater(quality["suspicious_count"], 0)
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability)
        self.assertEqual(got["scale_identifiable_from_gps_alone"]["status"], "weak")
        self.assertTrue(got["scale_identifiable_from_gps_alone"]["reasons"])

    def test_undeclared_fix_quality_is_reported_as_undeclared_not_as_precision(self):
        samples = self.trace(curve=0.05)
        quality = gnss.quality_report(samples, max_speed_m_s=6.0)
        observability = gnss.observability([s["position"] for s in samples], samples)
        refused = gnss.fix_quality_weights([None] * len(samples))
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability,
                                  fix_quality=refused)
        self.assertIs(got["inputs_supplied"]["fix_quality"], True)
        self.assertEqual(got["fix_quality"]["status"], "refused_no_quality_field")
        self.assertIn("rtk_fix", got["recommended_minimum"]["constraint_ids"])
        self.assertEqual(got["constraint_options"][3]["status"], "declare_or_obtain")
        absent = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability)
        self.assertIs(absent["inputs_supplied"]["fix_quality"], False)
        self.assertEqual(absent["fix_quality"]["status"], "not_supplied")

    def test_a_fixed_rtk_solution_still_needs_independent_evaluation_points(self):
        samples = self.trace(curve=0.05)
        quality = gnss.quality_report(samples, max_speed_m_s=6.0)
        observability = gnss.observability([s["position"] for s in samples], samples)
        fixed = gnss.fix_quality_weights(["rtk_fixed"] * len(samples), hdop=[0.9] * len(samples))
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability,
                                 fix_quality=fixed)
        self.assertEqual(got["scale_identifiable_from_gps_alone"]["status"], "supported")
        self.assertNotIn("rtk_fix", got["recommended_minimum"]["constraint_ids"])
        self.assertEqual(got["ground_control_points"]["for_reconstruction"],
                         "not assumed by this protocol")
        self.assertIs(got["ground_control_points"]["for_evaluation"]["required"], True)
        self.assertIs(got["claims_accuracy"], False)

    def test_an_unresolved_vertical_reference_demands_confirmation_of_the_datum(self):
        samples = self.trace(curve=0.05)
        quality = gnss.quality_report(samples, max_speed_m_s=6.0)
        observability = gnss.observability([s["position"] for s in samples], samples)
        vertical = gnss.vertical_reference_check(
            samples, {"altitude_datum": "ellipsoidal", "height_reference": "above_ground_level",
                      "source": "barometric"})
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability,
                                  vertical_reference=vertical)
        self.assertIs(got["vertical_reference"]["requires_confirmation"], True)
        self.assertEqual(got["vertical_reference"]["status"], "inconsistency_detected")
        self.assertIn("barometric_trend", got["recommended_minimum"]["constraint_ids"])
        self.assertIn("datum", " ".join(got["constraint_options"][1]["does_not_address"]).lower())

    def test_a_supplied_clock_bound_is_carried_into_the_answer(self):
        samples = self.trace()
        quality = gnss.quality_report(samples, max_speed_m_s=6.0)
        observability = gnss.observability([s["position"] for s in samples], samples)
        bounds = gnss.time_offset_bounds([s["t_sec"] for s in samples],
                                         [s["t_sec"] + 0.2 for s in samples], max_speed_m_s=6.0)
        got = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability,
                                  clock_bounds=bounds)
        self.assertEqual(got["clock_offset_and_lever_arm"]["bound_s"], bounds["offset_bounds_s"])
        self.assertEqual(got["clock_offset_and_lever_arm"]["worst_case_error_m"],
                         bounds["worst_case_along_track_error_m"])
        plain = acc.gcp_requirement(telemetry_quality=quality, trajectory_observability=observability)
        self.assertIsNone(plain["clock_offset_and_lever_arm"]["bound_s"])

    def test_forged_or_incomplete_upstream_reports_are_refused(self):
        quality, observability = self.straight_case()
        for kwargs in ({"telemetry_quality": {}, "trajectory_observability": observability},
                       {"telemetry_quality": dict(quality, provenance="x"),
                        "trajectory_observability": observability},
                       {"telemetry_quality": {"provenance": "survey_gnss.quality_report"},
                        "trajectory_observability": observability},
                       {"telemetry_quality": quality, "trajectory_observability": {}},
                       {"telemetry_quality": quality,
                        "trajectory_observability": dict(observability, provenance="x")}):
            with self.subTest(kwargs=sorted(kwargs)), self.assertRaises(ValueError):
                acc.gcp_requirement(**kwargs)
        broken = dict(quality)
        del broken["median_horizontal_std_m"]
        with self.assertRaises(ValueError):
            acc.gcp_requirement(telemetry_quality=broken, trajectory_observability=observability)
        for extra in ({"fix_quality": {"provenance": "survey_gnss.fix_quality_weights"}},
                      {"vertical_reference": {"provenance": "x"}},
                      {"clock_bounds": {"provenance": "survey_gnss.time_offset_bounds"}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                acc.gcp_requirement(telemetry_quality=quality,
                                    trajectory_observability=observability, **extra)


if __name__ == "__main__":
    unittest.main()
