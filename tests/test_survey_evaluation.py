"""CPU-only analytical evaluation tests; never import the reconstruction pipeline."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scripts import survey_evaluation as ev


class EvaluationTests(unittest.TestCase):
    def report(self, secs=899.9):
        return {"secs": secs, "status": "complete", "steps": [
            {"name": n, "status": "done", "secs": 1.0}
            for n in ("keyframes", "colmap", "poses", "train", "frame", "export")]}

    def test_checkpoint_analytical_errors_without_alignment(self):
        got = ev.checkpoint_metrics([[3, 4, 0], [0, 0, 12]], np.zeros((2, 3)))
        expected = {"count": 2, "horizontal_rmse_m": np.sqrt(12.5),
                    "vertical_rmse_m": np.sqrt(72), "rmse_3d_m": np.sqrt(84.5),
                    "median_3d_m": 8.5, "p95_3d_m": 11.65, "max_3d_m": 12}
        for key, value in expected.items():
            self.assertAlmostEqual(got[key], value)
        self.assertEqual(got["bias_xyz_m"], [1.5, 2.0, 6.0])
        self.assertEqual(got["alignment"], "none")

    def test_checkpoint_large_finite_reductions_and_strict_json(self):
        for value in (1e308, np.finfo(np.float64).max):
            for signs, bias in (([1, 1, 1, 1], value), ([1, 1, -1, -1], 0.0)):
                points = np.zeros((4, 3))
                points[:, 0] = np.array(signs) * value
                with self.subTest(value=value, signs=signs):
                    got = ev.checkpoint_metrics(points, np.zeros((4, 3)))
                    for key in ("horizontal_rmse_m", "rmse_3d_m", "median_3d_m", "p95_3d_m", "max_3d_m"):
                        self.assertEqual(got[key], value)
                    self.assertEqual(got["vertical_rmse_m"], 0.0)
                    self.assertEqual(got["bias_xyz_m"], [bias, 0.0, 0.0])
                    json.dumps(ev.build_evaluation(checkpoints=got), allow_nan=False)

    def test_checkpoint_unrepresentable_differences_or_distances_raise_valueerror(self):
        largest = np.finfo(np.float64).max
        for a, b in (([[largest, 0, 0]], [[-largest, 0, 0]]),
                     ([[-largest, 0, 0]], [[largest, 0, 0]]),
                     ([[largest, largest, 0]], [[0, 0, 0]])):
            with self.subTest(a=a), self.assertRaises(ValueError):
                ev.checkpoint_metrics(a, b)

    def test_points_reject_empty_nonfinite_shapes_and_unmatched_checkpoints(self):
        good = [[0, 0, 0]]
        for bad in ([], np.empty((0, 3)), [1, 2, 3], [[1, 2]],
                    [[0, np.nan, 0]], [[0, 0, np.inf]]):
            for fn in (ev.checkpoint_metrics, ev.surface_metrics):
                for args in ((bad, good), (good, bad)):
                    with self.subTest(fn=fn.__name__, args=args), self.assertRaises(ValueError):
                        fn(*args)
        with self.assertRaises(ValueError):
            ev.checkpoint_metrics(good, good * 2)

    def test_point_coverage_perfect_partial_asymmetric_and_block_boundaries(self):
        ref = [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0], [8, 0, 0]]
        cases = [(ref, ref, 1, 1, 1), (ref[:2], ref, 1, .4, 4 / 7),
                 (ref, ref[:2], .4, 1, 4 / 7), ([[20, 0, 0]], ref, 0, 0, 0)]
        for a, b, precision, recall, f1 in cases:
            for chunk in (1, 2, 3, 512):
                got = ev.surface_metrics(a, b, thresholds=(.1,), chunk_size=chunk)
                row = got["thresholds"][0]
                self.assertEqual(row["distance_m"], .1)
                for key, value in (("precision", precision), ("recall", recall), ("f1", f1)):
                    self.assertAlmostEqual(row[key], value)
                self.assertEqual((got["reconstruction_count"], got["reference_count"]), (len(a), len(b)))
                self.assertEqual(got["sampling"], "caller_supplied_points")
                self.assertEqual(got["scope"], "reference_points")
        self.assertEqual(ev.surface_metrics([[.1, 0, 0]], [[0, 0, 0]])["thresholds"][0]["f1"], 1)

    def test_surface_parameters_and_workload_limit(self):
        p = [[0, 0, 0]]
        for kwargs in ({"thresholds": ()}, {"thresholds": (0,)}, {"thresholds": (-1,)},
                       {"thresholds": (np.inf,)}, {"thresholds": (np.nan,)},
                       {"thresholds": (True,)}, {"chunk_size": True}, {"chunk_size": 0},
                       {"chunk_size": 1.5}, {"max_points": True}, {"max_points": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ev.surface_metrics(p, p, **kwargs)
        for a, b in ((p * 3, p), (p, p * 3)):
            with self.assertRaises(ValueError):
                ev.surface_metrics(a, b, max_points=2)
        with self.assertRaises(ValueError):
            ev.surface_metrics(np.zeros((10001, 3)), p)

    def test_speed_strict_gate_and_video_duration_qualification(self):
        for secs, official in ((899.9, "meets_target"), (900, "exceeds_target"), (901, "exceeds_target")):
            got = ev.speed_metrics(self.report(secs), 600)
            self.assertEqual(got["status"], "measured")
            self.assertEqual(got["official_status"], official)
            self.assertAlmostEqual(got["processing_ratio"], secs / 600)
        for duration in (60, 599.49, 600.51):
            got = ev.speed_metrics(self.report(), duration)
            self.assertEqual(got["official_status"], "not_evaluated")
            self.assertEqual(got["status"], "measured")
            self.assertAlmostEqual(got["processing_ratio"], 899.9 / duration)
        for duration in (599.5, 600.5):
            self.assertEqual(ev.speed_metrics(self.report(), duration)["official_status"], "meets_target")

    def test_speed_total_cannot_be_shorter_than_any_measured_stage(self):
        for optional in (False, True):
            report = self.report(secs=1)
            if optional:
                report["steps"].append({"name": "evaluation", "status": "done", "secs": 1000})
            else:
                report["steps"][3]["secs"] = 1000
            with self.subTest(optional=optional):
                got = ev.speed_metrics(report, 600)
                self.assertEqual(got["status"], "not_evaluated")
                self.assertEqual(got["official_status"], "not_evaluated")
                self.assertTrue(got["reason"])

    def test_speed_total_allows_only_tiny_rounding_difference(self):
        for elapsed, expected in ((1.0, "measured"), (1.0 - 1e-10, "measured"),
                                  (.99, "not_evaluated")):
            with self.subTest(elapsed=elapsed):
                self.assertEqual(ev.speed_metrics(self.report(elapsed), 600)["status"], expected)

    def test_speed_unrepresentable_processing_ratio_is_not_qualified(self):
        got = ev.speed_metrics(self.report(1e308), 1e-308)
        self.assertEqual(got["status"], "not_evaluated")
        self.assertEqual(got["official_status"], "not_evaluated")
        self.assertIsNone(got["processing_ratio"])
        json.dumps(ev.build_evaluation(speed=got), allow_nan=False)

    def test_speed_rejects_incomplete_cached_missing_and_invalid_timings(self):
        reports = []
        for status in ("skipped", "cached", "failed", "warning", "running"):
            report = self.report()
            report["steps"][0]["status"] = status
            reports.append(report)
        for field, value in (("steps", []), ("status", "partial"), ("status", "complete-with-warnings"),
                             ("secs", np.nan), ("secs", -1), ("secs", True)):
            reports.append({**self.report(), field: value})
        report = self.report()
        report["steps"][0].pop("secs")
        reports.extend([report, {}, {**self.report(), "steps": self.report()["steps"] * 2}])
        for report in reports:
            got = ev.speed_metrics(report, 600)
            self.assertEqual(got["status"], "not_evaluated")
            self.assertEqual(got["official_status"], "not_evaluated")
            self.assertTrue(got["reason"])
        for duration in (0, -1, np.nan, np.inf, True):
            self.assertEqual(ev.speed_metrics(self.report(), duration)["status"], "not_evaluated")

    def test_explicit_survey_stages_cannot_be_empty(self):
        report = {"secs": 20, "status": "complete", "steps": [{"name": "survey", "status": "done", "secs": 20}]}
        self.assertEqual(ev.speed_metrics(report, 600)["official_status"], "not_evaluated")
        self.assertEqual(ev.speed_metrics(report, 600, required_stages=["survey"])["official_status"], "meets_target")
        for stages in ([], "survey", ["survey", "survey"], [""]):
            with self.assertRaises(ValueError):
                ev.speed_metrics(report, 600, required_stages=stages)

    def test_six_criteria_unknown_honesty_and_no_inferred_accuracy(self):
        result = ev.build_evaluation()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual([c["id"] for c in result["criteria"]], ["accuracy", "completeness", "speed", "innovation", "scalability", "ui"])
        self.assertEqual([c["weight"] for c in result["criteria"]], [30, 20, 20, 15, 10, 5])
        self.assertEqual(sum(c["weight"] for c in ev.OFFICIAL_CRITERIA), 100)
        self.assertTrue(all(c["status"] == "not_evaluated" for c in result["criteria"]))
        self.assertNotIn("score", result)
        for distance in (1, 1.01):
            cp = ev.checkpoint_metrics([[distance, 0, 0]], [[0, 0, 0]])
            surface = ev.surface_metrics([[0, 0, 0]], [[0, 0, 0]])
            result = ev.build_evaluation(checkpoints=cp, surface=surface, speed=ev.speed_metrics(self.report(), 600), georeferenced=True)
            criteria = {c["id"]: c for c in result["criteria"]}
            self.assertEqual(criteria["accuracy"]["status"], "measured")
            self.assertEqual(criteria["accuracy"]["max_error_within_1m"], distance <= 1)
            self.assertEqual(criteria["completeness"]["status"], "measured")
            self.assertEqual(criteria["completeness"]["metrics"]["scope"], "reference_points")
            self.assertEqual(criteria["speed"]["status"], "meets_target")
            self.assertNotIn("max_error_within_1m", ev.build_evaluation(checkpoints=cp)["criteria"][0])
            json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            ev.build_evaluation(checkpoints={"telemetry_residual_m": .01}, georeferenced=True)
        speed = ev.speed_metrics({**self.report(), "telemetry_residual_m": .01}, 600)
        self.assertEqual(ev.build_evaluation(speed=speed)["criteria"][0]["status"], "not_evaluated")

    def test_evaluation_rejects_bare_forged_speed_status(self):
        with self.assertRaises(ValueError):
            ev.build_evaluation(speed={"official_status": "meets_target"})

    def test_evaluation_requires_complete_speed_schema(self):
        speed = ev.speed_metrics(self.report(), 600)
        for key in speed:
            incomplete = dict(speed)
            del incomplete[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                ev.build_evaluation(speed=incomplete)

    def test_evaluation_rejects_inconsistent_speed_gate_claims(self):
        speed = ev.speed_metrics(self.report(), 600)
        changes = [dict(elapsed_s=900, processing_ratio=1.5),
                   dict(video_duration_s=60, processing_ratio=899.9 / 60),
                   dict(official_status="exceeds_target"), dict(status="not_evaluated"),
                   dict(official_target_s=1000), dict(official_video_duration_s=60),
                   dict(processing_ratio=0), dict(elapsed_s=True), dict(video_duration_s=np.inf),
                   dict(required_stages=[]), dict(required_stages=["train", "train"]),
                   dict(required_stages=[""]), dict(reason=None)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ev.build_evaluation(speed={**speed, **change})

    def test_evaluation_keeps_non600_and_unqualified_runs_unofficial(self):
        for report, duration, expected in ((self.report(), 60, "measured"),
                                           ({}, 600, "not_evaluated"),
                                           (self.report(), 0, "not_evaluated"),
                                           (self.report(900), 600, "exceeds_target")):
            got = ev.build_evaluation(speed=ev.speed_metrics(report, duration))
            self.assertEqual(got["criteria"][2]["status"], expected)
            json.dumps(got, allow_nan=False)

    def test_streaming_fingerprint_detects_same_size_changes_without_path_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            for data in (b"a" * (1024 * 1024 + 1), b"b" * (1024 * 1024 + 1), b""):
                path.write_bytes(data)
                self.assertEqual(ev.file_fingerprint(path), {"name": "sample.bin", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})


if __name__ == "__main__":
    unittest.main()
