import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from plyfile import PlyData, PlyElement
import survey_workflow as survey


class SurveyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "videos" / "flight"
        self.source.mkdir(parents=True)
        (self.source / "flight.mp4").write_bytes(b"test-video-not-decoded")
        self.metadata = {
            "schema_version": 1, "time_reference": "video", "time_offset_s": 0,
            "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
            "single_pass": True, "video_duration_s": 600,
        }
        self.csv = (
            "t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m\n"
            "0,28,77,100,1,2\n1,28.0001,77,100,1,2\n"
            "2,28.0001,77.0001,101,1,2\n3,28,77.0001,102,1,2\n"
        )

    def save(self):
        return survey.save_inputs(self.root, "flight", self.csv, self.metadata)

    def test_missing_evidence_is_unknown_not_success(self):
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "not_prepared")
        self.assertFalse(state["gpu_execution_enabled"])
        self.assertEqual(sum(c["weight"] for c in state["evaluation"]["criteria"]), 100)
        self.assertTrue(all(c["status"] == "not_evaluated" for c in state["evaluation"]["criteria"]))

    def test_inputs_validate_before_saving_and_refuse_overwrite(self):
        bad = dict(self.metadata, altitude_datum="unknown")
        with self.assertRaises(ValueError):
            survey.save_inputs(self.root, "flight", self.csv, bad)
        self.assertFalse((self.source / "telemetry.csv").exists())
        self.save()
        with self.assertRaises(FileExistsError):
            self.save()

    def test_prepare_records_hashes_without_reconstruction(self):
        self.save()
        result = survey.prepare_scene(self.root, "flight")
        self.assertEqual(result["status"], "prepared")
        manifest = json.loads((self.root / "work/flight/survey/preparation.json").read_text())
        self.assertEqual(len(manifest["inputs"]), 3)
        self.assertTrue(all(len(i["sha256"]) == 64 for i in manifest["inputs"]))
        self.assertFalse(manifest["video_duration_verified"])
        self.assertFalse((self.root / "work/flight/splat.ply").exists())
        self.assertTrue(any(c["requires_gpu"] for c in result["commands"]))

    def test_source_changes_invalidate_preparation(self):
        self.save()
        survey.prepare_scene(self.root, "flight")
        (self.source / "telemetry.csv").write_text(self.csv.replace("100,1,2", "101,1,2"))
        self.assertEqual(survey.scene_status(self.root, "flight")["status"], "invalid")
        with self.assertRaisesRegex(ValueError, "changed"):
            survey.align_scene(self.root, "flight")

    def test_multiple_videos_not_single_pass(self):
        self.save()
        (self.source / "second.mp4").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "one video"):
            survey.prepare_scene(self.root, "flight")

    def test_scene_paths_reject_traversal(self):
        for name in ("../flight", "a/b", "a\\b", "", "..", "C:flight"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                survey.scene_status(self.root, name)

    def test_alignment_requires_real_registered_cameras(self):
        self.save()
        survey.prepare_scene(self.root, "flight")
        with self.assertRaisesRegex(ValueError, "camera"):
            survey.align_scene(self.root, "flight")

    def test_evaluation_does_not_promote_telemetry_fit(self):
        self.save()
        survey.prepare_scene(self.root, "flight")
        result = survey.evaluate_scene(self.root, "flight")
        accuracy = next(c for c in result["evaluation"]["criteria"] if c["id"] == "accuracy")
        self.assertEqual(accuracy["status"], "not_evaluated")

    def test_gpu_runner_refuses_before_work_or_subprocess(self):
        with self.assertRaisesRegex(PermissionError, "GPU"):
            survey.reconstruct_scene(self.root, "flight", allow_gpu=False)
        self.assertFalse((self.root / "work").exists())

    def test_dense_command_plan_uses_argv_and_geometry_outputs(self):
        commands = survey.dense_commands(self.root, self.root / "work/flight", self.root / "dense")
        self.assertEqual([c["stage"] for c in commands], ["undistort", "dense", "fusion"])
        self.assertTrue(all(isinstance(c["argv"], list) for c in commands))
        self.assertTrue(commands[1]["requires_gpu"])
        self.assertIn("fused.ply", commands[-1]["argv"][-1])

    def prepared_geometry(self):
        self.save()
        survey.prepare_scene(self.root, "flight")
        self.work = self.root / "work/flight"
        self.preparation = survey.read_json(self.work / "survey/preparation.json")
        self.write_geometry(self.work)
        return survey.align_scene(self.root, "flight")

    def write_geometry(self, workspace, scale=2.0):
        rows = []
        for i, sample in enumerate(self.preparation["telemetry"]["samples"]):
            center = (np.asarray(sample["position"]) - [10, 20, 30]) / scale
            rows.append({"file": f"{i}.jpg", "t_sec": sample["t_sec"],
                         "camera": {"R_rowmajor": np.eye(3).ravel().tolist(), "t": (-center).tolist()}})
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "keyframes_poses.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        sparse = workspace / "colmap/sparse/txt/points3D.txt"
        sparse.parent.mkdir(parents=True, exist_ok=True)
        sparse.write_text("# tiny COLMAP points\n1 0 0 0 10 20 30 0.1\n2 1 2 3 40 50 60 0.1\n")

    def references(self, frame):
        pairs = {"independent": True, "alignment": "none", "coordinate_frame": frame,
                 "checkpoints": [{"reconstructed": [10, 20, 30], "reference": [10.1, 20, 30]}]}
        surface = {"independent": True, "coordinate_frame": frame, "visible_reference": True,
                   "reconstructed": [[10, 20, 30], [12, 24, 36]],
                   "reference": [[10.1, 20, 30], [12, 24, 36]]}
        survey.write_json(self.work / "survey/checkpoints.json", pairs)
        survey.write_json(self.work / "survey/surface_reference.json", surface)
        return surface

    def test_golden_alignment_checkpoint_surface_and_stale_reference(self):
        state = self.prepared_geometry()
        self.assertAlmostEqual(state["alignment"]["scale"], 2)
        cloud = PlyData.read(str(self.work / "survey/sparse_points.ply"))["vertex"].data
        np.testing.assert_allclose([cloud["x"][0], cloud["y"][0], cloud["z"][0]], [10, 20, 30])
        self.references(state["alignment"]["coordinate_frame"])
        state = survey.evaluate_scene(self.root, "flight")
        criteria = {c["id"]: c for c in state["evaluation"]["criteria"]}
        self.assertAlmostEqual(criteria["accuracy"]["metrics"]["rmse_3d_m"], .1)
        self.assertEqual(criteria["completeness"]["status"], "measured")
        self.assertEqual(criteria["completeness"]["metrics"]["scope"], "reference_points")
        self.assertIn("not surface-area", criteria["completeness"]["reason"])
        result = state["evaluation"]
        self.assertEqual(result["surface_source"], "survey/surface_reference.json")
        self.assertEqual(len(result["surface_sha256"]), 64)
        self.assertIsNone(result["source_run_id"])
        path = self.work / "survey/surface_reference.json"
        path.write_text(path.read_text() + " ")
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("surface_reference.json", " ".join(state["blockers"]))
        self.assertFalse(any(a["name"] == "evaluation.json" for a in state["artifacts"]))

    def test_surface_contract_rejects_bad_declarations_frame_and_limits(self):
        state = self.prepared_geometry()
        good = self.references(state["alignment"]["coordinate_frame"])
        for changes in ({"independent": False}, {"visible_reference": False},
                        {"coordinate_frame": {}}, {"reconstructed": [[0, 0, 0]] * 10001},
                        {"reference": [[0, 0, 0]] * 10001}, {"reference": []},
                        {"extra": True}, {"reference": [[0, 1]]}):
            with self.subTest(changes=list(changes)):
                survey.write_json(self.work / "survey/surface_reference.json", dict(good, **changes))
                with self.assertRaises(ValueError):
                    survey.evaluate_scene(self.root, "flight")

    def test_sparse_mutation_removes_stale_artifact(self):
        self.prepared_geometry()
        sparse = self.work / "colmap/sparse/txt/points3D.txt"
        sparse.write_text(sparse.read_text() + "3 2 2 2 0 0 0 0.1\n")
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("Sparse", " ".join(state["blockers"]))
        self.assertFalse(any(a["name"] == "sparse_points.ply" for a in state["artifacts"]))
        with self.assertRaisesRegex(ValueError, "Sparse"):
            survey.evaluate_scene(self.root, "flight")

    def test_export_rejects_bad_arrays_without_replacing_valid_cloud(self):
        self.prepared_geometry()
        alignment = survey.read_json(self.work / "survey/georeference.json")
        output = self.work / "survey/sparse_points.ply"
        before = output.read_bytes()
        for xyz, rgb in (([[0, 0, 0], [1, 1, 1]], [[0, 0, 0]]),
                         ([[0, 0, 0]], [[-1, 0, 0]]), ([[0, 0, 0]], [[256, 0, 0]]),
                         ([[0, 0, 0]], [[float("nan"), 0, 0]]),
                         ([[0, 0, 0]], [[1.5, 0, 0]]), ([[0, 0]], [[0, 0, 0]]),
                         ([[float("inf"), 0, 0]], [[0, 0, 0]])):
            with self.subTest(xyz=xyz, rgb=rgb), self.assertRaises(ValueError):
                survey._export_points(xyz, rgb, alignment, output)
            self.assertEqual(output.read_bytes(), before)
        occupied = output.with_suffix(".tmp")
        occupied.write_text("owned by another export")
        survey._export_points([[0, 0, 0]], [[0, 0, 0]], alignment, output)
        self.assertEqual(occupied.read_text(), "owned by another export")

    def fake_reconstruction(self, *, width=1920, height=1080, fail_stage=None, omit_cloud=False):
        import cv2
        executable = self.root / "tools/colmap/bin/colmap.exe"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"never executed")
        scripts = self.root / "scripts"
        scripts.mkdir(exist_ok=True)
        for name in ("extract_keyframes.py", "run_colmap.py", "parse_colmap.py",
                     "survey_priors.py"):
            (scripts / name).write_text("# fake runner fixture; never executed\n")
        calls = []
        self.spawned = []

        def runner(argv, **kwargs):
            stage = Path(kwargs["stdout"].name).stem
            run = Path(kwargs["stdout"].name).parent
            calls.append((stage, run))
            self.spawned.append([str(part) for part in argv])
            if stage == "poses":
                self.write_geometry(run, scale=3)
            if stage == "fusion" and not omit_cloud:
                dense = run / "dense"
                dense.mkdir(exist_ok=True)
                vertices = np.array([(0, 0, 0, 10, 20, 30), (1, 2, 3, 40, 50, 60)],
                                    dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
                PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(str(dense / "fused.ply"))
            return SimpleNamespace(returncode=7 if stage == fail_stage else 0)

        metadata = {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 18000,
                    cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height}
        cap = SimpleNamespace(get=lambda key: metadata[key], release=lambda: None, isOpened=lambda: True)
        self.addCleanup(patch.stopall)
        patch("cv2.VideoCapture", return_value=cap).start()
        patch.object(survey.subprocess, "run", side_effect=runner).start()
        return calls

    def test_complete_run_is_current_for_display_alignment_and_evaluation(self):
        self.prepared_geometry()
        calls = self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        run = self.work / "survey/runs" / record["id"]
        self.assertEqual([c[0] for c in calls], ["keyframes", "priors", "colmap", "poses", "undistort", "dense", "fusion"])
        colmap = next(argv for argv in self.spawned if argv[1].endswith("run_colmap.py"))
        self.assertIn("mapper=pose_prior", colmap)
        priors = next(argv for argv in self.spawned if argv[1].endswith("survey_priors.py"))
        self.assertIn("pose_priors.jsonl", priors[-1])
        self.assertTrue(all(c[1] == run for c in calls))
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state.get("latest_run", {}).get("id"), record["id"])
        self.assertAlmostEqual(state["alignment"]["scale"], 3)
        for name in ("dense_points.ply", "georeference.json", "run.json"):
            artifact = next(a for a in state["artifacts"] if a["name"] == name)
            self.assertEqual(set(artifact), {"name", "url", "kind"})
            self.assertIn("/runs/" + record["id"] + "/", artifact["url"])
        self.assertFalse(any(a["name"] == "sparse_points.ply" for a in state["artifacts"]))
        self.assertAlmostEqual(survey.align_scene(self.root, "flight")["alignment"]["scale"], 3)
        # A distinct, valid current frame proves evaluation does not consult legacy georeference.
        legacy = survey.read_json(self.work / "survey/georeference.json")
        legacy["coordinate_frame"]["origin"]["altitude_m"] += 1
        survey.write_json(self.work / "survey/georeference.json", legacy)
        self.references(state["alignment"]["coordinate_frame"])
        state = survey.evaluate_scene(self.root, "flight")
        result = state["evaluation"]
        self.assertEqual(result["source_run_id"], record["id"])
        self.assertEqual(result["report_source"], f"survey/runs/{record['id']}/run.json")
        self.assertEqual(result["georeference_source"], f"survey/runs/{record['id']}/georeference.json")
        self.assertEqual(result["report_sha256"], survey._modules()[0].file_fingerprint(run / "run.json")["sha256"])
        self.assertEqual(record["hardware"]["status"], "recorded")
        self.assertFalse(record["benchmark_qualified"])
        self.assertFalse(record["speed"]["diagnostic_only"])
        self.assertEqual(record["speed"]["official_status"], "meets_target")
        self.assertEqual(record["video_duration_s"], 600)
        self.assertGreaterEqual(record["secs"], sum(s["secs"] for s in record["steps"]))
        self.assertEqual(record["steps"][0]["name"], "setup")
        self.assertTrue(record["code_sha256"])
        self.assertTrue(all(len(v) == 64 for v in record["code_sha256"].values()))
        cloud = PlyData.read(str(run / "dense_points.ply"))["vertex"].data
        np.testing.assert_allclose([cloud["x"][1], cloud["y"][1], cloud["z"][1]], [13, 26, 39])
        (run / "dense_points.ply").write_bytes(b"changed")
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertFalse(any(a["name"] == "dense_points.ply" for a in state["artifacts"]))
        with self.assertRaisesRegex(ValueError, "dense_points"):
            survey.evaluate_scene(self.root, "flight")

    def test_failed_latest_run_is_visible_and_never_promotes_old_dense_evidence(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        first = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        patch.stopall()
        self.fake_reconstruction(fail_stage="dense")
        with self.assertRaisesRegex(RuntimeError, "dense failed"):
            survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state.get("latest_run", {}).get("status"), "failed")
        self.assertNotEqual(state["latest_run"]["id"], first["id"])
        self.assertIn("FAILED", " ".join(state["blockers"]))
        self.assertFalse(any(a["name"] == "dense_points.ply" for a in state["artifacts"]))
        self.assertAlmostEqual(state["alignment"]["scale"], 2)
        self.references(state["alignment"]["coordinate_frame"])
        result = survey.evaluate_scene(self.root, "flight")["evaluation"]
        self.assertIsNone(result["source_run_id"])
        self.assertIsNone(result["report_source"])

    def test_run_rejects_unverified_and_sub_1080p_resolution(self):
        self.prepared_geometry()
        for width, height in ((1280, 720), (1920, 800), (0, 1080), (float("nan"), 1080)):
            with self.subTest(width=width, height=height):
                calls = self.fake_reconstruction(width=width, height=height)
                with self.assertRaisesRegex(ValueError, "1080p|resolution"):
                    survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
                self.assertFalse(calls)
                patch.stopall()

    def test_run_missing_fused_output_fails_without_old_cloud(self):
        self.prepared_geometry()
        self.fake_reconstruction(omit_cloud=True)
        with self.assertRaises((OSError, ValueError)):
            survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state.get("latest_run", {}).get("status"), "failed")
        self.assertFalse(any(a["name"] == "dense_points.ply" for a in state["artifacts"]))

    def test_latest_run_rejects_unsafe_ids_stale_preparation_and_missing_files(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        pointer_path = self.work / "survey/latest_run.json"
        pointer = survey.read_json(pointer_path)
        for changes in ({"id": "../outside"}, {"id": record["id"] + "/../" + record["id"]},
                        {"id": "not-an-exact-run-id"}, {"preparation_id": "stale"}):
            with self.subTest(changes=changes):
                survey.write_json(pointer_path, dict(pointer, **changes))
                self.assertEqual(survey.scene_status(self.root, "flight")["status"], "invalid")
                with self.assertRaises(ValueError):
                    survey.evaluate_scene(self.root, "flight")
        survey.write_json(pointer_path, pointer)
        (self.work / "survey/runs" / record["id"] / "georeference.json").unlink()
        self.assertEqual(survey.scene_status(self.root, "flight")["status"], "invalid")

    def test_status_uses_stat_but_action_rehashes_cloud(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        path = self.work / "survey/runs" / record["id"] / "dense_points.ply"
        before = path.stat()
        content = path.read_bytes()
        path.write_bytes(content.replace(b"13.000000000", b"14.000000000"))
        self.assertEqual(path.stat().st_size, before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        # Polling must not rehash large content, even when stat-preserving changes exist.
        self.assertEqual(survey.scene_status(self.root, "flight")["status"], "aligned")
        with self.assertRaisesRegex(ValueError, "dense_points"):
            survey.evaluate_scene(self.root, "flight")

    def test_failed_run_warning_survives_stale_legacy_geometry(self):
        self.prepared_geometry()
        self.fake_reconstruction(fail_stage="keyframes")
        with self.assertRaises(RuntimeError):
            survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        (self.work / "colmap/sparse/txt/points3D.txt").unlink()
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertEqual(state.get("latest_run", {}).get("status"), "failed")
        self.assertIn("FAILED", " ".join(state["blockers"]))

    def test_approval_requires_literal_true_before_scene_validation(self):
        for approval in (False, None, 1, "false"):
            with self.subTest(approval=approval), self.assertRaises(PermissionError):
                survey.reconstruct_scene(self.root, "../invalid", allow_gpu=approval)
        self.assertFalse((self.root / "work").exists())

    def test_report_hash_and_identity_validation(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        run = self.work / "survey/runs" / record["id"]
        pointer = survey.read_json(self.work / "survey/latest_run.json")
        report_path = run / "run.json"
        before = report_path.stat()
        payload = report_path.read_bytes()
        report_path.write_bytes(payload.replace(b'"point_count": 2', b'"point_count": 9'))
        os.utime(report_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        # run.json is a KB-scale survey JSON, so a poll rehashes it instead of
        # presenting a timing/provenance verdict that no longer matches its bytes.
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("run.json", " ".join(state["blockers"]))
        with self.assertRaisesRegex(ValueError, "run.json"):
            survey.evaluate_scene(self.root, "flight")
        for changes in ({"preparation_id": "old"}, {"id": "20200101T000000-00000000"},
                        {"status": "failed"}, {"inputs": []}):
            with self.subTest(changes=changes):
                survey.write_json(report_path, dict(record, **changes))
                pointer["report"] = survey._fingerprint(report_path, run)
                survey.write_json(self.work / "survey/latest_run.json", pointer)
                self.assertEqual(survey.scene_status(self.root, "flight")["status"], "invalid")

    def test_latest_run_directory_cannot_escape_or_alias(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        run = self.work / "survey/runs" / record["id"]
        moved = self.root / "moved-run"
        run.rename(moved)
        try:
            run.symlink_to(moved, target_is_directory=True)
        except OSError as error:
            self.skipTest("Directory symlinks unavailable: " + str(error))
        self.assertEqual(survey.scene_status(self.root, "flight")["status"], "invalid")
        with self.assertRaisesRegex(ValueError, "Unsafe|isolated"):
            survey.evaluate_scene(self.root, "flight")

    def test_status_does_not_hash_video_and_actions_detect_hidden_source_edits(self):
        self.prepared_geometry()
        video = self.source / "flight.mp4"
        before = video.stat()
        video.write_bytes(b"X" * before.st_size)
        os.utime(video, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(survey.scene_status(self.root, "flight")["status"], "aligned")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            survey.evaluate_scene(self.root, "flight")

    def test_checkpoint_contract_and_reference_invalidation(self):
        state = self.prepared_geometry()
        self.references(state["alignment"]["coordinate_frame"])
        path = self.work / "survey/checkpoints.json"
        good = survey.read_json(path)
        for changes in ({"independent": False}, {"alignment": "sim3"}, {"coordinate_frame": {}},
                        {"checkpoints": [{}]}, {"checkpoints": []}, {"extra": True}):
            with self.subTest(changes=changes):
                survey.write_json(path, dict(good, **changes))
                with self.assertRaises(ValueError):
                    survey.evaluate_scene(self.root, "flight")
        survey.write_json(path, good)
        survey.evaluate_scene(self.root, "flight")
        path.unlink()
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("checkpoints.json", " ".join(state["blockers"]))

    def test_new_valid_run_is_visible_even_when_previous_evaluation_is_stale(self):
        state = self.prepared_geometry()
        self.references(state["alignment"]["coordinate_frame"])
        survey.evaluate_scene(self.root, "flight")
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("Evaluation source run changed", " ".join(state["blockers"]))
        artifacts = {a["name"]: a for a in state["artifacts"]}
        self.assertIn("dense_points.ply", artifacts)
        self.assertIn(record["id"], artifacts["dense_points.ply"]["url"])
        self.assertNotIn("evaluation.json", artifacts)
        self.assertEqual(survey.evaluate_scene(self.root, "flight")["status"], "evaluated")

    def test_legacy_report_remains_diagnostic_and_stale_hash_is_rejected(self):
        self.prepared_geometry()
        report = self.work / "report.json"
        survey.write_json(report, {"secs": 12, "steps": [{"name": "keyframes", "status": "done", "secs": 12}]})
        state = survey.evaluate_scene(self.root, "flight")
        self.assertEqual(state["evaluation"]["report_source"], "report.json")
        speed = next(c for c in state["evaluation"]["criteria"] if c["id"] == "speed")
        self.assertEqual(speed["status"], "not_evaluated")
        self.assertIn("diagnostic", speed["metrics"]["reason"])
        report.write_text(report.read_text() + " ")
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("report.json", " ".join(state["blockers"]))

    def test_points3d_relative_path_uses_one_constant(self):
        self.prepared_geometry()
        self.assertEqual(survey.POINTS3D, "colmap/sparse/txt/points3D.txt")
        self.assertIn(survey.POINTS3D, survey.RUN_FILES)
        self.assertTrue((self.work / survey.POINTS3D).is_file())
        georeference = survey.read_json(self.work / "survey/georeference.json")
        self.assertEqual(georeference["sources"]["sparse"]["relative_path"], survey.POINTS3D)

    def test_truncated_points3d_row_reports_the_row_not_a_path(self):
        sparse = self.root / "work/flight/colmap/sparse/txt/points3D.txt"
        sparse.parent.mkdir(parents=True, exist_ok=True)
        for text in ("1 0 0 0 10 20 30 0.1\n2 1 2 3 40 50\n", "1 0 0 0 10 20\n",
                     "1 0 0\n2 1 2 3 4 5 6 0.1\n", "1 0 0 0 ten 20 30 0.1\n"):
            sparse.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError) as caught:
                survey._read_sparse_points(sparse, survey.POINTS3D)
            message = str(caught.exception)
            self.assertIn("row", message)
            self.assertNotIn(str(self.root), message)
            self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", message), message)
        self.prepared_geometry()
        sparse.write_text("1 0 0 0 10 20\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "row"):
            survey.align_scene(self.root, "flight")

    def test_evidence_artifacts_require_verified_provenance(self):
        state = self.prepared_geometry()
        evidence = self.work / "survey/evidence"
        self.assertTrue((evidence / "evidence_points.ply").is_file())
        self.assertTrue((evidence / "evidence_summary.json").is_file())
        names = [a["name"] for a in state["artifacts"]]
        self.assertIn("evidence_points.ply", names)
        self.assertIn("evidence_summary.json", names)
        # A re-align without the sparse model keeps the old ENU clouds on disk: they
        # are stale evidence for a coordinate frame nothing current attests to.
        (self.work / survey.POINTS3D).unlink()
        survey.align_scene(self.root, "flight")
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "aligned")
        self.assertTrue((evidence / "evidence_points.ply").is_file())
        names = [a["name"] for a in state["artifacts"]]
        self.assertNotIn("evidence_points.ply", names)
        self.assertNotIn("evidence_summary.json", names)
        self.assertNotIn("sparse_points.ply", names)

    def test_tampered_evidence_is_rejected_and_realign_restores_it(self):
        self.prepared_geometry()
        cloud = self.work / "survey/evidence/evidence_points.ply"
        summary = self.work / "survey/evidence/evidence_summary.json"
        for path in (cloud, summary):
            before = path.stat()
            path.write_bytes(b"X" * before.st_size)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.subTest(evidence=path.name), self.assertRaisesRegex(ValueError, "Evidence"):
                survey.evaluate_scene(self.root, "flight")
            survey.align_scene(self.root, "flight")
            self.assertEqual(survey.scene_status(self.root, "flight")["status"], "aligned")

    def test_tampered_evaluation_cannot_change_reported_criteria(self):
        self.prepared_geometry()
        survey.evaluate_scene(self.root, "flight")
        path = self.work / "survey/evaluation.json"
        data = survey.read_json(path)
        for criterion in data["criteria"]:
            criterion.update(status="meets_target", official_status="meets_target",
                             reason="hand-edited claim of an official pass")
        survey.write_json(path, data)
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "evaluated")
        self.assertNotIn("meets_target", json.dumps(state["evaluation"]["criteria"]))
        self.assertEqual(state["evaluation"]["criteria"],
                         survey._modules()[0].build_evaluation()["criteria"])

    def test_claimed_accuracy_pass_is_recomputed_from_the_checkpoint_evidence(self):
        state = self.prepared_geometry()
        self.references(state["alignment"]["coordinate_frame"])
        state = survey.evaluate_scene(self.root, "flight")
        accuracy = next(c for c in state["evaluation"]["criteria"] if c["id"] == "accuracy")
        self.assertEqual(accuracy["status"], "measured")
        self.assertAlmostEqual(accuracy["metrics"]["rmse_3d_m"], .1)
        path = self.work / "survey/evaluation.json"
        data = survey.read_json(path)
        data["criteria"][0].update(status="meets_target", official_pass=True,
                                   metrics={"rmse_3d_m": 0.001})
        survey.write_json(path, data)
        accuracy = next(c for c in survey.scene_status(self.root, "flight")["evaluation"]["criteria"]
                        if c["id"] == "accuracy")
        self.assertEqual(accuracy["status"], "measured")
        self.assertNotIn("official_pass", accuracy)
        self.assertAlmostEqual(accuracy["metrics"]["rmse_3d_m"], .1)

    def test_stat_preserving_checkpoint_edit_is_detected_on_a_poll(self):
        state = self.prepared_geometry()
        self.references(state["alignment"]["coordinate_frame"])
        self.assertEqual(survey.evaluate_scene(self.root, "flight")["status"], "evaluated")
        path = self.work / "survey/checkpoints.json"
        original = path.read_bytes()
        tampered = original.replace(b"10.1", b"10.2")
        self.assertEqual(len(tampered), len(original))
        before = path.stat()
        path.write_bytes(tampered)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("checkpoints.json", " ".join(state["blockers"]))

    def test_stat_preserving_georeference_edit_is_detected_on_a_poll(self):
        self.prepared_geometry()
        survey.evaluate_scene(self.root, "flight")
        path = self.work / "survey/georeference.json"
        original = path.read_bytes()
        # Nudge the fitted scale in place: a different transform, same bytes, and
        # nothing on the poll path recomputes a point cloud, so only the digest
        # of georeference.json itself can catch it.
        found = re.search(rb'"scale": (\d+\.\d+)', original)
        self.assertIsNotNone(found)
        token = found.group(1)
        changed = token[:-1] + (b"1" if token.endswith(b"0") else b"0")
        self.assertEqual(len(changed), len(token))
        self.assertNotEqual(float(changed), float(token))
        tampered = original[:found.start(1)] + changed + original[found.end(1):]
        self.assertEqual(len(tampered), len(original))
        before = path.stat()
        path.write_bytes(tampered)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        state = survey.scene_status(self.root, "flight")
        self.assertEqual(state["status"], "invalid")
        self.assertIn("georeference.json", " ".join(state["blockers"]))

    def test_run_records_host_identity_without_probing_the_gpu(self):
        self.prepared_geometry()
        calls = self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        hardware = record["hardware"]
        self.assertEqual(hardware["status"], "recorded")
        self.assertTrue(hardware["cpu"].strip())
        self.assertGreater(hardware["logical_cores"], 0)
        self.assertTrue(hardware["total_ram_bytes"] is None or hardware["total_ram_bytes"] > 0)
        self.assertEqual(hardware["gpu"], "not_recorded_gpu")
        spawned = " ".join(" ".join(argv) for argv in self.spawned).lower()
        self.assertNotIn("nvidia", spawned)
        self.assertNotIn("torch", spawned)
        self.assertEqual([c[0] for c in calls],
                         ["keyframes", "priors", "colmap", "poses", "undistort", "dense", "fusion"])

    def test_speed_gate_fires_only_on_a_verified_duration(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        self.assertEqual(record["video_duration_s"], 600)
        self.assertEqual(record["speed"]["status"], "measured")
        self.assertEqual(record["speed"]["official_status"], "meets_target")
        self.assertFalse(record["speed"]["diagnostic_only"])
        self.assertEqual(record["speed"]["required_stages"],
                         ["setup", "keyframes", "priors", "colmap", "poses", "undistort", "dense",
                          "fusion", "georeferenced_export"])
        criteria = {c["id"]: c for c in survey.evaluate_scene(self.root, "flight")["evaluation"]["criteria"]}
        self.assertEqual(criteria["speed"]["status"], "meets_target")
        self.assertEqual(criteria["accuracy"]["status"], "not_evaluated")
        self.assertFalse(criteria["speed"].get("metrics", {}).get("diagnostic_only"))


    def test_run_records_evidence_and_says_occlusion_was_not_checked(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        run = self.work / "survey/runs" / record["id"]
        self.assertTrue((run / "evidence/evidence_points.ply").is_file())
        self.assertIs(record["evidence"]["occlusion_checked"], False)
        self.assertIn("evidence/evidence_summary.json", record["files"])
        state = survey.scene_status(self.root, "flight")
        self.assertIn("evidence_points.ply", [a["name"] for a in state["artifacts"]])
        # With depth maps present the same call must claim occlusion checking.
        views = [{"K": np.eye(3) * 100.0, "viewmat": np.eye(4),
                  "depth": np.full((9, 9), 5.0, np.float32)}]
        with patch.object(survey, "_depth_views", return_value=views):
            record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        run = self.work / "survey/runs" / record["id"]
        self.assertIs(record["evidence"]["occlusion_checked"], True)
        self.assertGreaterEqual(record["evidence"]["visibility"]["views_after"], 0)

    def test_status_reports_crs_datum_and_operator_measurements(self):
        self.prepared_geometry()
        state = survey.scene_status(self.root, "flight")
        self.assertIn("EPSG:4979", state["crs"])
        self.assertEqual(state["vertical_datum"], "ellipsoidal")
        self.assertEqual(state["position_reference"], "camera_center")
        self.assertNotIn("measurements", state)
        preparation = json.loads((self.work / "survey/preparation.json").read_text(encoding="utf-8"))
        rows = [{"kind": "height", "value_m": 12.4, "uncertainty_m": 0.2, "valid": True}]
        (self.work / "survey/measurements.json").write_text(json.dumps(
            {"preparation_id": preparation["id"], "measurements": rows}), encoding="utf-8")
        self.assertEqual(survey.scene_status(self.root, "flight")["measurements"], rows)
        (self.work / "survey/measurements.json").write_text(json.dumps(
            {"preparation_id": "elsewhere", "measurements": rows}), encoding="utf-8")
        stale = survey.scene_status(self.root, "flight")
        self.assertEqual(stale["status"], "invalid")
        self.assertIn("different preparation", " ".join(stale["blockers"]))

    def test_complete_run_publishes_the_format_deliverables(self):
        self.prepared_geometry()
        self.fake_reconstruction()
        record = survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        run = self.work / "survey/runs" / record["id"]
        manifest = json.loads((run / "products" / "export_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["claims_textured_mesh"], False)
        written = {entry["format"] for entry in manifest["files"]}
        self.assertIn("las", written)
        self.assertIn("gltf", written)
        self.assertEqual({e["format"] for e in manifest["not_delivered"]} - {"obj", "geotiff"}, set())
        state = survey.scene_status(self.root, "flight")
        self.assertIn("cloud.las", [a["name"] for a in state["artifacts"]])
        # Tampering with a deliverable must invalidate the scene, not just the file.
        target = run / "products" / "cloud.las"
        target.write_bytes(target.read_bytes() + b"x")
        tampered = survey.scene_status(self.root, "flight")
        self.assertEqual(tampered["status"], "invalid")
        self.assertIn("cloud.las", " ".join(tampered["blockers"]))

    def test_dense_profile_keeps_consistency_and_fusion_in_agreement(self):
        work = self.root / "work/flight"
        commands = survey.dense_commands(self.root, work, work / "dense")
        consistency = "--PatchMatchStereo.geom_consistency"
        self.assertEqual(commands[1]["argv"][commands[1]["argv"].index(consistency) + 1], "true")
        self.assertEqual(commands[2]["argv"][commands[2]["argv"].index("--input_type") + 1],
                         "geometric")
        fast = survey.dense_commands(self.root, work, work / "dense", profile="fast")
        self.assertEqual(fast[1]["argv"][fast[1]["argv"].index(consistency) + 1], "false")
        self.assertEqual(fast[2]["argv"][fast[2]["argv"].index("--input_type") + 1], "photometric")
        self.assertLess(
            int(fast[1]["argv"][fast[1]["argv"].index("--PatchMatchStereo.max_image_size") + 1]),
            int(commands[1]["argv"][commands[1]["argv"].index("--PatchMatchStereo.max_image_size") + 1]))
        with self.assertRaises(ValueError):
            survey.dense_commands(self.root, work, work / "d", profile="instant")

    def test_refused_preflight_leaves_no_run_directory(self):
        import cv2
        self.prepared_geometry()
        executable = self.root / "tools/colmap/bin/colmap.exe"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"never executed")
        metadata = {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 360.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 1280.0, cv2.CAP_PROP_FRAME_HEIGHT: 720.0}
        with patch("cv2.VideoCapture", return_value=SimpleNamespace(
                get=lambda key: metadata[key], release=lambda: None, isOpened=lambda: True)):
            with self.assertRaisesRegex(ValueError, "1080p"):
                survey.reconstruct_scene(self.root, "flight", allow_gpu=True)
        runs = self.work / "survey/runs"
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_depth_views_returns_none_without_a_dense_model(self):
        self.assertIsNone(survey._depth_views(self.root / "work/flight/survey/runs/absent"))


if __name__ == "__main__":
    unittest.main()
