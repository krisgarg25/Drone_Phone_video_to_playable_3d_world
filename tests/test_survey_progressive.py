import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import survey_progressive as progressive
import survey_streaming as streaming


def make_database(path, names, *, priors=False):
    """A database shaped like COLMAP 4.1.1's for the tables the executor reads."""
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY "
                    "AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE, "
                    "camera_id INTEGER NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY "
                    "NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB)")
        if priors:
            con.execute("CREATE TABLE IF NOT EXISTS pose_priors (pose_prior_id INTEGER "
                        "PRIMARY KEY, corr_data_id INTEGER NOT NULL, corr_sensor_id "
                        "INTEGER NOT NULL, corr_sensor_type INTEGER NOT NULL, position "
                        "BLOB, position_covariance BLOB, gravity BLOB, coordinate_system "
                        "INTEGER NOT NULL)")
        known = dict(con.execute("SELECT name, image_id FROM images"))
        for name in names:
            if name in known:
                image_id = known[name]
            else:
                image_id = max(known.values(), default=0) + 1
                known[name] = image_id
                con.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?,?,1)",
                            (image_id, name))
            con.execute("INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data) "
                        "VALUES (?,?,6,NULL)", (image_id, 10))
            if priors:
                con.execute("INSERT OR REPLACE INTO pose_priors (pose_prior_id, "
                            "corr_data_id, corr_sensor_id, corr_sensor_type, position, "
                            "position_covariance, gravity, coordinate_system) "
                            "VALUES (NULL,?,?,0,NULL,NULL,NULL,1)", (image_id, 1))
        con.commit()
    finally:
        con.close()


def _flag(argv, name):
    return argv[argv.index(name) + 1]


class FakeColmap:
    """Interprets the executor's argv the way COLMAP 4.1.1 documents it.

    Models are directories with a manifest.txt of registered image names; the
    fake model_converter turns a manifest into images.txt/points3D.txt in exactly
    the layout the real writer produces (two lines per registered image, NAME in
    field 10). Failures are injected by log-file stem, because that is how the
    executor names each invocation.
    """

    def __init__(self, run, *, fail=(), merge_silently_fails=False, extractor_drops=0,
                 extract_priors=False):
        self.run = Path(run)
        self.calls = []
        self.snapshots = {}
        self.violations = []
        self.fail = set(fail)
        self.merge_silently_fails = merge_silently_fails
        self.extractor_drops = extractor_drops
        self.extract_priors = extract_priors

    def _manifest(self, model_dir):
        return (Path(model_dir) / "manifest.txt").read_text(encoding="utf-8").split()

    def _write_model(self, model_dir, names):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "images.bin").write_bytes(b"fake-binary-model")
        (model_dir / "manifest.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    def __call__(self, argv, **kwargs):
        argv = [str(part) for part in argv]
        sub = argv[1]
        stem = Path(kwargs["stdout"].name).stem
        self.calls.append((sub, argv, stem))
        checkpoint = self.run / "progressive/checkpoints.json"
        if checkpoint.is_file():
            self.snapshots.setdefault(stem, json.loads(checkpoint.read_text(encoding="utf-8")))
        if sub == "feature_extractor":
            names = _flag(argv, "--image_list_path")
            names = Path(names).read_text(encoding="utf-8").split()
            if self.extractor_drops:
                # COLMAP rejects disagreeing images with a warning and exit 0.
                names = names[self.extractor_drops:]
            make_database(_flag(argv, "--database_path"), names,
                          priors=self.extract_priors)
        elif sub in ("mapper", "pose_prior_mapper"):
            names = Path(_flag(argv, "--Mapper.image_list_path")).read_text(
                encoding="utf-8").split()
            self._write_model(Path(_flag(argv, "--output_path")) / "0", names)
        elif sub == "model_merger":
            source = self._manifest(_flag(argv, "--input_path1"))
            target = self._manifest(_flag(argv, "--input_path2"))
            merged = target if self.merge_silently_fails else sorted(set(source) | set(target))
            self._write_model(_flag(argv, "--output_path"), merged)
        elif sub == "image_registrator":
            output = Path(_flag(argv, "--output_path"))
            if not output.is_dir():
                # COLMAP 4.1.1 refuses a non-existent output directory; the
                # executor must create it before spawning.
                self.violations.append(stem + ": output_path did not exist")
            self._write_model(output, self._manifest(_flag(argv, "--input_path")))
        elif sub == "model_converter":
            self.convert(argv)
        elif sub != "sequential_matcher":
            self.violations.append(stem + ": unexpected subcommand " + sub)
        return SimpleNamespace(returncode=1 if stem in self.fail else 0)

    def convert(self, argv):
        assert _flag(argv, "--output_type") == "TXT"
        source, output = Path(_flag(argv, "--input_path")), Path(_flag(argv, "--output_path"))
        if not output.is_dir():
            self.violations.append("model_converter: output_path did not exist")
            return
        names = self._manifest(source)
        lines = ["# Image list with two lines of data per image:"]
        for index, name in enumerate(names, start=1):
            lines.append(f"{index} 1 0 0 0 0 0 0 1 {name}")
            lines.append("0 0 -1")
        (output / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        points = ["# 3D point list"] + [
            f"{index} 0 0 0 128 128 128 0.1" for index in range(1, len(names) * 7 + 1)]
        (output / "points3D.txt").write_text("\n".join(points) + "\n", encoding="utf-8")

    def subcommands(self):
        return [sub for sub, _, _ in self.calls]

    def argv_of(self, stem):
        return next(argv for _, argv, name in self.calls if name == stem)


class SurveyProgressiveTests(unittest.TestCase):
    FRAMES = 24

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        executable = self.root / "tools/colmap/bin/colmap.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"never executed")
        self.run = self.root / "work/flight/survey/runs/20260923T000000-aaaaaaaa"
        self.frames = self.run / "frames_match"
        self.frames.mkdir(parents=True)
        self.names = [f"{index:05d}.jpg" for index in range(self.FRAMES)]
        for name in self.names:
            (self.frames / name).write_bytes(b"not decoded")
        self.plan = streaming.progressive_plan(self.FRAMES, 2,
                                               rates=streaming.MEASURED_RATES, image_px=1000)

    def execute(self, fake, **kwargs):
        self.addCleanup(patch.stopall)
        patch.object(progressive.subprocess, "run", side_effect=fake).start()
        return progressive.execute_plan(self.root, self.run, self.plan, **kwargs)

    def test_argv_builders_state_the_verified_colmap_contract(self):
        colmap = str(self.root / "tools/colmap/bin/colmap.exe")
        extract = progressive.feature_extraction_command(
            self.root, "db", "imgs", "list.txt", mask_path="masks")
        self.assertEqual(extract["stage"], "feature_extractor")
        self.assertTrue(extract["requires_gpu"])
        self.assertEqual(extract["argv"][:4],
                         [colmap, "feature_extractor", "--database_path", "db"])
        self.assertEqual(_flag(extract["argv"], "--image_list_path"), "list.txt")
        self.assertEqual(_flag(extract["argv"], "--ImageReader.single_camera"), "1")
        self.assertEqual(_flag(extract["argv"], "--ImageReader.mask_path"), "masks")
        match = progressive.sequential_matching_command(self.root, "db")
        self.assertEqual(match["argv"][1], "sequential_matcher")
        self.assertEqual(_flag(match["argv"], "--SequentialMatching.overlap"), "20")
        for pose_priors, expected in ((False, "mapper"), (True, "pose_prior_mapper")):
            command = progressive.window_mapper_command(
                self.root, "db", "imgs", "list.txt", "out", pose_priors=pose_priors)
            self.assertEqual(command["stage"], expected)
            self.assertFalse(command["requires_gpu"])
            self.assertEqual(_flag(command["argv"], "--Mapper.image_list_path"), "list.txt")
            self.assertEqual(_flag(command["argv"], "--Mapper.multiple_models"), "0")
        priors = progressive.window_mapper_command(self.root, "db", "imgs", "l", "o",
                                                   pose_priors=True)
        self.assertEqual(_flag(priors["argv"], "--prior_position_std_x"), "0.15")
        self.assertEqual(_flag(priors["argv"], "--overwrite_priors_covariance"), "1")
        merge = progressive.model_merger_command(self.root, "sub", "acc", "out")
        # Verified against COLMAP 4.1.1 exe/model.cc: input_path1 is absorbed
        # into input_path2 and input_path2 is what gets written, so the submap
        # must be path1 and the accumulated model path2 to keep its frame stable.
        self.assertEqual(merge["argv"], [colmap, "model_merger", "--input_path1", "sub",
                                         "--input_path2", "acc", "--output_path", "out"])
        register = progressive.image_registrator_command(self.root, "db", "merged", "acc")
        self.assertEqual(register["argv"], [colmap, "image_registrator",
                                            "--database_path", "db",
                                            "--input_path", "merged",
                                            "--output_path", "acc"])
        convert = progressive.model_converter_command(self.root, "model", "txt")
        self.assertEqual(convert["argv"], [colmap, "model_converter", "--input_path",
                                           "model", "--output_path", "txt",
                                           "--output_type", "TXT"])
        for command in (extract, match, priors, merge, register, convert):
            self.assertTrue(all(isinstance(part, str) for part in command["argv"]))

    def test_windows_share_features_and_publish_measured_checkpoints(self):
        fake = FakeColmap(self.run)
        result = self.execute(fake)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["label"], "measured")
        self.assertEqual(result["plan"], self.plan)
        self.assertEqual([w["status"] for w in result["windows"]], ["done", "done"])
        self.assertEqual(fake.violations, [])
        # The plan's windows tile the frames; window 1 adds only frames 17..23.
        self.assertEqual(fake.subcommands(),
                         ["feature_extractor", "sequential_matcher", "mapper",
                          "model_converter", "model_converter",
                          "feature_extractor", "sequential_matcher", "mapper",
                          "model_converter", "model_merger", "model_converter",
                          "image_registrator", "model_converter"])
        first = Path(_flag(fake.argv_of("window_00_feature_extractor"), "--image_list_path"))
        second = Path(_flag(fake.argv_of("window_01_feature_extractor"), "--image_list_path"))
        self.assertEqual(first.read_text(encoding="utf-8").split(), self.names[:17])
        self.assertEqual(second.read_text(encoding="utf-8").split(), self.names[17:])
        # Merge direction and re-registration over the shared database.
        merger = fake.argv_of("window_01_model_merger")
        self.assertEqual(Path(_flag(merger, "--input_path1")).parent.name, "window_01")
        self.assertEqual(_flag(merger, "--input_path2"),
                         str(self.run / "progressive/models/accumulated_00"))
        self.assertEqual(_flag(merger, "--output_path"),
                         str(self.run / "progressive/models/merged_01"))
        registrator = fake.argv_of("window_01_image_registrator")
        self.assertEqual(_flag(registrator, "--database_path"), str(self.run / "database.db"))
        self.assertEqual(_flag(registrator, "--input_path"),
                         str(self.run / "progressive/models/merged_01"))
        self.assertEqual(_flag(registrator, "--output_path"),
                         str(self.run / "progressive/models/accumulated_01"))
        # Measured against predicted: both present, neither overwriting the other.
        for row, planned in zip(result["windows"], self.plan["windows"]):
            self.assertEqual(row["predicted_s"], planned["predicted_s"])
            self.assertIsNotNone(row["secs"])
            self.assertGreaterEqual(row["secs"], 0)
            self.assertEqual((row["start"], row["end"], row["frame_count"]),
                             (planned["start"], planned["end"], planned["frame_count"]))
            self.assertIsNone(row["error"])
        self.assertEqual([w["registered_images"] for w in result["windows"]], [17, 24])
        self.assertEqual([w["points"] for w in result["windows"]], [17 * 7, 24 * 7])
        self.assertEqual([w["model_path"] for w in result["windows"]],
                         ["progressive/models/accumulated_00",
                          "progressive/models/accumulated_01"])
        self.assertEqual(result["accumulated"],
                         {"registered_images": 24, "points": 168,
                          "windows_done": 2, "windows_total": 2})
        self.assertIsNotNone(result["first_useful_output_s"])
        self.assertGreaterEqual(result["first_useful_output_s"], result["windows"][0]["secs"])
        # The file on disk is the returned payload, atomically written, and the
        # dashboard saw the intermediate state: window 0 done while 1 was running.
        on_disk = json.loads((self.run / "progressive/checkpoints.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(on_disk, result)
        self.assertEqual(list(self.run.glob("progressive/*.tmp")), [])
        mid_run = fake.snapshots["window_01_mapper"]
        self.assertEqual([w["status"] for w in mid_run["windows"]], ["done", "running"])
        self.assertEqual(mid_run["accumulated"]["windows_done"], 1)
        self.assertEqual(mid_run["label"], "measured")

    def test_checkpoint_contract_keys_and_run_relative_paths_only(self):
        fake = FakeColmap(self.run)
        result = self.execute(fake)
        self.assertEqual(set(result), {"schema_version", "plan", "label", "windows",
                                       "accumulated", "first_useful_output_s"})
        self.assertEqual(set(result["accumulated"]), {"registered_images", "points",
                                                      "windows_done", "windows_total"})
        for row in result["windows"]:
            self.assertEqual(set(row), {"index", "start", "end", "frame_count",
                                        "overlap_with_previous", "status", "secs",
                                        "predicted_s", "registered_images", "points",
                                        "model_path", "error"})
        text = (self.run / "progressive/checkpoints.json").read_text(encoding="utf-8")
        self.assertNotIn(str(self.root), text)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text), text)
        for row in result["windows"]:
            self.assertNotIn("\\", row["model_path"])
            self.assertTrue((self.run / row["model_path"]).is_dir())

    def test_a_presolved_database_never_re_extracts_or_re_matches(self):
        make_database(self.run / "database.db", self.names)
        fake = FakeColmap(self.run)
        result = self.execute(fake)
        self.assertEqual([w["status"] for w in result["windows"]], ["done", "done"])
        self.assertNotIn("feature_extractor", fake.subcommands())
        self.assertNotIn("sequential_matcher", fake.subcommands())
        self.assertEqual(fake.subcommands().count("mapper"), 2)
        self.assertEqual(result["accumulated"]["registered_images"], 24)

    def test_pose_priors_select_the_prior_mapper_and_fall_back_on_failure(self):
        make_database(self.run / "database.db", self.names, priors=True)
        fake = FakeColmap(self.run, fail={"window_00_pose_prior_mapper"})
        result = self.execute(fake)
        self.assertEqual([w["status"] for w in result["windows"]], ["done", "done"])
        self.assertIn("pose_prior_mapper", fake.subcommands())
        self.assertIn("mapper", fake.subcommands())
        order = fake.subcommands()
        self.assertLess(order.index("pose_prior_mapper"), order.index("mapper"))

    def test_a_failed_window_is_recorded_with_its_reason_and_blocks_later_merges(self):
        plan = streaming.progressive_plan(self.FRAMES, 3,
                                          rates=streaming.MEASURED_RATES, image_px=1000)
        fake = FakeColmap(self.run, fail={"window_01_mapper"})
        self.addCleanup(patch.stopall)
        patch.object(progressive.subprocess, "run", side_effect=fake).start()
        result = progressive.execute_plan(self.root, self.run, plan)
        self.assertEqual([w["status"] for w in result["windows"]],
                         ["done", "failed", "failed"])
        failed = result["windows"][1]
        self.assertIn("colmap mapper failed with exit 1", failed["error"])
        self.assertIsNotNone(failed["secs"])  # the spent wall time was measured
        self.assertIsNone(failed["registered_images"])
        self.assertIsNone(failed["points"])
        self.assertIsNone(failed["model_path"])
        self.assertTrue(result["windows"][2]["error"].startswith("not attempted: window 1 failed"))
        self.assertEqual(result["windows"][0]["registered_images"], 13)
        self.assertEqual(result["accumulated"],
                         {"registered_images": 13, "points": 13 * 7,
                          "windows_done": 1, "windows_total": 3})
        self.assertIsNotNone(result["first_useful_output_s"])

    def test_model_merger_exit_zero_without_a_merge_is_detected(self):
        # COLMAP 4.1.1 writes input_path2 unchanged and exits 0 when alignment
        # fails; the executor must notice from the output model, not the exit code.
        fake = FakeColmap(self.run, merge_silently_fails=True)
        result = self.execute(fake)
        self.assertEqual([w["status"] for w in result["windows"]], ["done", "failed"])
        self.assertIn("did not merge", result["windows"][1]["error"])
        self.assertIn("exits 0", result["windows"][1]["error"])
        self.assertEqual(result["accumulated"]["windows_done"], 1)

    def test_a_hung_colmap_command_is_a_window_failure_not_a_dead_run(self):
        fake = FakeColmap(self.run)

        def runner(argv, **kwargs):
            if Path(kwargs["stdout"].name).stem == "window_01_model_merger":
                raise subprocess.TimeoutExpired(cmd=[str(part) for part in argv], timeout=1)
            return fake(argv, **kwargs)

        self.addCleanup(patch.stopall)
        patch.object(progressive.subprocess, "run", side_effect=runner).start()
        result = progressive.execute_plan(self.root, self.run, self.plan)
        self.assertEqual([w["status"] for w in result["windows"]], ["done", "failed"])
        self.assertIn("timed out", result["windows"][1]["error"])
        # The timeout text carries the argv; absolute paths must not reach the file.
        text = (self.run / "progressive/checkpoints.json").read_text(encoding="utf-8")
        self.assertNotIn(str(self.root), text)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text), text)
        self.assertEqual(result["accumulated"]["windows_done"], 1)

    def test_silent_frame_rejection_by_the_extractor_fails_the_window(self):
        fake = FakeColmap(self.run, extractor_drops=2)
        result = self.execute(fake)
        self.assertEqual([w["status"] for w in result["windows"]], ["failed", "failed"])
        self.assertIn("feature_extractor took 15 of 17", result["windows"][0]["error"])
        self.assertIn("exits 0", result["windows"][0]["error"])
        self.assertIsNone(result["first_useful_output_s"])
        self.assertEqual(result["accumulated"],
                         {"registered_images": None, "points": None,
                          "windows_done": 0, "windows_total": 2})

    def test_plan_and_disk_disagreements_are_refused_before_any_subprocess(self):
        fake = FakeColmap(self.run)
        self.addCleanup(patch.stopall)
        patch.object(progressive.subprocess, "run", side_effect=fake).start()
        wrong = streaming.progressive_plan(20, 2, rates=streaming.MEASURED_RATES, image_px=1000)
        with self.assertRaisesRegex(ValueError, "replan"):
            progressive.execute_plan(self.root, self.run, wrong)
        stretched = json.loads(json.dumps(self.plan))
        stretched["windows"][-1]["end"] = self.FRAMES + 5
        with self.assertRaisesRegex(ValueError, "does not address"):
            progressive.execute_plan(self.root, self.run, stretched)
        for bad in ({}, {"windows": []}, {"frame_count": self.FRAMES, "windows": "x"}):
            with self.assertRaises(ValueError):
                progressive.execute_plan(self.root, self.run, bad)
        self.assertEqual(fake.calls, [])
        self.assertFalse((self.run / "progressive").exists())

    def test_missing_inputs_refuse_before_writing_anything(self):
        fake = FakeColmap(self.run)
        self.addCleanup(patch.stopall)
        patch.object(progressive.subprocess, "run", side_effect=fake).start()
        (self.root / "tools/colmap/bin/colmap.exe").unlink()
        with self.assertRaisesRegex(ValueError, "COLMAP executable is missing"):
            progressive.execute_plan(self.root, self.run, self.plan)
        (self.root / "tools/colmap/bin/colmap.exe").write_bytes(b"never executed")
        with self.assertRaises(ValueError):
            progressive.execute_plan(self.root, self.run, self.plan, image_dir="nowhere")
        outside = self.root.parent / "not-under-the-project-root"
        with self.assertRaisesRegex(ValueError, "leaves the project root"):
            progressive.execute_plan(self.root, outside, self.plan)
        self.assertEqual(fake.calls, [])
        self.assertFalse((self.run / "progressive").exists())

    def test_frame_names_sorts_and_refuses_an_empty_directory(self):
        self.assertEqual(progressive.frame_names(self.frames), self.names)
        empty = self.run / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "no frames"):
            progressive.frame_names(empty)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            progressive.frame_names(self.run / "absent")

    def test_read_model_stats_reports_missing_counts_as_null_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            txt = Path(directory)
            stats = progressive.read_model_stats(txt)
            self.assertEqual(stats, {"registered_images": None, "points": None,
                                     "image_names": None})
            (txt / "images.txt").write_text(
                "# header\n1 1 0 0 0 0 0 0 1 a.jpg\n0 0 -1\n", encoding="utf-8")
            stats = progressive.read_model_stats(txt)
            self.assertEqual(stats["registered_images"], 1)
            self.assertEqual(stats["image_names"], {"a.jpg"})
            self.assertIsNone(stats["points"])
            (txt / "images.txt").write_text("# header\n1 1 0 0 0 0 0 0 1\n0 0 -1\n",
                                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ten fields"):
                progressive.read_model_stats(txt)
            (txt / "images.txt").write_text("# h\n1 1 0 0 0 0 0 0 1 a.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "odd number"):
                progressive.read_model_stats(txt)

    def test_cli_exposes_progressive_and_keeps_the_gpu_gate_first(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
        helper = subprocess.run([sys.executable, str(ROOT / "survey.py"),
                                 "reconstruct", "--help"], cwd=ROOT,
                                capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(helper.returncode, 0, helper.stderr)
        self.assertIn("--progressive", helper.stdout)
        refused = subprocess.run([sys.executable, str(ROOT / "survey.py"), "reconstruct",
                                  "not_a_real_survey", "--progressive"], cwd=ROOT,
                                 capture_output=True, text=True, env=env, timeout=60)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("GPU", refused.stderr)
        self.assertFalse((ROOT / "work/not_a_real_survey").exists())


if __name__ == "__main__":
    unittest.main()
