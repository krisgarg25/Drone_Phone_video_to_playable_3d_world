import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SurveyCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "survey.py"), *args],
                              cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=15)

    def test_fast_test_entrypoint_includes_survey_suites(self):
        spec = importlib.util.spec_from_file_location("check_all_survey", ROOT / "tests/check_all.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scripts = {script for _, script, _ in module.SUITES}
        self.assertTrue({"test_survey_georef.py", "test_survey_evaluation.py",
                         "test_camera_intrinsics.py", "test_survey_workflow.py",
                         "test_survey_api.py", "test_survey_cli.py"}.issubset(scripts))

    def test_training_cache_depends_on_intrinsics_helper(self):
        import pipeline
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scripts = root / "scripts"
            scripts.mkdir()
            for name in ("robust.py", "train_splat.py", "camera_intrinsics.py"):
                (scripts / name).write_text("version = 1\n")
            with patch.object(pipeline, "ROOT", root):
                before = pipeline.code_digest(["python", "scripts/train_splat.py"])
                (scripts / "camera_intrinsics.py").write_text("version = 2\n")
                after = pipeline.code_digest(["python", "scripts/train_splat.py"])
            self.assertNotEqual(before, after)

    def test_help_lists_safe_actions_and_gate(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for action in ("prepare", "align", "evaluate", "status", "reconstruct"):
            self.assertIn(action, result.stdout)

    def test_reconstruction_without_approval_fails_before_scene_io(self):
        result = self.run_cli("reconstruct", "not_a_real_survey")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU", result.stderr)
        self.assertFalse((ROOT / "work/not_a_real_survey").exists())

    def test_status_missing_scene_produces_json_without_creating_files(self):
        result = self.run_cli("status", "not_a_real_survey")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(result.stdout)
        self.assertEqual(state["status"], "not_prepared")
        self.assertFalse((ROOT / "work/not_a_real_survey").exists())


if __name__ == "__main__":
    unittest.main()
