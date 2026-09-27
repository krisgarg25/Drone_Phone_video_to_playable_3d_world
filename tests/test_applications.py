"""Mission-profile registry (playbook §4): suggestions, preset choice, ledger gating, CLI flags."""
import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.a = importlib.import_module("applications")

    def test_every_profile_is_complete(self):
        keys = {"label", "lanes", "dense_profile", "preset_candidates", "required_products", "analyses",
                "workspace_tabs", "acceptance_checks", "not_needed"}
        import survey_workflow
        for app_id, app in self.a.APPLICATIONS.items():
            self.assertTrue(keys <= set(app), app_id)
            self.assertIn(app["dense_profile"], survey_workflow.DENSE_PROFILES, app_id)
            self.assertTrue(set(app["required_products"]) <= set(self.a.OFFICIAL), app_id)
            self.assertIn("layers", app["workspace_tabs"], app_id)
        self.assertEqual(len(self.a.APPLICATIONS), 8)

    def test_suggestions_follow_the_flight_pattern(self):
        self.assertEqual(self.a.suggest({"pattern": "corridor"})[0]["id"], "border")
        self.assertEqual([s["id"] for s in self.a.suggest("grid")][:2], ["construction", "disaster"])
        self.assertEqual(self.a.suggest("hover"), [])

    def test_preset_choice_keeps_an_accepted_detection(self):
        self.assertEqual(self.a.preset_for("border", "corridor")[0], "corridor")
        preset, why = self.a.preset_for("construction", "room")
        self.assertEqual(preset, "drone_mapping")
        self.assertIn("not one of them", why)
        with self.assertRaises(ValueError):
            self.a.get("space")

    def test_ledger_marks_unrequested_formats(self):
        ledger = {"rows": [{"format": "fbx", "status": "not_delivered", "reason": "x"},
                           {"format": "las", "status": "delivered"}, {"format": "geotiff", "status": "not_delivered", "reason": "y"}]}
        out = self.a.ledger_for(ledger, "border")
        rows = {r["format"]: r for r in out["rows"]}
        self.assertEqual(rows["fbx"]["status"], "not_requested")
        self.assertEqual(rows["geotiff"]["status"], "not_delivered")      # requested and missing stays missing
        self.assertEqual(out["delivered_of_requested"], 1)

    def test_cli_flags_exist(self):
        for script, cmd in (("pipeline.py", ["run", "--help"]), ("survey.py", ["reconstruct", "--help"])):
            out = subprocess.run([sys.executable, str(ROOT / script), *cmd], capture_output=True, text=True, timeout=120)
            self.assertIn("--application", out.stdout, script)

    def test_project_saves_application_and_site(self):
        import tempfile
        import threading
        import workspace_api as api

        class Server:
            process_lock = threading.Lock()
            active_job_info = {}

            def job_busy_locked(self):
                return False
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, "summary", lambda *a, **k: {}):
            root = Path(tmp)
            (root / "work" / "site1").mkdir(parents=True)
            (root / "work" / "site1" / "project.json").write_text(json.dumps({"name": "Site 1"}))
            api.save_project(root, {"scene": "site1", "application": "inspection", "site": "bridge-12"}, Server())
            saved = json.loads((root / "work" / "site1" / "project.json").read_text())
            self.assertEqual((saved["application"], saved["site"]), ("inspection", "bridge-12"))
            with self.assertRaises(api.Error):
                api.save_project(root, {"scene": "site1", "application": "space"}, Server())
            self.assertEqual(api._application("inspection")["workspace_tabs"][2], "inspect")


if __name__ == "__main__":
    unittest.main()
