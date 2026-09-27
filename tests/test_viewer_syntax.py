"""Every viewer script parses as an ES module (node --check on a .mjs copy).

The node unit tests import only the engine-free modules, so a syntax error in a module that
imports PlayCanvas (rehearsal.js, session.js, vr.js...) was only caught by the headless arena.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ViewerSyntax(unittest.TestCase):
    def test_modules_parse(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        files = [ROOT / "viewer" / "pc.js", *sorted((ROOT / "viewer" / "pc" / "scripts").glob("*.js")),
                 ROOT / "viewer" / "plan_core.js", ROOT / "viewer" / "workspace_core.js"]
        with tempfile.TemporaryDirectory() as tmp:
            for path in files:
                if not path.is_file():
                    continue
                copy = Path(tmp) / (path.stem + ".mjs")
                shutil.copyfile(path, copy)
                out = subprocess.run([node, "--check", str(copy)], capture_output=True, text=True, timeout=60)
                self.assertEqual(out.returncode, 0, f"{path.relative_to(ROOT)}: {out.stderr.strip()[:400]}")


if __name__ == "__main__":
    unittest.main()
