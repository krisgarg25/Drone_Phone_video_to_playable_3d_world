"""End-to-end tests for the Phase 4 CLI and field packs (DIS-09, CON-08)."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import ops  # noqa: E402
import test_survey_change as tc  # noqa: E402
import test_survey_damage as td  # noqa: E402


class PackTests(unittest.TestCase):
    def test_change_pack_georeferenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "a.npy", tc.epoch(0))
            np.save(tmp / "b.npy", tc.epoch(1, pile=True, shift=(0.5, 0, 0.2)))
            code = ops.main(["change", str(tmp / "a.npy"), str(tmp / "b.npy"), "--out",
                             str(tmp / "pack"), "--origin", "30.7333", "76.7794", "300",
                             "--captured", "2026-09-26 10:00 IST"])
            self.assertEqual(code, 0)
            manifest = json.loads((tmp / "pack" / "pack.json").read_text())
            self.assertIn("change.kmz", manifest["files"])
            self.assertGreaterEqual(manifest["placemarks"], 1)
            self.assertTrue((tmp / "pack" / "report.pdf").read_bytes().startswith(b"%PDF"))
            with zipfile.ZipFile(tmp / "pack" / "change.kmz") as kmz:
                kml = kmz.read("doc.kml").decode()
            self.assertIn("2026-09-26 10:00 IST", kml)
            self.assertIn("76.7", kml)
            self.assertTrue((tmp / "pack.zip").exists())

    def test_local_scene_gets_no_kmz(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "post.npy", td.street())
            fp = {k: td.ring(k) for k in td.BOX}
            (tmp / "fp.json").write_text(json.dumps(fp))
            code = ops.main(["damage", str(tmp / "post.npy"), "--out", str(tmp / "pack"),
                             "--footprints", str(tmp / "fp.json")])
            self.assertEqual(code, 0)
            manifest = json.loads((tmp / "pack" / "pack.json").read_text())
            self.assertNotIn("damage.kmz", manifest["files"])
            self.assertTrue(any("not georeferenced" in c for c in manifest["caveats"]))
            rows = (tmp / "pack" / "damage.csv").read_text().splitlines()
            self.assertEqual(len(rows), 5)

    def test_refusal_is_an_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "a.npy", np.zeros((5, 3)))
            code = ops.main(["change", str(tmp / "a.npy"), str(tmp / "a.npy"), "--out",
                             str(tmp / "p")])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
