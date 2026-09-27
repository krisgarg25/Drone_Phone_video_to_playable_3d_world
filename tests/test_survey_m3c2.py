"""M3C2 on a wall that bulges 4 cm in one patch: the patch is significant, the rest is not."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def wall(seed, bulge=False, gap=False):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 6, 40000); z = rng.uniform(0, 10, 40000)
    x = rng.normal(0, 0.01, 40000)                          # a vertical wall in the y-z plane, 1 cm noise
    if bulge:
        x = x + 0.04 * ((np.abs(y - 3) < 1) & (np.abs(z - 5) < 1))
    pts = np.column_stack([x, y, z])
    if gap:
        pts = pts[~((y > 4.5) & (z > 8))]
    return pts


class M3C2Tests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module("survey_m3c2")

    def test_bulge_detected_rest_not(self):
        out = self.m.m3c2(wall(0), wall(1, bulge=True, gap=True), normal_scale_m=1.0, projection_diameter_m=0.4,
                          toward=[10, 3, 5], core_spacing_m=0.5)
        core, dist, sig = out["core"], out["distance"], out["significant"]
        patch = (np.abs(core[:, 1] - 3) < 0.7) & (np.abs(core[:, 2] - 5) < 0.7)
        self.assertGreater(sig[patch].mean(), 0.9)
        self.assertAlmostEqual(float(np.nanmedian(dist[patch])), 0.04, delta=0.008)   # +x towards the viewer
        away = (np.abs(core[:, 1] - 3) > 1.5) | (np.abs(core[:, 2] - 5) > 1.5)
        self.assertLess(sig[away].mean(), 0.06)
        hole = (core[:, 1] > 4.8) & (core[:, 2] > 8.3)
        self.assertFalse(out["observed"][hole].any())                  # unobserved, not "no change"
        self.assertLess(out["summary"]["median_lod95_m"], 0.02)

    def test_registration_error_raises_the_lod(self):
        a = self.m.m3c2(wall(0), wall(1), toward=[10, 3, 5], core_spacing_m=1.0)
        b = self.m.m3c2(wall(0), wall(1), toward=[10, 3, 5], core_spacing_m=1.0, registration_error_m=0.05)
        self.assertAlmostEqual(b["summary"]["median_lod95_m"] - a["summary"]["median_lod95_m"], 0.05, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
