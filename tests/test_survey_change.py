"""CPU-only tests for two-epoch change detection (M3) on scenes with known change."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def terrain(x, y):
    return 0.08 * x + 2.0 * np.sin(y / 9.0) + 1.5 * np.cos(x / 7.0)


def epoch(seed, *, pile=False, pit=False, noise=0.03, n=90_000, shift=(0.0, 0.0, 0.0)):
    """A rolling 100x100 m site; a cone pile (3 m, r 6) and/or a 1.5 m-deep 8x8 m pit."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2))
    z = terrain(xy[:, 0], xy[:, 1]) + rng.normal(0, noise, n)
    if pile:
        r = np.hypot(xy[:, 0] - 30, xy[:, 1] - 60)
        z = z + np.clip(3.0 * (1 - r / 6.0), 0, None)
    if pit:
        inside = (np.abs(xy[:, 0] - 70) < 4) & (np.abs(xy[:, 1] - 30) < 4)
        z = z - 1.5 * inside
    pts = np.column_stack([xy, z])
    return pts + np.asarray(shift)


class ChangeTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module("survey_change")

    def test_known_offset_is_recovered(self):
        before = epoch(0)
        after = epoch(1, shift=(1.3, -0.7, 0.9))
        out = self.c.detect_change(before, after, cell_m=1.0)
        reg = out["report"]["registration"]
        self.assertAlmostEqual(reg["dx_m"], -1.3, delta=0.15)
        self.assertAlmostEqual(reg["dy_m"], 0.7, delta=0.15)
        self.assertAlmostEqual(reg["dz_m"], -0.9, delta=0.05)
        # Without registration the slopes light up as false change.
        raw = self.c.detect_change(before, after, cell_m=1.0, register=False)
        self.assertLess(out["report"]["cells"]["gain"] + out["report"]["cells"]["loss"],
                        0.02 * out["report"]["cells"]["observed_both"])
        self.assertGreater(raw["report"]["cells"]["gain"], 20 * max(1, out["report"]["cells"]["gain"]))

    def test_pile_and_pit_are_found_with_volumes(self):
        before = epoch(0)
        after = epoch(1, pile=True, pit=True, shift=(0.8, 0.5, -0.4))
        out = self.c.detect_change(before, after, cell_m=1.0)
        regions = out["report"]["regions"]
        gains = [r for r in regions if r["kind"] == "gain"]
        losses = [r for r in regions if r["kind"] == "loss"]
        self.assertTrue(gains and losses, regions[:4])
        pile, pit = gains[0], losses[0]
        # Cone volume pi r^2 h / 3 = 113.1 m^3; cells below LoD at the rim are dropped.
        self.assertAlmostEqual(pile["volume_m3"], 113.1, delta=113.1 * 0.15)
        self.assertAlmostEqual(pile["centre_enu"][0], 30, delta=1.0)
        self.assertAlmostEqual(pit["volume_m3"], -96.0, delta=96 * 0.15)
        self.assertAlmostEqual(pit["centre_enu"][1], 30, delta=1.0)
        # Nothing else of size beyond noise.
        self.assertLessEqual(len([r for r in regions if abs(r["volume_m3"]) > 5]), 2)

    def test_unobserved_is_not_no_change(self):
        before = epoch(0)
        after = epoch(1)
        after = after[~((after[:, 0] > 80) & (after[:, 1] > 80))]   # a gap in epoch 2
        out = self.c.detect_change(before, after, cell_m=1.0)
        self.assertEqual(int(out["status"][2, 95]), self.c.UNOBSERVED)   # row 2 = north
        self.assertGreater(out["report"]["cells"]["unobserved"], 300)

    def test_region_volume_and_mgrs(self):
        import survey_measure
        frame = survey_measure.frame_for_origin(30.7333, 76.7794, 300.0)
        before, after = epoch(0), epoch(1, pile=True)
        vol = self.c.region_volume(before, after, [(22, 52), (38, 52), (38, 68), (22, 68)],
                                   cell_m=0.5)
        self.assertTrue(vol["valid"])
        self.assertAlmostEqual(vol["net_m3"], 113.1, delta=6.0)
        self.assertLess(vol["sigma_m3"], 5.0)
        out = self.c.detect_change(before, after, cell_m=1.0, frame=frame)
        self.assertTrue(out["report"]["regions"][0]["mgrs"].startswith("43R"))

    def test_bad_inputs_refused(self):
        with self.assertRaises(ValueError):
            self.c.detect_change(np.zeros((5, 3)), epoch(0))
        far = epoch(1, shift=(5000, 0, 0))
        with self.assertRaises(ValueError):
            self.c.detect_change(epoch(0), far, cell_m=5.0)


if __name__ == "__main__":
    unittest.main()
