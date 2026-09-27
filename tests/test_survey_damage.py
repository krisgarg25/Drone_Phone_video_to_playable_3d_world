"""CPU-only tests for the building damage heuristic (DIS-02) on a synthetic street."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

BOX = {"flat": (10, 10), "gable": (40, 10), "rubble": (70, 10), "half": (100, 10)}   # SW corners


def street(seed=0):
    """Four 16x12 m buildings: flat 9 m roof, gable 7-10 m, collapsed rubble, half collapsed."""
    rng = np.random.default_rng(seed)
    def fill(x0, y0, w, h, n):
        return np.column_stack([rng.uniform(x0, x0 + w, n), rng.uniform(y0, y0 + h, n)])
    ground = fill(0, 0, 130, 40, 60_000)
    parts = []
    for name, (x0, y0) in BOX.items():
        inside = (ground[:, 0] > x0) & (ground[:, 0] < x0 + 16) & (ground[:, 1] > y0) & (ground[:, 1] < y0 + 12)
        ground = ground[~inside]
        xy = fill(x0, y0, 16, 12, 5000)
        if name == "flat":
            z = np.full(len(xy), 9.0)
        elif name == "gable":
            z = 10.0 - 0.5 * np.abs(xy[:, 1] - (y0 + 6))
        elif name == "rubble":
            z = np.abs(rng.normal(1.0, 0.8, len(xy))) + 0.6 * np.sin(xy[:, 0] * 2.1) * np.cos(xy[:, 1] * 1.7)
        else:
            left = xy[:, 0] < x0 + 8
            z = np.where(left, 9.0, 0.0) + np.where(left, 0.0, np.abs(rng.normal(1.8, 1.0, len(xy)))
                                                   + 0.8 * np.sin(xy[:, 0] * 2.3) * np.cos(xy[:, 1] * 1.9))
        parts.append(np.column_stack([xy, z + rng.normal(0, 0.03, len(xy))]))
    g = np.column_stack([ground, rng.normal(0, 0.03, len(ground))])
    return np.vstack([g, *parts])


def ring(name):
    x0, y0 = BOX[name]
    return [(x0, y0), (x0 + 16, y0), (x0 + 16, y0 + 12), (x0, y0 + 12)]


class DamageTests(unittest.TestCase):
    def setUp(self):
        self.d = importlib.import_module("survey_damage")

    def test_grades_with_pre_event_footprints(self):
        out = self.d.assess(street(), footprints={k: ring(k) for k in BOX},
                            reference_heights={"flat": 9.0, "gable": 10.0, "rubble": 9.0, "half": 9.0})
        got = {b["id"]: b["grade"] for b in out["buildings"]}
        self.assertEqual(got, {"flat": "intact", "gable": "intact",
                               "rubble": "collapsed", "half": "partial"},
                         [(b["id"], b["reason"], b["rough"], b["roof_cover"]) for b in out["buildings"]])

    def test_unobserved_footprint_is_unknown(self):
        out = self.d.assess(street(), footprints={"far": [(200, 200), (210, 200), (210, 210), (200, 210)],
                                                   "flat": ring("flat")})
        got = {b["id"]: b["grade"] for b in out["buildings"]}
        self.assertEqual(got["far"], "unknown")
        self.assertEqual(got["flat"], "intact")

    def test_derived_footprints_miss_flat_collapse_and_say_so(self):
        out = self.d.assess(street())
        self.assertIn("not found", out["footprints"])
        self.assertGreaterEqual(out["counts"]["intact"], 2)

    def test_pre_event_footprints_find_the_flattened_building(self):
        before = street()
        intact = street(1)
        # A "before" street where every building stands: rebuild rubble/half as 9 m roofs.
        for name in ("rubble", "half"):
            x0, y0 = BOX[name]
            inside = (before[:, 0] > x0) & (before[:, 0] < x0 + 16) & (before[:, 1] > y0) & (before[:, 1] < y0 + 12)
            before[inside, 2] = 9.0
        rings, heights = self.d.footprints_from(before)
        self.assertEqual(len(rings), 4)
        out = self.d.assess(intact, footprints=rings, reference_heights=heights)
        grades = sorted(b["grade"] for b in out["buildings"])
        self.assertEqual(grades, ["collapsed", "intact", "intact", "partial"])

    def test_plane_residual_zero_on_tilted_plane(self):
        z = np.add.outer(np.arange(20) * 0.3, np.arange(30) * -0.2)
        r = self.d.plane_residual(z, 0.5)
        self.assertLess(np.nanmax(r), 1e-6)       # float cancellation only


if __name__ == "__main__":
    unittest.main()
