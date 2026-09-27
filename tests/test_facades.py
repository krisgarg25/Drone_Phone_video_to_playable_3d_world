"""Facade completeness (URB-17): a box seen on three sides reports the fourth as unseen."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def box_points(skip_north=True, weak_east=False):
    """10 x 8 m footprint at the origin, 6 m high; walls sampled every 0.25 m."""
    pts = []
    for y in np.arange(0.1, 6.0, 0.25):
        for x in np.arange(-5, 5, 0.25):
            if not skip_north:
                pts.append([x, y, -4.0])              # north wall (-z)
            pts.append([x, y, 4.0])                   # south wall
        for z in np.arange(-4, 4, 0.25):
            pts.append([-5.0, y, z])                  # west
            if not weak_east or (int(z * 4) % 16 == 0):
                pts.append([5.0, y, z])               # east (sparse when weak)
    return np.asarray(pts)


BUILDING = {"id": "b1", "centre": [0, 0], "footprint": [[-5, -4], [5, -4], [5, 4], [-5, 4]], "base_y": 0.0, "height_m": 6.0}


class FacadeTests(unittest.TestCase):
    def setUp(self):
        self.f = importlib.import_module("facades")

    def test_unseen_north_wall(self):
        out = self.f.completeness([BUILDING], box_points())
        b = out["buildings"][0]
        by = {f["facade"]: f for f in b["facades"]}
        self.assertEqual(set(by), {"N", "E", "S", "W"})
        self.assertEqual(by["N"]["unobserved_pct"], 100.0)
        for side in "ESW":
            self.assertEqual(by[side]["observed_pct"], 100.0, side)
        # North is 10 of 36 m of perimeter.
        self.assertAlmostEqual(b["unobserved_pct"], 100 * 10 / 36, delta=0.5)
        self.assertTrue(any(m["part"] == "facade_unobserved" for m in out["overlay"]))

    def test_sparse_wall_is_weak_not_observed(self):
        out = self.f.completeness([BUILDING], box_points(skip_north=False, weak_east=True))
        east = next(f for f in out["buildings"][0]["facades"] if f["facade"] == "E")
        self.assertLess(east["observed_pct"], 50)
        self.assertGreater(east["weak_pct"] + east["unobserved_pct"], 50)


if __name__ == "__main__":
    unittest.main()
