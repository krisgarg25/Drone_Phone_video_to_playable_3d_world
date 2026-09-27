import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import detect_rooms as dr


def grid(a0, a1, b0, b1, step):
    aa = np.arange(a0, a1 + 1e-9, step)
    bb = np.arange(b0, b1 + 1e-9, step)
    aa, bb = np.meshgrid(aa, bb)
    return aa.ravel(), bb.ravel()


def synthetic_room(ceiling=False, noise=0.01, seed=0):
    """A 4 x 3 m room with 2.4 m walls, as (points, splat normals).

    Built on a 0.1 m pitch so the floor is ~12 m2 of occupied VOX cells, which is what
    the occupancy step counts. Walls start 5 cm above the floor and rise to 2.40 m, so
    the lowest splats fall inside the floor's own inlier tolerance and must be dropped
    by the standing test rather than being fitted as a fifth wall.
    """
    rng = np.random.default_rng(seed)
    pts, nrm = [], []

    # floor at y = 0, normal up
    u, v = grid(-2.0, 2.0, -1.5, 1.5, 0.1)
    pts.append(np.stack([u, np.zeros_like(u), v], 1))
    nrm.append(np.tile([0.0, 1.0, 0.0], (len(u), 1)))

    # two walls 4 m apart (normals along x), two walls 3 m apart (normals along z)
    along_z, rise = grid(-1.5, 1.5, 0.05, 2.40, 0.1)
    along_x, rise2 = grid(-2.0, 2.0, 0.05, 2.40, 0.1)
    for fixed, normal in ((-2.0, (1.0, 0.0, 0.0)), (2.0, (-1.0, 0.0, 0.0))):
        pts.append(np.stack([np.full(len(along_z), fixed), rise, along_z], 1))
        nrm.append(np.tile(normal, (len(along_z), 1)))
    for fixed, normal in ((-1.5, (0.0, 0.0, 1.0)), (1.5, (0.0, 0.0, -1.0))):
        pts.append(np.stack([along_x, rise2, np.full(len(along_x), fixed)], 1))
        nrm.append(np.tile(normal, (len(along_x), 1)))

    if ceiling:
        u, v = grid(-2.0, 2.0, -1.5, 1.5, 0.2)
        pts.append(np.stack([u, np.full(len(u), 2.4), v], 1))
        nrm.append(np.tile([0.0, 1.0, 0.0], (len(u), 1)))

    P = np.concatenate(pts, 0) + rng.normal(0.0, noise, (sum(len(p) for p in pts), 3))
    N = np.concatenate(nrm, 0)
    return P, N


class DetectRoomsTests(unittest.TestCase):
    def setUp(self):
        self.P, self.N = synthetic_room()
        self.stats = {}
        self.floor = dr.detect_floor(self.P, self.N, self.stats)
        self.walls = dr.wall_candidates(self.P, self.N, self.floor, self.stats)

    def test_floor_is_found_flat_and_about_twelve_square_metres(self):
        self.assertIsNotNone(self.floor)
        self.assertIsNone(self.floor["rejected"])
        self.assertLess(self.floor["tilt"], 5.0)
        self.assertLess(self.floor["rms"], dr.FLOOR_RMS_MAX)
        self.assertAlmostEqual(self.floor["area"], 12.0, delta=1.5)
        self.assertAlmostEqual(self.floor["y"], 0.0, delta=0.1)

    def test_all_four_walls_are_recovered_as_vertical_planes(self):
        self.assertEqual(len(self.walls), 4)
        for w in self.walls:
            self.assertLess(w["rms"], dr.WALL_RMS_MAX)
            self.assertLess(w["tilt"], dr.WALL_TILT_MAX)
            self.assertGreater(w["length"], 1.5)
            self.assertGreater(w["height"], 1.5)

    def test_wall_plane_equation_reads_positive_inside_the_room(self):
        # the whole point of the sign convention: a consumer measures a sofa's
        # clearance as p . normal + offset and never has to guess which side is in
        for w in self.walls:
            d = float(self.floor["centroid"] @ w["normal"] + w["offset"])
            self.assertGreater(d, 0.5)

    def test_opposing_walls_give_the_room_dimensions(self):
        enc = dr.enclosure(self.walls, self.floor)
        seps = sorted(p["separation_m"] for p in enc["opposing_pairs"])
        self.assertEqual(len(seps), 2)
        self.assertAlmostEqual(seps[0], 3.0, delta=0.5)
        self.assertAlmostEqual(seps[1], 4.0, delta=0.5)

    def test_no_ceiling_points_means_no_ceiling_and_a_reason(self):
        ceil, why = dr.detect_ceiling(self.P, self.N, self.floor, self.walls, {})
        self.assertIsNone(ceil)
        self.assertIn("no ceiling observed", why)

    def test_a_scanned_ceiling_is_measured_not_guessed(self):
        P, N = synthetic_room(ceiling=True)
        st = {}
        fl = dr.detect_floor(P, N, st)
        walls = dr.wall_candidates(P, N, fl, st)
        ceil, why = dr.detect_ceiling(P, N, fl, walls, st)
        self.assertIsNotNone(ceil, why)
        self.assertAlmostEqual(ceil["height_m"], 2.4, delta=0.25)

    def test_terrain_like_floor_is_refused_not_reported(self):
        # a drone scene's "floor" is flat but broken into hundreds of patches, which is
        # the measured signature that separated rocks (3%) and the auditorium (19%) from
        # the two interiors (79% and 96%).
        rng = np.random.default_rng(3)
        x = rng.uniform(-10, 10, 900)
        z = rng.uniform(-10, 10, 900)
        P = np.stack([x, np.zeros_like(x), z], 1)
        N = np.tile([0.0, 1.0, 0.0], (len(P), 1))
        fl = dr.detect_floor(P, N, {})
        self.assertIsNotNone(fl["rejected"])
        self.assertIn("disconnected", fl["rejected"])

    def test_a_cloud_with_no_horizontal_surface_gives_no_floor(self):
        rng = np.random.default_rng(4)
        P = rng.uniform(-2, 2, (400, 3))
        N = np.tile([1.0, 0.0, 0.0], (len(P), 1))
        self.assertIsNone(dr.detect_floor(P, N, {}))

    def test_perimeter_walls_do_not_split_the_room(self):
        comps, splits = dr.split_components(self.floor, self.walls)
        self.assertEqual(splits, 0)
        self.assertEqual(len(comps), 1)

    def test_a_wall_line_across_the_floor_does_split_it(self):
        # the only case where two rooms may be reported, and it needs real geometry
        wall, rise = grid(-1.5, 1.5, 0.05, 2.40, 0.1)
        P = np.concatenate([self.P, np.stack([np.zeros_like(wall), rise, wall], 1)], 0)
        N = np.concatenate([self.N, np.tile([1.0, 0.0, 0.0], (len(wall), 1))], 0)
        st = {}
        fl = dr.detect_floor(P, N, st)
        walls = dr.wall_candidates(P, N, fl, st)
        comps, splits = dr.split_components(fl, walls)
        self.assertGreaterEqual(splits, 1)
        self.assertGreaterEqual(len(comps), 2)

    def test_an_oversized_wall_is_flagged_as_a_facade(self):
        # the auditorium measured a 21.37 m "wall" over a fragmented 31 m2 floor;
        # plausibility has to say so instead of printing it as a room wall
        w = dict(self.walls[0])
        w["length"] = 21.37
        w["height"] = 2.9
        out = dr.sanity_checks(self.floor, [w], Path("work/room_w_jsonl/viewer_assets"))
        self.assertTrue(any(c["check"] == "wall length plausibility" for c in out), str(out))

    def test_detector_is_deterministic(self):
        b = dr.detect_floor(self.P, self.N, {})
        self.assertAlmostEqual(self.floor["area"], b["area"], places=9)
        self.assertAlmostEqual(float(self.floor["normal"][0]), float(b["normal"][0]),
                               places=9)

    def test_fit_plane_refuses_a_point_set_it_cannot_fit(self):
        self.assertIsNone(dr.fit_plane(np.zeros((2, 3))))


if __name__ == "__main__":
    unittest.main()
