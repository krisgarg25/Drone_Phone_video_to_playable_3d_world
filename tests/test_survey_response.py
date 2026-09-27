"""CPU-only tests for access/blocked roads/vehicle routing (DIS-04) and flooding (DIS-05)."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CELL = 1.0
T = (0.0, CELL, 0.0, 100.0, 0.0, -CELL)      # 100 x 100 m, row 0 = north (y = 100)


def site():
    """Flat ground, an east-west road at y = 50 (6 m wide), a wall of buildings north and
    south of it except a gap at x = 80..90, and a 2 m debris pile on the road at x = 40..44."""
    dtm = np.zeros((100, 100))
    dsm = dtm.copy()
    road = np.zeros_like(dtm, bool)
    road[47:53, :] = True
    dsm[30:47, 5:80] = 8.0
    dsm[53:70, 5:80] = 8.0
    before = dsm.copy()
    dsm[47:53, 40:44] = 2.0
    return dtm, dsm, before, road


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.r = importlib.import_module("survey_response")

    def test_blocked_road_found_and_new(self):
        dtm, dsm, before, road = site()
        out = self.r.blocked_roads(road, dtm, dsm, T, before_dsm=before)
        self.assertEqual(len(out["blocked"]), 1)
        b = out["blocked"][0]
        self.assertAlmostEqual(b["area_m2"], 24.0)
        self.assertAlmostEqual(b["centre"][0], 42.0, delta=0.6)
        self.assertTrue(b["new_since_before"])
        self.assertEqual(self.r.blocked_roads(road, dtm, before, T, before_dsm=before)["blocked"], [])

    def test_route_detours_around_debris(self):
        dtm, dsm, before, road = site()
        clear = self.r.vehicle_route(dtm, before, T, (2, 50), (60, 50), road=road)
        self.assertAlmostEqual(clear["length_m"], 58.0, delta=1.0)
        self.assertAlmostEqual(clear["offroad_m"], 0.0, delta=1.0)
        # Debris spans the whole road: the vehicle must leave by the west end and go round
        # the building blocks (they reach y = 30..70), so the route is much longer.
        blocked = self.r.vehicle_route(dtm, dsm, T, (2, 50), (60, 50), road=road)
        self.assertGreater(blocked["length_m"], clear["length_m"] + 30)
        path = np.array(blocked["path"])
        self.assertFalse(np.any((path[:, 0] > 39) & (path[:, 0] < 45) & (np.abs(path[:, 1] - 50) < 3)))

    def test_too_narrow_gap_and_unreachable(self):
        dtm, dsm, before, road = site()
        dsm[:, 85:87] = 5.0            # wall north-south, full height of the map
        dsm[48:52, 85:87] = 0.0        # a 4 m gap in the wall on the road
        ok = self.r.vehicle_route(dtm, dsm, T, (60, 50), (95, 50), road=road, vehicle_width_m=2.5)
        self.assertGreater(ok["length_m"], 30)
        with self.assertRaises(ValueError):
            self.r.vehicle_route(dtm, dsm, T, (60, 50), (95, 50), road=road, vehicle_width_m=5.0)
        with self.assertRaises(ValueError):
            self.r.vehicle_route(dtm, dsm, T, (50, 40), (95, 50), road=road, snap_m=0)   # on a roof
        # With snapping, a start on the roof edge 7 m from the road moves to the road.
        snapped = self.r.vehicle_route(dtm, before, T, (50, 43), (70, 50), road=road)
        self.assertAlmostEqual(snapped["snapped_m"]["staging point"], 5.0, delta=1.5)
        with self.assertRaises(ValueError):
            self.r.vehicle_route(dtm, before, T, (50, 36), (70, 50), road=road, snap_m=5)

    def test_flood_connectivity_depth_and_buildings(self):
        x = np.arange(100) + 0.5
        dtm = np.tile(0.05 * x, (100, 1))              # rises eastwards from a river at x = 0
        dtm[:, 60:62] = 6.0                             # an embankment
        dtm[40:60, 70:80] = 0.5                         # a hollow behind it
        fp = {"H1": [(20, 40), (30, 40), (30, 50), (20, 50)], "H2": [(90, 40), (95, 40), (95, 45), (90, 45)]}
        out = self.r.flood(dtm, T, seed_xy=(1, 50), rise_m=2.0, footprints=fp)
        rep = out["report"]
        self.assertAlmostEqual(rep["level_m"], 2.075, delta=0.01)      # seed cell centre x = 1.5
        # Wet where 0.05 x < 2.075 -> 41 columns x 100 rows.
        self.assertAlmostEqual(rep["flooded_m2"], 4100, delta=100)
        self.assertGreater(rep["isolated_low_ground_m2"], 150)      # the hollow stays dry
        ids = [b["id"] for b in rep["buildings"]]
        self.assertEqual(ids, ["H1"])
        with self.assertRaises(ValueError):
            self.r.flood(dtm, T, seed_xy=(1, 50))


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.d = importlib.import_module("survey_detections")

    def test_boxes_from_mask(self):
        mask = np.full((60, 80), 255, np.uint8)
        mask[10:30, 20:26] = 0
        mask[50:52, 70:71] = 0
        boxes = self.d.boxes_from_mask(mask, min_px=12)
        self.assertEqual(boxes, [{"bbox": [20, 10, 26, 30], "pixels": 120}])

    def test_bottom_centre_lands_on_ground_and_merges(self):
        # Camera 20 m up at (0, 20, 0) looking straight down (-y); image up = -z (north).
        cam = dict(name="f0.jpg", t_sec=0.0, pos=[0, 20, 0], forward=[0, -1, 0], up=[0, 0, -1], right=[1, 0, 0],
                   fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
        cam2 = dict(cam, name="f1.jpg", t_sec=1.0, pos=[0.5, 20, 0])
        surface = lambda xz: (np.zeros(len(xz)), np.ones(len(xz), bool))
        # A person standing at x = 5, z = 3 -> pixel u = 320 + 5/20*500, v = 240 + 3/20*500.
        rows = [dict(frame="f0.jpg", **{"class": "person"}, bbox=[440, 280, 460, 315]),
                dict(frame="f1.jpg", **{"class": "person"}, bbox=[422.5, 280, 442.5, 315]),
                dict(frame="nope.jpg", **{"class": "person"}, bbox=[0, 0, 1, 1])]
        out = self.d.georeference(rows, [cam, cam2], surface)
        self.assertEqual(out["unplaced"], 1)
        self.assertEqual(len(out["objects"]), 1)
        obj = out["objects"][0]
        self.assertAlmostEqual(obj["position"][0], 5.0, delta=0.1)
        self.assertAlmostEqual(obj["position"][2], 3.0, delta=0.1)
        self.assertEqual(obj["sightings"], 2)
        self.assertFalse(obj["moving"])


if __name__ == "__main__":
    unittest.main()
