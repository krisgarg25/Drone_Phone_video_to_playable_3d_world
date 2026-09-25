import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import workspace_place as wp


def make_room(root, *, nx=100, nz=100, cell=0.05, floor=0.0, ceiling=2.5,
              open_box=(10, 10, 90, 90)):
    """A synthetic interior: a supported floor patch inside walls, ceiling above.

    Coordinates are viewer Y-up; the grid spans x=[0, nx*cell], z=[0, nz*cell].
    ``open_box`` is the (x0, z0, x1, z1) cell rectangle that is real floor.
    """
    (root / "viewer_assets").mkdir(parents=True, exist_ok=True)
    ground = np.full((nz, nx), floor, np.float32)
    top = np.full((nz, nx), floor + ceiling, np.float32)
    cover = np.zeros((nz, nx), np.uint8)
    x0, z0, x1, z1 = open_box
    cover[z0:z1, x0:x1] = 1
    ground[cover == 0] = floor + 1.5      # walls rise where there is no floor
    top[cover == 0] = floor + ceiling
    (root / "viewer_assets" / "collision.json").write_text(json.dumps({
        "nx": nx, "nz": nz, "cell": cell, "origin_xz": [0.0, 0.0],
        "character_height": 1.75, "scale_m_per_unit": 1.0, "object_colliders": 0}))
    (root / "viewer_assets" / "ground.f32").write_bytes(ground.tobytes())
    (root / "viewer_assets" / "heights.f32").write_bytes(top.tobytes())
    (root / "viewer_assets" / "coverage.u8").write_bytes(cover.tobytes())
    return root


class WorkspacePlaceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scene = make_room(self.root / "room")

    def tearDown(self):
        self._tmp.cleanup()

    def test_library_items_have_real_metric_dimensions(self):
        lib = wp.library()
        by_key = {i["item"]: i for i in lib}
        self.assertIn("sofa", by_key)
        length, height, depth = by_key["sofa"]["size"]
        # A sofa is ~2 m long, under a metre tall, under a metre and a half deep.
        self.assertAlmostEqual(length, 2.0, delta=0.4)
        self.assertTrue(0.5 <= height <= 1.1, height)
        self.assertTrue(0.6 <= depth <= 1.2, depth)
        for item in lib:
            self.assertEqual(len(item["size"]), 3)
            self.assertGreater(item["footprint_m2"], 0.0)

    def test_unknown_item_is_refused_not_invented(self):
        with self.assertRaises(ValueError):
            wp.item_spec("spaceship")

    def test_floor_snap_reads_the_heightfield(self):
        # Centre of the open patch sits at the synthetic floor level.
        y = wp.floor_y(self.scene, 2.0, 2.0)
        self.assertIsNotNone(y)
        self.assertAlmostEqual(y, 0.0, delta=0.05)

    def test_floor_snap_off_the_grid_is_none_not_a_guess(self):
        self.assertIsNone(wp.floor_y(self.scene, 999.0, 999.0))

    def test_placement_in_the_middle_of_the_floor_fits(self):
        rec = wp.make_placement(self.scene, "sofa", 2.0, 2.0, yaw_deg=0.0)
        self.assertTrue(rec["fit"]["supported"], rec["fit"])
        self.assertTrue(rec["fit"]["valid"], rec["fit"])
        self.assertGreater(rec["fit"]["floor_clearance_m"], 0.5)
        # Snapped so the box sits on the floor: centre y = floor + height/2.
        self.assertAlmostEqual(rec["center_y"], rec["size"][1] / 2, delta=0.1)

    def test_placement_against_a_wall_reports_clearance(self):
        # Near the open-patch edge (cell 10 -> x=0.5 m) the sofa is tight to a wall.
        rec = wp.make_placement(self.scene, "sofa", 0.7, 2.0, yaw_deg=0.0)
        self.assertLess(rec["fit"]["floor_clearance_m"], 0.5)

    def test_placement_on_a_wall_is_not_supported(self):
        # x=0.1 m is outside the supported patch (walls), so a drop there is refused.
        rec = wp.make_placement(self.scene, "sofa", 0.1, 0.1, yaw_deg=0.0)
        self.assertFalse(rec["fit"]["supported"])
        self.assertFalse(rec["fit"]["valid"])
        self.assertIn("reason", rec["fit"])

    def test_tall_item_hits_the_ceiling(self):
        # A 2.05 m wardrobe in a low 1.9 m room cannot stand up even on open floor.
        low = make_room(self.root / "lowroom", ceiling=1.9)
        rec = wp.make_placement(low, "wardrobe", 2.0, 2.0, yaw_deg=0.0)
        self.assertTrue(rec["fit"]["supported"])
        self.assertLessEqual(rec["fit"]["ceiling_height_m"], 1.95)
        self.assertFalse(rec["fit"]["fits_height"], rec["fit"])
        self.assertFalse(rec["fit"]["valid"])

    def test_rotation_keeps_the_footprint_on_supported_floor(self):
        flat = wp.make_placement(self.scene, "dining_table", 2.0, 2.0, yaw_deg=0.0)
        turned = wp.make_placement(self.scene, "dining_table", 2.0, 2.0, yaw_deg=90.0)
        self.assertTrue(flat["fit"]["valid"] and turned["fit"]["valid"])
        # Footprint area is rotation invariant; the length axis swaps.
        self.assertAlmostEqual(flat["footprint_m2"], turned["footprint_m2"], delta=0.05)

    def test_scale_multiplies_the_dimensions(self):
        base = wp.make_placement(self.scene, "armchair", 2.0, 2.0)
        big = wp.make_placement(self.scene, "armchair", 2.0, 2.0, scale=1.5)
        self.assertAlmostEqual(big["size"][0], base["size"][0] * 1.5, delta=0.01)
        self.assertAlmostEqual(big["center_y"], base["center_y"] * 1.5, delta=0.05)

    def test_non_square_grid_reads_the_right_cells(self):
        # A tall grid (nx=40 across x, nz=80 down z) with floor only in the near-z
        # half. If (row, col) were swapped this would read the wrong axis and the
        # verdict would flip — exactly the bug a square 100x100 fixture hides.
        import numpy as np
        root = self.root / "tall"
        (root / "viewer_assets").mkdir(parents=True)
        nx, nz, cell = 40, 80, 0.1
        ground = np.full((nz, nx), 0.0, np.float32)
        cover = np.zeros((nz, nx), np.uint8)
        cover[4:76, 4:36] = 1                       # supported across the whole x span
        (root / "viewer_assets" / "collision.json").write_text(json.dumps(
            {"nx": nx, "nz": nz, "cell": cell, "origin_xz": [0.0, 0.0], "character_height": 1.75}))
        (root / "viewer_assets" / "ground.f32").write_bytes(ground.tobytes())
        (root / "viewer_assets" / "coverage.u8").write_bytes(cover.tobytes())
        # z = 1.0 m is row ~10 (well inside the supported band) -> stool fits.
        near = wp.make_placement(root, "stool", 2.0, 1.0, yaw_deg=0.0)
        self.assertTrue(near["fit"]["supported"], near["fit"])
        # z = 7.9 m is row ~78 (outside the supported band) -> not supported.
        far = wp.make_placement(root, "stool", 2.0, 7.9, yaw_deg=0.0)
        self.assertFalse(far["fit"]["supported"], far["fit"])

    def test_single_surface_scan_reports_ceiling_unknown_not_a_false_failure(self):
        # A 2.5D scan skin has top == floor (no captured ceiling). A tall wardrobe
        # must NOT be failed on headroom it cannot measure — only the floor verdict.
        flat = make_room(self.root / "flat", ceiling=0.0)
        rec = wp.make_placement(flat, "wardrobe", 2.0, 2.0, yaw_deg=0.0)
        self.assertTrue(rec["fit"]["supported"], rec["fit"])
        self.assertIsNone(rec["fit"]["ceiling_height_m"])
        self.assertTrue(rec["fit"]["fits_height"], rec["fit"])
        self.assertTrue(rec["fit"]["valid"], rec["fit"])

    def test_placement_carries_the_viewer_box_schema(self):
        rec = wp.make_placement(self.scene, "bed_double", 2.0, 2.0, yaw_deg=12.0)
        for key in ("id", "item", "label", "center_xz", "center_y", "size", "yaw_deg", "fit"):
            self.assertIn(key, rec)
        self.assertEqual(len(rec["center_xz"]), 2)
        self.assertEqual(len(rec["size"]), 3)


if __name__ == "__main__":
    unittest.main()
