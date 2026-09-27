import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import workspace_place as wp


def make_glb(positions, indices, mn, mx, node=None):
    """A minimal, self-contained glTF 2.0 .glb whose POSITION accessor declares mn/mx."""
    binarr = bytearray(b"".join(struct.pack("<3f", *p) for p in positions))
    while len(binarr) % 4:
        binarr += b"\x00"
    idx_start = len(binarr)
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
    binarr += idx_bytes
    while len(binarr) % 4:
        binarr += b"\x00"
    nodes = node if node is not None else [{"mesh": 0, "name": "cube"}]
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions) * 12},
            {"buffer": 0, "byteOffset": idx_start, "byteLength": len(idx_bytes)},
        ],
        "buffers": [{"byteLength": len(binarr)}],
    }
    j = json.dumps(doc).encode("utf-8")
    while len(j) % 4:
        j += b" "
    jp = struct.pack("<II", len(j), 0x4E4F534A) + j
    bp = struct.pack("<II", len(binarr), 0x004E4942) + bytes(binarr)
    return b"glTF" + struct.pack("<II", 2, 12 + len(jp) + len(bp)) + jp + bp


def cube(size=1.0, offset=(0.0, 0.0, 0.0)):
    s = size / 2
    pts = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s), (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
    pts = [(x + offset[0], y + offset[1], z + offset[2]) for x, y, z in pts]
    idx = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
    mn = [min(p[a] for p in pts) for a in range(3)]
    mx = [max(p[a] for p in pts) for a in range(3)]
    return make_glb(pts, idx, mn, mx)


class GltfBoundsTests(unittest.TestCase):
    """A model's real metre size comes from its accessors, or it is refused — never guessed."""

    def test_reads_the_true_size_from_position_accessors(self):
        result = wp.read_model_bounds(cube(2.0), "sofa.glb")
        self.assertEqual(result["size"], [2.0, 2.0, 2.0])
        self.assertEqual(result["min"], [-1.0, -1.0, -1.0])
        self.assertEqual(result["max"], [1.0, 1.0, 1.0])

    def test_node_transform_is_applied_to_the_bounds(self):
        # A root node scaled 4x: the true size is the scaled box, not the raw extents.
        pts, idx = [(0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 3)], [0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3]
        scaled = make_glb(pts, idx, [0, 0, 0], [2, 1, 3], node=[{"scale": [4, 4, 4], "children": [1]}, {"mesh": 0}])
        self.assertEqual(wp.read_model_bounds(scaled, "scaled.glb")["size"], [8.0, 4.0, 12.0])

    def test_offset_model_bounds_follow_the_geometry(self):
        result = wp.read_model_bounds(cube(1.0, offset=(1.0, 0.5, -2.0)), "off.glb")
        self.assertEqual(result["size"], [1.0, 1.0, 1.0])
        self.assertEqual(result["min"], [0.5, 0.0, -2.5])

    def test_embedded_data_uri_gltf_is_accepted(self):
        doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
               "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
               "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                              "min": [0, 0, 0], "max": [0.5, 1.0, 0.4]}],
               "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
               "buffers": [{"byteLength": 36, "uri": "data:application/octet-stream;base64,AAAAAAAA"}]}
        self.assertEqual(wp.read_model_bounds(json.dumps(doc).encode(), "m.gltf")["size"], [0.5, 1.0, 0.4])

    def test_absurd_extent_is_refused_not_placed(self):
        pts, idx = cube_data_big()
        with self.assertRaises(ValueError) as ctx:
            wp.read_model_bounds(make_glb(pts, idx, [0, 0, 0], [5000, 1, 1]), "scene.glb")
        self.assertIn("beyond", str(ctx.exception))

    def test_nan_accessor_is_refused(self):
        with self.assertRaises(ValueError):
            wp.read_model_bounds(self._accessor_doc(min=[float("nan"), 0, 0]), "nan.gltf")

    def test_missing_bounds_never_default_to_one_metre(self):
        acc = {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
        doc = self._accessor_doc(acc=acc)
        with self.assertRaises(ValueError) as ctx:
            wp.read_model_bounds(json.dumps(doc).encode(), "nobounds.gltf")
        self.assertIn("cannot be read", str(ctx.exception))

    def test_external_and_traversal_buffers_are_refused(self):
        for uri in ("../secret.bin", "/etc/passwd", "meshes/geo.bin", "C:\\evil\\a.bin", "https://x/y.bin"):
            with self.subTest(uri=uri):
                doc = self._accessor_doc(extra_buffers=[{"byteLength": 36, "uri": uri}])
                doc["buffers"] = [{"byteLength": 36, "uri": uri}]
                with self.assertRaises(ValueError):
                    wp.read_model_bounds(json.dumps(doc).encode(), "ext.gltf")

    def test_self_referencing_node_graph_is_refused(self):
        doc = self._accessor_doc()
        doc["nodes"] = [{"mesh": 0, "children": [0]}]   # node 0 contains itself
        with self.assertRaises(ValueError) as ctx:
            wp.read_model_bounds(json.dumps(doc).encode(), "cycle.gltf")
        self.assertIn("self-referencing", str(ctx.exception))

    def test_glb_chunks_out_of_the_mandated_order_are_refused(self):
        # glTF 2.0: the JSON chunk MUST be first and the BIN chunk second. The
        # viewer's own glTF parser enforces this, so accepting the other order here
        # would import an item that silently renders as a box.
        swapped = swap_glb_chunks(cube(1.5))
        with self.assertRaises(ValueError) as ctx:
            wp.read_model_bounds(swapped, "swapped.glb")
        self.assertIn("JSON chunk first", str(ctx.exception))

    def test_geometry_needing_a_decoder_the_viewer_lacks_is_refused(self):
        for ext, where in (("KHR_draco_mesh_compression", "primitive"), ("EXT_meshopt_compression", "bufferView")):
            with self.subTest(ext=ext):
                doc = self._accessor_doc()
                if where == "primitive":
                    doc["meshes"][0]["primitives"][0]["extensions"] = {
                        ext: {"bufferView": 1, "attributes": {"POSITION": 0}}}
                else:
                    doc["bufferViews"][0]["extensions"] = {
                        ext: {"buffer": 0, "byteOffset": 0, "byteLength": 36, "byteStride": 12,
                              "count": 3, "mode": "ATTRIBUTES", "decoder": "Meshoptimizer"}}
                doc["extensionsUsed"] = [ext]
                with self.assertRaises(ValueError) as ctx:
                    wp.read_model_bounds(json.dumps(doc).encode(), "compressed.gltf")
                self.assertIn(ext, str(ctx.exception))

    def test_a_visual_only_extension_still_imports(self):
        # Refusing undecodable *geometry* must not become refusing every extension:
        # these change how a model looks, and the viewer draws them fine.
        doc = self._accessor_doc()
        doc["materials"] = [{"name": "oak", "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 2.0}}}]
        doc["meshes"][0]["primitives"][0]["material"] = 0
        doc["extensionsUsed"] = ["KHR_materials_emissive_strength"]
        self.assertEqual(wp.read_model_bounds(json.dumps(doc).encode(), "look.gltf")["size"], [1.0, 1.0, 1.0])

    def test_interleaved_vertices_under_a_rotated_scaled_root_read_their_real_size(self):
        # The shape a real exporter emits: a root node carrying the up-axis rotation
        # and unit scale, above a child that supplies a column-major matrix.
        pts, idx = [(0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 3)], [0, 1, 2, 0, 2, 3]
        root = [{"rotation": [0, 0.7071067811865476, 0, 0.7071067811865476], "scale": [2, 2, 2], "children": [1]},
                {"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "mesh": 0}]
        glb = make_glb(pts, idx, [0, 0, 0], [2, 1, 3], node=root)
        # A 90 deg turn about Y maps (x, y, z) to (z, y, -x), then the root doubles it.
        self.assertEqual(wp.read_model_bounds(glb, "rot.glb")["size"], [6.0, 2.0, 4.0])

    def test_garbage_and_empty_are_refused(self):
        for payload, name in ((b"", "e.glb"), (b"not a gltf model", "x.glb"),
                              (b"glTF" + (2).to_bytes(4, "little") + (40).to_bytes(4, "little") + b"\x00" * 24, "y.glb")):
            with self.subTest(name=name):
                self.assertRaises(ValueError, wp.read_model_bounds, payload, name)

    def _accessor_doc(self, acc=None, min=None, extra_buffers=None):
        accessors = [acc or {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                             "min": min or [0, 0, 0], "max": [1, 1, 1]}]
        return {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": accessors,
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
                "buffers": extra_buffers or [{"byteLength": 36}]}


def swap_glb_chunks(glb):
    """The same file with its two chunks emitted in the other order."""
    total = int.from_bytes(glb[8:12], "little")
    json_len = int.from_bytes(glb[12:16], "little")
    json_c, bin_c = glb[12:20 + json_len], glb[20 + json_len:total]
    assert int.from_bytes(bin_c[4:8], "little") == 0x004E4942, "fixture: second chunk is not BIN"
    return glb[:12] + bin_c + json_c


def cube_data_big():
    s = 2500
    pts = [(-s, -1, -1), (s, -1, -1), (s, 1, -1), (-s, 1, -1), (-s, -1, 1), (s, -1, 1), (s, 1, 1), (-s, 1, 1)]
    return pts, [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]


class ImportedPlacementTests(unittest.TestCase):
    """An imported model places with its real file-read size and the scene's scale caveat."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scene = make_room(self.root / "room")
        (self.scene / "frame.json").write_text(json.dumps(
            {"scale_m_per_unit": 4.7, "scale_source": "flight speed x clip duration"}))
        self.model = {
            "id": "cafe0000cafe0000", "label": "my sofa", "file": "cafe0000cafe0000.glb",
            "size": [2.4, 0.9, 1.1], "source": "gltf", "bytes": 1000, "created_at": "now",
        }
        (self.scene / "models").mkdir()
        (self.scene / "models" / "index.json").write_text(json.dumps([self.model]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_item_spec_resolves_the_imported_size(self):
        spec = wp.item_spec(self.model["id"], self.scene)
        self.assertTrue(spec["imported"])
        self.assertEqual(spec["size"], [2.4, 0.9, 1.1])

    def test_unknown_item_still_refused_even_with_a_scene(self):
        with self.assertRaises(ValueError):
            wp.item_spec("nope", self.scene)

    def test_scene_library_lists_imported_alongside_catalogue(self):
        items = {i["item"]: i for i in wp.scene_library(self.scene)}
        self.assertIn("sofa", items)                       # 12 primitives keep working
        self.assertTrue(items["sofa"].get("imported") is None)
        self.assertTrue(items[self.model["id"]]["imported"])
        self.assertEqual(items[self.model["id"]]["size"], [2.4, 0.9, 1.1])
        self.assertEqual(items[self.model["id"]]["model"]["scale_status"], "estimated")

    def test_placement_uses_real_size_and_carries_model_descriptor(self):
        rec = wp.make_placement(self.scene, self.model["id"], 2.0, 2.0)
        self.assertEqual(rec["size"], [2.4, 0.9, 1.1])
        self.assertEqual(rec["model"]["file"], "cafe0000cafe0000.glb")
        self.assertEqual(rec["model"]["scale_status"], "estimated")
        self.assertTrue(rec["fit"]["supported"], rec["fit"])

    def test_estimate_scale_leaves_an_honest_caveat_in_the_reason(self):
        rec = wp.make_placement(self.scene, self.model["id"], 2.0, 2.0)
        self.assertIn("estimated", rec["fit"]["reason"])
        self.assertIn("flight speed", rec["fit"]["reason"])
        # The 3D label splits on ';' — the caveat must live after the verdict, not in it.
        self.assertNotIn("estimated", rec["fit"]["reason"].split(";")[0])

    def test_metric_scene_caveat_says_metric(self):
        (self.scene / "frame.json").write_text(json.dumps(
            {"scale_m_per_unit": 1.0, "scale_source": "AR pose-prior metric path"}))
        rec = wp.make_placement(self.scene, self.model["id"], 2.0, 2.0)
        self.assertEqual(rec["model"]["scale_status"], "metric")
        self.assertIn("metric", rec["fit"]["reason"])

    def test_catalogue_placements_have_no_model_field(self):
        rec = wp.make_placement(self.scene, "sofa", 2.0, 2.0)
        self.assertNotIn("model", rec)


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

    def test_top_surface_is_not_evidence_of_a_ceiling(self):
        for top_height in (0.0, 1.9, 2.5):
            with self.subTest(top_height=top_height):
                room = make_room(self.root / str(top_height), ceiling=top_height)
                fit = wp.make_placement(room, "wardrobe", 2.0, 2.0)["fit"]
                self.assertTrue(fit["supported"])
                self.assertTrue(fit["valid"])  # floor suitability only
                self.assertIsNone(fit["ceiling_height_m"])
                self.assertIsNone(fit["fits_height"])
                self.assertIn("ceiling", fit["reason"])
                self.assertIn("unknown", fit["reason"])
                self.assertIn("coverage", fit["clearance_basis"])
                self.assertNotIn("wall", fit["reason"])

    def test_off_grid_samples_count_against_the_whole_footprint(self):
        room = make_room(self.root / "edge", open_box=(0, 0, 100, 100))
        # 5 x samples at -0.25, 0.25, 0.75, 1.25, 1.75: one column off-grid.
        fit = wp.fit_check(room, 0.75, 2.5, 2.0, 0.5, 0.0, 0.5)
        self.assertAlmostEqual(fit["support_fraction"], 0.8)
        self.assertFalse(fit["valid"])  # 80% is not permission to overhang the scan
        self.assertFalse(fit["supported"])
        self.assertIn("outside", fit["reason"])

    def test_small_unsupported_holes_are_not_silently_filled(self):
        va = self.scene / "viewer_assets"
        cover = np.fromfile(va / "coverage.u8", dtype=np.uint8).reshape(100, 100)
        cover[40, 40] = 0
        (va / "coverage.u8").write_bytes(cover.tobytes())
        fit = wp.fit_check(self.scene, 2.025, 2.025, 0.2, 0.2, 0, 0.5)
        self.assertFalse(fit["valid"])
        self.assertLess(fit["support_fraction"], 1.0)
        self.assertNotIn("wall", fit["reason"])

    def test_grid_cache_refreshes_when_any_payload_changes(self):
        import os
        va = self.scene / "viewer_assets"
        original_header = (va / "collision.json").stat().st_mtime_ns
        for name, dtype, value, field in (("ground.f32", "<f4", 0.4, "floor"),
                                           ("heights.f32", "<f4", 1.2, "top"),
                                           ("coverage.u8", "u1", 0, "supported")):
            with self.subTest(name=name):
                old = wp._grid(self.scene)
                path = va / name
                stamp = path.stat()
                np.full((100, 100), value, dtype=dtype).tofile(path)
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10000000))
                new = wp._grid(self.scene)
                self.assertIsNot(old, new)
                self.assertAlmostEqual(float(new[field][40, 40]), value, places=6)
        self.assertEqual((va / "collision.json").stat().st_mtime_ns, original_header)
        self.assertIsNone(wp.floor_y(self.scene, 2.025, 2.025))

    def test_grid_cache_is_bounded_across_revisions_and_scenes(self):
        with patch.object(wp, "_GRID_CACHE", {}):
            for index in range(20):
                scene = make_room(self.root / f"cached{index}", nx=10, nz=10,
                                  open_box=(0, 0, 10, 10))
                wp._grid(scene)
            self.assertLessEqual(len(wp._GRID_CACHE), 8)

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
        self.assertIsNone(rec["fit"]["fits_height"])
        self.assertTrue(rec["fit"]["valid"], rec["fit"])

    def test_placement_carries_the_viewer_box_schema(self):
        rec = wp.make_placement(self.scene, "bed_double", 2.0, 2.0, yaw_deg=12.0)
        for key in ("id", "item", "label", "center_xz", "center_y", "size", "yaw_deg", "fit"):
            self.assertIn(key, rec)
        self.assertEqual(len(rec["center_xz"]), 2)
        self.assertEqual(len(rec["size"]), 3)

    def test_an_edited_size_is_honoured_and_re_fit(self):
        # The 3D editor resizes a piece; the record must carry the edited box and
        # re-run the floor check against the new footprint, not the catalogue one.
        catalogue = wp.item_spec("coffee_table")["size"]
        rec = wp.make_placement(self.scene, "coffee_table", 2.0, 2.0, size=[2.4, 0.9, 1.2])
        self.assertEqual(rec["size"], [2.4, 0.9, 1.2])
        self.assertNotEqual(rec["size"], list(catalogue))
        self.assertEqual(rec["footprint_m2"], round(2.4 * 1.2, 3))
        self.assertTrue(rec["fit"]["supported"], rec["fit"])
        # Sitting on the floor means the box centre is half its own height up.
        self.assertAlmostEqual(rec["center_y"], 0.45, places=3)

    def test_a_lifted_placement_keeps_its_offset_above_the_floor(self):
        rec = wp.make_placement(self.scene, "side_table", 2.0, 2.0)
        self.assertEqual(rec["lift_m"], 0.0)
        lifted = wp.make_placement(self.scene, "side_table", 2.0, 2.0, center_y=rec["center_y"] + 0.4)
        self.assertAlmostEqual(lifted["lift_m"], 0.4, places=3)
        self.assertAlmostEqual(lifted["center_y"], rec["center_y"] + 0.4, places=3)
        # An item pushed down stops at the clamp instead of sinking through the slab.
        sunk = wp.make_placement(self.scene, "side_table", 2.0, 2.0, center_y=-5.0)
        self.assertAlmostEqual(sunk["lift_m"], -0.2, places=3)
        self.assertAlmostEqual(sunk["center_y"], rec["center_y"] - 0.2, places=3)

    def test_lift_is_clamped_to_a_sane_range(self):
        rec = wp.make_placement(self.scene, "stool", 2.0, 2.0, center_y=999.0)
        self.assertLessEqual(rec["lift_m"], 3.0)
        self.assertGreater(rec["center_y"], 0.0)


if __name__ == "__main__":
    unittest.main()
