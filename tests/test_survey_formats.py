"""CPU-only tests for the official-format writers, each checked by an independent reader.

The independent readers here use only ``struct``, ``csv``, ``json`` and ``re`` rather
than the module's own ``read_*`` helpers, so a mistake in a writer cannot be hidden by
the matching mistake in its reader. That still proves only self-consistency: no
PDAL/laspy/GDAL/rasterio is installed here, so nothing in this file proves spec
compliance or that any output opens in CloudCompare or QGIS.
"""
import csv
import importlib
import json
import math
import re
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0],
                     [0.0, 2.0, 0.0], [0.5, 1.0, 1.5]], dtype=float)
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1], [1, 4, 2], [2, 4, 3],
                      [3, 4, 0]], dtype=np.int64)
POINTS = np.array([[1.5, -2.25, 10.125], [0.0, 0.0, 0.0], [1000.0, 2.0, -3.5]], dtype=float)
GEO_WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984"],PRIMEM["Greenwich",0],UNIT["degree",0.01745]]'
UTM_WKT = ('PROJCS["WGS 84 / UTM zone 32N",GEOGCS["WGS 84",DATUM["WGS_1984"],'
           'PRIMEM["Greenwich",0],UNIT["degree",0.01745]],PROJ["Transverse_Mercator"],'
           'UNIT["metre",1],AUTHORITY["EPSG","32632"]]')


class SurveyFormatsTests(unittest.TestCase):
    """Byte-level checks of every writer, plus the honesty of what each one reports."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.fmt = importlib.import_module("scripts.survey_formats")
        except ModuleNotFoundError as exc:
            if exc.name != "scripts.survey_formats":
                raise
            raise unittest.SkipTest("scripts.survey_formats has not been implemented")

    def setUp(self):
        keep = tempfile.TemporaryDirectory()
        self.addCleanup(keep.cleanup)
        self.root = Path(keep.name)

    def path(self, name):
        return self.root / name

    # ----------------------------------------------------------- independent re-readers
    @staticmethod
    def obj_parse(text):
        vertices, uvs, faces, meta = [], [], [], []
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(value) for value in parts[1:4]])
            elif parts[0] == "vt":
                uvs.append([float(value) for value in parts[1:3]])
            elif parts[0] == "f":
                faces.append([token.split("/") for token in parts[1:]])
            elif parts[0] in ("usemtl", "mtllib", "o"):
                meta.append((parts[0], parts[1] if len(parts) > 1 else ""))
        return vertices, uvs, faces, meta

    @staticmethod
    def fbx_numbers(text, key):
        for line in text.splitlines():
            if line.strip().startswith(key + ":"):
                payload = line.split(":", 1)[1].strip()
                return [float(value) for value in payload.split(",") if value.strip()]
        return None

    @staticmethod
    def las_header(text):
        return dict(
            signature=text[0:4], version=(text[24], text[25]),
            header_size=struct.unpack_from("<H", text, 94)[0],
            offset_to_point_data=struct.unpack_from("<I", text, 96)[0],
            n_vlrs=struct.unpack_from("<I", text, 100)[0],
            vlr_bytes=struct.unpack_from("<I", text, 104)[0],
            point_format=text[108],
            record_length=struct.unpack_from("<H", text, 109)[0],
            n_points=struct.unpack_from("<I", text, 111)[0],
            n_points_64=struct.unpack_from("<Q", text, 135)[0],
            scale=struct.unpack_from("<3d", text, 143),
            offset=struct.unpack_from("<3d", text, 167),
            max_min=struct.unpack_from("<6d", text, 191),
            software_id=text[58:90].split(b"\x00")[0].decode())

    @staticmethod
    def las_vlrs(text, offset, count):
        out, cursor = [], offset
        for _ in range(count):
            reserved, user_id, record_id, description, length = \
                struct.unpack_from("<H16sH32sH", text, cursor)
            out.append(dict(reserved=reserved, user_id=user_id.split(b"\x00")[0].decode(),
                            record_id=record_id, description=description.split(b"\x00")[0].decode(),
                            length=length, payload=text[cursor + 54:cursor + 54 + length]))
            cursor += 54 + length
        return out

    @staticmethod
    def tiff_ifd(text):
        """Parse the IFD with struct only. Type 2 (ASCII) is ONE string of ``count`` bytes.

        The count of an ASCII tag includes its NUL terminator, so the terminator is kept
        here rather than stripped: a caller that drops it cannot prove it was written.
        """
        assert text[:2] == b"II", "GeoTIFF must be little-endian"
        assert struct.unpack_from("<H", text, 2)[0] == 42
        start = struct.unpack_from("<I", text, 4)[0]
        count = struct.unpack_from("<H", text, start)[0]
        sizes = {1: 1, 2: 1, 3: 2, 4: 4, 12: 8}
        codes = {1: "B", 2: "s", 3: "H", 4: "I", 12: "d"}
        out = {}
        for index in range(count):
            base = start + 2 + index * 12
            tag, type_id, length = struct.unpack_from("<HHI", text, base)
            inline = text[base + 8:base + 12]
            total = sizes[type_id] * length
            blob = inline[:total] if total <= 4 else text[struct.unpack_from("<I", inline)[0]:
                                                          struct.unpack_from("<I", inline)[0] + total]
            if type_id == 2:
                values = [blob.decode("ascii", errors="replace")]
            else:
                values = list(struct.unpack("<" + codes[type_id] * length, blob))
            out[tag] = dict(type=type_id, count=length, values=values)
        return out

    @staticmethod
    def geo_keys(tags):
        raw = tags[34735]["values"]
        return dict(header=list(raw[:4]),
                    keys=[tuple(raw[4 + i * 4:8 + i * 4]) for i in range(raw[3])])

    # --------------------------------------------------------------------- honesty first
    def test_module_reports_round_trip_only_for_binary_formats(self):
        summary = self.fmt.verification_summary()
        self.assertEqual(summary["verified"], "round-trip only")
        self.assertFalse(summary["externally_validated"])
        self.assertEqual(summary["reference_libraries_available"],
                         {"laspy": False, "rasterio": False, "pyproj": False, "osgeo": False})
        for name in ("obj", "gltf", "fbx", "las", "geotiff"):
            self.assertTrue(summary["formats"][name]["round_trip_only"], name)
            self.assertTrue(summary["formats"][name]["unverified"], name)
        for name in ("xyz", "csv"):
            self.assertIn("plain text", summary["formats"][name]["verified"])
        self.assertIn("write_las", summary["formats"]["las"]["validated_by"])

    def test_every_writer_returns_the_same_disclaimer(self):
        results = [self.fmt.write_obj(TRIANGLES, VERTICES, self.path("a.obj")),
                   self.fmt.write_gltf(POINTS, None, self.path("a.gltf")),
                   self.fmt.write_fbx(POINTS, self.path("a.fbx")),
                   self.fmt.write_las(POINTS, self.path("a.las"), scale=(.001, .001, .001),
                                      offsets=(0, 0, 0)),
                   self.fmt.write_geotiff(np.zeros((2, 2)), self.path("a.tif"),
                                          transform=(0, 1, 0, 0, 0, -1), crs_wkt=GEO_WKT),
                   self.fmt.write_xyz(POINTS, self.path("a.xyz")),
                   self.fmt.write_csv(POINTS, self.path("a.csv"))]
        for result in results:
            self.assertFalse(result["externally_validated"])
            self.assertIn("path", result)
        self.assertEqual(results[0]["verified"], "round-trip only")
        self.assertEqual(results[-1]["verified"], "plain text, externally verifiable by inspection")

    def test_writers_refuse_nan_and_none(self):
        calls = [lambda: self.fmt.write_obj(TRIANGLES, np.vstack([VERTICES, [np.nan, 0, 0]]),
                                            self.path("n1.obj")),
                 lambda: self.fmt.write_gltf(np.array([[0.0, np.nan, 0.0]]), self.path("n2.gltf")),
                 lambda: self.fmt.write_las(np.array([[0.0, 0.0, np.inf]]), self.path("n3.las"),
                                            scale=(.001, .001, .001), offsets=(0, 0, 0)),
                 lambda: self.fmt.write_fbx(np.array([[0.0, 0.0, None]], dtype=object),
                                            self.path("n4.fbx")),
                 lambda: self.fmt.write_xyz([None, None], self.path("n5.xyz"))]
        for call in calls:
            with self.assertRaises(ValueError):
                call()

    # --------------------------------------------------------------------------- OBJ
    def test_write_obj_round_trips_with_one_based_indices(self):
        out = self.path("mesh.obj")
        result = self.fmt.write_obj(TRIANGLES, VERTICES, out)
        vertices, uvs, faces, meta = self.obj_parse(out.read_text(encoding="utf-8"))
        np.testing.assert_allclose(vertices, VERTICES, rtol=1e-9, atol=1e-12)
        self.assertEqual([[int(part[0]) for part in row] for row in faces],
                         [[int(v) + 1 for v in row] for row in TRIANGLES])
        self.assertEqual(uvs, [])
        self.assertEqual(result["vertices"], len(VERTICES))
        self.assertEqual(result["triangles"], len(TRIANGLES))
        self.assertIn(("o", "mesh"), meta)
        read = self.fmt.read_obj(out)
        np.testing.assert_allclose(read["vertices"], VERTICES, rtol=1e-9)
        np.testing.assert_array_equal(read["triangles"], TRIANGLES)

    def test_write_obj_textures_per_vertex_and_per_corner(self):
        per_vertex = self.path("pv.obj")
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
        self.fmt.write_obj(TRIANGLES, VERTICES, per_vertex, texture_coords=uvs,
                           material="facade.mtl")
        _, parsed, faces, meta = self.obj_parse(per_vertex.read_text(encoding="utf-8"))
        np.testing.assert_allclose(parsed, uvs, rtol=1e-7)
        self.assertTrue(all(part[1] == part[0] for part in faces[0]),
                        "per-vertex UVs reuse the vertex index")
        self.assertIn(("mtllib", "facade.mtl"), meta)
        self.assertIn(("usemtl", "facade.mtl"), meta)
        read = self.fmt.read_obj(per_vertex)
        self.assertEqual(read["material"], "facade.mtl")
        self.assertEqual(read["triangles"].tolist(), TRIANGLES.tolist())

        per_corner = self.path("pc.obj")
        corner_uvs = np.array([[i / 18.0, 1.0 - i / 18.0] for i in range(len(TRIANGLES) * 3)])
        self.fmt.write_obj(TRIANGLES, VERTICES, per_corner, texture_coords=corner_uvs)
        _, parsed, faces, _ = self.obj_parse(per_corner.read_text(encoding="utf-8"))
        self.assertEqual(len(parsed), len(TRIANGLES) * 3)
        self.assertEqual([part[1] for part in faces[1]], ["4", "5", "6"])
        with self.assertRaisesRegex(ValueError, "texture_coords"):
            self.fmt.write_obj(TRIANGLES, VERTICES, self.path("bad.obj"),
                               texture_coords=np.zeros((7, 2)))

    def test_write_obj_rejects_bad_geometry(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.fmt.write_obj(np.array([[0, 1, 99]]), VERTICES, self.path("x.obj"))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.fmt.write_obj(np.array([[0, 1]]), VERTICES, self.path("y.obj"))
        with self.assertRaisesRegex(ValueError, "degenerate"):
            self.fmt.write_obj(np.array([[1, 1, 2]]), VERTICES, self.path("z.obj"))

    def test_read_obj_rejects_negative_and_out_of_range_indices(self):
        out = self.path("neg.obj")
        out.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf v1 v2 v3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a number"):
            self.fmt.read_obj(out)
        out.write_text("v 0 0 0\nv 1 0 0\nf 1 2 -3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "negative"):
            self.fmt.read_obj(out)
        out.write_text("v 0 0 0\nv 1 0 0\nf 1 2 7\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.fmt.read_obj(out)
        out.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "triangle"):
            self.fmt.read_obj(out)

    # ------------------------------------------------------------------------- glTF
    def test_write_gltf_point_cloud_with_external_buffer(self):
        out = self.path("cloud.gltf")
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        result = self.fmt.write_gltf(POINTS, colors, out)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(doc["asset"]["version"], "2.0")
        self.assertEqual(doc["meshes"][0]["primitives"][0]["mode"], 0)
        self.assertEqual(doc["buffers"][0]["uri"], "cloud.bin")
        blob = self.path("cloud.bin").read_bytes()
        self.assertEqual(doc["buffers"][0]["byteLength"], len(blob))
        self.assertEqual(len(blob) % 4, 0)
        primitive = doc["meshes"][0]["primitives"][0]
        position = doc["accessors"][primitive["attributes"]["POSITION"]]
        self.assertEqual([position["componentType"], position["type"], position["count"]],
                         [5126, "VEC3", 3])
        np.testing.assert_allclose(position["min"], POINTS.min(axis=0), atol=1e-6)
        np.testing.assert_allclose(position["max"], POINTS.max(axis=0), atol=1e-6)
        color = doc["accessors"][primitive["attributes"]["COLOR_0"]]
        self.assertEqual([color["componentType"], color["type"]], [5121, "VEC4"])
        self.assertEqual(result["externally_validated"], False)
        read = self.fmt.read_gltf(out)
        np.testing.assert_allclose(read["positions"], POINTS, atol=1e-6)
        np.testing.assert_array_equal(read["colors"][:, :3], colors)

    def test_write_gltf_triangle_mesh_with_embedded_uri(self):
        out = self.path("mesh.gltf")
        self.fmt.write_gltf(VERTICES, TRIANGLES, out, embed=True)
        doc = json.loads(out.read_text(encoding="utf-8"))
        primitive = doc["meshes"][0]["primitives"][0]
        self.assertEqual(primitive["mode"], 4)
        indices = doc["accessors"][primitive["indices"]]
        self.assertEqual(indices["componentType"], 5123)
        self.assertEqual(indices["count"], len(TRIANGLES) * 3)
        self.assertTrue(doc["buffers"][0]["uri"].startswith(
            "data:application/octet-stream;base64,"))
        self.assertFalse(self.path("mesh.bin").exists())
        read = self.fmt.read_gltf(out)
        np.testing.assert_allclose(read["positions"], VERTICES, atol=1e-6)
        np.testing.assert_array_equal(read["triangles"], TRIANGLES)

    def test_gltf_accessor_invariants_are_enforced_on_read(self):
        out = self.path("broken.gltf")
        self.fmt.write_gltf(POINTS, None, out)
        original = json.loads(out.read_text(encoding="utf-8"))

        def break_(**changes):
            doc = json.loads(json.dumps(original))
            for path, value in changes.items():
                node = doc
                keys = path.split(".")
                for key in keys[:-1]:
                    node = node[int(key)] if key.isdigit() else node[key]
                if value is None:
                    del node[keys[-1]]
                else:
                    node[keys[-1]] = value
            out.write_text(json.dumps(doc), encoding="utf-8")
            return doc

        break_(**{"accessors.0.byteOffset": 3})
        with self.assertRaisesRegex(ValueError, "align"):
            self.fmt.read_gltf(out)
        break_(**{"accessors.0.count": 4})
        with self.assertRaisesRegex(
                ValueError, r"accessor 0 count 4 needs 48 bytes but its bufferView "
                            r"byteLength is 36"):
            self.fmt.read_gltf(out)
        break_(**{"buffers.0.uri": "nowhere.bin"})
        with self.assertRaisesRegex(ValueError, "resolve"):
            self.fmt.read_gltf(out)
        break_(**{"accessors.0.min": None})
        with self.assertRaisesRegex(ValueError, "min"):
            self.fmt.read_gltf(out)
        break_(**{"accessors.0.componentType": 5125, "accessors.0.type": "VEC2"})
        with self.assertRaisesRegex(ValueError, "componentType"):
            self.fmt.read_gltf(out)
        break_(**{"bufferViews.0.byteOffset": 2})
        with self.assertRaisesRegex(ValueError, "align"):
            self.fmt.read_gltf(out)
        break_(**{"buffers.0.byteLength": 4})
        with self.assertRaisesRegex(ValueError, "byteLength"):
            self.fmt.read_gltf(out)

    def test_write_gltf_rejects_impossible_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.fmt.write_gltf(np.zeros((4, 2)), self.path("p.gltf"))
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.fmt.write_gltf(POINTS, np.array([[0, 1, 3]]), self.path("q.gltf"))
        with self.assertRaisesRegex(ValueError, "colors"):
            self.fmt.write_gltf(POINTS, np.zeros((2, 3), dtype=float), self.path("r.gltf"))
        with self.assertRaisesRegex(ValueError, "range"):
            self.fmt.write_gltf(POINTS, np.array([[0.0, 1.0, 9.0]] * 3), self.path("t.gltf"))
        ambiguous = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]])
        with self.assertRaisesRegex(ValueError, "mode"):
            self.fmt.write_gltf(VERTICES, ambiguous, self.path("u.gltf"))
        forced = self.fmt.write_gltf(VERTICES, ambiguous, self.path("v.gltf"), mode="triangles")
        self.assertEqual(forced["triangles"], len(ambiguous))

    # --------------------------------------------------------------------------- FBX
    def test_write_fbx_emits_only_the_constructs_it_claims(self):
        out = self.path("cloud.fbx")
        result = self.fmt.write_fbx(POINTS, out)
        text = out.read_text(encoding="utf-8")
        self.assertIn("; FBX 6.1.0 project file", text)
        self.assertIn("FBXHeaderExtension:", text)
        self.assertIn("FBXVersion: 6100", text)
        self.assertIn("Definitions:", text)
        self.assertIn("Objects:", text)
        self.assertIn("Connections:", text)
        self.assertIn('"Mesh"', text)
        self.assertIn("Vertices:", text)
        self.assertNotIn("PolygonVertexIndex", text)
        flat = self.fbx_numbers(text, "Vertices")
        self.assertEqual(len(flat), len(POINTS) * 3)
        np.testing.assert_allclose(flat, POINTS.reshape(-1), atol=1e-6)
        read = self.fmt.read_fbx(out)
        np.testing.assert_allclose(read["positions"], POINTS, atol=1e-6)
        self.assertEqual(read["fbx_version"], 6100)
        self.assertEqual(read["triangles"].size, 0)
        self.assertEqual(result["polygons"], 0)
        self.assertEqual(result["externally_validated"], False)
        self.assertIn("minimal", self.fmt.write_fbx.__doc__)
        self.assertIn("unverified", self.fmt.write_fbx.__doc__)

    def test_write_fbx_triangles_use_the_negative_last_corner_convention(self):
        out = self.path("mesh.fbx")
        result = self.fmt.write_fbx(VERTICES, out, triangles=TRIANGLES)
        line = next(row for row in out.read_text(encoding="utf-8").splitlines()
                    if "PolygonVertexIndex" in row)
        values = [int(v) for v in line.split(":", 1)[1].split(",")]
        self.assertEqual(len(values), len(TRIANGLES) * 3)
        self.assertEqual(values[:3], [0, 1, -3])   # FBX corners are 0-based, last is ~index
        self.assertEqual(result["polygons"], len(TRIANGLES))
        read = self.fmt.read_fbx(out)
        np.testing.assert_array_equal(read["triangles"], TRIANGLES)
        np.testing.assert_allclose(read["positions"], VERTICES, atol=1e-6)

    def test_read_fbx_refuses_a_binary_file(self):
        out = self.path("bad.fbx")
        out.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + bytes(40))
        with self.assertRaisesRegex(ValueError, "ASCII"):
            self.fmt.read_fbx(out)

    # --------------------------------------------------------------------------- LAS
    def test_write_las_header_and_points(self):
        out = self.path("cloud.las")
        offsets = (500000.0, 4000000.0, 100.0)
        # 5 mm, not 1 mm: LAS coordinates are int32, so at 1 mm one offset only reaches
        # +/-2147 km and a 4,000 km UTM northing cannot be represented at all. That refusal
        # is pinned down by test_write_las_refuses_a_utm_northing_at_one_millimetre_scale;
        # here the offsets are kept realistic and the scale is the one that fits them.
        scale = (0.005, 0.005, 0.005)
        result = self.fmt.write_las(POINTS, out, scale=scale, offsets=offsets)
        text = out.read_bytes()
        header = self.las_header(text)
        adjusted = POINTS - np.array(offsets)
        self.assertEqual(header["signature"], b"LASF")
        self.assertEqual(header["version"], (1, 4))
        self.assertEqual(header["header_size"], 239)
        self.assertEqual(header["n_points"], len(POINTS))
        self.assertEqual(header["n_points_64"], len(POINTS))
        self.assertEqual(header["point_format"], 3)
        self.assertEqual(header["record_length"], 34)
        self.assertEqual(header["scale"], scale)
        self.assertEqual(header["offset"], offsets)
        self.assertEqual(header["max_min"][:2], (float(adjusted[:, 0].max()),
                                                 float(adjusted[:, 0].min())))
        self.assertEqual(header["max_min"][4:], (float(adjusted[:, 2].max()),
                                                 float(adjusted[:, 2].min())))
        self.assertEqual(header["software_id"], "scripts/survey_formats.write_las")
        raw = np.frombuffer(text[header["offset_to_point_data"]:][:12], dtype="<3i4")
        np.testing.assert_array_equal(raw[0], np.round(adjusted[0] / 0.005).astype(np.int32))
        stored = np.array([struct.unpack_from("<3i", text[header["offset_to_point_data"]:],
                                              row * header["record_length"])
                           for row in range(len(POINTS))], dtype=np.int32)
        vlrs = self.las_vlrs(text, header["header_size"], header["n_vlrs"])
        self.assertEqual(header["offset_to_point_data"],
                         header["header_size"] + header["vlr_bytes"])
        # The declared VLR byte count is the VLR block padded up to a 4-byte boundary, which
        # is what keeps the point records that follow aligned. Asserting the padding exactly
        # rather than skipping it: a count below sum(54 + length) would put the point data
        # inside the VLRs.
        written = sum(54 + v["length"] for v in vlrs)
        self.assertEqual(header["vlr_bytes"], written + (-written % 4))
        self.assertEqual(vlrs[0]["user_id"], "LASF_proj")
        self.assertEqual(vlrs[0]["record_id"], 34735)
        self.assertEqual(self.fmt.read_las(out)["srs_wkt"], None)
        read = self.fmt.read_las(out)
        # read_las returns metres, i.e. int32 * scale + offset, so the error against the
        # input must stay inside the quantisation step the header declares.
        np.testing.assert_allclose(np.column_stack([read["points"][k] for k in "xyz"]),
                                   POINTS, atol=max(scale))
        np.testing.assert_array_equal(np.column_stack([read["raw_points"][k] for k in "xyz"]),
                                      stored)
        self.assertEqual(read["n_points"], len(POINTS))
        self.assertEqual(result["externally_validated"], False)
        self.assertTrue(any("field order" in item for item in result["unverified"]))

    def test_write_las_wkt_vlr_and_extra_columns(self):
        out = self.path("geo.las")
        data = {"x": POINTS[:, 0], "y": POINTS[:, 1], "z": POINTS[:, 2],
                "intensity": np.array([10, 20, 30], dtype=np.uint16),
                "classification": np.array([2, 6, 9], dtype=np.uint8),
                "gps_time": np.array([1.5, 2.5, 3.5]),
                "red": np.array([100, 200, 250], dtype=np.uint16),
                "green": np.array([100, 200, 250], dtype=np.uint16),
                "blue": np.array([100, 200, 250], dtype=np.uint16),
                "point_source_id": np.array([7, 8, 9], dtype=np.uint16)}
        self.fmt.write_las(data, out, scale=(0.001, 0.001, 0.001), offsets=(0.0, 0.0, 0.0),
                           srs_wkt=UTM_WKT)
        text = out.read_bytes()
        header = self.las_header(text)
        self.assertEqual(header["n_vlrs"], 2)
        vlrs = self.las_vlrs(text, header["header_size"], 2)
        self.assertEqual(vlrs[1]["user_id"], "LASF_Projection")
        self.assertEqual(vlrs[1]["record_id"], 2112)
        self.assertEqual(vlrs[1]["payload"].split(b"\x00")[0].decode(), UTM_WKT)
        keys = struct.unpack("<" + "H" * (len(vlrs[0]["payload"]) // 2), vlrs[0]["payload"])
        # struct.unpack hands back a tuple, so the byte-level view is compared as one; the
        # module's own readers return geokeys as a list everywhere (write_las's summary and
        # read_las agree on that), which the next block checks.
        self.assertEqual(keys[:4], (1, 1, 0, 3))
        read = self.fmt.read_las(out)
        self.assertEqual(read["srs_wkt"], UTM_WKT)
        self.assertIsInstance(read["geokeys"], list)
        self.assertEqual(read["geokeys"][:4], [1, 1, 0, 3])
        np.testing.assert_array_equal(read["points"]["classification"], [2, 6, 9])
        np.testing.assert_array_equal(read["points"]["intensity"], [10, 20, 30])
        np.testing.assert_array_equal(read["points"]["point_source_id"], [7, 8, 9])
        np.testing.assert_allclose(read["points"]["gps_time"], [1.5, 2.5, 3.5], atol=1e-9)
        np.testing.assert_array_equal(read["points"]["red"], [100, 200, 250])

    def test_write_las_refuses_bad_arguments(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            self.fmt.write_las(POINTS, self.path("a.las"), scale=(0.0, 0.001, 0.001),
                               offsets=(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "point_format"):
            self.fmt.write_las(POINTS, self.path("b.las"), scale=(.001, .001, .001),
                               offsets=(0, 0, 0), point_format=9)
        with self.assertRaisesRegex(ValueError, "int32"):
            self.fmt.write_las(np.array([[1e18, 0.0, 0.0]]), self.path("c.las"),
                               scale=(0.001, 0.001, 0.001), offsets=(0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "classification"):
            self.fmt.write_las({"x": [0.0], "y": [0.0], "z": [0.0], "classification": [999]},
                               self.path("d.las"), scale=(.001, .001, .001), offsets=(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "gps_time"):
            self.fmt.write_las({"x": [0.0], "y": [0.0], "z": [0.0], "gps_time": [1.0]},
                               self.path("e.las"), scale=(.001, .001, .001), offsets=(0, 0, 0),
                               point_format=0)
        with self.assertRaisesRegex(ValueError, "red"):
            self.fmt.write_las({"x": [0.0], "y": [0.0], "z": [0.0], "red": [5]},
                               self.path("f.las"), scale=(.001, .001, .001), offsets=(0, 0, 0),
                               point_format=3)
        with self.assertRaisesRegex(ValueError, "number_of_returns"):
            self.fmt.write_las({"x": [0.0], "y": [0.0], "z": [0.0], "number_of_returns": [9]},
                               self.path("g.las"), scale=(.001, .001, .001), offsets=(0, 0, 0))

    def test_read_las_many_points_and_truncation(self):
        rng = np.random.default_rng(2)
        points = rng.normal(0.0, 12.0, size=(500, 3))
        out = self.path("many.las")
        self.fmt.write_las(points, out, scale=(0.0005, 0.0005, 0.001), offsets=(0.0, 0.0, 0.0))
        read = self.fmt.read_las(out)
        self.assertEqual(read["n_points"], 500)
        np.testing.assert_allclose(read["points"]["x"], points[:, 0], atol=0.0005)
        np.testing.assert_allclose(read["points"]["z"], points[:, 2], atol=0.001)
        out.write_bytes(out.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError, "truncat"):
            self.fmt.read_las(out)
        other = self.path("wrongsig.las")
        other.write_bytes(b"LASX" + bytes(400))
        with self.assertRaisesRegex(ValueError, "LASF"):
            self.fmt.read_las(other)

    # ----------------------------------------------------------------------- GeoTIFF
    def test_write_geotiff_tags_and_raster(self):
        raster = np.array([[1.5, 2.5, np.nan], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=float)
        out = self.path("dsm.tif")
        self.fmt.write_geotiff(raster, out, transform=(1000.0, 2.0, 0.0, 5000.0, 0.0, -2.0),
                               crs_wkt=UTM_WKT, nodata=-9999.0)
        text = out.read_bytes()
        tags = self.tiff_ifd(text)
        self.assertEqual(tags[256]["values"], [3])
        self.assertEqual(tags[257]["values"], [3])
        self.assertEqual(tags[258]["values"], [32])
        self.assertEqual(tags[259]["values"], [1])
        self.assertEqual(tags[262]["values"], [1])
        self.assertEqual(tags[277]["values"], [1])
        self.assertEqual(tags[339]["values"], [3])
        self.assertEqual(tags[278]["values"], [3])
        self.assertEqual(tags[279]["values"], [raster.size * 4])
        self.assertEqual(tags[33550]["values"], [2.0, 2.0, 0.0])
        self.assertEqual(tags[33922]["values"], [0.0, 0.0, 0.0, 1000.0, 5000.0, 0.0])
        keys = self.geo_keys(tags)
        self.assertEqual(keys["header"], [1, 1, 0, len(keys["keys"])])
        self.assertIn((1024, 0, 1, 1), keys["keys"])
        self.assertIn((1025, 0, 1, 1), keys["keys"])
        self.assertIn((3072, 0, 1, 32632), keys["keys"])
        self.assertEqual(tags[42112]["count"], len("-9999") + 1)   # ASCII count holds the NUL
        self.assertEqual(tags[42112]["values"][0].strip("\x00"), "-9999")
        offset, length = tags[273]["values"][0], tags[279]["values"][0]
        flat = np.frombuffer(text[offset:offset + length], dtype="<f4")
        np.testing.assert_allclose(flat[:6], [1.5, 2.5, -9999.0, 4.5, 5.5, 6.5], atol=1e-6)
        read = self.fmt.read_geotiff(out)
        np.testing.assert_allclose(read["raster"][:2, :2], raster[:2, :2], atol=1e-6)
        self.assertTrue(math.isnan(read["raster"][0, 2]))
        self.assertEqual(read["nodata"], -9999.0)
        self.assertEqual(read["transform"], (1000.0, 2.0, 0.0, 5000.0, 0.0, -2.0))
        self.assertEqual(read["epsg"], 32632)
        self.assertEqual(read["crs_wkt"], UTM_WKT)
        self.assertEqual(read["shape"], (3, 3))

    def test_write_geotiff_refuses_unsupported_inputs(self):
        ok = dict(transform=(0, 1, 0, 0, 0, -1), crs_wkt=GEO_WKT)
        with self.assertRaisesRegex(ValueError, "2-D"):
            self.fmt.write_geotiff(np.zeros(6), self.path("a.tif"), **ok)
        with self.assertRaisesRegex(ValueError, "single-band|band"):
            self.fmt.write_geotiff(np.zeros((3, 3, 3)), self.path("b.tif"), **ok)
        with self.assertRaisesRegex(ValueError, "rotat"):
            self.fmt.write_geotiff(np.zeros((2, 2)), self.path("c.tif"),
                                   transform=(0, 1, 0, 0, 0.5, -1), crs_wkt=GEO_WKT)
        with self.assertRaisesRegex(ValueError, "nodata"):
            self.fmt.write_geotiff(np.array([[0.0, np.nan], [1.0, 2.0]]), self.path("d.tif"), **ok)
        with self.assertRaisesRegex(ValueError, "WKT"):
            self.fmt.write_geotiff(np.zeros((2, 2)), self.path("e.tif"), transform=ok["transform"],
                                   crs_wkt="not a crs")
        with self.assertRaisesRegex(ValueError, "row order"):
            self.fmt.write_geotiff(np.zeros((2, 2)), self.path("f.tif"),
                                   transform=(0, 1, 0, 0, 0, 1), crs_wkt=GEO_WKT)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.fmt.write_geotiff(np.zeros((2, 2)), self.path("g.tif"),
                                   transform=(0, 1, 0, 0, 0, -1, 5, 6), crs_wkt=GEO_WKT)

    def test_geotiff_without_an_epsg_code_declares_user_defined(self):
        out = self.path("ud.tif")
        self.fmt.write_geotiff(np.zeros((2, 2), dtype=float), out,
                               transform=(0, 1, 0, 0, 0, -1),
                               crs_wkt='GEOGCS["Local survey grid"]')
        keys = self.geo_keys(self.tiff_ifd(out.read_bytes()))["keys"]
        self.assertIn((1024, 0, 1, 2), keys)
        self.assertIn((1026, 0, 1, 32767), keys)
        read = self.fmt.read_geotiff(out)
        self.assertIsNone(read["epsg"])
        self.assertTrue(read["user_defined_crs"])

    def test_read_geotiff_refuses_a_big_endian_or_tagless_file(self):
        out = self.path("be.tif")
        out.write_bytes(struct.pack(">HHI", 77, 43, 8) + struct.pack("<H", 0))
        with self.assertRaisesRegex(ValueError, "little-endian|II"):
            self.fmt.read_geotiff(out)

    # ---------------------------------------------------------------- plain text outputs
    def test_write_xyz_is_a_plain_text_cloud(self):
        out = self.path("cloud.xyz")
        result = self.fmt.write_xyz(POINTS, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "1.5 -2.25 10.125")
        self.assertEqual(len(lines), 3)
        self.assertEqual(result["verified"], "plain text, externally verifiable by inspection")
        self.assertFalse(result["externally_validated"])
        np.testing.assert_allclose(self.fmt.read_xyz(out)["points"], POINTS, atol=1e-9)

    def test_write_csv_has_a_header_and_named_columns(self):
        out = self.path("cloud.csv")
        data = {"x": POINTS[:, 0], "y": POINTS[:, 1], "z": POINTS[:, 2],
                "intensity": np.array([1, 2, 3])}
        self.fmt.write_csv(data, out)
        rows = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(rows[0], ["x", "y", "z", "intensity"])
        self.assertEqual(rows[1], ["1.5", "-2.25", "10.125", "1"])
        self.assertEqual(len(rows), 4)
        read = self.fmt.read_csv(out)
        np.testing.assert_allclose(read["x"], POINTS[:, 0])
        np.testing.assert_array_equal(read["intensity"], [1, 2, 3])

    def test_plain_writers_reject_nonfinite_and_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.fmt.write_csv({"x": [0.0], "y": [np.nan], "z": [0.0]}, self.path("bad.csv"))
        with self.assertRaisesRegex(ValueError, "length"):
            self.fmt.write_csv({"x": [0.0, 1.0], "y": [0.0], "z": [0.0]}, self.path("len.csv"))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.fmt.write_xyz(np.array([[0.0, np.inf, 0.0]]), self.path("bad.xyz"))

    # -------------------------------------------------------------------- integration
    def test_a_metric_cloud_survives_las_quantisation(self):
        """1 mm LAS precision works when the offset sits at the data, not at a UTM origin."""
        rng = np.random.default_rng(3)
        points = rng.normal(0.0, 25.0, size=(120, 3))
        scale = (0.001, 0.001, 0.001)
        # The offset is the centroid, which is the only way 1 mm is representable: the
        # stored value is (coordinate - offset) / scale in int32, so a UTM northing of
        # 5,600,000 m at 1 mm needs 5.6e9 and overflows. See the refusal test above.
        offsets = tuple(float(value) for value in points.mean(axis=0))
        out = self.path("p.las")
        self.fmt.write_las(points, out, scale=scale, offsets=offsets)
        read = self.fmt.read_las(out)
        recovered = np.column_stack([read["points"][k] for k in "xyz"])
        error = np.linalg.norm(recovered - points, axis=1)
        self.assertLess(float(error.max()), 0.002)
        self.assertGreater(float(error.max()), 0.0)   # quantisation is real, not free

    def test_write_las_refuses_a_utm_northing_at_one_millimetre_scale(self):
        """The int32 guard stays: 1 mm over a UTM-scale northing is physically impossible.

        This test exists so the guard cannot be quietly relaxed to make a file writable:
        LAS stores coordinates as int32, so at 0.001 m a single offset spans only
        2**32 * 0.001 = 4294967.296 m, and 5,600,000 m northing does not fit in it.
        """
        out = self.path("impossible.las")
        with self.assertRaisesRegex(ValueError, "does not fit the int32") as caught:
            self.fmt.write_las(POINTS, out, scale=(0.001, 0.001, 0.001),
                               offsets=(500000.0, 5600000.0, 100.0))
        message = str(caught.exception)
        self.assertIn("4294967.296", message)          # the full span the scale allows
        self.assertIn("move the offset to the data centroid", message)
        self.assertFalse(out.exists())                 # refused before writing anything


if __name__ == "__main__":
    unittest.main()
