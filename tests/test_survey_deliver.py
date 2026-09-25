"""Tests for scripts/survey_deliver.py: the run's measured cloud becomes deliverables.

A synthetic alignment stands in for a reconstruction run. These tests prove the
wiring - which container is filled from which evidence, and what is refused when the
evidence is missing. They prove nothing about any real scene's accuracy.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from scripts import survey_deliver as deliver

FRAME = {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
         "altitude_datum": "ellipsoidal",
         "origin": {"latitude_deg": 28.6, "longitude_deg": 77.2, "altitude_m": 227.0}}
ALIGNMENT = {"schema_version": 1, "status": "aligned", "scale": 1.0,
             "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
             "coordinate_frame": FRAME}


def cloud(n=400, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.uniform([0, 0, 90], [120, 80, 110], (n, 3)),
            rng.integers(0, 256, (n, 3), dtype=np.uint8))


def write_mesh(path, points, colors, faces):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices = np.zeros(len(points), dtype=dtype)
    for column, name in zip(points.T, ("x", "y", "z")):
        vertices[name] = column
    for channel, name in zip(colors.T, ("red", "green", "blue")):
        vertices[name] = channel
    faces_array = np.zeros(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    faces_array["vertex_indices"] = faces.astype(np.int32)
    PlyData([PlyElement.describe(vertices, "vertex"),
             PlyElement.describe(faces_array, "face")], text=False).write(str(path))


class MeshReadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_vertex_only_ply_is_not_an_error_it_is_the_absence_of_a_mesh(self):
        points, colors = cloud()
        path = self.root / "cloud.ply"
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                 ("blue", "u1")]
        record = np.zeros(len(points), dtype=dtype)
        for column, name in zip(points.T, ("x", "y", "z")):
            record[name] = column
        for channel, name in zip(colors.T, ("red", "green", "blue")):
            record[name] = channel
        PlyData([PlyElement.describe(record, "vertex")], text=False).write(str(path))
        self.assertEqual(deliver.read_mesh_ply(path), (None, None, None))
        self.assertEqual(deliver.read_mesh_ply(self.root / "missing.ply"), (None, None, None))

    def test_faces_and_vertex_colors_come_back_intact(self):
        points, colors = cloud(64, seed=1)
        faces = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]], dtype=np.int64)
        path = self.root / "mesh.ply"
        write_mesh(path, points, colors, faces)
        read_points, read_faces, read_colors = deliver.read_mesh_ply(path)
        np.testing.assert_allclose(read_points, points, atol=1e-4)
        np.testing.assert_array_equal(read_faces, faces)
        np.testing.assert_array_equal(read_colors, colors)

    def test_a_polygon_mesh_is_refused_not_silently_triangulated(self):
        points, colors = cloud(8, seed=2)
        path = self.root / "quad.ply"
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                 ("blue", "u1")]
        record = np.zeros(len(points), dtype=dtype)
        for column, name in zip(points.T, ("x", "y", "z")):
            record[name] = column
        for channel, name in zip(colors.T, ("red", "green", "blue")):
            record[name] = channel
        quads = np.zeros(1, dtype=[("vertex_indices", "i4", (4,))])
        quads["vertex_indices"] = np.array([[0, 1, 2, 3]], dtype=np.int32)
        PlyData([PlyElement.describe(record, "vertex"),
                 PlyElement.describe(quads, "face")], text=False).write(str(path))
        with self.assertRaisesRegex(ValueError, "only triangles"):
            deliver.read_mesh_ply(path)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.points, self.colors = cloud()

    def meshed(self, **kwargs):
        faces = np.array([[i, i + 1, i + 2] for i in range(len(self.points) - 2)],
                         dtype=np.int64)
        mesh = self.root / "mesh.ply"
        write_mesh(mesh, self.points, self.colors, faces)
        return deliver.deliver(self.points, self.colors, ALIGNMENT, self.root / "products",
                               mesh_path=mesh, **kwargs)

    def test_a_meshed_run_delivers_every_named_format_in_a_real_crs(self):
        result = self.meshed()
        geo = result["georeferenced"]
        self.assertIsNotNone(geo, result["refusals"])
        self.assertEqual(geo["formats"]["delivered"], geo["formats"]["total"],
                         [row for row in geo["formats"]["rows"] if row["status"] != "delivered"])
        self.assertEqual(geo["crs"]["epsg"], 32643)
        self.assertEqual(geo["crs"]["zone"], 43)
        self.assertTrue((self.root / "products" / "georeferenced" / "surface.obj").is_file())
        self.assertTrue((self.root / "products" / "georeferenced" / "dsm.tif").is_file())
        self.assertTrue((self.root / "products" / "georeferenced" / "cloud.las").is_file())
        self.assertEqual(result["meshed"], True)
        self.assertGreater(result["triangle_count"], 0)

    def test_the_local_set_keeps_enu_metres_and_still_refuses_a_geotiff(self):
        result = self.meshed()
        local = result["local_enu"]
        statuses = {row["format"]: row["status"] for row in local["formats"]["rows"]}
        self.assertEqual(statuses["obj"], "delivered")
        self.assertEqual(statuses["geotiff"], "not_delivered")
        self.assertIn("CRS", " ".join(r["reason"] for r in local["formats"]["rows"]
                                      if r["status"] != "delivered"))
        self.assertEqual(local["manifest"]["coordinate_frame"]["type"], "ENU")

    def test_without_a_mesh_obj_stays_undelivered_in_both_sets(self):
        result = deliver.deliver(self.points, self.colors, ALIGNMENT, self.root / "flat")
        self.assertEqual(result["meshed"], False)
        for group in (result["local_enu"], result["georeferenced"]):
            statuses = {row["format"]: row["status"] for row in group["formats"]["rows"]}
            self.assertEqual(statuses["obj"], "not_delivered")
        self.assertEqual(result["georeferenced"]["formats"]["delivered"], 5)

    def test_a_scene_without_a_usable_origin_still_gets_its_local_products(self):
        broken = dict(ALIGNMENT, coordinate_frame={"type": "ENU", "units": "m",
                                                   "geodetic_crs": "EPSG:4979",
                                                   "altitude_datum": "ellipsoidal"})
        result = deliver.deliver(self.points, self.colors, broken, self.root / "noorigin")
        self.assertIsNone(result["georeferenced"])
        self.assertEqual(result["local_enu"]["formats"]["delivered"],
                         result["local_enu"]["formats"]["total"] - 2)
        self.assertTrue(result["refusals"])
        self.assertIn("origin", result["refusals"][0]["reason"])

    def test_wgs84_positions_are_decimated_and_say_so(self):
        result = self.meshed(max_wgs84_rows=50)
        text = (self.root / "products" / "georeferenced" /
                "positions_wgs84.csv").read_text(encoding="utf-8").splitlines()
        self.assertIn("rows_total=400", text[0])
        self.assertIn("rows_written=50", text[0])
        self.assertEqual(len(text), 52)  # comment, header, 50 rows
        self.assertEqual(result["georeferenced"]["manifest"]["point_count"], 400)

    def test_a_mesher_own_vertex_set_is_reported_separately_from_the_cloud(self):
        """Poisson rebuilds the surface: its counts must not be read as point counts."""
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        mesh_points = self.points[:8] * 0.5
        mesh_colors = self.colors[:8]
        mesh = self.root / "poisson.ply"
        write_mesh(mesh, mesh_points, mesh_colors, faces)
        result = deliver.deliver(self.points, self.colors, ALIGNMENT, self.root / "mixed",
                                 mesh_path=mesh)
        self.assertEqual(result["cloud_point_count"], 400)
        self.assertEqual(result["point_count"], 8)
        self.assertEqual(result["mesh"]["mesh_face_count"], 2)
        self.assertIn("isosurface", result["mesh"]["basis"])
        self.assertEqual(result["georeferenced"]["formats"]["delivered"], 6)

    def test_the_manifest_on_disk_matches_the_returned_ledger(self):
        import json
        result = self.meshed()
        written = json.loads((self.root / "products" / "delivery_manifest.json")
                             .read_text(encoding="utf-8"))
        # The returned ledger carries the format tuple; JSON has no tuples.
        expected = json.loads(json.dumps(result))
        self.assertEqual(written["georeferenced"]["formats"], expected["georeferenced"]["formats"])


if __name__ == "__main__":
    unittest.main()
