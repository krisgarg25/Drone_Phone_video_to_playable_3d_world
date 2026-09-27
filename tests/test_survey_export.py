"""Tests for the multi-format export step that turns a cloud into deliverables."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import survey_export as products


FRAME = {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
         "altitude_datum": "ellipsoidal",
         "origin": {"latitude_deg": 28.6, "longitude_deg": 77.2, "altitude_m": 227.0}}
ALIGNMENT = {"schema_version": 1, "status": "aligned", "scale": 1.0,
             "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
             "coordinate_frame": FRAME}
# A real CRS definition: the GeoTIFF writer refuses "EPSG:32643" and refuses None.
UTM_43N = ('PROJCS["WGS 84 / UTM zone 43N",GEOGCS["WGS 84",DATUM["WGS_1984",'
           'PRIMEM["Greenwich",0]]],PROJECTION["Transverse_Mercator"],'
           'UNIT["metre",1]]')


def cloud(n=500, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.uniform([0, 0, 90], [120, 80, 110], (n, 3)),
            rng.integers(0, 256, (n, 3), dtype=np.uint8))


class ExportProductsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def export(self, name="products", **kwargs):
        points, colors = cloud()
        return products.export_products(points, colors, ALIGNMENT, self.root / name, **kwargs)

    def test_a_point_cloud_fills_every_container_that_does_not_require_lying(self):
        manifest = self.export()
        written = {entry["format"] for entry in manifest["files"]}
        self.assertEqual(written, {"ply", "las", "xyz", "gltf", "fbx"})
        for entry in manifest["files"]:
            path = self.root / "products" / entry["path"]
            self.assertTrue(path.is_file() and path.stat().st_size > 0, entry["path"])

    def test_obj_and_geotiff_are_declared_not_delivered_with_the_real_reason(self):
        manifest = self.export("no_crs")
        reasons = {entry["format"]: entry["reason"] for entry in manifest["not_delivered"]}
        self.assertIn("obj", reasons)
        self.assertIn("geotiff", reasons)
        self.assertIn("mesh", reasons["obj"].lower())
        self.assertIn("crs", reasons["geotiff"].lower())
        self.assertTrue(all(entry["format"] != "geotiff" for entry in manifest["files"]))

    def test_geotiff_is_produced_once_a_real_crs_is_supplied(self):
        manifest = self.export("with_crs", crs_wkt=UTM_43N)
        entry = next(e for e in manifest["files"] if e["format"] == "geotiff")
        self.assertTrue((self.root / "with_crs" / entry["path"]).is_file())
        self.assertEqual(entry["geometry"], "dsm")
        self.assertGreater(entry["cells_filled"], 0)
        self.assertLessEqual(entry["cells_filled"], entry["cells_total"])
        self.assertEqual(manifest["crs_wkt"], UTM_43N)

    def test_dsm_rows_are_north_up_where_an_independent_reader_samples_them(self):
        """A pillar at the north edge must read tall at its own coordinate, not mirrored."""
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 100, 20000), rng.uniform(0, 100, 20000),
                               np.zeros(20000)])
        pts[(pts[:, 0] > 40) & (pts[:, 0] < 60) & (pts[:, 1] > 85), 2] = 30.0
        raster, transform, _, _ = products.dsm_grid(pts, 5.0)
        a, b, _, d, _, f = transform
        def at(x, y):
            return raster[int((y - d) / f), int((x - a) / b)]
        self.assertEqual(at(50, 95), 30.0)
        self.assertEqual(at(50, 5), 0.0)
        try:
            import rasterio
        except ImportError:
            return
        from scripts import survey_crs, survey_formats
        with tempfile.TemporaryDirectory() as tmp:
            survey_formats.write_geotiff(raster, Path(tmp) / "dsm.tif", transform=transform,
                                         crs_wkt=survey_crs.utm_wkt(43, "N"), nodata=-9999.0)
            with rasterio.open(Path(tmp) / "dsm.tif") as src:
                north, south = [v[0] for v in src.sample([(50, 95), (50, 5)])]
        self.assertEqual((north, south), (30.0, 0.0))

    def test_terrain_products_classify_ground_and_georeference_the_dtm(self):
        rng = np.random.default_rng(5)
        ground = np.column_stack([rng.uniform(0, 100, 15000), rng.uniform(0, 100, 15000),
                                  rng.normal(0, 0.05, 15000)])
        roof = np.column_stack([rng.uniform(60, 80, 3000), rng.uniform(60, 80, 3000),
                                np.full(3000, 12.0)])
        ground = ground[~((ground[:, 0] > 60) & (ground[:, 0] < 80)
                          & (ground[:, 1] > 60) & (ground[:, 1] < 80))]
        points = np.vstack([ground, roof])
        manifest = products.export_products(points, None, ALIGNMENT, self.root / "terrain",
                                            crs_wkt=UTM_43N, cell_size_m=1.0)
        names = {entry["path"] for entry in manifest["files"]}
        self.assertTrue({"dsm.tif", "dtm.tif", "ndsm.tif", "dtm_observed.tif"} <= names)
        self.assertGreater(manifest["terrain"]["ground_points"], 0)
        try:
            import laspy
            import rasterio
        except ImportError:
            return
        classes = np.asarray(laspy.read(self.root / "terrain/cloud.las").classification)
        self.assertTrue(np.all(classes[:len(ground)] == 2))
        self.assertTrue(np.all(classes[len(ground):] == 1))
        with rasterio.open(self.root / "terrain/ndsm.tif") as src:
            roof_height, open_ground = [v[0] for v in src.sample([(70, 70), (20, 20)])]
        with rasterio.open(self.root / "terrain/dtm_observed.tif") as src:
            under_roof, seen = [v[0] for v in src.sample([(70, 70), (20, 20)])]
        self.assertAlmostEqual(float(roof_height), 12.0, delta=0.3)
        self.assertLess(abs(float(open_ground)), 0.3)
        self.assertEqual((float(under_roof), float(seen)), (0.0, 1.0))

    def test_obj_is_produced_when_a_surface_mesh_exists(self):
        points, _ = cloud()
        triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        manifest = products.export_products(points, None, ALIGNMENT, self.root / "meshed",
                                            triangles=triangles)
        written = {entry["format"] for entry in manifest["files"]}
        self.assertIn("obj", written)
        self.assertEqual(manifest["claims_textured_mesh"], False)
        self.assertIn("no uv texture", " ".join(manifest["caveats"]).lower())

    def test_point_containers_never_claim_a_surface(self):
        manifest = self.export("points_only")
        self.assertFalse(manifest["claims_textured_mesh"])
        self.assertFalse(manifest["claims_surface_mesh"])
        self.assertIn("no surface mesh", " ".join(manifest["caveats"]).lower())
        for entry in manifest["files"]:
            if entry["format"] in ("gltf", "fbx"):
                self.assertEqual(entry["geometry"], "points")

    def test_las_offset_sits_at_the_data_centroid_not_the_crs_origin(self):
        points, _ = cloud()
        manifest = products.export_products(points, None, ALIGNMENT, self.root / "las")
        entry = next(e for e in manifest["files"] if e["format"] == "las")
        for axis, index in (("offset_x", 0), ("offset_y", 1), ("offset_z", 2)):
            self.assertAlmostEqual(entry[axis], float(np.median(points[:, index])), delta=20.0)
        self.assertLessEqual(entry["scale_mm"], 5.0)
        self.assertFalse(entry["crs_written"])

    def test_manifest_is_written_and_binds_the_provenance(self):
        manifest = self.export("bound", source_sha256="abc123")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source_sha256"], "abc123")
        self.assertEqual(manifest["coordinate_frame"], FRAME)
        self.assertEqual(manifest["point_count"], 500)
        on_disk = json.loads((self.root / "bound" / "export_manifest.json").read_text())
        self.assertEqual(on_disk["point_count"], 500)
        self.assertEqual({e["format"] for e in on_disk["files"]},
                         {e["format"] for e in manifest["files"]})

    def test_bad_inputs_refuse_before_any_directory_is_created(self):
        points, colors = cloud()
        cases = ((np.zeros((4, 2)), colors, "shape"),
                 (np.full((4, 3), np.nan), colors[:4], "nan"),
                 (points, np.zeros((len(points), 2)), "colors"),
                 (points[:3], colors, "mismatched colors"))
        for bad_points, bad_colors, label in cases:
            with self.subTest(case=label), self.assertRaises(ValueError):
                products.export_products(bad_points, bad_colors, ALIGNMENT, self.root / label)
            self.assertFalse((self.root / label).exists())
        with self.assertRaises(ValueError):
            products.export_products(points, colors, ALIGNMENT, self.root / "badtri",
                                     triangles=np.array([[0, 1, 999999]], dtype=np.int64))
        self.assertFalse((self.root / "badtri").exists())


if __name__ == "__main__":
    unittest.main()
