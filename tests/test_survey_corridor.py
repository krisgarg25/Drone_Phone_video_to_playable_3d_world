"""CPU-only tests for corridor tiles, profiles and post blind spots (BOR-02, BOR-04)."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def ridge_dsm(cell=1.0):
    """400 x 200 m flat ground with a 15 m ridge across x = 200..210 and a hole at x 300..320."""
    cols, rows = 400, 200
    z = np.zeros((rows, cols))
    z[:, 200:210] = 15.0
    z[:, 300:320] = np.nan
    transform = (0.0, cell, 0.0, float(rows) * cell, 0.0, -cell)
    return z, transform


class CorridorTests(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module("survey_corridor")

    def test_profile_marks_gaps(self):
        z, t = ridge_dsm()
        out = self.c.profile(z, t, [(0, 100), (399, 100)], step_m=1.0)
        s = out["summary"]
        self.assertAlmostEqual(s["max_z"], 15.0)
        self.assertAlmostEqual(s["length_m"], 399.0)
        self.assertLess(s["observed_fraction"], 0.96)
        self.assertTrue(np.isnan(out["z"][310]))

    def test_ridge_hides_the_far_side(self):
        z, t = ridge_dsm()
        line = [(10, 50), (390, 50)]
        post = dict(id="OP1", x=50, y=150, height_m=3.0, range_m=1000)
        out = self.c.blind_spots(z, t, line, [post], step_m=5.0)
        s = out["summary"]
        chain = out["chainage_m"]
        covered = out["covered"]
        self.assertTrue(covered[chain < 150].all())
        self.assertFalse(covered[(chain > 210) & (chain < 280)].any())
        stretch = s["blind_stretches"][0]
        self.assertAlmostEqual(stretch["from_m"], 200, delta=10)     # chainage 0 is x = 10
        # The hole blocks sight too, and is flagged as unobserved ground.
        self.assertTrue(any(b["unobserved_ground"] for b in s["blind_stretches"]))
        # A second post beyond the ridge fills the gap up to the hole.
        post2 = dict(id="OP2", x=260, y=150, height_m=3.0, range_m=1000,
                     bearing_deg=180.0, fov_deg=120.0)
        both = self.c.blind_spots(z, t, line, [post, post2], step_m=5.0)
        self.assertGreater(both["summary"]["covered_fraction"], s["covered_fraction"] + 0.15)

    def test_range_and_fov_limit_coverage(self):
        z = np.zeros((100, 400))
        t = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
        post = dict(x=200, y=90, height_m=3, range_m=100, bearing_deg=180.0, fov_deg=60.0)
        out = self.c.blind_spots(z, t, [(0, 50), (399, 50)], [post], step_m=2.0)
        seen = out["chainage_m"][out["covered"]]
        # 60 deg cone at 40 m standoff -> +-23 m of line; range 100 is not the limit.
        self.assertAlmostEqual(seen.min(), 200 - 23.1, delta=3)
        self.assertAlmostEqual(seen.max(), 200 + 23.1, delta=3)

    def test_tiles_are_written_and_readable(self):
        import survey_formats
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 2500, 30000), rng.uniform(0, 300, 30000),
                               rng.normal(100, 1, 30000)])
        wkt = 'PROJCS["WGS 84 / UTM zone 43N",AUTHORITY["EPSG","32643"]]'
        with tempfile.TemporaryDirectory() as tmp:
            m = self.c.tile_products(pts, tmp, tile_m=1000, cell_m=5.0, crs_wkt=wkt,
                                     offset=(500000.0, 3400000.0))
            tiles = [t for t in m["tiles"] if "files" in t]
            self.assertEqual(len(tiles), 3)
            data = json.loads((Path(tmp) / "tiles.json").read_text())
            self.assertEqual(len(data["tiles"]), 3)
            las = survey_formats.read_las(Path(tmp) / f"tile_{tiles[0]['tile']}" / "points.las")
            self.assertGreaterEqual(float(las["points"]["x"].min()), 500000.0)
            tif = survey_formats.read_geotiff(Path(tmp) / f"tile_{tiles[0]['tile']}" / "dsm.tif")
            self.assertEqual(tif["shape"], (200, 200))
            self.assertAlmostEqual(float(np.nanmedian(tif["raster"])), 100.0, delta=2.5)
            cog = tiles[0].get("cog")
            if cog is not None:                                   # GDAL present: a real COG
                self.assertEqual(cog["layout"], "COG")
                import rasterio
                with rasterio.open(Path(tmp) / f"tile_{tiles[0]['tile']}" / "dsm_cog.tif") as src:
                    self.assertEqual(src.crs.to_epsg(), 32643)
                    self.assertAlmostEqual(float(np.nanmedian(np.where(src.read(1) == -9999, np.nan, src.read(1)))), 100.0, delta=2.5)


if __name__ == "__main__":
    unittest.main()
