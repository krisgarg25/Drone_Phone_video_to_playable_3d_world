"""CPU-only tests for design-surface import and cut/fill (CON-02)."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

E0, N0 = 500000.0, 3400000.0          # a UTM-like design origin


def design_z(x, y):
    """Finished grade: a 2% ramp in x, flat pad at 10 m for x < 20."""
    return np.where(x < 20, 10.0, 10.0 + 0.02 * (x - 20))


def as_built(seed=0, n=60_000, extra=None):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 60, (n, 2))
    z = design_z(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.02, n)
    if extra is not None:
        z = z + extra(xy[:, 0], xy[:, 1])
    return np.column_stack([xy, z])


def tin_grid(step=5.0):
    xs = np.arange(0, 60.001, step)
    X, Y = np.meshgrid(xs, xs)
    verts = np.column_stack([X.ravel(), Y.ravel(), design_z(X.ravel(), Y.ravel())])
    n = len(xs)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces += [[i, i + 1, i + n + 1], [i, i + n + 1, i + n]]
    return verts, np.array(faces)


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.d = importlib.import_module("survey_design")

    def write_landxml(self, folder):
        verts, faces = tin_grid()
        pts = "".join(f'<P id="{i + 1}">{N0 + v[1]} {E0 + v[0]} {v[2]}</P>'
                      for i, v in enumerate(verts))
        fs = "".join(f"<F>{a + 1} {b + 1} {c + 1}</F>" for a, b, c in faces)
        text = ('<?xml version="1.0"?><LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2">'
                f'<Surfaces><Surface name="FG"><Definition surfType="TIN"><Pnts>{pts}</Pnts>'
                f'<Faces>{fs}</Faces></Definition></Surface></Surfaces></LandXML>')
        path = Path(folder) / "fg.xml"
        path.write_text(text)
        return path

    def write_dxf(self, folder):
        verts, faces = tin_grid(10.0)
        out = ["0", "SECTION", "2", "ENTITIES"]
        for tri in faces:
            p = [verts[i] for i in tri] + [verts[tri[2]]]
            out += ["0", "3DFACE", "8", "FG"]
            for k, v in enumerate(p):
                out += [f"1{k}", str(E0 + v[0]), f"2{k}", str(N0 + v[1]), f"3{k}", str(v[2])]
        out += ["0", "ENDSEC", "0", "EOF"]
        path = Path(folder) / "fg.dxf"
        path.write_text("\n".join(out))
        return path

    def test_on_grade_site_has_no_cut_or_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            design = self.d.read_design(self.write_landxml(tmp))
        out = self.d.cut_fill(as_built(), design, offset=(E0, N0, 0.0))
        r = out["report"]
        self.assertLess(r["cut_m3"] + r["fill_m3"], 2.0, r)
        self.assertGreater(r["design_observed_fraction"], 0.95)

    def test_known_mound_and_trench(self):
        mound = lambda x, y: np.where((np.abs(x - 40) < 5) & (np.abs(y - 30) < 5), 1.0, 0.0)
        trench = lambda x, y: np.where((np.abs(x - 10) < 2) & (np.abs(y - 30) < 10), -0.5, 0.0)
        pts = as_built(extra=lambda x, y: mound(x, y) + trench(x, y))
        with tempfile.TemporaryDirectory() as tmp:
            for design in (self.d.read_design(self.write_landxml(tmp)),
                           self.d.read_design(self.write_dxf(tmp))):
                r = self.d.cut_fill(pts, design, offset=(E0, N0, 0.0), sigma_reg_m=0.03)["report"]
                self.assertAlmostEqual(r["cut_m3"], 100.0, delta=6.0, msg=design["source"])
                self.assertAlmostEqual(r["fill_m3"], 40.0, delta=4.0, msg=design["source"])
                self.assertGreater(r["cut_sigma_m3"], 2.0)       # 100 m2 x 3 cm systematic

    def test_geotiff_design_and_landxml_axis_order(self):
        import survey_formats
        xs = np.arange(0.25, 60, 0.5)
        X, Y = np.meshgrid(xs, xs[::-1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fg.tif"
            survey_formats.write_geotiff(design_z(X, Y).astype(np.float32), path,
                                         transform=(E0, 0.5, 0.0, N0 + 60.0, 0.0, -0.5),
                                         crs_wkt='PROJCS["WGS 84 / UTM zone 43N",AUTHORITY["EPSG","32643"]]')
            raster = self.d.read_design(path)
            tin = self.d.read_design(self.write_landxml(tmp))
        self.assertAlmostEqual(float(tin["vertices"][:, 0].min()), E0)       # east, not north
        pts = as_built(extra=lambda x, y: np.where(x > 50, 0.4, 0.0))
        r = self.d.cut_fill(pts, raster, offset=(E0, N0, 0.0))["report"]
        self.assertAlmostEqual(r["cut_m3"], 0.4 * 10 * 60, delta=15.0)

    def test_no_overlap_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            design = self.d.read_design(self.write_landxml(tmp))
        with self.assertRaises(ValueError):
            self.d.cut_fill(as_built(), design)            # offset forgotten

    def test_zone_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            design = self.d.read_design(self.write_landxml(tmp))
        start = as_built(extra=lambda x, y: np.where(x > 30, 1.0, 0.0))
        half = as_built(1, extra=lambda x, y: np.where(x > 45, 1.0, 0.0))
        zones = {"east": [(30, 0), (60, 0), (60, 60), (30, 60)]}
        out = self.d.zone_progress(half, design, zones, baseline=start, offset=(E0, N0, 0.0))
        self.assertAlmostEqual(out[0]["progress_pct"], 50.0, delta=5.0)


class ModelDeviationTests(unittest.TestCase):
    def test_proud_panel_found(self):
        import plan_glb
        d = importlib.import_module("survey_design")
        v = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], np.float32)
        V, F = d.read_glb_mesh(plan_glb.glb_bytes([{"name": "wall", "positions": v, "indices": np.array([0, 1, 2, 0, 2, 3])}]))
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 4, 5000), rng.uniform(0, 3, 5000), rng.normal(0, 0.005, 5000)])
        pts[(pts[:, 0] > 2) & (pts[:, 1] > 1.5), 2] += 0.08
        far = np.column_stack([rng.uniform(0, 4, 200), rng.uniform(0, 3, 200), np.full(200, 0.45)])
        out = d.model_deviation(np.vstack([pts, far]), V, F, tolerance_m=0.03, max_distance_m=0.3)
        r = out["report"]
        self.assertAlmostEqual(r["proud_pct"], 25.0, delta=2.0)
        self.assertEqual(r["short_pct"], 0.0)
        self.assertEqual(r["scan_points_unrelated"], 200)
        with self.assertRaises(ValueError):
            d.model_deviation(pts + 100, V, F)


if __name__ == "__main__":
    unittest.main()
