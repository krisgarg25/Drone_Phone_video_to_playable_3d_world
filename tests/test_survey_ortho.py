"""CPU-only tests for the true orthomosaic (M1) on an exactly rendered synthetic scene.

A ground plane with an analytic pattern and a 3 m box building with a distinct roof;
every frame is ray-cast, so each ortho cell has a known right answer. The tests ask
for the ground where the ground is, the roof where the roof is, nothing painted onto
ground a view could not see, and a GeoTIFF that GDAL places correctly.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import survey_ortho as ortho  # noqa: E402
import survey_texture as tex  # noqa: E402

WIDTH, HEIGHT = 320, 240
BOX = (1.0, 3.0, -1.0, 1.0, 3.0)   # xmin, xmax, ymin, ymax, height
ROOF = np.array([230, 40, 200])
WALL = np.array([90, 90, 90])


def pattern(x, y):
    r = 128 + 100 * np.sin(x * 1.3)
    g = 128 + 100 * np.cos(y * 1.1)
    b = 60 + 6 * (x + 10)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255)


def look(centre, target):
    """World->camera looking from centre at target, image x roughly east."""
    forward = np.asarray(target, float) - np.asarray(centre, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1.0, 0]) if abs(forward[1]) < 0.9 else np.cross(forward, [1.0, 0, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    if R @ np.array([1.0, 0, 0]) @ np.array([1.0, 0, 0]) < 0:
        R = np.vstack([-right, -down, forward])
    return R, -R @ np.asarray(centre, float)


def render(image, camera):
    f, cx, cy = camera["params"]
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    rays = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], axis=-1) @ image["R"]
    c = tex.camera_centre(image)
    best_t = np.full(u.shape, np.inf)
    colour = np.zeros(u.shape + (3,))
    x0, x1, y0, y1, h = BOX

    def hit(t, cond, value):
        nonlocal best_t
        ok = cond & (t > 0) & (t < best_t)
        best_t = np.where(ok, t, best_t)
        colour[ok] = value if np.ndim(value) == 1 else value[ok]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = -c[2] / rays[..., 2]
        p = c + rays * t[..., None]
        hit(t, np.isfinite(t), pattern(p[..., 0], p[..., 1]))
        t = (h - c[2]) / rays[..., 2]
        p = c + rays * t[..., None]
        hit(t, (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1), ROOF)
        for axis, value, lo, hi in ((0, x0, y0, y1), (0, x1, y0, y1), (1, y0, x0, x1), (1, y1, x0, x1)):
            t = (value - c[axis]) / rays[..., axis]
            p = c + rays * t[..., None]
            other = p[..., 1 - axis]
            hit(t, (other >= lo) & (other <= hi) & (p[..., 2] >= 0) & (p[..., 2] <= h), WALL)
    return np.clip(colour, 0, 255).astype(np.uint8)


def dsm(cell=0.25, half=10.0):
    n = int(2 * half / cell)
    xs = -half + (np.arange(n) + 0.5) * cell
    ys = half - (np.arange(n) + 0.5) * cell
    xx, yy = np.meshgrid(xs, ys)
    x0, x1, y0, y1, h = BOX
    z = np.where((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1), h, 0.0)
    return z, (-half, cell, 0.0, half, 0.0, -cell)


class Scene:
    def __init__(self, views):
        self.cameras = {1: {"model": "SIMPLE_PINHOLE", "width": WIDTH, "height": HEIGHT,
                            "params": [300.0, WIDTH / 2, HEIGHT / 2]}}
        self.images = []
        for i, (centre, target) in enumerate(views):
            R, t = look(centre, target)
            self.images.append({"name": f"v{i}.png", "camera_id": 1, "R": R, "t": t})
        self.frames = {im["name"]: render(im, self.cameras[1]) for im in self.images}

    def load(self, name):
        return self.frames[name]


NADIR = [((x, y, 14.0), (x, y, 0.0)) for x, y in ((-4, -4), (4, -4), (-4, 4), (4, 4), (0, 0), (2, 0))]
IDENTITY = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}


def cell_value(rgb, alpha, transform, x, y):
    a, b, _, d, _, f = transform
    r, c = int((y - d) / f), int((x - a) / b)
    return rgb[r, c].astype(float), alpha[r, c]


def cell_centre(transform, x, y):
    """The pattern must be compared at the cell centre, not the probe point: it changes
    up to 130 levels per metre, so a 0.125 m offset alone costs 16."""
    a, b, _, d, _, f = transform
    r, c = int((y - d) / f), int((x - a) / b)
    return a + (c + 0.5) * b, d + (r + 0.5) * f


class OrthoTests(unittest.TestCase):
    def run_ortho(self, scene, **kwargs):
        z, transform = dsm()
        return ortho.orthomosaic(z, transform, ortho.enu_to_local(IDENTITY), scene.cameras,
                                 scene.images, scene.load, **kwargs)

    def test_ground_and_roof_come_back_where_they_are(self):
        rgb, alpha, transform, report = self.run_ortho(Scene(NADIR))
        errors = []
        for x, y in ((-5, -5), (-3, 4), (5, 3), (-6, 1), (0, -5)):
            value, a = cell_value(rgb, alpha, transform, x, y)
            self.assertEqual(a, 255)
            errors.append(np.abs(value - pattern(*cell_centre(transform, x, y))).mean())
        self.assertLess(np.mean(errors), 6.0, errors)
        for x, y in ((1.6, 0.0), (2.5, -0.6), (2.2, 0.7)):
            value, a = cell_value(rgb, alpha, transform, x, y)
            self.assertEqual(a, 255)
            np.testing.assert_allclose(value, ROOF, atol=4)
        self.assertGreater(report["observed_fraction"], 0.5)
        self.assertGreater(report["views_used"], 1)

    def test_ground_behind_the_building_is_not_painted_from_a_blocked_view(self):
        # One oblique camera west of the box looking east: the ground just east of the
        # box is in its frame but hidden behind the building.
        scene = Scene([((-8.0, 0.0, 6.0), (6.0, 0.0, 0.0))])
        rgb, alpha, transform, _ = self.run_ortho(scene)
        _, hidden = cell_value(rgb, alpha, transform, 3.6, 0.0)
        self.assertEqual(hidden, 0, "ground occluded by the building must stay transparent")
        value, seen = cell_value(rgb, alpha, transform, -2.0, 0.0)
        self.assertEqual(seen, 255)
        self.assertLess(np.abs(value - pattern(*cell_centre(transform, -2.0, 0.0))).mean(), 10)

    def test_cells_no_camera_saw_are_transparent(self):
        scene = Scene([((-4, -4, 10.0), (-4, -4, 0.0))])
        rgb, alpha, transform, report = self.run_ortho(scene)
        self.assertEqual(cell_value(rgb, alpha, transform, 8.0, 8.0)[1], 0)
        self.assertLess(report["observed_fraction"], 0.5)

    def test_geotiff_places_the_roof_where_gdal_reads_it(self):
        try:
            import rasterio
        except ImportError:
            self.skipTest("rasterio not installed")
        import survey_crs
        import survey_formats
        rgb, alpha, transform, _ = self.run_ortho(Scene(NADIR))
        # Shift the local grid to plausible UTM numbers; the pixels do not move.
        a, b, c, d, e, f = transform
        utm = (a + 700000.0, b, c, d + 3160000.0, e, f)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ortho.tif"
            written = survey_formats.write_rgba_geotiff(rgb, alpha, out, transform=utm,
                                                        crs_wkt=survey_crs.utm_wkt(43, "N"))
            with rasterio.open(out) as src:
                self.assertEqual(src.count, 4)
                roof = next(src.sample([(700002.0, 3160000.0)]))
                ground = next(src.sample([(699995.0, 3159995.0)]))
        self.assertTrue(written["externally_validated"])
        np.testing.assert_allclose(roof[:3], ROOF, atol=4)
        self.assertEqual(roof[3], 255)
        self.assertLess(np.abs(ground[:3] - pattern(*cell_centre(transform, -5.0, -5.0))).mean(), 8)


if __name__ == "__main__":
    unittest.main()


class DeliverOrthoTests(unittest.TestCase):
    """deliver() -> georeferenced/ortho.tif from a real COLMAP text model and frames."""

    def test_deliver_writes_an_ortho_gdal_places_correctly(self):
        try:
            import rasterio
        except ImportError:
            self.skipTest("rasterio not installed")
        import cv2
        import survey_crs
        import survey_deliver
        from scipy.spatial.transform import Rotation
        scene = Scene(NADIR)
        frame = {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                 "altitude_datum": "ellipsoidal",
                 "origin": {"latitude_deg": 28.6, "longitude_deg": 77.2, "altitude_m": 210.0}}
        alignment = {"schema_version": 1, "status": "aligned", "scale": 1.0,
                     "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0],
                     "coordinate_frame": frame}
        rng = np.random.default_rng(0)
        ground = np.column_stack([rng.uniform(-10, 10, 30000), rng.uniform(-10, 10, 30000),
                                  np.zeros(30000)])
        x0, x1, y0, y1, h = BOX
        ground = ground[~((ground[:, 0] > x0) & (ground[:, 0] < x1)
                          & (ground[:, 1] > y0) & (ground[:, 1] < y1))]
        roof = np.column_stack([rng.uniform(x0, x1, 3000), rng.uniform(y0, y1, 3000),
                                np.full(3000, h)])
        points = np.vstack([ground, roof])
        colors = np.full(points.shape, 128, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model, frames = tmp / "model", tmp / "images"
            model.mkdir()
            frames.mkdir()
            (model / "cameras.txt").write_text(f"1 SIMPLE_PINHOLE {WIDTH} {HEIGHT} 300 {WIDTH / 2} {HEIGHT / 2}\n")
            lines = []
            for i, image in enumerate(scene.images, 1):
                qx, qy, qz, qw = Rotation.from_matrix(image["R"]).as_quat()
                tx, ty, tz = image["t"]
                lines += [f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image['name']}", ""]
                cv2.imwrite(str(frames / image["name"]), scene.frames[image["name"]][:, :, ::-1])
            (model / "images.txt").write_text("\n".join(lines) + "\n")
            result = survey_deliver.deliver(points, colors, alignment, tmp / "products",
                                            cell_size_m=0.25,
                                            texture_source={"model_dir": model, "image_dir": frames})
            self.assertFalse([r for r in result["refusals"] if r["set"] == "orthomosaic"],
                             result["refusals"])
            geo = result["georeferenced"]["manifest"]
            self.assertEqual(geo["files"][0]["path"], "ortho.tif")
            roof_utm, _, _ = survey_crs.enu_to_crs(np.array([[2.0, 0.0, 3.0]]), alignment)
            ground_utm, _, _ = survey_crs.enu_to_crs(np.array([[-5.0, -5.0, 0.0]]), alignment)
            with rasterio.open(tmp / "products/georeferenced/ortho.tif") as src:
                roof_px = next(src.sample([tuple(roof_utm[0, :2])]))
                ground_px = next(src.sample([tuple(ground_utm[0, :2])]))
        np.testing.assert_allclose(roof_px[:3], ROOF, atol=6)
        self.assertEqual(ground_px[3], 255)
        self.assertLess(np.abs(ground_px[:3].astype(float) - pattern(-5.0, -5.0)).mean(), 20)
