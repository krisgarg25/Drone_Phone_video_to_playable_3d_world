"""Tests for scripts/survey_texture.py (G1) on a synthetic scene with a known answer.

A 20 m ground plane carries an analytic pattern; cameras look down at it and their
frames are rendered by ray-casting that pattern, so every frame is exact. The bake must
reproduce the pattern in a view it was not given, respect lens distortion, refuse a view
an occluder blocks, and count - not fill in - a face no camera saw. The real-footage
result (rocks: gradient agreement 0.45 textured vs 0.37 per-vertex) is recorded in
docs/GAPS_AND_OPTIMIZATIONS.md, not asserted here.
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import survey_texture as tex  # noqa: E402

WIDTH, HEIGHT = 320, 240


def pattern(x, y):
    """RGB ground colour: two sinusoids and a slow ramp, smooth enough to sample."""
    r = 128 + 100 * np.sin(x * 1.3)
    g = 128 + 100 * np.cos(y * 1.1)
    b = 60 + 6 * (x + 10)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255)


def look_down(centre, yaw=0.0):
    """World->camera for a camera at ``centre`` looking straight down (-z), x right."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]], dtype=float)
    return R, -R @ np.asarray(centre, float)


def render_frame(image, camera):
    """Exact frame: undistort each pixel's ray, intersect the z=0 plane, colour it."""
    p = camera["params"]
    u, v = np.meshgrid(np.arange(WIDTH) + 0.0, np.arange(HEIGHT) + 0.0)
    f, cx, cy = p[0], p[1], p[2]
    xd, yd = (u - cx) / f, (v - cy) / f
    x, y = xd.copy(), yd.copy()
    if camera["model"] == "SIMPLE_RADIAL":
        for _ in range(20):  # fixed-point inversion of x_d = x (1 + k r^2)
            r2 = x * x + y * y
            x, y = xd / (1 + p[3] * r2), yd / (1 + p[3] * r2)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ image["R"]   # camera -> world dirs
    centre = tex.camera_centre(image)
    t = -centre[2] / rays[..., 2]
    ground = centre + rays * t[..., None]
    return pattern(ground[..., 0], ground[..., 1]).astype(np.uint8)


def plane(n=12, half=10.0):
    grid = np.linspace(-half, half, n + 1)
    xx, yy = np.meshgrid(grid, grid)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    faces = []
    for row in range(n):
        for col in range(n):
            a = row * (n + 1) + col
            faces += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return vertices, np.array(faces)


class Scene:
    def __init__(self, model="PINHOLE", k=0.0):
        params = [300.0, WIDTH / 2, HEIGHT / 2] + ([k] if model == "SIMPLE_RADIAL" else [])
        if model == "PINHOLE":
            params = [300.0, 300.0, WIDTH / 2, HEIGHT / 2]
        self.cameras = {1: {"model": model, "width": WIDTH, "height": HEIGHT,
                            "params": params}}
        if model == "PINHOLE":
            self.cameras[1]["params"] = [300.0, WIDTH / 2, HEIGHT / 2]
            self.cameras[1]["model"] = "SIMPLE_PINHOLE"
        self.images = []
        for i, (x, y, yaw) in enumerate([(-3, -3, 0.0), (3, -2, 0.3), (-2, 3, -0.2),
                                          (3, 3, 0.1), (0, 0, 0.0)]):
            R, t = look_down([x, y, 14.0], yaw)
            self.images.append({"name": f"f{i}.png", "camera_id": 1, "R": R, "t": t})
        self.frames = {image["name"]: render_frame(image, self.cameras[1])
                       for image in self.images}

    def load(self, name):
        return self.frames[name]


class BakeTests(unittest.TestCase):
    def test_a_held_out_view_is_reproduced(self):
        scene = Scene()
        vertices, faces = plane()
        held = scene.images[-1]                     # the overhead view, never baked from
        texture, uvs, report = tex.bake(vertices, faces, scene.cameras, scene.images[:-1],
                                        scene.load, cell_px=24)
        self.assertEqual(report["textured_faces"] + report["untextured_faces"], len(faces))
        rgb, covered = tex.render(vertices, faces, held, scene.cameras[1], uvs=uvs,
                                  texture=texture)
        truth = scene.frames[held["name"]]
        textured = covered & ~np.all(rgb == 128, axis=2)
        self.assertGreater(textured.mean(), 0.5)
        error = np.abs(rgb[textured].astype(int) - truth[textured].astype(int)).mean()
        self.assertLess(error, 12.0, "texture must reproduce a view it never saw")
        # And it is doing real work: a flat mean colour is far worse.
        flat = np.abs(truth[textured].astype(int) - truth[textured].mean(axis=0)).mean()
        self.assertGreater(flat, 3 * error)

    def test_lens_distortion_is_applied_not_ignored(self):
        scene = Scene("SIMPLE_RADIAL", k=-0.25)
        vertices, faces = plane()
        held = scene.images[-1]
        texture, uvs, _ = tex.bake(vertices, faces, scene.cameras, scene.images[:-1],
                                   scene.load, cell_px=24)
        rgb, covered = tex.render(vertices, faces, held, scene.cameras[1], uvs=uvs,
                                  texture=texture)
        mask = covered & ~np.all(rgb == 128, axis=2)
        good = np.abs(rgb[mask].astype(int) - scene.frames[held["name"]][mask]).mean()
        # The same frames read as if they were pinhole put every texel in the wrong place.
        pinhole = {1: {**scene.cameras[1], "model": "SIMPLE_PINHOLE",
                       "params": scene.cameras[1]["params"][:3]}}
        texture_bad, uvs_bad, _ = tex.bake(vertices, faces, pinhole, scene.images[:-1],
                                           scene.load, cell_px=24)
        rgb_bad, _ = tex.render(vertices, faces, held, scene.cameras[1], uvs=uvs_bad,
                                texture=texture_bad)
        bad = np.abs(rgb_bad[mask].astype(int) - scene.frames[held["name"]][mask]).mean()
        self.assertLess(good, 0.6 * bad)

    def test_an_occluded_view_is_not_chosen(self):
        scene = Scene()
        vertices, faces = plane()
        # A 4 m square roof 4 m above the plane's centre: every camera flies at 14 m within
        # 4.3 m of the centre, so each ray to the ground under the roof passes through it.
        roof = np.array([[-2.0, -2.0, 4.0], [2.0, -2.0, 4.0], [2.0, 2.0, 4.0], [-2.0, 2.0, 4.0]])
        all_vertices = np.vstack([vertices, roof])
        base = len(vertices)
        all_faces = np.vstack([faces, [[base, base + 1, base + 2], [base, base + 2, base + 3]]])
        scores, _ = tex.visibility(all_vertices, all_faces, scene.cameras, scene.images)
        centroids = all_vertices[all_faces[:len(faces)]].mean(axis=1)
        under = np.flatnonzero((np.abs(centroids[:, 0]) < 0.8) & (np.abs(centroids[:, 1]) < 0.8))
        self.assertTrue(len(under))
        self.assertFalse((scores[under] > 0).any(), "no camera may see through the roof")
        # Without the roof the same faces are seen, so the refusal is the roof's doing.
        open_scores, _ = tex.visibility(vertices, faces, scene.cameras, scene.images)
        self.assertTrue((open_scores[under] > 0).any(axis=1).all())

    def test_a_face_nobody_saw_is_counted_not_invented(self):
        scene = Scene()
        vertices, faces = plane()
        # The cameras do not reach the plane's outer ring; those faces are the baseline.
        _, _, baseline = tex.bake(vertices, faces, scene.cameras, scene.images, scene.load,
                                  cell_px=16)
        far = np.array([[500.0, 500.0, 0.0], [501.0, 500.0, 0.0], [500.0, 501.0, 0.0]])
        all_vertices = np.vstack([vertices, far])
        all_faces = np.vstack([faces, [[len(vertices), len(vertices) + 1, len(vertices) + 2]]])
        colors = np.zeros((len(all_vertices), 3))
        colors[-3:] = [10, 200, 30]
        texture, uvs, report = tex.bake(all_vertices, all_faces, scene.cameras, scene.images,
                                        scene.load, vertex_colors=colors, cell_px=16)
        self.assertEqual(report["untextured_faces"], baseline["untextured_faces"] + 1)
        self.assertEqual(report["untextured_fill"], "vertex colour")
        u, v = uvs[-1].mean(axis=0)
        pixel = texture[int(v * texture.shape[0]), int(u * texture.shape[1])]
        np.testing.assert_allclose(pixel, [10, 200, 30], atol=1)


class ModelAndWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_text_and_binary_models_read_the_same(self):
        text, binary = self.dir / "txt", self.dir / "bin"
        text.mkdir()
        binary.mkdir()
        (text / "cameras.txt").write_text("# c\n1 SIMPLE_RADIAL 640 360 549.8 320 180 -0.18\n")
        (text / "images.txt").write_text(
            "# i\n1 0.9 0.1 -0.4 0.07 5.6 -0.8 0.4 1 rocks/00000.jpg\n0 0 -1\n")
        (binary / "cameras.bin").write_bytes(struct.pack("<Q", 1) + struct.pack(
            "<iiQQ", 1, 2, 640, 360) + struct.pack("<4d", 549.8, 320, 180, -0.18))
        (binary / "images.bin").write_bytes(
            struct.pack("<Q", 1) + struct.pack("<i7di", 1, 0.9, 0.1, -0.4, 0.07, 5.6, -0.8, 0.4, 1)
            + b"rocks/00000.jpg\x00" + struct.pack("<Q", 1) + struct.pack("<ddq", 0, 0, -1))
        cams_t, imgs_t = tex.read_model(text)
        cams_b, imgs_b = tex.read_model(binary)
        self.assertEqual(cams_t, cams_b)
        self.assertEqual(imgs_t[0]["name"], imgs_b[0]["name"])
        np.testing.assert_allclose(imgs_t[0]["R"], imgs_b[0]["R"])
        with self.assertRaisesRegex(ValueError, "not supported"):
            tex._camera("FISHEYE", 1, 1, [1])

    def test_obj_and_gltf_carry_one_uv_per_corner(self):
        scene = Scene()
        vertices, faces = plane(4)
        texture, uvs, _ = tex.bake(vertices, faces, scene.cameras, scene.images, scene.load,
                                   cell_px=16)
        names = tex.write_obj(self.dir, vertices, faces, uvs, texture)
        names += tex.write_gltf(self.dir, vertices, faces, uvs)
        self.assertEqual(names, ["textured.obj", "textured.mtl", "textured.jpg",
                                 "textured.gltf", "textured.bin"])
        lines = (self.dir / "textured.obj").read_text().splitlines()
        self.assertEqual(sum(line.startswith("v ") for line in lines), len(vertices))
        self.assertEqual(sum(line.startswith("vt ") for line in lines), 3 * len(faces))
        face_lines = [line for line in lines if line.startswith("f ")]
        self.assertEqual(face_lines[0], "f 1/1 2/2 7/3")
        self.assertIn("map_Kd textured.jpg", (self.dir / "textured.mtl").read_text())
        document = json.loads((self.dir / "textured.gltf").read_text())
        buffer = (self.dir / "textured.bin").read_bytes()
        self.assertEqual(document["buffers"][0]["byteLength"], len(buffer))
        counts = [accessor["count"] for accessor in document["accessors"]]
        self.assertEqual(counts, [3 * len(faces)] * 3)
        views = document["bufferViews"]
        self.assertEqual(views[-1]["byteOffset"] + views[-1]["byteLength"], len(buffer))
        self.assertTrue(all(view["byteOffset"] % 4 == 0 for view in views))
        uv = np.frombuffer(buffer, np.float32, count=6 * len(faces),
                           offset=views[1]["byteOffset"]).reshape(-1, 2)
        self.assertTrue(((uv >= 0) & (uv <= 1)).all())
        self.assertEqual(document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"],
                         {"index": 0})

    def test_atlas_refuses_a_mesh_it_cannot_hold(self):
        with self.assertRaisesRegex(ValueError, "decimate"):
            tex.atlas_layout(10_000_000, 16)


if __name__ == "__main__":
    unittest.main()
