import ast
import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


class CameraIntrinsicsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("camera_intrinsics"),
                             "The CPU calibration helper must exist")
        self.convert = importlib.import_module("camera_intrinsics").camera_matrix_and_distortion

    def test_supported_models_map_intrinsics_and_distortion(self):
        cases = [
            ("SIMPLE_PINHOLE", [800, 301, 219], 800, [0, 0, 0, 0]),
            ("PINHOLE", [800, 950, 301, 219], 950, [0, 0, 0, 0]),
            ("SIMPLE_RADIAL", [800, 301, 219, .12], 800, [.12, 0, 0, 0]),
            ("RADIAL", [800, 301, 219, .12, -.03], 800, [.12, -.03, 0, 0]),
            ("OPENCV", [800, 950, 301, 219, .12, -.03, .004, -.005],
             950, [.12, -.03, .004, -.005]),
            ("FULL_OPENCV", [800, 950, 301, 219, .12, -.03, .004, -.005,
                             .006, .007, -.008, .009],
             950, [.12, -.03, .004, -.005, .006, .007, -.008, .009]),
        ]
        for model, params, fy, expected_dist in cases:
            with self.subTest(model=model):
                K, dist = self.convert(model, params, (640, 480), (640, 480))
                self.assertIsInstance(K, np.ndarray)
                self.assertIsInstance(dist, np.ndarray)
                self.assertTrue(np.issubdtype(K.dtype, np.floating))
                self.assertTrue(np.issubdtype(dist.dtype, np.floating))
                np.testing.assert_array_equal(K, [[800, 0, 301], [0, fy, 219], [0, 0, 1]])
                np.testing.assert_array_equal(dist, expected_dist)

    def test_nonuniform_resize_scales_each_focal_and_principal_coordinate(self):
        K, dist = self.convert("OPENCV", [800, 950, 301, 219, .12, -.03, .004, -.005],
                               (640, 480), (320, 720))
        np.testing.assert_array_equal(K, [[400, 0, 150.5], [0, 1425, 328.5], [0, 0, 1]])
        np.testing.assert_array_equal(dist, [.12, -.03, .004, -.005])

    def test_resized_intrinsics_reject_float64_overflow_and_zero_focal_underflow(self):
        smallest = float(np.finfo(np.float64).smallest_subnormal)
        for params, size in (([1e308, 90, 38, 14], (160, 30)),
                             ([90, 1e308, 38, 14], (80, 60)),
                             ([90, 90, 1e308, 14], (160, 30)),
                             ([smallest, 90, 38, 14], (40, 30)),
                             ([90, smallest, 38, 14], (80, 15))):
            with self.subTest(params=params, size=size), self.assertRaises(ValueError):
                self.convert("PINHOLE", params, (80, 30), size)

    def test_representable_float64_focal_boundaries_remain_usable(self):
        for focal in (float(np.finfo(np.float64).max),
                      float(np.finfo(np.float64).smallest_subnormal)):
            with self.subTest(focal=focal):
                K, _ = self.convert("SIMPLE_PINHOLE", [focal, 0, 0], (80, 30), (80, 30))
                self.assertTrue(np.isfinite(K).all())
                self.assertEqual(K[0, 0], focal)
                self.assertEqual(K[1, 1], focal)

    def test_shared_focal_becomes_unequal_after_nonuniform_resize(self):
        K, _ = self.convert("SIMPLE_PINHOLE", [800, 301, 219], (640, 480), (320, 720))
        np.testing.assert_array_equal(K, [[400, 0, 150.5], [0, 1200, 328.5], [0, 0, 1]])

    def test_tangential_only_coefficients_are_preserved(self):
        _, dist = self.convert("OPENCV", [800, 950, 301, 219, 0, 0, .02, -.04],
                               (640, 480), (640, 480))
        np.testing.assert_array_equal(dist, [0, 0, .02, -.04])

    def test_input_arrays_are_not_mutated_or_aliased(self):
        params = np.array([800, 950, 301, 219, .12, -.03, .004, -.005])
        original = params.copy()
        size = np.array([640, 480])
        K, dist = self.convert("OPENCV", params, size, size)
        K[0, 0] = 1
        dist[0] = 1
        np.testing.assert_array_equal(params, original)
        np.testing.assert_array_equal(size, [640, 480])

    def test_unsupported_models_fail_clearly(self):
        for model in ("OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
                      "THIN_PRISM_FISHEYE", "FOV", "SIMPLE_OPENCV", "UNKNOWN", ""):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "[Uu]nsupported.*model"):
                    self.convert(model, [800, 950, 301, 219], (640, 480), (640, 480))

    def test_each_model_requires_exact_parameter_count(self):
        for model, count in (("SIMPLE_PINHOLE", 3), ("PINHOLE", 4), ("SIMPLE_RADIAL", 4),
                             ("RADIAL", 5), ("OPENCV", 8), ("FULL_OPENCV", 12)):
            for length in (count - 1, count + 1):
                with self.subTest(model=model, length=length):
                    with self.assertRaisesRegex(ValueError, "param"):
                        self.convert(model, [1] * length, (640, 480), (640, 480))

    def test_invalid_parameter_shapes_are_rejected(self):
        for params in (None, 800, [], [[800, 301, 219]], [[800], [301], [219]],
                       [[800, 301], [219]], [800, "bad", 219], [800, 301, 219j]):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, "param"):
                    self.convert("SIMPLE_PINHOLE", params, (640, 480), (640, 480))

    def test_nonfinite_parameters_including_distortion_are_rejected(self):
        for index in range(12):
            for value in (np.nan, np.inf, -np.inf):
                params = [800, 950, 301, 219, .12, -.03, .004, -.005, .006, .007, -.008, .009]
                params[index] = value
                with self.subTest(index=index, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.convert("FULL_OPENCV", params, (640, 480), (640, 480))

    def test_nonpositive_focals_are_rejected(self):
        for params in ([0, 950, 301, 219], [-1, 950, 301, 219],
                       [800, 0, 301, 219], [800, -1, 301, 219]):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, "focal"):
                    self.convert("PINHOLE", params, (640, 480), (640, 480))

    def test_invalid_sizes_are_rejected(self):
        bad_sizes = (None, 640, (), (640,), (640, 480, 3), [[640, 480]],
                     (0, 480), (640, -1), (640, np.nan), (np.inf, 480),
                     (640.5, 480), (640, "bad"), (640, 480j))
        for size in bad_sizes:
            for field in ("source_size", "image_size"):
                sizes = {"source_size": (640, 480), "image_size": (640, 480)}
                sizes[field] = size
                with self.subTest(field=field, size=size):
                    with self.assertRaisesRegex(ValueError, field):
                        self.convert("PINHOLE", [800, 950, 301, 219], **sizes)


def load_cpu_preparation():
    """Execute real preparation/parsing functions, omitting CUDA imports and sizing.

    sparse_depth_targets comes along because it is pure numpy: it is the whole
    geometry half of the D2 depth term, and it is the part a CPU lane can pin.
    DepthTerm itself allocates CUDA buffers and is only exercised on the GPU,
    so its per-pixel support weighting is measured in a real training run rather
    than claimed here (see the depth columns of work/<scene>/train_log.txt).
    """
    trainer_path = SCRIPTS / "train_splat.py"
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "load_colmap", "load_points3d", "prepare_dataset", "sparse_depth_targets"
        }:
            nodes.append(node)
        elif isinstance(node, ast.Import) and all(
            alias.name in {"cv2", "numpy", "hashlib", "json"} for alias in node.names
        ):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "pathlib", "PIL", "camera_intrinsics"
        }:
            nodes.append(node)
    parser_tree = ast.parse((SCRIPTS / "parse_colmap.py").read_text(encoding="utf-8"))
    nodes.extend(node for node in parser_tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "qvec2rot")
    namespace = {"fit_to_vram": lambda data: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(trainer_path), "exec"), namespace)
    return namespace


class PrepareDatasetCpuTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.txt = self.work / "colmap" / "sparse" / "txt"
        self.txt.mkdir(parents=True)
        # TRACK[] as (IMAGE_ID, POINT2D_IDX) pairs: two points with different
        # track lengths and different reported reprojection ERROR, because both
        # are part of what prepare_dataset hands the trainer - see
        # test_zero_distortion_preserves_source_pose_points_and_limit.
        (self.txt / "points3D.txt").write_text(
            "1 1 2 3 10 20 30 0.1 1 0 2 3\n"
            "2 4 5 6 40 50 60 1.5 1 4 2 5 3 6 4 7 5 8\n", encoding="utf-8")
        namespace = load_cpu_preparation()
        self.prepare = namespace["prepare_dataset"]
        self.project = namespace["sparse_depth_targets"]
        self.write_images()

    def write_camera(self, model="SIMPLE_RADIAL", params=(90, 38, 14, .2), size=(80, 30)):
        row = f"1 {model} {size[0]} {size[1]} " + " ".join(map(str, params)) + "\n"
        (self.txt / "cameras.txt").write_text(row, encoding="utf-8")

    def write_images(self, names=("frame.png",), size=(80, 30)):
        rows = []
        self.images = []
        y, x = np.indices((size[1], size[0]))
        for index, name in enumerate(names):
            rgb = np.stack(((x * 17 + y * 11 + index * 53) % 256,
                            (x * 7 + y * 23 + index * 31) % 256,
                            (x * 29 + y * 3 + index * 71) % 256), axis=-1).astype(np.uint8)
            path = self.work / "frames_train" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
            self.images.append(rgb)
            rows.append(f"{index + 1} 1 0 0 0 1 2 3 1 {name}\n\n")
        (self.txt / "images.txt").write_text("".join(rows), encoding="utf-8")

    def run_preparation(self, max_images=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.prepare(self.work, max_images=max_images)

    def assert_undistorted(self, frame, K, dist, source=None):
        if source is None:
            source = self.images[0]
        expected = cv2.undistort(source, np.array(K, dtype=np.float64),
                                 np.array(dist, dtype=np.float64))
        self.assertFalse(np.array_equal(expected, source), "Fixture must exercise distortion")
        np.testing.assert_array_equal(frame["img"], expected)
        np.testing.assert_allclose(frame["K"], K)
        self.assertEqual(frame["K"].dtype, np.float32)
        self.assertTrue(frame["path"].is_file())
        np.testing.assert_array_equal(np.array(Image.open(frame["path"])), expected)
        self.assertEqual((frame["width"], frame["height"]), (source.shape[1], source.shape[0]))

    def test_opencv_uses_unequal_focals_and_scaled_principal_point_before_undistort(self):
        self.write_camera("OPENCV", [90, 100, 76, 52, .2, -.03, .02, -.01], (160, 120))
        data, _ = self.run_preparation()
        self.assert_undistorted(data[0], [[45, 0, 38], [0, 25, 13], [0, 0, 1]],
                                [.2, -.03, .02, -.01])

    def test_radial_undistortion_uses_resized_pixel_intrinsics(self):
        self.write_camera("RADIAL", [90, 76, 56, .2, -.03], (160, 120))
        data, _ = self.run_preparation()
        self.assert_undistorted(data[0], [[45, 0, 38], [0, 22.5, 14], [0, 0, 1]], [.2, -.03, 0, 0])

    def test_tangential_only_distortion_is_applied(self):
        self.write_camera("OPENCV", [45, 50, 38, 14, 0, 0, .15, -.08])
        data, _ = self.run_preparation()
        self.assert_undistorted(data[0], [[45, 0, 38], [0, 50, 14], [0, 0, 1]], [0, 0, .15, -.08])

    def test_full_opencv_higher_order_only_distortion_is_applied(self):
        self.write_camera("FULL_OPENCV", [45, 50, 38, 14, 0, 0, 0, 0, .2, .05, -.01, .003])
        data, _ = self.run_preparation()
        self.assert_undistorted(data[0], [[45, 0, 38], [0, 50, 14], [0, 0, 1]],
                                [0, 0, 0, 0, .2, .05, -.01, .003])

    def test_changed_intrinsics_cannot_reuse_cached_pixels(self):
        self.write_camera()
        first, _ = self.run_preparation()
        self.write_camera(params=(45, 38, 14, .2))
        second, _ = self.run_preparation()
        self.assertNotEqual(first[0]["path"], second[0]["path"])
        self.assertFalse(np.array_equal(first[0]["img"], second[0]["img"]))
        self.assert_undistorted(second[0], [[45, 0, 38], [0, 45, 14], [0, 0, 1]], [.2, 0, 0, 0])

    def test_same_name_and_dimensions_with_new_pixels_invalidates_cache(self):
        self.write_camera()
        first, _ = self.run_preparation()
        replacement = 255 - self.images[0]
        Image.fromarray(replacement).save(self.work / "frames_train" / "frame.png")
        second, _ = self.run_preparation()
        self.assert_undistorted(second[0], [[90, 0, 38], [0, 90, 14], [0, 0, 1]],
                                [.2, 0, 0, 0], replacement)
        self.assertNotEqual(first[0]["path"], second[0]["path"])
        self.assertFalse(np.array_equal(first[0]["img"], second[0]["img"]))
        stamp = second[0]["path"].stat().st_mtime_ns
        third, _ = self.run_preparation()
        self.assertEqual(second[0]["path"], third[0]["path"])
        self.assertEqual(stamp, third[0]["path"].stat().st_mtime_ns)
        np.testing.assert_array_equal(second[0]["img"], third[0]["img"])

    def test_trainer_rejects_intrinsics_unrepresentable_in_float32(self):
        for params in ([1e40, 90, 38, 14], [90, 1e40, 38, 14],
                       [90, 90, 1e40, 14], [90, 90, 38, -1e40],
                       [1e-50, 90, 38, 14], [90, 1e-50, 38, 14]):
            self.write_camera("PINHOLE", params)
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.run_preparation()

    def test_trainer_preserves_representable_float32_focal_boundaries(self):
        for focal in (float(np.finfo(np.float32).max),
                      float(np.finfo(np.float32).smallest_subnormal)):
            self.write_camera("SIMPLE_PINHOLE", [focal, 38, 14])
            with self.subTest(focal=focal):
                data, _ = self.run_preparation()
                K = data[0]["K"]
                self.assertEqual(K.dtype, np.float32)
                self.assertTrue(np.isfinite(K).all())
                self.assertEqual(K[0, 0], focal)
                self.assertEqual(K[1, 1], focal)

    def test_cache_identity_includes_all_calibration_fields(self):
        variants = [
            ("SIMPLE_RADIAL", [90, 38, 14, .2], (80, 30)),
            ("SIMPLE_RADIAL", [90, 37, 14, .2], (80, 30)),
            ("SIMPLE_RADIAL", [90, 38, 13, .2], (80, 30)),
            ("SIMPLE_RADIAL", [90, 38, 14, .3], (80, 30)),
            ("SIMPLE_RADIAL", [90, 38, 14, .2], (160, 30)),
            ("SIMPLE_RADIAL", [90, 38, 14, .2], (80, 60)),
            ("RADIAL", [90, 38, 14, .2, 0], (80, 30)),
            ("OPENCV", [90, 91, 38, 14, .2, 0, 0, 0], (80, 30)),
            ("OPENCV", [90, 92, 38, 14, .2, 0, 0, 0], (80, 30)),
        ]
        paths = set()
        for model, params, size in variants:
            with self.subTest(model=model, params=params, size=size):
                self.write_camera(model, params, size)
                data, _ = self.run_preparation()
                self.assertNotIn(data[0]["path"], paths)
                paths.add(data[0]["path"])

    def test_cache_identity_includes_actual_image_resolution(self):
        self.write_camera()
        first, _ = self.run_preparation()
        self.write_images(size=(40, 60))
        second, _ = self.run_preparation()
        self.assertNotEqual(first[0]["path"], second[0]["path"])
        self.assert_undistorted(second[0], [[45, 0, 19], [0, 180, 28], [0, 0, 1]], [.2, 0, 0, 0])

    def test_nested_names_do_not_collide_with_flattened_names(self):
        self.write_camera()
        self.write_images(names=("flight/part/frame.png", "flight__part__frame.png"))
        data, _ = self.run_preparation()
        self.assertNotEqual(data[0]["path"], data[1]["path"])
        for frame, source in zip(data, self.images):
            self.assert_undistorted(frame, [[90, 0, 38], [0, 90, 14], [0, 0, 1]],
                                    [.2, 0, 0, 0], source)

    def test_unchanged_calibration_reuses_cache_without_rewriting(self):
        self.write_camera()
        first, _ = self.run_preparation()
        stamp = first[0]["path"].stat().st_mtime_ns
        second, _ = self.run_preparation()
        self.assertEqual(first[0]["path"], second[0]["path"])
        self.assertEqual(stamp, second[0]["path"].stat().st_mtime_ns)
        np.testing.assert_array_equal(first[0]["img"], second[0]["img"])

    def test_zero_distortion_preserves_source_pose_points_and_limit(self):
        self.write_camera("SIMPLE_PINHOLE", [90, 76, 56], (160, 120))
        self.write_images(names=("nested/frame.png", "second.png"))
        # prepare_dataset returns (training views, the observed geometry). The
        # cloud is FOUR arrays, not two: the sparse-depth term (challenge D2)
        # has to know how hard to believe each point, and COLMAP already
        # reports that per point as a reprojection ERROR in pixels and a track
        # length in VIEWS. They are part of the payload, not an internal detail,
        # so they come back in band -- this is the assertion that makes the next
        # array added to it a deliberate change rather than a caller's crash.
        data, points = self.run_preparation(max_images=1)
        self.assertEqual(len(data), 1)
        self.assertEqual(len(points), 4)
        xyz, rgb, err_px, n_views = points
        frame = data[0]
        self.assertEqual(frame["path"], self.work / "frames_train" / "nested/frame.png")
        np.testing.assert_array_equal(frame["img"], self.images[0])
        np.testing.assert_array_equal(frame["K"], [[45, 0, 38], [0, 22.5, 14], [0, 0, 1]])
        np.testing.assert_array_equal(frame["viewmat"], [[1, 0, 0, 1], [0, 1, 0, 2],
                                                         [0, 0, 1, 3], [0, 0, 0, 1]])
        np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(rgb, [[10, 20, 30], [40, 50, 60]])
        # The limit is on images; the whole cloud is still loaded.
        np.testing.assert_allclose(err_px, [0.1, 1.5])
        np.testing.assert_array_equal(n_views, [2, 5])
        self.assertEqual(err_px.dtype, np.float32)
        self.assertEqual(n_views.dtype, np.float32)
        self.assertGreater(frame["sharpness"], 0)
        self.assertEqual(list((self.work / "frames_undist").rglob("*.png")), [])

    def test_unparseable_error_is_not_silently_zero(self):
        """A point with no readable ERROR keeps NaN, so the depth term can see
        "unknown" instead of reading a missing measurement as a perfect point."""
        (self.txt / "points3D.txt").write_text(
            "1 1 2 3 10 20 30 notanumber 1 0\n"
            "2 4 5 6 40 50 60 0.4 1 4\n", encoding="utf-8")
        self.write_camera("SIMPLE_PINHOLE", [90, 76, 56], (160, 120))
        _, (_, _, err_px, n_views) = self.run_preparation()
        self.assertTrue(np.isnan(err_px[0]), "unknown error must stay unknown")
        np.testing.assert_allclose(err_px[1:], [0.4])
        np.testing.assert_array_equal(n_views, [1, 1])

    def test_unsupported_camera_is_rejected_in_preparation(self):
        self.write_camera("OPENCV_FISHEYE", [90, 100, 38, 14, .1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "[Uu]nsupported.*model"):
            self.run_preparation()


class SparseDepthTargetsTests(unittest.TestCase):
    """The geometry half of challenge D2, on CPU: projection, frustum gating and the
    per-point confidence that decides how hard a pixel is held.

    UNITS: depths here are COLMAP scene units (arbitrary until frame.json anchors
    them), errors are PIXELS, view counts are counts. Nothing in this function
    takes a metric constant, which is the point of the test.
    """

    W, H = 80, 60

    def setUp(self):
        self.project = load_cpu_preparation()["sparse_depth_targets"]

    def view(self, center=(0.0, 0.0, 0.0)):
        vm = np.eye(4, dtype=np.float32)
        # viewmat is world->cam (R | t) with t = -R C, so a camera at world
        # center c sees a point p at depth (p - c).z under identity rotation.
        vm[:3, 3] = -np.array(center, dtype=np.float32)
        K = np.array([[40.0, 0.0, 40.0], [0.0, 40.0, 30.0], [0.0, 0.0, 1.0]],
                     dtype=np.float32)
        return [dict(K=K, viewmat=vm, width=self.W, height=self.H)]

    def test_on_axis_point_lands_on_the_principal_pixel_at_its_own_depth(self):
        (idxs, zs, confs), rep = self.project(
            self.view(), np.array([[0.0, 0.0, 5.0]]), np.array([0.1]),
            np.array([4.0]), 0.01, 50.0, 1.5, 4.0)
        self.assertEqual(int(idxs[0][0]), 30 * self.W + 40)
        self.assertAlmostEqual(float(zs[0][0]), 5.0, places=5)
        self.assertEqual(rep["views_with_points"], 1)
        self.assertEqual(rep["n_points"], 1)
        # coverage is a FRACTION of the frame, never a length: one point over
        # 80x60 pixels.
        self.assertAlmostEqual(rep["coverage_median"], 1.0 / (self.W * self.H), places=9)

    def test_off_axis_projection_follows_the_intrinsics_and_the_pose_sign(self):
        # The camera is pulled back to z=-1, so the world point (0.5, 0.25, 5) sits
        # at depth 5+1=6 and projects to u=40*0.5/6+40, v=40*0.25/6+30 -- the same
        # intrinsics and the same (R|t) convention the rasterizer applies to the
        # gaussian centres, which is the whole reason the two depths are comparable.
        (idxs, zs, _), _ = self.project(
            self.view(center=(0.0, 0.0, -1.0)), np.array([[0.5, 0.25, 5.0]]),
            np.array([0.1]), np.array([4.0]), 0.01, 50.0, 1.5, 4.0)
        self.assertAlmostEqual(float(zs[0][0]), 6.0, places=5)
        u, v = 40.0 * 0.5 / 6.0 + 40.0, 40.0 * 0.25 / 6.0 + 30.0
        self.assertEqual(int(idxs[0][0]), int(v) * self.W + int(u))

    def test_points_behind_the_camera_or_past_far_are_dropped_not_supervised(self):
        """A view with no usable point returns None, so the term can report "nothing
        measured" instead of supervising a pixel with a negative or absurd depth.
        The near/far pair here is the rasterizer's own near_plane and far_plane, in
        the same scene units, so the two cannot disagree about what is visible."""
        (idxs, _, _), rep = self.project(
            self.view(), np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 500.0]]),
            np.array([0.1, 0.1]), np.array([4.0, 4.0]), 0.01, 50.0, 1.5, 4.0)
        self.assertIsNone(idxs[0])
        self.assertEqual(rep["views_with_points"], 0)
        self.assertEqual(rep["points_per_view_median"], 0)
        self.assertEqual(rep["points_per_view_min"], 0)
        # The gated points are still in the cloud count: the report separates
        # "how many points exist" from "how many constrain this view", and a
        # coverage of 0 here is a real 0 rather than a missing measurement.
        self.assertEqual(rep["n_points"], 2)
        self.assertEqual(rep["coverage_median"], 0.0)

    def test_only_the_visible_points_of_a_cloud_are_projected(self):
        (idxs, zs, _), rep = self.project(
            self.view(), np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 500.0], [0.0, 0.0, 5.0]]),
            np.array([0.1, 0.1, 0.1]), np.array([4.0, 4.0, 4.0]),
            0.01, 50.0, 1.5, 4.0)
        self.assertEqual(len(idxs[0]), 1)
        self.assertEqual(rep["views_with_points"], 1)
        self.assertAlmostEqual(float(zs[0][0]), 5.0, places=5)

    def test_views_with_nothing_in_frustum_leave_a_none_target_not_a_zero(self):
        (idxs, zs, confs), _ = self.project(
            self.view(), np.array([[0.0, 0.0, 5.0]]), np.array([0.1]),
            np.array([4.0]), 6.0, 50.0, 1.5, 4.0)   # near plane past the point
        self.assertIsNone(idxs[0])

    def test_three_views_is_weaker_evidence_than_three_thousand(self):
        """The disagreement scale the challenge asks for: the same pixel held by a
        thin cloud must not constrain as hard as one held by a dense cloud. Both
        numbers here are counts; neither is a length."""
        (_, _, confs), _ = self.project(
            self.view(),
            np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]),
            np.array([0.0, 0.0, 0.0]), np.array([3.0, 30.0, 3000.0]),
            0.01, 50.0, 1.5, 4.0)
        thin, mid, dense = (float(c) for c in confs[0])
        self.assertAlmostEqual(thin, 3.0 / 7.0, places=6)
        self.assertAlmostEqual(mid, 30.0 / 34.0, places=6)
        self.assertLess(thin, mid)
        self.assertLess(mid, dense)
        self.assertLess(thin, 0.5 * dense)

    def test_reprojection_error_and_a_missing_error_scale_the_confidence(self):
        (_, _, confs), _ = self.project(
            self.view(),
            np.array([[0.0, 0.0, 5.0]] * 3), np.array([0.0, 1.5, np.nan]),
            np.array([10.0, 10.0, 10.0]), 0.01, 50.0, 1.5, 4.0)
        good, worse, unknown = (float(c) for c in confs[0])
        self.assertAlmostEqual(good, 10.0 / 14.0, places=6)
        # err == err_ref_px halves it; an unparseable ERROR is treated as the worst
        # error the file could report, never as zero error.
        self.assertAlmostEqual(worse, 0.5 * good, places=6)
        self.assertLess(unknown, 1e-5)
        self.assertTrue((confs[0] >= 0).all() and (confs[0] <= 1).all())

    def test_report_medians_are_over_views_that_have_points_at_all(self):
        data = self.view() + self.view(center=(0.3, 0.0, 0.0))
        xyz = np.array([[0.0, 0.0, 5.0], [0.2, 0.1, 4.0]])
        (_, _, _), rep = self.project(
            data, xyz, np.array([0.1, 0.2]), np.array([4.0, 8.0]),
            0.01, 50.0, 1.5, 4.0)
        self.assertEqual(rep["n_views"], 2)
        self.assertGreaterEqual(rep["views_with_points"], 1)
        self.assertEqual(rep["points_per_view_median"], 2)
        self.assertGreater(rep["conf_median"], 0.0)
        self.assertLessEqual(rep["conf_median"], 1.0)
        self.assertEqual(rep["distinct_sizes"], 1)

    def test_a_mixed_size_view_set_is_reported_as_unsupervisable(self):
        """One observed-depth grid cannot index two frame sizes: a flattened pixel
        index from the wrong grid would supervise real geometry in the wrong place,
        which is worse than measuring nothing. The report carries the count so the
        trainer can disable the term and say why."""
        other = self.view()[0]
        other = dict(other, width=40, height=30)
        (_, _, _), rep = self.project(
            [self.view()[0], other], np.array([[0.0, 0.0, 5.0]]), np.array([0.1]),
            np.array([4.0]), 0.01, 50.0, 1.5, 4.0)
        self.assertEqual(rep["distinct_sizes"], 2)


if __name__ == "__main__":
    unittest.main()
