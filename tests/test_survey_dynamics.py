"""CPU-only tests for motion evidence against a static reconstruction (challenge iv).

Every scene here is synthetic and deterministic: three textured scene layers at 5, 6
and 8 metres, seen through a known pinhole camera, so "the camera moved" and "the
object moved" are truths we control instead of things we hope a network guessed. The
two background depths give real parallax (a translating camera moves the near layer
faster than the far one), which is exactly the case a moving drone has to survive.

Camera convention (as elsewhere in the survey scripts): x_cam = x_world - C, the
camera looks along +Z, a point in front of the camera has z_cam > 0, and it projects
to u = fx * (X - Cx) / Z + cx. A scene layer at depth Z is therefore a canvas that the
camera window slides over at fx * C / Z pixels per metre.

No torch, no GPU, no downloaded model, no network: motion evidence only, and the tests
are written so that deleting real static geometry fails the suite.
"""
import sys
import unittest

import numpy as np

_TORCH_BEFORE = "torch" in sys.modules

import cv2  # noqa: E402

from scripts import survey_dynamics as dyn  # noqa: E402

HEIGHT, WIDTH = 120, 160
CANVAS_HEIGHT, CANVAS_WIDTH = 280, 380
ORIGIN_X, ORIGIN_Y = CANVAS_WIDTH / 2.0, CANVAS_HEIGHT / 2.0
FX = 300.0
K = np.array([[FX, 0.0, WIDTH / 2.0], [0.0, FX, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
TOLERANCE = 2.0

Z_FAR, Z_NEAR, Z_OBJECT = 8.0, 5.0, 6.0
CAMERA_STEP = 0.10        # metres per frame along +X: 3.75..6 px of parallax per frame
OBJECT_HOME = (-0.45, 0.05)
OBJECT_STEP = 0.48        # along +X: 24 px/frame, i.e. along the epipolar lines
OBJECT_SIDE = 0.16        # along +Y: 8 px/frame, i.e. across the epipolar lines
OBJECT_HALF = (14, 10)    # the obstacle's half size in pixels at its own depth
# The scene really does live between 4 m and 10 m, so this prior is truthful here.
DEPTH_RANGE = (4.0, 10.0)


def _texture(seed, canvas, sigmas=(5.0, 2.0, 0.8), amps=(1.0, 0.6, 0.2), scale=0.72):
    """A multiscale, non-repeating texture, i.e. something with natural-image statistics.

    Plain blurred noise is a trap in a test like this: every patch looks like every other
    patch, so Lucas-Kanade happily lands on the neighbouring ripple and the "motion" it
    reports is tracker confusion rather than scene motion. Summing a few octaves gives
    corners that are plentiful and distinctive, which is what the geometry test needs.
    """
    rng = np.random.default_rng(seed)
    height, width = canvas
    acc = np.zeros((height, width), np.float32)
    for sigma, amp in zip(sigmas, amps):
        raw = rng.integers(0, 256, (height, width), dtype=np.uint8).astype(np.float32)
        acc += amp * cv2.GaussianBlur(raw, (0, 0), sigma)
    acc = (acc - float(acc.min())) * 255.0 / (float(acc.max()) - float(acc.min()) + 1e-6)
    return np.clip(acc * scale, 0, 255).astype(np.uint8)


def _blob_canvas(seed, count):
    rng = np.random.default_rng(seed)
    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), np.uint8)
    for _ in range(count):
        center = (int(rng.integers(40, CANVAS_WIDTH - 40)), int(rng.integers(40, CANVAS_HEIGHT - 40)))
        cv2.circle(mask, center, int(rng.integers(22, 40)), 255, -1)
    return mask


def _object_canvas():
    """The obstacle's own skin: its own texture, kept bright so it cannot be mistaken for
    the dim scene, and exactly the footprint ``_object_box`` claims for it."""
    patch = _texture(404, (2 * OBJECT_HALF[1] + 1, 2 * OBJECT_HALF[0] + 1),
                     sigmas=(1.6, 0.7), amps=(1.0, 0.7), scale=1.0)
    return np.clip(165.0 + patch.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)


_FAR = _texture(101, (CANVAS_HEIGHT, CANVAS_WIDTH))
_NEAR = _texture(202, (CANVAS_HEIGHT, CANVAS_WIDTH))
_NEAR_ALPHA = _blob_canvas(303, 9)
_OBJECT = _object_canvas()


def _object_centre(index, center, cross=False):
    """Pixel position of the obstacle's centre: its world position, seen from the camera."""
    x = OBJECT_HOME[0] + OBJECT_STEP * index
    y = OBJECT_HOME[1] + (OBJECT_SIDE * index if cross else 0.0)
    u = FX * (x - center[0]) / Z_OBJECT + WIDTH / 2.0
    v = FX * (y - center[1]) / Z_OBJECT + HEIGHT / 2.0
    return int(round(u)), int(round(v))


def _object_box(index, center, *, cross=False, pad=0):
    """Exact pixel footprint of the obstacle in this frame, optionally grown by ``pad``."""
    u, v = _object_centre(index, center, cross)
    return (u - OBJECT_HALF[0] - pad, u + OBJECT_HALF[0] + pad,
            v - OBJECT_HALF[1] - pad, v + OBJECT_HALF[1] + pad)


def _frame(index, center, *, moving_object=True, cross=False):
    far = _view(_FAR, Z_FAR, center)
    near = _view(_NEAR, Z_NEAR, center)
    alpha = _view(_NEAR_ALPHA, Z_NEAR, center)
    image = np.where(alpha > 127, near, far)
    if moving_object:
        # The obstacle is a rigid patch of skin carried through the image, so it stays
        # trackable: pasting a fixed window of a static canvas instead would show new
        # texture every frame and no tracker could follow it.
        u, v = _object_centre(index, center, cross)
        top, left = v - OBJECT_HALF[1], u - OBJECT_HALF[0]
        height, width = _OBJECT.shape
        source_top, source_left = max(0, -top), max(0, -left)
        target_top, target_left = max(0, top), max(0, left)
        rows = min(height - source_top, HEIGHT - target_top)
        columns = min(width - source_left, WIDTH - target_left)
        if rows > 0 and columns > 0:
            image[target_top:target_top + rows,
                  target_left:target_left + columns] = _OBJECT[source_top:source_top + rows,
                                                               source_left:source_left + columns]
    return np.ascontiguousarray(np.dstack([image] * 3))


def _view(canvas, depth, center):
    """What a camera at ``center`` sees of a scene canvas lying at ``depth`` metres."""
    return cv2.getRectSubPix(canvas, (WIDTH, HEIGHT),
                             (ORIGIN_X + FX * center[0] / depth, ORIGIN_Y + FX * center[1] / depth))


def _pan_sequence(frames=5, *, step=CAMERA_STEP, moving_object=True, cross=False):
    """The camera translates along +X; the world is static unless an obstacle is asked for."""
    centers = np.array([[step * index, 0.0, 0.0] for index in range(frames)])
    return ([_frame(index, centers[index], moving_object=moving_object, cross=cross)
             for index in range(frames)], centers)


def _hover_sequence(frames=5, *, moving_object=True):
    """Zero camera baseline: parallax can explain nothing, appearance is all there is."""
    centers = np.zeros((frames, 3))
    return ([_frame(index, centers[index], moving_object=moving_object)
             for index in range(frames)], centers)


def _crop(box, shape=(HEIGHT, WIDTH)):
    left, right, top, bottom = box
    return (slice(max(top, 0), min(bottom + 1, shape[0])),
            slice(max(left, 0), min(right + 1, shape[1])))


def _noise(frame, sigma=1.5, seed=5):
    rng = np.random.default_rng(seed)
    return np.clip(frame.astype(np.float32) + rng.normal(0.0, sigma, frame.shape),
                   0, 255).astype(np.uint8)


def _gray(frames):
    return [np.ascontiguousarray(frame[:, :, 0]) for frame in frames]


class ConstraintTests(unittest.TestCase):
    def test_importing_the_module_does_not_pull_in_torch(self):
        if _TORCH_BEFORE:
            self.skipTest("torch was already imported by the runner")
        self.assertNotIn("torch", sys.modules)


class BackgroundModelTests(unittest.TestCase):
    def test_median_over_the_window_recovers_the_static_scene(self):
        frames, centers = _hover_sequence(frames=5)
        model = dyn.background_model(frames)
        truth = _frame(0, centers[0], moving_object=False)
        difference = np.abs(model.image.astype(np.int16) - truth.astype(np.int16))
        # The obstacle covers any one pixel in fewer than half the frames, so the
        # per-pixel median is the static scene and not a ghost of the obstacle.
        self.assertLess(float(difference.mean()), 10.0)
        self.assertLess(float((difference > 60).mean()), 0.03)
        self.assertEqual(model.method, "median")
        self.assertEqual(model.frames_used, 5)
        self.assertEqual(model.image.shape, truth.shape)

    def test_window_is_capped_and_the_memory_cost_is_reported(self):
        frames, _ = _hover_sequence(frames=25)
        model = dyn.background_model(frames)
        self.assertEqual(model.window, dyn.MAX_BACKGROUND_WINDOW)
        self.assertEqual(model.frames_used, dyn.MAX_BACKGROUND_WINDOW)
        self.assertEqual(model.peak_bytes,
                         2 * model.frames_used * HEIGHT * WIDTH * 3)
        self.assertIsInstance(np.asarray(model), np.ndarray)

    def test_a_single_frame_is_a_legal_window(self):
        frames, _ = _hover_sequence(frames=1)
        self.assertTrue(np.array_equal(dyn.background_model(frames).image, frames[0]))

    def test_invalid_background_inputs_raise(self):
        frames, _ = _hover_sequence(frames=3)
        cases = [
            (lambda: dyn.background_model([]), "no frames"),
            (lambda: dyn.background_model(frames[:2] + [np.zeros((20, 20, 3), np.uint8)]),
             "same shape"),
            (lambda: dyn.background_model(frames, method="mean"), "method"),
            (lambda: dyn.background_model([np.full((9, 9, 3), np.nan)]), "finite"),
            (lambda: dyn.background_model([np.full((9, 9, 3), 128.0)]), "0..1"),
            (lambda: dyn.background_model(frames, window=0), "window"),
            (lambda: dyn.background_model(frames, window="eight"), "window"),
        ]
        for call, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                call()


class MotionMaskTests(unittest.TestCase):
    def test_a_moving_obstacle_is_localised_to_a_few_pixels(self):
        frames, centers = _hover_sequence(frames=5)
        background = dyn.background_model(frames)
        for index in (1, 2, 3, 4):
            result = dyn.motion_mask(frames[index], background)
            mask = result["mask"]
            rows, cols = _crop(_object_box(index, centers[index], pad=3))
            inside = mask[rows, cols]
            self.assertEqual(int(inside.sum()), int(mask.sum()),
                             "nothing outside the obstacle may be masked")
            self.assertGreater(float(inside.mean()), 0.55)
            self.assertGreater(result["threshold"], 0.0)
            self.assertEqual(result["threshold_method"], "otsu")
            self.assertGreaterEqual(int(result["raw_pixels"]), int(mask.sum()))

    def test_the_automatic_threshold_does_not_call_sensor_noise_motion(self):
        frames, _ = _hover_sequence(frames=5, moving_object=False)
        background = dyn.background_model(frames)
        for seed in (2, 3, 4):
            result = dyn.motion_mask(_noise(frames[2], sigma=1.5, seed=seed), background)
            self.assertLess(float(result["mask"].mean()), 0.005)
            self.assertGreaterEqual(result["threshold"], result["auto_floor"])

    def test_a_fixed_threshold_is_used_verbatim_and_reported(self):
        frames, _ = _hover_sequence(frames=5)
        background = dyn.background_model(frames)
        result = dyn.motion_mask(frames[1], background, threshold=20.0)
        self.assertEqual(result["threshold"], 20.0)
        self.assertEqual(result["threshold_method"], "fixed")
        self.assertTrue(np.array_equal(result["mask"],
                                       dyn.motion_mask(frames[1], background,
                                                       threshold=20.0)["mask"]))

    def test_min_area_and_morphology_remove_dust(self):
        frame = np.zeros((40, 40, 3), np.uint8)
        frame[10, 10] = 200
        frame[30, 30] = 200
        frame[18:28, 18:28] = 200
        cleaned = dyn.motion_mask(frame, np.zeros_like(frame), min_area_px=12)
        mask = cleaned["mask"]
        self.assertEqual(int(cleaned["raw_pixels"]), 102)
        self.assertFalse(mask[10, 10])
        self.assertFalse(mask[30, 30])
        self.assertTrue(mask[20:26, 20:26].all())
        self.assertEqual(int(mask.sum()), int(mask[16:30, 16:30].sum()))
        grown = dyn.motion_mask(frame, np.zeros_like(frame), min_area_px=1, grow_px=2)
        self.assertGreater(int(grown["mask"].sum()), int(cleaned["mask"].sum()))

    def test_a_frame_that_matches_its_background_masks_nothing(self):
        frames, _ = _hover_sequence(frames=3)
        result = dyn.motion_mask(frames[0], frames[0].copy())
        self.assertEqual(int(result["mask"].sum()), 0)
        self.assertEqual(result["area_fraction"], 0.0)

    def test_gray_and_color_frames_are_both_accepted(self):
        frames, _ = _hover_sequence(frames=3)
        color = dyn.motion_mask(frames[1], frames[2])["mask"]
        mono = dyn.motion_mask(_gray(frames)[1], _gray(frames)[2])["mask"]
        self.assertTrue(np.array_equal(color, mono))

    def test_motion_mask_validation(self):
        frames, _ = _hover_sequence(frames=2)
        background = frames[0]
        cases = [
            (lambda: dyn.motion_mask(frames[0], background[:, :60]), "same shape"),
            (lambda: dyn.motion_mask("frame", background), "array"),
            (lambda: dyn.motion_mask(np.zeros((9, 9), np.complex64),
                                     np.zeros((9, 9), np.complex64)), "dtype"),
            (lambda: dyn.motion_mask(np.zeros((9, 9, 2), np.uint8),
                                     np.zeros((9, 9, 2), np.uint8)), "channels"),
            (lambda: dyn.motion_mask(frames[0], background, threshold=300.0), "threshold"),
            (lambda: dyn.motion_mask(frames[0], background, threshold=-1.0), "threshold"),
            (lambda: dyn.motion_mask(frames[0], background, min_area_px=0), "min_area_px"),
            (lambda: dyn.motion_mask(frames[0], background, morph_open=-1), "morph_open"),
            (lambda: dyn.motion_mask(frames[0], background, grow_px=-1), "grow_px"),
            (lambda: dyn.motion_mask(frames[0], background, max_area_fraction=0.0),
             "max_area_fraction"),
        ]
        for call, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                call()


class EpipolarTests(unittest.TestCase):
    def _run(self, frames, centers, *, cross=False, depth_range=DEPTH_RANGE, **opts):
        kwargs = dict(tolerance_px=TOLERANCE, scene_depth_range=depth_range)
        kwargs.update(opts)
        return dyn.epipolar_inconsistency(_gray(frames)[1], _gray(frames)[2],
                                          centers[1:3], K, **kwargs)

    def test_pure_camera_motion_is_explained_by_the_cameras_own_geometry(self):
        frames, centers = _pan_sequence(frames=4, moving_object=False)
        result = self._run(frames, centers)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["usable"])
        self.assertGreater(result["tracks"]["count"], 60)
        self.assertGreater(result["tracks"]["inlier_ratio"], 0.6)
        # Real parallax is present, so the test has something to discriminate.
        self.assertGreater(result["parallax_px"], 2.0)
        self.assertLess(result["tracks"]["median_residual_px"], TOLERANCE)
        self.assertEqual(result["tracks"]["inconsistent_count"], 0)
        # A panning camera must not be read as a frame full of moving objects.
        self.assertLess(float(result["mask"].mean()), 0.01)
        self.assertGreater(float(result["static_veto"].mean()), 0.6)

    def test_parallax_really_differs_between_the_two_background_depths(self):
        """The scene must contain parallax, or the geometry test proves nothing."""
        frames, centers = _pan_sequence(frames=3, moving_object=False)
        first, second = _gray(frames)[0], _gray(frames)[1]
        shift = cv2.absdiff(second, first)
        self.assertGreater(float((shift > 8).mean()), 0.1)
        # And the two layers must move at different rates, which is the whole point.
        self.assertGreater(float((shift > 8).sum()), 1000)

    def test_motion_across_the_epipolar_lines_is_flagged_as_genuinely_moving(self):
        frames, centers = _pan_sequence(frames=4, cross=True)
        result = self._run(frames, centers, cross=True)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["tracks"]["inconsistent_count"], 3)
        rows, cols = _crop(_object_box(2, centers[2], cross=True, pad=8))
        flagged = np.zeros((HEIGHT, WIDTH), bool)
        flagged[result["tracks"]["inconsistent_pixels"][:, 1],
                result["tracks"]["inconsistent_pixels"][:, 0]] = True
        outside = flagged.copy()
        outside[rows, cols] = False
        self.assertEqual(int(outside.sum()), 0, "flagged points must sit on the obstacle")
        self.assertGreaterEqual(int(flagged[rows, cols].sum()), 3)
        self.assertGreater(float(result["mask"][rows, cols].mean()), 0.2)
        self.assertLess(float(result["static_veto"][rows, cols].mean()), 0.9)

    def test_the_veto_is_refused_where_the_geometry_says_parallax_is_impossible(self):
        """Along-epipolar motion beyond the parallax band must not be called static."""
        frames, centers = _pan_sequence(frames=4)
        result = self._run(frames, centers)
        rows, cols = _crop(_object_box(2, centers[2], pad=4))
        self.assertFalse(result["static_veto"][rows, cols].any(),
                         "the obstacle interior must not be vetoed as static parallax")
        self.assertGreater(result["tracks"]["inconsistent_count"], 3)
        self.assertIn("parallax", " ".join(result["reasons"]))

    def test_a_hovering_camera_reports_unreliable_instead_of_a_mask(self):
        frames, centers = _hover_sequence(frames=3)
        result = dyn.epipolar_inconsistency(_gray(frames)[0], _gray(frames)[1],
                                           centers[0:2], K, tolerance_px=TOLERANCE,
                                           scene_depth_range=DEPTH_RANGE)
        self.assertEqual(result["status"], "unreliable")
        self.assertFalse(result["usable"])
        self.assertEqual(int(result["mask"].sum()), 0)
        self.assertEqual(int(result["static_veto"].sum()), 0)
        self.assertIn("baseline", " ".join(result["reasons"]))

    def test_a_tiny_baseline_with_a_rotating_camera_reports_unreliable(self):
        frames = [_frame(index, np.array([1e-5 * index, 0.0, 0.0]), moving_object=False)
                  for index in range(2)]
        centers = np.array([[0.0, 0.0, 0.0], [1e-5, 0.0, 0.0]])
        result = dyn.epipolar_inconsistency(_gray(frames)[0], _gray(frames)[1], centers, K,
                                           tolerance_px=TOLERANCE,
                                           scene_depth_range=DEPTH_RANGE)
        self.assertEqual(result["status"], "unreliable")
        self.assertIn("parallax", " ".join(result["reasons"]))
        self.assertEqual(int(result["static_veto"].sum()), 0)

    def test_without_a_depth_prior_the_along_epipolar_test_is_declared_impossible(self):
        frames, centers = _pan_sequence(frames=3, moving_object=False)
        result = dyn.epipolar_inconsistency(_gray(frames)[0], _gray(frames)[1],
                                            centers[0:2], K, tolerance_px=TOLERANCE)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["parallax_band_px"])
        self.assertIn("no depth prior", " ".join(result["reasons"]))

    def test_a_blank_pair_cannot_be_judged(self):
        blank = np.zeros((HEIGHT, WIDTH), np.uint8)
        result = dyn.epipolar_inconsistency(blank, blank, [[0, 0, 0], [0.1, 0, 0]], K,
                                           tolerance_px=TOLERANCE)
        self.assertEqual(result["status"], "unreliable")
        self.assertIn("correspondences", " ".join(result["reasons"]))

    def test_the_result_is_reproducible_for_the_same_inputs(self):
        frames, centers = _pan_sequence(frames=3, moving_object=False)
        args = (_gray(frames)[0], _gray(frames)[1], centers[0:2], K)
        first = dyn.epipolar_inconsistency(*args, tolerance_px=TOLERANCE,
                                          scene_depth_range=DEPTH_RANGE)
        for _ in range(3):
            again = dyn.epipolar_inconsistency(*args, tolerance_px=TOLERANCE,
                                              scene_depth_range=DEPTH_RANGE)
            self.assertTrue(np.array_equal(first["mask"], again["mask"]))
            self.assertTrue(np.array_equal(first["static_veto"], again["static_veto"]))
            self.assertAlmostEqual(first["tracks"]["median_residual_px"],
                                   again["tracks"]["median_residual_px"], places=9)

    def test_epipolar_validation(self):
        frames, centers = _pan_sequence(frames=3, moving_object=False)
        gray = _gray(frames)[0]
        pair = centers[0:2]
        cases = [
            (lambda: dyn.epipolar_inconsistency(gray, gray[:, :40], pair, K,
                                               tolerance_px=TOLERANCE), "same shape"),
            (lambda: dyn.epipolar_inconsistency(np.zeros((8, 8), np.uint8),
                                                np.zeros((8, 8), np.uint8), pair, K,
                                                tolerance_px=TOLERANCE), "too small"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, centers, K,
                                               tolerance_px=TOLERANCE), "two camera centers"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, [[0, 0], [1, 0], [2, 0]], K,
                                               tolerance_px=TOLERANCE), "two camera centers"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, np.zeros((2, 2)),
                                               tolerance_px=TOLERANCE), "3x3"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, np.zeros((3, 3)),
                                               tolerance_px=TOLERANCE), "focal"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K * np.nan,
                                               tolerance_px=TOLERANCE), "finite"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K, tolerance_px=0.0),
             "tolerance_px"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K, tolerance_px=np.inf),
             "tolerance_px"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K,
                                               tolerance_px=TOLERANCE, min_tracks=1),
             "min_tracks"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K,
                                               tolerance_px=TOLERANCE, min_inlier_ratio=1.5),
             "min_inlier_ratio"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K,
                                               tolerance_px=TOLERANCE,
                                               scene_depth_range=(8.0, 4.0)),
             "scene_depth_range"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K,
                                               tolerance_px=TOLERANCE,
                                               scene_depth_range=(0.0, 5.0)),
             "scene_depth_range"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K,
                                               tolerance_px=TOLERANCE,
                                               scene_depth_range="near"),
             "scene_depth_range"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, [[0, 0], [np.nan, 1]], K,
                                               tolerance_px=TOLERANCE), "camera_centers"),
            (lambda: dyn.epipolar_inconsistency(gray, gray, pair, K,
                                               tolerance_px=TOLERANCE, rng_seed=-1),
             "rng_seed"),
        ]
        for call, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                call()


class SequenceTests(unittest.TestCase):
    def test_static_camera_localises_the_obstacle_and_leaves_the_scene_alone(self):
        frames, centers = _hover_sequence(frames=5)
        results = dyn.dynamic_masks_for_sequence(frames, centers, K,
                                                 tolerance_px=TOLERANCE)
        self.assertIn("no-preceding-frame", " ".join(results[0]["reasons"]))
        self.assertEqual(len(results), len(frames))
        for index, item in enumerate(results):
            rows, cols = _crop(_object_box(index, centers[index], pad=3))
            inside = item["mask"][rows, cols]
            self.assertEqual(int(inside.sum()), int(item["mask"].sum()))
            self.assertGreater(float(inside.mean()), 0.5)
            self.assertLessEqual(item["confidence"], 0.6)
            if index:
                self.assertIn("no-camera-baseline", " ".join(item["reasons"]))

    def test_a_panning_camera_does_not_erase_the_static_scene(self):
        frames, centers = _pan_sequence(frames=5, moving_object=False)
        background = dyn.background_model(frames)
        naive = [dyn.motion_mask(frame, background)["mask"] for frame in frames]
        results = dyn.dynamic_masks_for_sequence(frames, centers, K,
                                                 tolerance_px=TOLERANCE,
                                                 scene_depth_range=DEPTH_RANGE)
        naive_mean = float(np.mean([m.mean() for m in naive]))
        masked_mean = float(np.mean([item["mask"].mean() for item in results[1:]]))
        self.assertGreater(naive_mean, 0.15,
                           "appearance alone really does light up the frame here")
        self.assertLess(masked_mean, 0.03,
                        "parallax must not be mistaken for moving objects")
        self.assertLess(masked_mean, naive_mean / 5.0)
        for index, item in enumerate(results):
            if index == 0:
                continue
            self.assertEqual(item["epipolar_status"], "ok")
            self.assertIn("camera-motion-explained", item["reasons"])
            self.assertGreater(item["vetoed_fraction"], 0.7)
            self.assertGreater(item["confidence"], 0.6)

    def test_a_pan_with_a_real_obstacle_masks_the_obstacle_not_the_frame(self):
        for cross in (True, False):
            frames, centers = _pan_sequence(frames=5, cross=cross)
            results = dyn.dynamic_masks_for_sequence(frames, centers, K,
                                                    tolerance_px=TOLERANCE,
                                                    scene_depth_range=DEPTH_RANGE)
            covered, outside, areas = [], [], []
            for index, item in enumerate(results):
                if index == 0:
                    continue
                rows, cols = _crop(_object_box(index, centers[index], cross=cross, pad=8))
                mask = item["mask"]
                covered.append(float(mask[rows, cols].mean()))
                rest = mask.copy()
                rest[rows, cols] = False
                outside.append(float(rest.mean()))
                areas.append(float(mask.mean()))
            with self.subTest(cross=cross):
                self.assertGreater(float(np.mean(covered)), 0.25,
                                   "the obstacle must stay visible in the mask")
                self.assertLess(float(np.mean(outside)), 0.03,
                                "everything masked must sit on the moving obstacle")
                self.assertLess(float(np.mean(areas)), 0.08)

    def test_every_frame_carries_a_confidence_and_the_reasons_that_shaped_it(self):
        frames, centers = _hover_sequence(frames=4)
        results = dyn.dynamic_masks_for_sequence(frames, centers, K,
                                                 tolerance_px=TOLERANCE)
        expected = {"index", "mask", "confidence", "status", "reasons", "area_fraction",
                    "epipolar_status", "vetoed_fraction", "background_threshold", "epipolar"}
        for index, item in enumerate(results):
            self.assertEqual(set(item), expected)
            self.assertEqual(item["index"], index)
            self.assertGreaterEqual(item["confidence"], 0.0)
            self.assertLessEqual(item["confidence"], 1.0)
            self.assertIsInstance(item["reasons"], list)
            self.assertTrue(all(isinstance(r, str) for r in item["reasons"]))
            self.assertAlmostEqual(item["area_fraction"], float(item["mask"].mean()),
                                   places=9)
            self.assertEqual(item["mask"].dtype, np.bool_)
            self.assertEqual(item["status"], "appearance-only")
        self.assertEqual(results[2]["epipolar"]["status"], "unreliable")

    def test_sequence_validation(self):
        frames, centers = _hover_sequence(frames=3)
        cases = [
            (lambda: dyn.dynamic_masks_for_sequence([], centers[:0], K), "no frames"),
            (lambda: dyn.dynamic_masks_for_sequence(frames, centers[:2], K),
             "one camera center per frame"),
            (lambda: dyn.dynamic_masks_for_sequence(frames, np.zeros((3, 4)), K),
             "2 or 3 columns"),
            (lambda: dyn.dynamic_masks_for_sequence(frames, centers, np.eye(2)), "3x3"),
            (lambda: dyn.dynamic_masks_for_sequence(frames[:2] + [np.zeros((9, 9, 3), np.uint8)],
                                                   centers, K), "same shape"),
            (lambda: dyn.dynamic_masks_for_sequence(frames, centers, K, bogus=1), "bogus"),
            (lambda: dyn.dynamic_masks_for_sequence(frames, centers, K, tolerance_px=0),
             "tolerance_px"),
            (lambda: dyn.dynamic_masks_for_sequence(frames, centers, K, window=0),
             "window"),
        ]
        for call, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                call()


class CoverageTests(unittest.TestCase):
    def test_coverage_cost_grows_monotonically_with_masked_area(self):
        shape = (40, 50)
        previous = -1.0
        for side in (0, 5, 10, 20, 35):
            masks = []
            for _ in range(3):
                mask = np.zeros(shape, bool)
                if side:
                    mask[:side, :side] = True
                masks.append({"mask": mask})
            cost = dyn.coverage_cost(masks, shape)
            self.assertGreater(cost["discounted_fraction"], previous)
            previous = cost["discounted_fraction"]
            self.assertEqual(cost["discounted_area_px"], 3 * side * side)
        self.assertAlmostEqual(previous, 35 * 35 / (40 * 50))
        self.assertEqual(dyn.coverage_cost([np.zeros(shape, bool)], shape)["discounted_fraction"],
                         0.0)

    def test_coverage_cost_names_the_completeness_price(self):
        shape = (40, 50)
        masks = []
        for _ in range(4):
            mask = np.zeros(shape, bool)
            mask[:20, :] = True  # half of every frame discounted
            masks.append({"mask": mask})
        cost = dyn.coverage_cost(masks, shape, budget=0.25)
        self.assertAlmostEqual(cost["discounted_fraction"], 0.5)
        self.assertAlmostEqual(cost["max_frame_fraction"], 0.5)
        self.assertTrue(cost["over_budget"])
        self.assertIn("completeness", cost["statement"].lower())
        self.assertEqual(cost["frames"], 4)
        self.assertEqual(cost["shape"], shape)
        self.assertFalse(dyn.coverage_cost([np.zeros(shape, bool)], shape)["over_budget"])

    def test_coverage_cost_validation(self):
        shape = (20, 20)
        cases = [
            (lambda: dyn.coverage_cost([], shape), "no masks"),
            (lambda: dyn.coverage_cost([np.zeros((9, 9), bool)], shape), "has shape"),
            (lambda: dyn.coverage_cost([np.zeros((*shape, 3), bool)], shape), "2-D"),
            (lambda: dyn.coverage_cost(["nope"], shape), "mask"),
            (lambda: dyn.coverage_cost([np.zeros(shape, bool)], shape, budget=2.0), "budget"),
            (lambda: dyn.coverage_cost([np.zeros(shape, bool)], (20,)), "shape"),
        ]
        for call, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                call()


class WeightTests(unittest.TestCase):
    @staticmethod
    def _masks(fraction, shape=(20, 20), count=3):
        side = max(0, int(round(np.sqrt(fraction) * shape[0])))
        out = []
        for _ in range(count):
            mask = np.zeros(shape, bool)
            if side:
                mask[:side, :side] = True
            out.append({"mask": mask})
        return out

    def test_observations_are_down_weighted_never_deleted(self):
        masks = self._masks(1.0, count=2) + self._masks(0.0, count=2)
        weights = np.full(4, 0.8)
        reduced = dyn.apply_to_weights(masks, weights, factor=0.25)
        self.assertEqual(reduced.shape, weights.shape)
        self.assertAlmostEqual(reduced[0], 0.8 * 0.25)
        self.assertAlmostEqual(reduced[2], 0.8)
        self.assertTrue((reduced > 0).all())
        self.assertTrue((reduced <= weights + 1e-12).all())

    def test_a_floor_bounds_the_discount(self):
        masks = self._masks(1.0, count=1)
        reduced = dyn.apply_to_weights(masks, np.array([1.0]), factor=1e-6, floor=0.2)
        self.assertAlmostEqual(float(reduced[0]), 0.2)

    def test_the_discount_grows_with_the_masked_fraction(self):
        previous = 2.0
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            reduced = dyn.apply_to_weights(self._masks(fraction, count=1), np.ones(1),
                                           factor=0.2)
            self.assertLess(float(reduced[0]), previous)
            previous = float(reduced[0])

    def test_pixels_select_exactly_the_observations_inside_a_mask(self):
        mask = np.zeros((20, 20), bool)
        mask[5:10, 5:10] = True
        pixels = np.array([[0, 0], [7, 7], [15, 2], [9, 9]])
        reduced = dyn.apply_to_weights([mask, mask], np.ones((2, 4)), pixels=pixels,
                                      factor=0.1)
        self.assertTrue(np.allclose(reduced[:, 0], 1.0))
        self.assertTrue(np.allclose(reduced[:, 1], 0.1))
        self.assertTrue(np.allclose(reduced[:, 2], 1.0))
        self.assertTrue(np.allclose(reduced[:, 3], 0.1))

    def test_pixels_may_be_given_per_view(self):
        mask = np.zeros((10, 10), bool)
        mask[:5, :] = True
        pixels = np.array([[[6, 4], [1, 1]], [[1, 1], [6, 4]]])
        reduced = dyn.apply_to_weights([mask, mask], np.ones((2, 2)), pixels=pixels,
                                       factor=0.5)
        self.assertTrue(np.allclose(reduced, [[1.0, 0.5], [0.5, 1.0]]))

    def test_a_flattened_pixel_grid_is_applied_exactly(self):
        shape = (4, 5)
        mask = np.zeros(shape, bool)
        mask[:2, :] = True
        reduced = dyn.apply_to_weights([mask], np.ones((1, 20)), factor=0.5)
        self.assertEqual(float(reduced[0, 0]), 0.5)
        self.assertEqual(float(reduced[0, 9]), 0.5)
        self.assertEqual(float(reduced[0, 10]), 1.0)
        self.assertEqual(float(reduced[0, 19]), 1.0)

    def test_weight_validation(self):
        shape = (6, 6)
        mask = np.zeros(shape, bool)
        mask[:3, :3] = True
        cases = [
            (lambda: dyn.apply_to_weights([mask], np.ones(2)), "per mask"),
            (lambda: dyn.apply_to_weights([mask], np.ones((2, 6))), "per mask"),
            (lambda: dyn.apply_to_weights([mask], np.ones(1),
                                          pixels=np.zeros((4, 3))), "pixels"),
            (lambda: dyn.apply_to_weights([mask], np.ones((1, 4)),
                                          pixels=np.full((4, 2), 99)), "outside"),
            (lambda: dyn.apply_to_weights([mask], np.ones((1, 5))), "pixels"),
            (lambda: dyn.apply_to_weights([mask], np.array([np.nan])), "finite"),
            (lambda: dyn.apply_to_weights([mask], np.array([-1.0])), "non-negative"),
            (lambda: dyn.apply_to_weights([mask], np.ones(1), factor=1.5), "factor"),
            (lambda: dyn.apply_to_weights([mask], np.ones(1), floor=-0.1), "floor"),
            (lambda: dyn.apply_to_weights(["nope"], np.ones(1)), "mask"),
            (lambda: dyn.apply_to_weights([np.ones(shape, np.int32)], np.ones(1)), "boolean"),
            (lambda: dyn.apply_to_weights([], np.ones(1)), "no masks"),
        ]
        for call, needle in cases:
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                call()


class StaticObstacleNoteTests(unittest.TestCase):
    def test_the_note_says_what_the_module_refuses_to_do(self):
        frames, centers = _pan_sequence(frames=3, cross=True)
        results = dyn.dynamic_masks_for_sequence(frames, centers, K,
                                                 tolerance_px=TOLERANCE,
                                                 scene_depth_range=DEPTH_RANGE)
        note = dyn.static_obstacle_note(results, (HEIGHT, WIDTH))
        text = note["statement"].lower()
        self.assertIn("parked", text)
        self.assertIn("vegetation", text)
        self.assertIn("down-weight", text)
        self.assertTrue(note["stationary_content_retained"])
        self.assertEqual(note["frames"], 3)
        self.assertEqual(note["deleted_pixels"], 0)
        self.assertIn("class labels", " ".join(note["needs_a_model_for"]).lower())
        self.assertGreaterEqual(note["discounted_fraction"], 0.0)

    def test_note_validation(self):
        with self.assertRaisesRegex(ValueError, "no masks"):
            dyn.static_obstacle_note([], (4, 4))


if __name__ == "__main__":
    unittest.main()
