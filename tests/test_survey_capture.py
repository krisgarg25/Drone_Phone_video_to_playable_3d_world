"""Tests for scripts/survey_capture.py: the frame-level decisions of a survey run.

Synthetic frames on purpose: a moving square on a still background is a dynamic
object the geometry can prove, and a lit gradient is an illumination field. Neither
claims anything about a real flight.
"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts import survey_capture as capture
from scripts import survey_georef as georef

METADATA = {"schema_version": 1, "time_reference": "video", "time_offset_s": 0,
            "altitude_datum": "ellipsoidal", "position_reference": "camera_center",
            "single_pass": True, "video_duration_s": 60}


def telemetry_fixture(rows=61):
    lines = ["t_sec,latitude_deg,longitude_deg,altitude_m,horizontal_std_m,vertical_std_m"]
    for index in range(rows):
        # A lawnmower-ish pass: east at altitude, then a return leg, so the plan has
        # real geometry to spend its budget on rather than a straight line.
        east = index * 0.00002
        north = 0.0 if index < rows // 2 else (index - rows // 2) * 0.00002
        lines.append(f"{index},{28.6 + north:.7f},{77.2 + east:.7f},{230 + index * 0.05},"
                     f"1.5,3.0")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "telemetry.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return georef.normalize_telemetry(path, METADATA)


def sharp_frame(size=160, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size), dtype=np.uint8)


def shaded_frame(size=160):
    yy, xx = np.mgrid[0:size, 0:size]
    texture = np.random.default_rng(3).integers(90, 170, (size, size))
    field = (120 + 70 * (xx / size) + 40 * (yy / size)).astype(float)
    return np.clip(texture * field / 120.0, 0, 255).astype(np.uint8)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.telemetry = telemetry_fixture()

    def test_the_budget_is_spent_on_geometry_not_on_the_clock(self):
        result = capture.plan(self.telemetry, video_duration_s=60, budget=20)
        times = result["target_times_s"]
        self.assertLessEqual(len(times), 20)
        self.assertEqual(times, sorted(times))
        # path_coverage measures arc within one baseline of a chosen frame, so it is
        # the baseline that sets the scale here; what must grow with the budget is
        # the covered fraction and what must shrink is the largest gap.
        thin = capture.plan(self.telemetry, video_duration_s=60, budget=6)
        self.assertGreater(result["path_coverage"]["covered_fraction"],
                           thin["path_coverage"]["covered_fraction"])
        self.assertLess(result["path_coverage"]["max_gap_m"], thin["path_coverage"]["max_gap_m"])
        self.assertGreater(result["mean_parallax_deg_to_flight_box_centre"], 0.0)
        self.assertIn("flight-box centre", result["parallax_basis"])
        self.assertEqual(result["candidates_anchored"] + result["unanchored_dropped"],
                         result["candidates"])
        self.assertIn("not a measurement", result["basis"])

    def test_a_baseline_floor_thins_the_selection(self):
        loose = capture.plan(self.telemetry, video_duration_s=60, budget=30,
                             min_baseline_m=0.5)
        tight = capture.plan(self.telemetry, video_duration_s=60, budget=30,
                             min_baseline_m=25.0)
        self.assertLess(tight["frame_count"], loose["frame_count"])
        self.assertGreaterEqual(tight["path_coverage"]["max_gap_m"],
                                loose["path_coverage"]["max_gap_m"])

    def test_an_unusable_request_is_refused(self):
        for budget in (1, 0, -5):
            with self.assertRaises(ValueError):
                capture.plan(self.telemetry, video_duration_s=60, budget=budget)
        with self.assertRaises(ValueError):
            capture.plan({"samples": []}, video_duration_s=60, budget=10)
        with self.assertRaises(ValueError):
            capture.plan(self.telemetry, video_duration_s=float("nan"), budget=10)
        with self.assertRaises(ValueError):
            capture.plan(self.telemetry, video_duration_s=60, budget=10,
                         min_baseline_m=0.0)

    def test_a_plan_survives_json_and_reads_back(self):
        import json
        result = capture.plan(self.telemetry, video_duration_s=60, budget=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_plan.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(capture.read_plan(path)["target_times_s"],
                             result["target_times_s"])
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                capture.read_plan(path)


class FrameQualityTests(unittest.TestCase):
    def test_a_blurred_frame_scores_below_a_sharp_one(self):
        sharp = sharp_frame()
        blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
        good = capture.assess_frame(sharp)
        bad = capture.assess_frame(blurred, energy_floor=good["blur"]["tenengrad"])
        self.assertLess(bad["weight"], good["weight"])
        self.assertTrue(good["keep"])

    def test_the_floor_is_declared_not_invented(self):
        """Without a caller-supplied floor the label degrades, and says so."""
        report = capture.assess_frame(cv2.GaussianBlur(sharp_frame(), (0, 0), 6))
        self.assertIsNone(report["energy_floor"])
        self.assertIn(report["blur_label"], ("directional", "isotropic", "motion_blur",
                                             "defocus_blur", "low_texture_or_blur"))


class PhotometryTests(unittest.TestCase):
    def test_flattening_reduces_the_illumination_gradient(self):
        shaded = shaded_frame()
        flat, actions = capture.prepare_for_matching(shaded, flatten=True)
        self.assertEqual(flat.shape, shaded.shape)
        self.assertEqual(flat.dtype, shaded.dtype)
        self.assertEqual(actions, ["illumination_flattened"])
        before = np.asarray(shaded, float).reshape(4, -1).mean(axis=1)
        after = np.asarray(flat, float).reshape(4, -1).mean(axis=1)
        self.assertLess(after.max() - after.min(), before.max() - before.min())

    def test_both_operations_are_recorded_in_order(self):
        _, actions = capture.prepare_for_matching(shaded_frame(), flatten=True,
                                                  normalise=True, target_mean=128.0)
        self.assertEqual(len(actions), 2)
        self.assertTrue(actions[0].startswith("illumination_flattened"))
        self.assertTrue(actions[1].startswith("exposure_normalised"))

    def test_a_passthrough_copy_is_available(self):
        image, actions = capture.prepare_for_matching(shaded_frame(), flatten=False)
        self.assertEqual(actions, [])
        np.testing.assert_array_equal(image, shaded_frame())


class IntrinsicsTests(unittest.TestCase):
    def test_a_declared_focal_is_used_and_a_default_is_guessable(self):
        declared = capture.camera_matrix(1920, 1080, focal_px=1060.0)
        guessed = capture.camera_matrix(1920, 1080)
        self.assertEqual(declared[0, 0], 1060.0)
        self.assertEqual(guessed[0, 0], 1.2 * 1920)
        self.assertEqual(list(guessed[2]), [0.0, 0.0, 1.0])
        self.assertEqual(guessed[0, 2], 960.0)


class MaskTests(unittest.TestCase):
    def frames_with_a_moving_object(self, count=6, size=160):
        background = np.full((size, size), 60, dtype=np.uint8)
        for y in range(0, size, 16):
            background[y, :] = 200
        for x in range(0, size, 23):
            background[:, x] = 170
        frames = []
        for index in range(count):
            frame = background.copy()
            cx = 20 + index * 18
            frame[cx:cx + 22, cx:cx + 22] = 20
            frame[cx + 3:cx + 19, cx + 3:cx + 19] = 250
            frames.append(frame)
        return frames

    def setUp(self):
        self.frames = self.frames_with_a_moving_object()
        self.centers = np.column_stack([np.arange(len(self.frames)) * 2.0,
                                        np.zeros(len(self.frames)),
                                        np.full(len(self.frames), 40.0)])
        self.k = capture.camera_matrix(160, 160, focal_px=150.0)

    def test_a_moving_region_is_masked_and_the_rest_is_kept(self):
        results, meta = capture.masks_for_sequence(self.frames, self.centers, self.k)
        self.assertEqual(len(results), len(self.frames))
        self.assertTrue(any(row["mask"].any() for row in results))
        self.assertLess(meta["coverage_cost"]["discounted_fraction"], 0.9)
        self.assertTrue(meta["static_obstacle_note"]["stationary_content_retained"])

    def test_masks_are_written_with_colmap_semantics(self):
        results, _ = capture.masks_for_sequence(self.frames, self.centers, self.k)
        names = [f"frame_{index:05d}.jpg" for index in range(len(self.frames))]
        with tempfile.TemporaryDirectory() as directory:
            summary = capture.write_masks(results, names, Path(directory) / "masks")
            self.assertEqual(summary["count"], len(self.frames))
            self.assertIn("0 = excluded", summary["convention"])
            first = Path(directory) / "masks" / summary["files"][0]["mask"]
            self.assertTrue(first.is_file())
            pixels = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(pixels.shape, self.frames[0].shape)
            masked = results[0]["mask"]
            if masked.any():
                self.assertTrue((pixels[masked] == 0).all())
                self.assertTrue((pixels[~masked] == 255).all())

    def test_a_mask_is_written_at_the_image_size_not_the_working_size(self):
        """COLMAP refuses an image whose mask disagrees in pixel size - all 72 frames
        of a real scene were lost this way before the frame-count guard caught it."""
        results, _ = capture.masks_for_sequence(self.frames, self.centers, self.k)
        names = [f"frame_{index:05d}.jpg" for index in range(len(self.frames))]
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "frames_match"
            images.mkdir()
            # The images on disk are the full-resolution matching frames, twice the
            # size the masks were computed at - exactly the mismatch survey_frames
            # creates, and exactly what COLMAP will not accept.
            for frame, name in zip(self.frames, names):
                large = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2),
                                   interpolation=cv2.INTER_NEAREST)
                self.assertTrue(cv2.imwrite(str(images / name), large))
            summary = capture.write_masks(results, names, Path(directory) / "masks",
                                          images_dir=images)
            self.assertEqual(summary["files"][0]["mask_size"], [self.frames[0].shape[1] * 2,
                                                                self.frames[0].shape[0] * 2])
            for entry in summary["files"]:
                png = cv2.imread(str(Path(directory) / "masks" / entry["mask"]),
                                 cv2.IMREAD_GRAYSCALE)
                self.assertEqual(png.shape, (self.frames[0].shape[0] * 2,
                                             self.frames[0].shape[1] * 2))

    def test_a_mask_for_a_missing_image_is_refused(self):
        results, _ = capture.masks_for_sequence(self.frames, self.centers, self.k)
        names = [f"frame_{index:05d}.jpg" for index in range(len(self.frames))]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "empty").mkdir()
            with self.assertRaisesRegex(ValueError, "cannot read the image"):
                capture.write_masks(results, names, Path(directory) / "masks",
                                    images_dir=Path(directory) / "empty")

    def test_mask_paths_follow_colmaps_own_naming_convention(self):
        """Verified against the tool: <mask_path>/<db image name>.png, subdir kept."""
        import cv2
        results, _ = capture.masks_for_sequence(self.frames, self.centers, self.k)
        names = [f"rocks/frame_{index:05d}.jpg" for index in range(len(self.frames))]
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "frames_match" / "rocks"
            images.mkdir(parents=True)
            for frame, name in zip(self.frames, names):
                self.assertTrue(cv2.imwrite(str(Path(directory) / "frames_match" / name), frame))
            summary = capture.write_masks(results, names, Path(directory) / "masks",
                                          images_dir=Path(directory) / "frames_match")
            for entry, name in zip(summary["files"], names):
                self.assertEqual(entry["mask"], name + ".png")
                self.assertTrue((Path(directory) / "masks" / entry["mask"]).is_file())

    def test_a_single_frame_cannot_prove_motion(self):
        with self.assertRaises(ValueError):
            capture.masks_for_sequence(self.frames[:1], self.centers[:1], self.k)


if __name__ == "__main__":
    unittest.main()
