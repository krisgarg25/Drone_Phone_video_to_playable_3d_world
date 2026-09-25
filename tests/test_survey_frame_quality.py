"""Tests for no-reference frame-quality proxies (motion blur / compression).

Everything here is validated on synthetic images whose degradation we *control*:
a sharp checkerboard, a Gaussian-defocused copy, a horizontally box-filtered
(motion) copy and JPEG round-trips at two qualities. That is the point - on real
footage we have no ground truth, so the only honest claim available is "these
image statistics separate degradations we can construct on purpose".
"""
import copy
import unittest

import cv2
import numpy as np

from scripts import survey_frame_quality as fq


def checkerboard(size=256, cell=13, seed=7):
    """Sharp test target: both edge orientations present, not exactly periodic.

    Two deliberate choices. Per-cell random amplitude keeps the gradient
    orientation balance honest - a mathematically perfect checkerboard aliases
    against any fixed smoothing window and biases the anisotropy statistic. And
    a 13-pixel cell is not a multiple of the 8-pixel JPEG lattice, because a
    grid-aligned target puts every edge exactly on a boundary and makes any
    blockiness statistic read a huge value that has nothing to do with
    compression.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size] // cell
    parity = (rows + cols) % 2
    amplitude = rng.uniform(0.65, 1.0, (size // cell + 2, size // cell + 2))
    levels = np.where(parity, 235, 30) * amplitude[rows, cols]
    return np.clip(levels + rng.normal(0, 2.5, (size, size)), 0, 255).astype(np.uint8)


def smooth_scene(size=256, seed=4):
    """Low-contrast, gently varying content - what a wall or road decodes to.

    Blocking artefacts are only *visible* where the source is smooth, so this is
    the target the compression statistic has to be judged on.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    wave = 110 + 45 * np.sin(x / 37.0) + 25 * np.cos(y / 53.0)
    return np.clip(wave + rng.normal(0, 5, (size, size)), 0, 255).astype(np.uint8)


def gaussian_defocus(gray, sigma=3.0):
    return cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def motion_blur(gray, length=17):
    """Box filter along +x: attenuates vertical edges only, the directional case."""
    kernel = np.full((1, length), 1.0 / length, np.float64)
    return cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT)


def grating(size=256, period=26):
    """One-directional bars: sharp, high-energy, and wholly non-isotropic."""
    bars = ((np.arange(size) % period) < period // 2) * 180 + 30
    return np.tile(bars, (size, 1)).astype(np.uint8)


def jpeg_roundtrip(gray, quality):
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


class BlurMetricsTests(unittest.TestCase):
    def setUp(self):
        self.sharp = checkerboard()
        self.defocused = gaussian_defocus(self.sharp)
        self.motional = motion_blur(self.sharp)

    def test_reports_the_three_expected_signals(self):
        out = fq.blur_metrics(self.sharp)
        for key in ("laplacian_variance", "tenengrad", "anisotropy",
                    "orientation_deg", "spectral_centroid"):
            self.assertIn(key, out)
        self.assertTrue(0.0 <= out["anisotropy"] <= 1.0, out["anisotropy"])

    def test_laplacian_variance_matches_the_keyframe_extractors_definition(self):
        # extract_keyframes.py scores candidates with exactly this number, so the
        # two must not drift apart or the manifest's `sharpness` field means one
        # thing during selection and another during weighting.
        expected = float(cv2.Laplacian(self.sharp, cv2.CV_64F).var())
        self.assertAlmostEqual(fq.blur_metrics(self.sharp)["laplacian_variance"],
                               expected, places=6)

    def test_both_blurs_reduce_gradient_energy(self):
        sharp = fq.blur_metrics(self.sharp)
        defocused = fq.blur_metrics(self.defocused)
        motional = fq.blur_metrics(self.motional)
        self.assertLess(defocused["laplacian_variance"], sharp["laplacian_variance"])
        self.assertLess(motional["laplacian_variance"], sharp["laplacian_variance"])
        self.assertLess(defocused["tenengrad"], sharp["tenengrad"])
        self.assertLess(motional["tenengrad"], sharp["tenengrad"])

    def test_anisotropy_separates_directional_motion_from_isotropic_defocus(self):
        sharp = fq.blur_metrics(self.sharp)["anisotropy"]
        defocused = fq.blur_metrics(self.defocused)["anisotropy"]
        motional = fq.blur_metrics(self.motional)["anisotropy"]
        # Isotropic attenuation leaves the orientation balance alone; a box
        # filter along x removes one axis of the gradient and does not.
        self.assertLess(defocused, sharp + 0.10, (defocused, sharp))
        self.assertGreater(motional, defocused + 0.25, (motional, defocused))
        self.assertGreater(motional, fq.MOTION_ANISOTROPY_MIN)
        self.assertLess(defocused, fq.MOTION_ANISOTROPY_MIN)

    def test_motion_direction_is_reported_perpendicular_to_the_smearing(self):
        # Horizontal smearing kills horizontal gradients, so the surviving
        # gradient runs vertically: 90 degrees.
        out = fq.blur_metrics(self.motional)
        self.assertAlmostEqual(out["orientation_deg"], 90.0, delta=12.0)

    def test_isotropic_defocus_collapses_the_high_band(self):
        sharp = fq.blur_metrics(self.sharp)["spectral_centroid"]
        defocused = fq.blur_metrics(self.defocused)["spectral_centroid"]
        self.assertLess(defocused, sharp)

    def test_blur_label_names_the_two_failure_modes_differently(self):
        reference = fq.blur_metrics(self.sharp)
        floor = 0.5 * reference["laplacian_variance"]
        motion = fq.blur_metrics(self.motional)
        defocused = fq.blur_metrics(self.defocused)
        self.assertEqual(fq.blur_label(motion, energy_floor=floor), "motion_blur")
        self.assertEqual(fq.blur_label(defocused, energy_floor=floor,
                                       centroid_floor=reference["spectral_centroid"]),
                         "defocus_blur")
        self.assertEqual(fq.blur_label(reference, energy_floor=floor), "sharp")

    def test_defocus_without_a_spectral_reference_falls_back_to_the_honest_label(self):
        # Low energy and no direction could be defocus or a textureless wall.
        # Saying "defocus" without a reference would be inventing a measurement.
        reference = fq.blur_metrics(self.sharp)
        floor = 0.5 * reference["laplacian_variance"]
        self.assertEqual(fq.blur_label(fq.blur_metrics(self.defocused), energy_floor=floor),
                         "low_texture_or_blur")

    def test_without_an_energy_floor_absolute_blur_is_left_unlabelled(self):
        # A flat white wall has no gradient energy at all and statistics alone
        # cannot tell it apart from defocus, so no floor means no absolute verdict.
        label = fq.blur_label(fq.blur_metrics(gaussian_defocus(self.sharp, 3.0)))
        self.assertEqual(label, "isotropic")


class CompressionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.clean = smooth_scene()
        self.light = jpeg_roundtrip(self.clean, 90)
        self.heavy = jpeg_roundtrip(self.clean, 20)

    def test_reports_a_block_boundary_ratio(self):
        out = fq.compression_metrics(self.clean)
        for key in ("block_boundary_ratio", "boundary_delta", "intra_delta",
                    "flat_block_fraction", "grid_excess", "ratio_saturated", "score"):
            self.assertIn(key, out)
        self.assertTrue(0.0 <= out["score"] <= 1.0)
        self.assertTrue(np.isfinite(out["block_boundary_ratio"]))

    def test_a_clean_frame_sits_at_one_because_the_ratio_is_a_ratio(self):
        # Nothing about the lattice is special in uncompressed content, so the
        # baseline is 1.0 rather than a tuned constant.
        self.assertAlmostEqual(fq.compression_metrics(self.clean)["block_boundary_ratio"],
                               1.0, delta=0.15)

    def test_heavier_jpeg_scores_worse_on_every_axis(self):
        light, heavy = fq.compression_metrics(self.light), fq.compression_metrics(self.heavy)
        self.assertGreater(heavy["block_boundary_ratio"], light["block_boundary_ratio"])
        self.assertGreater(heavy["grid_excess"], light["grid_excess"])
        self.assertLess(heavy["score"], light["score"])
        self.assertGreaterEqual(heavy["flat_block_fraction"], light["flat_block_fraction"])
        self.assertTrue(heavy["blocking_suspect"])
        self.assertFalse(light["blocking_suspect"])

    def test_the_ratio_saturates_instead_of_dividing_by_zero(self):
        # At very low quality every block interior quantises to one value, which
        # is a division by nothing, not an infinitely blocky frame.
        out = fq.compression_metrics(jpeg_roundtrip(self.clean, 2))
        self.assertTrue(out["ratio_saturated"])
        self.assertTrue(np.isfinite(out["block_boundary_ratio"]))

    def test_smooth_scene_content_is_not_penalised_as_quantisation(self):
        # Real footage is full of large flat areas - painted walls, ceiling, sky -
        # which produce a high flat-block fraction with no encoder involved. The
        # dead-zone cue therefore only discounts a frame the lattice already
        # implicates; on one real 720p clip 3 of 8 sampled frames crossed the
        # fraction and only 1 crossed the boundary ratio.
        panels = (np.rint(smooth_scene().astype(np.float64) / 70.0) * 70).astype(np.uint8)
        out = fq.compression_metrics(panels)
        self.assertGreaterEqual(out["flat_block_fraction"], fq.QUANTISED_FLAT_MIN)
        self.assertFalse(out["blocking_suspect"])
        frame = fq.frame_quality(panels)
        self.assertNotIn("quantised_flat_regions", frame["reasons"])
        self.assertEqual(frame["weight"], 1.0, frame)

    def test_a_blurred_frame_is_not_mislabelled_as_compression(self):
        # The two failure modes must stay distinguishable: defocus smooths
        # *across* the 8-pixel grid, so the boundary ratio falls towards 1
        # instead of rising above it.
        blurred = fq.compression_metrics(gaussian_defocus(self.clean, 4.0))
        smeared = fq.compression_metrics(motion_blur(self.clean, 15))
        heavy = fq.compression_metrics(self.heavy)
        for name, out in (("defocus", blurred), ("motion", smeared)):
            with self.subTest(name=name):
                self.assertLess(out["block_boundary_ratio"], heavy["block_boundary_ratio"])
                self.assertFalse(out["blocking_suspect"])
                self.assertGreater(out["score"], heavy["score"])


class FrameQualityTests(unittest.TestCase):
    def test_clean_frame_gets_a_high_weight_and_no_reasons(self):
        out = fq.frame_quality(checkerboard())
        self.assertTrue(0.0 <= out["weight"] <= 1.0)
        self.assertGreater(out["weight"], 0.85, out)
        self.assertEqual(out["reasons"], [])

    def test_degraded_frames_score_below_the_clean_one(self):
        sharp = checkerboard()
        reference = fq.blur_metrics(sharp)
        floor = 0.5 * reference["laplacian_variance"]
        base = fq.frame_quality(sharp, energy_floor=floor)["weight"]
        # Compression is judged on the low-contrast scene, because that is where
        # blocking is visible at all: a hard-edged checkerboard hides the lattice
        # behind its own edges and would flatter the metric.
        smooth = smooth_scene()
        smooth_reference = fq.blur_metrics(smooth)
        cases = {
            "defocus": (gaussian_defocus(sharp, 3.0), floor, reference),
            "motion": (motion_blur(sharp, 17), floor, reference),
            "jpeg": (jpeg_roundtrip(smooth, 15),
                     0.5 * smooth_reference["laplacian_variance"], smooth_reference),
            "flat": (np.full((128, 128), 128, np.uint8), floor, reference),
        }
        for name, (gray, case_floor, case_reference) in cases.items():
            out = fq.frame_quality(gray, energy_floor=case_floor,
                                   centroid_floor=case_reference["spectral_centroid"])
            with self.subTest(name=name):
                self.assertLess(out["weight"], base, out)
                self.assertLess(out["weight"], 0.85, out)
                self.assertTrue(out["reasons"], out)

    def test_sharpness_penalty_is_graded_not_a_step_once_past_the_floor(self):
        # Two frames below the floor must still rank: the variance spans orders
        # of magnitude, so a linear ramp saturates after one halving.
        sharp = checkerboard()
        floor = 0.5 * fq.blur_metrics(sharp)["laplacian_variance"]
        weights = [fq.frame_quality(motion_blur(sharp, length), energy_floor=floor)["weight"]
                   for length in (3, 7, 11, 15, 17)]
        for first, second in zip(weights, weights[1:]):
            self.assertLess(second, first, weights)

    def test_the_two_verdicts_stay_in_separate_lanes(self):
        # A soft frame must not be reported as a compressed one, or the reverse:
        # challenge (ii) names two failures and the whole point of the second
        # axis is that they want opposite handling downstream.
        sharp = checkerboard()
        reference = fq.blur_metrics(sharp)
        kwargs = {"energy_floor": 0.5 * reference["laplacian_variance"],
                  "centroid_floor": reference["spectral_centroid"]}
        blurred = fq.frame_quality(gaussian_defocus(sharp, 3.0), **kwargs)
        compressed = fq.frame_quality(jpeg_roundtrip(sharp, 8), **kwargs)
        self.assertIn("defocus_blur_suspected", blurred["reasons"])
        self.assertNotIn("blocking_artifacts", blurred["reasons"])
        self.assertIn("blocking_artifacts", compressed["reasons"])
        self.assertNotIn("defocus_blur_suspected", compressed["reasons"])
        self.assertEqual(compressed["blur_label"], "sharp")

    def test_compression_that_empties_the_high_band_is_not_called_blur(self):
        # Heavy quantisation also destroys high frequencies, so without an
        # explicit arbitration the blur axis would claim these frames too.
        smooth = smooth_scene()
        reference = fq.blur_metrics(smooth)
        kwargs = {"energy_floor": 0.5 * reference["laplacian_variance"],
                  "centroid_floor": reference["spectral_centroid"]}
        for quality in (25, 15, 8):
            out = fq.frame_quality(jpeg_roundtrip(smooth, quality), **kwargs)
            with self.subTest(quality=quality):
                self.assertIn("blocking_artifacts", out["reasons"])
                self.assertNotIn("motion_blur_suspected", out["reasons"])
                self.assertNotIn("defocus_blur_suspected", out["reasons"])
                self.assertEqual(out["blur_label"], "compression_artifact")

    def test_weight_never_rises_as_motion_blur_grows(self):
        sharp = checkerboard()
        weights = [fq.frame_quality(motion_blur(sharp, length),
                                    energy_floor=0.5 * fq.blur_metrics(sharp)["laplacian_variance"])["weight"]
                   for length in (3, 9, 21, 41)]
        for first, second in zip(weights, weights[1:]):
            self.assertLessEqual(second, first, weights)
        self.assertLess(weights[-1], weights[0], weights)
        # Past three octaves under the floor the weight stops falling: those
        # frames are already below what extract_keyframes keeps, so ranking them
        # further would be precision the statistics do not have.
        self.assertAlmostEqual(weights[-1], fq.WEIGHT_MIN_FOR_BLUR * fq.WEIGHT_MIN_FOR_MOTION,
                               places=9)

    def test_previous_gray_adds_temporal_reasons(self):
        sharp = checkerboard()
        dupes = fq.frame_quality(sharp, previous_gray=sharp.copy())
        self.assertIn("near_duplicate_previous", dupes["reasons"])
        self.assertLess(dupes["weight"], fq.frame_quality(sharp)["weight"])

        shifted = np.roll(sharp, 90, axis=1)
        moving = fq.frame_quality(shifted, previous_gray=sharp)
        self.assertIn("large_frame_to_frame_change", moving["reasons"])
        self.assertNotIn("near_duplicate_previous", moving["reasons"])

    def test_one_directional_scene_is_not_called_motion_blur(self):
        # Anisotropy is content-relative: a scene whose gradients run one way -
        # a corrugated roof, horizontal siding - scores high while perfectly
        # sharp. Without an energy floor the answer is "directional texture", not
        # a blur verdict.
        content = grating(period=26)
        blur = fq.blur_metrics(content)
        self.assertTrue(blur["directional"], blur["anisotropy"])
        out = fq.frame_quality(content)
        self.assertEqual(out["blur_label"], "directional")
        self.assertEqual(out["reasons"], ["directional_texture"])
        self.assertEqual(out["weight"], 1.0, out)
        # Give it a floor and the same frame is still sharp: high energy survives.
        self.assertEqual(fq.frame_quality(content, energy_floor=0.5 * blur["laplacian_variance"])
                         ["blur_label"], "sharp")

    def test_eight_periodic_detail_reads_as_blocking_and_that_is_named(self):
        # A documented false positive, asserted rather than hidden: real content
        # whose period is a whole number of blocks puts every edge on the lattice,
        # which is indistinguishable from encoder blocking in one grey frame. The
        # same grating at period 26 reads clean.
        aligned = fq.compression_metrics(grating(period=24))
        off_lattice = fq.compression_metrics(grating(period=26))
        self.assertTrue(aligned["blocking_suspect"])
        self.assertFalse(off_lattice["blocking_suspect"])
        self.assertGreater(aligned["grid_excess"], off_lattice["grid_excess"])
        frame = fq.frame_quality(grating(period=24))
        self.assertIn("blocking_artifacts", frame["reasons"])
        self.assertNotIn("motion_blur_suspected", frame["reasons"])

    def test_unrelated_previous_frame_adds_no_temporal_reason(self):
        sharp = checkerboard()
        other = checkerboard(seed=99)
        out = fq.frame_quality(sharp, previous_gray=other)
        self.assertNotIn("near_duplicate_previous", out["reasons"])

    def test_result_carries_the_underlying_signals(self):
        gray = checkerboard()
        out = fq.frame_quality(gray)
        self.assertEqual(out["blur"], fq.blur_metrics(gray))
        self.assertEqual(out["compression"], fq.compression_metrics(gray))
        self.assertEqual(out["sharpness"], out["blur"]["laplacian_variance"])
        self.assertEqual(out["blur_label"], fq.blur_label(out["blur"]))


class WeightsForManifestTests(unittest.TestCase):
    ROWS = [
        {"file": "clip/00001.jpg", "clip": "clip", "frame_index": 6,
         "t_clip": 0.2, "t_sec": 0.2, "sharpness": 283.9},
        {"file": "clip/00002.jpg", "clip": "clip", "frame_index": 8,
         "t_clip": 0.267, "t_sec": 0.267, "sharpness": 226.3},
        {"file": "clip/00003.jpg", "clip": "clip", "frame_index": 18,
         "t_clip": 0.6, "t_sec": 0.6, "sharpness": 12.0},
    ]

    def test_manifest_is_not_mutated(self):
        rows = copy.deepcopy(self.ROWS)
        fq.weights_for_manifest(rows)
        self.assertEqual(rows, copy.deepcopy(self.ROWS))

    def test_returns_one_weight_row_per_input_row(self):
        out = fq.weights_for_manifest(self.ROWS)
        self.assertEqual(len(out), len(self.ROWS))
        self.assertEqual([r["file"] for r in out], [r["file"] for r in self.ROWS])
        for row in out:
            self.assertTrue(0.0 <= row["weight"] <= 1.0, row)
            self.assertIsInstance(row["reasons"], list)

    def test_sharpness_is_calibrated_against_the_manifest_not_an_absolute(self):
        # extract_keyframes.py derives its blur floor from the corpus percentiles;
        # a fixed absolute threshold would punish a whole low-texture clip and
        # call it measured.
        out = fq.weights_for_manifest(self.ROWS)
        self.assertGreater(out[0]["weight"], out[2]["weight"])
        self.assertLess(out[2]["weight"], 0.6, out[2])

    def test_rows_carrying_full_quality_are_used_verbatim(self):
        quality = fq.frame_quality(gaussian_defocus(checkerboard(), 3.0))
        rows = [{"file": "a.jpg", "sharpness": 50.0, "quality": quality}]
        out = fq.weights_for_manifest(rows)
        self.assertAlmostEqual(out[0]["weight"], quality["weight"], places=6)
        self.assertEqual(out[0]["reasons"], quality["reasons"])

    def test_rows_without_any_signal_are_reported_not_invented(self):
        out = fq.weights_for_manifest([{"file": "preextracted/00001.jpg"}])
        self.assertIsNone(out[0]["weight"])
        self.assertIn("no_quality_metrics", out[0]["reasons"])

    def test_empty_or_malformed_manifest_raises(self):
        with self.assertRaises(ValueError):
            fq.weights_for_manifest([])
        with self.assertRaises(ValueError):
            fq.weights_for_manifest([{"clip": "no file key"}])


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.gray = checkerboard()

    def test_rejects_empty_non_finite_and_multi_channel_input(self):
        bad = {
            "empty": np.zeros((0, 0), np.uint8),
            "nan": np.full((64, 64), np.nan),
            "inf": np.full((64, 64), np.inf),
            "colour": np.zeros((64, 64, 3), np.uint8),
            "volume": np.zeros((4, 64, 64), np.uint8),
        }
        for name, value in bad.items():
            for func in (fq.blur_metrics, fq.compression_metrics, fq.frame_quality):
                with self.subTest(case=name, func=func.__name__), self.assertRaises(ValueError):
                    func(value)

    def test_previous_gray_is_validated_too(self):
        with self.assertRaises(ValueError):
            fq.frame_quality(self.gray, previous_gray=np.full((64, 64), np.nan))
        with self.assertRaises(ValueError):
            fq.frame_quality(self.gray, previous_gray=np.zeros((64, 64, 3), np.uint8))

    def test_a_mismatched_previous_frame_is_refused(self):
        with self.assertRaises(ValueError):
            fq.frame_quality(self.gray, previous_gray=np.zeros((64, 64), np.uint8))

    def test_float_input_is_accepted(self):
        out = fq.blur_metrics(self.gray.astype(np.float64) / 255.0)
        self.assertTrue(np.isfinite(out["laplacian_variance"]))

    def test_tiny_frames_are_refused(self):
        with self.assertRaises(ValueError):
            fq.blur_metrics(np.zeros((4, 4), np.uint8))


if __name__ == "__main__":
    unittest.main()
