"""Tests for the photometry proxies: exposure, illumination field, shadows, NCC.

Synthetic frames with *known* degradations are the ground truth here. Two of the
cases below exist specifically to record what these methods cannot do: a
low-frequency estimator cannot tell shading from a dark material, and normalised
cross-correlation is blind to a tone curve by construction. A suite that only
tested the recoverable case would overstate the module.
"""
import unittest

import cv2
import numpy as np

from scripts import survey_photometry as ph


def texture(size=256, seed=5):
    """Stationary mid-frequency texture: smoothed noise.

    Stationary matters. Block medians can only isolate illumination if the scene
    reflectance does not itself swing on that scale, so the recovery tests use
    this and ``blobby_texture`` is there to show what happens when it does.
    """
    rng = np.random.default_rng(seed)
    return np.clip(60 + 120 * cv2.GaussianBlur(rng.random((size, size)), (0, 0), 1.0),
                   0, 255).astype(np.uint8)


def blobby_texture(size=256, seed=3):
    """Non-stationary content: large bright and dark patches of its own."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(220):
        cx, cy = rng.integers(0, size, 2)
        radius = rng.integers(6, 22)
        canvas[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = rng.random()
    smooth = cv2.GaussianBlur(canvas, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    return np.clip(40 + 170 * smooth, 0, 255).astype(np.uint8)


def shading_field(shape, low=0.45, high=0.95, cells=4, seed=11):
    """A band-limited darkening field that stays at or below 1.0.

    Two deliberate constraints. Never above unity, because a multiplicative
    brightening would clip against the 255 rail and the frame would no longer be
    the product the estimator is being asked to invert. And low-pass filtered
    well past the cell grid, because raw cubic upsampling of a 4x4 grid carries
    ringing that is not low frequency at all.
    """
    rng = np.random.default_rng(seed)
    grid = rng.random((cells, cells))
    big = cv2.resize(grid.astype(np.float32), (shape[1], shape[0]),
                     interpolation=cv2.INTER_CUBIC).astype(np.float64)
    big = cv2.GaussianBlur(big, (0, 0), max(shape) / 8.0)
    big = (big - big.min()) / (big.max() - big.min())
    return low + (high - low) * big


def apply_shading(gray, field):
    return np.clip(gray.astype(np.float64) * field, 0, 255).astype(np.uint8)


class ExposureStatsTests(unittest.TestCase):
    def test_reports_mean_histogram_and_clipping(self):
        gray = np.full((64, 64), 100, np.uint8)
        out = ph.exposure_stats(gray)
        self.assertAlmostEqual(out["mean"], 100.0, places=9)
        self.assertEqual(int(out["histogram"][100]), 64 * 64)
        self.assertEqual(sum(int(v) for v in out["histogram"]), 64 * 64)
        self.assertEqual(out["clipped_fraction"], 0.0)
        self.assertEqual(out["std"], 0.0)

    def test_clipped_fractions_count_both_rails(self):
        gray = np.full((64, 64), 128, np.uint8)
        gray[0:8] = 0                          # 8 rows of 64 = 512 of 4096 pixels
        gray[8:16] = 255
        out = ph.exposure_stats(gray)
        self.assertAlmostEqual(out["clipped_low_fraction"], 512 / 4096, places=9)
        self.assertAlmostEqual(out["clipped_high_fraction"], 512 / 4096, places=9)
        self.assertAlmostEqual(out["clipped_fraction"], 1024 / 4096, places=9)

    def test_dynamic_range_is_the_percentile_spread(self):
        gray = texture()
        out = ph.exposure_stats(gray)
        expected = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        self.assertAlmostEqual(out["dynamic_range"], expected, places=9)
        self.assertGreater(out["dynamic_range"], 20.0)


class NormaliseExposureTests(unittest.TestCase):
    def test_dark_frame_is_brightened_by_the_stated_gain(self):
        dark = (texture().astype(np.float64) * 0.35).astype(np.uint8)
        result = ph.normalise_exposure(dark, 120.0)
        self.assertGreater(result.gain, 1.0)
        self.assertAlmostEqual(float(result.image.mean()), 120.0, delta=6.0)
        # The tuple contract: the first two positions are image and gain.
        self.assertIs(result[0], result.image)
        self.assertEqual(result[1], result.gain)

    def test_target_defaults_to_a_mid_grey(self):
        image = ph.normalise_exposure(texture())[0]
        self.assertTrue(60.0 < float(image.mean()) < 200.0)

    def test_brightening_cannot_silently_saturate(self):
        # A frame mostly in shadow but carrying a small bright patch - a lit roof
        # tile in a shadowed street - is the real case. The gain that would reach
        # a bright target would spend that patch on the rail, so it stops early
        # and the shortfall is reported instead of absorbed.
        gray = (texture().astype(np.float64) * 0.3).astype(np.uint8)
        gray[0:24, 0:80] = 210                 # ~3% of pixels already bright
        result = ph.normalise_exposure(gray, 200.0)
        self.assertGreater(result.requested_gain, 4.0, result.requested_gain)
        self.assertLess(result.gain, result.requested_gain)
        self.assertLess(ph.exposure_stats(result.image)["clipped_high_fraction"], 0.02)
        self.assertLess(float(result.image.mean()), 200.0)
        self.assertEqual(result.clipped_fraction,
                         ph.exposure_stats(result.image)["clipped_fraction"])

    def test_the_cap_does_not_fire_when_there_is_headroom(self):
        gray = (texture().astype(np.float64) * 0.3).astype(np.uint8)
        result = ph.normalise_exposure(gray, 150.0)
        self.assertAlmostEqual(result.gain, result.requested_gain, places=9)
        self.assertAlmostEqual(float(result.image.mean()), 150.0, delta=6.0)

    def test_gain_can_also_darken(self):
        bright = np.clip(texture().astype(np.float64) * 1.6, 0, 255).astype(np.uint8)
        result = ph.normalise_exposure(bright, 90.0)
        self.assertLess(result.gain, 1.0)
        self.assertAlmostEqual(float(result.image.mean()), 90.0, delta=6.0)

    def test_output_is_uint8_and_shape_preserving(self):
        gray = texture()
        result = ph.normalise_exposure(gray, 130.0)
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertEqual(result.image.shape, gray.shape)

    def test_a_black_frame_has_no_exposure_to_normalise(self):
        with self.assertRaises(ValueError):
            ph.normalise_exposure(np.zeros((64, 64), np.uint8), 128.0)


class IlluminationFieldTests(unittest.TestCase):
    def setUp(self):
        self.base = texture(192)
        self.field = shading_field(self.base.shape)
        self.shaded = apply_shading(self.base, self.field)

    def test_field_is_smooth_positive_and_full_size(self):
        out = ph.illumination_field(self.shaded, sigma_cells=1.0)
        self.assertEqual(out.shape, self.shaded.shape)
        self.assertTrue(np.all(out > 0))
        self.assertTrue(np.isfinite(out).all())

    def test_field_recovers_a_stationary_scene_shading(self):
        out = ph.illumination_field(self.shaded, sigma_cells=1.0)
        truth = self.field / self.field.mean()
        estimate = out / out.mean()
        self.assertGreater(float(np.corrcoef(truth.ravel(), estimate.ravel())[0, 1]), 0.95)

    def test_the_field_cannot_separate_shading_from_scene_albedo(self):
        # Documented limitation, asserted so nobody quietly relies on it: with
        # large dark patches of its own, the scene's reflectance is itself a
        # low-frequency signal and the estimator attributes it to illumination.
        blobby = blobby_texture(192)
        shaded = apply_shading(blobby, self.field)
        out = ph.illumination_field(shaded, sigma_cells=1.0)
        estimate = out / out.mean()
        self.assertLess(float(np.corrcoef((self.field / self.field.mean()).ravel(),
                                          estimate.ravel())[0, 1]), 0.9)

    def test_a_uniform_frame_gets_a_uniform_field(self):
        even = ph.illumination_field(np.full((128, 128), 120, np.uint8), sigma_cells=2.0)
        self.assertTrue(np.allclose(even, 120.0, atol=1e-3))

    def test_shading_adds_low_frequency_spread_the_unshaded_frame_lacks(self):
        bare = ph.illumination_field(self.base, sigma_cells=1.0)
        shaded = ph.illumination_field(self.shaded, sigma_cells=1.0)
        self.assertGreater(float(np.std(shaded / shaded.mean())),
                           float(np.std(bare / bare.mean())))

    def test_a_larger_sigma_gives_a_smoother_field(self):
        def roughness(array):
            gy, gx = np.gradient(array)
            return float(np.mean(gx ** 2 + gy ** 2))

        narrow = ph.illumination_field(self.shaded, sigma_cells=0.5)
        wide = ph.illumination_field(self.shaded, sigma_cells=4.0)
        self.assertLess(roughness(wide), roughness(narrow))


class FlattenTests(unittest.TestCase):
    def setUp(self):
        self.base = texture(192)
        self.field = shading_field(self.base.shape)
        self.shaded = apply_shading(self.base, self.field)

    def test_flattening_removes_most_of_the_low_frequency_drift(self):
        before = float(np.std(ph.illumination_field(self.shaded, sigma_cells=1.0)))
        result = ph.flatten(self.shaded)
        after = float(np.std(ph.illumination_field(result.image, sigma_cells=1.0)))
        self.assertLess(after, 0.6 * before)
        self.assertGreater(after, 0.0)

    def test_global_mean_brightness_is_preserved(self):
        result = ph.flatten(self.shaded)
        self.assertLess(abs(float(result.image.mean()) - float(self.shaded.mean())),
                        0.08 * float(self.shaded.mean()))

    def test_flatten_returns_the_field_it_used(self):
        result = ph.flatten(self.shaded)
        image, field = result
        self.assertEqual(image.shape, field.shape)
        self.assertTrue(np.all(field > 0))

    def test_local_texture_survives_flattening(self):
        result = ph.flatten(self.shaded)
        self.assertGreater(ph.exposure_stats(result.image)["dynamic_range"], 20.0)

    def test_the_correction_is_capped_so_shadows_do_not_amplify_noise(self):
        # A deep, narrow shadow is a division by a small number: without the cap
        # flattening would hand downstream a block of amplified grain.
        dark = apply_shading(self.base, np.ones(self.base.shape))
        hole = np.ones(self.base.shape)
        hole[64:128, 64:128] = 0.02
        carved = apply_shading(dark, hole)
        image, field = ph.flatten(carved, max_correction=1.5)
        inside = image[80:112, 80:112]
        self.assertLessEqual(float(inside.max()),
                             float(carved[80:112, 80:112].max()) * 1.5 + 1.0)
        self.assertGreater(float(field.min()), 0.0)


class ShadowSuspectTests(unittest.TestCase):
    def setUp(self):
        self.base = texture(256)
        self.field = shading_field(self.base.shape, low=0.30, high=0.98, seed=5)
        self.shaded = apply_shading(self.base, self.field)

    def test_returns_a_conservative_mask_and_leaves_the_image_alone(self):
        out = ph.shadow_suspects(self.shaded)
        self.assertEqual(out["mask"].shape, self.shaded.shape)
        self.assertEqual(out["mask"].dtype, bool)
        self.assertLess(out["fraction"], 0.6)
        self.assertLess(out["pixels"], out["mask"].size)
        self.assertLess(out["mean_level_in_suspects"], out["mean_level_outside"])

    def test_a_shadow_keeps_local_contrast_and_is_flagged(self):
        yy, xx = np.mgrid[0:256, 0:256]
        inside = (xx - 128) ** 2 + (yy - 128) ** 2 <= 60 ** 2
        blob = np.ones((256, 256))
        blob[inside] = 0.32                      # multiplicative darkening
        shadowed = np.clip(self.base.astype(np.float64) * blob, 0, 255).astype(np.uint8)
        mask = ph.shadow_suspects(shadowed)["mask"]
        # The default catches the core of the disk, not its penumbra: 0.55 of the
        # shadow area at zero false positives outside it. A hint that names half a
        # shadow is worth more than one that names all of it and every dark wall.
        self.assertGreater(float(mask[inside].mean()), 0.45, float(mask.mean()))
        self.assertLess(float(mask[~inside].mean()), 0.02)

    def test_a_genuinely_dark_textureless_patch_is_not_flagged(self):
        panel = self.base.copy()
        panel[80:170, 80:170] = 18               # dark, flat: a black object, not a shadow
        mask = ph.shadow_suspects(panel)["mask"]
        self.assertLess(float(mask[95:155, 95:155].mean()), 0.2)

    def test_an_evenly_underexposed_frame_is_not_called_shadowy(self):
        dark = (self.base.astype(np.float64) * 0.35).astype(np.uint8)
        out = ph.shadow_suspects(dark)
        self.assertLess(out["fraction"], 0.15, out)

    def test_the_mask_is_reported_as_a_hint_not_a_repair(self):
        out = ph.shadow_suspects(self.shaded)
        self.assertEqual(out["pixels"], int(out["mask"].sum()))
        self.assertIn("not shadow removal", out["use"])


class PhotometricConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.base = texture(256, seed=8)
        self.field = shading_field(self.base.shape, low=0.4, high=0.95, seed=21)
        self.shaded = apply_shading(self.base, self.field)
        self.gamma = np.clip(255.0 * (self.base / 255.0) ** 0.55, 0, 255).astype(np.uint8)
        self.gamma_shaded = apply_shading(self.gamma, self.field)
        self.near_same = cv2.GaussianBlur(self.base, (3, 3), 0)

    def _ncc(self, others, flatten_first):
        return ph.photometric_consistency(self.base, others,
                                          flatten_first=flatten_first)

    def test_identical_illumination_scores_above_a_shaded_frame(self):
        out = self._ncc([self.near_same, self.shaded], False)
        self.assertGreater(out["ncc"][0], 0.95)
        self.assertGreater(out["ncc"][0], out["ncc"][1])

    def test_flattening_recovers_most_of_the_lost_consistency(self):
        raw = self._ncc([self.shaded], False)["median"]
        flat = self._ncc([self.shaded], True)["median"]
        self.assertGreater(flat, raw)
        # The ceiling is 1.0 - NCC of a frame against itself - so recovery is the
        # fraction of the gap to perfect agreement.
        self.assertGreater((flat - raw) / (1.0 - raw), 0.6, (raw, flat))

    def test_flattening_does_not_recover_a_tone_curve(self):
        # Honest limit: flattening divides out a low-frequency *multiplicative*
        # field. Shading plus a gamma curve flattens back to less than shading
        # alone, which is the difference those two effects actually make.
        both = self._ncc([self.gamma_shaded], True)["median"]
        shading_only = self._ncc([self.shaded], True)["median"]
        self.assertLess(both, shading_only)
        self.assertLess(both, 0.98)

    def test_a_pure_gain_and_a_pure_gamma_are_invisible_to_ncc(self):
        # Not a bug to fix but a property to know: NCC is invariant to a·x+b, and
        # a smooth monotone curve is locally near-linear. Anyone reading a high
        # consistency score as "the exposure was stable" is reading the wrong
        # thing; use exposure_stats' mean and clipped fraction for that.
        gain_shifted = np.clip(self.base.astype(np.float64) * 0.8 + 6, 0, 255).astype(np.uint8)
        raw = self._ncc([gain_shifted, self.gamma], False)["ncc"]
        self.assertGreater(min(raw), 0.99, raw)

    def test_explicit_patches_are_the_ones_used(self):
        patches = np.array([[10, 10], [40, 60], [100, 120]])
        out = ph.photometric_consistency(self.base, [self.near_same], patches=patches)
        self.assertEqual(out["patches_used"], 3)
        out2 = ph.photometric_consistency(self.base, [self.near_same], patches=patches,
                                          patch_size=16)
        self.assertEqual(out2["patches_used"], 3)
        self.assertNotEqual(out["ncc"], out2["ncc"])

    def test_summary_statistics_and_per_frame_rows(self):
        out = self._ncc([self.near_same, self.shaded, self.gamma_shaded], True)
        self.assertEqual(len(out["ncc"]), 3)
        self.assertAlmostEqual(out["median"], float(np.median(out["ncc"])), places=9)
        self.assertAlmostEqual(out["min"], min(out["ncc"]), places=9)
        self.assertAlmostEqual(out["max"], max(out["ncc"]), places=9)
        self.assertTrue(out["flattened"])

    def test_a_zero_variance_frame_scores_zero_rather_than_one(self):
        blank = np.zeros((128, 128), np.uint8)
        out = ph.photometric_consistency(blank, [blank.copy()], flatten_first=False)
        self.assertEqual(out["ncc"], [0.0])
        self.assertIn("zero_variance_patch", out["notes"])


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.gray = texture(128)

    def _call(self, func, value):
        if func is ph.illumination_field:
            return func(value, sigma_cells=1.0)
        return func(value)

    def test_every_public_function_rejects_empty_and_non_finite_input(self):
        bad = {
            "empty": np.zeros((0, 0), np.uint8),
            "nan": np.full((64, 64), np.nan),
            "inf": np.full((64, 64), np.inf),
            "colour": np.zeros((64, 64, 3), np.uint8),
            "volume": np.zeros((4, 64, 64), np.uint8),
            "too_small": np.zeros((8, 8), np.uint8),
            "out_of_range": np.full((64, 64), 900.0),
            "normalised_float": np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64),
        }
        for name, value in bad.items():
            for func in (ph.exposure_stats, ph.normalise_exposure, ph.flatten,
                         ph.shadow_suspects, ph.illumination_field,
                         ph.photometric_consistency):
                with self.subTest(case=name, func=func.__name__):
                    with self.assertRaises(ValueError):
                        self._call(func, value) if func is not ph.photometric_consistency \
                            else ph.photometric_consistency(value, [self.gray])

    def test_validation_reaches_the_compared_frames_too(self):
        with self.assertRaises(ValueError):
            ph.photometric_consistency(self.gray, [np.full((128, 128), np.nan)])
        with self.assertRaises(ValueError):
            ph.photometric_consistency(np.full((128, 128), np.nan), [self.gray])
        with self.assertRaises(ValueError):
            ph.photometric_consistency(self.gray, [])
        with self.assertRaises(ValueError):
            ph.photometric_consistency(self.gray, [np.zeros((64, 64), np.uint8)])

    def test_illumination_sigma_must_be_positive_and_finite(self):
        for sigma in (0.0, -1.0, float("nan")):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                ph.illumination_field(self.gray, sigma_cells=sigma)

    def test_target_mean_must_be_a_reachable_grey_level(self):
        for target in (-5.0, 300.0, float("nan")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                ph.normalise_exposure(self.gray, target)

    def test_patches_must_land_inside_the_frame(self):
        with self.assertRaises(ValueError):
            ph.photometric_consistency(self.gray, [self.gray],
                                       patches=np.array([[500, 500]]))
        with self.assertRaises(ValueError):
            ph.photometric_consistency(self.gray, [self.gray], patches=np.array([[1]]))

    def test_shadow_thresholds_are_validated(self):
        with self.assertRaises(ValueError):
            ph.shadow_suspects(self.gray, dark_ratio=1.5)
        with self.assertRaises(ValueError):
            ph.shadow_suspects(self.gray, contrast_ratio=0.0)


if __name__ == "__main__":
    unittest.main()
