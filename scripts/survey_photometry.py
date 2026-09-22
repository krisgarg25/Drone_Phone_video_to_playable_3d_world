"""Challenge (iii): variable illumination and shadows, CPU, no reference frame.

A drone crossing a field while the sun moves, or rolling into a building's
shadow, hands SfM and Gaussian-splat training a sequence whose brightness is not
a property of the scene. Two things break. Feature matching degrades where the
gain drops the local contrast under the noise floor, and a splat optimiser will
 happily bake the shading into albedo, so the model looks right from the shots it
 was trained on and wrong from anywhere else.

The standard, boring mitigation is the one implemented here: estimate the
*low-frequency* illumination and divide it out before anything downstream looks
at the frame. On top of that this module supplies the two numbers the pipeline
currently lacks:

* ``exposure_stats`` / ``normalise_exposure`` - how bright is this frame, how
  much of it is already clipped, and what gain would fix it without the
  brightening step quietly saturating highlights on the way past the target.
* ``illumination_field`` / ``flatten`` - the smooth shading estimate and its
  removal.
* ``shadow_suspects`` - where the frame is dark *and* still has local contrast,
  which is what a shadow looks like in a single grey image.
* ``photometric_consistency`` - normalised cross-correlation on sampled patches
  between a reference frame and the rest, so "the exposure drifted across this
  pass" becomes a measured number instead of an assumption.

WHAT IS MEASURED AND WHAT IS PROXIED, precisely:
  measured  - mean, histogram, clipped-pixel fractions, dynamic range, applied
              gain: exact functions of the pixel array.
  measured  - NCC between two given patches: exactly the textbook statistic.
  proxied   - the illumination *field*: a block-median estimate cannot tell a
              shadow edge from a material edge, so ``flatten`` removes both.
              Flattening a frame with real albedo boundaries dulls them.
  proxied   - ``shadow_suspects``: a mask of regions that behave like shadows.
              It is a down-weighting hint for frame selection, NOT shadow
              removal - recovering a shadow means inverting a per-pixel
              reflectance x illumination product, which needs either colour
              (shadows shift chromaticity far less than illumination does, and
              this module only sees one channel) or multiple views of the same
              surface. Do not present this mask as corrected imagery.
  not attempted - radiometric calibration, EXIF gain/shutter recovery, HDR
              merging, any per-pixel shadow removal.

Calibration status, measured rather than assumed: the field estimator recovers a
constructed band-limited shading field on stationary texture at correlation 0.998
and drops to 0.50 on non-stationary content, where the scene's own albedo is
itself a low-frequency signal - so ``flatten`` removes shading cleanly only where
the surface it lands on is evenly reflective. Eight frames sampled from one real
indoor 487-frame clip flagged 4.7-39.5% of their area as shadow-suspect (median
17.8%) with no shadow ground truth to arbitrate against, while the mean level
inside the suspect set was 45 against 83 outside: the mask separates *something*,
and calling it a validated shadow detector would be a claim this data does not
support. Every floor here is a knob to set per corpus, not a constant to trust.

Cost on this machine, 720x960 grey: exposure_stats ~21 ms, flatten ~26 ms,
shadow_suspects ~50 ms, photometric_consistency ~58 ms per compared frame - almost
all of that is the flattening pass it runs first, so skip it (``flatten_first=False``)
when only relative ranking matters. CPU only (numpy + cv2), no scipy, no GPU, no I/O.
"""
from __future__ import annotations

from collections import namedtuple

import cv2
import numpy as np

__all__ = ["exposure_stats", "normalise_exposure", "illumination_field", "flatten",
           "shadow_suspects", "photometric_consistency"]

# Frames smaller than this cannot carry a block grid plus a smoothing radius, so
# the "field" would be a couple of pixels stretched over the frame.
MIN_SIDE = 32
MID_GREY_TARGET = 128.0
# A gain is only useful if it does not spend the highlights to buy mid-tones.
DEFAULT_MAX_CLIPPED_FRACTION = 0.02
DEFAULT_MAX_GAIN = 8.0
# Kept a full grey level off each rail. "Clipped" is decided after rounding, so
# a gain that lands a pixel on exactly 254.5 would round onto the rail and be
# counted, and the budget would be spent by ties rather than by real highlights.
RAIL_MARGIN = 1.0
# Shadow rules. A cast shadow is a large multiplicative drop in illuminance - an
# outdoor one is typically 2-4x, so "at least 38% below the in-the-light level"
# is a conservative reading of it - and it leaves the *relative* local contrast
# nearly intact, which is the only thing that separates it from a dark material
# in a single grey frame. Loosening either number raises detection and real
# footage false positives together; there is no setting that does both, because
# a large dark surface and a shadow are the same pixel pattern.
SHADOW_DARK_RATIO = 0.62
SHADOW_CONTRAST_RATIO = 0.70
SHADOW_MIN_AREA_FRACTION = 0.002


Normalised = namedtuple("Normalised", "image gain requested_gain clipped_fraction")
Normalised.__doc__ = """``(image, gain)`` plus ``requested_gain`` and ``clipped_fraction``.

A namedtuple so ``img, gain = normalise_exposure(g)`` works while the honesty
annotations stay reachable as attributes.
"""

Flattened = namedtuple("Flattened", "image field")
Flattened.__doc__ = "``(image, field)`` - the frame with the shading divided back out."


def _as_gray(image, name: str = "gray") -> np.ndarray:
    """Validate one single-channel frame and return it in float64 grey levels.

    Every public function in this module runs through here, so an empty frame, a
    NaN/inf pixel, a colour triple or a mislabelled 0..1 float fails at the
    boundary with the reason in the message instead of becoming a plausible
    number two steps later.
    """
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D single-channel frame, got "
                         f"{array.ndim}-D with shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if min(array.shape) < MIN_SIDE:
        raise ValueError(f"{name} must be at least {MIN_SIDE}x{MIN_SIDE} pixels, "
                         f"got {array.shape[0]}x{array.shape[1]}")
    values = array.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite pixels (NaN or inf): "
                         f"{int((~np.isfinite(values)).sum())} of them")
    low, high = float(values.min()), float(values.max())
    if high > 255.0 or low < 0.0:
        raise ValueError(f"{name} is outside the 0..255 grey range "
                         f"(min {low}, max {high})")
    if array.dtype.kind == "f" and 0.0 < high <= 1.0:
        raise ValueError(f"{name} looks normalised to 0..1 (max {high}); this "
                         f"module works in 0..255 grey levels")
    return values


def _gaussian1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _separable_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian: a vertical then a horizontal 1-D pass, reflect at the
    border. Done with ``cv2.filter2D`` and explicit 1-D kernels rather than
    ``sepFilter2D`` because the coarse grid is only a handful of cells wide and
    the two-pass form keeps the border behaviour visible."""
    kernel = _gaussian1d(sigma).astype(np.float32)
    radius = kernel.size // 2
    work = array.astype(np.float32)
    work = cv2.filter2D(work, cv2.CV_32F, kernel[:, None], anchor=(0, radius),
                        borderType=cv2.BORDER_REFLECT)
    work = cv2.filter2D(work, cv2.CV_32F, kernel[None, :], anchor=(radius, 0),
                        borderType=cv2.BORDER_REFLECT)
    return work.astype(np.float64)


def _resolve_block(shape, block) -> int:
    """Cell size in pixels: caller's choice, else ~1/16 of the short side.

    A fixed 16-pixel cell is far too fine on a 4K frame (it follows texture
    instead of shading) and far too coarse on a 128-pixel thumbnail, so the
    default scales with the frame and ``sigma_cells`` stays the semantic knob.
    """
    if block is None:
        return int(np.clip(min(shape) // 16, 8, 64))
    size = int(block)
    if size < 4:
        raise ValueError("block must be at least 4 pixels")
    return size


def _block_grid(image: np.ndarray, block: int) -> np.ndarray:
    """Per-block medians on a coarse grid, reflecting the frame to fill edges."""
    size = int(block)
    h, w = image.shape
    if min(h, w) < size:
        size = max(4, min(h, w) // 2)
    pad_y = (-h) % size
    pad_x = (-w) % size
    padded = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect")
    grid = padded.reshape(padded.shape[0] // size, size,
                          padded.shape[1] // size, size)
    return np.median(grid, axis=(1, 3))


def exposure_stats(gray) -> dict:
    """Brightness, histogram, clipping and dynamic range of one frame.

    These four are the only *exactly measured* quantities in this module: they
    are functions of the pixel array with no model behind them. ``clipped`` means
    a pixel sitting on the 0 or 255 rail, i.e. information the sensor has already
    thrown away, and ``dynamic_range`` is the p95-p5 spread in grey levels.
    """
    image = _as_gray(gray)
    levels = np.clip(np.rint(image), 0, 255).astype(np.int64)
    total = image.size
    histogram = np.bincount(levels.ravel(), minlength=256)
    low, high = float(image.min()), float(image.max())
    stats = {
        "mean": float(image.mean()),
        "std": float(image.std()),
        "min": low,
        "max": high,
        "histogram": [int(v) for v in histogram],
        "clipped_low_fraction": float(np.count_nonzero(levels == 0) / total),
        "clipped_high_fraction": float(np.count_nonzero(levels == 255) / total),
        "p5": float(np.percentile(image, 5)),
        "p50": float(np.percentile(image, 50)),
        "p95": float(np.percentile(image, 95)),
    }
    stats["clipped_fraction"] = stats["clipped_low_fraction"] + stats["clipped_high_fraction"]
    stats["dynamic_range"] = stats["p95"] - stats["p5"]
    return stats


def normalise_exposure(gray, target_mean=None, *, max_gain: float = DEFAULT_MAX_GAIN,
                       max_clipped_fraction: float
                       = DEFAULT_MAX_CLIPPED_FRACTION) -> Normalised:
    """Apply one scalar gain to reach ``target_mean``; return image and gain.

    The gain is capped *before* it is applied: the largest value that keeps the
    clipped fraction inside ``max_clipped_fraction`` and ``max_gain``, so a
    brightening pass cannot blow the highlights on the way past the target. When
    the cap binds, ``requested_gain`` holds the gain the target wanted, so the
    shortfall is reported rather than silently absorbed.

    A single scalar is the whole model here. It cannot fix a frame whose
    illumination varies across it - that is ``flatten`` - and it cannot recover
    anything already on a rail.
    """
    image = _as_gray(gray)
    target = MID_GREY_TARGET if target_mean is None else float(target_mean)
    if not np.isfinite(target) or not 0.0 < target < 255.0:
        raise ValueError(f"target_mean must be a grey level strictly between 0 and "
                         f"255, got {target}")
    limit = float(max_clipped_fraction)
    if not np.isfinite(limit) or not 0.0 <= limit < 1.0:
        raise ValueError("max_clipped_fraction must be finite and in [0, 1)")
    cap = float(max_gain)
    if not np.isfinite(cap) or cap <= 1.0:
        raise ValueError("max_gain must be finite and greater than 1")

    current = float(image.mean())
    if current <= 0.0:
        raise ValueError("a black frame has no exposure to normalise: mean is 0")
    requested = target / current

    n = image.size
    allowed = int(np.floor(limit * n))
    gain = requested
    if requested > 1.0:
        brightest = np.sort(image.ravel())[::-1]
        # At most `allowed` pixels may reach the rail, so the (allowed+1)-th
        # brightest pixel sets the ceiling - held a grey level short of 255 so
        # rounding cannot put it on the rail anyway.
        ceiling = (255.0 - RAIL_MARGIN) / float(brightest[allowed]) \
            if allowed < n and brightest[allowed] > 0 else float("inf")
        gain = min(requested, ceiling, cap)
    elif requested < 1.0:
        darkest = np.sort(image.ravel())
        floor_gain = (0.5 + RAIL_MARGIN) / float(darkest[allowed]) \
            if allowed < n and darkest[allowed] > 0 else 0.0
        gain = max(requested, floor_gain, 1.0 / cap)

    out = np.rint(np.clip(image * gain, 0.0, 255.0))
    # Count the rails on the rounded values that are actually written out:
    # casting a float to uint8 truncates, so a pixel at 254.9 would have been
    # reported as clipped while the stored frame says 254.
    levels = out.astype(np.int64)
    clipped = float(np.count_nonzero((levels == 0) | (levels == 255)) / n)
    return Normalised(out.astype(np.uint8), float(gain), float(requested), clipped)


def illumination_field(gray, *, sigma_cells: float, block=None) -> np.ndarray:
    """Low-frequency shading estimate: block medians, smoothed, stretched back.

    Median over each ``block`` x ``block`` cell, a separable Gaussian of
    ``sigma_cells`` *cells* across the coarse grid, then bicubic upsample. A
    median rather than a mean is what makes it survive a shadow edge and a patch
    of dark texture without dragging the estimate along with them.

    The field is in grey levels and floored at 1.0 so it can be divided by.
    Proxy: it attributes anything low-frequency - shading, a large dark surface,
    vignetting - to illumination.
    """
    image = _as_gray(gray)
    sigma = float(sigma_cells)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_cells must be finite and positive")
    grid = _block_grid(image, _resolve_block(image.shape, block))
    smoothed = _separable_blur(grid, sigma)
    field = cv2.resize(smoothed.astype(np.float32), (image.shape[1], image.shape[0]),
                       interpolation=cv2.INTER_CUBIC).astype(np.float64)
    return np.maximum(field, 1.0)


def flatten(gray, *, sigma_cells: float = 2.0, block=None,
            max_correction: float = 4.0) -> Flattened:
    """Divide the estimated illumination field back out of the frame.

    ``image * mean(field) / field``, with the per-pixel correction capped at
    ``max_correction`` because amplifying a deep shadow amplifies its noise by
    the same factor. Returns ``(image, field)`` so the caller can see, and log,
    exactly what was removed.

    This is the mitigation for slow illumination drift across a flight. It is not
    shadow removal, it does not recover clipped highlights, and on a frame whose
    low-frequency variation *is* real albedo (a dark road next to a bright wall)
    it flattens the scene, not the lighting.
    """
    image = _as_gray(gray)
    cap = float(max_correction)
    if not np.isfinite(cap) or cap <= 1.0:
        raise ValueError("max_correction must be finite and greater than 1")
    field = illumination_field(image, sigma_cells=sigma_cells, block=block)
    correction = np.clip(float(field.mean()) / field, 1.0 / cap, cap)
    return Flattened(np.rint(np.clip(image * correction, 0.0, 255.0)).astype(np.uint8), field)


def shadow_suspects(gray, *, sigma_cells: float = 2.0, block=None,
                    contrast_window: int = 15, dark_ratio: float = SHADOW_DARK_RATIO,
                    contrast_ratio: float = SHADOW_CONTRAST_RATIO) -> dict:
    """Flag dark low-frequency regions that still have local contrast.

    A shadow multiplies the light arriving at a surface, so the texture inside it
    keeps its *relative* contrast while its level drops. A genuinely dark object
    has absorbed the light instead: low level *and* low relative contrast. The
    mask is the intersection of "below ``dark_ratio`` of the in-the-light level"
    and "at least ``contrast_ratio`` of the typical local contrast", opened and
    de-speckled.

    Down-weighting hint only, not shadow removal, and deliberately conservative:
    it reports the fraction of the frame it suspects rather than a cleaned image.
    Colour would make this far more reliable (chromaticity barely moves under a
    shadow) but a grey frame is what the pipeline has at this stage.
    """
    image = _as_gray(gray)
    dark = float(dark_ratio)
    keep = float(contrast_ratio)
    if not (0.0 < dark < 1.0 and np.isfinite(dark)):
        raise ValueError("dark_ratio must be inside (0, 1)")
    if not np.isfinite(keep) or keep <= 0:
        raise ValueError("contrast_ratio must be finite and positive")

    cell = _resolve_block(image.shape, block)
    field = illumination_field(image, sigma_cells=sigma_cells, block=cell)
    window = int(contrast_window) | 1
    local_mean = _box(image, window)
    local_var = np.maximum(_box(image * image, window) - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)
    # Relative contrast, so a multiplicative darkening leaves it alone.
    relative = local_std / (local_mean + 1.0)
    # "In the light" is the upper quartile of the field, not its median: with a
    # median reference a frame that is mostly shadow would call itself normal.
    reference = float(np.percentile(field, 75))
    lit = field >= reference
    typical = float(np.median(relative[lit])) if lit.any() else float(np.median(relative))
    mask = (field < dark * reference) & (relative >= keep * typical)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)
    mask = _drop_small_components(mask)

    fraction = float(mask.mean())
    return {
        "mask": mask,
        "fraction": fraction,
        "pixels": int(mask.sum()),
        "field": field,
        "reference_level": reference,
        "typical_relative_contrast": typical,
        "mean_level_in_suspects": float(image[mask].mean()) if mask.any() else None,
        "mean_level_outside": float(image[~mask].mean()) if (~mask).any() else None,
        "thresholds": {"dark_ratio": dark, "contrast_ratio": keep,
                       "sigma_cells": float(sigma_cells), "block": cell,
                       "contrast_window": window},
        "use": "down-weighting hint only: this is not shadow removal and the "
               "image is returned unchanged",
    }


def _box(array: np.ndarray, size: int) -> np.ndarray:
    return cv2.boxFilter(array, cv2.CV_64F, (size, size), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def _drop_small_components(mask: np.ndarray,
                           min_fraction: float = SHADOW_MIN_AREA_FRACTION) -> np.ndarray:
    """Remove specks: a suspicion over a handful of pixels is noise, not a shadow."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    floor = max(9.0, min_fraction * mask.size)
    keep = np.zeros(count, bool)
    for label in range(1, count):
        keep[label] = stats[label, cv2.CC_STAT_AREA] >= floor
    return keep[labels]


def photometric_consistency(reference, others, *, patches=None, patch_size: int = 32,
                            max_patches: int = 64, flatten_first: bool = True,
                            sigma_cells: float = 2.0, block=None) -> dict:
    """NCC between a reference frame and each of ``others``, per sampled patch.

    Patches are taken from the same fixed grid in every frame (or from
    ``patches``, an Nx2 array of top-left corners) and scored with normalised
    cross-correlation, which is invariant to a scalar gain and an offset - so a
    plain exposure change is invisible to it by design, and what remains is the
    part of the illumination change that is *not* a gain: shading, a shadow
    crossing the frame, a tone curve.

    With ``flatten_first`` (the default) both sides get ``flatten``'d first, which
    is what turns a low-frequency illumination drift back into a high score; that
    recovery is the number worth reporting, and the paired ``flatten_first=False``
    run is the number that says how much drift there was.

    Per-frame score is the mean over its patches, and zero variance in a patch
    makes NCC undefined - those patches are dropped, and a frame with nothing left
    scores 0.0 with a ``zero_variance_patch`` note rather than 1.0.
    """
    base = _as_gray(reference, "reference")
    if isinstance(others, np.ndarray) or isinstance(others, (bytes, str)):
        raise ValueError("others must be a sequence of frames, not a single array")
    frames = list(others)
    if not frames:
        raise ValueError("others is empty: there is nothing to compare the reference to")
    size = int(patch_size)
    if size < 4:
        raise ValueError("patch_size must be at least 4 pixels")
    limit = int(max_patches)
    if limit < 1:
        raise ValueError("max_patches must be at least 1")

    if flatten_first:
        base = flatten(base, sigma_cells=sigma_cells, block=block).image.astype(np.float64)
        frames = [flatten(_as_gray(frame, "others[i]"), sigma_cells=sigma_cells,
                          block=block).image.astype(np.float64) for frame in frames]
    else:
        frames = [_as_gray(frame, "others[i]") for frame in frames]
    for frame in frames:
        if frame.shape != base.shape:
            raise ValueError(f"a frame is {frame.shape} but the reference is {base.shape}")

    positions = _patch_positions(base.shape, size, patches=patches, max_patches=limit)
    scores, notes = [], set()
    for frame in frames:
        values = []
        for top, left in positions:
            first = base[top:top + size, left:left + size]
            second = frame[top:top + size, left:left + size]
            score = _ncc(first, second)
            if score is None:
                notes.add("zero_variance_patch")
            else:
                values.append(score)
        scores.append(float(np.mean(values)) if values else 0.0)

    return {
        "ncc": scores,
        "median": float(np.median(scores)) if scores else 0.0,
        "min": float(min(scores)) if scores else 0.0,
        "max": float(max(scores)) if scores else 0.0,
        "patches_used": int(len(positions)),
        "patch_size": size,
        "flattened": bool(flatten_first),
        "notes": sorted(notes),
    }


def _patch_positions(shape, size, *, patches, max_patches):
    h, w = shape
    if patches is not None:
        corners = np.asarray(patches)
        if corners.ndim != 2 or corners.shape[1] != 2 or not len(corners):
            raise ValueError("patches must be a non-empty Nx2 array of (row, col) "
                             "top-left corners")
        if not np.isfinite(corners.astype(np.float64)).all():
            raise ValueError("patches must be finite")
        corners = corners.astype(np.int64)
        if np.any(corners < 0) or np.any(corners[:, 0] + size > h) \
                or np.any(corners[:, 1] + size > w):
            raise ValueError(f"a patch does not fit: frames are {h}x{w} and patches "
                             f"are {size}x{size}")
        return [(int(y), int(x)) for y, x in corners]
    rows = np.arange(0, max(1, h - size + 1), max(1, (h - size + 1) // 4))
    cols = np.arange(0, max(1, w - size + 1), max(1, (w - size + 1) // 4))
    grid = [(int(y), int(x)) for y in rows for x in cols]
    if len(grid) > max_patches:                     # even stride, not random
        stride = int(np.ceil(len(grid) / max_patches))
        grid = grid[::stride]
    return grid


def _ncc(first: np.ndarray, second: np.ndarray):
    a = first - float(first.mean())
    b = second - float(second.mean())
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if norm <= 1e-9:
        return None
    return float((a * b).sum() / norm)
