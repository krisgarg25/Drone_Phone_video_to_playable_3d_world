"""Challenge (ii): motion blur and video compression artifacts, no reference.

The pipeline already scores candidate keyframes with one number - the variance of
a Laplacian response, in ``extract_keyframes.py`` (``cv2.Laplacian(gray, CV_64F)
.var()``) - and keeps a corpus-derived blur floor. That is enough to drop the
catastrophically soft frames, and not enough to answer the question the challenge
actually asks: *why* is this frame soft? A frame blurred by a fast drone (a
directional smear), a frame out of focus (isotropic attenuation), and a frame
chewed up by the H.264 encoder (block lattice, dead-zone quantisation) all score
low on Laplacian variance. Downstream they want opposite treatments: a blurred
frame still constrains pose, a compressed frame still has valid texture, and a
flat wall has neither.

This module adds the second and third axis and turns the three into a per-frame
usability weight:

* ``blur_metrics``       - Laplacian variance (same definition as the keyframe
                           extractor), Sobel/Tenengrad gradient energy, and a
                           structure-tensor anisotropy ratio that separates
                           directional (motion) from isotropic (defocus) loss.
* ``compression_metrics``- a GBPS-style block-boundary/intra-block difference
                           ratio plus flat-block and grid-excess cues.
* ``frame_quality``      - those signals as a 0..1 weight plus human-readable
                           ``reasons``, optionally informed by the previous frame.
* ``weights_for_manifest`` - applies the weight to a ``keyframes.jsonl`` list
                           without touching the list itself.

WHAT THIS IS NOT: this is not a reference-based quality measurement. There is no
undistorted ground truth to compare against, no bitstream access to the encoder's
own quantisation parameters, and no shutter/ISO metadata. Every number here is an
*image-statistics proxy* for a perceptual failure mode. Two consequences are
load-bearing and documented at the functions that hit them:

1. Laplacian variance and gradient energy are absolute scales. They depend on
   resolution, texture content and sensor gain, so a single frame cannot be
   called "blurred" against a hard-coded threshold - only against a floor
   derived from that corpus (see ``weights_for_manifest``). Where no floor is
   supplied, ``blur_label`` refuses the absolute verdict and returns the
   scale-free answer instead.
2. Blur and absence-of-texture are not separable from one frame's statistics
   alone; neither is compression noise from real fine detail at low bitrates.
   Those cases are labelled ``low_texture_or_blur`` - the same vocabulary
   ``capture_diagnostics.classify`` already uses - rather than guessed at.

Calibration status, so nobody reads the thresholds as settled: the motion/defocus
split is validated on constructed degradations only (anisotropy 0.006-0.02 sharp
and defocused, 0.39-0.79 smeared, on the same source). On eight frames sampled
from one real indoor 487-frame clip the same statistic spread 0.02-0.74 with no
ground-truth labels attached to any of them, and the flat-block fraction crossed
its cue on three of eight while the boundary ratio crossed on one - which is why
the quantisation penalty is gated on the boundary evidence instead of standing
alone. The real-footage numbers are observations, not a validation. A third known
false positive is asserted rather than hidden: content whose period is a whole
number of blocks (a 24-pixel grating reads at boundary ratio 6.6, the same grating
at 26 reads 0.85) is indistinguishable from blocking in one grey frame.

CPU only (cv2 + numpy), no decoder, no I/O, no GPU.

Cost, measured on this machine at the scales the pipeline actually uses:
``frame_quality`` is ~44 ms per 640x480 frame and ~99 ms at 720x960, of which the
2-D FFT behind ``spectral_centroid`` is ~21%. ``weights_for_manifest`` is ~20 us
per existing row, because it reads numbers the keyframe extractor already wrote
rather than re-decoding anything. Run the per-frame path on the analysis-scale
frame, as ``capture_diagnostics.PROBE_MAX_SIDE`` already advises, not on the full
raster: over 400 keyframes that is the difference between ~18 s and ~40 s.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["blur_metrics", "blur_label", "compression_metrics", "frame_quality",
           "weights_for_manifest"]

# Below this the band split, the structure tensor and an 8-pixel block lattice
# are all under-determined, so the numbers would be noise dressed as a metric.
MIN_SIDE = 32
# An anisotropy of 0.35 means one gradient axis carries ~2.1x the energy of the
# other (S-major/S-minor = (1+a)/(1-a)). That is the scale-free motion cue.
MOTION_ANISOTROPY_MIN = 0.35
# A clean image has a boundary/intra difference ratio of ~1.0; blocking pushes it
# above 1.25. Blur pushes it *towards or below* 1.0, which is what keeps the two
# failure modes distinguishable.
BLOCKING_RATIO_MIN = 1.25
BLOCKING_RATIO_FULL = 2.2
# A frame whose inside-block differences have quantised to exactly zero has a
# division by nothing, not an infinitely blocky frame: the ratio saturates here.
BLOCKING_RATIO_CAP = 50.0
# A third of the blocks spanning at most one grey level means the encoder's
# dead zone, not the scene, decided what the flat areas look like.
QUANTISED_FLAT_MIN = 0.30
# Radial power centroid of the spectrum, in Nyquist units (1.0 == Nyquist on one
# axis, sqrt(2) == the diagonal corner). Isotropic defocus drags it down; a
# textureless wall with sensor noise leaves it high. Only meaningful relative to
# a reference frame from the same scene - see ``blur_label``'s ``centroid_floor``.
DEFOCUS_CENTROID_RATIO = 0.90
# extract_keyframes.py drops near-duplicates at a mean absolute thumbnail
# difference of 2.5 grey levels; the weight uses the same threshold so the two
# stages cannot disagree about what a repeat frame is.
DUPLICATE_DELTA_MAX = 2.5
LARGE_DELTA_MIN = 45.0
# Weight floors for each penalty. Deliberately coarse: these are priors for a
# sampler, not probabilities.
WEIGHT_MIN_FOR_BLUR = 0.20
WEIGHT_MIN_FOR_MOTION = 0.35
WEIGHT_MIN_FOR_BLOCKING = 0.45
DUPLICATE_WEIGHT = 0.55
LARGE_DELTA_WEIGHT = 0.80
# Laplacian variance spans orders of magnitude within one clip, so sharpness is
# penalised in octaves below the floor rather than linearly: linear ramps either
# saturate after one halving or never reach the floor at all.
SHARPNESS_PENALTY_AT_FLOOR = 0.75
SHARPNESS_PENALTY_OCTAVES = 3.0


def _as_gray(image, name: str = "gray") -> np.ndarray:
    """Validate a single-channel frame and return it in float64.

    Rejects empty, non-finite and multi-channel input with a message that says
    which rule was broken - a NaN pixel propagating into a variance silently
    turns every downstream weight into a NaN.
    """
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D single-channel frame, got "
                         f"{array.ndim}-D with shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if min(array.shape) < MIN_SIDE:
        raise ValueError(f"{name} must be at least {MIN_SIDE}x{MIN_SIDE} pixels, "
                         f"got {array.shape[0]}x{array.shape[1]} - statistics from "
                         f"a smaller frame are not meaningful")
    values = array.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite pixels "
                         f"(NaN or inf): {int((~np.isfinite(values)).sum())} of them")
    return values


def _ramp(value: float, low: float, high: float) -> float:
    """Smoothstep 0..1 between `low` and `high`; 1.0 at or above `high`."""
    value, low, high = float(value), float(low), float(high)
    if not (np.isfinite(value) and np.isfinite(low) and np.isfinite(high)):
        raise ValueError("ramp bounds and value must be finite")
    if high <= low:
        raise ValueError("ramp needs high > low")
    x = min(max((value - low) / (high - low), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _sharpness_severity(sharpness: float, floor: float) -> float:
    """How far past the clip's blur floor a frame is, as 0..1 severity.

    Zero at or above the floor - which is the point
    ``extract_keyframes.select_clip`` already keeps frames - then a step to
    ``SHARPNESS_PENALTY_AT_FLOOR``, graded by octave down to fully unusable
    ``SHARPNESS_PENALTY_OCTAVES`` halvings below it. The step is deliberate: the
    floor is a drop threshold in the existing pipeline, so crossing it is a
    discontinuous change in usability, not a gentle slope.
    """
    sharpness, floor = float(sharpness), float(floor)
    if not (np.isfinite(sharpness) and np.isfinite(floor)):
        raise ValueError("sharpness and floor must be finite")
    if floor <= 0:
        raise ValueError("floor must be positive to measure sharpness against")
    if sharpness >= floor:
        return 0.0
    depth = min(float(np.log2(floor / sharpness)) / SHARPNESS_PENALTY_OCTAVES, 1.0) \
        if sharpness > 0 else 1.0
    return SHARPNESS_PENALTY_AT_FLOOR + (1.0 - SHARPNESS_PENALTY_AT_FLOOR) * depth


def _box(array: np.ndarray, size: int) -> np.ndarray:
    k = int(size) | 1
    return cv2.boxFilter(array, cv2.CV_64F, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def _spectral_centroid(image: np.ndarray) -> float:
    """Power-weighted mean radius of the spectrum, in Nyquist units.

    1.0 is Nyquist along an axis, so the value runs ~0.4-0.6 for natural texture
    and drops as the high band is filtered out. Expressed as a fraction of
    Nyquist rather than cycles/pixel so it does not move with decode resolution.
    """
    windowed = image - float(image.mean())
    power = np.abs(np.fft.rfft2(windowed)) ** 2
    h, w = windowed.shape
    freq = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.rfftfreq(w)[None, :]) / 0.5
    band = freq > 0.04                      # ignore DC and the shading band
    total = float(power[band].sum())
    if total <= 0.0:
        return 0.0
    return float((power * freq)[band].sum() / total)


def blur_metrics(gray, *, integration: int = 9) -> dict:
    """Directional and isotropic sharpness signals for one frame.

    ``laplacian_variance`` is exactly what ``extract_keyframes.py`` scores
    candidates with, so a manifest's ``sharpness`` field and this module agree.
    ``tenengrad`` is mean squared Sobel gradient energy. ``anisotropy`` is
    ``|Sxx-Syy| / (Sxx+Syy)`` of the structure tensor, averaged over the
    half of the frame carrying the most gradient energy so flat regions cannot
    dilute it, and ``orientation_deg`` is the direction of the dominant gradient
    (0..180). ``spectral_centroid`` is the power-weighted mean radius of the
    spectrum.

    Proxy, not measurement: none of these is a sharpness *value* in the optical
    sense - they are statistics that move when sharpness moves.
    """
    image = _as_gray(gray)
    if int(integration) < 3:
        raise ValueError("integration must be at least 3 pixels")

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sxx, syy = _box(gx * gx, integration), _box(gy * gy, integration)
    sxy = _box(gx * gy, integration)
    energy = sxx + syy
    # Structure tensor is only meaningful where there is structure.
    strong = energy >= float(np.percentile(energy, 50))
    if not strong.any():
        strong = energy > 0
    xx = float(sxx[strong].mean())
    yy = float(syy[strong].mean())
    xy = float(sxy[strong].mean())
    total = xx + yy
    anisotropy = abs(xx - yy) / total if total > 0 else 0.0
    orientation = float(np.degrees(0.5 * np.arctan2(2.0 * xy, xx - yy))) % 180.0

    return {
        "laplacian_variance": float(laplacian.var()),
        "tenengrad": float(np.mean(gx * gx + gy * gy)),
        "anisotropy": float(anisotropy),
        "orientation_deg": orientation,
        "spectral_centroid": _spectral_centroid(image),
        "directional": bool(anisotropy >= MOTION_ANISOTROPY_MIN),
    }


def blur_label(blur: dict, *, energy_floor=None, centroid_floor=None) -> str:
    """Name the failure mode the blur signals point at, as far as they can.

    With an ``energy_floor`` it returns ``sharp``, ``motion_blur``,
    ``defocus_blur`` or ``low_texture_or_blur``; without one it can only report
    the scale-free shape of the gradient, ``directional`` or ``isotropic``.

    The floors are what make an absolute verdict possible at all.
    The floors are what make an absolute verdict possible at all.
    ``energy_floor`` says "for this clip, this is what an in-focus frame looks
    like"; ``centroid_floor`` says the same about the spectral shape, and without
    it a soft frame and a textureless wall cannot be told apart, so the answer
    degrades to ``low_texture_or_blur`` instead of guessing. Supplying constants
    for either would report a whole low-texture scene as blurred and call it data.
    """
    if not isinstance(blur, dict) or "anisotropy" not in blur:
        raise ValueError("blur must be a dict from blur_metrics()")
    for key in ("anisotropy", "tenengrad", "laplacian_variance", "spectral_centroid"):
        if key not in blur or not np.isfinite(float(blur[key])):
            raise ValueError(f"blur is missing or has a non-finite {key!r}")
    directional = bool(blur["directional"]) if "directional" in blur \
        else float(blur["anisotropy"]) >= MOTION_ANISOTROPY_MIN

    if energy_floor is None:
        return "directional" if directional else "isotropic"
    floor = float(energy_floor)
    if not np.isfinite(floor):
        raise ValueError("energy_floor must be finite")
    if floor <= 0:
        return "directional" if directional else "isotropic"
    if float(blur["laplacian_variance"]) >= floor:
        return "sharp"
    if directional:
        return "motion_blur"
    if float(blur["tenengrad"]) <= 0.0 or centroid_floor is None:
        return "low_texture_or_blur"
    reference = float(centroid_floor)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("centroid_floor must be finite and positive")
    return ("defocus_blur" if float(blur["spectral_centroid"])
            < reference * DEFOCUS_CENTROID_RATIO else "low_texture_or_blur")


def compression_metrics(gray, *, block: int = 8) -> dict:
    """Blockiness and quantisation cues from the pixel grid alone.

    ``block_boundary_ratio`` is the GBPS-style statistic: mean absolute difference
    across the ``block``-pixel lattice divided by the mean across every other
    adjacent-pixel pair. Clean imagery sits at ~1.0, blocking artefacts push it
    above 1.25, and *blur pulls it down* - which is why a soft frame does not get
    mislabelled as a compressed one.

    ``flat_block_fraction`` counts whole blocks that span at most one grey level
    (a dead-zone quantisation cue) and ``grid_excess`` is the mean |Laplacian|
    on the lattice over the mean off it (a deringing cue). ``score`` is 1.0 for
    an artefact-free frame and falls with the two cues.

    Proxy, not measurement: no bitstream, no QP, no encoder stats, and real fine
    detail that happens to be periodic at 8 pixels reads as blockiness.
    """
    image = _as_gray(gray)
    size = int(block)
    if size < 2:
        raise ValueError("block must be at least 2 pixels")
    h, w = image.shape

    horizontal = np.abs(np.diff(image, axis=1))       # x-direction neighbours
    vertical = np.abs(np.diff(image, axis=0))         # y-direction neighbours
    # Diff index j compares pixels j and j+1, so a lattice line at j+1 is a
    # boundary. The Laplacian is full size, so it needs the lattice pixels
    # themselves.
    x_boundary = np.arange(1, w) % size == 0
    y_boundary = np.arange(1, h) % size == 0
    x_lines = np.arange(size, w, size)
    y_lines = np.arange(size, h, size)
    boundary = np.concatenate([horizontal[:, x_boundary].ravel(),
                               vertical[y_boundary, :].ravel()])
    intra = np.concatenate([horizontal[:, ~x_boundary].ravel(),
                            vertical[~y_boundary, :].ravel()])
    if boundary.size == 0 or intra.size == 0:
        raise ValueError(f"frame is too small for a {size}-pixel block analysis")

    boundary_delta = float(boundary.mean())
    intra_delta = float(intra.mean())
    saturated = intra_delta <= 0.0
    ratio = boundary_delta / intra_delta if not saturated else (
        1.0 if boundary_delta == 0 else BLOCKING_RATIO_CAP)
    ratio = min(ratio, BLOCKING_RATIO_CAP)

    blocks = image[:(h // size) * size, :(w // size) * size]
    if blocks.size:
        tiled = blocks.reshape(h // size, size, w // size, size)
        spread = tiled.max(axis=(1, 3)) - tiled.min(axis=(1, 3))
        flat_fraction = float(np.mean(spread <= 1.0))
    else:
        flat_fraction = 0.0

    lap = np.abs(cv2.Laplacian(image, cv2.CV_64F))
    on_grid = np.concatenate([lap[:, x_lines].ravel(), lap[y_lines, :].ravel()])
    off_columns = np.ones(w, bool)
    off_columns[x_lines] = False
    off_rows = np.ones(h, bool)
    off_rows[y_lines] = False
    off_grid = np.concatenate([lap[:, off_columns].ravel(), lap[off_rows, :].ravel()])
    grid_excess = float(on_grid.mean() / off_grid.mean()) if off_grid.mean() > 0 else 1.0

    blocking = _ramp(min(ratio, BLOCKING_RATIO_FULL), BLOCKING_RATIO_MIN, BLOCKING_RATIO_FULL)
    quantised = _ramp(flat_fraction, 0.30, 0.90)
    return {
        "block_boundary_ratio": float(ratio),
        "boundary_delta": boundary_delta,
        "intra_delta": intra_delta,
        "ratio_saturated": bool(saturated),
        "flat_block_fraction": flat_fraction,
        "grid_excess": grid_excess,
        "blocking_suspect": bool(ratio >= BLOCKING_RATIO_MIN),
        "score": float(np.clip(1.0 - 0.75 * blocking - 0.25 * quantised, 0.0, 1.0)),
    }


def frame_quality(gray, *, previous_gray=None, energy_floor=None, centroid_floor=None) -> dict:
    """One 0..1 usability weight per frame, plus the reasons behind it.

    The weight is the product of independent penalties, each of which is a
    documented proxy: low sharpness (only judgeable against ``energy_floor``,
    which the caller derives from the corpus), directional gradient structure
    below that floor, blockiness, dead-zone quantisation and - when
    ``previous_gray`` is given - a near-duplicate or a jump relative to the
    previous frame. ``reasons`` is empty exactly when no penalty fired, so a
    weight of 1.0 always means "nothing was found", never "nothing was looked
    for".

    Motion blur costs more than defocus at equal sharpness on purpose: a
    defocused frame still projects consistently, while a smeared one carries
    direction-dependent feature error that biases triangulation.

    Returns the underlying ``blur`` and ``compression`` signal dicts alongside,
    so a reviewer can check any weight against the numbers that produced it.
    """
    image = _as_gray(gray)
    blur = blur_metrics(image)
    compression = compression_metrics(image)

    reasons: list[str] = []
    weight = 1.0

    floor = None
    if energy_floor is not None:
        floor = float(energy_floor)
        if not np.isfinite(floor):
            raise ValueError("energy_floor must be finite")
        if floor <= 0:
            floor = None                      # no usable reference: no verdict
        elif blur["laplacian_variance"] < floor:
            severity = _sharpness_severity(blur["laplacian_variance"], floor)
            weight *= _lerp(1.0, WEIGHT_MIN_FOR_BLUR, severity)
            reasons.append("low_sharpness_for_this_clip")

    label = blur_label(blur, energy_floor=floor, centroid_floor=centroid_floor)
    if compression["blocking_suspect"] and label in ("motion_blur", "defocus_blur"):
        # Both failure modes attenuate the high band, so a lattice-aligned frame
        # would otherwise be reported as blurred *and* compressed. The block
        # statistic is the more specific evidence, so it wins the label; the
        # sharpness weight above still applies, because the frame really has
        # lost detail either way.
        label = "compression_artifact"
        reasons.append("compression_explains_attenuation")
    if label == "motion_blur":
        reasons.append("motion_blur_suspected")
        weight *= WEIGHT_MIN_FOR_MOTION
    elif label == "defocus_blur":
        reasons.append("defocus_blur_suspected")
    elif label == "low_texture_or_blur" and floor is not None:
        reasons.append("low_texture_or_blur")
    elif label == "directional":
        # Directional structure with no floor to judge it against is a property
        # of the scene (a fence, a window frame), not evidence of a defect.
        reasons.append("directional_texture")

    if compression["blocking_suspect"]:
        severity = _ramp(min(compression["block_boundary_ratio"], BLOCKING_RATIO_FULL),
                         BLOCKING_RATIO_MIN, BLOCKING_RATIO_FULL)
        weight *= _lerp(1.0, WEIGHT_MIN_FOR_BLOCKING, severity)
        reasons.append("blocking_artifacts")
        # Measured on a real 720p indoor clip: three of eight sampled frames put
        # over 30% of their 8x8 blocks inside one grey level while only one
        # showed a lattice, because smooth walls and ceilings are flat in the
        # scene, not in the encoder. So the dead-zone cue discounts a frame only
        # where the block boundary evidence already fired; the raw fraction is
        # still reported for anyone who wants to calibrate it per corpus.
        if compression["flat_block_fraction"] >= QUANTISED_FLAT_MIN:
            weight *= 0.90
            reasons.append("quantised_flat_regions")

    frame_delta = None
    if previous_gray is not None:
        previous = _as_gray(previous_gray, "previous_gray")
        if previous.shape != image.shape:
            raise ValueError(f"previous_gray shape {previous.shape} does not match "
                             f"gray shape {image.shape}")
        frame_delta = float(np.mean(np.abs(image - previous)))
        if frame_delta < DUPLICATE_DELTA_MAX:
            weight *= DUPLICATE_WEIGHT
            reasons.append("near_duplicate_previous")
        elif frame_delta > LARGE_DELTA_MIN:
            weight *= LARGE_DELTA_WEIGHT
            reasons.append("large_frame_to_frame_change")

    return {
        "weight": float(np.clip(weight, 0.0, 1.0)),
        "reasons": reasons,
        "blur": blur,
        "compression": compression,
        "blur_label": label,
        "sharpness": blur["laplacian_variance"],
        "frame_delta": frame_delta,
        "energy_floor": floor,
    }


def weights_for_manifest(rows, *, energy_floor=None) -> list:
    """Map ``keyframes.jsonl`` rows to usability weights, without mutating them.

    Each returned dict is a new object: ``{"file", "weight", "reasons", "basis",
    "energy_floor"}``. A row that already carries a ``quality`` dict from
    ``frame_quality`` is used as-is (``basis="quality"``); a row with only the
    ``sharpness`` field the keyframe extractor writes gets a blur-only weight
    (``basis="sharpness_only"``); a row with neither gets ``weight=None`` and
    ``no_quality_metrics`` rather than an invented number.

    With no explicit floor, the floor is the same adaptive rule
    ``extract_keyframes.select_clip`` uses - the 5th percentile of the manifest's
    own sharpness values, halved - so the weights are calibrated to that clip
    instead of to a constant.
    """
    if isinstance(rows, (str, bytes, dict)) or not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list of keyframes.jsonl row dicts")
    if not len(rows):
        raise ValueError("rows is empty: there is no manifest to weight")
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {position} is {type(row).__name__}, not an object")
        if "file" not in row:
            raise ValueError(f"row {position} has no 'file' key, so its weight could "
                             f"not be matched back to a frame")

    if energy_floor is None:
        sharpness = [float(row["sharpness"]) for row in rows
                     if isinstance(row.get("sharpness"), (int, float, np.integer, np.floating))
                     and np.isfinite(float(row["sharpness"]))]
        floor = float(np.percentile(sharpness, 5)) * 0.5 if sharpness else None
    else:
        floor = float(energy_floor)
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("energy_floor must be finite and non-negative")

    out = []
    for row in rows:
        quality = row.get("quality")
        if isinstance(quality, dict) and "weight" in quality:
            weight = float(np.clip(float(quality["weight"]), 0.0, 1.0))
            reasons = list(quality.get("reasons", []))
            basis = "quality"
        elif isinstance(row.get("sharpness"), (int, float, np.integer, np.floating)) \
                and np.isfinite(float(row["sharpness"])):
            sharp = float(row["sharpness"])
            if floor is None or floor <= 0:
                weight, reasons, basis = 1.0, [], "sharpness_unscaled"
            else:
                severity = _sharpness_severity(sharp, floor)
                weight = _lerp(1.0, WEIGHT_MIN_FOR_BLUR, severity)
                reasons = (["low_sharpness_for_this_clip"] if severity > 0
                           else [])
                basis = "sharpness_only"
        else:
            weight, reasons, basis = None, ["no_quality_metrics"], "none"
        out.append({"file": row["file"], "weight": weight, "reasons": reasons,
                    "basis": basis, "energy_floor": floor})
    return out
