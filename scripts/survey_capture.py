"""Decide which frames a single pass should contribute, and what to do with them.

Four of the brief's eight challenges are decided before reconstruction even starts,
by the frames that go in and what is masked on them:

* **limited viewing angles** - ``plan`` spends the frame budget along the *flight
  geometry* (``survey_selection``) instead of along the clock, so a constant-speed
  pass does not oversample the straight parts and starve the turns;
* **motion blur and compression** - ``assess_frame`` scores every candidate with
  ``survey_frame_quality`` and drops the ones no matcher can use;
* **variable illumination and shadows** - ``prepare_for_matching`` flattens the
  low-frequency illumination field of the *matching* copy only, so features survive
  a cloud edge while the colourised copy stays untouched;
* **dynamic objects** - ``write_masks`` turns ``survey_dynamics`` into PNG masks in
  the layout COLMAP's ``--ImageReader.mask_path`` expects.

Every output states what it is: a plan is a plan, a mask is a mask, and none of it
is evidence about accuracy.
"""
import json
import math
from pathlib import Path

import numpy as np

try:  # imported as scripts.survey_capture by the tests
    from scripts import survey_dynamics as dynamics
    from scripts import survey_frame_quality as quality
    from scripts import survey_photometry as photometry
    from scripts import survey_selection as selection
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_dynamics as dynamics
    import survey_frame_quality as quality
    import survey_photometry as photometry
    import survey_selection as selection

DEFAULT_MIN_BASELINE_M = 1.0
"""Floor on the distance between two frames that may both enter the model. One metre
is roughly a hover: past that the pair adds almost no new parallax for a drone at
tens of metres of altitude, and below it the frames are duplicates with a different
filename."""
DROP_WEIGHT = 0.25
"""A frame scoring below this carries more harm than information into matching."""
MASK_FILE_SUFFIX = ".png"
"""COLMAP's convention, verified against the tool rather than assumed: it looks for
``<mask_path>/<name stored in the database>.png``, where that name keeps its clip
subdirectory *and* its original extension - ``masks/rocks/00000.jpg.png``. Writing
``masks/00000.png`` produced 72 MASK_ERROR lines and an empty database that COLMAP
still exited 0 on."""
__all__ = ["plan", "assess_frame", "prepare_for_matching", "masks_for_sequence",
           "write_masks", "camera_matrix", "DEFAULT_MIN_BASELINE_M", "DROP_WEIGHT"]


def _number(value, name, *, positive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(name + " must be finite and positive")
    return result


def _positions(times, telemetry):
    """ENU camera centres at the requested times, via the same bracketed interpolation
    the georeference uses, so the plan and the alignment never disagree."""
    try:
        from scripts import survey_georef as georef
    except ImportError:
        import survey_georef as georef
    points, _, matched = georef.sample_positions([float(t) for t in times], telemetry)
    return np.asarray(points, dtype=np.float64), [bool(m) for m in matched]


def plan(telemetry, *, video_duration_s, budget, min_baseline_m=DEFAULT_MIN_BASELINE_M,
         candidate_stride_s=None):
    """Spend a frame budget along the flight path, and report what that bought.

    The trajectory is taken from the GPS that will also georeference the model, so
    the selection cannot quietly optimise for a path the reconstruction never saw.
    Times where GPS does not bracket a position are excluded rather than guessed.
    """
    duration = _number(video_duration_s, "video_duration_s")
    budget = int(budget)
    if budget < 2:
        raise ValueError("budget must be at least two frames")
    baseline = _number(min_baseline_m, "min_baseline_m")
    samples = telemetry.get("samples") if isinstance(telemetry, dict) else None
    if not samples:
        raise ValueError("telemetry carries no samples to plan against")
    stride = candidate_stride_s or max(0.25, duration / (budget * 20))
    candidates = np.arange(0.0, duration + stride / 2, stride)
    candidates[-1] = min(candidates[-1], duration)
    positions, matched = _positions(candidates, telemetry)
    usable = [index for index, hit in enumerate(matched) if hit]
    if len(usable) < 3:
        raise ValueError("only " + str(len(usable)) + " candidate times fall inside a GPS "
                         "bracket; widen the stride or fix the telemetry gaps")
    track = positions[usable]
    chosen_local = selection.select_keyframes(track, budget=budget, min_baseline_m=baseline)
    chosen = [usable[int(index)] for index in chosen_local]
    coverage = selection.path_coverage(track, chosen_local, min_baseline_m=baseline)
    # Parallax is quoted against the centre of the flight box: a stand-in for the
    # scene the pass is looking at, which is not known until it is reconstructed.
    centre = (track.min(axis=0) + track.max(axis=0)) / 2.0
    try:
        parallax = selection.mean_parallax_deg(track, chosen_local, centre)
        parallax_note = "mean angle at the flight-box centre between consecutive frames"
    except ValueError as error:
        parallax, parallax_note = None, "parallax not computable: " + str(error)
    return {"schema_version": 1, "budget": budget, "min_baseline_m": baseline,
            "candidate_stride_s": float(stride), "video_duration_s": duration,
            "target_times_s": [float(candidates[index]) for index in chosen],
            "target_indices": [int(index) for index in chosen],
            "candidates": int(len(candidates)), "candidates_anchored": len(usable),
            "unanchored_dropped": int(len(candidates) - len(usable)),
            "frame_count": len(chosen), "path_coverage": coverage,
            "mean_parallax_deg_to_flight_box_centre": parallax,
            "parallax_basis": parallax_note,
            "basis": ("arc-length spacing on the GPS trajectory with a minimum baseline; "
                      "a plan for frame selection, not a measurement of the scene"),
            "warnings": ["frames outside a GPS bracket are excluded, never interpolated "
                         "past the last fix"]}


def assess_frame(gray, *, previous_gray=None, energy_floor=None, centroid_floor=None):
    """Score one candidate frame. Returns the metrics, the label and whether to keep it."""
    report = quality.frame_quality(gray, previous_gray=previous_gray,
                                   energy_floor=energy_floor, centroid_floor=centroid_floor)
    report["keep"] = report["weight"] >= DROP_WEIGHT
    return report


def prepare_for_matching(gray, *, flatten=True, normalise=False, sigma_cells=2.0,
                         target_mean=None):
    """A copy of the frame that is easier to match, never the copy that is colourised.

    ``flatten`` divides out the low-frequency illumination field; ``normalise`` then
    pulls the mean towards a target grey. Both are gain operations on the *matching*
    image only: features are what COLMAP reads, and a shadow that moves across a
    facade is a change in gain, not a change in the scene.
    """
    image = np.asarray(gray)
    actions = []
    if flatten:
        image = photometry.flatten(image, sigma_cells=sigma_cells).image
        actions.append("illumination_flattened")
    if normalise:
        result = photometry.normalise_exposure(image, target_mean)
        image = result.image
        actions.append("exposure_normalised(gain=%.3f)" % result.gain)
    return image, actions


def camera_matrix(width, height, *, focal_px=None, principal=None):
    """Intrinsics for the epipolar geometry, from a declared focal or a stated guess.

    Without a calibration the caller passes the focal COLMAP itself would assume;
    the returned matrix records which of the two happened, because a mask computed
    from a made-up focal is a weaker veto than one computed from a measured lens.
    """
    focal = _number(focal_px if focal_px is not None else 1.2 * max(width, height), "focal_px")
    cx, cy = principal if principal else (width / 2.0, height / 2.0)
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def masks_for_sequence(frames, camera_centers, k_matrix, *, tolerance_px=2.0, **options):
    """Per-frame dynamic-object masks, with the sequence's own coverage cost."""
    if len(frames) < 2:
        raise ValueError("a moving object needs at least two frames to move between")
    results = dynamics.dynamic_masks_for_sequence(frames, camera_centers, k_matrix,
                                                  tolerance_px=tolerance_px, **options)
    cost = dynamics.coverage_cost(results, np.asarray(frames[0]).shape[:2])
    note = dynamics.static_obstacle_note(results, np.asarray(frames[0]).shape[:2])
    return results, {"coverage_cost": cost, "static_obstacle_note": note}


def write_masks(results, frames, out_dir, *, images_dir=None):
    """Write the masks COLMAP reads: one greyscale PNG per image, named after it.

    COLMAP's convention is that a *zero* pixel is excluded from feature extraction,
    so the boolean mask is inverted on the way out. Writing the complement would
    have the matcher lock onto the traffic and ignore the buildings.

    The mask must be the same pixel size as the image it names: masks are computed
    at a reduced working width for speed, and a size mismatch makes COLMAP refuse
    the image entirely - measured on real footage, where one mismatched mask cost
    all 72 frames of a scene and exited 0 until the frame-count guard caught it.
    """
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for entry, name in zip(results, frames):
        mask = np.asarray(entry["mask"], dtype=bool)
        path = out_dir / (str(name) + MASK_FILE_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        if images_dir is not None:
            image = cv2.imread(str(Path(images_dir) / name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("cannot read the image a mask is being written for: " + str(name))
            size = (int(image.shape[1]), int(image.shape[0]))
            if (mask.shape[1], mask.shape[0]) != size:
                mask = cv2.resize(mask.astype(np.uint8), size,
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        if not cv2.imwrite(str(path), np.where(mask, 0, 255).astype(np.uint8)):
            raise ValueError("could not write mask: " + str(path))
        written.append({"image": str(name), "mask": str(name) + MASK_FILE_SUFFIX,
                        "masked_fraction": float(mask.mean()),
                        "mask_size": [int(mask.shape[1]), int(mask.shape[0])],
                        "confidence": entry.get("confidence")})
    total = sum(row["masked_fraction"] for row in written) / max(len(written), 1)
    return {"dir": out_dir.name, "count": len(written), "mean_masked_fraction": total,
            "convention": "0 = excluded from feature extraction (COLMAP mask semantics)",
            "files": written}


def read_plan(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "target_times_s" not in data:
        raise ValueError(str(path.name) + " is not a capture plan")
    return data
