"""Straight-track georeferencing: fix the one rotation camera positions cannot (M11).

A similarity fit of camera centres to GPS (``survey_georef.align_camera_trajectory``)
needs a trajectory that spans at least a plane. A corridor flight - a border fence,
a road, a pipeline - is a line, and a line leaves exactly one rotation unobserved:
the roll about the track axis. ``survey_georef`` correctly refuses such a track.

This module supplies that single missing degree of freedom from gravity and nothing
else. Along-track direction, scale and translation still come from GPS; only the
rotation about the track axis is chosen so the reconstruction's "up" matches ENU up.
Gravity in the reconstruction frame comes from one of two sources, each recorded:

* ``attitude`` - gimbal pitch/roll per frame (DJI SRT / flight log), rotated into the
  reconstruction by each camera's own orientation and robustly averaged.
* ``ground_plane`` - the normal of the dominant plane of the sparse cloud, oriented
  toward the cameras. This assumes the terrain is level ACROSS the track; a cross
  slope becomes a tilt of the same angle, and the output says so.

A cross-track tilt error of d degrees moves a point at cross-track distance x by
x * tan(d) vertically. That is the uncertainty this method adds, and it is reported
as ``cross_track_tilt_deg`` with the distance at which it reaches one metre.

``tilt_check`` is the same gravity used as an independent diagnostic on an ordinary
(non-degenerate) fit: the angle between the fitted "up" and gravity. It never
changes an alignment.

Conventions. Camera rows are world-to-camera (x_cam = R x_world + t, OpenCV axes:
x right, y down, z forward), exactly as ``survey_georef.camera_positions`` reads them.
Gimbal pitch is degrees above the horizon (-90 = nadir), roll is degrees with the
camera's right side down positive. NumPy only; invalid input raises ValueError.
"""
import copy
import math
from itertools import combinations

import numpy as np

import survey_georef as georef

UP = np.array([0.0, 0.0, 1.0])
MIN_TRACK_TO_VERTICAL_DEG = 10.0
"""A track within this angle of vertical leaves yaw, not roll, unobserved; refuse."""
MAX_PAIRS = 512
PLANE_ITERATIONS = 400


def _unit(vector, name):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return vector / norm


def _axis_angle(axis, angle):
    axis = _unit(axis, "rotation axis")
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _rotation_between(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = _unit(a, "a"), _unit(b, "b")
    axis = np.cross(a, b)
    sine, cosine = float(np.linalg.norm(axis)), float(np.clip(a @ b, -1.0, 1.0))
    if sine > 1e-12:
        return _axis_angle(axis, math.atan2(sine, cosine))
    if cosine > 0:
        return np.eye(3)
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    return _axis_angle(np.cross(a, helper), math.pi)


def _angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(_unit(a, "a") @ _unit(b, "b"), -1.0, 1.0))))


def _robust_mean_direction(vectors, *, cutoff_deg=None):
    """Mean of unit vectors after dropping those far from the median-seeded mean."""
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    mean = _unit(np.median(vectors, axis=0), "median direction")
    for _ in range(10):
        angles = np.degrees(np.arccos(np.clip(vectors @ mean, -1.0, 1.0)))
        mad = float(np.median(np.abs(angles - np.median(angles))))
        limit = cutoff_deg if cutoff_deg is not None else max(3.0 * 1.4826 * mad, 0.5)
        keep = angles <= max(limit, float(np.median(angles)))
        updated = _unit(vectors[keep].mean(axis=0), "mean direction")
        if np.allclose(updated, mean, atol=1e-12):
            break
        mean = updated
    angles = np.degrees(np.arccos(np.clip(vectors @ mean, -1.0, 1.0)))
    return mean, keep, angles


def up_in_camera(pitch_deg, roll_deg):
    """World up expressed in OpenCV camera axes for a gimbal at (pitch, roll)."""
    p, r = math.radians(float(pitch_deg)), math.radians(float(roll_deg))
    return np.array([-math.sin(r) * math.cos(p), -math.cos(r) * math.cos(p), math.sin(p)])


def gravity_from_attitude(camera_rows, attitude, *, max_gap_s=0.5):
    """Up in the reconstruction frame from per-frame gimbal pitch/roll.

    ``attitude`` is a list of {t_sec, pitch_deg, roll_deg}; each camera takes the
    nearest sample within ``max_gap_s``. Each camera's own rotation carries its up
    vector into the reconstruction frame; the robust mean of those is gravity.
    """
    samples = sorted((float(a["t_sec"]), float(a["pitch_deg"]), float(a["roll_deg"]))
                     for a in attitude if a.get("pitch_deg") is not None
                     and a.get("roll_deg") is not None)
    if not samples:
        raise ValueError("No attitude samples carry both pitch and roll")
    times = np.array([s[0] for s in samples])
    ups = []
    for row in camera_rows:
        index = int(np.argmin(np.abs(times - float(row["t_sec"]))))
        if abs(times[index] - float(row["t_sec"])) > max_gap_s:
            continue
        rotation = np.asarray(row["camera"]["R_rowmajor"], dtype=np.float64).reshape(3, 3)
        ups.append(rotation.T @ up_in_camera(samples[index][1], samples[index][2]))
    if len(ups) < 3:
        raise ValueError(f"Only {len(ups)} cameras have attitude within {max_gap_s} s; need 3")
    mean, keep, angles = _robust_mean_direction(ups)
    used = int(keep.sum())
    spread = float(np.sqrt(np.mean(angles[keep] ** 2)))
    return dict(source="attitude", up_local=mean.tolist(), cameras=len(ups), used=used,
                spread_deg=round(spread, 4),
                # Per-camera scatter shrinks as sqrt(n) only if gimbal errors are
                # independent; a biased gimbal zero is not reduced at all.
                tilt_std_deg=round(spread / math.sqrt(used), 4),
                assumption="gimbal angles are horizon-referenced and unbiased")


def gravity_from_ground(points, camera_centres, *, inlier_fraction_of_scale=0.01, seed=0):
    """Up in the reconstruction frame as the dominant plane's normal, toward the cameras."""
    points = np.asarray(points, dtype=np.float64)
    centres = np.asarray(camera_centres, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 50:
        raise ValueError("Ground plane needs at least 50 finite 3D points")
    if not np.isfinite(points).all() or centres.ndim != 2 or not len(centres):
        raise ValueError("Ground plane needs finite points and at least one camera centre")
    lo, hi = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    threshold = float(np.linalg.norm(hi - lo)) * inlier_fraction_of_scale
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(PLANE_ITERATIONS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) <= 1e-12:
            continue
        normal /= np.linalg.norm(normal)
        count = int(np.sum(np.abs((points - a) @ normal) <= threshold))
        if count > best_count:
            best, best_count = (normal, a), count
    if best is None:
        raise ValueError("Ground plane RANSAC found no nondegenerate plane")
    normal, anchor = best
    inliers = points[np.abs((points - anchor) @ normal) <= threshold]
    centred = inliers - inliers.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    if np.mean((centres - inliers.mean(axis=0)) @ normal) < 0:
        normal = -normal
    rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))
    return dict(source="ground_plane", up_local=normal.tolist(),
                inlier_fraction=round(len(inliers) / len(points), 4),
                plane_rms_units=rms, tilt_std_deg=None,
                assumption="terrain is level across the track; a cross slope of d degrees "
                           "becomes a cross-track tilt of d degrees")


def _line_model(source, target, weights, up_local):
    """Similarity whose along-track part comes from positions and whose roll comes from up."""
    weights = weights / weights.sum()
    mx, my = weights @ source, weights @ target
    x, y = source - mx, target - my
    sx = np.linalg.svd(x * np.sqrt(weights[:, None]), full_matrices=False)
    sy = np.linalg.svd(y * np.sqrt(weights[:, None]), full_matrices=False)
    if sx[1][0] <= 1e-12 or sy[1][0] <= 1e-12:
        raise ValueError("Track has no extent; cameras are stationary")
    axis_s, axis_t = sx[2][0], sy[2][0]
    if float(np.sum(weights * (x @ axis_s) * (y @ axis_t))) < 0:
        axis_t = -axis_t
    if _angle_deg(axis_t, UP) < MIN_TRACK_TO_VERTICAL_DEG or \
            _angle_deg(axis_t, -UP) < MIN_TRACK_TO_VERTICAL_DEG:
        raise ValueError("Track is near vertical; gravity cannot fix its heading")
    first = _rotation_between(axis_s, axis_t)
    gravity = first @ _unit(up_local, "up_local")
    g_perp = gravity - (gravity @ axis_t) * axis_t
    u_perp = UP - (UP @ axis_t) * axis_t
    if np.linalg.norm(g_perp) <= math.sin(math.radians(MIN_TRACK_TO_VERTICAL_DEG)):
        raise ValueError("Gravity is nearly parallel to the track; it cannot fix the roll")
    roll = math.atan2(float(axis_t @ np.cross(g_perp, u_perp)), float(g_perp @ u_perp))
    rotation = _axis_angle(axis_t, roll) @ first
    scale = float(np.sum(weights[:, None] * (x @ rotation.T) * y) / np.sum(weights[:, None] * x * x))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Line fit produced a nonpositive scale")
    return scale, rotation, my - scale * (rotation @ mx)


def _residuals(source, target, model):
    scale, rotation, translation = model
    return np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)


def align_line_with_gravity(camera_rows, telemetry, gravity, *, max_gap_s=2.0,
                            inlier_threshold_m=3.0):
    """Georeference a near-collinear track: 6 DoF from GPS, roll about the track from gravity.

    Same inlier rule as the Sim3 path (>= 4 inliers AND a strict majority), same output
    schema, so ``survey_georef.transform_points`` and every exporter accept the result.
    """
    threshold = float(inlier_threshold_m)
    up_local = _unit(gravity["up_local"], "gravity up_local")
    source, target, variance, files = georef.camera_positions(camera_rows, telemetry,
                                                              max_gap_s=max_gap_s)
    n = len(source)
    if n < 4:
        raise ValueError("Insufficient overlap: need at least four bracketed/exact cameras")
    source, target, variance = np.asarray(source), np.asarray(target), np.asarray(variance)
    weights = variance.min() / variance
    needed = max(4, n // 2 + 1)
    if math.comb(n, 2) <= MAX_PAIRS:
        pairs = combinations(range(n), 2)
    else:
        rng = np.random.default_rng(0)
        pairs = (rng.choice(n, 2, replace=False) for _ in range(MAX_PAIRS))
    best_mask, best_score = None, (-1, -math.inf)
    for pair in pairs:
        indices = list(pair)
        try:
            model = _line_model(source[indices], target[indices], weights[indices], up_local)
        except ValueError:
            continue
        residuals = _residuals(source, target, model)
        mask = residuals <= threshold
        count = int(mask.sum())
        if count < needed:
            continue
        score = (count, -float(np.average(residuals[mask] ** 2, weights=weights[mask])))
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None:
        raise ValueError("No majority consensus with at least four inliers for the line fit")
    mask = best_mask
    for _ in range(20):
        model = _line_model(source[mask], target[mask], weights[mask], up_local)
        residuals = _residuals(source, target, model)
        updated = residuals <= threshold
        if int(updated.sum()) < needed:
            raise ValueError("Robust line refit lost sufficient inlier consensus")
        if np.array_equal(mask, updated):
            break
        mask = updated
    else:
        raise ValueError("Robust line refit did not converge")
    scale, rotation, translation = model
    fitted_up = rotation @ up_local
    tilt_std = gravity.get("tilt_std_deg")
    cross = _cross_track_extent(source[mask], rotation, scale)
    warnings = [
        "Straight track: roll about the track axis comes from " + gravity["source"]
        + " gravity, not from GPS. Assumption: " + gravity.get("assumption", "none stated") + ".",
        "A cross-track tilt of d degrees moves a point x metres off the track by x*tan(d) "
        "vertically; see cross_track_tilt_deg.",
        "Fit residuals are not independent accuracy validation; GNSS bias remains possible."]
    if n != len(camera_rows):
        warnings.append(f"{len(camera_rows) - n} unmatched cameras omitted (no brackets or gap too large).")
    if int(mask.sum()) != n:
        warnings.append(f"{n - int(mask.sum())} matched cameras rejected as outliers.")
    tilt = None if tilt_std is None else float(tilt_std)
    return dict(
        schema_version=1, status="aligned",
        method="deterministic_ransac_weighted_line_similarity_gravity_roll",
        scale=float(scale), rotation=rotation.tolist(), translation=translation.tolist(),
        coordinate_frame=copy.deepcopy(telemetry["coordinate_frame"]), matched_count=n,
        inlier_count=int(mask.sum()), matched_files=list(files),
        inlier_files=[name for name, keep in zip(files, mask.tolist()) if keep],
        residuals_m=residuals.tolist(), inlier_mask=mask.tolist(),
        fit_rmse_m=float(np.sqrt(np.mean(residuals[mask] ** 2))),
        gravity=dict(gravity, residual_along_track_deg=round(_angle_deg(fitted_up, UP), 4)),
        cross_track_tilt_deg=tilt,
        cross_track_distance_for_1m_error_m=(None if not tilt else
                                             round(1.0 / math.tan(math.radians(tilt)), 1)),
        camera_cross_track_spread_m=cross,
        warnings=warnings, accuracy_validated=False)


def _cross_track_extent(source, rotation, scale):
    mapped = scale * (source @ rotation.T)
    centred = mapped - mapped.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return round(float(singular[1] / math.sqrt(len(source))), 4)


def is_degenerate_track(error):
    """True when survey_georef refused a fit because the track is (near) a line."""
    text = str(error)
    return ("near-collinear" in text or "nondegenerate majority" in text
            or "Robust refit lost" in text)


def tilt_check(alignment, gravity):
    """Angle between the fitted up and gravity on an ordinary Sim3 fit. Diagnostic only."""
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    angle = _angle_deg(rotation @ _unit(gravity["up_local"], "gravity up_local"), UP)
    return dict(source=gravity["source"], angle_deg=round(angle, 4),
                tilt_std_deg=gravity.get("tilt_std_deg"),
                note="Disagreement between the GPS-fitted up and gravity. Large values point at "
                     "clock offset, GPS height noise or a biased gimbal; it does not change the fit.")
