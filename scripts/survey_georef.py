"""Bounded camera-center GPS georeferencing, using only NumPy and stdlib.

No speed ruler, antenna lever-arm inference, datum conversion, or accuracy claim.
Normalized timestamps are video seconds (input t_sec + declared offset), bounded
by the declared video duration. Invalid inputs/unsafe fits raise ValueError;
filesystem errors propagate. Inputs are never modified.
"""
import copy
import csv
from functools import wraps
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

_FIELDS = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m",
           "horizontal_std_m", "vertical_std_m")
_FRAME = dict(type="ENU", units="m", geodetic_crs="EPSG:4979",
              altitude_datum="ellipsoidal")
_MIN_SECOND_RATIO = 0.01  # Second / first centered singular value; planar is OK.


def _checked(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (KeyError, TypeError, OverflowError, FloatingPointError,
                np.linalg.LinAlgError) as exc:
            raise ValueError(f"Invalid or numerically unsafe georeference: {exc}") from exc
    return call


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (str, int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _version(value):
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be integer 1")


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _rotation(value):
    raw = np.asarray(value)
    if raw.shape == (9,):
        raw = raw.reshape(3, 3)
    result = _array(raw, (3, 3), "rotation")
    if not (np.allclose(result.T @ result, np.eye(3), atol=1e-6, rtol=0)
            and abs(np.linalg.det(result) - 1) <= 1e-6):
        raise ValueError("rotation must be SO(3): orthonormal with determinant +1")
    return result


def _geodetic(row):
    lat, lon, alt = [_number(row[k], k) for k in _FIELDS[1:4]]
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("latitude/longitude outside geodetic ranges")
    return lat, lon, alt


def _variance(row):
    # Scalar isotropic surrogate: h is assumed a per-axis E/N standard deviation.
    h = _number(row["horizontal_std_m"], "horizontal_std_m", positive=True)
    v = _number(row["vertical_std_m"], "vertical_std_m", positive=True)
    h2 = _number(h * h, "horizontal variance", positive=True)
    v2 = _number(v * v, "vertical variance", positive=True)
    return _number((2 * h2 + v2) / 3, "mean variance", positive=True)


@_checked
def validate_metadata(metadata: dict) -> dict:
    """Validate required declarations and return an independent metadata copy."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    _version(metadata["schema_version"])
    for key, required in dict(time_reference="video", altitude_datum="ellipsoidal",
                              position_reference="camera_center").items():
        if metadata[key] != required:
            raise ValueError(f"{key} must declare {required!r}")
    if metadata["single_pass"] is not True:
        raise ValueError("single_pass must be true")
    result = copy.deepcopy(metadata)
    result["time_offset_s"] = _number(metadata["time_offset_s"], "time_offset_s")
    result["video_duration_s"] = _number(metadata["video_duration_s"],
                                          "video_duration_s", positive=True)
    return result


def _json_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field")
    return result


@_checked
def normalize_telemetry(path: Path, metadata: dict) -> dict:
    """Read six-field ellipsoidal GPS CSV/JSONL into video-timed ENU from the first sample."""
    meta = validate_metadata(metadata)
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".jsonl"):
        raise ValueError("Telemetry must be CSV or JSONL")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if not reader.fieldnames or (len(reader.fieldnames) != len(_FIELDS)
                                         or set(reader.fieldnames) != set(_FIELDS)):
                raise ValueError(f"CSV requires exactly these fields: {_FIELDS}")
            rows = list(reader)
        else:
            rows = [json.loads(line, object_pairs_hook=_json_object)
                    for line in stream if line.strip()]
    if not rows:
        raise ValueError("Telemetry is empty")
    geodetic, samples = [], []
    previous_raw = previous_video = -math.inf
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_FIELDS):
            raise ValueError(f"Each telemetry row requires exactly these fields: {_FIELDS}")
        raw_time = _number(row["t_sec"], "t_sec")
        time = _number(raw_time + meta["time_offset_s"], "video t_sec")
        if raw_time <= previous_raw or time <= previous_video:
            raise ValueError("Telemetry times must be strictly increasing, without duplicates")
        if not 0 <= time <= meta["video_duration_s"]:
            raise ValueError("Offset telemetry times must lie within video duration")
        previous_raw, previous_video = raw_time, time
        geodetic.append(_geodetic(row))
        _variance(row)
        samples.append(dict(t_sec=time, position=[],
                            horizontal_std_m=float(row["horizontal_std_m"]),
                            vertical_std_m=float(row["vertical_std_m"])))
    coordinates = np.array(geodetic, dtype=np.float64)
    lat, lon = np.deg2rad(coordinates[:, :2]).T
    alt = coordinates[:, 2]
    a, e2 = 6378137.0, 6.6943799901413165e-3
    radius = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    ecef = np.column_stack(((radius + alt) * np.cos(lat) * np.cos(lon),
                            (radius + alt) * np.cos(lat) * np.sin(lon),
                            (radius * (1 - e2) + alt) * np.sin(lat)))
    slat, clat, slon, clon = np.sin(lat[0]), np.cos(lat[0]), np.sin(lon[0]), np.cos(lon[0])
    basis = np.array([[-slon, clon, 0], [-slat * clon, -slat * slon, clat],
                      [clat * clon, clat * slon, slat]], dtype=np.float64)
    enu = _array((ecef - ecef[0]) @ basis.T, (len(rows), 3), "ENU positions")
    for sample, position in zip(samples, enu):
        sample["position"] = position.tolist()
    origin = dict(zip(_FIELDS[1:4], geodetic[0]))
    return dict(schema_version=1, coordinate_frame=dict(_FRAME, origin=origin), samples=samples)


def _telemetry_arrays(telemetry):
    _version(telemetry["schema_version"])
    frame = telemetry["coordinate_frame"]
    if any(frame[k] != v for k, v in _FRAME.items()):
        raise ValueError("Expected WGS84 ellipsoidal ENU coordinate frame in metres")
    _geodetic(frame["origin"])
    samples = telemetry["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("Telemetry samples must be a nonempty list")
    times = np.array([_number(s["t_sec"], "t_sec") for s in samples])
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Telemetry times must be nonnegative and strictly increasing")
    positions = np.array([_array(s["position"], (3,), "position") for s in samples])
    variances = np.array([_variance(s) for s in samples])
    return times, positions, variances


def _noncollinear(points, *, centered=False):
    matrix = points if centered else points - points.mean(axis=0)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] <= 1e-12 or singular[1] / singular[0] < _MIN_SECOND_RATIO:
        raise ValueError("Degenerate straight/near-collinear trajectory (second/first < 0.01)")


def _fit(source, target, weights):
    _noncollinear(source)
    _noncollinear(target)
    weights = weights / weights.sum()
    mx, my = weights @ source, weights @ target
    x, y = source - mx, target - my
    # SVD the effective weighted geometry directly; recentering changes its rank ratio.
    _noncollinear(x * np.sqrt(weights[:, None]), centered=True)
    _noncollinear(y * np.sqrt(weights[:, None]), centered=True)
    u, singular, vt = np.linalg.svd((y * weights[:, None]).T @ x)
    sign = np.ones(3)
    sign[-1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = _rotation((u * sign) @ vt)
    scale = _number((singular @ sign) / np.sum(weights[:, None] * x * x),
                    "scale", positive=True)
    translation = _array(my - scale * (rotation @ mx), (3,), "translation")
    return scale, rotation, translation


def _residuals(source, target, model):
    scale, rotation, translation = model
    residuals = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    return _array(residuals, (len(source),), "residuals")


@_checked
def align_camera_trajectory(camera_rows: list, telemetry: dict, *,
                            max_gap_s=2.0, inlier_threshold_m=3.0) -> dict:
    """Fit x_enu = scale * rotation @ x_local + translation (column vectors).

    Interpolation requires exact samples or brackets at most max_gap_s apart;
    unmatched cameras are omitted, never extrapolated. Residuals/mask follow the
    matched subset in camera input order. Require >=4 inliers AND strict majority.
    Deterministic RANSAC uses up to 512 triples, then scalar-weighted SVD refits.
    Scalar weights are inverse (2*h_std**2 + v_std**2)/3, assuming h_std is per-axis
    E/N. Interpolated variance is a convex endpoint average, not a claimed gain
    from independent errors. Timing/interpolation error and correlated GNSS bias
    are unmodelled. This is not anisotropic covariance fitting or bundle adjustment.
    fit_rmse_m is UNWEIGHTED inlier fit RMSE, NOT independently measured accuracy.

    matched_files and inlier_files record which camera images entered and stayed
    in the fit, in camera input order, so a reviewer can audit which checkpoints
    were held out. Naming them does not make them independent: whether the held
    out poses, checkpoints or scenes were used anywhere else is declared by the
    caller and is not proven by this module.
    """
    gap = _number(max_gap_s, "max_gap_s", positive=True)
    threshold = _number(inlier_threshold_m, "inlier_threshold_m", positive=True)
    times, positions, variances = _telemetry_arrays(telemetry)
    if not isinstance(camera_rows, list):
        raise ValueError("camera_rows must be a list")
    source, target, variance, files, seen = [], [], [], [], set()
    for row in camera_rows:
        if not isinstance(row["file"], str) or not row["file"]:
            raise ValueError("Camera file must be a nonempty string")
        time = _number(row["t_sec"], "camera t_sec")
        if time < 0 or time in seen:
            raise ValueError("Camera times must be nonnegative and unique")
        seen.add(time)
        rotation = _rotation(row["camera"]["R_rowmajor"])
        center = _array(-rotation.T @ _array(row["camera"]["t"], (3,), "camera t"),
                        (3,), "camera center")
        right = int(np.searchsorted(times, time))
        if right < len(times) and times[right] == time:
            point, var = positions[right], variances[right]
        elif right == 0 or right == len(times) or times[right] - times[right - 1] > gap:
            continue
        else:
            alpha = (time - times[right - 1]) / (times[right] - times[right - 1])
            point = (1 - alpha) * positions[right - 1] + alpha * positions[right]
            var = (1 - alpha) * variances[right - 1] + alpha * variances[right]
        source.append(center)
        target.append(point)
        variance.append(var)
        files.append(row["file"])
    n = len(source)
    if n < 4:
        raise ValueError("Insufficient overlap: need at least four bracketed/exact cameras")
    source, target, variance = np.array(source), np.array(target), np.array(variance)
    # Check observability on hypotheses and consensus, not the contaminated global set.
    weights = variance.min() / variance  # Proportional inverse variance; avoid overflow.
    if np.any(weights <= 0):
        raise ValueError("Uncertainty dynamic range is numerically unsafe")
    needed = max(4, n // 2 + 1)
    if math.comb(n, 3) <= 512:
        triples = combinations(range(n), 3)
    else:
        rng = np.random.default_rng(0)
        triples = (rng.choice(n, 3, replace=False) for _ in range(512))
    best_mask, best_score = None, (-1, -math.inf)
    for triple in triples:
        indices = list(triple)
        try:
            model = _fit(source[indices], target[indices], weights[indices])
            residuals = _residuals(source, target, model)
            mask = residuals <= threshold
            count = int(mask.sum())
            if count < needed:
                continue
            _noncollinear(source[mask])
            _noncollinear(target[mask])
            score = (count, -float(np.average(residuals[mask] ** 2, weights=weights[mask])))
            if score > best_score:
                best_mask, best_score = mask, score
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if best_mask is None:
        raise ValueError("No nondegenerate majority consensus with at least four inliers")
    mask = best_mask
    for _ in range(20):
        model = _fit(source[mask], target[mask], weights[mask])
        residuals = _residuals(source, target, model)
        updated = residuals <= threshold
        if int(updated.sum()) < needed:
            raise ValueError("Robust refit lost sufficient inlier consensus")
        if np.array_equal(mask, updated):
            break
        mask = updated
    else:
        raise ValueError("Robust inlier refit did not converge")
    scale, rotation, translation = model
    inlier_files = [name for name, keep in zip(files, mask.tolist()) if keep]
    warnings = [
        "Fit residuals are not independent accuracy validation; GNSS bias remains possible.",
        "Scalar inverse mean-variance weights assume per-axis horizontal std; no anisotropic "
        "covariance fitting or bundle adjustment. Interpolated variance is conservatively "
        "averaged; timing, motion-model error and temporal correlations are unmodelled.",
        "Independence of the checkpoints held out of this alignment is declared by the caller "
        "and is not proven by this module; matched_files and inlier_files record which cameras "
        "entered and stayed in the fit so the claim can be audited."]
    if n != len(camera_rows):
        warnings.append(f"{len(camera_rows) - n} unmatched cameras omitted (no brackets or gap too large).")
    if int(mask.sum()) != n:
        warnings.append(f"{n - int(mask.sum())} matched cameras rejected as outliers.")
    return dict(schema_version=1, status="aligned", method="deterministic_ransac_weighted_sim3",
                scale=scale, rotation=rotation.tolist(), translation=translation.tolist(),
                coordinate_frame=copy.deepcopy(telemetry["coordinate_frame"]), matched_count=n,
                inlier_count=int(mask.sum()), matched_files=list(files),
                inlier_files=inlier_files, residuals_m=residuals.tolist(),
                inlier_mask=mask.tolist(),
                fit_rmse_m=float(np.sqrt(np.mean(residuals[mask] ** 2))), warnings=warnings,
                accuracy_validated=False)


@_checked
def transform_points(points, alignment) -> np.ndarray:
    """Return a new float64 Nx3 array using a validated positive proper similarity."""
    _version(alignment["schema_version"])
    if alignment["status"] != "aligned":
        raise ValueError("Alignment status must be aligned")
    scale = _number(alignment["scale"], "scale", positive=True)
    rotation = _rotation(alignment["rotation"])
    translation = _array(alignment["translation"], (3,), "translation")
    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("points must have shape Nx3")
    values = _array(raw, raw.shape, "points")
    return _array(scale * (values @ rotation.T) + translation, raw.shape, "transformed points")
