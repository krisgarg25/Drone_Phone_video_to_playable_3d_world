"""Occlusion-aware view support: does a surface point survive depth consistency?

COLMAP track length says a feature was matched in N views; it does not say the
observing camera could actually see that surface. This checks each projection
against the dense depth map and rejects points hidden behind the recorded
surface. Support here is measured visibility, still not surveyed accuracy.
"""
import struct
from pathlib import Path

import numpy as np

_INT64_LIMIT = 2 ** 62  # Keep |value| well inside int64 so casts cannot wrap.


def read_depth_map(path):
    """Read a COLMAP binary depth map as a float32 (rows, cols) array."""
    raw = Path(path).read_bytes()
    header = struct.calcsize("<ii")
    if len(raw) < header:
        raise ValueError(f"{path}: too short for a depth map header")
    rows, cols = struct.unpack("<ii", raw[:header])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{path}: invalid depth map size {rows}x{cols}")
    body = raw[header:]
    if len(body) != rows * cols * 4:
        raise ValueError(f"{path}: expected {rows * cols * 4} bytes, found {len(body)}")
    return np.frombuffer(body, dtype=np.float32).reshape(rows, cols).copy()


def _validated(view):
    K = np.asarray(view["K"], dtype=np.float64)
    viewmat = np.asarray(view["viewmat"], dtype=np.float64)
    depth = np.asarray(view["depth"], dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("view K must be a finite 3x3 matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("view K must carry positive focal lengths")
    if viewmat.shape != (4, 4) or not np.isfinite(viewmat).all():
        raise ValueError("view viewmat must be a finite 4x4 matrix")
    rotation = viewmat[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or abs(np.linalg.det(rotation) - 1) > 1e-6:
        raise ValueError("view viewmat rotation must be orthonormal with determinant +1")
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("view depth must be a finite 2-D array")
    return K, viewmat, depth


def _pixel_indices(numerator, offset, name, limit=_INT64_LIMIT):
    """Floor projected coordinates to integer pixels, refusing to wrap them."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        floored = np.floor(numerator + offset)
    if not np.isfinite(floored).all() or np.any(np.abs(floored) >= limit):
        raise ValueError(f"projected pixel {name} is not finite or not representable "
                         "as an integer index")
    return floored.astype(np.int64)


def visible_support(points, views, *, relative_tolerance=0.02) -> np.ndarray:
    """Per-point count of views whose recorded surface agrees with the geometry.

    A view counts when the point projects inside the frame, sits in front of the
    camera, and is no further along the viewing ray than the depth map allows
    (within relative_tolerance of the viewing distance).

    Distance is ||xyz_cam||, the length of the camera-space vector, because COLMAP
    stereo depth maps store depth along the unit ray rather than as camera-space z;
    comparing z instead would call off-axis points that clearly sit behind the
    recorded surface visible. That follows COLMAP's convention as implemented by
    its stereo/depth_map code and should be re-verified against tools/colmap (this
    checkout ships binaries only) before the number is relied on for a claim.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if not len(views):
        raise ValueError("at least one view is required")
    support = np.zeros(len(points), dtype=np.int32)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    for view in views:
        K, viewmat, depth = _validated(view)
        camera = homogeneous @ viewmat.T
        x, y, z = camera[:, 0], camera[:, 1], camera[:, 2]
        rows, cols = depth.shape
        front = np.flatnonzero(z > 0)
        if not len(front):
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled_x = K[0, 0] * x[front] / z[front]
            scaled_y = K[1, 1] * y[front] / z[front]
        u = _pixel_indices(scaled_x, K[0, 2], "column")
        v = _pixel_indices(scaled_y, K[1, 2], "row")
        inside = (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
        if not inside.any():
            continue
        index = front[inside]
        observed = depth[v[inside], u[inside]]
        # Depth along the unit ray, not camera-space z: see the docstring note.
        distance = np.linalg.norm(camera[index, :3], axis=1)
        measurable = observed > 0
        agrees = distance <= observed * (1.0 + tolerance) + tolerance
        support[index[measurable & agrees]] += 1
    return support


def _support_counts(values, name):
    """Validate counts as finite non-negative integers before the int64 cast."""
    try:
        as_float = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold numeric per-point counts") from exc
    if not np.isfinite(as_float).all():
        raise ValueError(f"{name} must contain only finite counts")
    if (np.any(as_float < 0) or np.any(as_float != np.floor(as_float))
            or np.any(as_float >= _INT64_LIMIT)):
        raise ValueError(f"{name} must hold non-negative integer counts representable in int64")
    return as_float.astype(np.int64)


def visibility_summary(support_before, support_after, *, min_views=2) -> dict:
    """How much claimed support disappears once occlusion is enforced."""
    before = _support_counts(support_before, "support_before")
    after = _support_counts(support_after, "support_after")
    if before.shape != after.shape or before.ndim != 1 or not len(before):
        raise ValueError("support arrays must be matching non-empty 1-D vectors")
    if np.any(after > before):
        raise ValueError("visibility-corrected support cannot exceed claimed support")
    min_views = int(min_views)
    if min_views < 1:
        raise ValueError("min_views must be at least 1")
    return {"point_count": int(len(before)), "min_views": min_views,
            "views_before": int(before.sum()), "views_after": int(after.sum()),
            "retained_view_fraction": round(float(after.sum() / max(before.sum(), 1)), 6),
            "points_losing_min_views": round(float(((before >= min_views) & (after < min_views)).mean()), 6),
            "points_fully_occluded": int((after == 0).sum())}
