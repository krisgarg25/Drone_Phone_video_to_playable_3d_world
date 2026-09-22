"""Occlusion-aware view support: does a surface point survive depth consistency?

COLMAP track length says a feature was matched in N views; it does not say the
observing camera could actually see that surface. This checks each projection
against the dense depth map and rejects points hidden behind the recorded
surface. Support here is measured visibility, still not surveyed accuracy.
"""
import struct
from pathlib import Path

import numpy as np


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


def visible_support(points, views, *, relative_tolerance=0.02) -> np.ndarray:
    """Per-point count of views whose recorded surface agrees with the geometry.

    A view counts when the point projects inside the frame, sits in front of the
    camera, and is no further away than the depth map allows (within
    relative_tolerance of the viewing distance).
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
        in_front = z > 0
        u = np.floor(K[0, 0] * x / np.where(in_front, z, 1.0) + K[0, 2]).astype(np.int64)
        v = np.floor(K[1, 1] * y / np.where(in_front, z, 1.0) + K[1, 2]).astype(np.int64)
        inside = in_front & (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
        if not inside.any():
            continue
        observed = depth[v[inside], u[inside]]
        distance = z[inside]
        measurable = observed > 0
        agrees = distance <= observed * (1.0 + tolerance) + tolerance
        support[np.where(inside)[0][measurable & agrees]] += 1
    return support


def visibility_summary(support_before, support_after, *, min_views=2) -> dict:
    """How much claimed support disappears once occlusion is enforced."""
    before = np.asarray(support_before, dtype=np.int64)
    after = np.asarray(support_after, dtype=np.int64)
    if before.shape != after.shape or before.ndim != 1 or not len(before):
        raise ValueError("support arrays must be matching non-empty 1-D vectors")
    if not (np.isfinite(before).all() and np.isfinite(after).all()):
        raise ValueError("support arrays must be finite")
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
