"""Baseline-aware keyframe selection from camera positions (CPU, geometry only).

Dense reconstruction cost scales with the number of images, so the frames chosen
matter more than the frame rate. This spreads a fixed budget evenly along the
flight path and enforces a minimum baseline, which keeps parallax usable while
dropping near-duplicate frames. It measures camera geometry, not scene content:
texture, blur and occlusion are handled elsewhere.
"""
import math

import numpy as np


def _positions(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("positions must be a finite Nx3 array with N >= 3")
    return array


def _params(budget, min_baseline_m):
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 2:
        raise ValueError("budget must be an integer of at least 2")
    baseline = float(min_baseline_m)
    if not math.isfinite(baseline) or baseline < 0:
        raise ValueError("min_baseline_m must be finite and non-negative")
    return int(budget), baseline


def _chosen_indices(positions, chosen):
    indices = np.asarray(chosen, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("chosen must be a non-empty 1-D index array")
    if np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(positions):
        raise ValueError("chosen must be increasing indices inside positions")
    return indices


def _arc(positions):
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def select_keyframes(positions, *, budget, min_baseline_m):
    """Pick at most `budget` frames, evenly spaced along the path, >= baseline apart."""
    positions = _positions(positions)
    budget, baseline = _params(budget, min_baseline_m)
    arc = _arc(positions)
    total = float(arc[-1])
    if total <= 0:
        raise ValueError("positions do not describe a path (all frames coincide)")
    spacing = max(total / (budget - 1), baseline)
    targets = np.arange(0.0, total + spacing / 2.0, spacing)
    targets[-1] = total
    nearest = np.abs(arc[None, :] - targets[:, None]).argmin(axis=1)
    kept = [int(nearest[0])]
    for index in nearest[1:]:
        index = int(index)
        if index <= kept[-1]:
            continue
        if np.linalg.norm(positions[index] - positions[kept[-1]]) >= baseline - 1e-9:
            kept.append(index)
    last = len(positions) - 1
    if kept[-1] != last:
        if np.linalg.norm(positions[last] - positions[kept[-1]]) >= baseline - 1e-9:
            kept.append(last)
        else:
            kept[-1] = last
    return np.asarray(kept, dtype=np.int64)


def path_coverage(positions, chosen, *, min_baseline_m):
    """Arc-length coverage of the flight path around the selected frames."""
    positions = _positions(positions)
    chosen = _chosen_indices(positions, chosen)
    baseline = float(min_baseline_m)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("min_baseline_m must be a positive distance")
    arc = _arc(positions)
    total = float(arc[-1])
    gaps = np.diff(chosen)
    max_gap = float(max((float(arc[c]) - float(arc[p])) for p, c in zip(chosen[:-1], chosen[1:]))) if len(gaps) else 0.0
    intervals = sorted((max(0.0, float(arc[c]) - baseline), min(total, float(arc[c]) + baseline))
                       for c in chosen)
    covered, cursor = 0.0, -math.inf
    for start, end in intervals:
        if end <= cursor:
            continue
        covered += end - max(start, cursor)
        cursor = end
    return {"selected": int(len(chosen)), "max_gap_m": round(max_gap, 6),
            "covered_fraction": round(covered / total if total else 1.0, 6),
            "min_baseline_m": baseline}


def mean_parallax_deg(positions, chosen, target):
    """Mean angle at a scene target between consecutive selected cameras."""
    positions = _positions(positions)
    chosen = _chosen_indices(positions, chosen)
    if len(chosen) < 2:
        raise ValueError("at least two selected frames are required")
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target must be a finite XYZ vector")
    rays = positions[chosen] - target
    angles = []
    for first, second in zip(rays[:-1], rays[1:]):
        norm = np.linalg.norm(first) * np.linalg.norm(second)
        if norm == 0:
            raise ValueError("a selected camera sits exactly on the target")
        angles.append(math.degrees(math.acos(float(np.clip(first @ second / norm, -1.0, 1.0)))))
    return float(np.mean(angles))
