"""CPU-only evaluation of caller-supplied metric-coordinate observations.

Checkpoints MUST be independent surveyed matches, never telemetry fit residuals.
Surface inputs require independent, visibility/area-representative sampling;
point recall is not surface-area coverage. No alignment or sampling is fitted.
"""
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

OFFICIAL_CRITERIA = [
    {"id": "accuracy", "label": "Reconstruction accuracy", "weight": 30},
    {"id": "completeness", "label": "Reconstruction completeness", "weight": 20},
    {"id": "speed", "label": "Processing speed", "weight": 20},
    {"id": "innovation", "label": "Innovation", "weight": 15},
    {"id": "scalability", "label": "Scalability", "weight": 10},
    {"id": "ui", "label": "User interface", "weight": 5},
]
_REQUIRED_STAGES = ("keyframes", "colmap", "poses", "train", "frame", "export")
_ERROR_FIELDS = ("horizontal_rmse_m", "vertical_rmse_m", "rmse_3d_m",
                 "median_3d_m", "p95_3d_m", "max_3d_m")


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected nonempty finite Nx3 points in metres")
    return points


def _number(value):
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        value = float(value)
        if math.isfinite(value) and value >= 0:
            return value
    return None


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def checkpoint_metrics(reconstructed, reference) -> dict:
    """Matched independent checkpoints; XYZ metres, XY horizontal, Z vertical."""
    a, b = _points(reconstructed), _points(reference)
    if a.shape != b.shape:
        raise ValueError("Checkpoints must have equal matched shapes")
    try:
        with np.errstate(over="raise", invalid="raise"):
            delta = a - b
            horizontal = np.hypot(delta[:, 0], delta[:, 1])
            errors = np.hypot(horizontal, delta[:, 2])
    except FloatingPointError as exc:
        raise ValueError("Checkpoint differences or distances are not representable") from exc
    def rms(values):
        scale = float(np.max(np.abs(values)))
        return float(scale * np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0
    # Interpolation avoids summing the two middle values; scaled means avoid overflow.
    median, p95 = np.percentile(errors, [50, 95])
    bias_scale = np.max(np.abs(delta), axis=0)
    bias_scale[bias_scale == 0] = 1
    bias = np.mean(delta / bias_scale, axis=0) * bias_scale
    return {"count": len(a), "horizontal_rmse_m": rms(horizontal),
            "vertical_rmse_m": rms(delta[:, 2]), "rmse_3d_m": rms(errors),
            "median_3d_m": float(median), "p95_3d_m": float(p95),
            "max_3d_m": float(errors.max()), "bias_xyz_m": bias.tolist(),
            "alignment": "none"}


def surface_metrics(reconstructed, reference, *, thresholds=(.1, .25, .5, 1.0),
                    chunk_size=512, max_points=10000) -> dict:
    """Exact bidirectional point NN; O(N*M) work, O(N+M+chunk_size**2) memory.

    max_points caps EACH input (raise explicitly for larger intentional workloads).
    Threshold comparisons are inclusive; inputs are never implicitly subsampled.
    """
    _positive_integer(chunk_size, "chunk_size")
    _positive_integer(max_points, "max_points")
    thresholds = tuple(_number(t) for t in thresholds)
    if not thresholds or any(t is None or t <= 0 for t in thresholds):
        raise ValueError("Thresholds must be finite positive distances")
    a, b = _points(reconstructed), _points(reference)
    if max(len(a), len(b)) > max_points:
        raise ValueError("Point limit exceeded; supply representative samples or raise max_points")
    forward, reverse = np.full(len(a), np.inf), np.full(len(b), np.inf)
    for i in range(0, len(a), chunk_size):
        for j in range(0, len(b), chunk_size):
            delta = a[i:i + chunk_size, None, :] - b[None, j:j + chunk_size, :]
            distances = np.hypot.reduce(delta, axis=2)
            forward[i:i + chunk_size] = np.minimum(forward[i:i + chunk_size], distances.min(axis=1))
            reverse[j:j + chunk_size] = np.minimum(reverse[j:j + chunk_size], distances.min(axis=0))
    rows = []
    for t in thresholds:
        precision, recall = float(np.mean(forward <= t)), float(np.mean(reverse <= t))
        rows.append({"distance_m": t, "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return {"reconstruction_count": len(a), "reference_count": len(b), "thresholds": rows,
            "sampling": "caller_supplied_points", "scope": "reference_points"}


def speed_metrics(report: dict, video_duration_s: float, *, required_stages=None) -> dict:
    """Legacy timing qualification, not a benchmark runner.

    Explicit required_stages must list the caller's ENTIRE geometry pipeline.
    status describes usable full-run timing; official_status applies the 600 +/-
    0.5 s video / strictly <900 s processing gate. exceeds_target means too slow.
    """
    required = _REQUIRED_STAGES if required_stages is None else required_stages
    if isinstance(required, str):
        raise ValueError("required_stages must be a nonempty sequence of unique names")
    required = list(required)
    if not required or any(not isinstance(n, str) or not n.strip() for n in required) or len(set(required)) != len(required):
        raise ValueError("required_stages must be a nonempty sequence of unique names")
    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")
    elapsed, duration = _number(report.get("secs")), _number(video_duration_s)
    result = {"status": "not_evaluated", "official_status": "not_evaluated", "reason": "",
              "elapsed_s": elapsed, "video_duration_s": duration, "processing_ratio": None,
              "required_stages": required, "official_target_s": 900.0, "official_video_duration_s": 600.0}
    if elapsed is None or duration is None or duration <= 0:
        result["reason"] = "Invalid elapsed time or video duration"
        return result
    result["processing_ratio"] = _number(elapsed / duration)
    if result["processing_ratio"] is None:
        result["reason"] = "Processing ratio is not representable"
        return result
    steps = report.get("steps")
    if report.get("status") != "complete" or not isinstance(steps, list):
        result["reason"] = "Report is not complete"
        return result
    for name in required:
        matches = [s for s in steps if isinstance(s, dict) and s.get("name") == name]
        if len(matches) != 1 or matches[0].get("status") != "done" or _number(matches[0].get("secs")) is None:
            result["reason"] = f"Required stage {name} missing, duplicated, not fully executed, or invalid timing"
            return result
    for step in steps:
        if isinstance(step, dict) and step.get("status") == "done":
            stage_time = _number(step.get("secs"))
            if stage_time is None or (stage_time > elapsed and not math.isclose(
                    stage_time, elapsed, rel_tol=1e-9, abs_tol=1e-9)):
                result["reason"] = "Measured stage timing is invalid or exceeds total elapsed time"
                return result
    result["status"] = "measured"
    if abs(duration - 600.0) > .5:
        result["reason"] = "Official gate requires a 600 +/- 0.5 second video"
    else:
        result["official_status"] = "meets_target" if elapsed < 900 else "exceeds_target"
        result["reason"] = "Qualified full run; official processing target is strictly <900 seconds"
    return result


def build_evaluation(*, checkpoints=None, surface=None, speed=None, georeferenced=False) -> dict:
    """Combine metric outputs without inventing scores or official accuracy passes.

    georeferenced=True attests independent checkpoints in a shared metric frame;
    telemetry residual dictionaries are not checkpoint observations.
    """
    criteria = [{**c, "status": "not_evaluated", "reason": "No independent evidence supplied"}
                for c in OFFICIAL_CRITERIA]
    if checkpoints is not None:
        if (not isinstance(checkpoints, dict) or checkpoints.get("alignment") != "none"
                or not (_number(checkpoints.get("count")) or 0) > 0
                or any(_number(checkpoints.get(k)) is None for k in _ERROR_FIELDS)):
            raise ValueError("Expected independent checkpoint_metrics output, never telemetry residuals")
        _points([checkpoints.get("bias_xyz_m")])
        criteria[0].update(status="measured", metrics=dict(checkpoints), georeferenced=bool(georeferenced),
                           reason="Accuracy target <=1m specifies no statistic; no official pass inferred")
        if georeferenced:
            criteria[0]["max_error_within_1m"] = bool(checkpoints["max_3d_m"] <= 1.0)
    if surface is not None:
        if not isinstance(surface, dict) or surface.get("scope") != "reference_points" or not surface.get("thresholds"):
            raise ValueError("Expected surface_metrics point-sample output")
        criteria[1].update(status="measured", metrics=dict(surface),
                           reason="Reference-point agreement only; not surface-area or full-scene coverage")
    if speed is not None:
        fields = {"status", "official_status", "reason", "elapsed_s", "video_duration_s",
                  "processing_ratio", "required_stages", "official_target_s", "official_video_duration_s"}
        if not isinstance(speed, dict) or not fields <= speed.keys():
            raise ValueError("Expected complete speed_metrics output, not a bare status claim")
        required = speed["required_stages"]
        if (not isinstance(required, list) or not required
                or any(not isinstance(n, str) or not n.strip() for n in required)
                or len(set(required)) != len(required)
                or not isinstance(speed["reason"], str)
                or speed["status"] not in ("measured", "not_evaluated")
                or _number(speed["official_target_s"]) != 900.0
                or _number(speed["official_video_duration_s"]) != 600.0):
            raise ValueError("Invalid speed qualification schema")
        for key in ("elapsed_s", "video_duration_s", "processing_ratio"):
            if speed[key] is not None and _number(speed[key]) is None:
                raise ValueError(f"Invalid speed {key}")
        elapsed, duration = _number(speed["elapsed_s"]), _number(speed["video_duration_s"])
        ratio = _number(elapsed / duration) if elapsed is not None and duration else None
        claimed_ratio = _number(speed["processing_ratio"])
        if ((ratio is None) != (claimed_ratio is None)
                or (ratio is not None and not math.isclose(ratio, claimed_ratio, rel_tol=1e-12))):
            raise ValueError("Inconsistent speed processing ratio")
        official = "not_evaluated"
        if speed["status"] == "measured":
            if ratio is None:
                raise ValueError("Measured speed requires valid elapsed time and video duration")
            if abs(duration - 600.0) <= .5:
                official = "meets_target" if elapsed < 900.0 else "exceeds_target"
        if speed["official_status"] != official:
            raise ValueError("Inconsistent official speed gate claim")
        status = speed["status"] if official == "not_evaluated" else official
        criteria[2].update(status=status, metrics=dict(speed), reason=speed["reason"])
    return {"schema_version": 1, "criteria": criteria}


def file_fingerprint(path) -> dict:
    """Streaming manifest entry; basename only, never an absolute source path."""
    path, digest, size = Path(path), hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"name": path.name, "size_bytes": size, "sha256": digest.hexdigest()}
