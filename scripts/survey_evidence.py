"""Per-point multi-view evidence: triangulation support and reprojection error.

Support counts how many registered views observe a point and reprojection error
is that point's mean pixel residual. Both describe internal consistency of the
reconstruction only - neither is measured accuracy against surveyed truth.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class EvidenceModel:
    xyz: np.ndarray
    rgb: np.ndarray
    error: np.ndarray
    support: np.ndarray
    confidence: np.ndarray | None = None
    rejected: int = 0


def parse_points3d(path) -> EvidenceModel:
    """Read a COLMAP points3D.txt. Raises ValueError when nothing is usable."""
    xyz, rgb, error, support, rejected = [], [], [], [], 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        try:
            point = [float(v) for v in fields[1:4]]
            color = [int(v) for v in fields[4:7]]
            residual = float(fields[7])
            pairs = fields[8:]
        except (IndexError, ValueError):
            rejected += 1
            continue
        valid = (all(":" in pair for pair in pairs)
                 and all(math.isfinite(v) for v in point + [residual])
                 and all(0 <= c <= 255 for c in color))
        if not valid:
            rejected += 1
            continue
        xyz.append(point)
        rgb.append(color)
        error.append(residual)
        support.append(len(pairs))
    if not xyz:
        raise ValueError(f"No usable points in {path} ({rejected} rows rejected)")
    return EvidenceModel(np.asarray(xyz, float), np.asarray(rgb, np.uint8),
                         np.asarray(error, float), np.asarray(support, np.int32),
                         rejected=rejected)


def support_summary(model: EvidenceModel, *, min_views=3, cell_size_m=1.0) -> dict:
    """Support fraction plus grid occupancy; occupancy is not surface coverage."""
    min_views = int(min_views)
    cell = float(cell_size_m)
    if min_views < 1 or not math.isfinite(cell) or cell <= 0:
        raise ValueError("min_views must be >= 1 and cell_size_m must be positive")
    well_supported = model.support >= min_views
    cells = np.floor(model.xyz / cell).astype(np.int64)
    unique = np.unique(cells, axis=0)
    occupied = np.unique(cells[well_supported], axis=0)
    return {"point_count": int(len(model.xyz)), "min_views": min_views,
            "well_supported_count": int(well_supported.sum()),
            "well_supported_fraction": round(float(well_supported.mean()), 6),
            "cell_size_m": cell, "cell_count": int(len(unique)),
            "well_supported_cell_count": int(len(occupied)),
            "empty_cell_fraction": round(float(1 - len(occupied) / max(len(unique), 1)), 6),
            "mean_reprojection_error_px": round(float(model.error.mean()), 6),
            "max_reprojection_error_px": round(float(model.error.max()), 6),
            "rejected_source_rows": int(model.rejected)}


def support_confidence(model: EvidenceModel, *, min_views=3, max_error_px=1.0) -> np.ndarray:
    """0..1 blend of view support and reprojection residual, not calibrated error."""
    min_views, max_error_px = int(min_views), float(max_error_px)
    if min_views < 1 or not math.isfinite(max_error_px) or max_error_px <= 0:
        raise ValueError("min_views must be >= 1 and max_error_px must be positive")
    views = np.minimum(model.support / min_views, 1.0)
    residual = np.clip(1.0 - model.error / max_error_px, 0.0, 1.0)
    return np.clip(views * residual, 0.0, 1.0).astype(np.float32)


def export_evidence_ply(model: EvidenceModel, confidence, path) -> Path:
    """Write XYZ/RGB plus support, reprojection error and confidence atomically."""
    confidence = np.asarray(confidence, np.float32)
    if confidence.shape != (len(model.xyz),):
        raise ValueError("confidence must hold one value per point")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros(len(model.xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
        ("blue", "u1"), ("support", "i4"), ("reprojection_error_px", "f4"),
        ("confidence", "f4")])
    for name, values in (("x", model.xyz[:, 0]), ("y", model.xyz[:, 1]), ("z", model.xyz[:, 2]),
                         ("red", model.rgb[:, 0]), ("green", model.rgb[:, 1]),
                         ("blue", model.rgb[:, 2]), ("support", model.support),
                         ("reprojection_error_px", model.error), ("confidence", confidence)):
        array[name] = values
    temporary = path.with_name(path.name + ".tmp")
    try:
        PlyData([PlyElement.describe(array, "vertex")]).write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_ply(path) -> EvidenceModel:
    """Read back an evidence PLY (or a plain XYZ/RGB cloud) for verification."""
    data = PlyData.read(path)["vertex"].data
    names = set(data.dtype.names)
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required <= names:
        raise ValueError(f"PLY lacks required properties: {sorted(required - names)}")
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")]).astype(np.float64)
    rgb = np.column_stack([data[k] for k in ("red", "green", "blue")]).astype(np.uint8)
    error = (np.asarray(data["reprojection_error_px"], float) if "reprojection_error_px" in names
             else np.zeros(len(xyz)))
    support = (np.asarray(data["support"], np.int32) if "support" in names
               else np.ones(len(xyz), np.int32))
    confidence = np.asarray(data["confidence"], float) if "confidence" in names else None
    return EvidenceModel(xyz, rgb, error, support, confidence=confidence)
