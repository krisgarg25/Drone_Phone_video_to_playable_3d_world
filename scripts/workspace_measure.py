"""Adapter that lets the workspace measure with the real survey engine.

``survey_measure`` is Z-up (ENU) and wants a point cloud to snap clicks onto;
the viewer and its ``sparse_points.json`` are Y-up. This module bridges the two:
it remaps axes, snaps each human click to the nearest measured point, calls the
engine's ``distance`` / ``ground_plane`` + ``height_above`` / ``polygon_area``,
and returns a normalised record (value, unit, uncertainty, validity, snapped
points back in the viewer frame). When the scene has no cloud to stand behind a
measurement, the record comes back invalid rather than fabricated.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import survey_measure as m


def _engine(pts):
    """Viewer Y-up (x, y, z) -> engine Z-up (x, z, y)."""
    a = np.asarray(pts, dtype=float).reshape(-1, 3)
    return np.column_stack([a[:, 0], a[:, 2], a[:, 1]])


def _viewer(pts):
    """Engine Z-up (x, y, z) -> viewer Y-up (x, z, y)."""
    a = np.asarray(pts, dtype=float).reshape(-1, 3)
    return np.column_stack([a[:, 0], a[:, 2], a[:, 1]])


def load_cloud(work):
    """The scene's sparse cloud in the viewer (Y-up) frame, or None if there is none.

    ``measure`` expects a viewer-frame cloud and remaps internally, so this stays
    in the viewer frame rather than the engine frame.
    """
    path = Path(work) / "viewer_assets" / "sparse_points.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    pts = data.get("points") or data.get("xyz")
    if not pts:
        return None
    cloud = np.asarray(pts, dtype=float).reshape(-1, 3)
    return cloud if len(cloud) >= 2 else None


def _uncertainty(record):
    u = record.get("uncertainty") or {}
    plus_m = u.get("plus_minus_m")
    if plus_m is None:
        return None
    return {"m": plus_m, "unit": u.get("unit", "m"), "value": u.get("value"),
            "confidence": u.get("confidence", "1-sigma"), "valid": u.get("valid")}


def _invalid(kind, reason, *, uncertainty=None):
    return {"kind": kind, "value": None, "unit": "m", "valid": False, "reason": reason,
            "components": {}, "snapped": [], "uncertainty": uncertainty}


def _spacing(cloud):
    if len(cloud) < 2:
        return 0.5
    dist, _ = cKDTree(cloud).query(cloud, k=2)
    return float(np.median(dist[:, 1])) or 0.5


def _snap(tree, cloud, point, radius):
    dist, idx = tree.query(point, k=1)
    return (int(idx), float(dist)) if dist <= radius else (None, float(dist))


def measure(kind, cloud_viewer, picked_viewer, *, radius_m=None, quality=None):
    """Measure ``kind`` from viewer-Y-up clicks against a viewer-Y-up cloud."""
    cloud = np.asarray(cloud_viewer, dtype=float)
    cloud = cloud if cloud.ndim == 2 and len(cloud) >= 2 else np.asarray(cloud).reshape(-1, 3)
    cloud_engine = _engine(cloud)
    picked = _engine(picked_viewer)
    spacing = _spacing(cloud_engine)
    radius = float(radius_m) if radius_m is not None else max(0.5, 4.0 * spacing)
    tree = cKDTree(cloud_engine)

    if kind == "distance":
        if len(picked) != 2:
            return _invalid("distance", "distance needs exactly two points")
        i, _ = _snap(tree, cloud_engine, picked[0], radius)
        j, _ = _snap(tree, cloud_engine, picked[1], radius)
        if i is None or j is None or i == j:
            return _invalid("distance", "a click is not on measured geometry within "
                            f"{radius:.2f} m", uncertainty={"m": spacing})
        rec = m.distance(cloud_engine, i, j, quality=quality)
        return {"kind": "distance", "value": rec.get("length_m"), "unit": "m",
                "valid": rec["valid"], "reason": rec.get("reason"),
                "components": {"horizontal_m": rec.get("horizontal_m"),
                               "vertical_m": rec.get("vertical_m"),
                               "rise_m": rec.get("rise_m")},
                "snapped": _viewer(cloud_engine[[i, j]]).tolist(),
                "uncertainty": _uncertainty(rec), "engine": rec}

    if kind == "height":
        if len(picked) != 1:
            return _invalid("height", "height needs exactly one point")
        plane = m.ground_plane(cloud_engine, min_inliers=30, ransac_distance=max(0.1, 2 * spacing),
                               quality=quality)
        if not plane["valid"]:
            return _invalid("height", plane.get("reason") or "no ground plane could be fitted",
                            uncertainty=_uncertainty(plane))
        rec = m.height_above(cloud_engine, plane, picked[0], quality=quality)
        return {"kind": "height", "value": rec.get("value"), "unit": "m",
                "valid": rec["valid"], "reason": rec.get("reason"),
                "components": {"perpendicular_m": rec.get("perpendicular_m"),
                               "plane_elevation_at_target_m": rec.get("plane_elevation_at_target_m")},
                "snapped": _viewer([picked[0]]).tolist(),
                "uncertainty": _uncertainty(rec), "engine": rec,
                "ground": {"elevation_m": plane.get("elevation_m"),
                           "inlier_count": plane.get("inlier_count")}}

    if kind == "area":
        if len(picked) < 3:
            return _invalid("area", "area needs at least three points")
        rec = m.polygon_area(picked[:, :2], quality=quality)  # engine x,y == viewer x,z (horizontal)
        return {"kind": "area", "value": rec.get("value"), "unit": "m²",
                "valid": rec["valid"], "reason": rec.get("reason"),
                "components": {"basis": "horizontal plan area"},
                "snapped": _viewer(picked).tolist(),
                "uncertainty": _uncertainty(rec), "engine": rec}

    if kind == "volume":
        if len(picked) < 3:
            return _invalid("volume", "volume needs a footprint of at least three points")
        plane = m.ground_plane(cloud_engine, min_inliers=30,
                               ransac_distance=max(0.1, 2 * spacing), quality=quality)
        if not plane["valid"]:
            return _invalid("volume", plane.get("reason") or "no ground plane could be fitted",
                            uncertainty=_uncertainty(plane))
        n = np.asarray(plane["normal"], float)
        pt = np.asarray(plane["point_enu"], float)
        if abs(n[2]) < 1e-6:
            return _invalid("volume", "the ground plane is vertical")
        poly = picked[:, :2]                       # engine x,y == viewer x,z
        xy, z = cloud_engine[:, :2], cloud_engine[:, 2]
        ground_z = (float(np.dot(n, pt)) - n[0] * xy[:, 0] - n[1] * xy[:, 1]) / n[2]
        above = z - ground_z
        sel = _in_polygon(xy, poly) & (above > 0.05)
        cell = max(0.2, 2 * spacing)
        value = 0.0
        cells = 0
        if sel.any():
            keys = np.floor((xy[sel] - xy.min(axis=0)) / cell).astype(int)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            heights = above[sel]
            value = float(sum(max(0.0, float(np.median(heights[inv == i]))) for i in range(len(uniq))) * cell * cell)
            cells = int(len(uniq))
        ax, ay = poly[1] - poly[0]
        bx, by = poly[2] - poly[0]
        footprint = float(abs(ax * by - ay * bx) or cell * cell)
        u = m.uncertainty(point_spacing_m=spacing, kind="volume", power=3,
                          size_m=math.sqrt(max(footprint, 1e-6)), n_cells=max(cells, 1))
        return {"kind": "volume", "value": value, "unit": "m³", "valid": True,
                "reason": None if cells else "no measured surface rises above the ground plane inside the footprint",
                "components": {"basis": "median surface height above the fitted ground plane, per grid cell",
                               "cells": cells, "cell_size_m": cell},
                "snapped": _viewer(picked).tolist(),
                "uncertainty": _uncertainty({"uncertainty": u}), "engine": plane}

    return _invalid(kind, f"unknown measurement kind {kind!r}")


def _in_polygon(xy, poly):
    """Vectorised ray-casting point-in-polygon for (N,2) points against a closed ring."""
    x, y = np.asarray(xy, float)[:, 0], np.asarray(xy, float)[:, 1]
    poly = np.asarray(poly, float)
    inside = np.zeros(len(x), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = (yj - yi) * (x - xi) - (xj - xi) * (y - yi)
        between = (y >= np.minimum(yi, yj)) & (y < np.maximum(yi, yj))
        side = np.where(xj != xi, xi + (y - yi) * (xj - xi) / (yj - yi + 1e-12), xi)
        inside ^= between & (x < side)
        j = i
    return inside


def class_summary(semantics, *, cell_m=None):
    """Per-class count, footprint area and mean height from semantics.json.

    ``semantics`` is the ``{coords, rgb, classes}`` payload written by label_semantics.
    Area is a grid-occupancy footprint (count of distinct cells the class covers),
    which is robust to far floaters — a convex hull would be destroyed by a single
    outlier. It is a coarse analytical aid, not a surveyed quantity.
    """
    import label_semantics as ls
    coords = np.asarray(semantics.get("coords", []), float).reshape(-1, 3)
    rgb = np.asarray(semantics.get("rgb", []), float).reshape(-1, 3)
    if len(coords) == 0 or len(rgb) != len(coords):
        return {}
    if len(coords) > 20000:  # a detail poll must not process a 200k cloud every refresh
        keep = np.linspace(0, len(coords) - 1, 20000).astype(int)
        coords, rgb = coords[keep], rgb[keep]
    to_class = {tuple(int(c) for c in color): name for name, color in ls.CLASS_RGB.items()}
    labels = np.array([to_class.get(tuple(int(v) for v in row), "obstacle") for row in rgb])
    ground = float(np.percentile(coords[:, 1], 5)) if len(coords) else 0.0
    cell = float(cell_m) if cell_m else max(0.5, 3.0 * _spacing(coords))
    out = {}
    for name in ls.CLASSES:
        pts = coords[labels == name]
        if len(pts) == 0:
            continue
        keys = np.floor((pts[:, [0, 2]] - coords[:, [0, 2]].min(axis=0)) / cell).astype(np.int64)
        occupied = len(np.unique(keys, axis=0))
        out[name] = {"count": int(len(pts)), "area_m2": round(occupied * cell * cell, 2),
                     "mean_height_m": round(float(np.mean(pts[:, 1] - ground)), 3)}
    return out


def _geometry(points, kind):
    pts = np.asarray(points, float).tolist()
    if kind == "point" or len(pts) == 1:
        return {"type": "Point", "coordinates": pts[0]}
    if kind in {"area", "volume"}:
        ring = pts + ([pts[0]] if pts[0] != pts[-1] else [])
        return {"type": "Polygon", "coordinates": [ring]}
    return {"type": "LineString", "coordinates": pts}


def measurements_geojson(measurements, *, scene=""):
    """A GeoJSON FeatureCollection in the local viewer frame (never faked as WGS84)."""
    features = []
    for mrec in measurements:
        features.append({
            "type": "Feature", "id": f"measure:{mrec.get('id', '')}",
            "properties": {"name": mrec.get("label"), "kind": mrec.get("kind"),
                           "value": mrec.get("value"), "unit": mrec.get("unit"),
                           "uncertainty_m": (mrec.get("uncertainty") or {}).get("m"),
                           "valid": mrec.get("valid"), "reason": mrec.get("reason"),
                           "stale": mrec.get("stale", False), "scene": scene},
            "geometry": _geometry(mrec.get("points", []), mrec.get("kind", "point"))})
    return {"type": "FeatureCollection", "name": f"{scene} measurements",
            "coordinate_reference_system": "local viewer Y-up (x, y, z) — not georeferenced; "
            "no surveyed CRS is claimed for these positions",
            "features": features}


def measurements_csv(measurements):
    """A flat CSV of the measurement set with uncertainty and validity intact."""
    import csv
    import io
    fields = ("name", "kind", "value", "unit", "uncertainty_m", "valid", "reason", "stale")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for mrec in measurements:
        writer.writerow({"name": mrec.get("label"), "kind": mrec.get("kind"),
                         "value": mrec.get("value"), "unit": mrec.get("unit"),
                         "uncertainty_m": (mrec.get("uncertainty") or {}).get("m"),
                         "valid": mrec.get("valid"), "reason": mrec.get("reason"),
                         "stale": mrec.get("stale", False)})
    return stream.getvalue()
