"""One transform chain per scene: viewer <-> COLMAP <-> ENU <-> WGS84 / UTM / MGRS.

The workspace viewer, the placement engine and the planning editor all work in the
viewer frame: Y-up metres, ``viewer = (colmap @ Rg.T) * scale`` (``solve_frame.py``,
``export_viewer_assets.py``). Survey products live in local ENU, UTM and WGS84. A road
drawn in the viewer has to come out of an export at real coordinates, and a click has
to become an MGRS reference, so this module owns the single chain between them:

    viewer --(/scale, @Rg)--> colmap --(GPS similarity)--> ENU --> WGS84 / UTM / MGRS

The GPS similarity is the one ``solve_frame.gps_scale`` fitted (ruler D). Without it
the scene is ``local``: viewer metres are still usable for lengths and areas when the
scale is metric, but no Earth coordinate is ever produced - a refusal, not a guess.

``status`` is one of ``georeferenced``, ``local_metric``, ``local_estimated`` or
``local_relative``, using the same metric rule ``workspace_api.summary`` applies.
"""
import json
from pathlib import Path

import numpy as np

import survey_coords as coords
import survey_crs as crs
import survey_georef as georef

METRIC_SOURCES = ("AR pose-prior metric path", "GPS telemetry similarity fit")


def _scale_status(value, source):
    if not (isinstance(value, (int, float)) and np.isfinite(value) and value > 0):
        return "relative"
    return "metric" if isinstance(source, str) and source.startswith(METRIC_SOURCES) else "estimated"


def load(work):
    """The scene's transform chain, read from ``frame.json``. Raises if there is none."""
    path = Path(work) / "frame.json"
    frame = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    rotation = np.asarray(frame.get("rotation_rowmajor"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{path}: rotation_rowmajor is not a finite 3x3")
    scale = frame.get("scale_m_per_unit")
    source = frame.get("scale_source") if isinstance(frame.get("scale_source"), str) else None
    status = _scale_status(scale, source)
    alignment = None
    gps = frame.get("scale_anchor_gps")
    if isinstance(gps, dict) and isinstance(gps.get("alignment"), dict) and "refused" not in gps:
        alignment = gps["alignment"]
    registry = {
        "schema_version": 1,
        "viewer": {"up_axis": "Y", "units": "m" if status == "metric" else "scene units x scale"},
        "rotation_rowmajor": rotation.tolist(),
        "scale_m_per_unit": float(scale) if status != "relative" else None,
        "scale_source": source, "scale_status": status,
        "alignment": alignment,
        "status": ("georeferenced" if alignment is not None and status == "metric"
                   else "local_" + status),
    }
    if alignment is not None:
        registry["origin"] = alignment["coordinate_frame"]["origin"]
    return registry


def _points(value):
    points = np.atleast_2d(np.asarray(value, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    return points


def viewer_to_colmap(points, registry):
    if registry["scale_m_per_unit"] is None:
        raise ValueError("the scene has no scale, so viewer coordinates cannot be inverted")
    rotation = np.asarray(registry["rotation_rowmajor"])
    return (_points(points) / registry["scale_m_per_unit"]) @ rotation


def colmap_to_viewer(points, registry):
    rotation = np.asarray(registry["rotation_rowmajor"])
    return (_points(points) @ rotation.T) * registry["scale_m_per_unit"]


def _require_geo(registry):
    if registry["status"] != "georeferenced":
        raise ValueError("scene is " + registry["status"] + ": no GPS similarity fit, so no "
                         "Earth coordinates are produced")


def viewer_to_enu(points, registry):
    _require_geo(registry)
    return georef.transform_points(viewer_to_colmap(points, registry), registry["alignment"])


def enu_to_viewer(points, registry):
    _require_geo(registry)
    a = registry["alignment"]
    local = ((_points(points) - np.asarray(a["translation"])) / float(a["scale"])) @ np.asarray(a["rotation"])
    return colmap_to_viewer(local, registry)


def viewer_to_geodetic(points, registry):
    """Nx3 (lat deg, lon deg, ellipsoidal height m)."""
    origin = registry["origin"]
    return crs.enu_to_geodetic(viewer_to_enu(points, registry), origin["latitude_deg"],
                               origin["longitude_deg"], origin["altitude_m"])


def describe(point, registry, *, sigma_h_m=None, sigma_v_m=None):
    """Every readable form of one viewer-frame point, or its local form with the reason."""
    point = _points(point)[0]
    if registry["status"] != "georeferenced":
        return {"status": registry["status"], "viewer_xyz": point.round(3).tolist(),
                "reason": "no GPS similarity fit for this scene: coordinates are local"}
    lat, lon, height = viewer_to_geodetic(point, registry)[0]
    out = coords.describe_point(lat, lon, height, height_datum="ellipsoidal",
                                sigma_east_m=sigma_h_m, sigma_north_m=sigma_h_m,
                                sigma_up_m=sigma_v_m)
    out["status"] = "georeferenced"
    out["viewer_xyz"] = point.round(3).tolist()
    return out
