"""The proposal layer: planned roads, buildings, zones, objects and demolitions (Phase 2).

Everything a planner draws lives here, never in measured data. A *proposal* is one
scenario ("Scheme A") of *features* over the reconstructed scene; the scene itself is
the "existing" baseline every proposal is compared against.

Coordinates are the workspace viewer's frame (Y-up metres, see ``scene_frames``): the
editor draws on the collision surface in that frame, so the stored geometry is exactly
what was drawn. Exports convert through ``scene_frames`` to ENU / WGS84 when the scene
is georeferenced, and say "local" when it is not.

The backend owns derived geometry. A road's ribbon draped (or graded) on the measured
ground, a building's extruded prism and roof, a zone's rule verdicts and a proposal's
metrics are all computed here from one source, so what the viewer draws, what the rule
check tests and what the export writes cannot disagree.

Storage: ``work/<scene>/proposals/index.json`` plus one ``<id>.json`` per proposal, each
with a ``revision`` counter that every write must match (optimistic concurrency, so two
tabs cannot silently overwrite each other).
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

import workspace_place

SCHEMA = 1
MAX_PROPOSALS = 50
MAX_FEATURES = 2000
MAX_VERTICES = 512
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FEATURE_TYPES = ("road", "building", "zone", "object", "clip", "symbol", "route", "phase_line")
PROPOSAL_KINDS = ("plan", "mission")
ROOF_TYPES = ("flat", "gable", "hip")
BUILDING_USES = ("residential", "commercial", "mixed", "institutional", "industrial")
USE_COLORS = {"residential": [233, 196, 140], "commercial": [140, 170, 222],
              "mixed": [196, 160, 214], "institutional": [150, 206, 170],
              "industrial": [196, 190, 180]}
ROAD_COLOR, FOOTPATH_COLOR, MEDIAN_COLOR = [60, 64, 72], [176, 174, 168], [110, 150, 96]
ZONE_COLOR, CLIP_COLOR, VIOLATION_COLOR = [80, 190, 230], [228, 70, 70], [230, 60, 60]
CLEARED_SITE_COLOR = [146, 134, 112]
SURFACE_LIFT_M = 0.06
"""Road surfaces float this far above the sampled ground so they do not z-fight it."""
PERSONS_PER_M2 = 1 / 30.0
"""Default residential occupancy for the population estimate; stored per proposal."""
FLOOR_HEIGHT_FOR_EXISTING_M = 3.2

# Urban catalogue: nominal [length, height, depth] in metres and a display colour.
URBAN_OBJECTS = {
    "street_light": ([0.3, 9.0, 0.3], [205, 205, 210]),
    "electric_pole": ([0.35, 11.0, 0.35], [120, 100, 80]),
    "transformer": ([2.0, 2.2, 1.5], [120, 130, 120]),
    "tree_small": ([3.0, 5.0, 3.0], [70, 140, 70]),
    "tree_large": ([7.0, 12.0, 7.0], [50, 120, 55]),
    "bench": ([1.8, 0.8, 0.6], [150, 110, 70]),
    "bus_stop": ([6.0, 2.8, 2.0], [110, 140, 190]),
    "bollard": ([0.25, 1.0, 0.25], [220, 190, 60]),
    "barrier": ([3.0, 1.0, 0.5], [230, 120, 40]),
    "water_tank": ([4.0, 6.0, 4.0], [150, 170, 190]),
    "cell_tower": ([3.0, 30.0, 3.0], [170, 170, 175]),
    "solar_array": ([10.0, 1.2, 4.0], [40, 60, 110]),
    "car": ([4.5, 1.5, 1.8], [180, 60, 60]),
    "truck": ([9.0, 3.2, 2.5], [200, 200, 205]),
    "container": ([12.2, 2.6, 2.4], [60, 110, 160]),
    "tent": ([6.0, 3.0, 4.0], [220, 220, 200]),
    "sandbag_wall": ([4.0, 1.2, 0.8], [180, 160, 120]),
    "checkpoint_booth": ([3.0, 2.6, 3.0], [120, 130, 100]),
    # Inspection access (INF-09).
    "scaffold_bay": ([2.5, 6.0, 1.2], [200, 160, 60]),
    "scissor_lift": ([2.5, 2.5, 1.2], [230, 120, 30]),
    "boom_lift": ([8.0, 16.0, 2.4], [240, 180, 20]),
    # Disaster relief (DIS-07), construction logistics (CON-07) and border surveillance
    # (BOR-05). Nominal sizes; flat pads are 0.1 m high so they read as ground marking.
    "helipad": ([20.0, 0.1, 20.0], [235, 235, 90]),
    "staging_area": ([30.0, 0.1, 30.0], [240, 170, 60]),
    "relief_tent": ([10.0, 3.5, 6.0], [70, 130, 200]),
    "medical_tent": ([8.0, 3.2, 5.0], [230, 240, 240]),
    "water_bladder": ([6.0, 1.5, 4.0], [60, 120, 200]),
    "generator": ([3.0, 2.0, 1.5], [220, 180, 40]),
    "portable_toilet": ([1.2, 2.3, 1.2], [60, 140, 90]),
    "tower_crane": ([4.0, 45.0, 4.0], [240, 190, 30]),
    "site_office": ([6.0, 2.6, 2.4], [200, 200, 190]),
    "material_yard": ([20.0, 0.1, 15.0], [170, 150, 120]),
    "observation_post": ([3.0, 4.0, 3.0], [120, 130, 90]),
    "camera_mast": ([0.4, 8.0, 0.4], [180, 180, 185]),
}


class ProposalError(ValueError):
    """A request the proposal layer refuses; ``status`` is the HTTP code to answer with."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ 2D geometry (x, z)
def _poly(value, name, *, closed_min=3):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ProposalError(400, f"{name} must be a list of finite [x, z] points")
    if len(arr) < closed_min or len(arr) > MAX_VERTICES:
        raise ProposalError(400, f"{name} needs {closed_min}..{MAX_VERTICES} points")
    if np.abs(arr).max() > 1e6:
        raise ProposalError(400, f"{name} is outside the scene")
    if closed_min >= 3 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def polygon_area(poly):
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def contains(poly, points):
    """Even-odd point-in-polygon for Nx2 points."""
    points = np.atleast_2d(points)
    x, z = points[:, 0][:, None], points[:, 1][:, None]
    x1, z1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, z2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((z1 > z) != (z2 > z)) & (x < (x2 - x1) * (z - z1) / (z2 - z1) + x1)
    return np.count_nonzero(crosses, axis=1) % 2 == 1


def distance_to_boundary(poly, points):
    points = np.atleast_2d(points)
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("pij,ij->pi", points[:, None, :] - a[None], ab)
                / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)[None], 0, 1)
    nearest = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(points[:, None, :] - nearest, axis=2), axis=1)


def _self_intersects(poly):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            def orient(p, q, r):
                return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
            if (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0):
                return True
    return False


def triangulate(poly):
    """Ear clipping for a simple polygon; returns index triples into ``poly``."""
    n = len(poly)
    order = list(range(n)) if polygon_area(poly) > 0 else list(range(n))[::-1]
    triangles = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(order) > 3 and guard < 10 * n * n:
        guard += 1
        for k in range(len(order)):
            i, j, l = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = poly[i], poly[j], poly[l]
            if cross(a, b, c) <= 1e-12:
                continue
            others = [poly[m] for m in order if m not in (i, j, l)]
            if others:
                pts = np.asarray(others)
                d1 = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (pts[:, 1] - b[1]) - (c[1] - b[1]) * (pts[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (pts[:, 1] - c[1]) - (a[1] - c[1]) * (pts[:, 0] - c[0])
                if np.any((d1 >= 0) & (d2 >= 0) & (d3 >= 0)):
                    continue
            triangles.append((i, j, l))
            order.pop(k)
            break
        else:
            break
    if len(order) == 3:
        triangles.append(tuple(order))
    return triangles


def min_area_rectangle(points):
    """(centre[2], [length, depth], angle_rad) of the smallest enclosing rectangle."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        lo, hi = points.min(axis=0), points.max(axis=0)
        return (lo + hi) / 2, np.maximum(hi - lo, 0.1), 0.0
    try:
        hull = points[ConvexHull(points).vertices]
    except Exception:
        lo, hi = points.min(axis=0), points.max(axis=0)
        return (lo + hi) / 2, np.maximum(hi - lo, 0.1), 0.0
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        angle = math.atan2(edge[1], edge[0])
        c, s = math.cos(angle), math.sin(angle)
        rot = hull @ np.array([[c, -s], [s, c]])
        lo, hi = rot.min(axis=0), rot.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            centre = ((lo + hi) / 2) @ np.array([[c, s], [-s, c]])
            best = (area, centre, hi - lo, angle)
    return best[1], best[2], best[3]


def rectangle(centre, size, angle):
    c, s = math.cos(angle), math.sin(angle)
    hx, hz = size[0] / 2, size[1] / 2
    corners = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]])
    return corners @ np.array([[c, s], [-s, c]]) + np.asarray(centre)


def densify(line, step):
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


# ------------------------------------------------------------------ ground
class Ground:
    """The scene's measured ground (viewer frame), sampled like the placement engine."""

    def __init__(self, work):
        self.grid = workspace_place._grid(work)

    def sample(self, xz):
        g = self.grid
        xz = np.atleast_2d(xz)
        gx = np.clip((xz[:, 0] - g["ox"]) / g["cell"] - 0.5, 0, g["nx"] - 1.001)
        gz = np.clip((xz[:, 1] - g["oz"]) / g["cell"] - 0.5, 0, g["nz"] - 1.001)
        x0, z0 = np.floor(gx).astype(int), np.floor(gz).astype(int)
        fx, fz = gx - x0, gz - z0
        x1, z1 = np.minimum(x0 + 1, g["nx"] - 1), np.minimum(z0 + 1, g["nz"] - 1)
        floor, ok = g["floor"], g["supported"]
        corners = ((z0, x0, (1 - fx) * (1 - fz)), (z0, x1, fx * (1 - fz)),
                   (z1, x0, (1 - fx) * fz), (z1, x1, fx * fz))
        # Interpolate from measured corners only, so a measured point beside a coverage
        # hole is not dragged toward whatever the grid holds in the hole.
        num = sum(np.where(ok[r, c], w * floor[r, c], 0.0) for r, c, w in corners)
        den = sum(np.where(ok[r, c], w, 0.0) for r, c, w in corners)
        plain = sum(w * floor[r, c] for r, c, w in corners)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.where(den > 1e-9, num / np.maximum(den, 1e-12), plain)
        inside = ((xz[:, 0] >= g["ox"]) & (xz[:, 0] <= g["ox"] + g["nx"] * g["cell"])
                  & (xz[:, 1] >= g["oz"]) & (xz[:, 1] <= g["oz"] + g["nz"] * g["cell"]))
        supported = inside & g["supported"][np.round(gz).astype(int), np.round(gx).astype(int)]
        if not np.isfinite(y).all():
            fallback = np.nanmedian(floor)
            y = np.where(np.isfinite(y), y, fallback)
        return y, supported

    def sample_top(self, xz):
        """Height of the scanned top surface (roofs, crowns) where it exists, else the floor."""
        g = self.grid
        top = g.get("top")
        if top is None:
            return self.sample(xz)
        xz = np.atleast_2d(xz)
        gx = np.clip(np.round((xz[:, 0] - g["ox"]) / g["cell"] - 0.5), 0, g["nx"] - 1).astype(int)
        gz = np.clip(np.round((xz[:, 1] - g["oz"]) / g["cell"] - 0.5), 0, g["nz"] - 1).astype(int)
        floor, supported = self.sample(xz)
        y = top[gz, gx]
        return np.where(np.isfinite(y) & (y > floor), y, floor), supported

    @property
    def cell(self):
        return self.grid["cell"]


# ------------------------------------------------------------------ existing inventory
def _class_rgb():
    try:
        import label_semantics
        return {tuple(v): k for k, v in label_semantics.CLASS_RGB.items()}
    except Exception:
        return {}


def existing_inventory(work, ground, *, cell_m=1.0, min_building_m2=12.0, tree_cell_m=2.0,
                       min_height_m=2.5):
    """Buildings and tree clusters found in the scene's semantic labels (cached).

    Buildings: building-labelled points on a ``cell_m`` occupancy grid, connected
    components, each reduced to its minimum-area rectangle, height = 95th percentile
    point height minus the measured ground under its centre. Trees likewise from
    vegetation. A heuristic inventory from heuristic/learned labels - it says so.
    """
    path = Path(work) / "viewer_assets" / "semantics.json"
    cache = Path(work) / "proposals" / "existing.json"
    if not path.is_file():
        return {"status": "unavailable", "reason": "scene has no semantics.json",
                "buildings": [], "trees": []}
    stat = path.stat()
    signature = [stat.st_size, stat.st_mtime_ns]
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("signature") == signature + [cell_m, min_building_m2, tree_cell_m, min_height_m]:
                return cached
        except (OSError, ValueError):
            pass
    data = json.loads(path.read_text(encoding="utf-8"))
    xyz = np.asarray(data.get("coords") or [], dtype=np.float64).reshape(-1, 3)
    rgb = [tuple(c) for c in data.get("rgb") or []]
    lookup = _class_rgb()
    labels = np.array([lookup.get(c, "unknown") for c in rgb]) if rgb else np.array([])
    signature = signature + [cell_m, min_building_m2, tree_cell_m, min_height_m]
    result = {"status": "measured", "signature": signature, "cell_m": cell_m,
              "basis": "semantic labels (heuristic + learned) clustered on a "
                       f"{cell_m:g} m grid; heights from the measured ground; not surveyed",
              "buildings": [], "trees": []}
    # A footprint under 12 m2 is rarely a building (and on rocks it is a rock pillar);
    # trees cluster on a coarser grid so one crown is not counted as many.
    for kind, name, grid, minimum in (
            ("building", "buildings", cell_m, max(1, int(round(min_building_m2 / cell_m ** 2)))),
            ("vegetation", "trees", tree_cell_m, 2)):
        pts = xyz[labels == kind] if len(labels) == len(xyz) else np.zeros((0, 3))
        if len(pts) < minimum:
            continue
        lo = pts[:, [0, 2]].min(axis=0)
        idx = np.floor((pts[:, [0, 2]] - lo) / grid).astype(int)
        shape = idx.max(axis=0) + 1
        occupied = np.zeros(shape, dtype=bool)
        occupied[idx[:, 0], idx[:, 1]] = True
        components, count = ndimage.label(occupied, structure=np.ones((3, 3)))
        member = components[idx[:, 0], idx[:, 1]]
        for label in range(1, count + 1):
            cells = int((components == label).sum())
            if cells < minimum:
                continue
            group = pts[member == label]
            centre, size, angle = min_area_rectangle(group[:, [0, 2]])
            base = float(ground.sample(centre[None])[0][0])
            height = float(np.percentile(group[:, 1], 95)) - base
            if kind == "building" and height < min_height_m:
                continue
            entry = {"id": f"existing-{kind}-{label}", "centre": centre.round(3).tolist(),
                     "size": size.round(3).tolist(), "angle_rad": round(float(angle), 5),
                     "footprint": rectangle(centre, size, angle).round(3).tolist(),
                     "base_y": round(base, 3), "height_m": round(height, 2),
                     "area_m2": round(cells * grid * grid, 2), "points": int(len(group))}
            if kind == "building":
                entry["floors_estimate"] = max(1, int(round(height / FLOOR_HEIGHT_FOR_EXISTING_M)))
            result[name].append(entry)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


# ------------------------------------------------------------------ features
def _number(params, key, default, lo, hi, name=None):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProposalError(400, f"{name or key} must be a finite number")
    if not lo <= value <= hi:
        raise ProposalError(400, f"{name or key} must be between {lo} and {hi}")
    return float(value)


def _choice(params, key, default, options):
    value = params.get(key, default)
    if value not in options:
        raise ProposalError(400, f"{key} must be one of {', '.join(options)}")
    return value


def validate_feature(feature):
    """Return a clean copy of a feature, or raise ProposalError(400)."""
    if not isinstance(feature, dict):
        raise ProposalError(400, "A feature must be an object")
    kind = feature.get("type")
    if kind not in FEATURE_TYPES:
        raise ProposalError(400, f"Feature type must be one of {', '.join(FEATURE_TYPES)}")
    params = feature.get("params") or {}
    if not isinstance(params, dict):
        raise ProposalError(400, "params must be an object")
    name = feature.get("name") or kind.title()
    if not isinstance(name, str) or len(name) > 120:
        raise ProposalError(400, "name must be a string of at most 120 characters")
    clean = {"type": kind, "name": name.strip() or kind.title(),
             "hidden": bool(feature.get("hidden", False)),
             "locked": bool(feature.get("locked", False))}
    if kind == "road":
        line = _poly(params.get("centerline"), "centerline", closed_min=2)
        length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        if length < 1.0:
            raise ProposalError(400, "A road must be at least 1 m long")
        clean["params"] = {
            "centerline": line.round(4).tolist(),
            "width_m": _number(params, "width_m", 7.0, 2.5, 60.0),
            "lanes": int(_number(params, "lanes", 2, 1, 12)),
            "footpath_m": _number(params, "footpath_m", 1.5, 0.0, 8.0),
            "median_m": _number(params, "median_m", 0.0, 0.0, 20.0),
            "mode": _choice(params, "mode", "drape", ("drape", "graded")),
            "max_grade_pct": _number(params, "max_grade_pct", 6.0, 0.5, 20.0),
            "surface": _choice(params, "surface", "asphalt", ("asphalt", "concrete", "gravel")),
        }
    elif kind == "building":
        poly = _poly(params.get("footprint"), "footprint")
        if abs(polygon_area(poly)) < 4.0:
            raise ProposalError(400, "A building footprint must cover at least 4 m²")
        if _self_intersects(poly):
            raise ProposalError(400, "The footprint crosses itself")
        clean["params"] = {
            "footprint": poly.round(4).tolist(),
            "floors": int(_number(params, "floors", 3, 1, 200)),
            "floor_height_m": _number(params, "floor_height_m", 3.2, 2.4, 8.0),
            "roof": _choice(params, "roof", "flat", ROOF_TYPES),
            "roof_pitch_deg": _number(params, "roof_pitch_deg", 30.0, 5.0, 60.0),
            "use": _choice(params, "use", "residential", BUILDING_USES),
        }
    elif kind == "zone":
        poly = _poly(params.get("polygon"), "polygon")
        if _self_intersects(poly):
            raise ProposalError(400, "The zone boundary crosses itself")
        rules = params.get("rules") or {}
        if not isinstance(rules, dict):
            raise ProposalError(400, "rules must be an object")
        clean_rules = {}
        for key, lo, hi in (("max_height_m", 1, 1000), ("max_fsi", 0.05, 30),
                            ("max_coverage_pct", 1, 100), ("setback_m", 0, 100),
                            ("max_floors", 1, 200)):
            if rules.get(key) is not None:
                clean_rules[key] = _number(rules, key, None, lo, hi)
        clean["params"] = {"polygon": poly.round(4).tolist(), "rules": clean_rules,
                           "label": str(params.get("label") or "Plot")[:60]}
    elif kind == "object":
        item = params.get("item")
        if item not in URBAN_OBJECTS:
            raise ProposalError(400, f"Unknown object item {item!r}")
        position = np.asarray(params.get("position"), dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ProposalError(400, "position must be a finite [x, z]")
        clean["params"] = {"item": item, "position": position.round(4).tolist(),
                           "yaw_deg": _number(params, "yaw_deg", 0.0, -360, 360),
                           "scale": _number(params, "scale", 1.0, 0.1, 10.0)}
        if item == "tower_crane":
            # CON-07: a crane is its jib as much as its mast.
            clean["params"]["jib_radius_m"] = _number(params, "jib_radius_m", 40.0, 5.0, 90.0)
            clean["params"]["clearance_m"] = _number(params, "clearance_m", 3.0, 0.0, 20.0)
    elif kind in ("symbol", "route", "phase_line"):
        import mission
        clean["params"] = mission.validate_params(kind, params)
    else:  # clip
        poly = _poly(params.get("polygon"), "polygon")
        clean["params"] = {"polygon": poly.round(4).tolist(),
                           "reason": str(params.get("reason") or "demolish")[:60]}
    return clean


# ------------------------------------------------------------------ derived geometry
def _mesh(kind, positions, indices, color, opacity=1.0, *, part=None):
    return {"kind": kind, "part": part or kind, "color": color, "opacity": opacity,
            "positions": np.asarray(positions, dtype=np.float64).round(3).ravel().tolist(),
            "indices": np.asarray(indices, dtype=np.int64).ravel().tolist()}


def _ribbon(line, heights, left, right, *, drape=None, lift=0.0):
    """Quad strip between signed offsets ``left`` and ``right`` along a polyline.

    ``heights`` is the level along the centreline (a level cross-section); with a
    ``drape`` Ground each edge vertex instead sits on the ground under it, plus ``lift``.
    """
    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    a, b = line + normal * left, line + normal * right
    if drape is not None:
        # Rest on measured ground only; over unmeasured cells keep the centreline level.
        ga, sa = drape.sample(a)
        gb, sb = drape.sample(b)
        ha, hb = np.where(sa, ga + lift, heights), np.where(sb, gb + lift, heights)
    else:
        ha = hb = heights
    verts = np.vstack([np.column_stack([a[:, 0], ha, a[:, 1]]),
                       np.column_stack([b[:, 0], hb, b[:, 1]])])
    n = len(line)
    tris = []
    for i in range(n - 1):
        tris += [(i, i + 1, n + i), (i + 1, n + i + 1, n + i)]
    return verts, tris, a, b


def _grade(ground_y, step, max_grade):
    """Longitudinal profile no steeper than ``max_grade`` that stays close to the ground."""
    z = ndimage.uniform_filter1d(ground_y, size=max(3, int(round(10.0 / step)) | 1), mode="nearest")
    limit = max_grade * step
    for _ in range(4):
        for i in range(1, len(z)):
            z[i] = np.clip(z[i], z[i - 1] - limit, z[i - 1] + limit)
        for i in range(len(z) - 2, -1, -1):
            z[i] = np.clip(z[i], z[i + 1] - limit, z[i + 1] + limit)
    return z


def derive_road(feature, ground):
    p = feature["params"]
    line = densify(np.asarray(p["centerline"]), max(ground.cell, 0.5))
    step = float(np.mean(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    ground_y, supported = ground.sample(line)
    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    # Unmeasured ground (outside the scan's coverage) is not a surface to drape on: bridge
    # it with a straight profile between the nearest measured stations, and say how much.
    if supported.any() and not supported.all():
        ground_y = np.interp(station, station[supported], ground_y[supported])
    graded = p["mode"] == "graded"
    surface = _grade(ground_y.copy(), step, p["max_grade_pct"] / 100) if graded else ground_y
    lift = surface + SURFACE_LIFT_M
    half = p["width_m"] / 2 + p["median_m"] / 2
    meshes = []
    # Draped: every vertex rests on the ground under it. Graded: a level cross-section
    # at the design profile, whose difference from the ground is the earthwork below.
    drape = None if graded else ground

    def strip(offset_lo, offset_hi, raise_m):
        return _ribbon(line, lift + raise_m, offset_lo, offset_hi, drape=drape,
                       lift=SURFACE_LIFT_M + raise_m)
    verts, tris, _, _ = strip(-half, half, 0.0)
    meshes.append(_mesh("road", verts, tris, ROAD_COLOR, part="carriageway"))
    total_half = half + p["footpath_m"]
    if p["footpath_m"] > 0:
        for lo, hi in ((-total_half, -half), (half, total_half)):
            v, t, _, _ = strip(lo, hi, 0.15)
            meshes.append(_mesh("road", v, t, FOOTPATH_COLOR, part="footpath"))
    if p["median_m"] > 0:
        v, t, _, _ = strip(-p["median_m"] / 2, p["median_m"] / 2, 0.12)
        meshes.append(_mesh("road", v, t, MEDIAN_COLOR, part="median"))
    _, _, outer_left, outer_right = _ribbon(line, lift, -total_half, total_half)
    corridor = np.vstack([outer_left, outer_right[::-1]])
    # Earthwork: every grid point inside the corridor against the profile at its station.
    cut = fill = 0.0
    if graded:
        cell = max(ground.cell, 0.5)
        lo, hi = corridor.min(axis=0), corridor.max(axis=0)
        gx, gz = np.meshgrid(np.arange(lo[0] + cell / 2, hi[0], cell),
                             np.arange(lo[1] + cell / 2, hi[1], cell))
        pts = np.column_stack([gx.ravel(), gz.ravel()])
        pts = pts[contains(corridor, pts)] if len(pts) else pts
        if len(pts):
            nearest = cKDTree(line).query(pts)[1]
            diff = surface[nearest] - ground.sample(pts)[0]
            fill = float(np.clip(diff, 0, None).sum() * cell * cell)
            cut = float(np.clip(-diff, 0, None).sum() * cell * cell)
    length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    slope = np.abs(np.diff(surface)) / np.maximum(np.linalg.norm(np.diff(line, axis=0), axis=1), 1e-9)
    return {"meshes": meshes, "footprint": corridor.round(3).tolist(),
            "metrics": {"length_m": round(length, 2), "width_total_m": round(2 * total_half, 2),
                        "carriageway_area_m2": round(length * 2 * half, 1),
                        "total_area_m2": round(abs(polygon_area(corridor)), 1),
                        "max_grade_pct": round(float(slope.max() * 100) if len(slope) else 0.0, 2),
                        "cut_m3": round(cut, 1), "fill_m3": round(fill, 1),
                        "ground_supported_fraction": round(float(supported.mean()), 3),
                        "profile_bridged_m": round(float(np.sum(np.diff(station)[~(supported[:-1] & supported[1:])])), 1)}}


def _roof(footprint, base_top, p):
    """(vertices, triangles) of the roof; pitched roofs only on four-cornered footprints."""
    if p["roof"] == "flat" or len(footprint) != 4:
        tris = triangulate(footprint)
        verts = np.column_stack([footprint[:, 0], np.full(len(footprint), base_top), footprint[:, 1]])
        return verts, tris, 0.0
    centre, size, angle = min_area_rectangle(footprint)
    length, depth = (size[0], size[1]) if size[0] >= size[1] else (size[1], size[0])
    if size[0] < size[1]:
        angle += math.pi / 2
    rise = depth / 2 * math.tan(math.radians(p["roof_pitch_deg"]))
    rect = rectangle(centre, [length, depth], angle)
    axis = np.array([math.cos(angle), math.sin(angle)])
    inset = depth / 2 if p["roof"] == "hip" else 0.0
    ridge = np.array([centre - axis * (length / 2 - inset), centre + axis * (length / 2 - inset)])
    verts = np.vstack([np.column_stack([rect[:, 0], np.full(4, base_top), rect[:, 1]]),
                       np.column_stack([ridge[:, 0], np.full(2, base_top + rise), ridge[:, 1]])])
    # rect corners: 0(-,-) 1(+,-) 2(+,+) 3(-,+) along (axis, normal); ridge 4 at -axis, 5 at +axis
    tris = [(0, 1, 5), (0, 5, 4), (2, 3, 4), (2, 4, 5), (1, 2, 5), (3, 0, 4)]
    return verts, tris, rise


def derive_building(feature, ground):
    p = feature["params"]
    footprint = np.asarray(p["footprint"])
    samples = np.vstack([footprint, footprint.mean(axis=0)[None]])
    ys, supported = ground.sample(samples)
    base = float(ys.min())
    height = p["floors"] * p["floor_height_m"]
    top = base + height
    n = len(footprint)
    ccw = footprint if polygon_area(footprint) > 0 else footprint[::-1]
    walls_v = np.vstack([np.column_stack([ccw[:, 0], np.full(n, base), ccw[:, 1]]),
                         np.column_stack([ccw[:, 0], np.full(n, top), ccw[:, 1]])])
    walls_t = []
    for i in range(n):
        j = (i + 1) % n
        walls_t += [(i, n + j, j), (i, n + i, n + j)]
    roof_v, roof_t, rise = _roof(ccw, top, p)
    color = USE_COLORS[p["use"]]
    area = abs(polygon_area(footprint))
    meshes = [_mesh("building", walls_v, walls_t, color, part="walls"),
              _mesh("building", roof_v, roof_t, [min(255, c + 18) for c in color], part="roof")]
    return {"meshes": meshes, "footprint": footprint.round(3).tolist(),
            "metrics": {"footprint_area_m2": round(area, 1), "floors": p["floors"],
                        "height_m": round(height + rise, 2), "eave_height_m": round(height, 2),
                        "gfa_m2": round(area * p["floors"], 1),
                        "volume_m3": round(area * height, 1), "base_y": round(base, 3),
                        "ground_supported_fraction": round(float(supported.mean()), 3)}}


def _outline(poly, ground, width, color, lift, kind):
    closed = np.vstack([poly, poly[:1]])
    line = densify(closed, max(ground.cell, 0.5))
    ys = ground.sample(line)[0] + lift
    verts, tris, _, _ = _ribbon(line, ys, -width / 2, width / 2)
    return _mesh(kind, verts, tris, color, 0.9, part="outline")


def derive_zone(feature, ground):
    poly = np.asarray(feature["params"]["polygon"])
    return {"meshes": [_outline(poly, ground, 0.4, ZONE_COLOR, 0.25, "zone")],
            "footprint": poly.round(3).tolist(),
            "metrics": {"area_m2": round(abs(polygon_area(poly)), 1)}}


def _box(x0, x1, y0, y1, z0, z1):
    """Closed box in local (u = length, y, v = depth) coordinates."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
         (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
    t = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        t += [(i, j, 4 + j), (i, 4 + j, 4 + i)]
    return np.asarray(v, dtype=np.float64), t


def _prism(radius, y0, y1, sides=8, cx=0.0, cz=0.0, top_radius=None):
    """Closed n-gon prism (or frustum/cone when ``top_radius`` differs)."""
    top_radius = radius if top_radius is None else top_radius
    a = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.column_stack([np.cos(a), np.sin(a)])
    v = np.vstack([np.column_stack([cx + radius * ring[:, 0], np.full(sides, y0), cz + radius * ring[:, 1]]),
                   np.column_stack([cx + top_radius * ring[:, 0], np.full(sides, y1), cz + top_radius * ring[:, 1]]),
                   [[cx, y0, cz], [cx, y1, cz]]])
    t = []
    for i in range(sides):
        j = (i + 1) % sides
        t += [(i, j, sides + j), (i, sides + j, sides + i), (2 * sides, j, i), (2 * sides + 1, sides + i, sides + j)]
    return v, t


def _ellipsoid(rx, ry, rz, cy, rings=5, sides=8):
    """Low-poly ellipsoid (a tree crown) centred at (0, cy, 0)."""
    v, t = [[0.0, cy + ry, 0.0]], []
    for r in range(1, rings):
        phi = math.pi * r / rings
        for s in range(sides):
            th = 2 * math.pi * s / sides + (math.pi / sides) * (r % 2)
            v.append([rx * math.sin(phi) * math.cos(th), cy + ry * math.cos(phi), rz * math.sin(phi) * math.sin(th)])
    v.append([0.0, cy - ry, 0.0])
    bottom = len(v) - 1
    for s in range(sides):
        t.append((0, 1 + (s + 1) % sides, 1 + s))
        t.append((bottom, 1 + (rings - 2) * sides + s, 1 + (rings - 2) * sides + (s + 1) % sides))
    for r in range(rings - 2):
        a0, b0 = 1 + r * sides, 1 + (r + 1) * sides
        for s in range(sides):
            s1 = (s + 1) % sides
            t += [(a0 + s, a0 + s1, b0 + s), (a0 + s1, b0 + s1, b0 + s)]
    return np.asarray(v), t


def _tree(size):
    """Trunk plus a two-lobed crown, sized to the catalogue box."""
    w, h, d = size
    trunk_h = 0.38 * h
    parts = [(*_prism(max(0.08, 0.05 * w), 0.0, trunk_h + 0.1 * h, 7), [104, 76, 50], "trunk")]
    crown_h = h - trunk_h
    v, t = _ellipsoid(w / 2, crown_h * 0.5, d / 2, trunk_h + crown_h * 0.5)
    parts.append((v, t, [64, 132, 66], "crown"))
    v, t = _ellipsoid(w * 0.32, crown_h * 0.28, d * 0.32, trunk_h + crown_h * 0.72, rings=4, sides=7)
    parts.append((v + [w * 0.08, 0, -d * 0.06], t, [84, 156, 78], "crown"))
    return parts


def _street_light(size):
    """Pole, outreach arm and lamp head; the arm points along local +u."""
    h = size[1]
    arm = max(1.2, 0.18 * h)
    return [(*_prism(0.2, 0.0, 0.5, 8), [150, 150, 156], "base"),
            (*_prism(0.09, 0.0, h, 8, top_radius=0.06), [188, 190, 196], "pole"),
            (*_box(0.0, arm, h - 0.12, h - 0.02, -0.04, 0.04), [188, 190, 196], "arm"),
            (*_box(arm - 0.55, arm + 0.1, h - 0.22, h - 0.08, -0.16, 0.16), [70, 72, 80], "head"),
            (*_box(arm - 0.5, arm + 0.05, h - 0.24, h - 0.22, -0.13, 0.13), [255, 236, 160], "lamp")]


def _car(size):
    """Body, cabin with glass band, four wheels: a readable car at planning scale."""
    L, H, W = size
    parts = [(*_box(-L / 2, L / 2, 0.32, 0.32 + 0.42 * H, -W / 2, W / 2), None, "body"),
             (*_box(-0.28 * L, 0.18 * L, 0.32 + 0.42 * H, H, -0.44 * W, 0.44 * W), [60, 76, 96], "cabin")]
    for u in (-0.32 * L, 0.3 * L):
        for v in (-W / 2 + 0.02, W / 2 - 0.26):
            box = _box(u - 0.34, u + 0.34, 0.0, 0.64, v, v + 0.24)
            parts.append((*box, [28, 28, 30], "wheel"))
    return parts


# Detailed meshes for the most-used catalogue items; everything else stays a box of its
# nominal size. Add an item by writing a generator here: it receives the scaled
# [length, height, depth] and returns (vertices, triangles, colour-or-None, part) in
# local coordinates (u along length, y up from the ground, v along depth).
OBJECT_MODELS = {"tree_small": _tree, "tree_large": _tree, "street_light": _street_light, "car": _car}


def derive_object(feature, ground):
    p = feature["params"]
    size, color = URBAN_OBJECTS[p["item"]]
    size = np.asarray(size) * p["scale"]
    centre = np.asarray(p["position"])
    angle = math.radians(p["yaw_deg"])
    footprint = rectangle(centre, [size[0], size[2]], angle)
    base = float(ground.sample(np.vstack([footprint, centre[None]]))[0].min())
    model = OBJECT_MODELS.get(p["item"])
    parts = model(size) if model else [(*_box(-size[0] / 2, size[0] / 2, 0.0, size[1], -size[2] / 2, size[2] / 2), None, "object")]
    # Same local->world map as ``rectangle``: (u, v) @ [[c, s], [-s, c]] + centre.
    c, s = math.cos(angle), math.sin(angle)
    meshes = []
    for verts, tris, part_color, part in parts:
        world = np.column_stack([verts[:, 0] * c - verts[:, 2] * s + centre[0], verts[:, 1] + base,
                                 verts[:, 0] * s + verts[:, 2] * c + centre[1]])
        meshes.append(_mesh("object", world, tris, part_color or color, part=part))
    metrics = {"height_m": round(float(size[1]), 2), "base_y": round(base, 3)}
    if p["item"] == "tower_crane":
        meshes += _crane_swing(p, centre, base, float(size[1]), angle, ground, metrics)
    return {"meshes": meshes, "footprint": footprint.round(3).tolist(), "metrics": metrics}


def _crane_swing(p, centre, base, height, angle, ground, metrics):
    """Jib, counter-jib and the swing circle; scanned structures inside the circle that reach
    within ``clearance_m`` of the jib are conflicts (CON-07). The scan's top surface is the
    test, so a structure the flight did not see is not checked, and proposed buildings are
    reported by the plan's own rules, not here."""
    radius, clear = p["jib_radius_m"], p["clearance_m"]
    jib_y = base + height - 1.5
    c, s = math.cos(angle), math.sin(angle)
    step = max(ground.cell, radius / 150.0)
    xs = np.arange(-radius, radius + step, step)
    gx, gz = np.meshgrid(xs, xs)
    inside = gx ** 2 + gz ** 2 <= radius ** 2
    pts = np.column_stack([gx[inside] + centre[0], gz[inside] + centre[1]])
    mast = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) < 3.0
    top, supported = ground.sample_top(pts)
    hit = supported & ~mast & (top > jib_y - clear)
    metrics.update({"jib_radius_m": radius, "jib_height_m": round(jib_y - base, 2), "clearance_m": clear,
                    "swing_conflict_m2": round(float(hit.sum()) * step * step, 1),
                    "swing_highest_obstacle_m": round(float(top[hit].max() - base), 2) if hit.any() else None,
                    "swing_unobserved_pct": round(100.0 * float((~supported).mean()), 1)})
    if hit.any():
        k = int(np.argmax(np.where(hit, top, -np.inf)))
        metrics["swing_conflict_at"] = [round(float(pts[k, 0]), 2), round(float(pts[k, 1]), 2)]
    parts = []
    jib = _box(-0.12 * radius, radius, height - 2.0, height - 1.0, -0.6, 0.6)
    parts.append((*jib, [240, 190, 30], "jib"))
    ring_v, ring_t = [], []
    sides = 96
    for k in range(sides):
        a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
        for r0, r1 in ((radius - 0.25, radius + 0.25),):
            b = len(ring_v)
            ring_v += [[r0 * math.cos(a0), 0, r0 * math.sin(a0)], [r1 * math.cos(a0), 0, r1 * math.sin(a0)],
                       [r1 * math.cos(a1), 0, r1 * math.sin(a1)], [r0 * math.cos(a1), 0, r0 * math.sin(a1)]]
            ring_t += [[b, b + 1, b + 2], [b, b + 2, b + 3], [b, b + 2, b + 1], [b, b + 3, b + 2]]
    ring_colour = [220, 50, 50] if hit.any() else [240, 190, 30]
    meshes = []
    for verts, tris, colour, part in parts:
        world = np.column_stack([verts[:, 0] * c - verts[:, 2] * s + centre[0], verts[:, 1] + base,
                                 verts[:, 0] * s + verts[:, 2] * c + centre[1]])
        meshes.append(_mesh("object", world, tris, colour, part=part))
    rv = np.asarray(ring_v, float)
    world = np.column_stack([rv[:, 0] + centre[0], np.full(len(rv), jib_y), rv[:, 2] + centre[1]])
    meshes.append(_mesh("object", world, np.asarray(ring_t), ring_colour, 0.8, part="swing"))
    return meshes


def _obb(poly, lo, hi):
    centre, size, angle = min_area_rectangle(poly)
    return {"cx": round(float(centre[0]), 3), "cz": round(float(centre[1]), 3),
            "hx": round(float(size[0]) / 2, 3), "hz": round(float(size[1]) / 2, 3),
            "angle": round(float(angle), 6), "y_min": round(float(lo), 3), "y_max": round(float(hi), 3)}


def derive_clip(feature, ground, top_y):
    poly = np.asarray(feature["params"]["polygon"])
    ys = ground.sample(poly)[0]
    lo, hi = float(ys.min()) - 1.0, top_y
    n = len(poly)
    ccw = poly if polygon_area(poly) > 0 else poly[::-1]
    verts = np.vstack([np.column_stack([ccw[:, 0], np.full(n, lo), ccw[:, 1]]),
                       np.column_stack([ccw[:, 0], np.full(n, hi), ccw[:, 1]])])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, n + j, j), (i, n + i, n + j)]
    tris += [(n + a, n + b, n + c) for a, b, c in triangulate(ccw)]
    # The cleared site: hiding the structure also hides whatever splats sat at its foot,
    # and a scan never saw the ground under a roof, so draw bare ground in their place.
    site_y = ground.sample(ccw)[0] + 0.03
    site = np.column_stack([ccw[:, 0], site_y, ccw[:, 1]])
    return {"meshes": [_mesh("clip", verts, tris, CLIP_COLOR, 0.16),
                       _mesh("clip", site, triangulate(ccw), CLEARED_SITE_COLOR, part="site")],
            "footprint": poly.round(3).tolist(),
            "clip": {"polygon": poly.round(3).tolist(), "y_min": round(lo, 3), "y_max": round(hi, 3),
                     # The viewer hides splats inside this oriented box on the GPU, so the
                     # demolished volume is the polygon's minimum-area rectangle.
                     "obb": _obb(poly, lo, hi)},
            "metrics": {"area_m2": round(abs(polygon_area(poly)), 1)}}


# ------------------------------------------------------------------ evaluation
def _intersects(poly_a, poly_b, step=0.5):
    """Approximate overlap area of two polygons by sampling the smaller one's box."""
    a, b = np.asarray(poly_a), np.asarray(poly_b)
    lo = np.maximum(a.min(axis=0), b.min(axis=0))
    hi = np.minimum(a.max(axis=0), b.max(axis=0))
    if np.any(hi <= lo):
        return 0.0
    gx, gz = np.meshgrid(np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step))
    pts = np.column_stack([gx.ravel(), gz.ravel()])
    if not len(pts):
        return 0.0
    return float(np.count_nonzero(contains(a, pts) & contains(b, pts)) * step * step)


def evaluate(proposal, ground, existing, *, scale_status="metric", north=None, site=None):
    """Derived geometry, rule verdicts, impacts and the before/after metric table.

    ``north`` is the viewer-frame [x, z] of true north (scene north when local); mission
    symbols turn their bearings into facing vectors with it. ``site`` is
    ``urban_analysis.surface_layers`` (impervious / green cover rows when labels exist).
    """
    features = [f for f in proposal["features"] if not f.get("hidden")]
    derived = {}
    for f in features:
        kind = f["type"]
        if kind in ("symbol", "route", "phase_line"):
            import mission
            derived[f["id"]] = mission.derive(f, ground, np.asarray(north if north is not None else [0.0, -1.0]))
        elif kind == "road":
            derived[f["id"]] = derive_road(f, ground)
        elif kind == "building":
            derived[f["id"]] = derive_building(f, ground)
        elif kind == "zone":
            derived[f["id"]] = derive_zone(f, ground)
        elif kind == "object":
            derived[f["id"]] = derive_object(f, ground)
        else:
            # A demolition is as tall as what it removes (+3 m for roof clutter), so the
            # hidden volume never eats a tree canopy or hillside far above it.
            poly = np.asarray(f["params"]["polygon"])
            inside = [b["base_y"] + b["height_m"] for b in existing.get("buildings", [])
                      if contains(poly, np.asarray(b["centre"])[None])[0]]
            ceiling = (max(inside) + 3.0) if inside else float(ground.sample(poly)[0].max()) + 30.0
            derived[f["id"]] = derive_clip(f, ground, ceiling)
    clips = [np.asarray(f["params"]["polygon"]) for f in features if f["type"] == "clip"]
    demolished = [b for b in existing.get("buildings", [])
                  if any(contains(c, np.asarray(b["centre"])[None])[0] for c in clips)]
    demolished_ids = {b["id"] for b in demolished}
    new_buildings = [f for f in features if f["type"] == "building"]
    roads = [f for f in features if f["type"] == "road"]

    # Impacts: what each road and building would displace in the existing scene.
    impacts = {}
    for f in roads + new_buildings:
        poly = derived[f["id"]]["footprint"]
        hit = [b["id"] for b in existing.get("buildings", [])
               if b["id"] not in demolished_ids and _intersects(poly, b["footprint"]) > 1.0]
        trees = [t["id"] for t in existing.get("trees", [])
                 if contains(np.asarray(poly), np.asarray(t["centre"])[None])[0]]
        impacts[f["id"]] = {"existing_buildings_hit": hit, "trees_removed": trees}
        derived[f["id"]]["metrics"]["existing_buildings_hit"] = len(hit)
        derived[f["id"]]["metrics"]["trees_removed"] = len(trees)

    # Rules: each zone's limits over the new buildings whose centre is inside it.
    violations = []
    for zone in (f for f in features if f["type"] == "zone"):
        poly = np.asarray(zone["params"]["polygon"])
        rules = zone["params"]["rules"]
        plot_area = abs(polygon_area(poly))
        inside = [b for b in new_buildings
                  if contains(poly, np.asarray(b["params"]["footprint"]).mean(axis=0)[None])[0]]
        coverage = sum(derived[b["id"]]["metrics"]["footprint_area_m2"] for b in inside)
        gfa = sum(derived[b["id"]]["metrics"]["gfa_m2"] for b in inside)
        zone_metrics = derived[zone["id"]]["metrics"]
        zone_metrics.update({"buildings": len(inside),
                             "coverage_pct": round(100 * coverage / plot_area, 1) if plot_area else None,
                             "fsi": round(gfa / plot_area, 3) if plot_area else None})
        checks = []

        def check(rule, value, limit, subject, unit, higher_is_worse=True):
            ok = value <= limit if higher_is_worse else value >= limit
            checks.append({"rule": rule, "value": round(value, 3), "limit": limit, "unit": unit,
                           "subject": subject, "ok": bool(ok)})
            if not ok:
                violations.append({"zone": zone["id"], "feature": subject, "rule": rule,
                                   "value": round(value, 3), "limit": limit, "unit": unit})
        if "max_fsi" in rules:
            check("max_fsi", gfa / plot_area, rules["max_fsi"], zone["id"], "")
        if "max_coverage_pct" in rules:
            check("max_coverage_pct", 100 * coverage / plot_area, rules["max_coverage_pct"], zone["id"], "%")
        for b in inside:
            m = derived[b["id"]]["metrics"]
            if "max_height_m" in rules:
                check("max_height_m", m["height_m"], rules["max_height_m"], b["id"], "m")
            if "max_floors" in rules:
                check("max_floors", m["floors"], rules["max_floors"], b["id"], "floors")
            if "setback_m" in rules:
                setback = float(distance_to_boundary(poly, np.asarray(b["params"]["footprint"])).min())
                outside = not contains(poly, np.asarray(b["params"]["footprint"])).all()
                check("setback_m", -setback if outside else setback, rules["setback_m"], b["id"], "m",
                      higher_is_worse=False)
        zone_metrics["checks"] = checks
    for crane in (f for f in features if f["type"] == "object" and f["params"]["item"] == "tower_crane"):
        m = derived[crane["id"]]["metrics"]
        if m.get("swing_conflict_m2"):
            violations.append({"zone": None, "feature": crane["id"], "rule": "crane_swing_clearance",
                               "value": m["swing_highest_obstacle_m"], "limit": round(m["jib_height_m"] - m["clearance_m"], 2),
                               "unit": "m"})
    violating = {v["feature"] for v in violations}
    for fid in violating:
        for mesh in derived.get(fid, {}).get("meshes", []):
            mesh["violation"] = True

    ex_b = existing.get("buildings", [])
    kept = [b for b in ex_b if b["id"] not in demolished_ids]
    removed_trees = {t for imp in impacts.values() for t in imp["trees_removed"]}
    hit_buildings = {b for imp in impacts.values() for b in imp["existing_buildings_hit"]}
    tree_area = {t["id"]: t["area_m2"] for t in existing.get("trees", [])}
    new_res_gfa = sum(derived[b["id"]]["metrics"]["gfa_m2"] for b in new_buildings
                      if b["params"]["use"] in ("residential", "mixed"))
    existing_gfa = sum(b["area_m2"] * b["floors_estimate"] for b in ex_b)
    kept_gfa = sum(b["area_m2"] * b["floors_estimate"] for b in kept)
    density = proposal.get("settings", {}).get("persons_per_m2", PERSONS_PER_M2)

    def row(metric, unit, before, after, note=None):
        entry = {"metric": metric, "unit": unit, "existing": round(before, 1),
                 "proposal": round(after, 1), "change": round(after - before, 1)}
        if note:
            entry["note"] = note
        return entry
    new_area = sum(derived[b["id"]]["metrics"]["footprint_area_m2"] for b in new_buildings)
    road_area = sum(derived[r["id"]]["metrics"]["total_area_m2"] for r in roads)
    canopy_before = sum(tree_area.values())
    canopy_after = canopy_before - sum(tree_area[t] for t in removed_trees)
    table = [
        row("Buildings", "count", len(ex_b), len(kept) + len(new_buildings)),
        row("Built-up footprint", "m²", sum(b["area_m2"] for b in ex_b),
            sum(b["area_m2"] for b in kept) + new_area),
        row("Gross floor area", "m²", existing_gfa,
            kept_gfa + sum(derived[b["id"]]["metrics"]["gfa_m2"] for b in new_buildings),
            "existing floors estimated from height / 3.2 m"),
        row("Tallest building", "m", max((b["height_m"] for b in ex_b), default=0.0),
            max([b["height_m"] for b in kept] + [derived[b["id"]]["metrics"]["height_m"]
                                                 for b in new_buildings], default=0.0)),
        row("New road surface", "m²", 0.0, road_area),
        row("Tree canopy", "m²", canopy_before, canopy_after),
        row("Trees", "count", len(tree_area), len(tree_area) - len(removed_trees)),
        row("Residents (estimate)", "people", 0.0, new_res_gfa * density,
            f"new residential/mixed GFA x {density:.3f} persons per m²"),
        row("Earthwork cut", "m³", 0.0, sum(derived[r["id"]]["metrics"]["cut_m3"] for r in roads)),
        row("Earthwork fill", "m³", 0.0, sum(derived[r["id"]]["metrics"]["fill_m3"] for r in roads)),
    ]
    import urban_analysis
    gfa_by_use = {}
    for b in new_buildings:
        use = b["params"]["use"]
        gfa_by_use[use] = gfa_by_use.get(use, 0.0) + derived[b["id"]]["metrics"]["gfa_m2"]
    ecs, _ = urban_analysis.parking_demand(gfa_by_use, proposal.get("settings"))
    table.append(row("Parking demand (new)", "ECS", 0.0, ecs,
                     "equivalent car spaces per 100 m² of new floor area by use (MPD-2021-style defaults)"))
    if site and site.get("status") == "measured" and site.get("site_observed_m2"):
        area = site["site_observed_m2"]
        gone = sum(b["area_m2"] for b in demolished)
        imp_after = max(0.0, site["impervious_m2"] - gone) + new_area + road_area
        table.append(row("Impervious surface", "% of site", 100 * site["impervious_m2"] / area,
                         min(100.0, 100 * imp_after / area),
                         "road + roof labels; new roofs and roads assumed to cover pervious ground"))
        removed_canopy = sum(tree_area[t] for t in removed_trees)
        table.append(row("Green cover", "% of site", 100 * site["green_m2"] / area,
                         100 * max(0.0, site["green_m2"] - removed_canopy) / area,
                         "vegetation seen from above; removed tree crowns subtracted"))
    notes = ["Existing buildings and trees come from semantic labels, not a survey; "
             "their floors are estimated from height."]
    if scale_status != "metric":
        notes.insert(0, "Scene scale is " + scale_status + ": lengths, areas and volumes are "
                        "only as right as that scale.")
    return {"features": {fid: {"meshes": d["meshes"], "metrics": d["metrics"],
                               "footprint": d["footprint"], **({"clip": d["clip"]} if "clip" in d else {}),
                               **({"facing_xz": d["facing_xz"]} if "facing_xz" in d else {})}
                         for fid, d in derived.items()},
            "violations": violations, "impacts": impacts,
            "demolished": sorted(demolished_ids), "buildings_hit": sorted(hit_buildings),
            "metrics": table, "notes": notes, "scale_status": scale_status}


# ------------------------------------------------------------------ storage
def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dir(work):
    return Path(work) / "proposals"


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _index(work):
    path = _dir(work) / "index.json"
    if not path.is_file():
        return {"schema_version": SCHEMA, "proposals": [], "active": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("proposals"), list):
        raise ProposalError(409, "Stored proposal index is damaged")
    return data


def _proposal_path(work, pid):
    if not isinstance(pid, str) or not ID.match(pid):
        raise ProposalError(400, "Invalid proposal id")
    return _dir(work) / f"{pid}.json"


def load_proposal(work, pid):
    path = _proposal_path(work, pid)
    if not path.is_file():
        raise ProposalError(404, "Proposal not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA or not isinstance(data.get("features"), list):
        raise ProposalError(409, "Stored proposal is damaged")
    return data


def list_proposals(work, kind=None):
    index = _index(work)
    if kind is None:
        return index
    # Schemes written before missions existed carry no kind: they are plans.
    index = {**index, "proposals": [p for p in index["proposals"] if p.get("kind", "plan") == kind]}
    if index["active"] not in {p["id"] for p in index["proposals"]}:
        index["active"] = index["proposals"][0]["id"] if index["proposals"] else None
    return index


def create_proposal(work, name, *, source=None, kind="plan"):
    index = _index(work)
    if len(index["proposals"]) >= MAX_PROPOSALS:
        raise ProposalError(413, f"At most {MAX_PROPOSALS} proposals per scene")
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ProposalError(400, "A proposal needs a name of at most 80 characters")
    if kind not in PROPOSAL_KINDS:
        raise ProposalError(400, "kind must be plan or mission")
    pid = "p-" + uuid.uuid4().hex[:10]
    features = []
    if source:
        original = load_proposal(work, source)
        features = copy.deepcopy(original["features"])
        kind = original.get("kind", "plan")          # a copy of a mission is a mission
    proposal = {"schema_version": SCHEMA, "id": pid, "name": name.strip(), "kind": kind,
                "revision": 0, "created_at": _now(), "updated_at": _now(),
                "settings": {"persons_per_m2": PERSONS_PER_M2}, "features": features}
    _atomic(_proposal_path(work, pid), proposal)
    index["proposals"].append({"id": pid, "name": proposal["name"], "kind": kind,
                               "created_at": proposal["created_at"]})
    index["active"] = pid
    _atomic(_dir(work) / "index.json", index)
    return proposal


def status_text(proposal):
    """What every export says a scheme is: a proposal, or an inferred reconstruction (ARC-05)."""
    return "inferred hypothesis, not measured" if proposal.get("inferred") else "proposed, not measured"


def set_inferred(work, pid, inferred, basis=""):
    """Mark a scheme as a hypothesis reconstruction (a missing tower, a restored wall)."""
    proposal = load_proposal(work, pid)
    proposal["inferred"] = bool(inferred)
    proposal["inferred_basis"] = str(basis or "")[:1000] if inferred else ""
    proposal["revision"] += 1
    proposal["updated_at"] = _now()
    _atomic(_proposal_path(work, pid), proposal)
    return proposal


def rename_proposal(work, pid, name):
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ProposalError(400, "A proposal needs a name of at most 80 characters")
    proposal = load_proposal(work, pid)
    proposal["name"] = name.strip()
    proposal["revision"] += 1
    proposal["updated_at"] = _now()
    _atomic(_proposal_path(work, pid), proposal)
    index = _index(work)
    for entry in index["proposals"]:
        if entry["id"] == pid:
            entry["name"] = proposal["name"]
    _atomic(_dir(work) / "index.json", index)
    return proposal


def delete_proposal(work, pid):
    path = _proposal_path(work, pid)
    if not path.is_file():
        raise ProposalError(404, "Proposal not found")
    path.unlink()
    index = _index(work)
    index["proposals"] = [p for p in index["proposals"] if p["id"] != pid]
    if index["active"] == pid:
        index["active"] = index["proposals"][0]["id"] if index["proposals"] else None
    _atomic(_dir(work) / "index.json", index)
    return index


def _commit(work, proposal, revision):
    if revision != proposal["revision"]:
        raise ProposalError(409, "This proposal changed in another window. Reload it before editing.")
    proposal["revision"] += 1
    proposal["updated_at"] = _now()
    _atomic(_proposal_path(work, proposal["id"]), proposal)
    return proposal


def upsert_feature(work, pid, feature, revision):
    """Create (no id) or replace (existing id) one feature; returns the proposal."""
    proposal = load_proposal(work, pid)
    clean = validate_feature(feature)
    fid = feature.get("id") if isinstance(feature, dict) else None
    if fid is None:
        if len(proposal["features"]) >= MAX_FEATURES:
            raise ProposalError(413, f"At most {MAX_FEATURES} features per proposal")
        clean.update(id=f"{clean['type']}-{uuid.uuid4().hex[:8]}", created_at=_now(), updated_at=_now())
        proposal["features"].append(clean)
    else:
        for i, existing in enumerate(proposal["features"]):
            if existing["id"] == fid:
                if existing.get("locked") and clean.get("locked"):
                    raise ProposalError(409, "This feature is locked")
                if existing["type"] != clean["type"]:
                    raise ProposalError(400, "A feature cannot change type")
                clean.update(id=fid, created_at=existing.get("created_at", _now()), updated_at=_now())
                proposal["features"][i] = clean
                break
        else:
            raise ProposalError(404, "Feature not found")
    return _commit(work, proposal, revision)


def delete_feature(work, pid, fid, revision):
    proposal = load_proposal(work, pid)
    kept = [f for f in proposal["features"] if f["id"] != fid]
    if len(kept) == len(proposal["features"]):
        raise ProposalError(404, "Feature not found")
    proposal["features"] = kept
    return _commit(work, proposal, revision)


def replace_features(work, pid, features, revision):
    """Whole-list replacement: the editor's undo/redo sends the state it wants back."""
    if not isinstance(features, list) or len(features) > MAX_FEATURES:
        raise ProposalError(400, "features must be a list of at most 2000")
    proposal = load_proposal(work, pid)
    clean = []
    seen = set()
    for feature in features:
        item = validate_feature(feature)
        fid = feature.get("id")
        if not isinstance(fid, str) or not re.match(r"^[a-z]+-[0-9a-f]{8}$", fid) or fid in seen:
            raise ProposalError(400, "Every feature in a replacement needs a unique existing-style id")
        seen.add(fid)
        item.update(id=fid, created_at=feature.get("created_at", _now()), updated_at=_now())
        clean.append(item)
    proposal["features"] = clean
    return _commit(work, proposal, revision)


def add_features(work, pid, features, revision):
    """Append already-validated features in one write (one undo step), e.g. an import."""
    proposal = load_proposal(work, pid)
    if len(proposal["features"]) + len(features) > MAX_FEATURES:
        raise ProposalError(413, f"At most {MAX_FEATURES} features per proposal")
    for feature in features:
        clean = validate_feature(feature)
        clean.update(id=f"{clean['type']}-{uuid.uuid4().hex[:8]}", created_at=_now(), updated_at=_now())
        proposal["features"].append(clean)
    return _commit(work, proposal, revision)


def array_objects(work, pid, item, line, spacing_m, revision, *, offset_m=0.0):
    """Place ``item`` every ``spacing_m`` along a polyline (e.g. street lights)."""
    line = _poly(line, "line", closed_min=2)
    if not (isinstance(spacing_m, (int, float)) and 1.0 <= spacing_m <= 500):
        raise ProposalError(400, "spacing_m must be between 1 and 500")
    dense = densify(line, 0.25)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    station = np.r_[0, np.cumsum(seg)]
    count = int(station[-1] // spacing_m) + 1
    if count > 500:
        raise ProposalError(413, "That would place more than 500 objects")
    proposal = load_proposal(work, pid)
    tangent = np.gradient(dense, axis=0)
    for k in range(count):
        i = int(np.searchsorted(station, k * spacing_m).clip(0, len(dense) - 1))
        t = tangent[i] / max(np.linalg.norm(tangent[i]), 1e-9)
        normal = np.array([-t[1], t[0]])
        pos = dense[i] + normal * float(offset_m)
        clean = validate_feature({"type": "object", "name": item.replace("_", " ").title(),
                                  "params": {"item": item, "position": pos.tolist(),
                                             "yaw_deg": math.degrees(math.atan2(t[1], t[0]))}})
        clean.update(id=f"object-{uuid.uuid4().hex[:8]}", created_at=_now(), updated_at=_now())
        proposal["features"].append(clean)
    if len(proposal["features"]) > MAX_FEATURES:
        raise ProposalError(413, f"At most {MAX_FEATURES} features per proposal")
    return _commit(work, proposal, revision)


# ------------------------------------------------------------------ export
def export_geojson(work, proposal, evaluation, registry=None):
    """GeoJSON of every feature's footprint: WGS84 when georeferenced, else local metres."""
    import scene_frames
    georeferenced = registry is not None and registry.get("status") == "georeferenced"
    features = []
    for f in proposal["features"]:
        derived = evaluation["features"].get(f["id"])
        if not derived:
            continue
        ring = np.asarray(derived["footprint"], dtype=np.float64)
        base = derived["metrics"].get("base_y", 0.0)
        pts3 = np.column_stack([ring[:, 0], np.full(len(ring), base), ring[:, 1]])
        if georeferenced:
            geo = scene_frames.viewer_to_geodetic(pts3, registry)
            coords = [[round(float(lon), 8), round(float(lat), 8)] for lat, lon, _ in geo]
        else:
            coords = [[round(float(x), 3), round(float(-z), 3)] for x, _, z in pts3]
        coords.append(coords[0])
        features.append({"type": "Feature", "id": f["id"],
                         "geometry": {"type": "Polygon", "coordinates": [coords]},
                         "properties": {"feature_type": f["type"], "name": f["name"],
                                        "status": status_text(proposal),
                                        **f["params"], **{k: v for k, v in derived["metrics"].items()
                                                          if not isinstance(v, (list, dict))}}})
    collection = {"type": "FeatureCollection", "name": proposal["name"], "features": features,
                  "proposal": {"id": proposal["id"], "revision": proposal["revision"]},
                  "metrics": evaluation["metrics"], "violations": evaluation["violations"]}
    if georeferenced:
        collection["crs_note"] = "RFC 7946: WGS84 longitude/latitude"
    else:
        collection["crs_note"] = ("LOCAL: viewer metres (x east-ish, y = -viewer z); the scene has no "
                                  "GPS similarity fit, so these are not Earth coordinates")
    return collection
