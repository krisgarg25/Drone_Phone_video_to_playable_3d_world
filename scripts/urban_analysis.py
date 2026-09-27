"""Urban layers measured from the scan: roof planes, solar potential, canopy, road surface.

URB-01 roof planes (area, slope, aspect) per existing building, URB-02 canopy cover and
tree heights, URB-03 road width at a section and the road-surface layer, URB-04 solar
potential per roof plane, and the impervious / green / parking rows of the URB-12 metric
table. Everything here reads the scan (splat centres, semantic labels, measured ground);
nothing writes into it. Labels are heuristic/learned, so every layer built on them says so.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import ndimage

ROOF_CELL_M = 0.5
ROOF_INLIER_M = 0.15
ROOF_MIN_PLANE_M2 = 4.0
SOLAR_CONSTANT_KW = 1.353
MODULE_EFFICIENCY = 0.20
PERFORMANCE_RATIO = 0.75
USABLE_ROOF_FRACTION = 0.7
# Equivalent car spaces (ECS) per 100 m2 of floor area: Delhi MPD-2021-style defaults,
# overridable per scheme in ``settings.parking_ecs_per_100m2``.
PARKING_ECS_PER_100M2 = {"residential": 2.0, "commercial": 3.0, "mixed": 3.0,
                         "institutional": 1.8, "industrial": 1.0}
IMPERVIOUS_LABELS = ("road", "building")


class UrbanError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _bearing(vec_xz, north):
    """Compass bearing (deg clockwise from ``north``) of a viewer-frame [x, z] vector."""
    east = np.array([-north[1], north[0]])
    return float(math.degrees(math.atan2(float(np.dot(vec_xz, east)), float(np.dot(vec_xz, north)))) % 360)


def _compass(bearing):
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[int(((bearing + 22.5) % 360) // 45)]


# ------------------------------------------------------------------ URB-01 roof planes
def _top_cells(points, lo, cell):
    """Highest point per grid cell: the roof a camera above sees, not the walls below it."""
    idx = np.floor((points[:, [0, 2]] - lo) / cell).astype(int)
    key = idx[:, 0] * 100_003 + idx[:, 1]
    order = np.lexsort((-points[:, 1], key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    return points[order][first]


def _fit_plane(p):
    centre = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centre, full_matrices=False)
    n = vt[2]
    return (n if n[1] >= 0 else -n), centre


def roof_planes(building, points, north, *, cell=ROOF_CELL_M, inlier=ROOF_INLIER_M,
                min_plane_m2=ROOF_MIN_PLANE_M2, max_planes=8, seed=0):
    """Planar roof faces of one existing building from the scan's top surface.

    The top point per ``cell`` inside the footprint, above 60% of the building's height,
    is split into planes by sequential RANSAC (``inlier`` m), each refined by least squares
    and kept only as one connected patch of at least ``min_plane_m2`` (plan area).
    Slope is the tilt from horizontal; aspect the compass direction the face looks down to.
    """
    import workspace_proposals as proposals
    poly = np.asarray(building["footprint"], float)
    inside = points[proposals.contains(poly, points[:, [0, 2]])]
    floor = building["base_y"] + 0.6 * building["height_m"]
    inside = inside[inside[:, 1] > floor]
    out = {"id": building["id"], "planes": [], "cell_m": cell}
    if len(inside) < 20:
        out["note"] = "too few scan points on the roof"
        return out
    lo = inside[:, [0, 2]].min(axis=0)
    tops = _top_cells(inside, lo, cell)
    rng = np.random.default_rng(seed)
    remaining = tops
    min_cells = max(6, int(min_plane_m2 / (cell * cell)))
    for _ in range(max_planes):
        if len(remaining) < min_cells:
            break
        best = None
        for _ in range(200):
            s = remaining[rng.choice(len(remaining), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n = n / np.linalg.norm(n)
            if abs(n[1]) < 0.25:            # steeper than ~75 deg: a wall, not a roof
                continue
            count = int((np.abs((remaining - s[0]) @ n) < inlier).sum())
            if best is None or count > best[0]:
                best = (count, n, s[0])
        if best is None or best[0] < min_cells:
            break
        n, c = best[1], best[2]
        members = remaining[np.abs((remaining - c) @ n) < inlier]
        n, c = _fit_plane(members)
        mask = np.abs((remaining - c) @ n) < inlier
        members = remaining[mask]
        # One connected patch: two parallel faces at the same height are not one plane.
        idx = np.floor((members[:, [0, 2]] - lo) / cell).astype(int)
        grid = np.zeros(idx.max(axis=0) + 1, bool)
        grid[idx[:, 0], idx[:, 1]] = True
        labels, count = ndimage.label(grid, structure=np.ones((3, 3)))
        if count > 1:
            sizes = ndimage.sum(grid, labels, range(1, count + 1))
            keep_label = 1 + int(np.argmax(sizes))
            keep = labels[idx[:, 0], idx[:, 1]] == keep_label
            members = members[keep]
            drop = np.flatnonzero(mask)[keep]
            mask = np.zeros(len(remaining), bool)
            mask[drop] = True
        remaining = remaining[~mask]
        plan_area = len(members) * cell * cell
        if plan_area < min_plane_m2:
            continue
        n, c = _fit_plane(members)
        slope = math.degrees(math.acos(min(1.0, abs(float(n[1])))))
        horiz = n[[0, 2]]
        aspect = None if slope < 3.0 or np.linalg.norm(horiz) < 1e-9 else _bearing(horiz / np.linalg.norm(horiz), north)
        rms = float(np.sqrt(np.mean(((members - c) @ n) ** 2)))
        out["planes"].append({
            "plan_area_m2": round(plan_area, 1),
            "area_m2": round(plan_area / max(math.cos(math.radians(slope)), 0.2), 1),
            "slope_deg": round(slope, 1), "aspect_deg": None if aspect is None else round(aspect, 0),
            "aspect": "flat" if aspect is None else _compass(aspect),
            "normal": [round(float(v), 4) for v in n], "centre": [round(float(v), 3) for v in c],
            "height_m": round(float(c[1] - building["base_y"]), 2), "rms_m": round(rms, 3),
            "cells": int(len(members)),
            "outline": _outline(members, cell)})
    out["planes"].sort(key=lambda p: -p["area_m2"])
    covered = sum(p["plan_area_m2"] for p in out["planes"])
    out["roof_covered_pct"] = round(100 * covered / max(building["area_m2"], 1e-9), 0)
    out["kind"] = ("flat" if all(p["slope_deg"] < 5 for p in out["planes"]) else
                   "pitched") if out["planes"] else "unresolved"
    return out


def _outline(members, cell):
    """Convex outline of a plane's cells, in viewer x/y/z (for a draped tint)."""
    from scipy.spatial import ConvexHull
    xz = members[:, [0, 2]]
    if len(xz) < 3:
        return []
    try:
        hull = ConvexHull(xz)
    except Exception:
        return []
    pts = members[hull.vertices]
    return [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)] for p in pts]


# ------------------------------------------------------------------ URB-04 solar
def _plane_irradiance(normal_up, lat, lon, year=2025, utc_offset=None):
    """Clear-sky annual irradiation (kWh/m2) on a plane with unit ENU normal ``normal_up``.

    Sun from the NOAA algorithm every hour of the 21st of each month (x days in month);
    beam from Meinel's air-mass model, diffuse 10% of beam on the sky view (1+cos b)/2.
    A clear-sky ceiling: no cloud, no haze beyond the model, no shading by neighbours.
    """
    import plan_shadow
    n = np.asarray(normal_up, float)
    tilt_cos = n[2]
    total = 0.0
    for month in range(1, 13):
        days = (datetime(year + (month == 12), month % 12 + 1, 1) - datetime(year, month, 1)).days
        daily = 0.0
        base = datetime(year, month, 21, tzinfo=timezone.utc)
        for hour in range(24):
            az, el = plan_shadow.sun_position(lat, lon, base + timedelta(hours=hour - lon / 15.0))
            if el <= 0.5:
                continue
            am = 1.0 / (math.sin(math.radians(el)) + 0.50572 * (el + 6.07995) ** -1.6364)
            beam = SOLAR_CONSTANT_KW * 0.7 ** (am ** 0.678)
            sun = plan_shadow.enu_direction(az, el)
            daily += beam * max(0.0, float(sun @ n)) + 0.1 * beam * (1 + tilt_cos) / 2
        total += daily * days
    return total


def solar_potential(planes, registry, *, lat=None, lon=None):
    """Per-plane clear-sky irradiation and a first estimate of PV yield.

    Needs the site's latitude (the scene's GPS fit, else entered). Aspect needs true north,
    which only a georeferenced scene has; a local scene uses scene north and says so.
    """
    import scene_frames
    geo = registry is not None and registry.get("status") == "georeferenced"
    if geo:
        o = registry["origin"]
        lat, lon = o["latitude_deg"], o["longitude_deg"]
    elif lat is None or lon is None:
        raise UrbanError(409, "This scene has no GPS fit: enter the site's latitude and longitude for solar potential.")
    results = []
    for p in planes:
        nv = np.asarray(p["normal"], float)
        if geo:
            a, b = scene_frames.viewer_to_enu(np.vstack([np.zeros(3), nv]), registry)
            enu = b - a
            enu = enu / np.linalg.norm(enu)
        else:
            # Local: viewer -Z is north, +X east, +Y up.
            enu = np.array([nv[0], -nv[2], nv[1]])
        kwh_m2 = _plane_irradiance(enu, lat, lon)
        usable = p["area_m2"] * USABLE_ROOF_FRACTION
        results.append({**p, "irradiation_kwh_m2_yr": round(kwh_m2, 0),
                        "usable_area_m2": round(usable, 1),
                        "pv_kwh_yr": round(kwh_m2 * usable * MODULE_EFFICIENCY * PERFORMANCE_RATIO, 0),
                        "pv_kwp": round(usable * MODULE_EFFICIENCY, 1)})
    basis = (f"clear-sky irradiation at {lat:.4f}, {lon:.4f} (Meinel beam + 10% diffuse, hourly on "
             f"the 21st of each month); {USABLE_ROOF_FRACTION:.0%} of each face usable, "
             f"{MODULE_EFFICIENCY:.0%} modules, performance ratio {PERFORMANCE_RATIO}. No cloud and no "
             "shading by neighbours: an upper bound, typically 25-40% above metered yield in India.")
    if not geo:
        basis += " Local scene: aspect uses scene north (viewer -Z), not true north."
    return results, basis


# ------------------------------------------------------------------ labels
def labelled_points(work):
    """(xyz, labels) from ``semantics.json``; empty arrays when the scene has none."""
    import json
    from pathlib import Path
    import workspace_proposals as proposals
    path = Path(work) / "viewer_assets" / "semantics.json"
    if not path.is_file():
        return np.zeros((0, 3)), np.array([], dtype=object)
    data = json.loads(path.read_text(encoding="utf-8"))
    xyz = np.asarray(data.get("coords") or [], np.float64).reshape(-1, 3)
    lookup = proposals._class_rgb()
    labels = np.array([lookup.get(tuple(c), "unknown") for c in data.get("rgb") or []], dtype=object)
    if len(labels) != len(xyz):
        return np.zeros((0, 3)), np.array([], dtype=object)
    return xyz, labels


def _occupancy(xz, lo, shape, cell):
    grid = np.zeros(shape, bool)
    if len(xz):
        idx = np.floor((xz - lo) / cell).astype(int)
        ok = (idx >= 0).all(axis=1) & (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1])
        grid[idx[ok, 0], idx[ok, 1]] = True
    return grid


def surface_layers(work, ground, existing, *, cell=1.0):
    """Site-wide cover from labels on a ``cell`` grid over the measured ground.

    Returns road surface, impervious (road + building), canopy and green (vegetation
    seen from above) areas and shares of the observed site, plus canopy statistics.
    """
    g = ground.grid
    xyz, labels = labelled_points(work)
    supported = g.get("supported")
    site_m2 = float(np.asarray(supported).sum()) * g["cell"] ** 2 if supported is not None else g["nx"] * g["nz"] * g["cell"] ** 2
    out = {"cell_m": cell, "site_observed_m2": round(site_m2, 1),
           "basis": "semantic labels (heuristic + learned) on a %g m grid; not surveyed" % cell}
    trees = existing.get("trees", [])
    heights = np.array([t["height_m"] for t in trees], float)
    out["canopy"] = {"trees": len(trees), "crown_area_m2": round(sum(t["area_m2"] for t in trees), 1),
                     "height_m": ({"min": round(float(heights.min()), 1), "median": round(float(np.median(heights)), 1),
                                   "max": round(float(heights.max()), 1)} if len(heights) else None)}
    if not len(xyz):
        out["status"] = "unavailable"
        out["note"] = "The scene has no semantic labels, so road, impervious and green cover cannot be mapped."
        return out
    lo = np.array([g["ox"], g["oz"]])
    shape = (max(1, int(math.ceil(g["nx"] * g["cell"] / cell))), max(1, int(math.ceil(g["nz"] * g["cell"] / cell))))
    grids = {k: _occupancy(xyz[labels == k][:, [0, 2]], lo, shape, cell) for k in ("road", "building", "vegetation")}
    area = {k: float(v.sum()) * cell * cell for k, v in grids.items()}
    impervious = grids["road"] | grids["building"]
    out.update({"status": "measured",
                "road_surface_m2": round(area["road"], 1),
                "building_roof_m2": round(area["building"], 1),
                "impervious_m2": round(float(impervious.sum()) * cell * cell, 1),
                "green_m2": round(area["vegetation"], 1)})
    for key in ("impervious", "green"):
        out[key + "_pct"] = round(100 * out[key + "_m2"] / site_m2, 1) if site_m2 else None
    out["canopy"]["cover_pct"] = round(100 * out["canopy"]["crown_area_m2"] / site_m2, 1) if site_m2 else None
    out["_grids"] = {"road": grids["road"], "lo": lo, "cell": cell}
    return out


def road_overlay(layers, ground, *, color=(90, 110, 210), budget=30_000):
    """Road-surface cells as flat quads just above the measured ground (one mesh)."""
    import workspace_proposals as proposals
    grids = layers.get("_grids")
    if not grids:
        return []
    cell, lo = grids["cell"], grids["lo"]
    ii, jj = np.nonzero(grids["road"])
    if len(ii) > budget:
        pick = np.random.default_rng(0).choice(len(ii), budget, replace=False)
        ii, jj = ii[pick], jj[pick]
    if not len(ii):
        return []
    x0 = lo[0] + ii * cell
    z0 = lo[1] + jj * cell
    corners = np.stack([np.c_[x0, z0], np.c_[x0 + cell, z0], np.c_[x0 + cell, z0 + cell], np.c_[x0, z0 + cell]], axis=1)
    y = ground.sample(corners.reshape(-1, 2))[0].reshape(-1, 4) + 0.08
    positions = np.dstack([corners[:, :, 0], y, corners[:, :, 1]]).reshape(-1, 3)
    base = np.arange(len(ii))[:, None] * 4
    idx = np.hstack([base + [0, 1, 2], base + [0, 2, 3]]).reshape(-1, 3)
    return [proposals._mesh("overlay", np.round(positions, 3).tolist(), idx.tolist(), list(color), 0.55, part="road_surface")]


# ------------------------------------------------------------------ URB-03 road width
def road_width(work, a, b, *, step=0.1, radius=0.6):
    """Width of the road crossed by the line a-b (viewer points), from road labels.

    Samples every ``step`` m along the line; each sample takes the majority label of the
    labelled points within ``radius`` m (plan distance). The road run nearest the line's
    middle is the answer; its ends are refined to the sample, so the stated uncertainty is
    the label spacing plus one step.
    """
    from scipy.spatial import cKDTree
    a, b = np.asarray(a, float), np.asarray(b, float)
    xyz, labels = labelled_points(work)
    if not len(xyz):
        raise UrbanError(409, "The scene has no semantic labels, so the road cannot be found.")
    span = b[[0, 2]] - a[[0, 2]]
    length = float(np.linalg.norm(span))
    if length < 1.0:
        raise UrbanError(400, "Draw the section at least 1 m long, across the road.")
    if length > 300:
        raise UrbanError(400, "Keep the section under 300 m.")
    tree = cKDTree(xyz[:, [0, 2]])
    t = np.arange(0, length + 1e-9, step)
    xz = a[[0, 2]] + np.outer(t / length, span)
    hits = tree.query_ball_point(xz, radius)
    is_road = np.zeros(len(t), bool)
    seen = np.zeros(len(t), bool)
    for i, h in enumerate(hits):
        if h:
            seen[i] = True
            lab = labels[h]
            is_road[i] = (lab == "road").sum() * 2 > len(lab)
    runs = []
    edges = np.flatnonzero(np.diff(np.r_[0, is_road.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        runs.append((s, e - 1))
    if not runs:
        return {"width_m": None, "section_m": round(length, 2),
                "note": "No road-labelled surface under this line." + ("" if seen.any() else " It crosses no labelled scan at all.")}
    mid = len(t) / 2
    s, e = min(runs, key=lambda r: 0 if r[0] <= mid <= r[1] else min(abs(r[0] - mid), abs(r[1] - mid)))
    width = (e - s) * step + step
    spacing = float(np.median(tree.query(xyz[:: max(1, len(xyz) // 2000), [0, 2]], k=2)[0][:, 1]))
    left, right = xz[s], xz[e]
    import workspace_proposals as proposals
    ground = proposals.Ground(work)
    ys = ground.sample(np.vstack([left, right]))[0]
    gaps = int((~seen[s:e + 1]).sum())
    return {"width_m": round(width, 2), "uncertainty_m": round(2 * spacing + step, 2),
            "section_m": round(length, 2), "runs": len(runs),
            "edges": [[round(float(left[0]), 3), round(float(ys[0]), 3), round(float(left[1]), 3)],
                      [round(float(right[0]), 3), round(float(ys[1]), 3), round(float(right[1]), 3)]],
            "unlabelled_samples": gaps,
            "basis": "road labels (heuristic + learned) sampled every %.1f m; label spacing %.2f m" % (step, spacing)}


# ------------------------------------------------------------------ URB-12 extra rows
def parking_demand(buildings_gfa_by_use, settings=None):
    """Equivalent car spaces for new floor area, by use (per-scheme override allowed)."""
    rates = dict(PARKING_ECS_PER_100M2)
    custom = (settings or {}).get("parking_ecs_per_100m2")
    if isinstance(custom, dict):
        for k, v in custom.items():
            if k in rates and isinstance(v, (int, float)) and 0 <= v <= 20:
                rates[k] = float(v)
    return sum(gfa / 100.0 * rates.get(use, 1.0) for use, gfa in buildings_gfa_by_use.items()), rates


def study(work, ground, existing, registry, *, lat=None, lon=None, with_solar=True):
    """Everything the Plan tab's *Site layers* card shows, in one call."""
    import facades
    import mission
    north = mission.north_vector(registry)
    layers = surface_layers(work, ground, existing)
    overlay = road_overlay(layers, ground)
    layers.pop("_grids", None)
    buildings = []
    notes = []
    try:
        points = facades.scene_points(work)
    except (OSError, ValueError) as error:
        points = None
        notes.append("No splat cloud to fit roofs to: " + str(error))
    solar_basis = None
    for b in existing.get("buildings", []):
        entry = {"id": b["id"], "height_m": b["height_m"], "floors_estimate": b.get("floors_estimate"),
                 "footprint_m2": b["area_m2"]}
        if points is not None:
            roof = roof_planes(b, points, north)
            entry.update(roof)
            if with_solar and roof["planes"]:
                try:
                    planes, solar_basis = solar_potential(roof["planes"], registry, lat=lat, lon=lon)
                    entry["planes"] = planes
                    entry["pv_kwh_yr"] = round(sum(p["pv_kwh_yr"] for p in planes), 0)
                    entry["pv_kwp"] = round(sum(p["pv_kwp"] for p in planes), 1)
                except UrbanError as error:
                    solar_basis = None
                    if str(error) not in notes:
                        notes.append(str(error))
        buildings.append(entry)
        for p in entry.get("planes", []):
            p["overlay"] = p.pop("outline", [])
    if not existing.get("buildings"):
        notes.append("No existing buildings in this scan's labels, so there are no roofs to analyse.")
    if registry.get("status") != "georeferenced":
        notes.append("Local scene: aspects use scene north (viewer -Z), not true north.")
    return {"buildings": buildings, "layers": layers, "overlay": overlay, "solar_basis": solar_basis,
            "notes": notes, "roof_basis": ("top scan point per %.1f m cell above 60%% of the building's height, "
                                           "sequential RANSAC planes (%.2f m inliers), faces under %g m² dropped"
                                           % (ROOF_CELL_M, ROOF_INLIER_M, ROOF_MIN_PLANE_M2))}
