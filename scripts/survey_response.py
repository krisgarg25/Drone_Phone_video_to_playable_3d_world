"""Disaster access and flooding on the measured surface (DIS-04, DIS-05).

Both work on north-up rasters of the ground (``dtm``) and the top surface (``dsm``), the
same grids ``survey_terrain`` and the workspace heightfield provide.

**Access (DIS-04).** A vehicle of width ``vehicle_width_m`` can use a cell when the cell
is observed, nothing on it stands higher than ``max_obstacle_m`` above the ground and
the ground slope is at most ``max_slope_pct``; the free space is then eroded by half the
vehicle width, so a gap narrower than the vehicle is closed. Road cells (from the
scene's road labels, or a drawn road mask) cost 1 per metre, other drivable ground
``offroad_cost``. ``blocked_roads`` lists road stretches that are obstructed now (and,
with a pre-event surface, were clear before). ``vehicle_route`` is Dijkstra over the
8-connected grid from a staging point to the site.

**Flood (DIS-05).** Given a water level (absolute, or a seed point plus a rise), the
flooded area is every observed ground cell below the level *connected to the seed* -
a hollow behind an embankment stays dry - with depth = level - ground. Buildings whose
footprint touches water are listed with their maximum depth. This is a "bathtub"
model: no flow, no rainfall, no drainage; it answers "what is under water at this
level", not "when".
"""
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _cell_of(transform, xy, shape):
    a, b, _, d, _, f = transform
    col = int(math.floor((xy[0] - a) / b))
    row = int(math.floor((xy[1] - d) / f))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError(f"point {tuple(round(v, 1) for v in xy)} is outside the surface grid")
    return row, col


def _centre(transform, rc):
    a, b, _, d, _, f = transform
    return [round(float(a + (rc[1] + 0.5) * b), 2), round(float(d + (rc[0] + 0.5) * f), 2)]


def drivable(dtm, dsm, transform, *, vehicle_width_m=2.5, max_obstacle_m=0.5, max_slope_pct=25.0):
    cell = transform[1]
    observed = np.isfinite(dtm) & np.isfinite(dsm)
    ground = np.where(np.isfinite(dtm), dtm, np.nanmedian(dtm))
    gy, gx = np.gradient(ground, cell)
    slope = 100.0 * np.hypot(gx, gy)
    clear = observed & ((np.nan_to_num(dsm, nan=np.inf) - ground) <= max_obstacle_m) & (slope <= max_slope_pct)
    radius = max(0, int(math.ceil((vehicle_width_m / 2) / cell - 0.5)))
    if radius:
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        clear = ndimage.binary_erosion(clear, structure=(yy ** 2 + xx ** 2) <= radius ** 2, border_value=0)
    return clear, slope


def blocked_roads(road, dtm, dsm, transform, *, before_dsm=None, max_obstacle_m=0.5, min_len_m=2.0):
    """Obstructed road stretches, as regions with their obstacle height and area."""
    cell = transform[1]
    height = np.nan_to_num(dsm - dtm, nan=0.0)
    blocked = road & (height > max_obstacle_m)
    if before_dsm is not None:
        before = np.nan_to_num(before_dsm - dtm, nan=0.0)
        blocked &= before <= max_obstacle_m                 # new since the pre-event flight
    labels, count = ndimage.label(blocked, np.ones((3, 3)))
    out = []
    for index in range(1, count + 1):
        rr, cc = np.nonzero(labels == index)
        extent = cell * max(np.ptp(rr) + 1, np.ptp(cc) + 1)
        if extent < min_len_m:
            continue
        out.append({"id": f"R{len(out) + 1}", "area_m2": round(float(rr.size * cell * cell), 1),
                    "max_height_m": round(float(height[rr, cc].max()), 2),
                    "centre": _centre(transform, (rr.mean(), cc.mean())),
                    "new_since_before": before_dsm is not None})
    road_cells = int(road.sum())
    return {"blocked": out, "road_m2": round(road_cells * cell * cell, 1),
            "blocked_m2": round(float(blocked.sum()) * cell * cell, 1),
            "mask": blocked}


def vehicle_route(dtm, dsm, transform, start_xy, end_xy, *, road=None, vehicle_width_m=2.5,
                  max_obstacle_m=0.5, max_slope_pct=25.0, offroad_cost=3.0, speeds_kmh=(20.0, 8.0),
                  snap_m=15.0):
    """Cheapest drivable path; refuses with the reason when there is none.

    A clicked point rarely lands exactly on a drivable cell (a kerb, the road edge the
    vehicle width erodes away); each end moves to the nearest drivable cell within
    ``snap_m`` and the distance moved is reported.
    """
    clear, slope = drivable(dtm, dsm, transform, vehicle_width_m=vehicle_width_m,
                            max_obstacle_m=max_obstacle_m, max_slope_pct=max_slope_pct)
    shape = clear.shape
    cell = transform[1]
    if not clear.any():
        raise ValueError(f"no ground in this scene is drivable for a {vehicle_width_m:g} m vehicle")
    distance, (near_r, near_c) = ndimage.distance_transform_edt(~clear, return_indices=True)
    snapped = {}
    ends = []
    for name, xy in (("staging point", start_xy), ("site", end_xy)):
        rc = _cell_of(transform, xy, shape)
        moved = float(distance[rc]) * cell
        if moved > snap_m:
            raise ValueError(f"the {name} is {moved:.0f} m from the nearest ground a {vehicle_width_m:g} m "
                             "vehicle can use (obstacle, steep, unobserved or too narrow)")
        ends.append((int(near_r[rc]), int(near_c[rc])))
        snapped[name] = round(moved, 1)
    s, e = ends
    road = np.zeros(shape, bool) if road is None else np.asarray(road, bool)
    weight = np.where(road, 1.0, offroad_cost) * (1.0 + slope / 100.0)
    index = -np.ones(shape, np.int64)
    ids = np.flatnonzero(clear.ravel())
    index.ravel()[ids] = np.arange(ids.size)
    rows, cols, costs = [], [], []
    r_all, c_all = np.nonzero(clear)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r1, c1 = r_all + dr, c_all + dc
        ok = (r1 >= 0) & (r1 < shape[0]) & (c1 >= 0) & (c1 < shape[1])
        r0, c0, r1, c1 = r_all[ok], c_all[ok], r1[ok], c1[ok]
        ok = clear[r1, c1]
        r0, c0, r1, c1 = r0[ok], c0[ok], r1[ok], c1[ok]
        length = cell * math.hypot(dr, dc)
        rows.append(index[r0, c0])
        cols.append(index[r1, c1])
        costs.append(length * 0.5 * (weight[r0, c0] + weight[r1, c1]))
    rows, cols, costs = np.concatenate(rows), np.concatenate(cols), np.concatenate(costs)
    graph = coo_matrix((np.r_[costs, costs], (np.r_[rows, cols], np.r_[cols, rows])),
                       shape=(ids.size, ids.size)).tocsr()
    dist, pred = dijkstra(graph, indices=index[s], return_predecessors=True)
    target = index[e]
    if not np.isfinite(dist[target]):
        raise ValueError("no drivable connection between the staging point and the site for a "
                         f"{vehicle_width_m:g} m vehicle; see the blocked roads")
    path = [target]
    while path[-1] != index[s]:
        path.append(pred[path[-1]])
    path = path[::-1]
    rc = np.column_stack(np.unravel_index(ids[path], shape))
    xy = np.array([_centre(transform, p) for p in rc])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    on_road = road[rc[:, 0], rc[:, 1]]
    road_m = float(seg[on_road[1:] & on_road[:-1]].sum())
    length = float(seg.sum())
    eta = (road_m / (speeds_kmh[0] / 3.6) + (length - road_m) / (speeds_kmh[1] / 3.6))
    keep = [0] + [i for i in range(1, len(xy) - 1)
                  if abs(_cross(xy[i] - xy[i - 1], xy[i + 1] - xy[i])) > 1e-6] + [len(xy) - 1]
    return {"path": xy[keep].tolist(), "snapped_m": snapped, "length_m": round(length, 1), "road_m": round(road_m, 1),
            "offroad_m": round(length - road_m, 1), "eta_s": round(eta, 1),
            "vehicle_width_m": vehicle_width_m, "max_obstacle_m": max_obstacle_m,
            "max_slope_pct": max_slope_pct,
            "notes": ["Drivable = observed, obstacles at most %g m, slope at most %g%%, wide enough "
                      "for the vehicle. Unobserved ground is never driven through." % (max_obstacle_m, max_slope_pct),
                      "ETA at %g km/h on road and %g km/h off road; bridges' load limits and soft "
                      "ground are not known." % speeds_kmh]}


def flood(dtm, transform, *, level_m=None, seed_xy=None, rise_m=None, observed=None,
          footprints=None, dsm=None):
    """Bathtub flood extent, depth and affected buildings."""
    dtm = np.asarray(dtm, float)
    observed = np.isfinite(dtm) if observed is None else (np.asarray(observed, bool) & np.isfinite(dtm))
    if seed_xy is None:
        raise ValueError("a seed point in the water body (river, low street) is required")
    seed = _cell_of(transform, seed_xy, dtm.shape)
    if not observed[seed]:
        raise ValueError("the seed point is on ground the flight did not observe")
    if level_m is None:
        if rise_m is None:
            raise ValueError("give a water level or a rise above the seed point")
        level_m = float(dtm[seed]) + float(rise_m)
    below = observed & (dtm < level_m)
    labels, _ = ndimage.label(below, np.ones((3, 3)))
    if labels[seed] == 0:
        raise ValueError("the seed point is above the water level")
    wet = labels == labels[seed]
    depth = np.where(wet, level_m - dtm, np.nan)
    cell = transform[1]
    area = cell * cell
    report = {"level_m": round(float(level_m), 3), "flooded_m2": round(float(wet.sum()) * area, 1),
              "volume_m3": round(float(np.nansum(depth)) * area, 1),
              "max_depth_m": round(float(np.nanmax(depth)), 2) if wet.any() else 0.0,
              "depth_bands_m2": {band: round(float(((depth > lo) & (depth <= hi)).sum()) * area, 1)
                                 for band, lo, hi in (("0-0.5 m", 0, 0.5), ("0.5-1.5 m", 0.5, 1.5),
                                                      ("1.5 m+", 1.5, np.inf))},
              "isolated_low_ground_m2": round(float((below & ~wet).sum()) * area, 1),
              "notes": ["Bathtub model: cells below the level connected to the seed. No flow, "
                        "rainfall or drainage; a culvert under a road is not known.",
                        "Low ground not connected to the seed is reported, not flooded."]}
    buildings = []
    if footprints:
        from matplotlib.path import Path as MplPath
        a, b, _, d, _, f = transform
        rows, cols = dtm.shape
        cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
        centres = np.column_stack([a + (cc.ravel() + 0.5) * b, d + (rr.ravel() + 0.5) * f])
        grown = ndimage.binary_dilation(wet, iterations=1)
        for key, ring in footprints.items():
            inside = MplPath(np.asarray(ring, float)[:, :2]).contains_points(centres).reshape(rows, cols)
            touch = inside & grown
            if touch.any():
                edge_depth = level_m - np.nanmin(np.where(touch, dtm, np.nan))
                buildings.append({"id": str(key), "max_depth_m": round(float(max(edge_depth, 0.0)), 2),
                                  "wet_perimeter_cells": int(touch.sum())})
        buildings.sort(key=lambda r: -r["max_depth_m"])
    report["buildings"] = buildings
    return {"depth": depth.astype(np.float32), "wet": wet, "report": report}
