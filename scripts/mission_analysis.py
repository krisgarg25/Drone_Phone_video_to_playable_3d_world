"""Terrain analysis for mission planning: line of sight, viewsheds, route exposure, a covered
route suggestion and helicopter landing zones.

Everything runs on the scene's own surface model in the viewer frame:

* ``top``  - the scanned top surface (``heights.f32``): roofs, crowns, terrain. It blocks sight.
* ``floor`` - the measured ground (``ground.f32``). People walk on it; routes drape on it.

A person is modelled as eyes 1.6 m and chest 1.2 m above the surface they stand on; sight
is blocked when the surface rises above the straight line between eye and target anywhere
along the way. Tree crowns are treated as solid (no seeing through leaves), and nothing
unobserved is assumed clear: cells outside the scan's coverage are reported, not guessed.

Walking time uses Tobler's hiking function, v = 6 exp(-3.5 |dh/dx + 0.05|) km/h, times a
pace factor. It is a planning estimate for fit adults on foot, and the report says so.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

import mission
import plan_shadow

EYE_M, CHEST_M = 1.6, 1.2
BLOCK_TOLERANCE_M = 0.15
BUILDING_CLUTTER_M = 1.2
"""Surface this far above the floor is not walkable (walls, parked trucks, dense crowns)."""


class AnalysisError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Terrain:
    def __init__(self, grid):
        work = plan_shadow._work_grid(grid)
        self.floor, self.top, self.supported = work["floor"], work["top"], work["supported"]
        self.cell, self.ox, self.oz = work["cell"], work["ox"], work["oz"]
        self.nz, self.nx = self.floor.shape
        gx, gz = plan_shadow._centres(work)
        self.cx, self.cz = gx, gz

    def index(self, xz):
        xz = np.atleast_2d(xz)
        i = np.clip(np.floor((xz[:, 0] - self.ox) / self.cell).astype(int), 0, self.nx - 1)
        j = np.clip(np.floor((xz[:, 1] - self.oz) / self.cell).astype(int), 0, self.nz - 1)
        return j, i

    def at(self, arr, xz):
        j, i = self.index(xz)
        return arr[j, i]

    def inside(self, xz):
        xz = np.atleast_2d(xz)
        return ((xz[:, 0] >= self.ox) & (xz[:, 0] < self.ox + self.nx * self.cell)
                & (xz[:, 1] >= self.oz) & (xz[:, 1] < self.oz + self.nz * self.cell))


# ------------------------------------------------------------------ line of sight
def los(terrain, a, b, *, ignore_m=0.6):
    """(visible, blocked_at [x, y, z] or None) between two 3D points in the viewer frame."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    horiz = math.hypot(b[0] - a[0], b[2] - a[2])
    n = max(2, int(math.ceil(horiz / (terrain.cell * 0.5))))
    t = np.linspace(0, 1, n + 1)[1:-1]
    pts = a[None] + (b - a)[None] * t[:, None]
    d_from_a, d_to_b = t * horiz, (1 - t) * horiz
    keep = (d_from_a > ignore_m) & (d_to_b > ignore_m)
    surface = terrain.at(terrain.top, pts[:, [0, 2]])
    blocked = keep & (surface > pts[:, 1] + BLOCK_TOLERANCE_M) & terrain.inside(pts[:, [0, 2]])
    if blocked.any():
        k = int(np.argmax(blocked))
        return False, [round(float(pts[k, 0]), 2), round(float(surface[k]), 2), round(float(pts[k, 2]), 2)]
    return True, None


def viewshed(terrain, eye, *, range_m, facing_xz=None, sector_deg=360.0, target_m=CHEST_M, max_samples=None):
    """Boolean grid of cells whose target point (surface + ``target_m``) the eye can see."""
    eye = np.asarray(eye, dtype=np.float64)
    dx, dz = terrain.cx - eye[0], terrain.cz - eye[2]
    dist = np.hypot(dx, dz)
    cand = (dist <= range_m) & (dist > 1e-6)
    if facing_xz is not None and sector_deg < 360:
        f = np.asarray(facing_xz, dtype=np.float64)
        f = f / max(np.linalg.norm(f), 1e-12)
        cosang = (dx * f[0] + dz * f[1]) / np.maximum(dist, 1e-9)
        cand &= cosang >= math.cos(math.radians(sector_deg) / 2)
    out = np.zeros(terrain.floor.shape, dtype=bool)
    jj, ii = np.nonzero(cand)
    if not len(jj):
        return out
    tx, tz = terrain.cx[jj, ii], terrain.cz[jj, ii]
    ty = terrain.top[jj, ii] + target_m
    d = dist[jj, ii]
    step = terrain.cell * 0.7
    blocked = np.zeros(len(jj), dtype=bool)
    steps = int(math.ceil(range_m / step)) if max_samples is None else max_samples
    for k in range(1, steps + 1):
        s = k * step
        live = (s < d - 0.6) & ~blocked
        if not live.any():
            if s >= d.max():
                break
            continue
        t = s / d[live]
        px = eye[0] + (tx[live] - eye[0]) * t
        pz = eye[2] + (tz[live] - eye[2]) * t
        py = eye[1] + (ty[live] - eye[1]) * t
        pj = np.clip(np.floor((pz - terrain.oz) / terrain.cell).astype(int), 0, terrain.nz - 1)
        pi = np.clip(np.floor((px - terrain.ox) / terrain.cell).astype(int), 0, terrain.nx - 1)
        hit = (terrain.top[pj, pi] > py + BLOCK_TOLERANCE_M) & (s > 0.6)
        idx = np.flatnonzero(live)
        blocked[idx[hit]] = True
    out[jj[~blocked], ii[~blocked]] = True
    return out


# ------------------------------------------------------------------ hostiles from a mission
def observers(proposal, evaluation):
    """Every planned hostile with an eye point, facing, sector and range."""
    out = []
    for f in proposal["features"]:
        if f["type"] != "symbol" or f.get("hidden") or f["params"]["affiliation"] != "hostile":
            continue
        d = evaluation["features"].get(f["id"])
        if not d:
            continue
        p = f["params"]
        if p["role"] in ("objective", "obstacle", "checkpoint", "rally_point", "hlz"):
            continue
        out.append({"id": f["id"], "name": f["name"], "eye": [p["position"][0], d["metrics"]["eye_y"], p["position"][1]],
                    "facing_xz": d["facing_xz"], "sector_deg": p["sector_deg"], "range_m": p["range_m"],
                    "behaviour": p["behaviour"]})
    return out


def exposure_grid(terrain, hostiles, target_m=CHEST_M):
    """Per cell: how many hostiles can see a person standing there."""
    count = np.zeros(terrain.floor.shape, dtype=np.int16)
    for h in hostiles:
        count += viewshed(terrain, h["eye"], range_m=h["range_m"], facing_xz=h["facing_xz"],
                          sector_deg=h["sector_deg"], target_m=target_m)
    return count


# ------------------------------------------------------------------ routes
def tobler_kmh(slope):
    return 6.0 * np.exp(-3.5 * np.abs(slope + 0.05))


def _densify(line, step=1.0):
    import workspace_proposals as proposals
    return proposals.densify(np.asarray(line, dtype=np.float64), step)


def route_report(terrain, waypoints, hostiles, *, pace="walk", phase_lines=(), names=()):
    """Length, Tobler ETA, per-waypoint times, exposure to each hostile and dead ground."""
    wps = np.asarray(waypoints, dtype=np.float64)
    dense = _densify(wps, 1.0)
    ys = terrain.at(terrain.floor, dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    rise = np.diff(ys)
    slope = rise / np.maximum(seg, 1e-9)
    speed = tobler_kmh(slope) / 3.6 * mission.PACES.get(pace, 1.0)
    dt = seg / speed
    time = np.r_[0.0, np.cumsum(dt)]
    station = np.r_[0.0, np.cumsum(seg)]
    target = np.column_stack([dense[:, 0], ys + CHEST_M, dense[:, 1]])
    mid_dt = np.r_[dt, 0.0] / 2 + np.r_[0.0, dt] / 2        # time each sample stands for
    seen_by = np.zeros(len(dense), dtype=np.int16)
    per_hostile = []
    for h in hostiles:
        eye = np.asarray(h["eye"], dtype=np.float64)
        rel = dense - eye[[0, 2]]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        f = np.asarray(h["facing_xz"], dtype=np.float64)
        cos_ok = (rel @ f) / np.maximum(dist, 1e-9) >= math.cos(math.radians(h["sector_deg"]) / 2) if h["sector_deg"] < 360 else np.ones(len(dense), bool)
        cand = (dist <= h["range_m"]) & cos_ok
        vis = np.zeros(len(dense), dtype=bool)
        for k in np.flatnonzero(cand):
            vis[k] = los(terrain, eye, target[k])[0]
        seen_by += vis
        exposure = float(mid_dt[vis].sum())
        runs = np.diff(np.r_[0, vis.astype(np.int8), 0])
        starts, stops = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
        longest = max((float(time[min(e, len(time) - 1)] - time[s]) for s, e in zip(starts, stops)), default=0.0)
        first = int(np.argmax(vis)) if vis.any() else None
        per_hostile.append({"id": h["id"], "name": h["name"], "exposure_s": round(exposure, 1),
                            "longest_exposure_s": round(longest, 1),
                            "first_seen_m": None if first is None else round(float(station[first]), 1),
                            "first_seen_s": None if first is None else round(float(time[first]), 1),
                            "closest_m": round(float(dist.min()), 1)})
    # Waypoint arrival times: the dense line passes through every waypoint in order.
    wp_station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wps, axis=0), axis=1))]
    wp_times = np.interp(wp_station, station, time)
    crossings = []
    for pl in phase_lines:
        line = np.asarray(pl["line"], dtype=np.float64)
        for k in range(len(dense) - 1):
            p, r = dense[k], dense[k + 1] - dense[k]
            hit = None
            for a, b in zip(line[:-1], line[1:]):
                s = b - a
                den = r[0] * s[1] - r[1] * s[0]
                if abs(den) < 1e-12:
                    continue
                q = a - p
                t = (q[0] * s[1] - q[1] * s[0]) / den
                u = (q[0] * r[1] - q[1] * r[0]) / den
                if 0 <= t <= 1 and 0 <= u <= 1:
                    hit = t
                    break
            if hit is not None:
                crossings.append({"label": pl["label"], "station_m": round(float(station[k] + hit * seg[k]), 1),
                                  "time_s": round(float(time[k] + hit * dt[k]), 1)})
    exposed = seen_by > 0
    return {
        "length_m": round(float(station[-1]), 1), "eta_s": round(float(time[-1]), 1),
        "climb_m": round(float(np.clip(rise, 0, None).sum()), 1), "descent_m": round(float(np.clip(-rise, 0, None).sum()), 1),
        "max_slope_pct": round(float(np.abs(slope).max() * 100) if len(slope) else 0.0, 1),
        "exposure_s": round(float(mid_dt[exposed].sum()), 1),
        "exposed_m": round(float(np.sum(seg[(exposed[:-1] | exposed[1:])])), 1),
        "dead_ground_pct": round(100 * float((~exposed).mean()), 1),
        "unscanned_m": round(float(np.sum(seg[~terrain.at(terrain.supported, dense)[:-1]])), 1),
        "per_hostile": per_hostile,
        "waypoints": [{"name": (names[i] if i < len(names) and names[i] else f"WP{i + 1}"), "time_s": round(float(t), 1),
                       "station_m": round(float(s), 1)} for i, (t, s) in enumerate(zip(wp_times, wp_station))],
        "phase_lines": crossings,
        "profile": {"station_m": station[::max(1, len(station) // 200)].round(1).tolist(),
                    "height_m": ys[::max(1, len(station) // 200)].round(2).tolist(),
                    "seen_by": seen_by[::max(1, len(station) // 200)].tolist()},
        "basis": "Tobler's hiking function x pace; sight from each hostile's eye (surface + 1.6 m) "
                 "to a person's chest (ground + 1.2 m), crowns solid",
    }


def covered_route(terrain, start, end, hostiles, *, exposure_weight=25.0, slope_weight=4.0):
    """A* over the ground grid trading length and slope against cells hostiles can see."""
    exposure = exposure_grid(terrain, hostiles)
    walkable = ((terrain.top - terrain.floor) < BUILDING_CLUTTER_M) & terrain.supported
    sj, si = (int(v[0]) for v in terrain.index([start[0], start[1]]))
    ej, ei = (int(v[0]) for v in terrain.index([end[0], end[1]]))
    walkable[sj, si] = walkable[ej, ei] = True
    nz, nx = walkable.shape
    cell = terrain.cell
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    INF = float("inf")
    best = np.full((nz, nx), INF)
    parent = np.full((nz, nx), -1, dtype=np.int64)
    best[sj, si] = 0.0
    heap = [(0.0, sj, si)]
    goal = (ej, ei)
    h = lambda j, i: math.hypot(j - ej, i - ei) * cell
    while heap:
        f, j, i = heapq.heappop(heap)
        if (j, i) == goal:
            break
        g = best[j, i]
        if f - h(j, i) > g + 1e-9:
            continue
        for dj, di, w in moves:
            nj, ni = j + dj, i + di
            if not (0 <= nj < nz and 0 <= ni < nx) or not walkable[nj, ni]:
                continue
            step = w * cell
            slope = abs(terrain.floor[nj, ni] - terrain.floor[j, i]) / step
            cost = step * (1 + slope_weight * slope) + step * exposure_weight * exposure[nj, ni]
            ng = g + cost
            if ng < best[nj, ni]:
                best[nj, ni] = ng
                parent[nj, ni] = j * nx + i
                heapq.heappush(heap, (ng + h(nj, ni), nj, ni))
    if not np.isfinite(best[ej, ei]):
        raise AnalysisError(422, "No walkable path joins the start and end over measured ground.")
    path = []
    k = ej * nx + ei
    while k >= 0:
        j, i = divmod(int(k), nx)
        path.append([terrain.cx[j, i], terrain.cz[j, i]])
        k = parent[j, i] if (j, i) != (sj, si) else -1
    path = np.asarray(path[::-1])
    path[0], path[-1] = start, end
    return _simplify(path, 0.6 * cell + 0.3)


def _simplify(points, tol):
    """Douglas-Peucker."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    rel = points - a
    d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n if n > 1e-9 else np.linalg.norm(rel, axis=1)
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.vstack([a, b])
    left, right = _simplify(points[:k + 1], tol), _simplify(points[k:], tol)
    return np.vstack([left[:-1], right])


# ------------------------------------------------------------------ helicopter landing zones
def hlz_candidates(terrain, *, diameter_m=25.0, max_slope_deg=7.0, max_obstacle_m=0.5, limit=5):
    """Clear, flat, measured circles of at least ``diameter_m``: centre, usable diameter, slope."""
    gy, gx = np.gradient(terrain.floor, terrain.cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    good = (slope <= max_slope_deg) & ((terrain.top - terrain.floor) <= max_obstacle_m) & terrain.supported
    # Grid edges are unknown beyond the scan: treat them as not clear.
    good[[0, -1], :] = False
    good[:, [0, -1]] = False
    clear = ndimage.distance_transform_edt(good) * terrain.cell
    out, taken = [], np.zeros_like(good)
    need = diameter_m / 2
    order = np.argsort(clear, axis=None)[::-1]
    for flat in order:
        j, i = divmod(int(flat), terrain.nx)
        r = clear[j, i]
        if r < need or len(out) >= limit:
            break
        if taken[j, i]:
            continue
        x, z = terrain.cx[j, i], terrain.cz[j, i]
        near = np.hypot(terrain.cx - x, terrain.cz - z) <= r * 2
        taken |= near
        disk = np.hypot(terrain.cx - x, terrain.cz - z) <= need
        out.append({"centre": [round(float(x), 2), round(float(z), 2)], "usable_diameter_m": round(float(2 * r), 1),
                    "max_slope_deg": round(float(slope[disk].max()), 1),
                    "max_obstacle_m": round(float((terrain.top - terrain.floor)[disk].max()), 2)})
    return {"candidates": out, "criteria": {"diameter_m": diameter_m, "max_slope_deg": max_slope_deg,
                                            "max_obstacle_m": max_obstacle_m},
            "note": "Measured ground only; approach and departure sectors and wires are not checked."}


# ------------------------------------------------------------------ threat heatmap (MIL-04)
def threat_heatmap(terrain, route, *, range_m=300.0, samples=40, limit=8, spacing_m=15.0):
    """Where an enemy would most likely sit to watch a planned approach. HEURISTIC.

    Per standable cell (open ground, or a flat roof), three terms:

    * overwatch - the share of the route (``samples`` points along it, a person's chest
      above the floor) that an enemy's eyes there could see, within ``range_m``. Sight is
      symmetric on a surface model, so this is the union of reverse viewsheds cast from
      the route points to eye height above the surface;
    * height advantage - how far the cell's surface stands above the route, saturating
      at 10 m;
    * concealment - the share of the 3 m ring around it whose surface rises 1 m or more
      above it (parapets, walls, crowns to hide behind).

    score = overwatch x (0.5 + 0.5 height) x (0.6 + 0.4 concealment). It ranks ground for
    a planner to check; it is not a detection and says so.
    """
    route = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    if len(route) < 2:
        raise AnalysisError(400, "A route needs two or more waypoints.")
    pts = _densify(route, max(1.0, float(np.hypot(*np.diff(route, axis=0).T).sum()) / samples))
    if len(pts) > samples:
        pts = pts[np.linspace(0, len(pts) - 1, samples).round().astype(int)]
    seen = np.zeros(terrain.floor.shape, dtype=np.int32)
    heights = []
    for x, z in pts:
        y = float(terrain.at(terrain.floor, [[x, z]])[0])
        heights.append(y)
        seen += viewshed(terrain, [x, y + CHEST_M, z], range_m=range_m, target_m=EYE_M)
    overwatch = seen / max(len(pts), 1)
    raised = terrain.top - terrain.floor
    gz, gx = np.gradient(terrain.top, terrain.cell)
    flat_roof = (raised > 2.5) & (np.hypot(gx, gz) < 0.15)
    standable = terrain.supported & ((raised < 0.5) | flat_roof)
    height = np.clip((terrain.top - float(np.median(heights))) / 10.0, 0.0, 1.0)
    r = max(1, int(round(3.0 / terrain.cell)))
    ring = np.zeros((2 * r + 1, 2 * r + 1), bool)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ring[(yy ** 2 + xx ** 2 <= r * r) & (yy ** 2 + xx ** 2 >= (r - 1) ** 2)] = True
    # Share of ring cells >= 1 m higher: a convolution of the indicator per offset.
    cover = np.zeros(terrain.top.shape)
    offsets = np.argwhere(ring) - r
    padded = np.pad(terrain.top, r, mode="edge")
    nz, nx = terrain.top.shape
    for dj, di in offsets:
        shifted = padded[r + dj:r + dj + nz, r + di:r + di + nx]
        cover += shifted >= terrain.top + 1.0
    concealment = cover / len(offsets)
    score = np.where(standable & (overwatch > 0), overwatch * (0.5 + 0.5 * height) * (0.6 + 0.4 * concealment), 0.0)
    # The route itself is where the friendly force is, not the enemy.
    on_route = np.zeros_like(standable)
    for x, z in _densify(route, terrain.cell):
        j, i = terrain.index([[x, z]])
        on_route[max(0, j[0] - 3):j[0] + 4, max(0, i[0] - 3):i[0] + 4] = True
    score[on_route] = 0.0
    picks = []
    order = np.argsort(score, axis=None)[::-1]
    for flat in order:
        if len(picks) >= limit or score.flat[flat] <= 0:
            break
        j, i = np.unravel_index(flat, score.shape)
        x, z = float(terrain.cx[j, i]), float(terrain.cz[j, i])
        if any(math.hypot(x - p["position"][0], z - p["position"][1]) < spacing_m for p in picks):
            continue
        picks.append({"position": [round(x, 2), round(z, 2)], "score": round(float(score[j, i]), 3),
                      "overwatch_pct": round(100 * float(overwatch[j, i]), 1),
                      "height_above_route_m": round(float(terrain.top[j, i] - np.median(heights)), 1),
                      "concealment_pct": round(100 * float(concealment[j, i]), 0),
                      "elevated": bool(flat_roof[j, i])})
    return {"score": score, "candidates": picks, "route_samples": int(len(pts)), "range_m": range_m,
            "basis": "heuristic: share of the route visible x height advantage x nearby cover; not a detection"}


# ------------------------------------------------------------------ obstacles (MIL-09)
def obstacles(terrain, *, min_height_m=3.0, min_area_m2=0.25, limit=200):
    """Everything standing ``min_height_m`` or more above the ground: structures, trees, poles."""
    raised = np.where(terrain.supported, terrain.top - terrain.floor, 0.0)
    labels, count = ndimage.label(raised >= min_height_m, np.ones((3, 3)))
    out = []
    area_cell = terrain.cell ** 2
    for index, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        mask = labels[sl] == index
        area = float(mask.sum()) * area_cell
        if area < min_area_m2:
            continue
        h = raised[sl][mask]
        jj, ii = np.nonzero(mask)
        j0, i0 = sl[0].start, sl[1].start
        x = float(terrain.cx[j0 + jj, i0 + ii].mean())
        z = float(terrain.cz[j0 + jj, i0 + ii].mean())
        top = float(terrain.top[sl][mask].max())
        kind = "pole or mast" if area <= 4.0 and h.max() >= 6.0 else "structure or tree"
        out.append({"position": [round(x, 2), round(z, 2)], "height_m": round(float(h.max()), 1),
                    "top_y": round(top, 2), "area_m2": round(area, 1), "kind": kind})
    out.sort(key=lambda o: -o["height_m"])
    return {"obstacles": out[:limit], "count": len(out), "min_height_m": min_height_m,
            "basis": "surface above measured ground on the scan's grid; wires and thin masts below the cell size are not seen"}


# ------------------------------------------------------------------ trafficability (MIL-10)
TRAFFIC = {
    "foot": {"go_slope_pct": 30.0, "slow_slope_pct": 60.0, "go_clutter_m": 0.5, "slow_clutter_m": 1.2},
    "wheeled": {"go_slope_pct": 12.0, "slow_slope_pct": 25.0, "go_clutter_m": 0.25, "slow_clutter_m": 0.5},
    "tracked": {"go_slope_pct": 30.0, "slow_slope_pct": 45.0, "go_clutter_m": 0.6, "slow_clutter_m": 1.0},
}


def trafficability(terrain, *, mobility="wheeled", road=None, water=None, roughness_window_m=2.0):
    """GO / SLOW-GO / NO-GO per cell for ``mobility`` (foot, wheeled, tracked).

    Slope from the measured floor; clutter = surface standing above the floor (walls,
    rubble, crowns, parked vehicles); roughness = local standard deviation of the floor.
    Roads upgrade SLOW-GO to GO for vehicles; water and unobserved ground are NO-GO.
    Soil strength, wetness and bridges' load limits are not known and the result says so.
    """
    if mobility not in TRAFFIC:
        raise AnalysisError(400, f"mobility is one of {', '.join(TRAFFIC)}")
    t = TRAFFIC[mobility]
    floor = np.where(terrain.supported, terrain.floor, np.nan)
    filled = np.where(np.isfinite(floor), floor, np.nanmedian(floor))
    gz, gx = np.gradient(filled, terrain.cell)
    slope = 100.0 * np.hypot(gx, gz)
    clutter = np.clip(terrain.top - terrain.floor, 0, None)
    size = max(3, int(round(roughness_window_m / terrain.cell)) | 1)
    mean = ndimage.uniform_filter(filled, size)
    rough = np.sqrt(np.maximum(ndimage.uniform_filter(filled ** 2, size) - mean ** 2, 0.0))
    klass = np.full(terrain.floor.shape, 0, np.int8)            # 0 GO, 1 SLOW, 2 NO-GO
    klass[(slope > t["go_slope_pct"]) | (clutter > t["go_clutter_m"]) | (rough > 0.15)] = 1
    klass[(slope > t["slow_slope_pct"]) | (clutter > t["slow_clutter_m"]) | (rough > 0.4)] = 2
    if road is not None and mobility != "foot":
        klass[(np.asarray(road, bool)) & (klass == 1) & (clutter <= t["slow_clutter_m"])] = 0
    if water is not None:
        klass[np.asarray(water, bool)] = 2
    klass[~terrain.supported] = 2
    area = terrain.cell ** 2
    return {"class": klass, "mobility": mobility, "thresholds": t,
            "area_m2": {"go": round(float((klass == 0).sum()) * area, 1), "slow_go": round(float((klass == 1).sum()) * area, 1),
                        "no_go": round(float((klass == 2).sum()) * area, 1),
                        "unobserved": round(float((~terrain.supported).sum()) * area, 1)},
            "basis": ("slope of the measured floor, clutter above it and local roughness; roads upgrade vehicles; "
                      "unobserved ground is NO-GO. Soil strength, wetness and load limits are not known.")}
