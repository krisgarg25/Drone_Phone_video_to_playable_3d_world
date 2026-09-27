"""Per-building facade completeness (URB-17): which walls the flight actually saw.

A drone orbit sees the facades that face it and the roofs; a courtyard wall or the side
against a neighbour may never be seen, and the reconstruction there is interpolated or
empty. Planning decisions (setbacks, heights, "demolish and replace") on a facade nobody
observed must say so.

For each existing building (the workspace inventory: footprint rectangle, base, height)
every footprint edge is a wall. The wall is cut into ``bin_m`` x ``bin_m`` cells (along
the edge x up the height); a cell is

* ``observed`` when more than ``weak`` scan points lie within ``band_m`` of the wall
  plane inside it,
* ``weak`` with 1..``weak`` points,
* ``unobserved`` with none.

Scan points are the splat centres of ``viewer_assets/scene.ply`` above a minimum opacity:
the same surface the planner looks at. The result is a ratio of wall area, per facade and
per building, and wall overlays (unobserved red, weak amber) that hatch the unseen walls
in the 3D view.
"""
import math
from pathlib import Path

import numpy as np

import workspace_proposals as proposals

UNSEEN, WEAK = [220, 60, 60], [240, 170, 50]


_POINTS_CACHE = {}


def scene_points(work, *, min_opacity=0.3, max_points=3_000_000, colors=False):
    """Splat centres (viewer frame) that are opaque enough to be surface.

    Cached per file (size + mtime): inspection tools read the same cloud for every click.
    With ``colors`` the splats' base colour (SH DC term) comes back as uint8 RGB too.
    """
    from plyfile import PlyData
    path = Path(work) / "viewer_assets" / "scene.ply"
    if not path.is_file():
        raise ValueError("the scene has no viewer_assets/scene.ply")
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, min_opacity, max_points)
    if key not in _POINTS_CACHE:
        _POINTS_CACHE.clear()
        _POINTS_CACHE[key] = _read_points(path, min_opacity, max_points)
    xyz, rgb = _POINTS_CACHE[key]
    return (xyz, rgb) if colors else xyz


def _read_points(path, min_opacity, max_points):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    names = v.dtype.names
    if all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float64)
        rgb = np.clip((0.5 + 0.28209479177387814 * dc) * 255, 0, 255).astype(np.uint8)
    elif all(k in names for k in ("red", "green", "blue")):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 160, np.uint8)
    keep = np.ones(len(xyz), bool)
    if "opacity" in names:
        keep = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64))) >= min_opacity
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        pick = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    return xyz, rgb


def _compass(vec_xz):
    """Scene-frame compass of an outward normal: -z is scene north."""
    bearing = math.degrees(math.atan2(vec_xz[0], -vec_xz[1])) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((bearing + 22.5) // 45) % 8], round(bearing, 1)


def completeness(buildings, points, *, bin_m=1.0, band_m=0.75, weak=2, overlay=True):
    points = np.asarray(points, dtype=np.float64)
    xz = points[:, [0, 2]]
    out, positions, indices, wpos, widx = [], [], [], [], []
    for b in buildings:
        foot = np.asarray(b["footprint"], dtype=np.float64)
        centre = foot.mean(axis=0)
        base, height = float(b["base_y"]), max(float(b["height_m"]), bin_m)
        # Points near this building at all: a cheap prefilter before per-wall tests.
        lo, hi = foot.min(0) - band_m - 0.1, foot.max(0) + band_m + 0.1
        near = (xz[:, 0] >= lo[0]) & (xz[:, 0] <= hi[0]) & (xz[:, 1] >= lo[1]) & (xz[:, 1] <= hi[1]) \
            & (points[:, 1] >= base - 0.1) & (points[:, 1] <= base + height + 0.1)
        local, heights = xz[near], points[near, 1]
        # Each point belongs to its nearest wall, so a corner's points count once, for
        # the wall they lie on, not for the unseen wall that meets it.
        dists = []
        for k in range(len(foot)):
            a, c = foot[k], foot[(k + 1) % len(foot)]
            seg = c - a
            t = np.clip(((local - a) @ seg) / max(float(seg @ seg), 1e-12), 0.0, 1.0)
            dists.append(np.hypot(*(local - (a + t[:, None] * seg)).T))
        if dists:
            stack = np.vstack(dists)
            nearest = np.argmin(stack, axis=0)
            ordered = np.sort(stack, axis=0)
            # A point on the corner itself is equally near two walls and evidence for neither.
            if len(foot) > 1:
                nearest = np.where(ordered[1] - ordered[0] < 0.05, -1, nearest)
        else:
            nearest = np.zeros(0, int)
        facades, totals = [], np.zeros(3)
        for k in range(len(foot)):
            a, c = foot[k], foot[(k + 1) % len(foot)]
            edge = c - a
            length = float(np.hypot(*edge))
            if length < 0.5:
                continue
            u = edge / length
            n = np.array([u[1], -u[0]])
            if np.dot(n, (a + c) / 2 - centre) < 0:
                n = -n
            rel = local - a
            along, off = rel @ u, rel @ n
            pick = (np.abs(off) <= band_m) & (along >= 0) & (along <= length) & (nearest == k)
            ns, nh = max(1, int(math.ceil(length / bin_m))), max(1, int(math.ceil(height / bin_m)))
            si = np.clip((along[pick] / length * ns).astype(int), 0, ns - 1)
            hi_ = np.clip(((heights[pick] - base) / height * nh).astype(int), 0, nh - 1)
            counts = np.zeros((nh, ns), dtype=int)
            np.add.at(counts, (hi_, si), 1)
            cells = counts.size
            observed = int((counts > weak).sum())
            weak_cells = int(((counts > 0) & (counts <= weak)).sum())
            unseen = cells - observed - weak_cells
            area = length * height / cells
            totals += np.array([observed, weak_cells, unseen]) * area
            label, bearing = _compass(n)
            facades.append({"facade": label, "bearing_deg": bearing, "length_m": round(length, 2),
                            "area_m2": round(length * height, 1),
                            "observed_pct": round(100 * observed / cells, 1),
                            "weak_pct": round(100 * weak_cells / cells, 1),
                            "unobserved_pct": round(100 * unseen / cells, 1)})
            if overlay:
                for (rows, cols), (pos, idx) in (((np.nonzero(counts == 0)), (positions, indices)),
                                                 ((np.nonzero((counts > 0) & (counts <= weak))), (wpos, widx))):
                    for r, s in zip(rows, cols):
                        p0 = a + u * (s * length / ns) + n * 0.08
                        p1 = a + u * ((s + 1) * length / ns) + n * 0.08
                        y0, y1 = base + r * height / nh, base + (r + 1) * height / nh
                        start = len(pos)
                        pos += [[p0[0], y0, p0[1]], [p1[0], y0, p1[1]], [p1[0], y1, p1[1]], [p0[0], y1, p0[1]]]
                        idx += [[start, start + 1, start + 2], [start, start + 2, start + 3],
                                [start, start + 2, start + 1], [start, start + 3, start + 2]]
        wall = float(totals.sum()) or 1.0
        out.append({"id": b["id"], "centre": b.get("centre"), "wall_area_m2": round(wall, 1),
                    "observed_pct": round(100 * totals[0] / wall, 1), "weak_pct": round(100 * totals[1] / wall, 1),
                    "unobserved_pct": round(100 * totals[2] / wall, 1), "facades": facades})
    meshes = []
    if overlay:
        # Keep each mesh inside the viewer's per-feature budget.
        for pos, idx, colour, part in ((positions, indices, UNSEEN, "facade_unobserved"), (wpos, widx, WEAK, "facade_weak")):
            if idx:
                meshes.append(proposals._mesh("overlay", pos, idx, colour, 0.55, part=part))
    return {"buildings": out, "overlay": meshes,
            "basis": (f"scan points within {band_m:g} m of each wall, {bin_m:g} m cells; observed = more than "
                      f"{weak} points, weak = 1-{weak}, unobserved = none"),
            "notes": ["Walls are the inventory's footprint rectangles, so a building that is not "
                      "rectangular has walls that are approximate.",
                      "An unobserved wall is not proof the wall is missing: the flight did not see it, "
                      "and anything drawn there is inferred."]}
