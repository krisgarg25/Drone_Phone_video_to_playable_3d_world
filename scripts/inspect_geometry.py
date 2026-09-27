"""Measured geometry for inspection and archaeology on the scan's points (viewer frame, y up).

* ``tilt`` (INF-04) - verticality of a pole, mast or tower: the points in a cylinder around a
  clicked base are fitted with a 3D line (PCA, twice, the second time in a tighter cylinder
  around the first axis); tilt from vertical, lean direction (scene bearing, -z north) and
  lean per metre of height.
* ``wire`` (INF-05) - a conductor between two clicked attachment points: points in a narrow
  vertical slab between them, above the ground, fitted with a catenary
  y = c + a (cosh((s - s0) / a) - 1) by robust least squares; sag against the chord, and the
  lowest clearance to anything scanned below it (vegetation, roofs, ground).
* ``section`` (ARC-02) - a vertical section through the scan along a two-point line: the
  points within a slab, projected to (chainage, height), the upper outline, the ground
  line, and the stretches over ground the flight never observed; exported as SVG (true
  scale, with a scale bar) and DXF R12 (layers POINTS / OUTLINE / GROUND / UNOBSERVED).
* ``terrain_rasters`` (ARC-01) - multi-directional hillshade, a local relief model (DTM
  minus its 15 m low-pass; Hesse 2010) and slope, from the scene's measured ground grid.
"""
import math

import numpy as np
from scipy import ndimage


class GeometryError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _bearing(dx, dz):
    return (math.degrees(math.atan2(dx, -dz)) + 360.0) % 360.0


# ------------------------------------------------------------------ INF-04 tilt
def tilt(points, base, *, radius_m=0.8, min_height_m=1.0, max_height_m=80.0):
    points = np.asarray(points, float)
    base = np.asarray(base, float)
    horiz = np.hypot(points[:, 0] - base[0], points[:, 2] - base[2])
    pick = (horiz <= radius_m) & (points[:, 1] >= base[1] + 0.2) & (points[:, 1] <= base[1] + max_height_m)
    if pick.sum() < 20:
        raise GeometryError(422, f"only {int(pick.sum())} scan points within {radius_m:g} m of the base; "
                                 "click the foot of the pole, or widen the radius")
    pts = points[pick]
    for _ in range(3):
        centre = pts.mean(0)
        axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
        if axis[1] < 0:
            axis = -axis
        rel = points - centre
        along = rel @ axis
        dist = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        band = (dist <= radius_m * 0.6) & (points[:, 1] >= base[1] + 0.2)
        if band.sum() < 20:
            break
        pts = points[band]
    centre = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0] if vt[0][1] >= 0 else -vt[0]
    along = (pts - centre) @ axis
    height = float(along.max() - along.min())
    if height < min_height_m:
        raise GeometryError(422, f"the fitted element is only {height:.2f} m tall; not a pole or mast")
    linearity = float((s[0] - s[1]) / max(s[0], 1e-9))
    angle = math.degrees(math.acos(min(1.0, abs(axis[1]))))
    resid = np.linalg.norm((pts - centre) - np.outer(along, axis), axis=1)
    return {"tilt_deg": round(angle, 2), "lean_bearing_deg": round(_bearing(axis[0], axis[2]), 1),
            "lean_mm_per_m": round(1000.0 * math.tan(math.radians(angle)), 1),
            "top_offset_m": round(height * math.sin(math.radians(angle)), 3), "height_m": round(height, 2),
            "points": int(len(pts)), "linearity": round(linearity, 3),
            "radius_rms_m": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            "axis": [round(float(v), 5) for v in axis], "centre": [round(float(v), 3) for v in centre],
            "basis": "PCA line through the scan points in a cylinder around the clicked base; bearings are scene "
                     "bearings (-z north). Splat centres are a few cm noisy, so tilts under ~0.5 deg are not "
                     "distinguishable from vertical on a scan."}


# ------------------------------------------------------------------ INF-05 wire
def _catenary(params, s):
    a, s0, c = params
    return c + a * (np.cosh((s - s0) / a) - 1.0)


def wire(points, a, b, *, half_width_m=0.6, min_above_ground_m=2.0, ground=None, clearance_limit_m=None):
    from scipy.optimize import least_squares
    points = np.asarray(points, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    span_xz = b[[0, 2]] - a[[0, 2]]
    span = float(np.hypot(*span_xz))
    if span < 2.0:
        raise GeometryError(400, "pick the two attachment points at least 2 m apart")
    u = span_xz / span
    n = np.array([-u[1], u[0]])
    rel = points[:, [0, 2]] - a[[0, 2]]
    s = rel @ u
    off = rel @ n
    chord = a[1] + (b[1] - a[1]) * np.clip(s / span, 0, 1)
    slab = (np.abs(off) <= half_width_m) & (s > 0.5) & (s < span - 0.5) & (points[:, 1] < chord + 1.0) & (points[:, 1] > chord - 0.35 * span - 3)
    if ground is not None:
        floor = ground.sample(points[:, [0, 2]])[0]
        slab &= points[:, 1] >= floor + min_above_ground_m
    pts, ss = points[slab], s[slab]
    if len(pts) < 8:
        raise GeometryError(422, f"only {len(pts)} scan points along the span; the wire may be too thin for the scan to have seen")
    # Keep the band nearest the chord in each metre: a wire, not the tree under it.
    bins = np.floor(ss).astype(int)
    keep = np.zeros(len(pts), bool)
    for k in np.unique(bins):
        idx = np.flatnonzero(bins == k)
        top = idx[np.argsort(-pts[idx, 1])[:3]]
        keep[top] = True
    ps, py = ss[keep], pts[keep, 1]
    ends_s = np.array([0.0, span]); ends_y = np.array([a[1], b[1]])
    s_fit, y_fit = np.r_[ps, ends_s], np.r_[py, ends_y]
    w = np.r_[np.ones(len(ps)), np.full(2, 5.0)]
    start = [max(span, 10.0), span / 2, min(a[1], b[1]) - 0.5]
    fit = least_squares(lambda p: w * (_catenary(p, s_fit) - y_fit), start, loss="soft_l1", f_scale=0.1,
                        bounds=([1.0, -span, -1e5], [1e5, 2 * span, 1e5]))
    sample_s = np.linspace(0, span, max(20, int(span * 2)))
    wire_y = _catenary(fit.x, sample_s)
    chord_y = a[1] + (b[1] - a[1]) * sample_s / span
    sag = float(np.max(chord_y - wire_y))
    resid = _catenary(fit.x, ps) - py
    # Clearance: anything scanned under the wire, not the wire's own points.
    xz = a[[0, 2]] + np.outer(sample_s, u)
    under = (np.abs(off) <= 1.5) & (s > 0) & (s < span)
    below = points[under]
    sb = s[under]
    clearance, where = None, None
    if len(below):
        wy = _catenary(fit.x, sb)
        gap = wy - below[:, 1]
        cand = gap > 0.35
        if cand.any():
            k = int(np.argmin(np.where(cand, gap, np.inf)))
            clearance, where = float(gap[k]), float(sb[k])
    if ground is not None:
        top, _ = ground.sample_top(xz)
        g = wire_y - top
        k = int(np.argmin(g))
        if clearance is None or g[k] < clearance:
            clearance, where = float(g[k]), float(sample_s[k])
    result = {"span_m": round(span, 2), "sag_m": round(sag, 3), "sag_pct_of_span": round(100 * sag / span, 2),
              "catenary_a_m": round(float(fit.x[0]), 2), "points": int(len(ps)),
              "fit_rms_m": round(float(np.sqrt(np.mean(resid ** 2))), 3),
              "min_clearance_m": None if clearance is None else round(clearance, 2),
              "clearance_at_m": None if where is None else round(where, 1),
              "line": [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for (x, z), y in zip(xz, wire_y)],
              "basis": "catenary fitted to the highest scan points per metre in a 1.2 m slab between the "
                       "attachment points; clearance to scan points and the top surface under the span"}
    if clearance_limit_m is not None and clearance is not None:
        result["clearance_ok"] = bool(clearance >= clearance_limit_m)
        result["clearance_limit_m"] = clearance_limit_m
    return result


# ------------------------------------------------------------------ ARC-02 section
def section(points, a, b, *, half_width_m=0.25, bin_m=0.1, ground=None, colors=None):
    points = np.asarray(points, float)
    a, b = np.asarray(a, float)[[0, 2]], np.asarray(b, float)[[0, 2]]
    length = float(np.hypot(*(b - a)))
    if length < 0.5:
        raise GeometryError(400, "a section needs two points at least 0.5 m apart")
    u = (b - a) / length
    n = np.array([-u[1], u[0]])
    rel = points[:, [0, 2]] - a
    s, off = rel @ u, rel @ n
    pick = (np.abs(off) <= half_width_m) & (s >= 0) & (s <= length)
    if pick.sum() < 10:
        raise GeometryError(422, f"only {int(pick.sum())} scan points in the section slab; widen it")
    ps, py = s[pick], points[pick, 1]
    nb = max(1, int(math.ceil(length / bin_m)))
    idx = np.clip((ps / bin_m).astype(int), 0, nb - 1)
    top = np.full(nb, np.nan)
    np.fmax.at(top, idx, py)
    stations = (np.arange(nb) + 0.5) * bin_m
    ground_line, observed = None, None
    if ground is not None:
        xz = a + np.outer(stations, u)
        g, sup = ground.sample(xz)
        ground_line, observed = g, sup
    out = {"length_m": round(length, 3), "half_width_m": half_width_m, "points": int(pick.sum()),
           "station_m": stations, "outline_m": top, "ground_m": ground_line, "observed": observed,
           "pts_s": ps, "pts_y": py, "pts_rgb": None if colors is None else np.asarray(colors)[pick],
           "azimuth_deg": round(_bearing(u[0], u[1]), 1), "a": a.tolist(), "b": b.tolist(),
           "y_range": [round(float(np.nanmin(py)), 3), round(float(np.nanmax(py)), 3)]}
    return out


def section_svg(sec, *, title="Section", scale_px_per_m=None):
    """True-scale SVG: points, upper outline, ground (dashed where unobserved), scale bar."""
    length = sec["length_m"]
    y0, y1 = sec["y_range"]
    if sec["ground_m"] is not None:
        y0 = min(y0, float(np.nanmin(sec["ground_m"])))
    height = max(y1 - y0, 0.5)
    k = scale_px_per_m or max(20.0, min(400.0, 1000.0 / max(length, height)))
    pad = 40
    W, H = int(length * k + 2 * pad), int(height * k + 2 * pad + 40)
    X = lambda s: pad + s * k
    Y = lambda y: pad + (y1 - y) * k
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{pad}" y="20" font-family="sans-serif" font-size="13">{title} — {length:.2f} m, azimuth '
             f'{sec["azimuth_deg"]:.0f}° (scene), slab ±{sec["half_width_m"]:.2f} m</text>']
    step = max(1, len(sec["pts_s"]) // 20000)
    rgb = sec["pts_rgb"]
    # Dots about 4 cm across at drawing scale, never under a pixel: a sparse slab still reads.
    radius = max(0.9, min(4.0, 0.02 * k))
    for i in range(0, len(sec["pts_s"]), step):
        colour = "#555" if rgb is None else "#%02x%02x%02x" % tuple(int(c) for c in rgb[i])
        parts.append(f'<circle cx="{X(sec["pts_s"][i]):.1f}" cy="{Y(sec["pts_y"][i]):.1f}" r="{radius:.1f}" fill="{colour}"/>')
    ok = np.isfinite(sec["outline_m"])
    path = " ".join(f'{"M" if j == 0 or not ok[j - 1] else "L"}{X(s):.1f},{Y(y):.1f}'
                    for j, (s, y) in enumerate(zip(sec["station_m"], sec["outline_m"])) if ok[j])
    parts.append(f'<path d="{path}" fill="none" stroke="#c0392b" stroke-width="2.4"/>')
    if sec["ground_m"] is not None:
        for j in range(1, len(sec["station_m"])):
            dash = "" if sec["observed"][j] and sec["observed"][j - 1] else ' stroke-dasharray="4 3"'
            parts.append(f'<line x1="{X(sec["station_m"][j - 1]):.1f}" y1="{Y(sec["ground_m"][j - 1]):.1f}" '
                         f'x2="{X(sec["station_m"][j]):.1f}" y2="{Y(sec["ground_m"][j]):.1f}" stroke="#2c3e50" stroke-width="1.2"{dash}/>')
    bar = next(v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if v * k >= 60 or v == 100)
    parts.append(f'<line x1="{pad}" y1="{H - 20}" x2="{pad + bar * k:.1f}" y2="{H - 20}" stroke="black" stroke-width="3"/>')
    parts.append(f'<text x="{pad}" y="{H - 26}" font-family="sans-serif" font-size="11">{bar:g} m</text>')
    parts.append(f'<text x="{W - pad}" y="{H - 8}" text-anchor="end" font-family="sans-serif" font-size="9" fill="#666">'
                 'Red: upper outline of the scan · solid: measured ground · dashed: ground never observed</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def section_dxf(sec):
    """DXF R12: section in its own plane (X = chainage, Y = height)."""
    out = ["0", "SECTION", "2", "ENTITIES"]

    def poly(layer, xs, ys):
        out.extend(["0", "POLYLINE", "8", layer, "66", "1", "10", "0", "20", "0", "30", "0"])
        for x, y in zip(xs, ys):
            out.extend(["0", "VERTEX", "8", layer, "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0"])
        out.extend(["0", "SEQEND", "8", layer])

    step = max(1, len(sec["pts_s"]) // 50000)
    for i in range(0, len(sec["pts_s"]), step):
        out.extend(["0", "POINT", "8", "POINTS", "10", f"{sec['pts_s'][i]:.4f}", "20", f"{sec['pts_y'][i]:.4f}", "30", "0"])
    ok = np.isfinite(sec["outline_m"])
    run = []
    for j in range(len(ok) + 1):
        if j < len(ok) and ok[j]:
            run.append(j)
        elif len(run) > 1:
            poly("OUTLINE", sec["station_m"][run], sec["outline_m"][run])
            run = []
        else:
            run = []
    if sec["ground_m"] is not None:
        obs = np.asarray(sec["observed"], bool)
        for layer, mask in (("GROUND", obs), ("UNOBSERVED", ~obs)):
            run = []
            for j in range(len(mask) + 1):
                if j < len(mask) and mask[j]:
                    run.append(j)
                elif len(run) > 1:
                    poly(layer, sec["station_m"][run], sec["ground_m"][run])
                    run = []
                else:
                    run = []
    out.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ ARC-01 terrain rasters
def hillshade(z, cell, *, azimuths=(315, 15, 75, 135, 195, 255), altitude=35.0, z_factor=1.0):
    """Multi-directional hillshade in 0..1 (rows = +z / south, cols = +x / east)."""
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    gz, gx = np.gradient(z * z_factor, cell)
    slope = np.arctan(np.hypot(gx, gz))
    # Viewer frame: +x east, +z south, so the aspect of steepest descent is atan2(-gx, gz) from north.
    aspect = np.arctan2(-gx, gz)
    alt = math.radians(altitude)
    shade = np.zeros_like(z)
    for az in azimuths:
        a = math.radians(az)
        shade += np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(shade / len(azimuths), 0, 1)


def terrain_rasters(floor, top, supported, cell, *, lrm_window_m=15.0):
    dtm = np.where(supported, floor, np.nan)
    filled = np.where(np.isfinite(dtm), dtm, np.nanmedian(dtm))
    size = max(3, int(round(lrm_window_m / cell)) | 1)
    lowpass = ndimage.uniform_filter(filled, size=size, mode="nearest")
    lrm = np.where(supported, filled - lowpass, np.nan)
    gz, gx = np.gradient(filled, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gz)))
    dsm = np.where(supported, np.maximum(top, floor), np.nan)
    return {"hillshade_dtm": np.where(supported, hillshade(filled, cell), np.nan),
            "hillshade_dsm": np.where(supported, hillshade(np.where(np.isfinite(dsm), dsm, filled), cell), np.nan),
            "lrm": lrm, "slope_deg": np.where(supported, slope, np.nan),
            "lrm_range_m": [round(float(np.nanpercentile(lrm, 2)), 3), round(float(np.nanpercentile(lrm, 98)), 3)],
            "basis": f"from the scene's measured ground grid ({cell:g} m); LRM = DTM minus its {lrm_window_m:g} m "
                     "mean filter (Hesse 2010); unobserved cells are left blank, never interpolated for display"}


def raster_png(grid, cmap="gray", limits=None):
    """PNG bytes (RGBA; transparent where NaN) of a grid, row 0 at the top."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from PIL import Image
    g = np.asarray(grid, float)
    lo, hi = limits if limits else (np.nanmin(g), np.nanmax(g))
    norm = np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = (colormaps[cmap](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[~np.isfinite(g), 3] = 0
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()
