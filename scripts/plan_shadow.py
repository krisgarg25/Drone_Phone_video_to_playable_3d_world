"""Shadow study for the planning editor: where the sun is, and what a scheme shades.

Two surface models are compared on the scene's own grid:

* **existing** - the scanned top surface (``heights.f32``: roofs, crowns, terrain);
* **proposal** - the same with every demolished outline dropped to the measured ground
  and every proposed building and object rasterized in at its derived height.

A cell is in shadow when a ray from it toward the sun passes under that surface
somewhere along the way (a height-field ray march, so terrain and existing buildings
cast shadows too). ``instant`` answers for one moment; ``day`` counts hours of direct
sun over a time window, which is how planning guidance (e.g. the BRE "2 hours of sun on
21 March" test) is usually posed.

Tree crowns are porous when their outlines are known (the scene's tree inventory and any
proposed trees): light crossing a crown is attenuated by Beer-Lambert, exp(-k * path), with
k = 0.24 per metre so a 5 m path through a crown passes ~30% - a typical broadleaf canopy in
leaf. Buildings, terrain and unlabelled vegetation stay solid. The ray is marched at 0.7 of
a cell, and a DSM cannot shade from under an overhang.

Sun position: NOAA's solar calculator equations (Meeus), geometric elevation without
refraction - within ~0.01 deg over 1800-2100, well inside what a cell-sized raster can
resolve.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

import workspace_proposals as proposals

MAX_WORK_CELLS = 250_000
MAX_OVERLAY_QUADS = 40_000
SURFACE_LIFT_M = 0.08
OPEN_GROUND_M = 0.5
"""A cell counts as open ground when its surface is within this of the measured floor:
shadows are drawn and measured there, not on roofs or crowns."""
CROWN_EXTINCTION_PER_M = 0.24
SHADOW_COLOR, NEW_SHADOW_COLOR, LIT_COLOR = [28, 36, 64], [150, 60, 190], [250, 214, 90]
LOSS_BINS = [(0.5, 1.0, [252, 220, 120]), (1.0, 2.0, [248, 160, 70]),
             (2.0, 3.0, [232, 96, 60]), (3.0, 99.0, [180, 40, 60])]


class ShadowError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ sun
def _julian_day(when):
    when = when.astimezone(timezone.utc)
    return (when - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 2451545.0


def sun_position(lat_deg, lon_deg, when):
    """(azimuth deg clockwise from true north, elevation deg) at an aware datetime."""
    jc = (_julian_day(when) - 2451545.0) / 36525.0
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    centre = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + math.sin(2 * mr) * (0.019993 - 0.000101 * jc) + math.sin(3 * mr) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * jc)
    apparent = l0 + centre - 0.00569 - 0.00478 * math.sin(omega)
    obliquity = (23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
                 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(math.radians(obliquity)) * math.sin(math.radians(apparent)))
    y = math.tan(math.radians(obliquity / 2)) ** 2
    l0r = math.radians(l0)
    eq_time = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * ecc * math.sin(mr)
                               + 4 * ecc * y * math.sin(mr) * math.cos(2 * l0r)
                               - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * ecc * ecc * math.sin(2 * mr))
    utc = when.astimezone(timezone.utc)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    solar = (minutes + eq_time + 4 * lon_deg) % 1440
    hour_angle = math.radians(solar / 4 - 180)
    lat = math.radians(lat_deg)
    cos_zenith = (math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
    zenith = math.acos(max(-1.0, min(1.0, cos_zenith)))
    sz = math.sin(zenith)
    if abs(math.cos(lat) * sz) < 1e-12:
        azimuth = 180.0 if lat_deg > math.degrees(decl) else 0.0
    else:
        cos_az = (math.sin(lat) * math.cos(zenith) - math.sin(decl)) / (math.cos(lat) * sz)
        a = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        azimuth = (a + 180) % 360 if hour_angle > 0 else (540 - a) % 360
    return azimuth, 90.0 - math.degrees(zenith)


def enu_direction(azimuth_deg, elevation_deg):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def viewer_direction(enu, registry=None, north_deg=0.0):
    """Unit vector toward the sun in the viewer frame.

    Georeferenced: through the scene's own transform chain. Local: viewer -Z is taken as
    north, turned clockwise (seen from above) by ``north_deg``; the caller says so.
    """
    if registry is not None and registry.get("status") == "georeferenced":
        import scene_frames
        a, b = scene_frames.enu_to_viewer(np.vstack([np.zeros(3), enu]), registry)
        d = b - a
    else:
        x, z = enu[0], -enu[1]
        t = math.radians(north_deg)
        d = np.array([x * math.cos(t) - z * math.sin(t), enu[2], x * math.sin(t) + z * math.cos(t)])
    return d / np.linalg.norm(d)


# ------------------------------------------------------------------ surfaces
def _work_grid(grid):
    """Floor, top and support on a grid no larger than MAX_WORK_CELLS (max-pooled)."""
    floor = np.asarray(grid["floor"], dtype=np.float64)
    top = grid.get("top")
    top = floor.copy() if top is None else np.asarray(top, dtype=np.float64)
    top = np.where(np.isfinite(top), np.maximum(top, np.nan_to_num(floor, nan=-1e9)), floor)
    supported = np.asarray(grid["supported"], dtype=bool)
    nz, nx = floor.shape
    k = max(1, int(math.ceil(math.sqrt(nz * nx / MAX_WORK_CELLS))))
    if k > 1:
        pz, px = (-nz) % k, (-nx) % k
        def pool(a, fn, fill):
            a = np.pad(a, ((0, pz), (0, px)), constant_values=fill)
            return fn(a.reshape(a.shape[0] // k, k, a.shape[1] // k, k), axis=(1, 3))
        top = pool(top, np.nanmax, -np.inf)
        floor = pool(floor, np.nanmin, np.inf)
        supported = pool(supported, np.any, False)
    fill = np.nanmedian(floor[np.isfinite(floor)]) if np.isfinite(floor).any() else 0.0
    floor = np.where(np.isfinite(floor), floor, fill)
    top = np.where(np.isfinite(top), top, floor)
    return {"floor": floor, "top": top, "supported": supported, "cell": grid["cell"] * k,
            "ox": grid["ox"], "oz": grid["oz"], "factor": k}


def _centres(work):
    nz, nx = work["floor"].shape
    xs = work["ox"] + (np.arange(nx) + 0.5) * work["cell"]
    zs = work["oz"] + (np.arange(nz) + 0.5) * work["cell"]
    return np.meshgrid(xs, zs)


def rasterize_max(surface, work, positions, indices):
    """Raise ``surface`` to every triangle's height over the cell centres it covers."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    cell, ox, oz = work["cell"], work["ox"], work["oz"]
    nz, nx = surface.shape
    for a, b, c in p[tris]:
        area = (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0])
        if abs(area) < 1e-9:
            continue                      # a vertical wall covers no cell centre
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        i0, i1 = max(0, int(math.floor((lo[0] - ox) / cell - 0.5))), min(nx - 1, int(math.ceil((hi[0] - ox) / cell - 0.5)))
        j0, j1 = max(0, int(math.floor((lo[2] - oz) / cell - 0.5))), min(nz - 1, int(math.ceil((hi[2] - oz) / cell - 0.5)))
        if i1 < i0 or j1 < j0:
            continue
        gx, gz = np.meshgrid(ox + (np.arange(i0, i1 + 1) + 0.5) * cell, oz + (np.arange(j0, j1 + 1) + 0.5) * cell)
        w1 = ((c[0] - b[0]) * (gz - b[2]) - (c[2] - b[2]) * (gx - b[0])) / area
        w2 = ((a[0] - c[0]) * (gz - c[2]) - (a[2] - c[2]) * (gx - c[0])) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if inside.any():
            y = w1 * a[1] + w2 * b[1] + w0 * c[1]
            block = surface[j0:j1 + 1, i0:i1 + 1]
            block[inside] = np.maximum(block[inside], y[inside])
    return surface


def surfaces(grid, evaluation):
    """(work grid, existing surface, proposal surface) for one evaluated proposal."""
    work = _work_grid(grid)
    existing = work["top"].copy()
    proposed = existing.copy()
    gx, gz = _centres(work)
    pts = np.column_stack([gx.ravel(), gz.ravel()])
    for derived in evaluation["features"].values():
        if "clip" in derived:
            inside = proposals.contains(np.asarray(derived["clip"]["polygon"]), pts).reshape(proposed.shape)
            proposed[inside] = work["floor"][inside]
    for derived in evaluation["features"].values():
        for mesh in derived["meshes"]:
            if mesh["kind"] in ("building", "object"):
                rasterize_max(proposed, work, mesh["positions"], mesh["indices"])
    return work, existing, proposed


def shade(surface, work, direction):
    """Boolean mask of cells whose ray toward the sun passes under ``surface``."""
    d = np.asarray(direction, dtype=np.float64)
    horizontal = math.hypot(d[0], d[2])
    if d[1] <= 0:
        return np.ones(surface.shape, dtype=bool)
    if horizontal < 1e-9:
        return np.zeros(surface.shape, dtype=bool)
    rise = d[1] / horizontal
    dx, dz = d[0] / horizontal, d[2] / horizontal
    nz, nx = surface.shape
    cell = work["cell"]
    step = 0.7 * cell
    reach = min(600.0, (float(surface.max()) - float(surface.min())) / rise + cell)
    jj, ii = np.meshgrid(np.arange(nz, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing="ij")
    start = surface + 0.02
    out = np.zeros(surface.shape, dtype=bool)
    tolerance = 0.12
    for k in range(1, int(reach / step) + 2):
        t = k * step
        si = np.rint(ii + dx * t / cell).astype(np.int64)
        sj = np.rint(jj + dz * t / cell).astype(np.int64)
        valid = (si >= 0) & (si < nx) & (sj >= 0) & (sj < nz)
        if not valid.any():
            break
        h = np.where(valid, surface[np.clip(sj, 0, nz - 1), np.clip(si, 0, nx - 1)], -np.inf)
        out |= h > start + t * rise + tolerance
    return out


def light(surface, work, direction, porous, *, k=CROWN_EXTINCTION_PER_M):
    """Fraction of direct sun reaching each cell: 0 behind a solid, exp(-k*path) through crowns."""
    d = np.asarray(direction, dtype=np.float64)
    horizontal = math.hypot(d[0], d[2])
    if d[1] <= 0:
        return np.zeros(surface.shape)
    if horizontal < 1e-9:
        return np.ones(surface.shape)
    rise = d[1] / horizontal
    dx, dz = d[0] / horizontal, d[2] / horizontal
    nz, nx = surface.shape
    cell = work["cell"]
    step = 0.7 * cell
    step3d = step * math.sqrt(1 + rise * rise)
    reach = min(600.0, (float(surface.max()) - float(surface.min())) / rise + cell)
    jj, ii = np.meshgrid(np.arange(nz, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing="ij")
    start = surface + 0.02
    solid = np.zeros(surface.shape, dtype=bool)
    crown = np.zeros(surface.shape)
    tolerance = 0.12
    # A cell's own crown does not shade the ground under it twice: path starts one step out.
    for kk in range(1, int(reach / step) + 2):
        t = kk * step
        si = np.rint(ii + dx * t / cell).astype(np.int64)
        sj = np.rint(jj + dz * t / cell).astype(np.int64)
        valid = (si >= 0) & (si < nx) & (sj >= 0) & (sj < nz)
        if not valid.any():
            break
        ci, cj = np.clip(si, 0, nx - 1), np.clip(sj, 0, nz - 1)
        h = np.where(valid, surface[cj, ci], -np.inf)
        under = h > start + t * rise + tolerance
        leafy = porous[cj, ci] & valid
        crown += np.where(under & leafy, step3d, 0.0)
        solid |= under & ~leafy
    return np.where(solid, 0.0, np.exp(-k * crown))


def porous_mask(work, polygons, surface):
    """Work-grid cells inside any crown outline that stand at least 1 m above the floor."""
    mask = np.zeros(work["floor"].shape, dtype=bool)
    if not polygons:
        return mask
    gx, gz = _centres(work)
    pts = np.column_stack([gx.ravel(), gz.ravel()])
    for poly in polygons:
        mask |= proposals.contains(np.asarray(poly, dtype=np.float64), pts).reshape(mask.shape)
    return mask & (surface - work["floor"] >= 1.0)


# ------------------------------------------------------------------ overlay meshes
def _quads(mask, ground, work, color, opacity, part, budget):
    """Run-length merged cell quads over ``mask``, just above the measured ``ground``.

    A run is cut wherever the ground steps by more than 0.3 m, and at 12 cells, so a
    merged quad never floats over a dip or buries itself in a rise.
    """
    cell, ox, oz = work["cell"], work["ox"], work["oz"]
    positions, indices = [], []
    for j in range(mask.shape[0]):
        row = mask[j]
        if not row.any():
            continue
        steps = np.abs(np.diff(ground[j])) > 0.3
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            cuts = [start] + [k + 1 for k in range(start, stop - 1) if steps[k]] + [stop]
            for a, b in zip(cuts[:-1], cuts[1:]):
                for s in range(a, b, 12):
                    if len(indices) >= 2 * budget:
                        break
                    e = min(b, s + 12)
                    y = float(ground[j, s:e].max()) + SURFACE_LIFT_M
                    x0, x1 = ox + s * cell, ox + e * cell
                    z0, z1 = oz + j * cell, oz + (j + 1) * cell
                    base = len(positions)
                    positions += [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]
                    indices += [[base, base + 2, base + 1], [base, base + 3, base + 2]]
    quads = len(indices) // 2
    if not quads:
        return None, 0
    return proposals._mesh("overlay", positions, indices, color, opacity, part=part), quads


def _open(surface, work):
    """Open ground: cells whose surface is the measured floor (no roof, crown or massing)."""
    return (surface - work["floor"]) < OPEN_GROUND_M


def _area(mask, work):
    return round(float(mask[work["supported"]].sum()) * work["cell"] ** 2, 1)


# ------------------------------------------------------------------ study
def _location(registry, lat, lon):
    if registry is not None and registry.get("status") == "georeferenced":
        o = registry["origin"]
        return o["latitude_deg"], o["longitude_deg"], "scene GPS fit"
    if lat is None or lon is None:
        raise ShadowError(409, "This scene has no GPS fit: enter the site's latitude and longitude for a shadow study.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ShadowError(400, "Latitude must be -90..90 and longitude -180..180.")
    return lat, lon, "entered by the user"


def _moment(date, time, utc_offset):
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
        hh, mm = (int(v) for v in time.split(":"))
    except (TypeError, ValueError):
        raise ShadowError(400, "Give the date as YYYY-MM-DD and the time as HH:MM.")
    if not (0 <= hh <= 23 and 0 <= mm <= 59) or not (-14 <= utc_offset <= 14):
        raise ShadowError(400, "Time must be 00:00..23:59 and the UTC offset -14..+14 h.")
    zone = timezone(timedelta(hours=utc_offset))
    return day.replace(hour=hh, minute=mm, tzinfo=zone)


def study(grid, evaluation, registry=None, *, date, time="12:00", utc_offset=0.0, mode="instant",
          start_h=9.0, end_h=15.0, step_min=30, lat=None, lon=None, north_deg=0.0, crowns_existing=None,
          crowns_proposed=None):
    lat, lon, source = _location(registry, lat, lon)
    georeferenced = registry is not None and registry.get("status") == "georeferenced"
    work, existing, proposed = surfaces(grid, evaluation)
    porous_before = porous_mask(work, crowns_existing or [], existing)
    porous_after = porous_mask(work, list(crowns_existing or []) + list(crowns_proposed or []), proposed)
    porous = porous_before.any() or porous_after.any()

    def sunlit(surface, direction, mask):
        return light(surface, work, direction, mask) if porous else (~shade(surface, work, direction)).astype(float)
    crown_note = (f"{int(porous_after.sum())} crown cells are porous (Beer-Lambert, {CROWN_EXTINCTION_PER_M} per m: a 5 m "
                  "path passes ~30%); unlabelled vegetation, buildings and terrain are solid." if porous
                  else "No tree outlines are known here, so crowns count as solid.")
    notes = ["Shadows from the scanned surface model plus the proposed massing; overhangs are not modelled.",
             crown_note,
             f"Sun from NOAA equations at {lat:.4f}, {lon:.4f} ({source}); raster {work['cell']:.2f} m."]
    if not georeferenced:
        notes.append(f"No GPS fit: north is assumed to be viewer -Z turned {north_deg:g} deg clockwise.")
    overlay, metrics = [], []
    if mode == "instant":
        when = _moment(date, time, utc_offset)
        az, el = sun_position(lat, lon, when)
        sun = {"azimuth_deg": round(az, 2), "elevation_deg": round(el, 2), "local_time": when.isoformat()}
        if el <= 0:
            notes.insert(0, "The sun is below the horizon at this time.")
            return {"mode": mode, "sun": sun, "overlay": [], "metrics": [], "notes": notes}
        direction = viewer_direction(enu_direction(az, el), registry if georeferenced else None, north_deg)
        sun["direction_viewer"] = direction.round(5).tolist()
        before = sunlit(existing, direction, porous_before) < 0.5
        after = sunlit(proposed, direction, porous_after) < 0.5
        open_before, open_after = _open(existing, work), _open(proposed, work)
        after_open = after & open_after
        new = after_open & ~before & open_before
        lit = before & ~after & open_after
        budget = MAX_OVERLAY_QUADS
        for mask, color, opacity, part in ((after_open & ~new, SHADOW_COLOR, 0.5, "shadow"),
                                           (new, NEW_SHADOW_COLOR, 0.6, "new_shadow"),
                                           (lit, LIT_COLOR, 0.55, "newly_lit")):
            mesh, used = _quads(mask, work["floor"], work, color, opacity, part, budget)
            budget -= used
            if mesh:
                overlay.append(mesh)
        metrics = [
            {"metric": "Shadow on open ground", "unit": "m²", "existing": _area(before & open_before, work), "proposal": _area(after_open, work)},
            {"metric": "Newly shaded by the scheme", "unit": "m²", "existing": 0.0, "proposal": _area(new, work)},
            {"metric": "Sunlit again (demolitions)", "unit": "m²", "existing": 0.0, "proposal": _area(lit, work)},
            {"metric": "Longest proposed shadow", "unit": "m", "existing": 0.0,
             "proposal": round(float(_tallest(evaluation)) / math.tan(math.radians(el)), 1)},
        ]
        truncated = budget <= 0
    elif mode == "day":
        if not (0 <= start_h < end_h <= 24) or not (5 <= step_min <= 120):
            raise ShadowError(400, "The day window must be 0..24 h and the step 5..120 min.")
        base = _moment(date, "00:00", utc_offset)
        hours_before = np.zeros(existing.shape)
        hours_after = np.zeros(existing.shape)
        samples, dt = [], step_min / 60.0
        t = start_h + dt / 2
        while t < end_h:
            when = base + timedelta(hours=t)
            az, el = sun_position(lat, lon, when)
            samples.append({"local_time": when.isoformat(), "azimuth_deg": round(az, 2), "elevation_deg": round(el, 2)})
            if el > 0:
                direction = viewer_direction(enu_direction(az, el), registry if georeferenced else None, north_deg)
                # Partial sun under a crown counts partially: an hour at 30% is 0.3 h.
                hours_before += dt * sunlit(existing, direction, porous_before)
                hours_after += dt * sunlit(proposed, direction, porous_after)
            t += dt
        open_before, open_after = _open(existing, work), _open(proposed, work)
        both = open_before & open_after
        loss = np.where(both, hours_before - hours_after, 0.0)
        budget = MAX_OVERLAY_QUADS
        for lo, hi, color in LOSS_BINS:
            mesh, used = _quads((loss >= lo) & (loss < hi), work["floor"], work, color, 0.62, f"loss_{lo:g}h", budget)
            budget -= used
            if mesh:
                overlay.append(mesh)
        two_before = float(((hours_before >= 2) & open_before & work["supported"]).sum()) * work["cell"] ** 2
        two_after = float(((hours_after >= 2) & open_after & work["supported"]).sum()) * work["cell"] ** 2
        sun = {"samples": samples}
        metrics = [
            {"metric": "Losing ≥ 1 h of sun", "unit": "m²", "existing": 0.0, "proposal": _area(loss >= 1, work)},
            {"metric": "Losing ≥ 2 h of sun", "unit": "m²", "existing": 0.0, "proposal": _area(loss >= 2, work)},
            {"metric": "Open ground with ≥ 2 h sun", "unit": "m²", "existing": round(two_before, 1), "proposal": round(two_after, 1)},
            {"metric": "Mean sun on open ground", "unit": "h", "existing": round(float(hours_before[open_before].mean()), 2) if open_before.any() else 0.0,
             "proposal": round(float(hours_after[open_after].mean()), 2) if open_after.any() else 0.0},
        ]
        truncated = budget <= 0
        notes.append(f"Sun hours counted from {start_h:g}:00 to {end_h:g}:00 every {step_min} min.")
    else:
        raise ShadowError(400, "mode must be instant or day")
    for row in metrics:
        row["change"] = round(row["proposal"] - row["existing"], 2)
    if truncated:
        notes.append("The overlay was trimmed to its size budget; the metrics cover the whole scene.")
    return {"mode": mode, "sun": sun, "overlay": overlay, "metrics": metrics, "notes": notes,
            "location": {"lat_deg": lat, "lon_deg": lon, "source": source}}


def _tallest(evaluation):
    heights = [d["metrics"].get("height_m", 0.0) for d in evaluation["features"].values()
               if any(m["kind"] == "building" for m in d["meshes"])]
    return max(heights, default=0.0)
