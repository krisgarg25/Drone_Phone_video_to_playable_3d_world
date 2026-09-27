"""Two-epoch change detection on measured clouds (M3; CON-03, BOR-03, DIS-08, DIS-03).

Two flights of the same place never land in exactly the same frame: GNSS alone leaves
metre-level offsets between epochs, which a naive DSM difference reports as change on
every slope. This module:

1. grids both epochs on one shared north-up raster (per-cell mean height and spread; the
   mean, not the maximum, because a maximum of a few noisy samples is biased upwards by
   noise and by the slope inside the cell, and that bias differs between epochs);
2. co-registers the second epoch to the first by a horizontal shift search (coarse
   whole cells, then sub-cell) plus a vertical median, scored by the NMAD of the height
   difference on *stable* cells. Stable cells are found iteratively (within 3 NMAD of
   the median) or given by the caller (e.g. ground class, roads, or a drawn mask). The
   shift search is the idea behind Nuth & Kaab 2011 (TC 5:271) done by brute force and
   scored by a trimmed mean absolute deviation (see ``_score``). Uniform planar terrain
   cannot fix a horizontal shift at all; relief is what makes it observable;
3. differences the surfaces and attaches a per-cell level of detection at 95%:
   ``LoD95 = 1.96 * sqrt(sigma_reg^2 + se_a^2 + se_b^2)`` where ``sigma_reg`` is the
   stable-cell NMAD after registration and ``se`` the within-cell standard error;
4. keeps only changes beyond LoD95, groups them into regions, and reports each region's
   area, mean/max height change, volume with uncertainty and (when georeferenced) its
   WGS84 centre and MGRS.

What it does not claim: cells empty in either epoch are ``unobserved``, never "no
change"; a difference inside LoD95 is "not detected", not "unchanged"; this is a 2.5D
surface difference, so change under canopy or on vertical faces is not measured
(M3C2 on the full clouds is not implemented here).
"""
import math

import numpy as np
from scipy import ndimage

NMAD = 1.4826
UNOBSERVED, NOT_DETECTED, GAIN, LOSS = -1, 0, 1, 2


def _cloud(value, name):
    cloud = np.asarray(value, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or len(cloud) < 10:
        raise ValueError(f"{name} must be an (N, 3) cloud with at least 10 points")
    cloud = cloud[:, :3]
    if not np.isfinite(cloud).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return cloud


def _nmad(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(NMAD * np.median(np.abs(values - np.median(values))))


def grid_surface(enu, *, cell_m, bounds):
    """Mean height, zmax, count and within-cell std on a north-up grid over ``bounds`` (x0, y0, x1, y1).

    Row 0 is the northern edge; the transform is GDAL-style, the same as
    ``survey_export.dsm_grid`` and ``survey_terrain``.
    """
    x0, y0, x1, y1 = bounds
    cols = max(1, int(math.ceil((x1 - x0) / cell_m)))
    rows = max(1, int(math.ceil((y1 - y0) / cell_m)))
    if rows * cols > 60_000_000:
        raise ValueError(f"{rows}x{cols} grid at {cell_m} m is too large; raise cell_m or tile")
    col = np.floor((enu[:, 0] - x0) / cell_m).astype(np.int64)
    row = rows - 1 - np.floor((enu[:, 1] - y0) / cell_m).astype(np.int64)
    keep = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    flat = row[keep] * cols + col[keep]
    z = enu[keep, 2]
    size = rows * cols
    count = np.bincount(flat, minlength=size).astype(np.float64)
    zmax = np.full(size, -np.inf)
    np.maximum.at(zmax, flat, z)
    total = np.bincount(flat, weights=z, minlength=size)
    square = np.bincount(flat, weights=z * z, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean * mean, 0.0))
    empty = count == 0
    zmax[empty] = np.nan
    std[empty] = np.nan
    mean[empty] = np.nan
    shape = (rows, cols)
    transform = (x0, cell_m, 0.0, y0 + rows * cell_m, 0.0, -cell_m)
    return dict(z=mean.reshape(shape), zmax=zmax.reshape(shape), count=count.reshape(shape).astype(np.int32),
                std=std.reshape(shape), transform=transform)


def _shifted(grid, dx_cells, dy_cells):
    """Move the content of ``grid`` by dx cells east and dy cells north.

    Bilinear; NaN wherever any contributing cell is empty, so a shift never invents
    surface from a hole.
    """
    valid = np.isfinite(grid).astype(np.float64)
    filled = np.where(np.isfinite(grid), grid, 0.0)
    shift = (-dy_cells, dx_cells)            # rows grow southwards
    moved = ndimage.shift(filled, shift, order=1, mode="constant", cval=0.0)
    weight = ndimage.shift(valid, shift, order=1, mode="constant", cval=0.0)
    return np.where(weight > 0.999, moved, np.nan)


def _score(ref, mov, stable, dx, dy):
    """Trimmed mean absolute deviation of dh about its median, plus a tiny shift penalty.

    Not the NMAD: on terrain that is mostly one planar slope a horizontal shift turns
    into a *constant* dh over most cells, whose median absolute deviation is zero at every
    shift - the search could not tell 2 m from 0 m. The mean is pulled up by the cells
    where the shift misaligns relief (edges, ridges, walls); trimming the top 5% keeps
    real change from steering it. Ties go to the smaller shift.
    """
    dh = _shifted(mov, dx, dy) - ref
    use = stable & np.isfinite(dh)
    if use.sum() < 50:
        return float("inf"), dh
    dev = np.abs(dh[use] - np.median(dh[use]))
    cut = np.percentile(dev, 95)
    return float(dev[dev <= cut].mean()) + 1e-9 * (dx * dx + dy * dy), dh


def coregister(ref, mov, *, cell_m, max_shift_m=3.0, stable=None, iterations=4):
    """Shift ``mov`` onto ``ref`` (both grids on the same transform).

    Returns ``(dx_m, dy_m, dz_m, stable_mask, sigma_reg_m)``: the east/north/up
    translation that, added to the second epoch, best matches the first.
    """
    both = np.isfinite(ref) & np.isfinite(mov)
    if both.sum() < 100:
        raise ValueError("the epochs overlap in fewer than 100 cells; they may not be "
                         "the same site, or the grids do not share bounds")
    mask = both if stable is None else (both & np.asarray(stable, bool))
    if mask.sum() < 100:
        raise ValueError("fewer than 100 stable cells observed in both epochs")
    reach = max(1, int(round(max_shift_m / cell_m)))
    best = (0.0, 0.0)
    for _ in range(iterations):
        scores = {}
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                scores[(dx, dy)] = _score(ref, mov, mask, float(dx), float(dy))[0]
        coarse = min(scores, key=scores.get)
        fine = {}
        for fx in np.arange(-1.0, 1.0001, 0.1):
            for fy in np.arange(-1.0, 1.0001, 0.1):
                key = (coarse[0] + fx, coarse[1] + fy)
                fine[key] = _score(ref, mov, mask, *key)[0]
        best = min(fine, key=fine.get)
        _, dh = _score(ref, mov, mask, *best)
        use = mask & np.isfinite(dh)
        centre, spread = float(np.median(dh[use])), _nmad(dh[use])
        if stable is not None:
            break
        # Refine the stable set: real change drops out, so the next search is not
        # pulled towards the pile that moved.
        new_mask = both & np.isfinite(dh) & (np.abs(dh - centre) <= 3.0 * max(spread, 1e-3))
        if new_mask.sum() < 100 or np.array_equal(new_mask, mask):
            mask = new_mask if new_mask.sum() >= 100 else mask
            break
        mask = new_mask
    _, dh = _score(ref, mov, mask, *best)
    use = mask & np.isfinite(dh)
    dz = float(np.median(dh[use]))
    sigma = _nmad(dh[use])
    return best[0] * cell_m, best[1] * cell_m, -dz, mask, sigma


def _regions(labels, count, dh, lod, cell_m, transform, frame, min_cells, sign):
    out = []
    area_cell = cell_m * cell_m
    a, b, _, d, _, f = transform
    for index in range(1, count + 1):
        rr, cc = np.nonzero(labels == index)
        if rr.size < min_cells:
            continue
        values = dh[rr, cc]
        volume = float(values.sum() * area_cell)
        # Registration error is common to every cell of a region (fully correlated);
        # within-cell error is independent. Sum both honestly.
        sigma_cells = lod[rr, cc] / 1.96
        vol_sigma = float(area_cell * math.sqrt(float(np.sum(sigma_cells ** 2))))
        x = a + (cc + 0.5) * b
        y = d + (rr + 0.5) * f
        entry = dict(id=f"{'G' if sign > 0 else 'L'}{len(out) + 1}",
                     kind="gain" if sign > 0 else "loss",
                     cells=int(rr.size), area_m2=round(float(rr.size * area_cell), 2),
                     mean_dh_m=round(float(values.mean()), 3),
                     extreme_dh_m=round(float(values.max() if sign > 0 else values.min()), 3),
                     volume_m3=round(volume, 2), volume_sigma_m3=round(vol_sigma, 2),
                     centre_enu=[round(float(x.mean()), 2), round(float(y.mean()), 2)],
                     bbox_enu=[round(float(x.min() - b / 2), 2), round(float(y.min() + f / 2), 2),
                               round(float(x.max() + b / 2), 2), round(float(y.max() - f / 2), 2)])
        if frame is not None:
            entry.update(_where(entry["centre_enu"], frame))
        out.append(entry)
    return out


def _where(xy, frame):
    import survey_coords
    import survey_measure
    lat, lon, _ = survey_measure.enu_to_geodetic([xy[0], xy[1], 0.0], frame)
    return dict(lat=round(float(lat), 7), lon=round(float(lon), 7),
                mgrs=survey_coords.to_mgrs(float(lat), float(lon)))


def detect_change(before, after, *, cell_m=1.0, max_shift_m=3.0, stable=None,
                  min_region_m2=4.0, register=True, frame=None, bounds=None):
    """Difference two epochs of ENU points (after minus before), with LoD95.

    ``stable`` optionally marks cells known not to change (same grid shape as the
    result, row 0 north). ``frame`` (the survey_measure ENU frame dict) adds WGS84 and
    MGRS to each region. Returns grids and a JSON-safe report.
    """
    before, after = _cloud(before, "before"), _cloud(after, "after")
    if not (math.isfinite(cell_m) and cell_m > 0):
        raise ValueError("cell_m must be positive")
    if bounds is None:
        lo = np.minimum(before[:, :2].min(0), after[:, :2].min(0))
        hi = np.maximum(before[:, :2].max(0), after[:, :2].max(0))
        bounds = (float(lo[0]), float(lo[1]), float(hi[0]) + 1e-9, float(hi[1]) + 1e-9)
    ga = grid_surface(before, cell_m=cell_m, bounds=bounds)
    gb = grid_surface(after, cell_m=cell_m, bounds=bounds)
    if register:
        dx, dy, dz, mask, sigma = coregister(ga["z"], gb["z"], cell_m=cell_m,
                                             max_shift_m=max_shift_m, stable=stable)
        moved = _shifted(gb["z"], dx / cell_m, dy / cell_m) + dz
        moved_std = _shifted(gb["std"], dx / cell_m, dy / cell_m)
        moved_n = _shifted(gb["count"].astype(np.float64), dx / cell_m, dy / cell_m)
    else:
        dx = dy = dz = 0.0
        moved, moved_std, moved_n = gb["z"], gb["std"], gb["count"].astype(np.float64)
        both = np.isfinite(ga["z"]) & np.isfinite(moved)
        mask = both if stable is None else both & np.asarray(stable, bool)
        diff = moved - ga["z"]
        sigma = _nmad(diff[mask])
    dh = moved - ga["z"]
    observed = np.isfinite(dh)
    with np.errstate(invalid="ignore", divide="ignore"):
        se_a = ga["std"] / np.sqrt(np.maximum(ga["count"], 1))
        se_b = moved_std / np.sqrt(np.maximum(moved_n, 1.0))
    se_a = np.nan_to_num(se_a, nan=0.0)
    se_b = np.nan_to_num(se_b, nan=0.0)
    lod = 1.96 * np.sqrt(sigma ** 2 + se_a ** 2 + se_b ** 2)
    lod = np.where(observed, lod, np.nan)
    status = np.full(dh.shape, UNOBSERVED, dtype=np.int8)
    status[observed] = NOT_DETECTED
    gain = observed & (dh > lod)
    loss = observed & (dh < -lod)
    status[gain] = GAIN
    status[loss] = LOSS
    min_cells = max(1, int(math.ceil(min_region_m2 / (cell_m * cell_m))))
    structure = np.ones((3, 3), bool)
    lg, ng = ndimage.label(gain, structure)
    ll, nl = ndimage.label(loss, structure)
    regions = (_regions(lg, ng, dh, lod, cell_m, ga["transform"], frame, min_cells, +1)
               + _regions(ll, nl, dh, lod, cell_m, ga["transform"], frame, min_cells, -1))
    regions.sort(key=lambda r: -abs(r["volume_m3"]))
    area_cell = cell_m * cell_m
    total = int(dh.size)
    report = {
        "method": "2.5D DSM difference after shift co-registration; LoD95 per cell",
        "cell_m": cell_m,
        "registration": {"applied": bool(register), "dx_m": round(dx, 3), "dy_m": round(dy, 3),
                         "dz_m": round(dz, 3), "stable_cells": int(mask.sum()),
                         "sigma_reg_m": round(float(sigma), 4),
                         "stable_source": "given" if stable is not None else "iterative 3-NMAD"},
        "cells": {"total": total, "observed_both": int(observed.sum()),
                  "unobserved": int(total - observed.sum()),
                  "gain": int(gain.sum()), "loss": int(loss.sum())},
        "lod95_median_m": round(float(np.nanmedian(lod)), 4) if observed.any() else None,
        "volume": {"gain_m3": round(float(dh[gain].sum() * area_cell), 2),
                   "loss_m3": round(float(-dh[loss].sum() * area_cell), 2)},
        "regions": regions,
        "notes": ["Cells empty in either epoch are unobserved, not unchanged.",
                  "A difference below LoD95 is 'not detected'; it is not proof of no change.",
                  "2.5D surface difference: change under canopy or on vertical faces is not "
                  "measured."],
    }
    return dict(dh=dh.astype(np.float32), lod95=lod.astype(np.float32), status=status,
                transform=ga["transform"], report=report)


def region_volume(before, after, polygon, *, cell_m=0.25, max_shift_m=0.0, frame=None):
    """Volume change inside a drawn polygon (debris pile, landslide, stockpile moved).

    Registration is off by default here: a polygon over the change itself has no stable
    ground to register on. Run ``detect_change`` over the whole site first and pass its
    shift if the epochs are not already co-registered.
    """
    from matplotlib.path import Path as MplPath
    ring = np.asarray(polygon, dtype=np.float64)[:, :2]
    if len(ring) < 3:
        raise ValueError("polygon needs at least three vertices")
    lo, hi = ring.min(0), ring.max(0)
    bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    out = detect_change(before, after, cell_m=cell_m, register=max_shift_m > 0,
                        max_shift_m=max(max_shift_m, cell_m), bounds=bounds, frame=frame,
                        min_region_m2=cell_m * cell_m)
    a, b, _, d, _, f = out["transform"]
    rows, cols = out["dh"].shape
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    centres = np.column_stack([a + (cc.ravel() + 0.5) * b, d + (rr.ravel() + 0.5) * f])
    inside = MplPath(ring).contains_points(centres).reshape(rows, cols)
    dh, lod = out["dh"].astype(np.float64), out["lod95"].astype(np.float64)
    seen = inside & np.isfinite(dh)
    area = cell_m * cell_m
    coverage = float(seen.sum() / max(inside.sum(), 1))
    # Empty cells inside the polygon are sampling gaps at fine cells, not zero change:
    # summing only the seen cells under-counts by the gap fraction. They are filled by
    # linear interpolation of dh (nearest at the rim) and the fraction filled is reported.
    gaps = inside & ~seen
    if gaps.any() and seen.sum() >= 3:
        from scipy.interpolate import griddata
        known = np.column_stack(np.nonzero(seen))
        holes = np.column_stack(np.nonzero(gaps))
        fill = griddata(known, dh[seen], holes, method="linear")
        near = griddata(known, dh[seen], holes, method="nearest")
        dh[gaps] = np.where(np.isfinite(fill), fill, near)
        lod[gaps] = np.nanmax(lod[seen])
    use = inside & np.isfinite(dh)
    net = float(dh[use].sum() * area)
    sigma = float(area * math.sqrt(float(np.sum((lod[use] / 1.96) ** 2)))) if use.any() else None
    return {"net_m3": round(net, 3),
            "gain_m3": round(float(np.clip(dh[use], 0, None).sum() * area), 3),
            "loss_m3": round(float(-np.clip(dh[use], None, 0).sum() * area), 3),
            "filled_fraction": round(float(gaps.sum() / max(inside.sum(), 1)), 4),
            "sigma_m3": None if sigma is None else round(sigma, 3),
            "polygon_area_m2": round(float(inside.sum() * area), 2),
            "observed_fraction": round(coverage, 4),
            "valid": coverage >= 0.8,
            "reason": None if coverage >= 0.8 else
            f"only {coverage:.0%} of the polygon is observed in both epochs"}
