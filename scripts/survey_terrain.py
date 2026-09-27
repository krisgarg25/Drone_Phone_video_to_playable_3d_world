"""Bare-earth terrain from a measured cloud: DTM, nDSM and a ground class (M2).

The DSM ``survey_export`` writes keeps rooftops and canopy. Border, construction,
archaeology and military users need the ground itself (profiles, volumes, slope, HLZ
search) and heights ABOVE ground (building and tree heights, obstacles). This module
separates the two with a Simple Morphological Filter (SMRF; Pingel, Clarke & McBride
2013, ISPRS J. 77:21-30), implemented on NumPy/SciPy only:

1. a minimum-height grid at ``cell_m``; empty cells take their nearest filled value
   (and are remembered as empty);
2. single low blunders (a cell far below its neighbourhood median - the classic MVS
   negative outlier) are removed so an opening cannot eat down to them;
3. greyscale openings with square windows of growing radius (1-4 cells, then doubling,
   up to ``max_window_m`` - the progressive schedule of Zhang et al. 2003); a cell is an
   object when the opening lowers it by more than ``slope * radius`` - buildings and
   trees drop out, terrain that rises no steeper than ``slope`` survives. SMRF uses
   disks; squares are separable, which is what makes a square kilometre take seconds;
4. the remaining ground cells are interpolated across object and empty cells;
5. every point within ``elevation_threshold_m + scaling * local DTM slope * cell_m`` of
   that surface is ground (LAS class 2); everything else is left unclassified (1).

What the output does and does not claim is carried in the result:

* ``ground_observed`` marks DTM cells that contain a ground point; everything else is
  ``interpolated`` - under a roof, under dense canopy, or never seen. A single pass
  cannot see the ground under a building and the DTM does not pretend to.
* ``max_window_m`` must exceed half the widest building; a larger structure keeps its
  roof in the DTM, which shows up as an nDSM near zero on a known building.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

GROUND, UNCLASSIFIED = 2, 1


def _radii(max_radius):
    """1..4 cells one at a time (small objects, steps), then doubling (Zhang et al. 2003)."""
    radii = list(range(1, min(4, max_radius) + 1))
    while radii[-1] < max_radius:
        radii.append(min(radii[-1] * 2, max_radius))
    return radii


def _grid_index(enu, cell, origin=None):
    xmin, ymin = (float(enu[:, 0].min()), float(enu[:, 1].min())) if origin is None else origin
    cols = int(np.floor((enu[:, 0].max() - xmin) / cell)) + 1
    rows = int(np.floor((enu[:, 1].max() - ymin) / cell)) + 1
    col = np.clip(np.floor((enu[:, 0] - xmin) / cell).astype(np.int64), 0, cols - 1)
    row = np.clip(np.floor((enu[:, 1] - ymin) / cell).astype(np.int64), 0, rows - 1)
    return (xmin, ymin), rows, cols, row, col


def _reduce(values, flat, size, how):
    out = np.full(size, np.nan)
    order = np.lexsort((values, flat))
    flat_sorted, values_sorted = flat[order], values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
    ends = np.r_[starts[1:], len(flat_sorted)]
    pick = starts if how == "min" else ends - 1
    out[flat_sorted[pick]] = values_sorted[pick]
    return out


def _fill_nearest(grid, valid):
    _, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return grid[ri, ci]


def _interpolate(grid, known):
    rows, cols = grid.shape
    rr, cc = np.nonzero(known)
    if len(rr) < 3:
        return _fill_nearest(grid, known)
    target_r, target_c = np.nonzero(~known)
    out = grid.copy()
    if len(target_r):
        values = griddata((rr, cc), grid[known], (target_r, target_c), method="linear")
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = _fill_nearest(grid, known)[target_r[missing], target_c[missing]]
        out[target_r, target_c] = values
    return out


def classify_ground(enu, *, cell_m=1.0, max_window_m=18.0, slope=0.15,
                    elevation_threshold_m=0.5, scaling=1.25, blunder_m=None):
    """SMRF ground filter. Returns a dict of grids, per-point classes and a report."""
    enu = np.asarray(enu, dtype=np.float64)
    if enu.ndim != 2 or enu.shape[1] != 3 or len(enu) < 10 or not np.isfinite(enu).all():
        raise ValueError("classify_ground needs at least 10 finite ENU points")
    for name, value in (("cell_m", cell_m), ("max_window_m", max_window_m), ("slope", slope),
                        ("elevation_threshold_m", elevation_threshold_m)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive")
    origin, rows, cols, row, col = _grid_index(enu, cell_m)
    if rows * cols > 60_000_000:
        raise ValueError(f"{rows}x{cols} grid at {cell_m} m is too large; raise cell_m or tile")
    flat = row * cols + col
    zmin = _reduce(enu[:, 2], flat, rows * cols, "min").reshape(rows, cols)
    zmax = _reduce(enu[:, 2], flat, rows * cols, "max").reshape(rows, cols)
    occupied = np.isfinite(zmin)
    surface = _fill_nearest(zmin, occupied)

    blunder = 2.0 * elevation_threshold_m + 5 * cell_m * slope if blunder_m is None else blunder_m
    local = ndimage.median_filter(surface, size=5, mode="nearest")
    low = occupied & (surface < local - blunder)
    if low.any():
        surface = _fill_nearest(np.where(low, np.nan, surface), occupied & ~low)

    objects = np.zeros_like(occupied)
    previous = surface
    for radius in _radii(max(1, int(round(max_window_m / cell_m)))):
        # Square windows are separable (O(radius) per cell), so a square-kilometre site
        # at 1 m opens in seconds; disks cost O(radius^2) per cell.
        size = 2 * radius + 1
        opened = ndimage.maximum_filter(ndimage.minimum_filter(previous, size=size, mode="nearest"),
                                        size=size, mode="nearest")
        objects |= (previous - opened) > slope * radius * cell_m
        previous = opened
    ground_cells = occupied & ~objects & ~low
    if ground_cells.sum() < 3:
        raise ValueError("SMRF found fewer than three ground cells; check max_window_m and slope")
    dtm = _interpolate(np.where(ground_cells, zmin, np.nan), ground_cells)

    gy, gx = np.gradient(dtm, cell_m)
    dtm_slope = np.hypot(gx, gy)
    tolerance = elevation_threshold_m + scaling * dtm_slope[row, col] * cell_m
    height = enu[:, 2] - dtm[row, col]
    is_ground = np.abs(height) <= tolerance
    classes = np.where(is_ground, GROUND, UNCLASSIFIED).astype(np.uint8)
    observed = np.zeros(rows * cols, dtype=bool)
    observed[flat[is_ground]] = True
    observed = observed.reshape(rows, cols)
    dsm = np.where(occupied, zmax, np.nan)
    ndsm = np.where(occupied, zmax - dtm, np.nan)
    transform = (origin[0], cell_m, 0.0, origin[1] + rows * cell_m, 0.0, -cell_m)
    report = {
        "method": "SMRF (Pingel et al. 2013) on NumPy/SciPy",
        "parameters": {"cell_m": cell_m, "max_window_m": max_window_m, "slope": slope,
                       "elevation_threshold_m": elevation_threshold_m, "scaling": scaling,
                       "blunder_m": blunder},
        "points": int(len(enu)), "ground_points": int(is_ground.sum()),
        "ground_fraction": round(float(is_ground.mean()), 4),
        "cells": int(rows * cols), "cells_occupied": int(occupied.sum()),
        "cells_ground_observed": int(observed.sum()),
        "cells_interpolated": int(rows * cols - observed.sum()),
        "low_blunder_cells": int(low.sum()),
        "notes": ["DTM cells without a ground point are interpolated, not observed; see the "
                  "ground_observed mask. A single pass does not see under roofs or dense canopy.",
                  f"Structures wider than about {2 * max_window_m:g} m can survive as terrain."]}
    # Rasters are north-up: row 0 is the northern edge, matching survey_export.dsm_grid.
    flip = lambda grid: grid[::-1].astype(np.float32)
    return {"dtm": flip(dtm), "dsm": flip(dsm), "ndsm": flip(ndsm),
            "ground_observed": observed[::-1].copy(), "transform": transform,
            "classification": classes, "height_above_ground": height, "report": report}
