"""True orthomosaic from the frames that built the model (M1).

"GeoTIFF" to most users of a drone map means an orthophoto, not only an elevation grid.
This module colours every cell of a north-up grid from the source frames:

1. **Surface.** Each ortho cell takes its height from the DSM (nearest cell), so the
   image is draped on rooftops and ground alike - a *true* ortho, in which buildings
   stand where their footprints are instead of leaning away from the camera. Cells
   where the DSM has no data stay transparent.
2. **Frame.** Grid coordinates (ENU, or UTM with ellipsoidal or EGM96 heights) are
   mapped back into the reconstruction frame by a caller-supplied ``to_local``, the
   exact inverse of how the product's positions were made.
3. **Visibility.** Each camera builds a coarse depth buffer from the same surface
   samples; a cell is visible to a view only if it is in front of that buffer, inside
   the frame with a margin, and faces the camera at a usable angle.
4. **View choice.** Each cell takes the visible view with the best
   ``cos(incidence) * pixels per metre`` - the texture module's data term, on a grid.
5. **Sampling.** The chosen frame is sampled bilinearly through the camera model,
   lens distortion included.

Speed: a camera only projects the grid window its frame covers, found from a coarse
stride of the grid, so a nadir mapping flight touches a small window per frame.

Not done, and reported: no exposure balancing or seam blending between views, and no
moving-object test (a car that moved can appear in the cell it left). Alpha is 0 where
no camera saw the surface; nothing is filled in.
"""
import math

import numpy as np
from scipy import ndimage

try:
    from scripts import survey_crs as crs
    from scripts import survey_texture as texture
except ImportError:
    import survey_crs as crs
    import survey_texture as texture

BUFFER_WIDTH = 480
DEPTH_TOLERANCE = 0.02
EDGE_MARGIN_PX = 2.0
MIN_COS = 0.2
WINDOW_STRIDE = 8
MAX_CELLS = 64_000_000


def enu_to_local(alignment):
    """Inverse of ``survey_georef.transform_points`` for ENU grid coordinates."""
    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    return lambda xyz: ((np.asarray(xyz, dtype=np.float64) - translation) / scale) @ rotation


def utm_to_local(alignment, scene_crs, *, egm96=False, geoid=None):
    """Inverse of ``survey_crs.enu_to_crs`` (+ the EGM96 height shift when used)."""
    origin = scene_crs["origin_geodetic"]
    origin_ecef = crs.ecef_from_geodetic([origin["latitude_deg"]], [origin["longitude_deg"]],
                                         [origin["height_m"]])[0]
    basis = crs.enu_basis(origin["latitude_deg"], origin["longitude_deg"])
    to_local = enu_to_local(alignment)

    def convert(xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        lat, lon = crs.utm_inverse(xyz[:, 0], xyz[:, 1], scene_crs["zone"], scene_crs["hemisphere"])
        height = geoid.to_ellipsoidal(lat, lon, xyz[:, 2]) if egm96 else xyz[:, 2]
        ecef = crs.ecef_from_geodetic(lat, lon, height)
        return to_local((ecef - origin_ecef) @ basis.T)
    return convert


def _focal(camera):
    p = camera["params"]
    return float(p[0]) if camera["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL") \
        else 0.5 * (float(p[0]) + float(p[1]))


def _surface(dsm, transform, cell_m):
    a, b, _, d, _, f = transform
    rows_dsm, cols_dsm = dsm.shape
    width, height = cols_dsm * b, rows_dsm * -f
    cols, rows = int(math.ceil(width / cell_m)), int(math.ceil(height / cell_m))
    if rows * cols > MAX_CELLS:
        raise ValueError(f"{rows}x{cols} ortho cells exceed {MAX_CELLS}; raise the cell size")
    xs = a + (np.arange(cols) + 0.5) * cell_m
    ys = d - (np.arange(rows) + 0.5) * cell_m
    ci = np.clip(((xs - a) / b).astype(np.int64), 0, cols_dsm - 1)
    ri = np.clip(((ys - d) / f).astype(np.int64), 0, rows_dsm - 1)
    valid_dsm = np.isfinite(dsm)
    if not valid_dsm.any():
        raise ValueError("DSM has no data")
    _, (fr, fc) = ndimage.distance_transform_edt(~valid_dsm, return_indices=True)
    filled = dsm[fr, fc]
    z = filled[ri[:, None], ci[None, :]]
    supported = valid_dsm[ri[:, None], ci[None, :]]
    gy, gx = np.gradient(filled.astype(np.float64), -f, b)   # d/dy (north), d/dx (east)
    grad_x = gx[ri[:, None], ci[None, :]]
    grad_y = -gy[ri[:, None], ci[None, :]]                    # rows run southward
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, z.astype(np.float64), supported, grad_x, grad_y, (a, cell_m, 0.0, d, 0.0, -cell_m)


def orthomosaic(dsm, dsm_transform, to_local, cameras, images, load_image, *, cell_m=None,
                metres_per_local_unit=1.0, buffer_width=BUFFER_WIDTH,
                depth_tolerance=DEPTH_TOLERANCE, min_cos=MIN_COS):
    """Return (rgb uint8 HxWx3, alpha uint8 HxW, transform, report)."""
    dsm = np.asarray(dsm, dtype=np.float64)
    cell_m = float(cell_m or dsm_transform[1])
    xx, yy, zz, supported, gx, gy, transform = _surface(dsm, dsm_transform, cell_m)
    rows, cols = zz.shape
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    normal_grid = np.column_stack([-gx.ravel(), -gy.ravel(), np.ones(rows * cols)])
    normal_grid /= np.linalg.norm(normal_grid, axis=1, keepdims=True)
    local = to_local(grid)
    wall_cells, wall_points = _wall_samples(xx, yy, zz, cell_m)
    wall_local = to_local(wall_points) if len(wall_points) else np.zeros((0, 3))
    normals = to_local(grid + normal_grid) - local
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    best = np.zeros(rows * cols)
    view = np.full(rows * cols, -1, dtype=np.int64)
    pixels = np.zeros((rows * cols, 2), dtype=np.float32)
    coarse_r = np.arange(0, rows, WINDOW_STRIDE)
    coarse_c = np.arange(0, cols, WINDOW_STRIDE)
    coarse = (coarse_r[:, None] * cols + coarse_c[None, :]).ravel()
    used_views = 0
    for index, image in enumerate(images):
        camera = cameras[image["camera_id"]]
        w, h = camera["width"], camera["height"]
        px, depth = texture.project(local[coarse], image, camera)
        inside = ((depth > 0) & np.isfinite(px).all(axis=1) & (px[:, 0] >= 0) & (px[:, 0] < w)
                  & (px[:, 1] >= 0) & (px[:, 1] < h))
        if not inside.any():
            continue
        hit_r, hit_c = coarse[inside] // cols, coarse[inside] % cols
        r0, r1 = max(0, hit_r.min() - WINDOW_STRIDE), min(rows, hit_r.max() + WINDOW_STRIDE + 1)
        c0, c1 = max(0, hit_c.min() - WINDOW_STRIDE), min(cols, hit_c.max() + WINDOW_STRIDE + 1)
        ids = (np.arange(r0, r1)[:, None] * cols + np.arange(c0, c1)[None, :]).ravel()
        px, depth = texture.project(local[ids], image, camera)
        ok = ((depth > 0) & np.isfinite(px).all(axis=1)
              & (px[:, 0] >= EDGE_MARGIN_PX) & (px[:, 0] < w - EDGE_MARGIN_PX)
              & (px[:, 1] >= EDGE_MARGIN_PX) & (px[:, 1] < h - EDGE_MARGIN_PX))
        if not ok.any():
            continue
        scale = buffer_width / float(w)
        bw, bh = buffer_width, max(1, int(round(h * scale)))
        bx = np.clip((px[ok, 0] * scale).astype(np.int64), 0, bw - 1)
        by = np.clip((px[ok, 1] * scale).astype(np.int64), 0, bh - 1)
        buffer = np.full(bh * bw, np.inf)
        np.minimum.at(buffer, by * bw + bx, depth[ok])
        if len(wall_local):
            wr, wc = wall_cells // cols, wall_cells % cols
            near = (wr >= r0) & (wr < r1) & (wc >= c0) & (wc < c1)
            if near.any():
                wpx, wdepth = texture.project(wall_local[near], image, camera)
                wok = ((wdepth > 0) & np.isfinite(wpx).all(axis=1) & (wpx[:, 0] >= 0)
                       & (wpx[:, 0] < w) & (wpx[:, 1] >= 0) & (wpx[:, 1] < h))
                wx = np.clip((wpx[wok, 0] * scale).astype(np.int64), 0, bw - 1)
                wy = np.clip((wpx[wok, 1] * scale).astype(np.int64), 0, bh - 1)
                np.minimum.at(buffer, wy * bw + wx, wdepth[wok])
        buffer = ndimage.minimum_filter(buffer.reshape(bh, bw), size=3, mode="nearest").ravel()
        front = depth[ok] <= buffer[by * bw + bx] * (1 + depth_tolerance)
        cand = ids[ok][front]
        centre = texture.camera_centre(image)
        ray = centre - local[cand]
        distance = np.linalg.norm(ray, axis=1)
        cos = np.einsum("ij,ij->i", normals[cand], ray) / distance
        score = cos * _focal(camera) / depth[ok][front]
        better = (cos >= min_cos) & (score > best[cand])
        if better.any():
            used_views += 1
            best[cand[better]] = score[better]
            view[cand[better]] = index
            pixels[cand[better]] = px[ok][front][better]
    rgb = np.zeros((rows * cols, 3), dtype=np.uint8)
    observed = (view >= 0) & supported.ravel()
    counts = {}
    for index in np.unique(view[observed]):
        cells = np.flatnonzero(observed & (view == index))
        frame = load_image(images[index]["name"])
        if frame is None:
            observed[cells] = False
            continue
        rgb[cells] = np.clip(np.rint(texture._sample(frame, pixels[cells])), 0, 255).astype(np.uint8)
        counts[images[index]["name"]] = int(len(cells))
    gsd = np.where(observed, _focal_per_cell(best, observed), np.nan)
    alpha = np.where(observed, 255, 0).astype(np.uint8).reshape(rows, cols)
    report = {
        "cell_m": cell_m, "rows": rows, "columns": cols,
        "cells_observed": int(observed.sum()), "cells_total": int(rows * cols),
        "observed_fraction": round(float(observed.mean()), 4),
        "cells_without_dsm": int((~supported).sum()),
        "views_available": len(images), "views_used": len(counts),
        "median_source_gsd_m": (None if not observed.any() else
                                round(float(np.nanmedian(gsd)) * metres_per_local_unit, 4)),
        "not_done": ["no exposure balancing or seam blending between views",
                     "no moving-object test: a moved car can appear where it was",
                     "occlusion is tested against a coarse DSM depth buffer per view"]}
    return rgb.reshape(rows, cols, 3), alpha, transform, report


def _wall_samples(xx, yy, zz, cell_m, max_levels=64):
    """Vertical occluder points where the surface steps down by more than a cell.

    A height grid stores roof tops and ground but not the walls between them, so a
    depth buffer built from its cells alone lets an oblique view see through a
    building's side. Each cell that drops to a lower neighbour gets a column of points
    down to that neighbour's height, on the shared cell edge. They occlude; they are
    never coloured.
    """
    points, cells = [], []
    rows, cols = zz.shape
    flat_index = np.arange(rows * cols).reshape(rows, cols)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        high = zz[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
        low = zz[max(0, dr):rows - max(0, -dr) or None, max(0, dc):cols - max(0, -dc) or None]
        drop = high - low
        mask = drop > cell_m
        if not mask.any():
            continue
        src = (slice(max(0, -dr), rows - max(0, dr)), slice(max(0, -dc), cols - max(0, dc)))
        ex, ey = xx[src][mask] + dc * cell_m / 2, yy[src][mask] - dr * cell_m / 2
        top, bottom = high[mask], low[mask]
        ids = flat_index[src][mask]
        levels = np.clip(np.ceil((top - bottom) / cell_m).astype(np.int64), 1, max_levels)
        for k in np.unique(levels):
            sel = levels == k
            fractions = (np.arange(k) + 0.5) / k
            z = bottom[sel][:, None] + (top[sel] - bottom[sel])[:, None] * fractions[None, :]
            points.append(np.column_stack([np.repeat(ex[sel], k), np.repeat(ey[sel], k), z.ravel()]))
            cells.append(np.repeat(ids[sel], k))
    if not points:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    return np.concatenate(cells), np.vstack(points)


def _focal_per_cell(best, observed):
    """Ground size of one source pixel, in local units, from the winning score's f/depth."""
    out = np.full(len(best), np.nan)
    out[observed & (best > 0)] = 1.0 / best[observed & (best > 0)]
    return out
