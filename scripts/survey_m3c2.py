"""M3C2 point-cloud distance with level of detection (Lague, Brodu & Leroux 2013).

2.5D DSM differencing (``survey_change``) cannot see change on a wall, an overhang or a
bridge soffit. M3C2 measures along each core point's local surface normal instead:

1. normals at core points from the reference cloud (PCA of neighbours within D/2), oriented
   towards ``toward`` (a viewpoint, e.g. the mean camera) so signs are consistent;
2. for each core point, the points of each epoch inside a cylinder of diameter ``d`` along
   the normal (up to ``max_depth_m`` either side) give a mean position along the normal,
   a spread (std) and a count;
3. distance = mean2 - mean1 along the normal;
   LoD95 = +-1.96 sqrt(s1^2/n1 + s2^2/n2) + reg (registration error, added linearly as the
   original paper does);
4. |distance| > LoD95 is significant; a core point with fewer than ``min_points`` in
   either cylinder is ``unobserved`` - not "no change".

Normals from a noisy splat cloud are the weak link: a normal scale D of 5-10x the point
spacing is the paper's guidance and the default here is 1.0 m for building-scale scans.
"""
import numpy as np
from scipy.spatial import cKDTree


def normals(points, core, *, scale_m=1.0, toward=None, min_neighbours=8):
    tree = cKDTree(points)
    out = np.full((len(core), 3), np.nan)
    planarity = np.full(len(core), np.nan)
    for i, idx in enumerate(tree.query_ball_point(core, scale_m / 2.0)):
        if len(idx) < min_neighbours:
            continue
        nb = points[idx]
        centred = nb - nb.mean(0)
        evals, evecs = np.linalg.eigh(centred.T @ centred / len(idx))
        n = evecs[:, 0]
        if toward is not None and np.dot(np.asarray(toward, float) - core[i], n) < 0:
            n = -n
        out[i] = n
        planarity[i] = (evals[1] - evals[0]) / max(evals[2], 1e-12)
    return out, planarity


def m3c2(cloud1, cloud2, core=None, *, normal_scale_m=1.0, projection_diameter_m=0.5, max_depth_m=2.0,
         registration_error_m=0.0, min_points=5, toward=None, core_spacing_m=None):
    cloud1, cloud2 = np.asarray(cloud1, float), np.asarray(cloud2, float)
    if core is None:
        spacing = core_spacing_m or projection_diameter_m
        keys = np.floor(cloud1 / spacing).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        core = cloud1[np.sort(first)]
    core = np.asarray(core, float)
    n, _ = normals(cloud1, core, scale_m=normal_scale_m, toward=toward)
    trees = (cKDTree(cloud1), cKDTree(cloud2))
    radius = projection_diameter_m / 2.0
    reach = float(np.hypot(radius, max_depth_m))
    stats = []
    for tree, cloud in zip(trees, (cloud1, cloud2)):
        mean = np.full(len(core), np.nan)
        std = np.full(len(core), np.nan)
        count = np.zeros(len(core), int)
        for i, idx in enumerate(tree.query_ball_point(core, reach)):
            if not np.isfinite(n[i, 0]) or not idx:
                continue
            rel = cloud[idx] - core[i]
            along = rel @ n[i]
            radial = np.linalg.norm(rel - np.outer(along, n[i]), axis=1)
            sel = (radial <= radius) & (np.abs(along) <= max_depth_m)
            k = int(sel.sum())
            count[i] = k
            if k:
                mean[i] = float(along[sel].mean())
                std[i] = float(along[sel].std(ddof=1)) if k > 1 else 0.0
        stats.append((mean, std, count))
    (m1, s1, c1), (m2, s2, c2) = stats
    observed = (c1 >= min_points) & (c2 >= min_points) & np.isfinite(n[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        lod = 1.96 * np.sqrt(s1 ** 2 / np.maximum(c1, 1) + s2 ** 2 / np.maximum(c2, 1)) + registration_error_m
    dist = np.where(observed, m2 - m1, np.nan)
    lod = np.where(observed, lod, np.nan)
    significant = observed & (np.abs(dist) > lod)
    return {"core": core, "normal": n, "distance": dist, "lod95": lod, "significant": significant,
            "observed": observed, "n1": c1, "n2": c2,
            "summary": {"core_points": int(len(core)), "observed": int(observed.sum()),
                        "significant": int(significant.sum()),
                        "moved_towards_viewer": int((significant & (dist > 0)).sum()),
                        "moved_away": int((significant & (dist < 0)).sum()),
                        "median_lod95_m": None if not observed.any() else round(float(np.nanmedian(lod)), 4),
                        "max_abs_significant_m": None if not significant.any() else round(float(np.nanmax(np.abs(dist[significant]))), 4),
                        "parameters": {"normal_scale_m": normal_scale_m, "projection_diameter_m": projection_diameter_m,
                                       "max_depth_m": max_depth_m, "registration_error_m": registration_error_m,
                                       "min_points": min_points},
                        "method": "M3C2 (Lague et al. 2013): distance along local normals in cylinders, LoD95 per core point"}}
