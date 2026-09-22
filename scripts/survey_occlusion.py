"""What a single drone pass cannot see, kept separate from what it can.

Occlusion is why reconstructions have holes, and the honest response is to measure
the hole instead of papering over it. Geometry is sorted into layers that stay
separate in every output:

  measured     enough claimed views *and* enough of them survive the depth-map
               occlusion check
  weak         some claim of support, but too little un-occluded observation
  unobserved   nothing observed it at all
  constrained  a surface implied by a plane fitted to measured points (a roof that
               continues past the part that was imaged). A geometric constraint
               derived from evidence, not a learned guess, and never a measurement.
  inferred     generative completion. This module publishes no such entry point;
               :func:`inference_policy` says why and :func:`assert_no_inferred`
               rejects the label wherever a measured export is assembled.

Occlusion follows the convention established in ``survey_visibility``: COLMAP depth
maps store distance along the *unit ray*, not camera-space z, so a sample farther
along the ray than the recorded surface sits behind something and was not seen by
that view. Pixel projection, the ray-distance comparison and the missing-depth
(``<= 0``) rule are deliberately the same checks, so a cell this module calls hidden
is a cell ``visible_support`` refuses to count.

None of this measures accuracy against surveyed truth. It measures how much of the
model has any right to be believed.
"""
import math

import numpy as np

_INT64_LIMIT = 2 ** 62  # Keep |value| well inside int64 so casts cannot wrap.

# The layers, and the integer code each one carries through a blended array.
LAYERS = ("measured", "weak", "unobserved", "constrained", "inferred")
LAYER_CODE = {name: index for index, name in enumerate(LAYERS)}
CODE_LAYER = {index: name for name, index in LAYER_CODE.items()}
CLASSIFIED_LAYERS = ("measured", "weak", "unobserved")
MEASURED_LAYER = "measured"
CONSTRAINED_LAYER = "constrained"
INFERRED_LAYER = "inferred"
MEASURED_EXPORT_LAYERS = (MEASURED_LAYER,)
ALLOWED_EXPORT_LAYERS = (MEASURED_LAYER, CONSTRAINED_LAYER)
PATCH_KIND = "constrained_patch"

REASON_OBSERVED = "observed"
REASON_NEVER_OBSERVED = "never_observed"
REASON_OCCLUDED = "occluded"
REASON_UNUSABLE_BASELINE = "unusable_baseline"
HIDDEN_REASONS = (REASON_NEVER_OBSERVED, REASON_OCCLUDED, REASON_UNUSABLE_BASELINE)

_RANSAC_TRIALS = 512  # Fixed trial count: the fit must not depend on data order.
EXACT_BASELINE_VIEW_LIMIT = 64  # Above this the pairwise spread is bounded, not exact.


# --------------------------------------------------------------------------------------
# validation helpers - every failure is a ValueError, so nothing degrades silently
# --------------------------------------------------------------------------------------

def _points(value, name="points"):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx3 array")
    if not len(array):
        raise ValueError(f"{name} must hold at least one point")
    return array


def _counts(values, length, name):
    """Per-point counts as int64: no negatives, no fractions, no unwrappable magnitudes."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold numeric per-point counts") from exc
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must hold exactly one count per point")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite counts")
    if np.any(array < 0) or np.any(array != np.floor(array)) or np.any(array >= _INT64_LIMIT):
        raise ValueError(f"{name} must hold non-negative integer counts representable in int64")
    return array.astype(np.int64)


def _integer(value, name, *, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a whole number of at least {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number of at least {minimum}") from exc
    if (not math.isfinite(number) or number != math.floor(number)
            or number < minimum or number >= _INT64_LIMIT):
        raise ValueError(f"{name} must be a whole number of at least {minimum}")
    return int(number)


def _scale(value, name, *, allow_zero=False):
    adjective = "non-negative" if allow_zero else "positive"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {adjective} number") from exc
    if not math.isfinite(number) or number < 0 or (number == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be a finite {adjective} number")
    return number


def _vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XYZ vector")
    return array


def _codes(labels):
    """Layer names to int32 codes without a Python loop over every point."""
    unique, inverse = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    table = np.asarray([LAYER_CODE[name] for name in unique], dtype=np.int32)
    return table[inverse]


def _grid_cells(xyz, cell):
    """Voxel indices for positions, refusing to wrap them into int64."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.floor(np.asarray(xyz, dtype=np.float64) / cell)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= _INT64_LIMIT):
        raise ValueError("cell_size_m is too small for these positions: cell indices would "
                         "overflow int64 instead of counting cells")
    return scaled.astype(np.int64)


def _validated_view(view, index):
    """The same checks survey_visibility applies: finite K, a true rotation, finite depth."""
    if not isinstance(view, dict):
        raise ValueError(f"view {index} must be a mapping with K, viewmat and depth")
    for key in ("K", "viewmat", "depth"):
        if key not in view:
            raise ValueError(f"view {index} is missing {key!r}")
    K = np.asarray(view["K"], dtype=np.float64)
    viewmat = np.asarray(view["viewmat"], dtype=np.float64)
    depth = np.asarray(view["depth"], dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"view {index} K must be a finite 3x3 matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"view {index} K must carry positive focal lengths")
    if viewmat.shape != (4, 4) or not np.isfinite(viewmat).all():
        raise ValueError(f"view {index} viewmat must be a finite 4x4 matrix")
    rotation = viewmat[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or abs(np.linalg.det(rotation) - 1) > 1e-6):
        raise ValueError(f"view {index} viewmat rotation must be orthonormal with determinant +1")
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError(f"view {index} depth must be a finite 2-D array")
    return K, viewmat, depth


def _pixel_indices(numerator, offset, name, limit=_INT64_LIMIT):
    """Floor projected coordinates to integer pixels, refusing to wrap them."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        floored = np.floor(numerator + offset)
    if not np.isfinite(floored).all() or np.any(np.abs(floored) >= limit):
        raise ValueError(f"projected pixel {name} is not finite or not representable "
                         "as an integer index")
    return floored.astype(np.int64)


def _layer_names(value, length, name):
    """Per-point layer names, from a scalar name, an array, or a classify result."""
    if isinstance(value, dict):
        if "labels" not in value:
            raise ValueError(f"{name} must be a classify result carrying 'labels'")
        value = value["labels"]
    if isinstance(value, str):
        value = np.full(length, value)
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(length, str(array.item()))
    array = np.atleast_1d(array).astype(str).ravel()
    if array.shape[0] != length:
        raise ValueError(f"{name} must hold one layer name per point")
    unknown = sorted(set(array.tolist()) - set(LAYERS))
    if unknown:
        raise ValueError(f"{name} holds unknown layers {unknown}; the layers are {list(LAYERS)}")
    return array


# --------------------------------------------------------------------------------------
# 1. per-point classification
# --------------------------------------------------------------------------------------

def classify(points, *, support, visible_support, min_views=3, min_visible=2) -> dict:
    """Label every point 'measured', 'weak' or 'unobserved', and count the three layers.

    ``support`` is the claimed view count (COLMAP track length); ``visible_support``
    is the occlusion-checked count from ``survey_visibility.visible_support``. A point
    is 'measured' only when both thresholds are met. Claimed multi-view support with
    too few views surviving the depth check is 'weak', never measured, because the
    depth map says that surface was behind something. 'unobserved' means no view
    claimed it and no view saw it.

    Returns the per-point labels, their integer layer codes, the counts and each
    layer's fraction of the cloud. Raises ValueError when visibility exceeds the claim
    it is supposed to be a subset of, when a threshold is not a positive whole number,
    when min_visible exceeds min_views, or when the arrays are not finite per-point
    integer counts.
    """
    points = _points(points)
    support = _counts(support, len(points), "support")
    visible = _counts(visible_support, len(points), "visible_support")
    min_views = _integer(min_views, "min_views", minimum=1)
    min_visible = _integer(min_visible, "min_visible", minimum=1)
    if np.any(visible > support):
        raise ValueError("visible_support cannot exceed claimed support for any point: an "
                         "occlusion-checked view is one of the views that claimed it")
    if min_visible > min_views:
        raise ValueError("min_visible cannot exceed min_views: a measured point would need more "
                         "occlusion-checked views than the support being claimed for it")
    measured = (support >= min_views) & (visible >= min_visible)
    claimed = (visible >= 1) | (support >= min_views)
    labels = np.where(measured, MEASURED_LAYER, np.where(claimed, "weak", "unobserved")).astype(str)
    counts = {layer: int((labels == layer).sum()) for layer in CLASSIFIED_LAYERS}
    return {
        "point_count": int(len(points)), "min_views": min_views, "min_visible": min_visible,
        "labels": labels, "layer": _codes(labels), "counts": counts,
        "fractions": {layer: round(counts[layer] / len(points), 6) for layer in CLASSIFIED_LAYERS},
        "claimed_views": int(support.sum()), "visible_views": int(visible.sum()),
        "retained_view_fraction": round(float(visible.sum() / max(int(support.sum()), 1)), 6),
        "support_basis": "claimed track length",
        "visibility_basis": "depth-map distance along the unit ray (survey_visibility)",
        "note": ("'measured' needs enough views AND enough of them un-occluded; 'weak' is a "
                 "support claim with too little observable evidence; 'unobserved' is nothing. No "
                 "layer here measures accuracy against surveyed truth"),
    }


# --------------------------------------------------------------------------------------
# 2. hidden regions: the hole a single pass fundamentally cannot see
# --------------------------------------------------------------------------------------

def _cells(points, cell):
    """Occupied voxel indices plus each cell's sample centroid and sample count."""
    indices = _grid_cells(points, cell)
    order = np.lexsort(indices.T[::-1])
    sorted_indices = indices[order]
    boundaries = np.flatnonzero(np.any(np.diff(sorted_indices, axis=0), axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    sizes = np.add.reduceat(np.ones(len(order), dtype=np.int64), starts)
    totals = np.add.reduceat(points[order], starts, axis=0)
    return sorted_indices[starts], totals / sizes[:, None], sizes


def _parallax_deg(centroids, centres, seen):
    """Largest angle at each cell between the camera centres that actually observed it.

    Exact maximum pairwise angle up to ``EXACT_BASELINE_VIEW_LIMIT`` views; above that,
    a two-sweep bound which under-reads the spread and can therefore only over-report a
    cell as unobservable - the conservative direction when the question is how big the
    hole is.
    """
    count = len(centres)
    rays = centroids[:, None, :] - centres[None, :, :]
    length = np.linalg.norm(rays, axis=2)
    valid = seen & (length > 0)
    unit = np.where(valid[..., None], rays / np.where(valid, length, 1.0)[..., None], 0.0)
    angles = np.zeros(len(centroids), dtype=np.float64)
    rows = np.flatnonzero(valid.sum(axis=1) >= 2)
    if not len(rows):
        return angles, "no cell was observed by two or more views"
    if count <= EXACT_BASELINE_VIEW_LIMIT:
        chunk = max(1, min(4096, int(4e6 // max(count * count, 1))))
        diagonal = np.arange(count)
        for offset in range(0, len(rows), chunk):
            block = rows[offset:offset + chunk]
            gram = np.einsum("kim,kjm->kij", unit[block], unit[block])
            pair_ok = valid[block][:, :, None] & valid[block][:, None, :]
            cosine = np.where(pair_ok, gram, 1.0)
            cosine[:, diagonal, diagonal] = 1.0  # a view never degrades its own baseline
            closest = np.clip(cosine.reshape(len(block), -1).min(axis=1), -1.0, 1.0)
            angles[block] = np.degrees(np.arccos(closest))
        basis = f"exact maximum pairwise angle over {count} observing views"
    else:
        for row in rows:
            directions = unit[row][valid[row]]
            mean = directions.mean(axis=0)
            mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
            first = directions[int(np.argmin(directions @ mean))]
            second = directions[int(np.argmin(directions @ first))]
            angles[row] = math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0))))
        basis = (f"two-sweep lower bound on the angular spread over {count} views "
                 f"(exact only up to {EXACT_BASELINE_VIEW_LIMIT})")
    return angles, basis


def hidden_regions(camera_centers, points, depth_views, *, cell_size_m, angular_threshold_deg,
                   relative_tolerance=0.02) -> dict:
    """Which scene cells a single pass cannot see, each with the reason why.

    ``points`` are candidate surface samples - the reconstruction plus any ground or
    facade hypotheses worth testing - voxelised at ``cell_size_m``. Each cell centroid
    is projected into every depth view and compared with the recorded surface along the
    unit ray, exactly as ``survey_visibility.visible_support`` does. A cell is:

      never_observed      no view framed it, or every view that framed it recorded no
                          surface at all there (the same absence of evidence)
      occluded            views framed it, but each one recorded a nearer surface: the
                          cell sits behind something from everywhere it appears
      unusable_baseline   a view does see the surface, but the observing cameras subtend
                          less than ``angular_threshold_deg`` at it, so depth there is
                          not triangulated usefully however many views framed it
      observed            seen with a usable baseline, so not hidden

    Returns every cell with its reason, the hidden subset, per-reason counts, the hidden
    fraction and the excluded footprint area. Raises ValueError on a non-positive cell
    size, a negative threshold, a malformed or missing view, or camera centres that are
    not one-per-view and do not match the centres implied by the view matrices - mixing
    pose sources is how a visibility claim silently becomes meaningless.
    """
    points = _points(points)
    centres = np.asarray(camera_centers, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3 or not np.isfinite(centres).all():
        raise ValueError("camera_centers must be a finite Nx3 array")
    if not len(centres):
        raise ValueError("camera_centers must hold at least one camera position")
    if depth_views is None or isinstance(depth_views, dict):
        raise ValueError("depth_views must be a non-empty sequence of depth-map views")
    views = list(depth_views)
    if not len(views):
        raise ValueError("depth_views must be a non-empty sequence of depth-map views")
    if len(centres) != len(views):
        raise ValueError(f"camera_centers must supply one centre per depth view: got "
                         f"{len(centres)} centres for {len(views)} views")
    cell = _scale(cell_size_m, "cell_size_m")
    threshold = _scale(angular_threshold_deg, "angular_threshold_deg", allow_zero=True)
    tolerance = _scale(relative_tolerance, "relative_tolerance")

    prepared = []
    for index, view in enumerate(views):
        K, viewmat, depth = _validated_view(view, index)
        rotation = viewmat[:3, :3]
        own_centre = -rotation.T @ viewmat[:3, 3]
        if not np.allclose(own_centre, centres[index], rtol=1e-6, atol=1e-6):
            raise ValueError(f"camera_centers[{index}] does not match the centre implied by view "
                             f"{index}'s viewmat: this pass mixes pose sources")
        prepared.append((K, viewmat, depth))

    cells, centroids, sizes = _cells(points, cell)
    framing = np.zeros(len(centroids), dtype=np.int64)
    agrees = np.zeros(len(centroids), dtype=np.int64)
    behind = np.zeros(len(centroids), dtype=np.int64)
    seen = np.zeros((len(centroids), len(prepared)), dtype=bool)
    homogeneous = np.column_stack([centroids, np.ones(len(centroids))])
    for index, (K, viewmat, depth) in enumerate(prepared):
        camera = homogeneous @ viewmat.T
        x, y, z = camera[:, 0], camera[:, 1], camera[:, 2]
        rows, cols = depth.shape
        front = np.flatnonzero(z > 0)
        if not len(front):
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled_x = K[0, 0] * x[front] / z[front]
            scaled_y = K[1, 1] * y[front] / z[front]
        u = _pixel_indices(scaled_x, K[0, 2], "column")
        v = _pixel_indices(scaled_y, K[1, 2], "row")
        inside = (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
        if not inside.any():
            continue
        framed = front[inside]
        framing[framed] += 1
        recorded = depth[v[inside], u[inside]]
        distance = np.linalg.norm(camera[framed, :3], axis=1)
        measured = recorded > 0  # COLMAP's no-measurement value: evidence of nothing
        # Ray distance, not camera-space z: COLMAP stores depth along the unit ray.
        agrees_surface = measured & (distance <= recorded * (1.0 + tolerance) + tolerance)
        agrees[framed[agrees_surface]] += 1
        seen[framed[agrees_surface], index] = True
        behind[framed[measured & ~agrees_surface]] += 1

    parallax, parallax_basis = _parallax_deg(centroids, centres, seen)
    no_evidence = (framing > 0) & (agrees == 0) & (behind == 0)
    reasons = np.full(len(centroids), REASON_OBSERVED, dtype=object)
    reasons[(agrees == 0) & (behind > 0)] = REASON_OCCLUDED
    reasons[(framing == 0) | no_evidence] = REASON_NEVER_OBSERVED
    reasons[(agrees > 0) & (parallax < threshold)] = REASON_UNUSABLE_BASELINE

    counts = {reason: int((reasons == reason).sum())
              for reason in (REASON_OBSERVED,) + HIDDEN_REASONS}
    entries = [{
        "cell": [int(value) for value in row],
        "centre": [round(float(value), 6) for value in centroid],
        "reason": str(reason),
        "views_framing": int(framing[index]),
        "views_observed": int(agrees[index]),
        "views_behind": int(behind[index]),
        "parallax_deg": round(float(parallax[index]), 6),
        "points_in_cell": int(sizes[index]),
    } for index, (row, centroid, reason) in enumerate(zip(cells, centroids, reasons))]
    hidden = [entry for entry in entries if entry["reason"] in HIDDEN_REASONS]
    return {
        "cell_size_m": cell, "angular_threshold_deg": threshold,
        "relative_tolerance": tolerance, "camera_count": int(len(centres)),
        "view_count": int(len(prepared)), "total_cells": int(len(entries)),
        "counts": counts, "hidden": hidden, "cells": entries,
        "hidden_cells": int(len(hidden)),
        "hidden_fraction": round(len(hidden) / len(entries), 6),
        "excluded_area_m2": round(len(hidden) * cell * cell, 6),
        "parallax_basis": parallax_basis,
        "visibility_basis": "depth-map distance along the unit ray (survey_visibility convention)",
        "reasons": {
            REASON_NEVER_OBSERVED: "no view framed the cell, or no view recorded a surface there",
            REASON_OCCLUDED: "every view that framed it recorded a nearer surface",
            REASON_UNUSABLE_BASELINE: "seen, but the observing cameras subtend too small an angle",
            REASON_OBSERVED: "seen with a usable baseline"},
        "note": ("this is the honest answer to reconstructing occluded surfaces: it lists what a "
                 "single pass cannot see instead of filling it in. 'occluded' cells are behind "
                 "something from every viewpoint, and a pass that flies one side of a wall can "
                 "never see what lies behind it at any altitude it actually flew"),
        "limitation": ("only cells sampled by the supplied points can be assessed: a hole with no "
                       "candidate sample in it is invisible to this estimate, so feed it ground and "
                       "facade hypotheses as well as reconstructed points"),
    }


# --------------------------------------------------------------------------------------
# 3. planes fitted to measured points, and the constrained extension of a partial one
# --------------------------------------------------------------------------------------

def _candidate_plane(points, sample):
    """Plane through three samples, or None when the three are collinear."""
    a, b, c = points[sample[0]], points[sample[1]], points[sample[2]]
    normal = np.cross(b - a, c - a)
    length = float(np.linalg.norm(normal))
    scale = max(1.0, float(np.linalg.norm(b - a)) + float(np.linalg.norm(c - a)))
    if not np.isfinite(length) or length <= 1e-9 * scale:
        return None
    normal = normal / length
    return normal, -float(normal @ a)


def _refit(points, inliers):
    """Total-least-squares plane on the inliers, with a canonical orientation."""
    subset = points[inliers]
    centroid = subset.mean(axis=0)
    _, _, vh = np.linalg.svd(subset - centroid, full_matrices=False)
    normal = np.asarray(vh[2], dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("plane fit produced a degenerate normal")
    normal = normal / length
    if normal[int(np.argmax(np.abs(normal)))] < 0:  # order-independent orientation
        normal = -normal
    offset = -float(normal @ centroid)
    residual = np.abs(points[inliers] @ normal + offset)
    basis = np.vstack([vh[0], vh[1]])
    along = (points[inliers] - centroid) @ basis.T
    extent = ((float(along[:, 0].min()), float(along[:, 0].max())),
              (float(along[:, 1].min()), float(along[:, 1].max())))
    return {"normal": normal, "offset": offset, "centroid": centroid, "basis": basis,
            "residual_m": float(residual.mean()),
            "rms_residual_m": float(np.sqrt((residual ** 2).mean())),
            "max_residual_m": float(residual.max()), "extent": extent}


def plane_complete(points, labels, *, ransac_distance=0.05, min_inliers=200, max_planes=8,
                   seed=0) -> list:
    """Fit the dominant planes (ground, roofs, facades) to MEASURED points only.

    Anything not labelled 'measured' is excluded before a single plane is estimated:
    weak and unobserved points are exactly the ones an occluded reconstruction gets
    wrong, and a plane fitted through them inherits that. Planes come out greedily,
    best-supported first, each carrying its inlier indices, inlier count and residual
    in metres so a reader can weigh how strong the constraint is.

    The result is a list of planes, not a completed surface. Extending one beyond the
    measured points is :func:`constrained_extension`'s job, and that output is labelled
    'constrained' for the rest of its life.

    Raises ValueError when nothing is measured, when there are fewer measured points
    than ``min_inliers``, when ``min_inliers`` is below three (a plane needs three),
    when the labels are not layer names, or when a threshold is not a finite positive
    number.
    """
    points = _points(points)
    labels = _layer_names(labels, len(points), "labels")
    distance = _scale(ransac_distance, "ransac_distance")
    min_inliers = _integer(min_inliers, "min_inliers", minimum=3)
    max_planes = _integer(max_planes, "max_planes", minimum=1)
    seed = _integer(seed, "seed", minimum=0)
    measured = np.flatnonzero(labels == MEASURED_LAYER)
    if not len(measured):
        raise ValueError("plane_complete has no measured points to fit: refusing to fit planes to "
                         "weak, unobserved or inferred geometry, because that is how invented "
                         "surfaces get dressed up as measured ones")
    if len(measured) < min_inliers:
        raise ValueError(f"plane_complete needs at least min_inliers ({min_inliers}) measured "
                         f"points; only {len(measured)} are measured")
    rng = np.random.default_rng(seed)
    pool = measured
    planes = []
    while len(planes) < max_planes and len(pool) >= min_inliers:
        best = None
        for _ in range(_RANSAC_TRIALS):
            candidate = _candidate_plane(points, rng.choice(pool, size=3, replace=False))
            if candidate is None:
                continue
            normal, offset = candidate
            inliers = pool[np.abs(points[pool] @ normal + offset) <= distance]
            if best is None or len(inliers) > len(best):
                best = inliers
            if len(inliers) >= len(pool):  # cannot be beaten: stop sampling
                break
        if best is None or len(best) < min_inliers:
            break
        fitted = _refit(points, best)
        refined = pool[np.abs(points[pool] @ fitted["normal"] + fitted["offset"]) <= distance]
        if len(refined) >= min_inliers:  # one re-collect at the refined plane
            best, fitted = refined, _refit(points, refined)
        keep = np.ones(len(points), dtype=bool)
        keep[best] = False
        pool = pool[keep[pool]]
        planes.append({
            "index": len(planes), "normal": fitted["normal"], "offset": fitted["offset"],
            "plane_point": fitted["centroid"], "basis": fitted["basis"],
            "inlier_indices": np.sort(best).astype(np.int64), "inlier_count": int(len(best)),
            "inlier_fraction": round(len(best) / len(measured), 6),
            "residual_m": round(fitted["residual_m"], 9),
            "rms_residual_m": round(fitted["rms_residual_m"], 9),
            "max_residual_m": round(fitted["max_residual_m"], 9),
            "extent": ((round(fitted["extent"][0][0], 9), round(fitted["extent"][0][1], 9)),
                       (round(fitted["extent"][1][0], 9), round(fitted["extent"][1][1], 9))),
            "ransac_distance": distance, "min_inliers": min_inliers, "seed": seed,
            "layer": MEASURED_LAYER, "fit": "total least squares on RANSAC inliers",
            "note": ("plane fitted to measured points only; extending it beyond them is a "
                     "constraint, not a measurement"),
        })
    return planes


def _lattice(low, high, cell):
    axes = [np.arange(start, stop + cell * 0.5, cell) for start, stop in zip(low, high)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def _polygon_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _points_in_polygon(points, polygon):
    """Even-odd ray test; points and polygon share the plane's 2-D frame."""
    inside = np.zeros(len(points), dtype=bool)
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if y1 == y2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x1 + (points[:, 1] - y1) * (x2 - x1) / (y2 - y1)
        crossing = ((y1 > points[:, 1]) != (y2 > points[:, 1])) & (points[:, 0] < at)
        inside = inside ^ crossing
    return inside


def _region_samples(region, cell, normal, basis, origin):
    """In-plane (u, v) grid samples covering the region, plus a description of it.

    ``origin`` is a point the plane passes through, so the plane's own frame is
    (origin, basis) and every sample produced here is on the plane by construction.
    """
    if not isinstance(region, dict):
        raise ValueError("region must be a mapping with 'min'/'max' world bounds or a 'polygon'")
    if "polygon" in region:
        vertices = np.asarray(region["polygon"], dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("region polygon must be a finite Nx3 array of world points")
        if len(vertices) < 3:
            raise ValueError("region polygon needs at least three vertices")
        projected = (vertices - origin) @ basis.T
        if _polygon_area(projected) <= cell * cell * 1e-6:
            raise ValueError("region polygon is degenerate: it encloses no area on the plane")
        grid = _lattice(projected.min(axis=0), projected.max(axis=0), cell)
        kept = _points_in_polygon(grid, projected)
        return grid[kept], {"kind": "polygon",
                            "vertices": [[round(float(v), 6) for v in row] for row in vertices]}
    if "min" not in region or "max" not in region:
        raise ValueError("region must supply 'min' and 'max' world bounds, or a 'polygon'")
    low = _vector(region["min"], "region min")
    high = _vector(region["max"], "region max")
    if np.any(high <= low):
        raise ValueError("region max must exceed region min on every axis")
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                        for z in (low[2], high[2])], dtype=np.float64)
    side = (corners - origin) @ normal
    slack = 1e-9 * max(1.0, float(np.max(high - low)))
    if np.all(side > slack) or np.all(side < -slack):
        raise ValueError("region does not reach the plane: refusing to extend a plane into a "
                         "volume it never intersects")
    in_plane = (corners - origin) @ basis.T
    grid = _lattice(in_plane.min(axis=0), in_plane.max(axis=0), cell)
    world = origin + grid[:, 0][:, None] * basis[0] + grid[:, 1][:, None] * basis[1]
    inside = (np.all(world >= low - slack, axis=1) & np.all(world <= high + slack, axis=1)
              & (np.abs((world - origin) @ normal) <= slack))
    return grid[inside], {"kind": "bounds", "min": [float(v) for v in low],
                          "max": [float(v) for v in high]}


def constrained_extension(plane, region, *, cell_size_m=0.5) -> dict:
    """A labelled 'constrained' patch where a known plane is only partly observed.

    This is the defensible middle ground between "hole" and "hallucination". The
    surface is not invented: it is the plane the measured points already establish (a
    roof, a facade, a ground plane) evaluated over a region the pass did not image. The
    patch carries the plane's inlier count and residual so a reader can weigh the
    evidence behind it, how far outside the measured footprint it reaches, and a
    'constrained' label with no route back to 'measured' - :func:`blend` refuses the
    relabel and :func:`assert_no_inferred` reports the count.

    Raises ValueError when the plane is missing its inlier evidence, when its normal is
    not unit (then the residual is not in metres), when fewer than three inliers support
    it, when the region never meets the plane or leaves no room for a sample, or when
    the sampling cell size is not positive.
    """
    if not isinstance(plane, dict):
        raise ValueError("plane must be a plane_complete result, not a hand-made plane")
    for key in ("normal", "offset", "inlier_count", "residual_m", "plane_point", "basis", "extent"):
        if key not in plane:
            raise ValueError(f"plane is missing {key!r}: a constrained patch must carry the "
                             "evidence it was derived from")
    normal = _vector(plane["normal"], "plane normal")
    length = float(np.linalg.norm(normal))
    if abs(length - 1.0) > 1e-6:
        raise ValueError("plane normal must be unit length, otherwise the recorded residual is not "
                         "in metres and the patch cannot state its own uncertainty")
    normal = normal / length
    offset = float(plane["offset"])
    if not math.isfinite(offset):
        raise ValueError("plane offset must be finite")
    inlier_count = _integer(plane["inlier_count"], "plane inlier_count", minimum=3)
    residual = _scale(plane["residual_m"], "plane residual_m", allow_zero=True)
    cell = _scale(cell_size_m, "cell_size_m")
    basis = np.asarray(plane["basis"], dtype=np.float64)
    if basis.shape != (2, 3) or not np.isfinite(basis).all():
        raise ValueError("plane basis must be a 2x3 array of in-plane unit vectors")
    origin = _vector(plane["plane_point"], "plane point")
    samples, described = _region_samples(region, cell, normal, basis, origin)
    if not len(samples):
        raise ValueError("the region leaves no room for a surface sample at this cell_size_m: "
                         "refusing to publish an empty patch as geometry")
    points = origin + samples[:, 0][:, None] * basis[0] + samples[:, 1][:, None] * basis[1]
    on_plane = np.abs(points @ normal + offset)
    if float(on_plane.max()) > max(1e-9, residual):
        raise ValueError("the generated patch does not lie on its own plane")
    extent = np.asarray(plane["extent"], dtype=np.float64)
    if extent.shape != (2, 2):
        raise ValueError("plane extent must hold the inlier footprint in plane coordinates")
    outside = np.zeros(len(samples), dtype=np.float64)
    for axis in range(2):
        outside = np.maximum(outside, extent[axis, 0] - samples[:, axis])
        outside = np.maximum(outside, samples[:, axis] - extent[axis, 1])
    return {
        "layer": CONSTRAINED_LAYER, "kind": PATCH_KIND, "points": points,
        "point_count": int(len(points)), "plane_normal": normal, "plane_offset": offset,
        "plane_inlier_count": inlier_count, "plane_residual_m": residual,
        "basis": "measured plane fit", "region": described, "cell_size_m": cell,
        "area_m2": round(len(points) * cell * cell, 6),
        "outside_inlier_extent_m": round(float(max(outside.max(), 0.0)), 6),
        "max_plane_residual_m": round(float(on_plane.max()), 9),
        "note": ("constrained, not measured: this patch is the known plane evaluated where the pass "
                 f"did not see surface, so it is a geometric constraint derived from "
                 f"{inlier_count} measured inliers with {residual:.4f} m residual - it is not a "
                 "measurement and no code path here may relabel it as one"),
    }


# --------------------------------------------------------------------------------------
# 4. layer assembly, and the guard that keeps inference out
# --------------------------------------------------------------------------------------

def _looks_like_patch(entry):
    return (entry.get("kind") == PATCH_KIND or "plane_inlier_count" in entry
            or "plane_residual_m" in entry or "plane_normal" in entry)


def blend(layers) -> dict:
    """Assemble measured and constrained geometry, keeping a per-point integer layer code.

    Each entry is a mapping with ``points`` and exactly one of ``layer`` (one name for
    the whole entry) or ``labels``/``layers`` (one name per point, as produced by
    :func:`classify`). The merged array can always be split again: ``layer`` holds the
    integer code from :data:`LAYER_CODE` and ``provenance`` records which rows came from
    where. A constrained patch that arrives labelled 'measured' is a bug or a lie, so it
    raises instead of blending.
    """
    if not isinstance(layers, (list, tuple)) or not len(layers):
        raise ValueError("layers must be a non-empty sequence of layer entries")
    arrays, names, provenance = [], [], []
    cursor = 0
    for index, entry in enumerate(layers):
        if not isinstance(entry, dict):
            raise ValueError(f"layer entry {index} must be a mapping with 'points' and a layer")
        if "points" not in entry:
            raise ValueError(f"layer entry {index} has no points")
        points = _points(entry["points"], f"layer entry {index} points")
        declared = [key for key in ("layer", "labels", "layers") if key in entry]
        if len(declared) > 1:
            raise ValueError(f"layer entry {index} declares its layer twice ({declared}): "
                             "refusing to guess which one the caller meant")
        if not declared:
            raise ValueError(f"layer entry {index} declares no layer: geometry without a provenance "
                             "layer cannot be published as measured")
        key = declared[0]
        if key == "layer":
            if not isinstance(entry[key], str):
                raise ValueError("'layer' must be a layer name from LAYERS, not a code or an array; "
                                 "use 'labels' for per-point layers")
            row = _layer_names(entry[key], len(points), f"layer entry {index}")
        else:
            row = _layer_names(entry[key], len(points), f"layer entry {index} labels")
        if _looks_like_patch(entry) and not np.all(row == CONSTRAINED_LAYER):
            raise ValueError(f"layer entry {index} is a constrained patch (it carries the plane's "
                             "inlier count and residual) and cannot be relabelled as measured: a "
                             "constraint from a plane fit is not an observation")
        arrays.append(points)
        names.append(row)
        provenance.append({"entry": index, "layer": str(row[0]), "start": cursor,
                           "count": int(len(points)), "distinct_layers": sorted(set(row.tolist()))})
        cursor += len(points)
    labels = np.concatenate(names)
    counts = {layer: int((labels == layer).sum()) for layer in LAYERS if (labels == layer).any()}
    return {
        "points": np.vstack(arrays), "layer": _codes(labels), "layer_names": labels,
        "counts": counts, "total_points": int(len(labels)), "provenance": provenance,
        "layers_kept_separate": True,
        "note": ("measured and constrained geometry share one array but never one label: split on "
                 "'layer' before publishing anything as surveyed"),
    }


def assert_no_inferred(result) -> dict:
    """Guarantee that no 'inferred' point leaked into an export; raise if one did.

    Accepts a :func:`blend` result, any mapping carrying a layer column, or a bare
    sequence of layer names or codes. Callers run this on the way out: it is the last
    line between a generative completion and a product labelled measured. Constrained
    points are counted and reported rather than rejected, because a plane constraint
    from measured evidence is allowed to ship beside measurements as long as it keeps
    its own label.
    """
    if isinstance(result, dict):
        for key in ("layer", "layers", "labels", "layer_names"):
            if key in result:
                values = result[key]
                break
        else:
            raise ValueError("assert_no_inferred needs a blend result or a mapping carrying a layer "
                             "column: refusing to certify something it cannot read")
    else:
        values = result
    array = np.asarray(values)
    if array.ndim != 1 or not len(array):
        raise ValueError("layer column must be a non-empty 1-D array")
    if array.dtype.kind in "iuf":
        codes = array.astype(np.int64)
        unknown = sorted(set(codes.tolist()) - set(CODE_LAYER))
        if unknown:
            raise ValueError(f"layer codes {unknown} are not in CODE_LAYER")
        names = np.asarray([CODE_LAYER[int(code)] for code in codes])
    else:
        names = array.astype(str)
        unknown = sorted(set(names.tolist()) - set(LAYERS))
        if unknown:
            raise ValueError(f"unknown layers {unknown}: the layers are {list(LAYERS)}")
    counts = {layer: int((names == layer).sum()) for layer in LAYERS}
    if counts[INFERRED_LAYER]:
        raise ValueError(f"{counts[INFERRED_LAYER]} inferred point(s) are present: inferred geometry "
                         "must never appear in a measured export, because presenting invented "
                         "surfaces as measured ones is the failure this project exists to avoid")
    return {
        "points": int(len(names)), "clean": True, **counts,
        "measured_export_layers": list(MEASURED_EXPORT_LAYERS),
        "note": (f"no inferred points; {counts[CONSTRAINED_LAYER]} constrained point(s) are present "
                 f"and stay labelled separately from the {counts[MEASURED_LAYER]} measured point(s)"),
    }


# --------------------------------------------------------------------------------------
# 5. the visibility-aware completeness denominator
# --------------------------------------------------------------------------------------

def coverage_denominator(hidden, total_cells, *, cell_size_m=None) -> dict:
    """Completeness denominator over what the pass could see - and what that excluded.

    A completeness score is only meaningful against surface a single pass was able to
    observe: scoring occluded and never-framed cells as missing punishes the method for
    physics, and scoring them as present is fabrication. So the denominator drops the
    hidden cells, and the excluded cell count and footprint area are published beside it.

    BOTH NUMBERS MUST ALWAYS BE PUBLISHED TOGETHER. A visibility-filtered completeness
    score on its own is how filtering hides missing coverage: "98% complete" of what the
    pass could see sounds like "98% of the site". That is why this function refuses to
    run without a cell size, and why ``excluded_area_m2`` is a required field of the
    result rather than an optional extra.

    ``hidden`` may be a :func:`hidden_regions` result (then the cell size comes from it)
    or an explicit count. Raises ValueError when more cells are hidden than exist, when
    either count is not a non-negative whole number, or when no cell size is available to
    state the excluded area.
    """
    if isinstance(hidden, dict):
        carried = hidden.get("cell_size_m")
        count = hidden.get("hidden_cells")
        if count is None:
            if "hidden" not in hidden:
                raise ValueError("hidden result carries neither 'hidden_cells' nor 'hidden'")
            count = len(hidden["hidden"])
        if cell_size_m is None:
            if carried is None:
                raise ValueError("cell_size_m is required to report the excluded area: a "
                                 "visibility-aware denominator without its excluded area can hide "
                                 "missing coverage")
            cell_size_m = carried
    elif isinstance(hidden, (list, tuple, np.ndarray)):
        raise ValueError("pass the hidden_regions result or its hidden_cells count, not a bare "
                         "list: the per-reason breakdown is part of what gets published")
    else:
        count = hidden
    total = _integer(total_cells, "total_cells", minimum=0)
    excluded = _integer(count, "hidden cells", minimum=0)
    if excluded > total:
        raise ValueError(f"hidden cells ({excluded}) cannot exceed total cells ({total}): that "
                         "denominator would be built from cells that were never counted")
    cell = _scale(cell_size_m, "cell_size_m")
    area = excluded * cell * cell
    if not math.isfinite(area):
        raise ValueError("excluded area overflows: cell_size_m or the cell count is too large")
    visible = total - excluded
    fraction = excluded / total if total else 0.0
    return {
        "total_cells": int(total), "hidden_cells": int(excluded), "excluded_cells": int(excluded),
        "visible_cells": int(visible), "denominator": int(max(visible, 1)),
        "denominator_was_floored": bool(visible < 1),
        "excluded_fraction": round(fraction, 6), "kept_fraction": round(1.0 - fraction, 6),
        "cell_size_m": cell, "excluded_area_m2": round(float(area), 6),
        "publish_together": True,
        "basis": "cells the pass could observe, with the hidden cells reported beside them",
        "note": (f"completeness divided by this denominator is visibility-aware, not whole-site: "
                 f"{excluded} of {total} sampled cells were excluded because the pass could not see "
                 f"them ({area:.2f} m^2 of footprint), and both numbers must always be published "
                 "together"),
        "warning": ("a visibility-filtered score with its excluded area omitted hides missing "
                    "coverage behind physics"),
    }


# --------------------------------------------------------------------------------------
# 6. the inference policy: there is no inpainting entry point, on purpose
# --------------------------------------------------------------------------------------

def inference_policy() -> dict:
    """Why generative completion is excluded from measured products.

    ``inpaint_holes`` deliberately does not exist in this module. There is no flag to
    set, no model to download and no gated code path that turns a hole into geometry: a
    completion would be a guess about surfaces no camera recorded, and the project's
    non-negotiable rule is that geometry we inferred must never be presented as geometry
    we measured. The defensible middle ground is :func:`constrained_extension`, which
    extends a plane the measured points already establish and labels it 'constrained'
    permanently.
    """
    return {
        "generative_completion_allowed_in_measured_products": False,
        "has_inpainting_entry_point": False,
        "inpainting_entry_point": None,
        "permitted_layers": ALLOWED_EXPORT_LAYERS,
        "excluded_layers": (INFERRED_LAYER,),
        "measured_export_layers": MEASURED_EXPORT_LAYERS,
        "gating": ("there is nothing to gate: no function in this module produces inferred "
                   "geometry, and blend/assert_no_inferred exist to reject it if it arrives from "
                   "somewhere else"),
        "reasons": (
            "a single pass records surfaces from one side, so a learned completion of the occluded "
            "rest encodes priors about buildings in general and never this site's surface",
            "inferred surfaces can never be validated against imagery that could not see them, so "
            "their error is unknowable and must not enter a metre-level accuracy claim",
            "generated geometry inside a measured model destroys the provenance layer that lets a "
            "reviewer decide what to trust",
            "plane constraints fitted to measured points are kept because they carry their inlier "
            "count and residual; learned shape priors carry a dataset's average instead",
        ),
        "single_pass_limits": (
            "surfaces behind an occluder from every viewpoint - the far side of a wall, a courtyard "
            "the pass flew past - can never be recovered from that pass",
            "the side of the scene behind the start and the end of the trajectory is in no frame",
            "a nadir-dominated pass never sees facade elevations, and no oblique pass sees directly "
            "under an eave, a parapet or a canopy",
            "a surface observed only at a near-zero triangulation angle has no usable depth "
            "evidence however many views framed it",
            "recovering any of the above needs another pass, another altitude, or ground-level "
            "capture - not a completion model",
        ),
        "policy": ("measured and constrained geometry may ship together while each keeps its own "
                   "label; inferred geometry may not enter a measured product at all"),
        "statement": ("holes stay reported as holes: hidden_regions quantifies what one pass could "
                      "not see, coverage_denominator publishes that area next to the score, and no "
                      "function here fills the hole in"),
    }
