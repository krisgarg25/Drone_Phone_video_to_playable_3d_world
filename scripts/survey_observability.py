"""Measure what a single flight path can actually triangulate - no promises.

Challenge (i) of SIH26158 is "limited viewing angles due to a single flight
path". This module does not fix that and does not pretend to: it quantifies it,
so the deliverable can say which regions this pass cannot measure and why.

Three measurements, all from camera geometry plus an existing point cloud:

* ``trajectory_quality``   - how much 3D spread the camera path itself has.
* ``baselines``            - per-camera range to a scene target and the effective
  (perpendicular) baseline to its nearest other camera.
* ``region_observability`` - per grid cell of the scene: how many cameras reach
  it, and the largest stereo intersection angle the whole pass offers there.

Everything is a measurement in metres and degrees, using only numpy. It is *not*
accuracy evidence: a cell reported observable only means rays cross there at an
angle wide enough to bound depth error. Surface completeness, occlusion and
georeferenced error are measured elsewhere (``survey_visibility``,
``survey_evaluation``), and the point cloud handed to ``region_observability``
comes from this same pass, so regions that never reconstructed are missing from
the denominator rather than reported as holes. That caveat is repeated in every
``summarise`` result.
"""
import math

import numpy as np

# --- thresholds (all reachable as keyword arguments; nothing is buried) -------
MIN_SECOND_RATIO = 0.01
"""Degeneracy floor: second/first singular value of the centred trajectory.

Deliberately the same convention and value as ``survey_georef._noncollinear``,
which refuses a Sim3 fit below this ratio. A tool that graded observability with
a different floor would disagree with the fitter that uses it.
"""

WEAK_SECOND_RATIO = 0.10
"""Below this the pass is 'weak': an order of magnitude above the rejection
floor, i.e. a straight line with nothing but GPS wobble for sideways baseline.
A half-orbit scores ~0.44 and a full orbit 1.0, so 0.10 only catches genuinely
line-like flights."""

MIN_VIEWS = 3
"""Cameras that must reach a cell for it to be considered at all. Two is the
bare minimum to triangulate; three is the minimum that lets a bad correspondence
be detected rather than absorbed."""

MIN_PARALLAX_DEG = 5.0
"""Smallest usable stereo intersection angle, in degrees.

From the standard depth-error relation sigma_z = z^2 * sigma_u / (f * B_eff),
with B_eff = z * r for relative parallax r: sigma_z / z = sigma_u / (f * r).
For 1000 px images (f ~ 700 px), a 1 px correspondence error and r = sin(5 deg)
= 0.0875, that is ~1.6 % of range: 0.49 m at 30 m, inside the <=1 m budget. The
same numbers at 2 deg give 4.1 %, i.e. 1.2 m at 30 m: outside the budget. 5 deg
is therefore the floor where per-view depth stops being bounded, not a taste
threshold."""

MAX_CAMERAS_PER_CELL = 256
"""Best-partner angles cost O(k^2) per cell. Above this many in-range cameras,
k is subsampled evenly in path order, which always keeps the first and last
camera and so cannot silently delete the widest angle. Reported as
``subsampling_applied``."""

MIN_RANGE_M = 1e-6
"""Numerical floor only: a camera within this distance of a target has no
defined viewing ray, so it is excluded rather than producing a NaN."""

PLANE_AXES = (0, 1)
"""ENU east/north: the grid is a map grid. Facades need a vertical plane."""

ROUND_DIGITS = 6

_NOT_MEASURED = (
    "This measures camera geometry against an existing point cloud, not surface "
    "completeness: no independent reference surface was compared.",
    "No accuracy claim here. Georeferenced metre error is measured by "
    "survey_evaluation against held-out checkpoints.",
    "The supplied points come from this same pass, so regions that never "
    "reconstructed are absent from the denominator instead of appearing as holes.",
    "Occlusion and per-point view support are measured by survey_visibility, not here.",
    "Cells are gridded on one plane; a facade needs plane_axes set to that wall, "
    "and unobserved faces (behind the camera, under the aircraft) stay unmeasured.",
)


# --- shared validation --------------------------------------------------------
def _matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError(f"{name} must be a non-empty Nx3 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def _ratio(value, name, *, allow_zero=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0 or (not allow_zero and number <= 0):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return number


def _count(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _round(value, digits=ROUND_DIGITS):
    return round(float(value), digits)


def _stats(values):
    """min / median / max of a numeric sequence, or None when it is empty."""
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": int(array.size), "min": _round(array.min()),
            "median": _round(np.median(array)), "max": _round(array.max())}


# --- 1. trajectory ------------------------------------------------------------
def trajectory_quality(enu_positions, *, min_second_ratio=MIN_SECOND_RATIO,
                       weak_second_ratio=WEAK_SECOND_RATIO, min_cameras=3) -> dict:
    """Grade the camera path itself for 3D observability, in metres.

    ``second_over_first`` is the ratio of the second to the first singular value
    of the *centred* trajectory - the identical quantity ``survey_georef`` tests
    before fitting a similarity, recomputed here rather than imported. Below
    ``min_second_ratio`` the verdict is 'degenerate' (the fitter would refuse
    it), below ``weak_second_ratio`` it is 'weak', otherwise 'observable'.

    ``straightness`` is the fraction of trajectory variance lying along the
    principal direction (1.0 for perfectly collinear cameras, 0.5 for a planar
    circle), so a 'weak' verdict always shows how much of the path is a line.
    """
    positions = _matrix(enu_positions, "enu_positions")
    if len(positions) < _count(min_cameras, "min_cameras", minimum=2):
        raise ValueError(f"enu_positions needs at least {min_cameras} cameras")
    floor = _ratio(min_second_ratio, "min_second_ratio")
    weak = _ratio(weak_second_ratio, "weak_second_ratio")
    if weak < floor:
        raise ValueError("weak_second_ratio must be >= min_second_ratio")

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    centred = positions - positions.mean(axis=0)
    # np.linalg.svd returns descending singular values; pad for the N == 3 edge.
    singular = np.linalg.svd(centred, compute_uv=False)
    singular = np.concatenate([singular, np.zeros(3 - len(singular))])[:3]
    first, second = float(singular[0]), float(singular[1])
    total_variance = float(np.sum(singular ** 2))
    straightness = singular[0] ** 2 / total_variance if total_variance > 0 else 0.0
    ratio = second / first if first > 1e-12 else 0.0
    path_length = float(steps.sum())

    if first <= 1e-12:
        verdict, reason = "degenerate", (
            f"the trajectory has no extent: {len(positions)} cameras all coincide, "
            f"path_length {path_length:.4g} m, so nothing is observable")
    elif ratio < floor:
        verdict, reason = "degenerate", (
            f"second/first centred singular value {ratio:.4g} < min_second_ratio="
            f"{floor}: the cameras are collinear, exactly the condition "
            f"survey_georef refuses, so neither a metric similarity nor stereo "
            f"depth is determined by this pass")
    elif ratio < weak:
        verdict, reason = "weak", (
            f"second/first centred singular value {ratio:.4g} >= min_second_ratio="
            f"{floor} but < weak_second_ratio={weak}: a near-straight path "
            f"(straightness {straightness:.4g}) carries almost no sideways "
            f"baseline, so only the forward sweep is measured and facade geometry "
            f"off the flight line is not")
    else:
        verdict, reason = "observable", (
            f"second/first centred singular value {ratio:.4g} >= weak_second_ratio="
            f"{weak} (straightness {straightness:.4g} of the variance lies along the "
            f"principal direction): the path itself spreads in more than one "
            f"horizontal direction")

    return {
        "camera_count": int(len(positions)),
        "path_length_m": _round(path_length),
        "displacement_m": _round(np.linalg.norm(positions[-1] - positions[0])),
        "extent_m": _round(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))),
        "singular_values_m": [_round(v) for v in singular],
        "second_over_first": _round(ratio, 9),
        "third_over_first": _round(float(singular[2]) / first if first > 1e-12 else 0.0, 9),
        "straightness": _round(straightness, 9),
        "verdict": verdict,
        "reason": reason,
        "thresholds": {"min_second_ratio": floor, "weak_second_ratio": weak,
                       "min_cameras": min_cameras},
    }


# --- 2. per-camera baselines --------------------------------------------------
def _partner_table(centers, target, *, eligible):
    """Per-camera nearest eligible partner and the geometry it buys at ``target``.

    Returns (partner_index, separation, effective_baseline, intersection angle,
    ratio) as lists; ``partner_index`` is -1 where no partner exists.
    """
    count = len(centers)
    rays = target[None, :] - centers                     # camera -> target
    ranges = np.linalg.norm(rays, axis=1)
    unit = rays / ranges[:, None]
    separation = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    mask = np.eye(count, dtype=bool) | (~eligible)[None, :]
    separation = np.where(mask, np.inf, separation)
    partners = separation.argmin(axis=1)
    has_partner = np.isfinite(separation[np.arange(count), partners])

    out = [[], [], [], [], []]
    for index in range(count):
        if not has_partner[index]:
            out[0].append(-1)
            out[1].append(0.0)
            out[2].append(0.0)
            out[3].append(0.0)
            out[4].append(0.0)
            continue
        partner = int(partners[index])
        vector = centers[partner] - centers[index]
        # Perpendicular distance from the partner camera to this camera's viewing
        # line: the baseline that actually appears in the depth-error relation.
        effective = float(np.linalg.norm(np.cross(vector, unit[index])))
        # Intersection angle of the two (undirected) light rays at the target.
        # Undirected matters: cameras on opposite sides of a point are just as
        # collinear as cameras on the same side, and a signed angle would call
        # that 180 degrees of parallax.
        cosine = abs(float(rays[index] @ rays[partner]
                           / (ranges[index] * ranges[partner])))
        angle = math.degrees(math.acos(min(1.0, cosine)))
        out[0].append(partner)
        out[1].append(float(separation[index, partner]))
        out[2].append(effective)
        out[3].append(angle)
        out[4].append(effective / max(ranges[index], ranges[partner]))
    out[0] = [int(v) for v in out[0]]
    for column in out[1:]:
        for position, value in enumerate(column):
            column[position] = _round(value)
    return tuple(out)


def baselines(camera_centers, target, *, max_range_m=None,
              min_range_m=MIN_RANGE_M) -> dict:
    """Range to a scene target and the effective baseline to the nearest camera.

    For each camera: its distance to ``target``, its nearest other camera (only
    cameras inside ``max_range_m`` are eligible partners when the cap is set),
    the perpendicular baseline that partner subtends across this camera's
    viewing ray, the stereo intersection angle at the target, and
    ``effective baseline / the longer of the two ranges`` - a relative parallax
    bounded by 1 that equals sin(intersection angle) for an equidistant pair.

    This measures local stereo density at one target. For "can this region be
    triangulated at all" use ``region_observability``, which asks for each
    camera's widest-angle partner instead of its nearest one.
    """
    centers = _matrix(camera_centers, "camera_centers")
    point = _vector(target, "target")
    floor = _ratio(min_range_m, "min_range_m")
    limit = None if max_range_m is None else _ratio(max_range_m, "max_range_m")
    ranges = np.linalg.norm(centers - point, axis=1)
    if np.any(ranges <= floor):
        raise ValueError("every camera must sit farther than min_range_m from the target")
    eligible = np.ones(len(centers), dtype=bool) if limit is None else ranges <= limit
    partner_index, separation, effective, angle, ratio = _partner_table(
        centers, point, eligible=eligible)
    return {
        "camera_count": int(len(centers)),
        "target": [_round(v) for v in point],
        "max_range_m": None if limit is None else _round(limit),
        "cameras_in_range": int(eligible.sum()),
        "range_m": [_round(v) for v in ranges],
        "in_range": [bool(v) for v in eligible],
        "partner_index": partner_index,
        "separation_m": separation,
        "effective_baseline_m": effective,
        "triangulation_angle_deg": angle,
        "baseline_range_ratio": ratio,
        "usable": [bool(reach and partner >= 0)
                   for reach, partner in zip(eligible.tolist(), partner_index)],
    }


# --- 3. per-region observability ---------------------------------------------
def _best_partner_geometry(centers, centroid, ranges):
    """Widest-angle partner per camera: (largest angle in degrees, relative parallax)."""
    unit = (centers - centroid) / ranges[:, None]        # target -> camera lines
    cosine = np.abs(unit @ unit.T)
    # Diagonal = 1, not 0: a camera against itself is a 0 degree intersection,
    # while arccos(0) would invent a 90 degree baseline out of nothing.
    np.fill_diagonal(cosine, 1.0)
    angles = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    if len(centers) < 2:
        return 0.0, np.zeros(len(centers))
    best = angles.argmax(axis=1)
    rows = np.arange(len(centers))
    offset = centers[best] - centers                     # toward the widest partner
    rays = centroid[None, :] - centers                   # camera -> target
    rays = rays / ranges[:, None]
    effective = np.linalg.norm(np.cross(offset, rays), axis=1)
    longest = np.maximum(ranges, ranges[best])
    return float(angles.max()), np.where(angles[rows, best] > 0.0, effective / longest, 0.0)


def region_observability(camera_centers, points, *, cell_size_m, min_views=MIN_VIEWS,
                         min_parallax_deg=MIN_PARALLAX_DEG, max_range_m=None,
                         max_cameras_per_cell=MAX_CAMERAS_PER_CELL,
                         plane_axes=PLANE_AXES, min_range_m=MIN_RANGE_M) -> dict:
    """Per grid cell: cameras in range and the widest stereo angle the pass offers.

    Cells are buckets of ``cell_size_m`` on the ``plane_axes`` axes of the point
    cloud (ENU east/north by default, so the output is a map). For each cell:

    * ``cameras_in_range`` - cameras within ``max_range_m`` of the cell centroid.
    * ``max_triangulation_angle_deg`` - the widest intersection angle any pair of
      those cameras subtends at the cell. This is the honest "can this cell be
      triangulated at all" signal: below ``min_parallax_deg`` nothing can fix its
      depth, however many frames looked at it.
    * ``median_baseline_range_ratio`` - median over the in-range cameras of
      effective baseline / range, each camera paired with its widest-angle
      partner. This is the density signal: a cell can be high here and still be
      measurable, or low here and be saved by a single wide pair.

    Verdicts: 'unobservable' when fewer than ``min_views`` cameras reach the
    cell, 'weak' when the widest angle is under ``min_parallax_deg``, otherwise
    'observable'. ``local_stereo_weak`` marks the cells whose median ratio is
    under sin(min_parallax_deg), i.e. measurable only thanks to an extreme pair.
    """
    centers = _matrix(camera_centers, "camera_centers")
    cloud = _matrix(points, "points")
    size = _ratio(cell_size_m, "cell_size_m")
    needed = _count(min_views, "min_views", minimum=1)
    parallax = _ratio(min_parallax_deg, "min_parallax_deg", allow_zero=True)
    floor = _ratio(min_range_m, "min_range_m")
    cap = _count(max_cameras_per_cell, "max_cameras_per_cell", minimum=2)
    axes = tuple(plane_axes)
    if len(axes) != 2 or any(isinstance(a, (bool, np.bool_))
                             or not isinstance(a, (int, np.integer)) for a in axes):
        raise ValueError("plane_axes must be two integer axes")
    if len(set(int(a) for a in axes)) != 2 or any(int(a) not in (0, 1, 2) for a in axes):
        raise ValueError("plane_axes must be two distinct axes from (0, 1, 2)")
    axes = tuple(int(a) for a in axes)
    limit = None if max_range_m is None else _ratio(max_range_m, "max_range_m")

    keys = np.floor(cloud[:, list(axes)] / size).astype(np.int64)
    order = {}
    for row, key in enumerate(keys):
        order.setdefault((int(key[0]), int(key[1])), []).append(row)

    cells, subsampled = [], False
    for key in sorted(order):
        members = cloud[order[key]]
        centroid = members.mean(axis=0)
        ranges = np.linalg.norm(centers - centroid, axis=1)
        in_range = ranges > floor
        if limit is not None:
            in_range &= ranges <= limit
        count = int(in_range.sum())
        subset = np.flatnonzero(in_range)
        if len(subset) > cap:
            subsampled = True
            subset = subset[np.unique(np.linspace(0, len(subset) - 1, cap).round().astype(int))]
        evaluated = centers[subset]
        if len(subset) >= 2:
            max_angle, ratios = _best_partner_geometry(evaluated, centroid, ranges[subset])
            median_ratio = float(np.median(ratios))
        else:
            max_angle, median_ratio = 0.0, 0.0
        if count < needed:
            verdict = "unobservable"
        elif max_angle < parallax:
            verdict = "weak"
        else:
            verdict = "observable"
        cells.append({
            "cell_index": [int(key[0]), int(key[1])],
            "origin_m": [round(key[0] * size, 6), round(key[1] * size, 6)],
            "center_m": [_round(v) for v in centroid],
            "point_count": int(len(members)),
            "cameras_in_range": count,
            "cameras_evaluated": int(len(subset)),
            "max_triangulation_angle_deg": _round(max_angle),
            "median_baseline_range_ratio": _round(median_ratio),
            "local_stereo_weak": bool(median_ratio < math.sin(math.radians(parallax))),
            "verdict": verdict,
        })

    total = len(cells)
    tallies = {verdict: sum(1 for cell in cells if cell["verdict"] == verdict)
               for verdict in ("observable", "weak", "unobservable")}
    return {
        "cell_size_m": _round(size),
        "plane_axes": [int(a) for a in axes],
        "point_count": int(len(cloud)),
        "cell_count": total,
        "min_views": needed,
        "min_parallax_deg": _round(parallax),
        "max_range_m": None if limit is None else _round(limit),
        "max_cameras_per_cell": cap,
        "subsampling_applied": subsampled,
        "method": ("per in-range camera the widest-angle partner sets both the cell's "
                   "maximum intersection angle and the median baseline/range ratio; "
                   "the verdict uses the maximum angle because a single wide pair is "
                   "enough to fix depth"
                   + ("; cameras above max_cameras_per_cell were subsampled evenly in "
                      "path order with endpoints retained" if subsampled else "")),
        "observable_cells": tallies["observable"],
        "weak_cells": tallies["weak"],
        "unobservable_cells": tallies["unobservable"],
        "weak_fraction": _round(tallies["weak"] / total) if total else 0.0,
        "unobservable_fraction": _round(tallies["unobservable"] / total) if total else 0.0,
        "unmeasurable_fraction": _round((tallies["weak"] + tallies["unobservable"]) / total)
        if total else 0.0,
        "cells": cells,
    }


# --- 4. combined summary ------------------------------------------------------
def summarise(trajectory, camera_centers, points, *, cell_size_m, target=None,
              min_second_ratio=MIN_SECOND_RATIO, weak_second_ratio=WEAK_SECOND_RATIO,
              min_views=MIN_VIEWS, min_parallax_deg=MIN_PARALLAX_DEG,
              max_range_m=None, max_cameras_per_cell=MAX_CAMERAS_PER_CELL,
              plane_axes=PLANE_AXES) -> dict:
    """Combine the three measurements and state the limits in plain language.

    ``trajectory`` is the camera path (ENU metres, typically the telemetry or the
    registered camera centres). ``points`` should be the reconstructed cloud the
    claim is about, in the same frame. When ``target`` is omitted the mean of
    ``points`` is used, the convention the frame-selection measurement used.
    """
    cloud = _matrix(points, "points")
    if target is None:
        point, source = cloud.mean(axis=0), "mean of points"
    else:
        point, source = _vector(target, "target"), "supplied"

    quality = trajectory_quality(trajectory, min_second_ratio=min_second_ratio,
                                 weak_second_ratio=weak_second_ratio)
    pairs = baselines(camera_centers, point, max_range_m=max_range_m)
    regions = region_observability(camera_centers, cloud, cell_size_m=cell_size_m,
                                   min_views=min_views, min_parallax_deg=min_parallax_deg,
                                   max_range_m=max_range_m,
                                   max_cameras_per_cell=max_cameras_per_cell,
                                   plane_axes=plane_axes)

    angles = np.asarray(pairs["triangulation_angle_deg"], dtype=np.float64)
    in_range = np.asarray(pairs["in_range"], dtype=bool)
    weak_cameras = int((angles[in_range] < min_parallax_deg).sum())
    limits = []
    if quality["verdict"] != "observable":
        limits.append(
            f"Trajectory verdict is {quality['verdict']}: {quality['reason']} "
            f"A single pass in this shape contributes forward sweep, not sideways "
            f"baseline, so geometry off the flight line is not recoverable from it.")
    if regions["weak_cells"]:
        worst = [f"({cell['center_m'][0]:.1f}, {cell['center_m'][1]:.1f})"
                 for cell in regions["cells"] if cell["verdict"] == "weak"][:5]
        limits.append(
            f"{regions['weak_cells']} of {regions['cell_count']} cells sit outside any "
            f"baseline > {regions['min_parallax_deg']}\u00b0 and cannot be reconstructed "
            f"from this single pass (widest intersection angle in them: "
            f"{max((c['max_triangulation_angle_deg'] for c in regions['cells']
                    if c['verdict'] == 'weak'), default=0.0):.2f}\u00b0). Regions: "
            f"{', '.join(worst)}{' ...' if regions['weak_cells'] > len(worst) else ''}.")
    if regions["unobservable_cells"]:
        reach = (f"within {max_range_m:.1f} m" if max_range_m is not None else "in range")
        limits.append(
            f"{regions['unobservable_cells']} of {regions['cell_count']} cells have fewer "
            f"than {min_views} cameras {reach}; nothing about them can be measured by "
            f"this pass at all.")
    if weak_cameras:
        limits.append(
            f"{weak_cameras} of {int(in_range.sum())} in-range cameras contribute a "
            f"baseline under {min_parallax_deg}\u00b0 at the scene target: they add frames "
            f"and runtime, not depth information.")
    thin = [cell for cell in regions["cells"] if cell["local_stereo_weak"]
            and cell["verdict"] == "observable"]
    if thin:
        limits.append(
            f"{len(thin)} cells are called observable only because one far pair opens "
            f"an angle there; their median baseline/range ratio is under "
            f"{math.sin(math.radians(min_parallax_deg)):.4f}, so local stereo there is "
            f"thin and depth will be noisy.")

    return {
        "verdict": ("degenerate" if quality["verdict"] == "degenerate"
                    else "limited" if (regions["unmeasurable_fraction"] > 0
                                       or quality["verdict"] == "weak")
                    else "adequate-view-geometry"),
        "target": {"position_m": [_round(v) for v in point], "source": source},
        "trajectory": quality,
        "baselines": {
            "camera_count": pairs["camera_count"],
            "cameras_in_range": pairs["cameras_in_range"],
            "usable_baselines": int(sum(pairs["usable"])),
            "range_m": _stats(pairs["range_m"]),
            "effective_baseline_m": _stats(pairs["effective_baseline_m"]),
            "triangulation_angle_deg": _stats(pairs["triangulation_angle_deg"]),
            "baseline_range_ratio": _stats(pairs["baseline_range_ratio"]),
            "cameras_below_min_parallax": weak_cameras,
        },
        "regions": {k: v for k, v in regions.items() if k != "cells"} | {
            "weakest_cells": sorted((cell for cell in regions["cells"]
                                     if cell["verdict"] != "observable"),
                                    key=lambda c: c["max_triangulation_angle_deg"])[:10]},
        "thresholds": {"min_second_ratio": quality["thresholds"]["min_second_ratio"],
                       "weak_second_ratio": quality["thresholds"]["weak_second_ratio"],
                       "min_views": min_views, "min_parallax_deg": min_parallax_deg,
                       "cell_size_m": regions["cell_size_m"],
                       "max_range_m": max_range_m,
                       "max_cameras_per_cell": max_cameras_per_cell},
        "limits": limits,
        "not_measured": list(_NOT_MEASURED),
    }
