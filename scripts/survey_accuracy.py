"""Metric accuracy without extensive ground control: the protocol, made executable.

stdlib + NumPy only, CPU only, no I/O. This module is the accuracy protocol from the
project's own evaluation plan: fit the alignment on controls ONLY, then measure errors
on held-out surveyed checkpoints, and label every number by what the alignment it
passed through removed.

Three tracks are always reported side by side and never interchangeably:

``unaligned``    the absolute metric result; nothing was fitted to the reference.
``se3_aligned``  rigid body alignment: global position/orientation error is gone,
                 metric SCALE error survives, so it is a secondary geometry check.
``sim3_aligned`` similarity alignment: the fitted scale absorbs any metric scale
                 error, so it can look perfect on a model that is 20 % the wrong
                 size. It is structurally incapable of supporting a metric or
                 absolute accuracy claim, and ``accuracy_verdict``/``report`` enforce
                 that by refusing rather than by documenting.

Nothing here measures a real scene. Accuracy needs surveyed checkpoints plus a
declared CRS and vertical datum; without those inputs every path ends in
``accuracy_validated=False``.
"""
import copy
import hashlib
import json
import math

import numpy as np

try:  # imported as scripts.survey_accuracy by the tests
    from scripts import survey_evaluation as evaluation
    from scripts import survey_georef as georef
    from scripts import survey_gnss as gnss
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_evaluation as evaluation
    import survey_georef as georef
    import survey_gnss as gnss

# Degeneracy threshold shared with the georeferencing lane (via survey_gnss).
MIN_SECOND_RATIO = gnss.MIN_SECOND_RATIO

ROW_FIELDS = ("id", "surveyed")
MODES = ("unaligned", "se3", "sim3")
TRACK_KEYS = {"unaligned": "unaligned", "se3": "se3_aligned", "sim3": "sim3_aligned"}
ALIGNMENTS = {"unaligned": "none", "se3": "rigid_se3", "sim3": "similarity_sim3"}
MODE_TEXT = {
    "unaligned": "unaligned (absolute: nothing was fitted to the reference; this is the only "
                 "track that can carry a metric accuracy statement)",
    "se3": "se3_aligned (rigid SE(3): global position and orientation error removed, metric "
           "SCALE error preserved - a secondary geometry check, not absolute accuracy)",
    "sim3": "sim3_aligned (similarity Sim(3): the fitted scale absorbs any metric scale error - "
            "never evidence of metric or absolute accuracy)",
}
METRIC_STATISTICS = ("horizontal_rmse_m", "vertical_rmse_m", "rmse_3d_m", "median_3d_m",
                     "p95_3d_m", "max_3d_m")
VERTICAL_DATUMS = ("ellipsoidal", "orthometric", "unknown")
STRATA = 4
MIN_TABLE_FRACTION_OF_SCENE = 0.5
MIN_SUBSET_FRACTION_OF_SCENE = 0.25
MIN_HEIGHT_SPREAD_FRACTION = 1e-9  # below this the survey has no height spread to test
SPREAD_MIN_SAMPLE = 8              # at least this many points before proportions are fair


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer,
                                                                      np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _text(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _vector(value, name):
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must hold three finite numbers")
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != 3:
        raise ValueError(f"{name} must hold exactly three numbers (E, N, U metres)")
    return np.array([_number(item, f"{name}[]") for item in raw], dtype=np.float64)


def _table(rows, name="rows", *, require_model=False):
    """Validate a survey table into (ids, surveyed Nx3, model Nx3 or None per row)."""
    if isinstance(rows, (str, bytes, dict)) or not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must be a nonempty list of surveyed point dicts")
    ids, surveyed, model = [], [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{name} entries must be dicts with fields {list(ROW_FIELDS)}")
        for field in ROW_FIELDS:
            if field not in row:
                raise ValueError(f"a {name} entry is missing the required field {field!r}")
        identifier = _text(row["id"], "id")
        if identifier in ids:
            raise ValueError(f"duplicate surveyed point id {identifier!r}")
        if require_model and "model" not in row:
            raise ValueError(f"{identifier} has no 'model' coordinates: a point that was never "
                             "reconstructed cannot be a checkpoint")
        ids.append(identifier)
        surveyed.append(_vector(row["surveyed"], "surveyed"))
        model.append(_vector(row["model"], "model") if "model" in row else np.zeros(3))
    return ids, np.array(surveyed, dtype=np.float64), np.array(model, dtype=np.float64)


def _diameter(points):
    """Exact maximum pairwise distance for the small tables this protocol accepts."""
    best = 0.0
    for start in range(0, len(points) - 1, 256):
        block = points[start:start + 256]
        tail = points[start:]
        deltas = tail[None, :, :] - block[:, None, :]
        best = max(best, float(np.sqrt((deltas * deltas).sum(axis=2)).max()))
    return best


def _spread(points):
    centred = points - points.mean(axis=0)
    return {"count": int(len(points)),
            "centroid_m": [float(value) for value in points.mean(axis=0)],
            "radius_of_gyration_m": float(np.sqrt(np.mean((centred * centred).sum(axis=1)))),
            "diameter_m": _diameter(points),
            "height_range_m": float(np.ptp(points[:, 2]))}


def _rank_ratios(points):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    first = float(singular[0])
    if first <= 1e-12:
        return 0.0, 0.0
    return float(singular[1] / first), float(singular[2] / first)


def _strata(points, stratify_by, ids):
    if stratify_by in (None, "none"):
        return np.zeros(len(points), dtype=int)
    order = sorted(range(len(points)), key=lambda index: (points[index, 2], ids[index]))
    labels = np.zeros(len(points), dtype=int)
    for position, index in enumerate(order):
        labels[index] = min(STRATA - 1, position * STRATA // len(points))
    return labels


def _seeded_spread_pick(points, candidates, budget, rng, already=()):
    """Draw controls with probability proportional to squared distance to what is
    already chosen (k-means++ seeding). That covers the scene instead of clustering,
    and makes the seed genuinely decide the partition instead of hiding a choice."""
    picked, pool, reference = [], list(candidates), list(already)
    for _ in range(min(budget, len(pool))):
        if not reference:
            chosen = pool[int(rng.integers(len(pool)))]
        else:
            anchors = points[reference]
            distances = np.array([float(np.min(((anchors - point) ** 2).sum(axis=1)))
                                  for point in points[pool]], dtype=np.float64)
            total = float(distances.sum())
            chosen = (pool[int(rng.integers(len(pool)))] if total <= 0.0 else
                      pool[int(rng.choice(len(pool), p=distances / total))])
        picked.append(chosen)
        reference.append(chosen)
        pool.remove(chosen)
    return picked


def split_reference(rows, *, fit_fraction=0.6, seed=0, stratify_by="height", min_controls=4,
                    min_checkpoints=4, min_spread_fraction=0.6, scene_extent_m=None):
    """Partition surveyed points into fitting controls and held-out CHECKPOINTS.

    Controls come from seeded, height-stratified, distance-proportional farthest-point
    sampling, so they are spread over position AND height rather than clustered beside
    the takeoff point, and the same (rows, fit_fraction, seed, stratify_by) reproduces
    exactly the same partition - ``replay`` echoes the parameters plus a content digest
    so a changed survey table cannot be replayed as if it were unchanged.

    The split is refused, never quietly degraded, when the result could not support an
    accuracy statement: too few points either side, near-collinear controls (the
    georeferencing lane would reject them too), checkpoints clustered in plan or in
    height, or a survey that does not cover the declared scene extent. Every threshold
    is a declared engineering minimum, not a statistical guarantee and not an official
    sample-size requirement.
    """
    fit_fraction = _number(fit_fraction, "fit_fraction", positive=True)
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between 0 and 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer for the split to be replayable")
    if stratify_by not in (None, "none", "height"):
        raise ValueError("stratify_by must be 'height', 'none' or None")
    for name, value in (("min_controls", min_controls), ("min_checkpoints", min_checkpoints)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
                or value < 3:
            raise ValueError(f"{name} must be an integer of at least 3")
    fraction = _number(min_spread_fraction, "min_spread_fraction", positive=True)
    if fraction > 1.0:
        raise ValueError("min_spread_fraction cannot exceed 1")
    scene = None if scene_extent_m is None else _number(scene_extent_m, "scene_extent_m",
                                                       positive=True)
    ids, points, _ = _table(rows)
    count = len(ids)
    budget = int(round(fit_fraction * count))
    if budget < min_controls or count - budget < min_checkpoints:
        raise ValueError(f"cannot split {count} surveyed points at fit_fraction={fit_fraction}: "
                         f"at least {min_controls} controls and {min_checkpoints} held-out "
                         "checkpoints are required")
    labels = _strata(points, stratify_by, ids)
    rng = np.random.default_rng(int(seed))
    chosen = []
    if stratify_by in (None, "none"):
        chosen = _seeded_spread_pick(points, list(range(count)), budget, rng)
    else:
        members = [[index for index in range(count) if labels[index] == stratum]
                   for stratum in range(STRATA)]
        quotas = [budget * len(group) // count for group in members]
        remaining = budget - sum(quotas)
        for stratum in range(STRATA):  # largest-remainder top-up in deterministic order
            if remaining > 0 and members[stratum]:
                quotas[stratum] += 1
                remaining -= 1
        for stratum in range(STRATA):
            taken = _seeded_spread_pick(points, members[stratum],
                                        min(quotas[stratum], len(members[stratum])), rng,
                                        already=chosen)
            chosen.extend(taken)
        if len(chosen) < budget:  # a stratum was smaller than its quota: fill globally
            chosen.extend(_seeded_spread_pick(points, [index for index in range(count)
                                                       if index not in chosen],
                                              budget - len(chosen), rng, already=chosen))
    control_indices = sorted(set(chosen))
    control_set = set(control_indices)
    checkpoint_indices = [index for index in range(count) if index not in control_set]
    controls, checkpoints = points[control_indices], points[checkpoint_indices]
    copied = copy.deepcopy(rows)
    second, _ = _rank_ratios(controls)
    if second < MIN_SECOND_RATIO:
        raise ValueError("the selected controls are near-collinear (second/first centred singular "
                         f"value {second:.3g} < {MIN_SECOND_RATIO}), so an SE(3)/Sim(3) alignment "
                         "would be degenerate: scripts/survey_georef.py would reject this spread "
                         "too. Survey points off the line and repeat.")
    whole = _spread(points)
    if scene is not None and whole["diameter_m"] < MIN_TABLE_FRACTION_OF_SCENE * scene:
        raise ValueError(f"the surveyed table spans {whole['diameter_m']:.1f} m, under "
                         f"{MIN_TABLE_FRACTION_OF_SCENE} of the declared scene extent of "
                         f"{scene:.1f} m: it cannot represent the scene it claims to validate")
    fractions = {}
    for name, subset in (("control", controls), ("checkpoint", checkpoints)):
        spread = _spread(subset)
        ratio = (spread["radius_of_gyration_m"] / whole["radius_of_gyration_m"]
                 if whole["radius_of_gyration_m"] > 0 else 0.0)
        fractions[f"{name}_radius_fraction"] = float(ratio)
        if ratio < fraction:
            raise ValueError(f"the {name} points are clustered: their radius of gyration is only "
                             f"{ratio:.2f} of the survey's (minimum {fraction}). An error "
                             "measured on one corner of the scene is not a scene accuracy "
                             "statement: spread the sample instead of reporting it.")
        if whole["height_range_m"] > MIN_HEIGHT_SPREAD_FRACTION:
            vertical = spread["height_range_m"] / whole["height_range_m"]
            fractions[f"{name}_height_fraction"] = float(vertical)
            # A handful of points cannot span a height range proportionally, so the
            # small-sample rule is the weaker but still meaningful one: at least two
            # different heights. From SPREAD_MIN_SAMPLE upwards the fraction applies.
            if len(subset) >= SPREAD_MIN_SAMPLE:
                fractions[f"{name}_height_rule"] = "fraction"
                if vertical < fraction:
                    raise ValueError(f"the {name} points cover only {vertical:.2f} of the "
                                     "surveyed height range (minimum "
                                     f"{fraction}): a vertical error would go unmeasured, which "
                                     "is the classic silent failure. Spread checkpoints across "
                                     "height too.")
            else:
                distinct = int(np.unique(np.round(subset[:, 2], 6)).size)
                fractions[f"{name}_height_rule"] = f"distinct_heights({distinct})"
                if distinct < 2:
                    raise ValueError(f"the {name} points all share one height: nothing about "
                                     "vertical error can be measured on them, which is the "
                                     "classic silent failure")
        if scene is not None and spread["diameter_m"] < MIN_SUBSET_FRACTION_OF_SCENE * scene:
            raise ValueError(f"the {name} points span {spread['diameter_m']:.1f} m, under "
                             f"{MIN_SUBSET_FRACTION_OF_SCENE} of the declared scene extent: that "
                             "is not a spread sample of this scene")
    control_strata = sorted({int(labels[index]) for index in control_indices})
    checkpoint_strata = sorted({int(labels[index]) for index in checkpoint_indices})
    if stratify_by not in (None, "none") and len(checkpoint_strata) < 2:
        raise ValueError("every held-out checkpoint sits in one height stratum, so the split "
                         "cannot detect a vertical error at all")
    canonical = json.dumps([[row["id"], row["surveyed"], row.get("model")] for row in rows],
                           sort_keys=True)
    warnings = ["Held out of the fit by construction here; whether these points also "
                "entered reconstruction, keyframe selection or training is declared by "
                "the caller and cannot be proven by this module.",
                "Spread minima, stratum count and sample size are declared engineering "
                "choices, not statistical guarantees and not an official requirement."]
    if whole["height_range_m"] <= MIN_HEIGHT_SPREAD_FRACTION:
        warnings.append("the surveyed table has no height variation at all, so no split of it "
                        "can measure a vertical error; add control points at different heights "
                        "before reporting a vertical number")
    return {"schema_version": 1, "provenance": "survey_accuracy.split_reference",
            "method": "seeded_stratified_farthest_point",
            "controls": [copied[index] for index in control_indices],
            "checkpoints": [copied[index] for index in checkpoint_indices],
            "control_ids": [ids[index] for index in control_indices],
            "checkpoint_ids": [ids[index] for index in checkpoint_indices],
            "control_indices": [int(index) for index in control_indices],
            "checkpoint_indices": [int(index) for index in checkpoint_indices],
            "counts": {"rows": count, "controls": len(control_indices),
                       "checkpoints": len(checkpoint_indices)},
            "fit_fraction": fit_fraction, "seed": int(seed), "stratify_by": stratify_by,
            "min_controls": int(min_controls), "min_checkpoints": int(min_checkpoints),
            "strata": {"count": STRATA if stratify_by not in (None, "none") else 1,
                       "controls_covered": control_strata,
                       "checkpoints_covered": checkpoint_strata},
            "spread": dict(fractions, all=whole, controls=_spread(controls),
                           checkpoints=_spread(checkpoints), minimum_fraction=fraction,
                           scene_extent_m=scene,
                           absolute="declared" if scene is not None else "not_declared"),
            "disjoint": not (control_set & set(checkpoint_indices)),
            "independent_checkpoints_held_out": True,
            "replay": {"fit_fraction": fit_fraction, "seed": int(seed),
                       "stratify_by": stratify_by, "n_rows": count,
                       "input_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest()},
            "warnings": warnings}


def _fit(source, target, *, allow_scale):
    """Least squares SE(3) or Sim(3) taking x_target ~ scale * (x_source @ R.T) + t.

    Same convention as scripts/survey_georef.py, and the same refusal of degenerate
    geometry: a near-collinear control set cannot fix an orientation.
    """
    if len(source) < 3:
        raise ValueError("at least three controls are required to align a 3-D trajectory; fewer "
                         "cannot fix an orientation")
    for points, name in ((source, "control model"), (target, "control surveyed")):
        second, _ = _rank_ratios(points)
        if second < MIN_SECOND_RATIO:
            raise ValueError(f"the {name} points are near-collinear (second/first centred singular "
                             f"value {second:.3g} < {MIN_SECOND_RATIO}): refusing to fit a "
                             "degenerate alignment to them")
    mx, my = source.mean(axis=0), target.mean(axis=0)
    x, y = source - mx, target - my
    left, singular, right = np.linalg.svd(y.T @ x / len(x))
    flip = np.array([1.0, 1.0, float(np.sign(np.linalg.det(left @ right)))])
    if flip[2] == 0.0:
        raise ValueError("reflection-ambiguous control geometry; refusing to choose a rotation")
    rotation = left @ np.diag(flip) @ right
    scale = 1.0 if not allow_scale else float((singular * flip).sum() / ((x * x).sum() / len(x)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("fitted scale is not a positive finite number")
    translation = my - scale * (rotation @ mx)
    residuals = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
    return scale, rotation, translation, residuals


def alignment_trackpoints(controls, *, allow_scale, checkpoint_ids=()):
    """Fit the alignment on CONTROLS ONLY, and refuse any checkpoint that leaked in.

    ``allow_scale`` has no default on purpose: deciding whether the fit may absorb a
    metric scale error is a claim-relevant choice that belongs to the caller. The
    result carries ``status="aligned"`` so scripts/survey_georef.transform_points can
    apply it, and ``kind`` records which track it belongs to.
    """
    if type(allow_scale) is not bool:
        raise ValueError("allow_scale must be an explicit True or False: it decides whether the "
                         "fit is allowed to absorb metric scale error")
    ids, surveyed, model = _table(controls, "controls", require_model=True)
    leaked = set(ids) & set(checkpoint_ids or ())
    if leaked:
        raise ValueError(f"checkpoint id(s) {sorted(leaked)[:5]} appear among the controls: a "
                         "point used to fit the alignment can no longer validate it")
    scale, rotation, translation, residuals = _fit(model, surveyed, allow_scale=allow_scale)
    return {"schema_version": 1, "status": "aligned",
            "provenance": "survey_accuracy.alignment_trackpoints",
            "kind": "sim3" if allow_scale else "se3", "from": "model", "to": "surveyed",
            "scale": float(scale), "rotation": [[float(v) for v in row] for row in rotation],
            "translation": [float(v) for v in translation],
            "control_count": len(ids), "control_ids": ids,
            "residual_m": [float(value) for value in residuals],
            "fit_rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
            "fitted_from": "controls_only",
            "checkpoint_ids_excluded": sorted(set(checkpoint_ids or ())),
            "scale_error_absorbed": allow_scale,
            "warnings": ["Fit residuals are self-consistency against the same controls used to "
                         "fit; they are not independent accuracy evidence."]}


def _applied(checkpoints, transform, mode):
    ids, surveyed, model = _table(checkpoints, "checkpoints", require_model=True)
    if mode == "unaligned":
        if transform is not None:
            raise ValueError("the primary absolute result must carry no transform: passing one "
                             "would hide an alignment behind the word 'unaligned'")
        return ids, model, surveyed, None
    if not isinstance(transform, dict) or transform.get("schema_version") != 1 \
            or transform.get("status") != "aligned":
        raise ValueError(f"mode {mode!r} needs a validated alignment from alignment_trackpoints")
    if transform.get("kind") != mode:
        raise ValueError(f"mode {mode!r} was given a {transform.get('kind')!r} transform: a "
                         "Sim(3) fit absorbs metric scale error and a rigid fit does not, so the "
                         "label and the transform must agree")
    if mode == "se3" and float(transform["scale"]) != 1.0:
        raise ValueError("an se3 track must not carry a fitted scale")
    return ids, georef.transform_points(model, transform), surveyed, transform


def evaluate_checkpoints(checkpoints, transform, mode):
    """Measure held-out checkpoint error in one explicitly labelled alignment track."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    ids, reconstructed, surveyed, used = _applied(checkpoints, transform, mode)
    metrics = evaluation.checkpoint_metrics(reconstructed, surveyed)
    metrics["alignment"] = ALIGNMENTS[mode]
    cannot = (["nothing about metric scale: the fitted scale is whatever best matched the "
               "controls, so an arbitrarily wrongly sized model can score perfectly"]
              if mode == "sim3" else
              ["global position and orientation error: those were removed by the fit"]
              if mode == "se3" else
              ["completeness or surface coverage, and anything about a scene that was not "
               "surveyed"])
    return {"schema_version": 1, "provenance": "survey_accuracy.evaluate_checkpoints",
            "mode": mode, "label": MODE_TEXT[mode], "alignment": ALIGNMENTS[mode],
            "metrics": metrics, "checkpoint_count": len(ids), "checkpoint_ids": ids,
            "transform": None if used is None else {
                "kind": used["kind"], "scale": float(used["scale"]),
                "control_count": used["control_count"], "fit_rmse_m": used["fit_rmse_m"]},
            "usable_as_accuracy_evidence": mode == "unaligned",
            "feeds_build_evaluation": mode == "unaligned",
            "scale_error_absorbed": mode == "sim3",
            "placement_error_removed": mode in ("se3", "sim3"),
            "what_this_cannot_prove": cannot + ["the survey's own quality, CRS or vertical datum"]}


def accuracy_verdict(track, *, statistic=None, target_m=1.0):
    """Turn ONE track into a verdict - and refuse to turn a Sim(3) track into any."""
    if not isinstance(track, dict) or track.get("provenance") != \
            "survey_accuracy.evaluate_checkpoints":
        raise ValueError("a verdict needs an evaluate_checkpoints track, not hand-made numbers")
    mode = track["mode"]
    if statistic is not None and (isinstance(statistic, bool) or not isinstance(statistic, str)
                                  or statistic not in METRIC_STATISTICS):
        raise ValueError(f"statistic must be one of {list(METRIC_STATISTICS)}")
    limit = _number(target_m, "target_m", positive=True)
    value = None if statistic is None else float(track["metrics"][statistic])
    if mode == "sim3":
        raise ValueError("a Sim(3)-aligned number cannot be reported as metric or absolute "
                         "accuracy: the fitted scale absorbs the very scale error being asked "
                         "about, so the score is compatible with an arbitrarily wrong size. "
                         "Report the unaligned track, or label this one as relative shape only.")
    if mode == "se3":
        return {"schema_version": 1, "kind": "relative_geometry_secondary", "mode": mode,
                "statistic": statistic, "value_m": value, "threshold_m": limit,
                "absolute_accuracy": None,
                "meets_relative_geometry_target": None if value is None else bool(value <= limit),
                "reason": "rigid alignment already removed global position and orientation error, "
                          "so this is a shape and scale diagnostic, not absolute accuracy"}
    if statistic is None:
        return {"schema_version": 1, "kind": "absolute_metric_accuracy", "mode": mode,
                "statistic": None, "value_m": None, "threshold_m": limit, "meets_target": None,
                "reason": "the brief states a <=1 m target without naming a statistic, so no pass "
                          "or fail is inferred. Declare the statistic (for example rmse_3d_m or "
                          "p95_3d_m) and this function will compare it"}
    return {"schema_version": 1, "kind": "absolute_metric_accuracy", "mode": mode,
            "statistic": statistic, "value_m": value, "threshold_m": limit,
            "meets_target": bool(value <= limit),
            "reason": f"unaligned {statistic} = {value} m against a declared {limit} m target on "
                      f"{track['checkpoint_count']} held-out checkpoints"}


def report(*, rows, crs, vertical_datum, fit_fraction=0.6, seed=0, stratify_by="height",
           withheld_from_reconstruction=False, target_statistic=None, accuracy_target_m=1.0,
           primary_track="unaligned", min_controls=4, min_checkpoints=4,
           min_spread_fraction=0.6, scene_extent_m=None):
    """Run the whole controls-only / held-out checkpoint protocol and label all tracks.

    ``accuracy_validated`` becomes True only when the checkpoints were genuinely held
    out of both alignments (audited by id disjointness), the CRS and vertical datum are
    declared, the primary track is the unaligned one, and the caller explicitly declares
    that the survey was withheld from reconstruction. That last item stays a declaration
    rather than a default True, because no computation here can check it.
    """
    crs = _text(crs, "crs")
    if isinstance(vertical_datum, bool) or not isinstance(vertical_datum, str) \
            or vertical_datum.strip().lower() not in VERTICAL_DATUMS:
        raise ValueError(f"vertical_datum must be one of {list(VERTICAL_DATUMS)}")
    vertical_datum = vertical_datum.strip().lower()
    if type(withheld_from_reconstruction) is not bool:
        raise ValueError("withheld_from_reconstruction must be an explicit True or False "
                         "declaration by the caller")
    if primary_track == TRACK_KEYS["sim3"]:
        raise ValueError("primary_track='sim3_aligned' asks for an accuracy verdict from a "
                         "Sim(3) fit, which absorbs metric scale error and can therefore never "
                         "support a metric or absolute accuracy claim")
    if primary_track not in TRACK_KEYS.values():
        raise ValueError(f"primary_track must be one of {sorted(TRACK_KEYS.values())}")
    if target_statistic is not None and target_statistic not in METRIC_STATISTICS:
        raise ValueError(f"target_statistic must be one of {list(METRIC_STATISTICS)} or None")
    _number(accuracy_target_m, "accuracy_target_m", positive=True)
    split = split_reference(rows, fit_fraction=fit_fraction, seed=seed, stratify_by=stratify_by,
                            min_controls=min_controls, min_checkpoints=min_checkpoints,
                            min_spread_fraction=min_spread_fraction, scene_extent_m=scene_extent_m)
    controls, checkpoints = split["controls"], split["checkpoints"]
    rigid = alignment_trackpoints(controls, allow_scale=False,
                                 checkpoint_ids=split["checkpoint_ids"])
    similarity = alignment_trackpoints(controls, allow_scale=True,
                                       checkpoint_ids=split["checkpoint_ids"])
    tracks = {"unaligned": evaluate_checkpoints(checkpoints, None, "unaligned"),
              "se3_aligned": evaluate_checkpoints(checkpoints, rigid, "se3"),
              "sim3_aligned": evaluate_checkpoints(checkpoints, similarity, "sim3")}
    verdict = accuracy_verdict(tracks[primary_track], statistic=target_statistic,
                              target_m=accuracy_target_m)
    reasons = []
    if primary_track != TRACK_KEYS["unaligned"]:
        reasons.append(f"primary_track={primary_track} removed error by fitting to the reference; "
                       "only the unaligned track can validate absolute accuracy")
    if not withheld_from_reconstruction:
        reasons.append("the caller has not declared that these independent surveyed checkpoints "
                       "were withheld from reconstruction, so their independence is unverified")
    if vertical_datum == "unknown":
        reasons.append("the vertical datum is undeclared, so no vertical error can be "
                       "interpreted: ellipsoidal and orthometric heights differ by the local "
                       "geoid undulation, metres to tens of metres")
    if not split["disjoint"]:
        reasons.append("controls and checkpoints are not disjoint")
    return {"schema_version": 1, "provenance": "survey_accuracy.report",
            "protocol": "controls_only_alignment_with_held_out_checkpoints",
            "crs": crs, "vertical_datum": vertical_datum, "coordinate_units": "m",
            "counts": split["counts"], "split": split["replay"],
            "checkpoint_spread": split["spread"]["checkpoints"],
            "control_spread": split["spread"]["controls"],
            "spread_fractions": {key: value for key, value in split["spread"].items()
                                 if key.endswith("_fraction")},
            "tracks": tracks, "primary_track": primary_track,
            "headline": dict(tracks[primary_track]["metrics"]),
            "verdict": verdict,
            "target": {"statistic": target_statistic, "threshold_m": float(accuracy_target_m)},
            "meets_accuracy_target": verdict.get("meets_target"),
            "accuracy_validated": not reasons,
            "accuracy_validation_reasons": reasons or [
                "checkpoints were held out of both alignments by id and are disjoint from the "
                "controls, and the caller declared them independent of the reconstruction"],
            "fit_rmse_m": {"se3_aligned": rigid["fit_rmse_m"],
                           "sim3_aligned": similarity["fit_rmse_m"]},
            "fitted_sim3_scale": similarity["scale"],
            "what_this_does_not_prove": [
                "that this accuracy holds on any other scene, altitude, weather or flight: one "
                "surveyed scene validates one scene",
                "completeness or surface coverage: checkpoints are points, and a missing facade "
                "has no checkpoint on it",
                "anything via the sim3_aligned track: its scale factor absorbs metric scale error, "
                "so a small sim3 number is not evidence of metric accuracy",
                "anything via the se3_aligned track as absolute accuracy: rigid alignment removed "
                "global position and orientation error before the number was computed",
                "the survey's own quality, CRS or vertical datum; nor does it separate the GNSS "
                "bias, clock offset and antenna lever arm that placed the model, all of which are "
                "absorbed into the unaligned error",
                "per-point or per-object accuracy: RMSE, median, P95 and max are aggregate "
                "statistics that can hide a systematic tilt or an outlier region",
                f"{split['counts']['checkpoints']} correlated checkpoints are not "
                f"{split['counts']['checkpoints']} independent scenes"],
            "warnings": split["warnings"] + [
                "unaligned errors mix reconstruction error with georeferencing error; this "
                "protocol measures the sum and cannot attribute either part"]}


def _require(value, provenance, required, name):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1 \
            or value.get("provenance") != provenance:
        raise ValueError(f"{name} must be the output of {provenance}, so this decision cannot be "
                         "fed invented numbers")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{name} is missing required field(s) {missing}")
    return value


def gcp_requirement(*, telemetry_quality, trajectory_observability, fix_quality=None,
                    vertical_reference=None, clock_bounds=None, min_evaluation_points=4):
    """Answer the brief's actual question: can GPS alone give metric scale this flight?

    Inputs are the reports produced by scripts/survey_gnss.py and are validated as such -
    a hand-written summary of a trace is refused, because the whole point of the question
    is that the numbers must come from the data. The output states whether metric scale
    is identifiable, how confident that statement may be, and the cheapest additional
    constraint that would settle it (one known baseline, a barometric trend, IMU
    attitude, or an RTK fix). It is decision support: it is not a measurement, it claims
    no accuracy, and it never states a required number of ground control points.
    """
    quality = _require(telemetry_quality, "survey_gnss.quality_report",
                       ("count", "median_step_m", "median_horizontal_std_m", "path_length_m",
                        "total_displacement_m", "suspicious_count", "gaps_over_threshold_count",
                        "displacement_below_noise"), "telemetry_quality")
    observed = _require(trajectory_observability, "survey_gnss.observability",
                        ("status", "geometry", "identifiability", "metric_scale_support",
                         "clock_offset_and_lever_arm_separable"), "trajectory_observability")
    fixes = _require(fix_quality, "survey_gnss.fix_quality_weights",
                     ("status", "codes", "horizontal_std_m", "all_rows_resolved",
                      "unusable_indices"), "fix_quality")
    vertical = _require(vertical_reference, "survey_gnss.vertical_reference_check",
                        ("status", "requires_confirmation", "detected_signals"),
                        "vertical_reference")
    bounds = _require(clock_bounds, "survey_gnss.time_offset_bounds",
                      ("status", "offset_bounds_s", "worst_case_along_track_error_m"),
                      "clock_bounds")
    support = (observed["metric_scale_support"]["status"]
               if observed["status"] == "evaluated" else "unknown")
    reasons, status = [], "unknown"
    if support == "unknown":
        status = "not_supported"
        reasons.append("the trajectory could not be assessed at all (too few cameras matched to "
                       "telemetry samples)")
    elif quality["displacement_below_noise"]:
        status = "not_supported"
        reasons.append("per-fix displacement is smaller than the reported position uncertainty, so "
                       "this trace cannot resolve its own trajectory")
    elif support == "not_supported":
        status = "not_supported"
        reasons.append("reported uncertainty is not small against the trajectory spread")
    elif support == "weak":
        status = "weak"
        reasons.append("only a few times better than the noise, so scale is noise-dominated")
    else:
        status = "supported"
        reasons.append(f"GPS fixes are {observed['metric_scale_support']['spread_over_sigma']:.1f}x "
                       "tighter than the trajectory spread")
    if quality["suspicious_count"]:
        status = "weak" if status == "supported" else status
        reasons.append(f"{quality['suspicious_count']} fix interval(s) imply impossible speeds: "
                       "those fixes are noise, and scale read from them is not trustworthy")
    if quality["gaps_over_threshold_count"]:
        reasons.append(f"{quality['gaps_over_threshold_count']} telemetry gap(s) mean cameras "
                       "inside them have no position constraint at all")
    if fixes is None:
        reasons.append("fix quality was never assessed, so 'GPS' here means an unknown receiver "
                       "solution class")
    elif fixes["status"] == "refused_no_quality_field":
        status = "weak" if status == "supported" else status
        reasons.append("this log carries no quality field: the solution class is undeclared and "
                       "any precision attached to it would be invented")
    elif fixes["status"] == "partially_unresolved" and status == "supported":
        status = "weak"
        reasons.append("some fixes have unrecognised quality codes")
    elif fixes["unusable_indices"] and status == "supported":
        status = "weak"
        reasons.append(f"{len(fixes['unusable_indices'])} fix(es) carry no position at all")
    separable = bool(observed["clock_offset_and_lever_arm_separable"])
    has_fixed_rtk = bool(fixes is not None and fixes["status"] == "mapped"
                         and all(code == "rtk_fixed" for code in fixes["codes"]))
    needs = {"known_baseline": status in ("weak", "not_supported"),
             "barometric_trend": vertical is None or bool(vertical["requires_confirmation"]),
             "imu_attitude": not separable,
             "rtk_fix": status != "supported" or (fixes is not None
                                                 and fixes["status"] != "mapped")}
    options = [
        {"id": "known_baseline",
         "label": "one measured baseline: two surveyed points a known distance apart",
         "addresses": ["metric scale of the reconstructed model"],
         "does_not_address": ["absolute placement on Earth", "the vertical datum",
                              "clock offset"],
         "status": "recommended" if needs["known_baseline"] else "not_indicated",
         "rationale": "a single known distance pins scale without becoming a control network",
         "cost": "one tape or total-station shot, reusable across scenes"},
        {"id": "barometric_trend",
         "label": "barometric altitude trend as an independent vertical cross-check",
         "addresses": ["vertical drift and gross height error", "the datum mismatch signal"],
         "does_not_address": ["the absolute vertical datum: a barometer is height above a "
                              "pressure reference, not above the ellipsoid"],
         "status": "recommended" if needs["barometric_trend"] else "not_indicated",
         "rationale": "cheap and usually already logged; it can disagree with GPS and reveal which "
                      "column is which, but it cannot name a datum",
         "cost": "read an existing log column"},
        {"id": "imu_attitude",
         "label": "IMU/attitude or yaw prior, or simply fly a non-straight leg",
         "addresses": ["separating clock offset from the antenna lever arm",
                       "rotation about the trajectory axis"],
         "does_not_address": ["metric scale", "the vertical datum"],
         "status": "recommended" if needs["imu_attitude"] else "not_indicated",
         "rationale": "on a straight constant-speed line these parameters are mathematically "
                      "aliased, so no fitting on this flight can separate them",
         "cost": "an optional sensor, or a change to the flight path"},
        {"id": "rtk_fix",
         "label": "RTK/PPK fixed-ambiguity solution (or at least declared fix quality plus DOP)",
         "addresses": ["position noise", "scale reliability", "most of the placement bias"],
         "does_not_address": ["camera timing", "the vertical datum without a geoid model",
                              "completeness"],
         "status": ("declare_or_obtain" if fixes is None or fixes["status"] != "mapped"
                    else "not_indicated" if has_fixed_rtk else "recommended"),
         "rationale": "the strongest single upgrade available, and the only one attacking noise "
                      "and bias together",
         "cost": "optional hardware and base corrections; the mandatory-input-only track must "
                 "still be reported separately"}]
    questions = ["Which receiver and solution class produced these positions, and is the quality "
                 "field in this log a real estimate or a placeholder?"]
    if vertical is None or vertical["requires_confirmation"]:
        questions.append("What is the altitude column a height above - ellipsoid, geoid, ground or "
                         "the boot point? Confirm it from the logger's own definition, not from a "
                         "plausible-looking value.")
    if not separable:
        questions.append("Do clock offset and antenna lever arm need to be separate? This geometry "
                         "cannot give them: accept one aliased sum, add attitude, or fly a leg "
                         "that turns.")
    questions.append("Were the surveyed checkpoints kept out of reconstruction, keyframe selection "
                     "and alignment? Nothing in this computation can check that.")
    return {"schema_version": 1, "provenance": "survey_accuracy.gcp_requirement",
            "kind": "decision_support", "claims_accuracy": False, "measured": False,
            "inputs_supplied": {"telemetry_quality": True, "trajectory_observability": True,
                                "fix_quality": fixes is not None,
                                "vertical_reference": vertical is not None,
                                "clock_bounds": bounds is not None},
            "scale_identifiable_from_gps_alone": {
                "status": status, "reasons": reasons,
                "basis": {"median_step_m": quality["median_step_m"],
                          "median_horizontal_std_m": quality["median_horizontal_std_m"],
                          "trajectory_spread_m": observed["metric_scale_support"].get(
                              "trajectory_spread_m"),
                          "geometry": observed["geometry"],
                          "suspicious_fix_intervals": quality["suspicious_count"]}},
            "absolute_placement_identifiable_from_gps_alone": {
                "status": "conditional" if status == "supported" else "not_supported",
                "reasons": ["GNSS placement is only as good as the receiver's real error, and "
                            "correlated multipath or ephemeris bias does not show up in the "
                            "reported sigma",
                            "an independent surveyed check is what turns 'conditional' into a "
                            "measured number"]},
            "clock_offset_and_lever_arm": {
                "separable": separable,
                "bound_s": None if bounds is None else bounds["offset_bounds_s"],
                "worst_case_error_m": None if bounds is None
                else bounds["worst_case_along_track_error_m"],
                "reasons": ([] if separable else
                            ["straight and/or constant-speed geometry aliases clock offset with the "
                             "along-track lever arm; refusing to call either one calibrated"])},
            "fix_quality": {"status": "not_supplied" if fixes is None else fixes["status"],
                            "unusable_indices": [] if fixes is None else fixes["unusable_indices"]},
            "vertical_reference": {
                "status": "not_supplied" if vertical is None else vertical["status"],
                "requires_confirmation": None if vertical is None
                else bool(vertical["requires_confirmation"])},
            "constraint_options": options,
            "recommended_minimum": {
                "constraint_ids": [option["id"] for option in options
                                   if option["status"] in ("recommended", "declare_or_obtain")],
                "rationale": ["each option states what it fixes and what it leaves alone"]
                if any(option["status"] in ("recommended", "declare_or_obtain")
                       for option in options) else
                ["no additional constraint is indicated by the supplied inputs, which is not proof "
                 "that none is needed: run the fix-quality and vertical-datum checks to make this "
                 "answer complete"]},
            "ground_control_points": {
                "for_reconstruction": "not assumed by this protocol",
                "for_evaluation": {"required": True, "minimum_declared": int(min_evaluation_points),
                                   "kind": "independent surveyed checkpoints, never used for "
                                           "fitting"},
                "count_claimed": None,
                "note": "the brief's question is about control points as reconstruction INPUTS; "
                        "evaluation checkpoints are withheld from the fit and constrain nothing"},
            "requires_confirmation": questions,
            "warnings": ["decision support for planning, derived from telemetry quality and "
                         "trajectory geometry: not a measurement, and it proves nothing about "
                         "accuracy",
                         "thresholds are declared engineering choices shared with survey_gnss, "
                         "not official requirements"]}
