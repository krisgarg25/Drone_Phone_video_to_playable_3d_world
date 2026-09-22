"""Stage-time prediction: can this pass finish before the deadline, honestly?

Challenge (vi) is near-real-time processing. Nothing in this module has been
measured end to end - the longest recording on this machine is 151 s, so no
600-second single-pass run exists to time. What *has* been measured is per-image
cost per stage (report section 9.1: 291 keyframes at 1000 px on an RTX 3050 6 GB
laptop, driver 616.56). This module multiplies those rates by a frame budget and
states the result as a prediction, including the honest negative answers: the
measured dense stage alone needs ~2952 s at 291 frames, so the <900 s gate is
unreachable at that budget on this hardware.

Design rules that keep the model honest:

* Rates are supplied by the caller as data (``MEASURED_RATES`` below is that data,
  with its provenance attached), never read from a constant inside a function.
  When the rates are re-measured on other hardware the model has to be re-fed, so
  the number changes instead of lying.
* Resolution scaling happens only when a rate declares the ``reference_px`` it was
  measured at, plus a caller-declared ``pixel_power``. Undeclared means unscaled.
* Every answer carries ``PREDICTED_LABEL``; ``NOT_MEASURED_LABEL`` travels with any
  assembled payload, so a prediction cannot be rendered as a measurement.

Only stdlib arithmetic: no torch, no GPU, no subprocess, no filesystem.
"""
from dataclasses import dataclass
import math

# --- explicit thresholds and defaults -----------------------------------------
DEFAULT_DEADLINE_S = 900.0
"""The official gate: < 15 minutes of processing, strictly less than."""

DEFAULT_VIDEO_DURATION_S = 600.0
"""The official input: a 10-minute single-pass video."""

DEFAULT_RATIO_TARGET = 1.5
"""Processing-time / input-duration ratio the speed protocol asks to stay under."""

DEFAULT_MATCH_NEIGHBORS = 20
"""Frames each image is matched against by the sequential matcher - this repo's
own COLMAP plan value ("overlap": 20 in scripts/run_colmap.py), so the pair count
in the model is the pair count the measured run actually used."""

DEFAULT_WINDOWS = 4
"""Submaps in a progressive plan: enough to show visible progress, few enough that
the duplicated overlap work stays small."""

MIN_OVERLAP_FRAMES = 5
"""Floor on shared frames between consecutive submaps. Two views are the bare
minimum to triangulate and a handful more is where a similarity fit stops being
under-determined, so five is the smallest defensible shared set."""

MIN_OVERLAP_FRACTION = 0.25
"""A submap must also share at least this fraction of the *new* frames the window
adds, otherwise the shared set is too small a sample of the new geometry to lock
onto."""

MIN_WINDOW_FRAMES = 8
"""A window smaller than this is not a submap: too few cameras to register a
trajectory of its own. Asking for more windows than the frame budget supports is
refused rather than silently producing fragments."""

PREDICTED_LABEL = "predicted from measured per-image rates"
NOT_MEASURED_LABEL = ("not measured end-to-end; no 600 s single-pass video exists "
                      "on this machine")

ROUND_DIGITS = 6

_RATE_MODELS = ("per_image", "pairs", "fixed")
_PAIR_STRATEGIES = ("sequential", "exhaustive")
_PRODUCTS = ("coarse", "final")
_SPEC_KEYS = ("model", "s_per_image", "s_per_pair", "fixed_s", "reference_px",
              "pixel_power", "pair_strategy", "product")

# The measured baseline, as data. Report section 9.1, clip room_w_jsonl:
# 291 keyframes at 1000 px - keyframes 21.5 s, sparse 946.9 s, undistort 16.3 s,
# dense 2951.8 s (patchmatch with geometric consistency), fusion 64.9 s, total
# 4001.4 s. "coarse" marks the stages that already yield a measurable (registered,
# georeferenceable) product; everything else belongs to the final dense product.
MEASURED_RATES = {
    "keyframes": {"model": "per_image", "s_per_image": 21.5 / 291,
                  "reference_px": 1000, "pixel_power": 0.0, "product": "coarse"},
    "sparse": {"model": "per_image", "s_per_image": 946.9 / 291,
               "reference_px": 1000, "product": "coarse"},
    "undistort": {"model": "per_image", "s_per_image": 16.3 / 291,
                  "reference_px": 1000},
    "dense": {"model": "per_image", "s_per_image": 2951.8 / 291,
              "reference_px": 1000},
    "fusion": {"model": "per_image", "s_per_image": 64.9 / 291,
               "reference_px": 1000},
}
MEASURED_PROVENANCE = {"clip": "room_w_jsonl", "keyframes": 291, "image_px": 1000,
                       "hardware": "RTX 3050 6 GB Laptop GPU, driver 616.56",
                       "source": "report section 9.1, measured 23 September 2026",
                       "measured_total_s": 4001.4}

# Sequential-only matching, measured separately at the same budget: 239.7 s for
# 291 frames at 20 neighbours (5820 pairs), registering 233 of 288 cameras. The
# per-pair rate is that division, so the model reproduces the measurement instead
# of approximating it.
MEASURED_SEQUENTIAL_MATCHING = {"seconds": 239.7, "frames": 291,
                                "neighbors": DEFAULT_MATCH_NEIGHBORS,
                                "pairs": 291 * DEFAULT_MATCH_NEIGHBORS,
                                "s_per_pair": 239.7 / (291 * DEFAULT_MATCH_NEIGHBORS),
                                "registered_cameras": 233, "total_cameras": 288}


# --- validation ---------------------------------------------------------------
def _number(value, name, *, positive=True):
    if isinstance(value, (bool, str)) or value is None:
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and "
                         f"{'positive' if positive else 'non-negative'}")
    return result


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, float, str)) or value is None or not hasattr(value, "__index__"):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    result = int(value.__index__())
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _round(value):
    return round(float(value), ROUND_DIGITS)


def _validate_spec(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"rate {name!r} must be a dict of rate fields")
    unknown = sorted(set(spec) - set(_SPEC_KEYS))
    if unknown:
        raise ValueError(f"rate {name!r} has unknown fields {unknown}")
    model = spec.get("model")
    if model not in _RATE_MODELS:
        raise ValueError(f"rate {name!r} needs model in {_RATE_MODELS}, got {model!r}")
    product = spec.get("product", "final")
    if product not in _PRODUCTS:
        raise ValueError(f"rate {name!r} product must be one of {_PRODUCTS}")
    required = {"per_image": "s_per_image", "pairs": "s_per_pair",
                "fixed": "fixed_s"}[model]
    if required not in spec:
        raise ValueError(f"rate {name!r} with model {model!r} needs {required}")
    for other in ("s_per_image", "s_per_pair", "fixed_s"):
        if other != required and other in spec:
            raise ValueError(f"rate {name!r} with model {model!r} cannot carry {other}")
    rate = _number(spec[required], f"{name}.{required}", positive=False)
    if "reference_px" in spec:
        reference = _number(spec["reference_px"], f"{name}.reference_px")
        power = _number(spec.get("pixel_power", 1.0), f"{name}.pixel_power", positive=False)
    elif "pixel_power" in spec:
        raise ValueError(f"rate {name!r} sets pixel_power without reference_px "
                         "(there is nothing to scale from)")
    else:
        reference, power = None, 1.0
    if model == "pairs":
        strategy = spec.get("pair_strategy", "sequential")
        if strategy not in _PAIR_STRATEGIES:
            raise ValueError(f"rate {name!r} pair_strategy must be one of "
                             f"{_PAIR_STRATEGIES}")
    elif "pair_strategy" in spec:
        raise ValueError(f"rate {name!r} pair_strategy only applies to model 'pairs'")
    else:
        strategy = None
    return {"model": model, "rate": rate, "reference_px": reference,
            "pixel_power": power, "pair_strategy": strategy, "product": product}


def _validate_rates(rates):
    if not isinstance(rates, dict) or not rates:
        raise ValueError("rates must be a non-empty dict of stage name -> rate spec")
    return {name: _validate_spec(name, spec) for name, spec in rates.items()}


def _options(values, name):
    if isinstance(values, (int, float, str)) or not hasattr(values, "__iter__"):
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    picked = tuple(values)
    if not picked:
        raise ValueError(f"{name} must not be empty")
    return tuple(_integer(value, f"{name} entry") for value in picked)


# --- the budget ---------------------------------------------------------------
@dataclass(frozen=True)
class StageBudget:
    """Predicted wall time for one configuration. Never a measured run.

    Built by ``plan_stages``; the fields are the answer, the label says what kind
    of answer it is.
    """

    frame_count: int
    image_px: int
    stages: dict
    bases: dict
    total_s: float
    coarse_s: float
    deadline_s: float
    video_duration_s: float
    label: str = PREDICTED_LABEL

    @property
    def fits_deadline(self):
        return self.total_s < self.deadline_s

    @property
    def overage_s(self):
        return _round(max(0.0, self.total_s - self.deadline_s))

    @property
    def headroom_s(self):
        return _round(max(0.0, self.deadline_s - self.total_s))

    @property
    def final_s(self):
        return self.total_s

    @property
    def processing_ratio(self):
        return _round(self.total_s / self.video_duration_s)

    @property
    def meets_ratio_target(self):
        return self.total_s / self.video_duration_s < DEFAULT_RATIO_TARGET

    @property
    def dominant_stage(self):
        return max(self.stages, key=lambda name: self.stages[name])

    @property
    def dominant_stage_fraction(self):
        return _round(self.stages[self.dominant_stage] / self.total_s)

    def to_dict(self):
        return {
            "label": self.label,
            "frame_count": self.frame_count,
            "image_px": self.image_px,
            "deadline_s": _round(self.deadline_s),
            "video_duration_s": _round(self.video_duration_s),
            "stages": {name: {"seconds": seconds, "status": PREDICTED_LABEL,
                              "model": self.bases[name]["model"]}
                       for name, seconds in self.stages.items()},
            "bases": self.bases,
            "total_s": self.total_s,
            "coarse_s": self.coarse_s,
            "final_s": self.final_s,
            "fits_deadline": self.fits_deadline,
            "overage_s": self.overage_s,
            "headroom_s": self.headroom_s,
            "processing_ratio": self.processing_ratio,
            "meets_ratio_target": self.meets_ratio_target,
            "ratio_target": DEFAULT_RATIO_TARGET,
            "dominant_stage": self.dominant_stage,
            "dominant_stage_fraction": self.dominant_stage_fraction,
        }


def _pair_count(strategy, frames, neighbors):
    """How many image pairs the matcher actually visits."""
    if strategy == "exhaustive":
        return frames * (frames - 1) // 2
    return frames * min(neighbors, frames - 1)


def plan_stages(frame_count, image_px, *, rates, deadline_s=DEFAULT_DEADLINE_S,
                video_duration_s=DEFAULT_VIDEO_DURATION_S,
                match_neighbors=DEFAULT_MATCH_NEIGHBORS) -> StageBudget:
    """Predict per-stage and total wall seconds for one configuration.

    ``rates`` maps a stage name to a spec dict, and the caller owns those numbers
    so a re-measurement changes the answer instead of silently disagreeing with a
    constant. Recognised fields, all explicit:

    * ``model``: "per_image" (needs ``s_per_image``), "pairs" (needs ``s_per_pair``;
      the pair count is ``frames * min(match_neighbors, frames - 1)`` for
      "sequential" and ``frames * (frames - 1) / 2`` for "exhaustive"), or "fixed"
      (needs ``fixed_s``: cold starts, model loads, anything per run not per image).
    * ``reference_px``: the resolution the rate was measured at. Declaring it turns
      on scaling by ``(image_px / reference_px) ** pixel_power``, with ``pixel_power``
      defaulting to 1.0 because per-image cost grows with pixel area. Omitting it
      means "do not scale this stage".
    * ``product``: "coarse" marks the last stage that already yields a measurable
      product; anything else belongs to the final product.

    ``coarse_s`` is the running total through the last coarse stage, or None when
    no stage declares itself coarse - the model refuses to invent an early product.
    Every returned second is a prediction from per-image rates.
    """
    frames = _integer(frame_count, "frame_count")
    pixels = _integer(image_px, "image_px")
    limit = _number(deadline_s, "deadline_s")
    duration = _number(video_duration_s, "video_duration_s")
    neighbors = _integer(match_neighbors, "match_neighbors")
    specs = _validate_rates(rates)

    stages, bases, running, coarse = {}, {}, 0.0, None
    for name, spec in specs.items():
        factor = 1.0 if spec["reference_px"] is None else (
            pixels / spec["reference_px"]) ** spec["pixel_power"]
        if spec["model"] == "per_image":
            seconds = frames * spec["rate"] * factor
            basis = {"model": "per_image", "s_per_image": _round(spec["rate"]),
                     "images": frames, "resolution_factor": _round(factor),
                     "reference_px": spec["reference_px"],
                     "pixel_power": spec["pixel_power"] if spec["reference_px"] else None}
        elif spec["model"] == "pairs":
            pairs = _pair_count(spec["pair_strategy"], frames, neighbors)
            seconds = pairs * spec["rate"] * factor
            basis = {"model": "pairs", "pair_strategy": spec["pair_strategy"],
                     "s_per_pair": _round(spec["rate"]), "pairs": pairs,
                     "neighbors": neighbors, "resolution_factor": _round(factor),
                     "reference_px": spec["reference_px"],
                     "pixel_power": spec["pixel_power"] if spec["reference_px"] else None}
        else:
            seconds = spec["rate"]
            basis = {"model": "fixed", "fixed_s": _round(spec["rate"])}
        running += seconds
        stages[name] = _round(seconds)
        bases[name] = basis
        if spec["product"] == "coarse":
            coarse = running
    total = sum(stages.values())
    return StageBudget(frame_count=frames, image_px=pixels, stages=stages, bases=bases,
                       total_s=_round(total),
                       coarse_s=None if coarse is None else _round(coarse),
                       deadline_s=limit, video_duration_s=duration)


# --- configuration search ----------------------------------------------------
def search_budget(rates, video_duration_s, deadline_s, *, frame_options,
                  resolution_options, max_frame_rate=None,
                  match_neighbors=DEFAULT_MATCH_NEIGHBORS) -> dict:
    """Find the cheapest configuration that meets the deadline, or say it cannot.

    "Cheapest" is the lowest predicted wall time. ``best_quality_within_deadline``
    is reported next to it because the fastest option is rarely the one you want:
    it keeps the most frames that still fit. ``config`` is None, with a reason,
    when nothing fits - "the target is unreachable on this hardware" is a result,
    not a failure of the search.

    ``max_frame_rate`` (frames per source second) models the frame supply: an
    option needing more frames than the clip can yield at that rate is dropped
    rather than predicted, and counted in
    ``candidates_rejected_for_frame_supply``.
    """
    duration = _number(video_duration_s, "video_duration_s")
    limit = _number(deadline_s, "deadline_s")
    _validate_rates(rates)
    frames = _options(frame_options, "frame_options")
    pixels = _options(resolution_options, "resolution_options")
    supply = None if max_frame_rate is None else _number(max_frame_rate, "max_frame_rate") * duration

    feasible, infeasible, rejected_for_supply = [], [], 0
    fastest = None
    for size in pixels:
        for count in frames:
            if supply is not None and count > supply:
                rejected_for_supply += 1
                continue
            budget = plan_stages(count, size, rates=rates, deadline_s=limit,
                                 video_duration_s=duration, match_neighbors=match_neighbors)
            entry = {"frame_count": count, "image_px": size,
                     "predicted_s": budget.total_s, "fits_deadline": budget.fits_deadline,
                     "processing_ratio": budget.processing_ratio}
            if fastest is None or entry["predicted_s"] < fastest["predicted_s"]:
                fastest = entry
            (feasible if budget.fits_deadline else infeasible).append(entry)
    rank = lambda entry: (entry["predicted_s"], -entry["frame_count"], entry["image_px"])
    feasible.sort(key=rank)
    infeasible.sort(key=rank)

    if feasible:
        best = feasible[0]
        richest = max(feasible, key=lambda e: (e["frame_count"], e["image_px"],
                                               -e["predicted_s"]))
        config = {"frame_count": best["frame_count"], "image_px": best["image_px"]}
        reason = (f"Cheapest configuration meeting {limit} s: {config['frame_count']} frames "
                  f"at {config['image_px']} px, {best['predicted_s']} s "
                  f"({PREDICTED_LABEL}). {len(feasible)} of "
                  f"{len(feasible) + len(infeasible)} candidates fit, {len(infeasible)} did not.")
    else:
        config, reason = None, _unreachable(limit, frames, pixels, fastest,
                                            rejected_for_supply, duration, supply)
    return {
        "status": "feasible" if feasible else "infeasible",
        "config": config,
        "predicted_s": None if not feasible else feasible[0]["predicted_s"],
        "budget": None if not feasible else plan_stages(
            config["frame_count"], config["image_px"], rates=rates, deadline_s=limit,
            video_duration_s=duration, match_neighbors=match_neighbors).to_dict(),
        "best_quality_within_deadline": None if not feasible else {
            "frame_count": richest["frame_count"], "image_px": richest["image_px"]},
        "best_quality_predicted_s": None if not feasible else richest["predicted_s"],
        "feasible": feasible,
        "infeasible": infeasible,
        "candidates_evaluated": len(feasible) + len(infeasible),
        "candidates_rejected_for_frame_supply": rejected_for_supply,
        "fastest_predicted_s": None if fastest is None else fastest["predicted_s"],
        "fastest_candidate": fastest,
        "deadline_s": _round(limit),
        "video_duration_s": _round(duration),
        "ratio_target": DEFAULT_RATIO_TARGET,
        "max_frames_available": None if supply is None else int(math.floor(supply)),
        "reason": reason,
        "label": PREDICTED_LABEL,
    }


def _unreachable(limit, frames, pixels, fastest, rejected, duration, supply):
    if fastest is None:
        return (f"Every frame option exceeds the frame supply: {duration} s of video at the "
                f"declared rate yields at most {int(math.floor(supply))} frames and the "
                f"smallest option asked for {min(frames)}. Nothing is predicted rather "
                f"than inventing a configuration the clip cannot provide, so the target "
                f"is unreachable for these options.")
    return (f"No configuration among the {len(frames) * len(pixels)} candidates predicts a "
            f"finish inside the {limit} s deadline. The fastest was {fastest['predicted_s']} s "
            f"at {fastest['frame_count']} frames and {fastest['image_px']} px, which is "
            f"{fastest['predicted_s'] / limit:.1f}x the deadline. The target is unreachable "
            f"on this hardware at these measured per-image rates: the frame budget or the "
            f"working resolution has to drop, or the dense stage has to get cheaper.")


# --- progressive output ------------------------------------------------------
def progressive_plan(frame_count, windows=DEFAULT_WINDOWS, *, overlap_frames=None,
                     min_overlap_frames=MIN_OVERLAP_FRAMES,
                     min_overlap_fraction=MIN_OVERLAP_FRACTION,
                     min_window_frames=MIN_WINDOW_FRAMES, rates=None, image_px=None) -> dict:
    """Ordered submap windows, with the overlap a merge needs to register them.

    Window ``i`` covers frames ``[i * stride, i * stride + window_frames)`` clipped
    to the clip, and consecutive windows share ``overlap_frames`` frames. That
    shared set is what lets submap ``i`` be registered onto the accumulated model
    instead of floating free: at least ``min_overlap_frames`` cameras in common and
    at least ``min_overlap_fraction`` of the new frames the window contributes.

    Duplicated frames are duplicated *cost*, so ``duplicate_frames`` and
    ``duplicate_fraction`` come out with the plan: progressive output is not free,
    it buys an early product with re-work. With ``rates`` and ``image_px`` supplied
    together each window also carries predicted seconds and a running total.
    """
    frames = _integer(frame_count, "frame_count")
    count = _integer(windows, "windows", minimum=2)
    floor = _integer(min_overlap_frames, "min_overlap_frames")
    fraction = _number(min_overlap_fraction, "min_overlap_fraction")
    smallest = _integer(min_window_frames, "min_window_frames", minimum=2)
    if (rates is None) != (image_px is None):
        raise ValueError("rates and image_px must be supplied together")
    if rates is not None:
        _validate_rates(rates)
        pixels = _integer(image_px, "image_px")

    stride = math.ceil(frames / count)
    if (count - 1) * stride >= frames:
        raise ValueError(f"windows={count} leaves an empty window at {frames} frames "
                         f"(stride {stride}); ask for fewer windows")
    overlap = (max(floor, math.ceil(fraction * stride)) if overlap_frames is None
               else _integer(overlap_frames, "overlap_frames"))
    window_frames = stride + overlap
    if window_frames > frames:
        raise ValueError(f"overlap_frames={overlap} makes windows of {window_frames} "
                         f"frames, more than the {frames} frames available")
    if window_frames < smallest:
        raise ValueError(f"windows of {window_frames} frames are below min_window_frames="
                         f"{smallest}: too few cameras to register a submap")

    rows, spent = [], 0.0
    for index in range(count):
        start = index * stride
        end = min(frames, start + window_frames)
        predicted = cumulative = None
        if rates is not None:
            predicted = plan_stages(end - start, pixels, rates=rates).total_s
            spent += predicted
            cumulative = _round(spent)
        rows.append({"index": index, "start": start, "end": end,
                     "frame_count": end - start,
                     "overlap_with_previous": 0 if index == 0 else overlap,
                     "predicted_s": predicted, "cumulative_s": cumulative})
    processed = sum(row["frame_count"] for row in rows)
    duplicated = processed - frames
    return {
        "label": PREDICTED_LABEL,
        "frame_count": frames,
        "windows_declared": count,
        "windows": rows,
        "window_frames": window_frames,
        "stride_frames": stride,
        "overlap_frames": overlap,
        "min_overlap_frames": floor,
        "min_overlap_fraction": fraction,
        "min_window_frames": smallest,
        "image_px": None if rates is None else pixels,
        "frames_processed": processed,
        "duplicate_frames": duplicated,
        "duplicate_fraction": _round(duplicated / processed) if processed else 0.0,
        "coverage_complete": bool(rows[-1]["end"] == frames),
        "total_predicted_s": None if rates is None else _round(spent),
    }


def first_useful_output_at(frame_count, image_px, *, rates, windows=DEFAULT_WINDOWS,
                           deadline_s=DEFAULT_DEADLINE_S,
                           video_duration_s=DEFAULT_VIDEO_DURATION_S) -> dict:
    """Time to a coarse measurable map, kept separate from time to the final one.

    ``coarse_ready_s`` is the whole scene through the last stage the caller
    declared product "coarse" - registered poses plus sparse geometry, measurable
    even though it is not a dense surface. ``first_window_coarse_ready_s`` is the
    same for the first submap of a progressive plan, which is the number that
    answers "is something usable on screen yet". ``final_ready_s`` is the full
    budget. Those are three different questions and are never collapsed into one.

    When no stage declares itself coarse, both coarse answers are None: the model
    will not claim an early product the caller cannot evidence.
    """
    limit = _number(deadline_s, "deadline_s")
    budget = plan_stages(frame_count, image_px, rates=rates, deadline_s=limit,
                         video_duration_s=video_duration_s)
    plan = progressive_plan(budget.frame_count, windows, rates=rates, image_px=image_px)
    first = plan["windows"][0]
    window_budget = plan_stages(first["frame_count"], budget.image_px, rates=rates,
                               deadline_s=limit, video_duration_s=budget.video_duration_s)
    coarse, window_coarse = budget.coarse_s, window_budget.coarse_s
    candidates = [value for value in (coarse, window_coarse) if value is not None]
    first_map = min(candidates) if candidates else None
    if first_map is None:
        statement = (f"No coarse-product stages were declared in these rates, so there is no "
                     f"early measurable map to promise; final product in {budget.total_s} s "
                     f"against a {limit} s deadline ({PREDICTED_LABEL}, {NOT_MEASURED_LABEL}).")
    else:
        statement = (f"First measurable submap in {window_coarse} s "
                     f"({first['frame_count']} of {budget.frame_count} frames); whole-scene "
                     f"coarse map in {coarse} s; final product in {budget.total_s} s against "
                     f"a {limit} s deadline ({PREDICTED_LABEL}, {NOT_MEASURED_LABEL}).")
    return {
        "label": PREDICTED_LABEL,
        "first_measurable_map_s": first_map,
        "coarse_ready_s": coarse,
        "final_ready_s": budget.total_s,
        "first_window_coarse_ready_s": window_coarse,
        "first_window_frames": first["frame_count"],
        "windows": plan["windows_declared"],
        "meets_deadline_first_map": bool(first_map is not None and first_map < limit),
        "meets_deadline_coarse": bool(coarse is not None and coarse < limit),
        "meets_deadline_final": budget.fits_deadline,
        "deadline_s": _round(limit),
        "statement": statement,
        "budget": budget.to_dict(),
        "progressive": plan,
    }


# --- report ------------------------------------------------------------------
def report(rates, frame_count, image_px, *, video_duration_s=DEFAULT_VIDEO_DURATION_S,
           deadline_s=DEFAULT_DEADLINE_S, windows=DEFAULT_WINDOWS, frame_options=None,
           resolution_options=None, match_neighbors=DEFAULT_MATCH_NEIGHBORS,
           hardware=MEASURED_PROVENANCE["hardware"]) -> dict:
    """One dictionary of predictions, with every number labelled as one.

    Seconds in here are ``PREDICTED_LABEL`` and the payload carries
    ``NOT_MEASURED_LABEL``: a caller that renders only ``budget["total_s"]`` is
    still showing a prediction, because the label travels inside the payload.
    """
    budget = plan_stages(frame_count, image_px, rates=rates, deadline_s=deadline_s,
                         video_duration_s=video_duration_s, match_neighbors=match_neighbors)
    output = first_useful_output_at(budget.frame_count, budget.image_px, rates=rates,
                                    windows=windows, deadline_s=deadline_s,
                                    video_duration_s=video_duration_s)
    search = None
    if frame_options is not None and resolution_options is not None:
        search = search_budget(rates, video_duration_s, deadline_s,
                               frame_options=frame_options,
                               resolution_options=resolution_options,
                               match_neighbors=match_neighbors)
    disclaimers = [
        f"Every second in this payload is {PREDICTED_LABEL}, and this specific run is "
        + NOT_MEASURED_LABEL + ".",
        f"The official gate is a {video_duration_s:.0f} s single-pass video processed in "
        f"under {deadline_s:.0f} s. No such recording exists on this machine, so that gate "
        f"is predicted here, not measured.",
        "Per-image rates are the only measured input, and they come from one clip at one "
        "working resolution; the pixel-area scaling is a caller-declared assumption, not a "
        "second measurement.",
        "Sustained throughput on a long run is typically worse than a short-run rate: "
        "thermal throttling, memory pressure, disk and matching-graph growth are not "
        "modelled.",
        f"Sequential-only matching registered "
        f"{MEASURED_SEQUENTIAL_MATCHING['registered_cameras']} of "
        f"{MEASURED_SEQUENTIAL_MATCHING['total_cameras']} cameras in the measured run; a "
        f"window that fails to register still spends its predicted seconds and yields no "
        f"product.",
        "Time to first submap assumes submaps can be merged; merge and georeferencing cost "
        "is not measured, so it is not counted here.",
    ]
    statement = (f"{budget.frame_count} frames at {budget.image_px} px: {budget.total_s} s "
                 f"predicted against a {deadline_s} s deadline "
                 f"({'met' if budget.fits_deadline else 'missed'}); dominant stage "
                 f"{budget.dominant_stage} at {budget.stages[budget.dominant_stage]} s "
                 f"({budget.dominant_stage_fraction:.1%} of the budget).")
    return {
        "status_label": PREDICTED_LABEL,
        "statement": statement,
        "deadline_s": _round(deadline_s),
        "video_duration_s": _round(video_duration_s),
        "hardware": hardware,
        "provenance": dict(MEASURED_PROVENANCE),
        "measured_sequential_matching": dict(MEASURED_SEQUENTIAL_MATCHING),
        "budget": budget.to_dict(),
        "dominant_stage": budget.dominant_stage,
        "dominant_stage_fraction": budget.dominant_stage_fraction,
        "first_output": {key: value for key, value in output.items()
                         if key not in ("budget", "progressive")},
        "progressive": output["progressive"],
        "search": search,
        "disclaimers": disclaimers,
    }
