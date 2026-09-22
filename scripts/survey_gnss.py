"""Honest quality, datum and observability checks for flight GNSS telemetry.

stdlib + NumPy only, CPU only, no I/O. Inputs are never modified and nothing here
is written back into the trace.

The four questions this module may answer are: (1) does this GPS trace even look
like the aircraft it claims to describe, (2) does the vertical column *look* like
the ellipsoidal height it was filed as, (3) how tightly can the two clocks be
bounded from timestamps alone, (4) is the trajectory geometry capable of separating
the parameters someone wants to calibrate. It cannot answer "is this accurate":
that needs an independent surveyed reference (see scripts/survey_accuracy.py), so
every result carries ``accuracy_validated=False``.

Conventions match scripts/survey_georef.py: ``position`` is local ENU metres in a
WGS84 (EPSG:4979) frame with ellipsoidal height, and ``horizontal_std_m`` is a
per-axis (east, north) standard deviation, not a 2-D radius.
"""
import copy
import math

import numpy as np

try:  # imported as scripts.survey_gnss by the tests
    from scripts import survey_georef as georef
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_georef as georef

# Owned by the georeferencing lane; read here so both lanes reject the same
# near-collinear geometry. A rename fails loudly at import instead of letting the
# two modules silently disagree about degeneracy.
MIN_SECOND_RATIO = georef._MIN_SECOND_RATIO

SAMPLE_FIELDS = ("t_sec", "position", "horizontal_std_m", "vertical_std_m")
SPEED_CV_THRESHOLD = 0.05        # below this the implied speed counts as constant
SCALE_STRONG_RATIO = 10.0        # trajectory spread / reported horizontal sigma
SCALE_MIN_RATIO = 3.0            # under this, metre scale is not transferable
VERTICAL_FLATNESS_RATIO = 1e-4   # vertical range vs horizontal extent
VERTICAL_RELIEF_FRACTION = 0.05  # vertical range vs the caller's expected relief
CONSTANT_OFFSET_TOLERANCE_M = 0.01

ELLIPSOIDAL_SLOTS = ("ellipsoidal", "wgs84_height", "gnss_height", "ecef", "itrf")
GROUND_RELATIVE_REFS = ("above_ground_level", "agl", "barometric", "relative_to_takeoff",
                        "height_above_takeoff", "terrain_clearance", "radio_altimeter")


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _text(value, name):
    if value is None:
        return "unknown"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string, or absent to mean 'unknown'")
    return value.strip().lower()


def _series(values, name):
    if isinstance(values, (str, bytes, dict)) or not isinstance(
            values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of numbers")
    result = np.array([_number(value, f"{name}[]") for value in values], dtype=np.float64)
    return result.reshape(len(result))


def _samples(samples):
    """Normalise the telemetry sample schema into (times, ENU, h_sigma, v_sigma)."""
    if isinstance(samples, (str, bytes, dict)) or not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list of telemetry sample dicts")
    times, positions, horizontal, vertical = [], [], [], []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != set(SAMPLE_FIELDS):
            raise ValueError(f"each sample requires exactly these fields: {list(SAMPLE_FIELDS)}")
        times.append(_number(sample["t_sec"], "t_sec"))
        positions.append(_series(sample["position"], "position"))
        horizontal.append(_number(sample["horizontal_std_m"], "horizontal_std_m", positive=True))
        vertical.append(_number(sample["vertical_std_m"], "vertical_std_m", positive=True))
    if any(point.size != 3 for point in positions):
        raise ValueError("position must hold three ENU metres")
    points = np.array(positions, dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample t_sec must be strictly increasing without duplicates")
    return np.array(times), points, np.array(horizontal), np.array(vertical)


def _stats(values):
    """(median, mean, max) of a 1-D array, or Nones for an empty one."""
    if values.size == 0:
        return None, None, None
    return float(np.median(values)), float(np.mean(values)), float(np.max(values))


def _intervals(times, points):
    deltas = np.diff(points, axis=0)
    dt = np.diff(times)
    steps = np.sqrt(np.sum(deltas * deltas, axis=1))
    return dt, steps, np.hypot(deltas[:, 0], deltas[:, 1]), steps / dt


def _warning(text, warnings):
    if text not in warnings:
        warnings.append(text)


def quality_report(samples, *, max_speed_m_s, gap_threshold_s=2.0):
    """Describe what a GPS trace implies, and name the fixes that imply nonsense.

    ``max_speed_m_s`` is the caller's physical limit for the aircraft and has no
    default on purpose: deciding what counts as impossible motion belongs to the
    caller, not to this module. ``gap_threshold_s`` defaults to 2.0 s because that is
    the bracket limit scripts/survey_georef.py will interpolate across, so a counted
    gap is a gap that really does lose cameras.

    Every ``*_index`` names the LATER of the two fixes of the interval it describes;
    the interval series (``step_m``, ``implied_speed_m_s``, ``time_gap_s``) have
    length ``count - 1`` with element j covering fixes j -> j + 1. Nothing is
    filtered, repaired or dropped: suspicious fixes are listed and left in place.
    """
    limit = _number(max_speed_m_s, "max_speed_m_s", positive=True)
    threshold = _number(gap_threshold_s, "gap_threshold_s", positive=True)
    times, points, horizontal, vertical = _samples(samples)
    dt, steps, hsteps, speeds = _intervals(times, points)
    median_step, mean_step, max_step = _stats(steps)
    median_speed, mean_speed, max_speed = _stats(speeds)
    median_gap, mean_gap, max_gap = _stats(dt)
    warnings = []
    if times.size == 1:
        _warning("Single fix: no displacement, speed or gap can be assessed. A lone position "
                 "constrains nothing and must not be used as a scale anchor.", warnings)
    over = (np.flatnonzero(dt > threshold) + 1).tolist()
    suspicious = (np.flatnonzero(speeds > limit) + 1).tolist()
    if suspicious:
        _warning(f"{len(suspicious)} of {speeds.size} consecutive intervals imply motion faster "
                 f"than the supplied {limit} m/s limit: those fixes are receiver noise or "
                 "dropouts, not flight. Nothing was removed from the trace.", warnings)
    if median_step is not None and median_step < float(np.median(horizontal)):
        _warning(f"Median inter-fix displacement {median_step} m is smaller than the reported "
                 f"horizontal uncertainty {float(np.median(horizontal))} m: per-fix motion sits "
                 "below the noise floor, so this telemetry cannot resolve the trajectory it "
                 "appears to describe.", warnings)
    if over:
        _warning(f"{len(over)} time gap(s) exceed {threshold} s; cameras inside them cannot be "
                 "bracketed and are dropped by the georeferencing lane.", warnings)
    return {
        "schema_version": 1,
        "provenance": "survey_gnss.quality_report",
        "count": int(times.size),
        "first_t_sec": float(times[0]), "last_t_sec": float(times[-1]),
        "duration_s": float(times[-1] - times[0]),
        "total_displacement_m": float(np.linalg.norm(points[-1] - points[0])),
        "path_length_m": float(steps.sum()),
        "median_step_m": median_step, "mean_step_m": mean_step, "max_step_m": max_step,
        "max_step_index": (int(np.argmax(steps)) + 1) if steps.size else None,
        "median_horizontal_step_m": _stats(hsteps)[0],
        "step_m": steps.tolist(), "horizontal_step_m": hsteps.tolist(),
        "implied_speed_m_s": speeds.tolist(),
        "median_implied_speed_m_s": median_speed, "mean_implied_speed_m_s": mean_speed,
        "max_implied_speed_m_s": max_speed,
        "max_speed_index": (int(np.argmax(speeds)) + 1) if speeds.size else None,
        "time_gap_s": dt.tolist(), "median_gap_s": median_gap, "mean_gap_s": mean_gap,
        "max_gap_s": max_gap, "gap_threshold_s": threshold,
        "gaps_over_threshold_count": len(over), "gaps_over_threshold_indices": over,
        "max_speed_m_s": limit, "suspicious": [int(index) for index in suspicious],
        "suspicious_count": len(suspicious),
        "suspicious_detail": [{"index": int(index), "interval": int(index) - 1,
                               "t_sec": float(times[index - 1]), "dt_s": float(dt[index - 1]),
                               "distance_m": float(steps[index - 1]),
                               "implied_speed_m_s": float(speeds[index - 1])}
                              for index in suspicious],
        "median_horizontal_std_m": float(np.median(horizontal)),
        "median_vertical_std_m": float(np.median(vertical)),
        "reported_std_over_median_step": (float(np.median(horizontal) / median_step)
                                         if median_step else None),
        "displacement_below_noise": bool(median_step is not None
                                         and median_step < float(np.median(horizontal))),
        "warnings": warnings,
        "speeds_are_implied_not_measured": True,
        "accuracy_validated": False,
    }


def _signal(detected, evidence, meaning):
    return {"detected": detected, "evidence": evidence, "meaning": meaning}


def vertical_reference_check(samples, declared):
    """Report the tell-tale signs of barometric/AGL height in an ellipsoidal slot.

    The truth is not obtainable from the trace, so this function never picks a datum,
    never edits an altitude and never says "confirmed". It lists named inconsistency
    signals with their numbers and demands confirmation; an undeclared source can only
    ever come back as ``insufficient_declarations``.

    ``declared`` keys: ``altitude_datum`` (required: the slot the values were filed
    into), ``height_reference``, ``source``, ``expected_relief_m``, ``terrain``,
    ``origin_altitude_m`` and ``alternative_altitudes_m`` (an optional second,
    same-length vertical series from another sensor or log column).
    """
    times, points, horizontal, vertical = _samples(samples)
    if not isinstance(declared, dict):
        raise ValueError("declared must be a dict of caller declarations")
    datum = declared.get("altitude_datum")
    if isinstance(datum, bool) or not isinstance(datum, str) or not datum.strip():
        raise ValueError("declared['altitude_datum'] must name the slot the heights went into")
    datum_key = datum.strip().lower()
    reference = _text(declared.get("height_reference"), "declared['height_reference']")
    source = _text(declared.get("source"), "declared['source']")
    _text(declared.get("terrain"), "declared['terrain']")
    relief = declared.get("expected_relief_m")
    relief = None if relief is None else _number(relief, "expected_relief_m", positive=True)
    heights = points[:, 2]
    vertical_range = float(heights.max() - heights.min())
    extent = float(np.hypot(*np.ptp(points[:, :2], axis=0)))
    signals = {}
    signals["declared_reference_mismatch"] = _signal(
        bool(datum_key in ELLIPSOIDAL_SLOTS
             and (reference in GROUND_RELATIVE_REFS or source in GROUND_RELATIVE_REFS)),
        {"altitude_datum": datum, "height_reference": reference, "source": source},
        "the declared source is ground-relative or barometric while the values were filed as an "
        "absolute ellipsoidal height; both cannot be true of the same column.")
    signals["vertical_series_invariant"] = _signal(
        bool(times.size >= 2 and vertical_range <= 1e-9),
        {"vertical_range_m": vertical_range, "first_heights_m": heights[:8].tolist()},
        "a height column that never moves is a held or hard-coded value, not a measured series.")
    trigger = (VERTICAL_RELIEF_FRACTION * relief if relief is not None
               else (VERTICAL_FLATNESS_RATIO * extent if extent > 100.0 else None))
    fired = bool(times.size >= 2 and trigger is not None and vertical_range < trigger)
    signals["vertical_relief_inconsistent_with_extent"] = _signal(
        fired,
        {"vertical_range_m": vertical_range, "horizontal_extent_m": extent,
         "expected_relief_m": relief, "trigger_m": trigger},
        "vertical variation is implausibly small for the distance flown, which is what a "
        "terrain-following height above ground looks like when it is read as ellipsoidal.")
    if "alternative_altitudes_m" in declared:
        second = _series(declared["alternative_altitudes_m"], "alternative_altitudes_m")
        if second.size != times.size:
            raise ValueError("alternative_altitudes_m must hold one value per sample")
        difference = heights - second
        spread = float(difference.max() - difference.min())
        signals["constant_offset_below_centimetre"] = _signal(
            bool(spread < CONSTANT_OFFSET_TOLERANCE_M),
            {"mean_difference_m": float(np.mean(difference)),
             "std_difference_m": float(np.std(difference)), "difference_spread_m": spread},
            "the two vertical sources differ by a constant below a centimetre, so one is the "
            "other plus a fixed datum offset: they are not independent evidence of either datum.")
    else:
        signals["constant_offset_below_centimetre"] = _signal(
            None, {"reason": "no second vertical source was supplied"},
            "not evaluated: supplying a second vertical series would make this computable.")
    static = {name: bool(values.size >= 3 and float(values.max() - values.min()) <= 1e-12)
              for name, values in (("horizontal_std_m", horizontal),
                                   ("vertical_std_m", vertical))}
    signals["static_reported_uncertainty"] = _signal(
        any(static.values()), static,
        "the reported uncertainties repeat to under a picometre: the receiver is echoing a "
        "placeholder, so those sigmas are not per-fix estimates.")
    detected = [name for name, value in signals.items() if value["detected"] is True]
    undeclared = [name for name, value in (("height_reference", reference), ("source", source))
                  if value == "unknown"]
    status = ("inconsistency_detected" if detected
              else "insufficient_declarations" if undeclared else "no_inconsistency_signal")
    actions = []
    if detected or undeclared:
        actions = [
            "Read the logger's own field definition and state in writing what the altitude "
            "column is a height above: ellipsoid, geoid, ground, or the boot point.",
            "Compare one static epoch against a known benchmark. A constant sub-centimetre gap to "
            "a barometric or AGL source is a datum offset, not agreement.",
            "If the column is barometric or AGL, convert it with a checked geoid/terrain model or "
            "stop reporting vertical accuracy; then re-declare metadata altitude_datum and re-run.",
            "Until somebody confirms it, treat every vertical number from this flight as "
            "unverified rather than merely noisy."]
    return {
        "schema_version": 1,
        "provenance": "survey_gnss.vertical_reference_check",
        "status": status,
        "declared": copy.deepcopy(declared),
        "signals": signals,
        "detected_signals": detected,
        "undeclared_fields": undeclared,
        "requires_confirmation": bool(detected or undeclared),
        "confirmation_actions": actions,
        "measurements": {"vertical_range_m": vertical_range, "horizontal_extent_m": extent,
                         "sample_count": int(times.size)},
        "vertical_datum_verified": False,
        "signals_absent_prove_consistency": False,
        "guessed": False,
        "accuracy_validated": False,
    }


def time_offset_bounds(camera_times, sample_times, *, max_speed_m_s):
    """Bound the camera/GNSS clock offset by what timestamps alone can prove.

    Convention: ``t_gnss = t_camera + offset_s``. Two time series establish exactly
    one thing - coverage. The georeferencing lane refuses to extrapolate, so an
    offset is admissible precisely when every camera frame still lands inside the
    GNSS span; that interval is returned as ``offset_bounds_s``. ``max_speed_m_s``
    converts the interval's width into the along-track position error that any two
    admissible offsets can hide, which is the honest metric consequence of the bound.

    No offset is estimated, fitted or chosen: ``offset_estimate_s`` is always None and
    a wide interval is the correct answer, not a failure to converge.
    """
    limit = _number(max_speed_m_s, "max_speed_m_s", positive=True)
    cameras, samples = _series(camera_times, "camera_times"), _series(sample_times, "sample_times")
    for name, values in (("camera_times", cameras), ("sample_times", samples)):
        if values.size == 0:
            raise ValueError(f"{name} must not be empty")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} must be strictly increasing without duplicates")
    low, high = float(samples[0] - cameras[0]), float(samples[-1] - cameras[-1])
    result = {"schema_version": 1, "provenance": "survey_gnss.time_offset_bounds",
              "convention": "t_gnss = t_camera + offset_s",
              "camera_count": int(cameras.size), "sample_count": int(samples.size),
              "camera_span_s": [float(cameras[0]), float(cameras[-1])],
              "sample_span_s": [float(samples[0]), float(samples[-1])],
              "max_speed_m_s": limit, "infeasibility_s": 0.0,
              "estimates_offset": False, "fitted": False, "offset_estimate_s": None,
              "accuracy_validated": False,
              "warnings": ["Coverage of the camera span by the GNSS span is the only thing these "
                           "two time series can bound; nothing here narrows the offset further "
                           "inside the interval."]}
    if low > high:
        result.update(status="infeasible", offset_bounds_s=None, offset_width_s=None,
                      worst_case_along_track_error_m=None,
                      infeasibility_s=float((cameras[-1] - cameras[0])
                                            - (samples[-1] - samples[0])))
        _warning(f"No offset can place every camera frame inside GNSS coverage: the camera series "
                 f"spans {result['infeasibility_s']} s more than the telemetry series. Do not shift "
                 "the clock to force coverage; extend the log or drop the uncovered frames.",
                 result["warnings"])
        return result
    result.update(status="bounded", offset_bounds_s=[low, high], offset_width_s=high - low,
                  worst_case_along_track_error_m=limit * (high - low),
                  seconds_per_metre_of_error=1.0 / limit)
    return result


# Conservative per-axis floors. NOT measured on this hardware or in this flight: they
# are deliberately inflated relative to vendor literature, and vertical is taken worse
# than horizontal because satellite geometry degrades height more than position.
FIX_QUALITY_TABLE = {
    "no_fix": {"label": "no fix / invalid solution", "horizontal_std_m": None,
               "vertical_std_m": None, "usable": False,
               "basis": "no position exists, so no standard deviation can be attached to one."},
    "single": {"label": "single / SPS / autonomous GNSS", "horizontal_std_m": 5.0,
               "vertical_std_m": 10.0, "usable": True,
               "basis": "order of magnitude for consumer GNSS under open sky, inflated from a "
                        "3-5 m figure, and assuming HDOP <= 1 geometry."},
    "dgps": {"label": "differential / SBAS corrected", "horizontal_std_m": 1.5,
             "vertical_std_m": 2.5, "usable": True,
             "basis": "corrections remove part of the common error, but decimetres to metres "
                      "remain normal near obstructions or on marginal corrections."},
    "pps": {"label": "PPS code solution", "horizontal_std_m": 1.0, "vertical_std_m": 2.0,
            "usable": True, "basis": "code only, no carrier phase: decimetre-to-metre class."},
    "rtk_float": {"label": "RTK/PPK float ambiguity", "horizontal_std_m": 0.5,
                  "vertical_std_m": 1.0, "usable": True,
                  "basis": "float fixes carry decimetre-to-metre ambiguity error; deliberately "
                           "far from the vendor centimetre claim."},
    "rtk_fixed": {"label": "RTK/PPK fixed ambiguity", "horizontal_std_m": 0.05,
                  "vertical_std_m": 0.10, "usable": True,
                  "basis": "vendor figures near 8 mm + 1 ppm are NOT adopted; 5 cm per axis is a "
                           "conservative floor and is unverified on this receiver."},
    "estimated": {"label": "dead reckoning / extrapolated", "horizontal_std_m": None,
                  "vertical_std_m": None, "usable": False,
                  "basis": "a model output, not a measurement; it cannot anchor metric scale."},
    "manual": {"label": "manual input", "horizontal_std_m": None, "vertical_std_m": None,
               "usable": False, "basis": "operator-entered positions are not GNSS observations."},
    "simulator": {"label": "simulator / test mode", "horizontal_std_m": None,
                  "vertical_std_m": None, "usable": False,
                  "basis": "synthetic positions must never enter a survey output."},
}
_CODE_ALIASES = {
    "no_fix": "no_fix", "nofix": "no_fix", "invalid": "no_fix", "none": "no_fix",
    "no_solution": "no_fix", "nosolution": "no_fix",
    "single": "single", "sps": "single", "autonomous": "single", "3d_fix": "single",
    "2d_fix": "single", "fix": "single", "gps": "single", "sps_fix": "single",
    "dgps": "dgps", "dgps_fix": "dgps", "sbas": "dgps", "ppp": "dgps",
    "pps": "pps", "pps_fix": "pps",
    "rtk_float": "rtk_float", "float": "rtk_float", "rtkfloat": "rtk_float",
    "rtk_fixed": "rtk_fixed", "fixed": "rtk_fixed", "rtk_fix": "rtk_fixed",
    "rtkfixed": "rtk_fixed",
    "estimated": "estimated", "dead_reckoning": "estimated", "dr": "estimated",
    "manual": "manual", "simulator": "simulator", "sim": "simulator",
}
_GGA_CODES = {0: "no_fix", 1: "single", 2: "dgps", 3: "pps", 4: "rtk_fixed", 5: "rtk_float",
              6: "estimated", 7: "manual", 8: "simulator"}


def _normalise_code(code):
    if code is None:
        return "unknown"
    if isinstance(code, bool):
        raise ValueError("quality codes must be fix-name strings or integer NMEA GGA codes")
    if isinstance(code, (int, np.integer)):
        return _GGA_CODES.get(int(code), "unknown")
    if not isinstance(code, str):
        raise ValueError("quality codes must be fix-name strings or integer NMEA GGA codes")
    text = code.strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return "unknown"
    if text.isdigit():
        return _GGA_CODES.get(int(text), "unknown")
    return _CODE_ALIASES.get(text, "unknown")


def fix_quality_weights(quality_codes, hdop=None, *, allow_unknown_fallback=False):
    """Map fix quality (and HDOP) to conservative per-axis standard deviations.

    Rows with no quality information come back as ``None`` rather than as a guessed
    sigma: inventing precision for an unknown receiver is exactly the failure this
    table exists to prevent. HDOP scaling is one-sided - poor geometry inflates the
    floor, good geometry never buys precision back below it. Passing
    ``allow_unknown_fallback=True`` substitutes the documented single/SPS floor and
    records that as a caller-declared assumption.
    """
    if isinstance(quality_codes, (str, bytes, dict)) or not isinstance(
            quality_codes, (list, tuple, np.ndarray)) or not len(quality_codes):
        raise ValueError("quality_codes must be a nonempty sequence")
    codes = [_normalise_code(code) for code in quality_codes]
    factors = [1.0] * len(codes)
    missing_dop = []
    if hdop is not None:
        if not isinstance(hdop, (list, tuple, np.ndarray)) or len(hdop) != len(codes):
            raise ValueError("hdop must be None or hold one value per quality code")
        for index, value in enumerate(hdop):
            if value is None:
                missing_dop.append(index)
                continue
            factors[index] = max(1.0, _number(value, "hdop[]", positive=True))
    fallback = [index for index, code in enumerate(codes) if code == "unknown"]
    if fallback and allow_unknown_fallback:
        codes = ["single" if index in fallback else code for index, code in enumerate(codes)]
    horizontal, vertical, unresolved, unusable = [], [], [], []
    for index, code in enumerate(codes):
        row = FIX_QUALITY_TABLE.get(code)
        if row is None:
            unresolved.append(index)
        elif not row["usable"]:
            unusable.append(index)
        else:
            horizontal.append(row["horizontal_std_m"] * factors[index])
            vertical.append(row["vertical_std_m"] * factors[index])
            continue
        horizontal.append(None)
        vertical.append(None)
    warnings = ["Table values are conservative engineering floors, NOT measured on this hardware; "
                "horizontal_std_m is per axis (east, north) as in scripts/survey_georef.py."]
    if hdop is None:
        _warning("No HDOP supplied: the floors assume HDOP <= 1 geometry, which is optimistic "
                 "rather than neutral. Declare DOP, or widen these numbers before weighting.",
                 warnings)
    elif missing_dop:
        _warning(f"{len(missing_dop)} row(s) report no HDOP and were left at the table floor "
                 "(factor 1.0) rather than being improved.", warnings)
    if any(factor > 1.0 for factor in factors):
        _warning("HDOP scaling was applied one-sided: sigma = floor * max(1, HDOP), so a low HDOP "
                 "can never manufacture precision the floor refuses to grant.", warnings)
    if fallback and not allow_unknown_fallback:
        _warning("No usable GNSS quality field for these rows: refusing to invent precision. "
                 "Supply fix codes or HDOP, or declare the positions as unqualified.", warnings)
    if fallback and allow_unknown_fallback:
        _warning(f"{len(fallback)} unknown row(s) were assigned the documented 'single' "
                 "conservative floor because the caller explicitly allowed the fallback; that is "
                 "a declared assumption, not a measurement.", warnings)
    if unusable:
        _warning(f"{len(unusable)} row(s) are no-fix/estimated/manual/simulated: they carry no "
                 "position and must be excluded from any fit or prior.", warnings)
    status = ("refused_no_quality_field" if not codes or all(value is None for value in horizontal)
              else "partially_unresolved" if unresolved else "mapped")
    return {"schema_version": 1, "provenance": "survey_gnss.fix_quality_weights",
            "status": status, "codes": codes,
            "quality_known": any(code != "unknown" for code in codes),
            "horizontal_std_m": horizontal, "vertical_std_m": vertical,
            "dop_factor": factors, "hdop_supplied": hdop is not None,
            "dop_missing_indices": missing_dop, "unresolved_indices": unresolved,
            "unusable_indices": unusable, "all_rows_resolved": not unresolved,
            "all_rows_usable": not unresolved and not unusable,
            "fallback": "single" if fallback and allow_unknown_fallback else None,
            "table": copy.deepcopy(FIX_QUALITY_TABLE),
            "dop_scaling": "one-sided: sigma = floor * max(1, HDOP), never below the floor",
            "measured_on_this_hardware": False, "precision_invented": False,
            "warnings": warnings, "accuracy_validated": False}


def _geometry(points):
    """Centred singular values plus the along/cross-track split of the trajectory."""
    centred = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    first = float(singular[0]) if singular.size else 0.0
    degenerate = first <= 1e-12

    def ratio(index):
        return 0.0 if degenerate or singular.size <= index else float(singular[index] / first)

    axis = vt[0]
    along = centred @ axis
    cross = np.linalg.norm(centred - np.outer(along, axis), axis=1)
    return {"first_singular_value_m": first,
            "second_to_first_ratio": ratio(1), "third_to_first_ratio": ratio(2),
            "singular_values_m": [float(value) for value in singular],
            "axial_length_m": 0.0 if degenerate else float(along.max() - along.min()),
            "max_cross_track_deviation_m": 0.0 if degenerate else float(cross.max()),
            "radius_of_gyration_m": float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))}


def observability(camera_positions_enu, samples):
    """Say whether this trajectory can separate clock offset from lever arm at all.

    On a straight, constant-speed line, shifting the clock by ``d`` moves every
    assigned position by ``v*d`` along the line - indistinguishable from a free
    translation or an along-track antenna offset - and the rotation about that line is
    not constrained by positions at all. ``camera_positions_enu`` is the Nx3 set of
    GNSS-assigned camera centres (the targets of
    ``scripts/survey_georef.camera_positions``); ``samples`` supplies the time
    parameterisation the speed test needs.

    Nothing is estimated here. The output is which claims this geometry can support
    and which it must never be quoted for, so an unobservable parameter can never be
    reported as calibrated.
    """
    raw = np.asarray(camera_positions_enu)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
        raise ValueError("camera_positions_enu must be a nonempty Nx3 array of ENU metres")
    points = np.array(raw, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("camera_positions_enu must contain only finite values")
    times, track, horizontal, _ = _samples(samples)
    geometry = _geometry(points)
    dt, _, _, speeds = _intervals(times, track)
    median_speed, _, _ = _stats(speeds)
    variation = (float(np.std(speeds) / median_speed) if speeds.size and median_speed else None)
    median_dt = _stats(dt)[0]
    noise_speed = (float(np.median(horizontal) * math.sqrt(2.0) / median_dt)
                   if median_dt else None)
    spread = geometry["radius_of_gyration_m"] * 2.0
    ratio = spread / float(np.median(horizontal))
    collinear = geometry["first_singular_value_m"] <= 1e-12 or (
        geometry["second_to_first_ratio"] < MIN_SECOND_RATIO)
    planar = not collinear and geometry["third_to_first_ratio"] < MIN_SECOND_RATIO
    constant_speed = bool(variation is not None and variation < SPEED_CV_THRESHOLD)
    supported = len(points) >= 4 and times.size >= 2
    scale_status = ("supported" if ratio >= SCALE_STRONG_RATIO else
                    "weak" if ratio >= SCALE_MIN_RATIO else
                    "not_supported") if supported else "unknown"
    separable = bool(supported and not (collinear and constant_speed))
    warnings, never_claim, statements = [], [], []
    if not supported:
        warnings.append(f"{len(points)} camera centre(s) and {times.size} telemetry sample(s) "
                        "cannot assess geometry or motion; the georeferencing lane needs at "
                        "least four matched cameras.")
        never_claim.append("That clock offset, antenna lever arm or trajectory geometry has been "
                           "calibrated: there is not enough data here to have observed anything.")
    else:
        if collinear:
            warnings.append(f"Trajectory is near-collinear (second/first centred singular value "
                            f"{geometry['second_to_first_ratio']:.3g} < {MIN_SECOND_RATIO}); "
                            "scripts/survey_georef.py will reject this geometry outright.")
        if collinear and constant_speed:
            never_claim.append("Clock offset calibrated from camera-to-GNSS timing: on a straight "
                               "constant-speed line an offset of d shifts every position by v*d, "
                               "which is exactly a free translation.")
            never_claim.append("Antenna lever arm (along-track) calibrated: only lever arm plus "
                               "speed times clock offset is observable, never either alone.")
        if collinear:
            never_claim.append("Cross-track or vertical antenna lever arm calibrated from "
                               "positions alone: rotation about the trajectory axis is "
                               "unconstrained, so those components are confounded with it.")
        if scale_status == "supported":
            statements.append(f"GPS fixes are {ratio:.0f}x tighter than the trajectory spread, so "
                              "metre scale is transferable from this telemetry IF the reported "
                              "uncertainties are real and the solution quality is as declared.")
        else:
            never_claim.append("Metric scale transferred from GPS on this flight: the reported "
                               f"uncertainty is only {ratio:.1f}x smaller than the trajectory "
                               "spread, so scale is dominated by noise.")
    if supported and noise_speed is not None and median_speed and noise_speed >= median_speed:
        warnings.append(f"GNSS noise alone implies about {noise_speed:.1f} m/s of motion, at or "
                        "above the median implied speed: apparent speed variation may be noise "
                        "rather than manoeuvres, and the constant-speed test is not decisive.")
    identifiability = {"clock_offset": separable,
                       "lever_arm_along_track": separable,
                       "lever_arm_cross_track": bool(supported and not collinear),
                       "rotation_about_trajectory_axis": bool(supported and not collinear),
                       "metric_scale": bool(scale_status == "supported")}
    return {"schema_version": 1, "provenance": "survey_gnss.observability",
            "status": "evaluated" if supported else "insufficient_data",
            "geometry": ("collinear" if collinear else "planar" if planar else "spatial")
            if supported else None,
            "geometry_stats": dict(geometry, camera_count=len(points), trajectory_spread_m=spread),
            "motion_stats": {"sample_count": int(times.size),
                             "implied_speed_m_s": speeds.tolist(),
                             "median_implied_speed_m_s": median_speed,
                             "median_time_gap_s": median_dt,
                             "speed_coefficient_of_variation": variation,
                             "constant_speed": (constant_speed if times.size > 1 else None),
                             "noise_implied_speed_m_s": noise_speed,
                             "speed_cv_threshold": SPEED_CV_THRESHOLD},
            "metric_scale_support": {"status": scale_status, "trajectory_spread_m": spread,
                                     "median_horizontal_std_m": float(np.median(horizontal)),
                                     "spread_over_sigma": ratio,
                                     "thresholds": {"supported": SCALE_STRONG_RATIO,
                                                    "weak": SCALE_MIN_RATIO}},
            "clock_offset_and_lever_arm_separable": separable,
            "georef_rejects_this_geometry": bool(collinear or len(points) < 4),
            "min_second_ratio": MIN_SECOND_RATIO,
            "identifiability": identifiability,
            "never_claim": never_claim, "supported_statements": statements,
            "warnings": warnings,
            "estimates_anything": False, "accuracy_validated": False}
