"""Takeoff-anchored heights: GNSS sets the level once, the barometer draws the shape (M12).

Vertical is usually the worst axis of a GPS-only georeference. A consumer receiver's
ellipsoidal height wanders by metres from fix to fix, while the aircraft's relative
(barometric, above-takeoff) altitude is smooth and precise over a ten-minute flight but
has no absolute level. Together they give:

    h(t) = h_takeoff + relative_altitude(t)

``h_takeoff`` is either declared by the operator (a surveyed benchmark at the takeoff
point) or estimated as the robust centre of ``h_gnss(t) - relative_altitude(t)`` over
the whole flight. Each estimate carries its own standard deviation, and the output
height's uncertainty is built only from quantities this module measured or was given:

* ``offset_std_m``  - 1.4826 * MAD of the detrended GNSS-minus-baro residual, divided by
  the square root of the number of *independent* samples. GNSS height errors are
  correlated over minutes; for a first-order Gauss-Markov error with time constant
  ``correlation_s`` (default 60 s, recorded) that number is ``span / (2 correlation_s)``,
  never the raw sample count. Measured on 200 synthetic 600 s flights with a 60 s,
  2 m process: 2-sigma coverage of the true offset is 88% (ideal 95%) - still slightly
  optimistic, because one correlated flight's scatter underestimates sigma. With
  ``span / correlation_s`` it was 79%.
* ``drift_m``       - a straight-line trend of the same residual across the flight. The
  barometer drifts with weather and temperature; the trend is the only evidence of it
  the log contains, so it is reported and folded into the height uncertainty rather
  than subtracted as if it were known to be barometer and not GNSS.

Heights are fused in the geodetic (ellipsoidal) domain, before the ENU conversion, so
earth curvature over a long corridor does not leak into the comparison. Nothing here
changes horizontal positions. NumPy only; bad input raises ValueError.
"""
import csv
import io
import math

import numpy as np

DEFAULT_CORRELATION_S = 60.0
MIN_OVERLAP_SAMPLES = 10
MAX_GAP_S = 2.0
TELEMETRY_FIELDS = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m",
                    "horizontal_std_m", "vertical_std_m")


def _series(rows, value_key, name):
    times, values = [], []
    for row in rows:
        t, v = row.get("t_sec"), row.get(value_key)
        if t is None or v is None:
            continue
        t, v = float(t), float(v)
        if math.isfinite(t) and math.isfinite(v):
            times.append(t)
            values.append(v)
    if not times:
        raise ValueError(f"{name} has no finite samples")
    order = np.argsort(times)
    times, values = np.asarray(times)[order], np.asarray(values)[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} times must be strictly increasing")
    return times, values


def _bracketed(times, values, wanted, max_gap_s):
    out = np.full(len(wanted), np.nan)
    for i, t in enumerate(wanted):
        right = int(np.searchsorted(times, t))
        if right < len(times) and times[right] == t:
            out[i] = values[right]
        elif 0 < right < len(times) and times[right] - times[right - 1] <= max_gap_s:
            a = (t - times[right - 1]) / (times[right] - times[right - 1])
            out[i] = (1 - a) * values[right - 1] + a * values[right]
    return out


def fuse_heights(gnss_rows, relative_rows, *, takeoff_height_m=None, takeoff_std_m=None,
                 correlation_s=DEFAULT_CORRELATION_S, max_gap_s=MAX_GAP_S):
    """Return (rows with fused altitude_m and vertical_std_m, report).

    ``gnss_rows`` are six-field telemetry rows with ellipsoidal ``altitude_m``;
    ``relative_rows`` are {t_sec, relative_altitude_m}. Rows with no bracketing
    relative sample within ``max_gap_s`` keep their GNSS height and are counted.
    """
    if correlation_s <= 0 or not math.isfinite(correlation_s):
        raise ValueError("correlation_s must be positive")
    if (takeoff_height_m is None) != (takeoff_std_m is None):
        raise ValueError("Declare takeoff_height_m and takeoff_std_m together")
    gnss_rows = sorted(gnss_rows, key=lambda r: float(r["t_sec"]))
    g_times, g_height = _series(gnss_rows, "altitude_m", "GNSS telemetry")
    if len(g_times) != len(gnss_rows):
        raise ValueError("Every GNSS row needs a finite t_sec and altitude_m")
    r_times, r_alt = _series(relative_rows, "relative_altitude_m", "relative altitude")
    relative = _bracketed(r_times, r_alt, g_times, max_gap_s)
    overlap = np.isfinite(relative)
    if int(overlap.sum()) < MIN_OVERLAP_SAMPLES:
        raise ValueError(f"Only {int(overlap.sum())} GNSS samples overlap the relative altitude "
                         f"log; need {MIN_OVERLAP_SAMPLES}")
    residual = g_height[overlap] - relative[overlap]
    t = g_times[overlap]
    span = float(t[-1] - t[0])
    # Variance of the mean of a first-order Gauss-Markov process over span T with time
    # constant tau is ~ 2 sigma^2 tau / T, i.e. T / (2 tau) independent samples.
    independent = max(1.0, span / (2.0 * correlation_s))
    centre = float(np.median(residual))
    slope = float(np.polyfit(t - t.mean(), residual, 1)[0]) if span > 0 else 0.0
    drift = slope * span
    # Scatter about the trend, so a drifting barometer cannot inflate its own yardstick.
    detrended = residual - slope * (t - t.mean())
    scatter = float(1.4826 * np.median(np.abs(detrended - np.median(detrended))))
    if takeoff_height_m is None:
        offset, offset_std, basis = centre, scatter / math.sqrt(independent), "gnss_median"
    else:
        offset, offset_std, basis = float(takeoff_height_m), float(takeoff_std_m), "declared"
        if offset_std <= 0 or not math.isfinite(offset_std):
            raise ValueError("takeoff_std_m must be positive")
    # The trend could be baro drift or GNSS drift; half the end-to-end change bounds how
    # far the smooth baro curve can sit from the truth at its worst point.
    vertical_std = math.sqrt(offset_std ** 2 + (abs(drift) / 2) ** 2)
    fused, kept = [], 0
    for row, rel in zip(gnss_rows, relative):
        row = dict(row)
        if math.isfinite(rel):
            row["altitude_m"] = offset + float(rel)
            row["vertical_std_m"] = max(vertical_std, 1e-3)
            kept += 1
        fused.append(row)
    report = dict(
        status="fused", basis=basis, takeoff_height_m=round(offset, 4),
        offset_std_m=round(offset_std, 4), gnss_minus_baro_scatter_m=round(scatter, 4),
        drift_m=round(drift, 4), fused_vertical_std_m=round(vertical_std, 4),
        samples_fused=kept, samples_kept_gnss=len(fused) - kept,
        span_s=round(span, 3), independent_samples=round(independent, 2),
        correlation_s=correlation_s,
        gnss_vertical_std_median_m=round(float(np.median(
            [float(r["vertical_std_m"]) for r in gnss_rows])), 4),
        notes=["Relative altitude is barometric above takeoff; its level comes from "
               + ("the declared takeoff height." if basis == "declared" else
                  "the median GNSS-minus-baro residual over the whole flight."),
               f"GNSS height errors are treated as correlated over {correlation_s:g} s, so "
               f"{independent:.1f} independent samples, not {int(overlap.sum())}.",
               "drift_m is the end-to-end trend of GNSS minus baro; it is folded into the "
               "uncertainty, not removed, because the log cannot say which sensor drifted."])
    if abs(drift) > 3 * max(scatter, 0.1):
        report["warnings"] = [f"GNSS and barometer disagree by a {drift:+.2f} m trend over "
                              f"{span:.0f} s; check for a weather front, a baro reset or a "
                              "GNSS height jump before trusting either."]
    return fused, report


def read_telemetry_csv(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or set(rows[0]) != set(TELEMETRY_FIELDS):
        raise ValueError(f"Telemetry CSV requires exactly {TELEMETRY_FIELDS}")
    return rows


def write_telemetry_csv(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(TELEMETRY_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (repr(float(row[key])) if key != "t_sec" else row[key])
                         for key in TELEMETRY_FIELDS})
    return buffer.getvalue()
