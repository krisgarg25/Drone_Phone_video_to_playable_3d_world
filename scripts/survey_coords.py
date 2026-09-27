"""Coordinates people read aloud: MGRS, DMS and a CE90 / LE90 error figure (M8).

Military, border and disaster users pass positions as MGRS grid references and judge
them by CE90 / LE90 (the radius / half-height that contains the true point with 90%
probability). This module turns a WGS84 position plus its per-axis standard deviations
into those forms, and reads an MGRS reference back.

MGRS is built on ``survey_crs``'s own UTM series and zone rules (Norway and Svalbard
exceptions included), so a grid reference and a UTM product can never disagree about
where a point is. UPS (|lat| > 84 deg / < -80 deg) is not implemented and raises, and
so do the far parts of the Norway / Svalbard exception zones (more than 3.5 deg from the
central meridian), where survey_crs's series is not validated.

CE90 is solved exactly for unequal east / north sigmas by integrating the bivariate
normal over the circle, not with the circular-only 2.146 sigma shortcut, which is wrong
when a GNSS error ellipse is elongated. Axes are assumed uncorrelated.
"""
import math
import re

import numpy as np

import survey_crs as crs

BANDS = "CDEFGHJKLMNPQRSTUVWX"
COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
ROWS = "ABCDEFGHJKLMNPQRSTUV"
MIN_LAT, MAX_LAT = -80.0, 84.0
LE90_K = 1.6448536269514722  # two-sided 90% for one normal axis
_MGRS = re.compile(r"^\s*(\d{1,2})\s*([C-HJ-NP-X])\s*([A-HJ-NP-Z])([A-HJ-NP-V])\s*(\d*)\s*(\d*)\s*$")


def _band(lat):
    if not MIN_LAT <= lat <= MAX_LAT:
        raise ValueError(f"latitude {lat} is outside MGRS/UTM bands (-80..84); UPS is not implemented")
    return BANDS[min(int((lat - MIN_LAT) // 8), len(BANDS) - 1)]


def _band_limits(letter):
    index = BANDS.index(letter)
    low = MIN_LAT + 8 * index
    return low, (MAX_LAT if letter == "X" else low + 8)


def to_mgrs(lat_deg, lon_deg, precision=5, *, spaced=False):
    """MGRS reference of a point; ``precision`` digits per axis (5 = 1 m, 0 = 100 km)."""
    if not isinstance(precision, int) or not 0 <= precision <= 5:
        raise ValueError("precision must be an integer 0..5")
    lat, lon = float(lat_deg), float(lon_deg)
    band = _band(lat)
    zone, hemisphere, _ = crs.utm_zone_for(lat, lon)
    try:
        easting, northing = crs.utm_forward([lat], [lon], zone, hemisphere)
    except ValueError as error:
        # The Norway (32V) and Svalbard (31X-37X) zones reach 4-6 deg from their central
        # meridian, past the checked validity of survey_crs's series. Refuse, never guess.
        raise ValueError(f"({lat}, {lon}) lies in exception zone {zone}{band} beyond the UTM "
                         f"series' validity: {error}") from error
    easting, northing = float(easting[0]), float(northing[0])
    column = COLUMN_SETS[(zone - 1) % 3][int(easting // 100_000) - 1]
    row = ROWS[(int(northing // 100_000) + (5 if zone % 2 == 0 else 0)) % 20]
    scale = 10 ** (5 - precision)
    e_digits = f"{int(easting % 100_000) // scale:0{precision}d}" if precision else ""
    n_digits = f"{int(northing % 100_000) // scale:0{precision}d}" if precision else ""
    parts = (f"{zone:02d}{band}", column + row, e_digits, n_digits)
    return " ".join(p for p in parts if p) if spaced else "".join(parts)


def from_mgrs(reference):
    """(lat_deg, lon_deg, cell_size_m) at the centre of the referenced MGRS cell."""
    match = _MGRS.match(str(reference).upper())
    if not match:
        raise ValueError(f"not an MGRS reference: {reference!r}")
    zone, band, column, row, first, second = match.groups()
    digits = first + second
    if len(digits) % 2 or len(digits) > 10:
        raise ValueError("MGRS easting and northing need the same number of digits (0..5 each)")
    zone, precision = int(zone), len(digits) // 2
    if not 1 <= zone <= 60:
        raise ValueError("MGRS zone must be 1..60")
    columns = COLUMN_SETS[(zone - 1) % 3]
    if column not in columns:
        raise ValueError(f"column letter {column} is not used in zone {zone}")
    cell = 10 ** (5 - precision)
    e_in = int(digits[:precision]) * cell if precision else 0
    n_in = int(digits[precision:]) * cell if precision else 0
    easting = (columns.index(column) + 1) * 100_000 + e_in + cell / 2
    row_offset = (ROWS.index(row) - (5 if zone % 2 == 0 else 0)) % 20
    base = row_offset * 100_000 + n_in + cell / 2
    hemisphere = "N" if band >= "N" else "S"
    low, high = _band_limits(band)
    for k in range(10):
        northing = base + k * 2_000_000
        lat, lon = crs.utm_inverse([easting], [northing], zone, hemisphere)
        lat, lon = float(lat[0]), float(lon[0])
        if low - 0.5 <= lat < high + 0.5:
            return lat, lon, float(cell)
    raise ValueError(f"MGRS row {row} does not fall in latitude band {band} of zone {zone}")


def to_dms(lat_deg, lon_deg, decimals=2):
    def one(value, positive, negative):
        hemisphere = positive if value >= 0 else negative
        value = abs(value)
        degrees = int(value)
        minutes_full = (value - degrees) * 60
        minutes = int(minutes_full)
        seconds = round((minutes_full - minutes) * 60, decimals)
        if seconds >= 60:
            seconds, minutes = 0.0, minutes + 1
        if minutes >= 60:
            minutes, degrees = 0, degrees + 1
        return f"{degrees}°{minutes:02d}'{seconds:0{3 + decimals}.{decimals}f}\"{hemisphere}"
    return one(float(lat_deg), "N", "S") + " " + one(float(lon_deg), "E", "W")


def _circle_probability(radius, sigma_a, sigma_b, samples=720):
    """P(|x| <= radius) for x ~ N(0, diag(sa^2, sb^2)), integrated over the angle exactly."""
    theta = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    a = np.cos(theta) ** 2 / (2 * sigma_a ** 2) + np.sin(theta) ** 2 / (2 * sigma_b ** 2)
    integrand = (1 - np.exp(-radius ** 2 * a)) / (2 * a)
    return float(np.mean(integrand) * 2 * np.pi / (2 * np.pi * sigma_a * sigma_b))


def ce90(sigma_east_m, sigma_north_m, probability=0.9):
    """Radius holding the horizontal error with ``probability`` (uncorrelated axes)."""
    sa, sb = float(sigma_east_m), float(sigma_north_m)
    if not (sa > 0 and sb > 0 and math.isfinite(sa) and math.isfinite(sb)):
        raise ValueError("sigmas must be finite and positive")
    low, high = 0.0, 10 * max(sa, sb)
    for _ in range(80):
        mid = (low + high) / 2
        if _circle_probability(mid, sa, sb) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def le90(sigma_up_m):
    sigma = float(sigma_up_m)
    if not (sigma > 0 and math.isfinite(sigma)):
        raise ValueError("sigma must be finite and positive")
    return LE90_K * sigma


def describe_point(lat_deg, lon_deg, height_m, *, height_datum, sigma_east_m=None,
                   sigma_north_m=None, sigma_up_m=None, precision=5):
    """Every human-readable form of one position, with its 90% error figures when known."""
    zone, hemisphere, epsg = crs.utm_zone_for(lat_deg, lon_deg)
    easting, northing = crs.utm_forward([lat_deg], [lon_deg], zone, hemisphere)
    out = {"lat_deg": float(lat_deg), "lon_deg": float(lon_deg), "height_m": float(height_m),
           "height_datum": height_datum, "mgrs": to_mgrs(lat_deg, lon_deg, precision, spaced=True),
           "dms": to_dms(lat_deg, lon_deg),
           "utm": {"zone": f"{zone}{hemisphere}", "epsg": epsg,
                   "easting_m": round(float(easting[0]), 3),
                   "northing_m": round(float(northing[0]), 3)}}
    if sigma_east_m is not None and sigma_north_m is not None:
        out["ce90_m"] = round(ce90(sigma_east_m, sigma_north_m), 3)
    if sigma_up_m is not None:
        out["le90_m"] = round(le90(sigma_up_m), 3)
    out["basis"] = ("CE90/LE90 from the stated per-axis sigmas, uncorrelated normal errors; "
                    "not an independently measured accuracy" if "ce90_m" in out or "le90_m" in out
                    else "no uncertainty supplied, so no CE90/LE90 is stated")
    return out
