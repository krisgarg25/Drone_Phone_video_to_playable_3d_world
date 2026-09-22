"""Normalise real drone telemetry into the survey_georef camera-position contract.

The contract is ``scripts/survey_georef.normalize_telemetry``: a CSV/JSONL carrying
exactly ``t_sec, latitude_deg, longitude_deg, altitude_m, horizontal_std_m,
vertical_std_m``, strictly increasing times inside the declared video duration,
positive finite uncertainties, WGS84 lat/lon and an **ellipsoidal** altitude, plus
metadata declaring ``time_reference='video'``, ``altitude_datum='ellipsoidal'`` and
``position_reference='camera_center'``.

This module is the honest ingest path in front of it:

* parsers for the sources a drone actually produces - GPX tracks, DJI telemetry
  ``.srt`` subtitles, ``dji_telemetry.*`` CSV exports, and the contract CSV itself;
* a documented, deliberately conservative uncertainty model that turns RTK/PPK fix
  quality and HDOP into the two standard deviations, and refuses to invent either
  when the file carries no evidence of precision;
* optional IMU and barometer readers that state what those sensors can and cannot
  constrain, and never turn them into a position;
* ``merge_metadata``, which declares only what there is evidence for and otherwise
  returns typed blockers instead of a value that would make the pipeline quiet.

Deliberate limits. No geoid model, so a height above mean sea level is never
relabelled ellipsoidal. No time-zone or leap-second handling: wall-clock stamps are
read as UTC and reported as such. No unit guessing: the DJI layouts supported are the
documented Pilot/Assistant and subtitle-telemetry dialects, and a disagreeing dialect
is rejected rather than approximated. Every rejected row names its file and line, and
a parser never returns a row containing NaN or None - a row whose precision is unknown
is returned *without* those keys (see ``strict``) so feeding it to survey_georef fails
loudly in the caller's hands rather than quietly here.
"""
from __future__ import annotations

import csv
import datetime
import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["TELEMETRY_FIELDS", "POSITIONAL_FIELDS", "FIX_QUALITY_STD_M",
           "DOP_TO_METRE_UERE", "VERTICAL_DOP_RATIO", "HDOP_DEGRADED_LIMIT",
           "MIN_SAMPLE_SPAN_M", "parse_fix_quality", "sniff_format", "source_info",
           "parse_gpx", "parse_telemetry_csv", "parse_srt", "parse_dji_csv",
           "from_flight_log", "attach_quality", "read_imu", "read_barometer",
           "merge_metadata", "probe_video", "DetectedFormat", "QualityResult",
           "SensorSeries", "Blocker"]

TELEMETRY_FIELDS = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m",
                    "horizontal_std_m", "vertical_std_m")
POSITIONAL_FIELDS = ("t_sec", "latitude_deg", "longitude_deg", "altitude_m")

# --------------------------------------------------------------------- uncertainty model
# Conservative per-axis standard deviations in metres, per 3DR/Sony/ArduPilot fix-type
# naming. Each figure is a documented judgement, deliberately worse than the best case a
# datasheet quotes: a consumer drone's antenna is small, its own body and arms block part
# of the sky, and multipath on a facade is the survey-relevant failure mode. These are
# NOT calibrated for any specific fleet and are NOT a measured error budget.
FIX_QUALITY_STD_M: dict[str, tuple[float, float] | None] = {
    "no_fix": None,          # no usable position: refused, never floored to a guess
    "unknown": (5.0, 10.0),  # fix type not recorded: open-sky SPS, degraded
    "single": (2.0, 4.0),    # SPS, no corrections
    "psdiff": (0.5, 1.0),    # SBAS / pseudo-range differential
    "float": (0.15, 0.25),   # RTK/PPK float ambiguity, a few L1 wavelengths
    "fixed": (0.03, 0.05),   # RTK/PPK fixed: datasheet 0.01/0.015 with a 3x margin
}
FIX_QUALITY_ORDER = ("no_fix", "unknown", "single", "psdiff", "float", "fixed")
_RANK = {name: order for order, name in enumerate(FIX_QUALITY_ORDER)}
# A pseudorange position is roughly HDOP x UERE per axis. 2 m is a conservative user
# equivalent range error for single-frequency C/A code; under foliage it is larger.
DOP_TO_METRE_UERE = 2.0
# Vertical geometry is worse than horizontal for any plausible constellation; 1.5 is the
# usual near-mid-latitude VDOP/HDOP approximation.
VERTICAL_DOP_RATIO = 1.5
# Above this dilution the geometry, and so any unverified phase-lock claim, is degraded.
HDOP_DEGRADED_LIMIT = 2.0
# A trajectory shorter than this cannot constrain scale however precise its points are.
MIN_SAMPLE_SPAN_M = 0.5

_FIX_TOKENS = {
    "none": "no_fix", "0": "no_fix", "no_fix": "no_fix", "nofix": "no_fix",
    "no fix": "no_fix", "no_position": "no_fix", "no solution": "no_fix",
    "invalid": "no_fix", "1": "single", "single": "single", "sps": "single",
    "autonomous": "single", "3d": "single", "2d": "single", "2": "psdiff",
    "psdiff": "psdiff", "psdiffcode": "psdiff", "sbas": "psdiff", "waas": "psdiff",
    "egnos": "psdiff", "gbs": "psdiff", "qzs": "psdiff", "3": "float",
    "float": "float", "rtk_float": "float", "float_rtk": "float", "rtkfloat": "float",
    "4": "fixed", "fixed": "fixed", "rtk": "fixed", "rtk_fixed": "fixed",
    "rtkfixed": "fixed", "fixed_rtk": "fixed", "ppk": "fixed",
}
# ArduPilot numbers 5-7 mean "None"/"custom" while 3DR/Sony read 5 as "No RTK", and
# ArduPilot's "3D" is a text fix type. Where one token has two meanings it is refused.
_FIX_REFUSED = {"5", "6", "7", "custom", "rtk_none", "no rtk", "static", "ppi3d",
                "gps+glonext", "3d fix", "gps + glonass", "gps + glonass sbas"}


def parse_fix_quality(token) -> str:
    """Map a GPSDIM/FIX_TYPE/rtk_quality token onto a canonical quality name.

    Integer codes are read as the ArduPilot / DJI Pilot 2 ladder: 0 no fix, 1 single,
    2 pseudo-range differential, 3 RTK float, 4 RTK fixed. Codes 5-7 and vendor-specific
    strings are refused because ArduPilot and 3DR/Sony assign opposite meanings to 5, so
    this module will not pick one for the caller.
    """
    if isinstance(token, (bool, np.bool_)):
        raise ValueError(f"unknown_fix_token: {token!r} is a boolean, not a fix quality")
    if isinstance(token, (int, np.integer)):
        key = str(int(token))
    elif isinstance(token, str):
        key = re.sub(r"\s+", " ", token.strip().lower()).replace("-", "_")
    else:
        raise ValueError(f"unknown_fix_token: cannot read a fix quality from {token!r}")
    if key in _FIX_REFUSED:
        raise ValueError(f"unknown_fix_token: {token!r} means different things in the "
                         "ArduPilot and 3DR/Sony tables; map it yourself or drop the row")
    mapped = _FIX_TOKENS.get(key) or _FIX_TOKENS.get(key.replace(" ", "_"))
    if mapped is None:
        raise ValueError(f"unknown_fix_token: {token!r} is not a recognised fix quality; "
                         "known tokens are 0-4, no_fix, single, 3D, psdiff, sbas, float, fixed")
    return mapped


def _worse(left: str, right: str) -> str:
    """The lower-ranked (worse) of two fix-quality names, for bracketing samples."""
    return left if _RANK[left] <= _RANK[right] else right


def _std_from_dilution(hdop: float | None, fix: str | None) -> tuple[float, float]:
    """Metres of per-axis standard deviation from dilution and/or fix-quality evidence.

    The model, conservative in every step:
      horizontal = max(floor, HDOP x UERE) with UERE = ``DOP_TO_METRE_UERE``, where the
      floor is the fix-quality table entry, or the single-fix entry when the fix type is
      unknown - a dilution figure alone cannot show a corrected or phase-locked solution;
      vertical = max(table vertical floor, horizontal x ``VERTICAL_DOP_RATIO``);
      a float/fixed claim beside HDOP above ``HDOP_DEGRADED_LIMIT`` is additionally scaled
      by HDOP / limit, because degraded geometry makes an unverified RTK claim unsafe.
    Values are rounded to 0.1 mm: a finer figure from a generic model is noise, and
    rounding down would be a false precision.
    """
    # fix=None means "no fix evidence at all", which is the HDOP-only case and is floored
    # at the single-fix class; fix="unknown" means a quality series exists but has a gap
    # here, which is worse and gets its own pessimistic floor.
    table = FIX_QUALITY_STD_M[fix if fix is not None else "single"]
    if table is None:
        raise ValueError("no_fix carries no usable position, so it cannot become an "
                         "uncertainty: it can only be refused")
    floor_h, floor_v = table
    if hdop is None:
        result_h = floor_h
    elif fix in ("float", "fixed"):
        # A phase-lock solution is not limited by code pseudorange error, so dilution
        # only degrades it once the geometry is bad; below the limit the table stands.
        if not math.isfinite(hdop) or hdop <= 0:
            raise ValueError(f"hdop {hdop!r} must be a positive finite dilution factor")
        result_h = floor_h * max(1.0, hdop / HDOP_DEGRADED_LIMIT)
    else:
        if not math.isfinite(hdop) or hdop <= 0:
            raise ValueError(f"hdop {hdop!r} must be a positive finite dilution factor")
        base = FIX_QUALITY_STD_M["single"][0] if fix is None else floor_h
        result_h = max(base, hdop * DOP_TO_METRE_UERE)
    result_v = max(floor_v, result_h * VERTICAL_DOP_RATIO)
    return round(result_h, 4), round(result_v, 4)


# ------------------------------------------------------------------------- diagnostics
class _RejectRow(Exception):
    """Internal: this record is unparseable, so it is counted and dropped."""


_UNCERTAINTY_HINT = (
    "this source carries no evidence of positioning precision. Either pass "
    "horizontal_std_m= and vertical_std_m= together (a declared conservative value for "
    "both), or parse with strict=False and call survey_inputs.attach_quality with "
    "fix_quality_by_time= or hdop_by_time=. No default uncertainty is filled in here, "
    "because an invented standard deviation is a false precision claim.")

_GEOSET = {"latitude_deg": (-90.0, 90.0), "longitude_deg": (-180.0, 180.0),
           "altitude_m": (-2e6, 2e6)}
_NOT_FINITE = {"nan", "inf", "-inf", "+inf", "infinity", "none", "null", "n/a", "na", "<na>"}


def _finite_number(token, name: str | None = None) -> float:
    """A finite float, a hard ValueError for missing/NaN/infinite, or _RejectRow for junk."""
    label = f"{name} " if name else ""
    text = "" if token is None else str(token).strip()
    if text == "" or text.lower() in _NOT_FINITE:
        raise ValueError(f"{label}value {token!r} is not a finite number; missing, NaN and "
                         "infinite readings are never filled in")
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise _RejectRow(f"{label}value {token!r} is not numeric") from None
    if not math.isfinite(value):
        raise ValueError(f"{label}value {token!r} is not a finite number")
    return value


def _number(token, name: str | None = None) -> float:
    """A required numeric field: absent or empty is a hard error, unparseable junk is not."""
    if token is None:
        raise ValueError(f"{name or 'a field'} is absent from this record")
    return _finite_number(token, name)


def _geo(value: float, name: str) -> float:
    low, high = _GEOSET[name]
    if not low <= value <= high:
        raise ValueError(f"{name} {value} is outside the range [{low}, {high}]")
    return value


def _where(path, line=None, block=None) -> str:
    parts = [Path(path).name]
    if line is not None:
        parts.append(f"line {line}")
    if block is not None:
        parts.append(f"block {block}")
    return " ".join(parts)


def _check_times(times: Sequence[float], path, line_of=lambda index: None) -> None:
    for index in range(1, len(times)):
        if not times[index] > times[index - 1]:
            raise ValueError(f"{_where(path, line=line_of(index))}: telemetry times must be "
                             f"strictly increasing; {times[index - 1]} is followed by "
                             f"{times[index]}. Duplicate or reversed stamps are never "
                             "silently dropped or reordered")


def _span_too_small(rows: Sequence[dict]) -> bool:
    """True when these positions cannot constrain scale."""
    if len(rows) < 2:
        return True
    lat = np.array([float(row["latitude_deg"]) for row in rows])
    lon = np.array([float(row["longitude_deg"]) for row in rows])
    alt = np.array([float(row["altitude_m"]) for row in rows])
    east = np.radians(lon - lon.mean()) * 6378137.0 * math.cos(math.radians(lat.mean()))
    north = np.radians(lat - lat.mean()) * 6356752.3142
    return max(float(np.ptp(east)), float(np.ptp(north)), float(np.ptp(alt))) < MIN_SAMPLE_SPAN_M


def _note_span(rows, warnings) -> None:
    if _span_too_small(rows):
        warnings.append(f"positions span less than {MIN_SAMPLE_SPAN_M} m: a stationary or "
                        "single-point record cannot constrain scale, so any similarity "
                        "fitted from it is meaningless")


def _finish(path: Path, rows: list[dict], rejected: list[dict], *, on_malformed: str,
            max_dropped_fraction: float, expected: int) -> list[dict]:
    """Apply the malformed-row policy: name the offending line and stop, or count and go on."""
    if on_malformed not in ("reject", "drop"):
        raise ValueError("on_malformed must be 'reject' or 'drop'")
    if rejected and on_malformed == "reject":
        first = rejected[0]
        extra = f" (plus {len(rejected) - 1} more)" if len(rejected) > 1 else ""
        raise ValueError(f"{_where(path, line=first.get('line'), block=first.get('block'))}"
                         f": {first['message']}{extra}")
    if not rows:
        raise ValueError(f"{path.name}: every telemetry row was rejected")
    fraction = len(rejected) / max(expected, 1)
    if fraction > max_dropped_fraction:
        raise ValueError(f"{path.name}: {len(rejected)} of {expected} rows dropped "
                         f"({fraction:.0%}) exceeds max_dropped_fraction="
                         f"{max_dropped_fraction:.0%}; a track that loses this many samples "
                         "is not the flight that was flown")
    return rows


# --------------------------------------------------------------------------- provenance
_SOURCE_INFO: dict[str, dict] = {}


def _key(path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def _record(path: Path, info: dict) -> None:
    _SOURCE_INFO[_key(path)] = info


def source_info(path) -> dict:
    """Provenance from the last parse of ``path`` - datum, clock, side channels, warnings.

    Kept out of the telemetry rows on purpose: survey_georef requires exactly six fields,
    so where a height came from and how its datum is qualified belong to the metadata
    this module hands to ``merge_metadata``.
    """
    return json.loads(json.dumps(_SOURCE_INFO.get(_key(path), {}), default=float))


# ------------------------------------------------------------------------------ sniffing
@dataclass(frozen=True)
class DetectedFormat:
    """What the bytes said. ``via`` is always "content"; the extension is only noted."""
    kind: str
    via: str
    extension: str
    extension_agrees: bool
    detail: str = ""


_SNIFF_BYTES = 65536
_TIMECODE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")
_SRT_MARKERS = ("Latitude:", "Longitude:", "AbsoluteAltitude:", "RelativeAltitude:")
_DJI_HINTS = {"latitude", "longitude", "absolute_altitude", "relative_altitude",
              "gimbal_pitch", "gimbal_yaw", "elevation", "gps_number", "rtk_quality"}
_IMU_HINTS = {"ax", "ay", "az", "gx", "gy", "gz", "accel_x", "gyro_x"}
_BARO_HINTS = {"pressure_hpa", "barometric_pressure", "baro_altitude", "abs_pressure"}
_EXTENSION_KINDS = {"gpx": {".gpx"}, "srt": {".srt"}, "dji_csv": {".csv", ".txt"},
                    "telemetry_csv": {".csv", ".jsonl"}, "imu": {".csv", ".txt"},
                    "barometer": {".csv", ".txt"}, "unknown": set()}


def _bare(cell: str) -> str:
    return cell.split(".")[-1] if cell.startswith("dji_telemetry") else cell


def sniff_format(path) -> DetectedFormat:
    """Identify a flight-log dialect from its content, never from the extension alone.

    A file named ``.srt`` whose bytes are GPX is read as GPX; ``extension_agrees`` says
    the name disagreed, so the caller can log it. Extension is never a tie-breaker.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path.name}: no such file")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        text = stream.read(_SNIFF_BYTES)
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    kind, detail = "unknown", "no recognised telemetry dialect"
    if "<gpx" in " ".join(lines[:40]).lower():
        kind, detail = "gpx", "XML GPX element found"
    elif any(_TIMECODE.match(line) for line in lines):
        if any(marker in text for marker in _SRT_MARKERS):
            kind, detail = "srt", "subtitle blocks carrying DJI telemetry fields"
        else:
            kind, detail = "unknown", "subtitle file with no telemetry fields"
    elif non_empty and "," in non_empty[0]:
        header = next(csv.reader([non_empty[0]], skipinitialspace=True))
        columns = [cell.strip().lower() for cell in header if cell.strip()]
        bare = [_bare(cell) for cell in columns]
        prefixed = any(cell.startswith("dji_telemetry") for cell in columns)
        if set(columns) <= set(TELEMETRY_FIELDS) and len(columns) > 1:
            kind, detail = "telemetry_csv", "survey_georef contract columns"
        elif sum(1 for cell in bare if cell in _DJI_HINTS) >= 2:
            kind, detail = "dji_csv", "DJI telemetry columns" + (" (dji_telemetry.* prefix)"
                                                                 if prefixed else "")
        elif sum(1 for cell in bare if cell in _IMU_HINTS) >= 2:
            kind, detail = "imu", "inertial measurement columns"
        elif any(cell in _BARO_HINTS for cell in bare):
            kind, detail = "barometer", "barometric columns"
    extension = path.suffix.lower()
    return DetectedFormat(kind=kind, via="content", extension=extension,
                          extension_agrees=extension in _EXTENSION_KINDS[kind], detail=detail)


# -------------------------------------------------------------------------- plumbing
def _open(path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path.name}: no such file")
    return path


def _text_of(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _csv_records(text: str) -> list[tuple[int, list[str]]]:
    """(physical line number, cells) per non-empty record.

    Line numbers stay aligned unless a quoted field spans a line break, which none of
    these dialects use; a multiline field would shift only the reported number.
    """
    out = []
    for number, cells in enumerate(csv.reader(text.splitlines()), start=1):
        if any(cell.strip() for cell in cells):
            out.append((number, cells))
    return out


def _columns(cells: Sequence[str]) -> dict[str, int]:
    return {cell.strip().lower(): index for index, cell in enumerate(cells)
            if cell.strip() != ""}


def _cell(cells: Sequence[str], index: int | None):
    return None if index is None or index >= len(cells) else cells[index]


def _pick(columns: dict[str, int], names: Iterable[str]) -> int | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _time_epoch(token, *, what: str) -> float:
    """Seconds from the Unix epoch, from an integer stamp or an ISO-ish wall clock.

    Integer stamps of 10/13/16 digits are read as seconds/milliseconds/microseconds,
    which for this century cannot collide. Timestamps without a zone are read as UTC and
    the provenance says so; no leap-second handling is applied.
    """
    raw = str(token).strip()
    if re.fullmatch(r"-?\d{10}", raw):
        return float(raw)
    if re.fullmatch(r"-?\d{13}", raw):
        return float(raw) / 1e3
    if re.fullmatch(r"-?\d{16}", raw):
        return float(raw) / 1e6
    if re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", raw):
        try:
            parsed = datetime.datetime.fromisoformat(raw.replace("T", " ").replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{what} timestamp {raw!r} is not parseable as ISO-8601") from None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)  # logged by the aircraft
        return parsed.timestamp()
    return _finite_number(raw)


def _apply_declared_uncertainty(rows: list[dict], horizontal, vertical) -> None:
    for name, value in (("horizontal_std_m", horizontal), ("vertical_std_m", vertical)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be a positive finite number of metres")
    for row in rows:
        row.update(horizontal_std_m=round(float(horizontal), 4),
                   vertical_std_m=round(float(vertical), 4))


def _positional_only(rows: list[dict]) -> list[dict]:
    return [{name: row[name] for name in POSITIONAL_FIELDS} for row in rows]


def _six_fields(rows: list[dict]) -> list[dict]:
    return [{name: row[name] for name in TELEMETRY_FIELDS if name in row} for row in rows]


def _declared_pair(provenance: str, horizontal, vertical):
    """The caller-declared [horizontal, vertical] pair, echoed into the provenance."""
    if provenance != "declared_by_caller" or horizontal is None or vertical is None:
        return None
    return [round(float(horizontal), 4), round(float(vertical), 4)]


def _check_pair(horizontal, vertical) -> None:
    if (horizontal is None) != (vertical is None):
        raise ValueError("horizontal_std_m and vertical_std_m must be given together: one "
                         "declared axis with a silent gap on the other is not a declaration")


# ------------------------------------------------------------------------------------ GPX
_HDOP_NAMES = {"hdop", "dhdop", "hdops"}
_DOP_ANY = _HDOP_NAMES | {"vdop", "dvop", "vdops", "pdop"}
_XML_FORBIDDEN = re.compile(r"<!\s*(DOCTYPE|ENTITY|ELEMENT)", re.IGNORECASE)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node, name: str):
    for child in node:
        if _local(child.tag) == name:
            return child.text
    return None


def _hdop_of(trkpt) -> float | None:
    """The trackpoint's dilution, or an error when the provenance is not unique."""
    found: dict[str, list[float]] = {}
    for node in trkpt.iter():
        name = _local(node.tag)
        if name in _DOP_ANY:
            try:
                found.setdefault(name, []).append(_finite_number(node.text))
            except (ValueError, _RejectRow):
                continue
    horizontal = {name: values for name, values in found.items() if name in _HDOP_NAMES}
    if len(horizontal) > 1:
        raise ValueError(f"ambiguous dilution provenance: {sorted(horizontal)} all appear in "
                         "one trackpoint, so which HDOP describes this position is unclear")
    for values in horizontal.values():
        if len(set(values)) > 1:
            raise ValueError(f"ambiguous dilution provenance: the same dilution name appears "
                             f"twice with different values {values}")
    if not horizontal:
        return None
    return next(iter(horizontal.values()))[0]


def parse_gpx(path, *, horizontal_std_m: float | None = None,
              vertical_std_m: float | None = None,
              time_origin_epoch: float | None = None, on_malformed: str = "reject",
              max_dropped_fraction: float = 0.05, strict: bool = True) -> list[dict]:
    """Read a GPX track into telemetry rows.

    ``<ele>`` is taken as the height, and the GPX schema does not define its datum -
    receivers log ellipsoidal or mean-sea-level heights depending on firmware - so the
    provenance records ``unspecified_by_gpx`` and no datum is claimed. Times are read as
    UTC and made relative to the first trackpoint, or to ``time_origin_epoch`` when the
    caller states where video t=0 falls on that clock.

    Uncertainty must come from the file (an ``hdop``/``dhdop``/``hdops`` element, any
    namespace) mapped through ``_std_from_dilution``, or from an explicit
    ``horizontal_std_m``/``vertical_std_m`` pair. With neither, the parse raises; pass
    ``strict=False`` for positional-only rows to hand to ``attach_quality``. DTDs and
    entity declarations are refused outright: a flight log is third-party input.
    """
    _check_pair(horizontal_std_m, vertical_std_m)
    path = _open(path)
    raw = _text_of(path)
    if _XML_FORBIDDEN.search(raw):
        raise ValueError(f"{path.name}: GPX with a DTD or entity declaration is refused "
                         "(entity-expansion risk in third-party logs)")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"{path.name}: not a GPX document ({exc})") from None
    if _local(root.tag) != "gpx" and root.find(".//" + "gpx") is None:
        raise ValueError(f"{path.name}: not a GPX file - the root element is "
                         f"{_local(root.tag)!r}")
    points = [node for node in root.iter() if _local(node.tag) == "trkpt"]
    if not points:
        raise ValueError(f"{path.name}: not a GPX track: no trkpt element to place a camera")
    # ElementTree's non-validating parser does not expose start positions through
    # fromstring, so line numbers come from matching <trkpt openings in document order.
    # When the counts disagree (a trkpt written across lines, or a tag name inside text)
    # errors name the file without a line rather than guessing one.
    raw_lines = [number for number, line in enumerate(raw.splitlines(), start=1)
                 if "<trkpt" in line.lower()]
    aligned = len(raw_lines) == len(points)

    def line_for(order: int):
        return raw_lines[order] if aligned else None

    rows, rejected, lines = [], [], []
    for order, trkpt in enumerate(points):
        position = {"line": line_for(order)}
        try:
            lat = _geo(_number(trkpt.attrib.get("lat"), "lat"), "latitude_deg")
            lon = _geo(_number(trkpt.attrib.get("lon"), "lon"), "longitude_deg")
            if _child_text(trkpt, "ele") is None:
                raise _RejectRow("trackpoint carries no <ele> height")
            alt = _finite_number(_child_text(trkpt, "ele"), "ele")
            if _child_text(trkpt, "time") is None:
                raise ValueError("trackpoint carries no <time>, so the track cannot be "
                                 "ordered against a video")
            wall = _time_epoch(_child_text(trkpt, "time"), what="GPX time")
            dilution = _hdop_of(trkpt)
        except _RejectRow as exc:
            if on_malformed == "reject":
                raise ValueError(f"{_where(path, **position)}: {exc}") from None
            rejected.append(dict(message=str(exc), **position))
            continue
        except ValueError as exc:
            raise ValueError(f"{_where(path, **position)}: {exc}") from None
        lines.append(position["line"])
        rows.append(dict(t_sec=wall, latitude_deg=lat, longitude_deg=lon, altitude_m=alt,
                         _dilution=dilution))
    origin = rows[0]["t_sec"] if time_origin_epoch is None else float(time_origin_epoch)
    for row in rows:
        row["t_sec"] = round(row["t_sec"] - origin, 6)
    _check_times([row["t_sec"] for row in rows], path, line_of=lines.__getitem__)
    _finish(path, rows, rejected, on_malformed=on_malformed,
            max_dropped_fraction=max_dropped_fraction, expected=len(rows) + len(rejected))
    reference = "video" if time_origin_epoch is not None else "gnss_wall_clock"
    warnings = ["GPX <ele> datum is unspecified by the GPX schema: neither ellipsoidal nor "
                "mean sea level is claimed, and no geoid model is available here to convert"]
    dilutions = [row.pop("_dilution") for row in rows]
    if horizontal_std_m is not None:
        _apply_declared_uncertainty(rows, horizontal_std_m, vertical_std_m)
        provenance = "declared_by_caller"
    elif any(value is not None for value in dilutions):
        known = [value for value in dilutions if value is not None]
        worst = max(known)  # a track's precision is its worst geometry, not its best
        for row, value in zip(rows, dilutions):
            row.update(zip(("horizontal_std_m", "vertical_std_m"),
                           _std_from_dilution(value if value is not None else worst, None)))
        provenance = "hdop_model"
        if len(known) != len(dilutions):
            warnings.append(f"{len(dilutions) - len(known)} trackpoints carry no dilution and "
                            f"inherit the track's worst HDOP ({worst})")
        else:
            warnings.append("one dilution per trackpoint is used where present; fix quality "
                            "is not recorded in GPX, so precision is floored at single-fix")
    elif strict:
        raise ValueError(f"{_where(path, line=line_for(0))}: " + _UNCERTAINTY_HINT)
    else:
        provenance = "none"
    _note_span(rows, warnings)
    _record(path, dict(source="gpx", time_reference=reference,
                       altitude_datum="unspecified_by_gpx", first_time_s=0.0,
                       rows=len(rows), rejected=len(rejected),
                       uncertainty_source=provenance,
                       declared_std_m=_declared_pair(provenance, horizontal_std_m,
                                                     vertical_std_m),
                       warnings=warnings))
    return _positional_only(rows) if provenance == "none" else _six_fields(rows)


# ------------------------------------------------------------------------ contract CSV
def parse_telemetry_csv(path, *, on_malformed: str = "reject",
                        max_dropped_fraction: float = 0.05) -> list[dict]:
    """Read a file already in the survey_georef six-field contract.

    The header must match the six fields exactly. An extra column is refused by name
    because survey_georef's own reader refuses it, and a missing one is refused rather
    than defaulted; side channels belong in a companion file, not in the telemetry.
    """
    path = _open(path)
    records = _csv_records(_text_of(path))
    if not records:
        raise ValueError(f"{path.name}: file is empty")
    line, cells = records[0]
    columns = _columns(cells)
    missing = [name for name in TELEMETRY_FIELDS if name not in columns]
    extra = sorted(set(columns) - set(TELEMETRY_FIELDS))
    if missing:
        raise ValueError(f"{_where(path, line=line)}: telemetry contract CSV is missing "
                         + ", ".join(missing))
    if extra:
        raise ValueError(f"{_where(path, line=line)}: telemetry contract CSV has unexpected "
                         f"extra columns {extra}; survey_georef requires exactly "
                         f"{list(TELEMETRY_FIELDS)}, so carry side channels elsewhere")
    rows, rejected = [], []
    for number, cells in records[1:]:
        position = {"line": number}
        try:
            row = {name: _finite_number(_cell(cells, columns[name]))
                   for name in TELEMETRY_FIELDS}
            row["latitude_deg"] = _geo(row["latitude_deg"], "latitude_deg")
            row["longitude_deg"] = _geo(row["longitude_deg"], "longitude_deg")
            for name in ("horizontal_std_m", "vertical_std_m"):
                if row[name] <= 0:
                    raise ValueError(f"{name} must be positive metres, got {row[name]}: "
                                     "survey_georef treats it as an inverse weight")
            if row["t_sec"] < 0:
                raise ValueError("t_sec is negative; put the recording origin in "
                                 "metadata time_offset_s instead")
            row["t_sec"] = round(row["t_sec"], 6)
        except _RejectRow as exc:
            rejected.append(dict(message=str(exc), **position))
            continue
        except ValueError as exc:
            raise ValueError(f"{_where(path, **position)}: {exc}") from None
        rows.append({name: row[name] for name in TELEMETRY_FIELDS})
    _check_times([row["t_sec"] for row in rows], path,
                 line_of=lambda index: records[index + 1][0] if index + 1 < len(records) else None)
    _finish(path, rows, rejected, on_malformed=on_malformed,
            max_dropped_fraction=max_dropped_fraction, expected=len(rows) + len(rejected))
    warnings: list[str] = []
    _note_span(rows, warnings)
    warnings.append("altitude datum is whatever the producer of this CSV wrote: this module "
                    "cannot tell ellipsoidal from mean sea level from a bare column name")
    _record(path, dict(source="telemetry_csv", time_reference="video",
                       altitude_datum="undeclared", first_time_s=rows[0]["t_sec"] if rows else 0.0,
                       rows=len(rows), rejected=len(rejected),
                       uncertainty_source="in_file", warnings=warnings))
    return rows


# ------------------------------------------------------------------------------------ SRT
_SRT_TIME = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$")
_SRT_TOKEN = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):(.*)$")


def _timecode_seconds(token: str) -> float:
    match = _SRT_TIME.match(str(token).strip())
    if not match:
        raise _RejectRow(f"timecode {token!r} is not HH:MM:SS,mmm")
    hours, minutes, seconds, fraction = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(fraction) / 10 ** len(fraction)


def _srt_blocks(text: str) -> list[tuple[int, int, float, float, dict[str, str]]]:
    """(block index, first line, window start s, window end s, fields) per SRT block.

    Blocks are separated by blank lines, as the format requires. The index is the position
    among blocks that carry a timecode, so an error can name the block the user sees
    numbered in the file.
    """
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            if current:
                groups.append(current)
                current = []
            continue
        current.append((number, line.strip()))
    if current:
        groups.append(current)
    out, index = [], 0
    for group in groups:
        header = next(((number, line) for number, line in group if _TIMECODE.match(line)), None)
        if header is None:
            continue
        index += 1
        data = next((line for number, line in group
                     if line is not header[1] and _SRT_TOKEN.match(line)), None)
        if data is None:
            continue
        fields = {}
        for token in data.split():
            match = _SRT_TOKEN.match(token)
            if match:
                fields[match.group(1)] = match.group(2)
        left, _, right = header[1].partition("-->")
        out.append((index, group[0][0], _timecode_seconds(left), _timecode_seconds(right),
                    fields))
    return out


def parse_srt(path, *, altitude_source: str | None = None,
              horizontal_std_m: float | None = None, vertical_std_m: float | None = None,
              on_malformed: str = "reject", max_dropped_fraction: float = 0.05,
              strict: bool = True) -> list[dict]:
    """Read a DJI telemetry ``.srt`` subtitle into telemetry rows.

    Blocks are ``index``, ``HH:MM:SS,mmm --> HH:MM:SS,mmm``, then ``Key:value`` pairs.
    The block timecode is the video clock, so ``time_reference='video'`` is honest for
    these rows; a ``Time:``/``UTCTime:`` field, if present, is recorded as provenance.

    DJI writes both ``AbsoluteAltitude`` (above sea level) and ``RelativeAltitude``
    (above the takeoff point) - two different vertical datums - so the caller must
    choose with ``altitude_source='relative'|'absolute'``. Neither is ever labelled
    ellipsoidal here: the first is refused by survey_georef until a geoid separation is
    applied, the second because AGL is not a datum at all.

    Subtitles carry no fix quality or dilution, so uncertainty must be declared or
    supplied later: see ``attach_quality`` and the ``strict`` flag.
    """
    _check_pair(horizontal_std_m, vertical_std_m)
    path = _open(path)
    blocks = _srt_blocks(_text_of(path))
    if not blocks:
        raise ValueError(f"{path.name}: not telemetry - no subtitle block carries both a "
                         "timecode and Key:value data")
    keys = set()
    for _, _, _, _, fields in blocks:
        keys |= {key.lower() for key in fields}
    if not {"latitude", "longitude"} <= keys:
        raise ValueError(f"{path.name}: not telemetry - no Latitude/Longitude fields "
                         f"(seen {sorted(keys)})")
    if altitude_source not in ("relative", "absolute"):
        both = {"absolutealtitude", "relativealtitude"} <= keys
        raise ValueError(
            f"{path.name}: altitude_source must be declared as 'relative' (RelativeAltitude: "
            "height above the takeoff point) or 'absolute' (AbsoluteAltitude: above sea "
            "level)"
            + (", and this file carries both so the choice is a real one" if both else "")
            + ". survey_georef only accepts an ellipsoidal height, which neither is")
    wanted = "RelativeAltitude" if altitude_source == "relative" else "AbsoluteAltitude"
    if wanted.lower() not in keys:
        raise ValueError(f"{path.name}: altitude_source='{altitude_source}' asks for "
                         f"{wanted}, which this file does not carry (fields: "
                         f"{sorted({key for _, _, _, _, f in blocks for key in f})})")
    rows, rejected, walls, fix_series, dilutions, lines, gimbal_rows = [], [], [], [], [], [], []
    clock_warnings: list[str] = []
    for index, line, start, end, fields in blocks:
        position = {"line": line, "block": index}
        try:
            lat = _geo(_number(fields.get("Latitude"), "Latitude"), "latitude_deg")
            lon = _geo(_number(fields.get("Longitude"), "Longitude"), "longitude_deg")
            alt = _number(fields.get(wanted), wanted)
            # DJI's Time: field is milliseconds since the recording started, which is the
            # sample instant; the subtitle window's start is the fallback for exports that
            # omit it, since a block's window is a display duration, not a measurement time.
            declared = fields.get("Time")
            video_time = (_finite_number(declared, "Time") / 1000.0 if declared not in
                          (None, "") else start)
            if declared in (None, ""):
                clock_warnings.append("block "
                                      f"{index} has no Time: field, so its subtitle window "
                                      "start is used as the sample instant")
            elif abs(video_time - start) > 1.0:
                clock_warnings.append(f"block {index}: Time:{declared} disagrees with the "
                                      "subtitle window start by more than a second, so which "
                                      "one is the sample instant is unresolved")
        except _RejectRow as exc:
            if on_malformed == "reject":
                raise ValueError(f"{_where(path, **position)}: {exc}") from None
            rejected.append(dict(message=str(exc), **position))
            continue
        except ValueError as exc:
            raise ValueError(f"{_where(path, **position)}: {exc}") from None
        angles = {}
        for name, key in (("pitch_deg", "GimbalPitch"), ("roll_deg", "GimbalRoll"),
                          ("yaw_deg", "GimbalYaw")):
            if fields.get(key) not in (None, ""):
                try:
                    angles[name] = _finite_number(fields[key], key)
                except (ValueError, _RejectRow):
                    continue  # a broken gimbal angle is a side channel, not a lost fix
        fix = fields.get("RtkQuality") or fields.get("PositioningType") or fields.get("FixType")
        dilution = None
        for key in ("Hdop", "HDOP", "dhdop"):
            if fields.get(key) not in (None, ""):
                try:
                    dilution = _finite_number(fields[key], key)
                except (ValueError, _RejectRow):
                    dilution = None
                break
        wall = None
        for key in ("UTCTime", "AbsTime", "GpsDateTime"):
            if fields.get(key):
                try:
                    wall = _time_epoch(fields[key], what=key)
                except (ValueError, _RejectRow):
                    wall = None
                break
        rows.append(dict(t_sec=video_time, latitude_deg=lat, longitude_deg=lon, altitude_m=alt))
        lines.append(line)
        gimbal_rows.append(angles)
        walls.append(wall)
        fix_series.append(fix if fix not in (None, "") else None)
        dilutions.append(dilution)
    _check_times([row["t_sec"] for row in rows], path, line_of=lines.__getitem__)
    _finish(path, rows, rejected, on_malformed=on_malformed,
            max_dropped_fraction=max_dropped_fraction, expected=len(rows) + len(rejected))
    warnings: list[str] = list(dict.fromkeys(clock_warnings))
    if altitude_source == "absolute":
        warnings.append("DJI AbsoluteAltitude is referenced to mean sea level: turning it "
                        "into an ellipsoidal height needs a geoid model (EGM96/EGM2008) "
                        "that is not available here, so survey_georef will refuse it until "
                        "that separation is applied")
    else:
        warnings.append("DJI RelativeAltitude is height above the takeoff point (AGL), which "
                        "is not a geodetic datum: it cannot be called ellipsoidal")
    provenance = "none"
    if horizontal_std_m is not None:
        _apply_declared_uncertainty(rows, horizontal_std_m, vertical_std_m)
        provenance = "declared_by_caller"
    elif any(f is not None for f in fix_series) or any(d is not None for d in dilutions):
        try:
            qualities = [None if f is None else parse_fix_quality(f) for f in fix_series]
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from None
        for row, quality, dilution in zip(rows, qualities, dilutions):
            row.update(zip(("horizontal_std_m", "vertical_std_m"),
                           _std_from_dilution(dilution, quality)))
        provenance = "fix_quality" if any(q for q in qualities) else "hdop_model"
    elif strict:
        raise ValueError(f"{_where(path, line=blocks[0][1])}: " + _UNCERTAINTY_HINT)
    if provenance != "none":
        _note_span(rows, warnings)
    _record(path, dict(source="srt", time_reference="video",
                       altitude_datum="relative_to_takeoff" if altitude_source == "relative"
                       else "mean_sea_level",
                       altitude_source=altitude_source,
                       first_time_s=rows[0]["t_sec"] if rows else 0.0,
                       rows=len(rows), rejected=len(rejected),
                       wall_clock_s=walls if any(w is not None for w in walls) else None,
                       gimbal=gimbal_rows[:len(rows)], fix_quality=[f for f in fix_series if f],
                       uncertainty_source=provenance,
                       declared_std_m=_declared_pair(provenance, horizontal_std_m,
                                                     vertical_std_m),
                       warnings=warnings))
    return _positional_only(rows) if provenance == "none" else _six_fields(rows)


# -------------------------------------------------------------------------------- DJI CSV
_DJI_LAT = {"lat", "latitude", "latitude_deg", "gps_latitude", "gps.latitude"}
_DJI_LON = {"lon", "lng", "longitude", "longitude_deg", "gps_longitude", "gps.longitude"}
_DJI_TIME = {"time", "timestamp", "datetime", "date", "utc_time", "gps_time", "abs_time",
             "system_time", "gps_date_time", "t_sec", "t_ms"}
_DJI_RELATIVE = {"relative_altitude", "relativealtitude", "elevation", "agl", "height",
                 "flight_agl", "above_start_alt"}
_DJI_ABSOLUTE = {"absolute_altitude", "absolutealtitude", "altitude", "alt", "real_asl",
                 "asml", "gps_altitude", "flight_asl"}
_DJI_FIX = {"rtk_quality", "rtk_fix", "rtkquality", "fix_type", "gpsdim", "gps_type",
            "positioning_type", "gps_quality"}
_DJI_HDOP = {"hdop", "dhdop", "hdops"}
_DJI_SATS = {"gps_number", "gps_number_total", "satellites_num", "satellite_number",
             "satellites_num_total"}


def parse_dji_csv(path, *, altitude_source: str = "auto",
                  horizontal_std_m: float | None = None,
                  vertical_std_m: float | None = None, on_malformed: str = "reject",
                  max_dropped_fraction: float = 0.05, strict: bool = True) -> list[dict]:
    """Read a ``dji_telemetry.*`` CSV export from DJI Pilot or DJI Assistant.

    Column names match case-insensitively with or without the ``dji_telemetry.`` prefix,
    so an export flattened by a spreadsheet still parses. Time is read as a UTC wall
    clock (milliseconds since the epoch, or ``YYYY-MM-DD HH:MM:SS``) and made relative to
    the first row, so the provenance says ``gnss_wall_clock`` and the video anchor stays
    the caller's assumption.

    Heights: with both present, ``relative_altitude``/``elevation`` (above the takeoff
    point) wins over ``absolute_altitude`` (above sea level), because AGL is the figure
    the tag states; pass ``altitude_source='absolute'`` to insist on sea level. Neither is
    ever labelled ellipsoidal here.

    Uncertainty: a fix column (``rtk_quality``, ``gpsdim``, ``fix_type``, ...) is mapped
    through the documented table, with HDOP folded in as degraded-geometry inflation.
    Otherwise the file needs declared standard deviations or ``strict=False``.
    """
    _check_pair(horizontal_std_m, vertical_std_m)
    if altitude_source not in ("auto", "relative", "absolute"):
        raise ValueError("altitude_source must be 'auto', 'relative' or 'absolute'")
    path = _open(path)
    records = _csv_records(_text_of(path))
    if not records:
        raise ValueError(f"{path.name}: file is empty")
    line, cells = records[0]
    columns = {name.split(".")[-1]: index
               for name, index in _columns(cells).items()}
    found = {"lat": _pick(columns, _DJI_LAT), "lon": _pick(columns, _DJI_LON),
             "time": _pick(columns, _DJI_TIME), "relative": _pick(columns, _DJI_RELATIVE),
             "absolute": _pick(columns, _DJI_ABSOLUTE), "fix": _pick(columns, _DJI_FIX),
             "hdop": _pick(columns, _DJI_HDOP), "sats": _pick(columns, _DJI_SATS),
             "gimbal_pitch": _pick(columns, {"gimbal_pitch"}),
             "gimbal_roll": _pick(columns, {"gimbal_roll"}),
             "gimbal_yaw": _pick(columns, {"gimbal_yaw"})}
    if found["lat"] is None or found["lon"] is None:
        raise ValueError(f"{_where(path, line=line)}: no latitude/longitude column in "
                         f"{sorted(columns)}; GPS position is a mandatory input")
    if found["relative"] is None and found["absolute"] is None:
        raise ValueError(f"{_where(path, line=line)}: no altitude column in {sorted(columns)}"
                         " (looked for relative_altitude/elevation/height and "
                         "absolute_altitude/altitude); GPS height is mandatory")
    if altitude_source == "absolute" and found["absolute"] is None:
        raise ValueError(f"{path.name}: altitude_source='absolute' but no absolute altitude "
                         "column is present")
    chosen = ("relative" if found["relative"] is not None and altitude_source != "absolute"
              else "absolute")
    height_at = found[chosen]
    rows, rejected = [], []
    walls, fixes, dilutions, gimbal, satellites = [], [], [], [], []
    for number, cells in records[1:]:
        position = {"line": number}
        try:
            raw_time = _cell(cells, found["time"])
            if raw_time is None or str(raw_time).strip() == "":
                raise _RejectRow("row carries no timestamp, so it cannot be placed on the "
                                 "video clock")
            wall = _time_epoch(raw_time, what="DJI time")
            lat = _geo(_number(_cell(cells, found["lat"])), "latitude_deg")
            lon = _geo(_number(_cell(cells, found["lon"])), "longitude_deg")
            alt = _number(_cell(cells, height_at))
            dilution = None
            if found["hdop"] is not None and str(_cell(cells, found["hdop"]) or "").strip():
                dilution = _finite_number(_cell(cells, found["hdop"]))
        except _RejectRow as exc:
            if on_malformed == "reject":
                raise ValueError(f"{_where(path, **position)}: {exc}") from None
            rejected.append(dict(message=str(exc), **position))
            continue
        except ValueError as exc:
            raise ValueError(f"{_where(path, **position)}: {exc}") from None
        raw_fix = _cell(cells, found["fix"]) if found["fix"] is not None else None
        walls.append(wall)
        fixes.append(raw_fix.strip() if isinstance(raw_fix, str) else raw_fix)
        dilutions.append(dilution)
        satellites.append(_safe(_cell(cells, found["sats"])) if found["sats"] is not None else None)
        angles = {}
        for name in ("pitch", "roll", "yaw"):
            index = found["gimbal_" + name]
            value = _safe(_cell(cells, index)) if index is not None else None
            if value is not None:
                angles[name + "_deg"] = value
        gimbal.append(angles)
        rows.append(dict(t_sec=wall, latitude_deg=lat, longitude_deg=lon, altitude_m=alt))
    origin = walls[0] if walls else 0.0
    for row, wall in zip(rows, walls):
        row["t_sec"] = round(wall - origin, 6)
    _check_times([row["t_sec"] for row in rows], path,
                 line_of=lambda index: records[index + 1][0] if index + 1 < len(records) else None)
    _finish(path, rows, rejected, on_malformed=on_malformed,
            max_dropped_fraction=max_dropped_fraction, expected=len(rows) + len(rejected))
    warnings: list[str] = []
    known_fixes = [f for f in fixes if f not in (None, "")]
    if horizontal_std_m is not None:
        _apply_declared_uncertainty(rows, horizontal_std_m, vertical_std_m)
        provenance = "declared_by_caller"
    elif known_fixes or any(d is not None for d in dilutions):
        try:
            qualities = [None if f in (None, "") else parse_fix_quality(f) for f in fixes]
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from None
        for row, quality, dilution in zip(rows, qualities, dilutions):
            row.update(zip(("horizontal_std_m", "vertical_std_m"),
                           _std_from_dilution(dilution, quality)))
        provenance = ("fix_quality" if known_fixes and not any(dilutions)
                      else "hdop_model" if not known_fixes else "fix_quality+hdop_model")
        if any(q is None for q in qualities):
            warnings.append("fix quality is absent on some rows, which are floored at the "
                            "unknown/single-fix class")
        if any(d is not None and d > HDOP_DEGRADED_LIMIT for d in dilutions):
            warnings.append("a phase-lock fix claim beside HDOP above "
                            f"{HDOP_DEGRADED_LIMIT} was inflated by HDOP/"
                            f"{HDOP_DEGRADED_LIMIT}: RTK fix cannot be confirmed from "
                            "geometry alone")
    elif strict:
        raise ValueError(f"{_where(path, line=line)}: " + _UNCERTAINTY_HINT)
    else:
        provenance = "none"
    if chosen == "absolute":
        warnings.append("absolute altitude is DJI's height above sea level, not a height "
                        "above the WGS84 ellipsoid: an EGM96/EGM2008 geoid separation would "
                        "be needed and none is applied here")
    else:
        warnings.append("relative altitude is height above the takeoff point (AGL), which is "
                        "not a geodetic datum: survey_georef needs an ellipsoidal height")
    if provenance != "none":
        _note_span(rows, warnings)
    _record(path, dict(source="dji_csv", time_reference="gnss_wall_clock",
                       altitude_datum="relative_to_takeoff" if chosen == "relative"
                       else "mean_sea_level", altitude_source=chosen, first_time_s=0.0,
                       rows=len(rows), rejected=len(rejected), wall_clock_s=walls,
                       gimbal=gimbal, satellites=[s for s in satellites if s is not None],
                       fix_quality=[f for f in fixes if f not in (None, "")],
                       uncertainty_source=provenance,
                       declared_std_m=_declared_pair(provenance, horizontal_std_m,
                                                     vertical_std_m),
                       warnings=warnings))
    return _positional_only(rows) if provenance == "none" else _six_fields(rows)


def _safe(token):
    try:
        return _finite_number(token)
    except (ValueError, _RejectRow):
        return None


# ---------------------------------------------------------------------------- dispatch
_PARSERS = {"gpx": "parse_gpx", "srt": "parse_srt", "dji_csv": "parse_dji_csv",
            "telemetry_csv": "parse_telemetry_csv"}


def from_flight_log(path, **kwargs) -> list[dict]:
    """Parse a drone flight log by sniffing its content, never its extension alone.

    Keyword arguments are forwarded to the parser that was selected; one a given parser
    does not take (``altitude_source`` for a GPX, say) is dropped, so a caller can pass
    one options dict at every stage. A DJI SRT subtitle still needs the caller to declare
    ``altitude_source``, because that is a datum choice, not a dispatch detail.
    """
    path = _open(path)
    detected = sniff_format(path)
    name = _PARSERS.get(detected.kind)
    if name is None:
        hint = {"imu": "use survey_inputs.read_imu",
                "barometer": "use survey_inputs.read_barometer"}.get(
                    detected.kind,
                    "recognised kinds are gpx, srt, dji_csv and telemetry_csv")
        raise ValueError(f"{path.name}: unrecognised flight-log format ({detected.detail}); {hint}")
    target = globals()[name]
    accepted = set(inspect.signature(target).parameters)
    return target(path, **{key: value for key, value in kwargs.items() if key in accepted})


# ------------------------------------------------------------------- attaching uncertainty
@dataclass(frozen=True)
class QualityResult:
    """Outcome of mapping evidence into the two standard deviations.

    ``rows`` satisfy the six-field contract and are safe to write for
    survey_georef.``normalize_telemetry``. ``dropped`` names every row that was refused
    and why, so an unusable stretch of a flight is visible rather than silently absent.
    """
    rows: list[dict]
    dropped: list[dict] = field(default_factory=list)
    fix_quality_basis: str = "unknown"
    uncertainty_source: str = "none"
    warnings: list[str] = field(default_factory=list)
    model: dict = field(default_factory=dict)


def _as_pairs(series, *, name: str) -> list[tuple[float, object]]:
    if series is None:
        return []
    if isinstance(series, dict):
        pairs = sorted((float(key), value) for key, value in series.items())
    else:
        pairs = []
        for row in series:
            if len(row) != 2:
                raise ValueError(f"{name} entries must be (t_sec, value) pairs")
            pairs.append((float(row[0]), row[1]))
    for index in range(1, len(pairs)):
        if not pairs[index][0] > pairs[index - 1][0]:
            raise ValueError(f"{name} times must be strictly increasing; got "
                             f"{pairs[index - 1][0]} then {pairs[index][0]}")
    return pairs


def _sample(pairs: Sequence[tuple], time: float, gap: float):
    """("exact", value, None) | ("bracket", left, right) | None. Never extrapolated."""
    if not pairs:
        return None
    index = int(np.searchsorted(np.array([row[0] for row in pairs], dtype=float), time))
    if index < len(pairs) and pairs[index][0] == time:
        return ("exact", pairs[index][1], None)
    if index == 0 or index >= len(pairs):
        return None
    if pairs[index][0] - pairs[index - 1][0] > gap:
        return None
    return ("bracket", pairs[index - 1][1], pairs[index][1])


def attach_quality(rows, fix_quality_by_time=None, hdop_by_time=None, *,
                   max_quality_gap_s: float = 10.0) -> QualityResult:
    """Turn RTK/PPK fix quality and HDOP evidence into honest standard deviations.

    ``rows`` may be positional-only (the four keys a parser returns with
    ``strict=False``) or already carry the two std-dev keys. ``fix_quality_by_time`` and
    ``hdop_by_time`` are ``[(t_sec, value), ...]`` or dicts keyed by time, strictly
    increasing; fix tokens go through ``parse_fix_quality``.

    Every rule points the conservative way, and the model constants are echoed in
    ``result.model`` so a report can quote them:
      * ``no_fix`` rows are dropped and listed in ``dropped`` with the reason - a position
        with no fix is not a position with poor precision;
      * a row between two *different* fix classes takes the worse class, since the
        ambiguity is exactly when the receiver was gaining or losing lock;
      * a row with no fix sample within ``max_quality_gap_s`` becomes ``unknown``, which
        floors at the pessimistic single-fix class, and warns;
      * HDOP alone is modelled as HDOP x ``DOP_TO_METRE_UERE``, floored at single-fix,
        because dilution cannot show that corrections were applied, how many satellites
        were tracked, or that a multipath-biased fix was any good;
      * where a row declares its own std dev, the **larger** of declared and modelled is
        kept: this function never reduces an uncertainty someone else asserted;
      * with no fix quality, no HDOP and no declared std dev anywhere, it raises rather
        than inventing precision.

    Interpolated HDOP takes the worse of the two bracketing samples. None of this is a
    measured error budget, and no part of it removes a correlated GNSS bias.
    """
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        raise ValueError("attach_quality needs a non-empty rows sequence")
    for row in source_rows:
        for name in POSITIONAL_FIELDS:
            if name not in row:
                raise ValueError(f"attach_quality needs {name} on every row; parse with "
                                 "strict=False and keep the positional fields")
            row[name] = _finite_number(row[name])
        row["latitude_deg"] = _geo(row["latitude_deg"], "latitude_deg")
        row["longitude_deg"] = _geo(row["longitude_deg"], "longitude_deg")
        for name in ("horizontal_std_m", "vertical_std_m"):
            if name in row and row[name] is not None:
                value = row[name] = _finite_number(row[name])
                if value <= 0:
                    raise ValueError(f"{name} at t={row['t_sec']} must be positive metres")
    times = [row["t_sec"] for row in source_rows]
    for index in range(1, len(times)):
        if not times[index] > times[index - 1]:
            raise ValueError("row times must be strictly increasing before quality is "
                             f"attached: {times[index - 1]} then {times[index]}")
    quality = [(stamp, parse_fix_quality(token)) for stamp, token
               in _as_pairs(fix_quality_by_time, name="fix_quality_by_time")]
    dilution = _as_pairs(hdop_by_time, name="hdop_by_time")
    dilution = [(stamp, _finite_number(value)) for stamp, value in dilution]
    for stamp, value in dilution:
        if value <= 0:
            raise ValueError(f"hdop {value} at t={stamp} must be a positive dilution factor")
    gap = float(max_quality_gap_s)
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("max_quality_gap_s must be a positive finite number of seconds")
    if not quality and not dilution:
        if all("horizontal_std_m" not in row for row in source_rows):
            raise ValueError("attach_quality has nothing to map: " + _UNCERTAINTY_HINT)
    warnings: list[str] = []
    kept, dropped = [], []
    sources: set[str] = set()
    basis = "per_sample_rtk_or_ppk" if quality else "unknown"
    transition = gapped = inflated = widened = False
    for row in source_rows:
        time = row["t_sec"]
        fix = None
        if quality:
            hit = _sample(quality, time, gap)
            if hit is None:
                fix, gapped = "unknown", True
            elif hit[0] == "exact":
                fix = hit[1]
            else:
                fix = _worse(hit[1], hit[2])
                transition = transition or hit[1] != hit[2]
            sources.add("fix_quality")
        hdop = None
        if dilution:
            hit = _sample(dilution, time, gap)
            if hit is None:
                hdop = max(value for _, value in dilution)  # pessimistic, never a guess
            else:
                hdop = float(hit[1]) if hit[2] is None else max(float(hit[1]), float(hit[2]))
            sources.add("hdop_model")
        declared_h = row.get("horizontal_std_m")
        declared_v = row.get("vertical_std_m")
        if (declared_h is None) != (declared_v is None):
            raise ValueError(f"the row at t={time} declares only one of horizontal_std_m / "
                             "vertical_std_m; an uncertainty is a pair")
        if fix == "no_fix":
            dropped.append(dict(time_s=time, reason="unusable_fix_quality", fix_quality=fix,
                                row={name: row[name] for name in POSITIONAL_FIELDS}))
            continue
        if fix is None and hdop is None:
            # Nothing was supplied to map, so this is the caller's own declaration being
            # carried through, not a modelled figure: no floor is applied to it.
            if declared_h is None:
                raise ValueError(f"attach_quality has nothing to map at t={time}: "
                                 + _UNCERTAINTY_HINT)
            sources.add("declared_by_caller")
            kept.append({**{name: row[name] for name in POSITIONAL_FIELDS},
                         "horizontal_std_m": declared_h, "vertical_std_m": declared_v})
            continue
        model_h, model_v = _std_from_dilution(hdop, fix)
        if fix in ("float", "fixed") and hdop is not None and hdop > HDOP_DEGRADED_LIMIT:
            inflated = True
        horizontal, vertical = model_h, model_v
        if declared_h is not None:
            horizontal = max(declared_h, model_h)
            vertical = max(declared_v, model_v)
            sources.add("declared_by_caller")
            # Any disagreement is worth reporting, in either direction: the row ends up
            # with the larger figure, so a tighter declared claim is being set aside.
            widened = widened or declared_h != model_h or declared_v != model_v
        kept.append({**{name: row[name] for name in POSITIONAL_FIELDS},
                     "horizontal_std_m": round(horizontal, 4),
                     "vertical_std_m": round(vertical, 4)})
    if not kept:
        raise ValueError("attach_quality dropped every row: this track has no usable fix "
                         "quality at any timestamp")
    if gapped:
        basis = "unknown"
        warnings.append("no fix-quality sample within max_quality_gap_s of some rows; those "
                        "rows are floored at the unknown/single-fix class instead of "
                        "inheriting a nearby RTK claim")
    if transition:
        warnings.append("a fix-quality transition falls inside the track: the worse of the "
                        "two bracketing classes is used, never the better")
    if inflated:
        warnings.append("a phase-lock fix claim beside degraded HDOP was inflated by "
                        f"HDOP/{HDOP_DEGRADED_LIMIT}: the fix cannot be confirmed from "
                        "geometry alone")
    if quality and not dilution:
        warnings.append("no dilution series was supplied, so fix quality is taken at face "
                        "value; a fixed-RTK flag with three satellites in view still biases")
    if dilution and not quality:
        warnings.append("HDOP alone says nothing about the corrections used or the satellite "
                        "count, so precision is floored at the single-fix class")
    if widened:
        warnings.append("the larger of declared and modelled uncertainty was kept: this "
                        "function never reduces an uncertainty the caller asserted")
    _note_span(kept, warnings)
    warnings.append("modelled standard deviations are a conservative fix-type/dilution "
                    "estimate, not a measured error budget, and cannot remove a correlated "
                    "GNSS bias that shifts the whole trajectory")
    active = sources - {"declared_by_caller"}
    if active:
        source = "+".join(sorted(active)) if len(active) > 1 else next(iter(active))
    else:
        source = "declared_by_caller"
        basis = "declared_by_caller"
    return QualityResult(rows=kept, dropped=dropped, fix_quality_basis=basis,
                         uncertainty_source=source, warnings=warnings,
                         model=dict(uere_m=DOP_TO_METRE_UERE,
                                    vertical_ratio=VERTICAL_DOP_RATIO,
                                    degraded_hdop_limit=HDOP_DEGRADED_LIMIT,
                                    min_sample_span_m=MIN_SAMPLE_SPAN_M,
                                    fix_quality_std_m=dict(FIX_QUALITY_STD_M)))


# ------------------------------------------------------------------ optional sensor streams
_TIME_HEADERS = ("t_sec", "timestamp", "time", "t", "t_ms", "timestamp_ms", "system_time",
                 "gps_time", "imu_time", "baro_time", "datetime", "date")
_IMU_AXES = (
    ("ax", "acceleration", ("ax", "accel_x", "acc_x", "accelerometer_x", "acceleration_x")),
    ("ay", "acceleration", ("ay", "accel_y", "acc_y", "accelerometer_y", "acceleration_y")),
    ("az", "acceleration", ("az", "accel_z", "acc_z", "accelerometer_z", "acceleration_z")),
    ("gx", "angular_velocity", ("gx", "gyro_x", "gyroscope_x", "angular_velocity_x")),
    ("gy", "angular_velocity", ("gy", "gyro_y", "gyroscope_y", "angular_velocity_y")),
    ("gz", "angular_velocity", ("gz", "gyro_z", "gyroscope_z", "angular_velocity_z")),
)
_BARO_COLUMNS = (
    ("pressure_hpa", "pressure", ("pressure_hpa", "barometric_pressure", "baro_pressure",
                                  "abs_pressure", "static_pressure", "pressure")),
    ("baro_altitude_m", "baro_altitude", ("baro_altitude", "baro_altitude_m",
                                          "barometric_altitude", "altimeter_m", "baro_alt")),
)
_SENSOR_UNITS = {"acceleration": "m/s^2", "angular_velocity": "rad/s", "pressure": "hPa",
                 "baro_altitude": "m relative to the logged reference"}
_GYRO_UNITS = {"rad/s": 1.0, "deg/s": math.pi / 180.0}


@dataclass(frozen=True)
class SensorSeries:
    """A validated, timestamped, unit-declared sensor stream. It is never a position."""
    source: str
    path: str
    columns: tuple[str, ...]
    values: np.ndarray
    times_s: np.ndarray
    units: dict
    source_units: dict
    source_time_unit: str
    n_rows: int
    dt_median_s: float | None
    rejected: int
    rejected_detail: list[dict]

    def summarise(self) -> dict:
        """What this stream can and cannot constrain. ``can`` is empty by design."""
        cannot = {
            "imu": [
                "Position by dead reckoning: a consumer MEMS IMU's bias drifts faster than "
                "it is observable on a single ~10 minute pass, so double-integrated "
                "acceleration diverges quadratically with time and is not a metric ruler",
                "Absolute height, geodetic position, or scale: an IMU senses specific force "
                "and angular rate, not where it is",
                "Camera timestamps in video time: nothing here ties these samples to video "
                "frames, so any offset would be invented",
                "Anything survey_georef consumes: this stream is never turned into a camera "
                "position by this module"],
            "barometer": [
                "Absolute or ellipsoidal height: pressure altitude is relative to whatever "
                "reference the aircraft chose at takeoff",
                "Height above ground: a barometer reads pressure, not distance to the "
                "surface beneath the drone",
                "Metre-level vertical accuracy: weather-driven pressure change alone is "
                "several metres of apparent height over a flight",
                "Position: a barometer is not a substitute for GNSS",
                "Height: pressure is not converted to height here, because no reference "
                "pressure, temperature profile or datum was declared"],
        }[self.source]
        cannot = list(cannot)
        if set(self.columns) >= {"pressure_hpa", "baro_altitude_m"}:
            cannot.append("the logged barometric altitude and the pressure column are never "
                          "combined or cross-checked into one height: both are reported as given")
        return dict(source=f"{self.source}:{Path(self.path).name}", can=[], cannot=cannot,
                    provides_absolute_position=False, provides_metric_scale=False,
                    units=dict(self.units), columns=list(self.columns), n_rows=self.n_rows,
                    dt_median_s=self.dt_median_s, rejected=self.rejected,
                    rejected_detail=list(self.rejected_detail),
                    note="optional input under SIH26158: corroborating evidence only, never "
                         "the source of the georeference")


def _read_sensor(path, *, kind: str, schema, time_unit="auto", gyro_unit="deg/s"):
    path = _open(path)
    if gyro_unit not in _GYRO_UNITS:
        raise ValueError(f"gyro_unit must be one of {sorted(_GYRO_UNITS)}, not {gyro_unit!r}")
    if time_unit not in ("auto", "s", "ms"):
        raise ValueError("time_unit must be 's', 'ms' or 'auto'")
    records = _csv_records(_text_of(path))
    if not records:
        raise ValueError(f"{path.name}: file is empty")
    line, cells = records[0]
    columns = _columns(cells)
    time_at = _pick(columns, _TIME_HEADERS)
    if time_at is None:
        raise ValueError(f"{_where(path, line=line)}: no time column found among columns "
                         f"{sorted(columns)}; tried " + ", ".join(_TIME_HEADERS))
    header_time = cells[time_at].split(".")[-1].lower()
    forced = time_unit
    if forced == "auto":
        forced = "ms" if re.search(r"(^|_)(ms|millis|millisecond)", header_time) else "s"
    resolved = []
    for name, family, aliases in schema:
        index = _pick(columns, aliases)
        if index is not None:
            resolved.append((name, family, index))
    if not resolved:
        raise ValueError(f"{_where(path, line=line)}: no {kind} columns in {sorted(columns)}; "
                         "expected a pressure or altitude column, or accelerometer and "
                         "gyroscope axes, one of " + ", ".join(
                             ", ".join(aliases) for _, _, aliases in schema))
    families = {family for _, family, _ in resolved}
    for family in ("acceleration", "angular_velocity"):
        if family in families and sum(1 for _, item, _ in resolved if item == family) != 3:
            raise ValueError(f"{_where(path, line=line)}: an {kind} stream needs all three "
                             f"{family} axes; a partial axis set is not an orientation or a "
                             "specific-force vector")
    units = {name: _SENSOR_UNITS[family] for name, family, _ in resolved}
    scale = _GYRO_UNITS[gyro_unit]
    gyro_names = {name for name, family, _ in resolved if family == "angular_velocity"}
    rows, rejected, stamps = [], [], []
    for number, cells in records[1:]:
        position = {"line": number}
        try:
            time = _finite_number(_cell(cells, time_at), cells[time_at])
            values = [_number(_cell(cells, index), name)
                      for (name, _, index) in resolved]
        except _RejectRow as exc:
            rejected.append(dict(message=str(exc), **position))
            continue
        except ValueError as exc:
            raise ValueError(f"{_where(path, **position)}: {exc}") from None
        stamps.append(time / 1000.0 if forced == "ms" else time)
        rows.append([value * (scale if name in gyro_names else 1.0)
                     for (name, _, _), value in zip(resolved, values)])
    if not rows:
        raise ValueError(f"{path.name}: no usable {kind} rows; all {len(rejected)} data rows "
                         "were rejected")
    times = np.asarray(stamps, dtype=float)
    for index in range(1, len(times)):
        if not times[index] > times[index - 1]:
            raise ValueError(f"{path.name}: {kind} times must be monotonic and strictly "
                             f"increasing; {times[index - 1]} is followed by {times[index]}. "
                             "Rejected rows are never reordered or deduplicated to fix this")
    source_units = {family: ("deg/s" if family == "angular_velocity" and gyro_unit == "deg/s"
                             else "rad/s" if family == "angular_velocity"
                             else _SENSOR_UNITS[family]) for family in sorted(families)}
    series = SensorSeries(source=kind, path=str(path),
                          columns=tuple(name for name, _, _ in resolved),
                          values=np.asarray(rows, dtype=float), times_s=times, units=units,
                          source_units=source_units, source_time_unit=forced,
                          n_rows=len(times),
                          dt_median_s=float(np.median(np.diff(times))) if len(times) > 1 else None,
                          rejected=len(rejected), rejected_detail=rejected)
    _record(path, dict(source=kind, rows=series.n_rows, rejected=len(rejected),
                       time_unit=forced, columns=list(series.columns), warnings=[]))
    return series


def read_imu(path, *, time_unit="auto", gyro_unit: str = "deg/s") -> SensorSeries:
    """Read an accelerometer and gyroscope log; the output is SI and never a position.

    Acceleration is taken as logged in m/s^2. Gyro defaults to DJI's deg/s and is
    converted to rad/s, which ``units`` and ``source_units`` both state; pass
    ``gyro_unit='rad/s'`` for a stream that is already SI instead of letting anyone infer
    units from magnitudes. ``time_unit='auto'`` reads milliseconds from a ``*_ms`` column
    header and seconds otherwise, and ``source_time_unit`` records the choice. Rows whose
    numbers cannot be read are counted in ``rejected``; NaN, empty or non-monotonic times
    are hard errors that name the line.
    """
    return _read_sensor(path, kind="imu", schema=_IMU_AXES, time_unit=time_unit,
                        gyro_unit=gyro_unit)


def read_barometer(path, *, time_unit="auto") -> SensorSeries:
    """Read a barometric log, keeping pressure and any logged altitude unconverted.

    A pressure column stays in hPa and a logged barometric altitude stays in metres;
    neither becomes the other, because pressure-to-height needs a reference pressure, a
    temperature profile and a declared datum, none of which the file supplies.
    ``summarise()`` states the limits, and ``can`` is empty for both sensors: an optional
    input under SIH26158 must not be mistaken for part of the georeference.
    """
    return _read_sensor(path, kind="barometer", schema=_BARO_COLUMNS, time_unit=time_unit)


# -------------------------------------------------------------------------- video probing
def probe_video(path) -> dict | None:
    """Frames, fps and duration from the container, or None when the probe fails.

    Duration is ``frame_count / fps`` with no re-decode. For variable-frame-rate footage
    that can drift by a few percent, and a wrong duration is a live hazard because
    survey_georef bounds every telemetry time by it, so a caller wanting exact bounds
    declares the duration instead.
    """
    try:
        import cv2
    except ImportError:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not (math.isfinite(frames) and math.isfinite(fps) and frames > 0 and fps > 0):
        return None
    return dict(native_frames=int(frames), fps=fps, duration_s=frames / fps,
                source="container_frames_and_fps")


# ---------------------------------------------------------------------------- metadata
@dataclass(frozen=True)
class Blocker:
    """A claim this pipeline cannot make yet, and the evidence that would change it."""
    code: str
    message: str
    field: str
    severity: str = "blocker"
    unblocking_evidence: str = ""


_TIME_REFERENCES = {"video", "gnss_wall_clock", "camera_internal", "gps_leap_unsmoothed"}
_ALTITUDE_DATUMS = {"ellipsoidal", "AGL-relative", "mean_sea_level", "unspecified_by_gpx",
                    "undeclared"}
_POSITION_REFERENCES = {"camera_center", "gnss_antenna"}
_ALLOWED_OVERRIDES = {"time_reference", "time_offset_s", "video_duration_s", "altitude_datum",
                      "altitude_datum_basis", "position_reference", "antenna_offset_applied",
                      "lever_arm_uncertainty_m", "lever_arm_negligible", "single_pass",
                      "horizontal_std_m", "vertical_std_m", "camera_id"}
_OVERREACH = {"measured_accuracy_m", "accuracy_m", "rmse_m", "ground_control",
              "control_points", "gcp_count", "validated", "externally_validated",
              "official_target_met", "ce_90", "le_90", "sub_metre"}
_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy", "width", "height")
_SOURCE_DATUMS = {"relative_to_takeoff": "AGL-relative", "mean_sea_level": "mean_sea_level",
                  "unspecified_by_gpx": "unspecified_by_gpx", "ellipsoidal": "ellipsoidal"}


def merge_metadata(video_path=None, calibration_json=None, overrides=None,
                   source_info=None) -> dict:
    """Build the metadata survey_georef demands, declaring only what has evidence.

    survey_georef validates ``schema_version``, ``time_reference``, ``time_offset_s``,
    ``altitude_datum``, ``position_reference``, ``single_pass`` and ``video_duration_s``;
    everything added here is extra and survives its deepcopy. ``blockers`` lists
    ``Blocker`` objects and ``unresolved`` repeats their codes for JSON consumers.

    A blocker is never cleared by relabelling:
      * ``altitude_datum`` becomes ``ellipsoidal`` only when the caller asserts it, with
        ``altitude_datum_basis`` naming the receiver output or the geoid model applied. A
        takeoff-relative height is declared ``AGL-relative`` and an above-sea-level one
        ``mean_sea_level``: both are blockers, because survey_georef accepts only
        ellipsoidal, and saying so is the point.
      * ``position_reference`` defaults to ``gnss_antenna``, which survey_georef rejects by
        design - a GNSS solution is an antenna phase centre, not a camera centre. It
        becomes ``camera_center`` only with a declared lever arm
        (``antenna_offset_applied=True`` plus ``lever_arm_uncertainty_m``) or a declared
        negligible offset; declaring ``camera_center`` with neither is blocked too.
      * ``time_reference`` is ``video`` for the rows the parsers emit, since they are
        seconds from a declared origin, but when that origin came from a wall-clock log the
        anchor itself is an assumption and is reported as ``telemetry_anchor_assumed``.
      * ``video_duration_s`` is probed from the container or required from the caller.
      * ``single_pass`` is a boolean; a string description of flight geometry is refused.
    """
    overrides = dict(overrides or {})
    overreaching = sorted(set(overrides) & _OVERREACH)
    if overreaching:
        raise ValueError(f"overrides {overreaching} would assert a measurement this pipeline "
                         "has not made (overreach): report measured error separately")
    unknown = sorted(set(overrides) - _ALLOWED_OVERRIDES)
    if unknown:
        raise ValueError(f"unknown override {unknown}; supported keys are "
                         f"{sorted(_ALLOWED_OVERRIDES)}")
    info = dict(source_info or {})
    blockers: list[Blocker] = []
    warnings: list[str] = []

    def block(code, message, name, evidence=""):
        blockers.append(Blocker(code=code, message=message, field=name,
                                unblocking_evidence=evidence))

    single_pass = overrides.get("single_pass", True)
    if not isinstance(single_pass, bool):
        raise ValueError(f"single_pass must be True or False; {single_pass!r} is an "
                         "unsupported flight description - declare the geometry as a "
                         "boolean and keep the prose in a manifest")
    if single_pass is False:
        block("not_single_pass",
              "single_pass is False and survey_georef requires True: the alignment contract "
              "here is for one pass, so declare the flight honestly and expect the "
              "georeferencer to refuse it", "single_pass",
              "either an actual single-pass flight or a separate contract for multi-pass "
              "fusion")

    duration = None
    duration_source = "undeclared"
    probe = probe_video(video_path) if video_path is not None else None
    if "video_duration_s" in overrides:
        duration = _finite_number(overrides["video_duration_s"])
        duration_source = "caller_override"
    elif probe is not None:
        duration, duration_source = probe["duration_s"], probe["source"]
    if duration is None or not math.isfinite(duration) or duration <= 0:
        block("missing_video_duration_s",
              "video_duration_s is unavailable and every telemetry time is bounded by it, so "
              "it is not invented here: pass overrides=dict(video_duration_s=...)",
              "video_duration_s",
              "a container duration the caller declares, or ffprobe output recorded in the "
              "run manifest")
        duration = 0.0

    offset = 0.0
    if "time_offset_s" in overrides:
        offset = _finite_number(overrides["time_offset_s"])
    elif info.get("first_time_s") is not None:
        offset = _finite_number(info["first_time_s"])
    reference = overrides.get("time_reference")
    source_reference = info.get("time_reference", "video")
    if reference is None:
        reference = "video"
        if source_reference != "video":
            block("telemetry_anchor_assumed",
                  f"the parsed log keeps its own clock ({source_reference!r}) and its "
                  "timestamps were made relative to a declared origin, so "
                  f"seconds-since-first-sample is *assumed* to equal video time 0. "
                  f"survey_georef accepts this metadata, which is why the assumption is "
                  f"stated rather than hidden", "time_reference",
                  "frame-sync or timecode evidence tying the first telemetry sample to video "
                  "frame 0, or a declared time_offset_s")
            warnings.append(f"video_duration_s {duration} is measured from frame 0 while the "
                            "telemetry clock origin is assumed to be frame 0 too: any anchor "
                            "error shifts the whole georeference by that amount")
    if reference not in _TIME_REFERENCES:
        raise ValueError(f"time_reference {reference!r} is not one of "
                         f"{sorted(_TIME_REFERENCES)}")
    if reference != "video":
        block("time_reference_not_video",
              f"time_reference is {reference!r}: the rows still carry an absolute or internal "
              f"clock with time_offset_s {offset}, so normalize_telemetry would be adding an "
              f"offset to the wrong quantity and survey_georef only accepts 'video'",
              "time_reference",
              "subtract the recording origin from t_sec in the rows, or declare a "
              "time_offset_s that lands every sample inside [0, video_duration_s]")

    datum = overrides.get("altitude_datum")
    datum_source = "asserted_by_caller"
    if datum is None:
        datum = _SOURCE_DATUMS.get(info.get("altitude_datum"), "undeclared")
        datum_source = "declared_by_source" if datum != "undeclared" else "undeclared"
    if datum not in _ALTITUDE_DATUMS:
        raise ValueError(f"altitude_datum {datum!r} is not one of {sorted(_ALTITUDE_DATUMS)}")
    basis = str(overrides.get("altitude_datum_basis", ""))
    if datum == "ellipsoidal":
        if not basis:
            warnings.append("altitude_datum 'ellipsoidal' is asserted with no basis: name the "
                            "receiver output or the geoid model applied")
    elif datum == "AGL-relative":
        block("altitude_datum_agl_relative",
              "the only height evidence is relative to the takeoff point (AGL), which is not a "
              "geodetic datum, so it is declared 'AGL-relative' instead of relabelled "
              "'ellipsoidal'; survey_georef rejects every other value", "altitude_datum",
              "a logged absolute or ellipsoidal GNSS height for the same samples, or a "
              "surveyed takeoff elevation plus a declared geoid separation")
    elif datum == "mean_sea_level":
        block("altitude_datum_not_ellipsoidal",
              "altitude is referenced to mean sea level (DJI AbsoluteAltitude / ASL) while "
              "survey_georef consumes an ellipsoidal height and applies no datum conversion",
              "altitude_datum",
              "an EGM96/EGM2008 geoid separation applied to these samples, or the receiver's "
              "raw ellipsoidal height (h = H + N)")
    elif datum == "unspecified_by_gpx":
        block("altitude_datum_unspecified",
              "the GPX <ele> datum is unspecified by the GPX schema: some receivers log "
              "ellipsoidal heights and some log mean sea level, so no datum is claimed here",
              "altitude_datum",
              "the receiver's documented output datum, or a <geoidalt> extension that shows "
              "which of the two <ele> holds")
    else:
        block("altitude_datum_undeclared",
              "no altitude datum is known: neither the parsed source nor the caller declared "
              "one", "altitude_datum",
              "the provenance of the height column, or a caller declaration with a basis")

    position = overrides.get("position_reference") or "gnss_antenna"
    applied = bool(overrides.get("antenna_offset_applied", False))
    negligible = overrides.get("lever_arm_negligible") is True
    lever = overrides.get("lever_arm_uncertainty_m")
    intrinsics = offset_vector = None
    if calibration_json is not None:
        payload = json.loads(_text_of(_open(calibration_json)))
        if not isinstance(payload, dict):
            raise ValueError("calibration_json must hold a JSON object")
        intrinsics = {key: payload[key] for key in _INTRINSIC_KEYS if key in payload}
        if len(intrinsics) != len(_INTRINSIC_KEYS):
            raise ValueError("calibration JSON lacks complete intrinsics: found "
                             f"{sorted(intrinsics)}, need all of {list(_INTRINSIC_KEYS)}")
        for key, value in intrinsics.items():
            _finite_number(value)
        intrinsics.update(camera_model=payload.get("camera_model", "unknown"),
                          distortion=payload.get("distortion"),
                          source=Path(calibration_json).name)
        offset_vector = payload.get("antenna_offset_m", payload.get("gnss_camera_offset_m"))
    if position not in _POSITION_REFERENCES:
        raise ValueError(f"position_reference {position!r} is not one of "
                         f"{sorted(_POSITION_REFERENCES)}")
    if position == "gnss_antenna":
        block("position_reference_gnss_antenna",
              "the position stream is a GNSS antenna phase centre: until the antenna-to-camera "
              "offset is subtracted or shown negligible these are not camera centres, and "
              "survey_georef currently rejects position_reference='gnss_antenna' with 'must "
              "declare camera_center' - a real blocker, not something to relabel",
              "position_reference",
              "a surveyed lever arm applied to every sample, or evidence that the offset is "
              "negligible against the required accuracy (10 cm against a 1 m target is 10% of "
              "the error budget and the fit cannot absorb it)")
    elif not applied and not negligible:
        block("camera_center_unasserted",
              "position_reference declares 'camera_center' but nothing says the antenna offset "
              "was applied or is negligible, so the declaration is unbacked",
              "position_reference",
              "antenna_offset_applied=True with lever_arm_uncertainty_m, or a measured offset "
              "small enough to declare negligible")
    if applied and lever is None:
        block("lever_arm_uncertainty_undeclared",
              "the lever arm is declared applied but its residual uncertainty is not stated, "
              "so it cannot be carried into the error budget", "position_reference",
              "overrides=dict(lever_arm_uncertainty_m=...) from a survey of the mount")
    if offset_vector is not None and not applied:
        warnings.append("the calibration file supplies an antenna offset that has not been "
                        "applied to the telemetry rows")

    declared_uncertainty = None
    if "horizontal_std_m" in overrides or "vertical_std_m" in overrides:
        _check_pair(overrides.get("horizontal_std_m"), overrides.get("vertical_std_m"))
        declared_uncertainty = [round(_finite_number(overrides["horizontal_std_m"]), 4),
                                round(_finite_number(overrides["vertical_std_m"]), 4)]
        if min(declared_uncertainty) <= 0:
            raise ValueError("declared standard deviations must be positive metres")
    elif info.get("uncertainty_source") == "declared_by_caller":
        # The parser was handed this pair; carry that provenance into the metadata so a
        # reviewer can see the uncertainty was declared rather than measured or modelled.
        declared_uncertainty = info.get("declared_std_m")
    return dict(schema_version=1, time_reference=reference, time_offset_s=float(offset),
                altitude_datum=datum, position_reference=position, single_pass=single_pass,
                video_duration_s=float(duration), altitude_datum_source=datum_source,
                altitude_datum_basis=basis, video_duration_source=duration_source,
                video_probe=probe, antenna_offset_applied=applied,
                lever_arm_negligible=negligible,
                lever_arm_uncertainty_m=None if lever is None else _finite_number(lever),
                antenna_offset_m=offset_vector, intrinsics=intrinsics,
                uncertainty_declared_by_caller=declared_uncertainty, source=dict(info),
                warnings=warnings, blockers=blockers,
                unresolved=[item.code for item in blockers])
