"""Surveyed checkpoints in, the evaluator's exact ``checkpoints.json`` out (challenge F2).

A GPS-only reconstruction is "unverified" until somebody compares it with points that
were surveyed independently - a total station, an RTK rover, a published benchmark. The
evaluator (``survey_workflow._evaluation_metrics``) already turns such pairs into an
RMSE and, from 8 points up, a fit/hold-out split; what was missing is a way to hand it
the pairs without hand-writing JSON in the scene's own ENU tangent frame.

This module reads one CSV row per checkpoint - the surveyed coordinate and the same
feature measured on the delivered model - in whichever frame each was recorded, and
converts both into the alignment's local ENU frame:

    reference (surveyed):  ref_lat_deg, ref_lon_deg, ref_height_m      geodetic
                       or  ref_easting_m, ref_northing_m, ref_height_m  the scene's UTM zone
                       or  ref_e_m, ref_n_m, ref_u_m                    local ENU
    model (read off the georeferenced product):
                           model_easting_m, model_northing_m, model_height_m   UTM set
                       or  model_e_m, model_n_m, model_u_m                     ENU set

plus an ``id`` column. Heights on either side may be ellipsoidal or EGM96 mean sea level
(``survey_geoid``); a GCP sheet in MSL is converted, never silently mixed with GPS
heights. Nothing is inferred: a row with a missing value, a duplicate id, a point
kilometres from its model partner (two different frames, not an error of 1 km), or a
UTM coordinate in the wrong zone is refused with the row number.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math

import numpy as np

try:  # imported as scripts.survey_checkpoints by the tests
    from scripts import survey_crs as crs
    from scripts import survey_geoid as geoid
except ImportError:  # imported flat by the workflow
    import survey_crs as crs
    import survey_geoid as geoid

MAX_ROWS = 10000
MAX_SEPARATION_M = 1000.0
"""A GPS-aligned model is metres off, not a kilometre: a larger gap between a surveyed
point and its model partner means the two columns are in different frames."""
HEIGHT_DATUMS = ("ellipsoidal", "egm96")

REFERENCE_FORMS = {
    "geodetic": ("ref_lat_deg", "ref_lon_deg", "ref_height_m"),
    "utm": ("ref_easting_m", "ref_northing_m", "ref_height_m"),
    "enu": ("ref_e_m", "ref_n_m", "ref_u_m"),
}
MODEL_FORMS = {
    "utm": ("model_easting_m", "model_northing_m", "model_height_m"),
    "enu": ("model_e_m", "model_n_m", "model_u_m"),
}
TEMPLATE = ("id,ref_lat_deg,ref_lon_deg,ref_height_m,model_easting_m,model_northing_m,"
            "model_height_m\n")


def _form(header, forms, side):
    found = [name for name, columns in forms.items() if set(columns) <= set(header)]
    if len(found) != 1:
        options = "; ".join(f"{name}: {', '.join(columns)}" for name, columns in forms.items())
        raise ValueError(f"the {side} coordinates need exactly one column set ({options}); "
                         f"found {len(found)}")
    return found[0]


def _origin(alignment):
    frame = (alignment or {}).get("coordinate_frame")
    origin = (frame or {}).get("origin")
    if not isinstance(origin, dict) or frame.get("geodetic_crs") != "EPSG:4979":
        raise ValueError("the scene has no WGS84 ENU alignment to evaluate checkpoints against; "
                         "align it first")
    return (float(origin["latitude_deg"]), float(origin["longitude_deg"]),
            float(origin.get("altitude_m", 0.0)))


def geodetic_to_enu(lat, lon, height, origin):
    lat0, lon0, h0 = origin
    xyz = crs.ecef_from_geodetic(lat, lon, height)
    xyz0 = crs.ecef_from_geodetic([lat0], [lon0], [h0])[0]
    return (xyz - xyz0) @ crs.enu_basis(lat0, lon0).T


def _to_enu(values, form, datum, origin, scene_crs):
    """Nx3 in ``form`` -> Nx3 local ENU of the alignment, heights made ellipsoidal."""
    if form == "enu":
        if datum != "ellipsoidal":
            raise ValueError("ENU coordinates are already relative to the ellipsoidal origin; "
                             "an EGM96 height datum does not apply to them")
        return values
    if form == "utm":
        lat, lon = crs.utm_inverse(values[:, 0], values[:, 1], scene_crs["zone"],
                                   scene_crs["hemisphere"])
    else:
        lat, lon = values[:, 0], values[:, 1]
    height = values[:, 2]
    if datum == "egm96":
        height = geoid.to_ellipsoidal(lat, lon, height)
    return geodetic_to_enu(lat, lon, height, origin)


def parse(text, alignment, *, reference_height_datum="ellipsoidal",
          model_height_datum="ellipsoidal"):
    """(ids, reconstructed_enu, reference_enu, report) from checkpoint CSV text."""
    for name, value in (("reference_height_datum", reference_height_datum),
                        ("model_height_datum", model_height_datum)):
        if value not in HEIGHT_DATUMS:
            raise ValueError(f"{name} must be one of {HEIGHT_DATUMS}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("checkpoint CSV is empty")
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = [name.strip() for name in (reader.fieldnames or [])]
    if "id" not in header:
        raise ValueError("checkpoint CSV needs an 'id' column")
    reader.fieldnames = header
    ref_form = _form(header, REFERENCE_FORMS, "reference (surveyed)")
    model_form = _form(header, MODEL_FORMS, "model")
    ids, ref, model = [], [], []
    for number, row in enumerate(reader, start=2):
        if not any((value or "").strip() for value in row.values()):
            continue
        label = (row.get("id") or "").strip()
        if not label:
            raise ValueError(f"row {number}: missing id")
        if label in ids:
            raise ValueError(f"row {number}: duplicate id {label!r}")
        try:
            a = [float(row[c]) for c in REFERENCE_FORMS[ref_form]]
            b = [float(row[c]) for c in MODEL_FORMS[model_form]]
        except (TypeError, ValueError):
            raise ValueError(f"row {number} ({label}): every coordinate must be a number") \
                from None
        if not all(math.isfinite(v) for v in a + b):
            raise ValueError(f"row {number} ({label}): coordinates must be finite")
        ids.append(label)
        ref.append(a)
        model.append(b)
        if len(ids) > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} checkpoints")
    if not ids:
        raise ValueError("checkpoint CSV has a header but no rows")
    origin = _origin(alignment)
    scene_crs = crs.crs_from_origin(*origin)
    try:
        reference = _to_enu(np.asarray(ref, float), ref_form, reference_height_datum, origin,
                            scene_crs)
        reconstructed = _to_enu(np.asarray(model, float), model_form, model_height_datum,
                                origin, scene_crs)
    except geoid.GeoidUnavailable as error:
        raise ValueError(f"EGM96 heights were declared but cannot be converted: {error}") \
            from None
    gaps = np.linalg.norm(reconstructed - reference, axis=1)
    worst = int(np.argmax(gaps))
    if gaps[worst] > MAX_SEPARATION_M:
        raise ValueError(f"checkpoint {ids[worst]!r} is {gaps[worst]:.0f} m from its model "
                         f"point: the surveyed and model columns are probably in different "
                         f"frames (UTM zone {scene_crs['zone']}{scene_crs['hemisphere']}, "
                         "ENU or degrees)")
    report = {"count": len(ids), "reference_form": ref_form, "model_form": model_form,
              "reference_height_datum": reference_height_datum,
              "model_height_datum": model_height_datum,
              "utm_zone": f"{scene_crs['zone']}{scene_crs['hemisphere']}",
              "epsg": scene_crs["epsg"], "max_separation_m": float(gaps[worst]),
              "csv_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return ids, reconstructed, reference, report


def document(alignment, reconstructed, reference):
    """The exact shape ``survey_workflow`` accepts; no other keys, by contract."""
    return {"independent": True, "alignment": "none",
            "coordinate_frame": alignment["coordinate_frame"],
            "checkpoints": [{"reconstructed": [float(v) for v in a],
                             "reference": [float(v) for v in b]}
                            for a, b in zip(reconstructed, reference)]}
