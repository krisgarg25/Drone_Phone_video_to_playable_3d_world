"""Ellipsoidal <-> mean-sea-level heights through the EGM96 geoid (challenge F3).

GPS gives heights above the WGS84 ellipsoid; maps, GCP sheets and building plans quote
heights above mean sea level. The difference is the geoid undulation N, which varies by
more than 190 m across the globe (-107 m south of India, +85 m over New Guinea), so an
ellipsoidal height handed to someone expecting MSL is wrong by tens of metres:

    H_orthometric = h_ellipsoidal - N(lat, lon)

N comes from the NGA EGM96 15-arc-minute grid that PROJ distributes as
``us_nga_egm96_15.tif`` (public domain, 721 x 1440 float32, pixel centres on the
quarter degree from 90 N and 180 W). Nothing is downloaded here: the file lives at
``data/geoid/us_nga_egm96_15.tif`` and, when it is missing, every function raises
``GeoidUnavailable`` so a caller keeps ellipsoidal heights and says why, instead of
guessing an undulation. Fetch it with::

    curl -L -o data/geoid/us_nga_egm96_15.tif https://cdn.proj.org/us_nga_egm96_15.tif

Interpolation is bilinear on the grid, which is how PROJ's ``vgridshift`` applies it;
``tests/test_survey_geoid.py`` checks the two agree. What this does NOT give: the grid's
own error (EGM96 is good to roughly +/-0.5-1 m in most places and worse in mountains),
or a national vertical datum (NAVD88, AHD, the Indian MSL datum), which differ from the
global geoid by up to a metre or two.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID_PATH = ROOT / "data" / "geoid" / "us_nga_egm96_15.tif"
GRID_URL = "https://cdn.proj.org/us_nga_egm96_15.tif"
MODEL = "EGM96"
VERTICAL_EPSG = 5773  # "EGM96 height"
ROWS, COLUMNS, STEP_DEG = 721, 1440, 0.25
RANGE_M = (-107.0, 85.4)  # the grid's own extremes, used to reject a wrong file

VERTICAL_DATUMS = ("ellipsoidal", "egm96")
EGM96_VERT_CS = ('VERT_CS["EGM96 height",VERT_DATUM["EGM96 geoid",2005,'
                 'AUTHORITY["EPSG","5171"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
                 'AXIS["Gravity-related height",UP],AUTHORITY["EPSG","5773"]]')


class GeoidUnavailable(ValueError):
    """No usable geoid grid: keep ellipsoidal heights and report the reason."""


@lru_cache(maxsize=2)
def _grid(path: str) -> tuple[np.ndarray, str]:
    file = Path(path)
    if not file.is_file():
        raise GeoidUnavailable(f"no {MODEL} grid at {file.as_posix()}: heights stay "
                               f"ellipsoidal. Fetch it from {GRID_URL}")
    import cv2  # already a pipeline dependency; reads the deflate-tiled float32 TIFF
    quiet = getattr(getattr(cv2, "utils", None), "logging", None)
    previous = quiet.getLogLevel() if quiet else None
    if quiet:  # libtiff warns about every GeoTIFF tag it does not know; they are expected
        quiet.setLogLevel(quiet.LOG_LEVEL_ERROR)
    try:
        values = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    finally:
        if quiet:
            quiet.setLogLevel(previous)
    if values is None or values.shape != (ROWS, COLUMNS) or values.dtype != np.float32:
        raise GeoidUnavailable(f"{file.name} is not the {ROWS}x{COLUMNS} float32 {MODEL} "
                               "15' grid this module interpolates")
    low, high = float(values.min()), float(values.max())
    if not (np.isfinite(values).all() and abs(low - RANGE_M[0]) < 1.0
            and abs(high - RANGE_M[1]) < 1.0):
        raise GeoidUnavailable(f"{file.name} spans {low:.1f}..{high:.1f} m, not the "
                               f"{MODEL} range {RANGE_M[0]}..{RANGE_M[1]} m")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    return values.astype(np.float64), digest


def available(path=None) -> bool:
    try:
        _grid(str(path or GRID_PATH))
        return True
    except GeoidUnavailable:
        return False


def undulation(latitudes_deg, longitudes_deg, path=None) -> np.ndarray:
    """Geoid height N in metres (geoid above ellipsoid) at each point, bilinear."""
    grid, _ = _grid(str(path or GRID_PATH))
    lat = np.asarray(latitudes_deg, dtype=np.float64)
    lon = np.asarray(longitudes_deg, dtype=np.float64)
    if lat.shape != lon.shape or not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("latitudes and longitudes must be finite arrays of one shape")
    if np.any(np.abs(lat) > 90.0):
        raise ValueError("latitude outside -90..90")
    # Row 0 is 90 N, column 0 is 180 W; longitude wraps, latitude clamps at the poles.
    row = (90.0 - lat) / STEP_DEG
    col = np.mod(lon + 180.0, 360.0) / STEP_DEG
    r0 = np.clip(np.floor(row).astype(np.int64), 0, ROWS - 2)
    c0 = np.floor(col).astype(np.int64) % COLUMNS
    c1 = (c0 + 1) % COLUMNS
    fr, fc = row - r0, col - np.floor(col)
    top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
    bottom = grid[r0 + 1, c0] * (1 - fc) + grid[r0 + 1, c1] * fc
    return top * (1 - fr) + bottom * fr


def to_orthometric(latitudes_deg, longitudes_deg, ellipsoidal_m, path=None) -> np.ndarray:
    """h (WGS84 ellipsoid) -> H (EGM96 mean sea level)."""
    return np.asarray(ellipsoidal_m, dtype=np.float64) - undulation(latitudes_deg,
                                                                    longitudes_deg, path)


def to_ellipsoidal(latitudes_deg, longitudes_deg, orthometric_m, path=None) -> np.ndarray:
    """H (EGM96 mean sea level) -> h (WGS84 ellipsoid)."""
    return np.asarray(orthometric_m, dtype=np.float64) + undulation(latitudes_deg,
                                                                    longitudes_deg, path)


def describe(path=None) -> dict:
    """What a product can say about its heights: the model, its file digest, its limits."""
    try:
        _, digest = _grid(str(path or GRID_PATH))
    except GeoidUnavailable as error:
        return {"model": MODEL, "available": False, "reason": str(error)}
    return {"model": MODEL, "available": True, "vertical_epsg": VERTICAL_EPSG,
            "grid": Path(path or GRID_PATH).name, "grid_sha256": digest,
            "interpolation": "bilinear on the 15' grid (as PROJ vgridshift)",
            "limits": ("EGM96 itself is good to roughly 0.5-1 m in most places and worse "
                       "in high relief; a national height datum may differ by 1-2 m more")}


def compound_wkt(horizontal_wkt: str, name: str) -> str:
    """WKT1 COMPD_CS pairing a projected CRS with EGM96 heights (EPSG:5773)."""
    return f'COMPD_CS["{name} + EGM96 height",{horizontal_wkt},{EGM96_VERT_CS}]'
