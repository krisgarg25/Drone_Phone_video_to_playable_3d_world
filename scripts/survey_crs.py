"""Put a measured cloud into a CRS it is allowed to name.

``survey_georef`` turns WGS84 telemetry into a local east/north/up tangent plane,
and every product so far has been exported in that plane: honest, but not
georeferenced. The GeoTIFF and LAS writers here refuse a CRS they were not given,
so the way to deliver those formats is not to stamp a label on local metres but to
invert the chain that produced them: ENU -> ECEF -> geodetic -> UTM.

Deliberate limits, all of them stated in the returned payload:

* heights stay **ellipsoidal**. There is no geoid model in this repository, so no
  mean-sea-level or orthometric height is ever produced or implied.
* the projection is the standard ellipsoidal transverse Mercator series (Snyder
  8-9/8-10), accurate to millimetres within the UTM zone and refused beyond it
  rather than extrapolated.
* a scene straddling a zone boundary, or sitting outside 84 deg latitude, raises.
  UPS, custom grids and national zones are not invented here; the caller declares
  one explicitly if it is really needed.
* the WGS84 parameters match ``survey_georef.normalize_telemetry`` exactly, so the
  round trip is the inverse of that transform and not a reimplementation of it.
"""
import math

import numpy as np

# Same constants survey_georef used to build ENU; a different value here would
# make the inverse transform disagree with the forward one by design.
WGS84_A = 6378137.0
WGS84_E2 = 6.6943799901413165e-3
WGS84_F = 1.0 / 298.257223563
K0 = 0.9996
FALSE_EASTING = 500000.0
FALSE_NORTHING_SOUTH = 10000000.0
# Beyond this latitude UTM is not defined. UPS is, and is not implemented.
MAX_UTM_LATITUDE_DEG = 84.0
# The series is a zone-limited expansion: past ~3 deg from the central meridian
# its truncation error grows, and the zone edge is at 3 deg anyway. A half degree
# of slack lets a scene touch the boundary without failing.
MAX_OFF_MERIDIAN_DEG = 3.5
DEG = math.pi / 180.0

__all__ = ["ecef_from_geodetic", "geodetic_from_ecef", "enu_basis",
           "enu_to_geodetic", "utm_zone_for", "central_meridian", "utm_forward",
           "utm_inverse", "utm_wkt", "crs_from_origin", "crs_from_alignment", "enu_to_crs"]


def _finite_array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(name + " must be numeric")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.shape != shape:
        raise ValueError(name + " must have shape " + str(shape) + ", got " + str(result.shape))
    if not np.isfinite(result).all():
        raise ValueError(name + " must contain only finite values")
    return result


def _degrees(value, name, *, low=-180.0, high=180.0):
    array = _finite_array(np.atleast_1d(value), (len(np.atleast_1d(value)),), name)
    if np.any(array < low) or np.any(array > high):
        raise ValueError(f"{name} must lie in [{low}, {high}] degrees, "
                         f"got {array.min()}..{array.max()}")
    return array


def ecef_from_geodetic(latitudes_deg, longitudes_deg, heights_m) -> np.ndarray:
    """WGS84 geodetic to earth-centred earth-fixed, exactly as survey_georef does it."""
    lat = np.deg2rad(_degrees(latitudes_deg, "latitude_deg", low=-90.0, high=90.0))
    lon = _degrees(longitudes_deg, "longitude_deg")
    height = _finite_array(np.atleast_1d(heights_m), lat.shape, "height_m")
    radius = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    lam = np.radians(lon)
    return np.column_stack(((radius + height) * cos_lat * np.cos(lam),
                            (radius + height) * cos_lat * np.sin(lam),
                            (radius * (1 - WGS84_E2) + height) * sin_lat))


def geodetic_from_ecef(xyz) -> np.ndarray:
    """ECEF metres to (lat_deg, lon_deg, ellipsoidal_height_m) by Bowring plus refinement.

    Bowring's closed form is already sub-millimetre for a drone-scale baseline; the
    two refinement passes make the round trip through ``ecef_from_geodetic`` agree
    to float precision, which is what the tests assert.
    """
    points = _finite_array(np.atleast_2d(xyz), (len(np.atleast_2d(xyz)), 3), "ecef xyz")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    e2p = WGS84_E2 / (1 - WGS84_E2)
    # Bowring's auxiliary angle is on the reduced latitude: tan(beta) = (1-f) z / p,
    # and (1-f) equals sqrt(1-e2) exactly for this ellipsoid.
    theta = np.arctan2(math.sqrt(1 - WGS84_E2) * z, p)
    lat = np.arctan2(z + e2p * WGS84_A * np.sin(theta) ** 3,
                     p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3)
    for _ in range(3):
        radius = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        height = np.where(np.abs(np.cos(lat)) > 1e-12, p / np.cos(lat) - radius,
                          np.abs(z) - radius * (1 - WGS84_E2))
        lat = np.arctan2(z, p * (1 - WGS84_E2 * radius / (radius + height)))
    radius = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    # Projection onto the ellipsoid normal rather than p/cos(lat) - N: the closed
    # form divides by cos(lat), so at 80 deg a 1e-12 rad latitude error becomes
    # 4e-5 m of height. This form is first-order insensitive to that error.
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    height = ((p - radius * cos_lat) * cos_lat
              + (z - radius * (1 - WGS84_E2) * sin_lat) * sin_lat)
    return np.column_stack([np.rad2deg(lat), np.rad2deg(lon), height])


def enu_basis(origin_lat_deg, origin_lon_deg) -> np.ndarray:
    """The 3x3 rotation whose rows are the east, north and up unit vectors."""
    lat = math.radians(_degrees([origin_lat_deg], "origin latitude", low=-90, high=90)[0])
    lon = math.radians(_degrees([origin_lon_deg], "origin longitude")[0])
    slat, clat, slon, clon = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([[-slon, clon, 0.0],
                     [-slat * clon, -slat * slon, clat],
                     [clat * clon, clat * slon, slat]], dtype=np.float64)


def enu_to_geodetic(enu, origin_lat_deg, origin_lon_deg, origin_height_m=0.0) -> np.ndarray:
    """Invert ``survey_georef``: local ENU metres back to geodetic degrees and height.

    ``survey_georef.normalize_telemetry`` computes ``enu = (ecef - ecef[0]) @ basis.T``,
    whose inverse is ``ecef = ecef[0] + enu @ basis`` because ``basis`` is orthonormal.
    """
    points = _finite_array(np.atleast_2d(enu), (len(np.atleast_2d(enu)), 3), "enu points")
    origin_ecef = ecef_from_geodetic([origin_lat_deg], [origin_lon_deg], [origin_height_m])[0]
    return geodetic_from_ecef(origin_ecef + points @ enu_basis(origin_lat_deg, origin_lon_deg))


def _zone(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("zone must be an integer 1..60")
    try:
        zone = int(value)
    except (TypeError, ValueError):
        raise ValueError("zone must be an integer 1..60") from None
    if (isinstance(value, float) and value != zone) or not 1 <= zone <= 60:
        raise ValueError(f"zone must be an integer 1..60, got {value!r}")
    return zone


def utm_zone_for(lat_deg, lon_deg):
    """(zone, hemisphere, epsg) with the Norway and Svalbard exception zones applied."""
    lat = _degrees([lat_deg], "latitude", low=-90, high=90)[0]
    lon = _degrees([lon_deg], "longitude")[0]
    if abs(lat) > MAX_UTM_LATITUDE_DEG:
        raise ValueError(f"latitude {lat} deg is outside the UTM domain (|lat| <= "
                         f"{MAX_UTM_LATITUDE_DEG}); UPS is not implemented and no CRS is invented")
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    if 74.0 <= lat <= 84.0:
        for lower, upper, exception in ((0.0, 9.0, 31), (9.0, 21.0, 33),
                                        (21.0, 33.0, 35), (33.0, 42.0, 37)):
            if lower <= lon < upper:
                zone = exception
                break
    hemisphere = "N" if lat >= 0 else "S"
    return zone, hemisphere, (32600 if hemisphere == "N" else 32700) + zone


def central_meridian(zone) -> float:
    return (_zone(zone) - 1) * 6.0 - 180.0 + 3.0


def utm_forward(latitudes_deg, longitudes_deg, zone):
    """(easting, northing) in metres on the ellipsoid, Snyder's 6th-order series.

    Points more than ``MAX_OFF_MERIDIAN_DEG`` from the zone's central meridian raise:
    the series is a truncated expansion around that meridian, and a silently
    extrapolated easting is a positioning error dressed up as a coordinate.
    """
    lat = _degrees(latitudes_deg, "latitude", low=-90, high=90)
    lon = _degrees(longitudes_deg, "longitude")
    zone = _zone(zone)
    lam0 = math.radians(central_meridian(zone))
    lam = np.radians(lon)
    off = np.degrees(np.arctan2(np.sin(lam - lam0), np.cos(lam - lam0)))
    if np.any(np.abs(off) > MAX_OFF_MERIDIAN_DEG):
        worst = float(np.max(np.abs(off)))
        raise ValueError(f"a point {worst:.2f} deg from central meridian "
                         f"{math.degrees(lam0):.0f} exceeds the {MAX_OFF_MERIDIAN_DEG} deg "
                         "validity of this series; split the scene by zone or declare a CRS")
    phi = np.radians(lat)
    ep2 = WGS84_E2 / (1 - WGS84_E2)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)
    t = np.tan(phi) ** 2
    c = ep2 * np.cos(phi) ** 2
    a = np.cos(phi) * (lam - lam0)
    m = WGS84_A * ((1 - WGS84_E2 / 4 - 3 * WGS84_E2 ** 2 / 64 - 5 * WGS84_E2 ** 3 / 256) * phi
                   - (3 * WGS84_E2 / 8 + 3 * WGS84_E2 ** 2 / 32 + 45 * WGS84_E2 ** 3 / 1024)
                   * np.sin(2 * phi)
                   + (15 * WGS84_E2 ** 2 / 256 + 45 * WGS84_E2 ** 3 / 1024) * np.sin(4 * phi)
                   - (35 * WGS84_E2 ** 3 / 3072) * np.sin(6 * phi))
    easting = K0 * n * (a + (1 - t + c) * a ** 3 / 6
                        + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * a ** 5 / 120) + FALSE_EASTING
    northing = K0 * (m + np.tan(phi) * n * (a ** 2 / 2
                                            + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                                            + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2)
                                            * a ** 6 / 720))
    return easting, np.where(lat < 0, northing + FALSE_NORTHING_SOUTH, northing)


def utm_inverse(eastings, northings, zone, hemisphere="N"):
    """(lat_deg, lon_deg) for a UTM coordinate, the inverse of ``utm_forward``."""
    e = _finite_array(np.atleast_1d(eastings), (len(np.atleast_1d(eastings)),), "easting_m")
    nn = _finite_array(np.atleast_1d(northings), (len(np.atleast_1d(northings)),), "northing_m")
    zone = _zone(zone)
    if hemisphere not in ("N", "S"):
        raise ValueError("hemisphere must be 'N' or 'S'")
    if np.any(e <= 0) or np.any(nn <= 0):
        raise ValueError("UTM easting and northing must be positive in the declared hemisphere")
    lam0 = math.radians(central_meridian(zone))
    y = nn - (FALSE_NORTHING_SOUTH if hemisphere == "S" else 0.0)
    x = e - FALSE_EASTING
    m = y / K0
    mu = m / (WGS84_A * (1 - WGS84_E2 / 4 - 3 * WGS84_E2 ** 2 / 64 - 5 * WGS84_E2 ** 3 / 256))
    e1 = (1 - np.sqrt(1 - WGS84_E2)) / (1 + np.sqrt(1 - WGS84_E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu))
    ep2 = WGS84_E2 / (1 - WGS84_E2)
    c1 = ep2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi1) ** 2)
    r1 = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * np.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * K0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (d ** 2 / 2
                                             - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2)
                                             * d ** 4 / 24
                                             + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2
                                                - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = lam0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2)
                  * d ** 5 / 120) / np.cos(phi1)
    return np.rad2deg(lat), np.rad2deg(lon)


def utm_wkt(zone, hemisphere) -> str:
    """WKT1 PROJCS text for the named UTM zone, carrying its EPSG code."""
    zone = _zone(zone)
    if hemisphere not in ("N", "S"):
        raise ValueError("hemisphere must be 'N' or 'S'")
    epsg = (32600 if hemisphere == "N" else 32700) + zone
    false_northing = FALSE_NORTHING_SOUTH if hemisphere == "S" else 0
    return ('PROJCS["WGS 84 / UTM zone {zone}{hemi}",'
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",{a},{invf},AUTHORITY["EPSG","7030"]],'
            'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
            'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            'AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",{cm}],'
            'PARAMETER["scale_factor",{k0}],PARAMETER["false_easting",{fe}],'
            'PARAMETER["false_northing",{fn}],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],AXIS["Northing",NORTH],'
            'AUTHORITY["EPSG","{epsg}"]]').format(
        zone=zone, hemi=hemisphere, a=WGS84_A, invf=1.0 / WGS84_F,
        cm=central_meridian(zone), k0=K0, fe=FALSE_EASTING, fn=false_northing, epsg=epsg)


def crs_from_origin(origin_lat_deg, origin_lon_deg, origin_height_m=0.0) -> dict:
    """The single CRS this scene can be exported in, derived from its GPS origin.

    Every field is a fact about the CRS, not a measurement of the model: the
    ``validity`` note is what tells a reviewer the zone was chosen from the origin
    and that a scene reaching far across it needs checking.
    """
    zone, hemisphere, epsg = utm_zone_for(origin_lat_deg, origin_lon_deg)
    easting, northing = utm_forward([origin_lat_deg], [origin_lon_deg], zone)
    return {"schema_version": 1, "kind": "projected", "epsg": epsg,
            "name": f"WGS 84 / UTM zone {zone}{hemisphere}", "zone": zone,
            "hemisphere": hemisphere, "central_meridian_deg": central_meridian(zone),
            "origin_easting_m": float(easting[0]), "origin_northing_m": float(northing[0]),
            "origin_geodetic": {"latitude_deg": float(origin_lat_deg),
                                "longitude_deg": float(origin_lon_deg),
                                "height_m": float(origin_height_m)},
            "units": "m", "height_datum": "ellipsoidal (WGS84)",
            "wkt": utm_wkt(zone, hemisphere),
            "validity": (f"transverse Mercator series valid within {MAX_OFF_MERIDIAN_DEG} deg "
                         f"of central meridian {central_meridian(zone):.0f} deg; heights are "
                         "ellipsoidal, not orthometric")}


def crs_from_alignment(alignment) -> dict:
    """``crs_from_origin`` for a ``survey_georef`` alignment, or a refusal that says why."""
    if not isinstance(alignment, dict):
        raise ValueError("alignment must be a survey_georef result dict")
    frame = alignment.get("coordinate_frame")
    if not isinstance(frame, dict):
        raise ValueError("alignment carries no coordinate_frame; nothing to georeference against")
    if frame.get("geodetic_crs") != "EPSG:4979":
        raise ValueError("only WGS84 geodetic telemetry (EPSG:4979) is projected here, got "
                         + repr(frame.get("geodetic_crs")))
    if frame.get("altitude_datum") != "ellipsoidal":
        raise ValueError("only ellipsoidal heights are carried through, got "
                         + repr(frame.get("altitude_datum")))
    origin = frame.get("origin")
    if not isinstance(origin, dict):
        raise ValueError("coordinate_frame has no origin; the ENU tangent point is unknown")
    return crs_from_origin(origin["latitude_deg"], origin["longitude_deg"],
                           origin.get("altitude_m", 0.0))


def enu_to_crs(enu, alignment, crs=None):
    """Local ENU metres to (easting, northing, ellipsoidal height) plus geodetic degrees.

    Returns ``(utm_nx3, geodetic_nx3, crs)``. The geodetic columns are the same
    points in degrees, so a position CSV needs no second transform.
    """
    points = _finite_array(np.atleast_2d(enu), (len(np.atleast_2d(enu)), 3), "enu points")
    crs = crs or crs_from_alignment(alignment)
    origin = crs["origin_geodetic"]
    geodetic = enu_to_geodetic(points, origin["latitude_deg"], origin["longitude_deg"],
                               origin["height_m"])
    easting, northing = utm_forward(geodetic[:, 0], geodetic[:, 1], crs["zone"])
    return (np.column_stack([easting, northing, geodetic[:, 2]]), geodetic, crs)
