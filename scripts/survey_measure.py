"""Measurement and analysis on metric ENU point clouds, using NumPy and stdlib.

The pipeline already puts the scene into a georeferenced frame (survey_georef:
type "ENU", units "m", WGS84/EPSG:4979, ellipsoidal heights) and writes clouds in
it (survey_products). This module is the missing half: what a person can *ask* of
that model once they can see it - how long, how high, how big, how steep, how much
has been cut or filled.

Two rules govern every function here.

1. A number is never returned on its own. Each result is a record carrying
   ``value``, ``unit``, ``uncertainty`` (itself a record with its own unit and
   validity flag), ``valid`` and ``reason``. When validity is False, ``reason``
   says what is missing; when it is True, ``reason`` is None.
2. Absent evidence is reported as absent, not as zero. A measurement whose
   endpoints sit in empty space, a polygon with collinear vertices, or a cut/fill
   difference taken over a partial overlap comes back ``valid=False`` - and where
   there is no geometry at all to measure (no neighbours to snap to, no grid cell
   holding both surfaces, no plane under the target) the value is None rather than
   an invented 0.0.

Uncertainty is a documented root-sum-square budget: the sampling of the cloud
(point spacing, measured from the cloud itself when not declared), the declared
registration residual (the georeference fit RMSE - internal consistency, never an
independently surveyed accuracy), and a term from the measurement's own geometry
(surface roughness, plane residual, column scatter). Registration error averages
down as 1/sqrt(n) with a median-efficiency factor; nothing else does.

Scope: CPU only, NumPy plus stdlib. Points in, records out. Clouds are never
modified, and nothing is written to disk unless the caller hands ``to_csv`` an
explicit path. Neighbour queries are voxel-indexed, so they stay linear in the
point count. Degenerate or non-finite input raises ValueError; geometrically
meaningless measurements return a flagged record.
"""
import csv
from functools import lru_cache
import io
import json
import math
from pathlib import Path

import numpy as np

# WGS84 geometry, identical to survey_georef so both directions agree exactly.
_A = 6378137.0
_E2 = 6.6943799901413165e-3
_FRAME = dict(type="ENU", units="m", geodetic_crs="EPSG:4979",
              altitude_datum="ellipsoidal")
_MEDIAN_FACTOR = math.sqrt(math.pi / 2.0)  # 1-sigma of a median of n samples.
_AXES = ("height", "length", "horizontal")
_PLANARITY = 1e-8  # second/first centred singular value of a neighbourhood.


# --------------------------------------------------------------------------- #
# Input validation: nothing is silently coerced, and a NaN never becomes data. #
# --------------------------------------------------------------------------- #
def _number(value, name, *, positive=False, non_negative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (str, int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if non_negative and result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _count(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _rows(value, name):
    """A private float64 copy of a numeric array, finite-only."""
    raw = np.asarray(value, dtype=np.float64)
    if raw.size and not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(raw, dtype=np.float64, copy=True)


def _cloud(value, name="points"):
    """Validated float64 (N, 3) copy of a cloud; accepts an .xyz carrier object."""
    raw = np.asarray(getattr(value, "xyz", value))
    if raw.ndim != 2 or raw.shape[1] != 3 or not raw.shape[0]:
        raise ValueError(f"{name} must be a non-empty Nx3 array of ENU metres")
    return _rows(raw, name)


def _vector(value, name):
    result = _rows(value, name).reshape(-1)
    if result.size != 3:
        raise ValueError(f"{name} must be three finite ENU metres (east, north, up)")
    return result


def _ring(value, name):
    """A vertex list as (N, 3), with up=0 when 2-D vertices are given."""
    raw = np.asarray(getattr(value, "xyz", value))
    if raw.ndim != 2 or raw.shape[1] not in (2, 3) or not raw.shape[0]:
        raise ValueError(f"{name} must be a non-empty Nx2 or Nx3 vertex list")
    result = _rows(raw, name)
    if result.shape[1] == 2:
        return np.column_stack((result, np.zeros(len(result))))
    return result


def _plain(value, *, depth=0):
    """Recursively convert NumPy scalars into JSON-safe Python values."""
    if depth > 12:
        raise ValueError("Measurement nesting is too deep to serialise")
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("Refusing to serialise a non-finite measurement value")
        return float(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth=depth + 1) for item in value]
    raise ValueError(f"Cannot serialise {type(value).__name__} in a measurement")


def _text(value):
    return None if value is None else str(value)


def _safe(value, name):
    """A reported number, or None when there was nothing to measure."""
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} came out non-finite")
    return result


# --------------------------------------------------------------------------- #
# Voxel index: linear-time neighbourhood queries over a fixed cloud.          #
# --------------------------------------------------------------------------- #
class _Grid:
    """Voxel hash over a point cloud whose cell size is at least the query radius.

    Built once per measurement; the query walks the adjacent cells, so a radius no
    larger than the cell size touches 27 buckets. Positions are never modified.
    """

    __slots__ = ("points", "cell", "base", "dims", "order", "cells")

    def __init__(self, points, cell_m):
        self.points = points
        self.cell = _number(cell_m, "cell_m", positive=True)
        scaled = points / self.cell
        if not np.all(np.abs(scaled) < 2.0 ** 50):
            raise ValueError("cloud extent is too large for this neighbour cell size")
        self.base = np.floor(scaled.min(axis=0)).astype(np.int64) - 1
        index = self._index(points)
        self.dims = tuple(int(edge) + 2 for edge in index.max(axis=0))
        if float(np.prod(np.array(self.dims, dtype=np.float64))) > 2.0 ** 55:
            raise ValueError("Neighbour search resolution is numerically unsafe")
        keys = np.ravel_multi_index(index.T, self.dims)
        self.order = np.argsort(keys, kind="stable")
        ordered = keys[self.order]
        unique, starts = np.unique(ordered, return_index=True)
        ends = np.append(starts[1:], ordered.size)
        self.cells = {int(key): (int(start), int(stop))
                      for key, start, stop in zip(unique, starts, ends)}

    def _index(self, positions):
        return (np.floor(np.asarray(positions, dtype=np.float64) / self.cell)
                .astype(np.int64) - self.base)

    def candidate_rows(self, xyz, radius):
        point = np.asarray(xyz, dtype=np.float64).reshape(1, 3)
        centre = self._index(point)[0]
        span = int(math.ceil(_number(radius, "radius", positive=True) / self.cell))
        offsets = _neighbour_offsets(span)
        limit = np.array(self.dims, dtype=np.int64) - 1
        coords = centre + offsets
        keep = np.all((coords >= 1) & (coords <= limit), axis=1)
        coords = coords[keep]
        if not coords.shape[0]:
            return np.zeros(0, dtype=np.int64)
        keys = np.unique(np.ravel_multi_index(coords.T, self.dims))
        buckets = [self.cells.get(int(key)) for key in keys]
        parts = [self.order[slice(*bucket)] for bucket in buckets if bucket]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def near(self, xyz, radius):
        point = np.asarray(xyz, dtype=np.float64).reshape(-1)
        rows = self.candidate_rows(point, radius)
        if not rows.size:
            return rows
        delta = self.points[rows] - point
        return rows[np.einsum("ij,ij->i", delta, delta)
                    <= radius * radius * (1.0 + 1e-12)]

    def nearest_neighbour_distances(self, sample):
        """Distance from each sample row to its closest other point (inf if none)."""
        out = np.full(len(sample), np.inf)
        for row, point in enumerate(sample):
            rows = self.near(point, self.cell)
            rows = rows[rows != row]
            if rows.size:
                delta = self.points[rows] - point
                out[row] = float(np.min(np.einsum("ij,ij->i", delta, delta)) ** 0.5)
        return out


@lru_cache(maxsize=8)
def _neighbour_offsets(span):
    steps = np.arange(-span, span + 1, dtype=np.int64)
    return np.stack(np.meshgrid(steps, steps, steps, indexing="ij"),
                    axis=-1).reshape(-1, 3)


def _subsample(points, limit):
    if len(points) <= limit:
        return points
    return points[::math.ceil(len(points) / limit)]


def _estimate_spacing(points, *, max_samples=2000, tries=6):
    """Median nearest-neighbour distance: what the cloud actually resolves."""
    sample = _subsample(points, max_samples)
    if len(sample) < 2:
        return None
    span = float(np.max(sample.max(axis=0) - sample.min(axis=0)))
    cell = span / 20.0 if span > 0 else 1.0
    median = None
    for _ in range(tries):
        try:
            grid = _Grid(sample, max(cell, 1e-9))
        except ValueError:
            return None
        distances = grid.nearest_neighbour_distances(sample)
        finite = distances[np.isfinite(distances)]
        if not finite.size:
            cell *= 4.0
            continue
        median = float(np.median(finite))
        if median <= 0.0:
            cell = max(cell / 4.0, 1e-9)
            continue
        if 0.5 * cell <= median <= 2.0 * cell:
            return median
        cell = median
    return median if median and median > 0 else None


def _spacing_of(points, declared):
    """Declared spacing wins; otherwise measure it, and say which happened."""
    if declared is not None:
        return _number(declared, "point_spacing_m", non_negative=True), "declared"
    measured = _estimate_spacing(points)
    if measured:
        return measured, "measured median nearest-neighbour distance"
    return 0.0, "not measurable from this cloud"


# --------------------------------------------------------------------------- #
# Error budget.                                                               #
# --------------------------------------------------------------------------- #
def uncertainty(point_spacing_m=0.0, registration_residual_m=0.0, *, kind="length",
                n_points=2, n_samples=1, geometry_m=0.0, reference_m=None, power=1,
                size_m=None, cell_area_m2=None, n_cells=1):
    """Combine sampling, registration and geometry into a stated +/- value.

    Root sum of squares in metres: ``n_points`` independent coordinate errors of
    ``point_spacing/sqrt(12)`` each (one surface sample is good to about half a
    spacing, treated as a rectangular distribution), the registration residual
    reduced by ``sqrt(n_samples)`` with a median-efficiency factor of sqrt(pi/2),
    and a term from the measurement's own geometry (surface roughness, plane
    residual, column scatter). The metre figure is then expressed in the
    measurement's unit: metres, degrees when ``reference_m`` supplies a lever arm,
    m2/m3 through ``power`` and ``size_m`` for an area or volume built from
    measured edges, or m3 accumulated over a grid via ``cell_area_m2``/``n_cells``.

    This is an error budget, not an accuracy claim: correlated GNSS bias, timing
    error and unmodelled scale are absent, and the registration residual it consumes
    is itself a fit residual. With no spacing, no residual and no geometric term the
    record is invalid: an unquantified uncertainty is not a zero one.
    """
    spacing = _number(point_spacing_m, "point_spacing_m", non_negative=True)
    residual = _number(registration_residual_m, "registration_residual_m",
                       non_negative=True)
    geometry = _number(geometry_m, "geometry_m", non_negative=True)
    points = _count(n_points, "n_points")
    samples = _count(n_samples, "n_samples")
    if power not in (1, 2, 3):
        raise ValueError("power must be 1 (length), 2 (area) or 3 (volume)")
    lever = None if reference_m is None else _number(reference_m, "reference_m",
                                                     positive=True)
    cell = None if cell_area_m2 is None else _number(cell_area_m2, "cell_area_m2",
                                                     positive=True)
    cells = _count(n_cells, "n_cells")
    if power == 1 and size_m is not None:
        raise ValueError("size_m is only meaningful with power 2 or 3")
    if power > 1 and cell is None and size_m is None:
        raise ValueError("size_m must be given when propagating an area or volume")
    sampling = math.sqrt(points * spacing ** 2 / 12.0)
    registration = math.sqrt(points) * _MEDIAN_FACTOR * residual / math.sqrt(samples)
    base = math.sqrt(sampling ** 2 + registration ** 2 + geometry ** 2)
    assumptions = [
        "1-sigma (approximately 68%) coverage, not a 95% confidence interval.",
        "Sampling treats one cloud sample as uniform within half a spacing along "
        "the measured axis.",
        f"Registration residual is isotropic per point and averaged as a median of "
        f"{samples} points; it is a fit residual, not surveyed truth.",
        "Correlated bias (GNSS, scale, timing) is not included and does not average "
        "down with more points."]
    unit = "deg" if lever is not None else "m"
    if cell is not None:
        if power != 1:
            raise ValueError("cell_area_m2 accumulates lengths, not powers")
        plus_minus = cell * math.sqrt(cells) * base
        unit = "m3"
        assumptions.append(f"Accumulated over {cells} grid cells of {cell} m2 each; "
                           "cell-boundary inclusion error is not modelled.")
    elif lever is not None:
        plus_minus = math.degrees(math.atan(base / lever))
        assumptions.append(f"Converted to an angle using a {lever} m lever arm.")
    elif power == 2:
        size = _number(size_m, "size_m", non_negative=True)
        plus_minus = math.sqrt(2.0) * math.sqrt(size) * base
        unit = "m2"
        assumptions.append("Area error propagated as two independent sides of one "
                           "measured length.")
    elif power == 3:
        size = _number(size_m, "size_m", non_negative=True)
        plus_minus = math.sqrt(3.0) * size ** (2.0 / 3.0) * base
        unit = "m3"
        assumptions.append("Volume error propagated as three independent edges.")
    else:
        plus_minus = base
    if base <= 0.0:
        valid, reason = False, ("no point spacing, registration residual or geometric "
                                "term was declared: the uncertainty is unquantified, "
                                "not zero")
    elif plus_minus <= 0.0:
        valid, reason = False, ("the measurement has no extent to carry the error "
                                "budget, so this stated uncertainty is meaningless")
    else:
        valid, reason = True, None
        if residual == 0.0:
            assumptions.append("No registration residual was declared, so georeference "
                               "fit error is absent from this budget.")
    return dict(kind="uncertainty", unit=unit, value=plus_minus, plus_minus=plus_minus,
                plus_minus_m=base, confidence="1-sigma",
                components_m=dict(sampling_m=sampling, registration_m=registration,
                                  geometry_m=geometry),
                n_points=points, n_samples=samples, measured=str(kind),
                reference_m=lever, cell_area_m2=cell, cells=cells, valid=valid,
                reason=reason, assumptions=assumptions)


def quality_from_products(*, alignment=None, summary=None, points=None):
    """Assemble the declared error inputs from georeference/summary products.

    ``registration_residual_m`` is the georeference fit RMSE (or a summary's own
    declaration); ``point_spacing_m`` is taken from a summary when recorded and
    otherwise measured as the median nearest-neighbour distance of the cloud. Both
    stay None when unavailable rather than defaulting to a flattering zero.
    """
    result = dict(point_spacing_m=None, registration_residual_m=None,
                  spacing_basis=None, registration_basis=None)
    for name, value in (("alignment", alignment), ("summary", summary)):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{name} must be a dict or None")
    for source, key in ((alignment, "fit_rmse_m"), (summary, "registration_residual_m"),
                        (summary, "fit_rmse_m")):
        if source is None or source.get(key) is None:
            continue
        if result["registration_residual_m"] is None:
            result["registration_residual_m"] = _number(source[key], key,
                                                        non_negative=True)
            result["registration_basis"] = ("georeference fit_rmse_m"
                                            if key == "fit_rmse_m" else str(key))
    declared = None
    for source in (summary, alignment):
        if source is not None and source.get("point_spacing_m") is not None:
            declared = _number(source["point_spacing_m"], "point_spacing_m",
                               non_negative=True)
            break
    if declared is not None:
        result["point_spacing_m"] = declared
        result["spacing_basis"] = "declared point_spacing_m"
    elif points is not None:
        measured = _estimate_spacing(_cloud(points))
        if measured:
            result["point_spacing_m"] = measured
            result["spacing_basis"] = "measured median nearest-neighbour distance"
    return result


def _quality(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("quality must be a mapping of declared error inputs")
    result = {}
    for key in ("point_spacing_m", "registration_residual_m"):
        if value.get(key) is not None:
            result[key] = _number(value[key], key, non_negative=True)
    return result


def _budget(quality, spacing, **kwargs):
    """One call site's error budget, from declared or measured cloud spacing."""
    residual = quality.get("registration_residual_m") or 0.0
    return uncertainty(point_spacing_m=spacing, registration_residual_m=residual,
                       **kwargs)


# --------------------------------------------------------------------------- #
# Geometry primitives and record building.                                    #
# --------------------------------------------------------------------------- #
def _svd(centred):
    """Singular values and vectors, with a non-convergence turned into ValueError."""
    try:
        return np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"shape fit did not converge: {exc}") from exc


def _plane_from_points(points):
    """Total-least-squares plane: centroid, unit normal with +up, rms, planarity."""
    middle = points.mean(axis=0)
    centred = points - middle
    _, singular, vt = _svd(centred)
    if singular.size < 2 or singular[0] <= 0.0:
        return None
    if singular[1] / singular[0] < _PLANARITY:
        return None
    normal = vt[2]
    if normal[2] < 0:
        normal = -normal
    residual = np.abs(centred @ normal)
    return dict(point=middle, normal=np.asarray(normal, dtype=np.float64),
                rms=float(np.sqrt(np.mean(residual ** 2))),
                planarity=float(singular[1] / singular[0]))


def _area_vector(vertices):
    """Newell's method: the area-weighted normal of a 3-D polygon."""
    return 0.5 * np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)


def _record(kind, unit, value, *, uncertainty, valid, reason=None, geometry=None,
            **fields):
    result = dict(kind=kind, unit=unit, value=_safe(value, f"{kind} value"),
                  uncertainty=uncertainty, valid=bool(valid), reason=_text(reason))
    if geometry is not None:
        result["geometry_enu"] = geometry
    result.update(fields)
    result.setdefault("geometry_enu", None)
    return result


def _line(start, end):
    if start is None or end is None:
        return None
    return dict(type="LineString",
                coordinates=[np.asarray(start, dtype=np.float64).tolist(),
                             np.asarray(end, dtype=np.float64).tolist()])


def _point(xyz):
    if xyz is None:
        return None
    return dict(type="Point",
                coordinates=np.asarray(xyz, dtype=np.float64).tolist())


def _polygon(vertices):
    ring = [list(np.asarray(vertex, dtype=np.float64).tolist()) for vertex in vertices]
    if len(ring) > 2 and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return dict(type="Polygon", coordinates=[ring])


def _footprint(vertices):
    """Closed horizontal ring around the given points, at their lowest elevation."""
    values = np.asarray(vertices, dtype=np.float64)
    low, high = values.min(axis=0), values.max(axis=0)
    corners = [[low[0], low[1], low[2]], [high[0], low[1], low[2]],
               [high[0], high[1], low[2]], [low[0], high[1], low[2]]]
    return _polygon(corners)


# --------------------------------------------------------------------------- #
# Point density: how much geometry actually supports a location.              #
# --------------------------------------------------------------------------- #
def local_density(points, xyz, *, radius_m=None, min_neighbours=6, spacing_m=None):
    """How much cloud sits within a radius of a location, and what it resolves.

    ``count`` includes the sample point itself, so an isolated noise point reports
    count 1. ``spacing_m`` is the measured median nearest-neighbour distance inside
    the neighbourhood - what a ruler could actually resolve there.
    ``points_per_m2`` assumes a locally planar neighbourhood, which is stated in
    ``basis`` instead of being silently baked into the number. This is a supporting
    record: it has its own unit and validity flag and shares the error budget of the
    measurement that embeds it, which is what ``distance`` and ``height_above`` do.
    """
    cloud = _cloud(points)
    centre = np.asarray(_vector(xyz, "xyz"), dtype=np.float64)
    minimum = _count(min_neighbours, "min_neighbours")
    declared = None if spacing_m is None else _number(spacing_m, "spacing_m",
                                                      non_negative=True)
    fallback = declared if declared is not None else _estimate_spacing(cloud)
    radius = radius_m if radius_m is not None else max(5.0 * (fallback or 0.0), 0.01)
    radius = _number(radius, "radius_m", positive=True)
    rows = _Grid(cloud, radius).near(centre, radius)
    count = int(rows.size)
    spacing = 0.0
    if count >= 2:
        sample = _subsample(cloud[rows], 400)
        difference = sample[:, None, :] - sample[None, :, :]
        squared = np.einsum("ijk,ijk->ij", difference, difference)
        np.fill_diagonal(squared, np.inf)
        spacing = float(np.median(np.sqrt(squared.min(axis=1))))
    area, volume = math.pi * radius ** 2, 4.0 / 3.0 * math.pi * radius ** 3
    sufficient = count >= minimum
    where = f"({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f})"
    reason = None if sufficient else (
        f"only {count} point(s) within {radius:.3f} m of {where}: this location "
        f"spans empty space (need {minimum})")
    return dict(kind="point_density", unit="points_per_m2", value=count / area,
                count=count, radius_m=radius, points_per_m2=count / area,
                points_per_m3=count / volume, spacing_m=spacing,
                spacing_basis="median nearest-neighbour distance in the neighbourhood",
                min_neighbours=minimum, sufficient=bool(sufficient), valid=sufficient,
                reason=reason,
                basis="points_per_m2 assumes a locally planar neighbourhood",
                geometry_enu=_point(centre))


# --------------------------------------------------------------------------- #
# Distance and snapped segments.                                              #
# --------------------------------------------------------------------------- #
def _index(value, name, size):
    index = _count(value, name, minimum=0) if isinstance(value, (int, np.integer)) \
        and not isinstance(value, bool) else None
    if index is None:
        raise ValueError(f"{name} must be an integer index into points")
    if not 0 <= index < size:
        raise ValueError(f"{name} must index points 0..{size - 1}")
    return index


def distance(points, a, b, *, radius_m=None, min_neighbours=10, quality=None):
    """3-D, horizontal and vertical separation of two cloud points, with support.

    Reports the local point density around each endpoint: a 5 m "distance" between
    two isolated noise points is arithmetic, not a measurement, so it comes back
    valid=False with the reason. The three numbers are present because the
    arithmetic is defined for any two points; the validity flag says whether
    measured geometry stands behind them.

    ``horizontal_m`` is the east-north separation, ``vertical_m`` the absolute up
    separation, ``rise_m`` its signed form, and ``value`` the 3-D length.
    """
    cloud = _cloud(points)
    if len(cloud) < 2:
        raise ValueError("distance needs at least two points in the cloud")
    first, second = _index(a, "a", len(cloud)), _index(b, "b", len(cloud))
    if first == second:
        raise ValueError("distance indices a and b must differ")
    declared = _quality(quality)
    spacing, spacing_basis = _spacing_of(cloud, declared.get("point_spacing_m"))
    radius = None if radius_m is None else _number(radius_m, "radius_m", positive=True)
    minimum = _count(min_neighbours, "min_neighbours")
    density_a = local_density(cloud, cloud[first], radius_m=radius,
                              min_neighbours=minimum, spacing_m=spacing)
    density_b = local_density(cloud, cloud[second], radius_m=radius,
                              min_neighbours=minimum, spacing_m=spacing)
    delta = cloud[second] - cloud[first]
    length = float(np.linalg.norm(delta))
    horizontal = float(math.hypot(float(delta[0]), float(delta[1])))
    budget = _budget(declared, max(density_a["spacing_m"], density_b["spacing_m"],
                                   spacing), kind="distance", n_points=2,
                     n_samples=max(min(density_a["count"], density_b["count"]), 1))
    complaints = [record["reason"] for record in (density_a, density_b)
                  if not record["sufficient"]]
    return _record("distance", "m", length, uncertainty=budget,
                   valid=not complaints,
                   reason=None if not complaints else "endpoint " +
                        "; endpoint ".join(complaints),
                   length_m=length, horizontal_m=horizontal,
                   vertical_m=float(abs(delta[2])), rise_m=float(delta[2]),
                   delta_enu=delta.tolist(), indices=[first, second],
                   coordinates_enu={"a": cloud[first].tolist(),
                                    "b": cloud[second].tolist()},
                   density={"a": density_a, "b": density_b},
                   radius_m=density_a["radius_m"], min_neighbours=minimum,
                   spacing_basis=spacing_basis, geometry_enu=_line(cloud[first],
                                                                   cloud[second]))


def _snap(grid, cloud, point, radius, minimum, name):
    """Median-of-neighbours snap of one click, with the distance moved reported."""
    rows = grid.near(point, radius)
    count = int(rows.size)
    clicked = np.asarray(point, dtype=np.float64).tolist()
    if count < minimum:
        detail = (f"{name} has no points within {radius:.3f} m of "
                  f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})" if not count
                  else f"{name} has only {count} points within {radius:.3f} m of "
                       f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) "
                       f"(need {minimum})")
        return dict(kind="snap", unit="m", value=None, clicked_enu=clicked,
                    snapped_enu=None, snap_distance_m=None, neighbour_count=count,
                    radius_m=radius, neighbourhood_rms_m=None, valid=False,
                    reason=detail, basis="median of the cloud within radius_m",
                    geometry_enu=_point(point)), None
    neighbourhood = cloud[rows]
    snapped = np.median(neighbourhood, axis=0)
    offset = float(np.linalg.norm(snapped - point))
    spread = float(np.sqrt(np.mean(np.sum((neighbourhood - snapped) ** 2, axis=1))))
    return dict(kind="snap", unit="m", value=offset, clicked_enu=clicked,
                snapped_enu=snapped.tolist(), snap_distance_m=offset,
                neighbour_count=count, radius_m=radius, neighbourhood_rms_m=spread,
                valid=True, reason=None,
                basis="median of the cloud within radius_m",
                geometry_enu=_point(snapped)), snapped


def measure_segment(points, start_xyz, end_xyz, *, radius_m, axis="auto",
                    min_neighbours=6, near_vertical_deg=30.0, quality=None):
    """Snap two human clicks onto the surface, then measure between the snaps.

    A click lands wherever the ray happened to hit, so each endpoint is replaced by
    the median of the cloud within ``radius_m`` (robust to one stray noise point),
    and ``snap_distance_m`` says how far the click was from real geometry. A large
    snap distance is the user telling you they pointed at nothing, so it is reported
    rather than swallowed, and a note says when it exceeded half the snap radius.

    ``axis='auto'`` chooses ``height`` when the snapped segment is within
    ``near_vertical_deg`` of vertical and ``length`` otherwise; ``axis_rule`` states
    the verticality it measured and the threshold it compared against, so the choice
    is auditable. ``horizontal`` (east-north only) can be requested explicitly. All
    three numbers are always in the record; ``value`` is the one the axis asks for.
    """
    cloud = _cloud(points)
    clicked = {"start": _vector(start_xyz, "start_xyz"),
               "end": _vector(end_xyz, "end_xyz")}
    radius = _number(radius_m, "radius_m", positive=True)
    minimum = _count(min_neighbours, "min_neighbours")
    tilt = _number(near_vertical_deg, "near_vertical_deg", positive=True)
    if axis not in ("auto",) + _AXES:
        raise ValueError(f"axis must be 'auto' or one of {_AXES}")
    declared = _quality(quality)
    spacing, spacing_basis = _spacing_of(cloud, declared.get("point_spacing_m"))
    grid = _Grid(cloud, radius)
    snapping, snapped, rough = {}, {}, []
    for name, point in clicked.items():
        record, position = _snap(grid, cloud, point, radius, minimum, name)
        snapping[name] = record
        if position is not None:
            snapped[name] = position
            rough.append(record["neighbourhood_rms_m"])
    complaints = [record["reason"] for record in snapping.values()
                  if not record["valid"]]
    if complaints:
        return _record("segment", "m", None,
                       uncertainty=_budget(declared, spacing, kind="segment",
                                           n_points=2, n_samples=1),
                       valid=False, reason="; ".join(complaints), axis=None,
                       axis_rule=None, verticality=None, local_roughness_m=None,
                       notes=[], length_m=None, horizontal_m=None, vertical_m=None,
                       radius_m=radius, min_neighbours=minimum,
                       near_vertical_deg=tilt, snapping=snapping,
                       max_snap_distance_m=None, spacing_basis=spacing_basis,
                       geometry_enu=None)
    delta = snapped["end"] - snapped["start"]
    length = float(np.linalg.norm(delta))
    horizontal = float(math.hypot(float(delta[0]), float(delta[1])))
    vertical = float(abs(delta[2]))
    counts = (snapping["start"]["neighbour_count"], snapping["end"]["neighbour_count"])
    budget = _budget(declared, spacing, kind=f"segment.{axis}", n_points=2,
                     n_samples=max(min(counts), 1), geometry_m=float(np.mean(rough)))
    common = dict(snapping=snapping, radius_m=radius, min_neighbours=minimum,
                  near_vertical_deg=tilt, length_m=length, horizontal_m=horizontal,
                  vertical_m=vertical, notes=[],
                  max_snap_distance_m=max(record["snap_distance_m"] for record in
                                          snapping.values()),
                  spacing_basis=spacing_basis,
                  local_roughness_m=budget["components_m"]["geometry_m"],
                  geometry_enu=_line(snapped["start"], snapped["end"]))
    if length <= 0.0:
        return _record("segment", "m", None, uncertainty=budget, valid=False,
                       reason="both clicks snapped to the same surface point: the "
                              "segment has no length to measure", axis=None,
                       axis_rule=None, verticality=0.0, **common)
    verticality = vertical / length
    threshold = math.cos(math.radians(min(tilt, 90.0)))
    if axis == "auto":
        chosen = "height" if verticality >= threshold else "length"
        rule = (f"auto: verticality |dU|/|d| = {verticality:.3f} "
                f"{'>=' if verticality >= threshold else '<'} {threshold:.3f} "
                f"({tilt:.1f} deg from vertical) -> {chosen}")
    else:
        chosen, rule = axis, f"explicit axis request: {axis}"
    snap_max = common["max_snap_distance_m"]
    if snap_max > radius / 2.0:
        common["notes"] = [f"a click was {snap_max:.3f} m from the measured surface, "
                           f"more than half the {radius:.3f} m snap radius: check what "
                           f"it was meant to touch"]
    fields = {"height": "vertical_m", "length": "length_m",
              "horizontal": "horizontal_m"}
    return _record("segment", "m", common[fields[chosen]], uncertainty=budget,
                   valid=True, reason=None, axis=chosen, axis_rule=rule,
                   verticality=verticality, verticality_threshold=threshold, **common)


# --------------------------------------------------------------------------- #
# Areas: footprint, true surface, projection.                                 #
# --------------------------------------------------------------------------- #
def polygon_area(points_2d, *, quality=None):
    """Shoelace area of a planar ring in the east-north plane.

    The signed value is kept (``signed_area_m2``, with ``orientation``) so a
    reversed or self-crossed outline is visible instead of being hidden by an
    ``abs`` call. A collinear or zero-width ring is invalid rather than reported as
    a 0 m2 floor, because 0 m2 is a claim a judge would repeat.

    Up-coordinates are ignored when 3-D vertices are given: this is the footprint.
    For a sloped surface use ``true_area`` or ``projected_area``, otherwise a roof
    measured on a slope gets reported as its shadow.
    """
    vertices = _ring(points_2d, "points_2d")
    if len(vertices) < 3:
        raise ValueError("polygon_area needs at least three vertices")
    declared = _quality(quality)
    spacing = declared.get("point_spacing_m") or 0.0
    closed = bool(np.all(vertices[0] == vertices[-1]))
    ring = vertices[:-1] if closed and len(vertices) > 3 else vertices
    points_ = max(len(ring), 3)

    def budget(area):
        return _budget(declared, spacing, kind="area", n_points=points_, power=2,
                       size_m=area)

    if len(ring) < 3:
        return _record("polygon_area", "m2", 0.0, uncertainty=budget(0.0), valid=False,
                       reason="fewer than three distinct vertices remain once the "
                              "ring is closed: there is no polygon",
                       signed_area_m2=0.0, orientation="none", vertex_count=len(ring),
                       closed=closed, plane="horizontal", perimeter_m=None,
                       extent_m=None, vertices_enu=ring.tolist(), geometry_enu=None)
    east, north = ring[:, 0], ring[:, 1]
    signed = 0.5 * float(np.sum(east * np.roll(north, -1)
                                - north * np.roll(east, -1)))
    value = abs(signed)
    middle = ring[:, :2].mean(axis=0)
    centred = ring[:, :2] - middle
    singular = _svd(centred)[1]
    extent = float(np.max(np.linalg.norm(centred, axis=1)))
    collinear = bool(singular[0] <= 0.0 or singular[-1] / singular[0] < 1e-9)
    reason = None
    if collinear:
        reason = ("all vertices are collinear: the ring is a line, so its area is "
                  "not a measurement of anything")
    elif value <= 1e-9 * max(extent ** 2, 1.0):
        reason = "the ring encloses no area at this scale: a zero-width polygon"
    perimeter = float(np.sum(np.linalg.norm(np.roll(ring, -1, axis=0)[:, :2]
                                            - ring[:, :2], axis=1)))
    return _record("polygon_area", "m2", value, uncertainty=budget(value),
                   valid=reason is None, reason=reason, signed_area_m2=signed,
                   orientation="ccw" if signed > 0 else ("cw" if signed < 0 else "none"),
                   vertex_count=len(ring), closed=closed, plane="horizontal",
                   perimeter_m=perimeter, extent_m=extent,
                   vertices_enu=ring.tolist(), geometry_enu=_polygon(ring))


def true_area(triangle_vertices, *, quality=None):
    """True surface area of one triangle in 3-D: no foreshortening, no shadow.

    This is the number for a sloped roof panel or a single facet, measured on the
    facet itself; ``footprint_area_m2`` is its horizontal shadow, so the gap between
    them is exactly the error a footprint-only measurement would make. Collinear
    vertices are invalid, not zero.
    """
    vertices = _ring(triangle_vertices, "triangle_vertices")
    if len(vertices) != 3:
        raise ValueError("true_area takes exactly three vertices of one triangle")
    declared = _quality(quality)
    spacing = declared.get("point_spacing_m") or 0.0
    crossed = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    norm = float(np.linalg.norm(crossed))
    edges = [float(np.linalg.norm(vertices[i] - vertices[j]))
             for i, j in ((0, 1), (1, 2), (2, 0))]
    longest = max(edges)
    value = 0.5 * norm
    budget = _budget(declared, spacing, kind="area", n_points=3, power=2, size_m=value)
    if norm <= 1e-12 * max(longest ** 2, 1.0):
        return _record("true_area", "m2", 0.0, uncertainty=budget, valid=False,
                       reason="triangle vertices are collinear: the facet has no "
                              "area, so tilt is undefined", normal=None, tilt_deg=None,
                       footprint_area_m2=None, area_vector=None, perimeter_m=None,
                       longest_edge_m=longest, vertices_enu=vertices.tolist(),
                       geometry_enu=None)
    normal = crossed / norm
    if normal[2] < 0:
        normal = -normal
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    return _record("true_area", "m2", value, uncertainty=budget, valid=True,
                   reason=None, normal=normal.tolist(), tilt_deg=tilt,
                   footprint_area_m2=value * abs(float(normal[2])),
                   area_vector=(0.5 * crossed).tolist(), perimeter_m=sum(edges),
                   longest_edge_m=longest, vertices_enu=vertices.tolist(),
                   geometry_enu=_polygon(vertices))


def projected_area(surface, normal, *, quality=None):
    """How much of a surface actually faces a plane: the roof-versus-shadow fix.

    ``surface`` is a 3-D polygon (its area and normal come from Newell's method) or
    a mapping with ``area_m2`` and ``normal``. The value is
    |A * n_surface . n_projection|, so projecting a slope onto [0, 0, 1] gives its
    footprint, onto a wall normal gives the elevation area, and onto the surface's
    own normal gives the true area.

    A surface seen edge-on projects to a line, not to a 0 m2 area, so that case is
    invalid with a reason rather than a number that reads like a result.
    """
    axis = _vector(normal, "normal")
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        raise ValueError("normal must have non-zero length")
    axis = axis / length
    declared = _quality(quality)
    spacing = declared.get("point_spacing_m") or 0.0
    vertices = None
    if isinstance(surface, dict):
        if surface.get("area_m2") is None or surface.get("normal") is None:
            raise ValueError("a surface mapping needs area_m2 and normal")
        area = _number(surface["area_m2"], "area_m2", non_negative=True)
        face = _vector(surface["normal"], "surface normal")
        norm = float(np.linalg.norm(face))
        if norm == 0.0:
            raise ValueError("surface normal must have non-zero length")
        face = face / norm
    else:
        vertices = _ring(surface, "surface")
        if len(vertices) < 3:
            raise ValueError("a surface polygon needs at least three vertices")
        vector = _area_vector(vertices)
        area = float(np.linalg.norm(vector))
        face = vector / area if area > 0 else vector
    cos_theta = float(abs(np.dot(face, axis)))
    value = area * cos_theta
    budget = _budget(declared, spacing, kind="projected_area", n_points=3, power=2,
                     size_m=value)
    if area <= 0.0:
        reason = "the surface itself has no area: there is nothing to project"
    elif cos_theta <= 1e-9:
        reason = ("the surface is edge-on to this projection plane: the projection is "
                  "a line, so no area can be reported")
    else:
        reason = None
    return _record("projected_area", "m2", None if reason else value,
                   uncertainty=budget, valid=reason is None, reason=reason,
                   surface_area_m2=area, surface_normal=face.tolist(),
                   projection_normal=axis.tolist(), cos_theta=cos_theta,
                   tilt_to_plane_deg=math.degrees(math.acos(max(-1.0, min(1.0,
                                                                          cos_theta)))),
                   vertices_enu=None if vertices is None else vertices.tolist(),
                   geometry_enu=None if vertices is None else _polygon(vertices))


# --------------------------------------------------------------------------- #
# Ground plane and heights.                                                   #
# --------------------------------------------------------------------------- #
def ground_plane(points, *, ransac_distance=0.05, min_inliers=200, max_tilt_deg=25.0,
                 max_below_fraction=0.05, iterations=300, seed=0, quality=None):
    """Robustly fit the ground: the lower envelope, not the lowest points.

    Deterministic RANSAC over triples, keeping only near-horizontal candidates
    (within ``max_tilt_deg`` of vertical) that have at most ``max_below_fraction``
    of the cloud hanging below them. That second condition is what makes this a
    ground detector rather than a plane detector: a flat roof has as much inlier
    support as the terrain and would win a plain "most inliers" fit, which would
    then make every building height come out near zero.

    The winner is refitted by total least squares on its inliers and the inlier set
    recomputed to convergence, reporting ``inlier_count``, ``residual_rms_m``,
    ``below_fraction`` and the ``support_bbox_enu`` it was measured over. Fewer than
    ``min_inliers`` inliers is invalid: the plane may be worth looking at, but it is
    not evidence of ground.
    """
    cloud = _cloud(points)
    if len(cloud) < 3:
        raise ValueError("ground_plane needs at least three points")
    tolerance = _number(ransac_distance, "ransac_distance", positive=True)
    minimum = _count(min_inliers, "min_inliers", minimum=3)
    tilt_limit = _number(max_tilt_deg, "max_tilt_deg", positive=True)
    below_limit = _number(max_below_fraction, "max_below_fraction", non_negative=True)
    trials = _count(iterations, "iterations", minimum=8)
    rng = np.random.default_rng(_count(seed, "seed", minimum=0))
    cos_limit = math.cos(math.radians(min(tilt_limit, 90.0)))
    scoring = _subsample(cloud, 50000)
    best = None
    for triple in rng.integers(0, len(scoring), size=(trials, 3)):
        origin, a, b = (scoring[int(triple[0])], scoring[int(triple[1])],
                        scoring[int(triple[2])])
        normal = np.cross(a - origin, b - origin)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        if float(normal[2]) < cos_limit:
            continue
        offset = -float(np.dot(normal, origin))
        residual = scoring @ normal + offset
        inside = np.abs(residual) <= tolerance
        if not inside.any() or float(np.mean(residual < -tolerance)) > below_limit:
            continue
        count = int(inside.sum())
        score = (count, -float(np.mean(scoring[inside, 2])))
        if best is None or score > best[0]:
            best = (score, normal, offset)
    declared = _quality(quality)
    empty = dict(normal=None, point_enu=None, coefficients=None, elevation_m=None,
                 residual_rms_m=None, inlier_count=0, total_points=len(cloud),
                 inlier_fraction=0.0, below_fraction=None, support_bbox_enu=None,
                 tilt_deg_from_vertical=None, planarity=None, ransac_distance_m=tolerance,
                 iterations=trials, max_tilt_deg=tilt_limit, min_inliers=minimum,
                 method="deterministic_ransac_lower_envelope",
                 geometry_enu=None)
    if best is None:
        return _record("ground_plane", "m", None,
                       uncertainty=_budget(declared, 0.0, kind="plane", n_points=1),
                       valid=False,
                       reason=f"no near-horizontal plane with at most "
                              f"{below_limit:.2f} of the cloud below it appeared in "
                              f"{trials} deterministic RANSAC trials", **empty)
    _, normal, offset = best
    rows = np.flatnonzero(np.abs(cloud @ normal + offset) <= tolerance)
    for _ in range(4):
        fit = _plane_from_points(cloud[rows])
        if fit is None:
            break
        candidate = fit["normal"]
        shift = -float(np.dot(candidate, fit["point"]))
        updated = np.flatnonzero(np.abs(cloud @ candidate + shift) <= tolerance)
        if updated.size < 3 or (updated.size == rows.size
                                and np.array_equal(updated, rows)):
            break
        normal, offset, rows = candidate, shift, updated
    else:
        fit = _plane_from_points(cloud[rows])
    spacing, spacing_basis = _spacing_of(cloud, declared.get("point_spacing_m"))
    if fit is None:
        return _record("ground_plane", "m", None,
                       uncertainty=_budget(declared, spacing, kind="plane", n_points=1),
                       valid=False,
                       reason="the ground inlier set is rank-deficient: a plane cannot "
                              "be fitted", **dict(empty, inlier_count=int(rows.size)))
    normal, offset = fit["normal"], -float(np.dot(fit["normal"], fit["point"]))
    residual = np.abs(cloud[rows] @ normal + offset)
    all_residual = cloud @ normal + offset
    rms = float(np.sqrt(np.mean(residual ** 2)))
    slope = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    elevation = float(fit["point"][2])
    budget = _budget(declared, spacing, kind="plane", n_points=1,
                     n_samples=max(int(rows.size), 1), geometry_m=rms)
    valid = int(rows.size) >= minimum
    low, high = cloud[rows].min(axis=0), cloud[rows].max(axis=0)
    return _record("ground_plane", "m", elevation, uncertainty=budget, valid=valid,
                   reason=None if valid else f"only {int(rows.size)} ground inliers "
                                             f"within {tolerance:.3f} m (min "
                                             f"{minimum}): this is not established "
                                             f"ground",
                   normal=normal.tolist(), point_enu=fit["point"].tolist(),
                   coefficients=[float(normal[0]), float(normal[1]), float(normal[2]),
                                 offset],
                   elevation_m=elevation, residual_rms_m=rms,
                   residual_max_m=float(residual.max()), inlier_count=int(rows.size),
                   total_points=len(cloud),
                   inlier_fraction=float(rows.size) / len(cloud),
                   below_fraction=float(np.mean(all_residual < -tolerance)),
                   support_bbox_enu=[[float(low[0]), float(low[1])],
                                     [float(high[0]), float(high[1])]],
                   tilt_deg_from_vertical=slope, planarity=fit["planarity"],
                   ransac_distance_m=tolerance, iterations=trials,
                   max_tilt_deg=tilt_limit, min_inliers=minimum,
                   method="deterministic_ransac_lower_envelope",
                   spacing_basis=spacing_basis,
                   geometry_enu=_footprint(cloud[rows]),
                   note="support_bbox_enu is where ground was actually measured; "
                        "heights outside it are extrapolation")


def _plane_of(value):
    """Accept a ground_plane record, [a, b, c, d] coefficients, or an elevation."""
    if isinstance(value, (int, float, np.integer, np.floating)) and \
            not isinstance(value, (bool, np.bool_)):
        elevation = _number(value, "ground elevation")
        return dict(normal=np.array([0.0, 0.0, 1.0]),
                    point=np.array([0.0, 0.0, elevation]), support=None, residual=0.0,
                    basis="declared horizontal ground elevation")
    if isinstance(value, dict):
        coefficients = np.asarray(value.get("coefficients", []), dtype=np.float64)
        if value.get("normal") is not None:
            normal = _vector(value["normal"], "plane normal")
        elif coefficients.size == 4:
            normal = _vector(coefficients[:3], "plane normal")
        else:
            raise ValueError("a plane record needs normal, or coefficients [a,b,c,d]")
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("plane normal must have non-zero length")
        normal = normal / norm
        if value.get("point_enu") is not None:
            point = _vector(value["point_enu"], "plane point")
        elif value.get("point") is not None:
            point = _vector(value["point"], "plane point")
        elif coefficients.size == 4:
            point = (-float(coefficients[3]) / norm) * normal
        else:
            raise ValueError("a plane record needs point_enu or coefficients "
                             "[a, b, c, d]")
        residual = value.get("residual_rms_m")
        return dict(normal=normal, point=point, support=value.get("support_bbox_enu"),
                    residual=0.0 if residual is None else _number(residual,
                                                                  "residual_rms_m",
                                                                  non_negative=True),
                    basis="fitted ground plane")
    coefficients = _rows(np.asarray(value, dtype=np.float64), "plane coefficients") \
        .reshape(-1)
    if coefficients.size != 4:
        raise ValueError("a plane is a record, [a, b, c, d], or a ground elevation")
    normal = _vector(coefficients[:3], "plane normal")
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("plane normal must have non-zero length")
    normal = normal / norm
    return dict(normal=normal, point=(-float(coefficients[3]) / norm) * normal,
                support=None, residual=0.0, basis="plane coefficients [a,b,c,d]")


def height_above(points, plane_or_ground, target, *, support_margin_m=0.0,
                 radius_m=None, quality=None):
    """Height of a facility above a fitted ground plane, not above the cloud floor.

    The lowest point in a cloud is usually a drain, a noise spike or an occlusion
    edge, and on sloping ground "bottom of the bounding box" makes every building
    the same height. This measures the vertical distance from the plane through the
    target's own footprint, and reports the perpendicular distance alongside.

    Invalid, with no value, when the target's footprint falls outside the ground the
    plane was actually fitted over - no measured ground under a building is not a
    height - and invalid with its (negative) value when the target sits below the
    plane, which is a cut, not a facility.
    """
    cloud = _cloud(points)
    declared = _quality(quality)
    spacing, spacing_basis = _spacing_of(cloud, declared.get("point_spacing_m"))
    if isinstance(target, (int, np.integer)) and not isinstance(target, (bool, np.bool_)):
        index = _index(target, "target", len(cloud))
        location, described = cloud[index], f"point index {index}"
    else:
        index, location = None, _vector(target, "target")
        described = "coordinate"
    plane = _plane_of(plane_or_ground)
    density = local_density(cloud, location, radius_m=radius_m, spacing_m=spacing)
    budget = _budget(declared, max(density["spacing_m"], spacing), kind="height",
                     n_points=2, n_samples=max(density["count"], 1),
                     geometry_m=float(plane["residual"]))
    base = dict(target_enu=location.tolist(), target_index=index, density=density,
                plane_basis=plane["basis"], spacing_basis=spacing_basis,
                ground_normal=[float(x) for x in plane["normal"]],
                perpendicular_m=None, plane_elevation_at_target_m=None,
                geometry_enu=None)
    vertical = float(plane["normal"][2])
    if abs(vertical) < 1e-6:
        return _record("height_above", "m", None, uncertainty=budget, valid=False,
                       reason="the supplied plane is vertical: height above it is "
                              "undefined", **base)
    east, north = float(plane["normal"][0]), float(plane["normal"][1])
    height_at = (float(np.dot(plane["normal"], plane["point"]))
                 - east * float(location[0]) - north * float(location[1])) / vertical
    value = float(location[2]) - height_at
    foot = np.array([location[0], location[1], height_at], dtype=np.float64)
    base.update(plane_elevation_at_target_m=height_at,
                perpendicular_m=float(np.dot(plane["normal"], location - plane["point"])),
                geometry_enu=_line(foot, location))
    margin = _number(support_margin_m, "support_margin_m", non_negative=True)
    if plane["support"] is not None:
        low, high = np.asarray(plane["support"], dtype=np.float64)
        if not (low[0] - margin <= location[0] <= high[0] + margin
                and low[1] - margin <= location[1] <= high[1] + margin):
            return _record("height_above", "m", None, uncertainty=budget, valid=False,
                           reason=f"the {described} target ({location[0]:.2f}, "
                                  f"{location[1]:.2f}) is outside the ground plane "
                                  f"support ({low[0]:.2f}..{high[0]:.2f} E, "
                                  f"{low[1]:.2f}..{high[1]:.2f} N): there is no "
                                  f"measured ground plane under it", **base)
    if value < 0:
        return _record("height_above", "m", value, uncertainty=budget, valid=False,
                       reason=f"the target sits {abs(value):.3f} m below the ground "
                              f"plane: either the plane is wrong or this is a cut, not "
                              f"a facility height", **base)
    return _record("height_above", "m", value, uncertainty=budget, valid=True,
                   reason=None, **base)


# --------------------------------------------------------------------------- #
# Volumes and change.                                                         #
# --------------------------------------------------------------------------- #
def _columns(cloud, cell, origin, dims):
    """Median surface height, point count and scatter for each grid column."""
    index = np.floor((cloud[:, :2] - origin) / cell).astype(np.int64)
    if np.any(index < 0) or not np.all(index < np.array(dims, dtype=np.int64)):
        raise ValueError("grid columns fell outside the computed extent")
    keys = np.ravel_multi_index(index.T, dims)
    order = np.lexsort((cloud[:, 2], keys))
    sorted_keys, heights = keys[order], cloud[order, 2]
    unique, starts = np.unique(sorted_keys, return_index=True)
    counts = np.append(starts[1:], sorted_keys.size) - starts
    middle = starts + counts // 2
    median = np.where(counts % 2 == 1, heights[middle],
                      0.5 * (heights[middle] + heights[middle - 1]))
    sums = np.add.reduceat(heights, starts)
    squares = np.add.reduceat(heights ** 2, starts)
    mean = sums / counts
    variance = np.maximum(squares / counts - mean ** 2, 0.0)
    return dict(keys=unique, median=np.asarray(median, dtype=np.float64),
                counts=counts, variance=np.asarray(variance, dtype=np.float64))


def volume_between(cloud_a, cloud_b, *, cell_size_m, min_overlap_fraction=0.6,
                   min_points_per_cell=1, quality=None):
    """Cut/fill and change volume on a regular east-north grid, overlap included.

    Each cloud becomes one surface height per grid column (the median of its z
    values there) and the two are differenced where both have data. ``cut_m3`` is
    material above the second surface and ``fill_m3`` material below it, with
    ``value`` the net (cut positive): ``cloud_a`` reads as the existing/earlier
    surface and ``cloud_b`` as the design/later one, and swapping them reverses the
    sign.

    The honesty term is the cell census - ``cells_both_count`` against
    ``cells_a_only_count`` and ``cells_b_only_count``. A volume over a partial
    overlap is a lie, so anything below ``min_overlap_fraction`` shared cells comes
    back valid=False with the fraction stated, and zero shared cells yields no value
    at all rather than a tidy 0.0 m3.
    """
    first, second = _cloud(cloud_a, "cloud_a"), _cloud(cloud_b, "cloud_b")
    cell = _number(cell_size_m, "cell_size_m", positive=True)
    threshold = _number(min_overlap_fraction, "min_overlap_fraction", non_negative=True)
    if threshold > 1:
        raise ValueError("min_overlap_fraction must be between 0 and 1")
    per_cell = _count(min_points_per_cell, "min_points_per_cell")
    east_low, north_low = np.minimum(first[:, :2], second[:, :2]).min(axis=0)
    east_high, north_high = np.maximum(first[:, :2], second[:, :2]).max(axis=0)
    origin = np.array([east_low, north_low], dtype=np.float64)
    shape = np.floor(np.array([east_high - east_low, north_high - north_low]) / cell)
    dims = tuple(int(edge) + 1 for edge in shape)
    if float(np.prod(np.array(dims, dtype=np.float64))) > 2.0 ** 55:
        raise ValueError("cell_size_m is too fine for this extent")
    columns_a = _columns(first, cell, origin, dims)
    columns_b = _columns(second, cell, origin, dims)
    shared, ia, ib = np.intersect1d(columns_a["keys"], columns_b["keys"],
                                    assume_unique=True, return_indices=True)
    both = int(shared.size)
    keep = np.where((columns_a["counts"][ia] >= per_cell)
                    & (columns_b["counts"][ib] >= per_cell))[0] if both else ia
    delta = (columns_a["median"][ia][keep] - columns_b["median"][ib][keep]) if both \
        else np.zeros(0)
    area = cell * cell
    cut = float(np.sum(np.clip(delta, 0.0, None)) * area)
    fill = float(np.sum(np.clip(-delta, 0.0, None)) * area)
    total_cells = int(len(np.union1d(columns_a["keys"], columns_b["keys"])))
    only_a = int(len(columns_a["keys"]) - both)
    only_b = int(len(columns_b["keys"]) - both)
    overlap = both / total_cells if total_cells else 0.0
    declared = _quality(quality)
    spacing, spacing_basis = _spacing_of(np.vstack((first, second)),
                                         declared.get("point_spacing_m"))
    scatter = 0.0
    thin = 0
    if keep.size:
        variance = np.concatenate((columns_a["variance"][ia][keep],
                                   columns_b["variance"][ib][keep]))
        counts = np.concatenate((columns_a["counts"][ia][keep],
                                 columns_b["counts"][ib][keep]))
        scatter = float(np.sqrt(np.mean(variance / counts)))
        if scatter <= 0.0:
            scatter = spacing / math.sqrt(12.0)
        thin = int(np.count_nonzero(np.min(np.stack((
            columns_a["counts"][ia][keep], columns_b["counts"][ib][keep])), axis=0)
            == 1))
    notes = []
    if keep.size and thin:
        notes.append(f"{thin} of {int(keep.size)} compared columns are supported by a "
                     f"single point on one side: there a surface height is one sample, "
                     f"not a measured surface, so raise min_points_per_cell if the "
                     f"volume needs to stand on its own")
    budget = _budget(declared, spacing, kind="volume", n_points=2, geometry_m=scatter,
                     cell_area_m2=area, n_cells=max(int(keep.size), 1))
    base = dict(cell_size_m=cell, cell_area_m2=area, cut_m3=cut, fill_m3=fill,
                change_m3=cut + fill,
                height_proxy="median z per grid column", cells_both_count=both,
                cells_used_count=int(keep.size), cells_a_only_count=only_a,
                cells_b_only_count=only_b, cells_total_count=total_cells,
                thin_column_count=thin, notes=notes,
                overlap_fraction=float(overlap), mean_delta_m=None, max_delta_m=None,
                column_scatter_m=scatter, min_overlap_fraction=threshold,
                min_points_per_cell=per_cell, spacing_basis=spacing_basis,
                geometry_enu=_footprint(np.column_stack(
                    (np.vstack((first[:, :2], second[:, :2])),
                     np.zeros(len(first) + len(second))))))
    if delta.size:
        base.update(mean_delta_m=float(np.mean(delta)),
                    max_delta_m=float(np.max(np.abs(delta))))
    if not both:
        return _record("volume_between", "m3", None, uncertainty=budget, valid=False,
                       reason="no grid cell holds points from both clouds: there is no "
                              "shared surface to difference, so the volume is "
                              "undefined rather than zero", **base)
    if overlap < threshold:
        return _record("volume_between", "m3", cut - fill, uncertainty=budget,
                       valid=False,
                       reason=f"only {both} of {total_cells} grid cells hold both "
                              f"surfaces ({overlap * 100:.0f}% overlap, min "
                              f"{threshold * 100:.0f}%): the rest of the footprint was "
                              f"measured by one cloud only, so this volume describes a "
                              f"surface that was never measured", **base)
    if not keep.size:
        return _record("volume_between", "m3", None, uncertainty=budget, valid=False,
                       reason=f"every shared column holds fewer than {per_cell} "
                              f"points: no column has a surface to compare", **base)
    return _record("volume_between", "m3", cut - fill, uncertainty=budget, valid=True,
                   reason=None, **base)


# --------------------------------------------------------------------------- #
# Slope and aspect.                                                           #
# --------------------------------------------------------------------------- #
def _compass(bearing):
    names = ("north", "north-east", "east", "south-east", "south", "south-west",
             "west", "north-west")
    return names[int(((bearing + 22.5) % 360.0) // 45.0)]


def _no_aspect(reason, basis="compass bearing of steepest descent"):
    return dict(kind="aspect", unit="deg", value=None, compass=None, valid=False,
                reason=reason, basis=basis)


def slope_aspect(points, *, radius_m, center=None, min_neighbours=6, flat_deg=0.05,
                 quality=None):
    """Local slope and downhill compass bearing from a plane fit inside a radius.

    ``points`` is the cloud; ``center`` selects the neighbourhood (its centroid when
    omitted) and ``radius_m`` bounds it, so this is a local measurement that says
    where it was taken. ``aspect`` is the direction of steepest descent as a compass
    bearing from north through east, carried as its own record: a flat surface has a
    perfectly good slope of 0.000 deg and a meaningless aspect, so the aspect is
    flagged rather than reporting a random bearing.

    Too few neighbours, or neighbours that cannot define a plane, returns
    ``insufficient neighbours`` / ``plane cannot be fitted`` with value None - never
    a slope of 0.0, which is exactly what a flat roof would look like.
    """
    cloud = _cloud(points)
    radius = _number(radius_m, "radius_m", positive=True)
    minimum = _count(min_neighbours, "min_neighbours")
    flat = _number(flat_deg, "flat_deg", non_negative=True)
    declared = _quality(quality)
    centre = cloud.mean(axis=0) if center is None else _vector(center, "center")
    rows = _Grid(cloud, radius).near(centre, radius)
    spacing, spacing_basis = _spacing_of(cloud, declared.get("point_spacing_m"))
    budget = _budget(declared, spacing, kind="slope", n_points=3,
                     n_samples=max(int(rows.size), 1), reference_m=radius)
    base = dict(centre_enu=centre.tolist(), radius_m=radius,
                neighbour_count=int(rows.size), min_neighbours=minimum,
                flat_deg=flat, spacing_basis=spacing_basis, geometry_enu=_point(centre))
    where = f"({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f})"
    if rows.size < minimum:
        reason = (f"insufficient neighbours: {int(rows.size)} points within "
                  f"{radius:.3f} m of {where} (need {minimum})")
        return _record("slope_aspect", "deg", None, uncertainty=budget, valid=False,
                       reason=reason, aspect=_no_aspect(reason), normal=None,
                       slope_deg=None, residual_rms_m=None, planarity=None,
                       gradient_en=None, **base)
    fit = _plane_from_points(cloud[rows])
    if fit is None:
        reason = (f"the {int(rows.size)} points within {radius:.3f} m of {where} are "
                  f"collinear: a plane cannot be fitted, so slope and aspect are "
                  f"undefined")
        return _record("slope_aspect", "deg", None, uncertainty=budget, valid=False,
                       reason=reason, aspect=_no_aspect(reason), normal=None,
                       slope_deg=None, residual_rms_m=None, planarity=None,
                       gradient_en=None, **base)
    normal = fit["normal"]
    vertical = abs(float(normal[2])) < 1e-9
    slope = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    bearing = (math.degrees(math.atan2(float(normal[0]), float(normal[1]))) + 360.0) \
        % 360.0
    if vertical:
        aspect = _no_aspect(f"the local surface is vertical (slope {slope:.3f} deg): "
                            f"it falls straight down, so no downhill bearing exists",
                            basis="compass bearing of steepest descent, 0=north "
                                  "through 90=east")
    elif slope <= flat:
        aspect = _no_aspect(f"the surface is flat (slope {slope:.3f} deg at or below "
                            f"{flat:.3f} deg): the downhill direction is undefined",
                            basis="compass bearing of steepest descent, 0=north "
                                  "through 90=east")
    else:
        aspect = dict(kind="aspect", unit="deg", value=bearing, compass=_compass(bearing),
                      valid=True, reason=None,
                      basis="compass bearing of steepest descent, 0=north through "
                            "90=east")
    return _record("slope_aspect", "deg", slope, uncertainty=budget, valid=True,
                   reason=None, slope_deg=slope, aspect=aspect,
                   aspect_deg=aspect["value"], normal=normal.tolist(),
                   residual_rms_m=fit["rms"],
                   planarity=fit["planarity"], vertical=bool(vertical),
                   gradient_en=None if vertical else
                   [-float(normal[0] / normal[2]), -float(normal[1] / normal[2])],
                   **base)


# --------------------------------------------------------------------------- #
# Geodetic conversion: the ENU inverse of survey_georef's forward transform.  #
# --------------------------------------------------------------------------- #
def frame_for_origin(latitude_deg, longitude_deg, altitude_m):
    """The coordinate frame contract this module speaks, from an origin."""
    result = dict(_FRAME, origin=dict(latitude_deg=_number(latitude_deg, "latitude_deg"),
                                      longitude_deg=_number(longitude_deg,
                                                            "longitude_deg"),
                                      altitude_m=_number(altitude_m, "altitude_m")))
    _origin_of(result)
    return result


def _origin_of(frame):
    if not isinstance(frame, dict):
        raise ValueError("frame must be the ENU coordinate_frame dict")
    for key, expected in _FRAME.items():
        if frame.get(key) != expected:
            raise ValueError(f"frame {key} must be {expected!r} to measure in "
                             f"georeferenced metres")
    origin = frame.get("origin")
    if not isinstance(origin, dict):
        raise ValueError("frame must carry an origin with latitude/longitude/altitude")
    lat = _number(origin.get("latitude_deg"), "latitude_deg")
    lon = _number(origin.get("longitude_deg"), "longitude_deg")
    alt = _number(origin.get("altitude_m"), "altitude_m")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError("frame origin is outside geodetic ranges")
    return lat, lon, alt


def _ecef(lat_rad, lon_rad, alt):
    radius = _A / np.sqrt(1.0 - _E2 * np.sin(lat_rad) ** 2)
    return np.column_stack(((radius + alt) * np.cos(lat_rad) * np.cos(lon_rad),
                            (radius + alt) * np.cos(lat_rad) * np.sin(lon_rad),
                            (radius * (1.0 - _E2) + alt) * np.sin(lat_rad)))


def _basis(lat_rad, lon_rad):
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
    return np.array([[-slon, clon, 0.0], [-slat * clon, -slat * slon, clat],
                     [clat * clon, clat * slon, slat]], dtype=np.float64)


def _geodetic_rows(value, name):
    """Accept mappings, rows or a single triple; return ((N, 3), was_single)."""
    if isinstance(value, dict) or hasattr(value, "get"):
        value = [value]
    if isinstance(value, (bytes, str)):
        raise ValueError(f"{name} must be geodetic rows, not text")
    if isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be (latitude_deg, longitude_deg, altitude_m) "
                         f"rows")
    if len(value) and isinstance(value[0], dict):
        missing = [key for key in ("latitude_deg", "longitude_deg", "altitude_m")
                   if any(key not in row for row in value)]
        if missing:
            raise ValueError(f"{name} rows lack {missing}")
        raw = np.array([[row["latitude_deg"], row["longitude_deg"], row["altitude_m"]]
                        for row in value], dtype=np.float64)
        single = len(raw) == 1
    else:
        raw = np.asarray(value, dtype=np.float64)
        single = raw.ndim == 1
    result = _rows(raw, name)
    if result.ndim == 1:
        result = result.reshape(1, 3)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must be (latitude_deg, longitude_deg, altitude_m) "
                         f"rows")
    if np.any(np.abs(result[:, 0]) > 90.0) or np.any(np.abs(result[:, 1]) > 180.0):
        raise ValueError(f"{name} is outside geodetic ranges")
    return result, single


def _enu_rows(value, name):
    raw = np.asarray(value, dtype=np.float64)
    single = raw.ndim == 1
    result = _rows(raw, name)
    if result.ndim == 1:
        result = result.reshape(1, 3)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must be (east, north, up) rows in metres")
    return result, single


def geodetic_to_enu(value, frame):
    """Latitude/longitude/ellipsoidal height to ENU metres, as survey_georef does.

    The forward direction, re-implemented from the same WGS84 constants and the same
    local basis so both halves of this module round-trip and agree with the
    georeferencer to well under a millimetre.
    """
    lat0, lon0, alt0 = _origin_of(frame)
    rows, single = _geodetic_rows(value, "geodetic")
    basis = _basis(math.radians(lat0), math.radians(lon0))
    origin = _ecef(np.array([math.radians(lat0)]), np.array([math.radians(lon0)]),
                   np.array([alt0]))[0]
    positions = _ecef(np.radians(rows[:, 0]), np.radians(rows[:, 1]), rows[:, 2])
    result = np.array((positions - origin) @ basis.T, dtype=np.float64)
    return result[0] if single else result


def enu_to_geodetic(value, frame):
    """Inverse of ``geodetic_to_enu``: ENU metres back to latitude/longitude.

    Rotates through the same local ECEF basis the forward direction uses, then solves
    the geodetic latitude by fixed-point iteration on the WGS84 ellipsoid (six
    passes; sub-micrometre over a kilometre extent). Heights stay ellipsoidal: no
    geoid model is applied here, so this is not mean-sea-level height.
    """
    lat0, lon0, alt0 = _origin_of(frame)
    rows, single = _enu_rows(value, "ENU")
    basis = _basis(math.radians(lat0), math.radians(lon0))
    origin = _ecef(np.array([math.radians(lat0)]), np.array([math.radians(lon0)]),
                   np.array([alt0]))[0]
    x, y, z = (origin + rows @ basis).T
    p = np.hypot(x, y)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(z, np.where(p > 0.0, p * (1.0 - _E2), 1.0))
    for _ in range(6):
        sine = np.sin(latitude)
        radius = _A / np.sqrt(1.0 - _E2 * sine ** 2)
        latitude = np.arctan2(z + _E2 * radius * sine, p)
    sine, cosine = np.sin(latitude), np.cos(latitude)
    radius = _A / np.sqrt(1.0 - _E2 * sine ** 2)
    altitude = p * cosine + (z + _E2 * radius * sine) * sine - radius
    result = np.column_stack((np.degrees(latitude), np.degrees(longitude), altitude))
    if np.any(np.abs(result[:, 0]) > 90.0 + 1e-9):
        raise ValueError("inverse geodetic solution left the valid latitude range")
    return result[0] if single else result


def _to_wgs84(geometry, frame):
    if geometry is None:
        return None
    kind = geometry["type"]
    if kind == "Point":
        coordinates = enu_to_geodetic(geometry["coordinates"], frame).tolist()
    elif kind == "LineString":
        coordinates = enu_to_geodetic(np.asarray(geometry["coordinates"],
                                                 dtype=np.float64).reshape(-1, 3),
                                      frame).tolist()
    elif kind == "Polygon":
        coordinates = []
        for ring in geometry["coordinates"]:
            positions = enu_to_geodetic(np.asarray(ring, dtype=np.float64).reshape(-1, 3),
                                        frame).tolist()
            if positions and positions[0] != positions[-1]:
                positions.append(list(positions[0]))
            coordinates.append(positions)
    else:
        raise ValueError(f"unsupported geometry type {kind}")
    return dict(type=kind, coordinates=coordinates)


# --------------------------------------------------------------------------- #
# Records out: report, GeoJSON, CSV.                                          #
# --------------------------------------------------------------------------- #
_CORE = ("kind", "unit", "value", "uncertainty", "valid", "reason", "geometry_enu")


def _is_record(value):
    return (isinstance(value, dict) and "kind" in value and "unit" in value
            and "valid" in value)


def _named(measurements):
    """Accept a mapping of name to record or a list; the caller's order is kept.

    Only records built by this module are accepted, and every measurement record
    carries its own uncertainty because ``_record`` requires one. Supporting
    records (``local_density``, snap and aspect sub-records) carry a unit and a
    validity flag but share the budget of the measurement that embedded them, so
    they may also be reported on their own.
    """
    if isinstance(measurements, dict):
        items = list(measurements.items())
    elif isinstance(measurements, (list, tuple)):
        seen, items = {}, []
        for value in measurements:
            if not _is_record(value):
                raise ValueError("measurements must be records built by this module")
            kind = str(value["kind"])
            seen[kind] = seen.get(kind, 0) + 1
            items.append((f"{kind}_{seen[kind]}", value))
    else:
        raise ValueError("measurements must be a dict of name to record, or a list")
    result = {}
    for name, record in items:
        if not _is_record(record):
            raise ValueError(f"{name} is not a measurement record")
        result[str(name)] = record
    if not result:
        raise ValueError("measurements must not be empty")
    return result


def _entry(record):
    """One report entry: value, unit, uncertainty, valid, reason, sub-records.

    Supporting quantities (endpoint density, snap distance, aspect) carry a unit
    and a validity flag but no budget of their own - they share the one on the
    measurement that embedded them, which is said in ``uncertainty_note``. A
    measurement built by ``_record`` always has its own, because it is a required
    argument there.
    """
    components, detail = {}, {}
    for key, value in record.items():
        if key in _CORE:
            continue
        if _is_record(value):
            components[key] = _entry(value)
        elif isinstance(value, dict) and value and all(_is_record(item)
                                                       for item in value.values()):
            components[key] = {str(sub): _entry(item) for sub, item in value.items()}
        else:
            detail[key] = value
    budget = record.get("uncertainty")
    if budget is not None and not isinstance(budget, dict):
        raise ValueError("an uncertainty record must be built by uncertainty()")
    if budget is not None and "plus_minus" not in budget:
        raise ValueError("an uncertainty record needs a plus_minus")
    return dict(kind=str(record["kind"]),
                value=None if record["value"] is None else _plain(record["value"]),
                unit=None if record["unit"] is None else str(record["unit"]),
                uncertainty=None if budget is None else dict(
                    plus_minus=_plain(budget["plus_minus"]),
                    unit=str(budget["unit"]),
                    plus_minus_m=_plain(budget["plus_minus_m"]),
                    confidence=str(budget.get("confidence", "1-sigma")),
                    valid=bool(budget["valid"]),
                    reason=_text(budget.get("reason")),
                    components_m=_plain(budget.get("components_m", {})),
                    assumptions=_plain(budget.get("assumptions", []))),
                uncertainty_note=None if budget is not None else
                "shares the parent measurement's budget",
                valid=bool(record["valid"]), reason=_text(record.get("reason")),
                components=components, detail=_plain(detail))


def report(measurements):
    """One entry per measurement: value, unit, uncertainty, valid, reason.

    Entries keep the caller's order, and every nested number that means something
    (endpoint density, snap distances, aspect) is carried as its own sub-entry with
    its own unit and validity flag, so nothing in the report can be read as a
    measured fact without its caveat. ``reason`` is None exactly when ``valid`` is
    True, and a measurement with no value reports None rather than zero. A
    supporting record (from ``local_density``, say) may also be reported on its
    own: it then says its budget is shared with the measurement that embeds it.
    """
    return {name: _entry(record) for name, record in _named(measurements).items()}


def _frame_summary(frame):
    lat, lon, alt = _origin_of(frame)
    return dict(type="ENU", units="m", geodetic_crs="EPSG:4979",
                altitude_datum="ellipsoidal",
                origin=dict(latitude_deg=lat, longitude_deg=lon, altitude_m=alt),
                position_order="longitude,latitude,ellipsoidal_height",
                note="heights are ellipsoidal; no geoid or orthometric correction "
                     "is applied")


def to_geojson(measurements, frame):
    """GeoJSON FeatureCollection with ENU results converted to latitude/longitude.

    Positions are [longitude, latitude, ellipsoidal height] per RFC 7946 and the
    frame (origin, EPSG:4979, ellipsoidal datum) is echoed in ``coordinate_frame``,
    because a height is meaningless without its datum. Each feature's properties are
    exactly that measurement's ``report`` entry, so a map click shows the same
    uncertainty and validity flag as the table. A measurement taken where there was
    no geometry keeps its properties and gets a null geometry.
    """
    _origin_of(frame)
    records = _named(measurements)
    features = []
    for name, record in records.items():
        entry = _entry(record)
        geometry = _to_wgs84(record.get("geometry_enu"), frame)
        properties = dict(entry, name=name, geometry_available=geometry is not None)
        features.append(dict(type="Feature", id=f"measure:{name}",
                             properties=properties, geometry=geometry))
    return dict(type="FeatureCollection", name="survey measurements",
                generator="scripts/survey_measure.py",
                coordinate_frame=_frame_summary(frame), features=features)


def to_csv(measurements, path=None, frame=None):
    """Flat CSV of the report: one row per measurement, flags and reasons intact.

    ``value`` always travels with ``unit``, ``plus_minus`` with its own unit and
    validity flag, and ``reason`` is filled exactly when ``valid`` is false. When a
    georeferenced ``frame`` is supplied the first measured position is converted to
    latitude/longitude/ellipsoidal height; the remaining per-measurement fields go
    into ``extra_json`` rather than becoming a column whose meaning changes between
    rows. Returns the text, and writes it when given a path.
    """
    records = _named(measurements)
    fields = ("name", "kind", "value", "unit", "plus_minus", "plus_minus_unit",
              "plus_minus_valid", "valid", "reason", "latitude_deg", "longitude_deg",
              "altitude_m", "extra_json")
    if frame is not None:
        _origin_of(frame)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for name, record in records.items():
        entry = _entry(record)
        budget = entry["uncertainty"] or {}
        position = None
        if frame is not None:
            anchor = entry["detail"].get("target_enu")
            if anchor is None:
                geometry = record.get("geometry_enu")
                if geometry:
                    anchor = np.asarray(geometry["coordinates"],
                                        dtype=np.float64).reshape(-1, 3)[0]
            if anchor is not None:
                position = enu_to_geodetic(np.asarray(anchor, dtype=np.float64), frame)
        writer.writerow(dict(
            name=name, kind=entry["kind"],
            value="" if entry["value"] is None else f"{entry['value']:.6f}",
            unit="" if entry["unit"] is None else entry["unit"],
            plus_minus="" if budget.get("plus_minus") is None
            else f"{budget['plus_minus']:.6f}",
            plus_minus_unit=budget.get("unit", ""),
            plus_minus_valid="" if not budget else
            ("true" if budget["valid"] else "false"),
            valid="true" if entry["valid"] else "false",
            reason=entry["reason"] or "",
            latitude_deg="" if position is None else f"{position[0]:.8f}",
            longitude_deg="" if position is None else f"{position[1]:.8f}",
            altitude_m="" if position is None else f"{position[2]:.4f}",
            extra_json=json.dumps(dict(detail=entry["detail"],
                                       components=entry["components"]),
                                  allow_nan=False, sort_keys=True,
                                  separators=(",", ":"))))
    text = stream.getvalue()
    if path is not None:
        Path(path).write_text(text, encoding="utf-8", newline="")
    return text
