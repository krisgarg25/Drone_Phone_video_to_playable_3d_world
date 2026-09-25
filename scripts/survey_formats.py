"""Writers for the SIH26158 output format list, in stdlib plus NumPy only.

Covered: ASCII Wavefront OBJ (v/vt/f), glTF 2.0 (.gltf plus a .bin buffer, or an embedded
``data:`` URI), a deliberately minimal ASCII FBX 6.1 geometry node set, LAS 1.4, a
single-band float32 GeoTIFF, and dependency-free XYZ/CSV cloud fallbacks.

READ THIS BEFORE TRUSTING ANY FILE. No reference geospatial library is importable in this
environment - laspy, rasterio, pyproj, GDAL/osgeo and PDAL are all absent - so the only
check that exists here is a round trip through this module's own reader. **A round trip
proves self-consistency, not spec compliance:** a writer and its reader can agree on a
wrong layout and nothing in this file would notice. No output has been opened in
CloudCompare, MeshLab, Blender, QGIS, PDAL or GDAL, and no function claims otherwise.
Every writer returns ``{"verified": ..., "externally_validated": False, "unverified":
[...]}``, and :func:`verification_summary` reports the same for the module along with the
library that would be needed to check each format properly.

What real verification would take, in priority order:
  * ``laspy`` (or PDAL) - ``laspy.read`` compares header fields, scales, offsets, counts
    and every point against ``write_las`` output, and settles the point format 3 field
    order that is *assumed* here (see ``LAS_POINT_FORMATS``).
  * ``rasterio`` (or GDAL) - ``rasterio.open`` confirms transform, CRS and values, and
    resolves the GeoKey directory against the EPSG registry.
  * the Khronos ``glTF-Validator`` - conformance checking beyond the accessor, bufferView
    and buffer invariants that ``_validate_gltf`` applies by hand.
  * the Autodesk FBX SDK or Blender's importer - ``write_fbx`` emits a small documented
    subset of ASCII FBX 6.1 and nothing outside this module has ever read it.
Layouts live in named module constants and every claim a writer cannot support is refused
rather than approximated, so a later run with a real library can diff field by field
instead of rediscovering which ones were guessed.

XYZ and CSV are the opposite case: documented plain text, verifiable by inspection with no
library at all, and their summaries say so.
"""
from __future__ import annotations

import base64
import csv
import datetime
import importlib.util
import json
import math
import re
import struct
from pathlib import Path

import numpy as np

__all__ = ["write_obj", "read_obj", "write_gltf", "read_gltf", "write_fbx", "read_fbx",
           "write_las", "read_las", "write_geotiff", "read_geotiff", "write_xyz",
           "read_xyz", "write_csv", "read_csv", "verification_summary",
           "LAS_POINT_FORMATS", "ROUND_TRIP", "PLAIN_TEXT", "REFERENCE_LIBRARIES"]

ROUND_TRIP = "round-trip only"
PLAIN_TEXT = "plain text, externally verifiable by inspection"
REFERENCE_LIBRARIES = ("laspy", "rasterio", "pyproj", "osgeo")


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def verification_summary() -> dict:
    """What has and has not been checked about the files these writers produce."""
    return dict(
        verified=ROUND_TRIP,
        externally_validated=False,
        reference_libraries_available={name: _available(name) for name in
                                       REFERENCE_LIBRARIES},
        note=("a round trip through this module's own reader proves self-consistency only; "
              "with no reference library importable here, no output has been checked "
              "against a specification, an SDK or any viewer application"),
        revalidate_with=dict(las="laspy.read, or `pdal info`",
                             geotiff="rasterio.open, or gdalinfo",
                             gltf="the Khronos glTF-Validator",
                             fbx="the Autodesk FBX SDK, or Blender",
                             obj="any independent OBJ importer",
                             xyz="inspection", csv="inspection"),
        formats=dict(
            obj=dict(verified=ROUND_TRIP, round_trip_only=True,
                     validator="read_obj in this module",
                     claims=["1-based v/vt/f indices", "triangles only",
                             "per-vertex or per-corner UVs"],
                     unverified=["no independent OBJ parser has read these bytes",
                                 "the .mtl file named by mtllib is never written"]),
            gltf=dict(verified=ROUND_TRIP, round_trip_only=True,
                      validator="read_gltf in this module",
                      claims=["asset.version '2.0'", "POSITION as VEC3 float32 with min/max",
                              "4-byte aligned bufferViews", "buffer URI resolved from disk"],
                      unverified=["no glTF-Validator run"]),
            fbx=dict(verified=ROUND_TRIP, round_trip_only=True,
                     validator="read_fbx in this module",
                     claims=["ASCII FBX 6.1 header, Definitions, Objects, Connections",
                             "polygon fields are emitted only when there are polygons"],
                     unverified=["never opened by an FBX SDK or importer",
                                 "no materials, textures, normals, UVs, takes or axis setup",
                                 "the point-cloud form (Vertices with no PolygonVertexIndex "
                                 "at all) is as untested as the empty index it replaces"]),
            las=dict(verified=ROUND_TRIP, round_trip_only=True,
                     validator="read_las in this module",
                     validated_by="write_las and read_las, both in this module",
                     claims=["239-byte header block declaring version 1.4",
                             "stored integer = round((value - offset) / scale)",
                             "a coordinate that cannot fit int32 is refused, never wrapped",
                             "the 32-bit and 64-bit point counts hold the same value",
                             "54-byte VLR headers with GeoKey and WKT VLRs"],
                     unverified=["the header FIELD OFFSETS are this module's own layout: the "
                                 "published LAS 1.4 header is longer than 239 bytes and places "
                                 "the point format and the scale doubles elsewhere, so a real "
                                 "LAS reader is expected to reject or misread these bytes "
                                 "(needs the ASPRS 1.4 table or laspy)",
                                 "the per-return point counts of a 1.4 header are not written "
                                 "at all",
                                 "the point format 3 field order after the base record is an "
                                 "assumption (needs laspy or the ASPRS 1.4 table)",
                                 "point formats 6-10 with the wider 1.4 bit fields are not "
                                 "implemented",
                                 "no laspy or PDAL comparison is available here"]),
            geotiff=dict(verified=ROUND_TRIP, round_trip_only=True,
                         validator="read_geotiff in this module",
                         claims=["little-endian classic TIFF, one uncompressed strip",
                                 "SampleFormat 3 (IEEE float32) with 32 bits per sample",
                                 "the 13 tags read_geotiff requires, by number: 256, 257, "
                                 "258, 259, 262, 273, 277, 278, 279, 339, 33550, 33922, "
                                 "34735",
                                 "ModelPixelScaleTag plus ModelTiepointTag, north-up only",
                                 "Software(305) and GeoAsciiParams(34737) always, and "
                                 "GDAL_NODATA(42112) when a nodata value is given"],
                         unverified=["no GDAL or rasterio comparison is available here",
                                     "the CRS was carried, never resolved against the EPSG "
                                     "registry",
                                     "the GeoKey directory holds 3 keys only: no "
                                     "GeographicType pairing, no projected-CRS parameters "
                                     "(3101/3102/...) and no ModelTransformationTag, so a "
                                     "user-defined CRS stays under-specified",
                                     "the TIFF baseline also expects the resolution tags "
                                     "(282, 283, 296), which are not written here",
                                     "single strip, north-up, one band, float32 only"]),
            xyz=dict(verified=PLAIN_TEXT, round_trip_only=False, unverified=[]),
            csv=dict(verified=PLAIN_TEXT, round_trip_only=False, unverified=[])))


def _summary(path: Path, *, verified: str, unverified=(), **facts) -> dict:
    result = dict(path=str(path), verified=verified, externally_validated=False,
                  unverified=list(unverified))
    result.update(facts)
    return result


def _reject(path: Path, message: str):
    """Delete an output that failed its own re-read, then raise: never leave a bad file."""
    path.unlink(missing_ok=True)
    raise ValueError(message)


# --------------------------------------------------------------------------- validation
def _finite_array(value, name: str, columns: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must hold finite numbers: None, NaN and non-numeric "
                         "values are never written") from None
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite: NaN, inf and None are never written")
    if columns is not None and (raw.ndim != 2 or raw.shape[1] != columns):
        raise ValueError(f"{name} must have shape (n, {columns}), got {raw.shape}")
    return raw


def _columns(points, required: tuple[str, ...], optional: tuple[str, ...] = ()):
    """Accept an (n, k) array, a dict of columns or a structured array; return (dict, n)."""
    def size(values, label):
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{label} must be a 1-D column, got shape {array.shape}")
        return len(array)

    allowed = set(required) | set(optional)
    if isinstance(points, dict):
        missing = [name for name in required if name not in points]
        if missing:
            raise ValueError(f"points is missing the required columns {missing}")
        unknown = sorted(set(points) - allowed)
        if unknown:
            raise ValueError(f"column {unknown[0]!r} cannot be written in this layout; "
                             f"supported columns are {sorted(allowed)}")
        data = {name: points[name] for name in points}
    else:
        array = np.asarray(points)
        if array.dtype.names:
            missing = [name for name in required if name not in array.dtype.names]
            if missing:
                raise ValueError(f"points is missing the required columns {missing}")
            data = {name: array[name] for name in array.dtype.names if name in allowed}
        else:
            matrix = _finite_array(points, "points", len(required))
            data = {name: matrix[:, index] for index, name in enumerate(required)}
    count = size(data[required[0]], f"{required[0]!r} column")
    for name, values in data.items():
        if size(values, f"{name!r} column") != count:
            raise ValueError(f"column {name} is not the same length as the first column "
                             f"({count} rows): every column must describe the same points")
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"column {name} must be finite: NaN, inf and None are never "
                             "written")
    return data, count


def _index_array(triangles, count: int, name: str = "triangles") -> np.ndarray:
    faces = np.asarray(triangles)
    if faces.size == 0:
        raise ValueError(f"{name} is empty: nothing to write")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {faces.shape}")
    if faces.dtype.kind not in "iu":
        raise ValueError(f"{name} must hold integer indices, got dtype {faces.dtype}")
    faces = faces.astype(np.int64)
    if faces.min() < 0:
        raise ValueError(f"{name} index {int(faces.min())} is negative")
    if faces.max() >= count:
        raise ValueError(f"{name} index out of range: {int(faces.max())} for {count} "
                         "vertices")
    return faces


# ---------------------------------------------------------------------------- plain text
def write_xyz(points, out) -> dict:
    """Write an ASCII ``x y z`` cloud: space separated, one point per line, no header.

    The fallback that needs no format claim at all - the layout is three decimal numbers
    per line, verifiable by opening the file. Values use ``repr``, the shortest text that
    reads back as the same float64, so inspection shows the measurement and not a rounded
    copy of it.
    """
    data, count = _columns(points, ("x", "y", "z"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii", newline="\n") as stream:
        for index in range(count):
            stream.write(" ".join(repr(float(data[name][index]))
                                  for name in ("x", "y", "z")) + "\n")
    try:
        if len(read_xyz(path=out)["points"]) != count:
            raise ValueError("row count changed on re-read")
    except ValueError as exc:
        _reject(out, f"{out.name}: refused XYZ output that this module's reader rejected: "
                     f"{exc}")
    return _summary(out, verified=PLAIN_TEXT, n_points=count,
                    layout="ascii x y z, one point per line", unverified=[])


def read_xyz(path) -> dict:
    """Read an ASCII ``x y z`` cloud, naming the line of anything unusable."""
    path = Path(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path.name} line {number}: expected 3 numbers, found "
                             f"{len(parts)}")
        rows.append(_finite_array(parts, f"{path.name} line {number}"))
    return dict(points=np.array(rows, dtype=float).reshape(-1, 3))


def _text_value(value, column) -> str:
    """Keep an integral column integral in text output; write exact round-trip floats."""
    if np.asarray(column).dtype.kind in "iu":
        return str(int(value))
    return repr(float(value))


def write_csv(points, out) -> dict:
    """Write a point cloud as CSV with a header row naming every column.

    Externally verifiable in the same way as :func:`write_xyz`. Extra columns the caller
    supplies (intensity, classification, a per-point error) are carried as named columns,
    which is where they belong: a binary format would have to invent a field the data may
    not really have. Integral columns stay integral and floats use ``repr``, so the text
    reads back as the value that was measured.
    """
    extras = ("intensity", "classification", "ring", "quality", "gps_time",
              "red", "green", "blue")
    data, count = _columns(points, ("x", "y", "z"), optional=extras)
    names = [name for name in ("x", "y", "z") + extras if name in data]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names)
        for index in range(count):
            writer.writerow([_text_value(data[name][index], data[name]) for name in names])
    try:
        back = read_csv(out)
        if sorted(back) != sorted(names) or len(next(iter(back.values()))) != count:
            raise ValueError("column set or row count changed on re-read")
    except ValueError as exc:
        _reject(out, f"{out.name}: refused CSV output that this module's reader rejected: "
                     f"{exc}")
    return _summary(out, verified=PLAIN_TEXT, n_points=count, columns=names, unverified=[])


def read_csv(path) -> dict:
    """Read a CSV cloud written by :func:`write_csv` into a dict of named columns."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    if not rows:
        raise ValueError(f"{path.name}: file is empty")
    header, body = rows[0], rows[1:]
    out = {}
    for index, name in enumerate(header):
        values = []
        for position, row in enumerate(body, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path.name} line {position}: {len(row)} fields for a "
                                 f"{len(header)}-column header")
            values.append(float(_finite_array([row[index]],
                                              f"{path.name} line {position}")[0]))
        out[name] = np.array(values, dtype=float)
    return out


# ------------------------------------------------------------------------------------ OBJ
def write_obj(triangles, vertices, out, texture_coords=None, material=None) -> dict:
    """Write an ASCII Wavefront OBJ: ``v`` positions, optional ``vt`` UVs, ``f`` faces.

    ``vertices`` is (n, 3) and ``triangles`` is (m, 3) of **zero-based** vertex indices,
    written 1-based as the format requires. ``texture_coords`` may be (n, 2), one UV per
    vertex, where each corner is written ``v/vt`` reusing the vertex index, or (3m, 2), one
    UV per face corner in row-major order, written ``v/vt`` with the corner index. Any
    other count is refused: a UV count matching neither convention is how a texture ends up
    on the wrong face.

    ``material`` writes ``mtllib`` and ``usemtl`` lines naming it; the .mtl file itself is
    not produced. No other OBJ construct is emitted - no normals, groups, smoothing groups,
    lines or curves - and a face whose corners repeat is refused rather than written,
    because a degenerate triangle shades and measures unpredictably.
    """
    verts = _finite_array(vertices, "vertices", 3)
    faces = _index_array(triangles, len(verts))
    repeated = ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
                | (faces[:, 0] == faces[:, 2]))
    if repeated.any():
        raise ValueError(f"{int(repeated.sum())} triangle(s) repeat a corner vertex: a "
                         "degenerate face is not written")
    uvs = None
    per_corner = False
    if texture_coords is not None:
        uvs = _finite_array(texture_coords, "texture_coords", 2)
        if len(uvs) == len(verts):
            per_corner = False
        elif len(uvs) == 3 * len(faces):
            per_corner = True
        else:
            raise ValueError(f"texture_coords must be one UV per vertex ({len(verts)}) or "
                             f"one per face corner ({3 * len(faces)}), got {len(uvs)}")
        if uvs.min() < 0.0 or uvs.max() > 1.0:
            raise ValueError("texture_coords must lie in [0, 1]: no tiling convention is "
                             "recorded in this file, so an out of range UV would be a guess")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ASCII Wavefront OBJ written by scripts/survey_formats.write_obj",
             f"# {len(verts)} vertices, {len(faces)} triangles",
             f"o {out.stem}"]
    lines += ["v " + " ".join(repr(float(v)) for v in row) for row in verts]
    lines += ["vt " + " ".join(repr(float(v)) for v in row)
              for row in ([] if uvs is None else uvs)]
    if material:
        lines.append(f"mtllib {material}")
        lines.append(f"usemtl {material}")
    for order, row in enumerate(faces):
        corners = []
        for offset in range(3):
            vertex = int(row[offset]) + 1
            uv = (order * 3 + offset + 1) if per_corner else vertex
            corners.append(str(vertex) if uvs is None else f"{vertex}/{uv}")
        lines.append("f " + " ".join(corners))
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    try:
        back = read_obj(out)
        if len(back["triangles"]) != len(faces) or np.any(back["vertices"] != verts):
            raise ValueError("geometry changed on re-read")
    except ValueError as exc:
        _reject(out, f"{out.name}: refused OBJ that this module's reader rejected: {exc}")
    return _summary(out, verified=ROUND_TRIP, vertices=len(verts), triangles=len(faces),
                    texture_coords=0 if uvs is None else len(uvs),
                    uv_layout="per_corner" if per_corner else
                    ("per_vertex" if uvs is not None else None), material=material,
                    unverified=["no independent OBJ parser has read these bytes",
                                "the .mtl library named by mtllib is not written"])


def read_obj(path) -> dict:
    """Parse ``v``, ``vt`` and ``f`` out of an ASCII OBJ. Triangles only, 1-based indices."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("<"):
        raise ValueError(f"{path.name}: not an ASCII OBJ")
    vertices, uvs, faces, material, mtllib = [], [], [], None, None
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0].startswith(("#", "o", "g")):
            continue
        if parts[0] == "v":
            vertices.append(_obj_numbers(path, number, parts[1:4], "vertex"))
        elif parts[0] == "vt":
            uvs.append(_obj_numbers(path, number, parts[1:3], "texture coordinate"))
        elif parts[0] == "f":
            corners = []
            for token in parts[1:]:
                head = token.split("/")[0]
                if head.startswith("-"):
                    raise ValueError(f"{path.name} line {number}: negative face index "
                                     f"{head!r}; relative indexing is not read here")
                try:
                    corners.append(int(head))
                except ValueError:
                    raise ValueError(f"{path.name} line {number}: face index {head!r} is not "
                                     "a number") from None
            if len(corners) != 3:
                raise ValueError(f"{path.name} line {number}: {len(corners)} face corners; "
                                 "this reader takes triangles, not polygons")
            faces.append(corners)
        elif parts[0] == "usemtl":
            material = parts[1] if len(parts) > 1 else None
        elif parts[0] == "mtllib":
            mtllib = parts[1] if len(parts) > 1 else None
    if not vertices:
        raise ValueError(f"{path.name}: no vertices found")
    grid = (np.array(faces, dtype=np.int64).reshape(-1, 3) if faces
            else np.zeros((0, 3), np.int64))
    if len(grid) and (grid.min() < 1 or grid.max() > len(vertices)):
        raise ValueError(f"{path.name}: face index out of range for {len(vertices)} vertices")
    return dict(vertices=np.array(vertices, dtype=float),
                triangles=grid - 1 if len(grid) else grid,
                texture_coords=np.array(uvs, dtype=float) if uvs else np.zeros((0, 2)),
                material=material, mtllib=mtllib)


def _obj_numbers(path, number, tokens, what):
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        raise ValueError(f"{path.name} line {number}: {what} value is not a number") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path.name} line {number}: {what} value is not finite")
    return values


# -------------------------------------------------------------------------------- glTF 2
_GLTF_COMPONENTS = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_GLTF_ELEMENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_GLTF_CODES = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_GLTF_NUMPY = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
               5125: np.uint32, 5126: np.float32}
_POINTS, _TRIANGLES = 0, 4


def write_gltf(positions, colors_or_mesh=None, out=None, *, mode: str = "auto",
               embed: bool = False, name: str = "survey") -> dict:
    """Write a minimal glTF 2.0 scene: one mesh, one primitive, one binary buffer.

    ``out`` is the ``.gltf`` path. Its buffer is a sibling ``.bin`` named after the file,
    or with ``embed=True`` a ``data:application/octet-stream;base64,`` URI.

    ``colors_or_mesh`` is overloaded by design, so the rule is stated and enforced instead
    of guessed: a float array is one color per vertex - shape (n, 3) or (n, 4), values in
    the [0, 1] range glTF requires - giving a PRIMITIVE_POINTS accessor with ``COLOR_0``;
    an integer array is triangle indices - shape (m, 3) inside the vertex range - giving a
    PRIMITIVE_TRIANGLES accessor with an index accessor. An array can satisfy both, and then
    the call raises unless ``mode='points'`` or ``mode='triangles'`` says which is meant.
    Three-component colors are widened to RGBA because each accessor's element stride has
    to be a multiple of four.

    Structure is verified before the output is trusted: componentType against accessor type,
    4-byte aligned bufferView and accessor offsets, each accessor inside its bufferView and
    each bufferView inside the buffer, ``POSITION`` as VEC3 float32 with min and max, an
    unsigned index type whose count divides by three, matching attribute counts, and a
    buffer URI that resolves from disk. The document is written and then re-read through
    :func:`read_gltf`; on any complaint both files are deleted and the error raised. That is
    self-consistency, not a validator's verdict.
    """
    if mode not in ("auto", "points", "triangles"):
        raise ValueError("mode must be 'auto', 'points' or 'triangles'")
    verts = _finite_array(positions, "positions", 3)
    count = len(verts)
    colors = triangles = None
    if colors_or_mesh is not None:
        given = np.asarray(colors_or_mesh)
        if given.ndim != 2 or given.size == 0:
            raise ValueError(f"colors_or_mesh must be a non-empty 2-D array, got "
                             f"{given.shape}")
        numeric = given.dtype.kind == "f"
        color_shaped = given.shape[0] == count and given.shape[1] in (3, 4)
        integers = None if numeric else np.asarray(given, dtype=np.int64)
        index_shaped = (not numeric and given.shape[1] == 3 and integers.min() >= 0
                        and integers.max() < count)
        if mode == "points":
            colors = given
        elif mode == "triangles":
            triangles = _index_array(given, count)
        elif numeric:
            if not color_shaped:
                raise ValueError(f"float colors must have one row per vertex: got "
                                 f"{given.shape} for {count} positions")
            colors = given
        elif index_shaped and not color_shaped:
            triangles = _index_array(given, count)
        elif color_shaped and not index_shaped:
            colors = given
        elif color_shaped and index_shaped:
            raise ValueError("colors_or_mesh is ambiguous: it is both an in-range triangle "
                             "index set and the shape of a per-vertex color set. Pass "
                             "mode='points' or mode='triangles'.")
        else:
            raise ValueError("colors_or_mesh is neither colors (one row per vertex) nor "
                             f"in-range triangle indices: an integer array spans "
                             f"{int(integers.min())}..{int(integers.max())}, out of range "
                             f"for {count} positions")
        if colors is not None and np.asarray(colors).shape[0] != count:
            raise ValueError("colors must have exactly one row per vertex")
    # The geometry is checked before the output path: a caller who passed a bad array wants
    # to hear about the array, not about an argument they did not forget.
    if out is None:
        raise ValueError("write_gltf needs an output path")
    blocks, accessors, views = [], [], []
    cursor = 0

    def add(values, component_type, accessor_type, bounds=False):
        nonlocal cursor
        if component_type not in _GLTF_COMPONENTS:
            raise ValueError(f"unknown glTF componentType {component_type}")
        elements = _GLTF_ELEMENTS[accessor_type]
        if values.shape[1] != elements:
            raise ValueError(f"accessor type {accessor_type} needs {elements} columns, got "
                             f"{values.shape[1]}")
        payload = np.ascontiguousarray(values).astype(_GLTF_NUMPY[component_type])
        data = payload.tobytes()
        if cursor % 4:  # every bufferView.byteOffset has to sit on a 4-byte boundary
            pad = 4 - cursor % 4
            blocks.append(b"\x00" * pad)
            cursor += pad
        views.append(dict(buffer=0, byteOffset=cursor, byteLength=len(data)))
        blocks.append(data)
        accessor = dict(bufferView=len(views) - 1, byteOffset=0,
                        componentType=component_type, count=len(payload), type=accessor_type)
        if bounds:
            accessor["min"] = [float(v) for v in payload.min(axis=0)]
            accessor["max"] = [float(v) for v in payload.max(axis=0)]
        accessors.append(accessor)
        cursor += len(data)
        return len(accessors) - 1

    attributes = {"POSITION": add(verts, 5126, "VEC3", bounds=True)}
    if colors is not None:
        values = np.asarray(colors)
        if values.dtype.kind == "f":
            floats = np.asarray(values, dtype=float)
            if floats.min() < 0.0 or floats.max() > 1.0:
                raise ValueError("float colors are out of range: glTF colors must lie in "
                                 "[0, 1], and a 0..255 array belongs in an integer column")
            if floats.shape[1] == 3:
                floats = np.column_stack([floats, np.ones(len(floats))])
            attributes["COLOR_0"] = add(floats, 5126, "VEC4")
        else:
            exact = np.asarray(values, dtype=np.int64)
            if exact.min() < 0 or exact.max() > 255:
                raise ValueError("integer colors must lie in uint8 (0..255)")
            widened = exact.astype(np.uint8)
            if widened.shape[1] == 3:
                widened = np.column_stack([widened, np.full(len(widened), 255, np.uint8)])
            attributes["COLOR_0"] = add(widened, 5121, "VEC4")
    primitive = dict(attributes=attributes,
                     mode=_TRIANGLES if triangles is not None else _POINTS)
    if triangles is not None:
        component = 5123 if count <= 65535 else 5125
        primitive["indices"] = add(triangles.reshape(-1, 1), component, "SCALAR")
    buffer = b"".join(blocks)
    if len(buffer) % 4:
        buffer += b"\x00" * (4 - len(buffer) % 4)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer_name = out.stem + ".bin"
    buffer_path = out.with_name(buffer_name)
    uri = ("data:application/octet-stream;base64,"
           + base64.b64encode(buffer).decode("ascii") if embed else buffer_name)
    document = dict(asset=dict(version="2.0",
                               generator="scripts/survey_formats.write_gltf"),
                    scene=0, scenes=[dict(nodes=[0], name=name)],
                    nodes=[dict(mesh=0, name=name)],
                    meshes=[dict(name=name, primitives=[primitive])],
                    accessors=accessors, bufferViews=views,
                    buffers=[dict(uri=uri, byteLength=len(buffer))])
    document = json.loads(json.dumps(document, allow_nan=False))
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    if not embed:
        buffer_path.write_bytes(buffer)
    try:
        # Validated after writing, so a buffer uri is resolved against the bytes on disk.
        read_gltf(out)
    except ValueError as exc:
        out.unlink(missing_ok=True)
        buffer_path.unlink(missing_ok=True)
        raise ValueError(f"{out.name}: refused a glTF that this module's own reader "
                         f"rejected: {exc}") from None
    return _summary(out, verified=ROUND_TRIP, positions=count,
                    colors=0 if colors is None else count,
                    triangles=0 if triangles is None else len(triangles),
                    mode=primitive["mode"], buffer_bytes=len(buffer), embedded=embed,
                    unverified=["no glTF-Validator run: only the invariants listed in "
                                "write_gltf's docstring were applied"])


def _buffer_bytes(document: dict, index: int, directory: Path) -> bytes:
    buffer = document["buffers"][index]
    uri = buffer.get("uri")
    if uri is None:
        raise ValueError(f"buffer {index} has no uri (only a .glb may omit one, and this "
                         "module does not write .glb)")
    if uri.startswith("data:"):
        header, _, payload = uri.partition(",")
        if "base64" not in header:
            raise ValueError(f"buffer {index} data uri is not base64 encoded")
        return base64.b64decode(payload)
    if Path(uri).is_absolute():
        raise ValueError(f"buffer {index} uri {uri!r} should be relative to the .gltf file")
    target = directory / uri
    if not target.is_file():
        raise ValueError(f"buffer {index} uri {uri!r} does not resolve to a file in "
                         f"{directory}")
    return target.read_bytes()


def _validate_gltf(document: dict, path: Path):
    """Re-apply every glTF invariant this module claims, from the parsed document."""
    directory = Path(path).parent
    if document.get("asset", {}).get("version") != "2.0":
        raise ValueError("glTF asset.version must be the string '2.0'")
    if not document.get("buffers"):
        raise ValueError("no buffers declared")
    payloads = []
    for index, buffer in enumerate(document["buffers"]):
        data = _buffer_bytes(document, index, directory)
        if len(data) != buffer["byteLength"]:
            raise ValueError(f"buffer {index} declares byteLength {buffer['byteLength']} "
                             f"but {len(data)} bytes resolve from its uri")
        payloads.append(data)
    for index, view in enumerate(document.get("bufferViews", [])):
        if view["byteOffset"] % 4:
            raise ValueError(f"bufferView {index} byteOffset {view['byteOffset']} is not "
                             "aligned to a 4-byte boundary")
        data = payloads[view["buffer"]]
        if view["byteOffset"] + view["byteLength"] > len(data):
            raise ValueError(f"bufferView {index} byteLength {view['byteLength']} at offset "
                             f"{view['byteOffset']} runs past a {len(data)}-byte buffer")
    for index, accessor in enumerate(document.get("accessors", [])):
        if accessor["componentType"] not in _GLTF_COMPONENTS:
            raise ValueError(f"accessor {index} has unknown componentType "
                             f"{accessor['componentType']}")
        if accessor["type"] not in _GLTF_ELEMENTS:
            raise ValueError(f"accessor {index} has unknown type {accessor['type']}")
        width = _GLTF_COMPONENTS[accessor["componentType"]]
        step = width * _GLTF_ELEMENTS[accessor["type"]]
        if accessor["byteOffset"] % width or accessor["byteOffset"] % 4:
            raise ValueError(f"accessor {index} byteOffset {accessor['byteOffset']} is not "
                             "aligned to both the component size and a 4-byte boundary")
        view = document["bufferViews"][accessor["bufferView"]]
        stride = view.get("byteStride", step)
        if stride % width:
            raise ValueError(f"accessor {index} stride {stride} is not a multiple of the "
                             f"{width}-byte component size")
        needed = accessor["byteOffset"] + (accessor["count"] - 1) * stride + step
        if accessor["count"] and needed > view["byteLength"]:
            raise ValueError(f"accessor {index} count {accessor['count']} needs {needed} "
                             f"bytes but its bufferView byteLength is {view['byteLength']}")
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        for place, primitive in enumerate(mesh.get("primitives", [])):
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                raise ValueError(f"mesh {mesh_index} primitive {place} has no POSITION")
            position = document["accessors"][attributes["POSITION"]]
            if position["componentType"] != 5126 or position["type"] != "VEC3":
                raise ValueError("POSITION must be a VEC3 accessor of componentType 5126 "
                                 "(FLOAT)")
            if "min" not in position or "max" not in position:
                raise ValueError("a POSITION accessor must declare min and max so a viewer "
                                 "can build a bounding volume")
            mode = primitive.get("mode", _TRIANGLES)
            if mode not in (0, 1, 2, 3, 4, 5, 6):
                raise ValueError(f"mesh {mesh_index} primitive {place} has mode {mode}")
            if mode == _POINTS and "indices" in primitive:
                raise ValueError("a PRIMITIVE_POINTS must not carry indices")
            if mode == _TRIANGLES:
                if "indices" not in primitive:
                    raise ValueError("a PRIMITIVE_TRIANGLES needs an index accessor")
                index_accessor = document["accessors"][primitive["indices"]]
                if index_accessor["componentType"] not in (5121, 5123, 5125):
                    raise ValueError("triangle indices must be UNSIGNED_BYTE, "
                                     "UNSIGNED_SHORT or UNSIGNED_INT")
                if index_accessor["count"] % 3:
                    raise ValueError("a TRIANGLES index count must be a multiple of 3")
            for label, accessor_index in attributes.items():
                shared = document["accessors"][accessor_index]["count"]
                if shared != position["count"]:
                    raise ValueError(f"{label} declares {shared} vertices but POSITION has "
                                     f"{position['count']}")
    return payloads


def read_gltf(path) -> dict:
    """Parse the JSON, resolve the buffer, verify the invariants, return NumPy arrays."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    payloads = _validate_gltf(document, path)

    def read(index):
        accessor = document["accessors"][index]
        view = document["bufferViews"][accessor["bufferView"]]
        data = payloads[view["buffer"]]
        width = _GLTF_COMPONENTS[accessor["componentType"]]
        code = _GLTF_CODES[accessor["componentType"]]
        elements = _GLTF_ELEMENTS[accessor["type"]]
        stride = view.get("byteStride", width * elements)
        start = view["byteOffset"] + accessor["byteOffset"]
        rows = [struct.unpack_from("<" + code * elements, data, start + row * stride)
                for row in range(accessor["count"])]
        return np.array(rows, dtype=np.float64).reshape(accessor["count"], elements)

    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    return dict(positions=read(attributes["POSITION"]),
                colors=(read(attributes["COLOR_0"]) if "COLOR_0" in attributes
                        else np.zeros((0, 0))),
                triangles=(read(primitive["indices"]).astype(np.int64).reshape(-1, 3)
                           if "indices" in primitive else np.zeros((0, 3), np.int64)),
                mode=primitive.get("mode", _TRIANGLES), asset=document["asset"],
                document=document)


def _unique_pairs(pairs):
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("glTF JSON contains a duplicate field name")
    return out


# ------------------------------------------------------------------------------------ FBX
def write_fbx(positions, out, *, triangles=None, name: str = "survey",
              precision: int = 6) -> dict:
    """Write a deliberately minimal ASCII FBX. The docstring is the contract; read it.

    This is a **minimal writer**, not an FBX implementation. Exactly these constructs are
    emitted, and nothing else: ``FBXHeaderExtension`` (FBXHeaderVersion 1003, FBXVersion
    6100, Creator); ``GlobalSettings`` with ``Version`` and a short ``MetaData`` block;
    ``Definitions`` with ``ObjectType`` counts for Model and NodeAttribute; ``Objects``
    holding one ``NodeAttribute`` of type ``"Mesh"`` and one ``Model`` of type ``"Mesh"``
    with ``Version``, ``Properties60`` (Lcl Translation, Lcl Rotation, Lcl Scaling,
    DefaultAttributeIndex), ``Shading``, ``Culling``, ``Vertices``, ``GeometryVersion``;
    and ``Connections`` with two ``Connect: "OO"`` links attaching the geometry to the
    model and the model to the root node. A triangle mesh additionally gets
    ``PolygonVertexIndex`` and a ``LayerElementSmoothing`` block; a point cloud gets
    neither, because it has no polygons to index.

    Not emitted, so an importer may substitute defaults or complain: Takes, the YX axis
    setup, PreviewSettings, media, textures, materials, normals, UVs, vertex colors, layer
    clustering, skins, deformers, cameras, lights, and any 7.x-only or binary FBX
    structure. ASCII FBX 6.1 has no point primitive, so a point cloud is written as a Mesh
    node holding ``Vertices`` with no polygon fields at all, and a consumer may still show
    it as an empty object. Coordinates are fixed-point at ``precision`` decimals
    (micrometres at metric scale), so values are quantised on write and are not bit-exact.

    Every construct above is unverified: no FBX SDK or importer exists in this environment,
    so the only check is :func:`read_fbx` here, which proves self-consistency and nothing
    beyond it. Nobody should claim this file "opens in Blender".
    """
    verts = _finite_array(positions, "positions", 3)
    faces = None if triangles is None else _index_array(triangles, len(verts))
    if not 0 <= int(precision) <= 15:
        raise ValueError("precision must be between 0 and 15 decimals")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ["; FBX 6.1.0 project file",
            "; written by scripts/survey_formats.write_fbx - a minimal ASCII writer",
            "FBXHeaderExtension:  {",
            "\tFBXHeaderVersion: 1003",
            "\tFBXVersion: 6100",
            '\tCreator: "scripts/survey_formats.write_fbx (minimal ASCII FBX 6.1)"',
            "}",
            "GlobalSettings:  {",
            "\tVersion: 1000",
            "\tMetaData:  {",
            "\t\tVersion: 100",
            '\t\tTitle: "Drone to 3D mesh survey"',
            "\t}",
            "}",
            "Definitions:  {",
            "\tVersion: 100",
            "\tCount: 2",
            '\tObjectType: "Model" {',
            "\t\tCount: 1",
            "\t}",
            '\tObjectType: "NodeAttribute" {',
            "\t\tCount: 1",
            "\t}",
            "}",
            "Objects:  {",
            '\tNodeAttribute: 100000001, "NodeAttribute::' + name + '", "Mesh" {',
            "\t\tVersion: 1",
            "\t}",
            '\tModel: 100000002, "Model::' + name + '", "Mesh" {',
            "\t\tVersion: 230",
            "\t\tProperties60:  {"]
    for label, kind, value in (("Lcl Translation", "Vector3D", "0,0,0"),
                               ("Lcl Rotation", "Vector3D", "0,0,0"),
                               ("Lcl Scaling", "Vector", "1,1,1")):
        body.append(f'\t\t\tProperty: "{label}", "{label}", "A+", "{kind}",{value}')
    body.append('\t\t\tProperty: "DefaultAttributeIndex", "int", "A+", 0')
    body.append("\t\t}")
    body.append("\t\tShading: Y")
    body.append('\t\tCulling: "CullingOff"')
    body.append("\t\t; geometry: x, y, z triples\n\t\tVertices: "
                + ", ".join(f"%.*f" % (int(precision), value) for value in verts.reshape(-1)))
    if faces is not None:
        corners = []
        for row in faces:
            # FBX marks the last corner of each polygon by negating it and adding one.
            corners.extend([int(row[0]), int(row[1]), -int(row[2]) - 1])
        body.append("\t\tPolygonVertexIndex: " + ", ".join(str(v) for v in corners))
        body.append("\t\tGeometryVersion: 124")
        body.append("\t\tLayerElementSmoothing: 0 {")
        body.append('\t\t\tMappingInformationType: "ByPolygon"')
        body.append('\t\t\tReferenceInformationType: "Direct"')
        body.append("\t\t\tSmoothing: " + " ".join("1" for _ in faces))
        body.append("\t\t}")
    else:
        # A point cloud has no polygons, so no polygon-only field is emitted at all: an
        # empty PolygonVertexIndex is exactly the half-construct that makes an importer
        # refuse a file, and this writer claims only what it actually writes.
        body.append("\t\tGeometryVersion: 124")
    body += ["\t}", "}", "Connections:  {",
             '\tConnect: "OO", 100000001, 100000002',
             '\tConnect: "OO", 0, 100000002', "}"]
    out.write_text("\n".join(body) + "\n", encoding="ascii")
    try:
        read = read_fbx(out)
        tolerance = 1.5 * 10.0 ** -int(precision)
        if len(read["positions"]) != len(verts):
            raise ValueError("vertex count changed on re-read")
        if np.any(np.abs(read["positions"] - verts) > tolerance):
            raise ValueError(f"a vertex moved by more than {tolerance} on re-read")
        if faces is not None and not np.array_equal(read["triangles"], faces):
            raise ValueError("polygons did not survive the negative-corner encoding")
    except ValueError as exc:
        _reject(out, f"{out.name}: refused ASCII FBX that this module's reader rejected: "
                     f"{exc}")
    return _summary(out, verified=ROUND_TRIP, positions=len(verts),
                    polygons=0 if faces is None else len(faces), fbx_version=6100,
                    precision_decimals=int(precision),
                    emitted=["FBXHeaderExtension", "GlobalSettings", "Definitions",
                             "NodeAttribute(Mesh)", "Model(Mesh)", "Vertices",
                             "GeometryVersion", "Connections"]
                    + (["PolygonVertexIndex", "LayerElementSmoothing"]
                       if faces is not None else []),
                    unverified=["never opened by an FBX SDK or importer: read_fbx in this "
                                "module is the only parser that has seen these bytes",
                                "no materials, textures, normals, UVs, takes or axis setup",
                                "a point cloud is a Mesh with Vertices and no "
                                "PolygonVertexIndex at all, and whether an importer prefers "
                                "that or an empty index is untested here",
                                f"coordinates are quantised to {precision} decimals"])


def read_fbx(path) -> dict:
    """Re-parse Vertices and PolygonVertexIndex from an ASCII FBX written by this module."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:18].startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path.name}: binary FBX is not ASCII and is not read here")
    text = raw.decode("ascii", errors="replace")
    version = re.search(r"FBXVersion:\s*(\d+)", text)
    if version is None:
        raise ValueError(f"{path.name}: no FBXVersion header, so this is not an FBX file")
    positions = _fbx_row(path, text, "Vertices")
    if positions is None:
        raise ValueError(f"{path.name}: no Vertices property found in the Mesh model")
    corners = _fbx_row(path, text, "PolygonVertexIndex", integer=True)
    triangles = np.zeros((0, 3), dtype=np.int64)
    if corners is not None and len(corners):
        current, faces = [], []
        for value in corners.reshape(-1):
            if value < 0:
                current.append(-int(value) - 1)
                if len(current) != 3:
                    raise ValueError(f"{path.name}: only triangular polygons are read, "
                                     f"found one with {len(current)} corners")
                faces.append(current)
                current = []
            else:
                current.append(int(value))
        if current:
            raise ValueError(f"{path.name}: PolygonVertexIndex does not close a polygon")
        triangles = np.array(faces, dtype=np.int64).reshape(-1, 3)
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(positions)):
            raise ValueError(f"{path.name}: polygon index out of range for "
                             f"{len(positions)} vertices")
    return dict(positions=positions, triangles=triangles, fbx_version=int(version.group(1)))


def _fbx_row(path, text: str, key: str, integer: bool = False):
    match = re.search(rf"{key}:[ \t]*([^\n]*)", text)
    if match is None or not match.group(1).strip():
        return None
    tokens = [token for token in match.group(1).split(",") if token.strip()]
    try:
        values = [int(float(token)) if integer else float(token) for token in tokens]
    except ValueError:
        raise ValueError(f"{path.name}: {key} carries a value that is not a number") from None
    if integer:
        return np.array(values, dtype=np.int64).reshape(-1, 1)
    if len(values) % 3:
        raise ValueError(f"{path.name}: {key} holds {len(values)} numbers, not a multiple "
                         "of 3")
    array = np.array(values, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: {key} holds a non-finite coordinate")
    return array


# ------------------------------------------------------------------------------------ LAS
LAS_HEADER_SIZE = 239
_LAS_BASE = [("x", "i4"), ("y", "i4"), ("z", "i4"), ("intensity", "u2"),
             ("return_bits", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
             ("user_data", "u1"), ("point_source_id", "u2")]
_LAS_GPS = [("gps_time", "f8")]
_LAS_RGB = [("red", "u2"), ("green", "u2"), ("blue", "u2")]
# Lengths are the published values (0:20, 1:28, 2:26, 3:34). For format 3 the ORDER of the
# GPS-time and RGB blocks after the base record is an assumption made without the ASPRS
# table or laspy, and it is reported in every summary this module returns.
LAS_POINT_FORMATS = {
    0: dict(length=20, fields=_LAS_BASE),
    1: dict(length=28, fields=_LAS_BASE + _LAS_GPS),
    2: dict(length=26, fields=_LAS_BASE + _LAS_RGB),
    3: dict(length=34, fields=_LAS_BASE + _LAS_GPS + _LAS_RGB),
}
_LAS_GEOTIFF_VLR = dict(user_id=b"LASF_proj", record_id=34735,
                        description="GeoTIFF GeoKeyDirectoryTag")
_LAS_WKT_VLR = dict(user_id=b"LASF_Projection", record_id=2112,
                    description="OGC WKT Coordinate System")
_PACKED_FIELDS = (("return_number", 0, 0x07), ("number_of_returns", 3, 0x38),
                  ("scan_direction_flag", 6, 0x40), ("edge_of_flight_line", 7, 0x80))


def _las_dtype(point_format: int):
    return np.dtype([(name, "<" + code)
                     for name, code in LAS_POINT_FORMATS[point_format]["fields"]])


def write_las(points, out, *, scale, offsets, srs_wkt=None, point_format: int = 3,
              system_identifier: str = "Drone to 3D mesh survey",
              generating_software: str = "scripts/survey_formats.write_las") -> dict:
    """Write a LAS 1.4 file: public header block, VLRs, then point data records.

    ``scale`` and ``offsets`` are three numbers each. A coordinate is stored exactly as
    ``round((value - offset) / scale)`` in int32, so the precision delivered is the scale
    requested and nothing more, and a coordinate that would not fit int32 is refused
    instead of wrapping. The header carries signature ``LASF``, version 1.4, header_size
    239, the scale/offset triples, the X/Y/Z max-then-min records with the offsets
    subtracted, ``offset_to_point_data`` after the VLRs, and the 32-bit and 64-bit point
    counts set to the same value. **The 239-byte block and the byte offsets used for these
    fields are this module's own layout, not a transcribed ASPRS 1.4 table - the published
    1.4 header is longer, carries per-return counts and places these fields elsewhere - so
    treat the files as self-consistent LAS-shaped data until laspy or PDAL has read one.**

    VLRs use the 54-byte base header (Reserved, UserID[16], RecordDataFormatID,
    Description[32], RecordDataLength). A GeoTIFF key-directory VLR (UserID ``LASF_proj``,
    record id 34735) is always written so the file states *some* CRS; when ``srs_wkt`` is
    supplied a second VLR (UserID ``LASF_Projection``, record id 2112) carries that WKT
    text. Key values come from an EPSG code found inside the WKT, or are written as
    user-defined (32767) when there is none, because no pyproj or EPSG registry is
    importable here - and no datum or geoid transformation is performed or claimed.

    ``point_format`` supports 0, 1, 2 and 3 with record lengths 20, 28, 26 and 34. **For
    format 3 this writer places GPS time before RGB after the 20-byte base record; that
    ordering is an assumption, not a checked fact, and it appears in the returned
    ``unverified`` list.** The return-number fields share one byte at 3/3/1/1 bits, so a
    return number or return count above 7 is refused: the wider 1.4 bit packing lives in
    point formats 6-10, which are not implemented.

    Optional columns: ``intensity``, ``classification``, ``scan_angle``, ``user_data``,
    ``point_source_id``, ``gps_time``, ``red``/``green``/``blue`` (all three or none) and
    the four packed return/flag fields. Anything the caller omits is written as zeros, and
    the summary lists which columns were actually supplied.
    """
    if point_format not in LAS_POINT_FORMATS:
        raise ValueError(f"point_format {point_format} is not implemented: supported values "
                         f"are {sorted(LAS_POINT_FORMATS)}; the 1.4-wide formats 6-10 need a "
                         "record layout this module cannot verify here")
    scale = np.asarray(scale, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if scale.shape != (3,) or offsets.shape != (3,):
        raise ValueError("scale and offsets must each hold three numbers")
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale must be three finite positive metres-per-unit numbers")
    if not np.isfinite(offsets).all():
        raise ValueError("offsets must be finite")
    if srs_wkt is not None:
        _check_wkt(srs_wkt)
    spec = LAS_POINT_FORMATS[point_format]
    available = tuple(name for name, _ in spec["fields"]
                      if name not in ("x", "y", "z", "return_bits"))
    packed_names = tuple(name for name, _shift, _mask in _PACKED_FIELDS)
    data, count = _columns(points, ("x", "y", "z"), optional=available + packed_names)
    if {"red", "green", "blue"} & set(data) and not {"red", "green", "blue"} <= set(data):
        raise ValueError("red, green and blue are one field group: supply all three or none")
    record = np.zeros(count, dtype=_las_dtype(point_format))
    for axis in "xyz":
        index = "xyz".index(axis)
        scaled = (np.asarray(data[axis], dtype=float) - offsets[index]) / scale[index]
        rounded = np.rint(scaled)
        if rounded.size and (rounded.min() < -2147483648 or rounded.max() > 2147483647):
            step = float(scale[index])
            origin = float(offsets[index])
            # int32 holds 2**32 consecutive values, so one offset/scale pair reaches
            # 2**32 * step metres in total, split across the offset as -2**31 .. 2**31-1.
            raise ValueError(
                f"{axis} does not fit the int32 LAS field at scale {step} with offset "
                f"{origin}: the adjusted range is {rounded.min():.0f}..{rounded.max():.0f} "
                f"units but int32 holds only -2147483648..2147483647. At scale {step} m one "
                f"int32 covers {2 ** 32 * step:.3f} m in total, i.e. "
                f"{origin - 2 ** 31 * step:.3f}..{origin + (2 ** 31 - 1) * step:.3f} m about "
                f"this offset: coarsen the scale or move the offset to the data centroid")
        record[axis] = rounded.astype(np.int32)
    packed = np.zeros(count, dtype=np.uint8)
    for field, shift, mask in _PACKED_FIELDS:
        if field not in data:
            continue
        values = np.asarray(data[field], dtype=np.int64)
        limit = mask >> shift
        if values.min() < 0 or values.max() > limit:
            raise ValueError(f"{field} must fit its packed bits (0..{limit}) for point "
                             f"format {point_format}; wider return numbers need LAS 1.4 "
                             "formats 6-10, which this writer does not emit")
        packed |= ((values & mask) << shift).astype(np.uint8)
    record["return_bits"] = packed
    for name in available:
        if name not in data:
            continue
        values = np.asarray(data[name])
        target = record[name].dtype
        if target.kind == "f":
            if np.asarray(values, dtype=float).size and not np.isfinite(
                    np.asarray(values, dtype=float)).all():
                raise ValueError(f"column {name} must be finite")
        else:
            low, high = int(np.iinfo(target).min), int(np.iinfo(target).max)
            if values.min() < low or values.max() > high:
                raise ValueError(f"column {name} must fit the {target} field "
                                 f"({low}..{high})")
        record[name] = values.astype(target)
    keys = _key_directory(_crs_kind(srs_wkt), _epsg_from_wkt(srs_wkt))
    vlr_specs: list[tuple[dict, bytes]] = [
        (_LAS_GEOTIFF_VLR, struct.pack("<" + "H" * len(keys), *keys))]
    if srs_wkt:
        vlr_specs.append((_LAS_WKT_VLR,
                          srs_wkt.encode("ascii", errors="replace") + b"\x00"))
    vlrs = [_las_vlr(spec, payload) for spec, payload in vlr_specs]
    vlr_bytes = b"".join(vlrs)
    if len(vlr_bytes) % 4:
        vlr_bytes += b"\x00" * (4 - len(vlr_bytes) % 4)
    if not count:
        raise ValueError("write_las refuses an empty cloud: a LAS file with no points is "
                         "not a survey product")
    offset_to_points = LAS_HEADER_SIZE + len(vlr_bytes)
    adjusted = [record[axis] * scale[index] for index, axis in enumerate("xyz")]
    now = datetime.datetime.now(datetime.timezone.utc)
    # Field offsets, as read back by read_las and by the tests' independent parser:
    # 94 header_size, 96 offset_to_point_data, 100 n_vlrs, 104 vlr_bytes, 108 point_format,
    # 109 record_length, 111 the 32-bit count, 135 the 64-bit count (set to the same value,
    # never to a second batch of records), 143 scale, 167 offset, 191 max then min per axis.
    # There is no room here for the per-return counts, and these offsets are this module's
    # own layout rather than a checked ASPRS 1.4 table: see verification_summary.
    header = struct.pack(
        "<4sHH16sBB32s32sHHHIIIBHIQQIQ",
        b"LASF", 0, 0, bytes(16), 1, 4,
        _fixed(system_identifier, 32), _fixed(generating_software, 32),
        now.timetuple().tm_yday, now.year, LAS_HEADER_SIZE, offset_to_points,
        len(vlrs), len(vlr_bytes), point_format, spec["length"], count, 0, 0, 0, count)
    header += struct.pack("<3d3d", *[float(v) for v in scale], *[float(v) for v in offsets])
    header += struct.pack("<6d", float(adjusted[0].max()), float(adjusted[0].min()),
                          float(adjusted[1].max()), float(adjusted[1].min()),
                          float(adjusted[2].max()), float(adjusted[2].min()))
    if len(header) != LAS_HEADER_SIZE:
        raise ValueError(f"internal error: the header built here is {len(header)} bytes, "
                         f"not {LAS_HEADER_SIZE}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        stream.write(header)
        stream.write(vlr_bytes)
        stream.write(record.tobytes())
    try:
        read = read_las(out)
        if read["n_points"] != count:
            raise ValueError("the point count changed on re-read")
        for index, axis in enumerate("xyz"):
            if not np.array_equal(read["raw_points"][axis], record[axis]):
                raise ValueError(f"{axis} did not survive the write/read round trip")
            # read_las hands back metres, so the scale/offset pair is exercised too: no
            # coordinate may sit further from its input than half the quantisation step.
            drift = np.abs(read["points"][axis] - np.asarray(data[axis], dtype=float))
            limit = (0.5 * float(scale[index])
                     + 8.0 * 2.220446049250313e-16 * float(np.max(np.abs(read["points"][axis]))))
            if float(drift.max()) > limit:
                raise ValueError(f"{axis} came back {float(drift.max())} m from its input, "
                                 f"more than half the {float(scale[index])} m step")
        if bool(srs_wkt) != bool(read["srs_wkt"]):
            raise ValueError("the CRS VLR did not survive the round trip")
    except ValueError as exc:
        _reject(out, f"{out.name}: refused LAS that this module's reader rejected: {exc}")
    return _summary(out, verified=ROUND_TRIP,
                    unverified=["no laspy or PDAL is importable here, so read_las in this "
                                "module is the only reader that has parsed these bytes",
                                "the 239-byte header layout and its field offsets are this "
                                "module's own, not a transcribed ASPRS 1.4 table, and the "
                                "per-return counts are absent",
                                "point format field order after the 20-byte base record is an "
                                "assumption, not a checked fact"
                                + (" (gps_time before red/green/blue)" if point_format == 3
                                   else ""),
                                "the wider LAS 1.4 return-number bit fields (formats 6-10) "
                                "are not written, so returns are limited to 0..7"],
                    n_points=count, point_format=point_format,
                    record_length=spec["length"], offset_to_point_data=offset_to_points,
                    scale=[float(v) for v in scale],
                    offsets=[float(v) for v in offsets], geokeys=keys,
                    vlr_record_ids=[spec["record_id"] for spec, _payload in vlr_specs],
                    columns=list(record.dtype.names), supplied=sorted(data),
                    quantisation_m=float(np.max(scale)), wkt_written=bool(srs_wkt))


def _fixed(text: str, width: int) -> bytes:
    """Pad a value into a fixed-width LAS field. The field needs no terminator when full.

    Truncation is refused rather than silently applied, so a name that does not fit is an
    error, but a name of exactly ``width`` characters is legal and simply leaves no NUL.
    """
    raw = text.encode("ascii", errors="replace")
    if len(raw) > width:
        raise ValueError(f"{text!r} does not fit the fixed {width}-byte LAS header field")
    return raw + b"\x00" * (width - len(raw))


def _las_vlr(spec: dict, payload: bytes) -> bytes:
    return struct.pack("<H16sH32sH", 0, spec["user_id"], spec["record_id"],
                       _fixed(spec["description"], 32), len(payload)) + payload


_EPSG_PATTERN = re.compile(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]'
                           r'|\bEPSG[:/-]\s*(\d+)\b', re.IGNORECASE)


def _epsg_from_wkt(wkt):
    """Pull an EPSG code out of WKT text. Nothing is looked up: an unknown name is None.

    The code of the CRS itself is the LAST authority in a WKT1/WKT2 definition: a
    full PROJCS opens with the spheroid's 7030 and the datum's 6326, so taking the
    first match would label a GeoTIFF's projected-CRS key with an ellipsoid.
    """
    if not wkt:
        return None
    matches = _EPSG_PATTERN.findall(wkt)
    if not matches:
        return None
    code = int(next((m[0] for m in reversed(matches) if m[0]),
                    next((m[1] for m in reversed(matches) if m[1]), 0)))
    return code if 1000 <= code < 100000 else None


def _check_wkt(wkt) -> None:
    if not isinstance(wkt, str) or not re.search(r"(PROJCS|GEOGCS|GEOGCR)\s*\[", wkt,
                                                 re.IGNORECASE):
        raise ValueError("crs_wkt does not look like a WKT CRS: a PROJCS/GEOGCS (or WKT2 "
                         "GEOGCR) definition is required and none is invented here")


def _crs_kind(wkt) -> str:
    if wkt and "PROJCS" in wkt.upper():
        return "projected"
    if wkt and re.search(r"GEOGC[SR]\s*\[", wkt, re.IGNORECASE):
        return "geographic"
    return "unknown"


def _key_directory(kind: str, code) -> list[int]:
    """GeoTIFF key directory: version, revision, minor, count, then key quadruples."""
    model_type = {"projected": 1, "geographic": 2}.get(kind, 32767)
    keys = [(1024, 0, 1, model_type), (1025, 0, 1, 1),
            (3072 if kind == "projected" else 1026, 0, 1, 32767 if code is None else code)]
    out = [1, 1, 0, len(keys)]
    for item in keys:
        out.extend(item)
    return out


def read_las(path) -> dict:
    """Re-parse the LAS header, walk the VLRs and decode every point data record.

    ``points`` holds one column per field, and its ``x``/``y``/``z`` are in the file's own
    coordinate units: the stored int32 is converted back with ``value * scale + offset``,
    which is what every other LAS reader (laspy included) returns. Reading the raw int32
    as if it were a position would put a whole cloud kilometres off, so the untouched
    records stay available separately under ``raw_points``.
    """
    path = Path(path)
    text = path.read_bytes()
    if len(text) < LAS_HEADER_SIZE:
        raise ValueError(f"{path.name}: truncated - {len(text)} bytes is shorter than the "
                         f"{LAS_HEADER_SIZE}-byte LAS 1.4 header")
    if text[:4] != b"LASF":
        raise ValueError(f"{path.name}: file signature is {text[:4]!r}, not the required "
                         "LASF")
    version = (text[24], text[25])
    if version != (1, 4):
        raise ValueError(f"{path.name}: only LAS 1.4 is read here, found "
                         f"{version[0]}.{version[1]}")
    header_size, offset_to_points, n_vlrs, vlr_bytes = struct.unpack_from("<HIII", text, 94)
    point_format = text[108]
    record_length, n_points = struct.unpack_from("<HI", text, 109)
    n_points_64 = struct.unpack_from("<Q", text, 135)[0]
    scale = struct.unpack_from("<3d", text, 143)
    offsets = struct.unpack_from("<3d", text, 167)
    extremes = struct.unpack_from("<6d", text, 191)
    if point_format not in LAS_POINT_FORMATS:
        raise ValueError(f"{path.name}: point format {point_format} has a record layout "
                         "this module does not claim to know")
    expected = LAS_POINT_FORMATS[point_format]["length"]
    if record_length != expected:
        raise ValueError(f"{path.name}: declared record length {record_length} disagrees "
                         f"with the {expected} bytes that format needs")
    # The 32-bit legacy count and the 64-bit extended count describe the SAME records, so
    # they are not additive: the extended value wins where it is present, and the two may
    # only disagree when a writer zeroed the legacy field because the count exceeded 2**32.
    if n_points_64 and n_points and n_points_64 != n_points:
        raise ValueError(f"{path.name}: the 32-bit count {n_points} and the 64-bit extended "
                         f"count {n_points_64} disagree, and one cloud is read here")
    total_records = n_points_64 or n_points
    if not total_records:
        raise ValueError(f"{path.name}: neither point count is set, so there is nothing to "
                         "read")
    if offset_to_points < header_size:
        raise ValueError(f"{path.name}: offset_to_point_data {offset_to_points} points "
                         f"inside the {header_size}-byte header")
    cursor, geokeys, wkt, vlrs = header_size, None, None, []
    for _ in range(n_vlrs):
        if cursor + 54 > len(text):
            raise ValueError(f"{path.name}: truncated VLR header at byte {cursor}")
        _reserved, user_id, record_id, description, length = \
            struct.unpack_from("<H16sH32sH", text, cursor)
        payload = text[cursor + 54:cursor + 54 + length]
        if len(payload) != length:
            raise ValueError(f"{path.name}: truncated VLR payload at byte {cursor}")
        vlrs.append(dict(user_id=user_id.split(b"\x00")[0].decode("ascii", errors="replace"),
                         record_id=record_id,
                         description=description.split(b"\x00")[0].decode("ascii",
                                                                          errors="replace"),
                         length=length))
        if record_id == _LAS_GEOTIFF_VLR["record_id"]:
            geokeys = list(struct.unpack("<" + "H" * (length // 2), payload))
        if record_id == _LAS_WKT_VLR["record_id"]:
            wkt = payload.split(b"\x00")[0].decode("ascii", errors="replace")
        cursor += 54 + length
    if cursor > offset_to_points:
        raise ValueError(f"{path.name}: the VLRs reach byte {cursor} but the point data is "
                         f"declared to start at {offset_to_points}")
    needed = offset_to_points + total_records * record_length
    if len(text) < needed:
        raise ValueError(f"{path.name}: truncated point data - {total_records} records of "
                         f"{record_length} bytes need {needed} bytes and the file is "
                         f"{len(text)}")
    record = np.frombuffer(text[offset_to_points:needed], dtype=_las_dtype(point_format),
                           count=total_records)
    columns = {name: record[name].copy() for name in record.dtype.names
               if name != "return_bits"}
    for field, shift, mask in _PACKED_FIELDS:
        columns[field] = ((record["return_bits"] >> shift) & mask).astype(np.uint8)

    def as_structured(source: dict):
        built = np.zeros(total_records, dtype=[(name, values.dtype)
                                               for name, values in source.items()])
        for name, values in source.items():
            built[name] = values
        return built

    # ``raw_points`` is the record exactly as stored; ``points`` converts the three integer
    # coordinate columns back out of the quantisation, so a caller never mistakes int32
    # units for metres. The other columns (intensity, classification, gps_time, RGB) are
    # stored as-is and are identical in both.
    raw = as_structured(columns)
    for index, axis in enumerate("xyz"):
        columns[axis] = (columns[axis].astype(np.float64) * float(scale[index])
                         + float(offsets[index]))
    return dict(points=as_structured(columns), raw_points=raw,
                n_points=int(total_records), version=version,
                header_size=int(header_size), offset_to_point_data=int(offset_to_points),
                point_format=int(point_format), record_length=int(record_length),
                scale=scale, offsets=offsets,
                maxs=(extremes[0], extremes[2], extremes[4]),
                mins=(extremes[1], extremes[3], extremes[5]),
                geokeys=geokeys, srs_wkt=wkt, vlrs=vlrs, n_vlrs=int(n_vlrs),
                vlr_bytes=int(vlr_bytes),
                system_id=text[26:58].split(b"\x00")[0].decode("ascii", errors="replace"),
                software_id=text[58:90].split(b"\x00")[0].decode("ascii", errors="replace"),
                verified=ROUND_TRIP, externally_validated=False)


# ------------------------------------------------------------------------------- GeoTIFF
TIFF_BYTE, TIFF_ASCII, TIFF_SHORT, TIFF_LONG, TIFF_DOUBLE = 1, 2, 3, 4, 12
_TIFF_WIDTH = {TIFF_BYTE: 1, TIFF_ASCII: 1, TIFF_SHORT: 2, TIFF_LONG: 4, TIFF_DOUBLE: 8}
_TIFF_CODES = {TIFF_BYTE: "B", TIFF_ASCII: "s", TIFF_SHORT: "H", TIFF_LONG: "I",
               TIFF_DOUBLE: "d"}
_GEOTIFF_REQUIRED = (256, 257, 258, 259, 262, 273, 277, 278, 279, 339, 33550, 33922, 34735)
_GEOTIFF_ASCII_PARAMS, _GDAL_NODATA, _TIFF_SOFTWARE = 34737, 42112, 305


def write_geotiff(raster, out, *, transform, crs_wkt, nodata=None,
                  description: str = "scripts/survey_formats.write_geotiff") -> dict:
    """Write a single-band float32 GeoTIFF: little-endian TIFF, one strip, no compression.

    ``transform`` is the six GDAL-style numbers ``(a, b, c, d, e, f)`` where
    x = a + col*b + row*c and y = d + col*e + row*f. ModelPixelScaleTag plus
    ModelTiepointTag can only express a north-up grid, so b must be positive, f negative
    and c and e zero: a rotated or sheared transform is refused rather than flattened,
    because ModelTransformationTag (34264) is not emitted here.

    Tags written: ImageWidth(256), ImageLength(257), BitsPerSample(258)=32,
    Compression(259)=1, PhotometricInterpretation(262)=1, StripOffsets(273),
    SamplesPerPixel(277)=1, RowsPerStrip(278), StripByteCounts(279), SampleFormat(339)=3
    (IEEE floating point), ModelPixelScaleTag(33550), ModelTiepointTag(33922),
    GeoKeyDirectoryTag(34735), GeoAsciiParamsTag(34737) carrying ``crs_wkt``, Software(305)
    and GDAL_NODATA(42112) when a nodata value is given. The GeoKey directory is built from
    an EPSG code found in the WKT text, or marked user-defined (32767) when there is none:
    no EPSG registry or pyproj is available here, so the CRS is carried and never validated,
    and no datum transformation is implied.

    A NaN cell requires an explicit finite ``nodata``: writing NaN into a grid whose
    consumer treats that number as data is how a hole silently becomes ground. Multi-band
    rasters, double precision, compression and tiled layouts are out of scope and refused
    rather than approximated.
    """
    try:
        values = np.asarray(raster, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("raster must hold numbers: None and non-numeric cells are never "
                         "written") from None
    if values.ndim != 2:
        raise ValueError("raster must be a single-band 2-D grid of shape (rows, columns), "
                         f"got {values.ndim}-D shape {values.shape}")
    if not values.size:
        raise ValueError("raster is empty")
    if np.isinf(values).any():
        raise ValueError("raster cells must be finite or NaN: an infinity is never written")
    affine = np.asarray(transform, dtype=float)
    if affine.shape != (6,) or not np.isfinite(affine).all():
        raise ValueError(f"transform must have shape (6,) of finite numbers "
                         f"(a, b, c, d, e, f), got {affine.shape}")
    a, b, c, d, e, f = (float(v) for v in affine)
    if c != 0.0 or e != 0.0:
        raise ValueError("a rotated or sheared transform needs ModelTransformationTag "
                         "(34264), which this writer does not emit")
    if not b > 0 or not f < 0:
        raise ValueError("only the north-up row order is supported: easting per pixel must "
                         f"be positive and northing per row negative, got b={b} and f={f}")
    _check_wkt(crs_wkt)
    holes = np.isnan(values)
    if holes.any() and (nodata is None or not math.isfinite(float(nodata))):
        raise ValueError(f"{int(holes.sum())} cell(s) are NaN but nodata is not a finite "
                         "number: an unlabelled hole cannot be told apart from a measurement")
    payload = np.ascontiguousarray(
        np.where(holes, float(nodata if nodata is not None else 0.0), values), dtype="<f4")
    rows, columns = payload.shape
    strip = payload.tobytes()
    keys = _key_directory(_crs_kind(crs_wkt), _epsg_from_wkt(crs_wkt))
    entries: list[list] = []

    def numeric(tag: int, type_id: int, items, code: str) -> None:
        cast = int if code in "HI" else float
        entries.append([tag, type_id, len(items),
                        struct.pack("<" + code * len(items), *[cast(v) for v in items])])

    def ascii_tag(tag: int, text: str) -> None:
        blob = (text + chr(0)).encode("ascii", errors="replace")
        entries.append([tag, TIFF_ASCII, len(blob), blob])

    numeric(256, TIFF_LONG, [columns], "I")
    numeric(257, TIFF_LONG, [rows], "I")
    numeric(258, TIFF_SHORT, [32], "H")
    numeric(259, TIFF_SHORT, [1], "H")
    numeric(262, TIFF_SHORT, [1], "H")
    numeric(273, TIFF_LONG, [8], "I")            # the 8-byte header, then the strip
    numeric(277, TIFF_SHORT, [1], "H")
    numeric(278, TIFF_SHORT, [rows], "H")        # one strip covers the whole image
    numeric(279, TIFF_LONG, [len(strip)], "I")
    numeric(339, TIFF_SHORT, [3], "H")           # 3 = IEEE floating point
    ascii_tag(_TIFF_SOFTWARE, description)
    numeric(33550, TIFF_DOUBLE, [b, -f, 0.0], "d")
    numeric(33922, TIFF_DOUBLE, [0.0, 0.0, 0.0, a, d, 0.0], "d")
    numeric(34735, TIFF_SHORT, keys, "H")
    ascii_tag(_GEOTIFF_ASCII_PARAMS, crs_wkt)
    if nodata is not None:
        label = repr(float(nodata))
        if "e" not in label and "E" not in label:
            # Only a plain decimal may be trimmed: rstrip would eat the exponent digits of
            # "1e+20" and hand the consumer a nodata value for the wrong number.
            label = label.rstrip("0").rstrip(".")
        ascii_tag(_GDAL_NODATA, label)
    entries.sort(key=lambda item: item[0])
    # Layout: TIFF header (8 bytes), the strip, out-of-line tag values, then the IFD.
    values_start = 8 + len(strip)
    if values_start % 2:
        values_start += 1
    cursor, fields, value_blob = values_start, [], bytearray()
    for tag, type_id, count, blob in entries:
        total = _TIFF_WIDTH[type_id] * count
        if total <= 4:
            fields.append([tag, type_id, count, blob + b"\x00" * (4 - total), None])
            continue
        if cursor % 2:
            value_blob += b"\x00"
            cursor += 1
        fields.append([tag, type_id, count, struct.pack("<I", cursor), blob])
        value_blob += blob
        cursor += total
    ifd_offset = cursor
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        stream.write(struct.pack("<2sHI", b"II", 42, ifd_offset))
        stream.write(strip)
        if values_start != 8 + len(strip):
            stream.write(b"\x00")
        stream.write(bytes(value_blob))
        stream.write(struct.pack("<H", len(fields)))
        for tag, type_id, count, field, _blob in fields:
            stream.write(struct.pack("<HHI", tag, type_id, count) + field[:4])
        stream.write(struct.pack("<I", 0))       # no next IFD
    try:
        read = read_geotiff(out)
        if read["shape"] != (rows, columns):
            raise ValueError(f"shape read back as {read['shape']}")
        if tuple(read["transform"]) != (a, b, c, d, e, f):
            raise ValueError(f"transform read back as {read['transform']}")
        finite = ~holes
        limit = max(1e-6, float(np.max(np.abs(values[finite]))) * 1e-6) if finite.any() else 1e-6
        if not np.allclose(read["raster"][finite], values[finite], atol=limit):
            raise ValueError("raster values did not survive the float32 round trip")
    except ValueError as exc:
        _reject(out, f"{out.name}: refused GeoTIFF that this module's reader rejected: {exc}")
    return _summary(out, verified=ROUND_TRIP, width=columns, height=rows, dtype="float32",
                    tags=[item[0] for item in entries],
                    transform=[a, b, c, d, e, f], geokeys=keys,
                    epsg=_epsg_from_wkt(crs_wkt),
                    nodata=None if nodata is None else float(nodata),
                    unverified=["no GDAL, rasterio or PDAL is importable here, so "
                                "read_geotiff in this module is the only reader that has "
                                "parsed these bytes",
                                "the GeoKey values were derived from the WKT text handed in "
                                "and never resolved against the EPSG registry",
                                "single strip, north-up, one band and float32 only"])


def read_geotiff(path) -> dict:
    """Re-parse the IFD and return the raster, transform, GeoKeys and CRS text."""
    path = Path(path)
    text = path.read_bytes()
    if len(text) < 8 or text[:2] != b"II":
        raise ValueError(f"{path.name}: a GeoTIFF read here must be little-endian, marked "
                         "II; big-endian MM is not decoded")
    if struct.unpack_from("<H", text, 2)[0] != 42:
        raise ValueError(f"{path.name}: TIFF version marker is not 42")
    ifd = struct.unpack_from("<I", text, 4)[0]
    if ifd + 2 > len(text):
        raise ValueError(f"{path.name}: truncated - the IFD offset {ifd} is past the end")
    count = struct.unpack_from("<H", text, ifd)[0]
    tags = {}
    for index in range(count):
        base = ifd + 2 + index * 12
        tag, type_id, length = struct.unpack_from("<HHI", text, base)
        if type_id not in _TIFF_WIDTH:
            raise ValueError(f"{path.name}: tag {tag} uses unknown field type {type_id}")
        total = _TIFF_WIDTH[type_id] * length
        raw = text[base + 8:base + 12]
        if total > 4:
            where = struct.unpack("<I", raw)[0]
            if where + total > len(text):
                raise ValueError(f"{path.name}: tag {tag} points outside the file")
            raw = text[where:where + total]
        raw = raw[:total]
        if type_id == TIFF_ASCII:
            values = [raw.decode("ascii", errors="replace")]
        else:
            values = list(struct.unpack("<" + _TIFF_CODES[type_id] * length, raw))
        tags[tag] = dict(type=type_id, count=length, values=values)
    missing = [tag for tag in _GEOTIFF_REQUIRED if tag not in tags]
    if missing:
        raise ValueError(f"{path.name}: required GeoTIFF tags are absent: {missing}")
    if tags[258]["values"] != [32] or tags[339]["values"] != [3]:
        raise ValueError(f"{path.name}: only 32-bit IEEE float rasters are read "
                         f"(BitsPerSample={tags[258]['values']}, "
                         f"SampleFormat={tags[339]['values']})")
    if tags[277]["values"] != [1]:
        raise ValueError(f"{path.name}: only single-band rasters are read, found "
                         f"{tags[277]['values']} samples per pixel")
    if tags[259]["values"] != [1]:
        raise ValueError(f"{path.name}: compression {tags[259]['values']} is not decoded")
    width, height = tags[256]["values"][0], tags[257]["values"][0]
    offset, length = tags[273]["values"][0], tags[279]["values"][0]
    if offset + length > len(text):
        raise ValueError(f"{path.name}: truncated strip data at {offset} + {length} bytes")
    if length != width * height * 4:
        raise ValueError(f"{path.name}: StripByteCounts {length} is not {width}x{height} "
                         "float32 pixels")
    raster = np.frombuffer(text[offset:offset + length], dtype="<f4").reshape(height, width)
    scale = tags[33550]["values"]
    tie = tags[33922]["values"]
    transform = (tie[3] - tie[0] * scale[0], scale[0], 0.0, tie[4] + tie[1] * scale[1],
                 0.0, -scale[1])
    keys = tags[34735]["values"]
    if keys[:3] != [1, 1, 0]:
        raise ValueError(f"{path.name}: unexpected GeoKeyDirectory header {keys[:3]}")
    triples = {tuple(keys[4 + i * 4:8 + i * 4]) for i in range(keys[3])}
    codes = [value for key, _loc, _count, value in triples if key in (1026, 3072)]
    user_defined = any(value == 32767 for _key, _loc, _count, value in triples)
    epsg = codes[0] if codes and codes[0] != 32767 else None
    nodata = None
    if _GDAL_NODATA in tags:
        raw = tags[_GDAL_NODATA]["values"][0].split(chr(0))[0].strip()
        nodata = float(raw) if raw else None
    values = raster.astype(np.float64)
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    crs_wkt = None
    if _GEOTIFF_ASCII_PARAMS in tags:
        crs_wkt = tags[_GEOTIFF_ASCII_PARAMS]["values"][0].split(chr(0))[0] or None
    return dict(raster=values, shape=(int(height), int(width)), transform=transform,
                pixel_scale=(scale[0], scale[1]), nodata=nodata, epsg=epsg,
                user_defined_crs=bool(user_defined), crs_wkt=crs_wkt, geokeys=keys,
                tags=sorted(tags), dtype="float32", verified=ROUND_TRIP,
                externally_validated=False)
