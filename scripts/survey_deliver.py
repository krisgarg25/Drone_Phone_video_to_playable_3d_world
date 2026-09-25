"""Turn one reconstruction run into the deliverables the brief names, in a real CRS.

The brief asks for OBJ, PLY, LAS, GeoTIFF, .glb/.gltf and .fbx of a **georeferenced**
model. ``survey_export`` already writes every container that needs no invention, but
it writes them in the local east/north/up tangent plane, which is not a CRS a GIS can
open. This module does the two things that closes the gap and nothing else:

* read the surface mesh COLMAP's Poisson mesher produced, so faces exist and OBJ can
  be written from measured geometry rather than from a guess;
* derive the scene's UTM CRS from its own GPS origin (``survey_crs``) and hand
  ``survey_export`` already-projected coordinates, so the same writers fill a
  georeferenced set alongside the local one.

A product set is never described as complete: ``official_ledger`` reports each of the
six formats as delivered or not delivered with the reason the exporter actually gave.
"""
import csv
import json
import os
import uuid
from pathlib import Path

import numpy as np

try:  # imported as scripts.survey_deliver by the tests
    from scripts import survey_crs as crs
    from scripts import survey_export as exporter
    from scripts import survey_georef as georef
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_crs as crs
    import survey_export as exporter
    import survey_georef as georef

# The six containers named in the problem statement, in the order it lists them.
OFFICIAL_FORMATS = ("obj", "ply", "las", "geotiff", "glb/gltf", "fbx")
_FORMAT_ALIASES = {"glb/gltf": ("gltf", "glb"), "geotiff": ("geotiff", "tif", "tiff")}
MAX_WGS84_ROWS = 200_000
"""A full-resolution cloud can be millions of points; a text position table is
useless at that size, so it is decimated and the decimation is reported."""


def read_mesh_ply(path):
    """(vertices, faces, colors) from a PLY that carries faces.

    Returns ``(None, None, None)`` when the file holds only vertices, which is the
    normal state before meshing: that is not an error, it is the reason OBJ is not
    delivered. A face list with anything other than three corners raises, because
    silently triangulating a polygon would invent geometry nobody measured.
    """
    from plyfile import PlyData
    path = Path(path)
    if not path.is_file():
        return None, None, None
    data = PlyData.read(str(path))
    if "face" not in data:
        return None, None, None
    vertices = data["vertex"].data
    points = np.column_stack([vertices[k].astype(np.float64) for k in ("x", "y", "z")])
    colors = None
    if all(k in vertices.dtype.names for k in ("red", "green", "blue")):
        colors = np.column_stack([vertices[k].astype(np.uint8) for k in
                                  ("red", "green", "blue")])
    faces_raw = np.asarray(list(data["face"].data["vertex_indices"]), dtype=object)
    counts = {len(row) for row in faces_raw}
    if counts and counts != {3}:
        raise ValueError(f"{path.name}: faces with {sorted(counts)} corners; only triangles "
                         "are read, and a polygon would have to be triangulated by guesswork")
    faces = np.vstack([row for row in faces_raw]).astype(np.int64) if len(faces_raw) \
        else np.zeros((0, 3), dtype=np.int64)
    if len(faces) and (faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError(f"{path.name}: a face references a vertex that is not in the file")
    return points, faces, colors


def _status_for(fmt, records, missing):
    for entry in records:
        if entry["format"] in _FORMAT_ALIASES.get(fmt, (fmt,)):
            return {"format": fmt, "status": "delivered", "path": entry["path"],
                    "geometry": entry.get("geometry"), "verified": entry.get("verified")}
    for entry in missing:
        if entry["format"] in _FORMAT_ALIASES.get(fmt, (fmt,)):
            return {"format": fmt, "status": "not_delivered", "reason": entry["reason"]}
    return {"format": fmt, "status": "not_delivered",
            "reason": "the exporter reported nothing for this format"}


def official_ledger(manifest):
    """Each officially named format, with the exporter's own reason for any gap."""
    rows = [_status_for(fmt, manifest["files"], manifest.get("not_delivered", []))
            for fmt in OFFICIAL_FORMATS]
    return {"formats": OFFICIAL_FORMATS, "delivered": sum(r["status"] == "delivered" for r in rows),
            "total": len(rows), "rows": rows}


def write_wgs84_positions(geodetic, path, *, max_rows=MAX_WGS84_ROWS):
    """A decimated lat/lon/height table, with the decimation stated in the header."""
    path = Path(path)
    total = len(geodetic)
    step = max(1, int(np.ceil(total / max_rows))) if max_rows else 1
    kept = geodetic[::step]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["# latitude_deg", "longitude_deg", "ellipsoidal_height_m",
                         "datum=WGS84", "height_not_orthometric=yes",
                         f"rows_written={len(kept)}", f"rows_total={total}",
                         f"decimation_every_nth_row={step}"])
        writer.writerow(["latitude_deg", "longitude_deg", "ellipsoidal_height_m"])
        for row in kept:
            writer.writerow([f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.4f}"])
    return {"path": path.name, "rows_written": int(len(kept)), "rows_total": int(total),
            "decimation_every_nth_row": int(step), "units": "degrees and ellipsoidal metres"}


def deliver(points, colors, alignment, output_dir, *, mesh_path=None, crs_wkt=None,
            cell_size_m=0.5, source_sha256=None, triangles=None, max_wgs84_rows=MAX_WGS84_ROWS):
    """Write the local and georeferenced product sets and the ledger between them.

    ``mesh_path`` is a meshed PLY from the run; when it is absent, or holds no faces,
    the products are point clouds and OBJ stays honestly undelivered. A CRS that
    cannot be derived from the scene's own GPS origin does not stop delivery: the
    local set is still written and the georeferenced set is reported as refused with
    the reason, rather than being faked in a made-up projection.
    """
    output_dir = Path(output_dir)
    cloud_count = int(len(points))
    mesh_note = None
    if mesh_path is not None and triangles is None:
        mesh_points, mesh_faces, mesh_colors = read_mesh_ply(mesh_path)
        if mesh_faces is not None and len(mesh_faces):
            # A mesher rebuilds the surface: its vertices are its own, and the faces
            # index into them. Export the mesh as the mesh, and keep the measured
            # cloud's count beside it so nobody reads 1.5 M points into a 90 k
            # triangle model and wonders which one the file holds.
            mesh_note = {"cloud_point_count": cloud_count,
                         "mesh_vertex_count": int(len(mesh_points)),
                         "mesh_face_count": int(len(mesh_faces)),
                         "basis": ("surface reconstructed from the fused cloud: its vertices "
                                   "are the mesher's isosurface, not individually measured "
                                   "points, and its colours are interpolated")}
            points, colors, triangles = mesh_points, mesh_colors, mesh_faces
    local_dir, geo_dir = output_dir / "enu", output_dir / "georeferenced"
    base = {"points": points, "colors": colors, "triangles": triangles,
            "source_sha256": source_sha256, "cell_size_m": cell_size_m}
    local = exporter.export_products(alignment=alignment, output_dir=local_dir, **base)
    result = {"schema_version": 1, "point_count": local["point_count"],
              "cloud_point_count": cloud_count,
              "meshed": triangles is not None, "mesh": mesh_note,
              "triangle_count": int(len(triangles)) if triangles is not None else 0,
              "local_enu": {"dir": "enu", "manifest": local,
                            "formats": official_ledger(local)},
              "georeferenced": None, "refusals": []}
    try:
        scene_crs = crs.crs_from_alignment(alignment) if crs_wkt is None else \
            {"wkt": crs_wkt, "name": "declared CRS", "epsg": None}
        enu = georef.transform_points(points, alignment)
        utm, geodetic, _ = crs.enu_to_crs(enu, alignment, crs=scene_crs)
        frame = {"type": "UTM", "units": "m", "epsg": scene_crs["epsg"],
                 "name": scene_crs["name"], "geodetic_crs": "EPSG:4979",
                 "altitude_datum": "ellipsoidal",
                 "origin": scene_crs["origin_geodetic"]}
        geo = exporter.export_products(points, colors, alignment, geo_dir,
                                       triangles=triangles, crs_wkt=scene_crs["wkt"],
                                       source_sha256=source_sha256, cell_size_m=cell_size_m,
                                       positions=utm, coordinate_frame=frame, crs=scene_crs)
        positions = write_wgs84_positions(geodetic, geo_dir / "positions_wgs84.csv",
                                          max_rows=max_wgs84_rows)
        geo["files"].append({"format": "csv", "path": positions["path"], "geometry": "points",
                             "verified": "written by this module, re-read by no one",
                             **positions})
        result["georeferenced"] = {"dir": "georeferenced", "crs": {k: scene_crs[k] for k in
                                                                   ("epsg", "name", "zone",
                                                                    "hemisphere", "validity")
                                                                   if k in scene_crs},
                                   "manifest": geo, "formats": official_ledger(geo)}
    except (ValueError, KeyError) as error:
        result["refusals"].append({"set": "georeferenced", "reason": str(error)})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, output_dir / "delivery_manifest.json")
    return result


def _write_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
