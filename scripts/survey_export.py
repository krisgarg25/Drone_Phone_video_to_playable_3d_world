"""Turn a measured cloud into the deliverable formats - without inventing anything.

A point cloud honestly fills PLY, LAS, XYZ, glTF and FBX. OBJ requires faces and
GeoTIFF requires a real CRS, so when either is absent the manifest records it as
not delivered with the reason, instead of manufacturing a surface or stamping a
local ENU frame with a CRS it does not have.
"""
import json
import os
import uuid
from pathlib import Path

import numpy as np

try:  # imported as scripts.survey_export by the tests
    from scripts import survey_formats as formats
    from scripts import survey_georef as georef
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_formats as formats
    import survey_georef as georef


def _write_ply(columns, path):
    from plyfile import PlyData, PlyElement
    names = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if "red" in columns:
        names += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    array = np.zeros(len(columns["x"]), dtype=names)
    for name, _ in names:
        array[name] = columns[name]
    PlyData([PlyElement.describe(array, "vertex")], text=False).write(path)


def dsm_grid(enu, cell):
    """Highest point per grid cell: a digital surface model, not bare earth."""
    xmin, ymin = float(enu[:, 0].min()), float(enu[:, 1].min())
    cols = int(np.floor((enu[:, 0].max() - xmin) / cell)) + 1
    rows = int(np.floor((enu[:, 1].max() - ymin) / cell)) + 1
    raster = np.full((rows, cols), np.nan, dtype=np.float32)
    col_index = np.clip(np.floor((enu[:, 0] - xmin) / cell).astype(np.int64), 0, cols - 1)
    row_index = np.clip(np.floor((enu[:, 1] - ymin) / cell).astype(np.int64), 0, rows - 1)
    order = np.lexsort((enu[:, 2], row_index, col_index))
    flat = (row_index * cols + col_index)[order]
    heights = enu[order, 2]
    keep = np.ones(len(flat), dtype=bool)
    keep[:-1] = flat[1:] != flat[:-1]          # the last value of each run is the maximum
    raster[flat[keep] // cols, flat[keep] % cols] = heights[keep]
    return (raster, (xmin, cell, 0.0, ymin + rows * cell, 0.0, -cell),
            int(np.isfinite(raster).sum()), int(raster.size))


def _write_json(value, path):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate(points, colors, triangles):
    """Return (points, rgb or None, faces or None) or raise before any file exists."""
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError("points must be a non-empty finite Nx3 array in reconstruction units")
    if colors is None:
        rgb = None
    else:
        rgb = np.asarray(colors)
        if rgb.shape != points.shape or not np.isfinite(rgb.astype(np.float64)).all():
            raise ValueError("colors must be a matching Nx3 array or None")
        if np.any(rgb < 0) or np.any(rgb > 255) or np.any(rgb != np.floor(rgb)):
            raise ValueError("colors must be integer RGB values in 0..255")
        rgb = rgb.astype(np.uint8)
    faces = None
    if triangles is not None:
        faces = np.asarray(triangles, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("triangles must be a non-empty Nx3 array of vertex indices")
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("triangles reference vertices that do not exist")
    return points, rgb, faces


def export_products(points, colors, alignment, output_dir, *, triangles=None, crs_wkt=None,
                    source_sha256=None, cell_size_m=0.5, scale_mm=1.0,
                    positions=None, coordinate_frame=None, crs=None):
    """Write the measured cloud into every container that needs no invention.

    ``positions`` pre-empts the ENU transform: pass an already-projected Nx3 array
    (see ``survey_deliver``) together with the ``coordinate_frame`` and ``crs`` that
    describe it, and the same writers fill a georeferenced product set. Leaving all
    three unset keeps the original local-ENU behaviour.
    """
    points, rgb, faces = validate(points, colors, triangles)
    if not 0 < float(scale_mm) <= 100.0:
        raise ValueError("scale_mm must be between 0.001 and 100 mm for a metric cloud")
    if float(cell_size_m) <= 0:
        raise ValueError("cell_size_m must be positive")

    if positions is None:
        enu = georef.transform_points(points, alignment)
    else:
        enu = np.asarray(positions, dtype=np.float64)
        if enu.shape != points.shape or not np.isfinite(enu).all():
            raise ValueError("positions must be a finite Nx3 array matching points")
    if coordinate_frame is None:
        coordinate_frame = alignment["coordinate_frame"]
    columns = {"x": enu[:, 0], "y": enu[:, 1], "z": enu[:, 2]}
    if rgb is not None:
        columns.update(red=rgb[:, 0], green=rgb[:, 1], blue=rgb[:, 2])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files, missing = [], []

    def record(fmt, name, **extra):
        path = output_dir / name
        files.append({"format": fmt, "path": name, "bytes": path.stat().st_size, **extra})

    ply = output_dir / "cloud.ply"
    _write_ply(columns, ply)
    record("ply", "cloud.ply", geometry="points", verified="round-trip only")
    # XYZ is a position-only container by convention; colour stays in the
    # formats that define a field for it.
    record("xyz", "cloud.xyz", geometry="points", rgb=False,
           verified=formats.write_xyz({k: columns[k] for k in "xyz"},
                                    output_dir / "cloud.xyz")["verified"])
    offsets = [float(np.median(columns[axis])) for axis in ("x", "y", "z")]
    scale = float(scale_mm) / 1000.0
    las = formats.write_las(columns, output_dir / "cloud.las", scale=[scale] * 3,
                            offsets=offsets, srs_wkt=crs_wkt)
    record("las", "cloud.las", geometry="points", verified=las["verified"],
           scale_mm=float(scale_mm), offset_x=offsets[0], offset_y=offsets[1],
           offset_z=offsets[2], crs_written=bool(crs_wkt))
    record("gltf", "cloud.gltf", geometry="mesh" if faces is not None else "points",
           vertex_colors=rgb is not None,
           verified=formats.write_gltf(enu, faces if faces is not None else rgb,
                                       out=output_dir / "cloud.gltf",
                                       mode="triangles" if faces is not None else
                                       "points")["verified"])
    record("fbx", "cloud.fbx", geometry="mesh" if faces is not None else "points",
           verified=formats.write_fbx(enu, output_dir / "cloud.fbx",
                                      triangles=faces)["verified"])
    if faces is None:
        missing.append({"format": "obj",
                        "reason": "no surface mesh exists; OBJ requires faces and inventing "
                                  "them would present inferred geometry as measured"})
    else:
        record("obj", "surface.obj", geometry="mesh", textured=False,
               vertex_colors=rgb is not None,
               verified=formats.write_obj(faces, enu, output_dir / "surface.obj")["verified"])

    caveats = []
    if faces is None:
        caveats.append("no surface mesh was produced: this is a measured point cloud, and no "
                       "faces were invented to fill the gap")
    else:
        caveats.append("the mesh carries no UV texture: "
                       + ("per-vertex colour from the fused cloud, not an image"
                          if rgb is not None else "geometry only"))
    if crs_wkt is None:
        missing.append({"format": "geotiff",
                        "reason": "no projected or geodetic CRS supplied: the cloud is in local "
                                  "ENU metres, and writing a GeoTIFF without a real CRS would be "
                                  "a false georeference"})
        caveats.append("local ENU metres; no projected CRS, so no GeoTIFF raster")
    else:
        raster, transform, filled, total = dsm_grid(enu, float(cell_size_m))
        geotiff = formats.write_geotiff(raster, output_dir / "dsm.tif", transform=transform,
                                        crs_wkt=crs_wkt, nodata=-9999.0)
        record("geotiff", "dsm.tif", geometry="dsm", verified=geotiff["verified"],
               cell_size_m=float(cell_size_m), cells_filled=filled, cells_total=total)
        caveats.append("the GeoTIFF is a digital surface model: it keeps rooftops and tree "
                       "canopy, so it is not bare-earth terrain")

    manifest = {"schema_version": 1, "point_count": int(len(enu)),
                "coordinate_frame": coordinate_frame, "files": files,
                "not_delivered": missing, "caveats": caveats,
                "claims_textured_mesh": False, "claims_surface_mesh": faces is not None,
                "externally_validated": False,
                "verification_note": "readers in scripts/survey_formats re-parse these files; "
                                     "no third-party library (laspy/GDAL/PDAL) was available to "
                                     "confirm spec compliance"}
    if crs is not None:
        manifest["crs"] = crs
    if crs_wkt is not None:
        manifest["crs_wkt"] = crs_wkt
    if source_sha256 is not None:
        manifest["source_sha256"] = source_sha256
    _write_json(manifest, output_dir / "export_manifest.json")
    return manifest
