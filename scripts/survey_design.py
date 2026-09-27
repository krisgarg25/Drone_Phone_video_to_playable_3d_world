"""Design surfaces and cut/fill against them (CON-02).

Earthworks are paid against a *design*: the engineer's finished-grade surface. This
module reads the three forms a design arrives in and compares an as-built cloud to it:

* GeoTIFF DEM (float32, north-up) through ``survey_formats.read_geotiff``;
* LandXML ``<Surface>`` TIN (``<Pnts><P id>N E Z</P>`` + ``<Faces><F>a b c</F>``;
  LandXML writes northing first, which is the classic swap and is handled here);
* DXF ``3DFACE`` entities (ASCII DXF, group codes 10/20/30 .. 13/23/33).

TINs are rasterised by barycentric interpolation over their own triangles, never by
re-triangulating the vertices (that would silently change the design's breaklines).

Coordinates: the design and the cloud must be in the same planar frame. Designs usually
come in a projected CRS (UTM or a local grid); pass ``offset`` = (east, north, up) to
subtract, e.g. the ENU origin's UTM coordinates, or run on the UTM products.

``cut_fill`` reports cut (as-built above design: material still to remove), fill
(below design: material still to place), both with an uncertainty from the as-built
cell error, only over cells where the design exists *and* the as-built was observed.
"""
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _tin(vertices, faces, source):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"{source}: a TIN needs at least three 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{source}: no triangles found")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{source}: a face refers to a vertex that does not exist")
    return dict(kind="tin", vertices=vertices, faces=faces, source=source)


def read_landxml(path):
    """First ``<Surface>`` of a LandXML file as a TIN (x = east, y = north)."""
    path = Path(path)
    root = ET.parse(path).getroot()
    strip = lambda tag: tag.rsplit("}", 1)[-1]
    surface = next((e for e in root.iter() if strip(e.tag) == "Surface"), None)
    if surface is None:
        raise ValueError(f"{path.name}: no <Surface> element")
    ids, rows = {}, []
    for element in surface.iter():
        if strip(element.tag) == "P":
            values = [float(v) for v in (element.text or "").split()]
            if len(values) != 3:
                raise ValueError(f"{path.name}: point {element.get('id')} is not 'N E Z'")
            north, east, z = values
            ids[element.get("id")] = len(rows)
            rows.append((east, north, z))
    faces = []
    for element in surface.iter():
        if strip(element.tag) == "F":
            if element.get("i") == "1":            # LandXML marks invisible (hole) faces
                continue
            refs = (element.text or "").split()
            try:
                faces.append([ids[r] for r in refs[:3]])
            except KeyError as missing:
                raise ValueError(f"{path.name}: face uses unknown point {missing}") from None
    return _tin(rows, faces, path.name)


def read_dxf_3dfaces(path):
    """Every 3DFACE of an ASCII DXF as TIN triangles (a quad becomes two)."""
    path = Path(path)
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    vertices, faces, index = [], [], 0
    while index < len(pairs):
        code, value = pairs[index]
        if code == "0" and value == "3DFACE":
            corner = {}
            index += 1
            while index < len(pairs) and pairs[index][0] != "0":
                c, v = pairs[index]
                if re.fullmatch(r"1[0-3]|2[0-3]|3[0-3]", c):
                    corner[c] = float(v)
                index += 1
            try:
                pts = [(corner[f"1{k}"], corner[f"2{k}"], corner[f"3{k}"]) for k in range(4)]
            except KeyError:
                raise ValueError(f"{path.name}: a 3DFACE is missing a corner coordinate") from None
            base = len(vertices)
            vertices.extend(pts)
            faces.append([base, base + 1, base + 2])
            if not np.allclose(pts[2], pts[3]):
                faces.append([base, base + 2, base + 3])
            continue
        index += 1
    return _tin(vertices, faces, path.name)


def read_design(path):
    """Dispatch on the suffix: .tif/.tiff, .xml/.landxml, .dxf."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import survey_formats
        data = survey_formats.read_geotiff(path)
        return dict(kind="raster", z=np.asarray(data["raster"], dtype=np.float64),
                    transform=tuple(data["transform"]), nodata=data.get("nodata"),
                    source=path.name)
    if suffix in (".xml", ".landxml"):
        return read_landxml(path)
    if suffix == ".dxf":
        return read_dxf_3dfaces(path)
    raise ValueError(f"{path.name}: design surfaces are read from GeoTIFF, LandXML or DXF")


def rasterise_tin(tin, *, transform, shape, offset=(0.0, 0.0, 0.0)):
    """Design height at every cell centre of a north-up grid; NaN outside the TIN."""
    a, b, _, d, _, f = transform
    rows, cols = shape
    v = tin["vertices"] - np.asarray(offset, dtype=np.float64)
    out = np.full(shape, np.nan)
    for i0, i1, i2 in tin["faces"]:
        p0, p1, p2 = v[i0], v[i1], v[i2]
        xs, ys = (p0[0], p1[0], p2[0]), (p0[1], p1[1], p2[1])
        c0 = max(0, int(math.floor((min(xs) - a) / b)))
        c1 = min(cols - 1, int(math.ceil((max(xs) - a) / b)))
        r0 = max(0, int(math.floor((max(ys) - d) / f)))
        r1 = min(rows - 1, int(math.ceil((min(ys) - d) / f)))
        if c1 < c0 or r1 < r0:
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        x = a + (cc + 0.5) * b
        y = d + (rr + 0.5) * f
        det = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(det) < 1e-12:
            continue
        w0 = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / det
        w1 = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        out[rr[inside], cc[inside]] = z[inside]
    return out


def resample_raster(design, *, transform, shape, offset=(0.0, 0.0, 0.0)):
    """Bilinear sample of a design DEM at every cell centre of the target grid."""
    from scipy import ndimage
    a, b, _, d, _, f = transform
    da, db, _, dd, _, df = design["transform"]
    z = np.array(design["z"], dtype=np.float64)
    if design.get("nodata") is not None:
        z[z == design["nodata"]] = np.nan
    rows, cols = shape
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    x = a + (cc + 0.5) * b + offset[0]
    y = d + (rr + 0.5) * f + offset[1]
    src_c = (x - da) / db - 0.5
    src_r = (y - dd) / df - 0.5
    # Inside the raster's footprint but outside its outermost cell centres, clamp to the
    # edge cell: the DEM covers that half cell, bilinear just has no second neighbour.
    height, width = z.shape
    within = (src_c >= -0.5) & (src_c <= width - 0.5) & (src_r >= -0.5) & (src_r <= height - 0.5)
    src_c = np.where(within, np.clip(src_c, 0, width - 1), src_c)
    src_r = np.where(within, np.clip(src_r, 0, height - 1), src_r)
    valid = np.isfinite(z)
    sampled = ndimage.map_coordinates(np.where(valid, z, 0.0), [src_r, src_c], order=1,
                                      mode="constant", cval=np.nan)
    weight = ndimage.map_coordinates(valid.astype(float), [src_r, src_c], order=1,
                                     mode="constant", cval=0.0)
    return np.where(weight > 0.999, sampled - offset[2], np.nan)


def design_grid(design, *, transform, shape, offset=(0.0, 0.0, 0.0)):
    if design["kind"] == "tin":
        return rasterise_tin(design, transform=transform, shape=shape, offset=offset)
    return resample_raster(design, transform=transform, shape=shape, offset=offset)


def cut_fill(points, design, *, cell_m=0.5, offset=(0.0, 0.0, 0.0), sigma_reg_m=0.0,
             tolerance_m=0.03, bounds=None):
    """As-built ENU points against a design. Returns grids and a JSON-safe report.

    ``sigma_reg_m`` is the as-built vertical registration error (e.g. the checkpoint LE
    or the M3 sigma); it is fully correlated across cells, so it adds area x sigma to the
    volume uncertainty rather than shrinking with the square root of the cell count.
    A cell is "on grade" when its difference is inside the larger of ``tolerance_m``
    (the contract's on-grade tolerance, 30 mm by default) and its own LoD95, which uses
    the pooled point noise over the cell's sample count (a one-point cell has no spread
    of its own, so its own std would claim a zero-width tolerance).
    """
    import survey_change
    cloud = survey_change._cloud(points, "points")
    if bounds is None:
        lo, hi = cloud[:, :2].min(0), cloud[:, :2].max(0)
        bounds = (lo[0], lo[1], hi[0] + 1e-9, hi[1] + 1e-9)
    grid = survey_change.grid_surface(cloud, cell_m=cell_m, bounds=bounds)
    target = design_grid(design, transform=grid["transform"], shape=grid["z"].shape,
                         offset=offset)
    diff = grid["z"] - target
    use = np.isfinite(diff)
    if use.sum() == 0:
        raise ValueError("the design and the as-built do not overlap; check the offset and "
                         "that both are in the same planar frame")
    with np.errstate(invalid="ignore", divide="ignore"):
        rich = use & (grid["count"] >= 3)
        pooled = float(np.median(grid["std"][rich])) if rich.any() else 0.0
        cell_sigma = pooled / np.sqrt(np.maximum(grid["count"], 1))
    tol = np.maximum(float(tolerance_m), 1.96 * np.hypot(cell_sigma, sigma_reg_m))
    area = cell_m * cell_m
    cut = use & (diff > tol)
    fill = use & (diff < -tol)
    cut_v = float(diff[cut].sum() * area)
    fill_v = float(-diff[fill].sum() * area)
    random = lambda m: float(area * math.sqrt(float(np.sum(cell_sigma[m] ** 2))))
    systematic = lambda m: float(m.sum() * area * sigma_reg_m)
    design_cells = np.isfinite(target)
    report = {
        "method": "per-cell mean as-built minus design, cells with both",
        "design": design.get("source"), "cell_m": cell_m, "tolerance_m": float(tolerance_m),
        "point_noise_m": round(pooled, 4),
        "cut_m3": round(cut_v, 2),
        "cut_sigma_m3": round(math.hypot(random(cut), systematic(cut)), 2),
        "fill_m3": round(fill_v, 2),
        "fill_sigma_m3": round(math.hypot(random(fill), systematic(fill)), 2),
        "net_m3": round(cut_v - fill_v, 2),
        "area_m2": {"design": round(float(design_cells.sum() * area), 1),
                    "compared": round(float(use.sum() * area), 1),
                    "cut": round(float(cut.sum() * area), 1),
                    "fill": round(float(fill.sum() * area), 1),
                    "on_grade": round(float((use & ~cut & ~fill).sum() * area), 1)},
        "design_observed_fraction": round(float(use.sum() / max(design_cells.sum(), 1)), 4),
        "notes": ["Cut = as-built above design (still to remove); fill = below (still to place).",
                  "Design cells the flight did not observe are not counted as on grade."]
        + ([] if sigma_reg_m else ["The as-built's vertical error against the design's datum is not "
                                   "included (none was given); take it from checkpoints before paying "
                                   "against these volumes."]),
    }
    return dict(diff=diff.astype(np.float32), design=target.astype(np.float32),
                transform=grid["transform"], report=report)


def zone_progress(points, design, zones, *, baseline=None, cell_m=0.5, offset=(0.0, 0.0, 0.0)):
    """Earthwork progress % per named zone (CON-05): moved / (moved + remaining).

    ``baseline`` is the pre-works cloud; without it progress cannot be stated and only the
    remaining cut/fill is reported.
    """
    from matplotlib.path import Path as MplPath
    now = cut_fill(points, design, cell_m=cell_m, offset=offset)
    base = cut_fill(baseline, design, cell_m=cell_m, offset=offset) if baseline is not None else None
    a, b, _, d, _, f = now["transform"]
    rows, cols = now["diff"].shape
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    centres = np.column_stack([a + (cc.ravel() + 0.5) * b, d + (rr.ravel() + 0.5) * f])
    area = cell_m * cell_m
    out = []
    for name, ring in zones.items():
        inside = MplPath(np.asarray(ring, float)[:, :2]).contains_points(centres).reshape(rows, cols)
        remaining = float(np.nansum(np.abs(now["diff"][inside])) * area)
        entry = {"zone": name, "remaining_m3": round(remaining, 2)}
        if base is not None:
            ba, bb, _, bd, _, bf = base["transform"]
            br, bc = base["diff"].shape
            gc, gr = np.meshgrid(np.arange(bc), np.arange(br))
            bcent = np.column_stack([ba + (gc.ravel() + 0.5) * bb, bd + (gr.ravel() + 0.5) * bf])
            binside = MplPath(np.asarray(ring, float)[:, :2]).contains_points(bcent).reshape(br, bc)
            start = float(np.nansum(np.abs(base["diff"][binside])) * area)
            entry["start_m3"] = round(start, 2)
            entry["progress_pct"] = None if start <= 0 else round(
                100.0 * max(0.0, min(1.0, 1.0 - remaining / start)), 1)
        out.append(entry)
    return out


# ------------------------------------------------------------------ design model deviation (CON-06)
def read_glb_mesh(data):
    """All triangles of a GLB in its world frame (node hierarchy applied): (vertices, faces)."""
    import plan_glb
    import workspace_place
    document, binary = plan_glb.read_glb(data)
    nodes = document.get("nodes", [])
    verts, faces = [], []
    offset = 0

    def walk(index, parent):
        nonlocal offset
        node = nodes[index]
        world = parent @ workspace_place._mat4_local(node)
        if "mesh" in node:
            for prim in document["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = plan_glb.accessor_array(document, binary, prim["attributes"]["POSITION"]).astype(np.float64)
                idx = (plan_glb.accessor_array(document, binary, prim["indices"]).astype(np.int64)
                       if "indices" in prim else np.arange(len(pos)))
                homo = np.column_stack([pos, np.ones(len(pos))]) @ world.T
                verts.append(homo[:, :3])
                faces.append(idx.reshape(-1, 3) + offset)
                offset += len(pos)
        for child in node.get("children", []):
            walk(child, world)

    scene = document.get("scenes", [{"nodes": list(range(len(nodes)))}])[document.get("scene", 0)]
    for root in scene.get("nodes", []):
        walk(root, np.eye(4))
    if not verts:
        raise ValueError("the model has no triangle meshes")
    return np.vstack(verts), np.vstack(faces)


def sample_surface(vertices, faces, *, spacing_m=0.05, max_samples=600_000, seed=0):
    """Area-weighted points on the mesh surface with each point's triangle normal."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 1e-12
    a, b, c, cross, area = a[keep], b[keep], c[keep], cross[keep], area[keep]
    n = int(min(max_samples, max(1000, area.sum() / (spacing_m ** 2))))
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(area), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    normals = cross[tri] / np.linalg.norm(cross[tri], axis=1, keepdims=True)
    return pts, normals, float(area.sum())


def model_deviation(points, vertices, faces, *, tolerance_m=0.05, margin_m=0.5, max_distance_m=1.0, spacing_m=0.05):
    """As-built scan points against a design model: signed distance along the model's normal.

    Positive = the built surface stands proud of the design (outside it); negative = short
    of it. Only scan points within ``margin_m`` of the model's bounding box are compared, and
    points farther than ``max_distance_m`` from any design surface are reported as
    "not part of this element" rather than as huge deviations. The model is sampled every
    ``spacing_m``, which bounds the distance error at about half that.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    lo, hi = vertices.min(0) - margin_m, vertices.max(0) + margin_m
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    near = points[inside]
    if len(near) < 20:
        raise ValueError("fewer than 20 scan points lie near the model; check that it is placed in the scan's frame")
    samples, normals, area = sample_surface(vertices, faces, spacing_m=spacing_m)
    dist, idx = cKDTree(samples).query(near)
    signed = np.einsum("ij,ij->i", near - samples[idx], normals[idx])
    related = dist <= max_distance_m
    s = signed[related]
    report = {"model_area_m2": round(area, 2), "scan_points_compared": int(related.sum()),
              "scan_points_unrelated": int((~related).sum()), "tolerance_m": tolerance_m,
              "within_tolerance_pct": round(100.0 * float(np.mean(np.abs(s) <= tolerance_m)), 1) if len(s) else None,
              "mean_m": round(float(np.mean(s)), 4) if len(s) else None,
              "rms_m": round(float(np.sqrt(np.mean(s ** 2))), 4) if len(s) else None,
              "p95_abs_m": round(float(np.percentile(np.abs(s), 95)), 4) if len(s) else None,
              "proud_pct": round(100.0 * float(np.mean(s > tolerance_m)), 1) if len(s) else None,
              "short_pct": round(100.0 * float(np.mean(s < -tolerance_m)), 1) if len(s) else None,
              "sampling_error_m": round(spacing_m / 2, 3),
              "basis": "signed distance from each scan point to the nearest design-surface sample, along that "
                       "surface's normal; unsigned where a surface is double-sided in the model"}
    return {"points": near[related], "deviation": s, "normals": normals[idx][related], "report": report}
