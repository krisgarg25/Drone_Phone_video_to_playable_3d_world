"""HTTP handlers for Phase 5: inspection, archaeology and the digital twin.

``/api/workspace/inspect/*`` - frames that saw a point, the defect register, crack
candidates, the inspection report, tilt, wire sag and clearance, sections (SVG/DXF), terrain
rasters (hillshade, LRM, slope), M3C2 change against another scan, the archive record.
``/api/workspace/twin/*`` - asset inventory with attribute cards, epochs, engine package.
Results carry viewer-frame overlays; nothing here writes into the reconstruction.
"""
import base64
import io
import math

import numpy as np

import inspect_geometry as geom
import inspection
import workspace_proposals as proposals

LRM_BANDS = [(-np.inf, -0.30, [30, 70, 200], "lrm_deep"), (-0.30, -0.08, [120, 160, 235], "lrm_low"),
             (0.08, 0.30, [240, 150, 120], "lrm_high"), (0.30, np.inf, [210, 40, 40], "lrm_raised")]


def _work(api, root, scene):
    return api.scene_paths(root, scene)[1]


def _call(api, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (inspection.InspectionError, geom.GeometryError, proposals.ProposalError) as error:
        raise api.Error(error.status, str(error))
    except ValueError as error:
        raise api.Error(422, str(error))


def _point(api, value, name="point"):
    if not (isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
        raise api.Error(400, f"{name} must be a finite [x, y, z]")
    return [float(v) for v in value]


def _num(api, data, key, default, lo, hi):
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise api.Error(400, f"{key} must be between {lo} and {hi}")
    return float(value)


def _registry(work):
    import scene_frames
    try:
        return scene_frames.load(work)
    except (OSError, ValueError):
        return {"status": "local_relative", "scale_status": "relative"}


def _points(api, work, colors=False):
    import facades
    try:
        return facades.scene_points(work, colors=colors)
    except (OSError, ValueError) as error:
        raise api.Error(409, str(error))


def _file(name, media, data):
    return {"filename": name, "media_type": media, "encoding": "base64", "content": base64.b64encode(data).decode("ascii")}


def _polyline_mesh(points, color, part, width=0.12):
    """A thin ribbon (both faces) through viewer points: sections, wires, tilt axes."""
    pts = np.asarray(points, float)
    positions, indices = [], []
    for p, q in zip(pts[:-1], pts[1:]):
        d = q - p
        if np.linalg.norm(d) < 1e-6:
            continue
        side = np.cross(d, [0, 1, 0])
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(d, [1, 0, 0])
        side = side / np.linalg.norm(side) * width / 2
        up = np.array([0, width / 2, 0]) if abs(d[1]) < np.linalg.norm(d) * 0.9 else side
        base = len(positions)
        positions += [(p - side).tolist(), (p + side).tolist(), (q + side).tolist(), (q - side).tolist(),
                      (p - up).tolist(), (p + up).tolist(), (q + up).tolist(), (q - up).tolist()]
        for o in (0, 4):
            indices += [[base + o, base + o + 1, base + o + 2], [base + o, base + o + 2, base + o + 3],
                        [base + o, base + o + 2, base + o + 1], [base + o, base + o + 3, base + o + 2]]
    return [proposals._mesh("overlay", positions, indices, color, 0.95, part=part)] if indices else []


def _with_urls(api, root, work, register):
    """Photo URLs on every register the API returns (GET and every write)."""
    for a in register.get("items", []):
        if a.get("photo"):
            a["photo"]["url"] = api.runtime_url(root, work / a["photo"]["file"])
    return register


def _where(work, registry, point):
    if registry.get("status") != "georeferenced":
        return None
    import scene_frames
    import survey_coords
    lat, lon, h = scene_frames.viewer_to_geodetic([point], registry)[0]
    return {"lat": round(float(lat), 7), "lon": round(float(lon), 7), "mgrs": survey_coords.to_mgrs(float(lat), float(lon))}


# ------------------------------------------------------------------ GET
def get(api, root, route, query):
    scene = query.get("scene", [""])[0]
    work = _work(api, root, scene)
    if route == "inspect/frames":
        try:
            point = [float(query.get(k, [""])[0]) for k in ("x", "y", "z")]
        except ValueError:
            raise api.Error(400, "Supply numeric x, y and z.")
        out = _call(api, inspection.frames_that_saw, work, point, ground=proposals.Ground(work))
        for f in out["frames"]:
            path = inspection.frame_path(work, f["name"])
            f["url"] = api.runtime_url(root, path) if path else None
        out["where"] = _where(work, _registry(work), point)
        return out
    if route == "inspect/annotations":
        return _with_urls(api, root, work, inspection.list_annotations(work))
    if route == "inspect/crack":
        item = next((a for a in inspection.list_annotations(work)["items"] if a["id"] == query.get("id", [""])[0]), None)
        if item is None or not item.get("photo"):
            raise api.Error(404, "That annotation has no photo to examine.")
        from PIL import Image
        image = Image.open(work / item["photo"]["file"]).convert("RGB")
        frame = item["frames"][0]
        gsd = frame["gsd_mm"] / max(item["photo"]["scale"][0], 1e-6)
        out = inspection.crack_candidates(image, gsd_mm=gsd)
        buf = io.BytesIO()
        Image.fromarray(out.pop("overlay")).save(buf, format="JPEG", quality=88)
        out["overlay_jpeg_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        out["gsd_mm"] = round(gsd, 3)
        return out
    if route == "inspect/report":
        project = api.read_json(root, work / "project.json", {}) or {}
        name = project.get("name", scene)
        data = inspection.report_pdf(work, title=f"{name} — inspection register", registry=_registry(work), scene_name=name)
        return _file("inspection-report.pdf", "application/pdf", data)
    if route == "inspect/terrain":
        return _terrain(api, work, query.get("layer", ["lrm"])[0])
    if route == "inspect/provenance":
        import provenance_record
        return provenance_record.build(work)
    if route == "twin/inventory":
        import twin
        out = twin.inventory(work, proposals.Ground(work))
        registry = _registry(work)
        ground = proposals.Ground(work)
        for a in out["assets"]:
            y = float(ground.sample_top([a["position"]])[0][0])
            a["viewer"] = [a["position"][0], y, a["position"][1]]
            where = _where(work, registry, a["viewer"])
            if where:
                a.update(where)
        return out
    if route == "twin/epochs":
        import twin
        return twin.epochs(work)
    raise api.Error(404, "Unknown workspace endpoint.")


def _terrain(api, work, layer):
    import plan_shadow
    ground = proposals.Ground(work)
    g = ground.grid
    top = g.get("top") if g.get("top") is not None else g["floor"]
    out = geom.terrain_rasters(g["floor"], top, np.asarray(g["supported"], bool), g["cell"])
    if layer == "ortho":
        import twin
        from PIL import Image
        points, rgb = _points(api, work, colors=True)
        colours = twin.cell_colors(g, points, rgb)
        rgba = np.dstack([colours, np.where(np.asarray(g["supported"], bool), 255, 0).astype(np.uint8)])
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        return {"layer": "ortho", "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "bounds": {"x0": g["ox"], "z0": g["oz"], "x1": g["ox"] + g["nx"] * g["cell"], "z1": g["oz"] + g["nz"] * g["cell"]},
                "cell_m": g["cell"], "lrm_range_m": out["lrm_range_m"], "overlay": [],
                "basis": f"top-down colour mosaic, {g['cell']:g} m cells, from the splats at the top surface - the "
                         "presentation layer's colours, not a radiometric orthomosaic; the survey lane's true "
                         "orthomosaic (ortho.tif) is the measured product"}
    if layer not in ("hillshade_dtm", "hillshade_dsm", "lrm", "slope_deg"):
        raise api.Error(400, "layer is ortho, hillshade_dtm, hillshade_dsm, lrm or slope_deg")
    grid = out[layer]
    lo, hi = out["lrm_range_m"] if layer == "lrm" else (0, 45) if layer == "slope_deg" else (0, 1)
    if layer == "lrm":
        lim = max(abs(lo), abs(hi), 0.05)
        png = geom.raster_png(grid, "RdBu_r", (-lim, lim))
    else:
        png = geom.raster_png(grid, "magma" if layer == "slope_deg" else "gray", (lo, hi))
    overlay = []
    if layer == "lrm":
        work_grid = {"cell": g["cell"], "ox": g["ox"], "oz": g["oz"]}
        budget = plan_shadow.MAX_OVERLAY_QUADS
        for a, b, colour, part in LRM_BANDS:
            mask = np.isfinite(grid) & (grid > a) & (grid <= b)
            mesh, used = plan_shadow._quads(mask, g["floor"], work_grid, colour, 0.55, part, budget)
            budget -= used
            if mesh:
                overlay.append(mesh)
    return {"layer": layer, "png_base64": base64.b64encode(png).decode("ascii"),
            "bounds": {"x0": g["ox"], "z0": g["oz"], "x1": g["ox"] + g["nx"] * g["cell"], "z1": g["oz"] + g["nz"] * g["cell"]},
            "cell_m": g["cell"], "lrm_range_m": out["lrm_range_m"], "overlay": overlay, "basis": out["basis"]}


# ------------------------------------------------------------------ POST
def post(api, root, route, data, server):
    scene = data.get("scene")
    work = _work(api, root, scene)
    with server.process_lock:
        if route == "inspect/annotation":
            register, entry = _call(api, inspection.add_annotation, work, data.get("item") or {},
                                    ground=proposals.Ground(work), revision=data.get("revision"))
            return {"register": _with_urls(api, root, work, register), "created": entry["id"]}
        if route == "inspect/annotation/update":
            return {"register": _with_urls(api, root, work, _call(api, inspection.update_annotation, work, str(data.get("id")),
                                                                    data.get("item") or {}, revision=data.get("revision")))}
        if route == "inspect/annotation/delete":
            return {"register": _with_urls(api, root, work, _call(api, inspection.delete_annotation, work, str(data.get("id")),
                                                                    revision=data.get("revision")))}
        if route == "inspect/tilt":
            base = _point(api, data.get("base"), "base")
            out = _call(api, geom.tilt, _points(api, work), base, radius_m=_num(api, data, "radius_m", 0.8, 0.1, 10))
            axis, c = np.array(out["axis"]), np.array(out["centre"])
            ends = [c - axis * out["height_m"] / 2, c + axis * out["height_m"] / 2]
            vertical = [ends[0], ends[0] + np.array([0, out["height_m"], 0])]
            return {"report": out, "overlay": _polyline_mesh(ends, [240, 90, 40], "tilt_axis", 0.08)
                    + _polyline_mesh(vertical, [80, 200, 255], "plumb_line", 0.05)}
        if route == "inspect/wire":
            a, b = _point(api, data.get("a"), "a"), _point(api, data.get("b"), "b")
            out = _call(api, geom.wire, _points(api, work), a, b, ground=proposals.Ground(work),
                        clearance_limit_m=_num(api, data, "clearance_limit_m", None, 0, 100))
            colour = [230, 60, 60] if out.get("clearance_ok") is False else [250, 210, 60]
            return {"report": out, "overlay": _polyline_mesh(out["line"], colour, "wire", 0.1)}
        if route == "inspect/section":
            a, b = _point(api, data.get("a"), "a"), _point(api, data.get("b"), "b")
            points, rgb = _points(api, work, colors=True)
            sec = _call(api, geom.section, points, a, b, ground=proposals.Ground(work), colors=rgb,
                        half_width_m=_num(api, data, "half_width_m", 0.25, 0.02, 5.0))
            title = str(data.get("title") or "Section")[:80]
            svg = geom.section_svg(sec, title=title)
            dxf = geom.section_dxf(sec)
            ok = np.isfinite(sec["outline_m"])
            report = {"length_m": sec["length_m"], "points": sec["points"], "azimuth_deg": sec["azimuth_deg"],
                      "y_range": sec["y_range"], "half_width_m": sec["half_width_m"],
                      "station_m": sec["station_m"][ok].round(3).tolist(), "outline_m": sec["outline_m"][ok].round(3).tolist(),
                      "ground_m": None if sec["ground_m"] is None else np.round(sec["ground_m"], 3).tolist(),
                      "observed_pct": None if sec["observed"] is None else round(100 * float(np.mean(sec["observed"])), 1)}
            ground = proposals.Ground(work)
            line = [[a[0], float(ground.sample_top([[a[0], a[2]]])[0][0]) + 0.3, a[2]],
                    [b[0], float(ground.sample_top([[b[0], b[2]]])[0][0]) + 0.3, b[2]]]
            return {"report": report, "svg": _file("section.svg", "image/svg+xml", svg.encode()),
                    "dxf": _file("section.dxf", "application/dxf", dxf.encode()),
                    "overlay": _polyline_mesh(line, [230, 60, 180], "section_line", 0.25)}
        if route == "inspect/m3c2":
            return _m3c2(api, root, work, data)
        if route == "inspect/provenance/verify":
            import provenance_record
            record = data.get("record")
            if not isinstance(record, dict):
                raise api.Error(400, "Supply the archive record to verify.")
            return {"changed": provenance_record.verify(work, record)}
        if route == "twin/attributes":
            import twin
            return {"attributes": _call(api, twin.set_attributes, work, data.get("id"), data.get("values"))}
        if route == "twin/package":
            return _package(api, root, work, data)
    raise api.Error(404, "Unknown workspace endpoint.")


def _m3c2(api, root, work, data):
    import ops_scene
    import survey_m3c2
    after = ops_scene.Scene(work)
    before_work = _work(api, root, data.get("before"))
    before = ops_scene.Scene(before_work)
    pa = _points(api, work)
    pb, basis = _call(api, ops_scene.viewer_points_between, after, before, _points(api, before_work))
    region = data.get("region")
    if region:
        from matplotlib.path import Path as MplPath
        ring = np.asarray(region, float)
        if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3:
            raise api.Error(400, "region is a polygon of [x, z] points")
        path = MplPath(ring)
        pa = pa[path.contains_points(pa[:, [0, 2]])]
        pb = pb[path.contains_points(pb[:, [0, 2]])]
    if len(pa) < 50 or len(pb) < 50:
        raise api.Error(422, "too few scan points in the region in one of the epochs")
    cap = 400_000
    rng = np.random.default_rng(0)
    if len(pa) > cap:
        pa = pa[rng.choice(len(pa), cap, replace=False)]
    if len(pb) > cap:
        pb = pb[rng.choice(len(pb), cap, replace=False)]
    import inspection as ins
    cams = ins._cameras(work)
    toward = np.mean([c["pos"] for c in cams], axis=0) if cams else pa.mean(0) + [0, 50, 0]
    # Defaults follow the cloud's own spacing: a cylinder must hold enough points of both
    # epochs (>= 5), and normals need a neighbourhood a few cylinders wide (Lague et al. 2013).
    from scipy.spatial import cKDTree
    sample = pb[rng.choice(len(pb), min(len(pb), 5000), replace=False)]
    spacing = float(np.median(cKDTree(pb).query(sample, k=2)[0][:, 1]))
    diameter = _num(api, data, "projection_diameter_m", None, 0.05, 5) or round(max(0.5, 3.5 * spacing), 2)
    normal_scale = _num(api, data, "normal_scale_m", None, 0.1, 10) or round(max(1.0, 2.0 * diameter), 2)
    out = survey_m3c2.m3c2(pb, pa, normal_scale_m=normal_scale,
                           projection_diameter_m=diameter,
                           max_depth_m=_num(api, data, "max_depth_m", 2.0, 0.1, 20),
                           registration_error_m=_num(api, data, "registration_error_m", 0.0, 0, 5),
                           core_spacing_m=_num(api, data, "core_spacing_m", 0.5, 0.1, 5), toward=toward)
    # One small square per significant core point, facing along its normal.
    positions = {1: [], -1: []}
    indices = {1: [], -1: []}
    sig = np.flatnonzero(out["significant"])
    if len(sig) > 15000:
        sig = sig[np.argsort(-np.abs(out["distance"][sig]))[:15000]]
    for i in sig:
        c, n = out["core"][i], out["normal"][i]
        side = np.cross(n, [0, 1, 0]) if abs(n[1]) < 0.9 else np.cross(n, [1, 0, 0])
        side /= np.linalg.norm(side)
        up = np.cross(n, side)
        k = 1 if out["distance"][i] > 0 else -1
        base = len(positions[k])
        h = 0.14
        c = c + n * 0.03
        positions[k] += [(c - side * h - up * h).tolist(), (c + side * h - up * h).tolist(),
                         (c + side * h + up * h).tolist(), (c - side * h + up * h).tolist()]
        indices[k] += [[base, base + 1, base + 2], [base, base + 2, base + 3], [base, base + 2, base + 1], [base, base + 3, base + 2]]
    overlay = []
    for k, colour, part in ((1, [214, 60, 50], "m3c2_towards"), (-1, [50, 110, 220], "m3c2_away")):
        if indices[k]:
            overlay.append(proposals._mesh("overlay", positions[k], indices[k], colour, 0.9, part=part))
    summary = dict(out["summary"], basis=basis, before=data.get("before"), point_spacing_m=round(spacing, 3),
                   note="distance along the local normal, positive = the surface moved towards the cameras")
    return {"report": summary, "overlay": overlay}


def _package(api, root, work, data):
    import twin
    ground = proposals.Ground(work)
    try:
        import facades
        points, rgb = facades.scene_points(work, colors=True)
    except (OSError, ValueError):
        points, rgb = None, None
    blob, manifest = _call(api, twin.package, work, ground.grid, _registry(work), points=points, rgb=rgb,
                           include_splats=bool(data.get("splats", True)))
    folder = work / "twin" / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "twin-package.zip"
    target.write_bytes(blob)
    return {"manifest": manifest, "bytes": len(blob), "url": api.runtime_url(root, target, True)}
