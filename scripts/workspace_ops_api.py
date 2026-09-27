"""HTTP handlers for the Operations tab (``/api/workspace/ops/*``, Phase 4).

Disaster, construction and border analyses on a workspace scene, each returning a
report, coloured overlays for the viewer (viewer frame) and map-frame markers. The last
result of each kind is cached under ``work/<scene>/ops/last/`` so a field pack can be
built from exactly what the operator saw. Analyses never write to the reconstruction.
"""
import base64
import io
import json
import math
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ops_scene

MAX_DESIGN_BYTES = 1_500_000
DESIGN_EXT = (".xml", ".landxml", ".dxf", ".tif", ".tiff", ".glb")
GAIN, LOSS, BLIND, SEEN = [214, 60, 50], [50, 110, 220], [230, 70, 60], [60, 190, 110]


def _work(api, root, scene):
    return api.scene_paths(root, scene)[1]


def _scene(api, root, name):
    work = _work(api, root, name)
    try:
        return ops_scene.Scene(work)
    except (OSError, ValueError) as error:
        raise api.Error(409, f"{name}: {error}")


def _run(api, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, KeyError) as error:
        raise api.Error(422, str(error))


def _xz_list(api, value, name, minimum):
    if not isinstance(value, list) or len(value) < minimum or len(value) > 512:
        raise api.Error(400, f"{name} needs {minimum}-512 points.")
    out = []
    for p in value:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p[:2])):
            raise api.Error(400, f"{name} points must be [x, z] numbers.")
        out.append([float(p[0]), float(p[-1] if len(p) == 2 else p[2])])
    return out


def _num(api, data, key, default, lo, hi):
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise api.Error(400, f"{key} must be a number between {lo} and {hi}.")
    return float(value)


def _cache(scene, kind, report, grids=None):
    folder = scene.work / "ops" / "last"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (folder / f"{kind}.json").write_text(json.dumps({"report": report, "at": stamp}, default=_plain))
    if grids:
        arrays = {k: v for k, v in grids.items() if isinstance(v, np.ndarray)}
        arrays["transform"] = np.asarray(grids["transform"], float)
        np.savez_compressed(folder / f"{kind}.npz", **arrays)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _geo(scene, rows, key="centre"):
    for row in rows:
        where = scene.where(row[key])
        if where:
            row.update(where)
        row["viewer"] = scene.viewer_point(row[key])[0]
    return rows


def _pins(scene, xys, color, *, height=7.0, part="pin"):
    """One unpickable mesh of vertical markers (a mast and a head) at map points."""
    import workspace_proposals as proposals
    positions, indices = [], []
    for xy in xys:
        x, y, z = scene.viewer_point(xy)[0]
        for (x0, x1, y0, y1, z0, z1) in ((x - 0.25, x + 0.25, y, y + height, z - 0.25, z + 0.25),
                                         (x - 1.0, x + 1.0, y + height, y + height + 2.0, z - 1.0, z + 1.0)):
            base = len(positions)
            positions += [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
                          [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]
            for a, b, c, d in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)):
                indices += [[base + a, base + b, base + c], [base + a, base + c, base + d]]
    if not indices:
        return []
    return [proposals._mesh("overlay", positions, indices, color, 0.95, part=part)]


# ---------------------------------------------------------------- summary
def summary(api, root, scene_name):
    scene = _scene(api, root, scene_name)
    project = scene.project()
    epochs = []
    work_root = scene.work.parent
    for other in sorted(work_root.iterdir()):
        if other == scene.work or not (other / "viewer_assets" / "collision.json").is_file():
            continue
        try:
            reg = ops_scene.scene_frames.load(other)
            geo = reg.get("status") == "georeferenced"
        except (OSError, ValueError):
            geo = False
        try:
            info = json.loads((other / "project.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        compatible = geo == scene.georeferenced
        same_site = bool(project.get("site")) and info.get("site") == project.get("site")
        epochs.append({"scene": other.name, "name": info.get("name", other.name), "georeferenced": geo,
                       "captured_at": info.get("captured_at"), "compatible": compatible, "same_site": same_site,
                       "reason": None if compatible else "one epoch is georeferenced and the other is not"})
    epochs.sort(key=lambda e: (not e["same_site"], not e["compatible"], e["scene"]))
    import survey_detections
    import survey_firstmap
    rows, source = survey_detections.load(scene.work)
    designs = sorted(p.name for p in (scene.work / "ops" / "designs").glob("*")) \
        if (scene.work / "ops" / "designs").is_dir() else []
    first = survey_firstmap.latest(scene.work)
    last = {}
    folder = scene.work / "ops" / "last"
    if folder.is_dir():
        for path in folder.glob("*.json"):
            try:
                last[path.stem] = json.loads(path.read_text())["at"]
            except (OSError, ValueError, KeyError):
                pass
    a, b, c, d = scene.bounds
    return {"scene": scene_name, "name": project.get("name", scene_name), "captured_at": project.get("captured_at"),
            "frame": {"status": scene.registry.get("status"), "scale_status": scene.registry.get("scale_status")},
            "grid": {"cell_m": scene.cell, "size": list(scene.shape[::-1]),
                     "observed_pct": round(100.0 * float(scene.observed.mean()), 1),
                     "extent_m": [round(c - a, 1), round(d - b, 1)]},
            "epochs": epochs, "designs": designs, "road_labels": int(scene.labels("road").sum()),
            "detections": {"raw": len(rows), "source": source},
            "firstmap": None if first is None else {k: first.get(k) for k in
                                                   ("run", "window", "georeferenced", "points", "registered_images",
                                                    "elapsed_s", "units", "crs")},
            "last": last}


# ---------------------------------------------------------------- disaster
def change(api, root, data):
    import survey_change
    after = _scene(api, root, data.get("scene"))
    before = _scene(api, root, data.get("before"))
    cell = _num(api, data, "cell_m", max(1.0, after.cell), after.cell, 10.0)
    b_pts, basis = _run(api, ops_scene.epoch_points_in, after, before)
    a_pts = after.surface_points()
    out = _run(api, survey_change.detect_change, b_pts, a_pts, cell_m=cell, bounds=after.bounds,
               max_shift_m=_num(api, data, "max_shift_m", 3.0, 0.0, 20.0), min_region_m2=_num(api, data, "min_region_m2", 4.0, 0.25, 1e4))
    report = out["report"]
    report["epochs"] = {"before": data.get("before"), "after": data.get("scene"), "basis": basis}
    _geo(after, report["regions"], "centre_enu")
    status = out["status"]
    overlay = after.overlay([(status == survey_change.GAIN, GAIN, 0.6, "change_gain"),
                             (status == survey_change.LOSS, LOSS, 0.6, "change_loss")],
                            transform=out["transform"])
    _cache(after, "change", report, dict(dh=out["dh"], transform=out["transform"]))
    return {"report": report, "overlay": overlay}


def volume(api, root, data):
    import survey_change
    after = _scene(api, root, data.get("scene"))
    before = _scene(api, root, data.get("before"))
    ring = after.xz_to_map(_xz_list(api, data.get("polygon"), "polygon", 3))
    b_pts, basis = _run(api, ops_scene.epoch_points_in, after, before)
    result = _run(api, survey_change.region_volume, b_pts, after.surface_points(), ring,
                  cell_m=_num(api, data, "cell_m", max(0.5, after.cell), after.cell, 5.0))
    result["basis"] = basis
    result["outline"] = [after.viewer_point(p)[0] for p in ring]
    return {"report": result, "overlay": [m for m in [after.line_mesh(np.vstack([ring, ring[:1]]), [250, 200, 40], part="volume")] if m]}


def damage(api, root, data):
    import survey_damage
    scene = _scene(api, root, data.get("scene"))
    cell = _num(api, data, "cell_m", max(0.5, scene.cell), scene.cell, 2.0)
    footprints = heights = None
    source = None
    if data.get("before"):
        before = _scene(api, root, data["before"])
        b_pts, basis = _run(api, ops_scene.epoch_points_in, scene, before)
        footprints, heights = _run(api, survey_damage.footprints_from, b_pts, cell_m=cell)
        source = f"pre-event footprints and heights from {data['before']} ({basis})"
    out = _run(api, survey_damage.assess, scene.surface_points(), footprints=footprints,
               reference_heights=heights, cell_m=cell)
    out.pop("grids")
    if source:
        out["footprints"] = source
    _geo(scene, out["buildings"], "centre_enu")
    colours = {"intact": [60, 180, 90], "partial": [240, 150, 40], "collapsed": [220, 50, 50], "unknown": [150, 150, 150]}
    overlay = []
    for grade, colour in colours.items():
        pts = [b["centre_enu"] for b in out["buildings"] if b["grade"] == grade]
        overlay += _pins(scene, pts, colour, height=12.0, part=f"damage_{grade}")
    if footprints:
        for key, ring in footprints.items():
            grade = next((b["grade"] for b in out["buildings"] if b["id"] == key), "unknown")
            mesh = scene.line_mesh(np.vstack([ring, ring[:1]]), colours[grade], width=0.6, part=f"footprint_{grade}")
            if mesh:
                overlay.append(mesh)
    _cache(scene, "damage", out)
    return {"report": out, "overlay": overlay}


def detections(api, root, query):
    import survey_detections
    scene = _scene(api, root, query.get("scene", [""])[0])
    ground = __import__("workspace_proposals").Ground(scene.work)
    layer = _run(api, survey_detections.layer, scene.work, ground.sample_top, registry=scene.registry)
    for obj in layer["objects"]:
        xy = scene.viewer_to_map([obj["position"]])[0][:2].tolist()
        obj["centre"] = xy
        where = scene.where(xy)
        if where:
            obj.update(where)
    colours = {"person": [240, 80, 200], "vehicle": [80, 200, 240]}
    overlay = []
    for cls in sorted({o["class"] for o in layer["objects"]}):
        overlay += _pins(scene, [o["centre"] for o in layer["objects"] if o["class"] == cls],
                         colours.get(cls, [240, 220, 80]), height=4.0, part=f"detection_{cls}")
    _cache(scene, "detections", layer)
    return {"report": layer, "overlay": overlay}


def access(api, root, data):
    import survey_response
    scene = _scene(api, root, data.get("scene"))
    road = scene.labels("road")
    road_source = "road labels of this flight" if road.any() else "no road labels in this scene"
    before_dsm = None
    if data.get("before"):
        before = _scene(api, root, data["before"])
        if before.shape == scene.shape and before.transform == scene.transform:
            before_dsm = before.dsm
            # Debris hides the road it covers from the labeller, so the pre-event road
            # network is the one that says where the road was.
            road = road | before.labels("road")
            road_source = f"road labels of this flight and of {data['before']} (pre-event)"
    width = _num(api, data, "vehicle_width_m", 2.5, 1.0, 6.0)
    obstacle = _num(api, data, "max_obstacle_m", 0.5, 0.1, 2.0)
    blocked = survey_response.blocked_roads(road, scene.dtm, scene.dsm, scene.transform,
                                            before_dsm=before_dsm, max_obstacle_m=obstacle)
    mask = blocked.pop("mask")
    report = dict(blocked, road_source=road_source)
    _geo(scene, report["blocked"])
    overlay = scene.overlay([(road & ~mask, [90, 90, 100], 0.35, "road"), (mask, [220, 50, 50], 0.75, "blocked")])
    if data.get("start") and data.get("end"):
        s, e = (scene.xz_to_map(_xz_list(api, [data[k]], k, 1))[0] for k in ("start", "end"))
        try:
            route = survey_response.vehicle_route(scene.dtm, scene.dsm, scene.transform, s, e, road=road,
                                                  vehicle_width_m=width, max_obstacle_m=obstacle,
                                                  max_slope_pct=_num(api, data, "max_slope_pct", 25.0, 2.0, 100.0))
            report["route"] = route
            mesh = scene.line_mesh(route["path"], [60, 200, 255], width=1.2, part="vehicle_route")
            if mesh:
                overlay.append(mesh)
            route["viewer_path"] = [scene.viewer_point(p)[0] for p in route["path"]]
        except ValueError as error:
            report["route_error"] = str(error)
        overlay += _pins(scene, [s], [60, 200, 120], height=5.0, part="staging")
        overlay += _pins(scene, [e], [230, 80, 60], height=5.0, part="site")
    _cache(scene, "access", report)
    return {"report": report, "overlay": overlay}


def flood(api, root, data):
    import survey_response
    scene = _scene(api, root, data.get("scene"))
    seed = scene.xz_to_map(_xz_list(api, [data.get("seed")], "seed", 1))[0]
    footprints = None
    if data.get("before") or data.get("with_buildings", True):
        import survey_damage
        try:
            footprints, _ = survey_damage.footprints_from(scene.surface_points(), cell_m=max(0.5, scene.cell))
        except ValueError:
            footprints = None
    out = _run(api, survey_response.flood, scene.dtm, scene.transform, seed_xy=seed,
               level_m=_num(api, data, "level_m", None, -1e4, 1e4), rise_m=_num(api, data, "rise_m", None, 0.0, 50.0),
               footprints=footprints)
    report = out["report"]
    for b in report["buildings"]:
        ring = np.asarray(footprints[b["id"]])
        b["centre"] = ring.mean(0).round(2).tolist()
    _geo(scene, report["buildings"])
    depth = out["depth"]
    overlay = scene.overlay([(np.nan_to_num(depth) > 1.5, [20, 60, 170], 0.7, "flood_deep"),
                             ((np.nan_to_num(depth) > 0.5) & (np.nan_to_num(depth) <= 1.5), [50, 110, 210], 0.6, "flood_mid"),
                             ((np.nan_to_num(depth) > 0) & (np.nan_to_num(depth) <= 0.5), [110, 170, 240], 0.5, "flood_shallow")],
                            surface=np.where(np.isfinite(depth), scene.floor_raw + depth, scene.floor_raw))
    _cache(scene, "flood", report, dict(depth=depth, transform=scene.transform))
    return {"report": report, "overlay": overlay}


# ---------------------------------------------------------------- construction
def save_design(api, root, data):
    scene = _scene(api, root, data.get("scene"))
    name = str(data.get("filename", ""))
    stem = re.sub(r"[^A-Za-z0-9_.-]", "-", Path(name).name)[:80]
    if not stem.lower().endswith(DESIGN_EXT) or stem.startswith("."):
        raise api.Error(400, "Design surfaces are LandXML (.xml), DXF 3DFACE (.dxf) or GeoTIFF (.tif).")
    try:
        body = base64.b64decode(str(data.get("content", "")), validate=True)
    except ValueError:
        raise api.Error(400, "The design file content is not valid base64.")
    if not body or len(body) > MAX_DESIGN_BYTES:
        raise api.Error(413, "Design files are limited to 1.5 MB here; clip it to the site.")
    folder = scene.work / "ops" / "designs"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / stem
    target.write_bytes(body)
    import survey_design
    if stem.lower().endswith(".glb"):
        try:
            verts, faces = survey_design.read_glb_mesh(body)
        except Exception as error:  # noqa: BLE001 - any parse failure refuses the file
            target.unlink(missing_ok=True)
            raise api.Error(422, f"{stem}: not a readable GLB model ({error})")
        return {"design": stem, "kind": "model", "faces": int(len(faces)),
                "designs": sorted(p.name for p in folder.glob("*"))}
    try:
        design = survey_design.read_design(target)
    except (ValueError, OSError, Exception) as error:  # noqa: BLE001 - any parse failure refuses the file
        target.unlink(missing_ok=True)
        raise api.Error(422, f"{stem}: not a readable design surface ({error})")
    faces = len(design.get("faces", [])) if design["kind"] == "tin" else None
    return {"design": stem, "kind": design["kind"], "faces": faces,
            "designs": sorted(p.name for p in folder.glob("*"))}


def cutfill(api, root, data):
    import survey_design
    scene = _scene(api, root, data.get("scene"))
    name = Path(str(data.get("design", ""))).name
    path = scene.work / "ops" / "designs" / name
    if not name or not path.is_file():
        raise api.Error(404, "Upload the design surface first.")
    design = _run(api, survey_design.read_design, path)
    frame = data.get("design_frame", "scene")
    cell = _num(api, data, "cell_m", max(0.5, scene.cell), scene.cell, 5.0)
    points = scene.surface_points()
    if frame == "utm":
        utm, crs = _run(api, scene.to_utm, points)
        out = _run(api, survey_design.cut_fill, utm, design, cell_m=cell,
                   sigma_reg_m=_num(api, data, "sigma_reg_m", 0.0, 0.0, 10.0))
        # Back onto the scene grid for the overlay: sample the UTM raster at each cell.
        a, b, _, d, _, f = out["transform"]
        cells, _ = scene.to_utm(points)
        col = ((cells[:, 0] - a) / b).astype(int)
        row = ((cells[:, 1] - d) / f).astype(int)
        ok = (row >= 0) & (row < out["diff"].shape[0]) & (col >= 0) & (col < out["diff"].shape[1])
        diff = np.full(scene.shape, np.nan)
        rr, cc = np.nonzero(scene.observed)
        diff[rr[ok], cc[ok]] = out["diff"][row[ok], col[ok]]
        transform, grid = scene.transform, diff
        out["report"]["design_frame"] = crs["name"]
    elif frame == "scene":
        out = _run(api, survey_design.cut_fill, points, design, cell_m=scene.cell, bounds=scene.bounds,
                   sigma_reg_m=_num(api, data, "sigma_reg_m", 0.0, 0.0, 10.0))
        transform, grid = out["transform"], out["diff"].astype(float)
        out["report"]["design_frame"] = "scene map frame (x east, -z north)"
    else:
        raise api.Error(400, "design_frame is 'scene' or 'utm'.")
    report = out["report"]
    if data.get("baseline"):
        base = _scene(api, root, data["baseline"])
        b_pts, basis = _run(api, ops_scene.epoch_points_in, scene, base)
        if frame == "utm":
            b_pts, _ = scene.to_utm(b_pts)
        ring = design["vertices"][:, :2] if design["kind"] == "tin" else None
        if ring is not None:
            from scipy.spatial import ConvexHull
            hull = ring[ConvexHull(ring).vertices]
            report["zones"] = _run(api, survey_design.zone_progress, points if frame == "scene" else scene.to_utm(points)[0],
                                   design, {"design area": hull.tolist()}, baseline=b_pts, cell_m=cell)
            report["baseline"] = {"scene": data["baseline"], "basis": basis}
    tol = report["tolerance_m"]
    overlay = scene.overlay([(np.nan_to_num(grid) > tol, [214, 70, 60], 0.6, "cut"),
                             (np.nan_to_num(grid) < -tol, [60, 110, 214], 0.6, "fill"),
                             (np.isfinite(grid) & (np.abs(np.nan_to_num(grid)) <= tol), [80, 200, 110], 0.35, "on_grade")],
                            transform=transform)
    _cache(scene, "cutfill", report, dict(diff=np.asarray(grid, np.float32), transform=transform))
    return {"report": report, "overlay": overlay}


def model_deviation(api, root, data):
    """CON-06: the scan against a design model (GLB), as a deviation map on the scan."""
    import facades
    import scene_frames
    import survey_design
    scene = _scene(api, root, data.get("scene"))
    name = Path(str(data.get("design", ""))).name
    path = scene.work / "ops" / "designs" / name
    if not name.lower().endswith(".glb") or not path.is_file():
        raise api.Error(404, "Upload the design model (GLB) first.")
    verts, faces = _run(api, survey_design.read_glb_mesh, path.read_bytes())
    frame = data.get("model_frame", "scene")
    if frame == "enu":
        if not scene.georeferenced:
            raise api.Error(409, "The scene has no GPS fit, so a model in ENU coordinates cannot be placed.")
        enu = np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])          # glTF is Y-up
        verts = scene_frames.enu_to_viewer(enu, scene.registry)
    elif frame != "scene":
        raise api.Error(400, "model_frame is 'scene' or 'enu'.")
    tol = _num(api, data, "tolerance_m", 0.05, 0.001, 1.0)
    points = facades.scene_points(scene.work)
    out = _run(api, survey_design.model_deviation, points, verts, faces, tolerance_m=tol,
               max_distance_m=_num(api, data, "max_distance_m", 1.0, 0.05, 20.0))
    import workspace_proposals as proposals
    dev, pts, nrm = out["deviation"], out["points"], out["normals"]
    order = np.argsort(-np.abs(dev))[:15000]
    groups = {"proud": ([], [], [214, 60, 50]), "short": ([], [], [50, 110, 220]), "ok": ([], [], [70, 190, 90])}
    for i in order:
        key = "proud" if dev[i] > tol else "short" if dev[i] < -tol else "ok"
        pos, idx, _ = groups[key]
        n = nrm[i]
        side = np.cross(n, [0, 1, 0]) if abs(n[1]) < 0.9 else np.cross(n, [1, 0, 0])
        side /= np.linalg.norm(side)
        up = np.cross(n, side)
        c = pts[i] + n * 0.02
        h = 0.05
        b = len(pos)
        pos += [(c - side * h - up * h).tolist(), (c + side * h - up * h).tolist(), (c + side * h + up * h).tolist(), (c - side * h + up * h).tolist()]
        idx += [[b, b + 1, b + 2], [b, b + 2, b + 3], [b, b + 2, b + 1], [b, b + 3, b + 2]]
    overlay = [proposals._mesh("overlay", pos, idx, colour, 0.9, part=f"deviation_{key}")
               for key, (pos, idx, colour) in groups.items() if idx]
    report = dict(out["report"], design=name, model_frame=frame)
    _cache(scene, "deviation", report)
    return {"report": report, "overlay": overlay}


# ---------------------------------------------------------------- border
def corridor(api, root, data):
    import survey_corridor
    scene = _scene(api, root, data.get("scene"))
    line = scene.xz_to_map(_xz_list(api, data.get("line"), "line", 2))
    posts = []
    for i, p in enumerate(data.get("posts") or []):
        if not isinstance(p, dict):
            raise api.Error(400, "Each post is an object.")
        x, z = _xz_list(api, [p.get("at")], "post", 1)[0]
        posts.append({"id": str(p.get("id") or f"OP{i + 1}")[:24], "x": x, "y": -z,
                      "height_m": _num(api, p, "height_m", 3.0, 0.0, 100.0),
                      "range_m": _num(api, p, "range_m", 800.0, 5.0, 20000.0),
                      "bearing_deg": _num(api, p, "bearing_deg", None, 0.0, 360.0),
                      "fov_deg": _num(api, p, "fov_deg", 360.0, 5.0, 360.0)})
    if not posts:
        raise api.Error(400, "Place at least one observation post.")
    out = _run(api, survey_corridor.blind_spots, scene.dsm, scene.transform, line, posts,
               step_m=_num(api, data, "step_m", max(1.0, scene.cell * 2), 0.5, 50.0),
               target_height_m=_num(api, data, "target_height_m", 1.7, 0.0, 10.0))
    report = out["summary"]
    for b in report["blind_stretches"]:
        b["centre"] = b["centre_xy"]
    _geo(scene, report["blind_stretches"])
    prof = survey_corridor.profile(scene.dtm, scene.transform, line, step_m=out["summary"]["step_m"])
    report["profile"] = {"chainage_m": prof["chainage_m"].round(1).tolist(),
                         "z": [None if not np.isfinite(v) else round(float(v), 2) for v in prof["z"]],
                         "covered": out["covered"].tolist(), **prof["summary"]}
    s, xy = survey_corridor._densify(line, out["summary"]["step_m"])
    overlay = []
    covered = out["covered"]
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or covered[i] != covered[start]:
            seg = xy[max(0, start - 1 if start else 0):i]
            if len(seg) >= 2:
                mesh = scene.line_mesh(seg, SEEN if covered[start] else BLIND, width=1.4,
                                       part="line_seen" if covered[start] else "line_blind")
                if mesh:
                    overlay.append(mesh)
            start = i
    overlay += _pins(scene, [[p["x"], p["y"]] for p in posts], [240, 220, 80], height=6.0, part="post")
    _cache(scene, "corridor", report)
    return {"report": report, "overlay": overlay}


def tiles(api, root, data):
    import survey_corridor
    scene = _scene(api, root, data.get("scene"))
    tile = _num(api, data, "tile_m", 250.0, 20.0, 5000.0)
    pts = scene.surface_points()
    wkt, offset = None, (0.0, 0.0)
    if scene.georeferenced:
        pts, crs = scene.to_utm(pts)
        wkt = crs["wkt"]
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _run(api, survey_corridor.tile_products, pts, tmp, tile_m=tile, cell_m=max(0.5, scene.cell),
                        crs_wkt=wkt, offset=offset)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in Path(tmp).rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(tmp).as_posix())
    if buffer.tell() > 60_000_000:
        raise api.Error(413, "The tiles are larger than 60 MB; raise the tile size or use ops.py tiles.")
    return {"filename": "tiles.zip", "media_type": "application/zip", "encoding": "base64",
            "content": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "tiles": len([t for t in manifest["tiles"] if "files" in t]), "crs": bool(wkt)}


# ---------------------------------------------------------------- packs
def pack(api, root, data):
    import ops_report
    scene = _scene(api, root, data.get("scene"))
    kind = data.get("kind")
    if kind not in ops_report.KINDS:
        raise api.Error(400, "Unknown pack kind.")
    cached = scene.work / "ops" / "last" / f"{kind}.json"
    if not cached.is_file():
        raise api.Error(404, "Run this analysis first; the pack is built from its last result.")
    report = json.loads(cached.read_text())["report"]
    grids = None
    npz = cached.with_suffix(".npz")
    if npz.is_file():
        with np.load(npz) as arrays:
            grids = {k: arrays[k] for k in arrays.files if k != "transform"}
            grids["transform"] = tuple(float(v) for v in arrays["transform"])
    project = scene.project()
    captured = project.get("captured_at") or "capture time not recorded"
    titles = {"change": "Change since the previous flight", "cutfill": "Earthworks against design",
              "damage": "Building damage triage", "corridor": "Border coverage and blind stretches",
              "access": "Road access and vehicle route", "flood": "Flood extent and depth",
              "detections": "People and vehicles seen"}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / f"{scene.work.name}-{kind}"
        _run(api, ops_report.build_pack, folder, kind=kind, title=f"{project.get('name', scene.work.name)} — {titles[kind]}",
             captured=captured, report=report, grids=grids)
        body = folder.with_suffix(".zip").read_bytes()
    return {"filename": f"{kind}-pack.zip", "media_type": "application/zip",
            "encoding": "base64", "content": base64.b64encode(body).decode("ascii")}


def firstmap(api, root, query):
    import survey_firstmap
    scene_name = query.get("scene", [""])[0]
    work = _work(api, root, scene_name)
    record = survey_firstmap.latest(work)
    if record is None:
        return {"available": False,
                "reason": "No progressive run for this scene. Start a survey run with --progressive to "
                          "get a first map while it continues."}
    run_dir = Path(record.pop("run_dir"))
    png = run_dir / record["path"] / "preview.png"
    record["png_base64"] = base64.b64encode(png.read_bytes()).decode("ascii") if png.is_file() else None
    record["available"] = True
    return record


def get(api, root, route, query):
    if route == "ops/summary":
        return summary(api, root, query.get("scene", [""])[0])
    if route == "ops/firstmap":
        return firstmap(api, root, query)
    if route == "ops/detections":
        return detections(api, root, query)
    raise api.Error(404, "Unknown workspace endpoint.")


POST = {"ops/change": change, "ops/volume": volume, "ops/damage": damage, "ops/access": access,
        "ops/flood": flood, "ops/design": save_design, "ops/cutfill": cutfill, "ops/corridor": corridor,
        "ops/tiles": tiles, "ops/pack": pack, "ops/deviation": model_deviation}


def post(api, root, route, data, server):
    handler = POST.get(route)
    if handler is None:
        raise api.Error(404, "Unknown workspace endpoint.")
    with server.process_lock:
        return handler(api, root, data)
