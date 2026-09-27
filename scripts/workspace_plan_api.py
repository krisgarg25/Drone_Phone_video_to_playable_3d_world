"""HTTP handlers for the planning editor (proposal layer) and click coordinates.

Dispatched from ``workspace_api.handle`` under ``/api/workspace/plan/*`` and
``/api/workspace/coords``. Every write returns the proposal *and* its fresh evaluation
(derived meshes, rule verdicts, metrics), so the editor never renders geometry the
backend did not compute. The workspace's loopback/Origin write guard applies unchanged.
"""
import base64
import math

import workspace_proposals as proposals
import scene_frames

EXPORT_FORMATS = ("geojson", "cityjson", "dxf", "3dtiles")


def _work(api, root, scene):
    return api.scene_paths(root, scene)[1]


def _context(work):
    ground = proposals.Ground(work)
    try:
        registry = scene_frames.load(work)
    except (OSError, ValueError):
        registry = {"status": "local_relative", "scale_status": "relative"}
    return ground, registry, proposals.existing_inventory(work, ground)


def _evaluate(work, proposal):
    """(evaluation, ground, registry, existing): one place that knows how to evaluate."""
    import mission
    import urban_analysis
    ground, registry, existing = _context(work)
    site = urban_analysis.surface_layers(work, ground, existing)
    site.pop("_grids", None)
    evaluation = proposals.evaluate(proposal, ground, existing,
                                    scale_status=registry.get("scale_status", "relative"),
                                    north=mission.north_vector(registry), site=site)
    return evaluation, ground, registry, existing


def _state(work, proposal):
    evaluation, ground, registry, existing = _evaluate(work, proposal)
    return {"proposal": proposal, "evaluation": evaluation,
            "frame": {k: registry.get(k) for k in ("status", "scale_status", "scale_source")},
            "existing": {"status": existing.get("status"), "basis": existing.get("basis"),
                         "buildings": existing.get("buildings", []),
                         "trees": len(existing.get("trees", []))}}


def _call(api, fn, *args, **kwargs):
    import plan_exports
    import plan_shadow
    try:
        return fn(*args, **kwargs)
    except (proposals.ProposalError, plan_exports.ExportError, plan_shadow.ShadowError) as error:
        raise api.Error(error.status, str(error))


def _number(api, query, key, default=None):
    raw = query.get(key, [None])[0]
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError:
        raise api.Error(400, f"{key} must be a number.")
    if not math.isfinite(value):
        raise api.Error(400, f"{key} must be finite.")
    return value


def _export(api, work, query):
    import plan_exports
    proposal = _call(api, proposals.load_proposal, work, query.get("id", [""])[0])
    evaluation, ground, registry, existing = _evaluate(work, proposal)
    fmt = query.get("format", ["geojson"])[0]
    if fmt not in EXPORT_FORMATS:
        raise api.Error(400, "format must be one of " + ", ".join(EXPORT_FORMATS))
    if fmt == "geojson" and "format" not in query:
        return proposals.export_geojson(work, proposal, evaluation, registry)
    stem = plan_exports._slug(proposal["name"])
    if fmt == "geojson":
        import json
        text = json.dumps(proposals.export_geojson(work, proposal, evaluation, registry), indent=1)
        return {"filename": f"{stem}.geojson", "media_type": "application/geo+json", "encoding": "utf-8", "content": text}
    if fmt == "cityjson":
        import json
        doc = _call(api, plan_exports.cityjson, proposal, evaluation, registry, ground, existing)
        return {"filename": f"{stem}.city.json", "media_type": "application/city+json", "encoding": "utf-8",
                "content": json.dumps(doc, separators=(",", ":"))}
    if fmt == "dxf":
        text = _call(api, plan_exports.dxf, proposal, evaluation, registry, ground)
        return {"filename": f"{stem}.dxf", "media_type": "image/vnd.dxf", "encoding": "utf-8", "content": text}
    data, _ = _call(api, plan_exports.tiles3d, proposal, evaluation, registry)
    return {"filename": f"{stem}-3dtiles.zip", "media_type": "application/zip", "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii")}


def _shadow(api, work, query):
    import plan_shadow
    proposal = _call(api, proposals.load_proposal, work, query.get("id", [""])[0])
    evaluation, ground, registry, existing = _evaluate(work, proposal)
    crowns_existing = [t["footprint"] for t in existing.get("trees", []) if t.get("footprint")]
    crowns_proposed = [evaluation["features"][f["id"]]["footprint"] for f in proposal["features"]
                       if f["type"] == "object" and str(f["params"].get("item", "")).startswith("tree")
                       and not f.get("hidden") and f["id"] in evaluation["features"]]
    return _call(api, plan_shadow.study, ground.grid, evaluation,
                 registry if registry.get("status") == "georeferenced" else None,
                 date=query.get("date", [""])[0], time=query.get("time", ["12:00"])[0],
                 utc_offset=_number(api, query, "utc_offset", 0.0),
                 mode=query.get("mode", ["instant"])[0],
                 start_h=_number(api, query, "start", 9.0), end_h=_number(api, query, "end", 15.0),
                 step_min=int(_number(api, query, "step", 30)),
                 lat=_number(api, query, "lat"), lon=_number(api, query, "lon"),
                 north_deg=_number(api, query, "north_deg", 0.0),
                 crowns_existing=crowns_existing, crowns_proposed=crowns_proposed)


def get(api, root, route, query):
    scene = query.get("scene", [""])[0]
    work = _work(api, root, scene)
    if route == "plan/proposals":
        kind = query.get("kind", [None])[0]
        if kind not in (None, "plan", "mission"):
            raise api.Error(400, "kind must be plan or mission")
        index = _call(api, proposals.list_proposals, work, kind)
        return {"index": index, "catalogue": {k: {"size": v[0], "color": v[1], "detailed": k in proposals.OBJECT_MODELS}
                                              for k, v in proposals.URBAN_OBJECTS.items()}}
    if route == "plan/proposal":
        proposal = _call(api, proposals.load_proposal, work, query.get("id", [""])[0])
        return _state(work, proposal)
    if route == "plan/export":
        return _export(api, work, query)
    if route == "plan/shadow":
        return _shadow(api, work, query)
    if route == "plan/facades":
        import facades
        ground, _, existing = _context(work)
        buildings = existing.get("buildings", [])
        if not buildings:
            return {"buildings": [], "overlay": [], "basis": existing.get("basis", ""),
                    "notes": ["No existing buildings in this scan's labels, so there are no facades to check."]}
        try:
            points = facades.scene_points(work)
        except (OSError, ValueError) as error:
            raise api.Error(409, str(error))
        return facades.completeness(buildings, points)
    if route == "plan/site":
        import urban_analysis
        ground, registry, existing = _context(work)
        try:
            return urban_analysis.study(work, ground, existing, registry,
                                        lat=_number(api, query, "lat"), lon=_number(api, query, "lon"),
                                        with_solar=query.get("solar", ["1"])[0] != "0")
        except urban_analysis.UrbanError as error:
            raise api.Error(error.status, str(error))
    if route == "plan/road-width":
        import urban_analysis
        try:
            a = [float(v) for v in query.get("a", [""])[0].split(",")]
            b = [float(v) for v in query.get("b", [""])[0].split(",")]
        except ValueError:
            raise api.Error(400, "Supply a and b as x,y,z.")
        if len(a) != 3 or len(b) != 3 or not all(api.finite(v) for v in a + b):
            raise api.Error(400, "Supply a and b as finite x,y,z.")
        try:
            return urban_analysis.road_width(work, a, b)
        except urban_analysis.UrbanError as error:
            raise api.Error(error.status, str(error))
    if route == "coords":
        try:
            point = [float(query.get(axis, [""])[0]) for axis in ("x", "y", "z")]
        except ValueError:
            raise api.Error(400, "Supply numeric x, y and z.")
        if not all(api.finite(v) for v in point):
            raise api.Error(400, "Supply finite x, y and z.")
        try:
            registry = scene_frames.load(work)
        except (OSError, ValueError) as error:
            raise api.Error(409, "The scene has no frame to convert from: " + str(error))
        return scene_frames.describe(point, registry)
    raise api.Error(404, "Unknown workspace endpoint.")


def _import(api, work, data):
    import plan_exports
    text, filename = data.get("content"), data.get("filename")
    if not isinstance(text, str) or not isinstance(filename, str) or not filename:
        raise api.Error(400, "Send the cadastral file as text with its filename.")
    binary = None
    if filename.lower().endswith(".zip"):
        import base64
        try:
            binary = base64.b64decode(text, validate=True)
        except ValueError:
            raise api.Error(400, "A zipped shapefile must be sent as base64.")
        text = ""
    declared = data.get("crs")
    if declared is not None and not (isinstance(declared, int) or (isinstance(declared, str) and len(declared) < 20000)):
        raise api.Error(400, "crs is an EPSG number or a WKT string.")
    ground, registry, _ = _context(work)
    g = ground.grid
    bounds = ((g["ox"], g["oz"]), (g["ox"] + g["nx"] * g["cell"], g["oz"] + g["nz"] * g["cell"]))
    features, skipped, basis = _call(api, plan_exports.parse_parcels, filename, text,
                                     registry if registry.get("status") == "georeferenced" else None, bounds,
                                     data=binary, declared_crs=declared)
    if not features:
        reasons = "; ".join(sorted({s["reason"] for s in skipped})) or "no parcels"
        raise api.Error(422, f"No parcels could be placed on this scene ({reasons}).")
    proposal = _call(api, proposals.add_features, work, data.get("id"), features, data.get("revision"))
    return proposal, {"imported": len(features), "skipped": skipped[:50], "skipped_count": len(skipped), "basis": basis}


def post(api, root, route, data, server):
    scene = data.get("scene")
    work = _work(api, root, scene)
    extra = None
    with server.process_lock:
        api.ensure_not_running(server, scene)
        if route == "plan/proposals":
            proposal = _call(api, proposals.create_proposal, work, data.get("name"),
                             source=data.get("source"), kind=data.get("kind") or "plan")
        elif route == "plan/proposals/inferred":
            proposal = _call(api, proposals.set_inferred, work, data.get("id"), bool(data.get("inferred")), data.get("basis", ""))
        elif route == "plan/proposals/rename":
            proposal = _call(api, proposals.rename_proposal, work, data.get("id"), data.get("name"))
        elif route == "plan/proposals/delete":
            return {"index": _call(api, proposals.delete_proposal, work, data.get("id"))}
        elif route == "plan/feature":
            proposal = _call(api, proposals.upsert_feature, work, data.get("id"),
                             data.get("feature"), data.get("revision"))
        elif route == "plan/feature/delete":
            proposal = _call(api, proposals.delete_feature, work, data.get("id"),
                             data.get("feature_id"), data.get("revision"))
        elif route == "plan/features":
            proposal = _call(api, proposals.replace_features, work, data.get("id"),
                             data.get("features"), data.get("revision"))
        elif route == "plan/array":
            proposal = _call(api, proposals.array_objects, work, data.get("id"), data.get("item"),
                             data.get("line"), data.get("spacing_m"), data.get("revision"),
                             offset_m=data.get("offset_m", 0.0))
        elif route == "plan/import":
            proposal, extra = _import(api, work, data)
        else:
            raise api.Error(404, "Unknown workspace endpoint.")
    state = _state(work, proposal)
    if extra:
        state["import"] = extra
    return state
