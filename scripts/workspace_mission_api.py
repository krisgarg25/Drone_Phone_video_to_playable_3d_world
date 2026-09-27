"""HTTP handlers for mission planning and rehearsal (``/api/workspace/mission/*``).

Missions are proposals of kind ``mission``; their features are written through the plan
routes. These routes add the analysis (sight, exposure, covered routes, HLZs), the mission
pack, the rehearsal scenario the viewer plays and the recorded runs for after-action review.
"""
import base64
import math
import struct
import zlib

import numpy as np

import mission
import mission_analysis as analysis
import plan_shadow
import workspace_plan_api as plan_api
import workspace_proposals as proposals

VIEWSHED_COLORS = [(1, [250, 214, 90]), (2, [240, 140, 50]), (3, [214, 50, 50])]


def _call(api, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (proposals.ProposalError, analysis.AnalysisError) as error:
        raise api.Error(error.status, str(error))


def _mission(api, work, pid):
    proposal = _call(api, proposals.load_proposal, work, pid)
    if proposal.get("kind") != "mission":
        raise api.Error(409, "That proposal is a planning scheme, not a mission.")
    evaluation, ground, registry, existing = plan_api._evaluate(work, proposal)
    return proposal, evaluation, ground, registry


def _float(api, query, key, default=None, lo=-1e7, hi=1e7):
    raw = query.get(key, [None])[0]
    if raw in (None, ""):
        if default is None:
            raise api.Error(400, f"Supply {key}.")
        return default
    try:
        value = float(raw)
    except ValueError:
        raise api.Error(400, f"{key} must be a number.")
    if not (math.isfinite(value) and lo <= value <= hi):
        raise api.Error(400, f"{key} must be between {lo} and {hi}.")
    return value


def _phase_lines(proposal):
    return [{"label": f["params"]["label"] or f["name"], "line": f["params"]["line"]}
            for f in proposal["features"] if f["type"] == "phase_line" and not f.get("hidden")]


def analyse(work, proposal, evaluation, ground):
    terrain = analysis.Terrain(ground.grid)
    hostiles = analysis.observers(proposal, evaluation)
    routes = []
    for f in proposal["features"]:
        if f["type"] == "route" and not f.get("hidden") and f["params"]["kind"] != "patrol":
            routes.append({"id": f["id"], "name": f["name"],
                           "report": analysis.route_report(terrain, f["params"]["waypoints"], hostiles,
                                                           pace=f["params"]["pace"], phase_lines=_phase_lines(proposal),
                                                           names=f["params"]["names"])})
    overlay = []
    if hostiles:
        count = analysis.exposure_grid(terrain, hostiles)
        open_ground = (terrain.top - terrain.floor) < plan_shadow.OPEN_GROUND_M
        work_grid = {"cell": terrain.cell, "ox": terrain.ox, "oz": terrain.oz}
        budget = plan_shadow.MAX_OVERLAY_QUADS
        for k, (level, color) in enumerate(VIEWSHED_COLORS):
            mask = open_ground & ((count >= level) if k == len(VIEWSHED_COLORS) - 1 else (count == level))
            mesh, used = plan_shadow._quads(mask, terrain.floor, work_grid, color, 0.34, f"seen_by_{level}", budget)
            budget -= used
            if mesh:
                overlay.append(mesh)
        seen = count > 0
        dead = float(((~seen) & open_ground & terrain.supported).sum()) * terrain.cell ** 2
        watched = float((seen & open_ground & terrain.supported).sum()) * terrain.cell ** 2
    else:
        dead = watched = 0.0
    return {"hostiles": len(hostiles), "routes": routes, "overlay": overlay,
            "ground": {"watched_m2": round(watched, 1), "dead_ground_m2": round(dead, 1)},
            "notes": ["Sight is computed on the scanned surface model: crowns are solid, overhangs are not modelled, "
                      "and enemy positions are the plan's, not observations.",
                      f"Raster {terrain.cell:.2f} m; viewsheds count who can see a person's chest (ground + 1.2 m)."]}


def scenario(proposal, evaluation, ground, registry):
    """What the rehearsal viewer plays: bots, route, phase lines, objective, labels."""
    feats = {f["id"]: f for f in proposal["features"] if not f.get("hidden")}

    def path3d(points):
        pts = np.asarray(points, dtype=np.float64)
        ys = ground.sample(pts)[0]
        return [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for (x, z), y in zip(pts, ys)]
    bots, markers = [], []
    for f in feats.values():
        if f["type"] != "symbol":
            continue
        p, d = f["params"], evaluation["features"][f["id"]]
        pos = [p["position"][0], d["metrics"]["base_y"], p["position"][1]]
        if p["affiliation"] == "hostile" and p["role"] not in ("objective", "obstacle", "checkpoint", "rally_point", "hlz"):
            patrol = feats.get(p["patrol_route"]) if p["patrol_route"] else None
            bots.append({"symbol": f["id"], "label": f["name"], "role": p["role"], "behaviour": p["behaviour"],
                         "count": p["count"], "position": [round(v, 3) for v in pos], "facing_xz": d["facing_xz"],
                         "sector_deg": p["sector_deg"], "range_m": p["range_m"], "alert_radius_m": p["alert_radius_m"],
                         "patrol": path3d(patrol["params"]["waypoints"]) if patrol and patrol["type"] == "route" else None})
        else:
            markers.append({"symbol": f["id"], "label": f["name"], "role": p["role"], "affiliation": p["affiliation"],
                            "position": [round(v, 3) for v in pos]})
    routes = [f for f in feats.values() if f["type"] == "route" and f["params"]["kind"] != "patrol"]
    route = None
    if routes:
        r = next((f for f in routes if f["params"]["kind"] == "approach"), routes[0])
        names = r["params"]["names"]
        route = {"id": r["id"], "name": r["name"], "pace": r["params"]["pace"],
                 "waypoints": [{"name": names[i] if i < len(names) and names[i] else f"WP{i + 1}", "at": pt}
                               for i, pt in enumerate(path3d(r["params"]["waypoints"]))]}
    return {"mission": {"id": proposal["id"], "name": proposal["name"], "revision": proposal["revision"]},
            "bots": bots, "route": route, "markers": markers,
            "phase_lines": [{"label": pl["label"], "line": path3d(pl["line"])} for pl in _phase_lines(proposal)],
            "north_xz": mission.north_vector(registry).round(6).tolist(),
            "frame": {k: registry.get(k) for k in ("status", "scale_status")},
            "label": "Terrain from reconstruction; enemy positions are planned, not observed; unscanned ground is not assumed clear."}


def _geo_rows(work, ground, rows):
    """Add MGRS to rows with an [x, z] position when the scene is georeferenced."""
    try:
        registry = plan_api._context(work)[1]
    except Exception:  # noqa: BLE001 - coordinates are a courtesy here
        return
    if registry.get("status") != "georeferenced":
        return
    import scene_frames
    import survey_coords
    for row in rows:
        x, z = row["position"]
        y = float(ground.sample_top([[x, z]])[0][0])
        lat, lon, _ = scene_frames.viewer_to_geodetic([[x, y, z]], registry)[0]
        row["mgrs"] = survey_coords.to_mgrs(float(lat), float(lon))


def _png(rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def basemap(ground):
    """Top-down hillshade of the scanned surface (roofs darker by height) for the AAR replay."""
    img, terrain = basemap_rgb(ground)
    return {"png_base64": base64.b64encode(_png(img)).decode("ascii"),
            "bounds": {"x0": terrain.ox, "z0": terrain.oz, "x1": terrain.ox + terrain.nx * terrain.cell,
                       "z1": terrain.oz + terrain.nz * terrain.cell}, "cell": terrain.cell}


def basemap_rgb(ground):
    terrain = analysis.Terrain(ground.grid)
    top = terrain.top
    gz, gx = np.gradient(top, terrain.cell)
    light = np.array([-0.5, -0.6, 0.62])
    normal = np.dstack([-gx, -gz, np.ones_like(top)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    shade = np.clip(normal @ (light / np.linalg.norm(light)), 0, 1)
    raised = np.clip((top - terrain.floor) / 12.0, 0, 1)
    base = np.dstack([108 + 40 * shade, 122 + 46 * shade, 96 + 34 * shade])
    roof = np.dstack([150 + 60 * shade, 146 + 58 * shade, 140 + 56 * shade])
    rgb = base * (1 - raised[..., None]) + roof * raised[..., None]
    rgb[~terrain.supported] = [40, 44, 52]
    # Row 0 of the grid is the smallest z (north in a local scene): the image's top row.
    return np.clip(rgb, 0, 255).astype(np.uint8), terrain


def get(api, root, route, query):
    scene = query.get("scene", [""])[0]
    work = plan_api._work(api, root, scene)
    pid = query.get("id", [""])[0]
    if route == "mission/candidates":
        ground = proposals.Ground(work)
        # Video detections first (they are observations with a time), then scan labels.
        return {"candidates": mission.detection_candidates(work, ground) + mission.label_candidates(work, ground)}
    if route == "mission/hlz":
        ground = proposals.Ground(work)
        return analysis.hlz_candidates(analysis.Terrain(ground.grid),
                                       diameter_m=_float(api, query, "diameter", 25.0, 5, 200),
                                       max_slope_deg=_float(api, query, "slope", 7.0, 1, 30))
    if route == "mission/basemap":
        return basemap(proposals.Ground(work))
    if route == "mission/trafficability":
        ground = proposals.Ground(work)
        terrain = analysis.Terrain(ground.grid)
        road = water = None
        try:
            import ops_scene
            scene = ops_scene.Scene(work)
            road, water = scene.labels("road"), scene.labels("water")
        except (OSError, ValueError, KeyError):
            pass
        out = _call(api, analysis.trafficability, terrain, mobility=query.get("mobility", ["wheeled"])[0], road=road, water=water)
        klass = out.pop("class")
        overlay, budget = [], plan_shadow.MAX_OVERLAY_QUADS
        work_grid = {"cell": terrain.cell, "ox": terrain.ox, "oz": terrain.oz}
        for k, colour, part in ((0, [70, 190, 90], "go"), (1, [240, 190, 50], "slow_go"), (2, [210, 60, 60], "no_go")):
            mask = (klass == k) & terrain.supported
            mesh, used = plan_shadow._quads(mask, terrain.floor, work_grid, colour, 0.35, part, budget)
            budget -= used
            if mesh:
                overlay.append(mesh)
        out["overlay"] = overlay
        return out
    if route == "mission/obstacles":
        ground = proposals.Ground(work)
        out = analysis.obstacles(analysis.Terrain(ground.grid), min_height_m=_float(api, query, "min_height", 3.0, 0.5, 100))
        _geo_rows(work, ground, out["obstacles"])
        return out
    if route == "mission/los":
        ground = proposals.Ground(work)
        terrain = analysis.Terrain(ground.grid)
        a = [_float(api, query, k) for k in ("x1", "y1", "z1")]
        b = [_float(api, query, k) for k in ("x2", "y2", "z2")]
        eye = [a[0], a[1] + _float(api, query, "eye", analysis.EYE_M, 0, 50), a[2]]
        tgt = [b[0], b[1] + _float(api, query, "target", analysis.CHEST_M, 0, 50), b[2]]
        visible, at = analysis.los(terrain, eye, tgt)
        return {"visible": visible, "blocked_at": at, "eye": eye, "target": tgt,
                "distance_m": round(math.dist(eye, tgt), 2)}
    if route == "mission/session":
        import mission_session as ms
        try:
            return ms.snapshot(query.get("session", [""])[0])
        except ms.SessionError as error:
            raise api.Error(error.status, str(error))
    if route == "mission/sessions":
        import mission_session as ms
        return {"sessions": ms.sessions_for(scene, pid)}
    if route == "mission/runs":
        return {"runs": _call(api, mission.list_runs, work, pid)}
    if route == "mission/run":
        return _call(api, mission.load_run, work, pid, query.get("run", [""])[0])
    if route == "mission/aar-pdf":
        import aar_report
        record = _call(api, mission.load_run, work, pid, query.get("run", [""])[0])
        proposal = _call(api, proposals.load_proposal, work, pid)
        img, terrain = basemap_rgb(proposals.Ground(work))
        bounds = (terrain.ox, terrain.oz, terrain.ox + terrain.nx * terrain.cell, terrain.oz + terrain.nz * terrain.cell)
        data = aar_report.aar_pdf(record, mission_name=proposal["name"], basemap_rgb=img, bounds=bounds)
        return {"filename": f"{record['id']}-aar.pdf", "media_type": "application/pdf", "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii")}
    proposal, evaluation, ground, registry = _mission(api, work, pid)
    if route == "mission/analysis":
        return _call(api, analyse, work, proposal, evaluation, ground)
    if route == "mission/scenario":
        return scenario(proposal, evaluation, ground, registry)
    if route == "mission/threat":
        source = next((f for f in proposal["features"] if f["id"] == query.get("route", [""])[0] and f["type"] == "route"), None)
        if source is None:
            raise api.Error(404, "Route not found in this mission.")
        terrain = analysis.Terrain(ground.grid)
        out = _call(api, analysis.threat_heatmap, terrain, source["params"]["waypoints"],
                    range_m=_float(api, query, "range", 300.0, 20, 3000))
        score = out.pop("score")
        top = float(score.max()) or 1.0
        overlay, budget = [], plan_shadow.MAX_OVERLAY_QUADS
        work_grid = {"cell": terrain.cell, "ox": terrain.ox, "oz": terrain.oz}
        for lo, hi, colour, part in ((0.33, 0.6, [250, 214, 90], "threat_low"), (0.6, 0.8, [240, 140, 50], "threat_mid"),
                                     (0.8, 1.01, [214, 50, 50], "threat_high")):
            mask = (score / top >= lo) & (score / top < hi)
            mesh, used = plan_shadow._quads(mask, terrain.top, work_grid, colour, 0.45, part, budget)
            budget -= used
            if mesh:
                overlay.append(mesh)
        _geo_rows(work, ground, out["candidates"])
        out["overlay"] = overlay
        return out
    if route == "mission/pack":
        data, _, _ = _call(api, mission.mission_pack, proposal, evaluation,
                           registry if registry.get("status") == "georeferenced" else None, ground)
        stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in proposal["name"]).strip("-") or "mission"
        return {"filename": f"{stem}-mission-pack.zip", "media_type": "application/zip", "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii")}
    raise api.Error(404, "Unknown workspace endpoint.")


def _lan_addresses():
    import socket
    out = set()
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                out.add(ip)
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("10.255.255.255", 1))
        out.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return sorted(ip for ip in out if not ip.startswith("127."))


def _join_links(scene, pid, sid, token, conditions=""):
    query = f"asset=/work/{scene}/viewer_assets&combat=1&bots=1&mission={pid}&scene={scene}&session={sid}&token={token}{conditions}"
    return {"path": f"/viewer/pc.html?{query}",
            "lan": [f"https://{ip}:8138/viewer/pc.html?{query}" for ip in _lan_addresses()],
            "note": "Open the LAN link on each headset or laptop on the same network (HTTPS so WebXR is allowed); "
                    "the certificate is the local _cert.pem, so accept it once."}


def session_post(api, route, data):
    """join / state: callable by LAN devices holding the session token (see workspace_api.handle)."""
    import mission_session as ms
    try:
        if route == "mission/session/join":
            return ms.join(str(data.get("session")), data.get("token"), data.get("name"), data.get("role", "player"))
        return ms.report(str(data.get("session")), data.get("token"), str(data.get("player")), pose=data.get("pose"),
                         health=data.get("health"), alive=data.get("alive", True), since=data.get("since", 0),
                         bots=data.get("bots"))
    except ms.SessionError as error:
        raise api.Error(error.status, str(error))


def post(api, root, route, data, server):
    scene = data.get("scene")
    work = plan_api._work(api, root, scene)
    if route in ("mission/session/join", "mission/session/state"):
        return session_post(api, route, data)
    if route in ("mission/session/open", "mission/session/command"):
        import mission_session as ms
        try:
            if route == "mission/session/open":
                _call(api, proposals.load_proposal, work, data.get("id"))
                opened = ms.open_session(scene, data.get("id"), name=data.get("name") or "Exercise")
                return {**opened, "links": _join_links(scene, data.get("id"), opened["session"], opened["token"])}
            return {"command": ms.command(str(data.get("session")), data.get("command"))}
        except ms.SessionError as error:
            raise api.Error(error.status, str(error))
    pid = data.get("id")
    with server.process_lock:
        if route == "mission/run":
            return _saved(api, work, pid, data.get("run"))
        if route == "mission/run/delete":
            return {"runs": _call(api, mission.delete_run, work, pid, data.get("run"))}
        if route == "mission/covered-route":
            api.ensure_not_running(server, scene)
            proposal, evaluation, ground, registry = _mission(api, work, pid)
            source = next((f for f in proposal["features"] if f["id"] == data.get("route") and f["type"] == "route"), None)
            if source is None:
                raise api.Error(404, "Route not found in this mission.")
            terrain = analysis.Terrain(ground.grid)
            hostiles = analysis.observers(proposal, evaluation)
            wps = np.asarray(source["params"]["waypoints"], dtype=np.float64)
            line = _call(api, analysis.covered_route, terrain, wps[0], wps[-1], hostiles)
            feature = {"type": "route", "name": f"{source['name']} (covered, suggested)",
                       "params": {"waypoints": np.asarray(line).round(3).tolist(), "kind": source["params"]["kind"],
                                  "pace": source["params"]["pace"], "names": []}}
            updated = _call(api, proposals.add_features, work, pid, [feature], data.get("revision"))
            state = plan_api._state(work, updated)
            new_id = updated["features"][-1]["id"]
            state["suggested"] = {
                "id": new_id,
                "planned": analysis.route_report(terrain, wps, hostiles, pace=source["params"]["pace"]),
                "covered": analysis.route_report(terrain, line, hostiles, pace=source["params"]["pace"])}
            return state
    raise api.Error(404, "Unknown workspace endpoint.")


def _saved(api, work, pid, run):
    record = _call(api, mission.save_run, work, pid, run)
    return {"id": record["id"], "summary": record["summary"]}
