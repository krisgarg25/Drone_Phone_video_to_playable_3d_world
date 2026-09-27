"""Mission planning features (Phase 3): tactical symbols, routes and phase lines.

A mission is a proposal of kind ``mission`` in the same store as urban-planning schemes
(``workspace_proposals``): the same revisions, undo, 3D editing and write guard. Nothing
here touches measured data; every export and every rehearsal says "planned, not observed".

Symbols sit on the scanned surface (roof or ground, ``heights.f32``) plus an optional
``elevation_m`` (a window, a tower), so a sentry dragged across a roof stays on it.
Bearings are degrees clockwise from true north when the scene is georeferenced, and from
viewer -Z ("scene north") otherwise; ``evaluate`` turns them into viewer-frame vectors
once, so the analysis, the 3D symbol and the rehearsal bot cannot disagree.
"""
from __future__ import annotations

import io
import json
import math
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

import workspace_proposals as proposals

MISSION_TYPES = ("symbol", "route", "phase_line")
AFFILIATIONS = ("hostile", "friendly", "unknown", "neutral")
ROLES = ("infantry", "sniper", "machine_gun", "vehicle", "observation_post", "objective",
         "rally_point", "hlz", "support_by_fire", "obstacle", "checkpoint")
BEHAVIOURS = ("sentry", "patrol", "overwatch", "reaction")
ROUTE_KINDS = ("approach", "withdrawal", "patrol")
PACES = {"walk": 1.0, "patrol": 0.8, "run": 1.5}
# Frame colours after MIL-STD-2525 / APP-6: hostile red, friendly blue, neutral green, unknown yellow.
AFFILIATION_COLORS = {"hostile": [224, 58, 58], "friendly": [64, 128, 232], "neutral": [70, 176, 90],
                      "unknown": [236, 200, 40]}
ROLE_COLORS = {"objective": [245, 190, 40], "rally_point": [64, 128, 232], "hlz": [250, 250, 250],
               "support_by_fire": [64, 128, 232], "obstacle": [40, 40, 40], "checkpoint": [150, 150, 160]}
ROUTE_COLORS = {"approach": [255, 140, 30], "withdrawal": [160, 110, 240], "patrol": [224, 58, 58]}
PHASE_COLOR = [20, 20, 24]
EYE_HEIGHT_M = 1.6
"""Standing observer's eye above the surface it stands on."""


def _clean_text(value, limit):
    return re.sub(r"[\x00-\x1f]", " ", str(value or ""))[:limit]


def validate_params(kind, params):
    """Clean params for a mission feature, or raise ProposalError(400)."""
    num = proposals._number
    if kind == "symbol":
        pos = np.asarray(params.get("position"), dtype=np.float64)
        if pos.shape != (2,) or not np.isfinite(pos).all() or np.abs(pos).max() > 1e6:
            raise proposals.ProposalError(400, "position must be a finite [x, z]")
        role = proposals._choice(params, "role", "infantry", ROLES)
        affiliation = proposals._choice(params, "affiliation",
                                        "hostile" if role in ("infantry", "sniper", "machine_gun", "vehicle", "observation_post") else "friendly",
                                        AFFILIATIONS)
        patrol = params.get("patrol_route")
        if patrol is not None and (not isinstance(patrol, str) or not re.match(r"^route-[0-9a-f]{8}$", patrol)):
            raise proposals.ProposalError(400, "patrol_route must be a route feature id")
        return {
            "position": pos.round(4).tolist(), "affiliation": affiliation, "role": role,
            "elevation_m": num(params, "elevation_m", 0.0, 0.0, 100.0),
            "bearing_deg": num(params, "bearing_deg", 0.0, -360.0, 720.0) % 360,
            "sector_deg": num(params, "sector_deg", 90.0, 10.0, 360.0),
            "range_m": num(params, "range_m", 120.0 if role != "sniper" else 300.0, 5.0, 2000.0),
            "behaviour": proposals._choice(params, "behaviour", "sentry", BEHAVIOURS),
            "alert_radius_m": num(params, "alert_radius_m", 25.0, 0.0, 300.0),
            "count": int(num(params, "count", 1, 1, 8)),
            "diameter_m": num(params, "diameter_m", 25.0, 5.0, 200.0),
            "patrol_route": patrol,
            "source": proposals._choice(params, "source", "planner", ("planner", "labels", "detections")),
            "notes": _clean_text(params.get("notes"), 400),
        }
    line = proposals._poly(params.get("waypoints" if kind == "route" else "line"),
                           "waypoints" if kind == "route" else "line", closed_min=2)
    if float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) < 1.0:
        raise proposals.ProposalError(400, "A line must be at least 1 m long")
    if kind == "route":
        names = params.get("names") or []
        if not isinstance(names, list):
            raise proposals.ProposalError(400, "names must be a list")
        return {"waypoints": line.round(4).tolist(),
                "names": [_clean_text(n, 40) for n in names[:len(line)]],
                "kind": proposals._choice(params, "kind", "approach", ROUTE_KINDS),
                "pace": proposals._choice(params, "pace", "walk", tuple(PACES)),
                "loop": bool(params.get("loop", False))}
    return {"line": line.round(4).tolist(), "label": _clean_text(params.get("label") or "PL", 40)}


# ------------------------------------------------------------------ frame helpers
def north_vector(registry):
    """Unit [x, z] of true north in the viewer frame (scene north when not georeferenced)."""
    if registry is not None and registry.get("status") == "georeferenced":
        import scene_frames
        a, b = scene_frames.enu_to_viewer(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), registry)
        v = (b - a)[[0, 2]]
        n = np.linalg.norm(v)
        if n > 1e-9:
            return v / n
    return np.array([0.0, -1.0])


def bearing_vector(bearing_deg, north):
    """Viewer [x, z] of a bearing measured clockwise from ``north`` (seen from above)."""
    east = np.array([-north[1], north[0]])          # north turned 90 deg clockwise
    b = math.radians(bearing_deg)
    return north * math.cos(b) + east * math.sin(b)


def bearing_of(vec_xz, north):
    east = np.array([-north[1], north[0]])
    return math.degrees(math.atan2(float(np.dot(vec_xz, east)), float(np.dot(vec_xz, north)))) % 360


# ------------------------------------------------------------------ derived geometry
def _disc(cx, cy, cz, r, sides=16):
    a = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    verts = np.vstack([[cx, cy, cz], np.column_stack([cx + r * np.cos(a), np.full(sides, cy), cz + r * np.sin(a)])])
    return verts, [(0, 1 + (i + 1) % sides, 1 + i) for i in range(sides)]


def _ring(cx, cy, cz, r0, r1, sides=32):
    a = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    inner = np.column_stack([cx + r0 * np.cos(a), np.full(sides, cy), cz + r0 * np.sin(a)])
    outer = np.column_stack([cx + r1 * np.cos(a), np.full(sides, cy), cz + r1 * np.sin(a)])
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris += [(i, sides + i, sides + j), (i, sides + j, j)]
    return np.vstack([inner, outer]), tris


def _head(kind, cx, cy, cz, size):
    """The symbol frame as a small solid: diamond (hostile), box (friendly), cube (neutral),
    octahedron-ish (unknown)."""
    s = size / 2
    if kind in ("hostile", "unknown"):
        v = np.array([[cx, cy + s, cz], [cx + s, cy, cz], [cx, cy, cz + s], [cx - s, cy, cz], [cx, cy, cz - s], [cx, cy - s, cz]])
        t = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)]
        return v, t
    w = s * (1.4 if kind == "friendly" else 1.0)
    return proposals._box(-w, w, cy - s, cy + s, -s, s)[0] + [cx, 0, cz], proposals._box(-w, w, cy - s, cy + s, -s, s)[1]


def _fan(cx, cy, cz, direction, sector_deg, radius, steps=24):
    """Sector of fire: a flat fan at the observer's surface height."""
    base = math.atan2(direction[1], direction[0])
    half = math.radians(min(sector_deg, 359.9)) / 2
    a = np.linspace(base - half, base + half, steps + 1)
    verts = np.vstack([[cx, cy, cz], np.column_stack([cx + radius * np.cos(a), np.full(len(a), cy), cz + radius * np.sin(a)])])
    return verts, [(0, i + 2, i + 1) for i in range(steps)]


def surface_y(ground, xz):
    return float(ground.sample_top(np.atleast_2d(xz))[0][0])


def derive_symbol(feature, ground, north):
    p = feature["params"]
    x, z = p["position"]
    base = surface_y(ground, [x, z]) + p["elevation_m"]
    direction = bearing_vector(p["bearing_deg"], north)
    color = ROLE_COLORS.get(p["role"]) or AFFILIATION_COLORS[p["affiliation"]]
    meshes = []
    pole_v, pole_t = proposals._prism(0.06, base, base + 3.0, 6, cx=x, cz=z)
    meshes.append(proposals._mesh("symbol", pole_v, pole_t, [230, 230, 235], part="pole"))
    head_v, head_t = _head(p["affiliation"], x, base + 3.6, z, 1.2)
    meshes.append(proposals._mesh("symbol", head_v, head_t, color, part="frame"))
    # Facing: a short arrow stub from the head along the bearing.
    tip = np.array([x, base + 3.6, z]) + np.array([direction[0], 0, direction[1]]) * 1.6
    stub_v = np.array([[x, base + 3.5, z], [x, base + 3.7, z], tip])
    meshes.append(proposals._mesh("symbol", stub_v, [(0, 1, 2)], color, part="facing"))
    if p["role"] == "hlz":
        v, t = _ring(x, base + 0.1, z, p["diameter_m"] / 2 - 0.5, p["diameter_m"] / 2)
        meshes.append(proposals._mesh("symbol", v, t, [250, 250, 250], 0.85, part="hlz"))
    elif p["affiliation"] in ("hostile", "friendly") and p["role"] not in ("objective", "rally_point", "obstacle", "checkpoint"):
        # Drawn to at most 150 m so a long-range post does not paint the whole scene; the
        # analysis and the rehearsal use the full range.
        v, t = _fan(x, base + 0.15, z, direction, p["sector_deg"], min(p["range_m"], 150.0))
        meshes.append(proposals._mesh("symbol", v, t, color, 0.14, part="sector"))
    elif p["role"] in ("objective", "rally_point"):
        v, t = _ring(x, base + 0.1, z, 2.2, 3.0)
        meshes.append(proposals._mesh("symbol", v, t, color, 0.9, part="marker"))
    footprint = proposals.rectangle([x, z], [1.0, 1.0], 0.0)
    return {"meshes": meshes, "footprint": footprint.round(3).tolist(),
            "metrics": {"base_y": round(base, 3), "eye_y": round(base + EYE_HEIGHT_M, 3),
                        "bearing_deg": round(p["bearing_deg"], 1)},
            "facing_xz": [round(float(direction[0]), 6), round(float(direction[1]), 6)]}


def _line_ribbon(line, ground, width, color, lift, part, *, dash=None):
    dense = proposals.densify(np.asarray(line, dtype=np.float64), max(ground.cell, 0.5))
    ys = ground.sample(dense)[0] + lift
    meshes = []
    if dash:
        station = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        on = (station // dash) % 2 == 0
        start = None
        for i, flag in enumerate(on):
            if flag and start is None:
                start = i
            if (not flag or i == len(on) - 1) and start is not None:
                end = i + 1 if flag else i
                if end - start >= 2:
                    v, t, _, _ = proposals._ribbon(dense[start:end], ys[start:end], -width / 2, width / 2)
                    meshes.append((v, t))
                start = None
        verts, tris = [], []
        for v, t in meshes:
            tris += [(a + len(verts), b + len(verts), c + len(verts)) for a, b, c in t]
            verts += v.tolist()
        return proposals._mesh("route", verts or [[0, 0, 0]] * 3, tris or [(0, 1, 2)], color, 0.95, part=part)
    v, t, _, _ = proposals._ribbon(dense, ys, -width / 2, width / 2)
    return proposals._mesh("route", v, t, color, 0.95, part=part)


def derive_route(feature, ground):
    p = feature["params"]
    wps = np.asarray(p["waypoints"], dtype=np.float64)
    color = ROUTE_COLORS[p["kind"]]
    meshes = [_line_ribbon(wps, ground, 0.7, color, 0.12, "path")]
    ys = ground.sample(wps)[0]
    # All waypoint posts in one mesh: a suggested route can have dozens of waypoints and
    # the viewer draws at most 16 meshes per feature.
    verts, tris = [], []
    for i, ((x, z), y) in enumerate(zip(wps, ys)):
        end = i in (0, len(wps) - 1)
        v, t = proposals._prism(0.35 if end else 0.22, y, y + (2.2 if end else 1.4), 8, cx=x, cz=z)
        tris += [(a + len(verts), b + len(verts), c + len(verts)) for a, b, c in t]
        verts += v.tolist()
    meshes.append(proposals._mesh("route", verts, tris, color, part="waypoint"))
    length = float(np.linalg.norm(np.diff(wps, axis=0), axis=1).sum())
    corridor = np.vstack([wps - 0.5, wps[::-1] + 0.5])
    return {"meshes": meshes, "footprint": corridor.round(3).tolist(),
            "metrics": {"length_m": round(length, 1), "waypoints": len(wps), "base_y": round(float(ys.min()), 3)}}


def derive_phase_line(feature, ground):
    line = np.asarray(feature["params"]["line"], dtype=np.float64)
    ys = ground.sample(line)[0]
    return {"meshes": [_line_ribbon(line, ground, 0.9, PHASE_COLOR, 0.14, "phase", dash=4.0)],
            "footprint": np.vstack([line - 0.5, line[::-1] + 0.5]).round(3).tolist(),
            "metrics": {"length_m": round(float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()), 1),
                        "base_y": round(float(ys.min()), 3)}}


def derive(feature, ground, north):
    if feature["type"] == "symbol":
        return derive_symbol(feature, ground, north)
    if feature["type"] == "route":
        return derive_route(feature, ground)
    return derive_phase_line(feature, ground)


# ------------------------------------------------------------------ candidates from labels
def label_candidates(work, ground, *, cell_m=1.5, min_points=3):
    """Person / vehicle clusters in the semantic labels, offered as *candidate* hostiles."""
    path = Path(work) / "viewer_assets" / "semantics.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    xyz = np.asarray(data.get("coords") or [], dtype=np.float64).reshape(-1, 3)
    rgb = [tuple(c) for c in data.get("rgb") or []]
    lookup = proposals._class_rgb()
    labels = np.array([lookup.get(c, "unknown") for c in rgb]) if rgb else np.array([])
    out = []
    if len(labels) != len(xyz):
        return out
    from scipy import ndimage
    for cls, role in (("person", "infantry"), ("vehicle", "vehicle")):
        pts = xyz[labels == cls]
        if len(pts) < min_points:
            continue
        lo = pts[:, [0, 2]].min(axis=0)
        idx = np.floor((pts[:, [0, 2]] - lo) / cell_m).astype(int)
        occ = np.zeros(idx.max(axis=0) + 1, dtype=bool)
        occ[idx[:, 0], idx[:, 1]] = True
        comp, n = ndimage.label(occ, structure=np.ones((3, 3)))
        member = comp[idx[:, 0], idx[:, 1]]
        for k in range(1, n + 1):
            group = pts[member == k]
            if len(group) < min_points:
                continue
            c = group[:, [0, 2]].mean(axis=0)
            out.append({"role": role, "position": c.round(3).tolist(), "points": int(len(group)),
                        "source": "labels",
                        "basis": f"{cls} semantic labels (heuristic), not a confirmed detection"})
    return out[:50]


def detection_candidates(work, ground):
    """People and vehicles from the video (MIL-03): the M10 detection layer as candidates.

    Each object is offered once with its sightings and time span; the class decides the
    role (person -> infantry, vehicle -> vehicle; an unclassed dynamic-mask blob stays a
    person-sized ``infantry`` candidate and says so). Always UNKNOWN until an analyst
    confirms it.
    """
    import survey_detections
    layer = survey_detections.layer(work, ground.sample_top)
    out = []
    for obj in layer["objects"]:
        role = "vehicle" if obj["class"] in ("vehicle", "car", "truck") else "infantry"
        seen = f"seen {obj['sightings']}x" + (f", {obj['first_t']}-{obj['last_t']} s" if obj["first_t"] is not None else "")
        out.append({"role": role, "position": [obj["position"][0], obj["position"][2]], "points": obj["sightings"],
                    "source": "detections",
                    "basis": f"{obj['class']} from {layer['source']}; {seen}"
                             + ("; moving" if obj["moving"] else "")})
    return out[:50]


# ------------------------------------------------------------------ mission pack (KMZ + GPX)
def _geo(points_viewer, registry):
    import scene_frames
    return scene_frames.viewer_to_geodetic(np.atleast_2d(points_viewer), registry)


def mission_pack(proposal, evaluation, registry, ground):
    """(zip bytes): KMZ overlay + GPX routes. WGS84 formats, so a GPS fit is required."""
    if registry is None or registry.get("status") != "georeferenced":
        raise proposals.ProposalError(409, "KMZ and GPX are WGS84 formats and this scene has no GPS fit. "
                                           "Use the GeoJSON/DXF exports, which say LOCAL.")
    placemarks, gpx_routes, gpx_points = [], [], []
    for f in proposal["features"]:
        d = evaluation["features"].get(f["id"])
        if not d or f["type"] not in MISSION_TYPES:
            continue
        name = escape(f["name"])
        if f["type"] == "symbol":
            p = f["params"]
            lat, lon, h = _geo([[p["position"][0], d["metrics"]["base_y"], p["position"][1]]], registry)[0]
            desc = escape(f"{p['affiliation']} {p['role'].replace('_', ' ')}; bearing {p['bearing_deg']:.0f} deg, "
                          f"sector {p['sector_deg']:.0f} deg, range {p['range_m']:.0f} m. Planned, not observed.")
            placemarks.append(f"<Placemark><name>{name}</name><description>{desc}</description>"
                              f"<styleUrl>#{p['affiliation']}</styleUrl><Point><coordinates>{lon:.8f},{lat:.8f},{h:.2f}"
                              f"</coordinates></Point></Placemark>")
            gpx_points.append(f'<wpt lat="{lat:.8f}" lon="{lon:.8f}"><ele>{h:.2f}</ele><name>{name}</name>'
                              f"<type>{p['affiliation']} {p['role']}</type></wpt>")
        else:
            key = "waypoints" if f["type"] == "route" else "line"
            line = np.asarray(f["params"][key], dtype=np.float64)
            ys = ground.sample(line)[0]
            geo = _geo(np.column_stack([line[:, 0], ys, line[:, 1]]), registry)
            coords = " ".join(f"{lon:.8f},{lat:.8f},{h:.2f}" for lat, lon, h in geo)
            style = f["params"]["kind"] if f["type"] == "route" else "phase"
            placemarks.append(f"<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl>"
                              f"<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString></Placemark>")
            if f["type"] == "route":
                names = f["params"]["names"]
                pts = "".join(f'<rtept lat="{lat:.8f}" lon="{lon:.8f}"><ele>{h:.2f}</ele><name>'
                              f"{escape(names[i] if i < len(names) and names[i] else f'WP{i + 1}')}</name></rtept>"
                              for i, (lat, lon, h) in enumerate(geo))
                gpx_routes.append(f"<rte><name>{name}</name>{pts}</rte>")
    styles = "".join(f'<Style id="{k}"><IconStyle><color>ff{c[2]:02x}{c[1]:02x}{c[0]:02x}</color></IconStyle>'
                     f'<LineStyle><color>ff{c[2]:02x}{c[1]:02x}{c[0]:02x}</color><width>3</width></LineStyle></Style>'
                     for k, c in {**AFFILIATION_COLORS, **ROUTE_COLORS, "phase": [20, 20, 24]}.items())
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>{escape(proposal['name'])} (planned, not observed)</name>{styles}{''.join(placemarks)}</Document></kml>")
    gpx = ('<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="Ground Control mission planner" '
           'xmlns="http://www.topografix.com/GPX/1/1">'
           f"<metadata><name>{escape(proposal['name'])}</name><desc>Planned, not observed</desc></metadata>"
           f"{''.join(gpx_points)}{''.join(gpx_routes)}</gpx>")
    kmz = io.BytesIO()
    with zipfile.ZipFile(kmz, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)
    buffer = io.BytesIO()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", proposal["name"]).strip("-") or "mission"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}.kmz", kmz.getvalue())
        z.writestr(f"{stem}.gpx", gpx)
    return buffer.getvalue(), kml, gpx


# ------------------------------------------------------------------ rehearsal runs (AAR)
MAX_RUN_FRAMES = 12000
MAX_RUNS = 100


def _runs_dir(work, pid):
    if not re.match(r"^p-[0-9a-f]{10}$", str(pid)):
        raise proposals.ProposalError(400, "Invalid mission id")
    return Path(work) / "proposals" / "runs" / pid


def _finite_list(value, width, name, limit):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != width or len(arr) > limit or not np.isfinite(arr).all():
        raise proposals.ProposalError(400, f"{name} must be a bounded list of finite {width}-tuples")
    return arr


def save_run(work, pid, run):
    """Validate a rehearsal recording, compute its after-action metrics and store it."""
    proposals.load_proposal(work, pid)
    if not isinstance(run, dict):
        raise proposals.ProposalError(400, "run must be an object")
    hz = run.get("hz")
    if not isinstance(hz, (int, float)) or not 1 <= hz <= 30:
        raise proposals.ProposalError(400, "hz must be 1..30")
    # player frames: [t, x, y, z, yaw, health]
    player = _finite_list(run.get("player"), 6, "player", MAX_RUN_FRAMES)
    if len(player) < 2:
        raise proposals.ProposalError(400, "A run needs at least two frames")
    bots = run.get("bots") or []
    if not isinstance(bots, list) or len(bots) > 64:
        raise proposals.ProposalError(400, "bots must be a list of at most 64")
    clean_bots = []
    for b in bots:
        if not isinstance(b, dict):
            raise proposals.ProposalError(400, "each bot must be an object")
        # bot frames: [t, x, y, z, yaw, sees(0/1), alive(0/1)]
        frames = _finite_list(b.get("frames"), 7, "bot frames", MAX_RUN_FRAMES)
        clean_bots.append({"id": int(b.get("id", len(clean_bots))), "symbol": _clean_text(b.get("symbol"), 64),
                           "label": _clean_text(b.get("label"), 80), "behaviour": _clean_text(b.get("behaviour"), 20),
                           "frames": frames.round(3).tolist()})
    events = run.get("events") or []
    if not isinstance(events, list) or len(events) > 5000:
        raise proposals.ProposalError(400, "events must be a list of at most 5000")
    clean_events = []
    for e in events:
        if not isinstance(e, dict) or not isinstance(e.get("t"), (int, float)) or not math.isfinite(e["t"]):
            raise proposals.ProposalError(400, "each event needs a finite time t")
        clean_events.append({"t": round(float(e["t"]), 2), "type": _clean_text(e.get("type"), 24),
                             "text": _clean_text(e.get("text"), 160),
                             **({"bot": int(e["bot"])} if isinstance(e.get("bot"), int) else {}),
                             **({"at": [round(float(v), 2) for v in e["at"][:3]]}
                                if isinstance(e.get("at"), list) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in e["at"][:3]) else {})})
    route = run.get("route")
    route_arr = _finite_list(route, 2, "route", 400) if route is not None else np.zeros((0, 2))
    conditions = run.get("conditions") if isinstance(run.get("conditions"), dict) else {}
    record = {
        "schema_version": 1, "id": "run-" + uuid.uuid4().hex[:10], "mission": pid,
        "mission_revision": int(run.get("mission_revision") or 0),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hz": float(hz), "player": player.round(3).tolist(), "bots": clean_bots, "events": clean_events,
        "route": route_arr.round(3).tolist(),
        "conditions": {k: _clean_text(v, 20) for k, v in conditions.items() if k in ("light", "fog_m", "filter")},
        "note": "Rehearsal on reconstructed terrain; enemy positions are the plan's, not observations.",
    }
    record["summary"] = run_summary(record)
    directory = _runs_dir(work, pid)
    existing = sorted(directory.glob("run-*.json")) if directory.is_dir() else []
    if len(existing) >= MAX_RUNS:
        raise proposals.ProposalError(413, f"At most {MAX_RUNS} runs per mission; delete old ones first")
    proposals._atomic(directory / f"{record['id']}.json", record)
    return record


def run_summary(record):
    p = np.asarray(record["player"], dtype=np.float64)
    t = p[:, 0]
    dt = np.diff(t, append=t[-1])
    steps = np.linalg.norm(np.diff(p[:, [1, 3]], axis=0), axis=1)
    # A respawn is a jump, not a walk: nobody covers 8 m between two frames on foot.
    dist = float(steps[steps <= 8.0].sum())
    per_bot, seen_any = [], np.zeros(len(p), dtype=bool)
    for b in record["bots"]:
        f = np.asarray(b["frames"], dtype=np.float64).reshape(-1, 7)
        if not len(f):
            continue
        sees = np.interp(t, f[:, 0], f[:, 5]) >= 0.5
        seen_any |= sees
        spotted = float(t[np.argmax(sees)] - t[0]) if sees.any() else None
        per_bot.append({"id": b["id"], "label": b["label"], "exposure_s": round(float(dt[sees].sum()), 1),
                        "first_seen_s": None if spotted is None else round(spotted, 1),
                        "neutralised": bool(f[-1, 6] < 0.5)})
    kinds = {}
    for e in record["events"]:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    waypoints = [e for e in record["events"] if e["type"] == "waypoint"]
    return {"duration_s": round(float(t[-1] - t[0]), 1), "distance_m": round(dist, 1),
            "exposure_s": round(float(dt[seen_any].sum()), 1),
            "exposed_pct": round(100 * float(dt[seen_any].sum()) / max(float(t[-1] - t[0]), 1e-9), 1),
            "per_bot": per_bot, "waypoints_reached": len(waypoints),
            "waypoint_times": [{"t": w["t"], "text": w["text"]} for w in waypoints],
            "hits_taken": kinds.get("hit_taken", 0), "casualty": kinds.get("casualty", 0),
            "neutralised": sum(1 for b in per_bot if b["neutralised"]),
            "completed": any(e["type"] == "complete" for e in record["events"])}


def list_runs(work, pid):
    directory = _runs_dir(work, pid)
    out = []
    for path in sorted(directory.glob("run-*.json")) if directory.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"id": data["id"], "created_at": data["created_at"], "summary": data["summary"],
                    "conditions": data.get("conditions", {})})
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def load_run(work, pid, rid):
    if not re.match(r"^run-[0-9a-f]{10}$", str(rid)):
        raise proposals.ProposalError(400, "Invalid run id")
    path = _runs_dir(work, pid) / f"{rid}.json"
    if not path.is_file():
        raise proposals.ProposalError(404, "Run not found")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_run(work, pid, rid):
    load_run(work, pid, rid)
    (_runs_dir(work, pid) / f"{rid}.json").unlink()
    return list_runs(work, pid)
