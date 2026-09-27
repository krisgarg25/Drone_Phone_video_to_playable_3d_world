"""Proposal exchange formats: CityJSON, DXF and 3D Tiles out; cadastral parcels in.

Every export goes through one ``OutFrame``:

* **georeferenced** scene - UTM easting / northing / WGS84 ellipsoidal height in the
  zone of the scene's GPS origin (``survey_crs``), and 3D Tiles placed on the globe
  through the ENU -> ECEF transform at that origin;
* **local** scene - viewer metres as (x, -z, y) = (east-ish, north-ish, up), with no CRS
  claimed anywhere, and the file says LOCAL.

The derived meshes the viewer draws are what gets written, so an export cannot disagree
with the screen. Heights are ellipsoidal, not orthometric, and every file says so.

Cadastral import reads parcel outlines from GeoJSON (WGS84, a declared UTM EPSG, or the
LOCAL frame this editor itself exports), KML (WGS84) or DXF (closed polylines, in the
scene's UTM zone when georeferenced, else local metres) and returns plot/zone features.
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from xml.etree import ElementTree

import numpy as np

import workspace_proposals as proposals

HEIGHT_NOTE = "heights are WGS84 ellipsoidal metres, not orthometric"
LOCAL_NOTE = ("LOCAL: viewer metres (x = east-ish, y = -viewer z = north-ish, z = up); the scene "
              "has no GPS similarity fit, so these are not Earth coordinates")
MAX_IMPORT_BYTES = (2 << 20) - 4096
"""The workspace refuses JSON bodies over 2 MiB; clip a district-wide file to the site first."""
MAX_PARCELS = 500


class ExportError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ frames
class OutFrame:
    def __init__(self, registry):
        import scene_frames
        import survey_crs as crs
        self._frames, self._crs = scene_frames, crs
        self.registry = registry
        self.georeferenced = registry is not None and registry.get("status") == "georeferenced"
        self.crs = crs.crs_from_alignment(registry["alignment"]) if self.georeferenced else None

    def enu(self, points):
        """Viewer points to ENU metres (georeferenced) or the local (x, -z, y) frame."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if self.georeferenced:
            return self._frames.viewer_to_enu(p, self.registry)
        return np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])

    def out(self, points):
        """Viewer points to the export CRS: UTM E/N/h, or local (x, -z, y)."""
        if self.georeferenced:
            return self._crs.enu_to_crs(self.enu(points), None, self.crs)[0]
        return self.enu(points)

    @property
    def epsg(self):
        return self.crs["epsg"] if self.crs else None

    def ecef_from_enu(self):
        """4x4 column-vector matrix taking ENU metres at the scene origin to ECEF."""
        o = self.registry["origin"]
        basis = self._crs.enu_basis(o["latitude_deg"], o["longitude_deg"])
        origin = self._crs.ecef_from_geodetic([o["latitude_deg"]], [o["longitude_deg"]], [o["altitude_m"]])[0]
        m = np.eye(4)
        m[:3, :3] = basis.T          # columns: east, north, up in ECEF
        m[:3, 3] = origin
        return m

    # -- inbound
    def _enu_to_viewer_xz(self, enu):
        v = self._frames.enu_to_viewer(enu, self.registry)
        return v[:, [0, 2]]

    def from_lonlat(self, lonlat):
        if not self.georeferenced:
            raise ExportError(409, "These parcels are in WGS84 longitude/latitude, but this scene has no GPS "
                                   "fit to place them with. Georeference the scene, or import them in the "
                                   "scene's LOCAL frame.")
        o = self.registry["origin"]
        ll = np.asarray(lonlat, dtype=np.float64)
        ecef = self._crs.ecef_from_geodetic(ll[:, 1], ll[:, 0], np.full(len(ll), o["altitude_m"]))
        origin = self._crs.ecef_from_geodetic([o["latitude_deg"]], [o["longitude_deg"]], [o["altitude_m"]])[0]
        enu = (ecef - origin) @ self._crs.enu_basis(o["latitude_deg"], o["longitude_deg"]).T
        return self._enu_to_viewer_xz(enu)

    def from_utm(self, en, zone=None, hemisphere=None):
        if not self.georeferenced:
            raise ExportError(409, "These parcels are in UTM, but this scene has no GPS fit to place them with.")
        en = np.asarray(en, dtype=np.float64)
        lat, lon = self._crs.utm_inverse(en[:, 0], en[:, 1], zone or self.crs["zone"],
                                         hemisphere or self.crs["hemisphere"])
        return self.from_lonlat(np.column_stack([lon, lat]))

    @staticmethod
    def from_local(xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.column_stack([xy[:, 0], -xy[:, 1]])


def _visible(proposal, evaluation):
    for f in proposal["features"]:
        derived = evaluation["features"].get(f["id"])
        if derived and not f.get("hidden"):
            yield f, derived


def _triangles(mesh):
    p = np.asarray(mesh["positions"], dtype=np.float64).reshape(-1, 3)
    return p, np.asarray(mesh["indices"], dtype=np.int64).reshape(-1, 3)


def _slug(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "scheme"


# ------------------------------------------------------------------ CityJSON 2.0
class _Vertices:
    def __init__(self):
        self.rows, self.index = [], {}

    def add(self, xyz):
        key = tuple(np.round(xyz, 3))
        if key not in self.index:
            self.index[key] = len(self.rows)
            self.rows.append(key)
        return self.index[key]


def _outward(tri_viewer, footprint):
    """Reorder a building triangle so its normal points out of the solid."""
    a, b, c = tri_viewer
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return None
    n /= length
    if abs(n[1]) > 0.5:                         # roof (up) or base (down): decided by caller
        return tri_viewer
    probe = (a + b + c)[[0, 2]] / 3 + n[[0, 2]] * 0.05
    return tri_viewer[::-1] if proposals.contains(footprint, probe[None])[0] else tri_viewer


def cityjson(proposal, evaluation, registry, ground, existing=None):
    frame = OutFrame(registry)
    verts = _Vertices()
    objects = {}

    def ring(points_viewer):
        return [verts.add(p) for p in frame.out(points_viewer)]

    for f, d in _visible(proposal, evaluation):
        kind = f["type"]
        attrs = {"name": f["name"], "status": proposals.status_text(proposal),
                 **{k: v for k, v in f["params"].items() if isinstance(v, (int, float, str))},
                 **{k: v for k, v in d["metrics"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}
        if kind == "building":
            footprint = np.asarray(f["params"]["footprint"], dtype=np.float64)
            base = d["metrics"]["base_y"]
            surfaces, values = [], []
            for mesh in d["meshes"]:
                p, tris = _triangles(mesh)
                for t in tris:
                    tri = _outward(p[t], footprint)
                    if tri is None:
                        continue
                    if mesh["part"] == "roof":
                        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                        if n[1] < 0:
                            tri = tri[::-1]
                    surfaces.append([ring(tri)])
                    values.append(1 if mesh["part"] == "roof" else 0)
            ccw = footprint if proposals.polygon_area(footprint) > 0 else footprint[::-1]
            for tri in proposals.triangulate(ccw):
                pts = np.column_stack([ccw[list(tri), 0], np.full(3, base), ccw[list(tri), 1]])
                n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                surfaces.append([ring(pts if n[1] < 0 else pts[::-1])])
                values.append(2)
            geometry = [{"type": "Solid", "lod": "2.2", "boundaries": [surfaces],
                         "semantics": {"surfaces": [{"type": "WallSurface"}, {"type": "RoofSurface"},
                                                    {"type": "GroundSurface"}], "values": [values]}}]
            objects[f["id"]] = {"type": "Building", "attributes": {
                **attrs, "storeysAboveGround": f["params"]["floors"], "measuredHeight": d["metrics"]["height_m"],
                "roofType": f["params"]["roof"], "function": f["params"]["use"]}, "geometry": geometry}
        elif kind in ("road", "object"):
            surfaces, values, sem = [], [], []
            for mesh in d["meshes"]:
                label = {"carriageway": "TrafficArea", "footpath": "AuxiliaryTrafficArea",
                         "median": "AuxiliaryTrafficArea"}.get(mesh["part"]) if kind == "road" else None
                if label and {"type": label} not in sem:
                    sem.append({"type": label})
                p, tris = _triangles(mesh)
                for t in tris:
                    surfaces.append([ring(p[t])])
                    values.append(sem.index({"type": label}) if label else None)
            geometry = {"type": "MultiSurface", "lod": "2" if kind == "object" else "1", "boundaries": surfaces}
            if kind == "road":
                geometry["semantics"] = {"surfaces": sem, "values": values}
                objects[f["id"]] = {"type": "Road", "attributes": attrs, "geometry": [geometry]}
            else:
                tree = f["params"]["item"].startswith("tree")
                objects[f["id"]] = {"type": "SolitaryVegetationObject" if tree else "CityFurniture",
                                    "attributes": attrs, "geometry": [geometry]}
        else:                                         # zone / clip: a flat outline on the ground
            poly = np.asarray(d["footprint"], dtype=np.float64)
            ys = ground.sample(poly)[0]
            ring_pts = np.column_stack([poly[:, 0], ys, poly[:, 1]])
            if proposals.polygon_area(poly) < 0:      # upward normal once mapped to (x, -z, y)
                ring_pts = ring_pts[::-1]
            geometry = [{"type": "MultiSurface", "lod": "1", "boundaries": [[ring(ring_pts)]]}]
            if kind == "zone":
                objects[f["id"]] = {"type": "LandUse", "attributes": {
                    **attrs, **{f"rule_{k}": v for k, v in f["params"]["rules"].items()}}, "geometry": geometry}
            else:
                gone = [b["id"] for b in (existing or {}).get("buildings", []) if b["id"] in evaluation["demolished"]
                        and proposals.contains(poly, np.asarray(b["centre"])[None])[0]]
                objects[f["id"]] = {"type": "GenericCityObject", "attributes": {
                    **attrs, "status": "proposed demolition", "demolishes_existing": gone}, "geometry": geometry}
    rows = np.asarray(verts.rows, dtype=np.float64).reshape(-1, 3)
    lo = rows.min(axis=0) if len(rows) else np.zeros(3)
    scale = 0.001
    doc = {"type": "CityJSON", "version": "2.0",
           "transform": {"scale": [scale] * 3, "translate": lo.round(3).tolist()},
           "metadata": {"title": f"{proposal['name']} ({'inferred hypothesis' if proposal.get('inferred') else 'proposed scheme'})",
                        "geographicalExtent": (lo.tolist() + rows.max(axis=0).tolist()) if len(rows) else [0] * 6},
           "CityObjects": objects,
           "vertices": np.rint((rows - lo) / scale).astype(np.int64).tolist()}
    if frame.georeferenced:
        doc["metadata"]["referenceSystem"] = f"https://www.opengis.net/def/crs/EPSG/0/{frame.epsg}"
        doc["+proposal"] = {"note": HEIGHT_NOTE}
    else:
        doc["+proposal"] = {"note": LOCAL_NOTE}
    doc["+proposal"].update(id=proposal["id"], revision=proposal["revision"], metrics=evaluation["metrics"])
    return doc


# ------------------------------------------------------------------ DXF (R12, ASCII)
LAYERS = {"PLAN-ROAD-CL": 1, "PLAN-ROAD-EDGE": 8, "PLAN-BLDG": 30, "PLAN-BLDG-3D": 31, "PLAN-PLOT": 4,
          "PLAN-OBJECT": 3, "PLAN-OBJECT-3D": 3, "PLAN-DEMOLISH": 1, "PLAN-LABEL": 7}


class _Dxf:
    def __init__(self):
        self.lines = []

    def g(self, code, value):
        if isinstance(value, float):
            value = f"{value:.4f}"
        self.lines += [str(code), str(value)]

    def polyline(self, layer, pts, closed):
        self.g(0, "POLYLINE"); self.g(8, layer); self.g(66, 1); self.g(70, 8 | (1 if closed else 0))
        self.g(10, 0.0); self.g(20, 0.0); self.g(30, 0.0)
        for x, y, z in pts:
            self.g(0, "VERTEX"); self.g(8, layer)
            self.g(10, float(x)); self.g(20, float(y)); self.g(30, float(z)); self.g(70, 32)
        self.g(0, "SEQEND"); self.g(8, layer)

    def face(self, layer, tri):
        self.g(0, "3DFACE"); self.g(8, layer)
        pts = list(tri) + [tri[-1]]
        for k, (x, y, z) in enumerate(pts):
            self.g(10 + k, float(x)); self.g(20 + k, float(y)); self.g(30 + k, float(z))

    def text(self, layer, xyz, height, value):
        self.g(0, "TEXT"); self.g(8, layer)
        self.g(10, float(xyz[0])); self.g(20, float(xyz[1])); self.g(30, float(xyz[2]))
        self.g(40, float(height)); self.g(1, value.replace("\n", " ")[:200])

    def point(self, layer, xyz):
        self.g(0, "POINT"); self.g(8, layer)
        self.g(10, float(xyz[0])); self.g(20, float(xyz[1])); self.g(30, float(xyz[2]))


def dxf(proposal, evaluation, registry, ground):
    frame = OutFrame(registry)
    body = _Dxf()
    for f, d in _visible(proposal, evaluation):
        kind = f["type"]
        poly = np.asarray(d["footprint"], dtype=np.float64)
        base = d["metrics"].get("base_y")
        ys = np.full(len(poly), base) if base is not None else ground.sample(poly)[0]
        ring = frame.out(np.column_stack([poly[:, 0], ys, poly[:, 1]]))
        label_at = ring.mean(axis=0)
        if kind == "road":
            line = np.asarray(f["params"]["centerline"], dtype=np.float64)
            body.polyline("PLAN-ROAD-CL", frame.out(np.column_stack([line[:, 0], ground.sample(line)[0], line[:, 1]])), False)
            body.polyline("PLAN-ROAD-EDGE", ring, True)
            text = f"{f['name']} {f['params']['width_m']:g} m"
        elif kind == "building":
            body.polyline("PLAN-BLDG", ring, True)
            for mesh in d["meshes"]:
                p, tris = _triangles(mesh)
                for t in tris:
                    body.face("PLAN-BLDG-3D", frame.out(p[t]))
            text = f"{f['name']} G+{f['params']['floors'] - 1} {d['metrics']['height_m']:g} m"
        elif kind == "zone":
            body.polyline("PLAN-PLOT", ring, True)
            text = f"{f['name']} {d['metrics']['area_m2']:g} m2"
        elif kind == "object":
            body.point("PLAN-OBJECT", frame.out([[*f["params"]["position"][:1], base, f["params"]["position"][1]]])[0])
            for mesh in d["meshes"]:
                p, tris = _triangles(mesh)
                for t in tris:
                    body.face("PLAN-OBJECT-3D", frame.out(p[t]))
            text = f["params"]["item"].replace("_", " ")
        else:
            body.polyline("PLAN-DEMOLISH", ring, True)
            text = "DEMOLISH"
        body.text("PLAN-LABEL", label_at, 1.2, text)
    head = _Dxf()
    head.g(999, "Proposed scheme " + proposal["name"] + " - " + (
        f"EPSG:{frame.epsg}, {HEIGHT_NOTE}" if frame.georeferenced else LOCAL_NOTE))
    head.g(0, "SECTION"); head.g(2, "HEADER")
    head.g(9, "$ACADVER"); head.g(1, "AC1009")
    head.g(9, "$INSUNITS"); head.g(70, 6)
    head.g(0, "ENDSEC")
    head.g(0, "SECTION"); head.g(2, "TABLES")
    head.g(0, "TABLE"); head.g(2, "LTYPE"); head.g(70, 1)
    head.g(0, "LTYPE"); head.g(2, "CONTINUOUS"); head.g(70, 0); head.g(3, "Solid line"); head.g(72, 65); head.g(73, 0); head.g(40, 0.0)
    head.g(0, "ENDTAB")
    head.g(0, "TABLE"); head.g(2, "LAYER"); head.g(70, len(LAYERS))
    for name, color in LAYERS.items():
        head.g(0, "LAYER"); head.g(2, name); head.g(70, 0); head.g(62, color); head.g(6, "CONTINUOUS")
    head.g(0, "ENDTAB"); head.g(0, "ENDSEC")
    head.g(0, "SECTION"); head.g(2, "ENTITIES")
    tail = ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(head.lines + body.lines + tail) + "\n"


def read_dxf_polygons(text):
    """Closed LWPOLYLINE / POLYLINE outlines as (layer, Nx2) pairs; everything else ignored."""
    tokens = text.replace("\r\n", "\n").split("\n")
    pairs = []
    for i in range(0, len(tokens) - 1, 2):
        try:
            pairs.append((int(tokens[i].strip()), tokens[i + 1].strip()))
        except ValueError:
            raise ExportError(400, "This does not look like an ASCII DXF file (binary DXF is not read).")
    out, i = [], 0
    while i < len(pairs):
        code, value = pairs[i]
        if code == 0 and value == "LWPOLYLINE":
            layer, flags, xs, ys = "", 0, [], []
            i += 1
            while i < len(pairs) and pairs[i][0] != 0:
                c, v = pairs[i]
                if c == 8: layer = v
                elif c == 70: flags = int(v)
                elif c == 10: xs.append(float(v))
                elif c == 20: ys.append(float(v))
                i += 1
            pts = np.column_stack([xs, ys[:len(xs)]]) if xs else np.zeros((0, 2))
            if len(pts) >= 3 and (flags & 1 or np.allclose(pts[0], pts[-1])):
                out.append((layer, pts))
            continue
        if code == 0 and value == "POLYLINE":
            layer, flags, pts = "", 0, []
            i += 1
            while i < len(pairs) and pairs[i][0] != 0:
                if pairs[i][0] == 8: layer = pairs[i][1]
                elif pairs[i][0] == 70: flags = int(pairs[i][1])
                i += 1
            while i < len(pairs) and pairs[i] == (0, "VERTEX"):
                i += 1
                x = y = None
                while i < len(pairs) and pairs[i][0] != 0:
                    if pairs[i][0] == 10: x = float(pairs[i][1])
                    elif pairs[i][0] == 20: y = float(pairs[i][1])
                    i += 1
                if x is not None and y is not None:
                    pts.append((x, y))
            pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            if len(pts) >= 3 and (flags & 1 or np.allclose(pts[0], pts[-1])):
                out.append((layer, pts))
            continue
        i += 1
    return out


# ------------------------------------------------------------------ 3D Tiles 1.1
def _shaded_colors(p, tris, color, opacity):
    light = np.array([0.35, 0.85, 0.4]) / np.linalg.norm([0.35, 0.85, 0.4])
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = 0.58 + 0.42 * np.abs(n @ light)
    rgb = np.clip(np.outer(shade, color), 0, 255)
    rgba = np.column_stack([rgb, np.full(len(rgb), 255 * opacity)]).astype(np.uint8)
    return np.repeat(rgba, 3, axis=0)


def tiles3d(proposal, evaluation, registry):
    """(zip bytes, tileset dict): tileset.json + one glTF (GLB) content, 3D Tiles 1.1."""
    from plan_glb import glb_bytes
    frame = OutFrame(registry)
    meshes, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
    for f, d in _visible(proposal, evaluation):
        if f["type"] == "clip":
            continue
        for k, mesh in enumerate(d["meshes"]):
            p, tris = _triangles(mesh)
            enu = frame.enu(p)
            lo, hi = np.minimum(lo, enu.min(axis=0)), np.maximum(hi, enu.max(axis=0))
            flat = enu[tris.ravel()]
            gl = np.column_stack([flat[:, 0], flat[:, 2], -flat[:, 1]])     # glTF is Y-up
            meshes.append({"name": f"{f['id']}:{mesh['part']}:{k}", "positions": gl,
                           "indices": np.arange(len(gl)),
                           "colors": _shaded_colors(p, tris, mesh["color"], mesh.get("opacity", 1.0))})
    if not meshes:
        raise ExportError(409, "The scheme has nothing to export as 3D Tiles yet.")
    centre, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.5)
    root = {"boundingVolume": {"box": centre.round(3).tolist() + [half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]},
            "geometricError": 0, "refine": "ADD", "content": {"uri": "proposal.glb"}}
    tileset = {"asset": {"version": "1.1", "generator": "Ground Control planning editor",
                         "tilesetVersion": f"{proposal['id']}@{proposal['revision']}"},
               "geometricError": float(np.linalg.norm(half) * 2), "root": root,
               "extras": {"scheme": proposal["name"], "status": proposals.status_text(proposal)}}
    if frame.georeferenced:
        root["transform"] = frame.ecef_from_enu().T.ravel().round(6).tolist()   # column-major
        tileset["extras"]["note"] = "placed at the scene's GPS origin; " + HEIGHT_NOTE
    else:
        tileset["extras"]["note"] = LOCAL_NOTE + "; no transform, so it is not placed on the globe"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("tileset.json", json.dumps(tileset, indent=1))
        archive.writestr("proposal.glb", glb_bytes(meshes, generator="Ground Control planning editor"))
    return buffer.getvalue(), tileset


# ------------------------------------------------------------------ cadastral import
RULE_KEYS = {"max_height_m": ("max_height_m", "max_height", "height_limit", "maxheight"),
             "max_fsi": ("max_fsi", "fsi", "far", "max_far"),
             "max_coverage_pct": ("max_coverage_pct", "coverage", "ground_coverage", "max_coverage"),
             "setback_m": ("setback_m", "setback", "min_setback"),
             "max_floors": ("max_floors", "floors", "storeys")}
NAME_KEYS = ("name", "plot_no", "plot", "parcel_id", "parcel", "khasra", "survey_no", "id", "label")


def _parcel_meta(props):
    lower = {str(k).lower(): v for k, v in (props or {}).items()}
    name = next((str(lower[k])[:100] for k in NAME_KEYS if lower.get(k) not in (None, "")), None)
    rules = {}
    for key, aliases in RULE_KEYS.items():
        for alias in aliases:
            value = lower.get(alias)
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    value = None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                rules[key] = float(value)
                break
    return name, rules


def _geojson_rings(doc):
    feats = doc.get("features") if doc.get("type") == "FeatureCollection" else [doc]
    for feat in feats or []:
        geom = feat.get("geometry") if feat.get("type") == "Feature" else feat
        props = feat.get("properties") if feat.get("type") == "Feature" else {}
        if not isinstance(geom, dict):
            continue
        if geom.get("type") == "Polygon":
            polys = [geom.get("coordinates") or []]
        elif geom.get("type") == "MultiPolygon":
            polys = geom.get("coordinates") or []
        else:
            continue
        for poly in polys:
            if poly and poly[0]:
                yield props, np.asarray([c[:2] for c in poly[0]], dtype=np.float64)


def _kml_rings(text):
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise ExportError(400, f"The KML file could not be read: {error}")
    for placemark in root.iter():
        if not placemark.tag.endswith("Placemark"):
            continue
        name = next((el.text for el in placemark if el.tag.endswith("name")), None)
        props = {"name": name} if name else {}
        for data in placemark.iter():
            if data.tag.endswith("Data") and data.get("name"):
                value = next((el.text for el in data if el.tag.endswith("value")), None)
                props[data.get("name")] = value
            if data.tag.endswith("SimpleData") and data.get("name"):
                props[data.get("name")] = data.text
        for boundary in placemark.iter():
            if boundary.tag.endswith("outerBoundaryIs"):
                coords = next((el.text for el in boundary.iter() if el.tag.endswith("coordinates")), "")
                pts = [tuple(float(v) for v in token.split(",")[:2]) for token in (coords or "").split() if "," in token]
                if len(pts) >= 3:
                    yield props, np.asarray(pts, dtype=np.float64)


def _epsg_from_geojson(doc):
    name = ((doc.get("crs") or {}).get("properties") or {}).get("name", "")
    match = re.search(r"EPSG:+(\d+)", str(name))
    return int(match.group(1)) if match else None


def to_lonlat(points, crs):
    """Any projected/geographic CRS (EPSG int or WKT) to WGS84 lon/lat through pyproj."""
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        raise ExportError(400, "Reading this coordinate system needs pyproj, which is not installed.")
    try:
        source = CRS.from_epsg(crs) if isinstance(crs, int) else CRS.from_wkt(crs) if "[" in str(crs) else CRS.from_user_input(crs)
    except Exception as error:  # noqa: BLE001 - pyproj raises its own types
        raise ExportError(400, f"The coordinate system could not be read: {error}")
    t = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
    pts = np.asarray(points, dtype=np.float64)
    lon, lat = t.transform(pts[:, 0], pts[:, 1])
    out = np.column_stack([lon, lat])
    if not np.isfinite(out).all():
        raise ExportError(400, "Some parcel vertices fall outside that coordinate system's area of use.")
    return out, source.name


def read_shapefile_zip(data):
    """Polygons and attributes of the first .shp in a zip, with its .prj WKT (or None).

    Reads shape types 5 (Polygon), 15 (PolygonZ) and 25 (PolygonM); every part is returned
    as a ring, and holes (counter-clockwise parts in the shapefile convention) are dropped.
    """
    import io
    import struct
    import zipfile
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ExportError(400, "A shapefile must be uploaded as a .zip holding the .shp, .dbf and .prj.")
    names = {n.lower(): n for n in archive.namelist()}
    shp = next((n for n in names if n.endswith(".shp")), None)
    if shp is None:
        raise ExportError(400, "The zip holds no .shp file.")
    stem = shp[:-4]
    buf = archive.read(names[shp])
    if len(buf) < 100 or struct.unpack(">i", buf[:4])[0] != 9994:
        raise ExportError(400, "The .shp file has no shapefile header.")
    shape_type = struct.unpack("<i", buf[32:36])[0]
    if shape_type not in (5, 15, 25):
        raise ExportError(400, f"Shape type {shape_type} is not a polygon layer (5, 15 or 25).")
    records, pos = [], 100
    while pos + 8 <= len(buf):
        _, length = struct.unpack(">ii", buf[pos:pos + 8])
        body = buf[pos + 8:pos + 8 + 2 * length]
        pos += 8 + 2 * length
        if len(body) < 44 or struct.unpack("<i", body[:4])[0] == 0:
            records.append([])
            continue
        n_parts, n_points = struct.unpack("<ii", body[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])) + [n_points]
        start = 44 + 4 * n_parts
        pts = np.frombuffer(body[start:start + 16 * n_points], dtype="<f8").reshape(-1, 2)
        rings = []
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            if len(ring) >= 4 and proposals.polygon_area(ring) < 0:        # clockwise = outer ring
                rings.append(ring[:-1] if np.allclose(ring[0], ring[-1]) else ring)
        records.append(rings)
    attrs = [{} for _ in records]
    dbf = names.get(stem + ".dbf")
    if dbf:
        d = archive.read(dbf)
        n_rec, header, rec_len = struct.unpack("<IHH", d[4:12])
        fields, off = [], 32
        while off < header - 1 and d[off] != 0x0D:
            fname = d[off:off + 11].split(b"\0")[0].decode("latin-1")
            fields.append((fname, chr(d[off + 11]), d[off + 16]))
            off += 32
        for k in range(min(n_rec, len(records))):
            row = d[header + k * rec_len:header + (k + 1) * rec_len]
            col = 1
            for fname, ftype, width in fields:
                raw = row[col:col + width].decode("latin-1").strip()
                col += width
                if ftype in "NF" and raw:
                    try:
                        attrs[k][fname] = float(raw)
                    except ValueError:
                        attrs[k][fname] = raw
                elif raw:
                    attrs[k][fname] = raw
    prj = names.get(stem + ".prj")
    wkt = archive.read(prj).decode("latin-1") if prj else None
    out = [(attrs[k], ring) for k, rings in enumerate(records) for ring in rings]
    return out, wkt


def parse_parcels(filename, text, registry, bounds, *, data=None, declared_crs=None):
    """(features, skipped): zone features in the viewer frame from a cadastral file.

    ``bounds`` is ((xmin, zmin), (xmax, zmax)) of the scene; parcels with no vertex within
    500 m of it are skipped, so a whole district file only brings in what is on the scan.
    """
    if len((text or "").encode("utf-8")) > MAX_IMPORT_BYTES or (data is not None and len(data) > MAX_IMPORT_BYTES):
        raise ExportError(413, "Cadastral files are limited to about 2 MB here: clip a district-wide file to the site first.")
    frame = OutFrame(registry)
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    rings, basis = [], ""
    if ext in ("geojson", "json"):
        try:
            doc = json.loads(text)
        except ValueError as error:
            raise ExportError(400, f"The GeoJSON file could not be read: {error}")
        epsg = _epsg_from_geojson(doc)
        raw = list(_geojson_rings(doc))
        if not raw:
            raise ExportError(400, "No Polygon or MultiPolygon parcels were found in this GeoJSON.")
        allpts = np.vstack([r for _, r in raw])
        local = str(doc.get("crs_note", "")).startswith("LOCAL")
        if epsg and 32601 <= epsg <= 32760:
            zone, hemi = epsg % 100, "N" if epsg < 32700 else "S"
            rings = [(p, frame.from_utm(r, zone, hemi)) for p, r in raw]
            basis = f"EPSG:{epsg} (UTM)"
        elif epsg and epsg not in (4326, 4979, 4258):
            lonlat_rings = [(p, to_lonlat(r, epsg)) for p, r in raw]
            rings = [(p, frame.from_lonlat(r)) for p, (r, _) in lonlat_rings]
            basis = f"EPSG:{epsg} ({lonlat_rings[0][1][1]}) through pyproj"
        elif not local and np.all(np.abs(allpts[:, 0]) <= 180) and np.all(np.abs(allpts[:, 1]) <= 90):
            # RFC 7946: an undeclared GeoJSON is WGS84. Only our own LOCAL export says otherwise.
            rings = [(p, frame.from_lonlat(r)) for p, r in raw]
            basis = "WGS84 longitude/latitude"
        elif frame.georeferenced and not local and np.all((allpts[:, 0] > 1e5) & (allpts[:, 0] < 9e5)):
            rings = [(p, frame.from_utm(r)) for p, r in raw]
            basis = f"UTM zone {frame.crs['zone']}{frame.crs['hemisphere']} (assumed from the scene)"
        else:
            rings = [(p, frame.from_local(r)) for p, r in raw]
            basis = "the scene's LOCAL frame (x east-ish, y north-ish, metres)"
    elif ext == "kml":
        rings = [(p, frame.from_lonlat(r)) for p, r in _kml_rings(text)]
        basis = "WGS84 longitude/latitude (KML)"
    elif ext == "dxf":
        raw = read_dxf_polygons(text)
        if not raw:
            raise ExportError(400, "No closed polylines were found in this DXF.")
        allpts = np.vstack([r for _, r in raw])
        if declared_crs:
            converted = [({"layer": layer}, to_lonlat(r, declared_crs)) for layer, r in raw]
            rings = [(p, frame.from_lonlat(r)) for p, (r, _) in converted]
            basis = f"{converted[0][1][1]} (declared) through pyproj"
        elif frame.georeferenced and np.abs(allpts).max() > 1e5:
            rings = [({"layer": layer}, frame.from_utm(r)) for layer, r in raw]
            basis = f"UTM zone {frame.crs['zone']}{frame.crs['hemisphere']} (assumed from the scene)"
        else:
            rings = [({"layer": layer}, frame.from_local(r)) for layer, r in raw]
            basis = "the scene's LOCAL frame (DXF x, y in metres)"
    elif ext == "zip":
        if data is None:
            raise ExportError(400, "Send the shapefile zip as base64.")
        raw, wkt = read_shapefile_zip(data)
        if not raw:
            raise ExportError(400, "No polygon parcels were found in this shapefile.")
        crs = declared_crs or wkt
        if crs:
            converted = [(p, to_lonlat(r, crs)) for p, r in raw]
            rings = [(p, frame.from_lonlat(r)) for p, (r, _) in converted]
            basis = f"shapefile in {converted[0][1][1]} through pyproj"
        else:
            rings = [(p, frame.from_local(r)) for p, r in raw]
            basis = "shapefile with no .prj: read as the scene's LOCAL frame"
    else:
        raise ExportError(400, "Import parcels from .geojson, .json, .kml, .dxf or a zipped shapefile (.zip).")
    (x0, z0), (x1, z1) = bounds
    features, skipped = [], []
    for k, (props, ring) in enumerate(rings):
        name, rules = _parcel_meta(props)
        label = name or f"Parcel {k + 1}"
        near = ((ring[:, 0] > x0 - 500) & (ring[:, 0] < x1 + 500) & (ring[:, 1] > z0 - 500) & (ring[:, 1] < z1 + 500)).any()
        if not near:
            skipped.append({"parcel": label, "reason": "outside the scanned area"})
            continue
        if len(features) >= MAX_PARCELS:
            skipped.append({"parcel": label, "reason": f"more than {MAX_PARCELS} parcels"})
            continue
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        try:
            clean = proposals.validate_feature({"type": "zone", "name": label[:120],
                                                "params": {"polygon": ring.tolist(), "rules": rules, "label": "Cadastral parcel"}})
        except proposals.ProposalError as error:
            skipped.append({"parcel": label, "reason": str(error)})
            continue
        features.append(clean)
    return features, skipped, basis
