"""Digital twin of a scanned site (TWN-02, TWN-04, TWN-05, TWN-07; CON-04).

* **Inventory with attribute cards (TWN-02).** Assets are what the scan measured:
  buildings and trees from the scene inventory, poles/masts from the obstacle list. Each has
  a stable id and measured facts (height, footprint, floors); owners add their own
  attributes (asset tag, owner, material, condition, notes), kept in
  ``work/<scene>/twin/attributes.json`` - never mixed with the measured facts.
* **Epochs (TWN-04 / CON-04).** Every scan of the same site (``project.json`` ``site``),
  ordered by capture time, with the change summary between neighbours when it has been
  computed. "Updating the twin" is flying again and appending an epoch; history is kept.
* **Surface mesh + LOD tiles (TWN-05).** The scan's measured top surface as a vertex-coloured
  triangle mesh (colours from the splats nearest the surface), and a two-level 3D Tiles 1.1
  quadtree (coarse root, four REPLACE children at full grid resolution), ENU->ECEF placed
  when the scene is georeferenced.
* **Engine package (TWN-07).** ``surface.glb`` (Y-up glTF with colours), ``surface.fbx``
  (geometry only - the minimal FBX writer carries no colours), ``tiles/``, the splat cloud
  in spatial chunks with an index for streaming, and a README that says which parts are
  measured and which are the photoreal presentation layer.
"""
import io
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import workspace_proposals as proposals

ATTRIBUTE_KEYS = ("name", "asset_tag", "owner", "use", "material", "condition", "built_year", "last_inspected", "notes")


# ------------------------------------------------------------------ inventory + attributes
def _attr_path(work):
    return Path(work) / "twin" / "attributes.json"


def load_attributes(work):
    try:
        data = json.loads(_attr_path(work).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def set_attributes(work, asset_id, values):
    if not isinstance(asset_id, str) or not asset_id or len(asset_id) > 80:
        raise proposals.ProposalError(400, "asset id is required")
    if not isinstance(values, dict):
        raise proposals.ProposalError(400, "attributes must be an object")
    clean = {}
    for key in ATTRIBUTE_KEYS:
        if key in values and values[key] not in (None, ""):
            clean[key] = str(values[key])[:2000 if key == "notes" else 120]
    data = load_attributes(work)
    if clean:
        data[asset_id] = {**clean, "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    else:
        data.pop(asset_id, None)
    path = _attr_path(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
    return data


def inventory(work, ground, *, pole_min_height_m=5.0):
    import mission_analysis
    existing = proposals.existing_inventory(work, ground)
    assets = []
    for b in existing.get("buildings", []):
        assets.append({"id": b["id"], "kind": "building", "position": b["centre"],
                       "measured": {"height_m": b["height_m"], "footprint_m2": b["area_m2"],
                                    "floors_estimate": b.get("floors_estimate"),
                                    "size_m": [round(v, 2) for v in b["size"]]},
                       "basis": existing.get("basis")})
    for t in existing.get("trees", []):
        assets.append({"id": t["id"], "kind": "tree", "position": t["centre"],
                       "measured": {"height_m": t["height_m"], "crown_m2": t["area_m2"]},
                       "basis": existing.get("basis")})
    terrain = mission_analysis.Terrain(ground.grid)
    obs = mission_analysis.obstacles(terrain, min_height_m=pole_min_height_m)
    for k, o in enumerate(o for o in obs["obstacles"] if o["kind"] == "pole or mast"):
        assets.append({"id": f"pole-{round(o['position'][0], 1)}_{round(o['position'][1], 1)}", "kind": "pole",
                       "position": o["position"], "measured": {"height_m": o["height_m"], "footprint_m2": o["area_m2"]},
                       "basis": obs["basis"]})
    try:
        import inspection
        register = inspection.list_annotations(work)["items"]
    except Exception:  # noqa: BLE001 - the register is optional
        register = []
    attrs = load_attributes(work)
    for a in assets:
        a["attributes"] = attrs.get(a["id"], {})
        x, z = a["position"]
        near = [r for r in register if math.hypot(r["position"][0] - x, r["position"][2] - z) < 8.0]
        a["defects_open"] = sum(1 for r in near if r["status"] != "closed")
        a["worst_severity"] = max((r["severity"] for r in near if r["status"] != "closed"), default=None)
    orphan = sorted(set(attrs) - {a["id"] for a in assets})
    return {"assets": assets, "counts": {k: sum(1 for a in assets if a["kind"] == k) for k in ("building", "tree", "pole")},
            "orphaned_attributes": orphan,
            "notes": ["Measured facts come from the scan and are recomputed each time; attributes are what people "
                      "typed and are kept per asset id.",
                      "Asset ids come from the inventory's clustering: rebuilding the scan can renumber them, and "
                      "attributes whose id no longer exists are listed as orphaned, not dropped."]}


# ------------------------------------------------------------------ epochs
def epochs(work):
    work = Path(work)
    try:
        me = json.loads((work / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        me = {}
    site = me.get("site")
    rows = []
    candidates = [p for p in work.parent.iterdir() if (p / "viewer_assets" / "collision.json").is_file()] if site else [work]
    for other in candidates:
        try:
            info = json.loads((other / "project.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        if site and info.get("site") != site:
            continue
        change = None
        cached = other / "ops" / "last" / "change.json"
        if cached.is_file():
            try:
                report = json.loads(cached.read_text())["report"]
                change = {"against": report.get("epochs", {}).get("before"), "gain_m3": report["volume"]["gain_m3"],
                          "loss_m3": report["volume"]["loss_m3"], "regions": len(report["regions"])}
            except (OSError, ValueError, KeyError):
                change = None
        rows.append({"scene": other.name, "name": info.get("name", other.name), "captured_at": info.get("captured_at"),
                     "current": other.resolve() == work.resolve(), "change": change})
    rows.sort(key=lambda r: (r["captured_at"] is None, r["captured_at"] or "", r["scene"]))
    return {"site": site, "epochs": rows,
            "note": None if site else "This scan has no site id in project.json, so no other epochs are linked to it."}


# ------------------------------------------------------------------ surface mesh
def cell_colors(grid, points=None, rgb=None):
    """RGB per grid cell: the splats within 0.6 m below the top surface, else neutral grey."""
    nz, nx = int(grid["nz"]), int(grid["nx"])
    colours = np.full((nz, nx, 3), 150, dtype=np.float64)
    if points is None or not len(points):
        return colours.astype(np.uint8)
    col = np.floor((points[:, 0] - grid["ox"]) / grid["cell"]).astype(int)
    row = np.floor((points[:, 2] - grid["oz"]) / grid["cell"]).astype(int)
    ok = (col >= 0) & (col < nx) & (row >= 0) & (row < nz)
    col, row, pts, cc = col[ok], row[ok], points[ok], np.asarray(rgb)[ok].astype(np.float64)
    top = np.maximum(grid["top"], grid["floor"]) if grid.get("top") is not None else grid["floor"]
    near = pts[:, 1] >= top[row, col] - 0.6
    flat = row[near] * nx + col[near]
    count = np.bincount(flat, minlength=nz * nx)
    for k in range(3):
        s = np.bincount(flat, weights=cc[near, k], minlength=nz * nx)
        channel = colours[:, :, k].ravel()
        channel[count > 0] = s[count > 0] / count[count > 0]
        colours[:, :, k] = channel.reshape(nz, nx)
    return np.clip(colours, 0, 255).astype(np.uint8)


def surface_mesh(grid, colours, *, stride=1, window=None):
    """Triangles over supported cells of the top surface (viewer frame). window = (r0, r1, c0, c1)."""
    top = np.maximum(grid["top"], grid["floor"]) if grid.get("top") is not None else grid["floor"]
    sup = np.asarray(grid["supported"], bool)
    r0, r1, c0, c1 = window or (0, top.shape[0], 0, top.shape[1])
    rows = np.arange(r0, r1, stride)
    cols = np.arange(c0, c1, stride)
    if len(rows) < 2 or len(cols) < 2:
        return None
    R, C = np.meshgrid(rows, cols, indexing="ij")
    x = grid["ox"] + (C + 0.5) * grid["cell"]
    z = grid["oz"] + (R + 0.5) * grid["cell"]
    y = top[R, C]
    ok = sup[R, C]
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    rgba = np.column_stack([colours[R, C].reshape(-1, 3), np.full(R.size, 255)]).astype(np.uint8)
    h, w = R.shape
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    quad_ok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    tris = np.concatenate([np.stack([a, c, b], -1)[quad_ok], np.stack([b, c, d], -1)[quad_ok]])
    if not len(tris):
        return None
    used = np.unique(tris)
    remap = -np.ones(len(positions), int)
    remap[used] = np.arange(len(used))
    return {"positions": positions[used], "colors": rgba[used], "indices": remap[tris].reshape(-1)}


def _to_gl(frame, positions):
    enu = frame.enu(positions)
    return enu, np.column_stack([enu[:, 0], enu[:, 2], -enu[:, 1]])


def tileset(grid, colours, registry):
    """(files dict, tileset dict): a coarse root and four full-resolution children."""
    from plan_exports import OutFrame
    from plan_glb import glb_bytes
    frame = OutFrame(registry)
    nz, nx = int(grid["nz"]), int(grid["nx"])
    coarse_stride = max(2, int(math.ceil(max(nz, nx) / 128)))
    files, children = {}, []

    def tile(mesh, name, error):
        enu, gl = _to_gl(frame, mesh["positions"])
        lo, hi = enu.min(0), enu.max(0)
        centre, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.5)
        files[name] = glb_bytes([{"name": name, "positions": gl, "indices": mesh["indices"], "colors": mesh["colors"]}],
                                generator="Ground Control twin")
        return {"boundingVolume": {"box": centre.round(3).tolist() + [half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]},
                "geometricError": error, "content": {"uri": name}}

    root_mesh = surface_mesh(grid, colours, stride=coarse_stride)
    if root_mesh is None:
        raise ValueError("the scene has no supported surface to tile")
    root = tile(root_mesh, "root.glb", grid["cell"] * coarse_stride * 4)
    root["refine"] = "REPLACE"
    for k, (r0, r1, c0, c1) in enumerate(((0, nz // 2 + 1, 0, nx // 2 + 1), (0, nz // 2 + 1, nx // 2, nx),
                                          (nz // 2, nz, 0, nx // 2 + 1), (nz // 2, nz, nx // 2, nx))):
        mesh = surface_mesh(grid, colours, stride=1, window=(r0, r1, c0, c1))
        if mesh is not None:
            children.append(tile(mesh, f"tile_{k}.glb", 0.0))
    root["children"] = children
    ts = {"asset": {"version": "1.1", "generator": "Ground Control twin"},
          "geometricError": root["geometricError"] * 4, "root": root,
          "extras": {"status": "measured top surface of the scan (2.5D); unobserved cells are holes, not filled",
                     "coarse_stride_cells": coarse_stride}}
    if frame.georeferenced:
        root["transform"] = frame.ecef_from_enu().T.ravel().round(6).tolist()
    else:
        ts["extras"]["note"] = "local scene: no transform, not placed on the globe"
    files["tileset.json"] = json.dumps(ts, indent=1).encode()
    return files, ts


def splat_chunks(work, *, chunk_m=50.0):
    """The splat PLY cut into spatial chunks (all properties kept) + an index for streaming."""
    from plyfile import PlyData, PlyElement
    path = Path(work) / "viewer_assets" / "scene.ply"
    if not path.is_file():
        return {}, None
    v = PlyData.read(str(path))["vertex"].data
    keys = np.floor(np.column_stack([v["x"], v["z"]]) / chunk_m).astype(int)
    files, index = {}, []
    for key in np.unique(keys, axis=0):
        pick = (keys[:, 0] == key[0]) & (keys[:, 1] == key[1])
        sub = v[pick]
        buf = io.BytesIO()
        PlyData([PlyElement.describe(np.asarray(sub), "vertex")]).write(buf)
        name = f"splats/chunk_{key[0]}_{key[1]}.ply"
        files[name] = buf.getvalue()
        index.append({"file": name, "count": int(pick.sum()),
                      "bounds_xz": [float(key[0] * chunk_m), float(key[1] * chunk_m), float((key[0] + 1) * chunk_m), float((key[1] + 1) * chunk_m)]})
    return files, {"chunk_m": chunk_m, "frame": "viewer (y up, -z scene north)", "chunks": index}


def package(work, grid, registry, *, points=None, rgb=None, include_splats=True):
    """Zip bytes of the engine package and its manifest."""
    import survey_formats
    import tempfile
    colours = cell_colors(grid, points, rgb)
    mesh = surface_mesh(grid, colours, stride=1)
    if mesh is None:
        raise ValueError("the scene has no supported surface to export")
    from plan_exports import OutFrame
    from plan_glb import glb_bytes
    frame = OutFrame(registry)
    _, gl = _to_gl(frame, mesh["positions"])
    files = {"surface.glb": glb_bytes([{"name": "surface", "positions": gl, "indices": mesh["indices"], "colors": mesh["colors"]}],
                                      generator="Ground Control twin")}
    with tempfile.TemporaryDirectory() as tmp:
        survey_formats.write_fbx(gl, Path(tmp) / "surface.fbx", triangles=mesh["indices"].reshape(-1, 3), name="surface")
        files["surface.fbx"] = (Path(tmp) / "surface.fbx").read_bytes()
    tiles, ts = tileset(grid, colours, registry)
    files.update({f"tiles/{k}": v for k, v in tiles.items()})
    chunk_index = None
    if include_splats:
        chunks, chunk_index = splat_chunks(work)
        files.update(chunks)
        if chunk_index:
            files["splats/index.json"] = json.dumps(chunk_index, indent=1).encode()
    manifest = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "frame": "ENU metres at the GPS origin (glTF Y-up)" if frame.georeferenced else "local scene metres (x, -z, y), glTF Y-up",
                "files": sorted(files), "triangles": int(len(mesh["indices"]) // 3),
                "measured": ["surface.glb", "surface.fbx", "tiles/"],
                "presentation": ["splats/"] if chunk_index else []}
    readme = ("Digital twin package\n\n"
              "surface.glb / surface.fbx / tiles/ - the scan's measured top surface (2.5D heightfield), "
              "vertex colours from the splats. Unobserved cells are holes. FBX carries geometry only.\n"
              "splats/ - the Gaussian-splat presentation layer in 50 m chunks with index.json; photoreal, not a "
              "measurement.\n" + ("Coordinates: " + manifest["frame"] + "\n"))
    files["README.txt"] = readme.encode()
    files["manifest.json"] = json.dumps(manifest, indent=1).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue(), manifest
