"""Place furniture on a scanned floor, without claiming unmeasured clearances.

The scene's ``viewer_assets`` carries a per-column floor height (``ground.f32``),
a top surface (``heights.f32``) and a supported/unsupported mask (``coverage.u8``)
on the same grid the walk-mode character uses. This module snaps and orients a
dropped item, then checks sampled floor support. A top surface is not evidence
of a ceiling: headroom remains unknown. Distances describe coverage edges, not
walls, and no furniture-to-furniture collision or clearance check is performed.

Coordinates are the viewer's Y-up frame (x, y, z); the grid is indexed exactly
as ``viewer/pc.js`` ``groundHF`` indexes it, so the box the backend places and
the box the character collides with are the same box. The output record reuses
the ``objects.json`` box schema (``center_xz`` / ``center_y`` / ``size`` /
``yaw_deg``) so the viewer renders a placement with the collider path it already
trusts.

Furniture sizes are nominal catalogue dimensions in metres — a planning aid, not
surveyed ground truth. The fit verdict is only as good as the scan's coverage
grid, and it says so in ``reason``.

A glTF/GLB may also be imported as an item. Unlike the catalogue, an imported
model carries its own real size: glTF is authored in metres and mandates that
every POSITION accessor declare ``min``/``max``, so the bounding box is read from
the file and pushed through the node graph's transforms — never guessed. A model
whose bounds cannot be read is refused, not defaulted to one metre — and so is one
whose vertices need a decoder this viewer does not ship (Draco, Meshopt) or whose
GLB chunks are out of the order glTF mandates, because the viewer would reject that
file and the user would be left looking at a coloured box. Scene scale
is anchored, not measured, so an imported item carries the same scale caveat the
rest of the product uses: a real 2.4 m sofa sitting in a scene whose metre is a
guess is only that wide if the guess is right.
"""
import json
import math
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# Nominal catalogue dimensions, metres, as [length, height, depth]. length is the
# major (yaw) axis, depth the minor axis, height vertical. Deliberately ordinary
# numbers a shopper would recognise from a spec sheet.
LIBRARY = {
    "sofa":          {"label": "Sofa (3-seat)",   "size": [2.00, 0.85, 0.90]},
    "armchair":      {"label": "Armchair",        "size": [0.85, 0.80, 0.90]},
    "coffee_table":  {"label": "Coffee table",    "size": [1.10, 0.45, 0.60]},
    "dining_table":  {"label": "Dining table",    "size": [1.60, 0.75, 0.90]},
    "dining_chair":  {"label": "Dining chair",    "size": [0.45, 0.90, 0.50]},
    "bed_double":    {"label": "Double bed",      "size": [2.00, 0.55, 1.60]},
    "desk":          {"label": "Desk",            "size": [1.20, 0.75, 0.60]},
    "wardrobe":      {"label": "Wardrobe",        "size": [1.00, 2.05, 0.60]},
    "bookshelf":     {"label": "Bookshelf",       "size": [0.80, 1.80, 0.30]},
    "side_table":    {"label": "Side table",      "size": [0.50, 0.55, 0.50]},
    "tv_stand":      {"label": "TV stand",        "size": [1.40, 0.50, 0.40]},
    "stool":         {"label": "Stool",           "size": [0.35, 0.45, 0.35]},
}

# Sample an N x N grid across the rotated rectangle; this is not a continuous
# footprint/obstacle intersection test.
_FOOTPRINT_SAMPLES = 5
# Proximity to unsupported coverage, not a measured wall or furniture clearance.
_TIGHT_M = 0.30
# Floor suitability requires a supported centre and this fraction of samples.
# Scan gaps stay unsupported; any off-grid footprint sample rejects the drop.
_MIN_SUPPORT = 0.6

# Cache at most eight parsed grids, versioned by the header and every payload.
_GRID_CACHE = {}
_GRID_CACHE_LIMIT = 8


# ---- imported glTF / GLB models ----------------------------------------------
# The only honest metre a model can have is one it states itself. glTF declares
# its unit as metres and requires every POSITION accessor to carry min/max, so
# the true bounding box comes from those accessors transformed by the node graph
# — a real size, or a refusal. A model never gets an inferred 1 metre.
GLB_MAGIC = b"glTF"
GLB_CHUNK_JSON = 0x4E4F534A
GLB_CHUNK_BIN = 0x004E4942
# A placeable item is one self-contained asset. Anything past these limits is a
# scanned scene, not a sofa, or a decompression trap; a JSON description this big
# is a hostile-allocation vector and is refused before it is parsed.
MODEL_MAX_BYTES = 256 << 20
MODEL_JSON_MAX_BYTES = 32 << 20
# Keep the accepted extent inside the placement engine's own size bound (20 m),
# so every later move/rotate/resize write that carries the size back stays valid.
MODEL_MIN_M, MODEL_MAX_M = 0.001, 20.0
MODEL_MAX_NODES = 20000
MODEL_MAX_MESHES = 20000
MODEL_MAX_ACCESSORS = 200000
MODEL_MAX_STEPS = 400000
MODEL_MAX_DEPTH = 200
# Geometry the shipped viewer cannot turn into vertices. Draco needs a WASM decoder
# that this build does not carry (the runtime fetch fails), and the bundled PlayCanvas
# has no Meshopt decoder at all. Both file shapes still declare honest min/max
# bounds, so a bounds reader alone waves them in — and the user gets a coloured box
# where their armchair should be. Refusing is the only outcome that matches the
# promise the import message makes.
MODEL_UNDECODABLE_PRIMITIVE_EXT = ("KHR_draco_mesh_compression", "EXT_meshopt_compression")
MODEL_UNDECODABLE_BUFFERVIEW_EXT = ("EXT_meshopt_compression",)


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and -1e308 <= value <= 1e308 and math.isfinite(value))


def _finite_triplet(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(_finite(v) for v in value))


def _gltf_document(data, filename):
    """Return (json_dict, embedded_bin_bytes) from a GLB or a .gltf, or raise.

    Only self-contained files are read; a .gltf that names an external buffer
    file is refused for traversal/aliasing reasons, so no buffer path is opened.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("The uploaded model is empty.")
    if len(data) > MODEL_MAX_BYTES:
        raise ValueError(f"A model file must be at most {MODEL_MAX_BYTES >> 20} MiB.")
    data = bytes(data)
    if len(data) >= 12 and data[:4] == GLB_MAGIC:
        version = int.from_bytes(data[4:8], "little")
        total = int.from_bytes(data[8:12], "little")
        if version != 2:
            raise ValueError(f"Only glTF 2.0 is supported; this .glb declares version {version}.")
        if total <= 12 or total > len(data):
            raise ValueError("The .glb length does not match its bytes — the file is truncated or corrupt.")
        offset, document, payload_bin = 12, None, b""
        seen = 0
        while offset + 8 <= total:
            length = int.from_bytes(data[offset:offset + 4], "little")
            kind = int.from_bytes(data[offset + 4:offset + 8], "little")
            offset += 8
            if length < 0 or length % 4 or offset + length > total:
                raise ValueError("A .glb chunk has an invalid or unaligned length.")
            chunk = data[offset:offset + length]
            offset += length
            seen += 1
            # glTF 2.0: "Chunks MUST appear in exactly the order given" — the JSON
            # chunk MUST be first, the BIN chunk second. This is not pedantry: the
            # viewer's own glTF parser rejects a file that breaks the order, so
            # accepting one here would import an item that silently renders as a box.
            if seen == 1 and kind != GLB_CHUNK_JSON:
                raise ValueError("A .glb must hold its JSON chunk first and its binary chunk second; "
                                 "this one starts with another chunk, and the viewer cannot render it.")
            if kind == GLB_CHUNK_JSON and document is None:
                if length > MODEL_JSON_MAX_BYTES:
                    raise ValueError("The model's JSON description is implausibly large.")
                try:
                    document = json.loads(chunk.decode("utf-8"))
                except (ValueError, UnicodeError):
                    raise ValueError("The .glb JSON chunk is not valid UTF-8 JSON.")
            elif kind == GLB_CHUNK_BIN and not payload_bin:
                payload_bin = chunk
            # Any later chunk has an unknown type, which the spec says readers MUST
            # ignore; ignoring it keeps this importer and the viewer in agreement.
        if document is None:
            raise ValueError("The .glb has no JSON chunk, so its geometry cannot be read.")
        return document, payload_bin
    if str(filename).lower().endswith(".glb"):
        raise ValueError("A .glb must begin with the glTF binary header; this file is not a valid .glb.")
    try:
        document = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise ValueError("The model is not a readable .glb or .gltf file.")
    return document, b""


def _reject_external_buffers(document):
    """Refuse any geometry stored outside the file: external, absolute, self- or
    traversal-referenced URIs alike. Only embedded .glb bins and data: URIs pass.
    """
    for index, buf in enumerate(document.get("buffers") or []):
        if not isinstance(buf, dict):
            raise ValueError(f"The model's buffer {index} is malformed.")
        uri = buf.get("uri")
        if uri is None:
            continue                                    # the .glb BIN chunk
        if isinstance(uri, str) and uri.startswith("data:"):
            continue                                    # embedded, safe
        traversal = (not isinstance(uri, str) or ".." in uri or "\\" in uri
                     or uri.startswith("/") or (len(uri) > 1 and uri[1] == ":"))
        what = ("a path-traversing or absolute buffer URI" if traversal
                else "an external buffer file")
        raise ValueError(f"The model loads {what} for its geometry; import a single "
                         "self-contained .glb, or a .gltf with buffers embedded as data URIs.")


def _reject_undecodable_geometry(document):
    """Refuse a model whose vertices only a decoder this viewer does not have can
    produce. Visual-only extensions (materials, texture transforms) are untouched:
    they change how a model looks, never whether it can be placed and bumped into.
    """
    def used(extensions, names):
        if not isinstance(extensions, dict):
            return None
        for name in names:
            if name in extensions:
                return name
        return None

    for mesh in (document.get("meshes") or []):
        if not isinstance(mesh, dict):
            continue
        for prim in (mesh.get("primitives") or []):
            if not isinstance(prim, dict):
                continue
            name = used(prim.get("extensions"), MODEL_UNDECODABLE_PRIMITIVE_EXT)
            if name:
                raise ValueError(f"This model's geometry is compressed with {name}, which this viewer "
                                 "cannot decode; export it again without compression and import that file.")
    for view in (document.get("bufferViews") or []):
        if not isinstance(view, dict):
            continue
        name = used(view.get("extensions"), MODEL_UNDECODABLE_BUFFERVIEW_EXT)
        if name:
            raise ValueError(f"This model's vertex data is compressed with {name}, which this viewer "
                             "cannot decode; export it again without compression and import that file.")


def _mat4_local(node):
    """A node's local transform as a 4x4 row-major matrix, or raise on non-finite."""
    matrix = node.get("matrix")
    if matrix is not None:
        if (not isinstance(matrix, (list, tuple)) or len(matrix) != 16
                or not all(_finite(v) for v in matrix)):
            raise ValueError("A model node carries a malformed or non-finite transform matrix.")
        return np.asarray(matrix, dtype=np.float64).reshape(4, 4).T
    t = node.get("translation", [0.0, 0.0, 0.0])
    r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = node.get("scale", [1.0, 1.0, 1.0])
    for name, value, size in (("translation", t, 3), ("rotation", r, 4), ("scale", s, 3)):
        if (not isinstance(value, (list, tuple)) or len(value) != size
                or not all(_finite(v) for v in value)):
            raise ValueError(f"A model node carries a malformed or non-finite {name}.")
    x, y, z, w = r
    norm = x * x + y * y + z * z + w * w
    rot = np.eye(3)
    if norm > 0:
        k = 2.0 / norm
        wx, wy, wz = k * w * x, k * w * y, k * w * z
        xx, xy, xz = k * x * x, k * x * y, k * x * z
        yy, yz, zz = k * y * y, k * y * z, k * z * z
        rot = np.array([[1 - (yy + zz), xy - wz, xz + wy],
                        [xy + wz, 1 - (xx + zz), yz - wx],
                        [xz - wy, yz + wx, 1 - (xx + yy)]])
    local = np.eye(4)
    local[:3, :3] = rot * np.array([s[0], s[1], s[2]])   # column scale (R @ S)
    local[:3, 3] = t
    return local


def _accessor_box(accessors, index):
    """The (min, max) of a POSITION accessor — read from the file, never decoded.

    glTF mandates min/max on POSITION, so the true extent is stated, not computed
    from raw buffer bytes; a file that omits or falsifies it gets no guessed size.
    """
    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(accessors)):
        raise ValueError("A mesh primitive references a POSITION accessor that does not exist.")
    acc = accessors[index]
    if not isinstance(acc, dict):
        raise ValueError("A POSITION accessor is malformed.")
    lo, hi = acc.get("min"), acc.get("max")
    if not (_finite_triplet(lo) and _finite_triplet(hi)):
        raise ValueError("The model's POSITION accessor has no finite min/max bounds, so its "
                         "real size cannot be read; it will not be placed at a guessed size.")
    if any(lo[i] > hi[i] for i in range(3)):
        raise ValueError("A POSITION accessor's min exceeds its max.")
    return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)


def _model_aabb(document):
    """The model-space world AABB (metres) over every positioned primitive."""
    accessors = document.get("accessors") or []
    meshes = document.get("meshes") or []
    nodes = document.get("nodes") or []
    for name, value in (("accessors", accessors), ("meshes", meshes), ("nodes", nodes)):
        if not isinstance(value, list):
            raise ValueError(f"The model's '{name}' is not a list.")
    if len(nodes) > MODEL_MAX_NODES or len(meshes) > MODEL_MAX_MESHES or len(accessors) > MODEL_MAX_ACCESSORS:
        raise ValueError("The model declares more parts than this importer will walk.")

    childed = set()
    for node in nodes:
        if isinstance(node, dict):
            for child in (node.get("children") or []):
                if isinstance(child, int) and not isinstance(child, bool):
                    childed.add(child)
    roots = None
    scenes = document.get("scenes")
    if isinstance(scenes, list) and scenes:
        index = document.get("scene")
        index = index if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(scenes) else 0
        scene = scenes[index]
        if isinstance(scene, dict) and isinstance(scene.get("nodes"), list):
            roots = [n for n in scene["nodes"] if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < len(nodes)]
    if roots is None:
        roots = [i for i in range(len(nodes)) if i not in childed]

    gmin = np.array([math.inf] * 3)
    gmax = np.array([-math.inf] * 3)
    found = False
    steps = 0
    axis = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)

    def walk(index, parent, path):
        nonlocal found, steps
        if index in path:
            raise ValueError("The model's node graph is self-referencing (a node contains itself).")
        if steps > MODEL_MAX_STEPS or len(path) > MODEL_MAX_DEPTH:
            raise ValueError("The model's node graph is too deep or too broad to import safely.")
        steps += 1
        node = nodes[index]
        if not isinstance(node, dict):
            return
        world = parent @ _mat4_local(node)
        mesh = node.get("mesh")
        if isinstance(mesh, int) and not isinstance(mesh, bool) and 0 <= mesh < len(meshes):
            definition = meshes[mesh]
            for prim in (definition.get("primitives") or []) if isinstance(definition, dict) else []:
                if not isinstance(prim, dict):
                    continue
                attributes = prim.get("attributes")
                position = attributes.get("POSITION") if isinstance(attributes, dict) else None
                lo, hi = _accessor_box(accessors, position)
                corner = lo * (1 - axis) + hi * axis
                transformed = corner @ world[:3, :3].T + world[:3, 3]
                if not np.all(np.isfinite(transformed)):
                    raise ValueError("Model geometry transformed to a non-finite position.")
                gmin[:] = np.minimum(gmin, transformed.min(axis=0))
                gmax[:] = np.maximum(gmax, transformed.max(axis=0))
                found = True
        next_path = path | {index}
        for child in (node.get("children") or []):
            if isinstance(child, int) and not isinstance(child, bool) and 0 <= child < len(nodes):
                walk(child, world, next_path)

    identity = np.eye(4)
    for root in roots:
        walk(root, identity, frozenset())
    if not found:
        raise ValueError("The model has no positioned geometry, so there is nothing to place.")
    size = gmax - gmin
    for v in size:
        if not np.isfinite(v):
            raise ValueError("The model's computed bounds are not finite.")
        if v < MODEL_MIN_M:
            raise ValueError(f"The model is degenerate ({v:.4f} m on an axis), too thin to be a real-scale item.")
        if v > MODEL_MAX_M:
            raise ValueError(f"The model spans {v:.1f} m — beyond the {MODEL_MAX_M:.0f} m an importable item may occupy.")
    return {"size": [round(float(v), 4) for v in size], "min": [round(float(v), 4) for v in gmin],
            "max": [round(float(v), 4) for v in gmax]}


def read_model_bounds(data, filename):
    """The model's true metre size [length, height, depth] + AABB, or raise a clear
    ValueError a caller can surface. Never invents a size it could not read."""
    document, _ = _gltf_document(data, filename)
    if not isinstance(document, dict):
        raise ValueError("The model description is not a JSON object.")
    _reject_external_buffers(document)
    _reject_undecodable_geometry(document)
    return _model_aabb(document)


def _frame_scale(scene_dir):
    """(status, source) read the same way ``workspace_api.summary`` reports scale.

    metric only for an AR pose-prior; estimated when a height/speed was assumed;
    relative otherwise. Imported items inherit this caveat verbatim.
    """
    try:
        frame = json.loads((Path(scene_dir) / "frame.json").read_text(encoding="utf-8"),
                           parse_constant=lambda _: None)
    except (ValueError, OSError):
        frame = {}
    if not isinstance(frame, dict):
        frame = {}
    source = frame.get("scale_source") if isinstance(frame.get("scale_source"), str) else "no scale reference"
    value = frame.get("scale_m_per_unit")
    status = "relative"
    if _finite(value) and value > 0:
        status = "metric" if source == "AR pose-prior metric path" else "estimated"
    return status, source


def models_path(scene_dir):
    return Path(scene_dir) / "models" / "index.json"


def read_models(scene_dir):
    """The scene's imported-model records, or [] if there are none / unreadable."""
    path = models_path(scene_dir)
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    except (ValueError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)
            and _finite_triplet(r.get("size")) and all(v > 0 for v in r["size"])]


def model_spec(scene_dir, item):
    for entry in read_models(scene_dir):
        if entry.get("id") == item:
            return entry
    return None


def scene_library(scene_dir):
    """The catalogue plus this scene's imported models, as a JSON-safe list."""
    out = library()
    for entry in read_models(scene_dir):
        out.append({"item": entry["id"], "label": entry.get("label") or "Imported model",
                    "size": list(entry["size"]), "footprint_m2": round(entry["size"][0] * entry["size"][2], 3),
                    "imported": True, "model": model_descriptor(entry, scene_dir)})
    return out


def model_descriptor(entry, scene_dir):
    """What the viewer and dock need to render and honestly caption an imported item."""
    status, source = _frame_scale(scene_dir)
    length, height, depth = entry["size"]
    return {"id": entry["id"], "file": entry.get("file"), "source": entry.get("source", "gltf"),
            "size": list(entry["size"]), "scale_status": status, "scale_source": source,
            "true_size_note": f"true size {length:.2f} × {height:.2f} × {depth:.2f} m, read from the model"}


def _scale_caveat(status, source):
    if status == "metric":
        return "model size is true (read from file); scene scale is metric"
    if status == "estimated":
        return (f"model size is true (read from file), but this scene's scale is estimated "
                f"({source}), not measured, so its on-scene size is only as accurate as that scale")
    return (f"model size is true (read from file), but this scene has only relative scale "
            f"({source}); treat its size as nominal, not measured")


def library():
    """The furniture catalogue as a JSON-safe list with footprint areas."""
    out = []
    for key, spec in LIBRARY.items():
        length, height, depth = spec["size"]
        out.append({"item": key, "label": spec["label"], "size": list(spec["size"]),
                    "footprint_m2": round(length * depth, 3)})
    return out


def item_spec(item, scene_dir=None):
    """The catalogue entry for ``item``, or an imported model's real spec, or raise.

    An unknown item with no matching imported model is refused — the caller can
    never place a piece whose size it invented.
    """
    spec = LIBRARY.get(item)
    if isinstance(spec, dict):
        return spec
    if scene_dir is not None:
        entry = model_spec(scene_dir, item)
        if entry is not None:
            return {"label": entry.get("label") or "Imported model", "size": list(entry["size"]),
                    "imported": True, "model": entry}
    raise ValueError(f"Unknown furniture item {item!r}.")


def _grid(scene_dir):
    """Load (and cache) the scene grid: floor, top surface, coverage, header."""
    va = Path(scene_dir) / "viewer_assets"
    col_path = va / "collision.json"
    if not col_path.is_file():
        raise ValueError("The scene has no collision grid to place furniture on.")
    signatures = []
    for name in ("collision.json", "ground.f32", "heights.f32", "coverage.u8"):
        try:
            stat = (va / name).stat()
            signatures.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
        except FileNotFoundError:
            signatures.append(None)
    key = (str(va.resolve()), tuple(signatures))
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    col = json.loads(col_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    nx, nz = int(col["nx"]), int(col["nz"])
    cell = float(col["cell"])
    ox, oz = (float(v) for v in col["origin_xz"])
    if nx <= 0 or nz <= 0 or not (cell > 0):
        raise ValueError("The scene's collision grid is unusable.")

    def read(name, dtype):
        path = va / name
        if not path.is_file():
            return None
        arr = np.fromfile(path, dtype=dtype)
        if arr.size != nx * nz:
            return None
        return arr.reshape(nz, nx).astype(np.float64)

    floor = read("ground.f32", "<f4")
    if floor is None:
        floor = read("heights.f32", "<f4")
    top = read("heights.f32", "<f4")
    cov_path = va / "coverage.u8"
    cover = np.fromfile(cov_path, dtype=np.uint8).reshape(nz, nx) if cov_path.is_file() and cov_path.stat().st_size == nx * nz else None
    if floor is None:
        raise ValueError("The scene has no floor heightfield to snap furniture to.")
    supported = (cover > 0) if cover is not None else np.isfinite(floor)
    # Preserve coverage holes: interpolating a height does not measure support.
    # This distance is to unsupported coverage or the grid edge, not to a wall.
    padded = np.pad(supported, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * cell
    grid = {"nx": nx, "nz": nz, "cell": cell, "ox": ox, "oz": oz,
            "floor": floor, "top": top, "supported": supported, "clearance": clearance}
    _GRID_CACHE[key] = grid
    while len(_GRID_CACHE) > _GRID_CACHE_LIMIT:
        _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
    return grid


def _cell_of(grid, x, z):
    """Grid cell (row=z, col=x) for a world (x, z), matching groundHF's convention.

    Returned as (z_index, x_index) because every grid array is shaped (nz, nx) and
    indexed [z, x]. Swapping these silently corrupts a square grid and throws on a
    non-square one — real room scans are never square.
    """
    gx = (x - grid["ox"]) / grid["cell"] - 0.5
    gz = (z - grid["oz"]) / grid["cell"] - 0.5
    if gx < -0.5 or gz < -0.5 or gx > grid["nx"] - 0.5 or gz > grid["nz"] - 0.5:
        return None
    return int(np.clip(round(gz), 0, grid["nz"] - 1)), int(np.clip(round(gx), 0, grid["nx"] - 1))


def _bilinear(arr, grid, x, z):
    gx = min(max((x - grid["ox"]) / grid["cell"] - 0.5, 0), grid["nx"] - 1.001)
    gz = min(max((z - grid["oz"]) / grid["cell"] - 0.5, 0), grid["nz"] - 1.001)
    x0, z0 = int(np.floor(gx)), int(np.floor(gz))
    fx, fz = gx - x0, gz - z0
    x1, z1 = min(x0 + 1, grid["nx"] - 1), min(z0 + 1, grid["nz"] - 1)
    a = arr[z0, x0] * (1 - fx) + arr[z0, x1] * fx
    b = arr[z1, x0] * (1 - fx) + arr[z1, x1] * fx
    return a * (1 - fz) + b * fz


def floor_y(scene_dir, x, z):
    """The floor height at world (x, z), or None without sampled floor support."""
    grid = _grid(scene_dir)
    cell = _cell_of(grid, x, z)
    if cell is None or not grid["supported"][cell]:
        return None
    y = _bilinear(grid["floor"], grid, x, z)
    return float(y) if np.isfinite(y) else None


def _footprint_cells(grid, x, z, length, depth, yaw_deg):
    """Sample points inside the rotated footprint rectangle, as (world x, z)."""
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    hu, hw = length / 2, depth / 2
    ts = np.linspace(-1, 1, _FOOTPRINT_SAMPLES)
    pts = []
    for qu in ts:
        for qv in ts:
            u, v = qu * hu, qv * hw
            pts.append((x + u * ca - v * sa, z + u * sa + v * ca))
    return pts


def fit_check(scene_dir, x, z, length, depth, yaw_deg, height):
    """Check sampled floor suitability, not collision-free furniture fit.

    ``supported`` and ``valid`` require a supported centre, the minimum fraction
    of supported samples and no off-grid samples. ``floor_clearance_m`` measures
    coverage-edge proximity. Ceiling height and height fit remain unknown: a top
    surface does not identify a ceiling, regardless of the item's ``height``.
    """
    grid = _grid(scene_dir)
    supported = grid["supported"]
    clearance = grid["clearance"]
    center_supported = False
    min_clear = float("inf")
    on_grid = 0
    supported_hits = 0
    samples = _footprint_cells(grid, x, z, length, depth, yaw_deg)
    for px, pz in samples:
        cell = _cell_of(grid, px, pz)
        if cell is None:
            continue
        ci, cj = cell
        on_grid += 1
        if supported[ci, cj]:
            supported_hits += 1
            min_clear = min(min_clear, float(clearance[ci, cj]))
    cc = _cell_of(grid, x, z)
    if cc is not None:
        center_supported = bool(supported[cc[0], cc[1]])
    # Off-grid samples are part of the whole footprint, never dropped from the
    # denominator. Even a mostly supported footprint cannot overhang the scan.
    fraction = supported_hits / len(samples)
    outside = on_grid != len(samples)
    is_supported = bool(not outside and center_supported and fraction >= _MIN_SUPPORT)
    clear = round(min_clear, 3) if np.isfinite(min_clear) else None
    tight = bool(clear is not None and clear < _TIGHT_M)
    if outside:
        reason = "part or all of the footprint is outside the scanned floor"
    elif not center_supported:
        reason = "the drop point has no measured floor support"
    elif not is_supported:
        reason = f"only {fraction * 100:.0f}% of the footprint samples land on measured floor"
    else:
        reason = "sampled floor support is suitable"
    reason += "; ceiling clearance unknown; obstacle and furniture collisions are not checked"
    return {"supported": is_supported, "support_fraction": round(fraction, 3),
            "floor_clearance_m": clear,
            "clearance_basis": "supported samples to unsupported coverage or grid boundary, not obstacle clearance",
            "ceiling_height_m": None, "fits_height": None, "tight": tight,
            "valid": is_supported, "reason": reason}


# The editor can lift a piece off the floor (onto a plinth) or push it into a
# surface it cannot pass. Both are clamped so a bad drag cannot fling furniture
# into the ceiling or under the slab.
_LIFT_MIN_M, _LIFT_MAX_M = -0.2, 3.0


def make_placement(scene_dir, item, x, z, *, yaw_deg=0.0, scale=1.0, label=None,
                   size=None, center_y=None):
    """Assemble a placement record: real size, snapped to the floor, fit-checked.

    ``x, z`` is the footprint centre on the floor; the returned ``center_y`` lifts
    the box so its base sits on the measured floor. If the floor cannot be read at
    that point the record is still produced but flagged invalid, never fabricated.

    ``size`` overrides the catalogue dimensions (the editor's resize handles);
    ``center_y`` carries an explicit vertical position, recorded as ``lift_m``
    above the measured floor so a later move or rotate keeps it.

    For an imported model the resolved spec carries a ``model`` entry: its true
    file size becomes ``size``, and the record gains a ``model`` descriptor plus a
    scale caveat on ``fit.reason`` — the model's own metres are read from the
    file, but the scene they sit in may only be scaled by a guess.
    """
    spec = item_spec(item, scene_dir)
    length, height, depth = spec["size"]
    s = float(scale) if scale and scale > 0 else 1.0
    length, height, depth = length * s, height * s, depth * s
    if size is not None:
        length, height, depth = (float(v) for v in size)
    yaw = float(yaw_deg) % 360.0
    floor = floor_y(scene_dir, x, z)
    fit = fit_check(scene_dir, x, z, length, depth, yaw, height)
    if floor is None:
        fit = {**fit, "supported": False, "valid": False,
               "reason": fit.get("reason") or "no measured floor at the drop point"}
        base_y = 0.0
    else:
        base_y = floor
    rest_y = base_y + height / 2
    lift = 0.0 if center_y is None else max(_LIFT_MIN_M, min(_LIFT_MAX_M, float(center_y) - rest_y))
    record = {
        "id": uuid.uuid4().hex,
        "item": item,
        "label": (label or spec["label"])[:200],
        "size": [round(length, 4), round(height, 4), round(depth, 4)],
        "center_xz": [round(float(x), 4), round(float(z), 4)],
        "center_y": round(rest_y + lift, 4),
        "yaw_deg": round(yaw, 2),
        "scale": s,
        "lift_m": round(lift, 4),
        "footprint_m2": round(length * depth, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit": fit,
    }
    if spec.get("imported"):
        descriptor = model_descriptor(spec["model"], scene_dir)
        record["model"] = descriptor
        record["fit"] = {**fit, "reason": fit["reason"] + "; " + _scale_caveat(
            descriptor["scale_status"], descriptor["scale_source"])}
    return record
