"""Minimal binary glTF 2.0 writer for triangle meshes (no dependencies beyond numpy).

Used by the flat planning test scene (its physics collider) and by the proposal
exports (3D Tiles content). Each mesh is one primitive: POSITION, optional NORMAL,
optional per-vertex COLOR_0 (RGBA uint8, normalized) and uint32 indices.
"""
from __future__ import annotations

import json
import struct

import numpy as np


def _pad(data: bytes, fill: bytes = b"\0") -> bytes:
    return data + fill * ((-len(data)) % 4)


def glb_bytes(meshes, *, generator="plan_glb.py", node_matrix=None, extras=None) -> bytes:
    """``meshes``: list of dicts with ``positions`` (Nx3), ``indices`` (Mx3 or flat),
    optional ``colors`` (Nx4 uint8), ``normals`` (Nx3) and ``name``."""
    blob = bytearray()
    views, accessors, gl_meshes = [], [], []

    def add_view(data: bytes, target):
        offset = len(blob)
        blob.extend(_pad(data))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data),
                      **({"target": target} if target else {})})
        return len(views) - 1

    for k, mesh in enumerate(meshes):
        pos = np.ascontiguousarray(np.asarray(mesh["positions"], dtype=np.float32).reshape(-1, 3))
        idx = np.ascontiguousarray(np.asarray(mesh["indices"], dtype=np.uint32).ravel())
        if not len(pos) or not len(idx) or len(idx) % 3 or idx.max() >= len(pos):
            raise ValueError(f"mesh {k}: needs vertices and triangle indices into them")
        attributes = {}
        accessors.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126,
                          "count": len(pos), "type": "VEC3",
                          "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()})
        attributes["POSITION"] = len(accessors) - 1
        if mesh.get("normals") is not None:
            nrm = np.ascontiguousarray(np.asarray(mesh["normals"], dtype=np.float32).reshape(-1, 3))
            accessors.append({"bufferView": add_view(nrm.tobytes(), 34962), "componentType": 5126,
                              "count": len(nrm), "type": "VEC3"})
            attributes["NORMAL"] = len(accessors) - 1
        if mesh.get("colors") is not None:
            col = np.ascontiguousarray(np.asarray(mesh["colors"], dtype=np.uint8).reshape(-1, 4))
            accessors.append({"bufferView": add_view(col.tobytes(), 34962), "componentType": 5121,
                              "normalized": True, "count": len(col), "type": "VEC4"})
            attributes["COLOR_0"] = len(accessors) - 1
        accessors.append({"bufferView": add_view(idx.tobytes(), 34963), "componentType": 5125,
                          "count": len(idx), "type": "SCALAR"})
        gl_meshes.append({"name": mesh.get("name", f"mesh{k}"),
                          "primitives": [{"attributes": attributes, "indices": len(accessors) - 1,
                                          "mode": 4, "material": 0}]})
    nodes = [{"mesh": i, "name": m["name"]} for i, m in enumerate(gl_meshes)]
    root = {"name": "root", "children": list(range(len(nodes)))}
    if node_matrix is not None:
        root["matrix"] = [float(v) for v in np.asarray(node_matrix, dtype=np.float64).T.ravel()]
    nodes.append(root)
    document = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0, "scenes": [{"nodes": [len(nodes) - 1]}], "nodes": nodes, "meshes": gl_meshes,
        # Vertex colours carry the look; the material only says "unlit, double sided".
        "materials": [{"name": "proposal", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0,
                                                "roughnessFactor": 1},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "extensionsUsed": ["KHR_materials_unlit"],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    if extras:
        document["extras"] = extras
    js = _pad(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    body = bytes(blob)
    total = 12 + 8 + len(js) + 8 + len(body)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(body), b"BIN\0") + body)


def read_glb(data: bytes):
    """(document, binary chunk) of a GLB; raises on anything that is not one."""
    magic, version, total = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("not a glTF 2.0 binary")
    jlen, jtype = struct.unpack("<I4s", data[12:20])
    if jtype != b"JSON":
        raise ValueError("first GLB chunk is not JSON")
    document = json.loads(data[20:20 + jlen])
    rest = data[20 + jlen:]
    binary = b""
    if rest:
        blen, btype = struct.unpack("<I4s", rest[:8])
        if btype != b"BIN\0":
            raise ValueError("second GLB chunk is not BIN")
        binary = rest[8:8 + blen]
    return document, binary


def accessor_array(document, binary, index):
    acc = document["accessors"][index]
    view = document["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(binary, dtype=dtype, count=acc["count"] * width, offset=start)
    return arr.reshape(-1, width) if width > 1 else arr
