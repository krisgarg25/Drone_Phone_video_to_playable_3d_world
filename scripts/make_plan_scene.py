"""Synthetic FLAT scene for exercising the planning editor end to end.

A real scan brings real problems (coverage holes, estimated scale, a "building" that is
a rock pile) that hide editor bugs behind data bugs. This scene has none of them:

* ground exactly at y = 0 over 120 x 120 m, fully covered, 0.5 m grid;
* two existing buildings (splat shells + semantic labels + solid colliders), so the
  existing inventory, "in the way" checks, demolition and walk collisions all have
  something real to act on;
* three trees (vegetation labels) for the canopy metrics;
* a *synthetic* GPS similarity fit at a fixed test origin, so MGRS, WGS84 exports and
  the sun position of the shadow study all have a latitude. It is labelled synthetic
  in ``frame.json`` and in the project name.

  python scripts/make_plan_scene.py --out work/flatplan
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

import label_semantics
from ground_mesh import build as ground_skin
from plan_glb import glb_bytes

SPAN, CELL = 120.0, 0.5
ORIGIN = {"latitude_deg": 30.7333, "longitude_deg": 76.7794, "altitude_m": 350.0}
# (centre x, centre z, length x, depth z, height, colour) - viewer frame, -z is north.
BUILDINGS = [(20.0, -10.0, 14.0, 10.0, 9.0, (0.62, 0.34, 0.28)),
             (-25.0, 15.0, 8.0, 8.0, 6.0, (0.78, 0.72, 0.60))]
TREES = [(0.0, 25.0, 3.2), (10.0, 28.0, 2.6), (-10.0, 30.0, 3.0)]   # x, z, crown radius
N_REST = 45


def _grid(lo, hi, step):
    return np.arange(lo + step / 2, hi, step)


def splats():
    rng = np.random.default_rng(7)
    parts = []   # (xyz, rgb, scale, label-or-None)
    xs = _grid(-SPAN / 2, SPAN / 2, 0.6)
    gx, gz = np.meshgrid(xs, xs)
    g = np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel()])
    rgb = np.column_stack([0.30 + 0.06 * rng.random(len(g)), 0.46 + 0.08 * rng.random(len(g)),
                           0.24 + 0.05 * rng.random(len(g))])
    # A faint 10 m survey grid and a paved plaza, so scale and position read at a glance.
    line = (np.abs((g[:, 0] + 0.3) % 10 - 5) > 4.6) | (np.abs((g[:, 2] + 0.3) % 10 - 5) > 4.6)
    rgb[line] = [0.55, 0.58, 0.50]
    plaza = (np.abs(g[:, 0]) < 8) & (np.abs(g[:, 2] - 5) < 8)
    rgb[plaza] = [0.62, 0.60, 0.56]
    parts.append((g, rgb, np.array([0.34, 0.02, 0.34]), None))
    for cx, cz, lx, dz, h, colour in BUILDINGS:
        pts = []
        for y in _grid(0, h, 0.4):
            for x in _grid(cx - lx / 2, cx + lx / 2, 0.4):
                pts += [[x, y, cz - dz / 2], [x, y, cz + dz / 2]]
            for z in _grid(cz - dz / 2, cz + dz / 2, 0.4):
                pts += [[cx - lx / 2, y, z], [cx + lx / 2, y, z]]
        roof = [[x, h, z] for x in _grid(cx - lx / 2, cx + lx / 2, 0.4) for z in _grid(cz - dz / 2, cz + dz / 2, 0.4)]
        p = np.asarray(pts + roof)
        shade = np.clip(np.asarray(colour) + rng.normal(0, 0.03, (len(p), 3)), 0, 1)
        shade[len(pts):] *= 0.8
        parts.append((p, shade, np.array([0.22, 0.22, 0.22]), "building"))
    for tx, tz, r in TREES:
        trunk = np.column_stack([tx + rng.normal(0, 0.12, 120), rng.uniform(0, 2.6, 120), tz + rng.normal(0, 0.12, 120)])
        parts.append((trunk, np.tile([0.36, 0.26, 0.16], (120, 1)), np.array([0.12, 0.12, 0.12]), None))
        u = rng.normal(size=(900, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        crown = np.column_stack([tx, 2.6 + r, tz]) + u * (r * rng.random((900, 1)) ** 0.3)
        green = np.column_stack([0.18 + 0.08 * rng.random(900), 0.42 + 0.14 * rng.random(900), 0.16 + 0.06 * rng.random(900)])
        parts.append((crown, green, np.array([0.35, 0.35, 0.35]), "vegetation"))
    return parts


def write_ply(path, parts):
    xyz = np.vstack([p[0] for p in parts])
    rgb = np.vstack([p[1] for p in parts])
    scale = np.vstack([np.tile(np.log(p[2]), (len(p[0]), 1)) for p in parts])
    dtype = ([(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2")]
             + [(f"f_rest_{i}", "f4") for i in range(N_REST)]
             + [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)])
    arr = np.zeros(len(xyz), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz.T
    dc = (rgb - 0.5) / 0.28209479177387814
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc.T
    arr["opacity"] = math.log(0.96 / 0.04)
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scale.T
    arr["rot_0"] = 1.0
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return len(arr)


def box_mesh(cx, cz, lx, dz, h):
    x0, x1, z0, z1 = cx - lx / 2, cx + lx / 2, cz - dz / 2, cz + dz / 2
    v = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1],
                  [x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1]], dtype=np.float32)
    quads = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0)]
    return np.array([[v[a], v[b], v[c]] for q in quads for a, b, c in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("work/flatplan"))
    args = ap.parse_args()
    work = args.out
    va, pc = work / "viewer_assets", work / "pc"
    va.mkdir(parents=True, exist_ok=True)
    pc.mkdir(parents=True, exist_ok=True)

    parts = splats()
    count = write_ply(va / "scene.ply", parts)

    n = int(round(SPAN / CELL))
    centres = -SPAN / 2 + (np.arange(n) + 0.5) * CELL
    xx, zz = np.meshgrid(centres, centres)
    ground = np.zeros((n, n), dtype="<f4")
    top = ground.copy()
    for cx, cz, lx, dz, h, _ in BUILDINGS:
        top[(np.abs(xx - cx) <= lx / 2) & (np.abs(zz - cz) <= dz / 2)] = h
    for tx, tz, r in TREES:
        d = np.hypot(xx - tx, zz - tz)
        top = np.maximum(top, np.where(d <= r, 2.6 + r + np.sqrt(np.clip(r * r - d * d, 0, None)), 0)).astype("<f4")
    ground.tofile(va / "ground.f32")
    top.tofile(va / "heights.f32")
    np.ones((n, n), dtype=np.uint8).tofile(va / "coverage.u8")
    (va / "collision.json").write_text(json.dumps({
        "origin_xz": [-SPAN / 2, -SPAN / 2], "cell": CELL, "nx": n, "nz": n,
        "rotation_rowmajor": np.eye(3).tolist(), "max_step": 0.8, "has_coverage": True,
        "spawn": {"x": 0.0, "z": 45.0, "face_xz": [0.0, 0.0]},
        "note": "synthetic flat planning scene (make_plan_scene.py)"}, indent=1))

    # Collider: the flat ground skin with a rim wall, plus each existing building as a
    # closed box, so the walker is stopped by what stands there and demolition can free it.
    coarse = 2.0
    m = int(round(SPAN / coarse))
    skin = ground_skin(np.zeros((m, m)), -SPAN / 2, -SPAN / 2, coarse, wall=2.0, skirt=1.0)
    tris = np.concatenate([skin] + [box_mesh(cx, cz, lx, dz, h) for cx, cz, lx, dz, h, _ in BUILDINGS])
    verts = tris.reshape(-1, 3)
    (pc / "collision.collision.glb").write_bytes(glb_bytes(
        [{"name": "collider", "positions": verts, "indices": np.arange(len(verts))}], generator="make_plan_scene.py"))

    coords, rgb = [], []
    for xyz, _, _, label in parts:
        if label:
            coords += xyz.round(3).tolist()
            rgb += [list(label_semantics.CLASS_RGB[label])] * len(xyz)
    counts = {k: sum(1 for c in rgb if tuple(c) == v) for k, v in label_semantics.CLASS_RGB.items()}
    (va / "semantics.json").write_text(json.dumps({
        "schema_version": 1, "classes": list(label_semantics.CLASS_RGB), "counts": counts,
        "note": "synthetic labels written with the scene", "coords": coords, "rgb": rgb}))
    (va / "cameras.json").write_text("[]")

    (work / "frame.json").write_text(json.dumps({
        "rotation_rowmajor": np.eye(3).tolist(), "scale_m_per_unit": 1.0,
        "scale_source": "GPS telemetry similarity fit (synthetic test origin, not a flight)",
        "scale_anchor_gps": {"scale": 1.0, "alignment": {
            "schema_version": 1, "status": "aligned", "method": "synthetic", "scale": 1.0,
            # viewer x = east, -z = north, y = up
            "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "translation": [0.0, 0.0, 0.0],
            "fit_rmse_m": 0.0,
            "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                                 "altitude_datum": "ellipsoidal", "origin": ORIGIN}}}}, indent=1))
    (work / "project.json").write_text(json.dumps({
        "name": "Flat plan test (synthetic)", "workflow": "general",
        "notes": "Flat 120 m test ground for the planning editor. Not a scan."}, indent=1))
    print(f"[plan-scene] {count} splats, {n}x{n} grid, {len(tris)} collider triangles -> {work}")


if __name__ == "__main__":
    main()
