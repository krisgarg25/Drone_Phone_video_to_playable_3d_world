"""Two synthetic epochs of one site for exercising the Operations tab end to end.

``work/opsbefore`` and ``work/opsafter`` share one *synthetic* GPS origin, so change,
damage, access, flood, corridor and construction tools all have a known answer:

* terrain: a 2% rise to the east, a river channel along the west edge (x < -65), a
  12 m ridge running north-south at x = 45..52 (the "border" sight test);
* a 7 m road along z = 0 (road labels), three buildings north of it;
* after the event: B2 has collapsed to rubble, B3 has lost its east half, a 2.5 m
  debris pile blocks the road at x = 5..11, a 3 m stockpile appeared at (-30, 45)
  (a cone, r = 7 m, 154 m^3) and a 20 x 20 m pad was excavated 1.5 m at (20, 45);
* the after epoch carries two nadir cameras and ``detections.json`` with two people
  and a vehicle, projected from known ground positions, so the layer has a truth;
* ``ops/designs/pad.xml`` (LandXML) in the after epoch: the pad's finished grade is
  2 m below original ground, so 0.5 m (200 m^3) of cut remains.

Everything is labelled synthetic in ``project.json`` and ``frame.json``.

  python scripts/make_ops_scene.py            # writes work/opsbefore and work/opsafter
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import label_semantics
from ground_mesh import build as ground_skin
from make_plan_scene import ORIGIN, box_mesh, write_ply
from plan_glb import glb_bytes

SPAN, CELL = 160.0, 0.5
BUILDINGS = {"B1": (-25.0, -20.0, 14.0, 10.0, 9.0), "B2": (0.0, -22.0, 12.0, 12.0, 7.0),
             "B3": (25.0, -20.0, 14.0, 10.0, 10.0)}           # cx, cz, lx, dz, h
PEOPLE = [(-10.0, 6.0), (30.0, 12.0)]
VEHICLE = (-40.0, -1.0)
STOCKPILE = (-30.0, 45.0, 7.0, 3.0)                            # x, z, r, h
PAD = (20.0, 45.0, 10.0)                                       # x, z, half size


def ground_y(x, z):
    y = 0.02 * (x + 80.0)
    y = y + 12.0 * np.clip(1 - np.abs(x - 48.5) / 3.5, 0, 1) ** 0.5 * (np.abs(x - 48.5) < 3.5)
    y = np.where(x < -65, y - 2.5, y)                           # river channel
    return y


def surfaces(after):
    n = int(round(SPAN / CELL))
    c = -SPAN / 2 + (np.arange(n) + 0.5) * CELL
    xx, zz = np.meshgrid(c, c)
    floor = ground_y(xx, zz)
    if after:
        px, pz, half = PAD
        floor = np.where((np.abs(xx - px) < half) & (np.abs(zz - pz) < half), floor - 1.5, floor)
    top = floor.copy()
    for key, (cx, cz, lx, dz, h) in BUILDINGS.items():
        inside = (np.abs(xx - cx) <= lx / 2) & (np.abs(zz - cz) <= dz / 2)
        roof = floor + h
        if after and key == "B2":
            roof = floor + 1.2 + 0.8 * np.abs(np.sin(xx * 2.3) * np.cos(zz * 1.9))
        if after and key == "B3":
            roof = np.where(xx > cx, floor + 1.0 + 0.9 * np.abs(np.sin(xx * 2.1 + zz)), floor + h)
        top = np.where(inside, roof, top)
    if after:
        top = np.where((xx >= 5) & (xx <= 11) & (np.abs(zz) <= 3.5), floor + 2.5, top)
        sx, sz, r, h = STOCKPILE
        d = np.hypot(xx - sx, zz - sz)
        top = np.where(d < r, floor + h * (1 - d / r), top)
        vx, vz = VEHICLE
        top = np.where((np.abs(xx - vx) < 2.2) & (np.abs(zz - vz) < 0.9), floor + 1.5, top)
    return c, floor.astype("<f4"), top.astype("<f4")


def splat_parts(c, floor, top):
    rng = np.random.default_rng(3)
    idx = np.arange(0, len(c), 1)                               # one splat per 0.5 m cell
    xx, zz = np.meshgrid(c[idx], c[idx])
    ys = top[np.ix_(idx, idx)]
    fl = floor[np.ix_(idx, idx)]
    pts = np.column_stack([xx.ravel() + rng.uniform(-0.2, 0.2, xx.size), ys.ravel(),
                           zz.ravel() + rng.uniform(-0.2, 0.2, xx.size)])
    raised = (ys - fl).ravel()
    road = (np.abs(zz.ravel()) <= 3.5) & (raised < 0.3)
    water = (xx.ravel() < -65)
    rgb = np.column_stack([0.36 + 0.05 * rng.random(len(pts)), 0.46 + 0.06 * rng.random(len(pts)),
                           0.28 + 0.05 * rng.random(len(pts))])
    rgb[road] = [0.32, 0.33, 0.35]
    rgb[water] = [0.25, 0.35, 0.45]
    rgb[raised > 0.3] = [0.66, 0.60, 0.55]
    label = np.full(len(pts), None, dtype=object)
    label[road] = "road"
    label[raised > 2.0] = "building"
    parts = []
    for name in (None, "road", "building"):
        pick = label == name
        parts.append((pts[pick], rgb[pick], np.array([0.3, 0.05, 0.3]), name))
    # Facade splats so buildings read as solids.
    for key, (cx, cz, lx, dz, h) in BUILDINGS.items():
        ring = []
        for y in np.arange(0.2, h, 0.5):
            for x in np.arange(cx - lx / 2, cx + lx / 2, 0.5):
                ring += [[x, y, cz - dz / 2], [x, y, cz + dz / 2]]
            for z in np.arange(cz - dz / 2, cz + dz / 2, 0.5):
                ring += [[cx - lx / 2, y, z], [cx + lx / 2, y, z]]
        ring = np.asarray(ring)
        ring[:, 1] += ground_y(ring[:, 0], ring[:, 2])
        g = np.interp(ring[:, 0], c, np.arange(len(c))).astype(int)
        k = np.interp(ring[:, 2], c, np.arange(len(c))).astype(int)
        keep = ring[:, 1] <= top[k, g] + 0.1                    # a collapsed wall has no facade
        parts.append((ring[keep], np.tile([0.7, 0.62, 0.55], (int(keep.sum()), 1)), np.array([0.2, 0.2, 0.2]), "building"))
    return parts


def cameras_and_detections(c, top):
    cams, dets = [], []
    for i, (x, z) in enumerate(((-20.0, 5.0), (15.0, 5.0))):
        cams.append({"id": i, "name": f"opsafter/{i:05d}.jpg", "t_sec": float(i * 4),
                     "pos": [x, 60.0, z], "forward": [0, -1, 0], "up": [0, 0, -1], "right": [1, 0, 0],
                     "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720,
                     "fov_x_deg": 77.3, "fov_y_deg": 48.5})
        # The frustum the viewer draws: image corners 1 m in front of the camera.
        c0 = np.array(cams[-1]["pos"], float) + np.array(cams[-1]["forward"], float)
        r = np.array(cams[-1]["right"], float) * 0.8
        u = np.array(cams[-1]["up"], float) * 0.45
        cams[-1]["corners"] = [(c0 - r + u).tolist(), (c0 + r + u).tolist(), (c0 + r - u).tolist(), (c0 - r - u).tolist()]
        cams[-1]["top_mark"] = (c0 + u * 1.4).tolist()
    targets = [("person", *PEOPLE[0]), ("person", *PEOPLE[1]), ("vehicle", *VEHICLE)]
    for cam in cams:
        for cls, x, z in targets:
            gi = int(np.clip(np.searchsorted(c, x), 0, len(c) - 1))
            gk = int(np.clip(np.searchsorted(c, z), 0, len(c) - 1))
            y = float(ground_y(np.array(x), np.array(z)))
            depth = cam["pos"][1] - y
            u = cam["cx"] + (x - cam["pos"][0]) / depth * cam["fx"]
            v = cam["cy"] + (z - cam["pos"][2]) / depth * cam["fy"]
            if not (0 <= u < cam["width"] and 0 <= v < cam["height"]):
                continue
            w = 22 if cls == "person" else 60
            dets.append({"frame": cam["name"], "class": cls, "bbox": [u - w / 2, v - 30, u + w / 2, v],
                         "score": 0.8 if cls == "person" else 0.9})
    return cams, dets


def landxml_pad():
    px, pz, half = PAD
    # Breaklines: a vertex just inside and just outside each pad edge, so the pad's wall
    # is vertical in the TIN, as a designer draws it, not a 2.5 m ramp.
    edge = lambda c: [c - half - 0.01, c - half + 0.01, c + half - 0.01, c + half + 0.01]
    xs = np.unique(np.r_[np.arange(px - half - 5, px + half + 5.01, 2.5), edge(px)])
    zs = np.unique(np.r_[np.arange(pz - half - 5, pz + half + 5.01, 2.5), edge(pz)])
    pts, ids = [], {}
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            y = float(ground_y(np.array(x), np.array(z)))
            if abs(x - px) < half and abs(z - pz) < half:
                y -= 2.0
            ids[(i, j)] = len(pts) + 1
            # LandXML: northing easting elevation. Map: east = x, north = -z.
            pts.append(f'<P id="{len(pts) + 1}">{-z} {x} {y:.3f}</P>')
    faces = []
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            a, b, cc, d = ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]
            faces += [f"<F>{a} {b} {cc}</F>", f"<F>{a} {cc} {d}</F>"]
    return ('<?xml version="1.0"?><LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2">'
            '<Surfaces><Surface name="Pad finished grade (synthetic)"><Definition surfType="TIN">'
            f"<Pnts>{''.join(pts)}</Pnts><Faces>{''.join(faces)}</Faces></Definition></Surface></Surfaces></LandXML>")


def render_frames(work: Path, cams, parts):
    """Synthetic source frames: every splat projected into each camera, far to near, 2x2 px.

    They exist so tools that open "the frame that saw this" have an image; they are drawn
    from the synthetic scene, not captured.
    """
    from PIL import Image
    xyz = np.vstack([p[0] for p in parts])
    rgb = (np.clip(np.vstack([p[1] for p in parts]), 0, 1) * 255).astype(np.uint8)
    for cam in cams:
        d = xyz - np.asarray(cam["pos"])
        z = d @ np.asarray(cam["forward"])
        ok = z > 0.5
        x = d[ok] @ np.asarray(cam["right"])
        y = -(d[ok] @ np.asarray(cam["up"]))
        u = (cam["cx"] + cam["fx"] * x / z[ok]).astype(int)
        v = (cam["cy"] + cam["fy"] * y / z[ok]).astype(int)
        col = rgb[ok]
        inside = (u >= 0) & (u < cam["width"] - 1) & (v >= 0) & (v < cam["height"] - 1)
        depth = z[ok][inside]
        order = np.argsort(-depth)
        u, v, col, depth = u[inside][order], v[inside][order], col[inside][order], depth[order]
        # Each splat covers its projected 0.6 m footprint, so the ground reads as a surface.
        size = np.clip(np.ceil(0.6 * cam["fx"] / depth), 1, 12).astype(int)
        img = np.full((cam["height"], cam["width"], 3), (120, 150, 190), np.uint8)
        for du in range(int(size.max())):
            for dv in range(int(size.max())):
                pick = size > max(du, dv)
                uu = np.clip(u[pick] + du, 0, cam["width"] - 1)
                vv = np.clip(v[pick] + dv, 0, cam["height"] - 1)
                img[vv, uu] = col[pick]
        path = work / "frames_full" / cam["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path, quality=90)


def write_epoch(work: Path, after: bool):
    va, pc = work / "viewer_assets", work / "pc"
    va.mkdir(parents=True, exist_ok=True)
    pc.mkdir(parents=True, exist_ok=True)
    c, floor, top = surfaces(after)
    parts = splat_parts(c, floor, top)
    count = write_ply(va / "scene.ply", parts)
    n = len(c)
    floor.tofile(va / "ground.f32")
    top.tofile(va / "heights.f32")
    np.ones((n, n), dtype=np.uint8).tofile(va / "coverage.u8")
    (va / "collision.json").write_text(json.dumps({
        "origin_xz": [-SPAN / 2, -SPAN / 2], "cell": CELL, "nx": n, "nz": n,
        "rotation_rowmajor": np.eye(3).tolist(), "max_step": 0.8, "has_coverage": True,
        "spawn": {"x": -50.0, "z": 20.0, "face_xz": [1.0, 0.0]},
        "note": "synthetic operations scene (make_ops_scene.py)"}, indent=1))
    coarse = 2.0
    step = int(coarse / CELL)
    skin = ground_skin(top[::step, ::step].astype(float), -SPAN / 2, -SPAN / 2, coarse, wall=2.0, skirt=1.0)
    verts = skin.reshape(-1, 3)
    (pc / "collision.collision.glb").write_bytes(glb_bytes(
        [{"name": "collider", "positions": verts, "indices": np.arange(len(verts))}], generator="make_ops_scene.py"))
    coords, rgb = [], []
    for xyz, _, _, label in parts:
        if label:
            sub = xyz[:: max(1, len(xyz) // 6000)]
            coords += sub.round(2).tolist()
            rgb += [list(label_semantics.CLASS_RGB[label])] * len(sub)
    counts = {k: sum(1 for v in rgb if tuple(v) == tuple(col)) for k, col in label_semantics.CLASS_RGB.items()}
    (va / "semantics.json").write_text(json.dumps({"schema_version": 1, "classes": list(label_semantics.CLASS_RGB),
                                                   "counts": counts, "note": "synthetic labels",
                                                   "coords": coords, "rgb": rgb}))
    if after:
        cams, dets = cameras_and_detections(c, top)
        (va / "cameras.json").write_text(json.dumps(cams, indent=1))
        (work / "detections.json").write_text(json.dumps(dets, indent=1))
        render_frames(work, cams, parts)
        (work / "ops" / "designs").mkdir(parents=True, exist_ok=True)
        (work / "ops" / "designs" / "pad.xml").write_text(landxml_pad())
    else:
        (va / "cameras.json").write_text("[]")
    (work / "frame.json").write_text(json.dumps({
        "rotation_rowmajor": np.eye(3).tolist(), "scale_m_per_unit": 1.0,
        "scale_source": "GPS telemetry similarity fit (synthetic test origin, not a flight)",
        "scale_anchor_gps": {"scale": 1.0, "alignment": {
            "schema_version": 1, "status": "aligned", "method": "synthetic", "scale": 1.0,
            "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "translation": [0.0, 0.0, 0.0],
            "fit_rmse_m": 0.0,
            "coordinate_frame": {"type": "ENU", "units": "m", "geodetic_crs": "EPSG:4979",
                                 "altitude_datum": "ellipsoidal", "origin": ORIGIN}}}}, indent=1))
    label = "after the event" if after else "before the event"
    (work / "project.json").write_text(json.dumps({
        "name": f"Ops site {label} (synthetic)", "workflow": "response",
        "site": "ops-synthetic", "captured_at": "2026-09-20T09:00:00+05:30" if not after else "2026-09-26T09:00:00+05:30",
        "notes": f"Synthetic operations test site, {label}. Not a scan."}, indent=1))
    print(f"[ops-scene] {work.name}: {count} splats, {n}x{n} grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("work"))
    args = ap.parse_args()
    write_epoch(args.root / "opsbefore", after=False)
    write_epoch(args.root / "opsafter", after=True)


if __name__ == "__main__":
    main()
