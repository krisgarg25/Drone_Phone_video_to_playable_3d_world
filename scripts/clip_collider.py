"""Clip a collision mesh to the GROUND layer, dropping airborne crusts.

Why this exists. Drone footage contains sky and haze, and the splat happily
reconstructs both as gaussians. Voxelised, they become a solid crust tens of
metres above the terrain — measured on work/rocks, 33% of grid columns had their
topmost collider surface more than 10 m above the ground, and at the spawn there
were two disjoint shells: real ground at y=-16..-11 and a crust at y=+5..+9 with
16 m of nothing between them. Every downward raycast in the viewer (spawn
placement, the grounded test, the terrain underlay's height sampling) takes the
FIRST hit, so all of them latched onto the crust. That is the whole "character
floats above a blown-out white sheet, nowhere near the splat" symptom: the
capsule really was resting on solid geometry, just 20 m too high, and the
underlay sheet was draped over the same crust, hiding the splat underneath it.

A y-max on splat-transform's `-B` box cannot separate the two: this scene's own
ground relief spans y=-13.3..+6.8, which overlaps the crust's range. The cut has
to be relative to the ground, not absolute.

Method: per grid column, occupancy is binned at the voxel pitch. Starting from
the exported heightfield's height, the nearest occupied bin is the ground seed;
marching up from there, the layer ends at the first void `--gap` metres tall.
Everything above that void is airborne and goes. This keeps real overhangs and
boulders (no void under them) and keeps the terrain's relief intact, because the
threshold follows the terrain instead of cutting a plane through it.

  python clip_collider.py --asset work/rocks/viewer_assets \
      --glb work/rocks/pc/col_cluster_shell.collision.glb \
      --out work/rocks/pc/collision.collision.glb
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_mesh(path: Path) -> np.ndarray:
    """All triangles of a GLB as (M, 3, 3) float32, in the file's own space."""
    data = path.read_bytes()
    clen, _ = struct.unpack("<II", data[12:20])
    js = json.loads(data[20:20 + clen])
    bin_off = 20 + clen + 8
    out = []
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            acc = js["accessors"][prim["attributes"]["POSITION"]]
            bv = js["bufferViews"][acc["bufferView"]]
            off = bin_off + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            pts = np.frombuffer(data, np.uint8, count=acc["count"] * 12, offset=off)
            pts = pts.copy().view(np.float32).reshape(-1, 3)
            iacc = js["accessors"][prim["indices"]]
            ibv = js["bufferViews"][iacc["bufferView"]]
            ioff = bin_off + ibv.get("byteOffset", 0) + iacc.get("byteOffset", 0)
            dt = np.uint32 if iacc["componentType"] == 5125 else np.uint16
            idx = np.frombuffer(data, dt, count=iacc["count"], offset=ioff)
            out.append(pts[idx.reshape(-1, 3).astype(np.int64)])
    return np.concatenate(out, axis=0)


def write_mesh(tris: np.ndarray, path: Path) -> None:
    """Minimal single-primitive GLB: welded POSITION + uint32 indices.

    PlayCanvas instantiates one render component per primitive and pc.js gives
    each one a static trimesh rigidbody, so welding into a single primitive also
    collapses N ammo bodies into one.
    """
    flat = tris.reshape(-1, 3).astype(np.float32)
    verts, idx = np.unique(flat, axis=0, return_inverse=True)
    verts = verts.astype(np.float32)
    idx = idx.astype(np.uint32).ravel()

    vb, ib = verts.tobytes(), idx.tobytes()
    pad = (-len(vb)) % 4  # both are 4-byte types, so this is always 0 — kept honest
    blob = vb + b"\0" * pad + ib
    gltf = {
        "asset": {"version": "2.0", "generator": "clip_collider.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "collider"}],
        "meshes": [{"name": "collider", "primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts),
             "type": "VEC3",
             "min": [float(v) for v in verts.min(0)],
             "max": [float(v) for v in verts.max(0)]},
            {"bufferView": 1, "componentType": 5125, "count": len(idx),
             "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vb), "target": 34962},
            {"buffer": 0, "byteOffset": len(vb) + pad, "byteLength": len(ib),
             "target": 34963},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    blob += b"\0" * ((-len(blob)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON")); f.write(js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0")); f.write(blob)
    print(f"[clip] wrote {path.name}: {len(tris)} tris, {len(verts)} welded verts "
          f"({total / 1e6:.1f} MB)")


def ground_ceiling(tris: np.ndarray, ref: np.ndarray, cov: np.ndarray,
                   ox: float, oz: float, cell: float,
                   pitch: float, gap: float, seek: float) -> tuple:
    """Per column: (ceiling y, seed y). NaN where no ground layer was found.

    `seek` bounds how far from the exported heightfield the ground seed may be —
    without it a column whose only geometry is crust would seed on the crust and
    the clip would keep exactly the wrong layer.
    """
    nz, nx = ref.shape
    V = tris.reshape(-1, 3)
    j = np.floor((V[:, 0] - ox) / cell).astype(int)
    i = np.floor((V[:, 2] - oz) / cell).astype(int)
    ok = (i >= 0) & (i < nz) & (j >= 0) & (j < nx)
    i, j, y = i[ok], j[ok], V[ok, 1].astype(np.float64)

    y0 = float(y.min()) - pitch
    nb = int(np.ceil((y.max() - y0) / pitch)) + 2
    occ = np.zeros((nz * nx, nb), bool)
    occ[i * nx + j, np.clip(((y - y0) / pitch).astype(int), 0, nb - 1)] = True

    # ground seed: nearest occupied bin to the reference height, within `seek`
    b_ref = np.clip(((ref.ravel() - y0) / pitch).astype(int), 0, nb - 1)
    r = int(np.ceil(seek / pitch))
    seed = np.full(nz * nx, -1, int)
    for d in range(r + 1):                    # expanding search, nearest wins
        for b in ((b_ref - d, b_ref + d) if d else (b_ref,)):
            bb = np.clip(b, 0, nb - 1)
            hit = occ[np.arange(nz * nx), bb] & (seed < 0)
            seed[hit] = bb[hit]

    # march up from the seed; stop at the first void `gap` metres tall
    need = max(int(np.ceil(gap / pitch)), 1)
    ceil_b = np.full(nz * nx, -1, int)
    live = seed >= 0
    cur = seed.copy()
    run = np.zeros(nz * nx, int)
    for b in range(nb):
        act = live & (b > cur) & (ceil_b < 0)
        if not act.any():
            continue
        filled = occ[:, b]
        run = np.where(act & ~filled, run + 1, np.where(act & filled, 0, run))
        done = act & (run >= need)
        ceil_b[done] = b - run[done]          # bottom of the void
    ceil_b[live & (ceil_b < 0)] = nb - 1      # layer runs to the top: keep it all

    ceiling = np.where(live, y0 + (ceil_b + 1) * pitch, np.nan).reshape(nz, nx)
    seed_y = np.where(live, y0 + seed * pitch, np.nan).reshape(nz, nx)
    n_sup = int((cov > 0).sum())
    got = int((live.reshape(nz, nx) & (cov > 0)).sum())
    print(f"[clip] ground layer found in {int(live.sum())}/{live.size} columns "
          f"({got}/{n_sup} of the supported ones)")
    return ceiling, seed_y


def clip(glb: Path, asset: Path, out: Path, pitch: float = 0.35,
         gap: float = 1.4, seek: float = 4.0, below: float = 4.0) -> Path:
    col = json.loads((asset / "collision.json").read_text(encoding="utf-8"))
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    ref = np.fromfile(asset / "heights.f32", np.float32).reshape(nz, nx).astype(np.float64)
    cov = np.fromfile(asset / "coverage.u8", np.uint8).reshape(nz, nx)

    tris = read_mesh(glb)
    print(f"[clip] {glb.name}: {len(tris)} tris, "
          f"y {tris[:, :, 1].min():.2f}..{tris[:, :, 1].max():.2f}")
    ceiling, seed_y = ground_ceiling(tris, ref, cov, ox, oz, cell, pitch, gap, seek)

    c = tris.mean(axis=1)
    j = np.clip(np.floor((c[:, 0] - ox) / cell).astype(int), 0, nx - 1)
    i = np.clip(np.floor((c[:, 2] - oz) / cell).astype(int), 0, nz - 1)
    top, bot = ceiling[i, j], seed_y[i, j] - below
    keep = np.isfinite(top) & (c[:, 1] <= top) & (c[:, 1] >= bot)
    print(f"[clip] keeping {keep.sum()} / {len(tris)} tris "
          f"({100 * keep.mean():.0f}%) — dropped {int((~keep).sum())} airborne/buried")
    if keep.sum() < 1000:
        raise SystemExit("[clip] almost nothing survived — check --seek/--gap")

    kept = tris[keep]
    write_mesh(kept, out)

    # report the thing that actually matters: what a downward ray now finds
    V = kept.reshape(-1, 3)
    jj = np.floor((V[:, 0] - ox) / cell).astype(int)
    ii = np.floor((V[:, 2] - oz) / cell).astype(int)
    m = (ii >= 0) & (ii < nz) & (jj >= 0) & (jj < nx)
    TOP = np.full((nz, nx), -np.inf)
    np.maximum.at(TOP, (ii[m], jj[m]), V[m, 1])
    has = np.isfinite(TOP)
    d = TOP[has] - ref[has]
    print(f"[clip] topmost surface minus ground, over {int(has.sum())} columns: "
          + " ".join(f"p{p}={np.percentile(d, p):+.2f}" for p in (5, 50, 95)))
    print(f"[clip] columns still >3 m above ground: {100 * (d > 3).mean():.1f}% "
          f"(was 46% before clipping on work/rocks)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--glb", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pitch", type=float, default=0.35,
                    help="occupancy bin height (m); use the collider voxel size")
    ap.add_argument("--gap", type=float, default=1.4,
                    help="void height that ends the ground layer (m)")
    ap.add_argument("--seek", type=float, default=4.0,
                    help="max distance from the heightfield to seed the ground (m)")
    ap.add_argument("--below", type=float, default=4.0,
                    help="keep this much shell under the seed (m)")
    a = ap.parse_args()
    clip(a.glb, a.asset, a.out, a.pitch, a.gap, a.seek, a.below)


if __name__ == "__main__":
    main()
