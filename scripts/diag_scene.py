"""Diagnose splat / heightfield / collider agreement for a viewer_assets dir.

Answers the questions that matter for "is the player standing on the splat?":
  - what are the splat's true bounds, and its robust (percentile) bounds?
  - how much of the splat is sky floaters above the ground surface?
  - does the heightfield cover the same footprint as the splat's ground?
  - where does the collider's top surface sit vs the splat's ground?

  python diag_scene.py --asset work/rocks/viewer_assets \
      --glb work/rocks/pc/collision.collision.glb
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from glb_bounds import accessor, load_glb


def splat_stats(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    P = np.stack([np.asarray(v["x"], np.float64),
                  np.asarray(v["y"], np.float64),
                  np.asarray(v["z"], np.float64)], axis=1)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64))) \
        if "opacity" in v.data.dtype.names else np.ones(len(P))
    return P, op


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--glb", type=Path)
    args = ap.parse_args()

    P, op = splat_stats(args.asset / "scene.ply")
    print(f"[diag] {len(P)} gaussians")
    for ax, nm in enumerate("xyz"):
        c = P[:, ax]
        print(f"  {nm}: abs [{c.min():8.2f} {c.max():8.2f}]  "
              f"p1/p99 [{np.percentile(c, 1):8.2f} {np.percentile(c, 99):8.2f}]  "
              f"p5/p95 [{np.percentile(c, 5):8.2f} {np.percentile(c, 95):8.2f}]")

    col = json.loads((args.asset / "collision.json").read_text(encoding="utf-8"))
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    H = np.frombuffer((args.asset / "heights.f32").read_bytes(), np.float32).reshape(nz, nx)
    print(f"[diag] heightfield {nx}x{nz} cell {cell:.3f} -> "
          f"x[{ox:.1f}..{ox + (nx - 1) * cell:.1f}] z[{oz:.1f}..{oz + (nz - 1) * cell:.1f}]")
    print(f"        H range [{H.min():.2f} {H.max():.2f}]")

    # how much of the heightfield footprint has real splat support?
    gx = np.clip(((P[:, 0] - ox) / cell).astype(int), 0, nx - 1)
    gz = np.clip(((P[:, 2] - oz) / cell).astype(int), 0, nz - 1)
    cnt = np.zeros((nz, nx), np.int64)
    np.add.at(cnt, (gz, gx), 1)
    print(f"[diag] heightfield cells with >=8 gaussians: "
          f"{(cnt >= 8).mean() * 100:.0f}%  (>=1: {(cnt >= 1).mean() * 100:.0f}%)")

    # floaters: gaussians well above the local ground height
    hy = H[gz, gx]
    above = P[:, 1] - hy
    for t in (1, 2, 3, 5, 8):
        print(f"[diag] gaussians more than {t} m above local ground: "
              f"{(above > t).mean() * 100:5.1f}%")

    if args.glb and args.glb.exists():
        gltf, bin_ = load_glb(args.glb)
        V = np.vstack([accessor(gltf, bin_, p["attributes"]["POSITION"]).astype(np.float64)
                       for m in gltf["meshes"] for p in m["primitives"]])
        print(f"[diag] collider {len(V)} verts  x[{V[:, 0].min():.1f}..{V[:, 0].max():.1f}] "
              f"y[{V[:, 1].min():.1f}..{V[:, 1].max():.1f}] "
              f"z[{V[:, 2].min():.1f}..{V[:, 2].max():.1f}]")
        top = V[:, 1].max()
        print(f"[diag] collider verts within 0.01 of top ({top:.2f}): "
              f"{(V[:, 1] > top - 0.01).mean() * 100:.1f}%")
        s = col.get("spawn") or {}
        if s:
            sx, sz = s["x"], s["z"]
            near = (np.abs(V[:, 0] - sx) < 1.0) & (np.abs(V[:, 2] - sz) < 1.0)
            gsel = (np.abs(P[:, 0] - sx) < 1.5) & (np.abs(P[:, 2] - sz) < 1.5)
            print(f"[diag] SPAWN ({sx:.2f}, {sz:.2f}):")
            print(f"        collider verts within 1 m: {near.sum()}, "
                  f"top y = {V[near, 1].max() if near.any() else float('nan'):.2f}")
            print(f"        splat gaussians within 1.5 m: {gsel.sum()}, "
                  f"y p20/p50/p95 = "
                  f"{np.percentile(P[gsel, 1], [20, 50, 95]).round(2) if gsel.any() else None}")
            print(f"        heightfield H = "
                  f"{H[int((sz - oz) / cell), int((sx - ox) / cell)]:.2f}")


if __name__ == "__main__":
    main()
