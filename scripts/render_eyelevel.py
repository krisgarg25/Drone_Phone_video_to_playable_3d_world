"""Render the splat from a walking-eye-height camera at the spawn, offline.

The eval renders answer "how good is the splat from the air it was trained at".
They cannot answer "what does the player see", because the player stands ON the
surface looking out at grazing incidence -- exactly where reconstruction noise,
haze gaussians and floaters stop hiding behind each other. This renders the
spawn view at 1.7 m eye height, looking at the route's first waypoint, so the
walkable-world question gets an image instead of a claim.

  python render_eyelevel.py --work work/temple [--ply scene.ply] [--out eye.png]
  python render_eyelevel.py --work work/temple --drop-big 1.0
    --drop-big S also writes a second image with near-ground gaussians whose
    max scale exceeds S metres culled, to preview a floater cull before
    committing it to strip_sky.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--ply", default="scene.ply")
ap.add_argument("--out", default="eye.png")
ap.add_argument("--eye", type=float, default=1.7, help="camera height above the collider surface")
ap.add_argument("--drop-big", type=float, default=0.0,
                help="also render a variant with near-ground gaussians of max "
                     "scale above this many metres culled (0 = skip)")
args = ap.parse_args()

work = args.work
asset = work / "viewer_assets"
dev = "cuda"

col = json.loads((asset / "collision.json").read_text(encoding="utf-8"))
nx, nz, cell = col["nx"], col["nz"], col["cell"]
ox, oz = col["origin_xz"]
G = np.fromfile(asset / "ground.f32", np.float32).reshape(nz, nx)
cov = np.fromfile(asset / "coverage.u8", np.uint8).reshape(nz, nx) > 0


def ground_at(x: float, z: float) -> float:
    j = int(np.clip((x - ox) / cell, 0, nx - 1))
    i = int(np.clip((z - oz) / cell, 0, nz - 1))
    return float(G[i, j])


sp = col["spawn"]
sx, sz = sp["x"], sp["z"]
fx, fz = sp.get("face_xz", [sx, sz + 10.0])
look = np.array([fx - sx, 0.0, fz - sz], np.float64)
look /= np.linalg.norm(look)
eye_y = ground_at(sx, sz) + args.eye
print(f"[eye] spawn ({sx:.1f}, {sz:.1f}) ground {ground_at(sx, sz):.2f} m, "
      f"camera at {eye_y:.2f} m, facing ({look[0]:.2f}, {look[2]:.2f})")


def load(path: Path, cull: float):
    v = PlyData.read(str(path))["vertex"].data
    keep = np.ones(len(v), bool)
    if cull > 0:
        P = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
        smax = np.exp(np.stack([np.asarray(v[f"scale_{k}"], np.float64)
                                for k in range(3)], 1)).max(1)
        j = np.clip(((P[:, 0] - ox) / cell).astype(int), 0, nx - 1)
        i = np.clip(((P[:, 2] - oz) / cell).astype(int), 0, nz - 1)
        near = cov[i, j] & (P[:, 1] - G[i, j] < 6.0)   # only over the walkable footprint
        keep = ~near | (smax <= cull)
        print(f"[eye] cull scale>{cull} m near ground: dropping {int((~keep).sum())} "
              f"({100 * (~keep).mean():.1f}%)")
    return v, keep


def render(v, keep, out_path: Path) -> None:
    means = torch.tensor(np.stack([np.asarray(v["x"][keep], np.float64),
                                   np.asarray(v["y"][keep], np.float64),
                                   np.asarray(v["z"][keep], np.float64)], 1),
                         dtype=torch.float32, device=dev)
    quats = torch.nn.functional.normalize(torch.tensor(
        np.stack([np.asarray(v[f"rot_{k}"][keep], np.float64) for k in range(4)], 1),
        dtype=torch.float32, device=dev), dim=1)
    scales = torch.tensor(np.stack([np.asarray(v[f"scale_{k}"][keep], np.float64)
                                    for k in range(3)], 1),
                          dtype=torch.float32, device=dev).exp()
    opac = torch.sigmoid(torch.tensor(np.asarray(v["opacity"][keep], np.float32), device=dev))
    sh0 = torch.tensor(np.stack([np.asarray(v[f"f_dc_{c}"][keep], np.float64) for c in range(3)], 1),
                       dtype=torch.float32, device=dev)
    colors = sh0[:, None, :].repeat(1, 16, 1)[None]  # SH degree 0: view-independent

    # OpenCV lookat -> world-to-cam. +Z forward, +X right, +Y down.
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(look, -up); r /= np.linalg.norm(r)
    d = np.cross(r, look)
    R = np.stack([r, -d, look], 0)  # rows: camera x/y/z axes in world coords
    C = np.array([sx, eye_y, sz])
    vm = torch.eye(4, dtype=torch.float32, device=dev)[None]
    vm[0, :3, :3] = torch.tensor(R, dtype=torch.float32, device=dev)
    vm[0, :3, 3] = torch.tensor(-R @ C, dtype=torch.float32, device=dev)
    W, H = 1280, 720
    fov_y = np.radians(60.0)
    fy = H / (2 * np.tan(fov_y / 2))
    K = torch.tensor([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)[None]
    from gsplat import rasterization  # noqa: E402
    with torch.no_grad():
        rgb, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, W, H,
                                  near_plane=0.01, far_plane=1000.0,
                                  render_mode="RGB", sh_degree=0, packed=True)
    Image.fromarray((rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()).save(str(out_path))
    print(f"[eye] wrote {out_path}")


v, _ = load(asset / args.ply, 0.0)
render(v, np.ones(len(v["x"]), bool), work / args.out)
if args.drop_big > 0:
    v2, keep2 = load(asset / args.ply, args.drop_big)
    render(v2, keep2, work / args.out.replace(".png", ".culled.png"))
