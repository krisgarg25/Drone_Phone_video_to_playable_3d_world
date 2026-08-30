"""Render poses 00000 & 00044 with EXACT training-style settings (full SH3,
packed, near/far) vs DC-only, from splat.ply + COLMAP poses."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
work = ROOT / "work" / "rocks"
dev = "cuda"

v = PlyData.read(str(work / "splat.ply"))["vertex"].data
N = len(v)
means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev)
quats = torch.nn.functional.normalize(
    torch.tensor(np.stack([v[f"rot_{k}"] for k in range(4)], 1), dtype=torch.float32, device=dev), dim=1)
scales = torch.tensor(np.stack([v[f"scale_{k}"] for k in range(3)], 1), dtype=torch.float32, device=dev).exp()
opac = torch.sigmoid(torch.tensor(np.asarray(v["opacity"], dtype=np.float32), device=dev))
sh0 = torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), dtype=torch.float32, device=dev)
n_rest = 15
shN = torch.zeros(N, n_rest, 3, dtype=torch.float32, device=dev)
for i in range(n_rest):
    shN[:, i, 0] = torch.tensor(np.asarray(v[f"f_rest_{i*3+0}"], dtype=np.float32), device=dev)
    shN[:, i, 1] = torch.tensor(np.asarray(v[f"f_rest_{i*3+1}"], dtype=np.float32), device=dev)
    shN[:, i, 2] = torch.tensor(np.asarray(v[f"f_rest_{i*3+2}"], dtype=np.float32), device=dev)
colors_sh = torch.cat([sh0[:, None, :], shN], dim=1)  # [N,16,3]
rgb_direct = (sh0 * 0.28209479177387814 + 0.5).clamp(0, 1)  # [N,3]

from gsplat import rasterization  # noqa: E402

rows = [json.loads(l) for l in (work / "keyframes_poses.jsonl").read_text().splitlines() if l.strip()]
extent = 113.56


def render(name, row, colors, sh_degree, packed):
    R = np.array(row["camera"]["R_rowmajor"], dtype=np.float32)
    t = np.array(row["camera"]["t"], dtype=np.float32)
    f, cx, cy = row["camera"]["fx"], row["camera"]["cx"], row["camera"]["cy"]
    K = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    vm = torch.eye(4, dtype=torch.float32, device=dev)[None]
    vm[0, :3, :3] = torch.tensor(R, device=dev)
    vm[0, :3, 3] = torch.tensor(t, device=dev)
    with torch.no_grad():
        rgb, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, 640, 360,
                                  near_plane=0.01, far_plane=10 * extent,
                                  render_mode="RGB", sh_degree=sh_degree, packed=packed)
        img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(str(work / name))
    print(f"{name}: mean={img.mean():.1f}")


r0 = next(r for r in rows if r["file"] == "00000.jpg")
r44 = next(r for r in rows if r["file"] == "00044.jpg")
render("t_A_00000_sh3.png", r0, colors_sh[None], 3, True)
render("t_B_00000_dc.png", r0, rgb_direct[None], None, True)
render("t_C_00044_sh3.png", r44, colors_sh[None], 3, True)
render("t_D_00044_dc.png", r44, rgb_direct[None], None, True)
