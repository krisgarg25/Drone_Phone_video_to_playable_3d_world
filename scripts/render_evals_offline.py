"""Render all eval poses offline with gsplat (the proven training path):
scene.ply + poses.json -> eval_renders/eval_XX.png (1280x720)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=ROOT / "work" / "rocks")
args = ap.parse_args()

work = args.work
asset = work / "viewer_assets"
out = work / "eval_renders"
out.mkdir(exist_ok=True)
dev = "cuda"

v = PlyData.read(str(asset / "scene.ply"))["vertex"].data
means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev)
quats = torch.nn.functional.normalize(
    torch.tensor(np.stack([v[f"rot_{k}"] for k in range(4)], 1), dtype=torch.float32, device=dev), dim=1)
scales = torch.tensor(np.stack([v[f"scale_{k}"] for k in range(3)], 1), dtype=torch.float32, device=dev).exp()
opac = torch.sigmoid(torch.tensor(np.asarray(v["opacity"], dtype=np.float32), device=dev))
sh0 = torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), dtype=torch.float32, device=dev)

n_rest = 15
shN = torch.zeros(len(v), n_rest, 3, dtype=torch.float32, device=dev)
for i in range(n_rest):
    for c in range(3):
        shN[:, i, c] = torch.tensor(np.asarray(v[f"f_rest_{i*3+c}"], dtype=np.float32), device=dev)
colors = torch.cat([sh0[:, None, :], shN], dim=1)[None]  # [1,N,16,3]

from gsplat import rasterization  # noqa: E402

poses = json.loads((asset / "poses.json").read_text())
for i, p in enumerate(poses):
    W, H = p["width"], p["height"]
    sx, sy = W / 640.0, H / 360.0
    fx, fy = p["fx"] * sx, p["fy"] * sy
    cx, cy = p["cx"] * sx, p["cy"] * sy
    R = np.array(p["R_rowmajor"], dtype=np.float32)
    t = np.array(p["t"], dtype=np.float32)
    vm = torch.eye(4, dtype=torch.float32, device=dev)[None]
    vm[0, :3, :3] = torch.tensor(R, device=dev)
    vm[0, :3, 3] = torch.tensor(t, device=dev)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    with torch.no_grad():
        rgb, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, W, H,
                                  near_plane=0.01, far_plane=1000.0,
                                  render_mode="RGB", sh_degree=3, packed=True)
        img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(str(out / f"eval_{i:02d}.png"))
    print(f"[eval] {i:02d} <- {p['name']} mean={img.mean():.1f}")
print("[eval] done")
