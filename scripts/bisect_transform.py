"""Bisect the export transform: render eval cam0 at each stage to find where it breaks.

A: splat.ply + raw COLMAP pose            (training path — expect GOOD)
B: rot only   (points @Rg, pose R@Rg^T)
C: rot+scale  (scene.ply frame, 640x360 K)
D: rot+scale+K2x (scene.ply frame, 1280x720 K — what poses.json says)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_heightfield import load_points, ransac_plane, rot_to_up  # noqa: E402
from parse_colmap import qvec2rot  # noqa: E402

from gsplat import rasterization  # noqa: E402

work = ROOT / "work" / "rocks"
dev = "cuda"
SH2RGB = 0.28209479177387814


def load_ply(path):
    v = PlyData.read(str(path))["vertex"].data
    return dict(
        means=torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev),
        quats=torch.tensor(np.stack([v[f"rot_{k}"] for k in range(4)], 1), dtype=torch.float32, device=dev),
        scales=torch.tensor(np.stack([v[f"scale_{k}"] for k in range(3)], 1), dtype=torch.float32, device=dev).exp(),
        opac=torch.sigmoid(torch.tensor(np.asarray(v["opacity"], dtype=np.float32), device=dev)),
        rgb=(torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1),
                          dtype=torch.float32, device=dev) * SH2RGB + 0.5).clamp(0, 1),
    )


def render(g, R, t, K, W, H, out):
    viewmat = torch.eye(4, dtype=torch.float32, device=dev)[None]
    viewmat[0, :3, :3] = torch.tensor(R, dtype=torch.float32, device=dev)
    viewmat[0, :3, 3] = torch.tensor(t, dtype=torch.float32, device=dev)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]
    with torch.no_grad():
        rgb, _, _ = rasterization(g["means"], g["quats"], g["scales"], g["opac"],
                                  g["rgb"][None], viewmat, Kt, W, H, render_mode="RGB")
        img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(str(work / out))
    print(f"[bisect] {out}  mean={img.mean():.1f}")


# raw COLMAP pose of 00000.jpg
rows = [json.loads(l) for l in (work / "keyframes_poses.jsonl").read_text().splitlines() if l.strip()]
row0 = next(r for r in rows if r["file"] == "00000.jpg")
R0 = np.array(row0["camera"]["R_rowmajor"])
t0 = np.array(row0["camera"]["t"])
f0, cx0, cy0 = row0["camera"]["fx"], row0["camera"]["cx"], row0["camera"]["cy"]
K640 = [[f0, 0, cx0], [0, f0, cy0], [0, 0, 1]]

g_full = load_ply(work / "splat.ply")      # COLMAP frame, all gaussians
g_scene = load_ply(work / "viewer_assets" / "scene.ply")  # transformed + pruned

xyz = load_points(work / "sparse_points.ply", None, 0.0, 10_000_000)
Rg = rot_to_up(ransac_plane(xyz))
print(f"[bisect] Rg=\n{Rg.round(4)}")

s = 110.0 / 149.14  # same scale export computed

# A: training path
render(g_full, R0, t0, K640, 640, 360, "bisect_A_colmap.png")
# B: rotate points+pose only (use full ply rotated on the fly)
g_rot = dict(g_full)
g_rot["means"] = (g_full["means"] @ torch.tensor(Rg.T, dtype=torch.float32, device=dev))
Rb = R0 @ Rg.T
render(g_rot, Rb, t0, K640, 640, 360, "bisect_B_rot.png")
# C: rot+scale on the fly (full ply), 640 K
g_rs = dict(g_rot)
g_rs["means"] = g_rot["means"] * s
render(g_rs, Rb, t0 * s, K640, 640, 360, "bisect_C_rotscale.png")
# D: same but 1280x720 K
K720 = [[f0 * 2, 0, cx0 * 2], [0, f0 * 2, cy0 * 2], [0, 0, 1]]
render(g_rs, Rb, t0 * s, K720, 1280, 720, "bisect_D_K720.png")
