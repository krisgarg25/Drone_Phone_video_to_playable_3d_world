"""Render eval cam 0 with gsplat directly (ground truth projection) + plot the
gravity-rotated point cloud with the camera arc, to diagnose pose/gauge issues."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_heightfield import load_points, ransac_plane, rot_to_up  # noqa: E402

work = ROOT / "work" / "rocks"
asset = work / "viewer_assets"

# ---- 1. offline gsplat render of eval cam 0 ----
import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402
from plyfile import PlyData  # noqa: E402
from PIL import Image  # noqa: E402

ply = PlyData.read(str(asset / "scene.ply"))["vertex"].data
dev = "cuda"
means = torch.tensor(np.stack([ply["x"], ply["y"], ply["z"]], 1), dtype=torch.float32, device=dev)
quats = torch.tensor(np.stack([ply["rot_0"], ply["rot_1"], ply["rot_2"], ply["rot_3"]], 1), dtype=torch.float32, device=dev)
scales = torch.tensor(np.stack([ply["scale_0"], ply["scale_1"], ply["scale_2"]], 1), dtype=torch.float32, device=dev).exp()
opac = torch.sigmoid(torch.tensor(np.asarray(ply["opacity"], dtype=np.float32), device=dev))
colors = (torch.tensor(np.stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]], 1),
                       dtype=torch.float32, device=dev) * 0.28209479177387814 + 0.5).clamp(0, 1)[None]

poses = json.loads((asset / "poses.json").read_text())
p = poses[0]
R = np.array(p["R_rowmajor"], dtype=np.float32)
t = np.array(p["t"], dtype=np.float32)
viewmat = torch.eye(4, dtype=torch.float32, device=dev)[None]
viewmat[0, :3, :3] = torch.tensor(R, device=dev)
viewmat[0, :3, 3] = torch.tensor(t, device=dev)
W, H = p["width"], p["height"]
fx, fy, cx, cy = p["fx"] * W / 640, p["fy"] * H / 360, p["cx"] * W / 640, p["cy"] * H / 360
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=dev)[None]

with torch.no_grad():
    rgb, _, _ = rasterization(means, quats, scales, opac, colors, viewmat, K, W, H, render_mode="RGB")
    img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
Image.fromarray(img).save(str(work / "gsplat_eval0.png"))
print(f"[gsplat] wrote gsplat_eval0.png (cam0 {W}x{H}, fx={fx:.0f})")

# ---- 2. point cloud + camera arc plot in gravity-rotated frame ----
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

xyz = load_points(work / "sparse_points.ply", None, 0.0, 10_000_000)
Rg = rot_to_up(ransac_plane(xyz))
X = xyz @ Rg.T
rows = [json.loads(l) for l in (work / "keyframes_poses.jsonl").read_text().splitlines() if l.strip()]
C = np.array([-np.array(r["camera"]["R_rowmajor"]).T @ np.array(r["camera"]["t"]) for r in rows]) @ Rg.T

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax1.scatter(X[::3, 0], X[::3, 1], s=0.5, c="g", alpha=0.3, label="points (side)")
ax1.scatter(C[:, 0], C[:, 1], s=8, c="r", label="cameras")
ax1.set_xlabel("x"); ax1.set_ylabel("y (gravity-up)"); ax1.legend(); ax1.set_title("side view")
ax2 = fig.add_subplot(122)
ax2.scatter(X[::3, 0], X[::3, 2], s=0.5, c="g", alpha=0.3, label="points (top)")
ax2.scatter(C[:, 0], C[:, 2], s=8, c="r", label="cameras")
ax2.set_xlabel("x"); ax2.set_ylabel("z"); ax2.legend(); ax2.set_title("top view"); ax2.set_aspect("equal")
plt.tight_layout()
plt.savefig(str(work / "gauge_check.png"), dpi=110)
print("[plot] wrote gauge_check.png")
print(f"[gauge] camera y range in gravity frame: [{C[:,1].min():.2f}, {C[:,1].max():.2f}], "
      f"point y range: [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
