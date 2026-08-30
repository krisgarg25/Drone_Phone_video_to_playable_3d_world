"""Verify eval camera poses: center, forward, up for each pose in poses.json."""
import json
from pathlib import Path

import numpy as np

asset = Path(__file__).resolve().parent.parent / "work" / "rocks" / "viewer_assets"
poses = json.loads((asset / "poses.json").read_text())
col = json.loads((asset / "collision.json").read_text())
nx, nz, cell = col["nx"], col["nz"], col["cell"]
ox, oz = col["origin_xz"]
hf = np.load(asset / "heightfield.npz")["heights"] if (asset / "heightfield.npz").exists() else None
if hf is None:
    import struct
    raw = (asset / "heights.f32").read_bytes()
    hf = np.array(struct.unpack(f"<{len(raw)//4}f", raw)).reshape(nz, nx)

print(f"heightfield y range: [{hf.min():.2f}, {hf.max():.2f}], mean {hf.mean():.2f}")
for i in (0, 4, 9):
    p = poses[i]
    R = np.array(p["R_rowmajor"])
    t = np.array(p["t"])
    C = -R.T @ t
    fwd = R.T @ np.array([0, 0, 1.0])
    up = -R.T @ np.array([0, 1.0, 0])
    gx = int(np.clip((C[0] - ox) / cell, 0, nx - 1))
    gz = int(np.clip((C[2] - oz) / cell, 0, nz - 1))
    ground = hf[gz, gx]
    print(f"cam {i} {p['name']}: C=[{C[0]:7.2f},{C[1]:7.2f},{C[2]:7.2f}] "
          f"ground_y={ground:7.2f} height={C[1]-ground:6.2f}m "
          f"fwd=[{fwd[0]:5.2f},{fwd[1]:5.2f},{fwd[2]:5.2f}] up=[{up[0]:5.2f},{up[1]:5.2f},{up[2]:5.2f}]")
