"""Sanity-check ORIGINAL COLMAP camera centers vs sparse point cloud extents."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_heightfield import load_points  # noqa: E402

work = Path(__file__).resolve().parent.parent / "work" / "rocks"
rows = [json.loads(l) for l in (work / "keyframes_poses.jsonl").read_text().splitlines() if l.strip()]

xyz = load_points(work / "sparse_points.ply", None, 0.0, 10_000_000)
print(f"points3D: {len(xyz)}  extent min {xyz.min(0).round(2)} max {xyz.max(0).round(2)}")

Cs = []
for r in rows[::10]:
    R = np.array(r["camera"]["R_rowmajor"])
    t = np.array(r["camera"]["t"])
    C = -R.T @ t
    Cs.append(C)
    print(f"{r['file']}: C=[{C[0]:7.2f},{C[1]:7.2f},{C[2]:7.2f}]  |t|={np.linalg.norm(t):6.2f}")
Cs = np.array(Cs)
print(f"camera centers: min {Cs.min(0).round(2)} max {Cs.max(0).round(2)}")
