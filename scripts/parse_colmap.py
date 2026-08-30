"""Parse COLMAP text model + our keyframes manifest -> keyframes_poses.jsonl.

Expects work/<video>/colmap/sparse/txt/{cameras.txt,images.txt} (from
`colmap model_converter --output_type TXT`) and work/<video>/keyframes.jsonl.

Output rows: {file, t_sec, camera:{R_rowmajor, t, fx, fy, cx, cy, width, height}}
where x_cam = R @ x_world + t  (COLMAP convention, world->camera).

Usage:
  python parse_colmap.py --work work/rocks
"""
import argparse
import json
from pathlib import Path

import numpy as np


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    txt = args.work / "colmap" / "sparse" / "txt"
    kf_file = args.work / "keyframes.jsonl"

    cams = {}
    for line in (txt / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        cam_id, model = int(p[0]), p[1]
        nums = list(map(float, p[4:]))
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = nums[0], nums[1], nums[2]
            fx = fy = f
        elif model == "PINHOLE":
            fx, fy, cx, cy = nums[:4]
        else:  # OPENCV etc: fx fy cx cy + distortion; ignore distortion for eval FOV
            fx, fy, cx, cy = nums[:4]
        cams[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy)

    reg = {}
    for line in (txt / "images.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 10:
            continue  # 2D-points continuation line (header lines have exactly 10 tokens)
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        cam_id = int(p[8])
        name = p[9]
        reg[name] = dict(R=qvec2rot(q), t=t, cam_id=cam_id)

    kf_meta = {}
    if kf_file.exists():
        for l in kf_file.read_text(encoding="utf-8").splitlines():
            if l.strip():
                m = json.loads(l)
                kf_meta[m["file"]] = m.get("t_sec")

    rows = []
    for name, r in sorted(reg.items(), key=lambda kv: kf_meta.get(kv[0], 0)):
        c = cams[r["cam_id"]]
        rows.append({
            "file": name,
            "t_sec": kf_meta.get(name),
            "camera": {
                "R_rowmajor": [list(map(float, row)) for row in r["R"]],
                "t": list(map(float, r["t"])),
                **{k: float(c[k]) for k in ("fx", "fy", "cx", "cy")},
            },
        })
    out = args.work / "keyframes_poses.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"[poses] {len(rows)} registered / {len(kf_meta)} keyframes -> {out}")


if __name__ == "__main__":
    main()
