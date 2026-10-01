"""Export COLMAP keypoints and verified matches for the film's camera-solve beat.

    .venv/Scripts/python.exe video_production/rig/export_features.py [--work work/rocks_quality]

Writes film/assets/pipeline/features.json: for two keyframes, a sample of the SIFT
keypoints COLMAP found (x, y in 0..1 of the frame, and scale in px) and a sample of the
geometrically verified matches between them. Numbers are the real totals, not the sample.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PAIR_MAX = 2147483647


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=ROOT / "work" / "rocks_quality")
    ap.add_argument("--a", default="rocks/00035.jpg")
    ap.add_argument("--b", default="rocks/00041.jpg")
    ap.add_argument("--keypoints", type=int, default=1400)
    ap.add_argument("--matches", type=int, default=260)
    ap.add_argument("--out", type=Path, default=ROOT / "video_production/film/assets/pipeline/features.json")
    args = ap.parse_args()

    db = sqlite3.connect(args.work / "database.db")
    ids = {name: i for i, name in db.execute("select image_id, name from images")}
    cam = db.execute("select width, height from cameras limit 1").fetchone()
    w, h = float(cam[0]), float(cam[1])
    rng = np.random.default_rng(7)

    def keypoints(name):
        rows, cols, blob = db.execute("select rows, cols, data from keypoints where image_id=?",
                                      (ids[name],)).fetchone()
        kp = np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)
        scale = np.sqrt(np.abs(kp[:, 2] * kp[:, 5] - kp[:, 3] * kp[:, 4])) if cols >= 6 else np.full(rows, 2.0)
        return kp[:, :2], scale

    a, b = ids[args.a], ids[args.b]
    flip = a > b
    lo, hi = (b, a) if flip else (a, b)
    rows, cols, blob = db.execute("select rows, cols, data from two_view_geometries where pair_id=?",
                                  (lo * PAIR_MAX + hi,)).fetchone()
    m = np.frombuffer(blob, dtype=np.uint32).reshape(rows, cols)
    if flip:
        m = m[:, ::-1]

    frames = {}
    for key, name in (("a", args.a), ("b", args.b)):
        xy, sc = keypoints(name)
        pick = rng.choice(len(xy), size=min(args.keypoints, len(xy)), replace=False)
        frames[key] = {"name": name, "total": int(len(xy)),
                       "points": [[round(float(xy[i, 0] / w), 4), round(float(xy[i, 1] / h), 4),
                                   round(float(sc[i]), 1)] for i in pick]}
    xa, _ = keypoints(args.a)
    xb, _ = keypoints(args.b)
    pick = rng.choice(len(m), size=min(args.matches, len(m)), replace=False)
    matches = [[round(float(xa[m[i, 0], 0] / w), 4), round(float(xa[m[i, 0], 1] / h), 4),
                round(float(xb[m[i, 1], 0] / w), 4), round(float(xb[m[i, 1], 1] / h), 4)] for i in pick]

    n_img = db.execute("select count(*) from images").fetchone()[0]
    n_pairs, n_inl = db.execute("select count(*), sum(rows) from two_view_geometries where rows > 0").fetchone()
    out = {"width": w, "height": h, "frames": frames,
           "matches": {"total": int(len(m)), "lines": matches},
           "scene": {"images": int(n_img), "verified_pairs": int(n_pairs), "verified_matches": int(n_inl)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{args.a}: {frames['a']['total']} keypoints, {args.b}: {frames['b']['total']}, "
          f"{len(m)} verified matches; scene {n_img} images, {n_pairs} pairs, {n_inl} matches -> {args.out}")


if __name__ == "__main__":
    main()
