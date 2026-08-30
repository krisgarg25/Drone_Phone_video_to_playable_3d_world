"""End-to-end selftest of the pose-prior path with REAL COLMAP.

Takes a frame subset from work/temple, synthesizes AR-style pose priors from
that scene's existing reconstruction (scaled to meters), then runs
run_colmap.py end-to-end: features -> prior injection -> sequential + spatial
matching -> pose_prior_mapper -> model_aligner -> TXT.

  .venv\\Scripts\\python.exe tests\\selftest_priors.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    src_txt = ROOT / "work/temple/colmap/sparse/txt"
    src_frames = ROOT / "work/temple/frames_train"
    if not src_txt.exists():
        sys.exit("run the temple scene once first (need its COLMAP TXT model)")

    # parse temple's registered cameras: name -> center C (world)
    centers = {}
    for line in (src_txt / "images.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 10:
            continue
        q = list(map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])))
        R = qvec2rot(q)
        centers[p[9]] = -R.T @ t

    names = sorted(n for n in centers if (src_frames / n).exists())
    take = names[:: max(1, len(names) // 30)][:30]
    print(f"[selftest] using {len(take)} temple frames as synthetic 'AR' scene")

    work = ROOT / "work/_selftest_priors"
    if work.exists():
        shutil.rmtree(work)
    clip = "a"
    (work / "frames_train" / clip).mkdir(parents=True)

    # scale camera centers so the walk spans ~12 m (AR-like metric scale)
    allC = np.array([centers[n] for n in take])
    span = float(np.linalg.norm(allC.max(0) - allC.min(0)))
    scale = 12.0 / max(span, 1e-9)

    manifest, priors = [], []
    for i, name in enumerate(take):
        fname = f"{i:05d}.jpg"
        shutil.copyfile(src_frames / name, work / "frames_train" / clip / fname)
        C = centers[name] * scale
        manifest.append({"file": f"{clip}/{fname}", "clip": clip,
                         "frame_index": i, "t_clip": round(i / 2.0, 3),
                         "t_sec": round(i / 2.0, 3)})
        priors.append({"file": f"{clip}/{fname}", "clip": clip,
                       "position": [float(x) for x in C],
                       "quat_xyzw_c2w": [0, 0, 0, 1],
                       "log_dt": 0.0, "std": 0.15})

    (work / "keyframes.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest), encoding="utf-8")
    (work / "pose_priors.jsonl").write_text(
        "\n".join(json.dumps(p) for p in priors), encoding="utf-8")

    r = subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"),
         str(ROOT / "scripts/run_colmap.py"), str(work),
         "--set", "overlap=10", "--set", "cross_clip=spatial",
         "--set", "exhaustive_max=10", "--set", "mapper=pose_prior"],
        cwd=ROOT)
    if r.returncode != 0:
        sys.exit("SELFTEST FAILED: run_colmap.py errored")

    txt = work / "colmap/sparse/txt/images.txt"
    reg = sum(1 for l in txt.read_text().splitlines()
              if l.strip() and not l.startswith("#") and len(l.split()) == 10)
    pct = 100 * reg / len(take)
    print(f"\n[selftest] registered {reg}/{len(take)} ({pct:.0f}%)")
    if pct < 80:
        sys.exit("SELFTEST FAILED: low registration")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
