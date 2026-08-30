"""Phone AR pose logs -> per-keyframe pose priors for COLMAP.

Reads work/<scene>/keyframes.jsonl (written by extract_keyframes.py), pairs
each keyframe with a sample from a per-clip AR log (ARCore / ARKit / Record3D),
and emits work/<scene>/pose_priors.jsonl consumed by run_colmap.py which
injects them into the COLMAP database as native pose priors (COLMAP >= 3.11).

Pairing rules:
  - video clips: log timestamps are matched to each frame's t_clip by
    interpolation (position lerp + quaternion slerp). Both log and frames are
    re-based to start at ~0 so epoch-stamped logs just work.
  - Record3D logs (metadata JSON): rows are 1:1 with exported rgb frames, so
    we match by frame index and stride (stride inferred from the manifest).

Usage:
  python import_phone_poses.py --work work/room \
      --log walk1=D:/logs/walk1_poses.jsonl --log walk2=D:/logs/walk2.csv \
      [--std 0.15] [--max-gap 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poses_lib import load_any, match_to_frames  # noqa: E402


def clip_stride(frame_indices: list[int]) -> int:
    diffs = [b - a for a, b in zip(frame_indices, frame_indices[1:]) if b > a]
    return max(1, round(sum(diffs) / len(diffs))) if diffs else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--log", action="append", default=[],
                    help="CLIP=PATH pose log for that clip (repeatable)")
    ap.add_argument("--std", type=float, default=0.15,
                    help="assumed prior position error in meters (BA softness)")
    ap.add_argument("--max-gap", type=float, default=0.25,
                    help="max seconds between frame and nearest log sample")
    args = ap.parse_args()

    kf_file = args.work / "keyframes.jsonl"
    if not kf_file.exists():
        sys.exit(f"missing {kf_file} - run extract_keyframes.py first")
    rows = [json.loads(l) for l in kf_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        sys.exit("empty manifest")

    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)

    clips_meta = {}
    cj = args.work / "clips.json"
    if cj.exists():
        clips_meta = {c["name"]: c for c in json.loads(cj.read_text(encoding="utf-8"))}

    logs: dict[str, tuple[list, str]] = {}
    for spec in args.log:
        name, _, path = spec.partition("=")
        samples, fmt = load_any(Path(path))
        if not samples:
            sys.exit(f"log {path}: no samples")
        logs[name] = (samples, fmt)

    unmatched = sorted(set(by_clip) - set(logs))
    if unmatched:
        print(f"[priors] WARNING no log given for clips: {unmatched} "
              f"(frames there get no prior)", flush=True)

    out_rows, stats = [], {}
    for clip, frames in by_clip.items():
        if clip not in logs:
            continue
        samples, fmt = logs[clip]
        n_before = len(out_rows)

        if fmt == "record3d":
            stride = clip_stride([f["frame_index"] for f in frames])
            total_dir = int(clips_meta.get(clip, {}).get("source_dir_count",
                                                         len(frames) * stride))
            for f in frames:
                orig_idx = f["frame_index"] * stride
                j = min(len(samples) - 1, round(orig_idx * len(samples) / max(total_dir, 1)))
                t, p, q = samples[j]
                out_rows.append({"file": f["file"], "clip": clip, "position": list(p),
                                 "quat_xyzw_c2w": list(q), "log_dt": 0.0})
        else:
            ts = [f["t_clip"] for f in frames]
            # re-base log to its own first sample so relative frame times line up
            base_t = samples[0][0]
            rebased = [(s[0] - base_t, s[1], s[2]) for s in samples]
            matched = match_to_frames(rebased, ts, max_gap=args.max_gap)
            for f, m in zip(frames, matched):
                if m is None or m["log_dt"] > args.max_gap:
                    continue
                out_rows.append({"file": f["file"], "clip": clip,
                                 "position": m["position"],
                                 "quat_xyzw_c2w": m["quat_xyzw_c2w"],
                                 "log_dt": m["log_dt"]})

        got = len(out_rows) - n_before
        stats[clip] = {"format": fmt, "samples": len(samples),
                       "matched": got, "of": len(frames)}
        print(f"[priors] {clip}: {got}/{len(frames)} frames matched "
              f"(fmt={fmt}, {len(samples)} log samples)")

    if not out_rows:
        sys.exit("[priors] nothing matched - wrong log/clip pairing?")

    # sanity: metric scale + non-degenerate spread
    P = [r["position"] for r in out_rows]
    extent = max((max(p[i] for p in P) - min(p[i] for p in P)) for i in range(3))
    if extent < 0.05:
        print(f"[priors] WARNING prior positions span only {extent:.3f}m - "
              "are these meters? Priors this tight are near-useless.", flush=True)

    out = args.work / "pose_priors.jsonl"
    for r in out_rows:
        r["std"] = args.std
    out.write_text("\n".join(json.dumps(r) for r in out_rows), encoding="utf-8")
    print(f"[priors] wrote {len(out_rows)} pose priors -> {out}")


if __name__ == "__main__":
    main()
