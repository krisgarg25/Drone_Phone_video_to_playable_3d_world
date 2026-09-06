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
import robust as rb  # noqa: E402
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
    ap.add_argument("--strict", action="store_true",
                    help="fail the step when a log cannot be paired. Off by "
                         "default: priors are an optional boost to the solve, and "
                         "colmap's rescue ladder can run without them, so a bad AR "
                         "log should not cost the whole scene.")
    args = ap.parse_args()

    def bail(msg: str) -> None:
        """Degrade, unless the operator asked for a hard failure."""
        if args.strict:
            raise rb.StepError(rb.EMPTY_INPUT, msg, returncode=3)
        rb.warn(msg + "  -> continuing WITHOUT pose priors")
        rb.write_json(args.work / "priors_skipped.json", {"reason": msg})
        sys.exit(0)

    rows = rb.jsonl_rows(args.work / "keyframes.jsonl", required=("clip", "file"))
    if not rows:
        bail(f"no usable keyframes manifest at {args.work / 'keyframes.jsonl'}")

    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)

    clips_meta = {}
    cj = args.work / "clips.json"
    if cj.exists():
        clips_meta = {c["name"]: c for c in (rb.read_json(cj, []) or [])
                      if isinstance(c, dict) and "name" in c}

    logs: dict[str, tuple[list, str]] = {}
    for spec in args.log:
        name, sep, path = spec.partition("=")
        if not sep:
            bail(f"--log '{spec}' is not CLIP=PATH form")
        p = Path(path)
        if not p.exists():
            bail(f"pose log '{name}' not found: {p}")
        try:
            samples, fmt = load_any(p)
        except Exception as e:
            bail(f"pose log '{name}' could not be read ({type(e).__name__}: {e})")
        # A non-finite sample must never reach COLMAP: max()/min() over a list
        # holding NaN gives NaN, the extent sanity check below compares NaN and
        # gets False, and the poisoned prior sails through to bundle adjustment.
        good = [(t, pos, quat) for t, pos, quat in samples
                if rb.finite(t, *pos, *quat)]
        if len(good) != len(samples):
            rb.warn(f"pose log '{name}': dropped {len(samples) - len(good)} "
                    f"non-finite samples of {len(samples)}")
        if not good:
            bail(f"log {p}: no finite samples")
        logs[name] = (good, fmt)

    unmatched = sorted(set(by_clip) - set(logs))
    if unmatched:
        print(f"[priors] WARNING no log given for clips: {unmatched} "
              f"(frames there get no prior)", flush=True)
    stale = sorted(set(logs) - set(by_clip))
    if stale:
        rb.warn(f"logs given for clips with no keyframes: {stale}")

    out_rows, stats = [], {}
    for clip, frames in by_clip.items():
        if clip not in logs:
            continue
        samples, fmt = logs[clip]
        n_before = len(out_rows)

        if fmt == "record3d":
            stride = clip_stride([f.get("frame_index", 0) for f in frames])
            total_dir = int(clips_meta.get(clip, {}).get("source_dir_count",
                                                         len(frames) * stride))
            for f in frames:
                orig_idx = f.get("frame_index", 0) * stride
                # clamp BOTH ends: only the top was bounded, so a short log made
                # a negative index read backwards from the end of the sample list
                j = min(len(samples) - 1, max(0, round(orig_idx * len(samples)
                                                       / max(total_dir, 1))))
                t, p, q = samples[j]
                out_rows.append({"file": f["file"], "clip": clip, "position": list(p),
                                 "quat_xyzw_c2w": list(q), "log_dt": 0.0})
        else:
            ts = [f.get("t_clip", f.get("t_sec", 0.0)) for f in frames]
            # re-base log to its own first sample so relative frame times line up
            base_t = samples[0][0]
            rebased = [(s[0] - base_t, s[1], s[2]) for s in samples]
            matched = match_to_frames(rebased, ts, max_gap=args.max_gap)
            for f, m in zip(frames, matched):
                if m is None or m["log_dt"] > args.max_gap:
                    continue
                if not rb.finite(*m["position"], *m["quat_xyzw_c2w"]):
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
        if got == 0:
            rb.warn(f"clip '{clip}': {len(samples)} log samples matched none of its "
                    f"{len(frames)} keyframes - check --max-gap and the clip naming")

    if not out_rows:
        bail("[priors] nothing matched - wrong log/clip pairing, or the log's "
             "clock does not line up with the frames")

    # sanity: metric scale + non-degenerate spread
    P = [r["position"] for r in out_rows]
    extent = max((max(p[i] for p in P) - min(p[i] for p in P)) for i in range(3))
    if extent < 0.05:
        print(f"[priors] WARNING prior positions span only {extent:.3f}m - "
              "are these meters? Priors this tight are near-useless.", flush=True)

    out = args.work / "pose_priors.jsonl"
    for r in out_rows:
        r["std"] = args.std
    rb.write_text(out, "\n".join(json.dumps(r) for r in out_rows) + "\n")
    stale_marker = args.work / "priors_skipped.json"
    if stale_marker.exists():
        stale_marker.unlink()
    print(f"[priors] wrote {len(out_rows)} pose priors -> {out}")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[priors] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
