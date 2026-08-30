"""V2: one or more video clips -> sharpness-filtered keyframes.

Multi-clip layout (what COLMAP wants):
  work/<scene>/frames_full/<clip>/NNNNN.jpg     ground truth copies
  work/<scene>/frames_train/<clip>/NNNNN.jpg    downscaled SfM/training input

Frames live in PER-CLIP SUBFOLDERS so lexicographic image order keeps every
clip contiguous -- exactly how colmap sequential_matcher expects multi-clip
input. The global keyframe budget is split across clips proportional to
duration x motion-weight (from capture_diagnostics.py when available), then
each clip runs the proven selection: temporal bins -> sharpest per bin ->
adaptive blur floor -> near-duplicate rejection.

Manifest rows carry GLOBAL t_sec (monotonic across all clips) plus per-clip
fields; downstream steps sort by t_sec so clip boundaries stay invisible.

Pre-extracted frame folders (e.g., Record3D rgb exports) are supported via
--frames-dir NAME=PATH[:STRIDE]: frames are copied as-is (optionally strided),
no selection.

Usage:
  python extract_keyframes.py --work work/room \
      --video walk1.mp4 walk2.mp4 --target 600 --train-width 640
  python extract_keyframes.py --work work/room \
      --frames-dir rec3d=D:/r3d_export/rgbd --target 400
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

CANDIDATES_PER_TARGET = 4   # decode ~4x more candidates than we keep per clip


def probe(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "nb_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration"] = meta["nb_frames"] / fps if fps else 0.0
    cap.release()
    return meta


def select_clip(cands, target: int, min_sharpness: float | None):
    """Temporal-bin selection + adaptive blur floor + near-duplicate rejection."""
    n_bins = min(target, max(1, len(cands)))
    edges = np.linspace(0, len(cands), n_bins + 1).astype(int)
    picked = []
    for b in range(n_bins):
        seg = cands[edges[b]:edges[b + 1]]
        if seg:
            picked.append(max(seg, key=lambda c: c[1]))

    if min_sharpness is not None:
        blur_floor = min_sharpness
    else:
        # drop only catastrophic outliers: bottom 5% of *selected* frames,
        # halved -- indoor phone footage has naturally low Laplacian variance.
        sharps = [c[1] for c in picked]
        blur_floor = float(np.percentile(sharps, 5)) * 0.5 if sharps else 0.0

    kept, last_thumb, dupes = [], None, 0
    for cand in picked:
        _, sharp, _, thumb = cand
        if sharp < blur_floor:
            continue
        if last_thumb is not None:
            diff = float(np.abs(thumb.astype(np.float32) - last_thumb.astype(np.float32)).mean())
            if diff < 2.5:
                dupes += 1
                continue
        kept.append(cand)
        last_thumb = thumb
    return kept, blur_floor, dupes


def save_frames(video: Path, keep_indices: set[int], full_dir: Path, train_dir: Path,
                train_width: int, meta: dict):
    cap = cv2.VideoCapture(str(video))
    scale = train_width / meta["width"]
    train_h = round(meta["height"] * scale / 2) * 2
    rows, idx, saved = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep_indices:
            name = f"{saved:05d}.jpg"
            cv2.imwrite(str(full_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            train = cv2.resize(frame, (train_width, train_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(train_dir / name), train, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append((name, idx))
            saved += 1
        idx += 1
    cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--video", action="append", nargs="+", default=[],
                    help="video clip(s); space-separated, repeatable")
    ap.add_argument("--frames-dir", action="append", default=[],
                    help="pre-extracted frames: NAME=PATH[:STRIDE] (repeatable)")
    ap.add_argument("--target", type=int, default=500, help="total keyframes across all clips")
    ap.add_argument("--per-clip-min", type=int, default=60,
                    help="every clip gets at least this many keyframes")
    ap.add_argument("--train-width", type=int, default=640)
    ap.add_argument("--min-sharpness", type=float, default=None)
    ap.add_argument("--diagnostics", type=Path, default=None,
                    help="capture_diagnostics.json for motion-weighted budget split")
    args = ap.parse_args()

    if not args.video and not args.frames_dir:
        sys.exit("nothing to do: pass --video and/or --frames-dir")
    video_paths = [Path(v) for grp in args.video for v in grp]

    full_root = args.work / "frames_full"
    train_root = args.work / "frames_train"
    for d in (full_root, train_root):
        d.mkdir(parents=True, exist_ok=True)

    # ---- budget split ------------------------------------------------------
    diag = {}
    if args.diagnostics and Path(args.diagnostics).exists():
        diag = {c["clip"]: c.get("motion_weight", 1.0)
                for c in json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))["clips"]}

    clips = []          # dicts: name, kind, weight, meta
    for v in video_paths:
        m = probe(v)
        w = float(m["duration"]) * diag.get(v.stem, 1.0)
        clips.append(dict(name=v.stem, kind="video", path=v, meta=m, weight=max(w, 1.0)))
    for spec in args.frames_dir:
        name, _, rest = spec.partition("=")
        parts = rest.split(":")
        path, stride = Path(parts[0]), int(parts[1]) if len(parts) > 1 else 1
        files = sorted(p for p in path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        files = files[::stride]
        if not files:
            sys.exit(f"--frames-dir {spec}: no images under {path}")
        h, wdt = cv2.imread(str(files[0])).shape[:2]
        clips.append(dict(name=name, kind="dir", files=files, weight=float(len(files)),
                          meta=dict(width=wdt, height=h, fps=30.0, nb_frames=len(files),
                                    duration=len(files) / 30.0)))

    wsum = sum(c["weight"] for c in clips) or 1.0
    # many clips: shrink the per-clip floor so the global target still rules,
    # otherwise 10+ clips inflate the frame count beyond matchable size
    per_clip_floor = args.per_clip_min
    if len(clips) > 4:
        per_clip_floor = min(per_clip_floor, max(12, args.target // (2 * len(clips))))
    budgets = {c["name"]: max(per_clip_floor,
                              round(args.target * c["weight"] / wsum)) for c in clips}

    # ---- select per clip ---------------------------------------------------
    manifest, clips_meta = [], []
    t_global = 0.0
    for c in clips:
        name = c["name"]
        for sub in (full_root / name, train_root / name):
            if sub.exists():
                shutil.rmtree(sub)
            sub.mkdir(parents=True)

        if c["kind"] == "video":
            meta = c["meta"]
            total = meta["nb_frames"] or int(meta["duration"] * meta["fps"])
            budget = min(budgets[name], max(10, total // CANDIDATES_PER_TARGET))
            stride = max(1, round(total / (budget * CANDIDATES_PER_TARGET)))
            print(f"[keyframes] {name}: {meta['width']}x{meta['height']} @ {meta['fps']:.2f} "
                  f"({meta['duration']:.0f}s), budget={budget}, stride={stride}", flush=True)
            cap = cv2.VideoCapture(str(c["path"]))
            cands, idx = [], 0
            while True:
                ok = False
                for _ in range(stride):
                    ok, frame = cap.read()
                    idx += 1
                    if not ok:
                        break
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
                cands.append((idx - 1, sharp, float(gray.mean()), thumb))
            cap.release()
            kept, blur_floor, dupes = select_clip(cands, budget, args.min_sharpness)
            keep_set = {i: k for k, i in enumerate(sorted(i for i, _, _, _ in kept))}
            sharp_by_idx = {i: s for i, s, _, _ in kept}
            saved_rows = save_frames(c["path"], set(keep_set), full_root / name,
                                     train_root / name, args.train_width, meta)
            clip_dur = meta["duration"]
        else:
            meta, files = c["meta"], c["files"]
            print(f"[keyframes] {name}: importing {len(files)} pre-extracted frames", flush=True)
            saved_rows = []
            for k, src in enumerate(files):
                dst_name = f"{k:05d}.jpg"
                img = cv2.imread(str(src))
                h, wdt = img.shape[:2]
                scale = args.train_width / wdt
                train_h = round(h * scale / 2) * 2
                shutil.copyfile(src, full_root / name / dst_name)
                cv2.imwrite(str(train_root / name / dst_name),
                            cv2.resize(img, (args.train_width, train_h),
                                       interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_rows.append((dst_name, k))
            blur_floor, dupes, clip_dur = 0.0, 0, len(files) / meta["fps"]

        fps_eff = meta["fps"] or 30.0
        for fname, fidx in sorted(saved_rows, key=lambda r: r[1]):
            t_local = fidx / fps_eff
            manifest.append({
                "file": f"{name}/{fname}",
                "clip": name,
                "frame_index": fidx,
                "t_clip": round(t_local, 3),
                "t_sec": round(t_global + t_local, 3),
                **({"sharpness": round(sharp_by_idx[fidx], 1)} if c["kind"] == "video" else {}),
            })
        t_global += clip_dur
        clips_meta.append({
            "name": name, "kind": c["kind"], "fps": meta["fps"],
            "duration": round(clip_dur, 2), "budget": budgets[name],
            "kept": len(saved_rows), "blur_floor": round(blur_floor, 2), "dupes_dropped": dupes,
            **({"source": str(c["path"]) } if c["kind"] == "video" else
               {"source_dir_count": len(files)}),
        })
        print(f"[keyframes] {name}: kept {len(saved_rows)} "
              f"(floor={blur_floor:.1f}, {dupes} dupes dropped)")

    (args.work / "keyframes.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest), encoding="utf-8")
    (args.work / "clips.json").write_text(json.dumps(clips_meta, indent=2), encoding="utf-8")
    (args.work / "video_meta.json").write_text(json.dumps({
        "width": clips[0]["meta"]["width"], "height": clips[0]["meta"]["height"],
        "fps": clips[0]["meta"]["fps"],
        "nb_clips": len(clips),
        "total_keyframes": len(manifest),
    }, indent=2), encoding="utf-8")
    print(f"[keyframes] wrote {len(manifest)} keyframes from {len(clips)} clip(s) to {args.work}")


if __name__ == "__main__":
    main()
