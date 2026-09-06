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
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

CANDIDATES_PER_TARGET = 4   # decode ~4x more candidates than we keep per clip


def probe(video: Path) -> dict:
    """Container metadata plus the size frames ACTUALLY decode at.

    The declared width/height is the stored raster, not the picture: a phone clip
    written 1920x1080 with a rotate hint decodes as 1080x1920. Sizing train
    frames from the metadata is where a "--width 1280" run ended up handing the
    trainer 1280x2772 images and filled the GPU.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise rb.StepError(rb.EMPTY_INPUT, f"cannot open {video}", returncode=3)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "nb_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        declared = (meta["width"], meta["height"])
        ok, frame = cap.read()
        if not ok or frame is None:
            raise rb.StepError(
                rb.EMPTY_INPUT,
                f"{video.name} opens but yields no decodable frame - it is "
                f"corrupt, audio-only, or in a codec this build cannot read",
                returncode=3)
        meta["height"], meta["width"] = frame.shape[:2]
        meta["rotate_hint"] = declared != (meta["width"], meta["height"])
        if meta["rotate_hint"]:
            rb.warn(f"{video.name}: container says {declared[0]}x{declared[1]} but "
                    f"frames decode as {meta['width']}x{meta['height']} "
                    f"(rotation metadata) - using the decoded size")

        # WebM and VFR phone containers do not report either number honestly.
        # Measured on this repo's own footage: data.webm gives fps=1000.0 and
        # frame count -9223372036854775808, a 64-bit underflow. Trusting that made
        # duration -9.2e15 s, which poisoned the frame budget and every t_sec.
        meta["fps_measured"] = True
        if not (0 < fps <= 240.0):
            rb.warn(f"{video.name}: container claims {fps} fps; falling back to 30 "
                    f"and deriving the rate from the timestamps")
            fps, meta["fps"], meta["fps_measured"] = 30.0, 30.0, False
        if meta["nb_frames"] < 0 or meta["nb_frames"] > 5_000_000:
            # Unknown length: ask the decoder where the end is instead.
            claimed = meta["nb_frames"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0x7FFFFFFF)
            tail = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            meta["nb_frames"] = max(0, min(tail, 5_000_000))
            rb.warn(f"{video.name}: container reports {claimed} frames; probing the "
                    f"end of the stream found {meta['nb_frames']}")
        if not meta["nb_frames"]:
            # Seeking this same handle back to 0 does not reliably reset a WebM
            # decoder, so count on a fresh one. grab() advances the demuxer
            # without paying to decode each frame.
            n, counter = 0, cv2.VideoCapture(str(video))
            try:
                while counter.grab():
                    n += 1
            finally:
                counter.release()
            meta["nb_frames"] = n
            rb.warn(f"{video.name}: no usable length, counted {n} frames by "
                    f"demuxing")
        meta["duration"] = meta["nb_frames"] / fps if fps else 0.0
        # re-open cleanly so the caller's own scan starts at frame 0
        cap.release()
        cap = cv2.VideoCapture(str(video))
        return meta
    finally:
        # Without this the mp4 stays open on an exception path, and the next run
        # fails in shutil.rmtree with a Windows PermissionError that looks like a
        # corrupt work directory rather than a leaked handle.
        cap.release()


def _train_size(w: int, h: int, train_width: int, max_pixels: int | None = None):
    """(train_w, train_h) from a DECODED frame size, capped by side and pixels."""
    max_dim = max(w, h, 1)
    scale = train_width / max_dim
    tw = max(16, round(w * scale / 2) * 2)
    th = max(16, round(h * scale / 2) * 2)
    if max_pixels and tw * th > max_pixels:
        from robust import pixels_for
        tw, th = pixels_for(tw, th, max_pixels)
    return tw, th


def _encode_and_save(full_path: Path, train_path: Path, frame: np.ndarray,
                     train_w: int, train_h: int) -> bool:
    """Write one frame at both resolutions. cv2.imwrite returns False on failure."""
    ok_full = cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    train = cv2.resize(frame, (train_w, train_h), interpolation=cv2.INTER_AREA)
    ok_train = cv2.imwrite(str(train_path), train, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bool(ok_full and ok_train)


def save_frames(video: Path, keep_indices: set[int], full_dir: Path, train_dir: Path,
                train_width: int, meta: dict, max_pixels: int | None = None):
    rows, idx, saved, failed = [], 0, 0, []
    tw = th = None
    # A bounded window is the difference between a long clip and an OOM kill:
    # this used to hold one future (and one frame.copy()) per kept frame, ~15 GB
    # for 600 4K frames, and the process died with nothing printed.
    window = 4 * min(12, os.cpu_count() or 8)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise rb.StepError(rb.EMPTY_INPUT, f"cannot re-open {video}",
                               returncode=3)
        with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 8)) as pool:
            inflight = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in keep_indices:
                    if tw is None:
                        # the first decoded frame decides the geometry, not the header
                        tw, th = _train_size(frame.shape[1], frame.shape[0],
                                             train_width, max_pixels)
                        if (tw, th) != _train_size(meta["width"], meta["height"],
                                                   train_width, max_pixels):
                            rb.warn(f"{video.name}: sized train frames from the "
                                    f"decoded frame, not the container metadata")
                    name = f"{saved:05d}.jpg"
                    fut = pool.submit(_encode_and_save, full_dir / name,
                                      train_dir / name, frame.copy(), tw, th)
                    inflight.append((name, idx, fut))
                    saved += 1
                    while len(inflight) >= window:
                        n, i, f = inflight.pop(0)
                        if f.result():
                            rows.append((n, i))
                        else:
                            failed.append(n)
                idx += 1
            for n, i, f in inflight:
                if f.result():
                    rows.append((n, i))
                else:
                    failed.append(n)
    finally:
        cap.release()
    if failed:
        rb.warn(f"{video.name}: {len(failed)} frames failed to write and were left "
                f"out of the manifest")
    return sorted(rows, key=lambda r: r[1])


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


def _extract_one_clip(c: dict, name: str, meta: dict, clip_budget: int,
                      args, full_root: Path, train_root: Path) -> dict:
    """Select and write one clip's keyframes. Raises StepError if unusable."""
    if c["kind"] == "video":
        total = meta["nb_frames"] or int(meta["duration"] * meta["fps"])
        if total <= 0:
            raise rb.StepError(rb.EMPTY_INPUT,
                               f"{name}: reports {total} frames - nothing to "
                               f"sample from", returncode=3)
        budget = min(clip_budget, max(10, total // CANDIDATES_PER_TARGET))
        stride = max(1, round(total / (budget * CANDIDATES_PER_TARGET)))
        print(f"[keyframes] {name}: {meta['width']}x{meta['height']} @ "
              f"{meta['fps']:.2f} ({meta['duration']:.0f}s), budget={budget}, "
              f"stride={stride}", flush=True)
        cands, idx = [], 0
        cap = cv2.VideoCapture(str(c["path"]))
        try:
            if not cap.isOpened():
                raise rb.StepError(rb.EMPTY_INPUT, f"cannot open {name} for "
                                                    f"candidate scan", returncode=3)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
                cands.append((idx - 1, sharp, float(gray.mean()), thumb))
                if stride > 1:
                    for _ in range(stride - 1):
                        if not cap.grab():
                            break
                        idx += 1
        finally:
            # an unreleased handle keeps the mp4 open on Windows and the next
            # run then fails to clear its own work folders
            cap.release()
        if not cands:
            raise rb.StepError(rb.EMPTY_INPUT,
                               f"{name}: decoded 0 frames - the container opens "
                               f"but yields nothing usable", returncode=3)
        kept, blur_floor, dupes = select_clip(cands, budget, args.min_sharpness)
        keep_set = {i for i, _, _, _ in kept}
        sharp_by_idx = {i: s for i, s, _, _ in kept}
        rows = save_frames(c["path"], keep_set, full_root / name,
                           train_root / name, args.train_width, meta,
                           max_pixels=args.train_max_pixels)
        return dict(rows=rows, clip_dur=meta["duration"], blur_floor=blur_floor,
                    dupes=dupes, sharp_by_idx=sharp_by_idx, budget=budget)

    def _process_preextracted(k, src):
        img = cv2.imread(str(src))
        if img is None:
            return None
        dst_name = f"{k:05d}.jpg"
        h, wdt = img.shape[:2]
        train_w, train_h = _train_size(wdt, h, args.train_width,
                                      args.train_max_pixels)
        shutil.copyfile(src, full_root / name / dst_name)
        ok = cv2.imwrite(str(train_root / name / dst_name),
                         cv2.resize(img, (train_w, train_h),
                                    interpolation=cv2.INTER_AREA),
                         [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (dst_name, k) if ok else None

    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 8)) as pool:
        rows = [r for r in pool.map(lambda item: _process_preextracted(*item),
                                    enumerate(c["files"])) if r]
    return dict(rows=rows, clip_dur=len(c["files"]) / (meta["fps"] or 30.0),
                blur_floor=0.0, dupes=0, sharp_by_idx={}, budget=clip_budget)


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
    ap.add_argument("--train-max-pixels", dest="train_max_pixels", type=int,
                    default=None,
                    help="total pixels per training frame. --train-width caps one "
                         "side and is blind to orientation: a 1280-wide portrait "
                         "clip is 1280x2772 = 3.5 MP, which is what filled the "
                         "GPU and killed a training run with no traceback. Default "
                         "is derived from the graphics card actually present.")
    ap.add_argument("--min-sharpness", type=float, default=None)
    ap.add_argument("--diagnostics", type=Path, default=None,
                    help="capture_diagnostics.json for motion-weighted budget split")
    args = ap.parse_args()
    if args.train_max_pixels is None:
        # the same budget the trainer applies, so a frame written here is never
        # one that step cannot render
        args.train_max_pixels = rb.train_budget()["pixels"]
        print(f"[keyframes] training frames capped at "
              f"{args.train_max_pixels / 1e6:.2f} MP on this GPU "
              f"({rb.available_vram_gb():.1f} GiB free)", flush=True)

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
        # capture_diagnostics keys clips by file NAME ("walk1.mp4"); this looks
        # them up by STEM ("walk1"), so without the normalisation below every
        # weight lookup missed and --preset auto silently budgeted every clip
        # equally, motion diagnostics printed and all.
        d = rb.read_json(args.diagnostics, {"clips": []}) or {"clips": []}
        for c in d.get("clips", []):
            key = Path(str(c.get("clip", ""))).stem
            if key:
                diag[key] = float(c.get("motion_weight", 1.0))

    clips = []          # dicts: name, kind, weight, meta
    skipped = []
    for v in video_paths:
        try:
            m = probe(v)
        except rb.StepError as e:
            # One unreadable clip must not cost the whole scene: reconstruct from
            # the rest and say clearly what was left out.
            rb.warn(f"skipping clip {v.name}: {e}")
            skipped.append(v.name)
            continue
        w = float(m["duration"]) * diag.get(v.stem, 1.0)
        clips.append(dict(name=v.stem, kind="video", path=v, meta=m, weight=max(w, 1.0)))
    for spec in args.frames_dir:
        name, _, rest = spec.partition("=")
        parts = rest.split(":")
        path = Path(parts[0])
        # "rec3d=C:\frames\x" partitioned on ":" leaves parts[1] == "C" and a
        # Windows path that exists reporting as "no images".
        if len(parts) > 2 and re.fullmatch(r"[A-Za-z]", parts[1]):
            path = Path(f"{parts[0]}:{parts[1]}")
            parts = [str(path)] + parts[2:]
        stride = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        files = sorted(p for p in path.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[::stride]
        first = None
        for f in files:
            first = cv2.imread(str(f))
            if first is not None:
                break
        if first is None:
            rb.warn(f"--frames-dir {spec}: no readable images under {path}, skipping")
            skipped.append(name or str(path))
            continue
        h, wdt = first.shape[:2]
        clips.append(dict(name=name, kind="dir", files=files, weight=float(len(files)),
                          meta=dict(width=wdt, height=h, fps=30.0, nb_frames=len(files),
                                    duration=len(files) / 30.0)))
    if not clips:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            "no usable source produced a single frame"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
            + " - check the clip paths and that the files are videos this build "
              "can decode", returncode=3)

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
        meta = c["meta"]
        for sub in (full_root / name, train_root / name):
            try:
                if sub.exists():
                    shutil.rmtree(sub)
            except PermissionError as e:
                # A viewer or an interrupted run can still hold a frame open on
                # Windows. Everything in here is about to be rewritten, so warn
                # and carry on rather than failing the whole scene.
                rb.warn(f"could not clear {sub.name} ({e}); reusing what is there")
            sub.mkdir(parents=True, exist_ok=True)

        try:
            one = _extract_one_clip(c, name, meta, budgets[name], args,
                                    full_root, train_root)
        except rb.StepError as e:
            rb.warn(f"clip '{name}' failed and was skipped: {e}")
            skipped.append(name)
            continue
        rows = one["rows"]
        clip_dur, blur_floor, dupes = one["clip_dur"], one["blur_floor"], one["dupes"]
        sharp_by_idx, budget = one["sharp_by_idx"], one["budget"]
        if not rows:
            rb.warn(f"clip '{name}' produced no keyframes "
                    f"(blur floor {blur_floor:.1f}, {dupes} duplicates)")
            skipped.append(name)
            continue
        fps_eff = meta["fps"] or 30.0
        for fname, fidx in rows:
            t_local = fidx / fps_eff
            entry = {
                "file": f"{name}/{fname}",
                "clip": name,
                "frame_index": fidx,
                "t_clip": round(t_local, 3),
                "t_sec": round(t_global + t_local, 3),
            }
            # sharpness is only known for decoded video, and only for a frame that
            # actually survived selection; a dropped write must not KeyError here.
            if c["kind"] == "video" and fidx in sharp_by_idx:
                entry["sharpness"] = round(sharp_by_idx[fidx], 1)
            manifest.append(entry)
        t_global += clip_dur
        clips_meta.append({
            "name": name, "kind": c["kind"], "fps": meta["fps"],
            "width": meta["width"], "height": meta["height"],
            "orientation": "portrait" if meta["height"] > meta["width"] else "landscape",
            "duration": round(clip_dur, 2), "budget": budget,
            "kept": len(rows), "blur_floor": round(blur_floor, 2),
            "dupes_dropped": dupes,
            **({"source": str(c["path"])} if c["kind"] == "video" else
               {"source_dir_count": len(c["files"])}),
        })
        print(f"[keyframes] {name}: kept {len(rows)} "
              f"(floor={blur_floor:.1f}, {dupes} dupes dropped)")

    if skipped:
        rb.warn(f"{len(skipped)} source(s) contributed nothing: "
                f"{', '.join(skipped)}")
    # An empty manifest used to be written with exit 0. The pipeline treats the
    # keyframes step as done because the file exists, so COLMAP then receives no
    # images and the failure surfaces several steps and one hour later as an
    # unrelated-looking error somewhere else.
    if not manifest:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"no keyframes from {len(clips)} source(s)"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
            + ". Nothing downstream can be built from an empty manifest, so "
              "stopping here rather than in colmap.", returncode=3)
    below = [cm["name"] for cm in clips_meta if cm["kept"] < 3]
    if below:
        rb.warn(f"clip(s) with too few frames to reconstruct on their own: "
                f"{', '.join(below)}")

    rb.write_text(args.work / "keyframes.jsonl",
                  "\n".join(json.dumps(m) for m in manifest) + "\n")
    rb.write_json(args.work / "clips.json", clips_meta)

    # Frame geometry per size group, because the size decides whether COLMAP can
    # use one shared camera: run with `single_camera` against a second geometry it
    # answers per image with CAMERA_SINGLE_DIM_ERROR and STILL EXITS 0, so those
    # frames never reach the database. Measured here on a two-clip scene: 120
    # keyframes written, 60 images in the database, and the mapper printed its
    # registration percentage against 120 — which reads like bad tracking, not
    # like half the footage having been dropped. run_colmap.py detects the split
    # and switches to one camera per clip; the warning is so it is never a surprise.
    by_size: dict[str, int] = {}
    for cm in clips_meta:
        by_size.setdefault(f"{cm['width']}x{cm['height']}", 0)
        by_size[f"{cm['width']}x{cm['height']}"] += cm["kept"]
    mixed = len(by_size) > 1
    if mixed:
        print(f"[keyframes] WARNING mixed frame geometries: "
              + ", ".join(f"{k} ({v} frames)" for k, v in sorted(by_size.items()))
              + " -> COLMAP gets one camera per clip", flush=True)
    (args.work / "video_meta.json").write_text(json.dumps({
        "width": clips[0]["meta"]["width"], "height": clips[0]["meta"]["height"],
        "fps": clips[0]["meta"]["fps"],
        "nb_clips": len(clips),
        "total_keyframes": len(manifest),
        "mixed_geometry": mixed,
        "frames_by_size": by_size,
        "clips": [{"name": cm["name"], "width": cm["width"], "height": cm["height"],
                   "orientation": cm["orientation"], "kept": cm["kept"]}
                  for cm in clips_meta],
    }, indent=2), encoding="utf-8")
    print(f"[keyframes] wrote {len(manifest)} keyframes from {len(clips)} clip(s) to {args.work}")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[keyframes] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
