"""Learned open-vocabulary segmentation of a scene's keyframes (challenge D6).

What this is. A `py=PY310` step that names what the geometric heuristic in
`label_semantics.py` cannot: `scripts/survey_dynamics.py:1196` records the gap as
"class labels - a SAM2-class segmenter would name vehicles, humans and animals".
Two permissive models run in two passes over the keyframes:

  pass 1  GroundingDINO-tiny (172 M params) - open-vocabulary *detection*: one text
          prompt of noun phrases, so all classes are scored in a single forward pass
          per image. Boxes + scores only; no segmentation yet.
  pass 2  SAM 2.1 hiera-small (38 M params) - promptable *segmentation*: each kept
          box becomes a mask, and the masks are unioned per class into one uint8
          label map per frame, written beside a manifest.

The two passes are deliberately sequential and free the first model before loading
the second: on a 6 GB card GroundingDINO peaks at 1999 MiB and SAM 2 at 751 MiB,
so keeping both resident buys nothing and risks the training environment's own
headroom. `label_semantics.py` does the 3-D half (back-projecting these masks onto
the sparse cloud and fusing with the heuristic) in the CPU lane, which is why this
file never touches a point cloud.

Licences, checked at the source rather than trusted from a brief:
  microsoft/MoGe (D2b) and `Ruicheng/moge-2-*` weights ..... MIT
  HF `GroundingDinoForObjectDetection` (transformers) ...... Apache-2.0
  `IDEA-Research/grounding-dino-tiny` weights .............. Apache-2.0 (HF tag)
  facebookresearch/sam2 code ............................. Apache-2.0
  `facebook/sam2.1-hiera-small` weights .................. Apache-2.0 (HF tag)
  BERT text encoder inside grounding-dino ................ Apache-2.0
Nothing AGPL (Ultralytics YOLO was rejected on that ground) and nothing CC-BY-NC.

What this is NOT. It is not a classifier for the shipped cloud and it does not
decide any 3-D label: `learned/summary.json` records that no shipped artefact was
modified. Where the learned mask and the geometry disagree, `label_semantics.py`
counts it and reports it; disagreement is data here, not something to average away.

Coordinates. Boxes and masks stay in the keyframe's own pixel grid, at
`--mask-scale` of it. No metric length appears anywhere in this file, so no scene
scale can be corrupted from here.

Usage:
  .venv310\\Scripts\\python.exe scripts/learned_semantics.py --work work/rocks
  ... --limit 20 --min-score 0.22 --no-masks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

# Learned class -> the noun phrases GroundingDINO is prompted with. The class names
# are chosen to overlap `label_semantics.CLASSES` where the two can mean the same
# thing (road / building / vegetation / obstacle) and to add the four the geometric
# ladder structurally cannot produce (vehicle, person, animal, furniture, water):
# a vertical planar surface is a wall whether it belongs to a house or a shed, and
# height-above-ground cannot tell grass from a mown lawn.
CLASS_PROMPTS: dict[str, list[str]] = {
    "vegetation": ["tree", "plant", "grass", "shrub", "hedge", "flower"],
    "road": ["road", "street", "pavement", "sidewalk"],
    "building": ["building", "house", "wall", "facade", "roof", "bridge"],
    "obstacle": ["rock", "stone", "boulder", "pole", "fence", "barrier"],
    "vehicle": ["car", "truck", "bus", "van", "motorcycle", "bicycle", "boat"],
    "person": ["person", "pedestrian"],
    "animal": ["dog", "cat", "horse", "cow", "bird"],
    "furniture": ["table", "chair", "sofa", "bed", "cabinet", "shelf"],
    "water": ["water", "river", "pond", "swimming pool"],
}
# ground is deliberately absent: it is a geometric relation (the walkable surface
# at this scene's gravity), not an appearance, and a detector that has never seen
# the camera trajectory cannot name it. The heuristic keeps that class alone.

DETECTOR = "IDEA-Research/grounding-dino-tiny"
SEGMENTER = "facebook/sam2.1-hiera-small"
LICENCES = {
    "code": "transformers 4.56 (Apache-2.0): GroundingDinoForObjectDetection, Sam2Model",
    DETECTOR: "Apache-2.0 weights (HF `license: apache-2.0`), 172.2 M params",
    SEGMENTER: "Apache-2.0 weights (HF `license: apache-2.0`), 38.5 M params",
}

# Detection gate. GroundingDINO's own scores are poorly calibrated across
# vocabularies; the probe on work/rocks scored grass 0.58, rock 0.43, road 0.43 and
# a tail of empty sub-word matches at 0.20-0.21, so 0.22 sits above the noise floor
# this prompt actually produces while keeping the 0.3-class true positives. It is a
# score, not a length, and the fusion re-gates on view count.
MIN_SCORE = 0.22
MAX_BOXES_PER_FRAME = 24    # mask cost is per box; the tail past ~20 is duplicates
MIN_BOX_AREA = 0.0004       # fraction of the frame: below this a box cannot vote
MAX_BOX_AREA = 0.98         # a box covering the whole frame is a background lie
MASK_SCALE = 0.5            # label map resolution vs the keyframe grid: 4x less
                            # disk, and the fusion samples a 3x3 window anyway

# Detector input size, in the processor's own shortest-edge pixels. The shipped
# preprocessor config is shortest 800 / longest 1333, which on this 6 GB laptop
# measured 1999 MiB allocated and ~2.2 GiB reserved for GroundingDINO-tiny, and
# 2.34 s per frame at 640x360 (the frame is upscaled to 1333x749 first). Dropping
# the shortest edge to 640 keeps every prompt alive but costs less memory and time;
# the numbers recorded in learned/summary.json say which was used.
DET_SHORTEST_EDGE = 800
DET_EDGE_LADDER = (800, 640, 512)
# CUDA context + torch's own residency, measured as the difference between the
# card's free memory before and after `import torch; torch.cuda.is_available()`.
CUDA_CONTEXT_ALLOWANCE_GIB = 0.4
MIN_FREE_GIB = 2.6          # 2.2 GiB reserved at shortest-edge 800 + the allowance


def build_prompt(classes: list[str]) -> tuple[str, dict[str, str]]:
    """One ' . '-joined prompt plus phrase -> class, so labels are traceable."""
    phrases, phrase_class = [], {}
    for cls in classes:
        for p in CLASS_PROMPTS[cls]:
            if p in phrase_class and phrase_class[p] != cls:
                rb.warn(f"prompt '{p}' claimed by both {phrase_class[p]} and {cls}; "
                        f"keeping {phrase_class[p]}")
                continue
            phrase_class[p] = cls
            phrases.append(p)
    return " . ".join(phrases), phrase_class


def detect_pass(frames, *, detector: str, min_score: float, device: str,
                shortest_edge: int = DET_SHORTEST_EDGE):
    """GroundingDINO over every keyframe: [{file, detections:[{cls, box, score}]}]."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    proc = AutoProcessor.from_pretrained(detector)
    if shortest_edge != DET_SHORTEST_EDGE:
        # The shipped config's 800/1333 is what cost 2.2 GiB; scaling both edges
        # together keeps the aspect behaviour and the padding identical.
        proc.image_processor.size = {"shortest_edge": int(shortest_edge),
                                     "longest_edge": int(shortest_edge * 1333 / 800)}
    model = GroundingDinoForObjectDetection.from_pretrained(detector).to(device).eval()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    prompt, phrase_class = build_prompt(list(CLASS_PROMPTS))
    rows, secs, peak = [], [], 0.0
    for i, fr in enumerate(frames):
        path = fr["path"]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        t0 = time.time()
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        det = proc.post_process_grounded_object_detection(
            out, threshold=min_score, target_sizes=[[h, w]])[0]
        labels = det.get("text_labels")
        if labels is None:
            labels = [proc.tokenizer.decode([int(x)], skip_special_tokens=True)
                      for x in det["labels"].tolist()]
        boxes = []
        for score, phrase, box in zip(det["scores"].tolist(), labels,
                                      det["boxes"].tolist()):
            phrase = (phrase or "").strip().lower()
            cls = phrase_class.get(phrase)
            if cls is None:
                continue                     # sub-word artefact of the tokenizer
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(w * h)
            if not (MIN_BOX_AREA <= area <= MAX_BOX_AREA):
                continue
            boxes.append(dict(cls=cls, score=round(float(score), 4),
                              box=[round(v / w if k % 2 == 0 else v / h, 6)
                                   for k, v in enumerate([x1, y1, x2, y2])],
                              area_fraction=round(area, 5)))
        boxes.sort(key=lambda b: -b["score"])
        rows.append(dict(file=fr["file"], width=w, height=h, detections=boxes[:MAX_BOXES_PER_FRAME]))
        peak = max(peak, torch.cuda.max_memory_allocated(device) / 2 ** 20
                   if device.startswith("cuda") else 0.0)
        secs.append(time.time() - t0)
        print(f"[learned] detect {i + 1}/{len(frames)} {fr['file']} "
              f"{len(boxes[:MAX_BOXES_PER_FRAME])} boxes {secs[-1]:.2f}s", flush=True)
    stats = dict(detect_secs=round(float(np.sum(secs)), 1),
                 detect_secs_per_frame_median=round(float(np.median(secs)), 3) if secs else None,
                 detector_shortest_edge=int(shortest_edge),
                 detector_peak_mib=round(peak, 1) or None,
                 detector_peak_reserved_mib=round(
                     torch.cuda.max_memory_reserved(device) / 2 ** 20, 1)
                 if device.startswith("cuda") else None)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, stats, prompt


def mask_pass(frames, rows, out_dir: Path, *, segmenter: str, device: str,
              min_score: float, mask_scale: float, save_masks: bool):
    """SAM 2 over the kept boxes, unioned per class into one label map per frame."""
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    proc = Sam2Processor.from_pretrained(segmenter)
    model = Sam2Model.from_pretrained(segmenter).to(device).eval()
    if device.startswith("cuda"):
        # Reset, or this reports GroundingDINO's peak as SAM 2's.
        torch.cuda.reset_peak_memory_stats(device)
    index = {r["file"]: r for r in rows}
    classes = list(CLASS_PROMPTS)
    per_class_px: dict[str, int] = {}
    mask_rows, secs = [], []
    for i, fr in enumerate(frames):
        rec = index.get(fr["file"])
        if not rec or not rec["detections"]:
            mask_rows.append(dict(file=fr["file"], masks=0, coverage_fraction=0.0))
            continue
        img = Image.open(fr["path"]).convert("RGB")
        w, h = img.size
        keep = [d for d in rec["detections"] if d["score"] >= min_score]
        boxes = [[d["box"][0] * w, d["box"][1] * h, d["box"][2] * w, d["box"][3] * h]
                 for d in keep]
        t0 = time.time()
        inputs = proc(images=img, input_boxes=[boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, multimask_output=False)
        masks = out.pred_masks
        if masks.dim() == 5:                       # (1, N, 1, H, W) low-res logits
            masks = masks[:, :, 0]
        masks = torch.nn.functional.interpolate(
            masks.float(), size=(int(h * mask_scale), int(w * mask_scale)),
            mode="bilinear", align_corners=False) > 0
        masks = masks[0].cpu().numpy()             # (N, mh, mw) bool
        secs.append(time.time() - t0)
        mh, mw = masks.shape[1:]
        label = np.zeros((mh, mw), np.uint8)       # 0 = no learned evidence
        for j, d in enumerate(sorted(keep, key=lambda d: -d["score"])):
            if j >= len(masks):
                break
            cls_i = classes.index(d["cls"]) + 1
            take = masks[j] & (label == 0)         # highest score wins an overlap
            label[take] = cls_i
        filled = int((label > 0).sum())
        for j, cls in enumerate(classes):
            per_class_px[cls] = per_class_px.get(cls, 0) + int((label == j + 1).sum())
        name = mask_name(fr["file"])
        if save_masks:
            np.savez_compressed(out_dir / "masks" / f"{name}.npz", label=label,
                                scale=np.array([mask_scale], np.float32),
                                width=np.array([w]), height=np.array([h]))
        mask_rows.append(dict(file=fr["file"], mask=name, masks=int(len(boxes)),
                              mask_width=mw, mask_height=mh, covered_px=filled,
                              coverage_fraction=round(filled / float(mw * mh), 4)))
        print(f"[learned] mask {i + 1}/{len(frames)} {fr['file']} {len(boxes)} boxes "
              f"{filled / (mw * mh):.1%} of frame {secs[-1]:.2f}s", flush=True)
    stats = dict(mask_secs=round(float(np.sum(secs)), 1) if secs else 0.0,
                 mask_secs_per_frame_median=round(float(np.median(secs)), 3) if secs else None,
                 segmenter_peak_mib=round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)
                 if device.startswith("cuda") else None)
    return mask_rows, per_class_px, stats


def mask_name(file: str) -> str:
    """A keyframe's own path flattened into one safe stem.

    Keyframe names are `<clip>/<idx>.jpg`, so two clips of a multi-video take both
    own a `00000.jpg`: a bare stem would let one clip's masks silently overwrite the
    other's and the fusion would back-project a different scene's labels.
    """
    return file.replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]


def pick_frames(work: Path, poses, limit: int):
    """Keyframes that exist on disk, in the copy the trainer and COLMAP already chose."""
    frames = []
    for row in poses:
        name = row["file"]
        for base in ("frames_train", "frames_full", "frames_undist", "frames_match"):
            cand = work / base / name
            if cand.is_file():
                frames.append(dict(file=name, path=cand, t_sec=row.get("t_sec")))
                break
    if not frames:
        rb.die(rb.EMPTY_INPUT, "no keyframe jpg found under frames_train/frames_full - "
                              "learned_semantics runs after `poses`", code=2)
    return frames if limit <= 0 else frames[:limit]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default work/<scene>/learned")
    ap.add_argument("--detector", default=DETECTOR)
    ap.add_argument("--segmenter", default=SEGMENTER)
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--mask-scale", type=float, default=MASK_SCALE)
    ap.add_argument("--limit", type=int, default=0, help="at most N keyframes (0 = all)")
    ap.add_argument("--no-masks", dest="save_masks", action="store_false", default=True,
                    help="detect only; no 3-D fusion will be possible")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: cuda if free)")
    ap.add_argument("--strict", action="store_true", help="non-zero if nothing segmented")
    args = ap.parse_args(argv)
    rb.configure_streams()

    work = Path(args.work)
    out_dir = Path(args.out) if args.out else work / "learned"
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    started = time.time()

    poses = rb.jsonl_rows(work / "keyframes_poses.jsonl", required=("file", "camera"))
    if not poses:
        rb.die(rb.EMPTY_INPUT, f"{work / 'keyframes_poses.jsonl'} empty or absent", code=2)
    frames = pick_frames(work, poses, args.limit)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        rb.warn("no CUDA device: open-vocabulary detection on CPU is ~30 s/frame; "
                "this step is meant for the 310 lane with the GPU free")
    free = rb.free_vram_gb() if device.startswith("cuda") else None
    # GroundingDINO-tiny reserves ~2.2 GiB at its shipped 800-px shortest edge. On a
    # 6 GB laptop with a browser holding 3 GiB that does not fit, so the step walks
    # its own detector-input ladder down until it does, and refuses - naming the
    # number it needed - rather than OOM-ing half-way through a take.
    edge = DET_SHORTEST_EDGE
    if free is not None:
        need = {e: 2.2 * e / DET_SHORTEST_EDGE + CUDA_CONTEXT_ALLOWANCE_GIB
                for e in DET_EDGE_LADDER}
        for e in DET_EDGE_LADDER:
            if free >= need[e]:
                edge = e
                break
        else:
            rb.die(rb.OOM, f"only {free:.1f} GiB of VRAM is free; the smallest detector "
                          f"input this step knows about ({min(DET_EDGE_LADDER)} px) "
                          f"still needs {need[min(DET_EDGE_LADDER)]:.1f} GiB. Close "
                          f"whatever holds the GPU - or run with --device cpu, which is "
                          f"~30 s per frame.", code=2)
        if edge != DET_SHORTEST_EDGE:
            rb.warn(f"free VRAM {free:.1f} GiB < {need[DET_SHORTEST_EDGE]:.1f} GiB: "
                    f"detector input dropped from {DET_SHORTEST_EDGE} to {edge} px "
                    f"shortest edge (recorded in summary.json)")

    rows, det_stats, prompt = detect_pass(frames, detector=args.detector,
                                          min_score=args.min_score, device=device,
                                          shortest_edge=edge)
    mask_rows, per_class_px, mask_stats = [], {}, {}
    if args.save_masks:
        mask_rows, per_class_px, mask_stats = mask_pass(
            frames, rows, out_dir, segmenter=args.segmenter, device=device,
            min_score=args.min_score, mask_scale=args.mask_scale, save_masks=True)

    rb.write_json(out_dir / "detections.json",
                  dict(schema_version=1, prompt=prompt,
                       # The uint8 label map stores 1 + the index into this list, so
                       # the fusion in label_semantics.py can decode masks without
                       # importing this file (which would pull torch into the CPU lane).
                       label_classes=list(CLASS_PROMPTS), frames=rows))
    with (out_dir / "masks.jsonl").open("w", encoding="utf-8") as f:
        for r in mask_rows:
            f.write(json.dumps(r) + "\n")
    detected = {}
    for r in rows:
        for d in r["detections"]:
            detected[d["cls"]] = detected.get(d["cls"], 0) + 1
    covered = [r for r in mask_rows if r.get("covered_px")]
    summary = dict(
        schema_version=1, step="learned_semantics",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        status="ran" if detected else "no_detections",
        purpose=("per-image learned class masks for label_semantics.py to back-project; "
                 "advisory - nothing shipped is modified by this step"),
        models=dict(detector=args.detector, segmenter=args.segmenter if args.save_masks
                    else None, licences=LICENCES, prompt=prompt),
        gate=dict(min_score=args.min_score, mask_scale=args.mask_scale,
                  max_boxes_per_frame=MAX_BOXES_PER_FRAME,
                  min_box_area_fraction=MIN_BOX_AREA, max_box_area_fraction=MAX_BOX_AREA),
        run=dict(frames=len(frames), frames_with_detections=sum(
            1 for r in rows if r["detections"]),
            frames_with_masks=len(covered),
            detections_by_class=detected,
            mask_pixels_by_class=per_class_px or None,
            device=device, free_vram_gb_at_start=round(free, 2) if free else None,
            total_wall_secs=round(time.time() - started, 1), **det_stats, **mask_stats),
        modifies_shipped_artefacts=False,
    )
    rb.write_json(out_dir / "summary.json", summary)
    rb.write_json(out_dir / "done.json", dict(step="learned_semantics",
                                              status=summary["status"],
                                              frames=len(frames),
                                              secs=summary["run"]["total_wall_secs"]))
    print(f"[learned] {summary['run']['frames_with_detections']}/{len(frames)} frames "
          f"with detections; classes: {json.dumps(detected)}")
    print(f"[learned] peak detect {det_stats.get('detector_peak_mib')} MiB / mask "
          f"{mask_stats.get('segmenter_peak_mib')} MiB, "
          f"{summary['run']['total_wall_secs']} s total")
    print(f"[learned] wrote {out_dir}")
    return 3 if (args.strict and not detected) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except rb.StepError as exc:
        print(f"[learned] {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)
