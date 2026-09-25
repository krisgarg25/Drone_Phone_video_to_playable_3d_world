"""Score, flatten and mask the frames a survey run actually matches on.

Runs between keyframe extraction and COLMAP. Everything the brief's frame-level
challenges need is decided here, on the frames that were written, so the legacy
splat pipeline is untouched:

``--select``   spends the frame budget on flight geometry (the capture plan) and
               drops frames whose blur/compression score says no matcher will use
               them - challenges (i) and (ii).
``--flatten``  writes a second image directory with the low-frequency illumination
               field divided out, which is what COLMAP reads; the colourised copies
               stay untouched, so a shadow never becomes a painted-over wall - (iii).
``--mask``     derives dynamic-object masks from epipolar inconsistency and writes
               them in the layout COLMAP's ``--ImageReader.mask_path`` expects - (iv).

The stage never invents a number: a frame with no GPS bracket is left unanchored and
counted, a mask that cannot be computed is reported as absent, and the report says
which of the three operations ran.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:  # imported as scripts.survey_frames by the tests
    from scripts import survey_capture as capture
    from scripts import survey_georef as georef
except ImportError:
    import survey_capture as capture
    import survey_georef as georef

MASK_WIDTH = 640
"""Masks are computed at this width and resized onto the matching images. A moving
car is a blob tens of pixels wide; full resolution buys the veto nothing and costs a
frame's memory per frame."""
MAX_MASK_FRAMES = 400
"""The mask pass holds downscaled greys, so it is bounded by frames as well as size."""


def _read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(str(Path(path).name) + " holds no rows")
    return rows


def _time_of(row):
    value = row.get("t_sec", row.get("t_clip"))
    if value is None:
        raise ValueError("keyframe " + str(row.get("file")) + " has no timestamp")
    return float(value)


def build(work, *, preparation, select=False, flatten=False, mask=False,
          plan=None, focal_px=None, energy_floor=None, min_baseline_m=None,
          progress=None):
    """Do the requested frame work and return the report the stage writes.

    ``work`` is the run directory holding ``keyframes.jsonl`` and ``frames_full``;
    ``preparation`` is the manifest from ``survey.py prepare``. Returns a dict that is
    safe to serialise and that never claims more than the flags asked for.
    """
    import cv2
    work = Path(work)
    rows = _read_rows(work / "keyframes.jsonl")
    times = [_time_of(row) for row in rows]
    telemetry = preparation["telemetry"]
    report = {"schema_version": 1, "frames_in": len(rows),
              "operations": {"select": bool(select), "flatten": bool(flatten),
                             "mask": bool(mask)}}

    positions, matched = capture._positions(times, telemetry)
    anchored = [index for index, hit in enumerate(matched) if hit]
    report["gps_anchored_frames"] = len(anchored)
    report["frames_without_a_gps_bracket"] = len(rows) - len(anchored)

    kept = list(range(len(rows)))
    if select:
        if plan is None:
            raise ValueError("--select needs a capture plan")
        targets = [float(value) for value in plan["target_times_s"]]
        baseline = plan.get("min_baseline_m", min_baseline_m or capture.DEFAULT_MIN_BASELINE_M)
        chosen, used = [], set()
        for target in targets:
            candidates = [index for index in anchored
                          if index not in used and index < len(times) - 1]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda index: abs(times[index] - target))
            if chosen and baseline:
                if np.linalg.norm(positions[nearest] - positions[chosen[-1]]) < baseline - 1e-9:
                    continue
            chosen.append(nearest)
            used.add(nearest)
        report["plan_targets"] = len(targets)
        report["plan_matched"] = len(chosen)
        report["unmatched_targets"] = len(targets) - len(chosen)
        kept = sorted(chosen)

    # frames_match is written from scratch and holds exactly the frames that survive.
    # COLMAP reads a directory, not this manifest, so a dropped frame left on disk
    # would still be matched; frames_train stays as extracted.
    match_dir = work / "frames_match"
    scores, previous_gray, dropped = [], None, []
    for index in kept:
        row = rows[index]
        path = work / "frames_full" / row["file"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("cannot read written keyframe " + str(path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = capture.assess_frame(gray, previous_gray=previous_gray,
                                     energy_floor=energy_floor)
        previous_gray = gray
        scores.append({"file": row["file"], "weight": score["weight"],
                       "blur_label": score["blur_label"], "reasons": score["reasons"],
                       "sharpness": score["sharpness"]})
        target = match_dir / row["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not score["keep"]:
            dropped.append(row["file"])
            target.unlink(missing_ok=True)
            continue
        source = work / "frames_train" / row["file"]
        if flatten:
            flat, actions = capture.prepare_for_matching(gray, flatten=True)
            train = cv2.imread(str(source)) if source.is_file() else None
            height, width = (train.shape[:2] if train is not None else gray.shape)
            resized = cv2.resize(flat, (width, height), interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(str(target), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("could not write matching frame " + str(target))
            scores[-1]["matching_copy"] = actions
        elif source.is_file():
            target.write_bytes(source.read_bytes())
    report["scored"] = len(scores)
    report["dropped_by_quality"] = dropped
    kept = [index for index in kept if rows[index]["file"] not in dropped]
    report["frames_out"] = len(kept)
    report["matching_dir"] = "frames_match"

    if mask and len(kept) >= 2:
        step = max(1, len(kept) // MAX_MASK_FRAMES)
        subset = kept[::step]
        grays, centres = [], []
        width = height = None
        for index in subset:
            path = match_dir / rows[index]["file"]
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("cannot read matching frame " + str(path))
            if width is None:
                height, width = image.shape[:2]
            scale = MASK_WIDTH / float(max(image.shape))
            grays.append(cv2.resize(image, (int(width * scale), int(height * scale)),
                                    interpolation=cv2.INTER_AREA))
            centres.append(positions[index])
        k_matrix = capture.camera_matrix(grays[0].shape[1], grays[0].shape[0],
                                         focal_px=(focal_px * scale if focal_px else None))
        results, meta = capture.masks_for_sequence(grays, np.asarray(centres), k_matrix)
        summary = capture.write_masks(results, [rows[index]["file"] for index in subset],
                                      work / "masks", images_dir=match_dir)
        # Every image COLMAP reads needs a mask entry, or the ones without a veto
        # would be silently trusted more than the ones with one.
        for index in kept:
            if str(rows[index]["file"]) + capture.MASK_FILE_SUFFIX not in {
                    entry["mask"] for entry in summary["files"]}:
                source = match_dir / rows[index]["file"]
                image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                path = work / "masks" / (str(rows[index]["file"]) + capture.MASK_FILE_SUFFIX)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.full(image.shape, 255, dtype=np.uint8))
                summary["files"].append({"image": str(rows[index]["file"]),
                                         "mask": str(rows[index]["file"]) + capture.MASK_FILE_SUFFIX,
                                         "masked_fraction": 0.0, "confidence": None,
                                         "note": "no motion evidence for this frame"})
        summary.update(mask_width=MASK_WIDTH, frames_masked=len(subset), frames_kept=len(kept),
                       coverage_cost=meta["coverage_cost"],
                       static_obstacle_note=meta["static_obstacle_note"])
        report["masks"] = summary

    if flatten:
        report["matching_dir"] = "frames_match"
    report["interpretation"] = ("frame-level preparation: what entered matching, not a "
                                "measurement of the reconstruction it produced")
    return report, kept, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--focal-px", dest="focal_px", type=float, default=None)
    parser.add_argument("--energy-floor", dest="energy_floor", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the report (default: <work>/survey/frames.json)")
    args = parser.parse_args(argv)
    preparation = json.loads(args.preparation.read_text(encoding="utf-8"))
    plan = capture.read_plan(args.plan) if args.plan else None
    try:
        report, kept, rows = build(args.work, preparation=preparation, select=args.select,
                                   flatten=args.flatten, mask=args.mask, plan=plan,
                                   focal_px=args.focal_px, energy_floor=args.energy_floor)
    except (ValueError, KeyError, IndexError) as error:
        print("[frames] " + str(error), file=sys.stderr)
        return 3
    out = args.out or args.work / "survey" / "frames.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if report.get("frames_out", len(rows)) != len(rows):
        # Rewrite the manifest COLMAP and the priors both read, so a frame dropped
        # here cannot reappear downstream under its old name. The extraction output
        # is kept beside it: this stage is a decision, and the decision is auditable.
        original = args.work / "keyframes_extracted.jsonl"
        manifest = args.work / "keyframes.jsonl"
        if not original.exists():
            manifest.replace(original)
            with original.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        with manifest.open("w", encoding="utf-8", newline="\n") as stream:
            for index in kept:
                stream.write(json.dumps(rows[index]) + "\n")
        report["manifest_rewritten"] = manifest.name
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("[frames] scored {}, kept {}, dropped {}, masks {}".format(
        report.get("scored", 0), report.get("frames_out", 0),
        len(report.get("dropped_by_quality", [])),
        report.get("masks", {}).get("count", 0)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
