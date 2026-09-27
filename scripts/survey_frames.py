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
               This is the step the main pipeline runs (``pipeline.py``, step
               ``frames``) when the capture probe measured moving light; it needs no
               telemetry and no reconstruction, and pairs with ``--keep-all`` so the
               matching directory stays one-for-one with ``keyframes.jsonl``.
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


def _colour_copy_digest(dirpath):
    """One digest over every byte of the directory the splat is colourised from.

    ``flatten`` exists to remove light, so the single outcome that must never happen is
    it reaching ``frames_train`` - a shadow painted into the colour copy becomes
    permanent albedo in the model. Nothing in this module writes there, so this is not
    the check that catches a bug, it is the receipt the step leaves in the scene
    directory: the digest is recorded before and after every pass, so any run can be
    audited later without re-deriving it from the extraction. Cost is one read of the
    JPEGs twice, about 0.1 s for 72 frames.
    """
    import hashlib
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(dirpath.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(dirpath).as_posix().encode("utf-8"))
            h.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return h.hexdigest()[:16]


def build(work, *, preparation=None, select=False, flatten=False, mask=False,
          plan=None, focal_px=None, energy_floor=None, min_baseline_m=None,
          camera_centres=None, k_matrix=None, scene_depth_range=None, keep_all=False,
          progress=None):
    """Do the requested frame work and return the report the stage writes.

    ``work`` is the run directory holding ``keyframes.jsonl`` and ``frames_full``;
    ``preparation`` is the manifest from ``survey.py prepare``. Returns a dict that is
    safe to serialise and that never claims more than the flags asked for.

    Camera centres normally come from the GPS that also georeferences the model. A
    caller with no telemetry can pass ``camera_centres`` instead - one centre per
    keyframe, in whatever units its own reconstruction uses - which unlocks ``--mask``
    but not ``--select``, because the frame budget is spent along a flight path that a
    pose set without a metric scale cannot describe. Neither is needed to flatten:
    the illumination pass is a per-frame operation, and the main pipeline has no
    telemetry and no reconstruction at the point it runs.

    ``keep_all`` writes the matching copy for every extracted keyframe instead of
    dropping the ones the quality score dislikes. The survey lane wants the drop; the
    pipeline lane does not, because the keyframe budget was already spent upstream and
    a matching directory that no longer lines up with ``keyframes.jsonl`` means COLMAP
    registers frames the trainer never sees, or the reverse.
    """
    import cv2
    work = Path(work)
    rows = _read_rows(work / "keyframes.jsonl")
    times = [_time_of(row) for row in rows]
    report = {"schema_version": 1, "frames_in": len(rows),
              "operations": {"select": bool(select), "flatten": bool(flatten),
                             "mask": bool(mask)}}

    if camera_centres is not None:
        positions = np.asarray(camera_centres, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3) or len(positions) != len(rows):
            raise ValueError("camera_centres must hold one 2- or 3-vector per keyframe, got "
                             f"{positions.shape} for {len(rows)} frames")
        anchored = list(range(len(rows)))
        report["position_source"] = ("reconstructed poses, not GPS. The epipolar test is "
                                     "scale-free in these units, so this is a mask and not "
                                     "a georeference")
    elif preparation is not None:
        telemetry = preparation["telemetry"]
        positions, matched = capture._positions(times, telemetry)
        anchored = [index for index, hit in enumerate(matched) if hit]
        report["gps_anchored_frames"] = len(anchored)
        report["frames_without_a_gps_bracket"] = len(rows) - len(anchored)
    elif select or mask:
        raise ValueError("--select and --mask both need camera centres: pass "
                         "--preparation (GPS) or --poses (a reconstruction)")
    else:
        positions, anchored = None, list(range(len(rows)))
        report["position_source"] = ("none needed: scoring and flattening are per-frame "
                                     "operations, so this run has no camera centres and "
                                     "says nothing about geometry")

    kept = list(range(len(rows)))
    if select:
        if camera_centres is not None:
            raise ValueError("--select spends the frame budget on the GPS flight path; it "
                             "cannot run on reconstructed poses")
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
    colour_dir = work / "frames_train"
    copy_before = _colour_copy_digest(colour_dir)
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
        if not score["keep"] and not keep_all:
            dropped.append(row["file"])
            target.unlink(missing_ok=True)
            continue
        if not score["keep"]:
            # Kept under --keep-all: the score still goes in the report, so a reviewer
            # can see the frame was weak without a silently shorter frame list.
            scores[-1]["kept_despite"] = "quality drop disabled by keep_all"
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
    report["frame_scores"] = scores
    report["dropped_by_quality"] = dropped
    kept = [index for index in kept if rows[index]["file"] not in dropped]
    # Anything still in frames_match that this pass did not write would still be
    # matched: COLMAP reads the directory, not this manifest. That is leftovers from a
    # previous extraction or from the frames --select dropped, and it silently costs
    # the run the correspondence between the images COLMAP sees and the keyframes the
    # trainer reads by name.
    stale = [p for p in sorted(match_dir.rglob("*")) if p.is_file()
             and p.relative_to(match_dir).as_posix() not in {rows[i]["file"] for i in kept}]
    for path in stale:
        path.unlink()
    if keep_all:
        report["frame_budget_note"] = ("every extracted keyframe kept its matching copy; "
                                      "weak frames are scored in this report, not dropped")
    if stale:
        report["stale_matching_files_removed"] = [p.name for p in stale]
    report["frames_out"] = len(kept)
    report["matching_dir"] = "frames_match"

    if mask and len(kept) >= 2:
        step = max(1, len(kept) // MAX_MASK_FRAMES)
        subset = kept[::step]
        grays, centres = [], []
        width = height = None
        scale = 1.0
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
        if k_matrix is None:
            k_matrix = capture.camera_matrix(grays[0].shape[1], grays[0].shape[0],
                                             focal_px=(focal_px * scale if focal_px else None))
        else:
            # A supplied matrix describes the full-size frames; the greys handed to
            # the epipolar test are downscaled, so it has to come along at the same scale.
            k_matrix = np.asarray(k_matrix, dtype=np.float64) * scale
            k_matrix[2, 2] = 1.0
        options = {} if scene_depth_range is None else {"scene_depth_range": scene_depth_range}
        results, meta = capture.masks_for_sequence(grays, np.asarray(centres), k_matrix,
                                                   **options)
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
    copy_after = _colour_copy_digest(colour_dir)
    report["colour_copy"] = {
        "dir": "frames_train",
        "read_by": "the trainer, which colourises the splat from the raw extraction",
        "digest_before": copy_before,
        "digest_after": copy_after,
        "unchanged": copy_before == copy_after,
        "note": ("this pass writes frames_match only. A flattened colour copy would turn "
                 "a shadow into painted-on albedo, so the digest of the copy that is "
                 "trained on is recorded on both sides of the pass"),
    }
    report["interpretation"] = ("frame-level preparation: what entered matching, not a "
                                "measurement of the reconstruction it produced")
    return report, kept, rows


def _poses_centres(path, rows):
    """Camera centres + intrinsics from a reconstruction, lined up with keyframes.jsonl.

    ``parse_colmap`` stores the world->camera pose, so the centre is ``-R^T t``. The
    reconstruction's own units carry through untouched: the epipolar test compares a
    baseline against a depth in the same units, which is what makes a scale-free model
    usable here.
    """
    import numpy as np
    by_name, focals = {}, set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cam = row["camera"]
        R = np.asarray(cam["R_rowmajor"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
        by_name[row["file"]] = (-R.T @ t, cam)
        focals.add((round(float(cam["fx"]), 3), round(float(cam["fy"]), 3)))
    missing = [row["file"] for row in rows if row["file"] not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} keyframes have no reconstructed pose, first missing "
                         f"is {missing[0]!r}; masking needs one camera centre per frame")
    centres = np.asarray([by_name[row["file"]][0] for row in rows])
    camera = by_name[rows[0]["file"]][1]
    k = [[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]]
    return centres, np.asarray(k, dtype=np.float64), len(focals) == 1


def mask_existing(work, *, images_dir, poses, out, depth_range=None):
    """Mask the frames a reconstruction actually matched on, and write nothing else.

    ``build`` always rewrites ``frames_match`` from the raw extraction. That is correct
    when this stage owns that directory and destructive when it does not: a dynamic
    pass running after photometric normalisation would silently hand COLMAP unflattened
    frames back. Masks are geometric, so they can be derived from whichever copy the
    matcher read without deciding to change what it reads next time.
    """
    import cv2
    work = Path(work)
    images = Path(work) / images_dir if not Path(images_dir).is_absolute() else Path(images_dir)
    if not images.is_dir():
        raise ValueError(f"--images {images_dir!r} does not exist under {work}")
    rows = _read_rows(work / "keyframes.jsonl")
    centres, k_matrix, shared_focal = _poses_centres(poses, rows)

    grays, scale = [], 1.0
    for row in rows:
        path = images / row["file"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("cannot read the image a mask is being computed for: " + str(path))
        scale = MASK_WIDTH / float(max(image.shape))
        grays.append(cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA))
    k_scaled = k_matrix * scale
    k_scaled[2, 2] = 1.0

    options = {} if depth_range is None else {"scene_depth_range": tuple(depth_range)}
    results, meta = capture.masks_for_sequence(grays, centres, k_scaled, **options)
    summary = capture.write_masks(results, [row["file"] for row in rows],
                                 work / "masks", images_dir=images)
    report = {"schema_version": 1, "stage": "mask-existing", "frames_in": len(rows),
              "frames_out": len(rows), "matched_dir": str(images_dir),
              "intrinsics": "reconstructed camera, shared across frames" if shared_focal
                            else "reconstructed cameras disagree on focal; the first frame's "
                                "matrix was used for every epipolar test",
              "position_source": ("reconstructed poses, not GPS. The epipolar test is "
                                  "scale-free in these units, so this is a mask and not a "
                                  "georeference"),
              "scene_depth_range": None if depth_range is None else list(depth_range),
              "masks": summary}
    summary.update(mask_width=MASK_WIDTH, frames_masked=len(rows), frames_kept=len(rows),
                   coverage_cost=meta["coverage_cost"],
                   static_obstacle_note=meta["static_obstacle_note"])
    if depth_range is None:
        report["depth_range_warning"] = ("no scene depth range supplied, so no pixel is vetoed "
                                        "as static parallax: a panning camera is not "
                                        "separable from a moving wall on this pass")
    report["interpretation"] = ("masks for the frames a reconstruction matched on; a decision "
                                "about what may be trusted, not a measurement of the scene")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("[frames] masked {} frames on {} ({:.2%} of pixels masked)".format(
        len(rows), images_dir, summary["mean_masked_fraction"]), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--preparation", type=Path, default=None,
                        help="survey.py prepare manifest; the GPS source of camera centres")
    parser.add_argument("--poses", type=Path, default=None,
                        help="keyframes_poses.jsonl from a reconstruction, to mask on "
                             "reconstructed centres instead of GPS")
    parser.add_argument("--depth-range", dest="depth_range", nargs=2, type=float,
                        default=None, metavar=("NEAR", "FAR"),
                        help="scene depth in the pose file's units; without it no pixel is "
                             "vetoed as static parallax and a panning camera is not separable")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--keep-all", dest="keep_all", action="store_true",
                        help="write a matching copy for every keyframe, even one the "
                             "quality score would drop: what the frame budget decided "
                             "upstream stays decided")
    parser.add_argument("--images", default=None, metavar="DIR",
                        help="mask an existing image directory in place, writing no frames; "
                             "needs --poses and --mask, and refuses --select/--flatten")
    parser.add_argument("--focal-px", dest="focal_px", type=float, default=None)
    parser.add_argument("--energy-floor", dest="energy_floor", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the report (default: <work>/survey/frames.json)")
    args = parser.parse_args(argv)
    if args.images:
        if not (args.mask and args.poses) or args.select or args.flatten or args.preparation:
            parser.exit(2, "--images is a mask-only pass: it needs --poses and --mask, and "
                           "refuses --select/--flatten/--preparation because those all "
                           "rewrite frames\n")
        try:
            mask_existing(args.work, images_dir=args.images, poses=args.poses,
                          out=args.out or args.work / "survey" / "frames.json",
                          depth_range=args.depth_range)
        except (ValueError, KeyError, IndexError) as error:
            print("[frames] " + str(error), file=sys.stderr)
            return 3
        return 0
    if bool(args.preparation) and args.poses:
        parser.exit(2, "pass one of --preparation (GPS) or --poses (reconstructed), "
                       "not both\n")
    if not (args.preparation or args.poses) and (args.select or args.mask):
        parser.exit(2, "--select and --mask both need camera centres: pass --preparation "
                       "(GPS) or --poses (a reconstruction)\n")
    preparation = (json.loads(args.preparation.read_text(encoding="utf-8"))
                   if args.preparation else None)
    plan = capture.read_plan(args.plan) if args.plan else None
    centres = k_matrix = shared_focal = None
    if args.poses:
        rows = _read_rows(args.work / "keyframes.jsonl")
        centres, k_matrix, shared_focal = _poses_centres(args.poses, rows)
    try:
        report, kept, rows = build(args.work, preparation=preparation, select=args.select,
                                   flatten=args.flatten, mask=args.mask, plan=plan,
                                   focal_px=args.focal_px, energy_floor=args.energy_floor,
                                   camera_centres=centres, k_matrix=k_matrix,
                                   scene_depth_range=args.depth_range,
                                   keep_all=args.keep_all)
    except (ValueError, KeyError, IndexError) as error:
        print("[frames] " + str(error), file=sys.stderr)
        return 3
    if args.poses:
        report["intrinsics"] = ("reconstructed camera, shared across frames" if shared_focal
                                else "reconstructed cameras disagree on focal; the first frame's "
                                     "matrix was used for every epipolar test")
    out = args.out or args.work / "survey" / "frames.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    copy = report.get("colour_copy") or {}
    if copy and not copy.get("unchanged", True):
        # Write the receipt, then fail: this is the one outcome that would make a
        # lighting artefact permanent, because the trainer colourises the splat from
        # frames_train. A pass that moved the colour copy has to stop the run rather
        # than let a model be built on pixels it did not intend to change.
        out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("[frames] frames_train changed during this pass ({} -> {}); refusing to "
              "continue. Flattening must never reach the colourised copy - see {}".format(
                  copy.get("digest_before"), copy.get("digest_after"), out), file=sys.stderr)
        return 4
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
