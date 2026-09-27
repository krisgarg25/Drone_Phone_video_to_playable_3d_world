"""Challenge D1: mask moving objects, and only re-match if the masks register more frames.

Dynamic objects are the one challenge that cannot be decided before reconstruction. A
mask needs to know how the camera moved, because the difference between "a car crossed
the frame" and "the drone panned" is only visible in the geometry - so the camera
centres come from a model that has to be reconstructed first. That makes this a second
pass, and a second pass costs a real re-extraction, so it runs only under one of two
conditions: registration is already weak (the same bar ``rescue_below`` uses, so
"weak" has one meaning in this codebase), or an operator asked for it.

The pass defends the model it started with. A mask that hides the facade along with
the traffic registers *fewer* frames, and the only honest test of that is to run both
and compare. On a tie the unmasked model wins, because it was already good enough and
the masks cost coverage.

Writes ``<work>/dynamics.json`` either way, including the skipped case: "we did not
mask this" is a fact a reviewer is entitled to.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402
import survey_frames  # noqa: E402
from run_colmap import DEFAULT_PLAN, count_model_images  # noqa: E402

DEPTH_PERCENTILES = (5, 95)
"""Depth bounds for the static-parallax veto, taken off the reconstructed cloud itself.
Both the baseline and the depth arrive in the model's own units, so their ratio - the
only thing the veto compares - is unaffected by however wrong the scene scale is."""


def nearest_depths(points, centres):
    """Distance from every 3D point to the camera that saw it closest, in model units."""
    if not len(points) or not len(centres):
        return np.zeros(0)
    out = []
    for block in range(0, len(points), 512):
        chunk = points[block:block + 512]
        out.append(np.linalg.norm(chunk[:, None, :] - centres[None, :, :], axis=2).min(axis=1))
    return np.concatenate(out)


def _read_sparse_txt(path):
    """XYZ of the reconstructed cloud, from the exported text model. No COLMAP import:
    the file is already plain text and the columns are fixed by its own header."""
    rows = []
    if not path.is_file():
        return np.zeros((0, 3))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            rows.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float64)


def _centres_from_poses(path):
    rows = [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    centres = []
    for row in rows:
        cam = row["camera"]
        R = np.asarray(cam["R_rowmajor"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
        centres.append(-R.T @ t)
    return np.asarray(centres, dtype=np.float64)


def run(work: Path, *, below: float, force: bool) -> dict:
    work = Path(work)
    txt = work / "colmap" / "sparse" / "txt"
    plan = dict(DEFAULT_PLAN)
    if (work / "plan.json").is_file():
        plan.update(json.loads((work / "plan.json").read_text(encoding="utf-8")))
    image_dir = plan["image_dir"]
    n_frames = sum(1 for _ in (work / image_dir).rglob("*.jpg")) + \
        sum(1 for _ in (work / image_dir).rglob("*.png"))
    registered = count_model_images(txt) if txt.is_dir() else 0
    ratio = registered / max(n_frames, 1)
    report = {"schema_version": 1, "frames": n_frames, "registered_pass1": registered,
              "registration_ratio": round(ratio, 4), "rescue_below": below, "forced": force,
              "decision": None}

    if not txt.is_dir():
        report["decision"] = "unavailable"
        report["reason"] = (f"no reconstructed model at {txt.relative_to(work)}; this pass "
                            f"masks against an existing reconstruction, it cannot make one")
        return report

    if not force and ratio >= below:
        report["decision"] = "skipped"
        report["reason"] = (f"{registered}/{n_frames} frames registered, at or above the "
                            f"{below:.0%} bar, so the capture is not failing on moving "
                            f"objects and a masked re-match would only cost coverage")
        return report

    poses = work / "keyframes_poses.jsonl"
    if not poses.is_file():
        report["decision"] = "unavailable"
        report["reason"] = ("no keyframes_poses.jsonl: this pass needs a reconstruction to "
                            "take camera centres from")
        return report

    centres = _centres_from_poses(poses)
    depths = nearest_depths(_read_sparse_txt(txt / "points3D.txt"), centres)
    near, far = (float(np.percentile(depths, DEPTH_PERCENTILES[0])),
                 float(np.percentile(depths, DEPTH_PERCENTILES[1]))) if len(depths) else (None, None)
    report["scene_depth_range_units"] = [near, far]

    argv = [sys.executable, str(Path(survey_frames.__file__)), "--work", str(work),
            "--poses", str(poses), "--mask", "--images", image_dir,
            "--out", str(work / "dynamics_frames.json")]
    if near is not None and far > near:
        argv += ["--depth-range", f"{near:.4f}", f"{far:.4f}"]
    else:
        # Without a depth range no pixel is vetoed as static parallax, which means a
        # pan is indistinguishable from a moving wall. Say so rather than masking the
        # scene on a guess.
        report["depth_range_warning"] = ("no usable depth spread from the reconstructed "
                                         "cloud; masks will be appearance-only and weaker")
    rb.run_cmd(argv, cwd=str(Path(work).resolve().parent.parent))
    frames_report = json.loads((work / "dynamics_frames.json").read_text(encoding="utf-8"))
    masks = frames_report.get("masks", {})
    report["masked_fraction_mean"] = round(float(masks.get("mean_masked_fraction", 0.0)), 5)
    report["masked_fraction_max"] = round(float(masks.get("coverage_cost", {})
                                                .get("max_frame_fraction", 0.0)), 5)
    report["frames_masked"] = masks.get("frames_masked", 0)
    report["masks_written"] = masks.get("count", 0)
    if not masks.get("count"):
        report["decision"] = "no-masks"
        report["reason"] = "the mask pass produced nothing to apply"
        return report

    # The backup has to live outside work/<scene>/colmap: run_colmap clears
    # work/<scene>/colmap/sparse before it writes the new model, and a backup kept in
    # there is deleted by the very run it exists to recover from.
    stash = work / "dynamics_pass1"
    backup = stash / "txt"
    poses_backup = stash / "keyframes_poses.jsonl"
    if stash.exists():
        shutil.rmtree(stash)
    stash.mkdir(parents=True)
    shutil.copytree(txt, backup)
    shutil.copyfile(poses, poses_backup)

    def restore():
        # Refuse to touch the model on disk unless the backup is actually there. An
        # unhedged copytree here once deleted a good reconstruction instead of
        # restoring it, which is the one outcome this step must never produce.
        if not backup.is_dir():
            rb.warn(f"dynamics: no pass-1 backup at {backup}; leaving the current model alone")
            return
        shutil.rmtree(txt, ignore_errors=True)
        shutil.copytree(backup, txt)
        if poses_backup.is_file():
            shutil.copyfile(poses_backup, poses)

    # The re-match wipes the database and the sparse tree, so a crash partway
    # through would otherwise leave the run holding a half-built pass-2 model and
    # no memory of the complete pass-1 one. Keep is set only once the comparison has
    # actually chosen the new model, so every other exit - exception included - puts
    # the original back.
    # Only mask_path is overridden. ``image_dir`` already sits in plan.json as whatever
    # pass 1 matched on, so passing it again would be a second source of truth - and an
    # earlier version pinned it to frames_match here, which re-matched a photometrically
    # different image set than the one being diagnosed.
    keep = False
    rematch = [sys.executable, str(Path(__file__).parent / "run_colmap.py"), str(work),
               "--plan", str(work / "plan.json"), "--set", "mask_path=masks"]
    try:
        rc = subprocess.run(rematch, cwd=str(Path(work).resolve().parent.parent)).returncode
        after = count_model_images(txt) if txt.is_dir() else 0
        report["registered_pass2"] = after
        report["rematch_exit"] = rc
        if rc == 0 and after > registered:
            keep = True
    finally:
        if not keep:
            restore()
            report["decision"] = "reverted"
            report["reason"] = (f"masked re-match registered {report.get('registered_pass2', 0)}"
                                f"/{n_frames} against {registered}/{n_frames} unmasked, so the "
                                f"masks bought nothing; the unmasked model is kept")
            report["pass1_restored_from"] = str(backup.relative_to(work))
            if report.get("rematch_exit", 0) != 0:
                report["reason"] += " (the re-match did not exit cleanly)"
            return report

    rb.run_cmd([sys.executable, str(Path(__file__).parent / "parse_colmap.py"),
                "--work", str(work)], cwd=str(Path(work).resolve().parent.parent))
    report["decision"] = "applied"
    report["reason"] = (f"masked re-match registered {after}/{n_frames} against "
                        f"{registered}/{n_frames}, so the model now excludes the moving "
                        f"objects; poses were re-parsed from it")
    report["pass1_kept_at"] = str(backup.relative_to(work))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--below", type=float, default=None,
                    help="registration ratio under which masking is worth trying "
                         "(default: the run's own rescue_below)")
    ap.add_argument("--force", action="store_true",
                    help="mask and re-match even when registration is healthy")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the report (default: <work>/dynamics.json)")
    args = ap.parse_args(argv)
    below = args.below
    if below is None:
        plan_path = args.work / "plan.json"
        below = (json.loads(plan_path.read_text(encoding="utf-8")).get("rescue_below")
                 if plan_path.is_file() else None) or DEFAULT_PLAN["rescue_below"]
    out = args.out or args.work / "dynamics.json"
    try:
        report = run(args.work, below=below, force=args.force)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # A failed mask pass must not read like a failed reconstruction: the model on
        # disk is still the pass-1 one, and downstream steps are entitled to it.
        rb.warn(f"dynamics: {error}")
        rb.write_json(out, {"schema_version": 1, "decision": "error", "reason": str(error)})
        print(f"[dynamics] {error}", file=sys.stderr)
        return 0
    rb.write_json(out, report)
    print("[dynamics] {}: {} ({} masks, {:.2%} of pixels)".format(
        report["decision"], report.get("reason", "")[:120], report.get("masks_written", 0),
        report.get("masked_fraction_mean", 0.0)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
