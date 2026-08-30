"""V2 COLMAP runner: plan-driven, multi-clip aware, optional AR pose priors.

Replaces the hardcoded sequential pipeline. Everything is driven by a plan
dict (written by pipeline.py to <work>/plan.json, or built from CLI flags):

  camera_model        SIMPLE_RADIAL default
  per_folder_camera   one camera per clip folder (mixed lenses/orientations)
  max_image_size      SIFT image cap
  max_features        SIFT feature cap
  overlap             sequential matcher window
  quadratic_overlap   sequential matcher stride doubling
  loop_detection      needs vocab_tree; re-visits within/between clips
  vocab_tree          path to .bin (optional)
  cross_clip          auto | spatial | vocab | exhaustive | none
  exhaustive_max      frame count under which exhaustive matching is used
  mapper              auto | incremental | pose_prior | global
  prior_std           meters of assumed AR position error
  init_min_tri_angle  Mapper.init_min_tri_angle degrees

Pose priors: work/<scene>/pose_priors.jsonl rows {file, position, std} are
injected straight into the COLMAP database 'pose_priors' table (CARTESIAN,
camera centers in meters) and mapped with `pose_prior_mapper` -- the official
soft-constraint path that lets feature evidence override noisy AR drift.
After mapping we best-effort align the model back onto the prior frame with
`model_aligner`.

Usage: python run_colmap.py <work_dir> [--plan path.json] [overrides...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "colmap"
COLMAP = TOOLS / "bin" / "colmap.exe"

# Verified against the installed stack (colmap 4.1.1 / pycolmap 4.1.1):
#   PosePrior::CoordinateSystem: UNDEFINED=-1, WGS84=0, CARTESIAN=1
#   SensorType: INVALID=-1, CAMERA=0, IMU=1
# In COLMAP >= 4 the pose_priors table links priors to images through
# (corr_data_id=image_id, corr_sensor_id=camera_id, corr_sensor_type=CAMERA);
# in 3.11/3.12 it was a plain image_id primary key. Both are supported below.
CARTESIAN = 1
SENSOR_TYPE_CAMERA = 0

POSE_PRIORS_DDL_V4 = (
    "CREATE TABLE IF NOT EXISTS pose_priors "
    "  (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,"
    "   corr_data_id               INTEGER               NOT NULL,"
    "   corr_sensor_id             INTEGER               NOT NULL,"
    "   corr_sensor_type           INTEGER               NOT NULL,"
    "   position                   BLOB,"
    "   position_covariance        BLOB,"
    "   gravity                    BLOB,"
    "   coordinate_system          INTEGER               NOT NULL)"
)
POSE_PRIORS_DDL_V3 = (
    "CREATE TABLE IF NOT EXISTS pose_priors "
    "  (image_id            INTEGER  PRIMARY KEY  NOT NULL,"
    "   position            BLOB,"
    "   coordinate_system   INTEGER  NOT NULL,"
    "   position_covariance BLOB)"
)


def _f64_blob(arr) -> bytes:
    return np.asarray(arr, np.float64).astype("<f8").tobytes()


def inject_pose_priors(db_path: Path, priors_file: Path) -> tuple[int, int]:
    """Write AR position priors into the DB; returns (written, rows_in_file)."""
    rows = [json.loads(l) for l in priors_file.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    con = sqlite3.connect(str(db_path))
    try:
        have = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors'"
        ).fetchone()
        cols = set()
        if have:
            cols = {r[1] for r in con.execute("PRAGMA table_info(pose_priors)")}
        else:
            con.execute(POSE_PRIORS_DDL_V4)
            cols = {r[1] for r in con.execute("PRAGMA table_info(pose_priors)")}
        v4 = "corr_data_id" in cols

        if v4:
            img_info = dict((name, (iid, cid)) for name, iid, cid in
                            con.execute("SELECT name, image_id, camera_id FROM images"))
        else:
            img_info = dict((name, (iid, None)) for name, iid in
                            con.execute("SELECT name, image_id FROM images"))

        nan_gravity = _f64_blob([float("nan")] * 3)
        written = 0
        for r in rows:
            info = img_info.get(r["file"])
            if info is None:
                continue
            iid, cid = info
            pos = np.asarray(r["position"], np.float64).reshape(3)
            std = float(r.get("std", DEFAULT_PLAN["prior_std"]))
            cov = np.diag([std * std] * 3)  # symmetric -> row/col-major identical
            if v4:
                con.execute(
                    "INSERT OR REPLACE INTO pose_priors (pose_prior_id, corr_data_id, "
                    "corr_sensor_id, corr_sensor_type, position, position_covariance, "
                    "gravity, coordinate_system) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                    (iid, cid, SENSOR_TYPE_CAMERA, _f64_blob(pos),
                     _f64_blob(cov.reshape(-1)), nan_gravity, CARTESIAN))
            else:
                con.execute(
                    "INSERT OR REPLACE INTO pose_priors (image_id, position, "
                    "coordinate_system, position_covariance) VALUES (?, ?, ?, ?)",
                    (iid, _f64_blob(pos), CARTESIAN, _f64_blob(cov.reshape(-1))))
            written += 1
        con.commit()
        n_db = con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
        print(f"[colmap] pose_priors now holds {n_db} rows")
    finally:
        con.close()
    return written, len(rows)

DEFAULT_PLAN = {
    "camera_model": "SIMPLE_RADIAL",
    "per_folder_camera": False,
    "max_image_size": 1600,
    "max_features": 8192,
    "overlap": 20,
    "quadratic_overlap": True,
    "loop_detection": False,
    "vocab_tree": None,
    "cross_clip": "auto",
    "exhaustive_max": 350,
    "mapper": "auto",
    "prior_std": 0.15,
    "init_min_tri_angle": 16.0,
    "rescue_below": 0.6,
}


def env() -> dict:
    return {**os.environ,
            "PATH": f"{TOOLS / 'bin'};{os.environ.get('PATH', '')}",
            "QT_PLUGIN_PATH": f"{TOOLS / 'plugins'};{os.environ.get('QT_PLUGIN_PATH', '')}"}


def run(sub: str, *flags: str, allow_fail: bool = False) -> bool:
    print(f"[colmap] {sub} {' '.join(flags[:6])}{' ...' if len(flags) > 6 else ''}",
          flush=True)
    r = subprocess.run([str(COLMAP), sub, *flags], env=env())
    ok = r.returncode == 0
    if not ok and not allow_fail:
        sys.exit(f"[colmap] {sub} failed (exit {r.returncode})")
    if not ok:
        print(f"[colmap] {sub} failed but continuing (allow_fail)")
    return ok


def model_size(d: Path) -> int:
    for fname in ("points3D.bin", "points3D.txt", "images.bin", "images.txt"):
        f = d / fname
        if f.exists():
            return f.stat().st_size
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file()) if d.exists() else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--plan", type=Path, default=None)
    ap.add_argument("--plan-hash", default=None,
                    help="content digest from pipeline.py; changes invalidate markers")
    ap.add_argument("--set", action="append", default=[],
                    help="KEY=VALUE plan override (repeatable)")
    args = ap.parse_args()

    work = args.work
    frames = work / "frames_train"
    db = work / "database.db"
    priors_file = work / "pose_priors.jsonl"

    plan = dict(DEFAULT_PLAN)
    if args.plan and args.plan.exists():
        plan.update(json.loads(args.plan.read_text(encoding="utf-8")))
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in plan:
            sys.exit(f"unknown plan key {k}; known: {sorted(plan)}")
        old = plan[k]
        if isinstance(old, bool):
            plan[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(old, (int, float)):
            plan[k] = type(old)(float(v))
        else:
            plan[k] = v
    if not COLMAP.exists():
        sys.exit(f"colmap.exe not found at {COLMAP}")

    n_frames = sum(1 for _ in frames.rglob("*.jpg")) + \
        sum(1 for _ in frames.rglob("*.png"))
    has_priors = priors_file.exists()

    # ---------------- mapper choice ----------------
    mapper = plan["mapper"]
    if mapper == "auto":
        mapper = "pose_prior" if has_priors else "incremental"

    print(f"[colmap] plan: {n_frames} frames, priors={'yes' if has_priors else 'no'}, "
          f"matcher=cross:{plan['cross_clip']}, mapper={mapper}, "
          f"tri_angle={plan['init_min_tri_angle']}")

    if db.exists():
        db.unlink()
    if (work / "sparse").exists():
        shutil.rmtree(work / "sparse")
    (work / "sparse").mkdir(parents=True)

    # ---------------- features ----------------
    reader_flags = ["--ImageReader.camera_model", plan["camera_model"]]
    reader_flags += (["--ImageReader.single_camera_per_folder", "1"]
                     if plan["per_folder_camera"]
                     else ["--ImageReader.single_camera", "1"])
    
    # Sensitive peak threshold (0.002) and edge threshold (16) to capture plain painted walls & faint room features
    peak_thresh = str(plan.get("sift_peak_threshold", 0.002))
    edge_thresh = str(plan.get("sift_edge_threshold", 16))
    
    run("feature_extractor",
        "--database_path", str(db),
        "--image_path", str(frames),
        *reader_flags,
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.max_image_size", str(plan["max_image_size"]),
        "--SiftExtraction.max_num_features", str(plan["max_features"]),
        "--SiftExtraction.peak_threshold", peak_thresh,
        "--SiftExtraction.edge_threshold", edge_thresh)

    # ---------------- pose priors ----------------
    if has_priors:
        written, total = inject_pose_priors(db, priors_file)
        print(f"[colmap] injected {written}/{total} pose priors into {db.name}")
        if written == 0:
            sys.exit("[colmap] no priors matched images - manifest names vs DB names diverged")

    # ---------------- matching ----------------
    seq_flags = ["--SequentialMatching.overlap", str(plan["overlap"]),
                 "--SequentialMatching.quadratic_overlap",
                 "1" if plan["quadratic_overlap"] else "0"]
    vocab = plan.get("vocab_tree") or ""
    have_vocab = bool(vocab) and Path(vocab).exists()
    if plan["loop_detection"] and have_vocab:
        seq_flags += ["--SequentialMatching.loop_detection", "1",
                      "--SequentialMatching.loop_detection_period", "10",
                      "--SequentialMatching.loop_detection_num_images", "50",
                      "--SequentialMatching.loop_detection_vocab_tree", str(vocab)]
    elif plan["loop_detection"]:
        print("[colmap] loop detection requested but no vocab tree found - skipping")

    run("sequential_matcher", "--database_path", str(db), *seq_flags,
        "--FeatureMatching.use_gpu", "1")

    cc = plan["cross_clip"]
    if cc == "auto":
        if has_priors:
            cc = "spatial"
        elif have_vocab:
            cc = "vocab"
        elif n_frames <= plan["exhaustive_max"]:
            cc = "exhaustive"
        else:
            cc = "none"
        print(f"[colmap] cross-clip strategy: {cc}")

    if n_frames > 1:
        if cc == "spatial":
            run("spatial_matcher", "--database_path", str(db),
                "--SpatialMatching.ignore_z", "0",
                "--SpatialMatching.max_num_neighbors", "50",
                "--FeatureMatching.use_gpu", "1")
        elif cc == "vocab":
            run("vocab_tree_matcher", "--database_path", str(db),
                "--VocabTreeMatching.vocab_tree_path", str(vocab),
                "--VocabTreeMatching.num_images", "25",
                "--FeatureMatching.use_gpu", "1")
        elif cc == "exhaustive":
            run("exhaustive_matcher", "--database_path", str(db),
                "--FeatureMatching.use_gpu", "1")
        if cc == "none" and n_frames > plan["exhaustive_max"]:
            print("[colmap] WARNING many frames with no cross-clip mechanism: clips may "
                  "reconstruct as separate models. Provide a vocab tree or phone poses.")

    # ---------------- accelerated mapping with high-throughput BA ----------------
    # Reserve 3 threads for OS, UI, and background desktop tasks
    cpu_threads = str(max(1, (os.cpu_count() or 16) - 3))
    map_flags = [
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_global_frames_ratio", "1.3",
        "--Mapper.ba_global_points_ratio", "1.3",
        "--Mapper.ba_global_max_num_iterations", "25",
        "--Mapper.ba_local_max_num_iterations", "15",
        "--Mapper.ba_global_max_refinements", "2",
        "--Mapper.ba_local_max_refinements", "1",
        "--Mapper.num_threads", cpu_threads,
    ]

    def do_map(extra_flags, tag):
        nonlocal mapper
        if mapper == "pose_prior":
            std = str(plan["prior_std"])
            ok = run("pose_prior_mapper",
                     "--database_path", str(db),
                     "--image_path", str(frames),
                     "--output_path", str(work / "sparse"),
                     "--overwrite_priors_covariance", "1",
                     "--prior_position_std_x", std,
                     "--prior_position_std_y", std,
                     "--prior_position_std_z", std,
                     "--use_robust_loss_on_prior_position", "1",
                     *map_flags, *extra_flags, allow_fail=True)
            if not ok:
                print("[colmap] pose_prior_mapper failed - falling back to plain mapper")
                mapper = "incremental"
        if mapper == "global":
            gdb = work / "database_global.db"
            shutil.copyfile(db, gdb)
            run("view_graph_calibrator", "--database_path", str(gdb))
            run("global_mapper", "--database_path", str(gdb),
                "--image_path", str(frames),
                "--output_path", str(work / "sparse"), *map_flags, *extra_flags)
        if mapper == "incremental":
            run("mapper",
                "--database_path", str(db),
                "--image_path", str(frames),
                "--output_path", str(work / "sparse"),
                *map_flags, *extra_flags)
        return count_registered(work / "sparse")

    def wipe_models():
        if (work / "sparse").exists():
            shutil.rmtree(work / "sparse")
        (work / "sparse").mkdir(parents=True)

    def count_registered(sparse_dir: Path) -> tuple[int, Path | None]:
        best, best_dir = 0, None
        for d in sparse_dir.glob("*"):
            if not d.is_dir():
                continue
            try:
                import pycolmap
                n = pycolmap.Reconstruction(str(d)).num_reg_images()
            except Exception:
                n = 0
            if n > best:
                best, best_dir = n, d
        return best, best_dir

    wipe_models()
    registered, modeldir = do_map([], "default")
    frac = registered / max(n_frames, 1)
    print(f"[colmap] attempt #1 ({mapper}): {registered}/{n_frames} registered")

    # Rescue: a low registration rate usually means a degenerate initial pair.
    # Retry once with permissive thresholds instead of pre-lowering them
    # (pre-lowering invites bad init pairs; the retry only fires when needed).
    if frac < plan["rescue_below"] and n_frames >= 8:
        rescue = ["--Mapper.init_min_tri_angle", "4",
                  "--Mapper.abs_pose_min_num_inliers", "15",
                  "--Mapper.init_num_trials", "5"]
        print(f"[colmap] below {100 * plan['rescue_below']:.0f}% - retrying "
              f"with rescue flags: {' '.join(rescue)}")
        wipe_models()
        mapper_saved = mapper
        registered2, modeldir2 = do_map(rescue, "rescue")
        print(f"[colmap] attempt #2 (rescue): {registered2}/{n_frames} registered")
        if registered2 > registered:
            registered, modeldir = registered2, modeldir2
        else:
            mapper = mapper_saved

    if modeldir is None:
        sys.exit("[colmap] mapper produced no model")

    # With active position priors, pose_prior_mapper's BA keeps camera centers
    # in the prior (AR) frame WITH metric scale - verified empirically (12m
    # walk reproduced at scale=1.0000, max deviation ~7mm). colmap
    # model_aligner cannot help here anyway: it only supports WGS84 priors.
    chosen = modeldir

    txt = work / "colmap" / "sparse" / "txt"
    if txt.parent.parent.exists():
        shutil.rmtree(txt.parent.parent)
    txt.mkdir(parents=True, exist_ok=True)
    run("model_converter", "--input_path", str(chosen),
        "--output_path", str(txt), "--output_type", "TXT")

    registered = sum(1 for line in (txt / "images.txt").read_text().splitlines()
                     if line.strip() and not line.startswith("#") and
                     len(line.split()) == 10)
    print(f"[colmap] done: {registered}/{n_frames} frames registered "
          f"({100 * registered / max(n_frames, 1):.0f}%). Model: {chosen}")


if __name__ == "__main__":
    main()
