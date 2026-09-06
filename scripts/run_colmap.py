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
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robust as rb  # noqa: E402

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
    "calibration_json": None,   # path to WebXR capture calibration.json
}


def image_size(path: Path):
    """Pixel size of a JPEG or PNG, read from its header. No image library: only
    the geometry is wanted here, and a full decode costs more than the answer."""
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            if head == b"\x89P":                                  # PNG
                f.seek(16)
                w, h = struct.unpack(">II", f.read(8))
                return int(w), int(h)
            if head != b"\xff\xd8":                               # not a readable JPEG
                return None
            while True:
                marker = f.read(2)
                if len(marker) < 2 or not marker.startswith(b"\xff"):
                    return None
                code = marker[1]
                if code == 0x01 or 0xD0 <= code <= 0xD9:           # standalone markers
                    continue
                seg = f.read(2)
                if len(seg) < 2:
                    return None
                (ln,) = struct.unpack(">H", seg)
                # SOFn is the only segment carrying the frame dimensions.
                if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                    body = f.read(5)                                # precision + h + w
                    if len(body) < 5:
                        return None
                    # SOFn lays the frame out as precision, height, width.
                    return int.from_bytes(body[3:5], "big"), int.from_bytes(body[1:3], "big")
                f.seek(ln - 2, 1)
    except (OSError, struct.error):
        return None


def frame_geometries(frames: Path) -> dict:
    """{folder: (w, h)} — the size of each clip folder's first frame.

    COLMAP gives one camera to every image it is told shares a camera, so this is
    the question the plan has to answer before feature extraction: can all of
    these frames describe the same lens?
    """
    dirs = sorted(p for p in frames.iterdir() if p.is_dir()) or [frames]
    geo = {}
    for d in dirs:
        first = next((f for f in sorted(d.iterdir())
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
        if first is None:
            continue
        size = image_size(first)
        if size:
            geo[d.name if d != frames else "."] = size
    return geo


def env() -> dict:
    return {**os.environ,
            "PATH": f"{TOOLS / 'bin'};{os.environ.get('PATH', '')}",
            "QT_PLUGIN_PATH": f"{TOOLS / 'plugins'};{os.environ.get('QT_PLUGIN_PATH', '')}"}


_OPTIONS: dict | None = None
_WORK: Path | None = None


def probe_options(work: Path | None = None) -> dict:
    """subcommand -> its advertised --Section.option names, probed once.

    Cached per scene because `colmap <sub> -h` for eleven subcommands is not
    free, and because the map is a property of the vendored binary, not of the
    run. Per-subcommand because the union of the sets is actively wrong: an
    option `mapper` accepts makes `global_mapper` abort its argument parsing.
    """
    global _OPTIONS, _WORK
    if work is not None:
        _WORK = Path(work)
    if _OPTIONS is None:
        # Only cache when a scene directory is known: otherwise the probe would
        # drop a file into whatever the current working directory happened to be.
        cache = (_WORK / ".colmap_options.json") if _WORK else None
        _OPTIONS = rb.colmap_option_map(COLMAP, env=env(), cache=cache)
        total = len(set().union(*_OPTIONS.values())) if _OPTIONS else 0
        if total:
            print(f"[colmap] probed {total} options across "
                  f"{sum(1 for v in _OPTIONS.values() if v)} subcommands", flush=True)
        else:
            rb.warn("could not probe COLMAP options; passing the plan flags as-is")
    return _OPTIONS


def known_options(sub: str | None = None) -> set:
    """The flags to validate an argv against: this subcommand's, or the union."""
    per_sub = probe_options()
    union = set().union(*per_sub.values()) if per_sub else set()
    if not union:
        return set()          # probe failed: drop nothing
    if sub is None:
        return union
    # A subcommand the probe could not read gets the union, not an empty set -
    # "I did not see this flag" is not evidence that the flag is invalid.
    return per_sub.get(sub) or union


def run(sub: str, *flags: str, allow_fail: bool = False,
        timeout: float = 7200) -> bool:
    """One COLMAP subcommand. Raises a classified StepError unless allow_fail.

    Unsupported flags are dropped rather than fatal: a newer build's optional
    extra used to abort a whole solve on the parsing line.
    """
    known = known_options(sub)
    argv, dropped = rb.split_flags([str(COLMAP), sub, *flags], known)
    if dropped:
        names = ", ".join(d for d in dropped if d.startswith("--"))
        rb.warn(f"{sub}: this COLMAP build has no {names} - continuing without it")
    print(f"[colmap] {sub} {' '.join(flags[:6])}{' ...' if len(flags) > 6 else ''}",
          flush=True)
    try:
        rb.run_cmd(argv, env=env(), timeout=timeout, retries=1,
                   retry_on=(rb.CRASH, rb.TIMEOUT))
        return True
    except rb.StepError as e:
        if allow_fail:
            print(f"[colmap] {sub} failed ({e.kind}) but continuing: "
                  f"{rb.status_name(e.returncode)}", flush=True)
            return False
        # Name the class so a caller can pick a different strategy instead of
        # watching the process exit.
        raise rb.StepError(e.kind, f"{sub} failed: {e.message}",
                           returncode=e.returncode, output=e.output)


def count_model_images(d: Path) -> int:
    """Registered images in a COLMAP model dir, with no optional dependency.

    pycolmap is optional - pipeline.py's doctor advertises a sqlite fallback -
    so the previous `except Exception: n = 0` meant that on any machine without
    the wheel every successful reconstruction reported zero images and the run
    died saying the mapper produced no model.

    images.bin is deliberately not parsed by hand: the record layout gained
    fields with COLMAP 4's rigs/frames refactor, and a parser that quietly
    under-counts would pick the wrong model, which is worse than the bug this
    replaces. Asking the same colmap binary that wrote the model to convert it
    cannot drift with the format.
    """
    txt = d / "images.txt"
    if not txt.exists():
        via = count_via_converter(d)
        return _count_with_pycolmap(d) if via is None else via
    # Layout: one comment header, then two lines per image (pose, then its 2D
    # points). Counting pose lines by content is ambiguous - the points line
    # also starts with an integer - so use the pairing.
    rows = [l for l in rb.read_text(txt).splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    return len(rows) // 2


def count_via_converter(d: Path) -> int | None:
    """Registered images via model_converter, or None if the conversion failed.

    The temp dir is outside work/<scene>/sparse on purpose: count_registered
    globs that tree for candidate models, and a scratch directory holding an
    images.txt would be scored as if it were a reconstruction.
    """
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="colmap_count_"))
    try:
        ok = run("model_converter", "--input_path", str(d),
                 "--output_path", str(out), "--output_type", "TXT", allow_fail=True)
        f = out / "images.txt"
        if not (ok and f.exists()):
            return None
        rows = [l for l in rb.read_text(f).splitlines()
                if l.strip() and not l.lstrip().startswith("#")]
        return len(rows) // 2
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _count_with_pycolmap(d: Path) -> int:
    try:
        import pycolmap
        return int(pycolmap.Reconstruction(str(d)).num_reg_images())
    except Exception:
        return 0


def count_registered(sparse_dir: Path) -> tuple:
    """(best registered-image count, its directory) under a sparse tree."""
    best, best_dir = 0, None
    for d in sorted(p for p in sparse_dir.glob("*") if p.is_dir()):
        n = count_model_images(d)
        if n > best:
            best, best_dir = n, d
    return best, best_dir


def _drop_priors(db_path: Path) -> int:
    """Delete pose_priors rows from a database copy.

    pose_prior_mapper treats the AR track as a soft constraint, so a mis-scaled
    or drifting prior can hold the bootstrap somewhere the imagery disagrees.
    One rung solves without it; the original database is left intact.
    """
    con = sqlite3.connect(str(db_path))
    try:
        try:
            n = con.execute("DELETE FROM pose_priors").rowcount
        except sqlite3.Error:
            n = 0
        con.commit()
    finally:
        con.close()
    return n


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
    probe_options(work)
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

    # ---------------- can one camera describe all of these frames? -----------
    # Asked of the frames, not of the plan, because the cost of getting it wrong
    # is invisible: COLMAP rejects every image whose dimensions disagree with the
    # shared camera with a CAMERA_SINGLE_DIM_ERROR *warning* and exits 0, so those
    # frames never reach the database and the rest of the run reconstructs the
    # footage it happens to have left. Measured here on a two-clip scene: 120
    # keyframes on disk, 60 images in the database, and the mapper printing its
    # registration rate against 120 — indistinguishable from a bad tracking take.
    geo = frame_geometries(frames)
    sizes = {size for size in geo.values()}
    if len(sizes) > 1 and not plan["per_folder_camera"]:
        plan["per_folder_camera"] = True
        groups = ", ".join(f"{name} {w}x{h}" for name, (w, h) in sorted(geo.items()))
        print(f"[colmap] {len(sizes)} frame geometries ({groups}) "
              f"-> one camera per clip, or the frames that disagree are dropped")

    # ---------------- mapper choice ----------------
    mapper = plan["mapper"]
    if mapper == "auto":
        mapper = "pose_prior" if has_priors else "incremental"

    print(f"[colmap] plan: {n_frames} frames, priors={'yes' if has_priors else 'no'}, "
          f"camera={'per-clip' if plan['per_folder_camera'] else 'shared'}, "
          f"matcher=cross:{plan['cross_clip']}, mapper={mapper}, "
          f"tri_angle={plan['init_min_tri_angle']}")

    if db.exists():
        db.unlink()
    if (work / "sparse").exists():
        shutil.rmtree(work / "sparse")
    (work / "sparse").mkdir(parents=True)

    # ---------------- calibration: seed COLMAP with the real focal length ----
    # COLMAP guesses focal ≈ 1.2 × max(width, height).  For a phone that is
    # typically 1536 px for a 1280 px wide frame, while the real lens is ~1060 px
    # (WebXR FOV ≈ 60°).  A 45 % focal error breaks geometric verification
    # (wrong essential matrix), corrupts the initial pair triangulation, and
    # every subsequent PNP fails — which looks like "no baseline" footage but is
    # really a wrong camera model.  When calibration.json is present we pass the
    # true focal / principal-point to the feature extractor so BA starts from a
    # good point.
    camera_params_str: str | None = None
    cal_json_path = plan.get("calibration_json")
    if cal_json_path:
        cal_p = Path(cal_json_path)
        if cal_p.exists():
            try:
                cal = json.loads(cal_p.read_text(encoding="utf-8"))
                cal_cams = cal.get("cameras", [])
                if cal_cams:
                    cc0 = cal_cams[0]
                    ref_w = int(cc0["imageWidth"])
                    ref_h = int(cc0["imageHeight"])
                    # Scale focal if extracted frames differ from calibration size
                    first_img = next(frames.rglob("*.jpg"), None) or next(frames.rglob("*.png"), None)
                    actual = image_size(first_img) if first_img else None
                    scale = (actual[0] / ref_w) if actual else 1.0
                    f_px = (cc0["focalLengthX"] + cc0["focalLengthY"]) / 2 * scale
                    cx = cc0["principalPointX"] * scale
                    cy = cc0["principalPointY"] * scale
                    # SIMPLE_RADIAL params: f, cx, cy, k1
                    camera_params_str = f"{f_px:.1f},{cx:.1f},{cy:.1f},0"
                    default_guess = 1.2 * (actual[0] if actual else ref_w)
                    print(f"[colmap] calibration.json: f={f_px:.0f}px "
                          f"(COLMAP default would be ~{default_guess:.0f}px, "
                          f"{abs(f_px - default_guess) / f_px * 100:.0f}% error), "
                          f"cx={cx:.0f} cy={cy:.0f}")
            except (KeyError, OSError, json.JSONDecodeError, IndexError) as e:
                print(f"[colmap] calibration.json unreadable ({e}) - using COLMAP default")

    # ---------------- features ----------------
    reader_flags = ["--ImageReader.camera_model", plan["camera_model"]]
    reader_flags += (["--ImageReader.single_camera_per_folder", "1"]
                     if plan["per_folder_camera"]
                     else ["--ImageReader.single_camera", "1"])
    if camera_params_str:
        reader_flags += ["--ImageReader.camera_params", camera_params_str]
    
    # Sensitive peak threshold (0.002) and edge threshold (16) to capture plain painted walls & faint room features
    peak_thresh = str(plan.get("sift_peak_threshold", 0.002))
    edge_thresh = str(plan.get("sift_edge_threshold", 16))
    
    # Compute optimal CPU threads (12 threads for i5-13450HX P-cores)
    cpu_threads = str(min(12, max(1, (os.cpu_count() or 16) - 2)))

    run("feature_extractor",
        "--database_path", str(db),
        "--image_path", str(frames),
        *reader_flags,
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
        "--FeatureExtraction.num_threads", cpu_threads,
        "--FeatureExtraction.max_image_size", str(plan["max_image_size"]),
        "--SiftExtraction.max_num_features", str(plan["max_features"]),
        "--SiftExtraction.peak_threshold", peak_thresh,
        "--SiftExtraction.edge_threshold", edge_thresh)

    # Every frame on disk has to be in the database before anything else gets an
    # opinion about this scene. A rejected image is a warning line and a zero exit
    # code, so this count is the only thing that can see the loss.
    exts = (".jpg", ".jpeg", ".png")
    on_disk = {}
    for p in frames.rglob("*"):
        if p.suffix.lower() in exts:
            on_disk[p.parent.name] = on_disk.get(p.parent.name, 0) + 1
    in_db = {}
    con = sqlite3.connect(str(db))
    try:
        for (name,) in con.execute("SELECT name FROM images"):
            folder = Path(name).parent.name
            in_db[folder] = in_db.get(folder, 0) + 1
    finally:
        con.close()
    n_db = sum(in_db.values())
    if n_db != n_frames:
        lost = ", ".join(f"{f}: kept {in_db.get(f, 0)} of {n}"
                         for f, n in sorted(on_disk.items()) if in_db.get(f, 0) != n)
        sys.exit(f"[colmap] feature_extractor took {n_db} of {n_frames} frames. COLMAP "
                 f"refuses an image whose dimensions disagree with the camera it was "
                 f"told to share, and still exits 0 - so this run would have "
                 f"reconstructed the footage it kept and looked like bad tracking. "
                 f"Dropped: {lost}. One camera per clip fixes mixed portrait and "
                 f"landscape scenes; rerun with --set per_folder_camera=true.")
    print(f"[colmap] database holds all {n_db} frames", flush=True)

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
    if have_vocab and _WORK is not None:
        # A tree this build refuses has to be proved once, not once per scene:
        # the first rejection is a 25 s COLMAP fast-fail in the log, and every
        # later one is the same crash re-enacted for nothing.
        rejected = sorted(_WORK.parent.glob("*/vocab_tree_skipped.json"),
                          key=lambda p: p.stat().st_mtime)
        if rejected:
            prior = rb.read_json(rejected[-1], {}) or {}
            if Path(str(prior.get("tree", ""))) == Path(vocab):
                have_vocab = False
                rb.warn(f"skipping {Path(vocab).name}: "
                        f"{rejected[-1].parent.name} already showed this COLMAP "
                        "build cannot read it (legacy flann format). Re-download a "
                        "current vocab tree - see pipeline.py doctor.")
    if plan["loop_detection"]:
        # The sequential matcher builds its own retrieval vocabulary from the
        # database; a vocab-tree FILE is only an argument vocab_tree_matcher
        # takes (below). Verified against COLMAP 4.1.1 -h: the
        # SequentialMatching.loop_detection_* family ends at max_num_features.
        seq_flags += ["--SequentialMatching.loop_detection", "1",
                       "--SequentialMatching.loop_detection_period", "10",
                       "--SequentialMatching.loop_detection_num_images", "50",
                       "--SequentialMatching.loop_detection_max_num_features", "4096"]
        if not have_vocab:
            print("[colmap] loop detection will build its vocabulary from the database "
                  "(slower); tools/vocab_tree.bin would also enable vocab cross-clip")

    run("sequential_matcher", "--database_path", str(db), *seq_flags,
        "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
        "--FeatureMatching.num_threads", cpu_threads)

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
                "--SpatialMatching.max_num_neighbors", "30",
                "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                "--FeatureMatching.num_threads", cpu_threads)
        elif cc == "vocab":
            try:
                run("vocab_tree_matcher", "--database_path", str(db),
                    "--VocabTreeMatching.vocab_tree_path", str(vocab),
                    "--VocabTreeMatching.num_images", "25",
                    "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                    "--FeatureMatching.num_threads", cpu_threads)
            except rb.StepError as e:
                if e.kind != rb.UNSUPPORTED_ASSET:
                    raise
                # Cross-clip matching via a tree is an enhancement. Losing it
                # costs some loop closures; losing the solve costs the scene.
                have_vocab = False
                why = next((ln for ln in (e.output or "").splitlines()
                            if "faiss" in ln.lower() or "flann" in ln.lower()),
                           e.message)
                rb.warn(
                    f"{vocab}: this COLMAP build cannot read it - the tree is the "
                    "pre-May-2025 flann format and COLMAP 4.x needs faiss. "
                    "Re-download vocab_tree_flickr100K_words*.bin from a current "
                    "release (see pipeline.py doctor). Continuing without vocab "
                    "cross-clip matching; sequential + quadratic overlap still "
                    "links the clips.")
                rb.write_json(_WORK / "vocab_tree_skipped.json",
                              {"reason": why[:500], "tree": str(vocab)})
                cc = "exhaustive" if n_frames <= plan["exhaustive_max"] else "none"
                print(f"[colmap] cross-clip strategy: {cc}", flush=True)
        if cc == "exhaustive":
            run("exhaustive_matcher", "--database_path", str(db),
                "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                "--FeatureMatching.num_threads", cpu_threads)
        if cc == "none" and n_frames > plan["exhaustive_max"]:
            print("[colmap] WARNING many frames with no cross-clip mechanism: clips may "
                  "reconstruct as separate models. Provide a vocab tree or phone poses.")

    # ---------------- accelerated mapping with high-throughput BA ----------------
    # Pin Ceres Solver Bundle Adjustment to 12 threads (P-core hyperthreads) to avoid E-core latency
    cpu_threads = str(min(12, max(1, (os.cpu_count() or 16) - 2)))
    map_flags = [
        "--Mapper.multiple_models", "0",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_global_frames_ratio", "1.3",
        "--Mapper.ba_global_points_ratio", "1.3",
        "--Mapper.ba_global_max_num_iterations", "25",
        "--Mapper.ba_local_max_num_iterations", "10",
        "--Mapper.ba_global_max_refinements", "2",
        "--Mapper.ba_local_max_refinements", "1",
        "--Mapper.num_threads", cpu_threads,
    ]

    def do_map(extra_flags, tag, out_dir: Path):
        """Run the configured mapper into its own directory.

        Per-attempt output dirs are the point: the old code wiped work/sparse
        before the rescue run, so a rescue that registered fewer frames than
        attempt #1 destroyed the better model and there was nothing to fall back
        to but the settings.
        """
        nonlocal mapper
        out_dir.mkdir(parents=True, exist_ok=True)
        if mapper == "pose_prior":
            std = str(plan["prior_std"])
            ok = run("pose_prior_mapper",
                     "--database_path", str(db),
                     "--image_path", str(frames),
                     "--output_path", str(out_dir),
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
            run("view_graph_calibrator", "--database_path", str(gdb), allow_fail=True)
            run("global_mapper", "--database_path", str(gdb),
                "--image_path", str(frames),
                "--output_path", str(out_dir), *map_flags, *extra_flags)
        if mapper == "incremental":
            run("mapper",
                "--database_path", str(db),
                "--image_path", str(frames),
                "--output_path", str(out_dir),
                *map_flags, *extra_flags)
        return count_registered(out_dir)

    # Under-registration has several distinct causes and each needs a different
    # repair, so one rescue rung is not a ladder. Every rung is cheap to skip and
    # the loop stops the moment a rung clears the bar.
    floor = max(8, -(-n_frames // 10))
    target = max(floor, int(n_frames * plan["rescue_below"]))
    # Degenerate initial pair: insist on real parallax and more inliers.
    # Dicts, not lists: two --Mapper.init_min_tri_angle in one argv is not a
    # looser setting, it is COLMAP refusing to parse the command at all.
    relaxed_opts = {"Mapper.init_min_tri_angle": "4",
                    "Mapper.abs_pose_min_num_inliers": "15",
                    "Mapper.init_num_trials": "5"}
    # Texture-poor views: let small angles and few inliers register, so views
    # that only weakly constrain each other still join the model.
    permissive_opts = {**relaxed_opts,
                       "Mapper.init_min_tri_angle": "1.5",
                       "Mapper.abs_pose_min_num_inliers": "8",
                       "Mapper.multiple_models": "1",
                       "Mapper.max_error": "4.0"}

    def opts(d: dict) -> list:
        return [tok for kv in d.items() for tok in (f"--{kv[0]}", kv[1])]

    relaxed, permissive = opts(relaxed_opts), opts(permissive_opts)
    ladder = [("default", mapper, [])]
    if priors_file.exists() and mapper != "pose_prior":
        # The mapper built for exactly this footage: a handset that turned too
        # fast for feature matching still left a metric camera track, and
        # pose_prior_mapper solves against the track instead of the parallax.
        ladder.append(("pose-priors", "pose_prior", []))
    ladder += [
        ("relaxed-init", "incremental", relaxed),
    ]
    if priors_file.exists():
        # A wrong or mis-scaled AR prior anchors the bootstrap somewhere the
        # imagery disagrees with; solving without it sometimes registers the rest.
        ladder.append(("no-priors", "incremental", permissive))
    ladder += [
        ("global", "global", []),
        ("global-permissive", "global", relaxed),
        ("incremental-permissive", "incremental", permissive),
    ]

    mapper_configured = mapper
    db_orig = db
    registered, modeldir, tried = 0, None, []
    for n, (tag, want_mapper, flags) in enumerate(ladder):
        mapper = want_mapper
        db = db_orig
        out_dir = work / "sparse" / f"try{n}-{tag}"
        print(f"\n[colmap] rung {n + 1}/{len(ladder)} '{tag}' "
              f"(mapper={want_mapper}{', priors off' if tag == 'no-priors' else ''})",
              flush=True)
        if tag == "no-priors":
            # A copy, so the rungs after this one still see the priors: silently
            # solving the rest of the ladder without them would report that the
            # AR data is what failed, when it was never offered.
            db = work / "database_nopriors.db"
            shutil.copyfile(db_orig, db)
            _drop_priors(db)
        try:
            got, mdir = do_map(flags, tag, out_dir)
        except rb.StepError as e:
            # A rung that dies outright (global_mapper on a thin view graph) is a
            # result, not a verdict: record it and move to the next repair.
            rb.warn(f"rung '{tag}' failed ({e.kind}): {str(e).splitlines()[-1][:160]}")
            tried.append((tag, 0, e.kind))
            continue
        tried.append((tag, got, ""))
        print(f"[colmap] rung '{tag}': {got}/{n_frames} registered")
        if got > registered:
            registered, modeldir = got, mdir
        if registered >= target:
            break

    mapper = mapper_configured
    if modeldir is None:
        detail = "; ".join(f"{t}={g}" + (f" ({k})" if k else "") for t, g, k in tried)
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"[colmap] no rung of the rescue ladder produced a model "
            f"({len(ladder)} tried: {detail}). This footage has no reconstructable "
            "multi-view geometry as captured - run `pipeline.py scan` for the "
            "capture verdict before re-shooting.", returncode=4)
    print(f"\n[colmap] best of {len(tried)} rungs: {registered}/{n_frames} from "
          f"{modeldir.name}")

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

    registered = count_model_images(txt)
    print(f"[colmap] done: {registered}/{n_frames} frames registered "
          f"({100 * registered / max(n_frames, 1):.0f}%). Model: {chosen}")

    # A solve this partial cannot build a world, and `train` does not fail
    # politely on it: too few registered views triangulate too few seed points,
    # the densifier prunes the cloud to nothing, and the next CUDA render
    # fail-fasts (exit 0xC0000409) with no traceback at all. Better to say so
    # here, in seconds, than after 20 minutes of GPU.
    if registered < floor:
        rung_report = ", ".join(f"{t}={g}" + (f"({k})" if k else "")
                                for t, g, k in tried)
        print(f"\n[colmap] FAILED: only {registered} of {n_frames} keyframes solved, "
              f"under the {floor} needed to train.\n"
              f"  Rungs tried: {rung_report}\n"
              f"  Nothing downstream can be built from this, and training would die "
              f"silently rather than report it.\n"
              f"  Check the take before blaming the solver: spin-in-place and "
              f"point-at-one-thing walks give SfM no baseline, and mixed "
              f"portrait/landscape footage now gets one camera per clip (see the "
              f"camera= line above). `python scripts/analyze_take.py <take>` "
              f"reports the path and the baseline.")
        # Named so the runner can distinguish footage that cannot be solved from
        # a resource failure it could retry with different settings.
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"only {registered}/{n_frames} keyframes registered, "
                           f"below the {floor} needed to train",
                           returncode=4)


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        # A traceback here hides the one line that matters: which rung failed,
        # and with what class of problem.
        print(f"\n[colmap] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
