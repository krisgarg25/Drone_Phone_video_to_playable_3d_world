"""Progressive reconstruction executor: the near-real-time story, actually executed.

``survey_streaming.progressive_plan`` answers "which windows, with which overlaps,
at what *predicted* cost". This module executes those windows against the frames
of a survey run and materialises each one as a COLMAP submap, merged into an
accumulated model with COLMAP's own incremental machinery, publishing
``progressive/checkpoints.json`` after every window so partial progress is
observable before the full model exists.

How the sharing and the merging work (this is integration, not invention):

* **One shared database.** Feature extraction and matching run against the run's
  COLMAP database - the same ``database.db`` the monolithic ``colmap`` stage
  already filled. A frame that already carries features is never re-extracted:
  each window hands ``feature_extractor`` an ``--image_list_path`` holding only
  the frames that are new to the database, and COLMAP's matchers skip pairs whose
  two-view geometry already exists ("SKIP: Matches for image pair already exist
  in database", feature_matching.cc). Duplicated frames in the plan therefore
  stay duplicated *mapper* work and never become duplicated feature work.
* **Window submaps without database surgery.** ``mapper``/``pose_prior_mapper``
  take ``--Mapper.image_list_path``, which ``OptionManager::PostParse`` reads into
  ``Mapper.image_names`` and the ``DatabaseCache`` honours as an image subset. One
  solve per window, over the shared features.
* **Merging is COLMAP's.** ``model_merger`` aligns the new submap onto the
  accumulated model through their commonly *registered* images - the plan's
  overlap frames, which carry identical image ids in both models because both
  were solved from the same database - and ``image_registrator`` then re-registers
  every pose-less image of the merged model against the shared matches. No
  transform is fitted in this file.

Honesty rules, mirroring ``survey_streaming``:

* ``secs`` per window is **measured** wall time; ``predicted_s`` is the plan's
  prediction and is carried alongside, never overwritten in either direction.
  The checkpoint's top-level ``label`` is ``"measured"`` because its window rows
  are; the embedded plan keeps its own predicted label.
* A window that fails is recorded as ``failed`` with its reason; later windows
  that can no longer merge are recorded as failed too. Nothing is fabricated:
  counts this module could not read back are ``null``, never 0.
* COLMAP 4.1.1's ``model_merger`` exits **0 even when the merge fails** (it then
  writes ``input_path2`` unchanged), and ``image_registrator`` always exits 0 -
  so success is decided by reading the output models back (registered-image
  names and counts via ``model_converter`` TXT), never by exit codes alone.

Verification status of the argv below: every flag was checked against the
vendored COLMAP 4.1.1 binary's own option strings and the 4.1.1 sources
(exe/model.cc, exe/image.cc, exe/sfm.cc, controllers/option_manager.cc,
scene/reconstruction_io_text.cc); ``feature_extractor``/``sequential_matcher``/
``mapper``/``model_converter`` flags are the ones ``run_colmap.py`` has already
used in real runs. No command in this module has been executed against a real
COLMAP on this machine - unit tests patch ``subprocess.run``, which proves the
argv, not that COLMAP accepts it.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

SCHEMA_VERSION = 1
MEASURED_LABEL = "measured"
CHECKPOINT_RELATIVE_PATH = "progressive/checkpoints.json"
"""Run-dir relative location of the checkpoint file the dashboard polls."""

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
WINDOW_STATUSES = ("pending", "running", "done", "failed")

# Values mirrored from scripts/run_colmap.py DEFAULT_PLAN and its extraction
# flags, so a standalone progressive run produces the same features the
# monolithic colmap stage would have.
DEFAULT_MATCH_OVERLAP = 20
DEFAULT_MAX_IMAGE_SIZE = 1600
DEFAULT_MAX_FEATURES = 8192
DEFAULT_PRIOR_STD_M = 0.15
DEFAULT_SIFT_PEAK_THRESHOLD = 0.002
DEFAULT_SIFT_EDGE_THRESHOLD = 16
COLMAP_TIMEOUT_S = 7200

_CHECKPOINT_ERRORS = (ValueError, KeyError, IndexError, RuntimeError, OSError,
                      sqlite3.Error, subprocess.SubprocessError)
"""What a window is allowed to die of. subprocess.SubprocessError covers a COLMAP
timeout: a hung window is a recorded failure, never a dead run."""


# --- paths and frames ----------------------------------------------------------
def colmap_binary(root):
    """The vendored COLMAP executable, built exactly like survey_workflow does."""
    executable = Path(root) / "tools" / "colmap" / "bin" / "colmap.exe"
    if not executable.is_file():
        raise ValueError("COLMAP executable is missing: tools/colmap/bin/colmap.exe")
    return str(executable)


def frame_names(image_root):
    """Sorted frame names of a written-frames directory, relative POSIX paths.

    The window indices of a plan address positions in *this* ordering; the same
    names are what the COLMAP database holds when the frames were extracted from
    the same directory, which is what makes ``--image_list_path`` and
    ``--Mapper.image_list_path`` work without translation.
    """
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise ValueError("image directory does not exist: " + image_root.name)
    names = [path.relative_to(image_root).as_posix() for path in image_root.rglob("*")
             if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    if not names:
        raise ValueError("image directory holds no frames: " + image_root.name)
    return sorted(names)


def _atomic_write_json(path, payload):
    """Whole-file rewrite via temp + os.replace, so a poller never sees a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


# --- argv builders (pure; every flag verified against COLMAP 4.1.1) ------------
def feature_extraction_command(root, database_path, image_path, image_list_path, *,
                               mask_path=None, max_image_size=DEFAULT_MAX_IMAGE_SIZE,
                               max_features=DEFAULT_MAX_FEATURES,
                               peak_threshold=DEFAULT_SIFT_PEAK_THRESHOLD,
                               edge_threshold=DEFAULT_SIFT_EDGE_THRESHOLD):
    """SIFT extraction for exactly the frames listed - the sharing mechanism.

    ``--image_list_path`` is COLMAP's own ImageReader option (documented for
    feature_extractor; the string is present in the vendored 4.1.1 binary), so a
    frame that was extracted for an earlier window - or by the monolithic colmap
    stage - is never handed to the extractor again.
    """
    argv = [colmap_binary(root), "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--image_list_path", str(image_list_path),
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.gpu_index", "0",
            "--FeatureExtraction.max_image_size", str(max_image_size),
            "--SiftExtraction.max_num_features", str(max_features),
            "--SiftExtraction.peak_threshold", str(peak_threshold),
            "--SiftExtraction.edge_threshold", str(edge_threshold)]
    if mask_path is not None:
        argv += ["--ImageReader.mask_path", str(mask_path)]
    return {"stage": "feature_extractor", "requires_gpu": True, "argv": argv}


def sequential_matching_command(root, database_path, *, overlap=DEFAULT_MATCH_OVERLAP,
                                quadratic_overlap=True):
    """Sequential matching over the shared database; already-matched pairs are skipped.

    COLMAP's matcher checks ``ExistsTwoViewGeometry`` per pair and logs
    "SKIP: Matches for image pair already exist in database", so re-running this
    per window only pays for pairs involving the frames this window added.
    """
    return {"stage": "sequential_matcher", "requires_gpu": True,
            "argv": [colmap_binary(root), "sequential_matcher",
                     "--database_path", str(database_path),
                     "--SequentialMatching.overlap", str(overlap),
                     "--SequentialMatching.quadratic_overlap",
                     "1" if quadratic_overlap else "0",
                     "--FeatureMatching.use_gpu", "1",
                     "--FeatureMatching.gpu_index", "0"]}


def window_mapper_command(root, database_path, image_path, image_list_path, output_path,
                          *, pose_priors=False, prior_std=DEFAULT_PRIOR_STD_M):
    """One window's submap solve, restricted by COLMAP's own image-list option.

    ``--Mapper.image_list_path`` is registered by ``OptionManager::AddMapperOptions``
    and read into ``Mapper.image_names`` by ``PostParse``; the ``DatabaseCache``
    then loads only those images (database_cache.h: "use only the data for a
    subset of the images"). ``multiple_models 0`` mirrors run_colmap.py: a window
    that splits into components keeps one model instead of fragments.
    """
    if pose_priors:
        argv = [colmap_binary(root), "pose_prior_mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_path),
                "--output_path", str(output_path),
                "--overwrite_priors_covariance", "1",
                "--prior_position_std_x", str(prior_std),
                "--prior_position_std_y", str(prior_std),
                "--prior_position_std_z", str(prior_std),
                "--use_robust_loss_on_prior_position", "1"]
        stage = "pose_prior_mapper"
    else:
        argv = [colmap_binary(root), "mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_path),
                "--output_path", str(output_path)]
        stage = "mapper"
    argv += ["--Mapper.image_list_path", str(image_list_path),
             "--Mapper.multiple_models", "0"]
    return {"stage": stage, "requires_gpu": False, "argv": argv}


def model_merger_command(root, submap_path, accumulated_path, output_path):
    """COLMAP's relative-model merge: submap (src) onto accumulated (tgt).

    Direction matters and comes from the 4.1.1 source: ``model_merger`` calls
    ``MergeAndFilterReconstructions(max_reproj_error, reconstruction1,
    reconstruction2)``, which aligns reconstruction1 onto reconstruction2 through
    commonly registered image ids and then **writes reconstruction2**. Passing the
    new submap as ``--input_path1`` and the accumulated model as ``--input_path2``
    therefore keeps the accumulated coordinate frame stable across windows.
    The output directory must exist (``Reconstruction::WriteBinary`` checks).
    """
    return {"stage": "model_merger", "requires_gpu": False,
            "argv": [colmap_binary(root), "model_merger",
                     "--input_path1", str(submap_path),
                     "--input_path2", str(accumulated_path),
                     "--output_path", str(output_path)]}


def image_registrator_command(root, database_path, merged_path, output_path):
    """COLMAP's own re-registration over the shared overlap matches.

    ``image_registrator`` iterates the images *of the input reconstruction* that
    have no pose and calls ``RegisterNextImage`` using the database's matches
    (exe/image.cc). Both directories must exist; it writes a binary model and
    always exits 0 when the paths are valid, so what it registered is measured
    from the output model, never from the exit code.
    """
    return {"stage": "image_registrator", "requires_gpu": False,
            "argv": [colmap_binary(root), "image_registrator",
                     "--database_path", str(database_path),
                     "--input_path", str(merged_path),
                     "--output_path", str(output_path)]}


def model_converter_command(root, model_path, txt_path):
    """BIN -> TXT so counts and image names can be read without parsing binaries.

    images.bin's record layout gained fields with COLMAP 4's rigs/frames
    refactor, so - exactly like run_colmap.count_via_converter - the binary that
    wrote the model converts it and the text format is parsed. The TXT directory
    must exist (``Reconstruction::WriteText`` checks).
    """
    return {"stage": "model_converter", "requires_gpu": False,
            "argv": [colmap_binary(root), "model_converter",
                     "--input_path", str(model_path),
                     "--output_path", str(txt_path),
                     "--output_type", "TXT"]}


# --- database inspection (read-only, stdlib sqlite3) ---------------------------
def _db_extracted_names(database_path):
    """Image names in the database that already carry SIFT features."""
    database_path = Path(database_path)
    if not database_path.is_file():
        return set()
    try:
        con = sqlite3.connect(str(database_path))
        try:
            rows = con.execute("SELECT images.name FROM images "
                               "JOIN keypoints ON keypoints.image_id = images.image_id "
                               "WHERE keypoints.rows > 0")
            return {name for (name,) in rows}
        finally:
            con.close()
    except sqlite3.Error:
        return set()


def _db_has_pose_priors(database_path):
    """Whether the shared database carries GPS pose priors (survey_priors stage)."""
    database_path = Path(database_path)
    if not database_path.is_file():
        return False
    try:
        con = sqlite3.connect(str(database_path))
        try:
            have = con.execute("SELECT COUNT(*) FROM sqlite_master "
                               "WHERE type='table' AND name='pose_priors'").fetchone()[0]
            if not have:
                return False
            return con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0] > 0
        finally:
            con.close()
    except sqlite3.Error:
        return False


# --- model read-back ------------------------------------------------------------
def read_model_stats(txt_dir):
    """Registered images, points and registered image names from a TXT model.

    images.txt holds exactly two lines per *registered* image (COLMAP writes
    ``RegImageIds()`` only), the second being its 2D points - the same pairing
    run_colmap.count_model_images relies on. Anything unreadable stays ``None``:
    a count that could not be taken is missing data, not a zero.
    """
    txt_dir = Path(txt_dir)
    stats = {"registered_images": None, "points": None, "image_names": None}
    images = txt_dir / "images.txt"
    if images.is_file():
        rows = [line for line in images.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if len(rows) % 2:
            raise ValueError("images.txt has an odd number of data rows; expected "
                             "the pose/points line pair COLMAP writes per image")
        names = set()
        for line in rows[::2]:
            fields = line.split()
            if len(fields) < 10:
                raise ValueError("images.txt pose row has fewer than the ten fields "
                                 "COLMAP writes (IMAGE_ID..NAME)")
            names.add(fields[9])
        stats["registered_images"] = len(rows) // 2
        stats["image_names"] = names
    points = txt_dir / "points3D.txt"
    if points.is_file():
        stats["points"] = len([line for line in points.read_text(encoding="utf-8").splitlines()
                               if line.strip() and not line.lstrip().startswith("#")])
    return stats


# --- the executor ----------------------------------------------------------------
def execute_plan(root, run, plan, *, image_dir="frames_match", database_path=None,
                 match_overlap=DEFAULT_MATCH_OVERLAP,
                 max_image_size=DEFAULT_MAX_IMAGE_SIZE,
                 max_features=DEFAULT_MAX_FEATURES, timeout=COLMAP_TIMEOUT_S):
    """Execute a ``progressive_plan()`` over a run's written frames.

    Returns the checkpoint payload (the same dict that is atomically rewritten at
    ``progressive/checkpoints.json`` after every window). Per-window failures are
    recorded, never raised; only plan/setup disagreements - a plan built for a
    different frame count, a missing binary, no frames on disk - raise, and the
    caller (``reconstruct_scene``) records those as ``progressive_error`` without
    endangering the monolithic reconstruction.
    """
    root, run = Path(root).resolve(), Path(run).resolve()
    if not run.is_relative_to(root):
        raise ValueError("run directory leaves the project root")
    if not isinstance(plan, dict) or not isinstance(plan.get("windows"), list) \
            or not plan["windows"]:
        raise ValueError("plan must be a progressive_plan() result with a non-empty "
                         "windows list")
    colmap_binary(root)  # refuse before any directory exists, like _preflight does
    frames = frame_names(run / image_dir)
    if plan.get("frame_count") != len(frames):
        raise ValueError(f"plan was built for {plan.get('frame_count')} frames but "
                         f"{image_dir} holds {len(frames)}; replan before executing")
    for row in plan["windows"]:
        if not (0 <= row["start"] < row["end"] <= len(frames)) \
                or row["end"] - row["start"] != row["frame_count"]:
            raise ValueError(f"plan window {row.get('index')} does not address the "
                             f"{len(frames)} frames on disk")

    prog = run / "progressive"
    models, lists, logs, txt = (prog / "models", prog / "lists", prog / "logs", prog / "txt")
    for directory in (models, lists, logs, txt):
        directory.mkdir(parents=True, exist_ok=True)
    database = Path(database_path) if database_path is not None else run / "database.db"

    def relative(path):
        return Path(path).resolve().relative_to(run).as_posix()

    def start(command, log_name, *, allow_fail=False):
        """One COLMAP invocation, logged; raises RuntimeError on a bad exit code."""
        log_path = logs / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print("[progressive] " + command["stage"] + " -> " + relative(log_path), flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(command["argv"], cwd=root, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=timeout,
                                      check=False)
        except subprocess.TimeoutExpired as error:
            # A hung window is a recorded failure, never a dead run - and the
            # checkpoint gets a run-relative reason instead of an argv dump that
            # would leak absolute local paths to the browser.
            raise RuntimeError(f"colmap {command['argv'][1]} timed out after "
                               f"{error.timeout} s; inspect {relative(log_path)}") from error
        if proc.returncode and not allow_fail:
            raise RuntimeError(f"colmap {command['argv'][1]} failed with exit "
                               f"{proc.returncode}; inspect {relative(log_path)}")
        return proc.returncode

    def convert_and_read(model_dir, txt_dir, log_name):
        txt_dir.mkdir(parents=True, exist_ok=True)
        start(model_converter_command(root, model_dir, txt_dir), log_name)
        return read_model_stats(txt_dir)

    rows = [{"index": row["index"], "start": row["start"], "end": row["end"],
             "frame_count": row["frame_count"],
             "overlap_with_previous": row["overlap_with_previous"],
             "status": "pending", "secs": None, "predicted_s": row.get("predicted_s"),
             "registered_images": None, "points": None, "model_path": None,
             "error": None}
            for row in plan["windows"]]
    started = time.perf_counter()
    state = {"first_useful_output_s": None}

    def payload():
        done = [row for row in rows if row["status"] == "done"]
        last = done[-1] if done else None
        return {"schema_version": SCHEMA_VERSION,
                "plan": plan,
                "label": MEASURED_LABEL,
                "windows": rows,
                "accumulated": {"registered_images": None if last is None else last["registered_images"],
                                "points": None if last is None else last["points"],
                                "windows_done": len(done), "windows_total": len(rows)},
                "first_useful_output_s": state["first_useful_output_s"]}

    def publish():
        _atomic_write_json(prog / "checkpoints.json", payload())

    publish()
    extracted = _db_extracted_names(database)
    if extracted:
        print(f"[progressive] shared database already carries features for "
              f"{len(extracted)} frames - they will not be re-extracted", flush=True)
    accumulated_dir, accumulated_stats, blocked = None, None, None

    for row, window in zip(plan["windows"], rows):
        index = row["index"]
        tag = f"window_{index:02d}"
        window.update(status="running")
        publish()
        t0 = time.perf_counter()
        try:
            if blocked is not None:
                raise RuntimeError(blocked)
            names = frames[row["start"]:row["end"]]
            window_list = lists / f"{tag}.txt"
            window_list.write_text("\n".join(names) + "\n", encoding="utf-8")
            predicted = row.get("predicted_s")
            print(f"[progressive] window {index}: {len(names)} frames "
                  f"({row['overlap_with_previous']} shared with previous), predicted "
                  f"{'n/a' if predicted is None else str(predicted) + ' s'}", flush=True)

            # --- shared features: extract and match only what is new ------------
            new_names = [name for name in names if name not in extracted]
            if new_names:
                new_list = lists / f"{tag}_new.txt"
                new_list.write_text("\n".join(new_names) + "\n", encoding="utf-8")
                mask_dir = run / "masks"
                start(feature_extraction_command(
                    root, database, run / image_dir, new_list,
                    mask_path=mask_dir if mask_dir.is_dir() else None,
                    max_image_size=max_image_size, max_features=max_features),
                    f"{tag}_feature_extractor.log")
                extracted_now = _db_extracted_names(database)
                missing = [name for name in names if name not in extracted_now]
                if missing:
                    # The guard run_colmap.py learned from the mask bug: COLMAP
                    # rejects images with a warning and exit 0, so the database
                    # itself is the only witness.
                    shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
                    raise RuntimeError(
                        f"feature_extractor took {len(names) - len(missing)} of "
                        f"{len(names)} window frames into the database; COLMAP refuses "
                        f"an image whose dimensions disagree with the camera it was "
                        f"told to share, and still exits 0. Missing: {shown}")
                extracted = extracted_now
                start(sequential_matching_command(root, database, overlap=match_overlap),
                      f"{tag}_sequential_matcher.log")
            else:
                print(f"[progressive] window {index}: all {len(names)} frames already "
                      f"carry features in the shared database - no extraction or "
                      f"matching repeated", flush=True)

            # --- the window's own submap -----------------------------------------
            sparse = prog / "windows" / tag
            sparse.mkdir(parents=True, exist_ok=True)
            priors = _db_has_pose_priors(database)
            code = start(window_mapper_command(root, database, run / image_dir,
                                               window_list, sparse, pose_priors=priors),
                         f"{tag}_pose_prior_mapper.log" if priors else f"{tag}_mapper.log",
                         allow_fail=priors)
            if code:
                print(f"[progressive] window {index}: pose_prior_mapper failed with exit "
                      f"{code} - falling back to the plain mapper", flush=True)
                start(window_mapper_command(root, database, run / image_dir,
                                            window_list, sparse, pose_priors=False),
                      f"{tag}_mapper.log")
            candidates = sorted((path for path in sparse.iterdir()
                                 if path.is_dir() and (path / "images.bin").is_file()),
                                key=lambda path: path.name)
            if not candidates:
                raise RuntimeError(f"mapper produced no model for window {index}; "
                                   f"inspect {relative(logs)}")
            best_dir, best_stats = None, None
            for position, candidate in enumerate(candidates):
                stats = convert_and_read(candidate, txt / f"{tag}_{candidate.name}",
                                         f"{tag}_model_converter_{position}.log")
                if best_stats is None or (stats["registered_images"] or 0) > \
                        (best_stats["registered_images"] or 0):
                    best_dir, best_stats = candidate, stats
            if not best_stats["registered_images"]:
                raise RuntimeError(f"window {index} submap registered "
                                   f"{best_stats['registered_images']} of {len(names)} "
                                   f"frames - no usable submap")

            # --- merge into the accumulated model, COLMAP's own machinery --------
            accumulated = models / f"accumulated_{index:02d}"
            if accumulated_dir is None:
                shutil.copytree(best_dir, accumulated, dirs_exist_ok=True)
            else:
                merged = models / f"merged_{index:02d}"
                merged.mkdir(parents=True, exist_ok=True)
                start(model_merger_command(root, best_dir, accumulated_dir, merged),
                      f"{tag}_model_merger.log")
                merged_stats = convert_and_read(merged, txt / f"merged_{index:02d}",
                                                f"{tag}_model_converter_merged.log")
                # model_merger exits 0 even when alignment fails (COLMAP 4.1.1 then
                # writes input_path2 unchanged), so the output is verified from the
                # models themselves: a real merge holds every registered image of
                # both parents.
                parents = (accumulated_stats["image_names"] or set()) | \
                          (best_stats["image_names"] or set())
                missing = parents - (merged_stats["image_names"] or set())
                if merged_stats["registered_images"] is None or missing:
                    raise RuntimeError(
                        f"model_merger did not merge window {index}: {len(missing)} "
                        f"registered images of the parents are missing from its output "
                        f"(COLMAP exits 0 even when the merge fails); the shared overlap "
                        f"frames did not make the submap registrable onto the "
                        f"accumulated model")
                accumulated.mkdir(parents=True, exist_ok=True)
                start(image_registrator_command(root, database, merged, accumulated),
                      f"{tag}_image_registrator.log")
            stats = convert_and_read(accumulated, txt / f"accumulated_{index:02d}",
                                     f"{tag}_model_converter_accumulated.log")
            if stats["registered_images"] is None:
                raise RuntimeError(f"accumulated model after window {index} could not "
                                   f"be read back; its counts would be invented")
            window.update(status="done", secs=round(time.perf_counter() - t0, 3),
                          registered_images=stats["registered_images"],
                          points=stats["points"],
                          model_path=f"progressive/models/accumulated_{index:02d}",
                          error=None)
            accumulated_dir, accumulated_stats = accumulated, stats
            if state["first_useful_output_s"] is None and stats["registered_images"] > 0:
                state["first_useful_output_s"] = round(time.perf_counter() - started, 3)
            print(f"[progressive] window {index}: done in {window['secs']} s (measured), "
                  f"{stats['registered_images']} registered images, {stats['points']} "
                  f"points", flush=True)
        except _CHECKPOINT_ERRORS as error:
            # The checkpoint is served to a browser: an unexpected OSError text
            # must not smuggle absolute local paths into it.
            message = str(error)
            for absolute, shown in ((str(run), "<run>"), (run.as_posix(), "<run>"),
                                    (str(root), "<root>"), (root.as_posix(), "<root>"),
                                    # repr-quoted forms (doubled backslashes), as
                                    # they appear in exception texts carrying argv.
                                    (str(run).replace("\\", "\\\\"), "<run>"),
                                    (str(root).replace("\\", "\\\\"), "<root>")):
                message = message.replace(absolute, shown)
            window.update(status="failed", secs=round(time.perf_counter() - t0, 3),
                          error=message)
            if blocked is None:
                blocked = (f"not attempted: window {index} failed ({message}); without its "
                           f"accumulated model this window shares no frames with anything "
                           f"registered, so model_merger has no common images to align")
            print(f"[progressive] window {index}: FAILED - {message}", flush=True)
        publish()

    result = payload()
    if state["first_useful_output_s"] is not None:
        print(f"[progressive] first useful output after "
              f"{state['first_useful_output_s']} s (measured wall time from the start "
              f"of the progressive executor)", flush=True)
    return result
