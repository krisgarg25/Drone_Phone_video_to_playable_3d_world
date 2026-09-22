import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}
MAX_INPUT_BYTES = 2 * 1024 * 1024
# One spelling for the COLMAP sparse model. It is opened, fingerprinted and
# verified through this constant alone, so provenance cannot silently stop
# matching because two call sites disagreed about the case of points3D.
POINTS3D = "colmap/sparse/txt/points3D.txt"
EVIDENCE_POINTS = "survey/evidence/evidence_points.ply"
EVIDENCE_SUMMARY = "survey/evidence/evidence_summary.json"
# These KB-scale survey verdicts are rehashed whenever the dashboard polls: a
# stat-only freshness check lets a hand-edited pass, or an evaluation.json left
# behind by an older run, keep rendering green. evaluation.json is green-proof a
# different way - its criteria are recomputed from the other three - and the video
# and the PLY clouds are far too large to digest per poll, so they stay on
# size-plus-mtime checks.
HASH_ON_READ = ("checkpoints.json", "surface_reference.json", "georeference.json",
                "evaluation.json", "run.json")


def scene_paths(root, scene):
    root = Path(root).resolve()
    if not isinstance(scene, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", scene):
        raise ValueError("Scene must contain 1–64 letters, numbers, underscores or hyphens.")
    paths = (root / "videos" / scene, root / "work" / scene)
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Scene path leaves the project.")
    return paths


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _modules():
    import survey_evaluation as evaluation
    import survey_georef as georef
    return evaluation, georef


def _source_files(root, scene):
    source, _ = scene_paths(root, scene)
    videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS) if source.is_dir() else []
    if len(videos) != 1:
        raise ValueError("A single-pass survey requires exactly one video in videos/" + scene + "/.")
    files = [videos[0], source / "telemetry.csv", source / "flight_metadata.json"]
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("Missing or unsafe input: " + path.name)
    return files


def _safe_path(base, relative):
    base = Path(base).resolve()
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise ValueError("Unsafe path leaves its workspace: " + str(relative))
    return path


def _fingerprint(path, root):
    evaluation, _ = _modules()
    path, root = Path(path).absolute(), Path(root).absolute()
    _safe_path(root, path.relative_to(root))
    result = evaluation.file_fingerprint(path)
    result.update(relative_path=path.relative_to(root).as_posix(), mtime_ns=path.stat().st_mtime_ns)
    return result


def _matches(path, evidence, full=False):
    # Polling is stat-only for the large inputs. Actions also verify the content
    # digest, including edits that preserve file size and timestamps, and the
    # KB-scale verdict files are digested on a poll too.
    if not isinstance(evidence, dict) or not path.is_file():
        return False
    if not re.fullmatch(r"[a-f0-9]{64}", str(evidence.get("sha256", ""))):
        return False
    stat = path.stat()
    if stat.st_size != evidence.get("size_bytes") or stat.st_mtime_ns != evidence.get("mtime_ns"):
        return False
    full = full or path.name in HASH_ON_READ
    return not full or _modules()[0].file_fingerprint(path)["sha256"] == evidence["sha256"]


def _unchanged(root, manifest, full=False):
    return len(manifest.get("inputs", [])) == 3 and all(
        _matches(_safe_path(root, item["relative_path"]), item, full)
        for item in manifest["inputs"]
    )


def save_inputs(root, scene, telemetry_csv, metadata):
    _, georef = _modules()
    source, _ = scene_paths(root, scene)
    if not isinstance(telemetry_csv, str) or len(telemetry_csv.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Telemetry must be CSV text below 2 MiB.")
    if not isinstance(metadata, dict):
        raise ValueError("Flight metadata must be a JSON object.")
    with tempfile.TemporaryDirectory(prefix="survey-input-") as temp:
        path = Path(temp) / "telemetry.csv"
        path.write_text(telemetry_csv, encoding="utf-8")
        georef.normalize_telemetry(path, metadata)
    target_csv, target_meta = source / "telemetry.csv", source / "flight_metadata.json"
    if target_csv.exists() or target_meta.exists():
        raise FileExistsError("Survey inputs already exist; use a new scene or edit them locally, then prepare again.")
    source.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(metadata, indent=2, allow_nan=False)
    with target_csv.open("x", encoding="utf-8", newline="") as stream:
        stream.write(telemetry_csv)
    try:
        with target_meta.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except Exception:
        target_csv.unlink(missing_ok=True)
        raise
    return scene_status(root, scene)


def prepare_scene(root, scene):
    _, georef = _modules()
    root = Path(root)
    files = _source_files(root, scene)
    metadata = read_json(files[2])
    telemetry = georef.normalize_telemetry(files[1], metadata)
    _, work = scene_paths(root, scene)
    manifest = {
        "schema_version": 1, "id": uuid.uuid4().hex,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": [_fingerprint(p, root) for p in files],
        "metadata": metadata, "video_duration_verified": False,
        "telemetry": telemetry, "gpu_execution_enabled": False,
    }
    write_json(work / "survey" / "preparation.json", manifest)
    return scene_status(root, scene)


def _prepared(root, scene):
    _, work = scene_paths(root, scene)
    path = work / "survey" / "preparation.json"
    if not path.is_file():
        raise ValueError("Prepare the video and telemetry before this action.")
    manifest = read_json(path)
    if not _unchanged(root, manifest, full=True):
        raise ValueError("Survey inputs changed; prepare again before using results.")
    return work, manifest


def _camera_rows(path):
    if not path.is_file():
        raise ValueError("Registered camera poses are missing. Prepare the GPU reconstruction command first.")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_sparse_points(path, label=None):
    """Parse COLMAP points3D rows: id, x y z, r g b, error, then the track."""
    import numpy as np
    label = label or Path(path).name
    xyz, rgb = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 7:
                raise ValueError("Truncated COLMAP points3D row " + str(number) + " in " + label
                                 + ": 7 fields required (id x y z r g b), found " + str(len(fields)))
            try:
                xyz.append([float(x) for x in fields[1:4]])
                rgb.append([int(x) for x in fields[4:7]])
            except ValueError as error:
                raise ValueError("Unreadable COLMAP points3D row " + str(number) + " in " + label
                                 + ": " + str(error)) from error
    points = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Sparse model has no finite XYZ points.")
    if colors.shape != points.shape or np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("Sparse model has invalid RGB values.")
    return points, colors


def _export_points(points, colors, alignment, destination):
    import numpy as np
    _, georef = _modules()
    points, colors = np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Export requires nonempty finite Nx3 XYZ points.")
    if (colors.shape != points.shape or not np.isfinite(colors).all()
            or np.any(colors < 0) or np.any(colors > 255) or np.any(colors != np.floor(colors))):
        raise ValueError("Export requires matching finite Nx3 integer RGB colors in 0..255.")
    mapped = georef.transform_points(points, alignment)
    if mapped.shape != points.shape or not np.isfinite(mapped).all():
        raise ValueError("Georeferenced export contains invalid XYZ points.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="ascii", newline="\n") as stream:
            stream.write("ply\nformat ascii 1.0\ncomment Coordinates: local ENU metres; see georeference.json\n")
            stream.write(f"element vertex {len(mapped)}\nproperty double x\nproperty double y\nproperty double z\n")
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for point, color in zip(mapped, colors):
                stream.write(" ".join(f"{x:.9f}" for x in point) + " " + " ".join(str(int(x)) for x in color) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def align_scene(root, scene):
    _, georef = _modules()
    work, preparation = _prepared(root, scene)
    run, latest = _latest_run(work, preparation, full=True)
    if latest and latest["status"] == "complete":
        # The dense run has its own poses and transform; never overwrite it
        # using unrelated legacy cameras.
        return scene_status(root, scene)
    pose_path = _safe_path(work, "keyframes_poses.jsonl")
    alignment = georef.align_camera_trajectory(_camera_rows(pose_path), preparation["telemetry"])
    alignment["preparation_id"] = preparation["id"]
    sources = {"poses": _fingerprint(pose_path, work)}
    alignment["pose_sha256"] = sources["poses"]["sha256"]
    sparse = _safe_path(work, POINTS3D)
    if sparse.is_file():
        points, colors = _read_sparse_points(sparse, POINTS3D)
        destination = _safe_path(work, "survey/sparse_points.ply")
        _export_points(points, colors, alignment, destination)
        sources.update(sparse=_fingerprint(sparse, work), sparse_points=_fingerprint(destination, work))
        alignment["sparse_sha256"] = sources["sparse"]["sha256"]
        import survey_products as products
        cloud, summary = products.evidence_for_sparse(sparse, alignment,
                                                      _safe_path(work, "survey/evidence"))
        sources.update(evidence_points=_fingerprint(cloud, work),
                       evidence_summary=_fingerprint(summary, work))
    alignment["sources"] = sources
    write_json(_safe_path(work, "survey/georeference.json"), alignment)
    return scene_status(root, scene)


def _current_alignment(work, preparation, full=False):
    path = _safe_path(work, "survey/georeference.json")
    if not path.is_file():
        return None
    data = read_json(path)
    if data.get("preparation_id") != preparation["id"]:
        return None
    sources = data.get("sources", {})
    for key, name, digest in (("poses", "keyframes_poses.jsonl", "pose_sha256"),
                              ("sparse", POINTS3D, "sparse_sha256"),
                              ("sparse_points", "survey/sparse_points.ply", None),
                              ("evidence_points", EVIDENCE_POINTS, None),
                              ("evidence_summary", EVIDENCE_SUMMARY, None)):
        if key != "poses" and not data.get("sparse_sha256"):
            continue
        if key.startswith("evidence") and key not in sources:
            continue
        evidence = sources.get(key, {})
        if (not _matches(_safe_path(work, name), evidence, full)
                or (digest and evidence.get("sha256") != data.get(digest))):
            label = ("Registered camera poses" if key == "poses"
                     else "Evidence cloud/summary" if key.startswith("evidence")
                     else "Sparse model/export")
            raise ValueError(label + " changed or lacks provenance; align again: " + name)
    return data


RUN_FILES = ("georeference.json", "dense_points.ply", "keyframes_poses.jsonl",
             POINTS3D, "dense/fused.ply",
             "evidence/evidence_summary.json", "evidence/evidence_points.ply")


def _latest_run(work, preparation, full=False):
    pointer_path = _safe_path(work, "survey/latest_run.json")
    if not pointer_path.is_file():
        return None, None
    pointer = read_json(pointer_path)
    run_id = pointer.get("id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[a-f0-9]{8}", run_id):
        raise ValueError("Latest run has an invalid exact run id.")
    if pointer.get("preparation_id") != preparation["id"]:
        raise ValueError("Latest run belongs to a different preparation; reconstruct again.")
    runs = _safe_path(work, "survey/runs")
    run = _safe_path(runs, run_id)
    # Reject aliases to another run, including junctions/symlinks inside runs.
    if run.resolve() != runs.resolve() / run_id:
        raise ValueError("Latest run path is not its exact isolated run directory.")
    report_path = _safe_path(run, "run.json")
    if not _matches(report_path, pointer.get("report"), full):
        raise ValueError("Latest run report missing or changed (hash/stat): run.json")
    record = read_json(report_path)
    if (record.get("id") != run_id or record.get("preparation_id") != preparation["id"]
            or record.get("status") != pointer.get("status")
            or record.get("status") not in {"running", "failed", "complete"}
            or record.get("inputs") != preparation["inputs"]):
        raise ValueError("Latest run identity, preparation, inputs or status mismatch.")
    if record["status"] == "complete":
        # Validate exactly what the run recorded: evidence products are optional
        # (they need a sparse model and depth maps), but anything present at
        # publish time must still match.
        for name in record.get("files", {}):
            if not _matches(_safe_path(run, name), record["files"][name], full):
                raise ValueError("Latest run evidence missing or changed (hash/stat): " + name)
        alignment = read_json(run / "georeference.json")
        if alignment.get("preparation_id") != preparation["id"] or alignment.get("run_id") != run_id:
            raise ValueError("Latest run georeference provenance mismatch.")
    return run, record


def _evidence_context(work, preparation, full=False, selection=None):
    run, latest = selection if selection is not None else _latest_run(work, preparation, full)
    if latest and latest["status"] == "complete":
        georeference, report = run / "georeference.json", run / "run.json"
        alignment = read_json(georeference)
    else:
        alignment = _current_alignment(work, preparation, full)
        georeference = work / "survey/georeference.json" if alignment else None
        # A failed/running current run must not borrow an old timing report.
        report = _safe_path(work, "report.json") if latest is None else None
        if report is not None and not report.is_file():
            report = None
    return {"run": run, "latest": latest, "alignment": alignment,
            "georeference": georeference, "report": report}


def _evaluation_paths(work, context):
    paths = {"georeference": context["georeference"], "report": context["report"]}
    for key, name in (("checkpoint", "checkpoints.json"), ("surface", "surface_reference.json")):
        path = _safe_path(work, "survey/" + name)
        paths[key] = path if path.is_file() else None
    return paths


def _total_ram_bytes():
    """Total physical RAM in bytes, read without a subprocess or a third-party dep."""
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError, ValueError):
            return None
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _hardware():
    """Host identity for a run, taken from the CPU only.

    No GPU is probed here: nvidia-smi and torch stay out of a CPU-only survey path,
    so the GPU entry is a placeholder the caller may fill. Timing from a run with no
    recorded hardware is never presented as an official benchmark.
    """
    import platform
    cpu = (platform.processor() or platform.machine() or "").strip()
    cores = os.cpu_count()
    ram = _total_ram_bytes()
    return {"status": "recorded" if cpu and cores else "not_recorded",
            "cpu": cpu or None, "logical_cores": cores, "total_ram_bytes": ram,
            "gpu": "not_recorded_gpu"}


def _required_stages(record):
    """Every stage the run's own published plan said it had to execute."""
    return (["setup"] + [command["stage"] for command in record.get("commands", [])]
            + ["georeferenced_export"])


def _diagnostic_speed(report, duration, required_stages=None):
    """Qualify timing from the evidence in the report, downgrading what is not it.

    A run records its hardware at setup and takes its duration from the decoder, so
    the official 600 s / <900 s gate can fire. Anything else - a legacy report.json
    whose duration came from declared metadata, or a run whose host identity could
    not be read - keeps the diagnostic label instead of a claimed pass.
    """
    speed = _modules()[0].speed_metrics(report, duration, required_stages=required_stages)
    hardware = report.get("hardware") if isinstance(report, dict) else None
    verified = bool(isinstance(report, dict)
                    and report.get("duration_source") == "decoder_metadata"
                    and isinstance(hardware, dict) and hardware.get("status") == "recorded")
    speed["diagnostic_only"] = not verified
    if not verified:
        speed.update(status="not_evaluated", official_status="not_evaluated",
                     reason="Timing diagnostic only; source duration or hardware identity not "
                            "recorded with this run, so no official benchmark or accuracy qualification.")
    return speed


def _evaluation_metrics(preparation, context, paths):
    """Recompute every presented criterion from the evidence itself.

    Both the evaluate action and the read path call this, so a stored evaluation can
    never present a status its own inputs do not support.
    """
    evaluation, _ = _modules()
    alignment, latest = context["alignment"], context["latest"]
    checkpoint_result = surface_result = None
    if paths["checkpoint"]:
        data = read_json(paths["checkpoint"])
        if alignment is None:
            raise ValueError("Align the scene before evaluating absolute checkpoints.")
        if (not isinstance(data, dict) or set(data) != {"independent", "alignment", "coordinate_frame", "checkpoints"}
                or data.get("independent") is not True or data.get("alignment") != "none"):
            raise ValueError("Checkpoints require exactly independent:true, alignment:'none', coordinate_frame, checkpoints.")
        if data.get("coordinate_frame") != alignment["coordinate_frame"]:
            raise ValueError("Checkpoint coordinate_frame must exactly match current georeference.json.")
        rows = data.get("checkpoints", [])
        if (not isinstance(rows, list) or not rows or len(rows) > 10000
                or any(not isinstance(r, dict) or set(r) != {"reconstructed", "reference"} for r in rows)):
            raise ValueError("Supply 1–10000 independent checkpoint pairs: reconstructed, reference.")
        checkpoint_result = evaluation.checkpoint_metrics([r["reconstructed"] for r in rows], [r["reference"] for r in rows])
    if paths["surface"]:
        data = read_json(paths["surface"])
        if alignment is None:
            raise ValueError("Align the scene before evaluating a surface reference.")
        if (not isinstance(data, dict)
                or set(data) != {"independent", "coordinate_frame", "visible_reference", "reconstructed", "reference"}
                or data.get("independent") is not True or data.get("visible_reference") is not True):
            raise ValueError("Surface reference requires exactly independent:true, visible_reference:true, coordinate_frame, reconstructed, reference.")
        if data["coordinate_frame"] != alignment["coordinate_frame"]:
            raise ValueError("Surface coordinate_frame must exactly match current georeference.json.")
        for key in ("reconstructed", "reference"):
            if not isinstance(data[key], list) or not 1 <= len(data[key]) <= 10000:
                raise ValueError("Surface " + key + " requires 1–10000 XYZ points.")
        surface_result = evaluation.surface_metrics(data["reconstructed"], data["reference"])
    speed = None
    if paths["report"]:
        if latest:
            speed = _diagnostic_speed(latest, latest["video_duration_s"], _required_stages(latest))
        else:
            speed = _diagnostic_speed(read_json(paths["report"]), preparation["metadata"]["video_duration_s"])
            speed["reason"] += " Historical source duration is declared, not independently verified."
    return evaluation.build_evaluation(checkpoints=checkpoint_result, surface=surface_result,
                                       speed=speed, georeferenced=alignment is not None)


def evaluate_scene(root, scene):
    work, preparation = _prepared(root, scene)
    context = _evidence_context(work, preparation, full=True)
    latest = context["latest"]
    paths = _evaluation_paths(work, context)
    # Capture provenance before computing metrics, then check again before publishing.
    sources = {key: _fingerprint(path, work) if path else None for key, path in paths.items()}
    result = _evaluation_metrics(preparation, context, paths)
    result.update(preparation_id=preparation["id"], sources=sources,
                  source_run_id=latest["id"] if latest and latest["status"] == "complete" else None)
    for key, evidence in sources.items():
        result[key + "_sha256"] = evidence["sha256"] if evidence else None
        result[key + "_source"] = evidence["relative_path"] if evidence else None
    for key, path in paths.items():
        if path and not _matches(path, sources[key], full=True):
            raise ValueError("Evaluation source changed during evaluation: " + path.name)
    write_json(_safe_path(work, "survey/evaluation.json"), result)
    return scene_status(root, scene)


def _current_evaluation(work, preparation, context=None, full=False):
    path = _safe_path(work, "survey/evaluation.json")
    if not path.is_file():
        return None
    data = read_json(path)
    if data.get("preparation_id") != preparation["id"]:
        raise ValueError("Evaluation belongs to a different preparation; evaluate again.")
    context = context or _evidence_context(work, preparation, full)
    latest = context["latest"]
    run_id = latest["id"] if latest and latest["status"] == "complete" else None
    if data.get("source_run_id") != run_id:
        raise ValueError("Evaluation source run changed; evaluate again.")
    paths = _evaluation_paths(work, context)
    for key, other in paths.items():
        evidence = data.get("sources", {}).get(key)
        if other is None and evidence is None and key in data.get("sources", {}):
            continue
        if (other is None or not evidence or evidence.get("relative_path") != other.relative_to(work).as_posix()
                or not _matches(other, evidence, full) or data.get(key + "_sha256") != evidence["sha256"]):
            name = other.name if other else (data.get(key + "_source") or key)
            raise ValueError("Evaluation source/hash changed or missing: " + name + "; evaluate again.")
    # The stored criteria are a snapshot of a past action. What is presented is
    # recomputed from these re-verified inputs, so a hand-edited or older
    # evaluation.json cannot render accuracy or speed as a pass.
    data["criteria"] = _evaluation_metrics(preparation, context, paths)["criteria"]
    return data


def scene_status(root, scene):
    evaluation, _ = _modules()
    source, work = scene_paths(root, scene)
    videos = [p for p in source.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()] if source.is_dir() else []
    readiness = [
        {"label": "Single video", "status": "ready" if len(videos) == 1 else "missing", "detail": f"{len(videos)} video files; exactly one required"},
        {"label": "GPS telemetry", "status": "ready" if (source / "telemetry.csv").is_file() else "missing", "detail": "telemetry.csv, camera-center GPS with explicit uncertainty"},
        {"label": "Flight metadata", "status": "ready" if (source / "flight_metadata.json").is_file() else "missing", "detail": "flight_metadata.json, clock and altitude conventions"},
        {"label": "Registered cameras", "status": "ready" if (work / "keyframes_poses.jsonl").is_file() else "missing", "detail": "Required for CPU alignment; produced by reconstruction"},
    ]
    state = {"schema_version": 1, "scene": scene, "status": "not_prepared", "readiness": readiness,
             "evaluation": evaluation.build_evaluation(), "alignment": None, "artifacts": [],
             "gpu_execution_enabled": False, "blockers": [], "commands": [
                 {"label": "Prepare inputs (CPU)", "argv": [".venv/Scripts/python.exe", "survey.py", "prepare", scene], "requires_gpu": False},
                 {"label": "Reconstruct dense points — approval required", "argv": [".venv/Scripts/python.exe", "survey.py", "reconstruct", scene, "--allow-gpu"], "requires_gpu": True},
             ]}
    prep_path = work / "survey" / "preparation.json"
    if prep_path.is_file():
        try:
            preparation = read_json(_safe_path(work, "survey/preparation.json"))
            if len(videos) != 1 or not _unchanged(root, preparation):
                raise ValueError("Inputs changed; prepare again. Previous results are stale.")
            state["status"] = "prepared"
            run, latest = _latest_run(work, preparation)
            names = [(prep_path, "Input provenance")]
            if latest:
                state["latest_run"] = {key: latest[key] for key in
                                       ("id", "status", "preparation_id", "secs", "error") if key in latest}
                names.append((run / "run.json", "Run timing/provenance; the official gate needs a "
                                                "decoder-verified duration and recorded hardware"))
                if latest["status"] != "complete":
                    state["blockers"].append("Latest reconstruction " + latest["status"].upper() + ": "
                                             + latest.get("error", "not complete") + "; no current dense evidence.")
            context = _evidence_context(work, preparation, selection=(run, latest))
            alignment = context["alignment"]
            if alignment:
                state["status"] = "aligned"
                state["alignment"] = {key: alignment.get(key) for key in ("scale", "matched_count", "inlier_count", "fit_rmse_m", "accuracy_validated", "coordinate_frame", "warnings")}
                readiness[3].update(status="ready", detail="Current registered cameras validated")
                names.append((context["georeference"], "ENU transform; GPS fit is not accuracy"))
                if latest and latest["status"] == "complete":
                    names.append((context["run"] / "dense_points.ply", "Current dense ENU cloud; not validated full-scene coverage"))
                    # _latest_run already verified every entry in record["files"],
                    # so these are provenance-backed rather than bare existence.
                    for name, kind in (("evidence/evidence_points.ply",
                                        "Sparse ENU cloud with view support, reprojection error and visibility"),
                                       ("evidence/evidence_summary.json",
                                        "Support and visibility summary; view counts are not measured accuracy")):
                        if name in (latest.get("files") or {}):
                            names.append((run / name, kind))
                elif alignment.get("sparse_sha256"):
                    names.append((work / "survey/sparse_points.ply", "Sparse diagnostic cloud, not dense coverage"))
                # Evidence products are listed only while the current alignment
                # carries a matching fingerprint for them, exactly as the sparse
                # cloud is gated on sparse_sha256. A re-align that lost the sparse
                # model leaves stale ENU clouds on disk for a coordinate frame
                # nothing current attests to, so they must not be offered.
                sources = alignment.get("sources", {}) if alignment.get("sparse_sha256") else {}
                for key, name, kind in (("evidence_points", EVIDENCE_POINTS,
                                         "ENU cloud with per-point view support and reprojection error"),
                                        ("evidence_summary", EVIDENCE_SUMMARY,
                                         "Support summary; view counts are not measured accuracy")):
                    if sources.get(key):
                        names.append((work / name, kind))
            try:
                result = _current_evaluation(work, preparation, context)
                if result:
                    state.update(status="evaluated", evaluation=result)
                    names.append((work / "survey/evaluation.json", "Independent evaluation report"))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                # Stale evaluation must not hide separately validated current
                # geometry, nor be presented as evidence for a new run.
                state["status"] = "invalid"
                state["blockers"].append(str(error))
            for path, kind in names:
                state["artifacts"].append({"name": path.name, "kind": kind,
                                           "url": "/work/" + quote(scene) + "/" + quote(path.relative_to(work).as_posix())})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            state.update(status="invalid", alignment=None, artifacts=[], evaluation=evaluation.build_evaluation())
            state["blockers"].append(str(error))
    state["blockers"] += [r["label"] + ": " + r["detail"] for r in readiness[:3] if r["status"] != "ready"]
    return state


def dense_commands(root, workspace, dense_dir):
    colmap = str(Path(root) / "tools" / "colmap" / "bin" / "colmap.exe")
    workspace, dense_dir = Path(workspace), Path(dense_dir)
    return [
        {"stage": "undistort", "requires_gpu": False, "argv": [colmap, "image_undistorter", "--image_path", str(workspace / "frames_train"), "--input_path", str(workspace / "colmap/sparse/txt"), "--output_path", str(dense_dir), "--output_type", "COLMAP", "--max_image_size", "1600"]},
        {"stage": "dense", "requires_gpu": True, "argv": [colmap, "patch_match_stereo", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"]},
        {"stage": "fusion", "requires_gpu": False, "argv": [colmap, "stereo_fusion", "--workspace_path", str(dense_dir), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(dense_dir / "fused.ply")]},
    ]


def _publish_run(work, run, record):
    write_json(_safe_path(run, "run.json"), record)
    write_json(_safe_path(work, "survey/latest_run.json"), {
        "id": record["id"], "status": record["status"], "preparation_id": record["preparation_id"],
        "report": _fingerprint(run / "run.json", run),
    })


def _depth_views(run, *, kind="geometric"):
    """Camera views carrying COLMAP stereo depth maps, for occlusion checking.

    Returns None when the dense stage or pycolmap is unavailable; the caller must
    then say occlusion was not checked rather than assume the cloud is clean.
    """
    try:
        import numpy as np
        import pycolmap
        import survey_visibility as visibility
    except ImportError:
        return None
    sparse, depths = _safe_path(run, "dense/sparse"), _safe_path(run, "dense/stereo/depth_maps")
    if not sparse.is_dir() or not depths.is_dir():
        return None
    views = []
    try:
        model = pycolmap.Reconstruction(str(sparse))
        for image in model.images.values():
            path = depths / (image.name + "." + kind + ".bin")
            if not path.is_file():
                continue
            viewmat = np.eye(4)
            viewmat[:3, :] = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
            views.append({"K": np.asarray(image.camera.calibration_matrix(), dtype=np.float64),
                          "viewmat": viewmat,
                          "depth": visibility.read_depth_map(path)})
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return None
    return views or None


def _preflight(root, video, expected_duration):
    """Verify tooling and source before any work exists: resolution, duration."""
    import cv2
    executable = root / "tools/colmap/bin/colmap.exe"
    if not executable.is_file():
        raise ValueError("COLMAP executable is missing: " + str(executable))
    cap = cv2.VideoCapture(str(video))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        cap.release()
    if not all(math.isfinite(x) and x > 0 for x in (fps, count, width, height)):
        raise ValueError("Cannot verify source video duration and resolution.")
    if min(width, height) < 1080 or max(width, height) < 1920:
        raise ValueError(f"Source video resolution {width:g}x{height:g} is below 1080p "
                         "(1920x1080); reconstruction refused.")
    duration = count / fps
    if not math.isfinite(duration) or abs(duration - expected_duration) > max(1.0, duration * .01):
        raise ValueError("Declared video duration disagrees with the decoded video metadata.")
    return duration, int(width), int(height)


def reconstruct_scene(root, scene, *, allow_gpu=False):
    if allow_gpu is not True:
        raise PermissionError("GPU reconstruction requires explicit approval and --allow-gpu. No work started.")
    started = time.perf_counter()
    import cv2
    import numpy as np
    from plyfile import PlyData
    evaluation, georef = _modules()
    root = Path(root).resolve()
    work, preparation = _prepared(root, scene)
    video = _source_files(root, scene)[0]
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run = _safe_path(work, "survey/runs/" + run_id)
    duration, source_width, source_height = _preflight(
        root, video, preparation["metadata"]["video_duration_s"])
    run.mkdir(parents=True, exist_ok=False)
    py = str(root / ".venv/Scripts/python.exe")
    commands = [
        {"stage": "keyframes", "argv": [py, str(root / "scripts/extract_keyframes.py"), "--work", str(run), "--video", str(video), "--target", "400", "--train-width", "1600"]},
        {"stage": "colmap", "argv": [py, str(root / "scripts/run_colmap.py"), str(run)]},
        {"stage": "poses", "argv": [py, str(root / "scripts/parse_colmap.py"), "--work", str(run)]},
        *dense_commands(root, run, run / "dense"),
    ]
    record = {"schema_version": 1, "id": run_id, "preparation_id": preparation["id"], "status": "running",
              "inputs": preparation["inputs"], "steps": [], "hardware": _hardware(),
              "benchmark_qualified": False, "duration_source": "decoder_metadata",
              "timing_scope": "From approval gate through setup, decode metadata, all uncached stages, export, hashes and manifest writes; final manifest/pointer commit excluded.",
              "warnings": ["Post-hoc GPS similarity alignment; no joint GPS bundle adjustment.",
                           "No dynamic masking or independently validated completeness yet.",
                           "Recorded host identity is CPU and RAM only; the GPU model is not recorded, "
                           "so this timing is not a benchmark on declared hardware and never an accuracy claim."],
              "versions": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__}, "commands": commands}
    try:
        _publish_run(work, run, record)
        record.update(video_duration_s=duration, video_resolution=[source_width, source_height])
        code_paths = [Path(__file__), Path(evaluation.__file__), Path(georef.__file__)]
        code_paths += [Path(command["argv"][1]) for command in commands[:3]]
        record["code_sha256"] = {str(path.resolve()): evaluation.file_fingerprint(path)["sha256"] for path in code_paths}
        record["steps"].append({"name": "setup", "status": "done", "secs": time.perf_counter() - started})
        for command in commands:
            t0 = time.perf_counter()
            print("[survey] " + command["stage"], flush=True)
            with (run / (command["stage"] + ".log")).open("w", encoding="utf-8") as log:
                proc = subprocess.run(command["argv"], cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=21600, check=False)
            record["steps"].append({"name": command["stage"], "status": "done" if proc.returncode == 0 else "failed", "secs": time.perf_counter() - t0})
            _publish_run(work, run, record)
            if proc.returncode:
                raise RuntimeError(f"{command['stage']} failed with exit {proc.returncode}; inspect {run / (command['stage'] + '.log')}")
        t0 = time.perf_counter()
        alignment = georef.align_camera_trajectory(_camera_rows(run / "keyframes_poses.jsonl"), preparation["telemetry"])
        alignment.update(preparation_id=preparation["id"], run_id=run_id,
                         pose_sha256=evaluation.file_fingerprint(run / "keyframes_poses.jsonl")["sha256"])
        write_json(run / "georeference.json", alignment)
        cloud = PlyData.read(str(run / "dense/fused.ply"))["vertex"].data
        points = np.column_stack([cloud[k] for k in ("x", "y", "z")])
        colors = np.column_stack([cloud[k] for k in ("red", "green", "blue")])
        _export_points(points, colors, alignment, run / "dense_points.ply")
        sparse_points = _safe_path(run, POINTS3D)
        if sparse_points.is_file():
            import survey_products as products
            views = _depth_views(run)
            evidence_cloud, evidence_report = products.evidence_for_sparse(
                sparse_points, alignment, _safe_path(run, "evidence"), views=views)
            record["evidence"] = read_json(evidence_report)
        record["files"] = {name: _fingerprint(_safe_path(run, name), run) for name in RUN_FILES
                           if _safe_path(run, name).is_file()}
        record["output"] = record["files"]["dense_points.ply"]
        if not _unchanged(root, preparation, full=True):
            raise ValueError("Survey inputs changed during reconstruction; prepare again.")
        record["steps"].append({"name": "georeferenced_export", "status": "done", "secs": time.perf_counter() - t0})
        record["point_count"] = len(points)
        # Include the large provenance/manifest write in elapsed time; the final
        # small commit cannot include its own duration and is explicitly excluded.
        _publish_run(work, run, record)
        record.update(status="complete", secs=time.perf_counter() - started)
        record["speed"] = _diagnostic_speed(record, duration, _required_stages(record))
    except Exception as error:
        record.update(status="failed", error=str(error), secs=time.perf_counter() - started)
        raise
    finally:
        _publish_run(work, run, record)
    return record
