"""Disk-backed workspace API. Reads never probe footage or start reconstruction."""
import hashlib
import json
import math
import mimetypes
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from email.parser import BytesHeaderParser
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
INPUT_EXTS = VIDEO_EXTS | {".gpx", ".srt", ".csv", ".jsonl", ".json"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_EXTS = {".ply", ".splat", ".glb", ".gltf", ".bin", ".obj", ".mtl", ".las", ".laz"}
GENERATED_EXTS = MODEL_EXTS | IMAGE_EXTS | {".json", ".f32", ".u8", ".rgb", ".npz"}
ROOT_OUTPUTS = {"splat.ply", "frame.json", "diagnostics.json", "report.json", "keyframes.jsonl", "keyframes_poses.jsonl", "eval_pairs.json", "video_meta.json"}
SURVEY_OUTPUTS = {"preparation.json", "georeference.json", "evaluation.json", "sparse_points.ply", "latest_run.json", "capture_plan.json"}
WORKFLOWS = {"general", "inspection", "survey", "response", "heritage"}
CAPTURES = {"unknown", "drone", "handheld", "phone"}
MAX_JSON = 2 << 20
MAX_UPLOAD = 2 << 30
MAX_FILES = 128
MAX_POINTS = 1000
PROXY_WARNING = "Measurements use the local model/physics-proxy in viewer Y-up coordinates, not independently verified survey accuracy. Area is projected onto XZ."


class Error(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def safe_path(root, relative):
    """Reject aliases as well as traversal, including Windows junctions and ADS."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise Error(400, "Invalid relative path.")
    parts = relative.split("/")
    for part in parts:
        if (not part or part in {".", ".."} or part.endswith((".", " "))
                or any(ord(c) < 32 for c in part) or re.search(r'[:<>"|?*]', part)
                or re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)):
            raise Error(400, "Invalid relative path.")
    root = Path(root).resolve()
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise Error(400, "Linked workspace paths are not allowed.")
    if not path.resolve().is_relative_to(root):
        raise Error(400, "Path leaves the workspace.")
    return path


def scene_paths(root, scene):
    if not isinstance(scene, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}|[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}", scene) or ".." in scene:
        raise Error(400, "Scene must be a safe 1–64 character name.")
    work = safe_path(root, "work/" + scene)
    source = safe_path(root, "videos/" + scene)
    if "." in scene and not (work.is_dir() or source.is_dir()):
        raise Error(400, "New scene names use letters, digits, underscores or hyphens.")
    return source, work


def runtime_url(root, path, download=False):
    relative = Path(path).relative_to(root).as_posix()
    return "/runtime/" + "/".join(quote(p, safe="") for p in relative.split("/")) + ("?download=1" if download else "")


def read_json(root, path, default=None):
    path = safe_path(root, Path(path).relative_to(root).as_posix())
    if not path.is_file():
        return default
    if path.stat().st_size > 16 << 20:
        return default
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=lambda _: None)
    except (ValueError, UnicodeError):
        return default


def write_json(root, path, data):
    path = safe_path(root, path.relative_to(root).as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def files_under(root, directory):
    if not directory.is_dir():
        return
    safe_path(root, directory.relative_to(root).as_posix())
    for parent, directories, names in os.walk(directory, followlinks=False):
        parent = safe_path(root, Path(parent).relative_to(root).as_posix())
        directories[:] = [d for d in sorted(directories)
                          if not (parent / d).is_symlink()
                          and not getattr(parent / d, "is_junction", lambda: False)()]
        for name in sorted(names):
            path = parent / name
            if not path.is_symlink() and path.is_file():
                yield path


def videos_for(root, scene):
    source, _ = scene_paths(root, scene)
    videos = [p for p in files_under(root, source) if p.suffix.lower() in VIDEO_EXTS]
    # Existing pipeline also accepts videos/<scene>.mp4. Do not move old inputs.
    video_root = safe_path(root, "videos")
    if video_root.is_dir():
        for path in sorted(video_root.iterdir()):
            if path.stem == scene and path.suffix.lower() in VIDEO_EXTS:
                try:
                    path = safe_path(root, path.relative_to(root).as_posix())
                    if path.is_file():
                        videos.append(path)
                except Error:
                    continue
    return videos


def file_allowed(relative):
    parts = relative.split("/")
    ext = Path(relative).suffix.lower()
    if relative in {"viewer/pc.html", "viewer/pc.js"} or re.fullmatch(r"viewer/workspace[A-Za-z0-9_.-]*\.js", relative):
        return True
    if relative.startswith("viewer/pc/"):
        return ext in MODEL_EXTS | {".js", ".mjs", ".wasm"}
    if relative.startswith("viewer/assets/"):
        return ext in MODEL_EXTS | IMAGE_EXTS
    if parts[0] == "videos":
        return (len(parts) >= 3 and ext in INPUT_EXTS) or (len(parts) == 2 and ext in VIDEO_EXTS)
    if len(parts) < 3 or parts[0] != "work":
        return False
    rest = parts[2:]
    if len(rest) == 1:
        return rest[0] in ROOT_OUTPUTS
    if rest[0] in {"viewer_assets", "pc"}:
        return ext in GENERATED_EXTS
    if rest[0] in {"frames_train", "frames_full", "frames_undist"}:
        return ext in IMAGE_EXTS
    if rest[0] == "survey":
        tail = rest[1:]
        if len(tail) == 1:
            return tail[0] in SURVEY_OUTPUTS
        if tail[0] == "evidence":
            return len(tail) == 2 and tail[1] in {"evidence_points.ply", "evidence_summary.json"}
        if tail[0] == "runs" and len(tail) >= 3 and re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", tail[1]):
            generated = tail[2:]
            if len(generated) == 1:
                return generated[0] in {"run.json", "georeference.json", "dense_points.ply", "keyframes_poses.jsonl"}
            if generated[0] == "evidence":
                return len(generated) == 2 and generated[1] in {"evidence_points.ply", "evidence_summary.json"}
            if generated[0] == "products":
                return ext in MODEL_EXTS | IMAGE_EXTS | {".json", ".csv", ".tif", ".tiff", ".prj"}
    return False


def serve_file(handler, root, query):
    relative = query.get("path", [""])[0]
    path = safe_path(root, relative)
    if not file_allowed(relative):
        raise Error(403, "This file is not a workspace media or deliverable.")
    if not path.is_file():
        raise Error(404, "File not found.")
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        start, end, status = 0, size - 1, 200
        value = handler.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
            valid = bool(match and (match[1] or match[2]) and size)
            if valid:
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), size - 1) if match[2] else size - 1
                    valid = start <= end and start < size
                else:
                    count = int(match[2])
                    valid = count > 0
                    start = max(0, size - count)
            if not valid:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.send_header("Content-Length", "0")
                handler.send_header("Accept-Ranges", "bytes")
                handler.end_headers()
                return
            status = 206
        handler.send_response(status)
        content_type = handler.extensions_map.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        handler.send_header("Content-Type", content_type)
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(max(0, end - start + 1)))
        if status == 206:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if query.get("download") == ["1"]:
            handler.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(path.name, safe=""))
        handler.end_headers()
        if handler.command == "HEAD":
            return
        stream.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def finite(value):
    # JSON integers are arbitrary precision; converting a huge one to float
    # would raise before the coordinate/anchor bounds can refuse it.
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and -1e308 <= value <= 1e308 and math.isfinite(value))


def rows_jsonl(root, path):
    path = safe_path(root, path.relative_to(root).as_posix())
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream):
                if index >= 100000:
                    break
                try:
                    row = json.loads(line, parse_constant=lambda _: None)
                    if isinstance(row, dict):
                        yield row
                except ValueError:
                    continue


def frames_for(root, work):
    cameras = read_json(root, work / "viewer_assets/cameras.json", [])
    camera_rows = {r["name"]: {**r, "camera_index": index} for index, r in enumerate(cameras) if isinstance(r, dict) and isinstance(r.get("name"), str)} if isinstance(cameras, list) else {}
    timestamps = {r.get("file"): r.get("t_sec") for r in rows_jsonl(root, work / "keyframes.jsonl")}
    frames = {}
    for folder in ("frames_train", "frames_full", "frames_undist"):
        directory = safe_path(root, (work / folder).relative_to(root).as_posix())
        for path in files_under(root, directory):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            name = path.relative_to(directory).as_posix()
            if name in frames:
                continue
            camera = camera_rows.get(name, {})
            timestamp = camera.get("t_sec", timestamps.get(name))
            row = {"name": name, "url": runtime_url(root, path), "t_sec": timestamp if finite(timestamp) else None,
                   "camera_index": camera.get("camera_index")}
            pos = camera.get("pos")
            if isinstance(pos, list) and len(pos) == 3 and all(finite(n) for n in pos):
                row["pos"] = pos
            frames[name] = row
        # These directories are alternate resolutions/undistortions of the
        # same extraction. Undistortion flattens and prefixes filenames, so
        # merging all three directories would invent additional observations.
        if frames:
            break
    return list(frames.values()), len(camera_rows)


def model_revision(root, work):
    digest = hashlib.sha256()
    candidates = [work / "frame.json"]
    for folder in ("viewer_assets", "pc"):
        for path in files_under(root, safe_path(root, (work / folder).relative_to(root).as_posix())):
            if path.suffix.lower() in MODEL_EXTS | {".f32", ".u8", ".npz"} or path.name in {"collision.json", "objects.json"}:
                candidates.append(path)
    for path in sorted(candidates):
        path = safe_path(root, path.relative_to(root).as_posix())
        digest.update(path.relative_to(work).as_posix().encode())
        if path.is_file():
            stat = path.stat()
            # Avoid rereading multi-GB geometry on every poll. Frame metadata is
            # content-hashed; geometry identity includes size, mtime and ctime.
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode())
            if path.suffix == ".json" and stat.st_size < MAX_JSON:
                digest.update(path.read_bytes())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def public_data(data, root):
    """Diagnostics and child logs may contain absolute tool paths: do not expose them."""
    if isinstance(data, dict):
        return {k: public_data(v, root) for k, v in data.items()}
    if isinstance(data, list):
        return [public_data(v, root) for v in data]
    if isinstance(data, str):
        data = data.replace(str(root), "[workspace]").replace(root.as_posix(), "[workspace]")
        return re.sub(r"[A-Za-z]:[\\/][^\r\n\"']*", "[local path]", data)
    if isinstance(data, float) and not math.isfinite(data):
        return None
    return data


def job_snapshot(server, root):
    with server.process_lock:
        return public_data({**server.active_job_info, "logs": list(server.active_job_info.get("logs", []))}, root)


def summary(root, scene, server, detail=False):
    source, work = scene_paths(root, scene)
    videos = videos_for(root, scene)
    if not work.is_dir() and not source.is_dir() and not videos:
        raise Error(404, "Project not found.")
    metadata = read_json(root, work / "project.json", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    frame = read_json(root, work / "frame.json", {})
    frame = frame if isinstance(frame, dict) else {}
    scale_source = frame.get("scale_source") if isinstance(frame.get("scale_source"), str) else "No scale reference"
    scale_value = frame.get("scale_m_per_unit")
    scale_status = "relative"
    if finite(scale_value) and scale_value > 0:
        scale_status = "metric" if scale_source == "AR pose-prior metric path" else "estimated"
    frames, camera_count = frames_for(root, work)
    poses = {r.get("file") for r in rows_jsonl(root, work / "keyframes_poses.jsonl") if isinstance(r.get("file"), str)}
    latest = {}
    times = [p.stat().st_mtime for p in (source, work, work / "project.json") if p.exists()]
    logs_dir = safe_path(root, (work / "logs").relative_to(root).as_posix())
    for log in server._log_files(logs_dir):
        log = safe_path(root, log.relative_to(root).as_posix())
        row = server._step_row(log, time.time())
        modified = log.stat().st_mtime_ns
        if row["name"] not in latest or modified >= latest[row["name"]][0]:
            latest[row["name"]] = (modified, row)
        times.append(log.stat().st_mtime)
    wrappers = {name: pair for name, pair in latest.items() if name.startswith("workspace_")}
    if wrappers:
        newest = max(wrappers, key=lambda name: wrappers[name][0])
        latest = {name: pair for name, pair in latest.items() if name not in wrappers or name == newest}
    steps = sorted((pair[1] for pair in latest.values()), key=lambda r: r["i"])
    model = safe_path(root, (work / "viewer_assets/scene.ply").relative_to(root).as_posix())
    trained = safe_path(root, (work / "splat.ply").relative_to(root).as_posix()).is_file()
    viewable = model.is_file() and model.stat().st_size > 0
    times += [p.stat().st_mtime for p in videos + ([model] if viewable else [])]
    job = job_snapshot(server, root)
    processing = job.get("scene") == scene and job.get("status") == "running"
    failed = any(s["status"] in {"failed", "interrupted"} for s in steps)
    if job.get("scene") == scene and str(job.get("status", "")).startswith(("failed", "error", "cancelled")):
        failed = True
    status = "processing" if processing or any(s["status"] == "running" for s in steps) else "failed" if failed else "ready" if viewable else "uploaded" if videos else "empty"
    # Conservative defaults: a scene is local/unverified until validated survey
    # evidence (detail path only) proves otherwise. The fast list never imports
    # the survey subsystem, so it always reports these defaults.
    georeference = {"status": "local", "crs": None}
    accuracy = {"status": "unverified", "rmse_m": None}
    project = {"id": scene, "name": metadata.get("name") or scene,
               "workflow": metadata.get("workflow") if metadata.get("workflow") in WORKFLOWS else "general",
               "capture": metadata.get("capture") if metadata.get("capture") in CAPTURES else "unknown",
               "status": status, "updated": datetime.fromtimestamp(max(times or [0]), timezone.utc).isoformat(),
               "thumbnail_url": frames[len(frames) // 2]["url"] if frames else None,
               "video_count": len(videos), "frame_count": len(frames), "registered_count": len(poses) or camera_count,
               "viewable": viewable, "trained": trained,
               "scale": {"status": scale_status, "source": scale_source, "unit": "units" if scale_status == "relative" else "m"},
               "georeference": georeference,
               "accuracy": accuracy}
    if not detail:
        return project
    warnings = [PROXY_WARNING]
    if scale_status != "metric":
        warnings.append("Scale is relative." if scale_status == "relative" else "Scale is estimated from an assumed height/speed; it is not independently measured accuracy.")
    artifacts = []
    for path in sorted(work.glob("*")) if work.is_dir() else []:
        if path.name in ROOT_OUTPUTS and path.is_file():
            safe_path(root, path.relative_to(root).as_posix())
            artifacts.append({"name": path.name, "url": runtime_url(root, path, True), "bytes": path.stat().st_size,
                              "kind": "model" if path.suffix in MODEL_EXTS else "report"})
    for folder in ("viewer_assets", "pc"):
        for path in files_under(root, safe_path(root, (work / folder).relative_to(root).as_posix())):
            if path.suffix.lower() in MODEL_EXTS:
                artifacts.append({"name": folder + "/" + path.name, "url": runtime_url(root, path, True), "bytes": path.stat().st_size, "kind": "model"})
    # Only consult survey evidence when it exists. An ordinary phone clip never
    # needs telemetry, preparation, NumPy imports or GPU diagnostics just to list.
    survey_summary = None
    if safe_path(root, (work / "survey/preparation.json").relative_to(root).as_posix()).is_file():
        try:
            import survey_workflow
            survey = survey_workflow.scene_status(root, scene)
            # scene_status is the only authority that validates staleness and
            # re-hashes evidence, so georeference and accuracy come from it and
            # never from a raw evaluation.json that a poll could render green.
            if survey.get("crs"):
                georeference = {"status": "georeferenced", "crs": survey["crs"],
                                "vertical_datum": survey.get("vertical_datum"),
                                "position_reference": survey.get("position_reference"),
                                "viewer_frame": "local Y-up; the interactive splat is not itself georeferenced"}
                warnings.append("Georeferenced survey deliverables use " + str(survey["crs"])
                                + "; the interactive viewer stays in local coordinates.")
            accuracy_criterion = next((c for c in (survey.get("evaluation") or {}).get("criteria", [])
                                       if c.get("id") == "accuracy"), None)
            metrics = accuracy_criterion.get("metrics") if accuracy_criterion and accuracy_criterion.get("status") == "measured" else None
            if metrics and finite(metrics.get("rmse_3d_m")):
                accuracy = {"status": "verified", "rmse_m": metrics["rmse_3d_m"],
                            "horizontal_rmse_m": metrics.get("horizontal_rmse_m"),
                            "vertical_rmse_m": metrics.get("vertical_rmse_m"),
                            "checkpoints": metrics.get("count"),
                            "basis": "independent surveyed checkpoints"}
            if survey.get("status") == "invalid":
                warnings.append("Survey evidence is stale or invalid; prepare/align/evaluate it separately.")
            survey_summary = {"status": survey.get("status"), "blockers": survey.get("blockers", []),
                              "fit_rmse_m": (survey.get("alignment") or {}).get("fit_rmse_m")}
            for artifact in survey.get("artifacts", []):
                relative = unquote(artifact["url"]).lstrip("/")
                path = safe_path(root, relative)
                if file_allowed(relative) and path.is_file():
                    artifacts.append({"name": artifact["name"], "url": runtime_url(root, path, True), "bytes": path.stat().st_size, "kind": artifact["kind"]})
        except (ImportError, ValueError, OSError, KeyError, TypeError, AttributeError):
            warnings.append("Survey evidence could not be validated; no georeference or accuracy claim is made.")
    revision = model_revision(root, work)
    measurements = read_json(root, work / "measurements.json", [])
    measurements = measurements if isinstance(measurements, list) else []
    measurements = [{**m, "stale": not viewable or m.get("model_revision") != revision} for m in measurements if isinstance(m, dict)]
    placements = read_json(root, work / "placements.json", [])
    placements = placements if isinstance(placements, list) else []
    placements = [{**p, "stale": not viewable or p.get("model_revision") != revision} for p in placements if isinstance(p, dict)]
    diagnostics = read_json(root, work / "diagnostics.json")
    sem = read_json(root, work / "viewer_assets/semantics.json")
    semantics = None
    if isinstance(sem, dict) and isinstance(sem.get("counts"), dict) and isinstance(sem.get("classes"), list):
        semantics = {"classes": sem["classes"], "counts": sem["counts"]}
        try:
            import workspace_measure
            semantics["summary"] = workspace_measure.class_summary(sem)
        except Exception:
            semantics["summary"] = None
    project.update(videos=[{"name": p.name, "url": runtime_url(root, p), "bytes": p.stat().st_size} for p in videos],
                   artifacts=artifacts, frames=frames, steps=steps, measurements=measurements, placements=placements,
                   furniture=furniture_library(), model_revision=revision,
                   diagnostics=public_data(diagnostics, root) if isinstance(diagnostics, dict) else None,
                   georeference=georeference, accuracy=accuracy, survey=survey_summary, semantics=semantics,
                   notes=metadata.get("notes", ""), viewer_url=("/runtime/viewer/pc.html?asset=/runtime/work/" + quote(scene, safe="") + "/viewer_assets&embed=1") if viewable else None,
                   warnings=public_data(warnings, root), job=job)
    return project


def projects(root, server):
    names = set()
    for folder in ("work", "videos"):
        directory = safe_path(root, folder)
        if directory.is_dir():
            for path in directory.iterdir():
                if path.is_dir():
                    names.add(path.name)
                elif folder == "videos" and path.suffix.lower() in VIDEO_EXTS:
                    names.add(path.stem)
    result = []
    for name in sorted(names):
        try:
            result.append(summary(root, name, server))
        except (Error, OSError):
            continue
    from pipeline import PRESETS, QUALITY
    return {"projects": result, "job": job_snapshot(server, root), "presets": PRESETS, "qualities": QUALITY}


def validate_text(value, label, limit, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()) or "\x00" in value:
        raise Error(400, f"{label} must be {'at most' if empty else '1–'}{limit} characters.")
    return value if empty else value.strip()


def ensure_not_running(server, scene):
    # IDs differing only in case address the same Windows directory. Treat them
    # conservatively as the same job on every platform, just like upload names.
    if str(server.active_job_info.get("scene", "")).casefold() == scene.casefold() and server.job_busy_locked():
        raise Error(409, "Wait for this scene's reconstruction to stop before changing it.")


def save_project(root, data, server):
    scene = data.get("scene")
    _, work = scene_paths(root, scene)
    updates = {}
    if "name" in data:
        updates["name"] = validate_text(data["name"], "Name", 200)
    if "notes" in data:
        updates["notes"] = validate_text(data["notes"], "Notes", 20000, empty=True)
    for key, allowed in (("workflow", WORKFLOWS), ("capture", CAPTURES)):
        if key in data:
            if not isinstance(data[key], str) or data[key] not in allowed:
                raise Error(400, "Invalid " + key + ".")
            updates[key] = data[key]
    with server.process_lock:
        ensure_not_running(server, scene)
        saved = read_json(root, work / "project.json", {})
        if not isinstance(saved, dict):
            raise Error(409, "Project metadata is not a JSON object.")
        saved.update(updates)
        write_json(root, work / "project.json", saved)
    return summary(root, scene, server, True)


def measurement_value(data, unit, cloud=None):
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in {"point", "distance", "height", "area", "volume"}:
        raise Error(400, "Unknown measurement kind.")
    label = validate_text(data.get("label"), "Label", 200)
    points = data.get("points")
    if (not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS
            or any(not isinstance(p, list) or len(p) != 3 or any(not finite(n) or abs(n) > 1e7 for n in p) for p in points)):
        raise Error(400, "Supply at most 1000 finite XYZ points within the model coordinate bounds.")
    if (kind == "point" and len(points) != 1 or kind == "height" and len(points) != 2
            or kind in {"distance", "area", "volume"} and len(points) < (2 if kind == "distance" else 3)):
        raise Error(400, "Wrong number of points for this measurement kind.")
    if kind == "volume" and cloud is None:
        raise Error(409, "Volume needs a point cloud to integrate against; run reconstruction first.")
    record = {"id": uuid.uuid4().hex, "kind": kind, "label": label, "points": points,
              "unit": unit, "created_at": datetime.now(timezone.utc).isoformat(),
              "geometry": "viewer-pick", "model_revision": data.get("model_revision"), "stale": False}
    # With a point cloud behind the scene, measure with the real engine: clicks are
    # snapped to measured geometry and carry an uncertainty budget + validity flag.
    # Without one, fall back to the raw viewer-pick arithmetic (still honest, just
    # unquantified). "point" is an annotation and never measured.
    if cloud is not None and kind in {"distance", "height", "area", "volume"}:
        import workspace_measure as wm
        engine_kind = "distance" if kind == "height" else kind
        engine = wm.measure(engine_kind, cloud, points[:2] if kind == "height" else points)
        record["engine"] = True
        record["uncertainty"] = engine.get("uncertainty")
        record["valid"] = engine.get("valid")
        record["reason"] = engine.get("reason")
        record["snapped"] = engine.get("snapped")
        if kind == "height":
            comp = engine.get("components") or {}
            record["value"] = comp.get("vertical_m")
            record["valid"] = record["value"] is not None
            record["support"] = engine.get("valid")
        else:
            record["value"] = engine.get("value")
            if kind in {"area", "volume"} and record["value"] is not None:
                record["unit"] = "m²" if kind == "area" else "m³"
        return record
    value = None
    if kind == "distance":
        value = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    elif kind == "height":
        value = abs(points[0][1] - points[1][1])
    elif kind == "area":
        value = abs(sum(a[0] * b[2] - b[0] * a[2] for a, b in zip(points, points[1:] + points[:1]))) / 2
        unit += "²"
    record["engine"] = False
    record["valid"] = True
    record["value"] = value
    record["unit"] = unit
    return record


def save_measurement(root, data, server, delete=False):
    scene = data.get("scene")
    _, work = scene_paths(root, scene)
    detail = summary(root, scene, server, True)
    cloud = None
    if not delete:
        try:
            import workspace_measure
            cloud = workspace_measure.load_cloud(work)
        except (ImportError, ValueError, OSError):
            cloud = None
    entry = None if delete else measurement_value(data, detail["scale"]["unit"], cloud)
    with server.process_lock:
        ensure_not_running(server, scene)
        path = work / "measurements.json"
        rows = read_json(root, path, [])
        if not isinstance(rows, list):
            raise Error(409, "Stored measurements are not a list.")
        if delete:
            identity = data.get("id")
            if not isinstance(identity, str) or not identity or len(identity) > 100:
                raise Error(400, "Supply a measurement id.")
            filtered = [m for m in rows if m.get("id") != identity]
            if len(filtered) == len(rows):
                raise Error(404, "Measurement not found.")
            rows = filtered
        else:
            if not detail["viewable"]:
                raise Error(409, "A viewable model is required for measurements.")
            if data.get("model_revision") != model_revision(root, work):
                raise Error(409, "The model changed. Reload before measuring.")
            if len(rows) >= 10000:
                raise Error(413, "This model already has 10000 measurements.")
            rows.append(entry)
        write_json(root, path, rows)
    return summary(root, scene, server, True)


def furniture_library():
    """The catalogue the Place tab offers, or [] if the backend lacks SciPy."""
    try:
        import workspace_place
        return workspace_place.library()
    except Exception:
        return []


def placement_record(work, data, revision):
    """Validate a drop and let the placement engine snap + fit-check it."""
    import workspace_place
    item = data.get("item")
    if not isinstance(item, str):
        raise Error(400, "A furniture item is required.")
    try:
        workspace_place.item_spec(item)
    except ValueError as error:
        raise Error(400, str(error))
    point = data.get("point")
    if (not isinstance(point, list) or len(point) != 3
            or any(not isinstance(n, (int, float)) or not math.isfinite(n) or abs(n) > 1e7 for n in point)):
        raise Error(400, "Supply a finite XYZ drop point on the floor.")
    yaw = data.get("yaw_deg", 0.0)
    scale = data.get("scale", 1.0)
    if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise Error(400, "Rotation must be a finite angle.")
    if not isinstance(scale, (int, float)) or not (0.05 <= scale <= 20):
        raise Error(400, "Scale must be between 0.05 and 20.")
    label = data.get("label")
    try:
        rec = workspace_place.make_placement(work, item, point[0], point[2],
                                             yaw_deg=float(yaw), scale=float(scale),
                                             label=label if isinstance(label, str) else None)
    except (ValueError, OSError) as error:
        raise Error(409, str(error))
    rec["model_revision"] = revision
    rec["stale"] = False
    return rec


def save_placement(root, data, server, mode="create"):
    scene = data.get("scene")
    _, work = scene_paths(root, scene)
    detail = summary(root, scene, server, True)
    revision = detail["model_revision"]
    with server.process_lock:
        ensure_not_running(server, scene)
        path = work / "placements.json"
        rows = read_json(root, path, [])
        if not isinstance(rows, list):
            raise Error(409, "Stored placements are not a list.")
        if mode == "delete":
            identity = data.get("id")
            if not isinstance(identity, str) or not identity or len(identity) > 100:
                raise Error(400, "Supply a placement id.")
            filtered = [p for p in rows if p.get("id") != identity]
            if len(filtered) == len(rows):
                raise Error(404, "Placement not found.")
            rows = filtered
        else:
            if not detail["viewable"]:
                raise Error(409, "A viewable model is required to place furniture.")
            if data.get("model_revision") != revision:
                raise Error(409, "The model changed. Reload before placing.")
            rec = placement_record(work, data, revision)
            if mode == "update":
                identity = data.get("id")
                if not isinstance(identity, str) or not identity:
                    raise Error(400, "Supply a placement id.")
                rec["id"] = identity
                for i, p in enumerate(rows):
                    if p.get("id") == identity:
                        rows[i] = rec
                        break
                else:
                    raise Error(404, "Placement not found.")
            else:
                if len(rows) >= 2000:
                    raise Error(413, "This model already has 2000 placed items.")
                rows.append(rec)
        write_json(root, path, rows)
    return summary(root, scene, server, True)


def run_job(root, data, server):
    if set(data) - {"scene", "preset", "quality", "anchor", "action", "engine", "dense_profile"}:
        raise Error(400, "Unsupported launch fields; arbitrary arguments are not allowed.")
    scene = data.get("scene")
    scene_paths(root, scene)
    from pipeline import PRESETS, QUALITY
    preset, quality = data.get("preset"), data.get("quality")
    action, engine, profile = data.get("action", "run"), data.get("engine", "pipeline"), data.get("dense_profile", "survey")
    for value, choices in ((preset, PRESETS), (quality, QUALITY), (action, {"run", "scan"}), (engine, {"pipeline", "survey"}), (profile, {"survey", "fast", "budget"})):
        if not isinstance(value, str) or value not in choices:
            raise Error(400, "Invalid preset, quality, action, engine or dense profile.")
    extra = []
    if "anchor" in data:
        anchor = data["anchor"]
        if not isinstance(anchor, dict) or set(anchor) != {"kind", "value"} or anchor.get("kind") not in ("height", "speed") or not finite(anchor.get("value")) or not 0 < anchor["value"] <= 10000:
            raise Error(400, "Anchor requires height or speed and a finite positive value at most 10000.")
        extra = ["--" + anchor["kind"] + "-anchor", str(anchor["value"])]
    if engine == "survey" and (action != "run" or extra):
        raise Error(400, "Survey reconstruction uses prepared telemetry, not scan or scale anchors.")
    videos = videos_for(root, scene)
    if not videos or any(path.stat().st_size == 0 for path in videos):
        raise Error(400, "Upload at least one nonempty video; empty video files cannot be processed.")
    if engine == "pipeline":
        extra += ["--video", *map(str, videos)]
    with server.process_lock:
        if server.job_busy_locked():
            raise Error(409, "A reconstruction job is already active.")
        if engine == "survey":
            import survey_workflow
            status = survey_workflow.scene_status(root, scene)
            if status.get("status") not in {"prepared", "aligned", "evaluated"} or any(r.get("status") != "ready" for r in status.get("readiness", [])[:3]):
                raise Error(409, "Survey inputs must be prepared and current before reconstruction.")
        token = server.reserve_job_locked(scene, preset, quality, action, engine)
    server.spawn_pipeline_job(scene, preset, quality, extra, root=root, action=action, engine=engine, dense_profile=profile, reservation=token)
    return {"status": "started", "scene": scene}


def body_length(handler, limit):
    if handler.headers.get("Transfer-Encoding"):
        raise Error(400, "Content-Length is required; chunked uploads are not accepted.")
    values = handler.headers.get_all("Content-Length", [])
    if len(values) != 1 or not re.fullmatch(r"[0-9]{1,12}", values[0]):
        raise Error(400, "Supply one valid Content-Length.")
    size = int(values[0])
    if size > limit:
        raise Error(413, "Upload exceeds the 2 GiB limit." if limit == MAX_UPLOAD else "JSON exceeds the 2 MiB limit.")
    return size


def json_body(handler):
    size = body_length(handler, MAX_JSON)
    raw = handler.rfile.read(size)
    if handler.headers.get_content_type() != "application/json":
        raise Error(400, "Expected application/json.")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise Error(400, "Supply a valid JSON object.")
    if not isinstance(data, dict):
        raise Error(400, "Supply a JSON object.")
    return data


class MultipartReader:
    """Bounded delimiter scanner; only complete MIME delimiter lines are framing."""
    def __init__(self, source, boundary):
        self.source = source
        self.boundary = b"\r\n--" + boundary
        self.buffer = b""
        self.eof = False

    def fill(self):
        chunk = self.source.read(1 << 16)
        self.buffer += chunk
        self.eof = not chunk

    def line(self, limit=16384):
        while b"\r\n" not in self.buffer and not self.eof and len(self.buffer) <= limit:
            self.fill()
        end = self.buffer.find(b"\r\n")
        if end < 0 or end > limit:
            raise Error(400, "Malformed multipart headers.")
        line, self.buffer = self.buffer[:end], self.buffer[end + 2:]
        return line

    def copy_part(self, output):
        keep = len(self.boundary) + 4
        while True:
            index = self.buffer.find(self.boundary)
            if index >= 0:
                end = index + len(self.boundary)
                while len(self.buffer) < end + 4 and not self.eof:
                    self.fill()
                suffix = self.buffer[end:end + 4]
                if suffix.startswith(b"\r\n") or suffix == b"--\r\n" or (self.eof and suffix == b"--"):
                    output.write(self.buffer[:index])
                    final = suffix.startswith(b"--")
                    self.buffer = self.buffer[end + (4 if final and suffix.endswith(b"\r\n") else 2):]
                    return final
                # Boundary-like bytes inside video data are not a delimiter.
                output.write(self.buffer[:index + 2])
                self.buffer = self.buffer[index + 2:]
            elif self.eof:
                raise Error(400, "Multipart upload ended before its closing boundary.")
            else:
                count = max(0, len(self.buffer) - keep)
                output.write(self.buffer[:count])
                self.buffer = self.buffer[count:]
                self.fill()


def upload(handler, root, server):
    size = body_length(handler, MAX_UPLOAD)
    content_type = handler.headers.get_content_type()
    boundary = handler.headers.get_param("boundary")
    if content_type != "multipart/form-data" or not isinstance(boundary, str) or not re.fullmatch(r"[A-Za-z0-9'()+_,./:=?-]{1,70}", boundary):
        # Small malformed requests are consumed for reliable Windows error replies.
        if size <= MAX_JSON:
            handler.rfile.read(size)
        raise Error(400, "Expected multipart/form-data with a valid boundary.")
    with tempfile.TemporaryDirectory(prefix="workspace-upload-") as temporary:
        temporary = Path(temporary)
        # Spool once so validation errors never leave a browser still sending a
        # large request. Neither the video nor its MIME body is held in memory.
        with (temporary / "body").open("w+b") as spool:
            remaining = size
            while remaining:
                chunk = handler.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    raise Error(400, "Upload body was truncated.")
                spool.write(chunk)
                remaining -= len(chunk)
            spool.seek(0)
            parser = MultipartReader(spool, boundary.encode("ascii"))
            if parser.line() != b"--" + boundary.encode("ascii"):
                raise Error(400, "Malformed multipart opening boundary.")
            files, scene, seen = [], None, set()
            for index in range(MAX_FILES + 2):
                header_lines = []
                while True:
                    line = parser.line()
                    if not line:
                        break
                    header_lines.append(line)
                    if sum(map(len, header_lines)) > 16384:
                        raise Error(400, "Multipart headers are too large.")
                headers = BytesHeaderParser().parsebytes(b"\r\n".join(header_lines) + b"\r\n\r\n")
                if headers.get_content_disposition() != "form-data":
                    raise Error(400, "Expected form-data parts.")
                field = headers.get_param("name", header="content-disposition")
                name = headers.get_filename()
                part = temporary / str(index)
                with part.open("wb") as stream:
                    final = parser.copy_part(stream)
                if name is None:
                    if field != "scene" or scene is not None or part.stat().st_size > 64:
                        raise Error(400, "Supply exactly one scene field.")
                    scene = part.read_text(encoding="utf-8")
                    scene_paths(root, scene)
                else:
                    if field not in {"files", "files[]", "file"} or not name or len(name) > 200 or "/" in name or "\\" in name or Path(name).suffix.lower() not in INPUT_EXTS:
                        raise Error(400, "Unsupported or unsafe upload filename.")
                    safe_path(root, "videos/_upload/" + name)
                    if name.casefold() in seen:
                        raise Error(409, "Duplicate upload filenames are not allowed.")
                    seen.add(name.casefold())
                    files.append((name, part))
                if final:
                    break
            else:
                raise Error(413, "Too many uploaded files; maximum 128.")
        if scene is None or not files or len(files) > MAX_FILES:
            raise Error(400, "Supply a scene and 1–128 files.")
        source, _ = scene_paths(root, scene)
        with server.process_lock:
            ensure_not_running(server, scene)
            existing = {p.name.casefold() for p in source.iterdir()} if source.is_dir() else set()
            if existing.intersection(seen):
                raise Error(409, "An uploaded filename already exists. Rename it or choose another scene.")
            targets = [(safe_path(root, "videos/" + scene + "/" + name), part) for name, part in files]
            source.mkdir(parents=True, exist_ok=True)
            created = []
            try:
                for target, part in targets:
                    with target.open("xb") as out, part.open("rb") as inp:
                        created.append(target)
                        while chunk := inp.read(1 << 20):
                            out.write(chunk)
            except Exception:
                for target in created:
                    target.unlink(missing_ok=True)
                raise
    return {"status": "success", "scene": scene, "saved_files": [name for name, _ in files]}


def handle(handler, server):
    root = Path(handler.directory).resolve()
    parsed = urlparse(handler.path)
    route = parsed.path.removeprefix("/api/workspace/")
    query = parse_qs(parsed.query, keep_blank_values=True)
    try:
        if handler.command in {"GET", "HEAD"}:
            if route == "file":
                serve_file(handler, root, query)
                return
            if route == "projects":
                result = projects(root, server)
            elif route == "project":
                result = summary(root, query.get("scene", [""])[0], server, True)
            else:
                raise Error(404, "Unknown workspace endpoint.")
        else:
            refused = handler._survey_refused()
            origin = urlparse(handler.headers.get("Origin", ""))
            if refused or origin.scheme not in {"http", "https"} or origin.path not in {"", "/"} or origin.query or origin.fragment or origin.username:
                try:
                    size = body_length(handler, MAX_JSON)
                    handler.rfile.read(size)
                except Error:
                    pass
                raise Error(403, "Workspace writes require loopback and a matching browser Origin/Host.")
            if route == "upload":
                result = upload(handler, root, server)
            else:
                data = json_body(handler)
                if route == "project":
                    result = save_project(root, data, server)
                elif route in {"measurements", "measurements/delete"}:
                    result = save_measurement(root, data, server, route.endswith("/delete"))
                elif route in {"placements", "placements/update", "placements/delete"}:
                    mode = "delete" if route.endswith("/delete") else "update" if route.endswith("/update") else "create"
                    result = save_placement(root, data, server, mode)
                elif route == "run":
                    result = run_job(root, data, server)
                elif route == "cancel":
                    scene_paths(root, data.get("scene"))
                    if not server.cancel_job(data["scene"]):
                        raise Error(409, "There is no matching active job to cancel.")
                    result = {"status": "cancelled", "scene": data["scene"]}
                else:
                    raise Error(404, "Unknown workspace endpoint.")
        handler._survey_json(result)
    except ConnectionError:
        handler.close_connection = True
    except Error as error:
        handler.close_connection = True
        handler._survey_json({"error": str(error)}, error.status)
    except FileExistsError:
        handler._survey_json({"error": "A file with this name already exists."}, 409)
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        print("[workspace] " + type(error).__name__ + ": " + str(error), file=server.sys.stderr, flush=True)
        handler.close_connection = True
        handler._survey_json({"error": "Workspace request could not be completed. Check the server log."}, 400)
