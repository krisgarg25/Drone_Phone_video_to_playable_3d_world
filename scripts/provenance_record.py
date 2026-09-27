"""Archive record for a scan (ARC-06): who, when, what device, which CRS, which processing, which files.

Heritage bodies archive a model only with its paradata: how it was made and whether it has
been altered since. ``build`` assembles, from what the pipeline already wrote,

* **descriptive** fields in Dublin Core terms (title, description, creator, date, coverage,
  rights, identifier) - from ``project.json`` where the operator filled them, blank where not;
* **capture**: source video (resolution, frame rate, duration, device where the container
  says), frame count, camera count, capture date;
* **spatial reference**: georeferenced or local, the scale source and its status, CRS and
  origin when there is one, the stated accuracy status (never an accuracy that was not
  checked against independent points);
* **processing chain**: the steps the run recorded with their status and wall time, the
  scenario (preset and who set each knob), the software's own file hashes where a run
  recorded them;
* **fixity**: SHA-256 and size of every delivered file, so a later copy can be checked.

Nothing is invented: a field the pipeline did not record is ``null`` with no guess.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

KEY_FILES = ("splat.ply", "frame.json", "report.json", "scenario.json", "video_meta.json", "keyframes_poses.jsonl",
             "viewer_assets/scene.ply", "viewer_assets/cameras.json", "viewer_assets/collision.json",
             "viewer_assets/ground.f32", "viewer_assets/heights.f32", "viewer_assets/coverage.u8",
             "viewer_assets/semantics.json", "pc/collision.collision.glb", "inspection/annotations.json",
             "twin/attributes.json")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(work):
    work = Path(work)
    project = _json(work / "project.json") or {}
    meta = _json(work / "video_meta.json") or {}
    report = _json(work / "report.json") or {}
    scenario = _json(work / "scenario.json") or {}
    cameras = _json(work / "viewer_assets" / "cameras.json") or []
    try:
        import scene_frames
        registry = scene_frames.load(work)
    except (OSError, ValueError):
        registry = None
    fixity = []
    for rel in KEY_FILES:
        path = work / rel
        if path.is_file():
            fixity.append({"file": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    georef = registry is not None and registry.get("status") == "georeferenced"
    crs = None
    if georef:
        import survey_crs
        c = survey_crs.crs_from_alignment(registry["alignment"])
        crs = {"name": c["name"], "epsg": c["epsg"], "origin": registry.get("origin"), "height_datum": c["height_datum"]}
    record = {
        "schema": "ground-control-archive-record/1",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dublin_core": {
            "title": project.get("name") or work.name, "description": project.get("notes") or None,
            "creator": project.get("operator") or project.get("creator") or None,
            "date": project.get("captured_at") or meta.get("creation_time") or None,
            "coverage": project.get("site") or None, "rights": project.get("rights") or None,
            "identifier": work.name, "type": "3D reconstruction (Gaussian splats + measured surface)",
            "format": ["PLY (splats)", "GLB (collision surface)", "float32 grids (surface model)"],
        },
        "capture": {"video": {k: meta.get(k) for k in ("width", "height", "fps", "duration_s", "codec", "device", "source")
                              if meta.get(k) is not None} or None,
                    "cameras_recovered": len(cameras) if isinstance(cameras, list) else None,
                    "first_frame_t": cameras[0].get("t_sec") if cameras else None,
                    "last_frame_t": cameras[-1].get("t_sec") if cameras else None},
        "spatial_reference": {"status": None if registry is None else registry.get("status"),
                              "scale_status": None if registry is None else registry.get("scale_status"),
                              "scale_source": None if registry is None else registry.get("scale_source"),
                              "crs": crs,
                              "accuracy": "not independently verified" if not project.get("accuracy_verified") else project["accuracy_verified"]},
        "processing": {"steps": [{"name": s.get("name"), "status": s.get("status"), "seconds": s.get("secs")} for s in steps] or None,
                       "scenario": {k: scenario.get(k) for k in ("preset", "application", "sources", "profile") if k in scenario} or None,
                       "software": "Ground Control (this repository)"},
        "fixity": fixity,
        "notes": ["Fields the pipeline did not record are null; nothing here is estimated.",
                  "Fixity hashes identify these exact files; any re-processing produces a new record."],
    }
    return record


def verify(work, record):
    """Files whose size or hash no longer matches the record (fixity check)."""
    work = Path(work)
    changed = []
    for entry in record.get("fixity", []):
        path = work / entry["file"]
        if not path.is_file():
            changed.append({"file": entry["file"], "problem": "missing"})
        elif path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            changed.append({"file": entry["file"], "problem": "changed"})
    return changed
