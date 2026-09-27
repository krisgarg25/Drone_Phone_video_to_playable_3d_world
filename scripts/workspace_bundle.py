"""One-click project bundle (G3): model, measurements, reports and CRS in one ZIP.

A reconstruction is only a deliverable once someone else can open it without this
machine. The bundle is built from exactly what the project page already lists and
vets - ``workspace_api.summary(..., detail=True)`` decides which files are
deliverables and which survey evidence is current - so the ZIP can never contain a
stale survey product the page would have refused to show.

Layout::

    README.txt          what is inside, in words, with every caveat the page shows
    manifest.json       machine-readable: project facts + every file's size and SHA-256
    crs/                coordinate reference: CRS JSON and a .prj when georeferenced,
                        otherwise a plain statement that the model is in local units
    model/              the model files (splat, viewer cloud, collision/nav meshes, rooms)
    survey/             georeferenced products and evidence, when a current run exists
    measurements/       JSON + GeoJSON + CSV, each carrying uncertainty and validity
    placements/         the furniture layout, when there is one
    reports/            quality gate, scenario record, audit, capture diagnostics

Files are written in their own formats, unchanged; the only files this module
authors are the README, the manifest, the CRS notes and the measurement exports.
Large binaries are stored rather than deflated (PLY/GLB/LAS gain little and cost a
lot of CPU); text is deflated.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

import workspace_api as api

MAX_BUNDLE_BYTES = 4 << 30
"""A bundle is streamed from a temporary file; beyond this the operator should copy the
work folder instead, and the refusal says so rather than filling the disk."""
STORED = {".ply", ".glb", ".las", ".laz", ".tif", ".tiff", ".splat", ".bin", ".png",
          ".jpg", ".jpeg", ".npz"}
REPORTS = {"world_check.json": "viewer_assets/world_check.json",
           "scenario.json": "scenario.json", "scenario_audit.json": "scenario_audit.json",
           "diagnostics.json": "diagnostics.json", "report.json": "report.json",
           "frame.json": "frame.json"}


def _folder(name: str, kind: str) -> str:
    if name.startswith("survey/") or kind.startswith(("Georeferenced", "Format manifest",
                                                      "Sparse", "Current dense", "ENU",
                                                      "Support", "Independent", "Run timing",
                                                      "Input provenance")):
        return "survey"
    if name in REPORTS or kind == "report":
        return "reports"
    return "model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(root: Path, scene: str, server) -> dict:
    """What the bundle will hold: (archive path -> source file) plus authored text."""
    project = api.summary(root, scene, server, True)
    _, work = api.scene_paths(root, scene)
    files: dict[str, Path] = {}
    for artifact in project.get("artifacts", []):
        relative = unquote(artifact["url"].split("?")[0].removeprefix("/runtime/"))
        if not api.file_allowed(relative):
            continue
        path = api.safe_path(root, relative)
        if not path.is_file():
            continue
        inside = relative.removeprefix(f"work/{scene}/")
        folder = _folder(inside, str(artifact.get("kind", "")))
        if folder == "survey":
            # Keep the run's own structure, minus the run id, so enu/ and georeferenced/
            # products with the same file names do not collide.
            parts = inside.split("/")
            if parts[:2] == ["survey", "runs"] and len(parts) > 3:
                inside = "/".join(parts[3:])
            inside = inside.removeprefix("survey/")
        files.setdefault(f"{folder}/{inside}", path)
    for name, relative in REPORTS.items():
        path = work / relative
        if path.is_file() and api.file_allowed(f"work/{scene}/{relative}"):
            files.setdefault(f"reports/{name}", path)
    authored: dict[str, str] = {}
    measurements = [m for m in project.get("measurements", []) if isinstance(m, dict)]
    if measurements:
        import workspace_measure
        authored["measurements/measurements.json"] = json.dumps(
            {"scene": scene, "coordinate_system": "local viewer Y-up", "scale": project["scale"],
             "measurements": measurements}, indent=2)
        authored["measurements/measurements.geojson"] = json.dumps(
            workspace_measure.measurements_geojson(measurements, scene=scene), indent=2)
        authored["measurements/measurements.csv"] = workspace_measure.measurements_csv(measurements)
    placements = [p for p in project.get("placements", []) if isinstance(p, dict)]
    if placements:
        authored["placements/placements.json"] = json.dumps(
            {"scene": scene, "coordinate_system": "local viewer Y-up", "placements": placements},
            indent=2)
    authored.update(_crs_notes(project,
                               files.get("survey/products/georeferenced/export_manifest.json")))
    return {"project": project, "files": files, "authored": authored}


def _crs_notes(project: dict, manifest: Path | None) -> dict:
    """CRS files for the bundle. ``manifest`` is the current run's georeferenced export
    manifest, already vetted by the project summary; its WKT becomes the .prj."""
    geo = project.get("georeference") or {}
    if geo.get("status") != "georeferenced":
        scale = project.get("scale") or {}
        return {"crs/README.txt": (
            "This project is NOT georeferenced.\n\n"
            "Every coordinate in model/ and measurements/ is in the viewer's local Y-up frame.\n"
            f"Scale: {scale.get('status', 'unknown')} ({scale.get('source', 'no source')}).\n"
            "Units are metres only when the scale above says 'metric' or 'estimated'; an "
            "estimated scale is an assumption, not a measurement.\n"
            "Add flight telemetry under Details -> Georeferenced survey to produce products "
            "in a real CRS.\n")}
    notes = {"crs/crs.json": json.dumps({key: geo.get(key) for key in
                                         ("crs", "vertical_datum", "position_reference",
                                          "viewer_frame")}, indent=2)}
    if manifest is not None:
        try:
            wkt = json.loads(manifest.read_text(encoding="utf-8")).get("crs_wkt")
        except (OSError, ValueError):
            wkt = None
        if isinstance(wkt, str):
            notes["crs/georeferenced.prj"] = wkt
    notes["crs/README.txt"] = (
        f"Georeferenced survey products (survey/georeferenced/) use {geo.get('crs')}.\n"
        f"Vertical datum: {geo.get('vertical_datum') or 'see survey/georeferenced/export_manifest.json'}.\n"
        "The interactive model in model/ stays in local Y-up coordinates; it is not itself "
        "georeferenced.\n")
    return notes


def _readme(project: dict, entries: list[dict]) -> str:
    accuracy = project.get("accuracy") or {}
    quality = project.get("quality") or {}
    lines = [f"{project.get('name', project['id'])} - project bundle",
             f"Scene id: {project['id']}",
             f"Bundled: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}", "",
             "WHAT THE NUMBERS MEAN",
             f"  Scale: {(project.get('scale') or {}).get('status')} - "
             f"{(project.get('scale') or {}).get('source')}",
             f"  Location: {(project.get('georeference') or {}).get('status')}",
             "  Accuracy: " + (f"{accuracy.get('rmse_m')} m 3D RMSE against "
                               f"{accuracy.get('checkpoints')} independent checkpoints"
                               if accuracy.get("status") == "verified"
                               else "not independently verified"),
             f"  Quality gate: {quality.get('status', 'not run')}", ""]
    warnings = project.get("warnings") or []
    if warnings:
        lines.append("CAVEATS (the same ones the project page shows)")
        lines += [f"  - {text}" for text in warnings]
        lines.append("")
    lines.append("CONTENTS (sizes and SHA-256 in manifest.json)")
    lines += [f"  {entry['path']}  ({entry['bytes']:,} bytes)" for entry in entries]
    return "\n".join(lines) + "\n"


def build(root: Path, scene: str, server, destination: Path) -> dict:
    """Write the ZIP to ``destination`` and return its manifest."""
    contents = plan(root, scene, server)
    project, files, authored = contents["project"], contents["files"], contents["authored"]
    total = sum(path.stat().st_size for path in files.values())
    if total > MAX_BUNDLE_BYTES:
        raise api.Error(413, f"The bundle would hold {total / 2**30:.1f} GiB; copy work/{scene} "
                             "directly instead.")
    entries = []
    for name, path in sorted(files.items()):
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path),
                        "source": path.relative_to(root).as_posix()})
    for name, text in sorted(authored.items()):
        data = text.encode("utf-8")
        entries.append({"path": name, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "source": "authored"})
    manifest = {"schema_version": 1, "scene": scene, "name": project.get("name"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scale": project.get("scale"), "georeference": project.get("georeference"),
                "accuracy": project.get("accuracy"), "quality": project.get("quality"),
                "scenario": {k: (project.get("scenario") or {}).get(k)
                             for k in ("preset", "label", "decided_by")},
                "warnings": project.get("warnings"), "files": entries}
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        archive.writestr("README.txt", _readme(project, entries), zipfile.ZIP_DEFLATED)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2), zipfile.ZIP_DEFLATED)
        for name, text in sorted(authored.items()):
            archive.writestr(name, text, zipfile.ZIP_DEFLATED)
        for name, path in sorted(files.items()):
            method = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            archive.write(path, name, compress_type=method)
    return manifest


def serve(handler, root: Path, query: dict, server):
    scene = query.get("scene", [""])[0]
    api.scene_paths(root, scene)
    handle, temporary = tempfile.mkstemp(prefix="bundle-", suffix=".zip")
    os.close(handle)
    temporary = Path(temporary)
    try:
        build(root, scene, server, temporary)
        size = temporary.stat().st_size
        handler.send_response(200)
        handler.send_header("Content-Type", "application/zip")
        handler.send_header("Content-Length", str(size))
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Content-Disposition",
                            f"attachment; filename=\"{scene}-bundle.zip\"")
        handler.end_headers()
        if handler.command == "HEAD":
            return
        with temporary.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                handler.wfile.write(block)
    finally:
        temporary.unlink(missing_ok=True)
