"""Infrastructure inspection on a scan (INF-01, INF-02, INF-06, INF-08; ARC-03).

* ``frames_that_saw`` (M6 / INF-01) - for a clicked point, the recovered cameras whose image
  contains it and whose line of sight is not blocked by the scanned surface, ranked by how
  well they saw it (close, near the image centre). Each comes with its pixel, so the UI can
  open the original frame at the defect. Occlusion is tested against the scene's top
  surface model: a thin occluder (a pole, a wire, a railing) is not seen by that test.
* the **defect register** (INF-02 / ARC-03) - annotations with type, severity, status, note,
  the frames that saw them and a photo crop cut from the best frame at full resolution,
  stored in ``work/<scene>/inspection/``. They are a layer on the scan, never part of it.
* ``crack_candidates`` (INF-06, heuristic) - thin dark lines in an annotation's photo crop
  (morphological black-hat, Otsu, elongated components); their pixel length times the
  ground sample distance of that frame at that range gives an approximate length in mm.
  A shadow line or a joint looks the same; the result is a prompt for the inspector.
* ``report_pdf`` (INF-08) - the register as a PDF: summary, register table, one page per
  defect with its photo, position, frames, measurements and notes.
"""
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TYPES = ("crack", "spalling", "corrosion", "exposed_rebar", "deformation", "vegetation", "water_ingress",
         "missing_element", "inscription", "carving", "erosion", "graffiti", "other")
STATUSES = ("open", "monitor", "repair_planned", "closed")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class InspectionError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ provenance (M6)
def _cameras(work):
    path = Path(work) / "viewer_assets" / "cameras.json"
    try:
        cams = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    keys = ("pos", "forward", "up", "right", "fx", "fy", "cx", "cy", "width", "height", "name")
    return [c for c in cams if isinstance(c, dict) and all(k in c for k in keys)]


def project(camera, point):
    """(u, v, depth) of a viewer-frame point in ``camera``'s image, or None behind it."""
    d = np.asarray(point, float) - np.asarray(camera["pos"], float)
    z = float(d @ np.asarray(camera["forward"], float))
    if z <= 1e-6:
        return None
    x = float(d @ np.asarray(camera["right"], float))
    y = -float(d @ np.asarray(camera["up"], float))
    return camera["cx"] + camera["fx"] * x / z, camera["cy"] + camera["fy"] * y / z, z


def _blocked(ground, a, b, *, skip_end_m=0.6, margin_m=0.3):
    """True when the scanned top surface rises above the segment a->b (last metres skipped)."""
    if ground is None:
        return False
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    n = max(2, int(length / max(ground.cell * 0.7, 0.05)))
    t = np.linspace(0.0, 1.0, n)[:-1]
    t = t[t * length < length - skip_end_m]
    if not len(t):
        return False
    pts = a[None] + (b - a)[None] * t[:, None]
    top, supported = ground.sample_top(pts[:, [0, 2]])
    return bool(np.any(supported & (top > pts[:, 1] + margin_m)))


def frames_that_saw(work, point, *, ground=None, limit=6):
    cams = _cameras(work)
    if not cams:
        raise InspectionError(409, "The scene has no recovered cameras, so no frame can be traced.")
    out, seen = [], 0
    for index, cam in enumerate(cams):
        p = project(cam, point)
        if p is None:
            continue
        u, v, depth = p
        if not (0 <= u < cam["width"] and 0 <= v < cam["height"]):
            continue
        if _blocked(ground, cam["pos"], point):
            continue
        seen += 1
        off = math.hypot((u - cam["cx"]) / cam["width"], (v - cam["cy"]) / cam["height"])
        gsd_mm = depth / cam["fx"] * 1000.0
        out.append({"camera_index": index, "name": cam["name"], "t_sec": cam.get("t_sec"),
                    "u": round(u, 1), "v": round(v, 1), "width": cam["width"], "height": cam["height"],
                    "distance_m": round(depth, 2), "gsd_mm": round(gsd_mm, 2),
                    "centrality": round(1.0 - min(1.0, off * 1.6), 3),
                    "score": round((1.0 / max(depth, 0.1)) * (0.4 + 0.6 * (1.0 - min(1.0, off * 1.6))), 5)})
    out.sort(key=lambda r: -r["score"])
    return {"frames": out[:limit], "seen_by": seen, "cameras": len(cams),
            "basis": "cameras whose image contains the point and whose sight line clears the scanned top "
                     "surface; thin occluders (poles, wires, railings) are not modelled"}


def frame_path(work, name):
    """The largest available image for a camera name (frames_full before frames_train)."""
    work = Path(work)
    for folder in ("frames_full", "frames_train", "frames_undist"):
        path = work / folder / name
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return path
        flat = work / folder / Path(name).name
        if flat.is_file():
            return flat
    return None


def crop(work, frame, *, half_px=160):
    """A crop of the original frame around the traced pixel, at the image's own resolution."""
    from PIL import Image
    path = frame_path(work, frame["name"])
    if path is None:
        raise InspectionError(404, f"Frame {frame['name']} is not on disk.")
    image = Image.open(path).convert("RGB")
    sx, sy = image.width / frame["width"], image.height / frame["height"]
    u, v = frame["u"] * sx, frame["v"] * sy
    half = int(half_px * max(sx, sy, 1.0))
    box = (max(0, int(u - half)), max(0, int(v - half)), min(image.width, int(u + half)), min(image.height, int(v + half)))
    return image.crop(box), {"box": box, "scale": [round(sx, 4), round(sy, 4)], "source": path.name,
                             "centre_px": [round(u - box[0], 1), round(v - box[1], 1)]}


# ------------------------------------------------------------------ defect register
def _dir(work):
    return Path(work) / "inspection"


def _load(work):
    path = _dir(work) / "annotations.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("items"), list) else {"items": [], "revision": 0}
    except (OSError, ValueError):
        return {"items": [], "revision": 0}


def _save(work, data):
    folder = _dir(work)
    folder.mkdir(parents=True, exist_ok=True)
    data["revision"] = int(data.get("revision", 0)) + 1
    tmp = folder / "annotations.json.tmp"
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(folder / "annotations.json")
    return data


def list_annotations(work):
    return _load(work)


def _clean(item, existing=None):
    base = dict(existing or {})
    pos = item.get("position", base.get("position"))
    if not (isinstance(pos, list) and len(pos) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)):
        raise InspectionError(400, "position must be a finite [x, y, z] in the viewer frame")
    kind = item.get("type", base.get("type", "other"))
    if kind not in TYPES:
        raise InspectionError(400, f"type must be one of {', '.join(TYPES)}")
    severity = item.get("severity", base.get("severity", 2))
    if not (isinstance(severity, int) and 1 <= severity <= 5):
        raise InspectionError(400, "severity is 1 (note) to 5 (urgent)")
    status = item.get("status", base.get("status", "open"))
    if status not in STATUSES:
        raise InspectionError(400, f"status must be one of {', '.join(STATUSES)}")
    measurements = item.get("measurements", base.get("measurements", {}))
    if not isinstance(measurements, dict) or len(measurements) > 20:
        raise InspectionError(400, "measurements must be a small object of name -> number")
    clean_m = {}
    for k, v in measurements.items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            clean_m[str(k)[:40]] = float(v)
    base.update(position=[round(float(v), 4) for v in pos], type=kind, severity=severity, status=status,
                title=str(item.get("title", base.get("title", kind.replace("_", " ").capitalize())))[:120],
                note=str(item.get("note", base.get("note", "")))[:4000], measurements=clean_m,
                element=str(item.get("element", base.get("element", "")))[:120])
    return base


def add_annotation(work, item, *, ground=None, revision=None):
    data = _load(work)
    if revision is not None and revision != data.get("revision", 0):
        raise InspectionError(409, "The register changed in another window; reload it.")
    if len(data["items"]) >= 2000:
        raise InspectionError(413, "The register holds at most 2000 items.")
    entry = _clean(item)
    entry["id"] = "a-" + uuid.uuid4().hex[:10]
    entry["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        trace = frames_that_saw(work, entry["position"], ground=ground)
    except InspectionError:
        trace = {"frames": [], "seen_by": 0}
    entry["frames"] = trace["frames"][:4]
    entry["seen_by"] = trace["seen_by"]
    entry["photo"] = None
    if entry["frames"]:
        try:
            image, info = crop(work, entry["frames"][0])
            folder = _dir(work) / "photos"
            folder.mkdir(parents=True, exist_ok=True)
            image.save(folder / f"{entry['id']}.jpg", quality=90)
            entry["photo"] = {"file": f"inspection/photos/{entry['id']}.jpg", **info, "frame": entry["frames"][0]["name"]}
        except (InspectionError, OSError):
            pass
    data["items"].append(entry)
    return _save(work, data), entry


def update_annotation(work, item_id, item, *, revision=None):
    data = _load(work)
    if revision is not None and revision != data.get("revision", 0):
        raise InspectionError(409, "The register changed in another window; reload it.")
    for k, existing in enumerate(data["items"]):
        if existing["id"] == item_id:
            data["items"][k] = _clean({**existing, **item}, existing)
            data["items"][k]["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return _save(work, data)
    raise InspectionError(404, "Annotation not found.")


def delete_annotation(work, item_id, *, revision=None):
    data = _load(work)
    if revision is not None and revision != data.get("revision", 0):
        raise InspectionError(409, "The register changed in another window; reload it.")
    before = len(data["items"])
    data["items"] = [a for a in data["items"] if a["id"] != item_id]
    if len(data["items"]) == before:
        raise InspectionError(404, "Annotation not found.")
    photo = _dir(work) / "photos" / f"{item_id}.jpg"
    photo.unlink(missing_ok=True)
    return _save(work, data)


# ------------------------------------------------------------------ crack candidates (INF-06)
def crack_candidates(image, *, gsd_mm=None, min_length_px=25, min_elongation=4.0):
    """Thin dark lines in an RGB PIL image or array. Heuristic; returns lines and an overlay."""
    import cv2
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if blackhat.max() < 18:                       # nothing darker than its surroundings at all
        mask[:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lines = []
    keep = np.zeros_like(mask)
    for k in range(1, count):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_length_px:
            continue
        ys, xs = np.nonzero(labels == k)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        centred = pts - pts.mean(0)
        evals = np.sort(np.linalg.eigvalsh(np.cov(centred.T)))[::-1]
        elongation = math.sqrt(evals[0] / max(evals[1], 1e-6))
        if elongation < min_elongation:
            continue
        # Length along the principal axis; width = area / length.
        axis = np.linalg.eigh(np.cov(centred.T))[1][:, -1]
        proj = centred @ axis
        length = float(proj.max() - proj.min())
        if length < min_length_px:
            continue
        width = area / max(length, 1.0)
        keep[labels == k] = 255
        entry = {"length_px": round(length, 1), "width_px": round(width, 1), "elongation": round(elongation, 1),
                 "centre_px": [round(float(v), 1) for v in pts.mean(0)]}
        if gsd_mm:
            entry.update(length_mm=round(length * gsd_mm, 1), width_mm=round(width * gsd_mm, 2))
        lines.append(entry)
    lines.sort(key=lambda e: -e["length_px"])
    overlay = rgb.copy() if rgb.ndim == 3 else np.dstack([rgb] * 3)
    overlay[keep > 0] = [230, 30, 30]
    return {"candidates": lines[:20], "overlay": overlay,
            "basis": "thin dark lines (morphological black-hat + Otsu, elongation >= %g); joints, cables and "
                     "shadow edges look the same - confirm on the photo" % min_elongation}


# ------------------------------------------------------------------ report (INF-08)
def report_pdf(work, *, title, registry=None, scene_name=""):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    data = _load(work)
    items = sorted(data["items"], key=lambda a: (-a["severity"], a["created_at"]))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = f"{scene_name} · inspection register rev {data.get('revision', 0)} · generated {now} · positions from the reconstruction"

    def where(a):
        if registry and registry.get("status") == "georeferenced":
            import scene_frames
            import survey_coords
            lat, lon, h = scene_frames.viewer_to_geodetic([a["position"]], registry)[0]
            return f"{survey_coords.to_mgrs(float(lat), float(lon))} · h {h:.1f} m (ellipsoidal)"
        x, y, z = a["position"]
        return f"scene x {x:.2f}, y {y:.2f}, z {z:.2f} (local, not georeferenced)"

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, title, fontsize=16, weight="bold")
        counts = {s: sum(1 for a in items if a["severity"] == s) for s in range(5, 0, -1)}
        by_status = {s: sum(1 for a in items if a["status"] == s) for s in STATUSES}
        fig.text(0.07, 0.915, f"{len(items)} items · severity " + ", ".join(f"{k}: {v}" for k, v in counts.items() if v)
                 + " · " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in by_status.items() if v), fontsize=9.5)
        rows = [[a["id"][2:], a["title"][:28], a["type"].replace("_", " "), str(a["severity"]), a["status"].replace("_", " "),
                 str(a.get("seen_by", 0))] for a in items[:36]]
        if rows:
            ax = fig.add_axes([0.07, max(0.08, 0.89 - 0.021 * (len(rows) + 1)), 0.86, 0.021 * (len(rows) + 1)])
            ax.axis("off")
            table = ax.table(cellText=rows, colLabels=["ID", "Title", "Type", "Sev.", "Status", "Frames"], loc="upper left", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
        else:
            fig.text(0.07, 0.85, "The register is empty.", fontsize=10)
        fig.text(0.07, 0.02, footer, fontsize=6.5, color="#666")
        pdf.savefig(fig)
        plt.close(fig)
        for a in items:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.95, f"{a['title']}  [{a['id']}]", fontsize=13, weight="bold")
            facts = [("Type", a["type"].replace("_", " ")), ("Severity", f"{a['severity']} of 5"), ("Status", a["status"].replace("_", " ")),
                     ("Element", a.get("element") or "-"), ("Position", where(a)), ("Recorded", a["created_at"]),
                     ("Seen by", f"{a.get('seen_by', 0)} frames; best: " + ", ".join(f"{f['name']} ({f['distance_m']} m, {f['gsd_mm']} mm/px)" for f in a.get("frames", [])[:2]))]
            facts += [(k, f"{v:g}") for k, v in a.get("measurements", {}).items()]
            y = 0.915
            for k, v in facts:
                fig.text(0.07, y, k, fontsize=9, color="#444")
                fig.text(0.25, y, str(v)[:110], fontsize=9)
                y -= 0.02
            photo = _dir(work).parent / a["photo"]["file"] if a.get("photo") else None
            if photo and photo.is_file():
                ax = fig.add_axes([0.07, 0.3, 0.86, y - 0.33])
                ax.imshow(Image.open(photo))
                cx, cy = a["photo"]["centre_px"]
                ax.plot([cx], [cy], marker="o", mfc="none", mec="red", ms=18, mew=2)
                ax.set_title(f"From {a['photo']['frame']} (original resolution)", fontsize=8)
                ax.axis("off")
            else:
                fig.text(0.07, 0.6, "No frame on disk saw this point.", fontsize=10)
            if a.get("note"):
                fig.text(0.07, 0.27, "Note", fontsize=9, weight="bold")
                fig.text(0.07, 0.25, a["note"][:900], fontsize=8.5, wrap=True, va="top")
            fig.text(0.07, 0.02, footer, fontsize=6.5, color="#666")
            pdf.savefig(fig)
            plt.close(fig)
    return buffer.getvalue()
