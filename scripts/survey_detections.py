"""People and vehicles as a georeferenced layer, never as geometry (M10, DIS-06, MIL-03).

Detections come from one of two places, in this order:

1. ``work/<scene>/detections.json`` - boxes from a detector run over the frames:
   ``[{"frame": "<image name>", "class": "person"|"vehicle"|..., "bbox": [x0, y0, x1, y1],
   "score": 0..1}]`` in pixel coordinates of that frame;
2. the dynamic-object masks the pipeline already writes (``dynamics_frames.json`` +
   ``masks/``; 0 = excluded) - connected masked regions become boxes of class
   ``moving object`` (the mask does not say person or vehicle).

Each box's bottom-centre pixel (where a standing person or a tyre meets the ground) is
cast as a ray from that frame's recovered camera and intersected with the scanned
surface. Hits from different frames within ``merge_m`` of each other are one object
seen several times: the layer reports each object once, with its first/last time, the
frames that saw it and the spread of the hits (a moving object spreads).

Nothing here edits the reconstruction: the masks already kept these pixels out of the
geometry, and this layer only says where they were.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def boxes_from_mask(mask, *, min_px=12):
    """Bounding boxes of masked (value 0) regions of a COLMAP-style mask image."""
    excluded = np.asarray(mask) == 0
    labels, count = ndimage.label(excluded)
    out = []
    for index, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        size = int((labels[sl] == index).sum())
        if size < min_px:
            continue
        out.append({"bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop], "pixels": size})
    return out


def _read_mask(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def load(work):
    """All raw detections of a scene: [{frame, class, bbox, score, source}], source note."""
    work = Path(work)
    explicit = work / "detections.json"
    if explicit.is_file():
        rows = json.loads(explicit.read_text(encoding="utf-8"))
        return [dict(r, source="detector") for r in rows], "detector boxes (detections.json)"
    listing = work / "dynamics_frames.json"
    if not listing.is_file():
        return [], "no detector output and no dynamic masks for this scene"
    info = json.loads(listing.read_text(encoding="utf-8"))
    masks = info.get("masks") or {}
    folder = work / masks.get("dir", "masks")
    rows = []
    for entry in masks.get("files", []):
        path = folder / entry["mask"]
        if not path.is_file() or not entry.get("masked_fraction"):
            continue
        mask = _read_mask(path)
        for box in boxes_from_mask(mask):
            rows.append({"frame": entry["image"], "class": "moving object", "bbox": box["bbox"],
                         "score": None, "mask_size": [mask.shape[1], mask.shape[0]],
                         "source": "dynamic mask"})
    return rows, "dynamic-object masks (class unknown: person or vehicle)"


def _ray(camera, u, v):
    right, up, forward = (np.asarray(camera[k], float) for k in ("right", "up", "forward"))
    d = forward + (u - camera["cx"]) / camera["fx"] * right - (v - camera["cy"]) / camera["fy"] * up
    return np.asarray(camera["pos"], float), d / np.linalg.norm(d)


def cast(origin, direction, surface, *, max_m=500.0, step_m=0.25):
    """First crossing of the ray below ``surface(xz) -> (y, supported)`` (viewer, y up)."""
    t = np.arange(step_m, max_m, step_m)
    pts = origin[None, :] + t[:, None] * direction[None, :]
    y, ok = surface(pts[:, [0, 2]])
    below = (pts[:, 1] <= y) & ok
    if not below.any():
        return None
    k = int(np.argmax(below))
    if k == 0:
        return pts[0]
    # Linear refinement between the last point above and the first below.
    a, b = pts[k - 1], pts[k]
    ha, hb = a[1] - y[k - 1], b[1] - y[k]
    f = ha / (ha - hb) if ha != hb else 0.0
    return a + f * (b - a)


def georeference(rows, cameras, surface, *, merge_m=3.0):
    """Hits per detection, then objects merged across frames. Viewer frame, metres."""
    by_name = {}
    for cam in cameras:
        by_name[cam["name"]] = cam
        by_name[Path(cam["name"]).name] = cam
    hits, missed = [], 0
    for row in rows:
        cam = by_name.get(row["frame"]) or by_name.get(Path(row["frame"]).name)
        if cam is None:
            missed += 1
            continue
        x0, y0, x1, y1 = row["bbox"]
        sx = sy = 1.0
        if row.get("mask_size"):
            sx, sy = cam["width"] / row["mask_size"][0], cam["height"] / row["mask_size"][1]
        u, v = (x0 + x1) / 2 * sx, y1 * sy                 # bottom centre: where it meets ground
        origin, direction = _ray(cam, u, v)
        point = cast(origin, direction, surface)
        if point is None:
            missed += 1
            continue
        hits.append({"at": point, "t": cam.get("t_sec"), "frame": cam["name"], "class": row["class"],
                     "score": row.get("score"), "height_px": (y1 - y0) * sy})
    objects = []
    for hit in sorted(hits, key=lambda h: (h["t"] is None, h["t"])):
        match = None
        for obj in objects:
            if obj["class"] == hit["class"] and np.hypot(*(obj["centre"][[0, 2]] - hit["at"][[0, 2]])) <= merge_m:
                match = obj
                break
        if match is None:
            objects.append({"class": hit["class"], "hits": [hit], "centre": hit["at"].copy()})
        else:
            match["hits"].append(hit)
            match["centre"] = np.mean([h["at"] for h in match["hits"]], axis=0)
    out = []
    for i, obj in enumerate(objects):
        pts = np.array([h["at"] for h in obj["hits"]])
        times = [h["t"] for h in obj["hits"] if h["t"] is not None]
        scores = [h["score"] for h in obj["hits"] if h["score"] is not None]
        spread = float(np.max(np.hypot(pts[:, 0] - obj["centre"][0], pts[:, 2] - obj["centre"][2])))
        out.append({"id": f"D{i + 1}", "class": obj["class"],
                    "position": [round(float(v), 3) for v in obj["centre"]],
                    "frames": sorted({h["frame"] for h in obj["hits"]}), "sightings": len(obj["hits"]),
                    "first_t": min(times) if times else None, "last_t": max(times) if times else None,
                    "spread_m": round(spread, 2), "moving": spread > merge_m / 2,
                    "score": round(float(np.mean(scores)), 3) if scores else None})
    return {"objects": out, "hits": len(hits), "unplaced": missed}


def layer(work, surface, *, registry=None, merge_m=3.0):
    """The scene's detection layer, with WGS84/MGRS when the scene is georeferenced."""
    work = Path(work)
    rows, source = load(work)
    cam_path = work / "viewer_assets" / "cameras.json"
    cameras = json.loads(cam_path.read_text(encoding="utf-8")) if cam_path.is_file() else []
    if not rows:
        return {"objects": [], "hits": 0, "unplaced": 0, "source": source,
                "notes": ["No detections to place."]}
    if not cameras:
        return {"objects": [], "hits": 0, "unplaced": len(rows), "source": source,
                "notes": ["The scene has no recovered cameras, so detections cannot be placed."]}
    result = georeference(rows, cameras, surface, merge_m=merge_m)
    if registry is not None and registry.get("status") == "georeferenced":
        import scene_frames
        import survey_coords
        for obj in result["objects"]:
            lat, lon, h = scene_frames.viewer_to_geodetic([obj["position"]], registry)[0]
            obj.update(lat=round(float(lat), 7), lon=round(float(lon), 7),
                       mgrs=survey_coords.to_mgrs(float(lat), float(lon)))
    result["source"] = source
    result["notes"] = ["Detections are a layer: the reconstruction does not contain these people or "
                       "vehicles (their pixels were masked out of the geometry).",
                       "Position = the box's bottom centre cast onto the scanned surface; a person "
                       "standing on a roof edge or a partly hidden vehicle can land a few metres off.",
                       "Objects seen within %g m in several frames are merged; 'moving' means their "
                       "hits spread more than half of that." % merge_m]
    return result
