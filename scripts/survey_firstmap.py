"""First map while the run continues (DIS-01 / GAPS H1).

``survey_progressive`` merges one window after another into an accumulated sparse
model. After each merge this module turns that model into something a response team
can look at *now*:

* ``preview.png`` - a top-down colour mosaic (mean point colour per cell, north up;
  cells without a point are transparent, never filled);
* ``dsm.tif`` - the highest sparse point per cell in UTM, when the window's cameras could
  be georeferenced against the flight's GPS (``georeference`` callable, supplied by
  ``survey_workflow``); otherwise nothing georeferenced is written and the preview
  says "local frame, unscaled";
* ``preview.json`` and ``progressive/preview/latest.json`` - window, counts, cell size,
  bounds, elapsed wall time, georeference status and every caveat.

It is a *sparse* first look: a few thousand triangulated points, coarse cells, heights
from the highest point per cell. It is labelled a preview everywhere and is replaced by
the dense products when the run finishes.
"""
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np

LABEL = ("First look from the sparse model while the run continues - coarse, "
         "replaced by the dense products when the run finishes.")


def _quat_to_r(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def read_txt_model(txt_dir):
    """Points (xyz, rgb) and registered images (name, R, t) from a COLMAP TXT model."""
    txt_dir = Path(txt_dir)
    xyz, rgb = [], []
    for line in (txt_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        xyz.append([float(v) for v in f[1:4]])
        rgb.append([int(v) for v in f[4:7]])
    images = []
    # A registered image is a pose line (nine numeric fields, then a file name) followed
    # by its 2D points, which is an EMPTY line when it observes none - so blank lines
    # cannot be dropped and pairs cannot be counted; pose lines are recognised by shape.
    # The name may itself contain spaces, so it is read from a capped split rather than
    # as the tenth whitespace token.
    for line in (txt_dir / "images.txt").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if line.startswith("#") or len(f) < 10:
            continue
        try:
            float(f[9])
            continue                      # a points line that happens to have ten numbers
        except ValueError:
            pass
        p = line.split(maxsplit=9)
        q = [float(v) for v in p[1:5]]
        t = [float(v) for v in p[5:8]]
        images.append({"name": p[9].strip(), "R": _quat_to_r(*q), "t": np.array(t)})
    return np.asarray(xyz, float).reshape(-1, 3), np.asarray(rgb, np.uint8).reshape(-1, 3), images


def camera_rows(images, times):
    """keyframes_poses.jsonl-shaped rows for survey_georef, timed by frame name."""
    rows = []
    for image in images:
        t = times.get(image["name"])
        if t is None:
            continue
        rows.append({"file": image["name"], "t_sec": float(t),
                     "camera": {"R_rowmajor": image["R"].tolist(), "t": image["t"].tolist()}})
    return rows


def _png_rgba(rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def rasterise(points, colors, *, cell_m=None, max_cells=512):
    """Colour mosaic + highest-point DSM on a north-up grid (x east, y north, z up)."""
    if len(points) < 10:
        raise ValueError(f"only {len(points)} sparse points so far; nothing to map yet")
    lo, hi = points[:, :2].min(0), points[:, :2].max(0)
    span = float(max(hi - lo))
    if cell_m is None:
        # Coarse on purpose: a sparse model has a few points per square metre at best.
        cell_m = max(span / max_cells, span / math.sqrt(len(points)) * 1.5, 1e-6)
    cols = max(1, int(math.ceil((hi[0] - lo[0]) / cell_m)) + 1)
    rows = max(1, int(math.ceil((hi[1] - lo[1]) / cell_m)) + 1)
    col = ((points[:, 0] - lo[0]) / cell_m).astype(int)
    row = rows - 1 - ((points[:, 1] - lo[1]) / cell_m).astype(int)
    flat = row * cols + col
    n = rows * cols
    count = np.bincount(flat, minlength=n)
    rgba = np.zeros((n, 4), np.float64)
    for k in range(3):
        rgba[:, k] = np.bincount(flat, weights=colors[:, k].astype(float), minlength=n)
    seen = count > 0
    rgba[seen, :3] /= count[seen, None]
    rgba[seen, 3] = 255
    dsm = np.full(n, -np.inf)
    np.maximum.at(dsm, flat, points[:, 2])
    dsm[~seen] = np.nan
    transform = (float(lo[0]), cell_m, 0.0, float(lo[1] + rows * cell_m), 0.0, -cell_m)
    return dict(rgba=rgba.reshape(rows, cols, 4).astype(np.uint8),
                dsm=dsm.reshape(rows, cols), cell_m=float(cell_m), transform=transform,
                observed_fraction=float(seen.mean()))


def publish(run, index, txt_dir, *, georeference=None, times=None, started=None,
            crs_wkt=None, stats=None):
    """Write ``progressive/preview/window_<i>/`` and ``latest.json``. Never raises."""
    run = Path(run)
    out = run / "progressive" / "preview" / f"window_{index:02d}"
    record = {"window": index, "label": LABEL, "status": "failed", "georeferenced": False,
              "elapsed_s": None if started is None else round(time.perf_counter() - started, 2)}
    try:
        out.mkdir(parents=True, exist_ok=True)
        xyz, rgb, images = read_txt_model(txt_dir)
        frame, reason = "local", "no GPS telemetry for this run"
        if georeference is not None and times:
            rows = camera_rows(images, times)
            try:
                alignment = georeference(rows)
                import survey_crs
                import survey_georef
                enu = survey_georef.transform_points(xyz, alignment)
                xyz, _, crs = survey_crs.enu_to_crs(enu, alignment)
                crs_wkt = crs_wkt or crs["wkt"]
                record["crs"] = crs["name"]
                frame, reason = "utm", None
                record["alignment"] = {k: alignment.get(k) for k in ("scale", "fit_rmse_m", "method")
                                       if k in alignment}
            except (ValueError, KeyError, TypeError) as error:
                reason = f"window cameras could not be georeferenced yet: {error}"
        grid = rasterise(xyz, rgb)
        (out / "preview.png").write_bytes(_png_rgba(grid["rgba"]))
        files = ["preview.png"]
        if frame == "utm" and crs_wkt:
            import survey_formats
            survey_formats.write_geotiff(grid["dsm"].astype(np.float32), out / "dsm.tif",
                                         transform=grid["transform"], crs_wkt=crs_wkt, nodata=-9999.0)
            files.append("dsm.tif")
        record.update(status="done", georeferenced=frame == "utm", frame=frame,
                      reason=reason, points=int(len(xyz)), registered_images=len(images),
                      cell_m=round(grid["cell_m"], 4), transform=grid["transform"],
                      size=[int(grid["rgba"].shape[1]), int(grid["rgba"].shape[0])],
                      observed_fraction=round(grid["observed_fraction"], 4),
                      height_range=[round(float(np.nanmin(grid["dsm"])), 2),
                                    round(float(np.nanmax(grid["dsm"])), 2)],
                      files=files, path=f"progressive/preview/window_{index:02d}",
                      units="m" if frame == "utm" else "model units (unscaled)")
        if stats:
            record["window_stats"] = {k: stats.get(k) for k in ("registered_images", "points")}
    except (ValueError, OSError, IndexError, KeyError) as error:
        record["reason"] = str(error)
    try:
        (out / "preview.json").write_text(json.dumps(record, indent=1))
        if record["status"] == "done":
            latest = run / "progressive" / "preview" / "latest.json"
            tmp = latest.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=1))
            tmp.replace(latest)
    except OSError:
        pass
    return record


def latest(work):
    """The newest preview of the scene's latest survey run, or None."""
    work = Path(work)
    try:
        run_id = json.loads((work / "survey" / "latest_run.json").read_text())["id"]
    except (OSError, ValueError, KeyError):
        return None
    run = work / "survey" / "runs" / run_id
    try:
        record = json.loads((run / "progressive" / "preview" / "latest.json").read_text())
    except (OSError, ValueError):
        return None
    record["run"] = run_id
    record["run_dir"] = run
    return record
