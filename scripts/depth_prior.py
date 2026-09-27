"""MoGe-2 metric depth as an INDEPENDENT SECOND RULER (challenge D2b).

What this is. A `py=PY310` pipeline step that runs the MIT-licensed MoGe-2
monocular geometry model (ViT-S, 35 M params) over a scene's keyframes, writes
the metric depth / point / normal maps it predicts into `work/<scene>/depth/`,
and then answers the only question a monocular metric prior can honestly answer
for a reconstruction that already exists:

    "the scene is currently labelled N metres per scene unit. Measured against a
     model that has never seen this scene's cameras, GPS, flight log or scale
     guess, what is N, and do the two numbers agree?"

The answer is a *fit of two rulers*: MoGe's metric radial distances (metres) and
the reconstructed cloud's radial distances (scene units), sampled at COLMAP's own
measured keypoints - the pixels SfM actually reconstructed from. That fit yields a
second `scale_m_per_unit`, with its residual, inlier count and confidence, written
into `work/<scene>/depth/summary.json` next to the existing number.

What this is NOT, stated because the repo has been burned by each of these:

  * NOT a mutator. It never writes `frame.json`, `solve_frame.py`'s anchors, the
    collider or any shipped artefact. It is a validator that reports. `frame.json`
    is opened read-only, and only to compare against it.
  * NOT a COLMAP regulariser. docs/GAPS_AND_OPTIMIZATIONS.md D2 proposed a
    monocular depth prior "to regularise COLMAP"; COLMAP 4.1 is a vendored exe
    whose mapper takes no per-pixel depth - its only prior channel is
    `pose_priors`. Trainer-side depth regularisation is already landed in
    `train_splat.py` (D2a). Only the *metric* half was left, and that is this file.
  * NOT a measurement of scene scale. It is a measurement of the *agreement*
    between one guessed scale and a learned prior's opinion. The learned prior is
    not a survey instrument: its metric head is trained on ground-level photography
    and has no stated absolute accuracy on aerial footage, which is exactly what a
    drone scene is. That is why the deliverable is a verdict with residuals, not a
    replacement number.

Why radial distance rather than Z-depth. MoGe also recovers its own intrinsics,
and on `work/rocks` it recovers 70.3 deg horizontal against COLMAP's 60.4 deg.
Sampling `depth[y, x]` (the coordinate along the optical axis) at a pixel whose
true ray MoGe has wrong directions makes the *same* surface point's depth
depend on that angular error, to first order. The distance from the camera
centre to the surface point seen at pixel (y, x) does not: it is a property of
the surface point, so the two rulers can be compared at the same pixel without
MoGe's focal length entering the ratio. Its FOV error is therefore reported as a
diagnostic (see `moge_vs_colmap_fov`) instead of being allowed to bias the fit.

Units. Every length in `depth/*.npz` and in the fit is either metres (MoGe's
frame, always labelled `_m`) or scene units (the reconstruction's frame, always
labelled `_units`). The single output of the fit, `scale_m_per_unit`, is
metres per scene unit - the same quantity as `frame.json`'s key, so the two are
directly comparable. Nothing here applies a metric constant to the geometry.

Failure mode. An absent measurement is reported as `null` with a reason, never as
0 or as a plausible-looking number: `status` is "measured" or "not_measurable",
and `scale_m_per_unit` is null in the latter case. The gate that refuses is
`fit_scale()` plus `verdict()`; its thresholds are all dimensionless ratios,
counts or pixels and are named constants below.

Usage:
  .venv310\\Scripts\\python.exe scripts/depth_prior.py --work work/rocks
  ... --resolution-level 4 --limit 20 --model Ruicheng/moge-2-vits-normal
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

# The checkpoint this step was developed and measured against. Both the code and
# the weights are permissively licensed - checked at the source rather than from
# the research notes: microsoft/MoGe @ 07444410 LICENSE is MIT (with the bundled
# DINOv2 tree under Apache-2.0), and every `Ruicheng/moge-2-*` Hub repo is tagged
# `license: mit` with base_model facebook/dinov2-{small,base,large}, which HF tags
# `apache-2.0`. No CC-BY-NC anywhere in the chain - unlike Pi3, VGGT's default
# checkpoint and Depth-Anything-V2 Base/Large, which this project rejected.
DEFAULT_MODEL = "Ruicheng/moge-2-vits-normal"
MODEL_LICENSE = {"Ruicheng/moge-2-vits-normal": ("MIT code + MIT weights (Apache-2.0 "
                                                 "DINOv2-small backbone)", 140_550_416)}

# --- Gate thresholds. All dimensionless: ratios, counts and pixels. ----------
MIN_OBSERVATIONS = 500        # pooled usable pixel-pairs; below this there is no fit
PER_FRAME_MIN_OBS = 40        # relaxed bar for the per-frame estimates, whose only job
                              # is to test whether frames agree with each other
MIN_FRAMES_WITH_DATA = 3      # one frame's ratio distribution proves nothing
MIN_INLIER_FRACTION = 0.20    # less than a fifth of the pixels agreeing = no ruler
MAX_LOG_MAD = 0.35            # 1.4826*MAD of log-ratio; 0.35 ~ +-42% scatter
MAX_FRAME_LOG_SPREAD = 0.40   # IQR of per-frame medians; frames must agree with each other
TRIM_SIGMA = 2.5              # inlier band, in robust sigmas of the log-ratio
FLOOR_REL_SIGMA = 0.05        # a 5% relative floor so a suspiciously tight cloud
                              # cannot make 2000 coincident pixels look like proof
MIN_TRACK_VIEWS = 3           # COLMAP points triangulated from fewer views are noisy
MAX_REPROJ_ERR_PX = 2.0       # COLMAP's own per-point ERROR, in pixels
MAX_WINDOW_SPREAD = 0.20      # 3x3 relative spread of MoGe radius: rejects its own
                              # edges, where a 1-px registration slip moves metres
MIN_RADIUS_UNITS = 1e-6       # guard on the ratio, not a physical threshold

# Agreement tiers, as relative distance between the two rulers. Deliberately not
# a "is it within 5%" survey bar: MoGe-2 is a prior, so the informative question
# is "same story" vs "different story", and the number is reported either way.
AGREE_WITHIN = 0.25           # <= 25% apart: the two rulers tell the same story
CONSISTENT_WITHIN = 0.60      # <= 60% apart: same order, materially different size

RESOLUTION_LEVELS = range(0, 10)


# --------------------------------------------------------------------------
# COLMAP text-model readers (this step needs the 2D observations, which the
# shared parsers drop, so it reads them itself instead of forking the model).
# --------------------------------------------------------------------------
def parse_images_with_points(path: Path) -> dict:
    """name -> {image_id, R, t, cam_id, obs:[(x, y, point3D_id)]}.

    images.txt holds two lines per image: a pose line and a POINTS2D line of
    (X, Y, POINT3D_ID) triples. POINT3D_ID is -1 for a keypoint that never
    triangulated, which is the definition of "not mutually observed".
    """
    from parse_colmap import qvec2rot
    out = {}
    rows = [l for l in rb.read_text(path).splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    for pair in range(0, len(rows) - 1, 2):
        p = rows[pair].split(maxsplit=9)
        q = rows[pair + 1].split()
        if len(p) < 10:
            continue
        try:
            R = qvec2rot(np.array([float(x) for x in p[1:5]]))
            t = np.array([float(x) for x in p[5:8]], np.float64)
            cam_id = int(p[8])
            image_id = int(p[0])
        except ValueError:
            continue
        nums = np.array([float(x) for x in q], np.float64) if q else np.empty(0)
        obs = nums.reshape(-1, 3) if nums.size % 3 == 0 else np.empty((0, 3))
        if obs.size:
            obs = obs[obs[:, 2] >= 0]                     # keep triangulated only
        out[p[9].strip()] = dict(image_id=image_id, R=R, t=t, cam_id=cam_id, obs=obs)
    return out


def parse_points3d_with_tracks(path: Path):
    """-> (ids, xyz, err, track_len). Track is the number of (image, point2D) pairs."""
    ids, xyz, err, tl = [], [], [], []
    for line in rb.read_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        ids.append(int(p[0]))
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        err.append(float(p[7]))
        tl.append(max(1, (len(p) - 8) // 2))
    return (np.array(ids, np.int64), np.asarray(xyz, np.float64),
            np.array(err, np.float64), np.array(tl, np.int64))


def collect_frame_observations(images: dict, pts_ids, pts_xyz, pts_err, pts_tl):
    """For every registered image, the pixels SfM reconstructed and the same
    points' radial distance from that camera, in scene units.

    Returns [{name, image_id, R, t, xy: (M,2) int, radius_units: (M,)}] filtered by
    COLMAP's own quality flags only - nothing here knows about metres yet.
    """
    order = np.argsort(pts_ids, kind="stable")
    sorted_ids = np.asarray(pts_ids)[order]

    def row_of(pid):
        """point3D_ID -> row index in pts_xyz, -1 when the id is absent."""
        i = np.clip(np.searchsorted(sorted_ids, pid), 0, max(0, len(sorted_ids) - 1))
        return np.where(sorted_ids[i] == pid, order[i], -1)

    keep = (pts_err <= MAX_REPROJ_ERR_PX) & (pts_tl >= MIN_TRACK_VIEWS)
    frames = []
    for name, im in sorted(images.items()):
        obs = im["obs"]
        if len(obs) == 0:
            continue
        ok = row_of(obs[:, 2].astype(np.int64))
        good = (ok >= 0) & keep[np.clip(ok, 0, None)]
        if not good.any():
            continue
        obs, ok = obs[good], ok[good]
        world = pts_xyz[ok]
        cam = world @ im["R"].T + im["t"]                 # COLMAP: x_cam = R x + t
        radius = np.linalg.norm(cam, axis=1)
        front = cam[:, 2] > 0.0
        sel = front & (radius > MIN_RADIUS_UNITS)
        if not sel.any():
            continue
        frames.append(dict(name=name, image_id=im["image_id"], R=im["R"], t=im["t"],
                           xy=np.rint(obs[sel][:, :2]).astype(np.int64),
                           radius_units=radius[sel]))
    return frames


# --------------------------------------------------------------------------
# The fit. Pure numpy so the CPU lane can unit-test it without torch.
# --------------------------------------------------------------------------
def sample_radius(radius_map, valid_map, xy, window=1):
    """MoGe radial distance (metres) at the given pixels of its metric point map.

    A (2*window+1)^2 patch per pixel: the median of the *valid* patch values is the
    estimate, and the patch's relative spread comes back too so the caller can drop
    pixels sitting on a depth edge, where a one-pixel registration slip between
    COLMAP's keypoint and MoGe's surface moves metres rather than millimetres.
    Pixels outside the frame, masked out, or with an invalid patch neighbour return
    usable=False - never a plausible-looking zero.
    """
    h, w = radius_map.shape
    u, v = np.asarray(xy)[:, 0], np.asarray(xy)[:, 1]
    n = len(u)
    rad = np.zeros(n, np.float64)
    spread = np.full(n, np.inf)
    usable = np.zeros(n, bool)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not inside.any():
        return rad, spread, usable
    idx_u, idx_v = u[inside], v[inside]
    vals, oks = [], []
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            vv = np.clip(idx_v + dy, 0, h - 1)
            uu = np.clip(idx_u + dx, 0, w - 1)
            vals.append(radius_map[vv, uu])
            oks.append(valid_map[vv, uu])
    vals = np.stack(vals)                                   # (P, M) metres
    oks = np.stack(oks)                                     # (P, M) bool
    all_valid = oks.all(axis=0)
    med = np.median(vals, axis=0)
    pmin = np.where(oks, vals, np.inf).min(axis=0)
    pmax = np.where(oks, vals, -np.inf).max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(med > 0, (pmax - pmin) / med, np.inf)
    good = (all_valid & np.isfinite(med) & (med > 0) & (rel <= MAX_WINDOW_SPREAD))
    out_i = np.flatnonzero(inside)
    rad[out_i] = med
    spread[out_i] = rel
    usable[out_i] = good
    return rad, spread, usable


def fit_scale(radius_units, radius_m, per_frame=None, *, max_iter=3,
              min_observations=MIN_OBSERVATIONS):
    """Robust scale-only fit of metres onto scene units, in log space.

    The model is a similarity with the poses held fixed - the two rulers share
    every camera, so only one free parameter (the ratio) exists. Least squares on
    log(radius_m / radius_units) is the scale-only LSQ; the median and the
    trimmed mean are both reported because their gap is itself a diagnostic.
    Returns a dict that is safe to serialise, with nulls where nothing was measured.
    """
    ru = np.asarray(radius_units, np.float64)
    rm = np.asarray(radius_m, np.float64)
    ok = np.isfinite(ru) & np.isfinite(rm) & (ru > MIN_RADIUS_UNITS) & (rm > 0)
    n_in = int(ok.sum())
    base = dict(observations=int(len(ru)), observations_used=n_in,
                status="not_measurable", scale_m_per_unit=None,
                scale_m_per_unit_lsq=None, scale_m_per_unit_median=None,
                inliers=0, inlier_fraction=None, log_mad=None,
                relative_residual_p50=None, relative_residual_p90=None,
                residual_m_p50=None, residual_m_p90=None, reasons=[])
    if n_in < min_observations:
        base["reasons"].append(
            f"only {n_in} pixels are observed by both rulers "
            f"(need {min_observations}); the cloud is too thin to compare")
        return base
    log = np.log(rm[ok] / ru[ok])
    keep = np.ones(len(log), bool)
    centre = float(np.median(log))
    sigma = FLOOR_REL_SIGMA
    for _ in range(max_iter):
        mad = float(np.median(np.abs(log[keep] - centre))) * 1.4826
        sigma = max(mad, FLOOR_REL_SIGMA)
        new = np.abs(log - centre) <= TRIM_SIGMA * sigma
        if new.sum() < min_observations or (new == keep).all():
            if new.sum() >= min_observations:
                keep = new
            break
        keep, centre = new, float(np.median(log[new]))
    n_keep = int(keep.sum())
    frac = n_keep / len(log)
    mad = float(np.median(np.abs(log[keep] - centre))) * 1.4826
    scale_median = float(math.exp(centre))
    scale_lsq = float(math.exp(float(log[keep].mean())))
    rel_res = np.abs(log[keep] - math.log(scale_lsq))
    s_units = ru[ok][keep]
    out = dict(observations=int(len(ru)), observations_used=n_in,
               status="measured", scale_m_per_unit=scale_median,
               scale_m_per_unit_lsq=scale_lsq, scale_m_per_unit_median=scale_median,
               inliers=n_keep, inlier_fraction=frac, log_mad=mad,
               relative_residual_p50=float(np.percentile(rel_res, 50)),
               relative_residual_p90=float(np.percentile(rel_res, 90)),
               residual_m_p50=float(np.median(np.abs(rm[ok][keep] - scale_median * s_units))),
               residual_m_p90=float(np.percentile(
                   np.abs(rm[ok][keep] - scale_median * s_units), 90)),
               ratio_lsq_over_median=scale_lsq / scale_median if scale_median else None,
               reasons=[])
    # Gates on the fit's quality, not on whether we liked the answer.
    if frac < MIN_INLIER_FRACTION:
        out["reasons"].append(
            f"only {frac:.1%} of compared pixels agree within {TRIM_SIGMA} robust sigma "
            f"- the two rulers have no common scale")
    if mad > MAX_LOG_MAD:
        out["reasons"].append(
            f"scatter of the metre/unit ratio is +-{(math.exp(mad) - 1) * 100:.0f}% "
            f"(robust), above the {((math.exp(MAX_LOG_MAD) - 1) * 100):.0f}% the fit may "
            f"be called a measurement")
    if per_frame:
        meds = np.array([f["scale_m_per_unit"] for f in per_frame
                         if f.get("scale_m_per_unit")], np.float64)
        if len(meds) >= MIN_FRAMES_WITH_DATA:
            iqr = float(np.percentile(np.log(meds), 75) - np.percentile(np.log(meds), 25))
            out["frame_log_spread_iqr"] = iqr
            out["frames_used"] = int(len(meds))
            if iqr > MAX_FRAME_LOG_SPREAD:
                out["reasons"].append(
                    f"per-frame scale estimates disagree with each other (IQR of the log "
                    f"is {iqr:.2f}); a consistent single ratio was not found")
        elif len(meds) > 0:
            out["frames_used"] = int(len(meds))
            out["reasons"].append(f"only {len(meds)} frames carry a per-frame estimate "
                                  f"(need {MIN_FRAMES_WITH_DATA})")
    if out["reasons"]:
        out["status"] = "not_measurable"
        out["scale_m_per_unit"] = None
        out["scale_m_per_unit_lsq"] = None
        out["scale_m_per_unit_median"] = None
    return out


def verdict(fit, frame: dict) -> dict:
    """Compare the learned ruler with the shipped one. Reports; never writes."""
    existing = frame.get("scale_m_per_unit")
    v = dict(
        existing_scale_m_per_unit=existing,
        existing_scale_source=frame.get("scale_source"),
        moge_scale_m_per_unit=None,
        ratio_moge_over_existing=None,
        agreement="not_comparable",
        agreement_note="",
        confidence="none",
        modifies_anything=False,
    )
    s = fit.get("scale_m_per_unit")
    if fit.get("status") != "measured" or not s or not rb.finite(existing) or existing <= 0:
        if not rb.finite(existing) or existing <= 0:
            v["agreement_note"] = ("frame.json has no usable scale_m_per_unit, so there "
                                   "is nothing to compare against")
        else:
            v["agreement_note"] = "; ".join(fit.get("reasons") or ["fit not measured"])
        return v
    ratio = s / existing
    d = abs(ratio - 1.0)
    v.update(moge_scale_m_per_unit=s, ratio_moge_over_existing=ratio)
    if d <= AGREE_WITHIN:
        v["agreement"] = "agree"
    elif d <= CONSISTENT_WITHIN:
        v["agreement"] = "within_order"
    else:
        v["agreement"] = "disagree"
    v["agreement_note"] = (
        f"MoGe's ruler says {s:.4g} m per scene unit; frame.json says "
        f"{existing:.4g} m per scene unit - {d * 100:.0f}% apart, "
        f"{'the learned ruler is the larger' if ratio > 1 else 'the learned ruler is the smaller'}")
    strong = (fit.get("inlier_fraction") or 0) >= 0.5 and (fit.get("log_mad") or 1) <= 0.12
    v["confidence"] = ("high" if strong and fit.get("frames_used", 0) >= 10 else
                       "moderate" if strong or (fit.get("inlier_fraction") or 0) >= 0.35
                       else "low")
    return v


def fov_diagnostic(intrinsics, width, height, fx, fy):
    """MoGe's recovered horizontal FOV against COLMAP's, both in degrees."""
    fx_m = float(intrinsics[0, 0]) * width
    fov_m = math.degrees(2 * math.atan(width / (2 * fx_m))) if fx_m > 0 else None
    fx_c = float(fx if fx else fy)
    fov_c = math.degrees(2 * math.atan(width / (2 * fx_c))) if fx_c > 0 else None
    if not (fov_m and fov_c):
        return None
    return dict(moge_fov_x_deg=round(fov_m, 2), colmap_fov_x_deg=round(fov_c, 2),
                moge_fx_px=round(fx_m, 2), colmap_fx_px=round(fx_c, 2),
                relative_focal_error=round(fx_m / fx_c - 1.0, 4))


# --------------------------------------------------------------------------
# The GPU half. torch and moge are imported inside the run, never at module
# scope: the CPU lane must keep importing this file for its fit tests, and
# tests/test_survey_dynamics.py asserts torch is absent from that lane.
# --------------------------------------------------------------------------
def run_inference(frames, work: Path, out_dir: Path, *, model_id: str,
                  resolution_level: int, fp16: bool, save_maps: bool, limit: int):
    import torch
    from moge.model.v2 import MoGeModel

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        rb.warn("no CUDA device: MoGe-2 on CPU is minutes per frame; this run will "
                "be slow and the summary records device=cpu")
    t0 = time.time()
    model = MoGeModel.from_pretrained(model_id).to(dev).eval()
    load_secs = time.time() - t0
    picked = frames if limit <= 0 else frames[:limit]
    rows, radii_m, radii_u = [], [], []
    for i, fr in enumerate(picked):
        path = None
        for base in ("frames_train", "frames_full", "frames_undist", "frames_match"):
            cand = work / base / fr["name"]
            if cand.is_file():
                path = cand
                break
        if path is None:
            rows.append(dict(file=fr["name"], status="frame_missing"))
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            rows.append(dict(file=fr["name"], status="unreadable"))
            continue
        h, w = img.shape[:2]
        t1 = time.time()
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)
        x = (x.float() / 255.0).to(dev)
        with torch.no_grad():
            out = model.infer(x, num_tokens=None, resolution_level=resolution_level,
                              use_fp16=fp16, apply_mask=False)
        unbatch = lambda t: t[0] if t is not None and t.dim() == 4 else t
        pts = unbatch(out["points"]).cpu().numpy().astype(np.float32)
        dep = unbatch(out["depth"]).cpu().numpy().astype(np.float32)
        msk = (unbatch(out["mask"]).cpu().numpy().astype(bool) if out.get("mask") is not None
               else np.ones((h, w), bool))
        nrm = (unbatch(out["normal"]).cpu().numpy().astype(np.float16)
               if out.get("normal") is not None else None)
        intr = out["intrinsics"].cpu().numpy().astype(np.float32).reshape(3, 3)
        secs = time.time() - t1
        if save_maps:
            # ~10 bytes per pixel: depth f16 (metres, 5e-4 relative - three orders
            # below the fit's own residual), the metric point map f16, normals f16,
            # and validity as packed bits. Full-res maps are what a later
            # trainer-side depth term would consume; --no-maps skips them.
            np.savez_compressed(
                out_dir / f"{Path(fr['name']).stem}.npz",
                depth_m=dep.astype(np.float16), points_m=pts.astype(np.float16),
                mask_valid=np.packbits(msk.ravel()), mask_shape=np.array(msk.shape),
                **({"normals": nrm} if nrm is not None else {}),
                intrinsics=intr,
                frame_name=np.array([fr["name"]]),
                t_sec=np.array([float(fr.get("t_sec") or 0.0)]))
        # The fit needs MoGe's radius at COLMAP's measured pixels.
        rad_map = np.linalg.norm(pts, axis=2).astype(np.float64)
        valid = msk & np.isfinite(rad_map) & (rad_map > 0)
        r_m, _spread, inside = sample_radius(rad_map, valid, fr["xy"])
        rows.append(dict(file=fr["name"], status="ok", width=int(w), height=int(h),
                         secs=round(secs, 3),
                         observations=int(len(fr["xy"])),
                         observations_on_map=int(inside.sum()),
                         fov=fov_diagnostic(intr, w, h, fr.get("fx"), fr.get("fy"))))
        fr["radius_m"] = r_m[inside]
        fr["radius_units_used"] = fr["radius_units"][inside]
        print(f"[depth] {i + 1}/{len(picked)} {fr['name']} {secs:.2f}s "
              f"{int(inside.sum())}/{len(fr['xy'])} px both-rulers", flush=True)
    stats = dict(model=str(model_id), resolution_level=int(resolution_level),
                 fp16=bool(fp16), device=dev.type, load_secs=round(load_secs, 2),
                 frames_requested=len(picked),
                 frames_ok=sum(1 for r in rows if r.get("status") == "ok"),
                 peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                 if dev.type == "cuda" else None,
                 peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
                 if dev.type == "cuda" else None)
    return rows, stats


def load_metric_map(npz_path: Path):
    """Read one saved frame back: (points f32 (H,W,3) metres, valid mask bool).

    Accepts both validity encodings - packed bits (what this version writes) and a
    plain bool array (what an earlier run left on disk) - so --report-only can
    re-fit a directory of maps without re-running inference.
    """
    with np.load(npz_path) as z:
        pts = z["points_m"].astype(np.float32)
        msk = z["mask_valid"]
        if "mask_shape" in z:
            msk = np.unpackbits(msk)[:int(np.prod(z["mask_shape"]))].astype(bool)
            msk = msk.reshape(tuple(int(v) for v in z["mask_shape"]))
        else:
            msk = msk.astype(bool)
        intr = z["intrinsics"].astype(np.float32).reshape(3, 3)
    return pts, msk, intr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True, type=Path, help="work/<scene> directory")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HuggingFace id (default {DEFAULT_MODEL})")
    ap.add_argument("--resolution-level", type=int, default=6, choices=list(RESOLUTION_LEVELS),
                    help="MoGe token knob, 0-9; 6 ~= 1500 tokens on ViT-S. Lower is "
                         "cheaper on a 6 GB card and degrades fine detail first.")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="run at most N keyframes (0 = all)")
    ap.add_argument("--no-maps", dest="save_maps", action="store_false", default=True,
                    help="skip writing the per-frame .npz files (fit only)")
    ap.add_argument("--out", type=Path, default=None, help="default work/<scene>/depth")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when the scale fit is not measurable")
    ap.add_argument("--report-only", action="store_true",
                    help="reuse the .npz maps already in the out dir instead of inferring")
    args = ap.parse_args(argv)
    rb.configure_streams()

    work = Path(args.work)
    out_dir = Path(args.out) if args.out else work / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    poses = rb.jsonl_rows(work / "keyframes_poses.jsonl", required=("file", "camera"))
    if not poses:
        rb.die(rb.EMPTY_INPUT, f"{work / 'keyframes_poses.jsonl'}: no poses - "
                              f"depth_prior runs after the `poses` step", code=2)
    txt = work / "colmap" / "sparse" / "txt"
    images = parse_images_with_points(txt / "images.txt")
    if not images:
        rb.die(rb.EMPTY_INPUT, f"no registered images readable in {txt / 'images.txt'}",
               code=2)
    ids, xyz, err, tl = parse_points3d_with_tracks(txt / "points3D.txt")
    if len(ids) < MIN_OBSERVATIONS:
        rb.die(rb.EMPTY_INPUT, f"{txt / 'points3D.txt'} holds {len(ids)} points: too "
                              f"thin for any ruler comparison", code=2)
    frames = collect_frame_observations(images, ids, xyz, err, tl)
    by_name = {p["file"]: p for p in poses}
    # Only frames the pipeline extracted AND COLMAP registered are usable.
    usable = []
    for f in frames:
        p = by_name.get(f["name"])
        if p is None:
            continue
        c = p["camera"]
        f["t_sec"] = p.get("t_sec")
        f["fx"], f["fy"] = c.get("fx"), c.get("fy")
        usable.append(f)
    frame_json = rb.read_json(work / "frame.json", {}) or {}

    print(f"[depth] {len(usable)}/{len(frames)} registered frames carry keyframes "
          f"poses; {len(ids)} cloud points, mean track {tl.mean():.1f}")
    if not usable:
        rb.die(rb.EMPTY_INPUT, "no frame is both registered and in "
                              "keyframes_poses.jsonl - nothing to compare", code=2)

    if args.report_only:
        rows, stats = [], dict(model=args.model + " (from saved maps)",
                               resolution_level=None, fp16=None, device="none",
                               load_secs=0.0, frames_requested=len(usable), frames_ok=0)
        for f in usable:
            npz = out_dir / f"{Path(f['name']).stem}.npz"
            if not npz.is_file():
                continue
            pts, msk, intr = load_metric_map(npz)
            rad_map = np.linalg.norm(pts, axis=2).astype(np.float64)
            valid = msk & np.isfinite(rad_map) & (rad_map > 0)
            r_m, _spread, inside = sample_radius(rad_map, valid, f["xy"])
            f["radius_m"] = r_m[inside]
            f["radius_units_used"] = f["radius_units"][inside]
            h, w = msk.shape
            rows.append(dict(file=f["name"], status="ok", width=int(w), height=int(h),
                             observations=int(len(f["xy"])),
                             observations_on_map=int(inside.sum()),
                             fov=fov_diagnostic(intr, w, h, f.get("fx"), f.get("fy"))))
            stats["frames_ok"] += 1
    else:
        rows, stats = run_inference(usable, work, out_dir, model_id=args.model,
                                    resolution_level=args.resolution_level,
                                    fp16=args.fp16, save_maps=args.save_maps,
                                    limit=args.limit)

    have = [f for f in usable if len(f.get("radius_m", [])) >= PER_FRAME_MIN_OBS]
    all_m = np.concatenate([f["radius_m"] for f in have]) if have else np.empty(0)
    all_u = np.concatenate([f["radius_units_used"] for f in have]) if have else np.empty(0)
    # Per-frame fits use the relaxed gate on purpose: they exist to test whether the
    # frames agree with EACH OTHER, which a 500-observation bar would silence on a
    # sparse drone cloud. A frame that clears it still cannot set the headline.
    per_frame = []
    for f in have:
        one = fit_scale(f["radius_units_used"], f["radius_m"],
                        min_observations=PER_FRAME_MIN_OBS)
        per_frame.append(dict(file=f["name"], observations_used=one["observations_used"],
                              scale_m_per_unit=one["scale_m_per_unit"],
                              inliers=one["inliers"],
                              inlier_fraction=one["inlier_fraction"],
                              log_mad=one["log_mad"], status=one["status"]))
    fit = fit_scale(all_u, all_m, per_frame=per_frame)
    v = verdict(fit, frame_json)

    timings = [r["secs"] for r in rows if r.get("secs")]
    summary = dict(
        schema_version=1,
        step="depth_prior",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        status=fit["status"],
        purpose=("independent metric-scale check; advisory - this step writes nothing "
                 "the reconstruction reads"),
        reads=dict(keyframes_poses=str(work / "keyframes_poses.jsonl"),
                   colmap_sparse=str(txt), frame_json_read_only=True),
        writes=dict(depth_dir=str(out_dir),
                    maps=[r["file"] for r in rows if r.get("status") == "ok"] or None),
        model=dict(id=stats.get("model"),
                   licence=MODEL_LICENSE.get(args.model, ("licence checked at install "
                                                          "time", None))[0],
                   resolution_level=stats.get("resolution_level"),
                   fp16=stats.get("fp16"), device=stats.get("device")),
        run=dict(peak_vram_mib=stats.get("peak_vram_mib"),
                 peak_reserved_mib=stats.get("peak_reserved_mib"),
                 load_secs=stats.get("load_secs"),
                 frames_inferred=stats.get("frames_ok"),
                 frames_requested=stats.get("frames_requested"),
                 frames_in_fit=len(have),
                 secs_per_frame_median=round(float(np.median(timings)), 3) if timings else None,
                 secs_per_frame_max=round(float(np.max(timings)), 3) if timings else None,
                 total_wall_secs=round(time.time() - started, 1)),
        fit=fit, verdict=v, per_frame=per_frame, rows=rows,
    )
    rb.write_json(out_dir / "summary.json", summary)
    rb.write_json(out_dir / "done.json", dict(step="depth_prior", status=fit["status"],
                                              secs=summary["run"]["total_wall_secs"]))
    s = v["moge_scale_m_per_unit"]
    e = v["existing_scale_m_per_unit"]
    print(f"[depth] {fit['status']}: MoGe {round(s, 4) if s is not None else None} m/unit"
          f" vs frame.json {round(e, 4) if rb.finite(e) else None} m/unit"
          f" -> {v['agreement']} (confidence {v['confidence']})")
    print(f"[depth]   {v['agreement_note']}")
    print(f"[depth]   {fit['observations_used']} px compared, inliers {fit['inliers']}"
          + (f" ({fit['inlier_fraction']:.1%}), robust scatter "
             f"+-{((math.exp(fit['log_mad']) - 1) * 100 if fit.get('log_mad') else 0):.0f}%"
             if fit.get("log_mad") else ""))
    if fit["status"] != "measured":
        print("[depth] NOT MEASURABLE: " + "; ".join(fit["reasons"]))
    print(f"[depth] wrote {out_dir / 'summary.json'}")
    return 3 if (args.strict and fit["status"] != "measured") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except rb.StepError as exc:
        print(f"[depth] {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)
