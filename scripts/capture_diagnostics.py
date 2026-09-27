"""Capture-style diagnostics: what did this clip actually do, and can SfM live with it?

Cheap probe pass per video (decodes a few dozen downscaled frame pairs):
  - sharpness (variance-of-Laplacian) percentiles -> motion-blur risk
  - ORB feature counts                            -> texture richness
  - consecutive-pair geometry: ORB matches ->
        homography inliers, essential-matrix rotation angle and normalized
        translation magnitude                     -> rotation- vs translation-dominant
  - illumination (challenge D3): shadow-suspect area, its change across the clip,
        frame level drift, patch NCC              -> does photometric normalisation
        have anything to remove here

Verdicts drive the "auto" preset: keyframe budget weighting, matcher plan,
Mapper.init_min_tri_angle, whether the matching copy gets flattened, and human
warnings ("you spun in place here").

Usage:
  python capture_diagnostics.py --video a.mp4 b.mp4 --out work/scene/diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:  # imported as scripts.capture_diagnostics by the tests, run as a script by the pipeline
    from scripts import survey_photometry as photometry
except ImportError:
    import survey_photometry as photometry

PROBE_PAIRS = 40          # pairs sampled across the clip
PROBE_MAX_SIDE = 640      # decode resolution for analysis
ORB_COUNT = 1500
NCC_PAIRS = 12            # adjacent-sample pairs scored for patch correlation
"""Bounded so the cost does not scale with clip length: the frames handed to
``probe_illumination`` are the stride sample the motion probe already took."""

SHADOW_AREA_TRIGGER = 0.25
"""Challenge D3 trigger, part 1: at least this fraction of a sampled frame's AREA
reading as a cast shadow, on the worst-decile of the clip. A fraction of pixels, not
a brightness, so it does not care about the camera's exposure, its gain or the
resolution the probe decoded at. ``shadow_suspects`` decides "shadow-like" by
comparing each block against the frame's OWN upper-quartile field level, so a clip
that is simply dim all over scores zero here."""

SHADOW_AREA_CHANGE_TRIGGER = 0.12
"""Part 2: and the suspect area must MOVE by at least this fraction of the frame
between the calmest and the worst-decile sampled frame (p90 - p10 of it). A dark
wall that sits in every frame is albedo, and flattening it only dulls the matching
copy; a shadow that crosses the scene is illuminance, and it lands on different
content in every view. One number cannot tell those apart - the module itself says
a large dark surface and a shadow are the same pixel pattern - so both are asked."""

EXPOSURE_DRIFT_WARN = 1.6
"""Warn (do not normalise) when the ratio between the brightest and the dimmest
sampled frame's MEAN grey level exceeds this. A whole-frame gain is exactly what
``flatten`` cannot remove - it divides by the field and multiplies by the field's own
mean, so a uniform gain cancels - so this is a shooting note, not a pipeline fix."""


def _percentiles(xs) -> dict:
    if not xs:
        return {}
    a = np.asarray(xs, np.float64)
    return {f"p{p}": round(float(np.percentile(a, p)), 2) for p in (5, 25, 50, 75, 95)}


# ---------------------------------------------------------------------------
# Scene signature: what is in front of the camera, and how is it being held?
#
# The preset table has eight capture styles and the motion classifier could only ever
# reach two of them, because `classify` keys off rotation-vs-translation ratios and an
# essential matrix normalises its translation: a drone at 12 m/s and a person at 1.2 m/s
# produce the same |t|. Nothing in the old probe measured the SCENE or the STEADINESS of
# the mount, which is where an aerial orbit, a blank-walled room, a corridor walked
# forward and a turntable object actually differ. These four numbers are computed from
# frames the motion probe has already decoded, so the cost is a few numpy ops per sample.
# ---------------------------------------------------------------------------

def sky_fraction(gray: np.ndarray) -> float:
    """Fraction of the upper frame that is bright AND locally flat: open sky.

    Deliberately not "bright": a white painted ceiling, a blown-out wall and an overcast
    sky are all bright. Sky is the part of the image that carries no structure at all, so
    both a luminance test and a local-contrast test have to hold, and only in the upper
    60% of the frame (water, snow and lit concrete sit low in frame and are not sky).
    This is the cheapest reliable indoor/outdoor line the footage itself can draw.
    """
    h, w = gray.shape
    g = gray.astype(np.float64)
    blur = cv2.blur(g, (9, 9))
    local_std = np.sqrt(np.maximum(
        cv2.blur(g * g, (9, 9)) - blur * blur, 0.0))
    lo, hi = np.percentile(g, [2, 98])
    span = max(hi - lo, 1.0)
    bright = g > lo + 0.78 * span
    flat = local_std < 0.045 * span
    band = np.zeros_like(bright)
    band[:int(0.60 * h)] = True
    return float((bright & flat & band).mean())


def horizon_score(gray: np.ndarray) -> float:
    """Strength of a single horizontal luminance transition, 0-1, sky-to-ground scale.

    An oblique outdoor view has one: the horizon. A room, a nadir map and a turntable
    object do not - their row-to-row changes are furniture, walls and texture, spread
    across the frame rather than concentrated on one line. Reported as the peak row
    divided by the mean of the rest, so a noisy frame cannot fake a horizon.
    """
    h = gray.shape[0]
    rows = gray.astype(np.float64).mean(axis=1)
    band = rows[int(0.15 * h):int(0.85 * h)]
    if len(band) < 6:
        return 0.0
    step = np.abs(np.diff(band))
    lo, hi = np.percentile(gray, [2, 98])
    span = max(hi - lo, 1.0)
    return float(np.sort(step)[-3:].mean() / (step.mean() + 1e-6)
                 * min(1.0, step.max() / (0.06 * span)))


def expansion_gain(dxs: list, dys: list, xs: list, ys: list,
                   cx: float, cy: float) -> float:
    """How much of the inter-frame motion radiates outward from the image centre.

    +1 is a camera driving forward (a corridor, a push-in): every feature moves away
    from the focus of expansion. ~0 is a lateral pass or an orbit. -1 is a pull-back.
    This is the one signal that separates "walk down a hall" from "circle a room"
    without knowing how fast either one was going.
    """
    if not dxs:
        return 0.0
    f = np.stack([np.asarray(dxs), np.asarray(dys)], 1)
    r = np.stack([np.asarray(xs) - cx, np.asarray(ys) - cy], 1)
    nf, nr = np.linalg.norm(f, axis=1), np.linalg.norm(r, axis=1)
    keep = (nf > 1e-3) & (nr > 1e-3)
    if keep.sum() < 20:
        return 0.0
    cosang = np.einsum("ij,ij->i", f[keep], r[keep]) / (nf[keep] * nr[keep])
    return float(np.median(cosang))


def mount_jitter(video: Path, start: int, n: int = 40) -> dict:
    """High-frequency shake, measured on `n` CONSECUTIVE frames from mid-clip.

    The stride sample cannot see this: at one frame every few seconds, a walking bob and
    a drone hover both look like smooth drift. A short burst at native frame rate can,
    because a hand-walked phone bobs at the step rate while a gimbal'd aerial shot does
    not. Reported in pixels of residual displacement after the smooth trajectory is
    subtracted, normalised by frame height so resolution cannot change the answer.
    """
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
        prev = None
        dxs, dys = [], []
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = PROBE_MAX_SIDE / max(h, w)
            small = cv2.resize(frame, (round(w * scale), round(h * scale)),
                               interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                try:
                    shift, _ = cv2.phaseCorrelate(prev.astype(np.float64),
                                                  gray.astype(np.float64))
                    dxs.append(float(shift[0]))
                    dys.append(float(shift[1]))
                except cv2.error:
                    pass
            prev = gray
    finally:
        cap.release()
    if len(dys) < 8:
        return {"measured": False, "reason": "too few consecutive frames to judge"}
    ser = np.asarray(dys, np.float64)
    # Residual after removing the trend over a 5-frame window: what a gimbal would not
    # have produced. The denominator is the whole travel, so a long smooth push-in does
    # not read as "shaky" merely because the camera moved a lot.
    trend = np.convolve(ser, np.ones(5) / 5.0, mode="same")
    resid = float(np.std(ser[2:-2] - trend[2:-2])) if len(ser) > 5 else 0.0
    travel = float(np.abs(np.diff(ser)).sum() + np.abs(ser[-1] - ser[0]))
    return {"measured": True, "frames": len(dys) + 1,
            "vertical_residual_px": round(resid, 3),
            "residual_per_frame_height": round(resid / max(prev.shape[0], 1), 5),
            "wander_px": round(travel, 2),
            "roughness": round(resid / max(travel, 1e-6), 3)}


def probe_illumination(frames) -> dict:
    """Measure challenge D3 on the stride sample the motion probe already took.

    Every number here comes out of ``survey_photometry``, so "variable lighting" has
    one definition in this codebase rather than two. What each one is a ratio of:

      ``shadow_area_*``   - fraction of the frame AREA that ``shadow_suspects`` flags,
                            which judges each block against that frame's own lit level:
                            scale- and gain-invariant, resolution-invariant.
      ``low_freq_contrast`` - (p95 - p5) of ``illumination_field`` over the field's own
                            mean: how much low-frequency signal ``flatten`` has to
                            remove. Reported, never decided on - the field estimator
                            cannot tell a shadow edge from a material edge, so on a
                            scene of big tonal areas (water beside rock) this is high
                            whether the light moved or not.
      ``exposure_drift_ratio`` - brightest frame mean over dimmest frame mean.
      ``ncc_*``   - ``photometric_consistency`` on neighbouring sampled frames, raw vs
                            flattened. Reported only: on a moving camera this scores
                            the scene change, not the light (measured on a real drone
                            clip: +0.35 raw -> +0.33 flattened, i.e. flattening *lowers*
                            it), and the module's own tests say NCC is not a level measure.

    Cost is two passes over frames the probe already decoded: about 0.05 s per sampled
    frame at 640 px, so roughly 2 s per clip.
    """
    frames = list(frames)
    if not frames:
        return {"frames_probed": 0, "variable_lighting": False,
                "reason": "no frames were sampled, so lighting was not measured"}
    if min(frames[0].shape) < photometry.MIN_SIDE:
        return {"frames_probed": 0, "variable_lighting": False,
                "reason": f"sampled frames are {frames[0].shape}, below the "
                          f"{photometry.MIN_SIDE} px the block grid needs, so lighting "
                          f"was not measured"}
    stats = [photometry.exposure_stats(f) for f in frames]
    means = np.array([s["mean"] for s in stats], dtype=np.float64)
    areas = np.array([photometry.shadow_suspects(f)["fraction"] for f in frames])
    fields = [photometry.illumination_field(f, sigma_cells=2.0) for f in frames]
    contrast = np.array([(float(np.percentile(f, 95)) - float(np.percentile(f, 5)))
                         / float(f.mean()) for f in fields])

    raw, flat = [], []
    step = max(1, (len(frames) - 1) // NCC_PAIRS)
    for i in range(0, len(frames) - 1, step):
        raw.append(photometry.photometric_consistency(frames[i], [frames[i + 1]],
                                                      flatten_first=False)["median"])
        flat.append(photometry.photometric_consistency(frames[i], [frames[i + 1]],
                                                        flatten_first=True)["median"])

    p10, p50, p90 = (float(np.percentile(areas, q)) for q in (10, 50, 90))
    spread = p90 - p10
    drift = float(means.max() / means.min()) if means.min() > 0 else None
    area_hit, change_hit = p90 >= SHADOW_AREA_TRIGGER, spread >= SHADOW_AREA_CHANGE_TRIGGER
    out = {
        "frames_probed": len(frames),
        "shadow_area_median": round(p50, 4),
        "shadow_area_p90": round(p90, 4),
        "shadow_area_spread": round(spread, 4),
        "low_freq_contrast_median": round(float(np.median(contrast)), 4),
        "exposure_drift_ratio": None if drift is None else round(drift, 3),
        "clipped_fraction_max": round(max(s["clipped_fraction"] for s in stats), 5),
        "ncc_neighbour_median": round(float(np.median(raw)), 4) if raw else None,
        "ncc_neighbour_flattened": round(float(np.median(flat)), 4) if flat else None,
        "variable_lighting": bool(area_hit and change_hit),
        "basis": (f"shadow-suspect AREA (fraction of the frame, against each frame's own "
                  f"lit level) reached {p90:.1%} of the frame on the worst decile of "
                  f"{len(frames)} sampled frames and moved by {spread:.1%} across the clip; "
                  f"{SHADOW_AREA_TRIGGER:.0%} and {SHADOW_AREA_CHANGE_TRIGGER:.0%} are the "
                  f"bars. Gain-invariant by construction, so this is not a brightness test."),
        "checks": {"area_over_bar": bool(area_hit), "changes_over_bar": bool(change_hit)},
        "use": ("drives the frames step: on -> COLMAP matches a copy of the keyframes with "
                "the low-frequency illumination field divided out (frames_match), while "
                "frames_train, the copy the splat is trained on, stays untouched"),
    }
    return out


def illumination_warnings(illum: dict) -> list[str]:
    """What the illumination probe says to a human, as opposed to to the pipeline."""
    w = []
    if not illum or illum.get("frames_probed", 0) == 0:
        return w
    drift = illum.get("exposure_drift_ratio")
    if drift is not None and drift >= EXPOSURE_DRIFT_WARN:
        w.append(f"Frame-to-frame mean level drifts by {drift:.2f}x across this clip. That "
                 "is a whole-frame gain (auto-exposure riding the sun or a cloud), and "
                 "flattening cannot remove it - it divides by the field and re-anchors on "
                 "the field's own mean, so a uniform gain cancels. Lock exposure and "
                 "auto-iris before flying; the model would otherwise bake the brightening "
                 "in as albedo.")
    if illum.get("clipped_fraction_max", 0) > 0.02:
        w.append(f"{illum['clipped_fraction_max']:.1%} of some frame sits on the black or "
                 "white rail: information the sensor threw away, which no normalisation "
                 "recovers. Expose for the highlights.")
    return w


def probe_pose_motion(path) -> dict:
    """What an AR pose log says the camera physically did, in metres and seconds.

    This is the only pre-run evidence of ABSOLUTE motion in the whole probe. An essential
    matrix normalises its translation, so two frames of video cannot tell a drone at
    12 m/s from a person at 0.3 m/s - and the difference between those two IS the
    scenario. A handset that ran ARCore / ARKit / Record3D recorded it, so where a log
    exists the question is answerable rather than guessable.

    `load_any` is the same reader `import_phone_poses` uses, on purpose: two parsers for
    one file format is how a scene ends up classified by one and solved by the other.
    """
    from poses_lib import load_any

    samples, fmt = load_any(Path(path))
    pos = np.array([s[1] for s in samples], dtype=float)
    tim = np.array([s[0] for s in samples], dtype=float)
    good = np.isfinite(pos).all(1) & np.isfinite(tim)
    pos, tim = pos[good], tim[good]
    if len(pos) < 3:
        return {"measured": False, "format": fmt,
                "reason": f"only {len(pos)} finite samples"}
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    # A teleported relocalisation is not a metre of walking; it would otherwise read as
    # the fastest, straightest path in the dataset and classify a room as a corridor.
    span = float(np.nanmax(tim) - np.nanmin(tim))
    moving = step[step < 5.0]
    return {
        "measured": True, "format": fmt, "samples": int(len(pos)),
        "path_m": round(float(moving.sum()), 1),
        "net_displacement_m": round(float(np.linalg.norm(pos[-1] - pos[0])), 1),
        "straightness": round(float(np.linalg.norm(pos[-1] - pos[0])
                                    / max(moving.sum(), 1e-6)), 3),
        "duration_s": round(span, 1),
        "mean_speed_m_per_s": round(float(moving.sum() / max(span, 1e-6)), 3),
        "height_range_m": round(float(pos[:, 1].max() - pos[:, 1].min()), 2),
        "height_range_note": "axis 1 of the log as stored; every SDK here writes it "
                             "as up, but no classifier input depends on it",
        "teleports_dropped": int((step >= 5.0).sum()),
    }


def probe_video(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps else 0.0

    stride = max(1, total // (PROBE_PAIRS + 1)) if total > PROBE_PAIRS else 1
    orb = cv2.ORB_create(nfeatures=ORB_COUNT)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    sharp, nfeat = [], []
    samples = []      # the stride sample (~41 frames), reused by the illumination pass
    sky, horiz = [], []
    flow_x, flow_y, ref_x, ref_y = [], [], [], []
    centre = (0.0, 0.0)
    """Optical centre of the probe-resolution frame, in the same pixels as the match
    coordinates. Assigned per frame (it is constant for one clip) and defaulted here so
    a video that decodes nothing reaches `expansion_gain` rather than a NameError."""
    pair_rows = []
    prev_gray = prev_kp = prev_des = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = PROBE_MAX_SIDE / max(h, w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        samples.append(gray)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        sky.append(sky_fraction(gray))
        horiz.append(horizon_score(gray))
        fh, fw = gray.shape
        centre = (fw / 2.0, fh / 2.0)
        kp, des = orb.detectAndCompute(gray, None)
        nfeat.append(len(kp))

        if prev_des is not None and des is not None and len(prev_kp) >= 8 and len(kp) >= 8:
            matches = bf.knnMatch(prev_des, des, k=2)
            good = [m for m, n in (pair for pair in matches if len(pair) == 2)
                    if m.distance < 0.75 * n.distance]
            row = {"t": round(idx / fps, 2)}
            if len(good) >= 8:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp[m.trainIdx].pt for m in good])
                flow_x.extend((pts2[:, 0] - pts1[:, 0]).tolist())
                flow_y.extend((pts2[:, 1] - pts1[:, 1]).tolist())
                ref_x.extend(pts1[:, 0].tolist())
                ref_y.extend(pts1[:, 1].tolist())
                H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
                inl_ratio = float(mask_h.sum()) / len(good) if mask_h is not None else 0.0
                flow_px = float(np.median(np.linalg.norm(pts2 - pts1, axis=1))) \
                    if len(good) else 0.0
                # rotation vs translation: decompose E with an assumed
                # focal (= width). |t| is up to scale but comparable.
                fx_assumed = float(gray.shape[1])
                K = np.array([[fx_assumed, 0, gray.shape[1] / 2],
                              [0, fx_assumed, gray.shape[0] / 2], [0, 0, 1]])
                rot_deg = trans_norm = np.nan
                try:
                    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC,
                                                     0.999, 3.0)
                    if E is not None and mask_e is not None and int(mask_e.sum()) >= 8:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask_e.copy())
                        rot_deg = float(np.degrees(np.arccos(
                            np.clip((np.trace(R) - 1) / 2, -1, 1))))
                        trans_norm = float(np.linalg.norm(t))
                except cv2.error:
                    pass
                row.update(inlier_ratio=round(inl_ratio, 3), flow_px=round(flow_px, 2),
                           rot_deg=None if np.isnan(rot_deg) else round(rot_deg, 2),
                           trans_norm=None if np.isnan(trans_norm) else round(trans_norm, 4))
            if len(row) > 1:
                pair_rows.append(row)
        prev_kp, prev_des = kp, des
        idx += 1

        # Fast demuxer grab for skipped interval frames
        for _ in range(stride - 1):
            if not cap.grab():
                break
            idx += 1
    cap.release()

    rot = [r["rot_deg"] for r in pair_rows if r.get("rot_deg") is not None]
    trn = [r["trans_norm"] for r in pair_rows if r.get("trans_norm") is not None]
    med_rot = float(np.median(rot)) if rot else 0.0
    med_trn = float(np.median(trn)) if trn else 0.0
    low_inl = [r for r in pair_rows if r.get("inlier_ratio", 0) < 0.30]

    rotation_dominant_pct = 0
    if pair_rows:
        rd_flags = [bool(r["rot_deg"] is not None and r["rot_deg"] > 3.0 and
                         r["trans_norm"] is not None and r["trans_norm"] < 0.02)
                    for r in pair_rows]
        rotation_dominant_pct = round(100 * sum(rd_flags) / len(rd_flags))
    weak_geo_pct = round(100 * len(low_inl) / len(pair_rows)) if pair_rows else 100

    blur_p = _percentiles(sharp)
    style = classify(med_rot, med_trn, rotation_dominant_pct, weak_geo_pct, blur_p)
    illumination = probe_illumination(samples)

    # Scene signature, from the same decoded frames. `orb_features` was already here and
    # unused for preset selection; per-pixel is the comparable form, because a 4K clip
    # and a 640 px clip with the same wall in front of them report very different counts.
    h_px = max((s.shape[0] for s in samples), default=1)
    w_px = max((s.shape[1] for s in samples), default=1)
    texture_per_10k = round(float(np.median(nfeat)) / max(h_px * w_px, 1) * 1e4, 2) \
        if nfeat else None
    signature = {
        "sky_fraction_median": round(float(np.median(sky)), 4) if sky else None,
        "sky_fraction_p90": round(float(np.percentile(sky, 90)), 4) if sky else None,
        "horizon_score_median": round(float(np.median(horiz)), 3) if horiz else None,
        "expansion_gain": round(expansion_gain(flow_x, flow_y, ref_x, ref_y,
                                               *[centre[0], centre[1]]), 3)
                          if flow_x else None,
        "texture_features_per_10k_px": texture_per_10k,
        "frame_px": [w_px, h_px],
        "mount": mount_jitter(video, max(0, total // 2)),
    }

    warnings = warnings_for(style, rotation_dominant_pct, weak_geo_pct, blur_p, nfeat)
    warnings += illumination_warnings(illumination)
    if illumination.get("variable_lighting"):
        warnings.append("Lighting moves ACROSS the frame here (see the illumination block): "
                        "the run matches a flattened copy of the keyframes and trains the "
                        "colour off the raw ones.")

    return {
        "clip": video.name,
        "duration_s": round(dur, 1), "fps": round(fps, 2), "frames": total,
        "sharpness": blur_p, "orb_features": _percentiles(nfeat),
        "median_pair_rot_deg": round(med_rot, 2),
        "median_pair_trans": round(med_trn, 4),
        "rotation_dominant_pct": rotation_dominant_pct,
        "weak_geometry_pct": weak_geo_pct,
        "pairs_probed": len(pair_rows),
        "style": style,
        "signature": signature,
        "illumination": illumination,
        "warnings": warnings,
        "motion_weight": motion_weight(blur_p, nfeat, med_trn),
    }


def classify(med_rot, med_trn, rot_dom_pct, weak_pct, blur_p) -> str:
    if weak_pct > 60:
        return "low_texture_or_blur"
    if rot_dom_pct > 40 and med_rot > 3.0:
        return "rotation_dominant"
    if med_rot < 1.0 and med_trn > 0.02:
        return "translation_sweep"      # healthy walk / drone push-forward
    if med_rot >= 1.0 and med_trn > 0.02:
        return "orbit_mixed"            # arcs around subject -- ideal
    return "static_or_unknown"


def warnings_for(style, rot_dom, weak, blur_p, nfeat) -> list[str]:
    w = []
    if style == "rotation_dominant":
        w.append("Mostly spinning in place (rotation >> translation). Pure rotation "
                 "carries no depth information: COLMAP cannot triangulate those "
                 "segments. Re-shoot moving sideways/arcs instead of pivoting.")
    if weak > 60:
        w.append("Weak pairwise geometry (low texture or heavy blur). Add texture/"
                 "lighting, slow down, lock exposure.")
    p25 = blur_p.get("p25")
    if p25 is not None and p25 < 30:
        w.append("Low sharpness (blur p25 < 30). Steadier movement / more light / "
                 "higher shutter will help.")
    f25 = (_percentiles(nfeat) or {}).get("p25")
    if f25 is not None and f25 < 80:
        w.append("Few ORB features on many frames (textureless walls?). COLMAP may "
                 "struggle; consider better light or adding objects/posters.")
    return w


def motion_weight(blur_p, nfeat, med_trn) -> float:
    """Per-clip weight for splitting the global keyframe budget."""
    q = blur_p.get("p50", 0.0)
    f = (_percentiles(nfeat) or {}).get("p50", 0.0)
    s = min(max(q / 120.0, 0.25), 1.5) * min(max(f / 400.0, 0.25), 1.5)
    return round(float(s), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, action="append", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    reports = []
    for v in args.video:
        print(f"[diag] probing {v.name} ...", flush=True)
        reports.append(probe_video(v))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"clips": reports}, indent=2), encoding="utf-8")

    for r in reports:
        print(f"\n=== {r['clip']} ===  {r['duration_s']}s @ {r['fps']}fps, style={r['style']}")
        print(f"  sharpness {r['sharpness']}  ORB {r['orb_features']}")
        print(f"  median pair: rot={r['median_pair_rot_deg']}deg  |t|={r['median_pair_trans']}  "
              f"rot-dominant {r['rotation_dominant_pct']}%  weak-pairs {r['weak_geometry_pct']}%")
        i = r.get("illumination", {})
        print(f"  illumination: {i.get('frames_probed')} frames probed, shadow area "
              f"median {i.get('shadow_area_median')} p90 {i.get('shadow_area_p90')} "
              f"spread {i.get('shadow_area_spread')}, low-freq contrast "
              f"{i.get('low_freq_contrast_median')}, exposure drift "
              f"{i.get('exposure_drift_ratio')}x -> "
              f"{'normalise the matching copy' if i.get('variable_lighting') else 'match the raw frames'}")
        for wmsg in r["warnings"]:
            print(f"  ! {wmsg}")
    print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
