"""Capture-style diagnostics: what did this clip actually do, and can SfM live with it?

Cheap probe pass per video (decodes a few dozen downscaled frame pairs):
  - sharpness (variance-of-Laplacian) percentiles -> motion-blur risk
  - ORB feature counts                            -> texture richness
  - consecutive-pair geometry: ORB matches ->
        homography inliers, essential-matrix rotation angle and normalized
        translation magnitude                     -> rotation- vs translation-dominant

Verdicts drive the "auto" preset: keyframe budget weighting, matcher plan,
Mapper.init_min_tri_angle, and human warnings ("you spun in place here").

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

PROBE_PAIRS = 40          # pairs sampled across the clip
PROBE_MAX_SIDE = 640      # decode resolution for analysis
ORB_COUNT = 1500


def _percentiles(xs) -> dict:
    if not xs:
        return {}
    a = np.asarray(xs, np.float64)
    return {f"p{p}": round(float(np.percentile(a, p)), 2) for p in (5, 25, 50, 75, 95)}


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
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
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
        "warnings": warnings_for(style, rotation_dominant_pct, weak_geo_pct, blur_p, nfeat),
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
        for wmsg in r["warnings"]:
            print(f"  ! {wmsg}")
    print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
