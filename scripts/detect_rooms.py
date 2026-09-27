"""Find the room in an interior scan: floor, ceiling, walls, and the metres between them.

The product can drop a sofa into a scanned room but cannot say how big the room is,
how high the ceiling is, or how much clearance the sofa has at the wall (GAPS doc,
item B5). This step answers those from the reconstructed cloud and writes
`viewer_assets/rooms.json`. It is the sibling of `build_objects.py`: same cloud,
same splat normals, same footprint-clustering idea, same rule that "nothing found"
is a legitimate answer with a printed reason.

Why these methods, measured on this repo's own scenes (2026-09-26):

*   **Splat shortest-axis normals, not kNN-PCA normals.** kNN-PCA on the subset of
    points standing off the floor returned only 3.7% vertical normals (517 of 14000)
    on `room_w_jsonl`, where the same test on the whole cloud gives 31.3% and 32.9%
    on the two interiors: with the walls and furniture thinned out, the k-nearest
    neighbours of a wall splat are in empty air, so its covariance is not planar.
    `build_objects.py` already reads the normal off the splat's shortest axis and
    needs no neighbour search at all, so this does too.
*   **The floor is found by a height histogram, then fitted as a plane.** A
    constant-y slab test loses a third of the floor because gravity alignment is
    not exact: `frame.json` reports `ground_tilt_deg` 5.94 (room_w_jsonl) and 9.68
    (roomscan), and the fitted floor normals measure 4.7 deg and 7.2 deg off
    vertical on the actual cloud. Free-normal RANSAC maximises inliers with no
    spatial coherence and won an 8.3 x 9.7 m "floor" holding 8% of its own bounding
    box over a room whose observed floor is 3.8 x 3.9 m. Histogram-then-least-squares
    gives 51001 splats at 4.7 deg and 4.1 cm rms with 79% of the cells in one patch.
    So: histogram for the candidate, plane fit for the numbers, occupancy for the
    truth test.
*   **Walls are plane fits, not blob clusters.** Height-agreement footprint
    clustering (the build_objects rule) welded the whole wall band of room_w_jsonl
    into one 48886-splat cluster whose fitted plane was 9.7 deg off *horizontal*
    with 0.437 m rms — not a wall. The planes that really are walls are tight: the
    two this step ships are 21685 splats at rms 0.048 m and 8443 at rms 0.045 m, and
    they meet at an interior angle of 92.3 deg, which is a room corner. So each wall
    is extracted as an (azimuth, offset) plane with a residual test, and a blob is
    rejected by `|n.y| < 0.2` plus `rms <= WALL_RMS_MAX`.

What this step refuses, on purpose:

*   **Ceiling height.** Neither interior has one. Horizontal splats above 1.2 m that
    sit over the observed floor number 1978 of 216496 (0.9%) on room_w_jsonl, and
    their tallest 30 cm slab holds 25% of them and covers 8% of the floor — that is
    wall tops and door headers, not a ceiling. `ceiling.height_m` is `null` with a
    reason. It is never a guess off the mean height and never a zero: the walls being
    observed up to 2.98 m is the scan's limit, not a measured ceiling.
*   **Wall clearance vs coverage edge.** `clearance` is perpendicular distance from
    a fitted wall plane. The coverage grid's own edge is a *visibility* boundary
    that happens to sit near walls in a good scan and nowhere near them in a bad
    one; it is reported next to the clearance for comparison and is never used as it.
*   **Rooms that are not rooms.** A scanned floor is one coherent sheet: the largest
    connected component of the floor occupancy covers 79% (room_w_jsonl) and 96%
    (roomscan) of all its floor cells. A drone scene's "floor" is terrain or terraced
    seating and fragments: 3% across 416 patches (rocks) and 19% across 362 patches
    (auditorium). `FLOOR_COHERENT_MIN` is set between those two measured groups, and a
    scene under it yields zero rooms with its counts printed.

Units: `viewer_assets/*` coordinates are **world metres** — `export_viewer_assets.py`
writes sparse points as `(pts_colmap @ Rg.T) * scale_m_per_unit` and the same for the
splat centres, so `collision.json.content_bounds` equals `frame.json.region_box_m`
digit for digit. Areas are therefore m^2, but a metre is only as good as its anchor,
so every artefact carries `units.scale_source` and a `metric_confidence` verdict:
`anchored` for the AR pose-prior path the interiors solved on, `provisional` when the
anchor is one assumed dimension (a drone's flight speed x clip duration, or "camera
1.6 m above ground"), and `scene units` if no scale survived to frame.json.

  python detect_rooms.py --asset work/room_w_jsonl/viewer_assets
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

# --- thresholds. Every one is stated in rooms.json so a reader can re-derive it.
VOX = 0.10              # m — footprint cell; same grid build_objects.py uses
OPACITY_MIN = 0.15      # sigmoid opacity below this is a fog gaussian, not geometry
UP_FACE = 0.82          # |n.y| above this = horizontal disc (floor, ceiling, table top)
WALL_FACE = 0.35        # |n.y| below this = vertical disc (wall-ish)
FLOOR_BIN = 0.05        # m — height histogram resolution for the floor candidate
FLOOR_BAND = 0.20       # m — splats within this of the modal floor height seed the fit
FLOOR_TOL = 0.12        # m — RANSAC-style inlier distance for the fitted floor plane
FLOOR_RMS_MAX = 0.12    # m — a floor that is not flat to 12 cm is not a floor
FLOOR_MIN_AREA = 1.0    # m^2 — smaller than this and the "room" is a doormat
FLOOR_COHERENT_MIN = 0.40   # largest patch / all floor cells; see the docstring
FLOOR_MAX_TILT = 25.0   # deg off vertical for the floor normal; beyond it it is a slope
WALL_STAND_MIN = 0.30   # m above the floor: below this it is skirting and floor noise
WALL_STAND_MAX = 3.20   # m: above this it is roof line or a ceiling edge
WALL_TOL = 0.12         # m — inlier distance from the candidate wall plane
WALL_RMS_MAX = 0.10     # m — measured: real walls 0.037-0.087, welded blobs 0.23+
WALL_N_MIN = 150        # splats on a wall before it counts as observed
WALL_LEN_MIN = 0.80     # m along the wall
WALL_HEIGHT_MIN = 0.90  # m from the floor to the top of its splats
WALL_TILT_MAX = 12.0    # deg from true vertical
WALL_AZ_BIN = 3.0       # deg — splat-normal azimuth bins; 5 deg split one wall across
WALL_OFF_BIN = 0.05     # m  — three bins = the +-12 cm search width
WALL_MERGE_AZ = 8.0     # deg — two candidates this close in azimuth are one wall
WALL_MERGE_OFF = 0.25   # m   — ...and this close in offset
WALL_NEAR = 2.5         # m — a wall must be this close to the observed floor to bound it
CEILING_MIN_H = 1.20    # m above the floor: below this a "ceiling" is a table
CEILING_MAX_H = 4.50    # m: above this it is not a room ceiling
CEILING_TOL = 0.15      # m — slab thickness for the ceiling candidate
CEILING_SLAB_MIN = 0.30   # share of the high horizontal splats in the modal slab
CEILING_COVER_MIN = 0.25  # and that slab must span this fraction of the floor area

VERTEX_PROPS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3")


def require_props(arr, names, src: Path) -> None:
    """A vertex table without these columns cannot give a normal or a position."""
    missing = [n for n in names if n not in (arr.dtype.names or ())]
    if missing:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{src.name} has no {', '.join(missing)} property, so neither a surface "
            f"normal nor a position can be measured.\n"
            f"  export_viewer_assets.py owns the vertex table; re-run train + export.",
            returncode=3)


def splat_normals_and_centres(arr):
    """Per-splat surface normal (shortest axis) and centre, in world metres.

    Identical derivation to build_objects.py so the two steps agree about which way
    a splat is facing. A quaternion with zero or NaN norm has no axis at all: those
    splats are dropped rather than being called flat.
    """
    op = 1.0 / (1.0 + np.exp(-np.asarray(arr["opacity"], np.float64)))
    keep = op >= OPACITY_MIN
    P = np.stack([np.asarray(arr[k], np.float64) for k in ("x", "y", "z")], axis=1)[keep]
    SC = np.exp(np.stack([np.asarray(arr[k], np.float64) for k in
                          ("scale_0", "scale_1", "scale_2")], axis=1))[keep]
    q = np.asarray(np.stack([np.asarray(arr[k], np.float64) for k in
                             ("rot_0", "rot_1", "rot_2", "rot_3")], axis=1))[keep]
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    qn = np.where(np.isfinite(qn) & (qn > 1e-9), qn, np.nan)
    q = q / qn
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], axis=1)
    nrm = R[np.arange(len(R)), :, SC.argmin(axis=1)]
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    good = np.isfinite(P).all(axis=1) & np.isfinite(nrm).all(axis=1)
    return P[good], nrm[good], int((~keep).sum()), int((~good).sum())


def fit_plane(pts, *, up_axis=1, face_up=True, iters=3):
    """Least-squares plane through a point set, re-fitted on its own inliers.

    Returns (normal, centroid, rms_m, inlier_mask_relative). `face_up` flips the
    arbitrary SVD sign so the normal points away from the floor for a ceiling and
    up for a floor; splat normals carry no reliable sign, so this is a convention,
    never a measurement.
    """
    if len(pts) < 3:
        return None
    cen = pts.mean(axis=0)
    sel = np.ones(len(pts), bool)
    for _ in range(iters):
        use = pts[sel] if sel.any() else pts
        if len(use) < 3:
            return None
        _, sv, vt = np.linalg.svd(use - use.mean(axis=0), full_matrices=False)
        nrm = vt[2]
        if face_up and nrm[up_axis] < 0:
            nrm = -nrm
        if not face_up and abs(nrm[up_axis]) > 0.5:
            nrm = -nrm
        d = np.abs((pts - use.mean(axis=0)) @ nrm)
        rms_all = float(np.sqrt(np.mean(d ** 2)))
        new = d < max(2.5 * rms_all, 1e-6)
        if new.sum() < 3 or (new == sel).all():
            sel = new if new.sum() >= 3 else sel
            break
        sel = new
    use = pts[sel] if sel.any() else pts
    cen = use.mean(axis=0)
    _, sv, vt = np.linalg.svd(use - cen, full_matrices=False)
    nrm = vt[2]
    if face_up and nrm[up_axis] < 0:
        nrm = -nrm
    d = np.abs((pts - cen) @ nrm)
    return nrm, cen, float(np.sqrt(np.mean(d[sel] ** 2))) if sel.any() else 0.0, sel


def occupancy_mask(pts, origin, shape, vox=VOX):
    """Rasterise a point set into a plan-view boolean grid."""
    ny, nx = shape
    m = np.zeros((ny, nx), bool)
    gj = np.floor((pts[:, 0] - origin[0]) / vox).astype(np.int64)
    gi = np.floor((pts[:, 2] - origin[1]) / vox).astype(np.int64)
    ok = (gj >= 0) & (gj < nx) & (gi >= 0) & (gi < ny)
    m[gi[ok], gj[ok]] = True
    return m


def largest_component(mask):
    """The biggest connected blob in a boolean grid, plus its share of the cells."""
    from scipy import ndimage as ndi
    lab, count = ndi.label(mask, structure=np.ones((3, 3), int))
    if count == 0:
        return mask, 0.0, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    out = lab == keep
    return out, float(out.sum()) / max(1, int(mask.sum())), int(count)


def contour_polygon(mask, origin, vox=VOX, max_pts=40):
    """Outer boundary of a plan-view mask as an [x, z] ring in world metres.

    cv2 gives a proper 8-connected trace; without it the fallback is the axis-aligned
    box of the mask, which is labelled in the artefact so nobody reads a rectangle as
    a traced outline.
    """
    if not mask.any():
        return [], "none"
    try:
        import cv2
    except ImportError:
        xs = [origin[0] + np.floor((np.nonzero(mask)[1]).min()) * vox,
              origin[0] + (np.nonzero(mask)[1]).max() * vox + vox]
        zs = [origin[1] + np.floor((np.nonzero(mask)[0]).min()) * vox,
              origin[1] + (np.nonzero(mask)[0]).max() * vox + vox]
        return [[xs[0], zs[0]], [xs[1], zs[0]], [xs[1], zs[1]], [xs[0], zs[1]]], "bbox-fallback"
    u8 = (mask.astype(np.uint8)) * 255
    # pad so the trace is closed even when the mask touches its own border
    u8 = np.pad(u8, 1, constant_values=0)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], "none"
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    eps = 0.02 * peri
    while eps < 0.5 * peri:
        simp = cv2.approxPolyDP(c, eps, True)
        if len(simp) <= max_pts:
            break
        eps *= 1.6
    simp = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    ring = [[round(float(origin[1] + r * vox), 3),
             round(float(origin[0] + cc * vox), 3)] for r, cc in simp]
    return [[p[1], p[0]] for p in ring], "cv2-contour"


def detect_floor(P, nrm, stats):
    """Modal horizontal slab -> plane -> coherence test. Returns dict or None."""
    flat = nrm[:, 1] ** 2 > UP_FACE ** 2
    if int(flat.sum()) < 50:
        stats["no_horizontal_splats"] = int(flat.sum())
        return None
    ys = P[flat, 1]
    lo, hi = float(np.min(ys)), float(np.max(ys))
    nb = max(8, int((hi - lo) / FLOOR_BIN) + 1)
    hist, edges = np.histogram(ys, bins=nb, range=(lo, hi))
    mode = int(rb.safe_argmax(hist, 0, label="floor height histogram"))
    y0 = 0.5 * (edges[mode] + edges[mode + 1])
    band = flat & (np.abs(P[:, 1] - y0) < FLOOR_BAND)
    cand = P[band]
    fit = fit_plane(cand, face_up=True)
    if fit is None:
        return None
    nrm_f, cen, rms, sel = fit
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(nrm_f[1]))))))
    # keep only splats near the candidate slab AND not facing sideways: a wall bottom
    # edge is within 12 cm of the floor plane too, and calling it floor would tilt it.
    inl = ((np.abs((P - cen) @ nrm_f) < FLOOR_TOL)
           & (nrm[:, 1] ** 2 > UP_FACE ** 2)
           & ((P[:, 1] - y0) < 3.0 * FLOOR_BAND))
    F = P[inl]
    if len(F) < 50:
        stats["floor_inliers"] = int(len(F))
        return None
    fit2 = fit_plane(F, face_up=True)
    if fit2 is None:
        return None
    nrm_f, cen, rms, sel = fit2
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(nrm_f[1]))))))
    d = np.abs((F - cen) @ nrm_f)
    rms = float(np.sqrt(np.mean(d ** 2)))
    ox0 = float(np.min(F[:, 0])) - VOX
    oz0 = float(np.min(F[:, 2])) - VOX
    shp = (int(np.ptp(F[:, 2]) / VOX) + 3, int(np.ptp(F[:, 0]) / VOX) + 3)
    raw = occupancy_mask(F, (ox0, oz0), shp)
    try:
        from scipy import ndimage as ndi
        closed = ndi.binary_closing(raw, np.ones((3, 3), bool))
    except ImportError:
        closed = raw
    patch, share, ncomp = largest_component(closed)
    # Closing bridges 1-cell gaps, so the patch can spill onto cells that hold no
    # splat at all. Intersect it back with the real cells or "largest patch" can
    # exceed the floor it was cut from (measured on roomscan: 7.39 m2 of patch from
    # 6.06 m2 of floor), which is exactly the kind of number this artefact must not print.
    patch = patch & raw
    share = float(patch.sum()) / max(1, int(raw.sum()))
    area = float(raw.sum()) * VOX * VOX
    stats.update(floor_splats=int(len(F)), floor_rms=round(rms, 4),
                 floor_tilt_deg=round(tilt, 2), floor_cells=int(raw.sum()),
                 floor_components=ncomp, floor_coherent_share=round(share, 3))
    rejected = None
    if rms > FLOOR_RMS_MAX or tilt > FLOOR_MAX_TILT or area < FLOOR_MIN_AREA:
        rejected = (f"the horizontal slab is not a floor: rms {rms:.3f} m (max "
                    f"{FLOOR_RMS_MAX}), normal {tilt:.1f} deg off vertical (max "
                    f"{FLOOR_MAX_TILT}), area {area:.2f} m2 (min {FLOOR_MIN_AREA})")
    elif share < FLOOR_COHERENT_MIN:
        rejected = (f"floor is {ncomp} disconnected patches and the largest holds "
                    f"{100*share:.0f}% of the {area:.1f} m2 of floor cells (needs "
                    f"{100*FLOOR_COHERENT_MIN:.0f}% in one patch) — a scanned room floor "
                    "is one sheet; terrain and terraced seating fragment like this")
    # The fit is returned even when rejected, so the caller can still test for walls
    # and give a refusal with two independent measurements behind it, not one.
    return {"normal": nrm_f, "centroid": cen, "rms": rms, "tilt": tilt,
            "y": float(np.median(F[:, 1])), "inliers": int(len(F)),
            "mask_raw": raw, "mask_patch": patch, "share": share, "ncomp": ncomp,
            "origin": (ox0, oz0), "cells": raw, "area": area, "rejected": rejected}


def wall_candidates(P, nrm, floor, stats, rej=None):
    """Fit vertical planes to the splats standing on the floor.

    Returns a list of dicts. A candidate survives only if it is near-vertical, its
    splats are actually coplanar, and it is long and tall enough to be a wall.
    `rej` tallies every rejection reason so the artefact can say what was tried and
    discarded instead of printing a bare zero.
    """
    if rej is None:
        rej = {}
    up = floor["normal"]
    cen = floor["centroid"]
    h = (P - cen) @ up
    flat_in = np.abs((P - cen) @ up) < FLOOR_TOL
    vertical = (nrm[:, 1] ** 2 < WALL_FACE ** 2) & ~flat_in
    stand = vertical & (h > WALL_STAND_MIN) & (h < WALL_STAND_MAX)
    W = P[stand]
    stats["standing_vertical_splats"] = int(stand.sum())
    if len(W) < WALL_N_MIN:
        stats["walls_skipped"] = f"only {len(W)} vertical splats stand off the floor"
        return []
    az = np.degrees(np.arctan2(nrm[stand][:, 2], nrm[stand][:, 0])) % 180.0
    out = []
    nbins = int(180.0 / WALL_AZ_BIN)
    for b in range(nbins):
        a_lo = b * WALL_AZ_BIN
        keep = (az >= a_lo) & (az < a_lo + WALL_AZ_BIN)
        if int(keep.sum()) < WALL_N_MIN:
            continue
        Q = W[keep]
        th = np.radians(a_lo + 0.5 * WALL_AZ_BIN)
        mv = np.array([np.cos(th), 0.0, np.sin(th)])
        d = (Q - cen) @ mv
        o_lo, o_hi = float(np.min(d)), float(np.max(d))
        if not np.isfinite(o_lo) or o_hi - o_lo <= 0:
            continue
        noff = max(8, int((o_hi - o_lo) / WALL_OFF_BIN) + 1)
        hh, ee = np.histogram(d, bins=noff, range=(o_lo, o_hi))
        order = np.argsort(hh)[::-1]
        tried = []
        for j in order[:4]:
            if hh[j] < WALL_N_MIN:
                rej["no offset mode with enough splats"] = rej.get(
                    "no offset mode with enough splats", 0) + 1
                continue
            off = 0.5 * (ee[j] + ee[j + 1])
            if any(abs(off - t) < WALL_MERGE_OFF for t in tried):
                continue
            tried.append(off)
            inl = np.abs(d - off) < WALL_TOL
            Q2 = Q[inl]
            if len(Q2) < WALL_N_MIN:
                continue
            fit = fit_plane(Q2, face_up=False)
            if fit is None:
                rej["plane unfit"] = rej.get("plane unfit", 0) + 1
                continue
            nv, wc, rms, sel = fit
            if abs(float(nv[1])) > 0.2:          # it welded a horizontal blob
                rej["welded blob, normal not horizontal"] = rej.get(
                    "welded blob, normal not horizontal", 0) + 1
                continue
            if nv[1] > 0:
                nv = -nv
            nv2 = nv / max(float(np.linalg.norm(nv)), 1e-12)
            # Orient the horizontal normal toward the observed floor. A consumer then
            # reads  distance = p . normal + offset  in metres, positive inside the
            # room — which is what "clearance at the wall" needs, and is why the sign
            # is not left to whatever SVD returned.
            if float((cen - wc) @ nv2) < 0.0:
                nv2 = -nv2
            # one reference point for both the inlier test and the residual, or the
            # two are parallel planes a metre apart and every rms is nonsense.
            dd = np.abs((W - wc) @ nv2)
            inl2 = dd < WALL_TOL
            pts = W[inl2]
            if len(pts) < WALL_N_MIN:
                rej["refit inliers under min splats"] = rej.get(
                    "refit inliers under min splats", 0) + 1
                continue
            along = np.array([-nv2[2], 0.0, nv2[0]])
            s = (pts - wc) @ along
            hv = (pts - cen) @ up
            ln = float(np.max(s) - np.min(s))
            top = float(np.max(hv))
            base = float(np.min(hv))
            resid = float(np.sqrt(np.mean(np.abs((pts - wc) @ nv2) ** 2)))
            # observed fraction: 10 cm x 10 cm cells on the wall's own face
            ai = np.floor((s - np.min(s)) / VOX).astype(int)
            hi_ = np.floor((hv - base) / VOX).astype(int)
            if ln <= 0 or top - base <= 0:
                continue
            gw = np.zeros((int((top - base) / VOX) + 1, int(ln / VOX) + 1), bool)
            gw[np.minimum(hi_, gw.shape[0] - 1), np.minimum(ai, gw.shape[1] - 1)] = True
            obs = float(gw.sum()) / max(1, gw.size)
            key = round(a_lo + 0.5 * WALL_AZ_BIN, 1)
            if resid > WALL_RMS_MAX:
                rej["rms over the wall face"] = rej.get("rms over the wall face", 0) + 1
                stats.setdefault("rejected_wall_examples", []).append(
                    {"azimuth_deg": key, "offset_m": round(float(off), 2),
                     "splats": int(len(pts)), "fit_rms_m": round(resid, 3),
                     "length_m": round(ln, 2), "height_m": round(top - base, 2),
                     "why": f"rms {resid:.3f} m > {WALL_RMS_MAX} m"})
                continue
            if ln < WALL_LEN_MIN or top - base < WALL_HEIGHT_MIN:
                rej["too short or too low"] = rej.get("too short or too low", 0) + 1
                continue
            tilt = float(np.degrees(np.arcsin(min(1.0, abs(float(nv2[1]))))))
            if tilt > WALL_TILT_MAX:
                rej["not vertical enough"] = rej.get("not vertical enough", 0) + 1
                continue
            # must be near the observed floor to bound it at all
            dx, dz = _line_to_mask_distance(pts, floor)
            dist_floor = float(np.hypot(dx, dz))
            if dist_floor > WALL_NEAR:
                stats.setdefault("walls_far", []).append(
                    {"azimuth_deg": key, "offset_m": round(float(off), 2),
                     "splats": int(len(pts)), "fit_rms_m": round(resid, 3),
                     "length_m": round(ln, 2), "height_m": round(top - base, 2),
                     "distance_to_floor_m": round(dist_floor, 2)})
                rej["too far from the observed floor"] = rej.get(
                    "too far from the observed floor", 0) + 1
                continue
            out.append({"normal": nv2, "point": wc, "rms": resid, "inliers": int(len(pts)),
                        "length": ln, "height": top - base, "top": top, "base": base,
                        "tilt": tilt, "observed": obs, "along": along,
                        "offset": float(-np.dot(nv2, wc)),
                        "s_min": float(np.min(s)), "s_max": float(np.max(s)),
                        "dist_floor": float(dist_floor)})
    # merge duplicates across neighbouring azimuth bins
    merged = []
    for c in sorted(out, key=lambda c: -c["inliers"]):
        dupe = False
        for m in merged:
            dang = abs((float(np.degrees(np.arctan2(c["normal"][2], c["normal"][0])) % 180)
                        - float(np.degrees(np.arctan2(m["normal"][2], m["normal"][0])) % 180)
                        + 90) % 180 - 90)
            doff = abs(float((c["point"] - m["point"]) @ c["normal"]))
            if dang < WALL_MERGE_AZ and doff < WALL_MERGE_OFF:
                dupe = True
                break
        if not dupe:
            merged.append(c)
    stats["wall_planes_fitted"] = len(merged)
    stats["wall_rejections"] = rej
    return merged


def _line_to_mask_distance(pts, floor):
    """Plan-view gap from a wall's splats to the nearest observed floor cell.

    Returns (dx, dz) components of the shortest connector, so the caller can use the
    magnitude. A wall that does not come near the floor it is supposedly enclosing is
    a façade, a hedge row or a boulder face, and the room test should not accept it.
    """
    mask, origin = floor["mask_raw"], floor["origin"]
    ny, nx = mask.shape
    cell = np.argwhere(mask)
    if len(cell) == 0 or len(pts) == 0:
        return (np.inf, np.inf)
    fx = origin[0] + (cell[:, 1] + 0.5) * VOX
    fz = origin[1] + (cell[:, 0] + 0.5) * VOX
    wx = pts[:, 0]
    wz = pts[:, 2]
    # subsample both sets so this stays a millisecond, not a matrix
    rng = np.random.default_rng(0)
    if len(wx) > 3000:
        wx, wz = wx[rng.choice(len(wx), 3000, replace=False)], wz[rng.choice(len(wz), 3000, replace=False)]
    if len(fx) > 4000:
        idx = rng.choice(len(fx), 4000, replace=False)
        fx, fz = fx[idx], fz[idx]
    dx = wx[:, None] - fx[None, :]
    dz = wz[:, None] - fz[None, :]
    r2 = dx * dx + dz * dz
    k = int(np.argmin(r2))
    i, j = np.unravel_index(k, r2.shape)
    return (float(dx[i, j]), float(dz[i, j]))


def detect_ceiling(P, nrm, floor, walls, stats):
    """Is there a horizontal slab over the observed floor? Measure it or say null."""
    up, cen = floor["normal"], floor["centroid"]
    h = (P - cen) @ up
    flat = nrm[:, 1] ** 2 > UP_FACE ** 2
    mask, origin = floor["mask_raw"], floor["origin"]
    ny, nx = mask.shape
    gj = np.floor((P[:, 0] - origin[0]) / VOX).astype(int)
    gi = np.floor((P[:, 2] - origin[1]) / VOX).astype(int)
    inside = (gj >= 0) & (gj < nx) & (gi >= 0) & (gi < ny)
    over = np.zeros(len(P), bool)
    over[inside] = mask[gi[inside], gj[inside]]
    high = flat & over & (h > CEILING_MIN_H) & (h < CEILING_MAX_H)
    stats["ceiling_candidate_splats"] = int(high.sum())
    stats["ceiling_search_band_m"] = [CEILING_MIN_H, CEILING_MAX_H]
    if int(high.sum()) < 100:
        return None, (f"no ceiling observed: only {int(high.sum())} horizontal splats sit "
                      f"between {CEILING_MIN_H:.2f} and {CEILING_MAX_H:.2f} m above the "
                      "floor inside the observed floor outline, so the scan never looked "
                      "up far enough to measure one")
    hv = h[high]
    nb = max(8, int((float(np.max(hv)) - float(np.min(hv))) / CEILING_TOL) + 1)
    hist, edges = np.histogram(hv, bins=nb)
    mode = int(rb.safe_argmax(hist, 0, label="ceiling height histogram"))
    share = float(hist[mode]) / max(1, int(high.sum()))
    h_ceiling = 0.5 * (edges[mode] + edges[mode + 1])
    slab = high & (np.abs(h - h_ceiling) < CEILING_TOL)
    pts = P[slab]
    cover = 0.0
    if len(pts) >= 20:
        cm = occupancy_mask(pts, origin, (ny, nx))
        cover = float((cm & mask).sum()) / max(1, int(mask.sum()))
    stats["ceiling_modal_slab_share"] = round(share, 3)
    stats["ceiling_slab_cover_of_floor"] = round(cover, 3)
    stats["ceiling_slab_height_m"] = round(float(h_ceiling), 3)
    if share < CEILING_SLAB_MIN or cover < CEILING_COVER_MIN:
        wall_top = max([w["top"] for w in walls], default=0.0)
        return None, (
            f"no ceiling observed: {int(high.sum())} horizontal splats sit "
            f"{CEILING_MIN_H:.2f}-{CEILING_MAX_H:.2f} m above the floor inside the "
            f"observed floor outline, but the modal {CEILING_TOL*2:.2f} m band holds only "
            f"{100*share:.0f}% of them (needs {100*CEILING_SLAB_MIN:.0f}%) and spans "
            f"{100*cover:.0f}% of the floor area (needs {100*CEILING_COVER_MIN:.0f}%). "
            f"The walls are observed up to {wall_top:.2f} m above the floor, which is "
            "where the scan ran out, not a measured ceiling. A number here would be a "
            "guess.")
    fit = fit_plane(pts, face_up=True)
    if fit is None:
        return None, "the ceiling candidate could not be fitted"
    nv, cc, rms, _ = fit
    return {"normal": nv, "height_m": float(h_ceiling), "inliers": int(len(pts)),
            "rms_m": rms, "share": share, "cover": cover}, None


def room_clearances(floor, walls, objects):
    """Distances from real wall surfaces. Never from the coverage-grid edge."""
    res = []
    for w in walls:
        nv, off = w["normal"], w["offset"]
        along = w["along"]
        # 1) opposite wall: the parallel wall on the other side of this floor patch
        span = None
        for o in walls:
            if o is w:
                continue
            if abs(float(np.dot(o["normal"], nv))) > 0.94:
                sep = abs(float(o["offset"] - off))
                if span is None or sep < span:
                    span = sep
        # 2) furniture: nearest detected object box, from objects.json
        furn = None
        if objects:
            for b in objects:
                cx, cz = b.get("center_xz", [None, None])
                if cx is None or cz is None:
                    continue
                # the plane equation wall_candidates guarantees: positive inside the
                # room, so this is the same number a viewer would compute for a sofa
                perp = float(np.dot([cx, 0.0, cz], nv) + off)
                s = float(np.dot([cx, 0.0, cz] - w["point"], along))
                if perp < -0.05:                 # the box is behind the wall face
                    continue
                if not (w["s_min"] - 0.3 <= s <= w["s_max"] + 0.3):
                    continue
                # conservative: the box's largest horizontal half-extent, so a rotated
                # sofa is treated as if its corner pointed straight at the wall
                half = 0.5 * max(b.get("size", [0.0, 0.0, 0.0])[0],
                                 b.get("size", [0.0, 0.0, 0.0])[2])
                gap = perp - half
                if gap >= 0 and (furn is None or gap < furn):
                    furn = gap
        res.append({"wall_id": len(res),
                    "inward_normal_xz": [round(float(nv[0]), 4), round(float(nv[2]), 4)],
                    "to_opposite_wall_m": round(span, 3) if span is not None else None,
                    "to_nearest_detected_object_m": round(furn, 3) if furn is not None else None,
                    "opposite_wall_observed": span is not None,
                    "object_source": "objects.json" if objects else None,
                    "object_distance_basis": "perpendicular distance from the fitted wall "
                                             "plane to the nearest corner of the box's "
                                             "bounding square; not a path distance"})
    return res


def split_components(floor, walls):
    """Does a detected wall line actually cut the observed floor into separate rooms?

    A wall that only borders the floor (every perimeter wall does) must NOT split it.
    The test is therefore not "a wall line touches the floor cells" but: erase the
    cells the wall's own face covers, and the floor must fall into two pieces that are
    each at least a quarter of the bigger one. That is the only reading of "the
    geometry supports it" that a false positive cannot sneak through.
    """
    from scipy import ndimage as ndi
    cells, origin = floor["mask_patch"], floor["origin"]
    ii, jj = np.nonzero(cells)
    if len(ii) < 20 or not walls:
        return [cells], 0
    px = origin[0] + (jj + 0.5) * VOX
    pz = origin[1] + (ii + 0.5) * VOX
    best, splitting = [cells], 0
    for w in walls:
        nv, wc, al = w["normal"], w["point"], w["along"]
        d = (px - wc[0]) * nv[0] + (pz - wc[2]) * nv[2]
        s = (px - wc[0]) * al[0] + (pz - wc[2]) * al[2]
        on = (np.abs(d) < VOX) & (s >= w["s_min"] - VOX) & (s <= w["s_max"] + VOX)
        if on.sum() < 2:
            continue
        cut = cells.copy()
        cut[ii[on], jj[on]] = False
        lab, count = ndi.label(cut, structure=np.ones((3, 3), int))
        if count < 2:
            continue
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        # bincount is indexed by label, so label k has size sizes[k], not sizes[k-1]
        big = [k for k in range(1, count + 1) if sizes[k] >= 0.25 * sizes.max()]
        if len(big) >= 2:
            best = [lab == k for k in big]
            splitting += 1
    return best, splitting


def enclosure(walls, floor):
    """What the detected walls actually say about the room's shape.

    A pair of walls whose normals are near-parallel gives a distance across the room
    — a real dimension, from two fitted surfaces. A near-perpendicular pair gives a
    corner. Neither is invented: an opposing separation is only reported when the floor
    patch lies between the two planes, which is the difference between a room and two
    unrelated walls that happen to face the same way.
    """
    corners, opposing = [], []
    cen = floor["centroid"]
    ii, jj = np.nonzero(floor["mask_patch"])
    ox, oz = floor["origin"]
    px = ox + (jj + 0.5) * VOX
    pz = oz + (ii + 0.5) * VOX
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            na, nb = a["normal"], b["normal"]
            dot = float(abs(np.dot(na, nb)))
            dang = float(np.degrees(np.arccos(min(1.0, dot))))
            if dang > 80.0:                        # normals near-perpendicular -> a corner
                # inward normals 90 deg apart means the faces meet at 90 deg
                corners.append({"walls": [i, j],
                                "interior_angle_deg": round(180.0 - dang, 1)})
            elif dang < 8.0:                       # near-parallel -> a possible width
                sep = float(abs((b["point"] - a["point"]) @ na))
                da = (px - a["point"][0]) * na[0] + (pz - a["point"][2]) * na[2]
                db = (px - b["point"][0]) * nb[0] + (pz - b["point"][2]) * nb[2]
                between = float(np.mean((da > 0) & (db > 0)) if len(da) else 0.0)
                if between > 0.5:                  # the floor really is between them
                    opposing.append({"walls": [i, j], "separation_m": round(sep, 3),
                                     "floor_cells_between": round(between, 3)})
    return {"wall_count": len(walls), "corners": corners, "opposing_pairs": opposing,
            "width_m": (opposing[0]["separation_m"] if opposing else None),
            "note": ("no opposing wall pair was observed, so the room's width is not "
                     "measured" if not opposing else
                     "width is the perpendicular distance between two fitted wall planes")
            }


def sanity_checks(floor, walls, a):
    """Cross-check the measured floor against the artefacts that already claim a size."""
    out = []
    col = rb.read_json(a / "collision.json", {})
    if isinstance(col, dict) and floor is not None:
        nx, nz, cell = col.get("nx"), col.get("nz"), col.get("cell")
        if rb.finite(nx or 0, nz or 0, cell or 0) and cell:
            g_area = float(nx) * float(cell) * float(nz) * float(cell)
            out.append({
                "check": "collider grid extent vs measured floor",
                "collider_grid_m2": round(g_area, 2),
                "floor_observed_m2": round(floor["area"], 2),
                "ratio": round(floor["area"] / g_area, 3) if g_area else None,
                "note": ("the floor the cloud shows is smaller than the grid the "
                         "collider was built on: that grid is padded, and a scan only "
                         "shows the surface its camera path actually looked at"),
            })
        cb = col.get("content_bounds") or {}
        if isinstance(cb, dict):
            mn, mx = cb.get("min", []), cb.get("max", [])
            if len(mn) == 3 and len(mx) == 3:
                out.append({
                    "check": "floor inside content_bounds",
                    "content_bounds_xz_m": [round(float(mx[0] - mn[0]), 2),
                                            round(float(mx[2] - mn[2]), 2)],
                    "floor_bbox_xz_m": [
                        round(float(np.ptp(np.nonzero(floor["cells"])[1])) * VOX + VOX, 2),
                        round(float(np.ptp(np.nonzero(floor["cells"])[0])) * VOX + VOX, 2)],
                })
    cg = rb.read_json(a / "coverage_grid.json", {})
    if isinstance(cg, dict) and cg.get("bounds_min") and cg.get("bounds_max"):
        bm, bx = cg["bounds_min"], cg["bounds_max"]
        out.append({
            "check": "coverage grid extent (a VISIBILITY bound, not a wall)",
            "bounds_xz_m": [round(float(bx[0] - bm[0]), 2), round(float(bx[2] - bm[2]), 2)],
            "covered_pct": cg.get("covered_pct"),
            "never_used_for_clearance": True,
        })
    diag = 0.0
    if floor is not None and floor.get("cells") is not None:
        ii, jj = np.nonzero(floor["cells"])
        if len(ii):
            diag = float(np.hypot(np.ptp(ii) * VOX, np.ptp(jj) * VOX))
    for w in walls:
        if w["height"] > 4.0:
            out.append({"check": "wall height plausibility",
                        "wall_id": walls.index(w),
                        "height_m": round(w["height"], 2),
                        "verdict": "façade height, not a room ceiling height"})
        if diag and w["length"] > 2.5 * diag:
            out.append({"check": "wall length plausibility", "wall_id": walls.index(w),
                        "length_m": round(w["length"], 2),
                        "floor_patch_diagonal_m": round(diag, 2),
                        "verdict": "a wall longer than 2.5x the diagonal of the observed "
                                   "floor bounds no room; it is a façade or a scarp"})
    area = floor["area"] if floor else 0.0
    if area and not (1.0 <= area <= 400.0):
        out.append({"check": "floor area plausibility", "floor_observed_m2": round(area, 2),
                    "verdict": "outside the 1-400 m2 band a single scanned room falls in"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    a = args.asset
    if not a.is_dir():
        raise rb.StepError(rb.EMPTY_INPUT,
                          f"{a} is not a directory. export_viewer_assets.py writes "
                          f"work/<scene>/viewer_assets; point --asset at that.",
                          returncode=3)
    scene_ply = a / "scene.ply"
    if not scene_ply.exists():
        raise rb.StepError(rb.EMPTY_INPUT, f"{scene_ply} does not exist.\n"
                           f"  export_viewer_assets.py writes it; run export first.",
                           returncode=3)
    d = PlyData.read(str(scene_ply))["vertex"].data
    if len(d) == 0:
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"{scene_ply.name} holds 0 vertices — nothing to detect a "
                           f"room in.", returncode=3)
    require_props(d, VERTEX_PROPS, scene_ply)
    P, N, dropped_op, dropped_nan = splat_normals_and_centres(d)

    col = rb.read_json(a / "collision.json", {})
    cb = col.get("content_bounds") if isinstance(col, dict) else None
    bounds_note = "unbounded"
    if isinstance(cb, dict) and {"min", "max"} <= set(cb):
        lo = np.array(cb["min"], float)
        hi = np.array(cb["max"], float)
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo):
            ins = np.all((P >= lo) & (P <= hi), axis=1)
            P, N = P[ins], N[ins]
            bounds_note = "collision.json content_bounds"
    if len(P) < 200:
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"only {len(P)} usable gaussians survive inside "
                           f"{bounds_note} ({a / 'scene.ply'}); a room cannot be "
                           f"measured from that.", returncode=3)

    work = a.parent
    fr = rb.read_json(work / "frame.json", {})
    fr = fr if isinstance(fr, dict) else {}
    scale = fr.get("scale_m_per_unit")
    src = str(fr.get("scale_source", "unknown"))
    credible = ("pose-prior" in src.lower()) or ("pose prior" in src.lower())
    units = {
        "coordinate_frame": "viewer_assets world metres (y up)",
        "why": "export_viewer_assets.py multiplies every point by scale_m_per_unit "
               "before writing scene.ply / sparse_points.json, and collision.json's "
               "content_bounds equals frame.json's region_box_m exactly.",
        "scale_m_per_unit": float(scale) if rb.finite(scale) else None,
        "scale_source": src,
        "metric_confidence": ("anchored" if credible else
                              ("provisional" if rb.finite(scale) and scale > 0 else "none")),
        "length_unit": "m" if rb.finite(scale) and scale > 0 else "scene unit",
        "area_unit": "m2" if rb.finite(scale) and scale > 0 else "scene unit^2",
        "caveat": ("scale_m_per_unit came from " + src + "; " + (
            "AR pose priors are metric, so metres here are the same metres the "
            "furniture placement uses." if credible else
            "this anchor is one assumed dimension, not a surveyed one, so treat the "
            "metres as provisional and do not quote the area to a client."))
        if rb.finite(scale) and scale > 0 else
        "frame.json carries no usable scale_m_per_unit, so every number in this file "
        "is in scene units and squared scene units, not metres.",
    }

    stats = {"gaussians_used": int(len(P)), "dropped_low_opacity": int(dropped_op),
             "dropped_non_finite": int(dropped_nan), "bounds_filter": bounds_note}
    floor = detect_floor(P, N, stats)
    objects = []
    ojb = rb.read_json(a / "objects.json", {})
    if isinstance(ojb, dict) and isinstance(ojb.get("boxes"), list):
        objects = ojb["boxes"]
    stats["object_boxes_available"] = len(objects)

    walls, ceiling, rooms = [], (None, None), []
    if floor is not None:
        walls = wall_candidates(P, N, floor, stats)
        ceiling, why = detect_ceiling(P, N, floor, walls, stats)

    plausibility = []
    if floor is not None and floor.get("rejected"):
        plausibility.append("floor: " + floor["rejected"])

    comps, splits = ([], 0)
    if floor is not None and walls and not floor.get("rejected"):
        comps, splits = split_components(floor, walls)
    if floor is not None and walls and not floor.get("rejected"):
        cells = floor["cells"]
        origin = floor["origin"]
        obs_area = float(cells.sum()) * VOX * VOX
        if obs_area < 4.0:
            plausibility.append(
                f"observed floor {obs_area:.1f} m2 is below a plausible living room. "
                "This is the area the cloud actually shows, not the room's plan area — "
                f"only {len(walls)} wall(s) were found, so the outline is not closed "
                "and no plan area can be derived from it.")
        if obs_area > 400.0:
            plausibility.append(f"floor area {obs_area:.1f} m2 is larger than any room; "
                                "this reads as terrain, not an interior")
    for ri, comp in enumerate(comps):
            ring, ring_src = contour_polygon(comp, origin)
            patch_area = float(comp.sum()) * VOX * VOX
            clear = room_clearances(floor, walls, objects)
            rooms.append({
            "id": ri,
            "floor": {
                "plane_normal_xyz": [round(float(v), 4) for v in floor["normal"]],
                "plane_offset_m": round(float(-np.dot(floor["normal"], floor["centroid"])), 4),
                "height_m": round(float(floor["y"]), 3),
                "tilt_from_vertical_deg": round(float(floor["tilt"]), 2),
                "fit_rms_m": round(float(floor["rms"]), 4),
                "inlier_splats": int(floor["inliers"]),
                "observed_area_m2": round(obs_area, 2),
                "largest_coherent_patch_m2": round(patch_area, 2),
                "coherent_share_of_floor": round(float(floor["share"]), 3),
                "patches": int(floor["ncomp"]),
                "cell_m": VOX,
                "area_basis": ("count of 10 cm cells containing a floor splat, i.e. the "
                               "floor the scan actually saw. It is NOT the room's plan "
                               "area: it excludes unobserved floor and any floor under "
                               "furniture."),
                "outline_source": ring_src,
                "outline_xz": ring,
            },
            "ceiling": ({"height_m": round(float(ceiling["height_m"]), 3),
                         "clear_height_above_floor_m":
                             round(float(ceiling["height_m"]), 3),
                         "inlier_splats": int(ceiling["inliers"]),
                         "fit_rms_m": round(float(ceiling["rms_m"]), 4),
                         "slab_share_of_high_splats": round(float(ceiling["share"]), 3),
                         "slab_cover_of_floor": round(float(ceiling["cover"]), 3)}
                        if ceiling else
                       {"height_m": None, "clear_height_above_floor_m": None,
                        "status": "no_ceiling_observed",
                        "reason": why or "no ceiling candidate"}),
            "walls": [{"id": i,
                       "plane_normal_xyz": [round(float(v), 4) for v in w["normal"]],
                       "plane_offset_m": round(w["offset"], 4),
                       "plane_equation": "signed distance in metres = "
                                         "p . plane_normal_xyz + plane_offset_m; "
                                         "positive on the room side, so a point's "
                                         "wall clearance is that value",
                       "length_m": round(w["length"], 3),
                       "height_above_floor_m": round(w["height"], 3),
                       "top_m": round(w["top"], 3),
                       "observed_fraction_of_face": round(w["observed"], 3),
                       "inlier_splats": int(w["inliers"]),
                       "fit_rms_m": round(w["rms"], 4),
                       "tilt_from_vertical_deg": round(w["tilt"], 2),
                       "azimuth_deg": round(float(np.degrees(np.arctan2(w["normal"][2],
                                                                        w["normal"][0])) % 180), 1),
                       "distance_to_floor_patch_m": round(w["dist_floor"], 3)}
                      for i, w in enumerate(walls)],
            "clearance": clear,
            "enclosure": enclosure(walls, floor),
            "segmentation": {
                "rooms_found": len(comps),
                "walls_crossing_the_floor": splits,
                "note": ("one room: no detected wall line cuts the observed floor into "
                         "two substantial pieces, so there is no geometric evidence for "
                         "a second room and none was invented"
                         if splits == 0 else
                         f"{splits} detected wall line(s) cut the observed floor, so it "
                         f"is reported as {len(comps)} rooms"),
            },
            "not_measured": {
                "plan_area": (None if len(walls) >= 3 else
                              f"the room's plan area is not derivable from "
                              f"{len(walls)} wall(s): observed_area_m2 is what the cloud "
                              "shows, not the size of the room"),
                "opposite_walls": [i for i, c in enumerate(clear)
                                   if c["to_opposite_wall_m"] is None],
            },
        })
    if floor is not None and not floor.get("rejected") and not walls:
        stats["no_rooms_reason"] = ("a coherent floor was measured but no vertical plane "
                                    "passed the wall tests, so there is no enclosure to "
                                    "call a room")
    if floor is not None and floor.get("rejected"):
        stats["no_rooms_reason"] = floor["rejected"] + (
            f"; and {len(walls)} wall plane(s) passed the wall tests against that surface"
            if walls is not None else "")

    payload = {
        "schema_version": 1,
        "generator": "scripts/detect_rooms.py",
        "scene": work.name,
        "units": units,
        "thresholds": {
            "vox_m": VOX, "opacity_min": OPACITY_MIN, "up_face": UP_FACE,
            "wall_face": WALL_FACE, "floor_bin_m": FLOOR_BIN, "floor_band_m": FLOOR_BAND,
            "floor_inlier_tol_m": FLOOR_TOL, "floor_rms_max_m": FLOOR_RMS_MAX,
            "floor_min_area_m2": FLOOR_MIN_AREA, "floor_max_tilt_deg": FLOOR_MAX_TILT,
            "floor_coherent_min": FLOOR_COHERENT_MIN,
            "wall_stand_m": [WALL_STAND_MIN, WALL_STAND_MAX],
            "wall_inlier_tol_m": WALL_TOL, "wall_rms_max_m": WALL_RMS_MAX,
            "wall_min_splats": WALL_N_MIN, "wall_min_length_m": WALL_LEN_MIN,
            "wall_min_height_m": WALL_HEIGHT_MIN, "wall_max_tilt_deg": WALL_TILT_MAX,
            "wall_az_bin_deg": WALL_AZ_BIN, "wall_offset_bin_m": WALL_OFF_BIN,
            "wall_near_floor_max_m": WALL_NEAR,
            "wall_merge_az_deg": WALL_MERGE_AZ,
            "wall_merge_offset_m": WALL_MERGE_OFF,
            "ceiling_band_m": [CEILING_MIN_H, CEILING_MAX_H],
            "ceiling_slab_tol_m": CEILING_TOL,
            "ceiling_slab_min_share": CEILING_SLAB_MIN,
            "ceiling_slab_min_cover_of_floor": CEILING_COVER_MIN,
        },
        "rooms": rooms,
        "refusal": (None if rooms else {
            "rooms_found": 0,
            "floor": (floor["rejected"] if floor else
                      "no horizontal slab of at least 50 splats could be fitted"),
            "walls_found": len(walls or []),
            "standing_vertical_splats": stats.get("standing_vertical_splats"),
            "wall_rejections": stats.get("wall_rejections", {}),
        }),
        "sanity": sanity_checks(floor, walls, a),
        "diagnostics": stats,
        "plausibility": plausibility,
        "not_measured": {
            "wall_clearance_is_not": "the coverage-grid edge is a visibility boundary, "
                                     "not a wall; clearance here is measured from a "
                                     "fitted wall plane only",
            "ceiling": "null wherever the scan produced no horizontal slab over the floor",
            "room_plan_area": "only derivable when opposing walls are both detected",
        },
    }
    out = args.out or (a / "rooms.json")
    rb.write_json(out, payload)

    print(f"[rooms] {len(P)} gaussians inside {bounds_note} "
          f"({dropped_op} low-opacity, {dropped_nan} non-finite dropped)")
    if floor is None:
        print("[rooms] NO FLOOR MEASURED — no horizontal slab of at least 50 splats "
              f"could be fitted (horizontal splats: {stats.get('no_horizontal_splats')})")
    else:
        print(f"[rooms] floor at y={floor['y']:+.2f} m, {floor['inliers']} splats, "
              f"tilt {floor['tilt']:.1f} deg, rms {floor['rms']:.3f} m, "
              f"{floor['area']:.2f} m2 observed in "
              f"{floor['ncomp']} patches (largest {100*floor['share']:.0f}% of cells)"
              + (f"  [REJECTED: {floor['rejected']}]" if floor.get("rejected") else ""))
    print(f"[rooms] {len(walls)} wall plane(s); "
          + "; ".join(f"az {float(np.degrees(np.arctan2(w['normal'][2], w['normal'][0]))) % 180:.0f}deg "
                      f"{w['length']:.2f} x {w['height']:.2f} m "
                      f"{100*w['observed']:.0f}% seen, rms {w['rms']:.3f} m, "
                      f"{w['inliers']} splats" for w in walls) if walls
          else "[rooms] 0 wall planes passed the tests")
    if rooms:
        c = rooms[0]["ceiling"]
        print(f"[rooms] ceiling: " + (f"{c['clear_height_above_floor_m']:.2f} m clear, "
                                      f"{c['inlier_splats']} splats"
                                      if c.get("height_m") is not None
                                      else f"null — {c['reason']}"))
        print(f"[rooms] floor area {rooms[0]['floor']['observed_area_m2']:.2f} m2, "
              f"largest coherent patch "
              f"{rooms[0]['floor']['largest_coherent_patch_m2']:.2f} m2, "
              f"units {units['area_unit']} ({units['metric_confidence']}: {src})")
    else:
        print(f"[rooms] 0 rooms. {stats.get('no_rooms_reason') or stats.get('floor_rejected') or 'no floor'}")
        if stats.get("wall_rejections"):
            tally = ", ".join(f"{v} x {k}" for k, v in
                              sorted(stats["wall_rejections"].items(),
                                     key=lambda kv: -kv[1]))
            print(f"[rooms] wall candidates rejected: {tally}")
            for ex in (stats.get("rejected_wall_examples") or [])[:6]:
                print(f"          az {ex['azimuth_deg']:5.1f} off {ex['offset_m']:+6.2f} m "
                      f"{ex['splats']:6d} splats len {ex['length_m']:5.2f} m "
                      f"h {ex['height_m']:4.2f} m — {ex['why']}")
            for ex in (stats.get("walls_far") or [])[:4]:
                print(f"          az {ex['azimuth_deg']:5.1f} {ex['splats']:6d} splats, "
                      f"rms {ex['fit_rms_m']:.3f} m, {ex['length_m']:.2f} m long but "
                      f"{ex['distance_to_floor_m']:.2f} m from the observed floor")
    for p in plausibility:
        rb.warn(f"[rooms] {p}")
    print(f"[rooms] wrote {out}")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[rooms] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
