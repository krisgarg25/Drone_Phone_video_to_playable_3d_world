"""Geometric + colour semantic labelling of a reconstructed point cloud.

Classifies each 3D point into ground / road / building / vegetation / obstacle
using only surface normals (local PCA), height above the ground plane, planarity
and a colour vegetation index. Deliberately dependency-light (numpy + scipy, both
present offline) so it runs on any machine that already ran the pipeline, and
deterministic so it is unit-testable. It is a heuristic classifier: labels are a
navigational and analytical aid, not a surveyed ground truth.

Challenge D6 added the learned half. When `scripts/learned_semantics.py` (the
PY310 step: GroundingDINO-tiny + SAM 2.1, both Apache-2.0) has written per-keyframe
class masks into `work/<scene>/learned/`, this step back-projects them onto the
cloud and fuses them with the geometry. The rules are explicit because the last
thing this project needs is a learned label quietly rewriting a measured one:

  * every point carries `provenance`: "heuristic" (no learned evidence reached it),
    "both" (learned and geometric chose the same class), or "learned" (only the
    model had an opinion that this policy accepts). The heuristic is the fallback
    wherever the learned path has no evidence, which on a thin cloud is most points.
  * by default (`--conflict specificity`) a learned label may only replace the two
    classes the geometric ladder cannot genuinely claim - `obstacle`, its catch-all,
    and `ground`, its "flat thing below you" - and never a positive geometric claim
    like `building` or `road`. Every other collision is counted in `disagreements`
    and left alone. `--conflict heuristic` reports without ever changing a label;
    `--conflict learned` lets the model win everywhere.
  * learned evidence is gated on a per-point view count, a vote share, a grazing
    angle and a coarse depth-consistency test, because a mask is 2-D: without those
    a person in the foreground paints every wall behind them.
  * `vehicle`, `person`, `animal`, `furniture` and `water` are new classes the
    heuristic cannot produce at all - exactly the gap `survey_dynamics.py:1196`
    names ("a SAM2-class segmenter would name vehicles, humans and animals").
Everything about the learned path is reported in `semantics.json["learned"]`,
including the reason it was skipped when it was.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

CLASSES = ("ground", "road", "building", "vegetation", "obstacle")
# Classes only the learned path can name. Kept separate from CLASSES so the
# heuristic ladder's output set is unchanged and its unit tests still mean what
# they say.
LEARNED_ONLY = ("vehicle", "person", "animal", "furniture", "water")
ALL_CLASSES = CLASSES + LEARNED_ONLY
CLASS_RGB = {
    "ground": (124, 108, 84),
    "road": (150, 152, 158),
    "building": (214, 138, 60),
    "vegetation": (62, 168, 84),
    "obstacle": (206, 74, 74),
    "vehicle": (78, 120, 226),
    "person": (202, 60, 176),
    "animal": (176, 141, 60),
    "furniture": (86, 190, 196),
    "water": (40, 96, 190),
}

# --- Fusion gates. All dimensionless: counts, ratios and cosines. ------------
# A single view can be a detector hallucination; two independent keyframes voting
# the same class for the same 3-D point is the cheapest real corroboration
# available, since each must also survive its own depth-consistency test.
LEARNED_MIN_VIEWS = 2
# The winning class must hold at least this share of a point's learned votes. A
# point seen half as "tree" and half as "car" has no learned answer, and the
# heuristic keeps it - which is the honest result, not a coin flip.
LEARNED_MIN_SHARE = 0.5
# |cos(angle between the surface normal and the view ray)| under this is a grazing
# view, where a mask boundary one pixel wide already projects metres along the
# surface, and the label belongs to whatever the ray passed through first.
LEARNED_MIN_FACING = 0.2
# A point may sit at most this much further from the camera than the nearest cloud
# point in the same coarse mask cell, before its label is treated as leaked through
# that nearer surface. Relative, so it is unit-free like every other gate here.
LEARNED_DEPTH_SLACK = 1.25
# Cell size of that depth test, in mask pixels: 8 px at the default 0.5 mask scale
# is 16 keyframe pixels, i.e. ~2.5% of a 640-px frame.
LEARNED_DEPTH_CELL = 8
# How many keyframes to spend on the fusion (0 = all). Each frame costs one pass
# over the whole cloud, so a 480-frame interior take is the case this protects.
LEARNED_MAX_FRAMES = 0
# Classes a learned label is allowed to overwrite under `--conflict specificity`.
OVERRIDABLE = ("obstacle", "ground")
# Floors for `geometry_selfcheck`. Measured on work/rocks with a correct
# frame.json: cloud gap 1.5e-08 of the diagonal, median re-projection error 0.42 px
# (COLMAP's own mean per-point error there is 0.36 px, so this is the same number
# seen from the other side). Both bars are ~3 orders of magnitude above the
# measurement, so they fire on a wrong transform, not on float noise.
MAX_CLOUD_GAP = 1e-4        # fraction of the cloud diagonal
MAX_PIXEL_ERROR = 3.0       # pixels, vs COLMAP's own measured keypoints


def point_normals(points, k=12):
    """Unit surface normals from the smallest eigenvector of the kNN covariance."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return np.zeros((len(points), 3))
    k = max(3, min(int(k), len(points)))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    neighbours = points[idx]                                  # (N, k, 3)
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nji,njk->nik", centred, centred) / len(neighbours)
    _, vecs = np.linalg.eigh(cov)                              # ascending eigenvalues
    return vecs[:, :, 0]                                       # smallest-eigenvalue axis


def _eigen_planarity(points, k):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=max(3, min(k, len(points))))
    centred = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nji,njk->nik", centred, centred) / len(points[idx])
    vals = np.linalg.eigvalsh(cov)                             # ascending (s3<=s2<=s1)
    s1, s3 = vals[:, 2], vals[:, 0]
    return np.where(s1 > 0, (s1 - s3) / np.maximum(s1, 1e-12), 0.0)


def classify_points(points, colors, *, ground_height=0.0, k=12,
                    veg_index=0.12, ground_tol=None, building_height=1.0,
                    up_face=0.85, gray_chroma=22.0, planar=0.75, normals=None):
    """Return one class label per point. Y is up (viewer_assets metric frame).

    ground_tol defaults to a fraction of the cloud's vertical extent so the same
    rules behave on a 2 m room and a 20 m aerial scene; pass a number to override.
    `normals` lets the caller reuse the kNN normals it already computed (the D6
    fusion needs them for its grazing test); it is not otherwise recalculated.
    """
    points = np.asarray(points, float)
    colors = np.asarray(colors, float)
    n = len(points)
    if n == 0:
        return []
    if n < 4:
        return ["obstacle"] * n
    if ground_tol is None:
        ground_tol = float(np.clip(0.05 * (points[:, 1].max() - points[:, 1].min()), 0.30, 2.0))
    if normals is None:
        normals = point_normals(points, k)
    abs_n = np.abs(np.asarray(normals, float))
    verticality = abs_n[:, 1]                                # 1 = up-facing, 0 = wall
    planarity = _eigen_planarity(points, k)
    height = points[:, 1] - ground_height
    rgb = colors if colors.max() <= 255.001 else colors / 255.0
    if rgb.max() <= 1.0001:
        rgb = rgb * 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    exg = (2 * g - r - b) / 255.0                              # excess-green index
    chroma = rgb.max(axis=1) - rgb.min(axis=1)
    up = verticality > up_face
    green = exg > veg_index
    near = height <= ground_tol
    wall = (verticality < 0.5) & (height > building_height)
    gray = (chroma < gray_chroma) & (planarity > planar)

    labels = []
    for i in range(n):
        if green[i]:
            labels.append("vegetation")
        elif near[i] and up[i]:
            labels.append("road" if gray[i] else "ground")
        elif wall[i]:
            labels.append("building")
        else:
            labels.append("obstacle")
    return labels


def _ground_height(work):
    """Prefer the surveyed ground surface; fall back to the cloud's low percentile."""
    for name in ("ground.f32", "heights.f32"):
        path = work / "viewer_assets" / name
        meta = work / "viewer_assets" / "collision.json"
        if path.is_file() and meta.is_file():
            col = json.loads(meta.read_text())
            data = np.fromfile(path, dtype="<f4")
            if data.size == col.get("nx", 0) * col.get("nz", 0) and data.size:
                return float(np.nanpercentile(data, 10))
    return None


def load_cloud(work):
    path = work / "viewer_assets" / "sparse_points.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    pts = data.get("points") or data.get("xyz")
    cols = data.get("colors") or data.get("rgb") or []
    if not pts:
        return None
    pts = np.asarray(pts, float)
    cols = np.asarray(cols, float) if cols else np.full((len(pts), 3), 150.0)
    if len(cols) != len(pts):
        cols = np.full((len(pts), 3), 150.0)
    return pts, cols


# ---------------------------------------------------------------------------
# D6 fusion: back-project the learned per-image class masks onto the cloud.
# ---------------------------------------------------------------------------
def viewer_to_colmap(pts_viewer, rotation, scale):
    """Undo what export_viewer_assets did: p_colmap = (p_viewer / s) @ Rg.

    `semantics.json` is written in the viewer frame because that is what the viewer
    draws, but every camera pose and intrinsics this fusion needs lives in the
    COLMAP frame, so the cloud comes here rather than the poses going there.
    """
    return (np.asarray(pts_viewer, float) / float(scale)) @ np.asarray(rotation, float)


def load_learned(work: Path):
    """The learned step's output, or a reason it is absent. Never a silent zero.

    Returns (frames, classes, meta) on success, else (None, None, {reason:...}).
    """
    d = work / "learned"
    det = rb.read_json(d / "detections.json", None)
    if not isinstance(det, dict):
        return None, None, dict(status="absent", reason=(
            f"no learned/detections.json under {work}: run scripts/learned_semantics.py "
            f"on .venv310 first (the `learned` pipeline step)"))
    classes = det.get("label_classes") or []
    if not classes:
        return None, None, dict(status="invalid",
                                reason="learned/detections.json has no label_classes")
    frames = []
    for row in det.get("frames", []):
        name = row.get("file")
        stem = str(name).replace("/", "_").replace("\\", "_").rsplit(".", 1)[0]
        npz = d / "masks" / f"{stem}.npz"
        if npz.is_file():
            frames.append(dict(file=name, mask_path=npz, detections=len(row.get("detections") or [])))
    if not frames:
        return None, None, dict(status="no_masks", reason=(
            f"learned/detections.json lists {len(det.get('frames') or [])} frames but "
            f"learned/masks/ holds none: the learned step ran detection only (--no-masks)"))
    return frames, classes, dict(status="present", detections_file=str(d / "detections.json"))


def _cell_minimum(cell_flat, values, n_cells):
    """Per-cell minimum of `values`, without np.minimum.at's per-element cost."""
    out = np.full(n_cells, np.inf, np.float64)
    order = np.argsort(-values, kind="stable")          # far first, near last
    out[cell_flat[order]] = values[order]               # last write per cell wins
    return out


def geometry_selfcheck(work: Path, pts_c, poses, frames, *, sample_frames=8):
    """Prove the cloud being labelled and the images being masked are one geometry.

    `semantics.json`'s cloud lives in the viewer frame, and this step brings it back
    to COLMAP with `frame.json`'s rotation and scale. If either is stale, absent or
    from another solve, every back-projected mask lands on the wrong surface - and
    the output still looks plausible. So it is checked against COLMAP's own
    measurements instead of trusted:

      1. every cloud point must coincide with a points3D.txt point. The median gap,
         as a fraction of the cloud's own diagonal, is unit-free;
      2. re-projecting those points through each frame's pose must land on the pixel
         SfM actually measured that point at. Median error, in pixels.

    Returns (gap_fraction, median_pixel_error, reason); a non-None reason means the
    caller applies not one learned label.
    """
    try:
        from depth_prior import parse_images_with_points, parse_points3d_with_tracks
    except Exception as exc:                                   # noqa: BLE001
        return None, None, f"cannot import the COLMAP text-model readers ({exc})"
    txt = work / "colmap" / "sparse" / "txt"
    ids, xyz, _err, _tl = parse_points3d_with_tracks(txt / "points3D.txt")
    if len(ids) < 100:
        return None, None, f"{txt / 'points3D.txt'} holds only {len(ids)} points"
    images = parse_images_with_points(txt / "images.txt")
    if not images:
        return None, None, f"no registered images readable in {txt / 'images.txt'}"
    tree = cKDTree(xyz)
    gap, _row = tree.query(np.asarray(pts_c, float), k=1)
    diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    gap_fraction = float(np.median(gap)) / max(diag, 1e-9)
    if gap_fraction > MAX_CLOUD_GAP:
        return gap_fraction, None, (
            f"the viewer-frame cloud does not sit on this COLMAP model: median nearest-"
            f"point gap is {gap_fraction:.2e} of the cloud diagonal "
            f"(bar {MAX_CLOUD_GAP:.0e})")
    id2row = np.full(int(ids.max()) + 1, -1, np.int64)
    id2row[ids] = np.arange(len(ids))
    errs = []
    for rec in frames:
        if len(errs) >= sample_frames:
            break
        name = rec["file"]
        cam = poses.get(name)
        if not cam or name not in images or cam.get("fx") is None:
            continue
        obs = images[name]["obs"][:500]
        if len(obs) < 50:
            continue
        rows = id2row[np.clip(obs[:, 2].astype(np.int64), 0, len(id2row) - 1)]
        have = rows >= 0
        rows, src = rows[have], obs[have][:, :2]
        if len(rows) < 50:
            continue
        # `rows` indexes the points3D.txt array, not the exported cloud: the viewer
        # copy is pruned, so it is smaller and its positions are unrelated to COLMAP's
        # ids. Projecting it through COLMAP's R/t would also mix the Y-up metre frame
        # with the reconstruction's own units, which is a second way to be wrong.
        x = xyz[rows] @ np.array(cam["R_rowmajor"], float).T + np.array(cam["t"], float)
        front = x[:, 2] > 1e-9
        x, src = x[front], src[front]
        if len(x) < 50:
            continue
        u = float(cam["fx"]) * x[:, 0] / x[:, 2] + float(cam["cx"])
        v = float(cam["fy"]) * x[:, 1] / x[:, 2] + float(cam["cy"])
        errs.append(float(np.median(np.hypot(u - src[:, 0], v - src[:, 1]))))
    if not errs:
        return gap_fraction, None, ("no masked frame has a usable pose to check the "
                                    "projections against")
    med = float(np.median(errs))
    if med > MAX_PIXEL_ERROR:
        return gap_fraction, med, (
            f"re-projecting the cloud lands a median {med:.1f} px from where COLMAP "
            f"measured it (bar {MAX_PIXEL_ERROR} px): frame.json's rotation or scale "
            f"does not match the reconstruction the masks were made for")
    return gap_fraction, med, None


def apply_learned(work: Path, pts, normals_v, frames, classes, args):
    """Back-project the learned masks onto the cloud and vote each point a class.

    Returns (learned_label_or_None_per_point, info-for-semantics.json). Anything
    that could not be established comes back as an `info` reason and an all-None
    label list, never as a confident blank.
    """
    n = len(pts)
    src = rb.read_json(work / "learned" / "summary.json", {}) or {}
    info = dict(status="inactive", reason=None,
                models=(src.get("models") or {}).get("detector") and dict(
                    detector=(src["models"].get("detector")),
                    segmenter=(src["models"].get("segmenter")),
                    licences=src["models"].get("licences")))
    fr = rb.read_json(work / "frame.json", None)
    if not isinstance(fr, dict) or not rb.finite(fr.get("scale_m_per_unit")) \
            or fr.get("rotation_rowmajor") is None:
        info["reason"] = ("frame.json has no rotation/scale, so the viewer-frame cloud "
                          "cannot be brought back into the COLMAP frame the masks live in")
        return [None] * n, info
    Rg = np.array(fr["rotation_rowmajor"], float).reshape(3, 3)
    s = float(fr["scale_m_per_unit"])
    poses = {r["file"]: r.get("camera") or {} for r in
             rb.jsonl_rows(work / "keyframes_poses.jsonl", required=("file",))}
    if not poses:
        info["reason"] = "keyframes_poses.jsonl unreadable"
        return [None] * n, info

    pts_c = viewer_to_colmap(pts, Rg, s)
    gap, px_err, why = geometry_selfcheck(work, pts_c, poses, frames)
    info["geometry_check"] = dict(cloud_gap_fraction_of_diagonal=gap,
                                  median_reprojection_px=px_err,
                                  bars=dict(max_cloud_gap=MAX_CLOUD_GAP,
                                            max_pixel_error_px=MAX_PIXEL_ERROR))
    if why:
        info.update(status="rejected_geometry", reason=why)
        print(f"[semantics] learned labels REFUSED: {why}")
        return [None] * n, info
    normals_c = np.asarray(normals_v, float) @ Rg
    votes = np.zeros((len(classes), n), np.int32)
    used = skipped = 0
    votes_per_frame = []
    cell = LEARNED_DEPTH_CELL
    limit = args.max_fuse_frames if args.max_fuse_frames > 0 else len(frames)
    for rec in frames[:limit]:
        cam = poses.get(rec["file"])
        if not cam or cam.get("fx") is None:
            skipped += 1
            continue
        with np.load(rec["mask_path"]) as z:
            lab = z["label"]
            w, h = int(z["width"][0]), int(z["height"][0])
        mh, mw = lab.shape
        R = np.array(cam["R_rowmajor"], float).reshape(3, 3)
        t = np.array(cam["t"], float)
        x = pts_c @ R.T + t
        depth = x[:, 2]
        front = depth > 1e-9
        safe = np.where(front, depth, 1.0)
        u = (float(cam["fx"]) * x[:, 0] / safe + float(cam["cx"])) * (mw / float(w))
        v = (float(cam["fy"]) * x[:, 1] / safe + float(cam["cy"])) * (mh / float(h))
        inside = front & (u >= 0) & (u < mw) & (v >= 0) & (v < mh)
        if inside.sum() < 20:
            skipped += 1
            continue
        # Grazing views: a mask edge one pixel wide projects metres along a surface
        # seen side-on, and the label belongs to whatever the ray met first.
        radius = np.linalg.norm(x, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = x / np.where(radius > 0, radius, 1.0)[:, None]
        facing = np.abs(np.einsum("ij,ij->i", normals_c, d)) >= args.min_facing
        # Depth order, per coarse cell: only the near envelope of each cell may be
        # labelled by this frame, so a foreground object cannot paint the wall
        # behind it. Cell minima via one far-to-near scatter assignment.
        ncy, ncx = -(-mh // cell), -(-mw // cell)
        cell_flat = ((v // cell).astype(np.int64) * ncx + (u // cell).astype(np.int64))
        zbuf = np.full(ncy * ncx, np.inf)
        sel_all = np.flatnonzero(inside & facing)
        if sel_all.size == 0:
            skipped += 1
            continue
        zbuf = _cell_minimum(cell_flat[sel_all], radius[sel_all], zbuf.size)
        near = radius[sel_all] <= zbuf[cell_flat[sel_all]] * LEARNED_DEPTH_SLACK
        cls_at = lab[v[sel_all].astype(np.int64), u[sel_all].astype(np.int64)]
        keep = near & (cls_at > 0) & (cls_at <= len(classes))
        idx = sel_all[keep]
        if idx.size == 0:
            skipped += 1
            continue
        np.add.at(votes, (cls_at[keep] - 1, idx), 1)
        votes_per_frame.append(int(idx.size))
        used += 1

    total = votes.sum(axis=0)
    winner = votes.argmax(axis=0)
    won = votes[winner, np.arange(n)] if n else np.zeros(0, np.int32)
    ok = ((total >= args.min_views) & (won >= args.min_views)
          & (won >= total * LEARNED_MIN_SHARE))
    learned = [None] * n
    counts_by_class = {c: 0 for c in classes}
    for i in np.flatnonzero(ok):
        learned[i] = classes[int(winner[i])]
        counts_by_class[classes[int(winner[i])]] += 1
    med_support = float(np.median(supports)) if supports else None
    if med_support is not None and med_support < MIN_KEYPOINT_SUPPORT:
        info.update(status="rejected_geometry", reason=(
            f"only {med_support:.1%} of the cloud projects onto a COLMAP keypoint in the "
            f"masked frames (bar {MIN_KEYPOINT_SUPPORT:.0%}): the viewer-frame cloud and "
            f"the masked images are not the same geometry, so no learned label was applied"),
            frames_used=used, keypoint_support=round(med_support, 4))
        return [None] * n, info
    info.update(status="present", frames_with_masks=len(frames), frames_used=used,
                frames_skipped=skipped, points_with_any_vote=int((total > 0).sum()),
                points_passing_gates=int(ok.sum()),
                mean_views_per_point=round(float(np.mean(votes_per_frame)), 1)
                if votes_per_frame else None,
                keypoint_support_median=round(med_support, 4) if med_support else None,
                votes_by_class={c: int(votes[i].sum()) for i, c in enumerate(classes)},
                points_by_class=counts_by_class,
                gates=dict(min_views=args.min_views, min_share=LEARNED_MIN_SHARE,
                           min_facing=LEARNED_MIN_FACING,
                           depth_slack=LEARNED_DEPTH_SLACK, depth_cell=LEARNED_DEPTH_CELL))
    return learned, info


def fuse(labels_heuristic, learned_labels, *, policy="specificity"):
    """Combine the two opinions and record where they disagree. Pure and testable.

    Returns (final_labels, provenance, disagreements, counts_by_provenance).
    `labels_heuristic` is a list of class names; `learned_labels` is the same length
    with None where the learned path had no accepted answer.
    """
    final, prov = [], []
    disagreements: Counter = Counter()
    by_prov: Counter = Counter()
    for h, l in zip(labels_heuristic, learned_labels):
        if l is None:
            final.append(h)
            prov.append("heuristic")
            by_prov["heuristic"] += 1
            continue
        if l == h:
            final.append(h)
            prov.append("both")
            by_prov["both"] += 1
            continue
        disagreements[f"{h}->{l}"] += 1
        may = (policy == "learned"
               or (policy == "specificity" and h in OVERRIDABLE))
        if may and policy != "heuristic":
            final.append(l)
            prov.append("learned")
            by_prov["learned"] += 1
        else:
            final.append(h)
            prov.append("heuristic")
            by_prov["heuristic"] += 1
    return final, prov, disagreements, by_prov


def main():
    ap = argparse.ArgumentParser(description="Label the reconstructed point cloud semantically.")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--no-learned", action="store_true",
                    help="skip the D6 fusion even if learned masks exist")
    ap.add_argument("--conflict", choices=("specificity", "heuristic", "learned"),
                    default="specificity",
                    help="who wins when the model and the geometry disagree: "
                         "specificity (default) lets the model take only the "
                         "heuristic's catch-alls, heuristic never lets it change a "
                         "label, learned always lets it")
    ap.add_argument("--min-views", type=int, default=LEARNED_MIN_VIEWS,
                    help="keyframes that must agree before a learned label is accepted")
    ap.add_argument("--min-facing", type=float, default=LEARNED_MIN_FACING,
                    help="|normal . view| floor: grazing views are dropped")
    ap.add_argument("--max-fuse-frames", type=int, default=LEARNED_MAX_FRAMES,
                    help="cap the keyframes used for the fusion (0 = all)")
    ap.add_argument("--strict", action="store_true", help="non-zero if nothing learned")
    args = ap.parse_args()
    rb.configure_streams()
    work = Path(args.work)
    cloud = load_cloud(work)
    if cloud is None:
        print("[semantics] no sparse_points.json; skipping")
        return 0
    pts, cols = cloud
    ground = _ground_height(work)
    if ground is None:
        ground = float(np.percentile(pts[:, 1], 5))
    normals = point_normals(pts, k=12)
    labels = classify_points(pts, cols, ground_height=ground, normals=normals)

    learned_info: dict = {}
    learned_labels: list = [None] * len(pts)
    if args.no_learned:
        learned_info = dict(status="skipped", reason="--no-learned")
    else:
        frames, classes, meta = load_learned(work)
        if frames is None:
            learned_info = meta
        else:
            learned_labels, learned_info = apply_learned(
                work, pts, normals, frames, classes, args)
    final, prov, disagreements, by_prov = fuse(labels, learned_labels,
                                               policy=args.conflict)
    counts = {c: int(sum(1 for f in final if f == c)) for c in ALL_CLASSES}
    counts = {c: n for c, n in counts.items() if n} or {"obstacle": len(final)}
    heur_counts = {c: labels.count(c) for c in CLASSES}
    out = {"schema_version": 2, "classes": list(ALL_CLASSES), "counts": counts,
           "counts_heuristic_only": heur_counts,
           "counts_by_provenance": dict(by_prov),
           "provenance": prov,
           "ground_height_m": round(ground, 3), "coordinate_frame": "viewer Y-up",
           "note": ("Heuristic geometric+colour classification, upgraded where a "
                    "learned open-vocabulary mask (GroundingDINO+SAM 2) reached the "
                    "point and the conflict policy allowed it; not surveyed ground "
                    "truth."),
           "learned": learned_info,
           "disagreements": dict(disagreements.most_common(40)) if disagreements else {},
           "conflict_policy": args.conflict,
           "coords": [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for x, y, z in pts],
           "rgb": [[*CLASS_RGB[label]] for label in final]}
    (work / "viewer_assets" / "semantics.json").write_text(json.dumps(out))
    # A class-coloured PLY so the labelling is exportable and inspectable in MeshLab/CloudCompare.
    header = ("ply\nformat ascii 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pts))
    with (work / "viewer_assets" / "semantics.ply").open("w") as f:
        f.write(header)
        for (x, y, z), label in zip(pts, final):
            cr, cg, cb = CLASS_RGB[label]
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {cr} {cg} {cb}\n")
    print("[semantics] " + json.dumps(counts))
    print("[semantics] provenance " + json.dumps(dict(by_prov)) +
          (f"; disagreements {sum(disagreements.values())}" if disagreements else ""))
    if learned_info.get("status") not in ("present",):
        print("[semantics] learned path inactive: "
              f"{learned_info.get('reason', learned_info.get('status'))}")
    return 4 if (args.strict and learned_info.get("status") != "present") else 0



if __name__ == "__main__":
    raise SystemExit(main())
