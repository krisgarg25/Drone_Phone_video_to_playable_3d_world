"""Geometric + colour semantic labelling of a reconstructed point cloud.

Classifies each 3D point into ground / road / building / vegetation / obstacle
using only surface normals (local PCA), height above the ground plane, planarity
and a colour vegetation index. Deliberately dependency-light (numpy + scipy, both
present offline) so it runs on any machine that already ran the pipeline, and
deterministic so it is unit-testable. It is a heuristic classifier: labels are a
navigational and analytical aid, not a surveyed ground truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CLASSES = ("ground", "road", "building", "vegetation", "obstacle")
CLASS_RGB = {
    "ground": (124, 108, 84),
    "road": (150, 152, 158),
    "building": (214, 138, 60),
    "vegetation": (62, 168, 84),
    "obstacle": (206, 74, 74),
}


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
                    up_face=0.85, gray_chroma=22.0, planar=0.75):
    """Return one class label per point. Y is up (viewer_assets metric frame).

    ground_tol defaults to a fraction of the cloud's vertical extent so the same
    rules behave on a 2 m room and a 20 m aerial scene; pass a number to override.
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
    normals = np.abs(point_normals(points, k))
    verticality = normals[:, 1]                                # 1 = up-facing, 0 = wall
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


def main():
    ap = argparse.ArgumentParser(description="Label the reconstructed point cloud semantically.")
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    work = Path(args.work)
    cloud = load_cloud(work)
    if cloud is None:
        print("[semantics] no sparse_points.json; skipping")
        return 0
    pts, cols = cloud
    ground = _ground_height(work)
    if ground is None:
        ground = float(np.percentile(pts[:, 1], 5))
    labels = classify_points(pts, cols, ground_height=ground)
    counts = {c: labels.count(c) for c in CLASSES}
    out = {"schema_version": 1, "classes": list(CLASSES), "counts": counts,
           "ground_height_m": round(ground, 3), "coordinate_frame": "viewer Y-up",
           "note": "Heuristic geometric+colour classification; not surveyed ground truth.",
           "coords": [[round(float(x), 3), round(float(y), 3), round(float(z), 3)] for x, y, z in pts],
           "rgb": [[*CLASS_RGB[label]] for label in labels]}
    (work / "viewer_assets" / "semantics.json").write_text(json.dumps(out))
    # A class-coloured PLY so the labelling is exportable and inspectable in MeshLab/CloudCompare.
    header = ("ply\nformat ascii 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pts))
    with (work / "viewer_assets" / "semantics.ply").open("w") as f:
        f.write(header)
        for (x, y, z), label in zip(pts, labels):
            cr, cg, cb = CLASS_RGB[label]
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {cr} {cg} {cb}\n")
    print("[semantics] " + json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
