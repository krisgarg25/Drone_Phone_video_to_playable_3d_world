"""Point cloud -> walkable ground heightfield (library + CLI).

Steps:
  1. Load a PLY point cloud (splats' means), optionally filtered by opacity.
  2. Estimate gravity direction: RANSAC plane fit on the dominant surface.
  3. Rasterize points to an XZ grid; per-cell ground height = low percentile of Y
     (floaters live ABOVE ground; taking p_low digs through them).
  4. Fill holes by neighbor diffusion, light smoothing.
  5. Save .npz + heights.f32 (browser-ready raw float32) + preview PNG +
     collision.json (bounds, rotation, cell, max_step).

The viewer samples this bilinearly: character_y = ground(x,z), and blocks steps
steeper than max_step so you can't climb cliffs/walls.

CLI (standalone use):
  python build_heightfield.py --ply work/rocks/splat.ply --out work/rocks --res 320
Library use: see export_viewer_assets.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:  # optional, nicer PLY handling incl. binary
    from plyfile import PlyData
    HAVE_PLYFILE = True
except ImportError:
    HAVE_PLYFILE = False


def load_points(ply_path: Path, opacity_attr: str | None, min_opacity: float,
                max_points: int) -> np.ndarray:
    if not HAVE_PLYFILE:
        raise SystemExit("pip install plyfile")
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    if opacity_attr and opacity_attr in names:
        op = np.asarray(v[opacity_attr])
        if op.dtype != np.float32 and op.max() > 1:
            op = op.astype(np.float32) / np.iinfo(op.dtype).max
        mask = op >= min_opacity
        xyz = xyz[mask]
        print(f"[heightfield] opacity filter kept {mask.sum()}/{len(mask)} points")
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[sel]
    return xyz.astype(np.float64)


def ransac_plane(xyz: np.ndarray, iters: int = 12, thresh: float | None = None) -> np.ndarray:
    """Return plane normal (unit) of the dominant plane."""
    n = len(xyz)
    rng = np.random.default_rng(0)
    best_inl, best_n = -1, np.array([0.0, 1.0, 0.0])
    sub = xyz[rng.choice(n, min(n, 150_000), replace=False)]
    if thresh is None:
        extent = np.ptp(sub, axis=0).max()
        thresh = max(extent * 0.01, 1e-3)
    for _ in range(iters):
        p = sub[rng.choice(len(sub), 2000, replace=False)]
        try:
            c = p.mean(axis=0)
            _, _, vt = np.linalg.svd(p - c, full_matrices=False)
            normal = vt[-1]
        except np.linalg.LinAlgError:
            continue
        d = np.abs((sub - c) @ normal)
        inl = int((d < thresh).sum())
        if inl > best_inl:
            best_inl, best_n = inl, normal
    if best_n @ np.array([0.0, 1.0, 0.0]) < 0:
        best_n = -best_n
    print(f"[heightfield] plane inliers {best_inl}/{len(sub)} "
          f"({100 * best_inl / len(sub):.0f}%), normal={np.round(best_n, 3)}")
    return best_n


def rot_to_up(normal: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping `normal` onto +Y."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, y)
    s = np.linalg.norm(v)
    c = float(normal @ y)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0]).astype(float)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def cull_floaters(P: np.ndarray, cell_frac: float = 0.006, min_nbr: int = 12,
                  keep_frac: float = 0.35) -> np.ndarray:
    """Boolean keep-mask dropping sparse sky junk, keeping the terrain body.

    Splat training leaves a halo of isolated gaussians far outside the real
    scene (this clip: content sits in x[-17,11] but stray splats reach x=±57).
    They are harmless when you look from the training cameras — which is why
    the offline eval renders looked fine — but they wreck everything derived
    from BOUNDS: the metric rescale, the heightfield footprint, the voxel
    collider's grid box.

    Deliberately density/connectivity based, never height based: the boulder
    stack is 8-10 m of real geometry standing on the ground, so any "drop
    gaussians more than N metres above the ground" rule deletes the subject.
    Sky junk is distinguished by being SPARSE and DISCONNECTED, not by height.

      1. voxelize at cell_frac of the scene span,
      2. keep voxels whose 3x3x3 neighbourhood holds >= min_nbr gaussians,
      3. keep only the largest 26-connected component of those voxels.
    """
    span = float(np.ptp(P, axis=0).max())
    cell = max(span * cell_frac, 1e-6)
    g = np.floor((P - P.min(axis=0)) / cell).astype(np.int64)
    dims = g.max(axis=0) + 1
    if int(np.prod(dims)) > 60_000_000:  # keep the dense grid affordable
        cell *= 2.0
        g = np.floor((P - P.min(axis=0)) / cell).astype(np.int64)
        dims = g.max(axis=0) + 1
    occ = np.zeros(tuple(dims), np.int32)
    np.add.at(occ, (g[:, 0], g[:, 1], g[:, 2]), 1)

    # 3x3x3 box filter by separable cumulative sums
    dens = occ.astype(np.int32)
    for ax in range(3):
        dens = dens + np.roll(dens, 1, ax) + np.roll(dens, -1, ax)
    solid = (occ > 0) & (dens >= min_nbr)
    print(f"[cull] voxel {cell:.3f} u, grid {tuple(dims)}, "
          f"{solid.sum()}/{int((occ > 0).sum())} occupied voxels dense enough")

    # largest 26-connected component over `solid`
    lbl = np.zeros(tuple(dims), np.int32)
    cur, best, best_n = 0, 0, 0
    idx = np.argwhere(solid)
    solid_set = solid
    from collections import deque
    for seed in map(tuple, idx):
        if lbl[seed]:
            continue
        cur += 1
        n = 0
        dq = deque([seed])
        lbl[seed] = cur
        while dq:
            cx, cy, cz = dq.popleft()
            n += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        a, b, c = cx + dx, cy + dy, cz + dz
                        if (0 <= a < dims[0] and 0 <= b < dims[1] and 0 <= c < dims[2]
                                and solid_set[a, b, c] and not lbl[a, b, c]):
                            lbl[a, b, c] = cur
                            dq.append((a, b, c))
        if n > best_n:
            best_n, best = n, cur
    body = lbl == best
    keep = body[g[:, 0], g[:, 1], g[:, 2]]
    print(f"[cull] largest component {best_n}/{int(solid.sum())} voxels -> "
          f"kept {keep.sum()}/{len(keep)} gaussians ({100 * keep.mean():.1f}%)")
    if keep.mean() < keep_frac:
        print(f"[cull] WARNING: culled more than {100 * (1 - keep_frac):.0f}% — "
              f"min_nbr={min_nbr} is probably too aggressive; keeping all")
        return np.ones(len(P), bool)
    return keep


def rasterize_ground(pts: np.ndarray, res: int = 320, percentile: float = 20.0,
                     min_support: int = 4, cell: float | None = None,
                     per_cell: float = 10.0):
    """Rasterize to XZ grid; returns (H, lo, cell, cover). pts must be up=+Y.

    `cover` is a uint8 mask: 1 where the cell had >= min_support real gaussians,
    0 where its height is diffused guesswork. Downstream consumers MUST honour
    it — the viewer's ground underlay used to be drawn over the whole grid, and
    since only ~4% of cells had real support the result was a big white sheet
    covering the actual reconstruction.

    Cell size defaults to whatever the gaussian density can actually support
    (~`per_cell` gaussians per cell) rather than a fixed `res`. A constant res is
    a trap: tuned on a scene whose bounds the floater halo had inflated 4x, the
    same number later produced 0.17 m cells over real terrain with 24 gaussians
    per m2 — 70% of the grid came out as holes and got quietly extrapolated.
    """
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    if cell is None:
        area = max((hi[0] - lo[0]) * (hi[2] - lo[2]), 1e-9)
        cell = float(np.sqrt(per_cell * area / max(len(pts), 1)))
        span_xz = max(hi[0] - lo[0], hi[2] - lo[2])
        cell = float(np.clip(cell, span_xz / 512.0, span_xz / 16.0))
        print(f"[heightfield] auto cell {cell:.3f} for {len(pts)} pts over "
              f"{area:.0f} sq units (~{per_cell:.0f}/cell)")
    nx = max(int(round((hi[0] - lo[0]) / cell)) + 1, 8)
    nz = max(int(round((hi[2] - lo[2]) / cell)) + 1, 8)
    print(f"[heightfield] grid {nx}x{nz}, cell={cell:.3f} scene units")

    ix = np.clip(((pts[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iz = np.clip(((pts[:, 2] - lo[2]) / cell).astype(int), 0, nz - 1)
    flat = iz * nx + ix
    order = np.argsort(flat)
    flat_s, ys_s = flat[order], pts[order, 1]
    starts = np.searchsorted(flat_s, np.arange(nx * nz), side="left")
    ends = np.searchsorted(flat_s, np.arange(nx * nz), side="right")
    H = np.full(nx * nz, np.nan)
    counts = (ends - starts).astype(float)
    # vectorized percentile per cell via sorting within cells is overkill;
    # loop is fine at these sizes
    for k in np.nonzero(counts > 0)[0]:
        a, b = starts[k], ends[k]
        H[k] = np.percentile(ys_s[a:b], percentile)
    H = H.reshape(nz, nx)
    cover = (counts.reshape(nz, nx) >= min_support).astype(np.uint8)
    # a supported cell needs supported company — lone cells are splat specks
    cp = np.pad(cover, 1)
    nbr = sum(cp[1 + dy:1 + dy + nz, 1 + dx:1 + dx + nx]
              for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - cover
    cover = (cover & (nbr >= 3)).astype(np.uint8)
    print(f"[heightfield] real support (>={min_support} gaussians + neighbours): "
          f"{100 * cover.mean():.0f}% of cells")

    holes = np.isnan(H)
    n_holes0 = int(holes.sum())
    for _ in range(400):
        if not holes.any():
            break
        Hp = np.pad(H, 1, mode="edge")
        nb_sum = np.zeros_like(H)
        nb_cnt = np.zeros_like(H)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                shift = Hp[1 + dy: 1 + dy + nz, 1 + dx: 1 + dx + nx]
                m = ~np.isnan(shift)
                nb_sum += np.where(m, shift, 0.0)
                nb_cnt += m
        fill = (nb_cnt > 0) & np.isnan(H)
        H[fill] = nb_sum[fill] / np.maximum(nb_cnt[fill], 1)
        holes = np.isnan(H)
    H[np.isnan(H)] = np.nanmean(H)
    Hp = np.pad(H, 1, mode="edge")
    H = (Hp[:-2, 1:-1] + Hp[2:, 1:-1] + Hp[1:-1, :-2] + Hp[1:-1, 2:] + 4 * H) / 8.0
    if n_holes0:
        print(f"[heightfield] filled {n_holes0} hole cells")
    return H, lo, cell, cover


def save_heightfield(H: np.ndarray, lo: np.ndarray, cell: float, R: np.ndarray,
                     out_dir: Path, max_step: float = 0.8,
                     cover: np.ndarray | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nz, nx = H.shape
    Hf = H.astype(np.float32)
    np.savez_compressed(out_dir / "heightfield.npz", heights=Hf,
                        origin=lo[[0, 2]].astype(np.float32),
                        cell=np.float32(cell), max_step=np.float32(max_step),
                        cover=(cover if cover is not None
                               else np.ones_like(Hf, np.uint8)))
    (out_dir / "heights.f32").write_bytes(Hf.tobytes())
    if cover is not None:
        (out_dir / "coverage.u8").write_bytes(cover.astype(np.uint8).tobytes())

    hmin, hmax = np.percentile(H, 2), np.percentile(H, 98)
    img = np.clip((H - hmin) / max(hmax - hmin, 1e-9), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(out_dir / "heightfield_preview.png")

    cfg = {
        "origin_xz": [float(lo[0]), float(lo[2])],
        "cell": float(cell), "nx": int(nx), "nz": int(nz),
        "rotation_rowmajor": [list(map(float, row)) for row in R],
        "max_step": float(max_step),
        "has_coverage": cover is not None,
    }
    (out_dir / "collision.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[heightfield] wrote {out_dir/'heightfield.npz'} (+heights.f32, coverage.u8, preview, collision.json)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--percentile", type=float, default=20.0)
    ap.add_argument("--max-step", type=float, default=0.8)
    ap.add_argument("--opacity-attr", default="opacity")
    ap.add_argument("--min-opacity", type=float, default=0.35)
    args = ap.parse_args()

    pts = load_points(args.ply, args.opacity_attr, args.min_opacity, 800_000)
    print(f"[heightfield] {len(pts)} points")
    R = rot_to_up(ransac_plane(pts))
    pts = pts @ R.T
    pts = pts[cull_floaters(pts)]
    H, lo, cell, cover = rasterize_ground(pts, args.res, args.percentile)
    save_heightfield(H, lo, cell, R, args.out, args.max_step, cover)


if __name__ == "__main__":
    main()
