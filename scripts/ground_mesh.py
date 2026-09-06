"""Turn the clipped voxel shell into a walkable ground mesh.

The problem this solves, measured rather than assumed. splat-transform's
voxeliser emits a shell of axis-aligned voxel faces, so every height change is a
VERTICAL WALL. On work/rocks, sampled along the planned route at 0.2 m, the raw
shell's top surface alternates between -11.2 and -12.2 m on adjacent 0.64 m
cells: a 1.05 m wall every 0.6 m, with the first one 20 cm from the spawn. 17% of
route samples crossed a step taller than the capsule's 0.34 m radius. The capsule
was pinned within 2 m of its spawn for 150 s — grounded the whole time, 10%
travel efficiency. No amount of step-up logic in the controller walks a bed of
nails; the surface itself is the defect.

That noise is not terrain. The splat scatters gaussians roughly +/-0.5 m about
the true surface, and taking per-cell maxima of a 0.35 m voxelisation of that
turns the scatter into 3-voxel pillars. Filtering the same data gives max inter-
cell steps of 0.35 m along the route (p90 = 0.16 m) — and, crucially, a mesh of
SLOPES instead of walls, which a capsule climbs.

The other thing this fixes is a category of bug rather than a number: the route
was planned on a smoothed surface while ammo collided with the raw one, so the
autopilot was walking a map of a different world. Here the collider IS the
planning surface, so plan and physics cannot disagree.

By default the surface IS the exported heightfield, not the shell's top face. The shell is
a per-cell maximum of a 0.25 m voxelisation of a cloud that scatters half a metre about the
true surface, which is what turned the shipped collider into a field of vertical spikes; the
heightfield is the smooth measured surface the router already plans on, so physics, plan and
the visible underlay are literally the same array. `--surface shell` keeps the old path for
measuring against. Lost either way: overhangs and caves, which no heightfield can express.
This capture has none worth the trade.

  python ground_mesh.py --asset work/rocks/viewer_assets \
      --glb work/rocks/pc/collision.collision.glb \
      --out work/rocks/pc/ground.collision.glb
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402
from clip_collider import write_mesh
from walk_path_from_glb import read_glb_tris, smooth_surface, top_surface


def fill_holes(A: np.ndarray, ref: np.ndarray, passes: int = 24) -> np.ndarray:
    """Grow the measured surface into its holes, then fall back to `ref`.

    Dropping `ref` straight into the holes would leave a step wherever the two
    disagree — they differ by up to several metres, since `ref` is a diffused
    heightfield and the shell's top is a voxel maximum. Dilating the measured
    values outward first keeps the seam continuous; `ref` only supplies regions
    the shell never reached at all, so the mesh still spans the whole footprint
    and there is nowhere to fall through.
    """
    H = A.copy()
    for _ in range(passes):
        holes = ~np.isfinite(H)
        if not holes.any():
            break
        F = np.nan_to_num(H, nan=0.0)
        W = np.isfinite(H).astype(np.float64)
        fs, ws = np.zeros_like(F), np.zeros_like(W)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            fs += np.roll(np.roll(F, di, 0), dj, 1)
            ws += np.roll(np.roll(W, di, 0), dj, 1)
        grow = holes & (ws > 0)
        H[grow] = (fs / np.maximum(ws, 1e-9))[grow]
    n = int((~np.isfinite(H)).sum())
    if n:
        H[~np.isfinite(H)] = ref[~np.isfinite(H)]
    print(f"[ground] filled {int((~np.isfinite(A)).sum())} empty cells "
          f"({n} of them from the exported heightfield)")
    return H


def build(H: np.ndarray, ox: float, oz: float, cell: float,
          wall: float, skirt: float) -> np.ndarray:
    """Two triangles per grid quad, plus a boundary wall and a downward skirt.

    Vertices sit at cell CENTRES, matching how every other script in this
    pipeline indexes the grid (floor((x - origin) / cell)), so a position that
    tests walkable is over the triangle pair that carries its own height.

    The wall is why the walk test can no longer record a fall: the mesh spans the
    full footprint with no holes, and its rim is closed, so there is no route off
    the edge. The skirt gives the rim thickness — a zero-thickness wall lets a
    fast capsule tunnel through in one substep.
    """
    nz, nx = H.shape
    X = ox + (np.arange(nx) + 0.5) * cell
    Z = oz + (np.arange(nz) + 0.5) * cell
    XX, ZZ = np.meshgrid(X, Z)
    V = np.stack([XX, H, ZZ], axis=-1)          # (nz, nx, 3)

    a = V[:-1, :-1]; b = V[:-1, 1:]; c = V[1:, :-1]; d = V[1:, 1:]
    tris = [np.stack([a, c, b], axis=-2).reshape(-1, 3, 3),
            np.stack([b, c, d], axis=-2).reshape(-1, 3, 3)]

    if wall > 0:
        # four rim strips, each extruded up by `wall` and down by `skirt`
        rims = [V[0, :], V[-1, ::-1], V[:, 0][::-1], V[:, -1]]
        for r in rims:
            top = r.copy(); top[:, 1] += wall
            bot = r.copy(); bot[:, 1] -= skirt
            for lo, hi in ((bot, r), (r, top)):
                p0, p1 = lo[:-1], lo[1:]
                q0, q1 = hi[:-1], hi[1:]
                tris.append(np.stack([p0, q0, p1], axis=-2))
                tris.append(np.stack([p1, q0, q1], axis=-2))
    return np.concatenate(tris, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--glb", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--surface", choices=("hf", "shell"), default="hf",
                    help="hf = the exported heightfield, which is the surface the "
                         "route is planned on and the underlay is drawn from, so "
                         "physics and plan are the same array. shell = the old "
                         "voxel-shell top face, kept only to measure against.")
    ap.add_argument("--band", type=float, default=2.5,
                    help="max distance from the exported ground to count as surface (m)")
    ap.add_argument("--smooth", type=int, default=3,
                    help="box-filter width in cells; 1 disables")
    ap.add_argument("--wall", type=float, default=6.0,
                    help="height of the invisible boundary wall (m); 0 disables")
    ap.add_argument("--skirt", type=float, default=3.0,
                    help="depth the rim is extruded downward (m)")
    a = ap.parse_args()

    col = json.loads((a.asset / "collision.json").read_text(encoding="utf-8"))
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    ref = np.fromfile(a.asset / "heights.f32", np.float32).reshape(nz, nx).astype(np.float64)

    raw = top_surface(read_glb_tris(a.glb), ref, ox, oz, cell, a.band)
    y_lo, y_hi = rb.safe_min(raw, float("nan")), rb.safe_max(raw, float("nan"))
    print(f"[ground] shell top surface: {np.isfinite(raw).mean() * 100:.0f}% of "
          f"{nz}x{nx} cells, y {y_lo:.2f}..{y_hi:.2f}")
    if a.surface == "hf":
        # The heightfield is already complete: rasterize_ground diffused it into its
        # holes and camera_ground filled more from the height the camera was held at,
        # so there is nothing to grow and no seam to hide.
        H = smooth_surface(ref, a.smooth) if a.smooth > 1 else ref.copy()
    else:
        H0 = fill_holes(raw, ref)
        H = smooth_surface(H0, a.smooth) if a.smooth > 1 else H0

    # how far the collider now sits from the voxel shell it came from
    m = np.isfinite(raw)
    off = H[m] - raw[m]
    if off.size:
        print(f"[ground] offset from the shell over measured cells: "
              f"median {np.median(off):+.2f} m, "
              f"p95 {np.percentile(np.abs(off), 95):.2f} m")
    else:
        # A shell that contributes no cell inside `--band` of the heightfield is
        # not a surface to agree with - test2horizontal's splat collapsed to 377
        # gaussians, so its voxel shell was one 0.8 m cube in a room-sized grid.
        # The mesh built below comes from `ref`, so the world still ships a floor.
        rb.warn("[ground] no shell cell falls inside the band around the "
                f"heightfield ({a.band:g} m) - there is nothing to measure the "
                "offset against. Building the ground from the heightfield alone.")
    dz = np.abs(np.diff(H, axis=0)); dx = np.abs(np.diff(H, axis=1))
    step = np.concatenate([dz.ravel(), dx.ravel()])
    if step.size:
        print(f"[ground] inter-cell step: "
              + " ".join(f"p{p}={np.percentile(step, p):.2f}" for p in (50, 90, 99))
              + f" max={step.max():.2f} m  (raw shell was 1.05 m every other cell)")
        print(f"[ground] equivalent grade: p50={np.degrees(np.arctan(np.percentile(step, 50) / cell)):.0f}deg "
              f"p90={np.degrees(np.arctan(np.percentile(step, 90) / cell)):.0f}deg")
    else:
        print("[ground] grid is a single cell wide - no inter-cell step to measure")

    tris = build(H, ox, oz, cell, a.wall, a.skirt)
    write_mesh(tris, a.out)
    H.astype(np.float32).tofile(a.asset / "ground.f32")
    print(f"[ground] wrote ground.f32 ({nz}x{nx}) beside it — the surface the "
          f"physics, the route and the visible underlay all share")


if __name__ == "__main__":
    main()
