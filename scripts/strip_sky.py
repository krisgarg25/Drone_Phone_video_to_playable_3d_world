"""Strip the airborne canopy out of the exported splat.

Drone footage contains sky and haze, and 3DGS reconstructs both as gaussians. On
work/rocks that produced a distinct second layer over the walkable footprint:
7867 gaussians (6.2%) more than 12 m up, spread over 28% of the grid columns.
Overhead and opaque, it is a canopy — from a walking eye height it is a sheet
across the top of the view — and it is what the physics collider was being built
from before clip_collider.py cut it (see that file for the mesh side).

The cut height is NOT hardcoded, and the test is bimodality rather than
sparseness: over the footprint the band density falls 2159/m at +4 to 135/m at
+12 and then RECOVERS to 1727/m by +19, so the two layers are genuinely
separated and the gap is where to cut. Footage shot under overcast or indoors has
no such gap, only a gradient, and correctly loses nothing. See find_void for what
happened when the test was merely "sparse".

Only gaussians over the footprint are judged or cut. Outside it, the nearest
supported column's ground is a fiction — the distant backdrop reads as airborne
against it — and the backdrop is most of the painted frame and none of the
problem.

  python strip_sky.py --asset work/rocks/viewer_assets
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def local_ground(asset: Path, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ground height under each point, from the exported heightfield.

    Columns with no gaussian support carry a diffused height, which is fine as a
    reference but not as a source of truth — so unsupported and out-of-footprint
    positions both borrow the nearest SUPPORTED column instead.
    """
    col = json.loads((asset / "collision.json").read_text(encoding="utf-8"))
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    H = np.fromfile(asset / "heights.f32", np.float32).reshape(nz, nx).astype(np.float64)
    cov = np.fromfile(asset / "coverage.u8", np.uint8).reshape(nz, nx) > 0

    si, sj = np.nonzero(cov)
    sx, sz = ox + (sj + 0.5) * cell, oz + (si + 0.5) * cell
    sh = H[si, sj]

    j = np.floor((x - ox) / cell).astype(int)
    i = np.floor((z - oz) / cell).astype(int)
    inside = (i >= 0) & (i < nz) & (j >= 0) & (j < nx)
    good = inside.copy()
    good[inside] &= cov[i[inside], j[inside]]

    g = np.empty(len(x))
    g[good] = H[i[good], j[good]]
    # nearest supported column for the rest; chunked so the pairwise distance
    # matrix stays small on a 126k-gaussian cloud
    rest = np.nonzero(~good)[0]
    for k in range(0, len(rest), 4096):
        c = rest[k:k + 4096]
        d2 = (x[c, None] - sx[None, :]) ** 2 + (z[c, None] - sz[None, :]) ** 2
        g[c] = sh[d2.argmin(axis=1)]
    return g, inside


def find_void(d: np.ndarray, lo: float, hi: float, step: float = 1.0) -> float | None:
    """Height of the void SEPARATING a distinct canopy layer, or None.

    The test has to be bimodality, not just sparseness. A first version only
    asked whether the emptiest band held under a tenth of the ground layer's
    density, and on work/rocks that fired at +9.5 m on a perfectly smooth decline
    (4361/m at +4 tapering to a flat ~1900/m out to +19) — no void anywhere.
    It would have deleted 27% of the gaussians and 86% of the painted area:
    haze, backdrop and all. Wrong by an order of magnitude, and silently.

    So a band only counts as a void if the density RECOVERS above it: it must be
    under a tenth of the ground layer AND under a tenth of the densest band
    higher up. That is what "two layers with a gap between them" means, and it is
    the only shape where a height cut removes a canopy rather than a gradient.
    """
    base = float(((d >= -1) & (d < 2)).sum()) / 3.0
    if base <= 0:
        return None
    edges = np.arange(lo, hi + step, step)
    cnt = np.histogram(d, edges)[0] / step
    if len(cnt) < 3:
        return None
    print(f"[sky] band density per metre (ground layer = {base:.0f}/m):")
    print("      " + "  ".join(f"{edges[m]:+.0f}:{cnt[m]:.0f}" for m in range(len(cnt))))
    k = int(cnt[:-1].argmin())          # never the last band: nothing above it
    above = float(cnt[k + 1:].max())
    if cnt[k] > 0.10 * base or cnt[k] > 0.10 * above:
        why = ("below the ground layer but not below what sits above it "
               f"({cnt[k]:.0f}/m vs {above:.0f}/m up top) — a gradient, not a gap"
               if cnt[k] <= 0.10 * base else
               f"{100 * cnt[k] / base:.0f}% of ground density — still populated")
        print(f"[sky] emptiest band {edges[k]:+.0f} m holds {cnt[k]:.0f}/m: {why}.")
        print("[sky] no separable canopy — keeping everything")
        return None
    return float(edges[k] + step / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--search", type=float, nargs=2, default=(4.0, 20.0),
                    help="height range (m above ground) to look for the void in")
    ap.add_argument("--cut", type=float, default=0.0,
                    help="force a cut height (m above ground); 0 = auto-detect")
    a = ap.parse_args()

    scene = a.asset / "scene.ply"
    full = a.asset / "scene.full.ply"
    src = full if full.exists() else scene   # idempotent: always cut from the original
    ply = PlyData.read(str(src))
    arr = ply["vertex"].data
    x, y, z = (np.asarray(arr[k], np.float64) for k in "xyz")
    op = 1.0 / (1.0 + np.exp(-np.asarray(arr["opacity"], np.float64)))
    smax = np.exp(np.stack([np.asarray(arr[f"scale_{k}"], np.float64)
                            for k in range(3)], 1)).max(1)
    print(f"[sky] {src.name}: {len(arr)} gaussians")

    g, inside = local_ground(a.asset, x, z)
    d = y - g
    print(f"[sky] height above local ground: "
          + " ".join(f"p{p}={np.percentile(d, p):+.1f}" for p in (50, 90, 95, 99))
          + f"  ({int(inside.sum())} of {len(d)} inside the footprint)")

    # Only gaussians over the walkable footprint are judged, and only they are
    # ever cut. Outside it, `local_ground` borrows the nearest supported column,
    # which is a fiction for a ridge a hundred metres away — measured against it,
    # the backdrop reads as "airborne" and 26118 distant gaussians were the bulk
    # of the first cut. The backdrop is most of the painted frame and none of the
    # problem: the crust that mattered was over the terrain, and clip_collider.py
    # already removes it from the physics.
    cut = a.cut if a.cut > 0 else find_void(d[inside], a.search[0], a.search[1])
    if cut is None:
        if not full.exists():
            print("[sky] nothing to strip")
        return
    keep = ~inside | (d < cut)
    # paint weight ~ opacity x projected area: the honest measure of how much of
    # the frame this removes, since a canopy is few gaussians but large ones
    w = op * smax ** 2
    print(f"[sky] cut at ground +{cut:.1f} m -> dropping {int((~keep).sum())} "
          f"gaussians ({100 * (~keep).mean():.1f}%), "
          f"{100 * w[~keep].sum() / w.sum():.1f}% of painted area")
    print(f"[sky]   dropped y {y[~keep].min():.1f}..{y[~keep].max():.1f}, "
          f"opacity p50={np.median(op[~keep]):.2f}")
    if keep.sum() < 0.5 * len(keep):
        raise SystemExit("[sky] the cut would remove over half the scene — refusing")

    if not full.exists():
        shutil.copyfile(scene, full)
        print(f"[sky] kept the unstripped original as {full.name}")
    # PlyData.read mmaps its source, and on Windows that makes writing back to the
    # same path Errno 22. Write beside it and rename.
    kept = arr[keep].copy()
    tmp = scene.with_suffix(".ply.tmp")
    PlyData([PlyElement.describe(kept, "vertex")], text=False).write(str(tmp))
    del arr, ply
    os.replace(tmp, scene)
    print(f"[sky] wrote {scene.name} ({len(kept)} gaussians)")


if __name__ == "__main__":
    main()
