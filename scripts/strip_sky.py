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

This step is optional, so every outcome that is "the scene is simply not a
canopy" is a visible no-op that exits 0: no supported ground column, no void,
a void with nothing above it, a cut judged too greedy. Only an input this file
cannot read at all (no splat, no heightfield, a vertex table without the 3DGS
properties) is a StepError, because that names an upstream step that broke.

  python strip_sky.py --asset work/rocks/viewer_assets
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

VERTEX_PROPS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")


def require_props(arr, names, src: Path) -> None:
    """A vertex table without these columns is not a gaussian splat."""
    missing = [n for n in names if n not in (arr.dtype.names or ())]
    if missing:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{src.name} has no {', '.join(missing)} property, so airborne height "
            f"and paint weight cannot be measured.\n"
            f"  export_viewer_assets.py owns the vertex table; re-run train + "
            f"export rather than this step.",
            returncode=3)


def grid_header(asset: Path, what: str = "strip_sky"):
    """(nx, nz, cell, ox, oz) from collision.json, or a named upstream failure."""
    col = rb.read_json(asset / "collision.json")
    if not isinstance(col, dict) or not {"nx", "nz", "cell", "origin_xz"} <= set(col):
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{(asset / 'collision.json').name} is missing or has no grid header "
            f"(nx/nz/cell/origin_xz), so {what} has no footprint to judge against.\n"
            f"  export_viewer_assets.py writes it; re-run export.",
            returncode=3)
    nx, nz, cell = int(col["nx"]), int(col["nz"]), float(col["cell"])
    ox, oz = (float(v) for v in col["origin_xz"])
    if nx <= 0 or nz <= 0 or not rb.finite(cell, ox, oz) or cell <= 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"collision.json declares an unusable grid "
            f"(nx={nx}, nz={nz}, cell={cell!r}, origin=({ox!r}, {oz!r})).\n"
            f"  export_viewer_assets.py wrote it; re-run export.",
            returncode=3)
    return nx, nz, cell, ox, oz


def local_ground(asset: Path, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ground height under each point, from the exported heightfield.

    Columns with no gaussian support carry a diffused height, which is fine as a
    reference but not as a source of truth — so unsupported and out-of-footprint
    positions both borrow the nearest SUPPORTED column instead.

    Returns (ground, judgeable). With no supported column anywhere there is no
    ground to be above, so judgeable comes back all-False and the caller cuts
    nothing — the honest answer, rather than a cut measured against a fiction.
    """
    nx, nz, cell, ox, oz = grid_header(asset)
    shape = (nz, nx)
    H = rb.load_array(asset / "heights.f32", np.float32, shape,
                      label="heights.f32 (export_viewer_assets)").astype(np.float64)
    cov = rb.load_array(asset / "coverage.u8", np.uint8, shape,
                        label="coverage.u8 (export_viewer_assets)") > 0
    # a NaN height would poison `d` and then np.histogram's autodetected range,
    # which raises on [nan, nan]; such a column is simply not a usable reference.
    sup = cov & np.isfinite(H)

    si, sj = np.nonzero(sup)
    if si.size == 0:
        rb.warn("coverage.u8 marks no supported column at all, so nothing can be "
                "judged airborne — the scene keeps every gaussian")
        return np.zeros(len(x)), np.zeros(len(x), bool)
    sx, sz = ox + (sj + 0.5) * cell, oz + (si + 0.5) * cell
    sh = H[si, sj]

    j = np.floor((x - ox) / cell).astype(int)
    i = np.floor((z - oz) / cell).astype(int)
    inside = (i >= 0) & (i < nz) & (j >= 0) & (j < nx)
    good = inside.copy()
    good[inside] &= sup[i[inside], j[inside]]

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
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        rb.warn("no in-footprint gaussian with a finite height above ground — "
                "there is nothing to measure a canopy against")
        return None
    if step <= 0:
        rb.warn(f"band step {step:g} m is not positive; cannot histogram")
        return None
    if hi <= lo:
        rb.warn(f"search range {lo:.1f}..{hi:.1f} m is empty; pass --search LO HI")
        return None
    base = float(((d >= -1) & (d < 2)).sum()) / 3.0   # per metre over the ground band
    if base <= 0:
        rb.warn("the ground band (-1..2 m) holds no gaussian, so there is no "
                "ground layer to compare a canopy with")
        return None
    edges = np.arange(lo, hi + step, step)
    cnt = np.histogram(d, edges)[0] / step
    if len(cnt) < 3:
        return None
    print(f"[sky] band density per metre (ground layer = {base:.0f}/m):")
    print("      " + "  ".join(f"{edges[m]:+.0f}:{cnt[m]:.0f}" for m in range(len(cnt))))
    k = int(cnt[:-1].argmin())          # never the last band: nothing above it
    above = rb.safe_max(cnt[k + 1:], 0.0, label="densest band above the void")
    if above <= 0:
        print(f"[sky] emptiest band {edges[k]:+.0f} m has nothing above it inside "
              f"{lo:.0f}..{hi:.0f} m — no second layer to separate")
        print("[sky] no separable canopy — keeping everything")
        return None
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
    if not src.exists():
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"neither {full.name} nor {scene.name} exists in {a.asset}.\n"
            f"  export_viewer_assets.py writes scene.ply; run export before this step.",
            returncode=3)
    ply = PlyData.read(str(src))
    arr = ply["vertex"].data
    if len(arr) == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{src.name} holds 0 gaussians — the training or the export produced an "
            f"empty splat, so there is no canopy to look for and nothing to write.\n"
            f"  Fix the train/export step for this scene.",
            returncode=3)
    require_props(arr, VERTEX_PROPS, src)
    x, y, z = (np.asarray(arr[k], np.float64) for k in "xyz")
    op = 1.0 / (1.0 + np.exp(-np.asarray(arr["opacity"], np.float64)))
    smax = np.exp(np.stack([np.asarray(arr[f"scale_{k}"], np.float64)
                            for k in range(3)], 1)).max(1)
    print(f"[sky] {src.name}: {len(arr)} gaussians")

    g, inside = local_ground(a.asset, x, z)
    d = y - g
    print(f"[sky] height above local ground: "
          + " ".join(f"p{p}={rb.safe_pct(d, p, 0.0, label=f'd p{p}'):+.1f}"
                     for p in (50, 90, 95, 99))
          + f"  ({int(inside.sum())} of {len(d)} inside the footprint)")
    if not inside.any():
        print("[sky] nothing sits over the footprint — a canopy outside it is the "
              "painted backdrop, which this step never cuts")

    # Only gaussians over the walkable footprint are judged, and only they are
    # ever cut. Outside it, `local_ground` borrows the nearest supported column,
    # which is a fiction for a ridge a hundred metres away — measured against it,
    # the backdrop reads as "airborne" and 26118 distant gaussians were the bulk
    # of the first cut. The backdrop is most of the painted frame and none of the
    # problem: the crust that mattered was over the terrain, and clip_collider.py
    # already removes it from the physics.
    cut = a.cut if a.cut > 0 else find_void(d[inside], a.search[0], a.search[1])
    if cut is None:
        print("[sky] nothing to strip; scene.ply left as exported")
        return
    keep = ~inside | (d < cut)
    n_drop = int((~keep).sum())
    if n_drop == 0:
        print(f"[sky] cut at ground +{cut:.1f} m drops 0 gaussians — every gaussian "
              f"over the footprint is already below it")
        print("[sky] nothing to strip; scene.ply left as exported")
        return
    if keep.sum() < 0.5 * len(keep):
        rb.warn(f"the cut at ground +{cut:.1f} m would remove {100 * n_drop / len(keep):.0f}% "
                f"of the splat, over half of it — refusing to strip, keeping every "
                f"gaussian. That is a canopy verdict this scene does not support; "
                f"check --search and --cut before forcing it.")
        return
    # paint weight ~ opacity x projected area: the honest measure of how much of
    # the frame this removes, since a canopy is few gaussians but large ones
    w = op * smax ** 2
    wsum = float(w.sum())
    paint = 100 * float(w[~keep].sum()) / wsum if wsum > 0 else 0.0
    print(f"[sky] cut at ground +{cut:.1f} m -> dropping {n_drop} "
          f"gaussians ({100 * n_drop / len(keep):.1f}%), "
          f"{paint:.1f}% of painted area")
    print(f"[sky]   dropped y {rb.safe_min(y[~keep], 0.0, label='dropped y min'):.1f}"
          f"..{rb.safe_max(y[~keep], 0.0, label='dropped y max'):.1f}, "
          f"opacity p50={rb.safe_median(op[~keep], 0.0, label='dropped opacity'):.2f}")

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
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[sky] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
