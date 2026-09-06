"""Cull the reconstructed cloud/fog fill out of the exported splat.

Sibling of strip_sky.py for a different failure: footage flown ABOVE a cloud
sea. The clouds barely move during the clip, so they are multi-view-consistent
content and 3DGS reconstructs them as real geometry — on work/temple ~28% of
all gaussians are airborne, desaturated fog filling the air around the hill.
The player spawns inside that volume and every view direction is white mush.
No trainer fixes this: the fog is faithful reconstruction. See
PROBLEM-temple.md for the measurements.

Cloud signature (measured on temple, absent on rocks):
  desaturated   HSV saturation p50 = 0.04          (cut: sat < 0.20)
  airborne      > 1 m above the local ground       (from the exported heightfield)
  over the footprint only — the distant backdrop outside the grid is never
  judged and never cut (same rule as strip_sky).

One trap: the temple's own grey stone walls are ALSO desaturated and airborne.
So before cutting, compact tall structures are auto-detected and protected:
columns holding many high gaussians form dense connected clusters a small
fraction of the grid (the temple reads as a ~9x19 m box, ~4% of the grid),
while cloud fill spreads thinly over nearly every column. Points belonging to
a detected structure's columns are never cut.

The whole thing is gated on population size: real cloud fill is 15%+ of the
in-footprint gaussians, a normal clear-air scene is a couple percent. Under
the gate the step reports and leaves the splat alone, so rocks (and any
well-shot capture) passes through untouched.

Idempotent exactly like strip_sky: the pristine export is kept as
scene.full.ply, and every run re-derives scene.ply from it, so the sky cut
and the cloud cut compose instead of eating each other.

This step is optional, so a refusal is a recorded, visible no-op that exits 0
rather than a SystemExit that fails a run which already has a working splat.
Refusing to cut is always the SAFE answer here — what it costs is a scene that
keeps its fog, never a scene that loses its ceiling. Only an input this file
cannot read is a StepError.

  python strip_clouds.py --asset work/temple/viewer_assets [--dry-run]
"""
import argparse
import os
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strip_sky import grid_header, require_props  # noqa: E402
from strip_sky import local_ground  # noqa: E402
import robust as rb  # noqa: E402

VERTEX_PROPS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "f_dc_0", "f_dc_1", "f_dc_2")


def saturation_of(arr) -> np.ndarray:
    """HSV saturation of each gaussian's base colour (SH DC -> RGB)."""
    SH0_TO_RGB = 0.28209479177387814
    rgb = np.stack([np.asarray(arr[f"f_dc_{k}"], np.float64) for k in range(3)], 1)
    rgb = (rgb * SH0_TO_RGB + 0.5).clip(0.0, 1.0)
    mx, mn = rgb.max(1), rgb.min(1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def protect_structures(x, y, z, inside, fog, col, protect_frac):
    """Columns of compact built structures; their fog-looking points survive.

    A built structure is a compact cluster of DENSELY-populated columns: stone
    surfaces are seen by many cameras and reconstruct with hundreds of
    gaussians per column, while fog fills volume thinly (a handful per column
    spread across nearly every column). Height alone cannot be the test —
    measured on the 800 px temple reconstruction the walls rise under 3 m
    above the courtyard, so a ">3 m above highest ground" scan sees nothing —
    but column density separates them in both reconstructions.

    Column-density threshold + connected components + an area fraction test:
    the temple comes out as one tight cluster of columns, the cloud sea fails
    the area test (it touches most of the grid).
    """
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]

    j = np.floor((x[fog] - ox) / cell).astype(int)
    i = np.floor((z[fog] - oz) / cell).astype(int)
    ok = (i >= 0) & (i < nz) & (j >= 0) & (j < nx)
    counts = np.zeros((nz, nx), np.int64)
    np.add.at(counts, (i[ok], j[ok]), 1)

    pos = counts[counts > 0]
    if len(pos) == 0:
        return np.zeros_like(fog), []
    # only DENSE columns seed a structure: fog-smeared columns sit near the
    # median, surfaces well above it
    thr = max(8.0, rb.safe_pct(pos, 75, 8.0, label="fog column density p75"))
    occ = counts >= thr
    print(f"[clouds]   dense columns (>= {thr:.0f} fog-looking pts): "
          f"{int(occ.sum())} of {len(pos)} occupied "
          f"({100 * occ.sum() / max(nz * nx, 1):.1f}% of grid)")

    # one-column dilation bridges doorway/wall gaps inside a structure
    dil = occ.copy()
    dil[1:, :] |= occ[:-1, :]
    dil[:-1, :] |= occ[1:, :]
    dil[:, 1:] |= occ[:, :-1]
    dil[:, :-1] |= occ[:, 1:]

    grid_cols = max(nz * nx, 1)
    seen = np.zeros_like(dil, bool)
    structs = []
    for sz in range(nz):
        for sx in range(nx):
            if not dil[sz, sx] or seen[sz, sx]:
                continue
            comp = []
            q = deque([(sz, sx)])
            seen[sz, sx] = True
            while q:
                a, b = q.popleft()
                comp.append((a, b))
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c, d = a + dz, b + dx
                    if 0 <= c < nz and 0 <= d < nx and dil[c, d] and not seen[c, d]:
                        seen[c, d] = True
                        q.append((c, d))
            cols = np.array(comp)
            n_pts = int(counts[cols[:, 0], cols[:, 1]].sum())
            structs.append(dict(cells=comp, area_frac=len(comp) / grid_cols,
                                points=n_pts))

    prot = np.zeros_like(fog)
    for s in structs:
        verdict = "STRUCTURE" if s["area_frac"] <= protect_frac else "too widespread"
        s["verdict"] = verdict
        print(f"[clouds]   component: {len(s['cells'])} cols ({100 * s['area_frac']:.1f}% of grid), "
              f"{s['points']} fog-looking gaussians -> {verdict}")
        if s["area_frac"] <= protect_frac:
            keep_cells = {tuple(c) for c in s["cells"]}
            ti = np.floor((z - oz) / cell).astype(int)
            tj = np.floor((x - ox) / cell).astype(int)
            ok2 = inside & (ti >= 0) & (ti < nz) & (tj >= 0) & (tj < nx)
            m = np.zeros(len(x), bool)
            m[ok2] = [(int(a), int(b)) in keep_cells
                       for a, b in zip(ti[ok2], tj[ok2])]
            prot |= m
    return prot, structs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--sat", type=float, default=0.20, help="max HSV saturation of fog")
    ap.add_argument("--height", type=float, default=1.0, help="min metres above local ground")
    ap.add_argument("--gate", type=float, default=0.05,
                    help="fire only if fog is this fraction of the scene's TOTAL "
                         "painted area (opacity x area) — measured: cloud sea "
                         "paints 18.6%%, a clear-air capture 0.8%%")
    ap.add_argument("--max-cut", type=float, default=0.55,
                    help="cap on the cut as a fraction of the footprint count; over "
                         "it the step reports and leaves the splat alone")
    ap.add_argument("--protect-frac", type=float, default=0.10,
                    help="column cluster under this grid fraction counts as a structure")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true", help="ignore the population gate")
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
            f"empty splat, so there is no fog to look for and nothing to write.\n"
            f"  Fix the train/export step for this scene.",
            returncode=3)
    require_props(arr, VERTEX_PROPS, src)
    x, y, z = (np.asarray(arr[k], np.float64) for k in "xyz")
    op = 1.0 / (1.0 + np.exp(-np.asarray(arr["opacity"], np.float64)))
    smax = np.exp(np.stack([np.asarray(arr[f"scale_{k}"], np.float64)
                            for k in range(3)], 1)).max(1)
    print(f"[clouds] {src.name}: {len(arr)} gaussians")

    g, inside = local_ground(a.asset, x, z)
    d = y - g
    sat = saturation_of(arr)
    nx, nz, cell, ox, oz = grid_header(a.asset, what="strip_clouds")
    colj = {"nx": nx, "nz": nz, "cell": cell, "origin_xz": [ox, oz]}
    shape = (nz, nx)
    Hf = rb.load_array(a.asset / "heights.f32", np.float32, shape,
                       label="heights.f32 (export_viewer_assets)").astype(np.float64)
    cov = rb.load_array(a.asset / "coverage.u8", np.uint8, shape,
                        label="coverage.u8 (export_viewer_assets)") > 0
    supported = Hf[cov & np.isfinite(Hf)]
    # The fallback is the top of the splat, not the top of a fictional ground:
    # with no supported column there is nothing to be airborne relative to.
    gmax = rb.safe_max(supported, rb.safe_max(y, 0.0, label="y max"),
                       label="highest supported ground")

    n_in = int(inside.sum())
    sat_in = rb.safe_pct(sat[inside], 50, 1.0, label="saturation p50 in footprint")
    sat_in90 = rb.safe_pct(sat[inside], 90, 1.0, label="saturation p90 in footprint")
    print(f"[clouds] saturation p50/p90 over footprint: {sat_in:.3f}/{sat_in90:.3f}"
          + ("" if n_in else "  (no gaussian is over the footprint)"))
    print(f"[clouds] highest supported ground +{gmax:.1f} m; "
          f"{n_in} of {len(arr)} inside the footprint")

    fog = inside & (d > a.height) & (sat < a.sat)
    print(f"[clouds] fog candidates (airborne >{a.height:.0f} m, sat <{a.sat:.2f}): "
          f"{int(fog.sum())} ({100 * fog.sum() / max(n_in, 1):.1f}% of footprint)")
    if not fog.any() and not a.force:
        print("[clouds] nothing is both desaturated and airborne over the footprint "
              "— this scene has no cloud fill; keeping every gaussian")
        return

    print(f"[clouds] structure scan on the fog candidates themselves "
          f"(highest supported ground +{gmax:.1f} m):")
    prot, structs = protect_structures(x, y, z, inside, fog, colj, a.protect_frac)
    n_prot = int((fog & prot).sum())
    if n_prot:
        print(f"[clouds] protecting {n_prot} fog-looking gaussians that belong to structures")

    cut = fog & ~prot
    frac = float(cut.sum()) / max(n_in, 1)
    w = op * smax ** 2
    paint_frac = float(w[cut].sum()) / max(float(w.sum()), 1e-12)
    print(f"[clouds] would drop {int(cut.sum())} gaussians ({100 * frac:.1f}% of footprint, "
          f"{100 * paint_frac:.1f}% of painted area)")
    if cut.any():
        qs = rb.safe_pct(d[cut], (5, 50, 95), (0.0, 0.0, 0.0), label="cut height above ground")
        print(f"[clouds]   cut points sit {qs[0]:+.1f}/{qs[1]:+.1f}/{qs[2]:+.1f} m above local ground")

    if not cut.any() and not a.force:
        print("[clouds] every fog candidate belongs to a detected structure — "
              "nothing left to cut; keeping every gaussian")
        return
    if not a.force and paint_frac < a.gate:
        print(f"[clouds] below the {100 * a.gate:.0f}% painted-area gate — this is not a "
              f"cloud-sea capture, keeping everything")
        return
    if frac > a.max_cut and not a.force:
        rb.warn(f"the cut would take {100 * frac:.0f}% of the footprint, over the "
                f"{100 * a.max_cut:.0f}% cap — that reads as a mis-set --sat/--height "
                f"eating real geometry, not as cloud fill. Not cutting; the splat is "
                f"left as it is. Re-run with --max-cut or --force only if you have "
                f"looked at the report above.")
        return
    if int(cut.sum()) > 0.5 * len(arr):
        rb.warn(f"the cut would remove {100 * int(cut.sum()) / len(arr):.0f}% of the "
                f"whole splat — refusing to strip that much of a scene. The fog stays; "
                f"clip_collider.py still removes it from the physics mesh.")
        return

    if a.dry_run:
        print("[clouds] dry-run: nothing written")
        return

    if not full.exists():
        shutil.copyfile(scene, full)
        print(f"[clouds] kept the unculled original as {full.name}")
    kept = arr[~cut].copy()
    tmp = scene.with_suffix(".ply.tmp")
    PlyData([PlyElement.describe(kept, "vertex")], text=False).write(str(tmp))
    del arr, ply
    os.replace(tmp, scene)
    print(f"[clouds] wrote {scene.name} ({len(kept)} gaussians)")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[clouds] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
