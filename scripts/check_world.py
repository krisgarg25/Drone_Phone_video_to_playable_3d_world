"""Is the exported world actually right way up and standable? Pass/fail checks.

This is the regression guard for the bug class that made the character hover
over the scene: gravity derived from an unsigned plane normal came out inverted,
so the "ground" heightfield was fitted to the sky side of the terrain shell and
every downstream artifact agreed with it, consistently and wrongly.

Each check prints PASS/FAIL with the number it judged on, so a claim of "fixed"
is backed by something other than a screenshot that happens to look plausible.

  python check_world.py --asset work/rocks/viewer_assets
"""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

FAILS = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.10,
                    help="minimum fraction of supported cells (default 0.10)")
    args = ap.parse_args()
    w = args.asset
    col = json.loads((w / "collision.json").read_text(encoding="utf-8"))
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    H = np.fromfile(w / "heights.f32", np.float32).reshape(nz, nx)
    cov = (np.fromfile(w / "coverage.u8", np.uint8).reshape(nz, nx)
           if (w / "coverage.u8").exists() else np.ones((nz, nx), np.uint8))

    print(f"grid {nx}x{nz} cell {cell:.3f} m, origin ({ox:.1f}, {oz:.1f}), "
          f"footprint {nx * cell:.0f} x {nz * cell:.0f} m")
    check("heightfield coverage", cov.mean() >= args.min_coverage,
          f"{100 * cov.mean():.0f}% of cells have real gaussian support (threshold {100 * args.min_coverage:.0f}%)")
    print(f"  H over supported cells: min {H[cov > 0].min():.2f} "
          f"max {H[cov > 0].max():.2f} median {np.median(H[cov > 0]):.2f} m")

    # ---- gravity sanity: is the terrain a floor with stuff above it? ----
    v = PlyData.read(str(w / "scene.ply"))["vertex"]
    P = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
    gj = np.clip(((P[:, 0] - ox) / cell).astype(int), 0, nx - 1)
    gi = np.clip(((P[:, 2] - oz) / cell).astype(int), 0, nz - 1)
    inside = ((P[:, 0] >= ox) & (P[:, 0] < ox + nx * cell)
              & (P[:, 2] >= oz) & (P[:, 2] < oz + nz * cell) & (cov[gi, gj] > 0))
    d = P[inside, 1] - H[gi[inside], gj[inside]]
    print(f"  {inside.sum()} gaussians over supported cells; height above surface:")
    print("   " + "  ".join(f"p{p}={np.percentile(d, p):+.2f}" for p in (1, 5, 50, 95, 99)))
    below = float((d < -0.5).mean())
    check("terrain is a floor, not a ceiling", below < 0.25,
          f"{100 * below:.1f}% of gaussians sit >0.5 m BELOW the ground surface "
          f"(inverted gravity drives this toward 100%)")

    # ---- cameras must be above the ground they filmed ----
    # NOT the ground directly beneath them. This clip is a straight sideways
    # pass: the drone flies alongside the terrain looking across and 10 deg
    # down, so most cameras sit outside the ground footprint entirely and
    # "height above the cell under the camera" is a meaningless number.
    if (w / "poses.json").exists():
        cams = json.loads((w / "poses.json").read_text(encoding="utf-8"))
        sup = cov > 0
        zi, xi = np.nonzero(sup)
        gx, gz, gh = ox + (xi + 0.5) * cell, oz + (zi + 0.5) * cell, H[zi, xi]
        drops = []
        for c in cams:
            R = np.array(c["R_rowmajor"], np.float64)
            C = -R.T @ np.array(c["t"], np.float64)
            fwd = R.T @ np.array([0.0, 0.0, 1.0])
            # supported ground within the forward half-space of this camera
            to = np.stack([gx - C[0], gz - C[2]], 1)
            ahead = to @ np.array([fwd[0], fwd[2]]) > 0
            if ahead.sum() < 10:
                continue
            drops.append(C[1] - np.median(gh[ahead]))
        drops = np.array(drops)
        cam_ok = bool(np.median(drops) > 0 and (drops > -5.0).mean() >= 0.8)
        check("cameras above the ground they filmed", cam_ok,
              f"height above the terrain in front of them: min {drops.min():.1f} m, "
              f"median {np.median(drops):.1f} m over {len(drops)} cameras")

    # ---- spawn must be on supported, flat ground ----
    sp = col.get("spawn")
    if sp:
        # floor(), not round(). Grid samples are cell centres, so a spawn written
        # by walk_path_from_glb.py lands at exactly (j + 0.5) * cell and round()
        # is a coin flip between j and j+1 -- measured (45.5, 18.5) rounding to
        # cell (46, 19) when the spawn is in cell (45, 18). These checks were
        # judging a 5x5 neighbourhood 0.9 m diagonally away from the spawn.
        j = int(np.floor((sp["x"] - ox) / cell))
        i = int(np.floor((sp["z"] - oz) / cell))
        ok_ij = 2 <= i < nz - 2 and 2 <= j < nx - 2
        sup = bool(cov[i - 2:i + 3, j - 2:j + 3].all()) if ok_ij else False
        relief = float(np.ptp(H[i - 2:i + 3, j - 2:j + 3])) if ok_ij else float("nan")
        check("spawn on supported ground", sup,
              f"x={sp['x']:.1f} z={sp['z']:.1f} H={H[i, j]:.2f} m, "
              f"5x5 neighbourhood fully supported={sup}")
        max_relief = max(1.8, float(cell * 3.0))
        check("spawn is flat", relief < max_relief, f"5x5 relief {relief:.2f} m (limit {max_relief:.2f} m)")

    # ---- collider, if built ----
    # Must be the file the viewer actually loads. The glob fallback used to pick
    # whichever *collision*.glb sorted first, which after the build grew a
    # `clipped.collision.glb` intermediate — so the gate was passing on a mesh
    # nothing collides with. Named path first, and the intermediates excluded.
    STAGES = {"clipped", "col"}
    glb = next((p for p in (w / "collision.collision.glb", w / "collision.glb")
                if p.exists()), None)
    if glb is None and args.work:
        glb = next((p for p in sorted((args.work / "pc").glob("*collision*.glb"))
                    if p.name.split(".")[0].split("_")[0] not in STAGES), None)
    if glb and glb.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from glb_bounds import accessor, load_glb
        gltf, bin_ = load_glb(glb)
        V = np.concatenate([accessor(gltf, bin_, p["attributes"]["POSITION"])
                            for m in gltf["meshes"] for p in m["primitives"]])
        print(f"  collider {glb.name}: {len(V)} verts "
              f"x[{V[:, 0].min():.1f}..{V[:, 0].max():.1f}] "
              f"y[{V[:, 1].min():.1f}..{V[:, 1].max():.1f}] "
              f"z[{V[:, 2].min():.1f}..{V[:, 2].max():.1f}]")
        ymax = V[:, 1].max()
        at_top = float((V[:, 1] > ymax - 1e-3).mean())
        check("collider has no ceiling slab", at_top < 0.02,
              f"{100 * at_top:.1f}% of verts sit exactly at y_max={ymax:.2f} "
              f"(a floor-fill slab shows up here as a big number)")
        # collider should straddle the heightfield, not float above it
        cj = np.clip(((V[:, 0] - ox) / cell).astype(int), 0, nx - 1)
        ci = np.clip(((V[:, 2] - oz) / cell).astype(int), 0, nz - 1)
        dv = V[:, 1] - H[ci, cj]
        check("collider tracks the ground", abs(float(np.median(dv))) < 3.0,
              f"median collider vert is {float(np.median(dv)):+.2f} m from the "
              f"heightfield surface")
        if sp:
            near = (np.hypot(V[:, 0] - sp["x"], V[:, 2] - sp["z"]) < 1.5)
            check("collider exists at the spawn", int(near.sum()) > 0,
                  f"{int(near.sum())} collider verts within 1.5 m of the spawn")
    else:
        print("  [skip] no collider GLB found yet")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
