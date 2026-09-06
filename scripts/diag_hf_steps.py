"""Where do the heightfield's big steps come from?

The A/B says the heightfield collider is smoother than the voxel shell on the
auditorium (grade p50 16 deg vs 50 deg) and ROUGHERR on rocks and
room_multi_video. Two very different causes fit that:
  (a) the measured surface really is spiky there -> the export is the problem,
      and a per-scene escape hatch would be papering over it;
  (b) the spikes live in cells nothing was ever measured in, which the export
      filled by diffusion, so a floor built from the heightfield is a floor
      built partly from guesses.
Split the steps by coverage state to tell them apart, and attribute every big
step to the worst state among the two cells it joins. coverage.u8:
0 = nothing, 1 = a height the gaussians measured, 2 = camera-derived floor.

  python scripts/diag_hf_steps.py rocks room_multi_video auditorium
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk_path_from_glb import read_glb_tris, top_surface

NAMES = {1: "measured", 2: "camera", 0: "nothing"}

for scene in sys.argv[1:] or ["rocks", "room_multi_video", "auditorium"]:
    a = Path("work") / scene / "viewer_assets"
    col = json.loads((a / "collision.json").read_text())
    nx, nz, cell = col["nx"], col["nz"], col["cell"]
    ox, oz = col["origin_xz"]
    ref = np.fromfile(a / "heights.f32", np.float32).reshape(nz, nx).astype(np.float64)
    cov = np.fromfile(a / "coverage.u8", np.uint8).reshape(nz, nx)
    print(f"=== {scene}: grid {nz}x{nx} cell {cell:.3f} m, footprint "
          f"{nx * cell:.0f}x{nz * cell:.0f} m, span {ref.max() - ref.min():.1f} m")
    for s in (1, 2, 0):
        m = cov == s
        print(f"  {NAMES[s]:9s} {100 * m.mean():5.1f}% of cells  "
              + (f"span {np.nanmin(ref[m]):.1f}..{np.nanmax(ref[m]):.1f} m"
                 if m.any() else "-"))

    # attribute every neighbour step to the worst coverage state of its two cells
    for nm, H in (("heightfield", ref),):
        worst, step = [], []
        for ax in (0, 1):
            d = np.abs(np.diff(H, axis=ax))
            c = np.maximum(cov[:-1, :] if ax == 0 else cov[:, :-1],
                           cov[1:, :] if ax == 0 else cov[:, 1:])
            worst.append(c.ravel()); step.append(d.ravel())
        step = np.concatenate(step); worst = np.concatenate(worst)
        big = step > 3 * cell
        print(f"  {nm}: step p50={np.median(step):.2f} p90={np.percentile(step, 90):.2f} "
              f"max={step.max():.2f} m; {100 * np.degrees(np.arctan(np.median(step) / cell)):.0f}deg p50")
        print(f"    steps > {3 * cell:.1f} m: {100 * big.mean():.2f}% of all joins")
        for s in (1, 2, 0):
            both = worst == s
            if both.any():
                print(f"    between {NAMES[s]:9s} cells: {100 * big[both].mean():5.2f}% "
                      f"of those joins are big ({100 * both.mean():.1f}% of joins)")

    glb = Path("work") / scene / "pc" / "regress_shell.glb"
    if glb.exists():
        raw = top_surface(read_glb_tris(glb), ref, ox, oz, cell, 2.5)
        hs = np.where(np.isfinite(raw), raw, ref)
        st = np.concatenate([np.abs(np.diff(hs, axis=0)).ravel(),
                             np.abs(np.diff(hs, axis=1)).ravel()])
        print(f"  shell+ref: step p50={np.median(st):.2f} p90={np.percentile(st, 90):.2f} "
              f"max={st.max():.2f} m; "
              f"{100 * (st > 3 * cell).mean():.2f}% of joins > {3 * cell:.1f} m "
              f"(shell covers {100 * np.isfinite(raw).mean():.0f}% of cells)")
