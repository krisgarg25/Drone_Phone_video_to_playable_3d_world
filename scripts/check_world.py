"""Is the exported world actually right way up and standable? Pass/fail checks.

This is the regression guard for the bug class that made the character hover
over the scene: gravity derived from an unsigned plane normal came out inverted,
so the "ground" heightfield was fitted to the sky side of the terrain shell and
every downstream artifact agreed with it, consistently and wrongly.

Each check prints PASS/FAIL with the number it judged on, so a claim of "fixed"
is backed by something other than a screenshot that happens to look plausible.

TWO SEVERITIES, because not every failure is the same kind of news.

  hard  the world is wrong: inverted gravity, no measured ground, no collider.
        Worth nothing as shipped, so exit non-zero and say so.
  soft  the world is usable but limited: a short walk loop, thin coverage, a
        spawn standing on a table. The scene still loads and still walks.

Before the split, every FAIL raised SystemExit(1) and pipeline.py treated that
as the run having failed -- so a room scan with a correct 6 m loop was reported
as a failure and its evals, pairs and walktest never ran, discarding a build
that had already cost the GPU 20 minutes. The thresholds were absolute metres
too: a 15 m minimum rejects a room whose walk loop is 3-5 m *by construction*,
and a "1.5 m to the nearest collider vert" test fails any scene whose grid cell
is coarser than that, whatever the geometry is like. Thresholds now scale off
the scene's own cell size and character height unless the operator pins them.

The verdict is also written to world_check.json in the asset dir so the runner
and the dashboard read the same answer the console printed.

  python check_world.py --asset work/rocks/viewer_assets
  python check_world.py --asset work/room/viewer_assets --min-perimeter 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

HARD = "hard"
SOFT = "soft"

RESULTS = []


def check(name: str, ok: bool, detail: str, severity: str = SOFT) -> bool:
    """Record one verdict. `ok` may be None, which prints as "n/a"."""
    if ok is None:
        print(f"  [ n/a] {name}: {detail}")
        RESULTS.append({"name": name, "status": "na", "detail": detail,
                        "severity": severity})
        return True
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    RESULTS.append({"name": name, "status": "pass" if ok else "fail",
                    "detail": detail, "severity": severity})
    if not ok:
        print(f"         ^ {'blocking' if severity == HARD else 'quality warning'}")
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="minimum fraction of measured cells. Default scales with "
                         "the scene: a room scan legitimately sees through no "
                         "walls, so an absolute 0.10 fails it for nothing.")
    ap.add_argument("--min-perimeter", type=float, default=None,
                    help="minimum walk-loop length in m. Default is 15%% of this "
                         "scene's own grid footprint, capped at 15 m, because a "
                         "room's loop is 3-5 m by construction and the old fixed "
                         "15 m outdoor default failed every correct room.")
    ap.add_argument("--json", type=Path, default=None,
                    help="where to write the verdict (default: <asset>/world_check.json)")
    args = ap.parse_args()
    w = args.asset

    col = rb.read_json(w / "collision.json")
    if not col:
        # Nothing downstream can be judged without the grid. This is the one
        # case where exiting early is right: there is no world here at all.
        rb.die(rb.EMPTY_INPUT,
               f"collision.json is missing or empty in {w} - the export step "
               f"produced no world to check", code=3)
    nx, nz, cell = int(col["nx"]), int(col["nz"]), float(col["cell"])
    ox, oz = col["origin_xz"]
    H = rb.load_array(w / "heights.f32", np.float32, (nz, nx),
                      label="heights.f32 (written by export)")
    cov = (rb.load_array(w / "coverage.u8", np.uint8, (nz, nx), required=False)
           if (w / "coverage.u8").exists() else None)
    if cov is None:
        cov = np.ones((nz, nx), np.uint8)

    # A scene's own scale, so a hamster diorama and an auditorium are judged by
    # the same physical rule rather than by a constant that fits neither.
    char_h = float(col.get("character_height") or 1.75)
    footprint = max(nx, nz) * cell
    min_coverage = (args.min_coverage if args.min_coverage is not None
                    else max(0.02, min(0.10, 6.0 / max(footprint, 1.0))))
    # A fraction of this scene's own circuit, capped at the 15 m the outdoor
    # takes were tuned to. Deliberately not keyed to --character-height, which is
    # a collider clearance knob (the room preset ships 0.15 m for a space a human
    # cannot turn around in) and would make the minimum meaningless.
    min_perimeter = (args.min_perimeter if args.min_perimeter is not None
                     else max(2.0, min(15.0, 0.15 * footprint)))

    print(f"grid {nx}x{nz} cell {cell:.3f} m, origin ({ox:.1f}, {oz:.1f}), "
          f"footprint {nx * cell:.0f} x {nz * cell:.0f} m")
    print(f"time  thresholds derived: coverage>={100 * min_coverage:.0f}% "
          f"perimeter>={min_perimeter:.1f} m (footprint {footprint:.0f} m) -- "
          f"pin with --min-coverage / --min-perimeter")

    # coverage.u8 carries two different things: 1 = a height the gaussians
    # actually measured, 2 = a floor the camera's known height filled in. Averaging
    # them and calling the result "real gaussian support" let a grid that is 15%
    # measured report 98%, because an invented cell counted double.
    meas = float((cov == 1).mean())
    pose = float((cov == 2).mean())
    # Zero measured surface means there is no world; a low percentage means a
    # sparse one, which is a quality note, not a defect.
    check("heightfield has measured ground", meas > 0,
          f"{100 * meas:.0f}% of cells have a measured surface, "
          f"{100 * pose:.0f}% more is camera-derived floor, "
          f"{100 * max(0.0, 1 - meas - pose):.0f}% nothing", HARD)
    check("heightfield coverage", meas >= min_coverage,
          f"{100 * meas:.0f}% measured (threshold {100 * min_coverage:.0f}%)")
    sup = cov > 0
    if sup.any():
        print(f"  H over supported cells: min {rb.safe_min(H[sup], 0.0):.2f} "
              f"max {rb.safe_max(H[sup], 0.0):.2f} "
              f"median {rb.safe_median(H[sup], 0.0):.2f} m")
    else:
        rb.warn("no supported cells at all - the heightfield is empty")
    if meas > 0:
        m = H[cov == 1]
        print(f"  H over measured cells only: min {rb.safe_min(m, 0.0):.2f} "
              f"max {rb.safe_max(m, 0.0):.2f} median {rb.safe_median(m, 0.0):.2f} m")

    # ---- gravity sanity: is the terrain a floor with stuff above it? ----
    ply = w / "scene.ply"
    P = np.zeros((0, 3), np.float64)
    if ply.exists():
        v = PlyData.read(str(ply))["vertex"]
        P = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
    else:
        check("scene.ply present", False, f"{ply} missing - export wrote nothing", HARD)
    gj = np.clip(((P[:, 0] - ox) / cell).astype(int), 0, nx - 1) if len(P) else np.zeros(0, int)
    gi = np.clip(((P[:, 2] - oz) / cell).astype(int), 0, nz - 1) if len(P) else np.zeros(0, int)
    inside = (((P[:, 0] >= ox) & (P[:, 0] < ox + nx * cell)
               & (P[:, 2] >= oz) & (P[:, 2] < oz + nz * cell) & (cov[gi, gj] > 0))
              if len(P) else np.zeros(0, bool))
    d = (P[inside, 1] - H[gi[inside], gj[inside]]) if inside.any() else np.zeros(0)
    # Half a metre is a metre-scale assumption; on a coarse grid one cell of
    # noise reads as "below the floor".
    below_band = max(0.5, 2.0 * cell)
    below = float((d < -below_band).mean()) if d.size else None
    print(f"  {int(inside.sum())} gaussians over supported cells; height above surface:")
    if d.size:
        print("   " + "  ".join(f"p{p}={rb.safe_pct(d, p, 0.0):+.2f}"
                                for p in (1, 5, 50, 95, 99)))
    check("terrain is a floor, not a ceiling",
          None if below is None else below < 0.25,
          f"{'' if below is None else f'{100 * below:.1f}%'} of gaussians sit "
          f">{below_band:.2f} m BELOW the ground surface (inverted gravity drives "
          f"this toward 100%)"
          + (" - nothing landed on a supported cell to judge" if below is None else ""),
          HARD)

    # ---- cameras must be above the ground they filmed ----
    # NOT the ground directly beneath them. This clip is a straight sideways
    # pass: the drone flies alongside the terrain looking across and 10 deg
    # down, so most cameras sit outside the ground footprint entirely and
    # "height above the cell under the camera" is a meaningless number.
    poses_f = next((f for f in (w / "poses.json",
                                 (args.work / "poses.json") if args.work else None)
                    if f and Path(f).exists()), None)
    if poses_f:
        cams = rb.read_json(poses_f, [])
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
        # A camera may legitimately film ground far below it; what would signal
        # inverted gravity is sitting *under* the surface it looks at, past the
        # depth error a coarse grid allows.
        sink_limit = max(5.0, 10.0 * cell)
        if drops:
            arr = np.array(drops)
            cam_ok = (rb.safe_median(arr, -1.0) > 0
                      and (arr > -sink_limit).mean() >= 0.8)
            check("cameras above the ground they filmed", cam_ok,
                  f"height above the terrain in front of them: min {arr.min():.1f} m, "
                  f"median {rb.safe_median(arr, 0.0):.1f} m over {len(arr)} cameras "
                  f"(allowing {sink_limit:.1f} m of relief)")
        else:
            check("cameras above the ground they filmed", None,
                  f"no camera has 10+ supported cells ahead of it across "
                  f"{len(cams)} poses - the footage never looks at the ground it built")

    # ---- spawn must be on supported, flat ground ----
    sp = col.get("spawn")
    if sp:
        # Clamp rather than index out of range: a spawn just off the grid edge is
        # a real outcome for a scene walked to its boundary, and an IndexError
        # here reads as a corrupt world instead of a marginal one.
        j0 = int(np.floor((sp["x"] - ox) / cell))
        i0 = int(np.floor((sp["z"] - oz) / cell))
        i, j = rb.clamp_index(i0, j0, (nz, nx))
        if (i, j) != (i0, j0):
            rb.warn(f"spawn cell ({i0},{j0}) is outside the grid ({nz}x{nx}); "
                    f"judging the clamped ({i},{j})")
        # A grid too small for a 3x3 or 5x5 neighbourhood cannot fail a check
        # that presumes one -- a tiny scene is not an unsafe scene.
        want = 2 if (nz >= 5 and nx >= 5) else (1 if (nz >= 3 and nx >= 3) else 0)
        if want:
            sl = (slice(i - want, i + want + 1), slice(j - want, j + want + 1))
            sup_nb = bool(cov[sl].all())
            relief = float(np.ptp(H[sl]))
        else:
            sup_nb, relief = bool(cov[i, j] > 0), 0.0
        check("spawn on supported ground", sup_nb,
              f"x={sp['x']:.1f} z={sp['z']:.1f} H={H[i, j]:.2f} m, "
              f"{2 * want + 1}x{2 * want + 1} spawn neighbourhood supported={sup_nb}",
              HARD)
        max_relief = max(2.0 * char_h, float(cell * 4.0))
        check("spawn is flat", relief < max_relief,
              f"spawn local relief {relief:.2f} m (limit {max_relief:.2f} m "
              f"= 2x character height)")

    # ---- collider, if built ----
    # Must be the file the viewer actually loads. The glob fallback used to pick
    # whichever *collision*.glb sorted first, which after the build grew a
    # `clipped.collision.glb` intermediate — so the gate was passing on a mesh
    # nothing collides with. Named path first, and the intermediates excluded.
    STAGES = {"clipped", "col"}
    glb = next((p for p in (w / "collision.collision.glb", w / "collision.glb")
                if p.exists()), None)
    if glb is None and args.work:
        pc = args.work / "pc"
        if pc.is_dir():
            glb = next((p for p in sorted(pc.glob("*collision*.glb"))
                        if p.name.split(".")[0].split("_")[0] not in STAGES), None)
    if glb is None:
        check("collider exists", False,
              f"no collision GLB in {w} or {args.work / 'pc' if args.work else w} "
              f"- the viewer has nothing to stand on", HARD)
    else:
        import glb_bounds
        try:
            gltf, bin_ = glb_bounds.load_glb(glb)
            parts = [glb_bounds.accessor(gltf, bin_, p["attributes"]["POSITION"])
                     for m in gltf.get("meshes", []) for p in m["primitives"]
                     if "POSITION" in p.get("attributes", {})]
            V = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
        except Exception as e:
            # A truncated or malformed GLB is a defect to report, not a reason to
            # lose the rest of the verdict table.
            check("collider is readable", False,
                  f"{glb.name} could not be decoded ({type(e).__name__}: {e}) - "
                  f"the physics floor does not exist", HARD)
            V = np.zeros((0, 3), np.float32)
        print(f"  collider {glb.name}: {len(V)} verts "
              f"x[{rb.safe_min(V[:, 0], 0.0):.1f}..{rb.safe_max(V[:, 0], 0.0):.1f}] "
              f"y[{rb.safe_min(V[:, 1], 0.0):.1f}..{rb.safe_max(V[:, 1], 0.0):.1f}] "
              f"z[{rb.safe_min(V[:, 2], 0.0):.1f}..{rb.safe_max(V[:, 2], 0.0):.1f}]")
        check("collider has vertices", len(V) > 0,
              f"{glb.name} decoded to {len(V)} verts - an empty mesh means the "
              f"physics floor does not exist", HARD)

    if glb is not None and len(V):
        ymax = rb.safe_max(V[:, 1], 0.0)
        at_top = float((V[:, 1] > ymax - 1e-3).mean())
        check("collider has no ceiling slab", at_top < 0.02,
              f"{100 * at_top:.1f}% of verts sit exactly at y_max={ymax:.2f} "
              f"(a floor-fill slab shows up here as a big number)")
        cj = np.clip(((V[:, 0] - ox) / cell).astype(int), 0, nx - 1)
        ci = np.clip(((V[:, 2] - oz) / cell).astype(int), 0, nz - 1)
        # The collider is built from ONE array - ground.f32, whichever candidate
        # tune_collider picked - and that same array is what the route was planned
        # on and what the browser draws as the underlay. Judged against its own
        # source the tolerance can be a fraction of a cell, which is what actually
        # catches a mesh and a plan that have drifted apart.
        cs = col.get("collider_surface") or {}
        print(f"  collider surface: {cs.get('source', 'UNKNOWN (never tuned)')}"
              + (f" - {cs['reason']}" if cs.get("reason") else ""))
        G = rb.load_array(w / "ground.f32", np.float32, (nz, nx),
                          label="ground.f32", required=False)
        if G is not None:
            dg = V[:, 1] - G.astype(np.float64)[ci, cj]
            check("collider is the surface the route was planned on",
                  abs(rb.safe_median(dg, 99.0)) < 0.2 * cell,
                  f"median collider vert is {rb.safe_median(dg, 0.0):+.3f} m from "
                  f"ground.f32 (limit {0.2 * cell:.3f} m, 20% of the "
                  f"{cell:.3f} m cell)", HARD)
        else:
            check("collider is the surface the route was planned on", False,
                  "no ground.f32 beside it - run scripts/tune_collider.py, or the "
                  "physics and the autopilot are on different surfaces by accident",
                  HARD)
        dv = V[:, 1] - H[ci, cj]
        check("collider sits on the measured heightfield",
              abs(rb.safe_median(dv, 99.0)) < max(char_h, 2 * cell),
              f"median collider vert is {rb.safe_median(dv, 0.0):+.2f} m from "
              f"heights.f32, p95 {rb.safe_pct(np.abs(dv), 95, 0.0):.2f} m "
              f"(limit {max(char_h, 2 * cell):.2f} m)")
        rm = col.get("route_metrics") or {}
        if rm:
            perim = float(rm.get("perimeter_m", 0) or 0)
            check("route is long enough to be worth autopiloting",
                  perim >= min_perimeter,
                  f"loop {perim:.0f} m, {rm.get('waypoints', 0)} waypoints, "
                  f"{rm.get('routed_m2', 0):.0f} m2 routed, "
                  f"{rm.get('walkable_pct', 0):.1f}% of the grid walkable, "
                  f"{rm.get('loop_bad_pct', 0):.1f}% of loop samples bad, spawn "
                  f"{rm.get('spawn_above_floor_m', float('nan')):.2f} m above its "
                  f"floor ({min_perimeter:.1f} m is the shortest loop worth "
                  f"walking here)")
            check("route has a loop to follow", perim > 0,
                  f"{perim:.0f} m loop with {rm.get('waypoints', 0)} waypoints - "
                  f"autopilot has nowhere to go", HARD)
        else:
            check("route metrics recorded", None,
                  "collision.json has no route_metrics - tune_collider did not "
                  "report a walk loop, so the autopilot route is unjudged")
        # "Near the spawn" must mean near on this scene's scale: a coarse grid
        # puts its cell centres further apart than a fixed 1.5 m.
        reach = max(1.5, 2.0 * cell)
        if sp:
            near = int((np.hypot(V[:, 0] - sp["x"], V[:, 2] - sp["z"]) < reach).sum())
            check("collider exists at the spawn", near > 0,
                  f"{near} collider verts within {reach:.2f} m of the spawn")

    print()
    fails = [r for r in RESULTS if r["status"] == "fail"]
    hard = [r for r in fails if r["severity"] == HARD]
    soft = [r for r in fails if r["severity"] == SOFT]
    for r in hard:
        print(f"  HARD  {r['name']}: {r['detail'].splitlines()[0]}")
    for r in soft:
        print(f"  warn  {r['name']}: {r['detail'].splitlines()[0]}")

    verdict = {
        "status": "failed" if hard else ("warnings" if soft else "pass"),
        "checks": RESULTS,
        "hard_failures": [r["name"] for r in hard],
        "warnings": [r["name"] for r in soft],
        "thresholds": {"min_coverage": round(min_coverage, 4),
                       "min_perimeter_m": round(min_perimeter, 2),
                       "character_height_m": char_h,
                       "cell_m": cell,
                       "pinned_by_operator": [k for k, v in (
                           ("min-coverage", args.min_coverage),
                           ("min-perimeter", args.min_perimeter)) if v is not None]},
    }
    out = args.json or (w / "world_check.json")
    try:
        rb.write_json(out, verdict)
    except OSError as e:
        rb.warn(f"could not write {out}: {e}")

    print(f"\n{len(hard)} hard, {len(soft)} quality warning(s) -> {verdict['status']}"
          f"  (verdict: {out.name})")
    if hard:
        raise SystemExit(1)
    if soft:
        # Exit 0: the world is shipped. The runner reads world_check.json to
        # report the scene as complete-with-warnings rather than discarding it.
        print("world is usable but limited; see the warnings above")
    else:
        print("all checks passed")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[gate] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
