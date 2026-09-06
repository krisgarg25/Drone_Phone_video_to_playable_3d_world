"""Choose the collider's ground surface by measuring which one the route walks.

Two candidate surfaces exist for every scene and both are defensible:
  hf     — the exported heightfield (heights.f32), the surface the router, the
           visible underlay and the object floors are all already built on;
  shell  — the clipped voxel shell's per-cell top face, what the collider used to
           be made of exclusively.

Hand-picking one per scene is how this took an hour today. The A/B, five scenes,
both surfaces built the same way and planned on the same array, so the ONLY thing
that differs is the source (loop = the route the router publishes for that mesh):

  scene            hf loop   shell loop   winner   cell size   old config
  auditorium            68 m        51 m   hf        0.122 m    27 m
  room_w_jsonl          60 m        55 m   tie->hf   1.016 m    55 m
  rocks                 69 m        94 m   shell     0.430 m    93 m
  temple                 6 m        12 m   shell     2.946 m    36 m (*)
  room_multi_video       7 m         8 m   shell     0.294 m    24 m (*)

  (*) those two baselines are not comparable and are not losses - see the third
  bullet below.

Three things that measurement killed, recorded here because all three were
believed until it:

* The heightfield is NOT universally the smoother surface. On rocks its p90
  inter-cell step is 0.85 m against the shell's 0.61 m; on room_multi_video 3.83
  against 3.84. It wins only where the grid cell is finer than the 0.25 m voxel
  the shell was quantised with, which is where per-cell maxima become a bed of
  nails. The auditorium's cell is 0.122 m; every scene that prefers the shell has
  a coarser cell than that.
* No local roughness statistic predicts the outcome, so none is used as a judge.
  "Fraction of joins taller than the 0.9 m climb limit" ranks the SHELL smoother
  on four scenes out of five, including the auditorium where the heightfield
  clearly walks better - because what actually blocked that route was a constant
  46 degree median grade, not rare tall steps. The router is the only thing that
  knows how a surface walks, and it takes 0.8 s, so it is the judge.
* The old per-scene baselines are not comparable to these numbers, so two of them
  are not losses. `top_surface` takes the maximum over the vertex POSITIONS that
  fall in a cell, and a ground mesh has exactly one vertex per cell (vertices sit
  at cell centres), so on the meshes built here it is near-identity - only the rim
  wall lands in more than one cell, and `--band` rejects it. What actually differs
  is how many smoothing passes the surface got and what it was applied to: the old
  route was raw voxel shell -> max over many vertices per cell -> one box filter,
  the new one is heightfield -> one box filter at mesh build -> sampled at cell
  centres, with none in the router. Same shipped mesh with only the router's
  `--smooth` flipped: room_multi_video 8 m -> 28 m, room_w_jsonl 60 m -> 2 m,
  auditorium 68 m -> 68 m. On scenes this marginal (room_multi_video routes 11 m2
  of 75x75 m, temple 43 m2 of 106x106 m) loop length is chaotic in that knob -
  a different region wins the pick - so it cannot be quoted as a regression
  without the browser walk test behind it. Both scenes are also mis-scaled (see
  the gate in scripts/check_world.py), which step 4 of this pass changes anyway.
  What IS established: under one fixed procedure the tuner's winner beats or
  matches the old config on the three scenes with usable floors, and nothing was
  left on a surface chosen by hand.

Ties go to the heightfield within --tie-band, for a reason that is not about
roughness: heights.f32 is also what the browser draws as the ground underlay and
what build_objects.py measures every furniture floor against, so when the walk is
the same length, the heightfield is the choice that keeps one array end to end.

A run that already has a working collider never ends here in failure. When the
shell candidate cannot be measured at all - --src missing, unreadable, or a mesh
that puts no cell within --band of the heightfield - or when it turns out to BE
the heightfield, there is nothing to compare and the tuner keeps the collider
build_collider shipped and records `compared: false` with the reason. A candidate
build or route that fails likewise drops that candidate instead of the run, and
if both fail the collider and ground.f32 are rolled back byte for byte to what
this step found. Only a forced surface that cannot be built, or a scene with no
shipped collider to keep, is a StepError.

  python scripts/tune_collider.py --work work/auditorium \\
      --asset work/auditorium/viewer_assets \\
      --src work/auditorium/pc/collision.collision.glb \\
      --mesh-smooth 3 --pick best
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402
from walk_path_from_glb import read_glb_tris, top_surface  # noqa: E402

SOURCES = ("hf", "shell")

# What a candidate that produced no usable route is scored as: a 0 m loop, and
# both spawn warnings set, so it can never win a comparison.
NO_ROUTE = {"perimeter_m": 0.0, "waypoints": 0, "routed_m2": 0.0,
            "walkable_pct": 0.0, "loop_bad_pct": 100.0,
            "spawn_above_floor_m": float("nan"),
            "spawn_hole": True, "spawn_on_object": True}


def run(argv, soft: bool = False) -> bool:
    """Run a child step. `soft` returns False instead of ending the tuning."""
    name = Path(str(argv[1])).name if len(argv) > 1 else str(argv[0])
    try:
        p = rb.run_cmd(argv)
    except rb.StepError as e:
        sys.stdout.write(e.output or "")
        if not soft:
            raise
        rb.warn(f"{name} failed ({e.kind}) - {str(e).splitlines()[0]}")
        return False
    sys.stdout.write(p.stdout or "")
    return True


def grid_of(asset: Path) -> tuple:
    """(col, nx, nz, cell, ox, oz) with the header actually usable, or a named failure."""
    col_file = asset / "collision.json"
    col = rb.read_json(col_file)
    if not isinstance(col, dict) or not {"nx", "nz", "cell", "origin_xz"} <= set(col):
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{col_file} is missing or has no grid header (nx/nz/cell/origin_xz), so "
            f"there is no surface to compare candidates on.\n"
            f"  export_viewer_assets.py writes it; re-run export.",
            returncode=3)
    nx, nz, cell = int(col["nx"]), int(col["nz"]), float(col["cell"])
    ox, oz = (float(v) for v in col["origin_xz"])
    if nx <= 0 or nz <= 0 or not rb.finite(cell, ox, oz) or cell <= 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"collision.json declares an unusable grid (nx={nx}, nz={nz}, "
            f"cell={cell!r}, origin_xz=({ox!r}, {oz!r})).\n"
            f"  export_viewer_assets.py wrote it; re-run export.",
            returncode=3)
    return col, nx, nz, cell, ox, oz


def shell_gap(glb: Path, ref, ox, oz, cell, band) -> tuple:
    """Median metres from this mesh's top face to the heightfield.

    Returns (gap, covered_fraction, why_unusable). gap is None when the two
    surfaces cannot be compared at all, which is a different fact from "they are
    the same surface" and is reported as such.
    """
    if not glb.exists():
        return None, 0.0, f"{glb.name} does not exist"
    if glb.stat().st_size == 0:
        return None, 0.0, f"{glb.name} is empty"
    try:
        tris = read_glb_tris(glb)
        raw = top_surface(tris, ref, ox, oz, cell, band)
    except Exception as e:
        return None, 0.0, f"{glb.name} is not a readable mesh ({type(e).__name__}: {e})"
    if tris.size == 0:
        return None, 0.0, f"{glb.name} has no triangles"
    m = np.isfinite(raw) & np.isfinite(ref)
    if not m.any():
        return None, 0.0, (f"{glb.name} puts no cell within {band:.2f} m of the "
                           f"heightfield, so the two surfaces share nothing to "
                           f"compare (raise --band if that is wrong for this scene)")
    return float(np.median(np.abs(raw[m] - ref[m]))), float(m.mean()), ""


def num(c: dict, key: str, default: float = 0.0) -> float:
    v = c.get(key, default)
    return float(v) if rb.finite(v) else default


def candidate_metrics(col_file: Path, ok: bool, reason: str) -> dict:
    """The route this candidate published, or the record that it published none.

    A 0 m loop with both spawn warnings set can never win the A/B, which is what
    makes a failed candidate build or a failed route a soft event instead of a
    dead run.
    """
    rm = (rb.read_json(col_file, {}) or {}).get("route_metrics") if ok else None
    if not isinstance(rm, dict) or not rb.finite(rm.get("perimeter_m")):
        if ok:
            rb.warn("the router exited 0 but published no usable route_metrics")
        return dict(NO_ROUTE, note=reason)
    out = dict(NO_ROUTE)
    out.update(rm)
    return out


def show(c: dict) -> str:
    stand = c.get("spawn_above_floor_m")
    return (f"loop {num(c, 'perimeter_m'):5.0f} m, {int(num(c, 'waypoints'))} wp, "
            f"{num(c, 'routed_m2'):4.0f} m2 routed, "
            f"{num(c, 'walkable_pct'):4.1f}% walkable, "
            f"{num(c, 'loop_bad_pct'):.1f}% bad, spawn "
            + (f"{float(stand):.2f} m above floor" if rb.finite(stand) else "not placed")
            + (" ON AN OBJECT" if c.get("spawn_on_object") else "")
            + (" OVER A HOLE" if c.get("spawn_hole") else ""))


def print_candidates(cand: dict) -> None:
    for s, c in cand.items():
        print(f"[tune] {s:5s}: {show(c)}"
              + (f"  [{c['note']}]" if c.get("note") else ""))


def snapshot(paths) -> dict:
    """Byte-for-byte copies of the files this step overwrites.

    A tuning that gives up mid-way has to leave exactly what build_collider
    shipped - the mesh AND its matching ground.f32 - because a collider from one
    surface beside a planning array from the other is the bug this whole step
    exists to prevent.
    """
    snap = {}
    for p in paths:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            t = p.with_name(p.name + ".pre-tune")
            shutil.copyfile(p, t)
            snap[p] = t
    return snap


def rollback(snap: dict) -> None:
    for orig, tmp in snap.items():
        if tmp.exists():
            shutil.copyfile(tmp, orig)
            rb.warn(f"restored {orig.name} to the mesh this step started from")


def drop(snap: dict) -> None:
    for tmp in snap.values():
        try:
            tmp.unlink()
        except OSError:
            pass


def record(col_file: Path, payload: dict) -> None:
    col = rb.read_json(col_file, {}) or {}
    col["collider_surface"] = payload
    rb.write_json(col_file, col)


def keep_shipped(ship: Path, a, ref, ox, oz, cell, why: str, cand: dict = None) -> None:
    """Ship the collider that is already there, and record why nothing was chosen.

    Two roads lead here: no comparison was possible at all, or both candidates
    walked a 0 m loop so there was nothing to rank.
    """
    if not ship.exists() or ship.stat().st_size == 0:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{why}, and there is no shipped collider to keep instead "
            f"({ship} is missing or empty).\n"
            f"  build_collider.py writes it; run the collider step before tuning.",
            returncode=3)
    rb.warn(f"[tune] {why} - keeping the shipped {ship.name} untouched")
    kept, kept_cov, _ = shell_gap(ship, ref, ox, oz, cell, a.band)
    if rb.finite(kept):
        print(f"[tune] the kept {ship.name} sits {kept:.2f} m (median) from the "
              f"heightfield over {100 * kept_cov:.0f}% of cells")
    # The route still has to be published: the gate reads route_metrics, and a
    # collider nobody planned a loop on looks like a broken world rather than an
    # untuned one.
    run([PY, ROOT / "scripts" / "walk_path_from_glb.py", "--asset", a.asset,
         "--glb", ship, "--surface", "hf", "--band", a.band,
         "--smooth", a.smooth, "--pick", a.pick], soft=True)
    col_file = a.asset / "collision.json"
    rm = (rb.read_json(col_file, {}) or {}).get("route_metrics")
    record(col_file, {
        "source": "kept", "reason": why, "compared": bool(cand),
        "kept_mesh": ship.name,
        "kept_gap_to_heightfield_m": round(kept, 4) if rb.finite(kept) else None,
        "tie_band": None,
        "candidates": cand or {s: (rm if isinstance(rm, dict) else
                                   dict(NO_ROUTE, note="no comparison was possible"))
                               for s in SOURCES},
    })
    print(f"[tune] collider surface = kept ({ship.name}); nothing was re-tuned for "
          f"this scene, and the reason is recorded in {col_file.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--asset", required=True, type=Path)
    ap.add_argument("--src", type=Path, default=None,
                    help="the VOXEL SHELL mesh to derive from; defaults to the "
                         "shipped collider, which is only correct right after "
                         "build_collider has written it")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the winning collider goes (default: "
                         "<work>/pc/collision.collision.glb, the file the viewer loads)")
    ap.add_argument("--tie-band", type=float, default=0.10,
                    help="loop lengths within this fraction of each other count as "
                         "a tie, and the heightfield wins ties (see the header)")
    ap.add_argument("--force", choices=("auto",) + SOURCES, default="auto")
    ap.add_argument("--band", type=float, default=2.5)
    ap.add_argument("--wall", type=float, default=6.0)
    ap.add_argument("--skirt", type=float, default=3.0)
    ap.add_argument("--mesh-smooth", dest="mesh_smooth", type=int, default=3,
                    help="box filter on the collider surface itself (ground_mesh)")
    ap.add_argument("--smooth", type=int, default=3,
                    help="only reaches the router, which ignores it: it plans on "
                         "ground.f32, already smoothed by the mesh build. One knob "
                         "for both silently re-tuned the wrong script (measured: "
                         "the auditorium loop went 68 m -> 52 m when the mesh lost "
                         "its smoothing)")
    ap.add_argument("--pick", choices=("largest", "best"), default="best")
    a = ap.parse_args()

    ship = a.out or (a.work / "pc" / "collision.collision.glb")
    src_glb = a.src or ship
    col_file = a.asset / "collision.json"
    _col, nx, nz, cell, ox, oz = grid_of(a.asset)
    ref = rb.load_array(a.asset / "heights.f32", np.float32, (nz, nx),
                        label="heights.f32 (export_viewer_assets)").astype(np.float64)

    # Guard the one failure that would make the whole comparison meaningless: if
    # --src is already a heightfield mesh, the "shell" candidate is the "hf"
    # candidate, and the tuner would be awarding a prize for beating itself.
    # The threshold is a fraction of the grid cell, not a fixed 5 cm, because a
    # 5 cm gap is nothing on a 3 m cell and enormous on a 2 cm diorama grid.
    same_thr = max(0.005, min(0.05, 0.25 * cell))
    gap, cover, unusable = shell_gap(src_glb, ref, ox, oz, cell, a.band)
    if gap is None:
        print(f"[tune] {src_glb.name}: {unusable}")
        if a.force != "auto":
            raise rb.StepError(
                rb.EMPTY_INPUT,
                f"--force {a.force} asked for a surface built from {src_glb}, but "
                f"{unusable}. Drop --force to keep the shipped collider, or pass "
                f"--src the voxel shell (build_collider's clipped output).",
                returncode=3)
        keep_shipped(ship, a, ref, ox, oz, cell,
                     f"there is no shell candidate to compare against ({unusable})")
        return
    print(f"[tune] {src_glb.name}: top face sits {gap:.2f} m (median) from the "
          f"heightfield over {100 * cover:.0f}% of cells "
          f"(same-surface threshold {same_thr:.3f} m)")
    if gap < same_thr and a.force == "auto":
        keep_shipped(ship, a, ref, ox, oz, cell,
                     f"that mesh IS the heightfield ({100 * cover:.0f}% of cells "
                     f"agree to {gap:.3f} m), so there is no shell candidate to "
                     f"compare against; pass --src the voxel shell "
                     f"(build_collider's output) to tune from scratch")
        return

    snap = snapshot([ship, a.asset / "ground.f32"])
    try:
        if a.force != "auto":
            winner, cand = a.force, {}
            why = f"--force {a.force}"
        else:
            cand = {}
            for s in SOURCES:
                print(f"\n===== candidate: {s}")
                out_glb = a.work / "pc" / f"surface_{s}.glb"
                built = run([PY, ROOT / "scripts" / "ground_mesh.py", "--asset", a.asset,
                             "--glb", src_glb, "--out", out_glb,
                             "--surface", s, "--band", a.band, "--smooth", a.mesh_smooth,
                             "--wall", a.wall, "--skirt", a.skirt], soft=True)
                if not built or not out_glb.exists() or out_glb.stat().st_size == 0:
                    cand[s] = dict(NO_ROUTE, note="surface build failed")
                    rb.warn(f"candidate {s}: no mesh produced, it cannot win the A/B")
                    continue
                shutil.copyfile(out_glb, ship)
                # ground_mesh wrote ground.f32 for this candidate, so the router plans
                # on exactly the array the physics will use - for both candidates. A
                # surface that cannot produce a valid loop is not a candidate.
                ok = run([PY, ROOT / "scripts" / "walk_path_from_glb.py", "--asset", a.asset,
                          "--glb", ship, "--surface", "hf", "--band", a.band,
                          "--smooth", a.smooth, "--pick", a.pick], soft=True)
                cand[s] = candidate_metrics(
                    col_file, ok, "the router could not plan a loop on this surface")

            def key(s):
                c = cand[s]
                # spawn warnings are a tie-break, not a veto: a longer loop with the
                # spawn 0.3 m off the floor still beats a short one on solid ground.
                return (num(c, "perimeter_m"),
                        0 if (c.get("spawn_hole") or c.get("spawn_on_object")) else 1)

            best, other = max(SOURCES, key=key), min(SOURCES, key=key)
            pb, po = num(cand[best], "perimeter_m"), num(cand[other], "perimeter_m")
            print(f"\n[tune] hf {num(cand['hf'], 'perimeter_m'):.0f} m vs "
                  f"shell {num(cand['shell'], 'perimeter_m'):.0f} m")
            print_candidates(cand)
            if pb <= 0:
                rollback(snap)
                keep_shipped(ship, a, ref, ox, oz, cell,
                             "no candidate walked a loop, so this step cannot rank "
                             "them; the collider shipped by build_collider is kept",
                             cand)
                return
            margin = (pb - po) / max(pb, 1e-9)
            if margin <= a.tie_band:
                winner, why = "hf", (f"{po:.0f} m vs {pb:.0f} m is inside the "
                                     f"{100 * a.tie_band:.0f}% tie band, so the "
                                     f"shared-array preference decides")
            else:
                winner = best
                why = f"{100 * margin:.0f}% longer loop than {other} " \
                      f"({pb:.0f} m vs {po:.0f} m)"

        print(f"[tune] collider surface = {winner} ({why})")

        # Whichever lost, its ground_mesh run left ground.f32 holding its surface, and
        # that file is the underlay the browser draws. Rebuild the winner's so the
        # mesh, the planning array and the recorded decision all say the same thing.
        win_glb = a.work / "pc" / f"surface_{winner}.glb"
        rebuilt = run([PY, ROOT / "scripts" / "ground_mesh.py", "--asset", a.asset,
                       "--glb", src_glb, "--out", win_glb,
                       "--surface", winner, "--band", a.band, "--smooth", a.mesh_smooth,
                       "--wall", a.wall, "--skirt", a.skirt], soft=True)
        if not rebuilt or not win_glb.exists() or win_glb.stat().st_size == 0:
            if a.force != "auto":
                raise rb.StepError(
                    rb.FAILED,
                    f"--force {winner} could not be built from {src_glb} by "
                    f"ground_mesh.py; nothing was shipped. See the child error above.",
                    returncode=1)
            rollback(snap)
            rb.warn("[tune] the winner's surface could not be rebuilt after its A/B "
                    "run - keeping the shipped collider instead of a mesh with no "
                    "matching ground.f32 beside it")
            record(col_file, {
                "source": "kept", "reason": "the winning surface failed to rebuild",
                "compared": True, "kept_mesh": ship.name, "tie_band": a.tie_band,
                "candidates": cand,
            })
            return
        shutil.copyfile(win_glb, ship)
        run([PY, ROOT / "scripts" / "walk_path_from_glb.py", "--asset", a.asset,
             "--glb", ship, "--surface", "hf", "--band", a.band,
             "--smooth", a.smooth, "--pick", a.pick], soft=True)

        cands = cand or {winner: (rb.read_json(col_file, {}) or {}).get("route_metrics")
                         or dict(NO_ROUTE, note="forced without an A/B")}
        record(col_file, {
            "source": winner, "reason": why,
            "compared": a.force == "auto",
            "tie_band": a.tie_band if a.force == "auto" else None,
            "candidates": cands,
        })
        print(f"[tune] shipped {ship.name} + recorded the decision in {col_file.name}")
    finally:
        drop(snap)


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[tune] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
