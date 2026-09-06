"""Build the physics collider from scene.ply via @playcanvas/splat-transform.

Replaces the hand-tuned `-B -20,-1,-1,15,7,32` constant that mvp.bat carried:
that box was measured in a frame the pipeline no longer uses, and its ceiling
landed just above the terrain, so voxelisation produced a flat slab at the grid
top which the character then spawned on — hovering over the whole scene. The box
now comes from collision.json's `collider_box`, derived from the multi-view
supported region.

THE FRAME. splat-transform works in an "engine space" equal to (-x, -y, +z) of
the input PLY. Measured four independent ways: `-B`, `-S`, `-t` and the
`sceneBounds` written into the voxel JSON all agree. Consequences:
  * `-B` and `--seed-pos` are NOT in the PLY's own coordinates. Handing `-B`
    the exact content bounds of a +Y-up PLY rejects every gaussian, which is
    what stalled this script's first version.
  * the collision GLB is emitted in that same flipped space, with an identity
    node transform — so a +Y-up scene comes out upside down.
  * `--voxel-floor-fill` marches +Y from the grid floor *of that space*, i.e.
    downward through our scene. On an upside-down input it fills the sky.
Rather than carry three separate corrections, we hand the tool a source PLY
pre-flipped 180 deg about Z. Its engine space then coincides with our scene
frame, and box, seed and output GLB are all plain scene coordinates.

`--variant` exists because the fill flags interact in ways worth measuring
rather than assuming; `--compare` builds several and reports which one actually
yields a standable surface.

The voxeliser's output is then passed through clip_collider.py, which cuts every
airborne crust the splat reconstructed out of sky and haze. That step is not
optional: without it the topmost collider surface at the spawn was 20 m above
the terrain, so the character stood on debris in mid-air. `--compare` reports
the RAW variants (the clip would mask the differences between them); the single
`--variant` path writes the clipped mesh.

  python build_collider.py --work work/rocks --compare
  python build_collider.py --work work/rocks --variant shell
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import robust as rb  # noqa: E402

# splat-transform is an npm package. `npx -y` downloads it on every call, so a
# network hiccup fails a scene that already spent 20 minutes training. Install
# it once (`npm i @playcanvas/splat-transform --prefix tools`) and this prefers
# the vendored JS entrypoint. The .bin/*.cmd shims are deliberately not used:
# subprocess can only launch those through cmd.exe, which is the quoting hazard
# this module just removed.
PKG = "@playcanvas/splat-transform"
# The package's bin field is bin/cli.mjs. dist/cli.mjs exists too but is not an
# entrypoint - launching it exits 0 having written nothing, which surfaces
# several layers away as "no variant produced a collision mesh".
CLI_REL = ("@playcanvas", "splat-transform", "bin", "cli.mjs")
LOCAL_CLI = ROOT / "tools" / "node_modules" / Path(*CLI_REL)
NPX_CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "npm-cache" / "_npx"

TIMEOUT = 1800  # a voxeliser that wedges must cost a run minutes, not an afternoon


def voxelizer() -> list:
    """argv prefix for splat-transform: a local install or a cached npx copy, else [].

    Launched through node with an explicit script path rather than `npx -y`,
    because npm's `npx` is a .cmd shim: subprocess cannot start it without
    shell=True, and `npx -y` also re-checks the registry on every collider build.
    Returning [] when neither copy exists is deliberate - the last attempt used to
    hand back `npx` anyway, which is the one thing said it could not be launched.
    """
    node = shutil.which("node")
    cands = [LOCAL_CLI]
    if NPX_CACHE.is_dir():
        cands += sorted(NPX_CACHE.glob("*/node_modules/" + "/".join(CLI_REL)))
    if node:
        for c in cands:
            if c.exists():
                return [node, str(c)]
    return []

# Each variant is the list of extra args after the box/voxel basics.
#
# The fill flags are listed but expected to lose here: floor-fill makes every
# column solid from the grid floor up to its lowest occupied voxel, and columns
# with NO geometry are filled the full grid height. Our heightfield has ~56%
# unsupported cells, so filling would erect an invisible plateau at the grid
# ceiling — the exact defect the old collider had. Left in so the comparison
# proves it rather than assuming it.
VARIANTS = {
    # tight shell around occupied voxels only; nothing pinned, nothing filled
    "shell": [],
    # drop gaussians that contribute to no solid voxel, then shell
    "floaters_shell": ["-F", "0.35,0.3,0.004"],
    # keep only the terrain's connected cluster (explicit params: the -C
    # defaults use an opacity of 0.999, which keeps almost nothing)
    "cluster_shell": ["-C", "1.0,0.3,0.001"],
    # solid ground by filling each column up from the grid floor
    "floorfill": ["--voxel-floor-fill"],
}


def flipped_source(scene: Path, dst: Path) -> Path:
    """Write scene.ply rotated 180 deg about Z, so tool-space == scene-space.

    Positions (x, y, z) -> (-x, -y, z), and each gaussian's orientation
    quaternion left-multiplied by the same rotation. In (w, x, y, z) storage
    order that Hamilton product reduces exactly to (w,x,y,z) -> (-z,-y,x,w) —
    no trig, no matrix round-trip, no normalisation drift. Per-axis `scale_*`
    are unaffected: the rotation lives entirely in the quaternion.

    Spherical harmonics are deliberately left alone. They encode view-dependent
    colour and this PLY is only ever consumed by the voxeliser, which cares
    about occupancy; the output splat is discarded.
    """
    if dst.exists() and dst.stat().st_mtime >= scene.stat().st_mtime:
        return dst
    arr = PlyData.read(str(scene))["vertex"].data.copy()
    arr["x"] = -arr["x"]
    arr["y"] = -arr["y"]
    names = set(arr.dtype.names)
    if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        w, x, y, z = (np.asarray(arr[f"rot_{k}"], np.float32).copy() for k in range(4))
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = -z, -y, x, w
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(dst))
    print(f"[collider] flipped source -> {dst.name} ({len(arr)} gaussians)")
    return dst


class VoxelOverflow(Exception):
    """The grid exceeded Node's Map limit at this voxel size: coarsen and retry."""

    def __init__(self, voxel: float, cells: int) -> None:
        super().__init__(f"voxel {voxel:g} m -> {cells:,} cells exceeds the "
                         f"voxeliser's Map limit")
        self.voxel = voxel
        self.cells = cells


def run(cmd: list) -> str:
    """One splat-transform invocation, no shell, classified failure.

    shell=True here used to join argv into a cmd.exe string, and every path in
    this repo carries spaces ("Drone to 3d mesh"), so the tool received a
    half-path and the step died with a message about an unrecognised command.
    """
    print("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd),
          flush=True)
    try:
        p = rb.run_cmd(cmd, timeout=TIMEOUT)
    except rb.StepError as e:
        if e.kind == rb.VOXEL_OVERFLOW:
            raise
        if e.kind == rb.MISSING_TOOL:
            raise rb.StepError(
                rb.MISSING_TOOL,
                f"{e.message}\n  splat-transform could not be launched. Install it "
                f"once so the collider stops depending on npm at run time:\n"
                f"    npm install {PKG} --prefix tools",
                returncode=e.returncode)
        raise
    for line in [l for l in (p.stdout or "").splitlines() if l.strip()][-4:]:
        print("    " + line.strip())
    return p.stdout or ""


def measure(glb: Path, H, cov, ox, oz, cell, spawn) -> dict:
    """Judge a candidate collider against the heightfield it must agree with."""
    from glb_bounds import accessor, load_glb
    gltf, bin_ = load_glb(glb)
    parts = [accessor(gltf, bin_, p["attributes"]["POSITION"])
             for m in gltf["meshes"] for p in m["primitives"]
             if "POSITION" in p.get("attributes", {})]
    V = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
    if len(V) == 0:
        # An empty mesh is a real outcome (the box rejected every gaussian), not
        # a crash: report it so the caller can pick another variant.
        rb.warn(f"{glb.name} has no vertices")
        return {"verts": 0, "bounds": [[0, 0, 0], [0, 0, 0]], "pct_at_ceiling": 0.0,
                "median_offset_from_ground": None, "verts_near_spawn": 0}
    nz, nx = H.shape
    ymax = float(V[:, 1].max())
    cj = np.clip(((V[:, 0] - ox) / cell).astype(int), 0, nx - 1)
    ci = np.clip(((V[:, 2] - oz) / cell).astype(int), 0, nz - 1)
    sup = cov[ci, cj] > 0
    d = V[sup, 1] - H[ci[sup], cj[sup]]
    near = int((np.hypot(V[:, 0] - spawn["x"], V[:, 2] - spawn["z"]) < 1.5).sum())
    return {"verts": len(V),
            "bounds": [[round(float(V[:, k].min()), 1) for k in range(3)],
                       [round(float(V[:, k].max()), 1) for k in range(3)]],
            "pct_at_ceiling": round(100 * float((V[:, 1] > ymax - 1e-3).mean()), 2),
            "median_offset_from_ground": (round(rb.safe_median(d, None, label="dGround"), 2)
                                          if sup.any() and d.size else None),
            "verts_near_spawn": near}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--voxel-size", type=float, default=0.35)
    ap.add_argument("--voxel-opacity", type=float, default=0.3)
    ap.add_argument("--variant", default="shell", choices=list(VARIANTS))
    ap.add_argument("--no-clip", action="store_true",
                    help="skip the ground-layer clip (diagnostic only)")
    ap.add_argument("--clip-gap", dest="clip_gap", type=float, default=None,
                    help="void size (m) that ends the ground column; "
                         "default 1.4 (outdoor sky-crust removal). Rooms need "
                         "~4.0 so the floor-to-ceiling air is not mistaken for "
                         "airborne crust and the walls/ceiling survive in the "
                         "collider.")
    ap.add_argument("--compare", action="store_true",
                    help="build every variant and report the measurements")
    args = ap.parse_args()

    assets = args.work / "viewer_assets"
    rb.require_file(assets / "collision.json", "collision.json (written by export)")
    col = rb.read_json(assets / "collision.json")
    if not col or "collider_box" not in col:
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"collision.json has no collider_box - the export step "
                           f"produced no usable geometry: {assets / 'collision.json'}",
                           returncode=3)
    box, spawn = col["collider_box"], col["spawn"]
    nz, nx = int(col["nz"]), int(col["nx"])
    H = rb.load_array(assets / "heights.f32", np.float32, (nz, nx),
                      label="heights.f32 (written by export)")
    cov_f = assets / "coverage.u8"
    cov = (rb.load_array(cov_f, np.uint8, (nz, nx), label="coverage.u8",
                         required=False)
           if cov_f.exists() else np.ones((nz, nx), np.uint8))
    ox, oz = col["origin_xz"]
    cell = col["cell"]

    outdir = args.work / "pc"
    src = flipped_source(assets / "scene.ply", outdir / "collider_src.ply")

    # box faces offset off any round number: the flip is applied as a rotation,
    # so a face lying exactly on a geometry plane sees ~1e-8 sign noise and
    # keeps a coin-flip half of it
    bstr = ",".join(f"{v + 1e-3:.4f}" for v in box["min"]) + "," + \
           ",".join(f"{v - 1e-3:.4f}" for v in box["max"])
    i, j = rb.clamp_index(round((spawn["z"] - oz) / cell),
                          round((spawn["x"] - ox) / cell), H.shape)
    if (i, j) != (round((spawn["z"] - oz) / cell), round((spawn["x"] - ox) / cell)):
        rb.warn("spawn falls outside the heightfield; seeding the voxeliser from "
                "the nearest supported cell instead")
    ground = H[i, j]
    sy = (float(ground) if rb.finite(ground) else float(box["min"][1])) + 1.0
    seed = f"{spawn['x']:.2f},{sy:.2f},{spawn['z']:.2f}"

    # A fixed --voxel-size cannot be right for every scene: the cell count grows
    # with the cube of the extent, and past ~16.7M entries splat-transform's own
    # Map throws. Fit the size to this box before paying for a failed attempt.
    voxel, cells = rb.fit_voxel_size(box["min"], box["max"], args.voxel_size)
    if voxel > args.voxel_size * 1.02:
        rb.warn(f"--voxel-size {args.voxel_size:g} m would put {cells:,} cells in "
                f"this box, past the voxeliser's limit; using {voxel:g} m")
    print(f"[collider] box  -B {bstr}")
    print(f"[collider] seed --seed-pos {seed}  (spawn, 1 m up)")

    names = list(VARIANTS) if args.compare else [args.variant]
    results = {}
    ladder = [voxel] + rb.voxel_ladder(voxel)
    vrx = voxelizer()
    if not vrx:
        # Named here instead of letting subprocess raise "[WinError 2] The system
        # cannot find the file specified" over an argv with no program in it.
        rb.die(rb.MISSING_TOOL,
               f"splat-transform is not installed and no cached copy was found. "
               f"Fix it with: npm i {PKG} --prefix {ROOT / 'tools'} "
               f"(or run python scripts/bootstrap.py). node on PATH: "
               f"{'yes' if shutil.which('node') else 'no'}")
    for name in names:
        tag = outdir / f"col_{name}"
        for stale in outdir.glob(f"col_{name}.*"):
            stale.unlink()
        print(f"\n[collider] variant '{name}'")
        # NB: -w is --overwrite (boolean). The output is the trailing positional
        # argument, and `.voxel.json` is what asks for voxelisation + sidecars.
        for attempt, v in enumerate(ladder):
            try:
                run([*vrx, "-w", str(src), "-B", bstr,
                     "--voxel-size", str(v),
                     "--voxel-opacity", str(args.voxel_opacity),
                     "--seed-pos", seed,
                     *VARIANTS[name],
                     "--collision-mesh", "faces",
                     str(tag.with_suffix(".voxel.json"))])
            except rb.StepError as e:
                # Fit -> coarser rungs. Only a grid that is genuinely too fine
                # for the voxeliser's Map is worth retrying; anything else the
                # next rung would fail on too.
                if e.kind != rb.VOXEL_OVERFLOW or attempt == len(ladder) - 1:
                    raise
                nxt = ladder[attempt + 1]
                rb.warn(f"{name}: {v:g} m grid too fine for the voxeliser, "
                        f"retrying at {nxt:g} m")
                continue
            if attempt:
                rb.warn(f"{name} needed a coarser grid: {v:g} m "
                        f"(asked for {args.voxel_size:g} m)")
            voxel = v
            break
        glb = tag.with_suffix(".collision.glb")
        if not glb.exists():
            rb.warn(f"no GLB produced for variant '{name}'")
            continue
        results[name] = measure(glb, H, cov, ox, oz, cell, spawn)
        results[name]["glb"] = str(glb)
        results[name]["voxel"] = voxel
        print("    " + json.dumps({k: v for k, v in results[name].items()
                                   if k != "glb"}))

    if not results:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"[collider] no variant produced a collision mesh from {src.name} "
            f"({len(names)} tried at voxel {voxel:g} m).\n"
            f"  Either scene.ply is empty or the -B box rejects every gaussian; "
            f"check the export and frame steps for this scene.",
            returncode=3)

    if args.compare:
        print("\n=== collider variants ===")
        print(f"{'variant':<18}{'verts':>8}{'ceil%':>8}{'dGround':>9}{'@spawn':>8}")
        for n, r in results.items():
            print(f"{n:<18}{r['verts']:>8}{r['pct_at_ceiling']:>8.1f}"
                  f"{r['median_offset_from_ground'] or 0:>9.2f}{r['verts_near_spawn']:>8}")
        print("\nwant: low ceil% (no slab), dGround near 0, >0 verts at spawn")
        (args.work / "collider_variants.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    else:
        source = Path(results[names[0]]["glb"])
        dst = outdir / "collision.collision.glb"
        if args.no_clip:
            # Room scans: the walls and ceiling are the point of the scene, so
            # the ground-layer clip that strips sky crust on outdoor captures
            # would drop 95% of the geometry instead. Write the raw shell to
            # both names so tune_collider's --src still resolves and can do a
            # real A/B between the shell (with object edges) and the heightfield
            # (smoother, but floor-only).
            shutil.copyfile(source, dst)
            shutil.copyfile(source, outdir / "clipped.collision.glb")
            print(f"\n[collider] {source.name} -> {dst.name} (raw shell, unclipped — "
                  f"also written as clipped.collision.glb for tune_collider's --src)")
            return
        # Two post-passes, each fixing a defect the voxeliser cannot:
        #
        # 1. clip. Sky and haze gaussians voxelise into a crust tens of metres
        #    up, and this scene's ground relief overlaps that crust's height
        #    range, so `-B` cannot separate them with a flat plane. Unclipped,
        #    the topmost surface at the spawn was 8 m ABOVE the ground; every
        #    downward ray in the viewer found the crust first, so the character
        #    stood on it 20 m up and the underlay sheet was draped over it,
        #    hiding the splat. Post-clip the spawn column is 4 cm out.
        #
        # 2. ground mesh. A voxel shell renders every height change as a
        #    VERTICAL WALL — measured along the route, 1.05 m walls every 0.6 m,
        #    the first one 20 cm from the spawn. The capsule stayed pinned within
        #    2 m of its spawn for 150 s. The filtered heightfield mesh is the same
        #    data as slopes instead of walls, and it is the surface the route is
        #    planned on, so plan and physics cannot disagree.
        from clip_collider import clip
        from ground_mesh import main as _  # noqa: F401  (imported for the module)
        clipped = outdir / "clipped.collision.glb"
        print()
        # pitch = the grid the mesh was actually built on, which a resource
        # retry may have coarsened away from what --voxel-size asked for
        clip_kwargs = {"pitch": voxel}
        if args.clip_gap is not None:
            clip_kwargs["gap"] = args.clip_gap
        clip(source, assets, clipped, **clip_kwargs)
        print()
        rb.run_cmd([sys.executable, str(Path(__file__).parent / "ground_mesh.py"),
                    "--asset", str(assets), "--glb", str(clipped),
                    "--out", str(dst)], timeout=TIMEOUT)


if __name__ == "__main__":
    main()
