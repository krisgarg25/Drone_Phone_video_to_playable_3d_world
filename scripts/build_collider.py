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
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def run(cmd: list[str]) -> str:
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"splat-transform failed ({r.returncode})")
    for l in [l for l in r.stdout.splitlines() if l.strip()][-4:]:
        print("    " + l.strip())
    return r.stdout


def measure(glb: Path, H, cov, ox, oz, cell, spawn) -> dict:
    """Judge a candidate collider against the heightfield it must agree with."""
    from glb_bounds import accessor, load_glb
    gltf, bin_ = load_glb(glb)
    V = np.concatenate([accessor(gltf, bin_, p["attributes"]["POSITION"])
                        for m in gltf["meshes"] for p in m["primitives"]])
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
            "median_offset_from_ground": round(float(np.median(d)), 2) if sup.any() else None,
            "verts_near_spawn": near}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--voxel-size", type=float, default=0.35)
    ap.add_argument("--voxel-opacity", type=float, default=0.3)
    ap.add_argument("--variant", default="shell", choices=list(VARIANTS))
    ap.add_argument("--no-clip", action="store_true",
                    help="skip the ground-layer clip (diagnostic only)")
    ap.add_argument("--compare", action="store_true",
                    help="build every variant and report the measurements")
    args = ap.parse_args()

    assets = args.work / "viewer_assets"
    col = json.loads((assets / "collision.json").read_text(encoding="utf-8"))
    box, spawn = col["collider_box"], col["spawn"]
    H = np.fromfile(assets / "heights.f32", np.float32).reshape(col["nz"], col["nx"])
    cov = np.fromfile(assets / "coverage.u8", np.uint8).reshape(col["nz"], col["nx"])
    ox, oz = col["origin_xz"]
    cell = col["cell"]

    outdir = args.work / "pc"
    src = flipped_source(assets / "scene.ply", outdir / "collider_src.ply")

    # box faces offset off any round number: the flip is applied as a rotation,
    # so a face lying exactly on a geometry plane sees ~1e-8 sign noise and
    # keeps a coin-flip half of it
    bstr = ",".join(f"{v + 1e-3:.4f}" for v in box["min"]) + "," + \
           ",".join(f"{v - 1e-3:.4f}" for v in box["max"])
    j = int(round((spawn["x"] - ox) / cell))
    i = int(round((spawn["z"] - oz) / cell))
    sy = float(H[i, j]) + 1.0
    seed = f"{spawn['x']:.2f},{sy:.2f},{spawn['z']:.2f}"
    print(f"[collider] box  -B {bstr}")
    print(f"[collider] seed --seed-pos {seed}  (spawn, 1 m up)")

    names = list(VARIANTS) if args.compare else [args.variant]
    results = {}
    for name in names:
        tag = outdir / f"col_{name}"
        for stale in outdir.glob(f"col_{name}.*"):
            stale.unlink()
        print(f"\n[collider] variant '{name}'")
        # NB: -w is --overwrite (boolean). The output is the trailing positional
        # argument, and `.voxel.json` is what asks for voxelisation + sidecars.
        cmd = ["npx", "-y", "@playcanvas/splat-transform", "-w",
               str(src), "-B", bstr,
               "--voxel-size", str(args.voxel_size),
               "--voxel-opacity", str(args.voxel_opacity),
               "--seed-pos", seed,
               *VARIANTS[name],
               "--collision-mesh", "faces",
               str(tag.with_suffix(".voxel.json"))]
        run(cmd)
        glb = tag.with_suffix(".collision.glb")
        if not glb.exists():
            print(f"    NO GLB PRODUCED for '{name}'")
            continue
        results[name] = measure(glb, H, cov, ox, oz, cell, spawn)
        results[name]["glb"] = str(glb)
        print("    " + json.dumps({k: v for k, v in results[name].items() if k != "glb"}))

    if not results:
        raise SystemExit("[collider] no variant produced a collision mesh")

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
            shutil.copyfile(source, dst)
            print(f"\n[collider] {source.name} -> {dst.name} (raw shell, unclipped)")
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
        clip(source, assets, clipped, pitch=args.voxel_size)
        print()
        subprocess.run([sys.executable, str(Path(__file__).parent / "ground_mesh.py"),
                        "--asset", str(assets), "--glb", str(clipped),
                        "--out", str(dst)], check=True)


if __name__ == "__main__":
    main()
