"""Checks for the collider hardening: the voxel ladder and the no-shell launch.

  .venv\\Scripts\\python.exe tests\\test_collider.py    (or tests\\check_all.py)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import robust as R  # noqa: E402
import build_collider as BC  # noqa: E402
import clip_collider as CC  # noqa: E402

R.configure_streams()

fails = []


def ok(cond, label, detail=""):
    print(("pass  " if cond else "FAIL  ") + label + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(label)


# --- the tool must be launchable without a shell ---------------------------
# npx on Windows is a .cmd shim; subprocess can only start it via cmd.exe, which
# is the same quoting hazard that split "Drone to 3d mesh" in half.
prefix = BC.voxelizer()
probe = None
if not prefix:
    # A tree without tools/node_modules is a legitimate place to run this suite - the
    # rest of it drives a fake voxeliser - but it must not crash on the launch checks,
    # and it must not quietly claim it verified a tool that is not installed.
    print("SKIP  voxeliser launch checks: splat-transform is not installed here\n"
          "      fix: npm i @playcanvas/splat-transform --prefix tools")
else:
    ok(prefix[0].lower().endswith("node.exe") and prefix[1].endswith("cli.mjs")
       and os.sep + "bin" + os.sep in prefix[1],
       "voxelizer resolves to node + bin/cli.mjs, not npx", str(prefix))
    try:
        probe = subprocess.run([*prefix, "--help"], capture_output=True, text=True,
                               timeout=120)
    except OSError as e:
        ok(False, "resolved voxelizer actually runs", f"{type(e).__name__}: {e}")
if probe is not None:
    ok(probe.returncode == 0 and probe.stdout.strip(),
       "resolved voxelizer actually runs", probe.stdout[:80] or probe.stderr[:200])

# --- run() classifies the voxel Map limit as retryable, not fatal ----------
tmp = Path(tempfile.mkdtemp(prefix="collider_"))
fake = tmp / "fake_voxeliser.py"
fake.write_text(
    "import sys\n"
    "n = int(sys.argv[1])\n"
    "if n < 3:\n"
    "    sys.stderr.write('Extracting voxel faces\\n')\n"
    "    sys.stderr.write('RangeError: Map maximum size exceeded\\n')\n"
    "    sys.exit(1)\n"
    "print('done')\n", encoding="utf-8")

attempts = {"n": 0}
ladder = [0.25] + R.voxel_ladder(0.25)
try:
    for v in ladder:
        attempts["n"] += 1
        try:
            R.run_cmd([sys.executable, str(fake), str(attempts["n"] - 1)])
        except R.StepError as e:
            if e.kind != R.VOXEL_OVERFLOW or attempts["n"] == len(ladder):
                raise
            continue
        break
    ok(attempts["n"] == 4, "overflow retried on coarser rungs until it fit",
       f"{attempts['n']} attempts")
    ok(len(ladder) == 6 and all(b > a for a, b in zip(ladder, ladder[1:])),
       "ladder is strictly coarser", str(ladder))
except R.StepError as e:
    ok(False, "overflow retried on coarser rungs until it fit", str(e))

# a non-resource failure must NOT burn the whole ladder
try:
    R.run_cmd([sys.executable, "-c", "import sys; sys.stderr.write('some other error'); sys.exit(1)"])
    ok(False, "unrelated failure raises immediately")
except R.StepError as e:
    ok(e.kind != R.VOXEL_OVERFLOW, "unrelated failure is not mistaken for overflow",
       e.kind)

# --- fit_voxel_size must protect the box that actually crashed -------------
# work/auditorium/logs/08-collider.log: 84.9K gaussians, -B
# -195.6,-68.5,-162.7,167.4,71.7,164.9 (363 x 140 x 328 m) at --voxel-size 0.25.
# That grid is ~1.07 billion cells; splat-transform died in
# "Extracting voxel faces" with RangeError: Map maximum size exceeded.
AUD_BOX = ([-195.5767, -68.4997, -162.6870], [167.4355, 71.7431, 164.9345])
naive = 1
for lo, hi in zip(*AUD_BOX):
    naive *= int((hi - lo) / 0.25) + 1
ok(naive > 20 * R.VOXEL_CELL_LIMIT, "the logged box is far past the Map limit as asked",
   f"{naive:,} cells at 0.25 m")
av, acells = R.fit_voxel_size(*AUD_BOX, 0.25)
ok(acells <= R.VOXEL_CELL_LIMIT and av > 0.25,
   "auditorium box coarsened into budget instead of crashing",
   f"voxel={av:.3f} cells={acells:,}")
room = R.fit_voxel_size([-7, -1.9, -2.6], [2.8, 6.5, 6.6], 0.25)
ok(room[0] == 0.25, "roomscan's box keeps the requested 0.25 m (no needless coarsening)",
   str(room))

# --- the pitch handed to clip is the voxel that was actually used ----------
src = (ROOT / "scripts" / "build_collider.py").read_text(encoding="utf-8")
ok('clip_kwargs = {"pitch": voxel}' in src and '"pitch": args.voxel_size' not in src,
   "clip pitches at the used voxel, not the requested one")
ok("import subprocess" not in src,
   "build_collider cannot call subprocess directly, so no child bypasses run_cmd")
ok("--voxel-size\", str(v)" in src, "ladder value is what reaches the voxeliser")

# --- measure() reports an empty mesh instead of tracebacking --------------
try:
    BC.measure(Path(__file__), np.zeros((4, 4), np.float32),
               np.ones((4, 4), np.uint8), 0.0, 0.0, 0.5, {"x": 0.0, "z": 0.0})
    ok(False, "measure on a non-GLB file should fail loudly, not silently")
except Exception:
    ok(True, "measure rejects a non-GLB input rather than returning junk")
src_guard = (ROOT / "scripts" / "build_collider.py").read_text(encoding="utf-8")
ok("has no vertices" in src_guard and 'np.zeros((0, 3)' in src_guard,
   "measure guards the empty-mesh case")

# --- clip degrades instead of taking the collider down with a bad cut ------
# room_w_jsonl produced NO world because of an absolute 200-triangle floor: its
# whole voxelised room shell was 954 tris, so a legitimate 5% cut "failed".
def _clip_fixture(tmp: Path, n_floor: int, n_crust: int):
    asset = tmp / "viewer_assets"
    asset.mkdir(parents=True, exist_ok=True)
    n, cell = 8, 0.5
    (asset / "collision.json").write_text(json.dumps(
        {"nx": n, "nz": n, "cell": cell, "origin_xz": [0.0, 0.0]}), encoding="utf-8")
    np.full((n, n), 0.0, np.float32).tofile(asset / "heights.f32")
    np.ones((n, n), np.uint8).tofile(asset / "coverage.u8")

    def slab(y, k, i0):
        out = np.zeros((k, 3, 3), np.float32)
        for j in range(k):
            x = 0.1 + 0.01 * ((i0 + j) % 30)
            z = 0.1 + 0.01 * ((i0 + j) % 25)
            out[j] = [[x, y, z], [x + 0.02, y, z], [x, y, z + 0.02]]
        return out

    tris = np.concatenate([slab(0.0, n_floor, 0), slab(9.0, n_crust, 7)])
    glb = tmp / "shell.glb"
    CC.write_mesh(tris, glb)
    return glb, asset, tris


with tempfile.TemporaryDirectory() as td:
    t = Path(td)
    glb, asset, all_tris = _clip_fixture(t, n_floor=4, n_crust=200)
    out = t / "clipped.glb"
    try:
        CC.clip(glb, asset, out, pitch=0.4, gap=1.4, seek=4.0)
        ok(out.exists(), "a cut that eats the mesh still writes a collider",
           str(out))
        ok(len(CC.read_mesh(out)) == len(all_tris),
           "and ships it UNCLIPPED rather than leaving the scene with no mesh",
           f"{len(CC.read_mesh(out))} of {len(all_tris)} tris")
    except SystemExit as e:
        ok(False, "a gutted cut must not abort the collider step any more",
           f"SystemExit({e})")

    glb2, asset2, tris2 = _clip_fixture(t / "b", n_floor=300, n_crust=6)
    out2 = t / "b" / "clipped.glb"
    CC.clip(glb2, asset2, out2, pitch=0.4, gap=1.4, seek=4.0)
    kept2 = len(CC.read_mesh(out2))
    ok(0 < kept2 < len(tris2),
       "an ordinary crust cut still cuts", f"{kept2} of {len(tris2)} kept")

    # test2horizontal: the splat collapsed to 377 gaussians, so the voxel shell
    # came out an 0.8 m cube at the origin while the heightfield grid covers the
    # whole room. Not one vert mapped into a column, and ground_ceiling's y.min()
    # over the resulting empty array aborted the collider step twice.
    glb3, asset3, tris3 = _clip_fixture(t / "c", n_floor=12, n_crust=0)
    glb3_shifted = t / "c" / "shell.glb"
    CC.write_mesh(tris3 + np.array([500.0, 0.0, 500.0], np.float32), glb3_shifted)
    out3 = t / "c" / "clipped.glb"
    try:
        CC.clip(glb3_shifted, asset3, out3, pitch=0.4, gap=1.4, seek=4.0)
        ok(out3.exists() and len(CC.read_mesh(out3)) == len(tris3),
           "a mesh wholly outside the grid ships unclipped, not a traceback",
           f"{len(CC.read_mesh(out3)) if out3.exists() else 'no file'} tris")
    except ValueError as e:
        ok(False, "a mesh wholly outside the grid ships unclipped, not a traceback",
           str(e))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all collider checks passed")
