"""Checks the hardened gate against the degenerate scenes that used to crash it.

  .venv\\Scripts\\python.exe tests\\test_gate.py        (or tests\\check_all.py)

Every case here is built from a real failure shape: an empty support set, a
spawn off the grid, a grid too small for a 3x3 neighbourhood, a collider with no
vertices. Before the guards these raised ValueError/IndexError from inside a
numpy reduction, which is what made the gate look randomly broken.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import robust as R  # noqa: E402

R.configure_streams()

GATE = ROOT / "scripts" / "check_world.py"
PY = sys.executable
fails = []


def ok(cond, label, detail=""):
    print(("pass  " if cond else "FAIL  ") + label
          + (("  " + detail.replace("\n", " | ")[:260]) if detail else ""))
    if not cond:
        fails.append(label)


def run_gate(asset: Path, work: Path = None):
    cmd = [PY, str(GATE), "--asset", str(asset)]
    if work:
        cmd += ["--work", str(work)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                       encoding="utf-8", errors="replace")
    return p


def make_scene(tmp: Path, *, nx=20, nz=20, cell=0.25, cov_val=1, H_val=1.0,
               spawn=(1.0, 1.0), extra_col=None, gaussians=200,
               collider=True) -> Path:
    """A minimal but complete viewer_assets dir the gate can judge."""
    w = tmp / "viewer_assets"
    w.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    import build_collider_glb as _h
    H = _h.ramp(nz, nx, base=H_val, cell=cell)
    cov = np.full((nz, nx), cov_val, np.uint8)
    H.tofile(w / "heights.f32")
    cov.tofile(w / "coverage.u8")
    col = {"nx": nx, "nz": nz, "cell": cell,
           "origin_xz": [0.0, 0.0],
           "collider_box": {"min": [0, -1, 0], "max": [nx * cell, 2, nz * cell]},
           "spawn": {"x": spawn[0], "z": spawn[1]},
           "route_metrics": {"perimeter_m": 20.0, "waypoints": 8, "routed_m2": 30,
                             "walkable_pct": 40, "loop_bad_pct": 0,
                             "spawn_above_floor_m": 0.2},
           "collider_surface": {"source": "hf", "reason": "test"}}
    if extra_col:
        col.update(extra_col)
    (w / "collision.json").write_text(json.dumps(col), encoding="utf-8")
    # a tiny splat, above the ground plane
    arr = np.zeros(gaussians, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                     ("opacity", "f4"), ("scale_0", "f4"),
                                     ("scale_1", "f4"), ("scale_2", "f4"),
                                     ("rot_0", "f4"), ("rot_1", "f4"),
                                     ("rot_2", "f4"), ("rot_3", "f4"),
                                     ("f_dc_0", "f4"), ("f_dc_1", "f4"),
                                     ("f_dc_2", "f4")])
    arr["x"] = rng.uniform(0, nx * cell, gaussians).astype("f4")
    arr["z"] = rng.uniform(0, nz * cell, gaussians).astype("f4")
    arr["y"] = (H_val + rng.uniform(0.1, 1.0, gaussians)).astype("f4")
    arr["rot_0"] = 1
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(w / "scene.ply"))
    # ground.f32 must match heights exactly or the route-agreement check fails
    H.astype(np.float32).tofile(w / "ground.f32")
    if collider:
        import build_collider_glb
        build_collider_glb.write_flat_glb(w / "collision.collision.glb",
                                          nx * cell, nz * cell, H_val)
    return w


# a hand-written flat quad GLB, so the gate has a real collider to read
HELPER = Path(tempfile.mkdtemp(prefix="gatehelper_")) / "build_collider_glb.py"
HELPER.write_text('''
import json, struct
from pathlib import Path
import numpy as np

def ramp(nz, nx, base=1.0, cell=0.25, gain=0.02, xgain=0.001):
    """A gently sloping surface, shared by the mesh and the grid it must agree with.

    A planar fixture is not usable: every vertex sits exactly at y_max, which
    correctly reads as a ceiling slab, and the median then has nothing to agree
    with. A pure z slope is not usable either, because a whole boundary row ties
    at the maximum and reads as a slab too - so the surface tilts in both axes.
    """
    i = np.arange(nz)[:, None] * np.ones((1, nx))
    j = np.ones((nz, 1)) * np.arange(nx)[None, :]
    return (base + gain * i * cell + xgain * j * cell).astype(np.float32)

def write_flat_glb(out, ex, ez, y, empty=False, steps=12, gain=0.02, xgain=0.001):
    """A tilted grid floor at ~y with a boundary skirt hanging below it."""
    g = np.linspace(0, 1, steps)
    xs, zs = g * ex, g * ez
    X, Z = np.meshgrid(xs, zs)
    Y = y + gain * Z + xgain * X
    floor = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    ring = np.concatenate([np.arange(steps), np.arange(steps) * steps,
                           (steps - 1) + np.arange(steps) * steps,
                           (steps - 1) * steps + np.arange(steps)])
    skirt = floor[ring] - np.array([0, 0.4, 0], np.float32)
    V = np.concatenate([floor, skirt], 0)
    def quad(i, j):
        a, b = i * steps + j, i * steps + j + 1
        c, d = (i + 1) * steps + j, (i + 1) * steps + j + 1
        return [a, b, d, a, d, c]
    IDX = np.array([q for i in range(steps - 1) for j in range(steps - 1)
                    for q in quad(i, j)], np.uint32)
    if empty:
        V = np.zeros((0, 3), np.float32)
        IDX = np.zeros(0, np.uint32)
    ylo, yhi = (float(V[:, 1].min()), float(V[:, 1].max())) if len(V) else (0.0, 0.0)
    xlo = [float(V[:, k].min()) for k in range(3)] if len(V) else [0.0] * 3
    xhi = [float(V[:, k].max()) for k in range(3)] if len(V) else [0.0] * 3
    Vb, Ib = V.tobytes(), IDX.tobytes()
    gltf = {"asset": {"version": "2.0"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "buffers": [{"byteLength": len(Vb) + len(Ib)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(Vb)},
                {"buffer": 0, "byteOffset": len(Vb), "byteLength": len(Ib)}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(V),
                 "type": "VEC3", "min": xlo, "max": xhi},
                {"bufferView": 1, "componentType": 5125, "count": len(IDX),
                 "type": "SCALAR"}],
            "meshes": [{"primitives": ([] if empty else
                                        [{"attributes": {"POSITION": 0},
                                          "indices": 1, "mode": 4}])}]}
    hb = json.dumps(gltf).encode()
    hb += b" " * (-len(hb) % 4)
    bin_data = Vb + Ib
    bin_data += b"\\x00" * (-len(bin_data) % 4)
    def chunk(tag, data):
        return struct.pack("<I", len(data)) + tag + data
    jc, bc = chunk(b"JSON", hb), chunk(b"BIN\\x00", bin_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    # header is magic + version + totalLength; the JSON chunk's own length is the
    # first 4 bytes after it, so it must not be packed here as well
    out.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(jc) + len(bc))
                    + jc + bc)
''', encoding="utf-8")
sys.path.insert(0, str(HELPER.parent))
import build_collider_glb  # noqa: E402


def fresh(name):
    tmp = Path(tempfile.mkdtemp(prefix=f"gate_{name}_"))
    return tmp


# ---------------------------------------------------------------- happy path
tmp = fresh("ok")
asset = make_scene(tmp)
p = run_gate(asset, tmp)
ok(p.returncode == 0 and "all checks passed" in p.stdout,
   "healthy scene passes", p.stdout + p.stderr)
verdict = json.loads((asset / "world_check.json").read_text(encoding="utf-8"))
ok(verdict["status"] == "pass" and not verdict["hard_failures"],
   "verdict json written alongside the console output", str(verdict["status"]))

# ------------------------------------------------- no measured support anywhere
tmp = fresh("nocov")
asset = make_scene(tmp, cov_val=0)          # every cell unsupported
p = run_gate(asset, tmp)
ok("ValueError" not in p.stderr and "zero-size" not in p.stderr,
   "empty support set does not traceback", p.stderr[-400:])
ok(p.returncode == 1 and "heightfield has measured ground" in p.stdout,
   "empty support set is a HARD failure, not a crash")

# --------------------------------------------------------- coverage only via poses
tmp = fresh("poses")
asset = make_scene(tmp, cov_val=2)          # 100% camera-filled, 0% measured
p = run_gate(asset, tmp)
ok(p.returncode == 1 and "measured surface" in p.stdout,
   "camera-filled-only grid warns-and-fails rather than reporting 100% support",
   p.stdout[-500:])

# ------------------------------------------------------- spawn outside the grid
tmp = fresh("spawn")
big = 40.0                                   # well past nx*cell = 5 m
asset = make_scene(tmp, spawn=(big, big))
p = run_gate(asset, tmp)
ok("IndexError" not in p.stderr, "spawn outside the grid does not IndexError",
   p.stderr[-400:])
ok("outside the grid" in p.stdout, "spawn outside the grid is reported, not hidden")

# --------------------------------------------------- grid too small for a 5x5
tmp = fresh("tiny")
asset = make_scene(tmp, nx=3, nz=3, cell=0.1)
p = run_gate(asset, tmp)
ok("IndexError" not in p.stderr and p.returncode == 0,
   "a 3x3 grid is judged, not crashed", p.stderr[-300:] or p.stdout[-300:])
ok("3x3 spawn neighbourhood" in p.stdout,
   "the gate says which neighbourhood size it actually used")

# -------------------------------------------------------- no collider at all
tmp = fresh("nocollider")
asset = make_scene(tmp, collider=False)
shutil.rmtree(tmp / "pc", ignore_errors=True)
p = run_gate(asset, tmp)
ok("collider exists" in p.stdout and p.returncode == 1,
   "no collider is a HARD failure with a named cause")

# ------------------------------------------------------- empty collider mesh
tmp = fresh("emptyglb")
asset = make_scene(tmp)
build_collider_glb.write_flat_glb(asset / "collision.collision.glb", 0.0, 0.0, 0.0,
                                  empty=True)
p = run_gate(asset, tmp)
ok("ValueError" not in p.stderr and "concatenate" not in p.stderr,
   "an empty collider mesh does not traceback", p.stderr[-400:])
ok("collider has vertices" in p.stdout,
   "an empty collider mesh is reported as a hard failure")

# --------------------------------------------------------- quality shortfall is soft
tmp = fresh("short")
asset = make_scene(tmp, nx=60, nz=60, cell=0.5,
                   extra_col={"route_metrics": {"perimeter_m": 4.0, "waypoints": 4,
                                                "routed_m2": 2, "walkable_pct": 3,
                                                "loop_bad_pct": 0,
                                                "spawn_above_floor_m": 0.4}})
p = run_gate(asset, tmp)
body = json.loads((asset / "world_check.json").read_text(encoding="utf-8"))
ok(body["warnings"] == ["route is long enough to be worth autopiloting"]
   and not body["hard_failures"],
   "a short loop is a warning, not a blocking failure", str(body["warnings"]))
ok(p.returncode == 0, "a warning-only gate still exits 0 so the run ships")

# --------------------------------------------------- no assets at all
tmp = fresh("nothing")
p = run_gate(tmp / "viewer_assets", tmp)
ok(p.returncode != 0 and "collision.json" in (p.stdout + p.stderr)
   and "Traceback" not in p.stderr,
   "a scene with no assets names the step that owes them, without a traceback",
   (p.stdout + p.stderr)[-400:])

# --------------------------------------------------- zero-perimeter route
tmp = fresh("noloop")
asset = make_scene(tmp, extra_col={"route_metrics": {
    "perimeter_m": 0.0, "waypoints": 0, "routed_m2": 0, "walkable_pct": 0,
    "loop_bad_pct": 0, "spawn_above_floor_m": 0.0}})
p = run_gate(asset, tmp)
ok(p.returncode == 1 and "route has a loop to follow" in p.stdout,
   "a scene with no walkable route at all is HARD, not a warning")

# --------------------------------------------------- source-level guarantees
src = (ROOT / "scripts" / "check_world.py").read_text(encoding="utf-8")
ok("SystemExit(1)" in src and src.count("raise SystemExit") == 1,
   "only one place can fail the run")
for banned in ("H[cov > 0].min()", "np.percentile(d,", "drops.min()"):
    ok(banned not in src, f"raw empty-unsafe reduction removed: {banned}")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all gate checks passed")
