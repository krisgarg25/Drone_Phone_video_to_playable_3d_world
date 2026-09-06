"""One command from a clean clone to a runnable pipeline.

  python scripts/bootstrap.py                # pipeline environment
  python scripts/bootstrap.py --with-train   # plus the CUDA training environment
  python scripts/bootstrap.py --check        # change nothing, say what is missing

Stdlib-only on purpose: this runs before any dependency exists, so it cannot import
the rest of the repo. It ends by handing off to `pipeline.py doctor`, which is the
single place toolchain health is judged - duplicating those checks here is how two
sources of truth start disagreeing.

Two things this exists for. First, there is no single `pip install` that works: the
pipeline needs Python 3.12 plus a separate 3.10 environment holding a CUDA build of
gsplat, a Chromium download for the walk test, and Node packages for the collider.
Second, and worse, a plain clone can *look* complete while a 313 MB binary tracked in
git-lfs came down as a 130-byte text pointer - which COLMAP reports as a crash with no
mention of the real cause.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
VENV310 = ROOT / ".venv310"
LFS_FILE = Path("tools/colmap/bin/onnxruntime_providers_cuda.dll")
LFS_POINTER_MAX = 400          # the fetched file is 313 MB; an unfetched one is ~130 B
MIN_NODE = 18

ok_count = warn_count = fail_count = 0


def report(tag: str, msg: str, fix: str = "") -> None:
    global ok_count, warn_count, fail_count
    if tag == "ok":
        ok_count += 1
    elif tag == "FAIL":
        fail_count += 1
    else:
        warn_count += 1
    print(f"  [{tag:<4}] {msg}")
    if fix:
        print(f"         fix: {fix}")


def run(argv, *, cwd=None, timeout=3600) -> tuple[int, str]:
    """Never shell=True: the repo root contains spaces and a shell splits it in half."""
    try:
        p = subprocess.run([str(a) for a in argv], cwd=str(cwd or ROOT),
                           capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def py_version(launcher) -> str:
    code, out = run(list(launcher) + ["-c", "import sys;print('%d.%d'%sys.version_info[:2])"])
    return out.strip() if code == 0 else ""


def find_python(want: str) -> list[str] | None:
    """A launcher command for CPython `want` (e.g. "3.12"), or None."""
    cands: list[list[str]] = []
    pyw = shutil.which("py")
    if pyw:
        cands.append([pyw, f"-{want}"])
    cands.append([sys.executable])
    for name in (f"python{want}", "python3", "python"):
        w = shutil.which(name)
        if w:
            cands.append([w])
    for c in cands:
        if py_version(c) == want:
            return c
    return None


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def make_env(venv: Path, want: str, req: str, *, check_only: bool) -> Path | None:
    """Create `venv` and install `req` into it. Returns its python, or None."""
    py = venv_python(venv)
    if py.exists():
        report("ok", f"{venv.name} exists ({py_version([py])})")
    else:
        launcher = find_python(want)
        if launcher is None:
            report("FAIL", f"{venv.name} missing and CPython {want} is not installed",
                   f"install Python {want}, then re-run: python scripts/bootstrap.py")
            return None
        if check_only:
            report("info", f"{venv.name} would be created from "
                           f"{' '.join(launcher)} and {req} installed into it")
            return None
        print(f"  ......... creating {venv.name} with {' '.join(launcher)}")
        code, out = run(launcher + ["-m", "venv", str(venv)], timeout=600)
        if code != 0 or not py.exists():
            report("FAIL", f"could not create {venv.name}: {out.strip()[:200]}")
            return None
        report("ok", f"created {venv.name}")
    if check_only:
        return py
    code, out = run([py, "-m", "pip", "install", "-r", req], timeout=5400)
    if code != 0:
        report("FAIL", f"`pip install -r {req}` failed:\n{out.strip()[-1200:]}")
        return None
    report("ok", f"installed {req}")
    return py


def check_lfs() -> None:
    f = ROOT / LFS_FILE
    if not f.exists():
        report("info", f"{LFS_FILE} is not present (COLMAP's CUDA provider; only "
                       "needed by the ONNX-assisted matcher)")
        return
    size = f.stat().st_size
    if size < LFS_POINTER_MAX:
        report("FAIL", f"{LFS_FILE} is a {size}-byte git-lfs pointer, not the binary",
               "git lfs install && git lfs pull   "
               "(COLMAP otherwise dies with a stack buffer overrun that names no cause)")
    else:
        report("ok", f"{LFS_FILE.name} fetched ({size / 1048576:.0f} MB)")


def check_node(*, check_only: bool) -> None:
    npm = shutil.which("npm")
    if npm is None:
        report("FAIL", "npm not on PATH - the collider's package cannot be installed",
               "install Node 18+ from nodejs.org (npm ships with it), then re-run")
        return
    node = shutil.which("node")
    if node is None:
        report("FAIL", "node not on PATH - the collider is built by a Node CLI",
               "install Node 18+ from nodejs.org, then re-run this script")
        return
    code, out = run([node, "-e", "console.log(process.versions.node)"], timeout=120)
    ver = out.strip()
    major = "".join(ch for ch in ver.split(".")[0] if ch.isdigit())
    if code != 0 or not major or int(major) < MIN_NODE:
        report("FAIL", f"node reports {ver[:40] or 'nothing'}",
               f"Node {MIN_NODE}+ is required (@playcanvas/splat-transform)")
        return
    report("ok", f"node v{ver}")
    for pkg in (Path("tools"), Path("tools/navbake")):
        if not (ROOT / pkg / "package.json").exists():
            continue
        if (ROOT / pkg / "node_modules").is_dir():
            report("ok", f"{pkg}/node_modules present")
            continue
        if check_only:
            report("info", f"{pkg} would get `npm install`")
            continue
        code, out = run([npm, "install"], cwd=ROOT / pkg, timeout=1800)
        if code != 0:
            report("FAIL", f"`npm install` in {pkg} failed:\n{out.strip()[-800:]}")
        else:
            report("ok", f"npm install in {pkg}")


def check_media() -> None:
    for rel, what in (("tools/colmap/bin/colmap.exe", "COLMAP"),
                      ("tools/vocab_tree.bin", "vocab tree (loop closure)")):
        f = ROOT / rel
        report("ok" if f.exists() else "FAIL", f"{what} at {rel}",
               "" if f.exists() else "this came with the repo; a shallow or partial "
               "checkout can miss it - re-clone or `git checkout -- tools`")
    vids = sorted(p.name for p in (ROOT / "videos").glob("*.mp4")) if (ROOT / "videos").is_dir() else []
    if vids:
        report("ok", f"{len(vids)} take(s) in videos/: {', '.join(vids[:6])}")
    else:
        report("info", "no videos in videos/ (they are gitignored - 86 MB a take)",
               "drop a clip in as videos/<name>.mp4, then "
               "python pipeline.py run <name> --preset auto")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-train", action="store_true",
                    help="also build .venv310 with torch + gsplat (CUDA, ~3 GB download)")
    ap.add_argument("--check", action="store_true", help="report only, install nothing")
    ap.add_argument("--skip-node", action="store_true")
    args = ap.parse_args()

    if not (ROOT / "pipeline.py").exists():
        print(f"pipeline.py not found above {ROOT} - run this from inside the clone.")
        return 2

    print("\n=== clone integrity ===")
    check_lfs()
    check_media()
    if (ROOT / "tools/pc-engine").is_dir() and not any((ROOT / "tools/pc-engine").iterdir()):
        report("info", "tools/pc-engine submodule is empty (engine source)",
               "only needed to read or patch the engine: git submodule update --init --recursive")

    print("\n=== python environments ===")
    py = make_env(VENV, "3.12", "requirements.txt", check_only=args.check)

    if args.with_train:
        if make_env(VENV310, "3.10", "requirements-train.txt",
                    check_only=args.check) and not args.check:
            code, out = run([VENV310 / "Scripts" / "python.exe" if os.name == "nt"
                             else VENV310 / "bin" / "python",
                             "-c", "import torch;print(torch.cuda.is_available())"],
                            timeout=600)
            tail = out.strip()
            report("ok" if tail == "True" else "info",
                   f"torch sees the GPU: {tail or 'no answer'}",
                   "" if tail == "True" else "train will fall back or fail; check the "
                   "driver supports CUDA 12.4")
    elif not VENV310.exists():
        report("info", ".venv310 not created - `train` and `evals` need it",
               "re-run with --with-train")

    print("\n=== browser for the walk test ===")
    if py and not args.check:
        code, out = run([py, "-m", "playwright", "install", "chromium"], timeout=3600)
        report("ok" if code == 0 else "FAIL", "chromium for the headless walk test",
               "" if code == 0 else out.strip()[-400:])
    elif args.check:
        report("info", "`python -m playwright install chromium` would run for .venv")

    print("\n=== node toolchain ===")
    if args.skip_node:
        report("info", "node checks skipped (--skip-node)")
    else:
        check_node(check_only=args.check)

    print(f"\n{ok_count} ok, {warn_count} to note, {fail_count} blocking")
    if args.check:
        # An absent .venv is the expected finding here, and it was already reported
        # as one. Only a blocking count makes --check itself fail.
        print("--check: nothing installed. Re-run without it to apply the steps above.")
        return 0 if fail_count == 0 else 1
    if py is None:
        print("Stopped: the pipeline interpreter could not be created, so `doctor` "
              "could not run.")
        return 1

    print("\n=== pipeline.py doctor ===")
    code, out = run([py, "pipeline.py", "doctor"], timeout=1800)
    print(out.rstrip())
    if fail_count == 0 and code == 0:
        print("\nReady:  python pipeline.py run <name>   (a clip in videos/ named <name>.mp4)")
    return code


if __name__ == "__main__":
    sys.exit(main())
