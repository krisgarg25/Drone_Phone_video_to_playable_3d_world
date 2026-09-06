"""One command to verify the pipeline: every fast suite, in one report.

  .venv\\Scripts\\python.exe tests\\check_all.py            # minutes, no GPU work
  .venv\\Scripts\\python.exe tests\\check_all.py --e2e       # + every take in videos/
  .venv\\Scripts\\python.exe tests\\check_all.py --only robust,gate

Each suite is its own process on purpose. A hardening check that segfaults the
interpreter, or a numpy reduction that takes the heap with it, must not be able
to take the report down with it - the whole point of this work is that failures
stopped being invisible.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import robust as rb  # noqa: E402

# The repo's own .venv when it is there, otherwise whichever interpreter is running
# this file. A CI runner and anyone who installs requirements.txt into their own
# environment have no .venv at the repo root, and insisting on one made the suites
# unrunnable outside the machine they were written on.
_VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if _VENV_PY.exists():
    PY = _VENV_PY
else:
    PY = Path(sys.executable)
    print(f"[check_all] no .venv at {ROOT} - running the suites on {PY}")

# name -> (script, minutes allowed). Ordered cheapest first so a long failure at
# the end cannot hide that the first ten suites were clean.
SUITES = [
    ("robust", "test_robust.py", 5),
    ("capture", "test_capture.py", 10),
    ("collider", "test_collider.py", 10),
    ("gate", "test_gate.py", 10),
    ("unit", "run_tests.py", 10),
]
E2E = ("e2e", "test_e2e.py", 180)


def run_suite(script: str, minutes: float, extra: list[str]) -> tuple[bool, float, str]:
    t0 = time.time()
    cmd = [str(PY), str(ROOT / "tests" / script), *extra]
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=minutes * 60, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        ok = p.returncode == 0
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode("utf-8", "replace")
               + (e.stderr or b"").decode("utf-8", "replace"))
        out += f"\n!!! exceeded the {minutes:.0f} min budget"
        ok = False
    return ok, time.time() - t0, out


def main() -> int:
    rb.configure_streams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true",
                    help="also run the end-to-end smoke matrix over videos/ "
                         "(hours: this is the real 'any video gets an output' test)")
    ap.add_argument("--only", default=None, help="comma-separated suite names")
    ap.add_argument("--scenes", default=None, help="pass through to the e2e matrix")
    ap.add_argument("--verbose", action="store_true", help="print full suite output")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.only.split(",")] if args.only else None
    catalogue = SUITES + [E2E]
    known = {n for n, _, _ in catalogue}
    if wanted:
        bad_names = [w for w in wanted if w not in known]
        if bad_names:
            sys.exit(f"unknown suite(s): {bad_names}. Available: {', '.join(known)}")
    pick = [s for s in catalogue
            if (not wanted and (s[0] != "e2e" or args.e2e))
            or (wanted and s[0] in wanted)]

    results = []
    for name, script, minutes in pick:
        extra = (["--scenes", args.scenes] if name == "e2e" and args.scenes else [])
        print(f"\n{'=' * 70}\n>>> {name}  (tests/{script})\n{'=' * 70}", flush=True)
        ok, secs, out = run_suite(script, minutes, extra)
        keep = [l for l in out.splitlines()
                if "FAILED" in l or l.startswith(("FAIL", "!!!", "Traceback",
                                                  "    - ", "    ! "))]
        print(out.strip() if args.verbose else
              ("\n".join(keep[-40:]) if keep else out.strip()[-1500:]))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {secs / 60:.1f} min")
        results.append((name, ok))

    print(f"\n{'=' * 70}\nSUITE SUMMARY\n{'=' * 70}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} suites clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
