"""End-to-end shakedown: every take in videos/, every step, at smoke quality.

  .venv\\Scripts\\python.exe tests\\test_e2e.py
  .venv\\Scripts\\python.exe tests\\test_e2e.py --scenes rocks,temple
  .venv\\Scripts\\python.exe tests\\test_e2e.py --list

The claim this suite exists to test is the one the user actually cares about:
"any video input gets an output, without random failures". Nothing in the
per-step unit checks proves that, because each one runs a fixture that is known
to be well-formed. This runs the real graph over the real takes.

Train is NOT skipped. `smoke` runs 300 real steps on the real solve, because
frame, export, colors, collider, objects, surface, gate, evals, pairs and
walktest all consume splat.ply - skipping train would leave most of the graph
unexercised and the suite would pass vacuously.

What counts as a failure here is deliberately narrower than "the scene looks
good". A blurry or shallow world from thin footage is the pipeline answering the
question it was asked. A traceback, a step that never reached a terminal state,
or a run that produced nothing when the solve registered hundreds of frames -
those are the random failures being hunted.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PY = ROOT / ".venv" / "Scripts" / "python.exe"
TERMINAL = ("done", "skipped", "warning", "failed")

import robust as rb  # noqa: E402  (the failure taxonomy the runner reports in)

# Kinds that mean "this evidence cannot exist for this input", as opposed to
# "the code that makes this evidence is broken".
BENIGN_EVIDENCE = {rb.EMPTY_INPUT, rb.MISSING_TOOL, rb.UNSUPPORTED_ASSET}


def discover() -> list[str]:
    """Every take under videos/ that has at least one decodable clip."""
    import pipeline
    out = []
    vdir = ROOT / "videos"
    if not vdir.is_dir():
        # The clips are gitignored at 86 MB a take, so a missing videos/ is the
        # normal state of a fresh clone, not a broken checkout. The caller reports
        # it as "nothing was tested", which is deliberately not a pass.
        return out
    for p in sorted(vdir.iterdir()):
        name = p.stem if p.is_file() else p.name
        if not p.is_dir() and p.suffix.lower() not in pipeline.VIDEO_EXTS:
            continue
        try:
            src = pipeline.resolve_sources(name, None, None)
        except SystemExit:
            continue
        if src["videos"] or src["frames_dirs"]:
            out.append(name)
    return out


_EXC_RX = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit))\b")
_FRAME_RX = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
# A number in value position that no JSON parser accepts.
_NONFINITE_RX = re.compile(r":\s*(-?(?:NaN|Infinity))\b")


def tracebacks(scene: Path, declared: list[str], t0: float = 0.0) -> list[str]:
    """Any Python traceback in a step log is a crash we did not foresee.

    Scoped to the current graph: logs are NN-step.log, and a scene reusing work/
    data from an older pipeline still has orphans (12-collider.log when collider
    is step 09) whose contents say nothing about this run. Scoped to this run as
    well: a step that skipped kept its old log, and a crash the current code no
    longer produces is not this run's failure.

    One finding per line: the runner prints these individually, so a multi-line
    excerpt loses everything but the traceback header's own colon.
    """
    header = "Traceback (most recent call last)"
    expect = {f"{i + 1:02d}-{name}": name for i, name in enumerate(declared)}
    hits = []
    for log in sorted((scene / "logs").glob("*.log")):
        if log.stem not in expect:
            continue
        if t0 and log.stat().st_mtime < t0 - 5:
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        for block in text.split(header)[1:]:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            exc = next((ln for ln in reversed(lines) if _EXC_RX.match(ln)), lines[-1])
            frames = [m for ln in lines if (m := _FRAME_RX.search(ln))]
            # the deepest frame in this repo is the line that actually broke
            site = next((m for m in reversed(frames)
                         if str(ROOT) in (m.group(1) or "")), frames[-1] if frames else None)
            where = (f"  |  {Path(site.group(1)).name}:{site.group(2)} "
                     f"in {site.group(3)}") if site else ""
            hits.append(f"{log.name}: {exc}{where}")
    return hits


def evaluate(name: str, rc: int, secs: float, t0: float = 0.0) -> dict:
    """Turn one run into a verdict. Returns a row for the summary table."""
    import pipeline
    work = ROOT / "work" / name
    problems: list[str] = []

    rep = json.loads((work / "report.json").read_text(encoding="utf-8")) \
        if (work / "report.json").exists() else None
    if rep is None:
        problems.append("no report.json - the runner died before it could write one")

    # The declared graph, with the sources this run actually used.
    args = argparse.Namespace(cmd="run", name=name, quality="smoke", preset="auto",
                              timeout_scale=1.0, video=None, poses=None, fresh=False,
                              from_step=None, only=None, variant="cluster_shell",
                              cull="auto", mapper=None)
    src = pipeline.resolve_sources(name, None, None)
    cfg = pipeline.build_config(args, src, allow_auto_diag=False)
    declared = [s["name"] for s in pipeline.build_steps(cfg)]

    got = [s["name"] for s in (rep or {}).get("steps", [])]
    problems += [f"step '{m}' never reached a terminal state"
                 for m in declared if m not in got]
    problems += [f"step '{m}' is not in the declared graph"
                 for m in got if m not in declared]
    gate_hard: list[str] = []
    evidence_gap: list[str] = []
    for s in (rep or {}).get("steps", []):
        if s["status"] not in TERMINAL:
            problems.append(f"step '{s['name']}' has a non-terminal status "
                            f"'{s['status']}'")
        elif s["status"] == "failed":
            if s.get("kind") == "world-gate":
                # This one is a verdict, not a defect. The gate judged the world
                # unwalkable and named the rules; the runner carries on to the
                # evidence steps precisely so that judgement is checkable. What
                # this suite measures is whether the pipeline survived the
                # input - and a scene that yields assets plus a stated reason is
                # the answer, not a failure to report as one.
                gate_hard.append(s.get("detail", "world gate"))
            elif (s["name"] in pipeline.ADVISORY
                  and s.get("kind") in BENIGN_EVIDENCE):
                # The same reasoning one layer down. evals and pairs make the run
                # readable, not the world walkable, and the runner deliberately
                # ships without them - so a step that says "there was nothing to
                # score" is a gap in the evidence, while a crash in that same step
                # falls through below and stays a defect.
                evidence_gap.append(f"{s['name']}: [{s.get('kind')}] "
                                    f"{s.get('detail', '')[:120]}")
            else:
                problems.append(f"step '{s['name']}' failed "
                                f"[{s.get('kind', '?')}]: {s.get('detail', '')[:160]}")

    problems += [f"crash: {t}" for t in tracebacks(work, declared, t0)]

    # ---- did a world come out? ----
    asset = work / "viewer_assets"
    ply, heights = asset / "scene.ply", asset / "heights.f32"
    glb = next((p for p in (work / "pc" / "collision.collision.glb",
                            asset / "collision.glb") if p.exists()), None)
    n_reg = 0
    poses_file = work / "keyframes_poses.jsonl"
    if poses_file.exists():
        n_reg = sum(1 for l in poses_file.read_text(encoding="utf-8",
                                                    errors="replace").splitlines()
                    if l.strip())
    have = [ply.exists() and ply.stat().st_size > 1000,
            heights.exists() and heights.stat().st_size > 0,
            bool(glb) and glb.stat().st_size > 1000]
    missing = [w for w, h in zip(("viewer_assets/scene.ply", "heights.f32",
                                  "collision GLB"), have) if not h]
    if n_reg < 20:
        # Nothing to reconstruct from: the solve never registered a usable set of
        # cameras. That is an input problem, but the run still owes an honest
        # failure rather than a silent exit 0.
        if (rep or {}).get("status") == "complete":
            problems.append(f"status says complete on {n_reg} registered frames")
    elif missing:
        problems.append(f"{n_reg} frames registered but no world: missing "
                        + ", ".join(missing))

    # ---- is the JSON the browser loads actually JSON? ----
    # json.dumps writes a non-finite float as a bare NaN, json.loads reads it
    # back without complaint, and JSON.parse in the viewer rejects the whole
    # file. test2horizontal's walk test died that way on one unmeasured
    # "spawn_above_floor_m" in an otherwise healthy 300 KB collision.json.
    for jf in sorted(asset.glob("*.json")):
        raw = jf.read_text(encoding="utf-8", errors="replace")
        bad = _NONFINITE_RX.findall(raw)
        if bad:
            problems.append(f"{jf.name} holds {len(bad)} bare "
                            f"{bad[0]} token(s) - not parseable by the viewer")

    verdict = {
        "scene": name, "rc": rc, "secs": round(secs, 0),
        "registered": n_reg, "status": (rep or {}).get("status", "no-report"),
        "steps": f"{len(got)}/{len(declared)}",
        "assets": "ok" if not missing else ("none" if n_reg < 20 else "PARTIAL"),
        # "stale" means the world on disk predates this run - honest for a
        # resumed pass, but it must not be read as this run having produced it.
        "written": bool(ply.exists() and t0 and ply.stat().st_mtime >= t0 - 5),
        # The gate's own words, when it judged the world unwalkable. Kept out of
        # `problems` because it is a verdict about the footage, not a crash.
        "gate": "; ".join(gate_hard),
        # An evidence step that could not make evidence, as opposed to one that
        # broke doing it. Shown on the row so a quiet gap is still a visible one.
        "evidence": "; ".join(evidence_gap),
        "problems": problems,
    }
    if rc != 0 and not missing and n_reg >= 20 and not gate_hard:
        verdict["problems"] = verdict["problems"] + [f"exit code {rc}"]
    return verdict


def run(name: str, timeout: float) -> dict:
    cmd = [str(PY), str(ROOT / "pipeline.py"), "run", name, "--quality", "smoke"]
    print(f"\n{'=' * 70}\n=== {name}: {' '.join(cmd[2:])}\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        rc = p.returncode
        tail = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = -1
        tail = ((e.stdout or b"").decode("utf-8", "replace")
                + (e.stderr or b"").decode("utf-8", "replace"))
        print(f"!!! {name}: exceeded the {timeout / 60:.0f} min budget", flush=True)
    print(tail[-4000:], flush=True)
    t1 = time.time()
    v = evaluate(name, rc, t1 - t0, t0)
    assets = v["assets"] + ("/fresh" if v["written"]
                            else "/stale" if v["assets"] == "ok" else "")
    flag = "PASS" if not v["problems"] else "FAIL"
    gate = f", gate={v['gate']}" if v.get("gate") else ""
    ev = f", evidence missing: {v['evidence']}" if v.get("evidence") else ""
    print(f"--- {flag} {name}: {v['status']}, {v['registered']} registered, "
          f"assets={assets}{gate}{ev}, {v['secs'] / 60:.1f} min", flush=True)
    for prob in v["problems"]:
        print(f"    ! {prob}", flush=True)
    return v


def main() -> int:
    import robust as rb
    rb.configure_streams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None, help="comma-separated subset")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--per-scene-minutes", type=float, default=90.0)
    args = ap.parse_args()

    scenes = args.scenes.split(",") if args.scenes else discover()
    if args.list:
        print("\n".join(scenes))
        return 0
    if not scenes:
        # "0/0 takes clean" exiting 0 is the worst outcome a test harness can produce:
        # it is a green light for nothing. pipeline.py's own comment on --quality smoke
        # says why this suite may not quietly pass without a reconstruction to check.
        print("[e2e] no takes found in videos/ - the clips are gitignored (86 MB each), "
              "so a clean clone ships no test data.\n"
              "      Put a clip at videos/<name>.mp4 and re-run, or pass "
              "--scenes <name>.\n"
              "      Nothing was reconstructed, so this is not a pass.")
        return 1
    print(f"[e2e] {len(scenes)} takes: {', '.join(scenes)}")

    rows = [run(s, args.per_scene_minutes * 60) for s in scenes]

    print(f"\n{'=' * 70}\nE2E SUMMARY\n{'=' * 70}")
    print(f"{'scene':<18}{'status':<24}{'steps':<8}{'reg':>6}  "
          f"{'assets':<10}{'min':>7}  {'rc':<3} gate")
    for r in rows:
        assets = r["assets"] + ("/fresh" if r["written"]
                                else "/stale" if r["assets"] == "ok" else "")
        print(f"{r['scene']:<18}{r['status']:<24}{r['steps']:<8}"
              f"{r['registered']:>6}  {assets:<10}{r['secs'] / 60:>7.1f}  "
              f"{r['rc']:<3} {r['gate'] or '-'}")
    bad = [r for r in rows if r["problems"]]
    print()
    for r in bad:
        print(f"FAIL {r['scene']}")
        for p in r["problems"]:
            print(f"    - {p}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} takes clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
