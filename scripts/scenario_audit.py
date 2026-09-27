"""E2: audit a finished scene against the capture preset it claims it was built as.

The preset table has always existed and the gate has always existed, and neither one
knew about the other. A `room` run could produce a 107 m aerial world and both files
would be happily wrong in different directions, because nothing compared a preset's
promise with what the reconstruction turned out to be. That comparison is this script:
cheap, read-only, and entirely CPU - it reads artefacts a finished run already left
behind and says where they disagree with the scenario that was chosen.

It answers three questions and records them in `work/<scene>/scenario_audit.json`:

  what was this built as?     scenario.json (E0), or "unknown" for a run that predates it
  what did it actually become? footprint, grid cell, camera working height, registration
                               rate, scale regime, gate verdict - all measured, not inferred
  does the first agree with    a regime band per preset, plus consistency rules that are
  the second?                  bugs rather than judgement calls (a navmesh baked for a
                               1.7 m human under a preset that shipped a 0.15 m character)

`--all` walks every scene on disk and prints the ledger the E2 roadmap line asks for:
which capture style has real footage behind it, and which one passes its own gate.

  python scripts/scenario_audit.py rocks
  python scripts/scenario_audit.py --all
  python scripts/scenario_audit.py --all --json work/scenario_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402

# Working-height and footprint bands, in metres, per capture preset.
#
# These are stated expectations about the physical world, not measurements of this
# repository's footage - the honest label for them is "what a scene of this kind is".
# A hand holds a phone at 1.2-2.2 m; a multirotor surveyed at 40 m AGL is not the same
# object, and 40 m of rope cannot make a bedroom. Where a band is wide it is because
# the legitimate range is wide, not because the number is soft.
REGIMES = {
    "room":           {"agl": (0.4, 3.0),   "footprint": (1.5, 25.0),
                       "basis": "a phone is held at arm height inside one room"},
    "indoor_large":   {"agl": (0.8, 4.0),   "footprint": (15.0, 200.0),
                       "basis": "a floorplate walked, not a room orbited"},
    "outdoor_building": {"agl": (1.5, 40.0), "footprint": (5.0, 150.0),
                         "basis": "a facade circled from the ground or low air"},
    "drone":          {"agl": (5.0, 150.0), "footprint": (10.0, 600.0),
                       "basis": "an airframe in orbit or push-forward"},
    "drone_mapping":  {"agl": (15.0, 250.0), "footprint": (25.0, 2000.0),
                       "basis": "a nadir grid flown at mapping altitude"},
    "object":         {"agl": (0.05, 1.5),  "footprint": (0.02, 4.0),
                       "basis": "a turntable circle around one small subject"},
    "sky_heavy":      {"agl": (3.0, 150.0), "footprint": (10.0, 600.0),
                       "basis": "an aerial pass with open sky in frame"},
    "corridor":       {"agl": (0.8, 6.0),   "footprint": (8.0, 300.0),
                       "basis": "a long straight run down a hall or street"},
}

# A cell coarser than this cannot represent the thing the same scene's walk model is
# asked to avoid: it is larger than a doorway, a person, or a parkable sofa.
MAX_USEFUL_CELL_M = 1.5

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv")
"""Same list `pipeline.py` resolves sources with; duplicated here only so the ledger can
name a scene that has footage but no work directory yet. robust.py does not own this."""


def registration(work: Path) -> tuple[int, int] | None:
    """(registered, extracted) frames, from the two manifests this pipeline already
    treats as the truth about which views made it into the model.

    The obvious source - count the positive `POINT3D_ID` rows in COLMAP's images.txt -
    was tried first here and is wrong, because that column is the LAST field on a line
    whose 10th field is a filename, and a filename may contain spaces. Splitting on
    whitespace and reading `parts[-1]` therefore parses the tail of a name, raises, and
    silently under-counts whatever it does not raise on. The repo already solves this
    correctly in `parse_colmap.py`: `keyframes.jsonl` lists every frame that was given to
    COLMAP, `keyframes_poses.jsonl` lists every one the mapper placed. Those two counts
    are also exactly what the project API already reports as `registered_count`, so the
    audit and the UI cannot drift.
    """
    total = sum(1 for _ in rb.jsonl_rows(work / "keyframes.jsonl", required=("file",)))
    reg = sum(1 for _ in rb.jsonl_rows(work / "keyframes_poses.jsonl",
                                        required=("file",)))
    return (reg, total) if total else None


def measured_facts(root: Path, scene: str) -> dict:
    """Everything this scene demonstrably is, read out of the files a run leaves behind."""
    work = root / "work" / scene
    asset = work / "viewer_assets"
    out: dict = {"scene": scene, "has_assets": asset.is_dir()}
    scenario = rb.read_json(work / "scenario.json", {}) or {}
    gate = rb.read_json(asset / "world_check.json", {}) or {}
    col = rb.read_json(asset / "collision.json", {}) or {}
    frame = rb.read_json(work / "frame.json", {}) or {}
    rooms = rb.read_json(asset / "rooms.json", {}) or {}

    out["declared_preset"] = scenario.get("preset")
    out["declared_by"] = scenario.get("decided_by")
    out["recorded"] = bool(scenario)
    out["quality"] = scenario.get("quality")
    out["applied"] = scenario.get("applied") or {}

    nx, nz = col.get("nx"), col.get("nz")
    cell = col.get("cell")
    if isinstance(nx, int) and isinstance(nz, int) and rb.finite(cell):
        out["grid"] = [nx, nz]
        out["cell_m"] = round(float(cell), 4)
        out["footprint_m"] = round(max(nx, nz) * float(cell), 1)
    out["character_height_m"] = col.get("character_height")
    out["scale_m_per_unit"] = frame.get("scale_m_per_unit")
    out["scale_source"] = frame.get("scale_source")
    reg = registration(work)
    if reg:
        out["registered"], out["camera_frames"] = reg
        out["registration_pct"] = round(100.0 * reg[0] / reg[1], 1)

    thr = gate.get("thresholds") or {}
    out["gate_status"] = gate.get("status")
    out["gate_hard"] = gate.get("hard_failures") or []
    out["gate_warnings"] = gate.get("warnings") or []
    # The camera working height is the single number that says what kind of capture this
    # was. The gate already computes it and used to throw it away into a prose string.
    by_name = {c.get("name"): c for c in (gate.get("checks") or [])}
    cam = (by_name.get("cameras above the ground they filmed") or {}).get("metric")
    if cam and rb.finite(cam.get("value")):
        out["camera_agl_m"] = float(cam["value"])
    cov = (by_name.get("heightfield coverage") or {}).get("metric")
    if cov and rb.finite(cov.get("value")):
        out["measured_coverage_pct"] = float(cov["value"])
    head = by_name.get("headroom above the walk surface") or {}
    out["headroom"] = {"status": head.get("status"),
                       "value_m": (head.get("metric") or {}).get("value")}
    ceil = [r["ceiling"] for r in (rooms.get("rooms") or [])
            if isinstance(r.get("ceiling"), dict)]
    out["ceiling_observed"] = any(rb.finite(c.get("clear_height_above_floor_m"))
                                  for c in ceil)

    # The bake already writes the agent it used into nav.json's `build` block, so the
    # two path planners in one scene can be compared against each other rather than
    # against a guess. Parsed rather than grepped: an 8-figure number in a prose string
    # is how the coverage bug stayed invisible for so long.
    try:
        nav = json.loads((work / "pc" / "nav.json").read_text(encoding="utf-8"))
        build = nav.get("build") or {}
        if isinstance(build.get("walkableHeight"), (int, float)):
            out["nav_agent_height_m"] = float(build["walkableHeight"])
        if isinstance(build.get("walkableRadius"), (int, float)):
            out["nav_agent_radius_m"] = float(build["walkableRadius"])
        if isinstance(build.get("cellSize"), (int, float)):
            out["nav_cell_m"] = float(build["cellSize"])
    except (OSError, ValueError):
        pass
    return out


def findings(facts: dict) -> list[dict]:
    """Where the declared scenario and the measured world disagree, and why that matters."""
    f: list[dict] = []

    def add(kind, severity, msg, fix=None):
        f.append({"kind": kind, "severity": severity, "message": msg,
                  **({"fix": fix} if fix else {})})

    if not facts.get("has_assets"):
        add("incomplete", "info", "no viewer assets here yet - nothing to audit")
        return f
    preset = facts.get("declared_preset")
    if not facts.get("recorded"):
        add("unrecorded", "warn",
            "this run predates the scenario record, so which preset built it is not "
            "knowable from the directory - re-run `pipeline.py scan` to record what the "
            "footage says now", None)
        regime = None
    else:
        regime = REGIMES.get(preset)

    agl, fp = facts.get("camera_agl_m"), facts.get("footprint_m")
    if regime and agl is not None:
        lo, hi = regime["agl"]
        if not lo <= agl <= hi:
            add("regime", "warn",
                f"built as `{preset}` (expects a working height of {lo:g}-{hi:g} m, "
                f"{regime['basis']}) but the cameras sat {agl:g} m above the ground they "
                f"filmed",
                _nearest_preset(facts, preset))
    if regime and fp is not None:
        lo, hi = regime["footprint"]
        if not lo <= fp <= hi:
            add("regime", "warn",
                f"built as `{preset}` (expects {lo:g}-{hi:g} m across) but the grid it "
                f"produced is {fp:g} m across",
                _nearest_preset(facts, preset))

    cell = facts.get("cell_m")
    if rb.finite(cell) and cell > MAX_USEFUL_CELL_M:
        add("resolution", "warn",
            f"the walk grid resolves at {cell:.2f} m per cell, coarser than the "
            f"{MAX_USEFUL_CELL_M:g} m doorway a walk model has to see: at this resolution "
            f"the collider cannot represent an obstacle smaller than a room",
            "set `cell_meters` on this preset, or cull the floater cloud that is "
            "inflating the bounds")
    nav_cell = facts.get("nav_cell_m")
    if rb.finite(cell) and rb.finite(nav_cell) and cell > 4 * nav_cell:
        add("inconsistent-resolution", "warn",
            f"the physics grid resolves at {cell:.2f} m per cell while the navigation mesh "
            f"baked out of it resolves at {nav_cell:.2f} m - the route is planned "
            f"{cell / nav_cell:.0f}x finer than the surface it will actually collide with",
            "one scene, one agent resolution: derive the bake's --cell from the preset's "
            "grid instead of a constant")
    char_h = facts.get("character_height_m")
    nav_h = facts.get("nav_agent_height_m")
    if rb.finite(char_h) and rb.finite(nav_h) and abs(char_h - nav_h) > 0.05 * nav_h:
        add("inconsistent-scale", "warn",
            f"the physics floor was built for a {char_h:g} m character while the "
            f"navigation mesh was baked for a {nav_h:g} m agent - the two path planners "
            f"in one scene disagree about how tall the walker is",
            "nav_params() must take the preset's character height instead of its own "
            "constant")
    if char_h is None and facts.get("recorded"):
        add("lost-knob", "warn",
            "collision.json carries no character height, so the gate and the viewer "
            "defaulted it independently - the preset's value never survived the "
            "re-export", "forward the export knobs on the reexport step")

    scale_src = (facts.get("scale_source") or "").lower()
    if "flight speed" in scale_src or "clip duration" in scale_src:
        add("scale", "warn",
            f"every metre figure here is anchored on `\"{facts.get('scale_source')}\"`, "
            f"which is an assumed flight speed multiplied by a clip length and has never "
            f"been measured against anything",
            "land a surveyed checkpoint or set a known camera height before quoting an "
            "area or a volume")
    if facts.get("gate_hard"):
        add("gate", "fail",
            "the world gate blocked this scene: " + ", ".join(facts["gate_hard"]),
            "open the quality panel for the number each check judged on")
    elif facts.get("gate_warnings"):
        add("gate", "warn",
            "the world gate shipped this scene with warnings: "
            + ", ".join(facts["gate_warnings"]), None)

    pct, total = facts.get("registration_pct"), facts.get("camera_frames")
    if rb.finite(pct) and pct < 80 and rb.finite(total):
        add("registration", "warn",
            f"only {pct:.0f}% of the {total:.0f} frames COLMAP was given actually "
            f"registered ({facts.get('registered')}): the views that did not are views "
            f"the model cannot see",
            "re-shoot with slower arcs and more overlap, or force the dynamics pass")
    return f


def _nearest_preset(facts: dict, chosen: str) -> str:
    """Which preset the measured world actually fits - the operator's next click."""
    scores = []
    for name, band in REGIMES.items():
        s = 0.0
        for key, (lo, hi) in (("camera_agl_m", band["agl"]),
                              ("footprint_m", band["footprint"])):
            v = facts.get(key)
            if not rb.finite(v):
                continue
            s += 0.0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi)) / max(hi, 1.0)
        scores.append((s, name))
    scores.sort()
    best = scores[0][1] if scores else chosen
    return (f"`{best}` fits what this scene measured" if best != chosen
            else "the chosen preset already fits best")


def ledger(root: Path) -> list[dict]:
    if not (root / "work").is_dir():
        return []
    scenes = {p.name for p in (root / "work").iterdir() if p.is_dir()}
    vids = root / "videos"
    if vids.is_dir():
        scenes |= {p.stem for p in vids.iterdir()
                   if p.suffix.lower() in VIDEO_EXTS}
    rows = []
    for s in sorted(scenes):
        facts = measured_facts(root, s)
        rows.append({**facts, "findings": findings(facts)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", help="scene name under work/")
    ap.add_argument("--all", action="store_true", help="audit every scene on disk")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--json", type=Path, default=None,
                    help="where to write the audit (default: work/<scene>/scenario_audit.json)")
    args = ap.parse_args()

    if args.all:
        rows = ledger(args.root)
        print(f"{'scene':22}{'preset':15}{'gate':10}{'AGL':>7}{'foot':>8}{'cell':>7}"
              f"{'reg%':>6}  findings")
        for r in rows:
            flags = ",".join(sorted({f["kind"] for f in r["findings"]})) or "-"
            print(f"{r['scene'][:21]:22}{str(r.get('declared_preset') or '-')[:14]:15}"
                  f"{str(r.get('gate_status') or '-')[:9]:10}"
                  f"{r.get('camera_agl_m') if r.get('camera_agl_m') is not None else '-':>7}"
                  f"{r.get('footprint_m') if r.get('footprint_m') is not None else '-':>8}"
                  f"{r.get('cell_m') if r.get('cell_m') is not None else '-':>7}"
                  f"{r.get('registration_pct') if r.get('registration_pct') is not None else '-':>6}"
                  f"  {flags}")
        if args.json:
            rb.write_json(args.json, rows)
            print(f"\nwrote {args.json}")
        covered = {r.get("declared_preset") for r in rows if r.get("recorded")}
        untouched = sorted(set(REGIMES) - (covered or set()))
        print(f"\nscenarios with a real scene behind them: "
              f"{', '.join(sorted(c for c in covered if c)) or 'none'}")
        if untouched:
            print(f"scenarios NO footage on this machine has ever exercised: "
                  f"{', '.join(untouched)} - their knobs are assertions, not results")
        return

    if not args.scene:
        sys.exit("give a scene name, or --all")
    facts = measured_facts(args.root, args.scene)
    fs = findings(facts)
    out = args.json or (args.root / "work" / args.scene / "scenario_audit.json")
    rb.write_json(out, {**facts, "findings": fs,
                        "regime_expectation": REGIMES.get(facts.get("declared_preset") or "")})
    print(f"[audit] {args.scene}: {facts.get('declared_preset') or 'unrecorded'}"
          f" -> gate {facts.get('gate_status') or 'unknown'}")
    for key in ("camera_agl_m", "footprint_m", "cell_m", "registration_pct",
                "measured_coverage_pct", "character_height_m", "scale_source",
                "headroom"):
        if facts.get(key) is not None:
            print(f"  {key:22} {facts[key]}")
    for f in fs:
        print(f"  [{f['severity']}] {f['message']}")
        if f.get("fix"):
            print(f"      fix: {f['fix']}")
    if not fs:
        print("  no disagreement between the scenario chosen and the world produced")
    print(f"verdict: {out}")


if __name__ == "__main__":
    rb.configure_streams()
    main()
