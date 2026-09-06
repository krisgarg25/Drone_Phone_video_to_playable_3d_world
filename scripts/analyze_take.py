"""A captured take -> was it scanned well, measured off the phone's own pose log.

  .venv\\Scripts\\python.exe scripts\\analyze_take.py [FOLDER ...] [--report-only]

With no arguments it analyses videos/test1 — the take that came back with an
unmapped ceiling — so every take after it gets asked the same three questions:

  * how far did you walk while looking up (and down). A ceiling seen from one
    spot has no baseline: no amount of staring at it lets COLMAP triangulate it,
    because every observation is the same observation.
  * how high the phone was carried, which is what the page judges "floor" versus
    "ceiling" against.
  * whether data_room.json describes a room a tape measure would agree with.

Lines come out as pass / warn / fail. The exit code is 1 when anything failed
unless --report-only is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poses_lib import load_any, quat_to_R  # noqa: E402

# Keep these in step with ROOM_DEFAULTS in viewer/coverage_map.js
MIN_ROOM_H, MAX_ROOM_H = 1.85, 5.5
BASELINE_WANTED = 1.5      # m of walking under a level surface before it measures
STEEP = 25.0               # deg off-level that counts as looking up or down


def find_log(folder: Path):
    for pat in ("*poses*.jsonl", "*poses*.csv", "*.jsonl", "*.csv"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def pitch_deg(samples) -> np.ndarray:
    """Where each pose points, +90 straight up to -90 straight down. A WebXR
    camera looks down its own -Z, so the height row of that axis is the answer."""
    out = []
    for (_t, _p, q) in samples:
        f = -quat_to_R(q)[:, 2]
        out.append(np.degrees(np.arcsin(np.clip(f[1], -1.0, 1.0))))
    return np.asarray(out)


def band(pos: np.ndarray, idx: np.ndarray) -> dict:
    """Frames, share of the take, metres walked inside the band, and the widest
    horizontal gap between two of its points — the baseline a surface gets."""
    if idx.size == 0:
        return {"frames": 0, "frac": 0.0, "walk": 0.0, "span": 0.0}
    steps = np.linalg.norm(np.diff(pos[idx], axis=0), axis=1)
    consec = np.diff(idx) == 1          # a jump over lost frames is not a walk
    xz = pos[idx][:, [0, 2]]
    span = float(np.linalg.norm(xz.max(0) - xz.min(0))) if idx.size > 1 else 0.0
    return {"frames": int(idx.size), "frac": idx.size / max(1, len(pos)),
            "walk": float(steps[consec].sum()) if consec.any() else 0.0,
            "span": span}


class Report:
    def __init__(self, name: str):
        self.name, self.rows, self.failed = name, [], 0
        self.data = {}

    def head(self, line: str):
        self.rows.append(f"  {line}")

    def add(self, level: str, label: str):
        if level == "fail":
            self.failed += 1
        self.rows.append(f"  {level:<4}  {label}")

    def show(self):
        print(self.name)
        print("\n".join(self.rows))


def analyse(folder: Path) -> Report:
    rep = Report(str(folder))
    log = find_log(folder)
    if log is None:
        rep.add("fail", "no pose log in the folder — nothing to measure")
        return rep

    samples, fmt = load_any(log)
    pos = np.array([p for (_t, p, _q) in samples], float)
    t = np.array([s[0] for s in samples], float)
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    walked = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    pitch = pitch_deg(samples)
    ys = pos[:, 1]
    rep.head(f"{len(pos)} poses ({fmt}), {dur:.1f} s at {len(pos) / max(dur, 1e-6):.0f} Hz, "
             f"{walked:.1f} m walked")

    rep.add("pass" if walked > 2.0 else "fail",
            f"walked {walked:.1f} m in total" if walked > 2.0 else
            f"barely moved: {walked:.1f} m of walking - nothing in this take can be triangulated")

    rep.add("pass", f"phone height {ys.min():.2f}-{ys.max():.2f} m, median {np.median(ys):.2f} m "
                    f"(the floor/ceiling split is judged against this)")

    up = band(pos, np.nonzero(pitch > STEEP)[0])
    dn = band(pos, np.nonzero(pitch < -STEEP)[0])
    lv = band(pos, np.nonzero(np.abs(pitch) <= STEEP)[0])
    for b, name, said, need in ((up, "ceiling", "pointed up", BASELINE_WANTED),
                                (dn, "floor", "pointed down", BASELINE_WANTED),
                                (lv, "walls", "level", 1.2)):
        line = (f"{name}: {b['frac'] * 100:.0f}% of the take {said}, "
                f"{b['walk']:.1f} m walked in it, {b['span']:.1f} m of spread")
        if b["frac"] < 0.05:
            rep.add("warn", line + " - barely looked at")
        elif b["span"] < need:
            rep.add("warn" if name == "walls" else "fail",
                    line + f" (needs {need:.1f} m) - turning on the spot gives one "
                           f"viewpoint, and one viewpoint is not a measurement")
        else:
            rep.add("pass", line)

    rep.data = {"poses": len(pos), "seconds": dur, "walked": walked,
                "medianY": float(np.median(ys)),
                "ceilingFrac": up["frac"], "ceilingSpan": up["span"],
                "floorFrac": dn["frac"], "floorSpan": dn["span"],
                "wallSpan": lv["span"]}

    room_p = folder / "data_room.json"
    if not room_p.exists():
        rep.add("warn", "no data_room.json - this take predates the room fit")
        return rep
    room = json.loads(room_p.read_text(encoding="utf-8"))
    h = room.get("heightMeters")
    src = room.get("source", "none")
    rep.data.update({"roomH": h, "roomSource": src})
    if h is None:
        rep.add("warn", f"room ({src}): no height worked out - no ceiling was paired with the floor")
    elif MIN_ROOM_H <= h <= MAX_ROOM_H:
        rep.add("pass", f"room ({src}): {h:.2f} m tall, floor {room.get('floorMeters'):.2f} m, "
                        f"ceiling {room.get('ceilingMeters'):.2f} m")
    else:
        rep.add("fail", f"room ({src}): {h:.2f} m tall is not a room - the {src} evidence paired "
                        f"two surfaces that are not above each other")
    for key in ("floorSeenPct", "ceilingSeenPct"):
        v = room.get(key)
        if v is None:
            continue
        rep.add("pass" if v >= 30 else "warn",
                f"{key.replace('SeenPct', '')} {v}% of its area observed")
    walls = room.get("walls") or []
    spans = [s for s in (room.get("roomSpanMeters") or []) if s is not None]
    rep.add("pass" if walls else "warn",
            f"{len(walls)} walls" + (f", room {spans[0]:.1f} m across" if spans else ""))
    planes = room.get("planes") or []
    if planes and all("label" not in p for p in planes):
        rep.add("warn", "this export did not record plane labels - nothing to read either way")
    else:
        labels = sorted({p.get("label") for p in planes if p.get("label")})
        rep.add("pass" if labels else "warn",
                f"{len(labels)}/{len(planes)} planes carry the phone's own label" +
                (f" ({', '.join(labels)})" if labels
                 else " - every floor and ceiling is a guess"))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", type=Path, default=[Path("videos/test1")])
    ap.add_argument("--report-only", action="store_true",
                    help="print the report without a failing exit code")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    failed = 0
    for f in args.folders:
        folder = f if f.is_absolute() else root / f
        rep = analyse(folder)
        rep.show()
        failed += rep.failed
    print(f"\n{failed} rule(s) failed" if failed else "\nall rules passed")
    return 1 if failed and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
