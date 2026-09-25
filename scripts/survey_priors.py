"""Turn survey GPS into COLMAP pose priors so positions enter bundle adjustment.

A post-hoc similarity fit can only rotate/scale/translate an already-built model;
it cannot stop the model from drifting. Feeding ENU camera-centre priors with
their own reported uncertainty lets the mapper trade visual evidence against
position evidence per camera, which is the difference between "aligned to GPS"
and "constrained by GPS". std is the square root of the mean of the two
horizontal variances and the vertical variance: COLMAP's prior table takes one
scalar, so the anisotropy the receiver reports is deliberately averaged down.
"""
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np

try:  # imported as scripts.survey_priors by the tests
    from scripts import survey_georef as georef
except ImportError:  # imported flat by the workflow, which puts scripts/ on sys.path
    import survey_georef as georef


def _rows(path, required):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing input file: {path.name}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{number} is not JSON") from error
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError(f"{path.name}:{number} needs the fields {list(required)}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} holds no rows")
    return rows


def write_priors(keyframes_path, preparation_path, out_path, *, max_gap_s=2.0):
    """Write pose_priors.jsonl for keyframes that GPS can actually anchor."""
    keyframes = _rows(keyframes_path, ("file",))
    preparation_path = Path(preparation_path)
    if not preparation_path.is_file():
        raise ValueError(f"Missing input file: {preparation_path.name}")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    times = []
    for row in keyframes:
        value = row.get("t_sec", row.get("t_clip"))
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Keyframe {row['file']} has no usable timestamp")
        times.append(float(value))
    points, variance, matched = georef.sample_positions(times, preparation["telemetry"],
                                                        max_gap_s=max_gap_s)
    kept = [(row["file"], point, math.sqrt(float(spread)))
            for row, point, spread, hit in zip(keyframes, points, variance, matched) if hit]
    if not kept:
        raise ValueError("No keyframe falls inside a GPS bracket; refusing to map without anchors "
                         "rather than guessing positions between fixes.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for name, point, std in kept:
                stream.write(json.dumps({"file": name, "position": [float(v) for v in point],
                                         "std": std}) + "\n")
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"written": len(kept), "total": len(keyframes), "skipped": len(keyframes) - len(kept),
            "coordinate_frame": preparation["telemetry"]["coordinate_frame"],
            "std_basis": "sqrt(mean of two horizontal variances and the vertical variance)",
            "max_gap_s": float(max_gap_s)}


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Write COLMAP pose priors from survey GPS.")
    parser.add_argument("--keyframes", required=True, type=Path)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    arguments = parser.parse_args(argv)
    try:
        summary = write_priors(arguments.keyframes, arguments.preparation, arguments.out,
                               max_gap_s=arguments.max_gap_s)
    except (ValueError, OSError) as error:
        print(f"[priors] {error}", file=sys.stderr)
        return 3
    print(f"[priors] {summary['written']}/{summary['total']} keyframes anchored "
          f"({summary['skipped']} skipped)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
