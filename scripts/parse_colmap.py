"""Parse COLMAP text model + our keyframes manifest -> keyframes_poses.jsonl.

Expects work/<video>/colmap/sparse/txt/{cameras.txt,images.txt} (from
`colmap model_converter --output_type TXT`) and work/<video>/keyframes.jsonl.

Output rows: {file, t_sec, camera:{R_rowmajor, t, fx, fy, cx, cy, width, height}}
where x_cam = R @ x_world + t  (COLMAP convention, world->camera).

This row set is what every downstream metre comes from: solve_frame reads the
rotations to find gravity and the translations to find camera height, so a pose
silently dropped here is a camera that never constrained the ground fit, and a
t_sec that comes out None sorts badly rather than failing loudly.

Usage:
  python parse_colmap.py --work work/rocks
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as rb  # noqa: E402


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def parse_cameras(path: Path) -> dict:
    """camera_id -> {fx, fy, cx, cy}. Unknown models keep their first four params."""
    cams = {}
    for ln, line in enumerate(rb.read_text(path).splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        try:
            cam_id = int(p[0])
            model = p[1]
            nums = [float(x) for x in p[4:]]
        except (IndexError, ValueError):
            rb.warn(f"cameras.txt:{ln} is not a CAM_ID MODEL W H PARAMS... row, "
                    f"skipped: {line[:60]}")
            continue
        if len(nums) < 3:
            rb.warn(f"cameras.txt:{ln} has {len(nums)} params, needs at least 3 "
                    f"(f, cx, cy), skipped")
            continue
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_OPENCV"):
            fx = fy = nums[0]
            cx, cy = nums[1], nums[2]
        else:  # PINHOLE / OPENCV / FULL_OPENCV: fx fy cx cy [+ distortion]
            if len(nums) < 4:
                rb.warn(f"cameras.txt:{ln} model {model} needs 4 params, got "
                        f"{len(nums)}, skipped")
                continue
            fx, fy, cx, cy = nums[0], nums[1], nums[2], nums[3]
        cams[cam_id] = dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    return cams


def parse_images(path: Path) -> dict:
    """name -> pose. images.txt holds TWO lines per image after the header.

    The pairing is the definition of which line is the pose; filtering by
    content instead (the old "keep lines with exactly 10 tokens") both discarded
    every image whose file name contains a space and let a 2D-points line whose
    ninth field happened to be integer-shaped pass as a pose. This model's 419
    images came back as 838.

    The pose line splits to 9 fields plus a remainder, and that remainder IS the
    name, so it must be split with a cap rather than by all whitespace.
    """
    reg, bad = {}, 0
    rows = [l for l in rb.read_text(path).splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    for pair in range(0, len(rows) - 1, 2):
        line = rows[pair]
        p = line.split(maxsplit=9)
        if len(p) < 10:
            bad += 1
            continue
        try:
            q = np.array([float(x) for x in p[1:5]])
            t = np.array([float(x) for x in p[5:8]])
            cam_id = int(p[8])
        except ValueError:
            bad += 1
            continue
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))) or cam_id < 0:
            rb.warn(f"{path.name}: non-finite or unregistered pose for "
                    f"{p[9][:40]}, skipped")
            bad += 1
            continue
        reg[p[9].strip()] = dict(
            R=qvec2rot(q / max(np.linalg.norm(q), 1e-12)), t=t, cam_id=cam_id)
    if bad:
        rb.warn(f"{bad} image records in {path.name} were not usable")
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    txt = args.work / "colmap" / "sparse" / "txt"
    kf_file = args.work / "keyframes.jsonl"

    for f, label in ((txt / "cameras.txt", "cameras.txt (written by the colmap step)"),
                     (txt / "images.txt", "images.txt (written by the colmap step)")):
        rb.require_file(f, label)

    cams = parse_cameras(txt / "cameras.txt")
    reg = parse_images(txt / "images.txt")

    kf_meta = {}
    for m in rb.jsonl_rows(kf_file, required=("file",)):
        kf_meta[m["file"]] = m.get("t_sec")

    # A missing t_sec must not become None in a sort key: comparing None with a
    # float raises TypeError, which is how a name mismatch between the manifest
    # and COLMAP turned into a crash here instead of a warning.
    def when(name: str) -> float:
        v = kf_meta.get(name)
        return float(v) if rb.finite(v) else 0.0

    rows, orphan_cam = [], 0
    for name, r in sorted(reg.items(), key=lambda kv: when(kv[0])):
        c = cams.get(r["cam_id"])
        if c is None:
            orphan_cam += 1
            continue
        R = np.asarray(r["R"], np.float64)
        if not (np.all(np.isfinite(R)) and np.all(np.isfinite(r["t"]))):
            rb.warn(f"{name}: non-finite pose, skipped")
            continue
        rows.append({
            "file": name,
            "t_sec": (round(when(name), 3) if name in kf_meta else None),
            "camera": {
                "R_rowmajor": [list(map(float, row)) for row in R],
                "t": list(map(float, r["t"])),
                **{k: float(c[k]) for k in ("fx", "fy", "cx", "cy")},
            },
        })
    if orphan_cam:
        rb.warn(f"{orphan_cam} registered images named a camera id absent from "
                f"cameras.txt and were dropped")
    out = args.work / "keyframes_poses.jsonl"
    rb.write_text(out, "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    print(f"[poses] {len(rows)} registered / {len(kf_meta)} keyframes -> {out}")
    if not rows:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"COLMAP reported {len(reg)} registered images but none produced a "
            f"usable pose - check cameras.txt camera ids against images.txt",
            returncode=3)
    missing_t = sum(1 for r in rows if r["t_sec"] is None)
    if missing_t:
        rb.warn(f"{missing_t}/{len(rows)} poses have no timestamp - clip ordering "
                f"and any speed-based scale ruler will be less reliable "
                f"(manifest names vs COLMAP names diverged?)")


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[poses] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
