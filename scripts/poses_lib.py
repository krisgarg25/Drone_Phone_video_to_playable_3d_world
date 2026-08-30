"""Shared phone-pose math + log IO for the AR-prior pipeline.

Canonical in-memory sample: (t: float seconds, pxyz: (3,) meters, qxyzw: (4,))
with the quaternion in scalar-last (x, y, z, w) order and the pose being
CAMERA-TO-WORLD in a right-handed Y-up metric frame (ARCore / ARKit native).

Canonical on-disk log (what converters emit and what we re-read):
  JSONL: {"t": 1.234, "pxyz": [x, y, z], "qxyzw": [qx, qy, qz, qw]}

Also reads, via auto-detection:
  - generic ARCore logger CSV: header (or bare) rows
        timestamp, qx, qy, qz, qw, tx, ty, tz          (camera-to-world)
  - Record3D metadata JSON ("poses": N x 7 [qx,qy,qz,qw,tx,ty,tz]) which has
    NO timestamps -- those samples get t = row_index / assumed_fps.

COLMAP notes (why only positions end up as priors today): COLMAP >= 3.11
pose_priors store the CAMERA CENTER (world frame) plus covariance; orientation
priors are not consumed by the mapper. We keep orientations anyway so
downstream tools (spatial gating, future COLMAP) can use them.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- quaternion
def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_slerp(q0, q1, u: float) -> np.ndarray:
    """Slerp between scalar-last quaternions; shortest arc."""
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:  # take the short way around
        q0, d = -q0, -d
    if d > 1.0 - 1e-9:  # nearly identical: nlerp is stable
        return quat_normalize((1 - u) * q0 + u * q1)
    th = math.acos(d)
    s = math.sin(th)
    return quat_normalize((math.sin((1 - u) * th) / s) * q0 + (math.sin(u * th) / s) * q1)


def quat_to_R(q) -> np.ndarray:
    """Scalar-last quaternion -> 3x3 rotation matrix."""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R) -> np.ndarray:
    """3x3 rotation matrix -> scalar-last quaternion (numerically safe branch)."""
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize([x, y, z, w])


def c2w_center(pxyz, _q=None) -> np.ndarray:
    """AR camera center in world coords == the c2w translation itself."""
    return np.asarray(pxyz, np.float64)


# ------------------------------------------------------------------ loading
def _samples_from_rows(rows) -> list[tuple]:
    out = []
    for t, p, q in rows:
        p = np.asarray(p, np.float64).reshape(3)
        q = quat_normalize(q)
        out.append((float(t), p, q))
    out.sort(key=lambda r: r[0])
    return out


def load_canonical_jsonl(path: Path) -> list[tuple]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append((d["t"], d["pxyz"], d["qxyzw"]))
    return _samples_from_rows(rows)


_CSV_ALIASES = {
    "t": ["t", "time", "timestamp", "ts", "t_sec", "time_sec", "seconds",
          "timestamp_ns", "timestamp_s"],
    "qx": ["qx", "q_x", "quaternionx", "rotationx"],
    "qy": ["qy", "q_y", "quaterniony", "rotationy"],
    "qz": ["qz", "q_z", "quaternionz", "rotationz"],
    "qw": ["qw", "q_w", "quaternionw", "rotationw"],
    "tx": ["tx", "t_x", "px", "positionx", "x_m", "posx"],
    "ty": ["ty", "t_y", "py", "positiony", "y_m", "posy"],
    "tz": ["tz", "t_z", "pz", "positionz", "z_m", "posz"],
}
# bare-file fallback order: ARCore-Data-Logger convention
_BARE_ORDER = ["t", "qx", "qy", "qz", "qw", "tx", "ty", "tz"]


def load_csv_log(path: Path) -> list[tuple]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if not first:
            raise ValueError(f"{path} is empty")
        header = [h.strip().lower() for h in first]
        has_alpha_header = any(
            h.replace("_", "").replace("-", "").isalpha() for h in header)
        colmap = {}
        if has_alpha_header:
            for want, names in _CSV_ALIASES.items():
                for i, h in enumerate(header):
                    if h in names and want not in colmap:
                        colmap[want] = i
            missing = set(_BARE_ORDER) - set(colmap)
            if missing:
                raise ValueError(f"{path}: CSV header unrecognized, missing {sorted(missing)}")
        else:
            if not all(_is_num(v) for v in first):
                raise ValueError(f"{path}: cannot parse header {header[:8]}")
            colmap = {k: i for i, k in enumerate(_BARE_ORDER)}
        rows = []
        ns_div = None

        def consume(rec):
            nonlocal ns_div
            if not rec or all(not c.strip() for c in rec):
                return

            def num(key):
                return float(rec[colmap[key]])

            t = num("t")
            if ns_div is None:  # epoch nanoseconds / microseconds heuristic
                if abs(t) > 1e15:
                    ns_div = 1e9
                elif abs(t) > 1e11:
                    ns_div = 1e6
                else:
                    ns_div = 1.0
            rows.append((t / ns_div,
                         (num("tx"), num("ty"), num("tz")),
                         (num("qx"), num("qy"), num("qz"), num("qw"))))

        if not has_alpha_header:  # we consumed a data row already
            consume(first)
        for rec in reader:
            consume(rec)
    return _samples_from_rows(rows)


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def load_record3d_metadata(path: Path, assumed_fps: float = 30.0) -> list[tuple]:
    """Record3D export 'metadata' JSON. poses rows: [qx,qy,qz,qw,tx,ty,tz].

    Row i corresponds to rgbd/i.jpg 1:1, so timestamps are synthesized from
    row order (assumed_fps) -- good enough because the record3d import path
    uses frames directly rather than syncing against an external video.
    """
    meta = json.loads(path.read_text(encoding="utf-8"))
    poses = np.asarray(meta["poses"], np.float64).reshape(-1, 7)
    ts = meta.get("frameTimestamps")
    rows = []
    for i, p in enumerate(poses):
        t = float(ts[i]) if ts is not None else i / assumed_fps
        rows.append((t, tuple(p[4:7]), tuple(p[0:4])))
    return _samples_from_rows(rows)


def load_spectacular_jsonl(path: Path) -> list[tuple]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "time" in d and "position" in d:
            t = float(d["time"])
            pos = d["position"]
            if isinstance(pos, dict):
                p = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
            else:
                p = (float(pos[0]), float(pos[1]), float(pos[2]))
            ori = d.get("orientation", {})
            if isinstance(ori, dict):
                # Spectacular AI orientation is w, x, y, z -> canonical is x, y, z, w
                q = (float(ori.get("x", 0.0)), float(ori.get("y", 0.0)),
                     float(ori.get("z", 0.0)), float(ori.get("w", 1.0)))
            elif isinstance(ori, (list, tuple)) and len(ori) == 4:
                q = (float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
            else:
                q = (0.0, 0.0, 0.0, 1.0)
            rows.append((t, p, q))
    return _samples_from_rows(rows)


def extract_spectacular_poses(folder: Path, out_path: Path) -> list[tuple]:
    """Fast extraction of VIO camera poses directly from Spectacular AI recording in-process."""
    import spectacularAI as sai
    ffmpeg_dir = Path(__file__).resolve().parent.parent / "tools/ffmpeg/ffmpeg-9.0.1-essentials_build/bin"
    os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"

    poses = []
    def on_vio(out):
        cam = out.getCameraPose(0)
        p = cam.pose.position
        q = cam.pose.orientation
        poses.append({
            'time': float(cam.pose.time),
            'position': {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)},
            'orientation': {'x': float(q.x), 'y': float(q.y), 'z': float(q.z), 'w': float(q.w)}
        })

    print(f"[priors] Extracting VIO trajectory from {folder}...", flush=True)
    replay = sai.Replay(str(folder), configuration={'useSlam': 'True'})
    replay.setOutputCallback(on_vio)
    replay.runReplay()
    replay.close()

    out_path.write_text('\n'.join(json.dumps(p) for p in poses), encoding='utf-8')
    print(f"[priors] Extracted {len(poses)} VIO camera poses -> {out_path}", flush=True)
    return load_spectacular_jsonl(out_path)


def load_any(path: Path) -> tuple[list[tuple], str]:
    """Auto-detect one of: canonical jsonl / spectacular jsonl / record3d metadata / arcore csv."""
    suffix = path.suffix.lower()
    name = path.name.lower()
    if name.startswith("metadata") or suffix == ".r3d":
        return load_record3d_metadata(path), "record3d"
    text = path.read_text(encoding="utf-8-sig").lstrip()
    if not text:
        raise ValueError(f"{path} is empty")
    if text[0] in "[{":
        first_line = text.splitlines()[0]
        d = json.loads(first_line)
        if "poses" in d and isinstance(d["poses"], list) and len(d) <= 8:
            return load_record3d_metadata(path), "record3d"
        if "time" in d and "position" in d:
            return load_spectacular_jsonl(path), "spectacular"
        if "sdkVersion" in d or "frames" in d or "sensor" in d:
            # Raw Spectacular AI recording -> check for smoothed poses or auto-generate
            parent = path.parent
            smoothed = parent / f"{path.stem}_smoothed.jsonl"
            if not smoothed.exists():
                alt = parent / "poses.jsonl"
                if alt.exists():
                    smoothed = alt
            if smoothed.exists():
                return load_spectacular_jsonl(smoothed), "spectacular"
            # Auto-extract trajectory in-process
            return extract_spectacular_poses(parent, smoothed), "spectacular"
        return load_canonical_jsonl(path), "jsonl"
    return load_csv_log(path), "csv"



# -------------------------------------------------------------- interpolation
def interpolate_pose(samples: list[tuple], t: float):
    """Position lerp + quaternion slerp at time t; None outside coverage."""
    if not samples:
        return None
    if t <= samples[0][0]:
        return (samples[0][1].copy(), samples[0][2].copy()) if abs(t - samples[0][0]) < 0.5 else None
    if t >= samples[-1][0]:
        return (samples[-1][1].copy(), samples[-1][2].copy()) if abs(t - samples[-1][0]) < 0.5 else None
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, p0, q0 = samples[lo]
    t1, p1, q1 = samples[hi]
    u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    p = np.asarray(p0, np.float64) + (np.asarray(p1, np.float64) - np.asarray(p0, np.float64)) * u
    q = quat_slerp(q0, q1, u)
    return p, q


def match_to_frames(samples: list[tuple], frame_t: list[float], max_gap: float = 0.25):
    """One prior dict (or None) per frame timestamp."""
    out = []
    for tf in frame_t:
        ip = interpolate_pose(samples, tf)
        if ip is None:
            out.append(None)
            continue
        p, q = ip
        nearest_dt = min(abs(s[0] - tf) for s in (samples[0], samples[-1])) if len(samples) < 2 else \
            min(abs(samples[min(len(samples) - 1, max(0, _bisect(samples, tf)))][0] - tf),
                abs(samples[max(0, _bisect(samples, tf) - 1)][0] - tf))
        out.append({"position": [float(x) for x in p],
                    "quat_xyzw_c2w": [float(x) for x in q],
                    "log_dt": round(float(nearest_dt), 4)})
    return out


def _bisect(samples: list[tuple], t: float) -> int:
    lo, hi = 0, len(samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    # tiny smoke: identity roundtrip
    R = quat_to_R([0, 0, 0, 1])
    assert np.allclose(R, np.eye(3)), R
    assert np.allclose(R_to_quat(np.eye(3)), [0, 0, 0, 1], atol=1e-9)
    print("poses_lib OK", file=sys.stderr)
