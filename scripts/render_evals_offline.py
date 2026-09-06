"""Render all eval poses offline with gsplat (the proven training path):
scene.ply + poses.json -> eval_renders/eval_XX.png (at each camera's own size).

Evidence, not geometry. A render that cannot be made costs one blind A/B pair,
not the scene - so an unusable input names the step that owes the file, and one
bad camera skips its own render instead of ending the loop for the rest.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import robust as rb  # noqa: E402

NEED = ["x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
        "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2"]
N_REST = 15


def load_splat(ply: Path):
    """Gaussian tensors for gsplat, on whatever device this machine has."""
    rb.require_file(ply, "viewer_assets/scene.ply (written by the export step)")
    try:
        v = PlyData.read(str(ply))["vertex"].data
    except Exception as e:
        # plyfile's own parse errors subclass neither OSError nor ValueError.
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"{ply.name} could not be parsed: {type(e).__name__}: {e}",
                           returncode=3)
    missing = [p for p in NEED if p not in v.dtype.names]
    if missing:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{ply.name} has no {missing} - not a splat the export step wrote",
            returncode=3)
    if len(v) == 0:
        raise rb.StepError(rb.EMPTY_INPUT,
                           f"{ply.name} holds 0 gaussians - nothing to render",
                           returncode=3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        rb.warn("no CUDA device visible - eval renders run on the CPU and will be "
                "slow. The images are still valid. Check `pipeline.py doctor` for "
                "the torch build in .venv310.")

    keep = np.isfinite(np.stack([v["x"], v["y"], v["z"]], 1)).all(1)
    if not keep.any():
        raise rb.StepError(
            rb.EMPTY_INPUT, f"every gaussian in {ply.name} has a non-finite centre",
            returncode=3)
    if (~keep).sum():
        rb.warn(f"{int((~keep).sum())}/{len(v)} gaussians have a non-finite centre "
                "and are dropped from the eval renders")
        v = v[keep]

    def t(name):
        return torch.tensor(np.ascontiguousarray(v[name], dtype=np.float32),
                            device=dev)

    means = torch.stack([t("x"), t("y"), t("z")], 1)
    quats = torch.nn.functional.normalize(
        torch.stack([t(f"rot_{k}") for k in range(4)], 1), dim=1)
    # exp() of a trained log-scale overflows past ~88 and yields inf, which the
    # rasteriser turns into a black frame rather than an error.
    scales = torch.stack([t(f"scale_{k}") for k in range(3)], 1).clamp(max=10.0).exp()
    opac = torch.sigmoid(t("opacity"))
    sh0 = torch.stack([t("f_dc_0"), t("f_dc_1"), t("f_dc_2")], 1)
    rest_names = [f"f_rest_{k}" for k in range(N_REST * 3)]
    if all(name in v.dtype.names for name in rest_names):
        rest = np.stack([v[n] for n in rest_names], 1).reshape(len(v), N_REST, 3)
        shN = torch.tensor(rest.astype(np.float32), device=dev)
    else:
        shN = torch.zeros(len(v), N_REST, 3, dtype=torch.float32, device=dev)
    colors = torch.cat([sh0[:, None, :], shN], dim=1)[None]  # [1,N,16,3]
    return means, quats, scales, opac, colors, dev


def camera_of(p: dict):
    """(viewmat, K, W, H) numpy pieces from one poses.json row, or None."""
    try:
        W, H = int(p["width"]), int(p["height"])
        R = np.array(p["R_rowmajor"], np.float32)
        tv = np.array(p["t"], np.float32)
        fx, fy = float(p["fx"]) * W / 640.0, float(p["fy"]) * H / 360.0
        cx, cy = float(p["cx"]) * W / 640.0, float(p["cy"]) * H / 360.0
    except (KeyError, TypeError, ValueError) as e:
        rb.warn(f"pose {p.get('name', '?')} is unusable ({type(e).__name__}: {e}) "
                "- render skipped")
        return None
    if (W < 2 or H < 2
            or not rb.finite(fx, fy, cx, cy, *R.reshape(-1).tolist(),
                             *tv.tolist())):
        rb.warn(f"pose {p.get('name', '?')} has a degenerate camera "
                f"({W}x{H}, fx={fx:.1f}) - render skipped")
        return None
    return R, tv, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32), W, H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=ROOT / "work" / "rocks")
    ap.add_argument("--max-cams", type=int, default=None,
                    help="render at most this many poses (default: all)")
    args = ap.parse_args()

    asset = args.work / "viewer_assets"
    out = args.work / "eval_renders"
    out.mkdir(parents=True, exist_ok=True)

    means, quats, scales, opac, colors, dev = load_splat(asset / "scene.ply")
    poses = rb.read_json(asset / "poses.json", None)
    if not isinstance(poses, list) or not poses:
        raise rb.StepError(
            rb.EMPTY_INPUT,
            f"{asset / 'poses.json'} is missing, unreadable or empty - the export "
            "step writes the eval cameras there.", returncode=3)
    if args.max_cams:
        poses = poses[:args.max_cams]

    from gsplat import rasterization

    made, failed = 0, 0
    for i, p in enumerate(poses):
        cam = camera_of(p)
        if cam is None:
            failed += 1
            continue
        R, tv, K, W, H = cam
        vm = torch.eye(4, dtype=torch.float32, device=dev)[None]
        vm[0, :3, :3] = torch.tensor(R, device=dev)
        vm[0, :3, 3] = torch.tensor(tv, device=dev)
        Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]
        try:
            with torch.no_grad():
                rgb, _, _ = rasterization(
                    means, quats, scales, opac, colors, vm, Kt, W, H,
                    near_plane=0.01, far_plane=1000.0,
                    render_mode="RGB", sh_degree=3, packed=True)
                img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
        except RuntimeError as e:
            # gsplat reports a device assert as RuntimeError; one out-of-range
            # camera must not cost every render after it.
            rb.warn(f"pose {p.get('name', i)} failed to render "
                    f"({type(e).__name__}: {str(e).splitlines()[0][:180]}) - skipped")
            failed += 1
            continue
        # Named by pose index, not by render count: export_viewer_assets'
        # eval_pairs.json refers to these files as eval_<i>.png.
        if not rb.save_image(Image.fromarray(img), out / f"eval_{i:02d}.png"):
            rb.warn(f"eval_{i:02d}.png could not be written - skipped")
            failed += 1
            continue
        print(f"[eval] {i:02d} <- {p.get('name', i)} mean={img.mean():.1f}",
              flush=True)
        made += 1

    if not made:
        rb.warn(f"none of the {len(poses)} eval cameras produced a file - the "
                "visual bar has nothing to score from this run")
        return
    print(f"[eval] done: {made} renders" + (f", {failed} skipped" if failed else ""))


if __name__ == "__main__":
    rb.configure_streams()
    try:
        main()
    except rb.StepError as e:
        print(f"\n[eval] {e}", file=sys.stderr, flush=True)
        sys.exit(e.returncode)
