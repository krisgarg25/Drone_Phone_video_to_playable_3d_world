"""Turbocharged 3D Gaussian Splatting trainer for high quality & fast convergence.

Optimizations:
  1. Pre-cached GPU tensors: all viewmats, Ks, and images reside directly in CUDA VRAM.
  2. Native PyTorch Adam optimizer with seamless parameter resizing via DefaultStrategy.
  3. Fast channel-grouped vectorized SSIM in PyTorch.
  4. Random background regularization: destroys smoky floaters in unobserved air/ceiling.
  5. AbsGrad strategy (absgrad=True) + opacity reset intervals for razor-sharp geometric detail.
  6. Numerical stability clamps on scales, quats, and opacities.

Usage:
  python train_splat.py --work work/room_w_jsonl --steps 12000 --cap 650000
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_colmap import qvec2rot  # noqa: E402
import robust as rb  # noqa: E402

SH_DEG = 3


# ---------------- COLMAP TXT parsing ----------------
def load_colmap(txt_dir: Path):
    cams = {}
    for line in (txt_dir / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        # p[2], p[3] are the width/height COLMAP solved at — the only way to know
        # whether the images on disk still match these intrinsics.
        cams[int(p[0])] = dict(model=p[1], params=list(map(float, p[4:])),
                               w0=int(p[2]), h0=int(p[3]))

    imgs = []
    for line in (txt_dir / "images.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 10:
            continue
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        imgs.append(dict(name=p[9], R=qvec2rot(q), t=t, cam_id=int(p[8])))
    return cams, imgs


def load_points3d(txt_dir: Path):
    xyz, rgb = [], []
    for line in (txt_dir / "points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.array(xyz), np.array(rgb, dtype=np.float32)


# ---------------- data preparation ----------------
def prepare_dataset(work: Path, max_images: int | None = None):
    txt = work / "colmap" / "sparse" / "txt"
    cams, imgs = load_colmap(txt)

    und_dir = work / "frames_undist"
    und_dir.mkdir(exist_ok=True)

    data = []
    src_sizes = set()
    for im in imgs:
        c = cams[im["cam_id"]]
        if c["model"] in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
            f = c["params"][0]
            cx, cy = c["params"][1], c["params"][2]
            if c["model"] == "SIMPLE_RADIAL":
                dist = np.array([c["params"][3], 0, 0, 0], np.float64)
            elif c["model"] == "RADIAL":
                dist = np.array([c["params"][3], c["params"][4], 0, 0], np.float64)
            else:
                dist = np.array(c["params"][4:8], np.float64)
        else:
            f = c["params"][0]
            cx, cy = c["params"][2], c["params"][3]
            dist = np.zeros(4)
        K = np.array([[f, 0, cx], [0, f if c["model"] != "PINHOLE" else c["params"][1], cy], [0, 0, 1]], dtype=np.float32)

        # The undistortion cache is keyed on the source pixel size: a cache made
        # from 1280 px frames must not answer a 1920 px request, or the run trains
        # at the old size while asking for the new one.
        src_path = work / "frames_train" / im["name"]
        if np.any(dist[:2] != 0):
            src = cv2.imread(str(src_path))
            hs, ws = src.shape[:2]
            cache = und_dir / f"{ws}x{hs}__{im['name'].replace('/', '__')}"
            if cache.exists():
                img_path = cache
                img = cv2.cvtColor(cv2.imread(str(cache)), cv2.COLOR_BGR2RGB)
            else:
                img = cv2.undistort(src, K, dist)
                cache.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(cache), img)
                img_path = cache
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_path = src_path
            img = cv2.cvtColor(cv2.imread(str(src_path)), cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        # cameras.txt solved at (w0, h0); a resize scales fx, fy, cx and cy with
        # it, so a model solved at 1600 px is exactly as good at 1920 px once the
        # intrinsics come along. Without this the principal point stays at the old
        # centre and every gaussian lands off by the ratio.
        sx, sy = w / c["w0"], h / c["h0"]
        if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
            K = np.array([[K[0, 0] * sx, 0, K[0, 2] * sx],
                          [0, K[1, 1] * sy, K[1, 2] * sy],
                          [0, 0, 1]], dtype=np.float32)
            src_sizes.add((c["w0"], c["h0"], w, h))
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = im["R"]
        viewmat[:3, 3] = im["t"]
        sharpness = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

        data.append(dict(name=im["name"], K=K, viewmat=viewmat, path=img_path, width=w, height=h,
                         sharpness=sharpness, img=img))
        if max_images and len(data) >= max_images:
            break
    print(f"[train] dataset: {len(data)} images {data[0]['width']}x{data[0]['height']}")
    for w0, h0, w, h in sorted(src_sizes):
        print(f"[train] COLMAP solved at {w0}x{h0}, training images are {w}x{h}: "
              f"intrinsics scaled x{w / w0:.3f} to match")
    sharp_arr = sorted(d["sharpness"] for d in data)
    print(f"[train] frame sharpness (Laplacian var): min {sharp_arr[0]:.1f} "
          f"median {sharp_arr[len(sharp_arr)//2]:.1f} max {sharp_arr[-1]:.1f}")
    fit_to_vram(data)
    return data, load_points3d(txt)


def fit_to_vram(data: list) -> int:
    """Cap per-frame pixels so the rasterizer can actually render this set.

    A "width" limit cannot see a portrait frame coming: 1280x2772 is 3.5 MP,
    nearly 4x a 640x763 budget, and the card was filled mid-run with the process
    killed before it could print anything. The keyframe step sizes from video
    metadata; this is the last place that knows the real decoded size.

    Returns the chosen per-frame pixel count. K scales with the resize because
    the projected gaussian positions are only right if the intrinsics follow the
    pixel grid.
    """
    from robust import available_vram_gb, pixels_for, train_budget
    w0, h0 = data[0]["width"], data[0]["height"]
    want = min(w0 * h0, 1_100_000)
    budget = train_budget(vram_gb=available_vram_gb(), max_pixels=want,
                          n_frames=len(data))
    nw, nh = pixels_for(w0, h0, budget["pixels"])
    if (nw, nh) == (w0, h0):
        return w0 * h0
    sx, sy = nw / w0, nh / h0
    for d in data:
        d["K"] = np.array([[d["K"][0, 0] * sx, 0, d["K"][0, 2] * sx],
                           [0, d["K"][1, 1] * sy, d["K"][1, 2] * sy],
                           [0, 0, 1]], dtype=np.float32)
        if d.get("img") is not None:
            d["img"] = cv2.resize(d["img"], (nw, nh), interpolation=cv2.INTER_AREA)
        d["width"], d["height"] = nw, nh
    print(f"[train] {w0}x{h0} = {w0 * h0 / 1e6:.2f} MP/frame x {len(data)} frames does "
          f"not fit {budget['vram_gb']:.1f} GiB free: re-sized to {nw}x{nh} "
          f"({nw * nh / 1e6:.2f} MP), intrinsics scaled with it")
    return nw * nh


# ---------------- fast vectorized ssim ----------------
_W_CACHE = {}


def fast_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch SSIM across RGB channels using grouped 2D convolution."""
    ch = img1.shape[1]
    dev = img1.device
    if ch not in _W_CACHE or _W_CACHE[ch].device != dev:
        coords = torch.arange(11, device=dev, dtype=torch.float32) - 5.0
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        _W_CACHE[ch] = (g[:, None] @ g[None, :]).expand(ch, 1, 11, 11) / (g.sum() ** 2)
    w = _W_CACHE[ch]

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(img1, w, groups=ch, padding=5)
    mu2 = F.conv2d(img2, w, groups=ch, padding=5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, groups=ch, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, groups=ch, padding=5) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, groups=ch, padding=5) - mu12
    ssim_val = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_val.mean()


# ---------------- initialization ----------------
def init_from_points(xyz: np.ndarray, rgb: np.ndarray, data: list, device: str, n_init: int = 150_000, room_anchors: bool = True):
    pts = torch.tensor(xyz, dtype=torch.float32)
    if len(pts) > n_init:
        sel = torch.randperm(len(pts))[:n_init]
        pts = pts[sel]
        rgb = rgb[sel.cpu().numpy()]
    
    # Inject spatial anchor seeds for indoor room walls and corners
    if room_anchors and data and len(data) > 5:
        cam_centers = []
        cam_fwds = []
        for d in data:
            R = d["viewmat"][:3, :3]
            t = d["viewmat"][:3, 3]
            C = -R.T @ t
            fw = R[2, :]  # +Z forward
            cam_centers.append(C)
            cam_fwds.append(fw)
        cam_centers = np.array(cam_centers, dtype=np.float32)
        cam_fwds = np.array(cam_fwds, dtype=np.float32)
        
        # Sample forward rays at 1.5m, 2.5m, 3.5m to seed wall planes
        wall_pts = []
        wall_rgbs = []
        mean_c = rgb.mean(axis=0) if len(rgb) else np.array([160.0, 160.0, 160.0])
        for dist in (1.5, 2.5, 3.5):
            w = cam_centers + cam_fwds * dist
            # Add small random jitter
            w += np.random.randn(*w.shape).astype(np.float32) * 0.15
            wall_pts.append(w)
            wall_rgbs.append(np.tile(mean_c, (len(w), 1)))
        
        if wall_pts:
            wall_pts = np.vstack(wall_pts)
            wall_rgbs = np.vstack(wall_rgbs)
            pts = torch.cat([pts, torch.tensor(wall_pts, dtype=torch.float32)], dim=0)
            rgb = np.vstack([rgb, wall_rgbs])
            print(f"[train] seeded {len(wall_pts)} spatial anchor points on room walls")

    pts = pts.to(device)
    rgb_t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)

    sub = pts[torch.randperm(len(pts), device=device)[:min(20000, len(pts))]]
    d = torch.cdist(sub, sub)
    d.fill_diagonal_(float("inf"))
    nn3 = d.topk(min(3, len(sub) - 1), largest=False).values.mean(1)
    med = float(nn3.median().clamp(min=1e-6))
    scales0 = torch.full((len(pts),), math.log(med), dtype=torch.float32, device=device)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "quats": torch.nn.Parameter(torch.cat([torch.ones(len(pts), 1, device=device),
                                               torch.zeros(len(pts), 3, device=device)], dim=1)),
        "scales": torch.nn.Parameter(scales0[:, None].repeat(1, 3)),
        "opacities": torch.nn.Parameter(torch.full((len(pts),), 0.1, device=device)),
        "sh0": torch.nn.Parameter(_rgb_to_sh0(rgb_t)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(len(pts), (SH_DEG + 1) ** 2 - 1, 3, device=device)),
    })
    extent = float((pts.max(0).values - pts.min(0).values).max())
    return params, extent


def _rgb_to_sh0(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def export_ply(means, quats, scales, opacities, sh0, shN, path: Path):
    N = len(means)
    f_dc = sh0[:, 0, :]
    n_rest = (SH_DEG + 1) ** 2 - 1
    f_rest = torch.zeros(N, n_rest * 3, dtype=torch.float32)
    if shN is not None and shN.shape[1] > 0:
        k = min(shN.shape[1], n_rest)
        for i in range(k):
            f_rest[:, i * 3:(i + 1) * 3] = shN[:, i, :]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")] + \
            [(f"f_rest_{i}", "f4") for i in range(n_rest * 3)] + \
            [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    arr = np.zeros(N, dtype=dtype)
    m = means.detach().cpu().numpy()
    arr["x"], arr["y"], arr["z"] = m[:, 0], m[:, 1], m[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc.detach().cpu().numpy().T
    fr = f_rest.detach().numpy()
    for i in range(n_rest * 3):
        arr[f"f_rest_{i}"] = fr[:, i]
    arr["opacity"] = opacities.detach().cpu().numpy()
    sc = scales.detach().cpu().numpy()
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = sc[:, 0], sc[:, 1], sc[:, 2]
    qt = quats.detach().cpu().numpy()
    qt = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"[train] wrote {path} ({N} gaussians)")


# ---------------- main trainer ----------------
def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--cap", type=int, default=650_000, help="hard max gaussian count")
    ap.add_argument("--init-pts", type=int, default=150_000)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--grow-grad", dest="grow_grad", type=float, default=0.0006,
                    help="2D positional gradient a gaussian must exceed to be cloned or "
                         "split. This, not --cap, is what decides the splat count: the "
                         "auditorium run plateaued at 247k with an 850k cap and 18k steps "
                         "left. Lower it for detail, at the cost of VRAM and sort time.")
    ap.add_argument("--refine-stop", type=int, default=9000)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--antialias", action=argparse.BooleanOptionalAction, default=True,
                    help="gsplat antialiased rasterization (big sharpness win when "
                         "training resolution differs from capture resolution)")
    ap.add_argument("--opa-reg", type=float, default=0.002,
                    help="L1 opacity regularization weight (kills semi-transparent floaters)")
    ap.add_argument("--random-bkgd", action="store_true", default=True,
                    help="use random background color to kill floaters in unobserved air/ceiling")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    # Hardware acceleration flags for Ampere (RTX 30-series) Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    from gsplat import rasterization as rendering, DefaultStrategy
    from gsplat.strategy.ops import remove as _remove_gs

    data, (p_xyz, p_rgb) = prepare_dataset(args.work)
    if len(data) < 3:
        sys.exit(f"[train] only {len(data)} frames are registered in this model - "
                 f"training needs at least 3. See {args.work / 'logs'} for the "
                 f"colmap registration count.")

    W, H = data[0]["width"], data[0]["height"]

    # Pre-cache camera viewmats and Ks in GPU VRAM
    gpu_viewmats = torch.stack([torch.tensor(d["viewmat"], dtype=torch.float32, device=device) for d in data])
    gpu_Ks = torch.stack([torch.tensor(d["K"], dtype=torch.float32, device=device) for d in data])

    # One budget for the whole card, from measured free VRAM: pixels, image
    # cache and gaussian cloud are three claims on the same 6 GiB, and the two
    # heuristics this replaces each looked at only one of them. The cap clamp in
    # particular could not express what killed a run -- 465 frames at 1280x2772
    # is 3.5 MP of render intermediates per step, which no change to the
    # gaussian count touches, so the card filled and the process died with a
    # bare exit -1 and nothing printed.
    from robust import train_budget
    _free_b, _total_b = torch.cuda.mem_get_info()
    vram_gb = _free_b / 1024**3
    budget = train_budget(vram_gb=vram_gb, max_pixels=W * H, n_frames=len(data))
    img_bytes = sum(d["img"].nbytes for d in data if d.get("img") is not None)
    # A frame count alone says nothing about VRAM pressure once the images are
    # not resident, so streaming is decided against the budget, not a constant.
    stream_images = budget["stream_images"] or any(d.get("img") is None for d in data)
    if args.cap > budget["cap"]:
        print(f"[train] cap {args.cap} -> {budget['cap']}: {W}x{H} frames "
              f"x {len(data)} on {vram_gb:.1f} GiB free of {_total_b / 1024**3:.1f} GiB")
        args.cap = budget["cap"]
    if stream_images:
        print(f"[train] image set {img_bytes / 1024**2:.0f} MB will cost the "
              f"gaussians their room on a {vram_gb:.1f} GiB budget "
              f"— streaming frames from disk instead")
        for d in data:
            d["img"] = None
        gpu_imgs_u8 = None
    else:
        print(f"[train] pre-caching {img_bytes / 1024**2:.1f} MB images directly in GPU VRAM")
        gpu_imgs_u8 = torch.stack([torch.from_numpy(np.array(d.pop("img"))).to(device).permute(2, 0, 1) for d in data])

    params, extent = init_from_points(p_xyz, p_rgb, data, device, args.init_pts, room_anchors=True)
    print(f"[train] init {len(params['means'])} gaussians, scene extent {extent:.2f} m")
    if not len(params["means"]):
        # An empty cloud is not a slow start: the rasterizer's CUDA kernel
        # fail-fasts on zero gaussians, and the process dies with a bare
        # 0xC0000409 that looks like the GPU or the video format failing.
        sys.exit("[train] the reconstruction has no 3D points to seed from, so there "
                 "is nothing to train. The colmap solve is the problem, not this step.")
    lrs = {
        "means": 1.6e-4 * extent, "quats": 1e-3, "scales": 5e-3,
        "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": params[name], "lr": lrs[name], "name": name}],
                               eps=1e-15, betas=(0.9, 0.999))
        for name in params.keys()
    }

    strategy = DefaultStrategy(
        verbose=False,
        absgrad=True,
        grow_grad2d=args.grow_grad,
        prune_opa=0.02,
        prune_scale3d=0.10 * extent,
        reset_every=3000,
        refine_start_iter=500,
        refine_stop_iter=args.refine_stop,
        refine_every=150
    )
    strategy_state = strategy.initialize_state()

    if len(params["means"]) > args.cap:
        # Seeding can overshoot a cap that was only just settled, and the first
        # refine would then push straight into an out-of-memory. Trim to the most
        # opaque seeds rather than starting over-budget.
        print(f"[train] seeds {len(params['means'])} > cap {args.cap}: trimming "
              f"to the most opaque")
        opa = torch.sigmoid(params["opacities"]).reshape(-1)
        drop = torch.ones_like(opa, dtype=torch.bool)
        drop[opa.topk(args.cap).indices] = False
        _remove_gs(params, optimizers, strategy_state, drop)

    prog = args.work / "train_progress"
    prog.mkdir(exist_ok=True)
    log = open(args.work / "train_log.txt", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    # Checkpoint the seeds immediately. A card that fills, or a driver that
    # takes the process with a bare 0xC0000409, is only survivable if something
    # exists to rescue; with --save-every 3000 a death in the first minutes left
    # the runner nothing but a failure to report.
    export_ply(params["means"], F.normalize(params["quats"], dim=1),
               params["scales"], params["opacities"],
               params["sh0"], params["shN"], prog / "splat.partial.ply")

    say(f"[train] start steps={args.steps} cap={args.cap} {W}x{H} "
        f"ssim_w={args.ssim_weight} absgrad=True grow_grad2d={args.grow_grad} "
        f"refine_stop={args.refine_stop} "
        f"antialias={args.antialias} opa_reg={args.opa_reg}")

    # Blur-aware sampling: sharper frames train more often (directly attacks the
    # "splat looks soft" problem caused by motion-blurred frames polluting SH colors)
    sharp_w = np.clip(np.array([d["sharpness"] for d in data], dtype=np.float64), 15.0, None) ** 0.5
    sample_p = sharp_w / sharp_w.sum()
    say(f"[train] blur-aware sampling: effective weight range "
        f"{sample_p.min() * len(data):.2f}x - {sample_p.max() * len(data):.2f}x per frame")

    def means_lr(step):
        return max(1.6e-6 * extent, 1.6e-4 * extent * math.exp(-step * 0.00023))

    n_images = len(data)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        optimizers["means"].param_groups[0]["lr"] = means_lr(step)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            params["scales"].clamp_(max=math.log(10.0 * extent), min=-15.0)

        img_idx = int(np.random.choice(n_images, p=sample_p))
        viewmat = gpu_viewmats[img_idx:img_idx + 1]
        K = gpu_Ks[img_idx:img_idx + 1]
        if not stream_images:
            img_gt = gpu_imgs_u8[img_idx:img_idx + 1].to(dtype=torch.float32) * (1.0 / 255.0)
        else:
            # The renderer is asked for W x H, so the target must be W x H. The
            # frame on disk is not necessarily that size: fit_to_vram may have
            # re-sized the set, and a multi-clip scene can carry mixed native
            # resolutions in the first place.
            frame_bgr = cv2.imread(str(data[img_idx]["path"]))
            if frame_bgr is None:
                sys.exit(f"[train] cannot read {data[img_idx]['path']} - the keyframe "
                         f"step listed a file that is not decodable.")
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (W, H):
                frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_gt = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) * (1.0 / 255.0)

        if args.random_bkgd and np.random.rand() < 0.5:
            bkgd = torch.rand(3, device=device)
        else:
            bkgd = None

        quats_n = F.normalize(params["quats"], dim=-1)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)

        sh_deg_cur = min(step // 1000, SH_DEG)
        out = rendering(
            params["means"], quats_n, torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, viewmat, K,
            W, H, near_plane=0.01, far_plane=10 * extent,
            render_mode="RGB", sh_degree=sh_deg_cur,
            packed=True, absgrad=True, backgrounds=bkgd,
            rasterize_mode="antialiased" if args.antialias else "classic",
        )
        rgb_pred, alpha, info = out
        rgb_pred = rgb_pred.permute(0, 3, 1, 2)  # [1, 3, H, W]
        target_comp = img_gt

        l1 = F.l1_loss(rgb_pred, target_comp)
        ssim_loss = 1.0 - fast_ssim(rgb_pred.clamp(0, 1), target_comp) if args.ssim_weight > 0 else 0.0
        loss = (1 - args.ssim_weight) * l1 + args.ssim_weight * ssim_loss
        if args.opa_reg > 0:
            # weak global pull on opacity: prevents soft semi-transparent floater
            # crusts while letting real surfaces stay fully opaque
            loss = loss + args.opa_reg * torch.sigmoid(params["opacities"]).mean()

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
        if not len(params["means"]):
            # The densifier prunes on opacity; when it takes the last gaussian
            # the next backward pass fail-fasts inside CUDA with a bare
            # 0xC0000409 and no traceback. Say which step emptied it instead.
            say(f"[train] COLLAPSED at step {step}: every gaussian was pruned "
                f"(loss {loss.item():.4f}). The seeds cannot support a surface - "
                f"this is a solve-quality problem, not a GPU one.")
            sys.exit(1)
        # gsplat 1.5.3's DefaultStrategy has no max_gaussians, so --cap used to be
        # a number that got printed and never enforced: this cloud grew to 1.05 M
        # against a stated 850 k and kept climbing until the card filled.
        n = len(params["means"])
        if n > args.cap:
            opa = torch.sigmoid(params["opacities"]).reshape(-1)
            drop = torch.ones_like(opa, dtype=torch.bool)
            drop[opa.topk(args.cap).indices] = False
            _remove_gs(params, optimizers, strategy_state, drop)
            say(f"[train] --cap {args.cap} reached at step {step}: "
                f"{n} -> {len(params['means'])} on lowest opacity")

        for opt in optimizers.values():
            opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                mse = F.mse_loss(rgb_pred.clamp(0, 1), img_gt)
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
            it_s = step / max(time.time() - t0, 1e-4)
            say(f"[train] step {step:6d} loss {loss.item():.4f} psnr {psnr.item():.2f} "
                f"N {len(params['means'])} {time.time()-t0:.0f}s ({it_s:.1f} it/s) "
                f"mem {torch.cuda.max_memory_allocated()/2**30:.2f}GiB")

        if step % args.save_every == 0 or step == args.steps:
            # Not splat.ply: a run that dies at step 12k would otherwise leave a
            # half-trained cloud where the last complete one was, with every
            # exported asset downstream still built from that one.
            export_ply(params["means"], F.normalize(params["quats"], dim=1),
                       params["scales"], params["opacities"],
                       params["sh0"], params["shN"], prog / "splat.partial.ply")
            with torch.no_grad():
                rgb_eval, _, _ = rendering(
                    params["means"], F.normalize(params["quats"], dim=1),
                    torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                    torch.cat([params["sh0"], params["shN"]], dim=1),
                    viewmat, K, W, H, render_mode="RGB",
                    sh_degree=SH_DEG, packed=True,
                    rasterize_mode="antialiased" if args.antialias else "classic")
                rgb_eval = rgb_eval.permute(0, 3, 1, 2)
                prev = (rgb_eval[0].clamp(0, 1) * 255).byte()
                prev = prev.permute(1, 2, 0).cpu().numpy()
                # Instrumentation, not output: the checkpoint above is what a
                # rescue reads, so a refused preview must not end the solve.
                if not rb.save_image(Image.fromarray(prev),
                                     prog / f"step_{step:06d}.jpg", quality=92):
                    say(f"[train] step {step}: preview jpg refused, continuing")

    export_ply(params["means"], F.normalize(params["quats"], dim=1),
               params["scales"], params["opacities"],
               params["sh0"], params["shN"], args.work / "splat.ply")
    say(f"[train] DONE in {time.time()-t0:.0f}s ({args.steps / (time.time()-t0):.1f} it/s), final N={len(params['means'])}")


if __name__ == "__main__":
    main()
