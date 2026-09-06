"""Comprehensive 3D Gaussian Splatting Hardware Benchmark & Throughput Profiler.

Profiles raw CUDA rasterization, backward autograd pass, Adam optimizer updates,
and spherical harmonics math across the full lifecycle of a 3D reconstruction.

Usage:
  python scripts/benchmark_hardware.py
"""

import math
import sys
import time
import torch
import torch.nn.functional as F

def run_benchmark():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not torch.cuda.is_available():
        print("[error] No CUDA-capable GPU found!")
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_vram_gb = props.total_memory / (1024**3)

    # Enable Ampere TensorFloat-32 & cuDNN optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    try:
        from gsplat import rasterization as rendering
    except ImportError:
        print("[error] gsplat is not installed in current environment!")
        sys.exit(1)

    print("=" * 74)
    print(" 🚀 Drone3D Hardware Speed Test & Profiler")
    print(f"    - GPU:           {props.name}")
    print(f"    - VRAM:          {total_vram_gb:.2f} GiB")
    print(f"    - Compute Arch:  sm_{props.major}{props.minor} (Ampere Tensor Cores: {'YES (TF32)' if props.major >= 8 else 'NO'})")
    print(f"    - PyTorch:       {torch.__version__} (CUDA {torch.version.cuda})")
    print("=" * 74)

    # Benchmark configurations: (Label, N_gaussians, SH_deg, Width, Height, Warmup, Steps)
    configs = [
        ("Early Stage (100k, SH0, 640p)",  100_000, 0,  640,  360, 15, 60),
        ("Early Stage (100k, SH0, 1280p)", 100_000, 0, 1280,  720, 15, 60),
        ("Mid Stage   (350k, SH1, 1280p)", 350_000, 1, 1280,  720, 15, 60),
        ("Late Stage  (650k, SH3, 1280p)", 650_000, 3, 1280,  720, 15, 60),
    ]
    if total_vram_gb >= 7.5:
        configs.append(("Ultra Res   (800k, SH3, 1440p)", 800_000, 3, 1440,  810, 10, 40))

    print("\n" + "-" * 74)
    print(f"{'Stage / Workload':<36} | {'Speed (it/s)':<12} | {'Step Time':<10} | {'Peak VRAM'}")
    print("-" * 74, flush=True)

    results = []

    for label, n_pts, sh_deg, w, h, warmup, n_steps in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Initialize realistic spatial Gaussians
        means = (torch.randn(n_pts, 3, device=device) * 2.5).requires_grad_(True)
        quats = F.normalize(torch.randn(n_pts, 4, device=device), dim=-1).requires_grad_(True)
        scales = (torch.randn(n_pts, 3, device=device) * 0.2 - 5.2).requires_grad_(True)
        opacities = (torch.randn(n_pts, device=device) * 0.5 - 0.5).requires_grad_(True)

        n_sh_coeffs = (sh_deg + 1) ** 2
        sh0 = (torch.randn(n_pts, 1, 3, device=device) * 0.5).requires_grad_(True)
        shN = (torch.randn(n_pts, n_sh_coeffs - 1, 3, device=device) * 0.05).requires_grad_(True) if n_sh_coeffs > 1 else torch.empty(n_pts, 0, 3, device=device, requires_grad=True)

        params = [means, quats, scales, opacities, sh0]
        if n_sh_coeffs > 1:
            params.append(shN)
        opt = torch.optim.Adam(params, lr=1e-3, eps=1e-15)

        # Camera viewmat & K looking at scene
        viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        viewmat[0, 2, 3] = 6.0
        K = torch.tensor([[800.0, 0.0, w / 2.0], [0.0, 800.0, h / 2.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32).unsqueeze(0)
        target = torch.rand(1, 3, h, w, device=device, dtype=torch.float32)

        # Warmup iterations
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            quats_n = F.normalize(quats, dim=-1)
            cols = torch.cat([sh0, shN], dim=1) if n_sh_coeffs > 1 else sh0
            out = rendering(
                means, quats_n, torch.exp(scales), torch.sigmoid(opacities), cols,
                viewmat, K, w, h, near_plane=0.01, far_plane=50.0,
                render_mode="RGB", sh_degree=sh_deg, packed=True, absgrad=True,
                rasterize_mode="antialiased"
            )
            render_img = out[0] if isinstance(out, (tuple, list)) else out
            pred_chw = render_img.permute(0, 3, 1, 2) if render_img.ndim == 4 else render_img.permute(2, 0, 1).unsqueeze(0)
            loss = F.l1_loss(pred_chw, target)
            loss.backward()
            opt.step()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # Timed benchmark loop
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            quats_n = F.normalize(quats, dim=-1)
            cols = torch.cat([sh0, shN], dim=1) if n_sh_coeffs > 1 else sh0
            out = rendering(
                means, quats_n, torch.exp(scales), torch.sigmoid(opacities), cols,
                viewmat, K, w, h, near_plane=0.01, far_plane=50.0,
                render_mode="RGB", sh_degree=sh_deg, packed=True, absgrad=True,
                rasterize_mode="antialiased"
            )
            render_img = out[0] if isinstance(out, (tuple, list)) else out
            pred_chw = render_img.permute(0, 3, 1, 2) if render_img.ndim == 4 else render_img.permute(2, 0, 1).unsqueeze(0)
            loss = F.l1_loss(pred_chw, target)
            loss.backward()
            opt.step()

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        it_per_sec = n_steps / dt
        ms_per_it = (dt / n_steps) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        row = {
            "label": label,
            "it_s": it_per_sec,
            "ms": ms_per_it,
            "vram": peak_vram_mb
        }
        results.append(row)
        print(f"{row['label']:<36} | {row['it_s']:>8.1f} it/s | {row['ms']:>6.1f} ms | {row['vram']:>6.0f} MiB", flush=True)

    print("-" * 74)

    # Realistic 15,000 Step Weighted Projection:
    speed_early = results[1]["it_s"]
    speed_mid = results[2]["it_s"]
    speed_late = results[3]["it_s"]

    weighted_sec = (2000 / speed_early) + (4000 / speed_mid) + (9000 / speed_late)
    weighted_avg_it_s = 15000 / weighted_sec
    proj_min = weighted_sec / 60.0

    print(f"\n📊 Projected Full 15,000 Step 3DGS Reconstruction (High Quality 1280p):")
    print(f"   - Weighted Average Speed:  {weighted_avg_it_s:.1f} it/s")
    print(f"   - Total Training Duration: {proj_min:.1f} minutes ({weighted_sec:.0f} seconds)")
    print(f"   - 6GB VRAM Safety Margin:  +{((total_vram_gb * 1024) - results[3]['vram']):.0f} MiB free headroom")
    print("=" * 74 + "\n", flush=True)

if __name__ == "__main__":
    run_benchmark()
