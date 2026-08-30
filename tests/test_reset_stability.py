import math
import torch
import numpy as np
from pathlib import Path
import gsplat
from gsplat import rasterization, DefaultStrategy
from gsplat.optimizers import SelectiveAdam

device = "cuda"
N = 100000
extent = 10.0
W, H = 960, 720

means = torch.randn((N, 3), device=device, requires_grad=True)
quats = torch.randn((N, 4), device=device, requires_grad=True)
scales = torch.full((N, 3), -3.0, device=device, requires_grad=True)
opacities = torch.full((N,), -2.0, device=device, requires_grad=True)
sh0 = torch.randn((N, 1, 3), device=device, requires_grad=True)
shN = torch.randn((N, 15, 3), device=device, requires_grad=True)

params = torch.nn.ParameterDict({
    "means": means, "quats": quats, "scales": scales,
    "opacities": opacities, "sh0": sh0, "shN": shN
})

optimizers = {
    name: SelectiveAdam([{"params": params[name], "lr": 1e-3, "name": name}], eps=1e-15, betas=(0.9, 0.999))
    for name in params
}

strategy = DefaultStrategy(
    verbose=False,
    absgrad=True,
    grow_grad2d=0.0008,
    prune_opa=0.01,
    prune_scale3d=0.10 * extent,
    reset_every=500,
    refine_start_iter=100,
    refine_stop_iter=2500,
    refine_every=50
)
strategy_state = strategy.initialize_state()

viewmat = torch.eye(4, device=device)[None]
K = torch.tensor([[600.0, 0.0, W/2], [0.0, 600.0, H/2], [0.0, 0.0, 1.0]], device=device)[None]
gt = torch.rand((1, 3, H, W), device=device)

print("Testing 1000 steps through densification & opacity reset...")
for step in range(1, 1001):
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    with torch.no_grad():
        params["scales"].clamp_(max=math.log(10.0 * extent), min=-15.0)

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    quats_n = torch.nn.functional.normalize(params["quats"], dim=-1)
    render_colors, alphas, info = rasterization(
        params["means"], quats_n, torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, viewmat, K,
        W, H, packed=True, absgrad=True, sh_degree=min(step // 500, 3), render_mode="RGB"
    )
    rgb = render_colors.permute(0, 3, 1, 2)
    loss = (rgb - gt).abs().mean()

    strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
    loss.backward()
    strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

    vis_mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=device)
    if "gaussian_ids" in info and info["gaussian_ids"] is not None:
        vis_mask[info["gaussian_ids"]] = True
    else:
        vis_mask.fill_(True)

    for opt in optimizers.values():
        opt.step(vis_mask)

    if step % 200 == 0:
        print(f"Step {step:4d}: N={len(params['means'])} loss={loss.item():.4f}")

print("SUCCESS: 1000 steps completed without error!")
