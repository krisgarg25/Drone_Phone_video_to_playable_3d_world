import time
import torch
import numpy as np

# Test mock scene
N = 150000
W, H = 960, 720
device = "cuda"

from gsplat import rasterization, DefaultStrategy
from gsplat.optimizers import SelectiveAdam

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

strategy = DefaultStrategy(verbose=False, absgrad=True, reset_every=3000)
strategy_state = strategy.initialize_state()

viewmat = torch.eye(4, device=device)[None]
K = torch.tensor([[600.0, 0.0, W/2], [0.0, 600.0, H/2], [0.0, 0.0, 1.0]], device=device)[None]
gt = torch.rand((1, 3, H, W), device=device)

# Coarse step (0.5x resolution)
W_c, H_c = W // 2, H // 2
K_c = K.clone()
K_c[0, 0, 0] /= 2
K_c[0, 1, 1] /= 2
K_c[0, 0, 2] /= 2
K_c[0, 1, 2] /= 2
gt_c = torch.nn.functional.interpolate(gt, (H_c, W_c), mode="bilinear", align_corners=False)

torch.cuda.synchronize()
t0 = time.time()
n_coarse = 200
for step in range(1, n_coarse + 1):
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    quats_n = torch.nn.functional.normalize(params["quats"], dim=-1)
    render_colors, alphas, info = rasterization(
        params["means"], quats_n, torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, viewmat, K_c,
        W_c, H_c, packed=True, absgrad=True, sh_degree=min(step // 1000, 3), render_mode="RGB"
    )
    rgb = render_colors.permute(0, 3, 1, 2)
    loss = (rgb - gt_c).abs().mean()
    strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
    loss.backward()
    strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
    vis_mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=device)
    vis_mask[info["gaussian_ids"]] = True
    for opt in optimizers.values():
        opt.step(vis_mask)

torch.cuda.synchronize()
dt_c = time.time() - t0
print(f"COARSE: {n_coarse} steps in {dt_c:.3f}s -> {n_coarse / dt_c:.1f} it/s ({dt_c * 2000 / n_coarse:.1f}s for 2k coarse steps)")
