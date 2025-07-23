import math
from typing import List

import torch
from core.ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth     = cfg.min_depth
        self.max_depth     = cfg.max_depth

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        # Number of rays in this batch
        B = ray_bundle.origins.shape[0]

        # 1) Create N depths between near and far
        z_vals = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.n_pts_per_ray,
            device=ray_bundle.origins.device,
        )               # (N,)
        z_vals = z_vals.unsqueeze(0).expand(B, -1)  # (B, N)

        # 2) Compute sample points per ray, preserving (B, N, 3)
        #    origin: (B, 3) → (B, 1, 3)
        #    dirs:   (B, 3) → (B, 1, 3)
        #    z_vals: (B, N) → (B, N, 1)
        origins = ray_bundle.origins.unsqueeze(1)       # (B, 1, 3)
        dirs    = ray_bundle.directions.unsqueeze(1)    # (B, 1, 3)
        depths  = z_vals.unsqueeze(-1)                  # (B, N, 1)

        sample_points  = origins + dirs * depths        # (B, N, 3)
        sample_lengths = depths                         # (B, N, 1)

        # 3) Return a new RayBundle with the correct shapes
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=sample_lengths,
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}