import math
from typing import List, NamedTuple

import torch
import torch.nn.functional as F
from pytorch3d.renderer.cameras import CamerasBase


# Convenience class wrapping several ray inputs:
#   1) Origins -- ray origins
#   2) Directions -- ray directions
#   3) Sample points -- sample points along ray direction from ray origin
#   4) Sample lengths -- distance of sample points from ray origin

class RayBundle(object):
    def __init__(
        self,
        origins,
        directions,
        sample_points,
        sample_lengths,
    ):
        self.origins = origins
        self.directions = directions
        self.sample_points = sample_points
        self.sample_lengths = sample_lengths

    def __getitem__(self, idx):
        return RayBundle(
            self.origins[idx],
            self.directions[idx],
            self.sample_points[idx],
            self.sample_lengths[idx],
        )

    @property
    def shape(self):
        return self.origins.shape[:-1]

    @property
    def sample_shape(self):
        return self.sample_points.shape[:-1]

    def reshape(self, *args):
        return RayBundle(
            self.origins.reshape(*args, 3),
            self.directions.reshape(*args, 3),
            self.sample_points.reshape(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.reshape(*args, self.sample_lengths.shape[-2], 3),
        )

    def view(self, *args):
        return RayBundle(
            self.origins.view(*args, 3),
            self.directions.view(*args, 3),
            self.sample_points.view(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.view(*args, self.sample_lengths.shape[-2], 3),
        )

    def _replace(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        return self


# Sample image colors from pixel values
def sample_images_at_xy(
    images: torch.Tensor,
    xy_grid: torch.Tensor,
):
    batch_size = images.shape[0]
    spatial_size = images.shape[1:-1]

    xy_grid = -xy_grid.view(batch_size, -1, 1, 2)

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=True,
        mode="bilinear",
    )

    return images_sampled.permute(0, 2, 3, 1).view(-1, images.shape[-1])


# Generate pixel coordinates from in NDC space (from [-1, 1])
def get_pixels_from_image(image_size, camera):
    W, H = image_size[0], image_size[1]

    # TODO (Q1.3): Generate pixel coordinates from [0, W] in x and [0, H] in y
    x = torch.arange(0, W, 1)
    y = torch.arange(0, H, 1)
    # TODO (Q1.3): Convert to the range [-1, 1] in both x and y
    x = (x / (W - 1)) * 2 - 1
    y = (y / (H - 1)) * 2 - 1
    
    # Create grid of coordinates
    xy_grid = torch.stack(
        tuple( reversed( torch.meshgrid(y, x) ) ),
        dim=-1,
    ).view(W * H, 2)
    return -xy_grid


# Random subsampling of pixels from an image
def get_random_pixels_from_image(n_pixels, image_size, camera):
    xy_grid = get_pixels_from_image(image_size, camera)
    
    # TODO (Q2.1): Random subsampling of pixel coordinaters
    pass

    # Return
    return xy_grid_sub.reshape(-1, 2)[:n_pixels]


# Get rays from pixel values
def get_rays_from_pixels(xy_grid, image_size, camera):
    W, H = image_size[0], image_size[1]
    device = xy_grid.device
    print(xy_grid.shape)
    # TODO (Q1.3): Map pixels to points on the image plane at Z=1
    ndc_points = torch.cat(
        [
            xy_grid,
            torch.ones_like(xy_grid[..., -1:])
        ],
        dim=-1
    )
    ndc_points = ndc_points.to(device)

    # TODO (Q1.3): Use camera.unproject to get world space points from NDC space points
    world_coord = camera.unproject_points(ndc_points, world_coordinates=True)

    # TODO (Q1.3): Get ray origins from camera center
    rays_o = camera.get_camera_center().expand_as(world_coord)

    # TODO (Q1.3): Get ray directions as image_plane_points - rays_o
    rays_d = torch.nn.functional.normalize(world_coord - rays_o, dim=-1)

    # Create and return RayBundle
    return RayBundle(
        rays_o,
        rays_d,
        torch.zeros_like(rays_o).unsqueeze(1),
        torch.zeros_like(rays_o).unsqueeze(1),
    )