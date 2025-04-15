#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer



def renderThuman(data, idx, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(data['target']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['target']['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['target']['height'][idx]),
        image_width=int(data['target']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['target']['world_view_transform'][idx],
        projmatrix=data['target']['full_proj_transform'][idx],
        sh_degree=3,
        campos=data['target']['camera_center'][idx],
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, _, depth_image, mask_image = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return rendered_image, depth_image,mask_image

def renderThuman_R(data, idx,view_idx, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['ref']['FovX'][idx,view_idx] * 0.5)
    tanfovy = math.tan(data['ref']['FovY'][idx,view_idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['ref']['height'][idx,view_idx]),
        image_width=int(data['ref']['width'][idx,view_idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['ref']['world_view_transform'][idx,view_idx],
        projmatrix=data['ref']['full_proj_transform'][idx,view_idx],
        sh_degree=3,
        campos=data['ref']['camera_center'][idx,view_idx],
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, _, depth_image, mask_image = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return rendered_image, depth_image,mask_image
