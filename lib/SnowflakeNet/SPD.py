# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
import torch.nn as nn
from .utils import MLP_Res, MLP_CONV
from .skip_transformer import SkipTransformer


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=10, bounding=True, global_feat=True, in_rgb=False,
                 rgb_dim=3):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.in_rgb = in_rgb
        self.bounding = bounding
        self.radius = radius

        self.global_feat = global_feat
        self.ps_dim = 32 if global_feat else 64

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat if self.global_feat else 128, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, self.ps_dim])
        self.ps = nn.ConvTranspose1d(self.ps_dim, 128, up_factor, up_factor, bias=False)  # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.smpl_max = torch.tensor([[[0.8731], [0.5555], [0.1680]]],device='cuda')+0.05
        self.smpl_min = torch.tensor([[[-0.8720], [-1.1618], [-0.1224]]],device='cuda')-0.05
        if self.in_rgb:
            self.mlp_delta_rgb = MLP_CONV(in_channel=128, layer_dims=[64, rgb_dim])

    def forward(self, pcd_prev, feat_global=None, K_prev=None, rgb=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        # import ipdb
        # ipdb.set_trace()
        # bbox_min=pcd_prev.min(-1)[0].unsqueeze(-1)
        # bbox_max=pcd_prev.max(-1)[0].unsqueeze(-1)
        # radius=torch.norm(bbox_max-bbox_min,dim=1).item()

        # pcd_prev = (pcd_prev - self.smpl_min) / (self.smpl_max - self.smpl_min) * 2 - 1
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        # feat_1 = torch.cat([feat_1,
        #                     torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
        #                     feat_global.repeat(1, 1, feat_1.size(2))], 1) if self.global_feat else feat_1
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global], 1) if self.global_feat else feat_1
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        # ipdb.set_trace()
        delta = self.mlp_delta(torch.relu(K_curr))

        if self.bounding:
            delta = torch.tanh(delta) / self.radius ** self.i  # (B, 3, N_prev * up_factor)

        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta
        # pcd_child = (pcd_child * 0.5 + 0.5) * (self.smpl_max - self.smpl_min) + self.smpl_min

        if self.in_rgb:
            delta_rgb = self.mlp_delta_rgb(torch.relu(K_curr))
            rgb_child = self.up_sampler(rgb)
            rgb = rgb_child + torch.tanh(delta_rgb) * 0.1
            # rgb = torch.clamp(rgb_child + delta_rgb, -1, 1)

        return pcd_child, K_curr, rgb
