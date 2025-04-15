from torch import nn
from .SPD import SPD


class ModelPU(nn.Module):
    def __init__(self, up_factors=None, in_rgb=False,rgb_dim=3,dim_feat=32,use_global_feat=True):
        super(ModelPU, self).__init__()
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(up_factor=factor, i=i, global_feat=use_global_feat, dim_feat=dim_feat, in_rgb=in_rgb,rgb_dim=rgb_dim))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, x, global_feat, rgb):
        """
        Args:
            x: Tensor, (b, n_coarse, 3), coarse point cloud
        """
        arr_pcd = []
        # if rgb is not None:
        #     rgb = rgb.permute(0, 2, 1).contiguous()
        pcd = x.permute(0, 2, 1).contiguous()
        feat_prev = None
        for upper in self.uppers:
            pcd, feat_prev, rgb = upper(pcd, K_prev=feat_prev, feat_global=global_feat, rgb=rgb)

            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        # if rgb is not None:
        #     rgb = rgb.permute(0, 2, 1)

        return pcd.permute(0, 2, 1).contiguous(), feat_prev.permute(0, 2, 1), rgb
