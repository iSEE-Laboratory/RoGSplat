import torch
from torch import nn
from lib.modules.embedder import get_embedder
from lib.utils import initseq

class GSRegresser(nn.Module):
    def __init__(self, in_ch=128, hidden_dim=256, in_rgb=False,num_gaussian=1,require_opacity=True,opacity_acti='sigmoid',require_offset=False,require_rgb=True):
        super().__init__()

        self.out_relu = nn.ReLU(inplace=True)
        rgb_dim = 3
        self.require_rgb=require_rgb
        self.opacity_acti_name=opacity_acti
        self.rgb_embedder, rgb_dim = get_embedder(6, rgb_dim)
        self.rot_head = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4*num_gaussian),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3*num_gaussian),
            nn.Softplus(beta=100)
        )
        
        self.require_opacity=require_opacity
        if require_opacity:
            self.opacity_head = nn.Sequential(
                nn.Linear(in_ch, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1*num_gaussian),
            )
            initseq(self.opacity_head)
        if opacity_acti=='sigmoid':
            self.opacity_acti=nn.Sigmoid()
        elif opacity_acti=='tanh':
            self.opacity_acti=nn.Tanh()
            
        else:
            raise NotImplementedError
        
        if require_offset:
            self.xyz_resi=nn.Sequential(
                nn.Linear(in_ch, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1*num_gaussian),
            )
        if require_rgb:
            if not in_rgb:
                rgb_dim = 0
            self.rgb_head = nn.Sequential(
                nn.Linear(in_ch + rgb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 3*num_gaussian),
                nn.Sigmoid()
            )
        self.tanh=nn.Tanh()


    def forward(self, rgb_feat, rgb=None,xyz=None):
        opacity=None
        if self.require_opacity:
            opacity = self.opacity_head(rgb_feat)
            opacity=self.opacity_acti(opacity)
            
            if self.opacity_acti_name=='tanh':
                valid_mask=opacity>0
                opacity=opacity[:,valid_mask[0,:,0],:]
                rgb_feat=rgb_feat[:,valid_mask[0,:,0],:]
                rgb=rgb[:,valid_mask[0,:,0],:]

        if xyz is not None:
            # xyz=xyz[:,valid_mask[0,:,0],:]
            xyz=xyz+self.tanh(self.xyz_resi(rgb_feat))*0.05

        rot = self.rot_head(rgb_feat)
        rot = torch.nn.functional.normalize(rot, dim=-1)

        scale = self.scale_head(rgb_feat)
        # scale=torch.clamp_max(scale, 0.01)
        if self.require_rgb:
            if rgb is not None:
                rgb = self.rgb_embedder(rgb)
                rgb_feat = torch.cat([rgb_feat, rgb], dim=-1)
            rgb = self.rgb_head(rgb_feat)
        if xyz is not None:
            return rot, scale, opacity, rgb,xyz
        else:
            return rot, scale, opacity, rgb
