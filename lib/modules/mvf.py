import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class MVF(nn.Module):
    def __init__(self,  in_ch=32, hidden_dim=128):
        super(MVF, self).__init__()

        activation_func = nn.ReLU(inplace=False)
        self.in_ch=in_ch

        self.weight_fc = nn.Sequential(nn.Linear(in_ch + 1, hidden_dim),
                                       activation_func,
                                       nn.Linear(hidden_dim, in_ch + 1),
                                       )
        self.weight_fc.apply(weights_init)

    def forward(self, f, v):

        """
        q:B,1,N,3
        k:B,4,N,3
        v:B,4,N,32
        """
        feat = torch.cat([f, v], dim=-1)
        x = self.weight_fc(feat)
        h, w = x[..., :self.in_ch], x[..., -1:]
        v = torch.clamp(v * w,0,1)
        v = torch.softmax(v, dim=1)
        feat = torch.sum(h * v, dim=1)
        return feat, v
