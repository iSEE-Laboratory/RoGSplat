import torch
from torch import nn


class DINOv2(nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='large', freeze=False, load_from=None):
        super().__init__()

        if version == 'large':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        else:
            raise NotImplementedError

        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)

        self.freeze = freeze

    def forward(self, inputs):

        inputs = torch.nn.functional.interpolate(inputs, (518, 518), mode='bilinear', align_corners=False)
        if inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        B, _, h, w = inputs.shape

        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 1)
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 1)

        outs = []
        for feature in features:
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            outs.append(feature)
        outs = torch.stack(outs, dim=0)
        return outs
