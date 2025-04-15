import torch
import torch.nn.functional as F


class Projector():
    def __init__(self) -> None:
        pass

    def project(self, pts, extrinsic, intrinsic, h, w):
        pts = pts.permute(0, 2, 1)
        calib = intrinsic @ extrinsic
        pts = calib[..., :3, :3] @ pts[:, None]
        pts = pts + calib[..., :3, 3:4]
        pts[..., :2, :] /= (pts[..., 2:, :] + 1e-8)

        # depth = 1.0 / (pts[..., 2, :] + 1e-8)
        depth = pts[..., 2, :]
        pts = pts[..., :2, :].permute(0, 1, 3, 2)
        return pts, self.uv_normalize(pts, h, w), depth

    def uv_normalize(self, uv, h, w):
        resize_factor = torch.tensor([w - 1., h - 1.]).to(uv.device)[None, None, None, :]
        normalized_uv = 2 * uv / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_uv

    def project_to_depth(self,pts, extrinsic, intrinsic, h, w, zbuffer=None):
        uv, normalized_uv, depth = self.project(pts, extrinsic, intrinsic, h, w)
        normalized_uv = normalized_uv.view(-1, 1, *normalized_uv.shape[-2:])

        zbuf=F.grid_sample(zbuffer, normalized_uv, mode='nearest')
        zbuf=zbuf[:, :, 0].permute(0, 2, 1)
        return zbuf,depth.unsqueeze(-1)

    def apply(self, pts, extrinsic, intrinsic, h, w, img_feat=None, img=None):
        uv, normalized_uv, depth = self.project(pts, extrinsic, intrinsic, h, w)
        normalized_uv = normalized_uv.view(-1, 1, *normalized_uv.shape[-2:])

        rgb_sampled = None
        if img is not None:
            rgbs_sampled = F.grid_sample(img, normalized_uv, align_corners=True)

            rgb_sampled = rgbs_sampled[:, :, 0].permute(0, 2, 1)

        # deep feature sampling
        feat_sampled = None
        if img_feat is not None:
            if isinstance(img_feat, list):
                feat_sampled = []
                for feat_map in img_feat:
                    feat = F.grid_sample(feat_map, normalized_uv, align_corners=True)
                    feat = feat.squeeze(2).permute(0, 2, 1)
                    feat_sampled.append(feat)
            else:
                feat_sampled = F.grid_sample(img_feat, normalized_uv, align_corners=True)
                feat_sampled = feat_sampled.squeeze(2).permute(0, 2, 1)


        return feat_sampled,rgb_sampled
