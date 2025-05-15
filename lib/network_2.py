import torch
from torch import nn
from lib.modules.extractor import Unet
import lib.utils as utils
from gaussian_renderer import renderThuman
from pytorch3d.ops.knn import knn_points
import torch.nn.functional as F
from lib.network_1 import HumanModel as net1
from lib.modules.mvf import MVF
from lib.modules.projector import Projector
from lib.modules.gs_regresser import GSRegresser
import spconv.pytorch as spconv
from lib.modules.sp_conv import SPConv2

class HumanModel(nn.Module):
    def __init__(self, cfg, training=True):
        super().__init__()
        self.cfg = cfg
        self.training = training
        self.net1 = net1(cfg)   

        self.gs_regresser = GSRegresser(in_ch=64, hidden_dim=256, in_rgb=True, require_opacity=True,opacity_acti='sigmoid',require_rgb=True)

        self.projector = Projector()
        self.ref_view_num = cfg.dataset.valid_ref_view_num

        # image encoder
        self.img_encoder=Unet(3,encoder_dim=self.cfg.model.encoder.gsnet.encoder_dims,
                                           decoder_dim=self.cfg.model.encoder.gsnet.decoder_dims)
        # depth refine
        self.depth_refiner = Unet(4, encoder_dim=self.cfg.model.encoder.gsnet.encoder_dims,
                                           decoder_dim=self.cfg.model.encoder.gsnet.decoder_dims, predict_depth=True)

        # mvf
        self.MVF = MVF(in_ch=64)

        self.geo_mlp=nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        self.sp_conv = SPConv2()

    def smplx2render(self, data, i, xyz, rot, scale, opacity, rgb):
        render_i = renderThuman(data, i, xyz, rgb, rot, scale, opacity,
                                bg_color=self.cfg.dataset.bg_color)
        img_pred = render_i[0].unsqueeze(0)
        mask_pred = render_i[1].unsqueeze(0)
        depth_pred = render_i[2].unsqueeze(0)

        return img_pred, mask_pred, depth_pred

    def compute_visibility(self,points,extr, intr, H, W, zbuffer):
        zbuf, depth = self.projector.project_to_depth(points, extr, intr, H, W, zbuffer)
        zbuf = zbuf.reshape(*depth.shape[:2], *zbuf.shape[-2:])
        with torch.no_grad():
            for i in range(len(zbuf)):
                out_pts_mask = (zbuf[i] == 0).sum(0).bool()
                if out_pts_mask.sum() != 0:
                    out_pts = points[i, out_pts_mask[:, 0]]
                    in_pts = points[i, ~out_pts_mask[:, 0]]
                    ret = knn_points(out_pts[None], in_pts[None], K=5)
                    dists, idx = ret.dists.sqrt(), ret.idx
                    disp = 1 / (dists + 1e-8)
                    weight = disp / disp.sum(dim=-1, keepdim=True)
                    k_zbuf = zbuf[i][:, ~out_pts_mask[:, 0]][:, idx[0]]
                    replace_zbuf = torch.sum(k_zbuf[..., 0] * weight, dim=-1)
                    zbuf[i][:, out_pts_mask] = replace_zbuf
        visibility = 1 - torch.abs(zbuf - depth) / (zbuf + 1e-9)
        return visibility
    
    def multi_view_fusion(self, points, extr, intr, H, W, feat_map, image,visibility,func_mvf):
        feat_sampled,rgb_sampled = self.projector.apply(points, extr, intr, H, W, feat_map,image)

        if image is not None:
            rgb_sampled = rgb_sampled.reshape(*visibility.shape[:2], *rgb_sampled.shape[-2:])

        feat_sampled = feat_sampled.reshape(*visibility.shape[:2], *feat_sampled.shape[-2:])
        
        feat, attn = func_mvf(feat_sampled, visibility)

        rgb = None
        if image is not None:
            rgb = torch.sum(attn * rgb_sampled, dim=1)

        return feat, rgb

    def world2smpl(self, xyz, param):
        # if easymocap
        xyz = torch.matmul(xyz - param['Th'], param['Rh']) 

        # if THuman
        # xyz[..., 1] += param['y_transl'] 
        # xyz[..., :3] = xyz[..., :3] / param['height'] * param['v_scale']
        # xyz = (xyz - param['transl']) / param['scale']

        return xyz

    def smpl2world(self, xyz, param):
        # if easymocap
        xyz = torch.matmul(xyz, param['Rh'].transpose(1, 2)) + param['Th']

        # THuman
        # xyz = xyz * param['scale'] + param['transl']
        # xyz[..., :3] = xyz[..., :3] / param['v_scale'] * param['human_height']
        # xyz[..., 1] -= param['y_transl']
        return xyz

    def normalize_depth(self, depth, near, far):

        depth = (depth - near[..., None, None]) / (far - near)[..., None, None]
        depth = depth * 2 - 1
        return depth

    def unnormalize_depth(self, depth, near, far):
        depth = 0.5 * depth + 0.5
        depth = depth * (far - near)[..., None, None] + near[..., None, None]
        return depth

    def get_near_far(self, depth, depth_mask):
        near = []
        far = []
        for i in range(depth.shape[0]):
            near_i = depth[i][depth_mask[i]].min()
            far_i = depth[i][depth_mask[i]].max()
            far_i = far_i + ((far_i - near_i) * 0.1)
            near.append(near_i)
            far.append(far_i)
            depth[i][~depth_mask[i]] = far_i

        near = torch.stack(near, dim=0)
        far = torch.stack(far, dim=0)
        near = near - ((far - near) * 0.02)
        return near, far

    def construct_feature_volume(self, pts, vox_size, feat,conv_func):
        bound = utils.get_bound(pts)
        coord, out_sh, world_coord = utils.get_voxel_coord(pts, bound, vox_size)
        batch_size = 1

        xyzc = spconv.SparseConvTensor(feat, coord, out_sh, batch_size)
        feature_volume = conv_func(xyzc)
        # feature_volume=xyzc.dense()
        return feature_volume, bound, out_sh

    def voxelize(self, pts, volume_pts, volume_feat,conv_func):

        coarse_vox_size = self.cfg.coarse_vox_size

        feature_volume, bound, out_sh = self.construct_feature_volume(volume_pts, coarse_vox_size, volume_feat[0],conv_func)

        grid = utils.get_grid_coord(pts, bound, out_sh, coarse_vox_size)
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        features = features.permute(0, 2, 1)
        
        return features
    
    def predict_gs(self,data,render=False):
        bs = data['ref']['image'].shape[0]
        n_channels, H, W = data['ref']['image'].shape[2:]
        image = data['ref']['image'].reshape(-1, n_channels, H, W)
        zbuffer = data['ref']['depth'].reshape(-1, 1, H, W)
        mask = data['ref']['mask'].to(torch.bool)
        extr = data['ref']['extr']
        intr = data['ref']['intr']
        with torch.no_grad():
            imgs, depth_mask, depth, encoder_feat, decoder_feat, coarse_pcd, coarse_rgb_feat, coarse_pcd_feat = self.net1.render_ref_view(
                data)
        
        depth_mask = depth_mask > 0.95
        depth = depth.flatten(0, 1)
        h, w = depth.shape[-2:]
        depth = depth.reshape(*depth.shape[:2], -1)
        depth_mask = depth_mask.reshape(*depth.shape[:2], -1)
        
        near, far = self.get_near_far(depth, depth_mask)

        depth = self.normalize_depth(depth, near, far)

        depth = depth.reshape(*depth.shape[:2], h, w)
        depth_mask = depth_mask.reshape(bs*self.ref_view_num,1, h, w)
        # depth refine
        refined_depth, depth_encoder_feat, depth_decoder_feat = self.depth_refiner(image, depth)

        depth_loss = None

        if refined_depth.requires_grad:
            depth_mask=depth_mask.reshape(*depth_mask.shape[:2],H,W)
            valid_mask = depth_mask & mask.flatten(0, 1)
            refine_loss = (depth[valid_mask].detach() - refined_depth[valid_mask]).abs().mean()
            out_refined_depth = refined_depth.clone()

            out_pred_depth = image
            depth_loss = (refine_loss, out_refined_depth, out_pred_depth)
            if not render:
                return depth_loss
        
        depth = depth.reshape(*depth.shape[:2], -1)
        refined_depth = refined_depth.reshape(*depth.shape[:2], -1)

        depth = self.unnormalize_depth(depth, near, far)
        refined_depth = self.unnormalize_depth(refined_depth, near, far)

        depth = depth.reshape(bs, self.ref_view_num, 1, H, W)
        refined_depth = refined_depth.reshape(-1, 1, H, W)
        refined_depth[~mask[0]]=0

        if render:
            _, encoder_feat, decoder_feat = self.img_encoder(image, None)
            xyz = utils.depth2pc(refined_depth, extr.flatten(0, 1), intr.flatten(0, 1))

            xyz = xyz.reshape(bs, self.ref_view_num, H, W, -1)
            # import ipdb
            # ipdb.set_trace()
            encoder_feat = encoder_feat.reshape(bs, self.ref_view_num, *encoder_feat.shape[1:])
            decoder_feat = decoder_feat.reshape(bs, self.ref_view_num, *decoder_feat.shape[1:])#.permute(0,1,3,4,2)
            depth_decoder_feat=depth_decoder_feat.reshape(bs, self.ref_view_num, *depth_decoder_feat.shape[1:])
            image = image.reshape(bs, self.ref_view_num, *image.shape[1:])#.permute(0,1,3,4,2)

            for i in range(bs):
                xyz_i = xyz[i]
                coarse_pcd_i = coarse_pcd[i][None]
                mask_i = mask[i].permute(0, 2, 3, 1).squeeze(-1)
                xyz_i = xyz_i[mask_i][None]
    
                xyz_i=self.world2smpl(xyz_i,data['target'])
                coarse_pcd_i=self.world2smpl(coarse_pcd_i.clone(),data['target'])

                guidance_feat=self.voxelize(xyz_i,coarse_pcd_i,coarse_pcd_feat[i][None],self.sp_conv)
                xyz_i=self.smpl2world(xyz_i,data['target'])
                offset=self.geo_mlp(guidance_feat)
                xyz_i=xyz_i+offset
                
                visibility=self.compute_visibility(xyz_i,extr,intr,H,W,refined_depth)
                rgb_feat,rgb=self.multi_view_fusion(xyz_i,extr[i], intr[i], H, W, torch.cat([decoder_feat[i],depth_decoder_feat[i]],dim=1),image[i], visibility, self.MVF)
                rot_i, scale_i, opacity_i, rgb_i = self.gs_regresser(rgb_feat, rgb)
        return xyz_i,rot_i,scale_i,opacity_i,rgb_i,depth_loss

    def forward(self, data, render=False):
        if not render:
            return self.predict_gs(data,render)
        img_list,mask_list,depth_list=[],[],[]
        xyz_i,rot_i,scale_i,opacity_i,rgb_i,depth_loss=self.predict_gs(data,render)
        img_pred, mask_pred, depth_pred = self.smplx2render(data, 0, xyz_i[0], rot_i[0], scale_i[0],
                                                            opacity_i[0], rgb_i[0])
        img_list.append(img_pred)
        mask_list.append(mask_pred)
        depth_list.append(depth_pred)

        img_pred = torch.concat(img_list, dim=0)
        depth_pred = torch.concat(depth_list, dim=0)
        mask_pred = torch.concat(mask_list, dim=0)

        return img_pred, mask_pred, depth_loss
