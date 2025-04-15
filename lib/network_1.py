import torch
from torch import nn
from lib.modules.extractor import Unet
from lib.modules.projector import Projector
from lib.modules.gs_regresser import GSRegresser
from lib.SnowflakeNet.model_pu import ModelPU
from lib.modules.mvf import MVF
import torch.nn.functional as F
import lib.utils as utils
from gaussian_renderer import renderThuman, renderThuman_R
import spconv.pytorch as spconv
from lib.modules.sp_conv import SPConv


def to_cuda(data):
    # ipdb.set_trace()
    for k in data:
        if k in ['scale', 'transl', 'y_transl', 'human_height', 'v_scale','Rh','Th']:
            if isinstance(data[k],torch.Tensor):
                data[k] = data[k].cuda()
            else:
                data[k] = torch.tensor(data[k]).cuda()
    return data

class HumanModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ref_view_num = cfg.dataset.valid_ref_view_num
        self.img_encoder = Unet(encoder_dim=self.cfg.model.encoder.gsnet.encoder_dims,
                                         decoder_dim=self.cfg.model.encoder.gsnet.decoder_dims)
        self.depth_refiner = Unet(4, rgb_dim=3, encoder_dim=self.cfg.model.encoder.gsnet.encoder_dims,
                                           decoder_dim=self.cfg.model.encoder.gsnet.decoder_dims, predict_depth=True)
        self.MVF = MVF(in_ch=32)
        self.sp_conv = SPConv()
        self.gs_regresser = GSRegresser(in_ch=160, in_rgb=True)
        self.PU = ModelPU(up_factors=[1, 4], in_rgb=True,dim_feat=384, rgb_dim=cfg.dataset.valid_ref_view_num,use_global_feat=True) # 384
        self.projector = Projector()
      

    def render_ref_view(self, data):
        xyz, rot, scale, opacity, rgb, encoder_feat, decoder_feat, rgb_feat, pcd_feat,_ = self.predict_gaussian(data)
        bs = xyz.shape[0]
        img_list = []
        depth_list = []
        mask_list = []
        views = data['ref']['image'].shape[1]
        for i in range(bs):
            for v in range(views):
                render_i = renderThuman_R(data, i, v, xyz[i], rgb[i], rot[i], scale[i], opacity[i],
                                          bg_color=self.cfg.dataset.bg_color)
                img_list.append(render_i[0].unsqueeze(0))
                depth_list.append(render_i[1].unsqueeze(0))
                mask_list.append(render_i[2].unsqueeze(0))
        img_pred = torch.concat(img_list, dim=0)
        depth_pred = torch.concat(depth_list, dim=0)
        mask_pred = torch.concat(mask_list, dim=0)

        img_pred = img_pred.reshape(bs, views, *img_pred.shape[1:])
        depth_pred = depth_pred.reshape(bs, views, *depth_pred.shape[1:])
        mask_pred = mask_pred.reshape(bs, views, *mask_pred.shape[1:])
        return img_pred, mask_pred, depth_pred, encoder_feat, decoder_feat, xyz, rgb_feat, pcd_feat

    def smplx2render(self, data, xyz, rot, scale, opacity, rgb):
        bs = xyz.shape[0]
        img_list = []
        depth_list = []
        mask_list = []

        for i in range(bs):
            render_i = renderThuman(data, i, xyz[i], rgb[i], rot[i], scale[i], opacity[i],
                                    bg_color=self.cfg.dataset.bg_color)
            img_list.append(render_i[0].unsqueeze(0))
            depth_list.append(render_i[1].unsqueeze(0))
            mask_list.append(render_i[2].unsqueeze(0))
        img_pred = torch.concat(img_list, dim=0)
        depth_pred = torch.concat(depth_list, dim=0)
        mask_pred = torch.concat(mask_list, dim=0)
        return img_pred, mask_pred, depth_pred

    def compute_visibility(self,points,extr, intr, H, W, zbuffer):

        zbuf, depth = self.projector.project_to_depth(points, extr, intr, H, W, zbuffer)
        zbuf = zbuf.reshape(*depth.shape[:2], *zbuf.shape[-2:])
        visibility = 1 - torch.abs(zbuf - depth) / (zbuf + 1e-9)
        visibility=torch.clamp(visibility,0,1)
        return visibility.to(torch.float32)

    def multi_view_fusion(self, points, extr, intr, H, W, feat_map, image,visibility):
        feat_sampled,rgb_sampled = self.projector.apply(points, extr, intr, H, W, feat_map,image)

        if image is not None:
            rgb_sampled = rgb_sampled.reshape(*visibility.shape[:2], *rgb_sampled.shape[-2:])

        feat_sampled = feat_sampled.reshape(*visibility.shape[:2], *feat_sampled.shape[-2:])
        
        feat, attn = self.MVF(feat_sampled, visibility)

        rgb = None
        if image is not None:
            rgb = torch.sum(attn * rgb_sampled, dim=1)

        return feat, rgb

    def world2smpl(self, xyz, param):
        xyz = torch.matmul(xyz - param['Th'], param['Rh'])
        return xyz

    def smpl2world(self, xyz, param):
        xyz = torch.matmul(xyz, param['Rh'].transpose(1, 2)) + param['Th']
        return xyz

    def normalize_depth(self, depth, near, far):
        depth = (depth - near[..., None, None]) / (far - near)[..., None, None]
        depth = depth * 2 - 1
        return depth

    def unnormalize_depth(self, depth, near, far):
        depth = 0.5 * depth + 0.5
        depth = depth * (far - near)[..., None, None] + near[..., None, None]
        return depth

    def get_near_far(self, depth, depth_mask=None):
        near = []
        far = []
        if depth_mask is None:
            depth_mask = depth != 0
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

    def construct_feature_volume(self, pts, vox_size, feat):
        bound = utils.get_bound(pts)
        coord, out_sh, world_coord = utils.get_voxel_coord(pts, bound, vox_size)
        batch_size = 1

        xyzc = spconv.SparseConvTensor(feat, coord, out_sh, batch_size)
        feature_volume = self.sp_conv(xyzc)
        return feature_volume, bound, out_sh

    def voxelize(self, pts, volume_pts, volume_feat):

        coarse_vox_size = self.cfg.coarse_vox_size
        feature_volume, bound, out_sh = self.construct_feature_volume(volume_pts, coarse_vox_size, volume_feat[0])
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
    
    def predict_gaussian(self, data,render=True):
        data['target'] = to_cuda(data['target'])
        bs = data['ref']['image'].shape[0]
        n_channels, H, W = data['ref']['image'].shape[2:]
        image = data['ref']['image'].reshape(-1, n_channels, H, W)

        mask = data['ref']['mask'].to(torch.bool)

        if isinstance(data['target']['smpl'],torch.Tensor):
            smpl = data['target']['smpl'].cuda()
        else:
            smpl = torch.from_numpy(data['target']['smpl'])[None].cuda()

        extr = data['ref']['extr']
        intr = data['ref']['intr']

        depth = data['ref']['depth'].reshape(-1, H, W, 1).permute(0, 3, 1, 2)

        depth = depth.reshape(*depth.shape[:2], -1)
        depth_loss=None
        depth_mask = depth != 0
        abs_near, abs_far = self.get_near_far(depth,depth_mask)
        depth = self.normalize_depth(depth, abs_near, abs_far)

        depth=depth.reshape(*depth.shape[:2],H,W)
        if render:
            with torch.no_grad():
                refined_depth, _, depth_feat_map = self.depth_refiner(image, depth)
        else:
            refined_depth, _, depth_feat_map = self.depth_refiner(image, depth)

        depth_mask=depth_mask.reshape(*depth_mask.shape[:2],H,W)

        if refined_depth.requires_grad:
            
            valid_mask = depth_mask & mask.flatten(0, 1)
            refine_loss = (depth[valid_mask].detach() - refined_depth[valid_mask]).abs().mean()
            out_refined_depth = refined_depth.clone()

            out_pred_depth = image
            depth_loss = (refine_loss, out_refined_depth, out_pred_depth)
            if not render:
                return depth_loss
        
        refined_depth = refined_depth.reshape(*depth.shape[:2], -1)
        refined_depth = self.unnormalize_depth(refined_depth, abs_near, abs_far)
        refined_depth = refined_depth.reshape(-1, 1, H, W)

        depth = depth.reshape(*depth.shape[:2], -1)
        depth = self.unnormalize_depth(depth, abs_near, abs_far)
        depth = depth.reshape(-1, 1, H, W)

        refined_depth[~mask[0]]=0
        depth[~depth_mask]=0

        dep2xyz = utils.depth2pc(refined_depth, extr.flatten(0, 1), intr.flatten(0, 1))
        dep2xyz = dep2xyz.reshape(bs, self.ref_view_num, H, W, -1)
        depth_feat_map=depth_feat_map.reshape(bs,-1,*depth_feat_map.shape[-3:])

        vox_feat=[]
        
        for i in range(bs):
            mask_i=mask[i].permute(0,2,3,1).squeeze(-1)
            pcd_i=dep2xyz[i][mask_i][None]

            geo_feat_i=depth_feat_map[i].permute(0,2,3,1)[mask_i][None]
            smpl_i=smpl[i][None]

            # filter outlier points
            smpl_i=self.world2smpl(smpl_i.clone(), data['target'])
            pcd_i=self.world2smpl(pcd_i.clone(),data['target'])
            
            bbox_max=smpl_i.max(1)[0]+0.1
            bbox_min=smpl_i.min(1)[0]-0.1
            valid_pcd_mask=((pcd_i<bbox_max[:,None]).sum(2)+(pcd_i>bbox_min[:,None]).sum(2))==6
            pcd_i=pcd_i[valid_pcd_mask][None]
            geo_feat_i=geo_feat_i[valid_pcd_mask][None]
            vox_feat.append(self.voxelize(smpl_i,pcd_i,geo_feat_i))

        vox_feat=torch.cat(vox_feat,dim=0)

        visibility=self.compute_visibility(smpl,extr,intr,H,W,depth) # (1,4,6890,1)

        _, encoder_feat, decoder_feat = self.img_encoder(image, None)

        rgb_feat, _ = self.multi_view_fusion(smpl, extr[0], intr[0], H, W, encoder_feat,None, visibility)
        pcd=self.world2smpl(smpl, data['target'])

        pcd,pcd_feat,visibility=self.PU(pcd, torch.cat([rgb_feat,vox_feat],dim=-1).permute(0, 2, 1),visibility.squeeze(-1))
        
        pcd=self.smpl2world(pcd, data['target'])

        rgb_feat,rgb=self.multi_view_fusion(pcd,extr[0], intr[0], H, W, decoder_feat,image, visibility.unsqueeze(-1))
        local_feat=torch.cat([rgb_feat,pcd_feat],dim=-1)
        rot, scale, opacity, rgb = self.gs_regresser(local_feat, rgb)
        return pcd, rot, scale, opacity, rgb, encoder_feat, decoder_feat, rgb_feat, pcd_feat,depth_loss


    def forward(self, data, render=True):
        if not render:
            return self.predict_gaussian(data,render)
        
        pcd, rot, scale, opacity, rgb, _, _, _, _,depth_loss = self.predict_gaussian(data,render)
        img_pred, mask_pred, depth_pred = self.smplx2render(data, pcd, rot, scale, opacity, rgb)

        return img_pred, mask_pred, depth_pred,depth_loss
