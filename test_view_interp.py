from __future__ import print_function, division

import argparse
import logging

import ipdb
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from lib.loss import l1_loss, l2_loss, ssim, psnr
from lib.human_loader import HumanDataset
# from lib.zjumocap import ZJUMOCAPDataset as HumanDataset
# from lib.human_loader_pixelsplat import HumanDataset
from lib.network_2 import HumanModel
# from lib.net_pixelsplat import HumanModel
from torch.utils.data import DataLoader
from gaussian_renderer import renderThuman
from config.config import Config as config
# from lib.utils import get_novel_calib
# from lib.GaussianRender import pts2render
import lpips
import torch
import warnings
import json
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

warnings.filterwarnings("ignore", category=UserWarning)

def get_novel_calib(data,intr0,intr1,extr0,extr1, opt, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = 1
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        # intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        # intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        # extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        # extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        # if opt.use_hr_img:
        #     intr_new[:2] *= 2
        intr_new[:2]*=0.5
        width, height = 512,512
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))
    # ipdb.set_trace()
    data['target']['FovX'] = torch.tensor(np.array(fovx_list)).cuda()
    data['target']['FovY'] = torch.tensor(np.array(fovy_list)).cuda()
    data['target']['height']=torch.tensor(np.array([data['target']['height']]))
    data['target']['width']=torch.tensor(np.array([data['target']['width']]))
    data['target']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['target']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['target']['camera_center'] = torch.concat(camera_center_list).cuda()

    return data

class HumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = HumanModel(self.cfg)
        self.dataset = HumanDataset(self.cfg.dataset, phase=phase)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True)

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

    def smplx2render(self, data, i, ret):
        xyz, rot, scale, opacity, rgb,_=ret
        render_i = renderThuman(data, i, xyz[i], rgb[i], rot[i], scale[i], opacity[i],
                                bg_color=self.cfg.dataset.bg_color)
        img_pred = render_i[0].unsqueeze(0)
        mask_pred = render_i[1].unsqueeze(0)
        depth_pred = render_i[2].unsqueeze(0)

        return img_pred, mask_pred, depth_pred

    def infer_static(self):
        total_samples = len(os.listdir(os.path.join(self.cfg.dataset.data_root,'val', 'img')))
        novel_view_nums=10
        for idx in tqdm(range(total_samples)[::16]):
            item = self.dataset.get_test_item(idx)
            data = self.fetch_data(item)
            all_extr=data['all_extr']
            all_intr=data['all_intr']
            ret=self.model.predict_gs(data, render=True)
            name_idx=0
            
            source_ids=[[0,3],[3,6],[6,9],[9,12],[12,15],[15,0]]
            for id in range(6):
                source_id=source_ids[id]
                intr0=all_intr[source_id[0]]
                intr1=all_intr[source_id[1]]
                extr0=all_extr[source_id[0]]
                extr1=all_extr[source_id[1]]
                for i in range(novel_view_nums):
                    ratio_tmp = (i+0.5)*(1/novel_view_nums)
                    data_i = get_novel_calib(data,intr0,intr1,extr0,extr1, self.cfg.dataset, ratio=ratio_tmp, intr_key='intr_ori', extr_key='extr_ori')
                    with torch.no_grad():
                        img_pred, mask_pred, depth_pred = self.smplx2render(data_i,0, ret)
                    # ipdb.set_trace()
                    tmp_novel = img_pred[0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    cv2.imwrite(self.cfg.test_out_path + '/%s_%s.jpg' % (data['target']['sample_name'], str(name_idx).zfill(2)), tmp_novel[:, :, ::-1].astype(np.uint8))
                    name_idx+=1

      
    def fetch_data(self, data):
        # ipdb.set_trace()
        for view in ['ref']:
            for item in data[view].keys():
                if isinstance(data[view][item], torch.Tensor):
                    data[view][item] = data[view][item][None].cuda()
                else:
                    data[view][item] = torch.from_numpy(data[view][item][None]).cuda()
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    # parser.add_argument('--novel_view_nums', type=int, default=5)
    arg = parser.parse_args()

    cfg = config()
    # cfg_for_train = os.path.join('./config', 'zju.yaml')
    # cfg_for_train = os.path.join('./config', 'pixelsplat.yaml')
    cfg_for_train = os.path.join('./config', 'config.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    # cfg.dataset.test_data_root = arg.test_data_root

    cfg.restore_ckpt = os.path.join(arg.ckpt_path, 'ckpt', cfg.name + '_final.pth')
    cfg.test_out_path = os.path.join(arg.ckpt_path, 'test_out')
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = HumanRender(cfg, phase='test')
    render.infer_static()
