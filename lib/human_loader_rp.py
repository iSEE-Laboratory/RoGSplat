from torch.utils.data import Dataset
import numpy as np
import os
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import lib.utils as utils
import cv2
import json
import random

def project(pts, extrinsic, intrinsic):
    pts = pts.T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    return pts[:2].T


def check_cam(smplx, extr, intr, img):
    uv = project(smplx, extr, intr)
    uv = uv.astype(np.int32)
    import cv2 as cv

    img = img * 255
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = img.astype(np.uint8)
    for c in uv:
        cv.circle(img, c[:2], 1, (0, 255, 255), 1)
    cv.imwrite('a.jpg', img)


class HumanDataset(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        if self.phase == 'train':
            self.data_root = os.path.join(opt.data_root, 'train')
            self.intv = 1
            self.begin = 0
        elif self.phase == 'val':
            self.data_root = os.path.join(opt.data_root, 'val')
            self.intv = 4890
            self.begin = 2
        elif self.phase == 'test':

            self.data_root = os.path.join(opt.data_root, 'val')
            self.intv = 10
            self.begin = 0

        self.cam_num = 36
        self.pose_num = 4
        self.pose_intv = 5
        self.pose_start = 1

        self.human_list = os.listdir(self.data_root)
        self.human_list.sort()

        self.img_path = os.path.join(self.data_root, '%s/img/%s/%s.jpg')
        self.mask_path = os.path.join(self.data_root, '%s/mask/%s/%s.png')
        self.param_path = os.path.join(self.data_root,'../easymocap_smpl', '%s/easymocap_smpl/%d.npy')
        self.depth_path = os.path.join(self.data_root,'../easymocap_smpl', '%s/smpl_depth/%s/%s.jpg')
        self.vertex_path = os.path.join(self.data_root,'../easymocap_smpl', '%s/vertex/%d.npy')

        self.cams_all = []
        for subject_root in self.human_list:
            camera_file = os.path.join(self.data_root, subject_root, 'cameras.json')
            camera = json.load(open(camera_file))
            self.cams_all.append(camera)

    def read_smpl_param(self,path,pose_index):
        params_ori = dict(np.load(path, allow_pickle=True))['smpl'].item()
        params = {}
        params['betas'] = np.array(params_ori['betas']).astype(np.float32)
        params['poses'] = np.zeros((1,72)).astype(np.float32)
        params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
        params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
        params['transl'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
        return params

    def load_single_view(self, human_idx, pose_index, view_index, require_vertex=False, require_depth=True):
        human_name = self.human_list[human_idx]
        cam_name = f'camera{str(view_index).zfill(4)}'
        pose_name = f'{str(pose_index).zfill(4)}'
        img_name = self.img_path % (human_name, cam_name, pose_name)
        mask_name = self.mask_path % (human_name, cam_name, pose_name)
        param_name = self.param_path % (human_name,pose_index)
        depth_name = self.depth_path % (human_name, cam_name, pose_name)
        vertex_name = self.vertex_path % (human_name, pose_index)
        cam = self.cams_all[human_idx]
        img = utils.read_img(img_name)

        K = np.array(cam[cam_name]['K']).astype(np.float32)
        R = np.array(cam[cam_name]['R']).astype(np.float32)
        T = np.array(cam[cam_name]['T']).astype(np.float32)
        intr = K
        extr = np.concatenate([R, T[:, None]], axis=-1)
        intr[:2] *= self.opt.ratio
        H, W = int(img.shape[0] * self.opt.ratio), int(img.shape[1] * self.opt.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        mask = utils.read_img(mask_name)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        img[mask != 255] = 0
        vertex = None
        param=None
        if require_vertex is not None:
            vertex = np.load(vertex_name).astype(np.float32)
            # param=self.read_smpl_param(param_name,pose_index)
            param=np.load(param_name,allow_pickle=True).item()

        depth = None

        if require_depth:
            depth = utils.read_img(depth_name).astype(np.float32)
            valid_mask = depth > 10
            depth[valid_mask] = 255 / depth[valid_mask]
            depth[~valid_mask] = 0
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        return img, mask, intr, extr, vertex, depth,param

    def get_novel_view_tensor(self, human_idx, pose_index, view_index):
        img, mask, intr, extr, vertex, _,param = self.load_single_view(human_idx, pose_index, view_index,
                                                                     require_vertex=True, require_depth=False)
        width, height = img.shape[:2]

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        mask = mask[..., None]
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        mask = mask / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height,
                                                w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0,
                                                                                                                      1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'sample_name': str(human_idx) + '_' + str(view_index) + '_' + str(pose_index),
            'image': img,
            'mask': mask,
            'extr': torch.FloatTensor(extr),
            'intr': torch.FloatTensor(intr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center,
            'smpl': vertex,
            'Rh':cv2.Rodrigues(param['Rh'])[0].astype(np.float32),
            'Th':param['Th'].astype(np.float32)
        }

        return novel_view_data

    def get_ref_view_tensor(self, human_idx, pose_index, view_index):
        img, mask, intr, extr, vertex, depth,param = self.load_single_view(human_idx, pose_index, view_index)

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = 2 * (img / 255.0) - 1.0

        mask = mask[..., None]
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        mask = mask / 255.0
        width, height = img.shape[1:]
        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)
        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height,
                                                w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0,
                                                                                                                      1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        return img, mask, torch.from_numpy(intr), torch.from_numpy(extr), camera_center, torch.from_numpy(
            depth), torch.FloatTensor([FovX]), torch.FloatTensor(
            [FovY]), world_view_transform, full_proj_transform, torch.FloatTensor([height]), torch.FloatTensor(
            [width])


    def get_item(self, index):

        human_idx = index // (self.pose_num * self.cam_num)
        if '130' in self.human_list[human_idx]: # wrong smpl
            i=human_idx
            while human_idx==i:
                human_idx=random.randint(0,len(self.human_list))
        pose_index = (index % (
                self.pose_num * self.cam_num)) // self.cam_num * self.pose_intv + self.pose_start
        view_index = index % self.cam_num

        # target view
        target_data = self.get_novel_view_tensor(human_idx, pose_index, view_index)
        # reference views
        ref_data = {
            'image': [],
            'mask': [],
            'intr': [],
            'extr': [],
            'camera_center': [],
            'depth': [],
            'FovX': [],
            'FovY': [],
            'height': [],
            'width': [],
            'world_view_transform': [],
            'full_proj_transform': [],
        }
        if self.phase == 'train':
            ref_view = np.arange(0, self.opt.training_view_num)
            np.random.shuffle(ref_view)
            ref_view = ref_view[:self.opt.ref_view_num]
        else:
            ref_view = np.array(self.opt.ref_view)

        for view in ref_view:
            img, mask, intr, extr, camera_center, depth, FovX, FovY, world_view_transform, full_proj_transform, height, width = self.get_ref_view_tensor(
                human_idx, pose_index, view)

            ref_data['image'].append(img)
            ref_data['mask'].append(mask)
            ref_data['intr'].append(intr)
            ref_data['extr'].append(extr)
            ref_data['camera_center'].append(camera_center)
            ref_data['depth'].append(depth)
            ref_data['FovX'].append(FovX)
            ref_data['FovY'].append(FovY)
            ref_data['world_view_transform'].append(world_view_transform)
            ref_data['full_proj_transform'].append(full_proj_transform)
            ref_data['height'].append(height)
            ref_data['width'].append(width)

        ref_data['image'] = torch.stack(ref_data['image'], dim=0).to(torch.float32)
        ref_data['mask'] = torch.stack(ref_data['mask'], dim=0).to(torch.float32)
        ref_data['intr'] = torch.stack(ref_data['intr'], dim=0).to(torch.float32)
        ref_data['extr'] = torch.stack(ref_data['extr'], dim=0).to(torch.float32)
        ref_data['camera_center'] = torch.stack(ref_data['camera_center'], dim=0).to(torch.float32)
        ref_data['depth'] = torch.stack(ref_data['depth'], dim=0)
        ref_data['FovX'] = torch.stack(ref_data['FovX'], dim=0)
        ref_data['FovY'] = torch.stack(ref_data['FovY'], dim=0)
        ref_data['world_view_transform'] = torch.stack(ref_data['world_view_transform'], dim=0)
        ref_data['full_proj_transform'] = torch.stack(ref_data['full_proj_transform'], dim=0)
        ref_data['height'] = torch.stack(ref_data['height'], dim=0)
        ref_data['width'] = torch.stack(ref_data['width'], dim=0)

        return {
            'target': target_data,
            'ref': ref_data
        }

    def __getitem__(self, index):
        index = index * self.intv

        return self.get_item(index)

    def __len__(self):
        total_len=len(self.human_list) * self.cam_num * self.pose_num
        return total_len // self.intv

