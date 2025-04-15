from torch.utils.data import Dataset
import numpy as np
import os
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import lib.utils as utils
from lib.utils import check_cam
import cv2


def transform_can_smpl(xyz):
    # center = np.array([0, 0, 0]).astype(np.float32)
    # rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    # rot = rot.astype(np.float32)
    # trans = np.array([0, 0, 0]).astype(np.float32)
    # if np.random.uniform() > cfg.rot_ratio:
    #     return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans

class HumanDataset(Dataset):
    def __init__(self, opt, phase='train', subject=None):
        self.opt = opt
        # phase='train'
        self.phase = phase
        if self.phase == 'train':
            self.data_root = os.path.join(opt.data_root, 'train')
            self.intv = 1
            self.begin=0
        elif self.phase == 'val':
            self.data_root = os.path.join(opt.data_root, 'val')
            self.intv = 200
            self.begin = 2
        elif self.phase == 'test':
            # self.data_root = opt.test_data_root
            self.data_root = os.path.join(opt.data_root, 'val')
            self.intv = 1
            self.begin = 0

        self.smpl_name='easymocap_'
        self.img_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        self.img_hr_path = os.path.join(self.data_root, 'img/%s/%d_hr.jpg')
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%d.png')
        self.intr_path = os.path.join(self.data_root, 'parm/%s/%d_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parm/%s/%d_extrinsic.npy')
        self.smpl_path = os.path.join(self.data_root, self.smpl_name+'smpl/%s/vertices.npy')
        self.target_depth_name=self.smpl_name+'smpl_depth'
        self.depth_path = os.path.join(self.data_root, self.target_depth_name+'/%s/%d.jpg')
        
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))[self.begin::self.intv]
        if subject is not None:
            self.sample_list = [self.sample_list[(subject - 400) * 16]]

    def load_single_view(self, sample_name, subject_idx=None, source_id=0, hr_img=False, require_mask=True,
                         require_depth=False):
        img_name = self.img_path % (sample_name, source_id)
        image_hr_name = self.img_hr_path % (sample_name, source_id)
        mask_name = self.mask_path % (sample_name, source_id)
        intr_name = self.intr_path % (sample_name, source_id)
        extr_name = self.extr_path % (sample_name, source_id)
        smpl_name = self.smpl_path % (subject_idx)
        depth_name = self.depth_path % (sample_name, source_id)
        intr, extr = np.load(intr_name), np.load(extr_name)
        mask, pts = None, None
        if hr_img:
            img = utils.read_img(image_hr_name)
            intr[:2] *= 2
        else:
            img = utils.read_img(img_name)

        intr[:2] *= self.opt.ratio
        H, W = int(img.shape[0] * self.opt.ratio), int(img.shape[1] * self.opt.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if require_mask:
            mask = utils.read_img(mask_name)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # get smplx vertices
        smpl = None
        smpl_param = None
        if subject_idx is not None:
            smpl = np.load(smpl_name).astype(np.float32)
            smpl_param = np.load(smpl_name.replace('vertices','param'), allow_pickle=True).item()

        depth = None

        if require_depth:
            depth = utils.read_img(depth_name).astype(np.float32)
            valid_mask = depth > 10
            depth[valid_mask] = 255 / depth[valid_mask]
            depth[~valid_mask] = 0
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        
        return img, mask, intr, extr, pts, smpl, smpl_param, depth

    def get_novel_view_tensor(self, sample_name, subject_idx, view_id):
        img, mask, intr, extr, _, smpl, smpl_param, _= self.load_single_view(sample_name, subject_idx, view_id,
                                                                              hr_img=self.opt.use_hr_img,
                                                                              require_mask=True, require_depth=False,require_viz=False)
        width, height = img.shape[:2]

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        if mask.shape[2] == 3:
            mask = mask[:, :, [0]]
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

        min_xyz = np.min(smpl, axis=0)
        max_xyz = np.max(smpl, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bound = np.stack([min_xyz, max_xyz], axis=0)

        # get bounding mask and bcakground mask
        bound_mask = utils.get_bound_2d_mask(world_bound, intr, extr, img.shape[1], img.shape[2])

        smpl_param={
            'Rh':cv2.Rodrigues(smpl_param['Rh'])[0].astype(np.float32),
            'Th':smpl_param['Th'].astype(np.float32)
        }

        novel_view_data = {
            'sample_name': sample_name,
            'view_id': torch.IntTensor([view_id]),
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
            'smpl': smpl,
            # 'normal': normal,          
            'bound_mask': bound_mask,
        }
        novel_view_data.update(smpl_param)

        return novel_view_data

    def get_ref_view_tensor(self, sample_name, view_id):
        img, mask, intr, extr, _, _, _, depth = self.load_single_view(sample_name, None, view_id,
                                                                                  hr_img=self.opt.use_hr_img,
                                                                                  require_mask=True, require_depth=True,require_viz=True)

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = 2 * (img / 255.0) - 1.0

        if mask.shape[2] == 3:
            mask = mask[:, :, [0]]
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


    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        subject_idx, view_idx = sample_name.split('_')

        # target view
        target_id = np.random.choice(novel_id)
        target_data = self.get_novel_view_tensor(sample_name, subject_idx, target_id)

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
            ref_view = ref_view[:self.opt.valid_ref_view_num]
        else:
            ref_view = np.array(self.opt.ref_view)

        for view in ref_view:
            sample_name = subject_idx + '_' + f'{view:03d}'

            img, mask, intr, extr, camera_center, depth, FovX, FovY, world_view_transform, full_proj_transform, height, width = self.get_ref_view_tensor(
                sample_name, self.opt.source_id[0])

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
    
    def get_test_item(self,index):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        subject_idx, view_idx = sample_name.split('_')
        # novel_id=4
        # target view
        # target_id = np.random.choice(novel_id)
        target_id=3
        target_data = self.get_novel_view_tensor(sample_name, subject_idx, target_id)
        # self.get_rectified_stereo_data(subject_idx)
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
            ref_view = ref_view[:self.opt.valid_ref_view_num]
        else:
            ref_view = np.array(self.opt.ref_view)

        for view in ref_view:
            sample_name = subject_idx + '_' + f'{view:03d}'

            img, mask, intr, extr, camera_center, depth, FovX, FovY, world_view_transform, full_proj_transform, height, width = self.get_ref_view_tensor(
                sample_name, self.opt.source_id[0])

            # check_cam(target_data['smplx'],extr.numpy(),intr.numpy(),img)
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


        all_extr,all_intr=[],[]
        for i in range(16):
            sample_name = subject_idx + '_' + f'{i:03d}'
            extr_name = self.extr_path % (sample_name, 0)
            intr_name = self.intr_path % (sample_name, 0)
            intr, extr = np.load(intr_name), np.load(extr_name)
            all_intr.append(intr)
            all_extr.append(extr)

        return {
            'target': target_data,
            'ref': ref_data,
            'all_intr':all_intr,
            'all_extr':all_extr
        }

    def __getitem__(self, index):

        if self.phase == 'train':
            return self.get_item(index, novel_id=self.opt.train_novel_id)
        elif self.phase == 'val' or self.phase == 'test':
            return self.get_item(index, novel_id=self.opt.val_novel_id)


    def __len__(self):

        self.train_boost = 50
        self.val_boost = 1
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)
