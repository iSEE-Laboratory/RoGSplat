import json
import math
import ipdb
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm


def read_img(name):
    img = np.array(Image.open(name))
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15


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
    import cv2

    img = img * 255
    img = img.astype(np.uint8)
    # ipdb.set_trace()
    new_img=np.zeros_like(img)
    for c in uv:
        new_img[c[1],c[0]]=img[c[1],c[0]]
        # cv.circle(img, c[:2], 1, (0, 255, 255), 1)

    err_img=np.sum((new_img-img)**2,axis=2)
    err_img=np.sqrt(err_img)
    err_img=err_img/np.max(err_img)*255
    err_img=err_img.astype(np.uint8)
    err_img = cv2.applyColorMap(err_img, cv2.COLORMAP_JET)
    cv2.imwrite('a.jpg', img[:,:,[2,1,0]])
    cv2.imwrite('b.jpg',new_img[:,:,[2,1,0]])
    cv2.imwrite('err.jpg',err_img)
    # ipdb.set_trace()


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def depth2pcN(depth, extrinsic, intrinsic):
    B, C, S, S, N = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S - 0.5, S, device=depth.device),
                          torch.linspace(0.5, S - 0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, 1, N)  # B S S 3

    pts_2d[..., 2, :] = depth
    pts_2d[:, :, :, 0, :] -= intrinsic[:, None, None, 0, 2, None]
    pts_2d[:, :, :, 1, :] -= intrinsic[:, None, None, 1, 2, None]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:, :]], dim=-2)

    pts_2d[..., 0, :] /= intrinsic[:, 0, 0][:, None, None, None]
    pts_2d[..., 1, :] /= intrinsic[:, 1, 1][:, None, None, None]

    pts_2d = pts_2d.permute(0, 3, 1, 2, 4).reshape(B, 3, -1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S - 0.5, S, device=depth.device),
                          torch.linspace(0.5, S - 0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def get_bound_2d_mask(bounds, intr, extr, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, extr, intr)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1)  # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


class DepthNormalizerBase:
    is_relative = None
    far_plane_at_max = None

    def __init__(
            self,
            norm_min=-1.0,
            norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError


class NearFarMetricNormalizer(DepthNormalizerBase):
    """
    depth in [0, d_max] -> [-1, 1]
    """

    is_relative = True
    far_plane_at_max = True

    def __init__(
            self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        # Take quantiles as min and max
        _min, _max = torch.quantile(
            depth_linear[valid_mask],
            torch.tensor([self.min_quantile, self.max_quantile]).to(depth_linear.device),
        )

        # scale and shift
        depth_norm_linear = (depth_linear - _min) / (
                _max - _min
        ) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        return self.scale_back(depth_norm=depth_norm)


def get_bound(pts):
    min=pts.min(1)[0]-0.05
    max=pts.max(1)[0]+0.05
    bound=torch.cat([min,max],dim=0)
    return bound[None]

def get_voxel_coord(pts, bound, vox_size):
    min_xyz = bound[..., 0, :]
    max_xyz = bound[..., 1, :]
    dhw = pts[..., [2, 1, 0]]
    min_dhw = min_xyz[..., [2, 1, 0]]
    max_dhw = max_xyz[..., [2, 1, 0]]
    voxel_size = torch.tensor([vox_size, vox_size, vox_size]).to(pts.device)
    coord = torch.round((dhw - min_dhw) / voxel_size).type(torch.int32)
    world_coord = coord * voxel_size + min_dhw
    world_coord = world_coord[..., [2, 1, 0]]
    sh = coord.shape
    idx = [torch.full([sh[1]], i) for i in range(sh[0])]
    idx = torch.cat(idx).to(coord)
    coord = coord.view(-1, sh[-1])
    coord = torch.cat([idx[:, None], coord], dim=1)

    out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).type(torch.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    return coord, out_sh.view(-1).tolist(), world_coord


def get_world_anchor(bound, vox_size):
    min_xyz = bound[..., 0, :]
    max_xyz = bound[..., 1, :]
    min_dhw = min_xyz[..., [2, 1, 0]]
    max_dhw = max_xyz[..., [2, 1, 0]]
    voxel_size = torch.tensor([vox_size, vox_size, vox_size]).to(bound.device)
    out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).type(torch.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    size = 2 / out_sh
    size = size.view(-1).tolist()
    import ipdb
    ipdb.set_trace()

    coord_d = torch.arange(-1 + size[0] / 2, 1, size[0]).to(bound.device)
    coord_h = torch.arange(-1 + size[1] / 2, 1, size[1]).to(bound.device)
    coord_w = torch.arange(-1 + size[2] / 2, 1, size[2]).to(bound.device)
    d, h, w = torch.meshgrid(coord_d, coord_h, coord_w)
    dhw = torch.stack([d, h, w], dim=-1).flatten(0, 2)
    ipdb.set_trace()
    coord = torch.round((max_dhw - min_dhw) / voxel_size).type(torch.int32)
    world_coord = coord * voxel_size + min_dhw
    world_coord = world_coord[..., [2, 1, 0]]

    return world_coord


def get_grid_coord(pts, bound, sh, vox_size):
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]

    min_dhw = bound[:, 0, [2, 1, 0]]
    dhw = dhw - min_dhw[:, None]
    dhw = dhw / torch.tensor(vox_size).to(dhw)
    # convert the voxel coordinate to [-1, 1]
    out_sh = torch.tensor(sh).to(dhw)

    dhw = dhw / out_sh * 2 - 1
    # convert dhw to whd, since the occupancy is indexed by dhw
    grid_coords = dhw[..., [2, 1, 0]]
    return grid_coords[:, None, None]

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]



def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])