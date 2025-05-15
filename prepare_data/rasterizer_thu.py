# Imports
import os
import matplotlib.pyplot as plt
import ipdb
import numpy as np
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer
)
import tqdm
from pytorch3d.renderer import (
    AlphaCompositor,
    NDCMultinomialRaysampler,
    PointsRasterizationSettings,
    PointsRasterizer,
    ray_bundle_to_ray_points,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection
import pytorch3d.implicitron.tools.point_cloud_utils as pcd_util
import cv2
import pickle
from scipy.spatial.transform import Rotation
import smplx as smplx

device = torch.device('cuda:0')


def render_thu_depth(data_path):
    cam_path = data_path + 'parm'
    smpl_path = data_path + 'smpl'
    file_lst = os.listdir(cam_path)
    depth_path = data_path + 'smpl_depth'
    os.makedirs(depth_path, exist_ok=True)
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=-1
    )
    smpl = smplx.SMPL(os.path.join(os.getcwd(), 'smpl'))

    face = torch.from_numpy(smpl.faces.astype(np.int32)).cuda()

    for file in tqdm.tqdm(file_lst):
        os.makedirs(os.path.join(depth_path, file), exist_ok=True)
        vertex = torch.from_numpy(np.load(os.path.join(smpl_path, file.split('_')[0], 'vertices.npy'))).cuda()
        # face=torch.from_numpy(np.load(os.path.join(smpl_path, file.split('_')[0], 'faces.npy')).astype(np.int32)).cuda()
        # cam
        for i in range(1):
            extr = torch.from_numpy(np.load(os.path.join(cam_path, file, f'{i}_extrinsic.npy'))).to(
                torch.float32).cuda()
            intr = torch.from_numpy(np.load(os.path.join(cam_path, file, f'{i}_intrinsic.npy'))).to(
                torch.float32).cuda()
            cam = _cameras_from_opencv_projection(extr[:3, :3][None], extr[:3, 3][None], intr[None],
                                                  torch.tensor([[1024, 1024]]))
            rasterizer = MeshRasterizer(
                cameras=cam,
                raster_settings=raster_settings
            )

            meshes = Meshes(vertex[None].to(torch.float32), face[None])
            fragments = rasterizer(meshes)

            depth = fragments.zbuf[0]
            depth = 1 / depth
            depth[depth == -1] = 0
            cv2.imwrite(os.path.join(depth_path, file, f'{i}.jpg'), depth.detach().cpu().numpy() * 255)
            # np.save(os.path.join(depth_path, file, f'{i}.npy'), depth.detach().cpu().numpy())


def world2smpl(xyz, param):
    # ipdb.set_trace()
    xyz[..., 1] += param['y_transl']
    xyz[..., :3] = xyz[..., :3] / param['height'] * param['v_scale']
    xyz = (xyz - param['transl']) / param['scale']
    return xyz


def show_pcd(point):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="pcd")
    # 设置点云大小
    vis.get_render_option().point_size = 4
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()

    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(point)
    # 设置点的颜色为白色
    pcd.paint_uniform_color([0.745, 0.723, 0.952])

    # r=weights[:,:8].sum(1)*255
    # g=weights[:,8:16].sum(1)*255
    # b=weights[:,16:].sum(1)*255
    # color=np.stack([r,g,b],axis=-1).astype(np.uint8)
    # pcd.colors=o3d.open3d.utility.Vector3dVector(color)
    vis.add_geometry(pcd)

    vis.run()
    vis.capture_screen_image('e.jpg')
    vis.destroy_window()


with torch.no_grad():
    data_path = 'D:/dataset/Thuman2/render/train/'
    render_thu_depth(data_path)
    data_path = 'D:/dataset/Thuman2/render/train/'
    # # disturb_smpl(data_path)
    render_thu_depth(data_path)

