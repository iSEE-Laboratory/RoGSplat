import ipdb
import torch

import smplx as smplx
import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import pickle
import trimesh

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def save(pid, data_id, vid, save_path, extr, intr, depth, img, mask, img_hr=None):
    img_save_path = os.path.join(save_path, 'img', data_id + '_' + '%03d' % pid)
    depth_save_path = os.path.join(save_path, 'depth', data_id + '_' + '%03d' % pid)
    mask_save_path = os.path.join(save_path, 'mask', data_id + '_' + '%03d' % pid)
    parm_save_path = os.path.join(save_path, 'parm', data_id + '_' + '%03d' % pid)
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(parm_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    depth = depth * 2.0 ** 15
    cv2.imwrite(os.path.join(depth_save_path, '{}.png'.format(vid)), depth.astype(np.uint16))
    img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
    mask = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(img_save_path, '{}.jpg'.format(vid)), img)
    if img_hr is not None:
        img_hr = (np.clip(img_hr, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(img_save_path, '{}_hr.jpg'.format(vid)), img_hr)
    cv2.imwrite(os.path.join(mask_save_path, '{}.png'.format(vid)), mask)
    np.save(os.path.join(parm_save_path, '{}_intrinsic.npy'.format(vid)), intr)
    np.save(os.path.join(parm_save_path, '{}_extrinsic.npy'.format(vid)), extr)


class StaticRenderer:
    def __init__(self):
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        self.scene = t3.Scene()

        self.smpl = smplx.SMPL(os.path.join(os.getcwd(), 'smpl'))
        self.N = 10

    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        print('init')
        self.scene = t3.Scene()

        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()

    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)

    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()

    def camera_light(self):
        camera = t3.Camera(res=(1024, 1024))
        self.scene.add_camera(camera)

        camera_hr = t3.Camera(res=(2048, 2048))
        self.scene.add_camera(camera_hr)

        light_dir = np.array([0, 0, 1])
        light_list = []
        for l in range(6):
            rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                               rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            light_list.append(light)
        lights = t3.Lights(light_list)
        self.scene.add_lights(lights)


def render_data(renderer, data_path, phase, data_id, save_path, cam_nums, res, dis=1.0, is_thuman=False):
    obj_path = os.path.join(data_path, phase, data_id, '%s.obj' % data_id)
    texture_path = data_path
    img_path = os.path.join(texture_path, phase, data_id, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # # randomly move the scan
    # move_range = 0.1 if human_height < 1.80 else 0.05
    # delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    # delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    # if delta_x > 1.0 or delta_z > 1.0:
    #     move_range = 0.01
    # obj['vi'][:, 0] += np.random.uniform(-move_range, move_range, 1)
    # obj['vi'][:, 2] += np.random.uniform(-move_range, move_range, 1)

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    degree_interval = 360 / cam_nums
    angle_list1 = list(range(int(360 - degree_interval // 2), 360))
    angle_list2 = list(range(0, int(0 + degree_interval // 2)))
    angle_list = angle_list1 + angle_list2
    angle_base = np.random.choice(angle_list, 1)[0]
    if is_thuman:
        # thuman needs a normalization of orientation
        smpl_path = os.path.join(data_path, 'THuman2.0_Smpl_X_Paras', data_id, 'smplx_param.pkl')
        with open(smpl_path, 'rb') as f:
            smpl_para = pickle.load(f)

        y_orient = smpl_para['global_orient'][0][1]
        angle_base += (y_orient * 180.0 / np.pi)

        # resize smplx vertices
        smpl_path = os.path.join(data_path, 'Thuman2.0_smpl', data_id + '_smpl.pkl')

        with open(smpl_path, 'rb') as f:
            smpl_para = pickle.load(f)

        smplx_np = {}
        for key in smpl_para:
            smplx_np[key] = np.array(smpl_para[key], dtype=np.float32)

        output = renderer.smpl(global_orient=torch.from_numpy(smplx_np['global_orient']),
                               betas=torch.from_numpy(smplx_np['betas']),
                               body_pose=torch.from_numpy(smplx_np['body_pose']).flatten(1), pose2rot=True)
        smpl_verts = output.vertices[0].detach().numpy() * smplx_np['scale'] + smplx_np['transl']
        smpl_verts[:, :3] = smpl_verts[:, :3] / (vy_max - vy_min) * human_height
        y_transl = np.min(smpl_verts[:, 1])
        smpl_verts[:, 1] -= y_transl
        face = renderer.smpl.faces_tensor.numpy()
        normal = compute_normal(smpl_verts, face)
        path = os.path.join(save_path, 'smpl', data_id)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'vertices.npy'), smpl_verts)
        np.save(os.path.join(path, 'faces.npy'), face)
        np.save(os.path.join(path, 'normal.npy'), normal)
        smplx_np['v_scale'] = vy_max - vy_min
        smplx_np['height'] = human_height
        smplx_np['y_transl'] = y_transl
        np.save(os.path.join(path, 'param.npy'), smplx_np)

        smplx_obj = trimesh.load(os.path.join(data_path, 'THuman2.0_Smpl_X_Paras', data_id, 'mesh_smplx.obj'))
        smplx_verts = np.array(smplx_obj.vertices)
        smplx_verts[:, :3] = smplx_verts[:, :3] / (vy_max - vy_min) * human_height
        smplx_verts[:, 1] -= np.min(smplx_verts[:, 1])

        face = np.array(smplx_obj.faces)
        normal = compute_normal(smplx_verts, face)
        path = os.path.join(save_path, 'smplx', data_id)
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'vertices.npy'), smplx_verts)
        np.save(os.path.join(path, 'faces.npy'), face)
        np.save(os.path.join(path, 'normal.npy'), normal)

        # mesh = trimesh.Trimesh(smplx_verts, face, vertex_colors=normal * 0.5 + 0.5)
        # mesh.show()

    for pid in range(cam_nums):
        angle = angle_base + pid * degree_interval

        def render(dis, angle, look_at_center, p, renderer, render_2k=False):
            ori_vec = np.array([0, 0, dis])
            rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
            fwd = np.matmul(rotate, ori_vec)
            cam_pos = look_at_center + fwd

            x_min = 0
            y_min = -25
            cx = res[0] * 0.5
            cy = res[1] * 0.5
            fx = res[0] * 0.8
            fy = res[1] * 0.8
            _cx = cx - x_min
            _cy = cy - y_min
            renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
            renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
            renderer.scene.cameras[0]._init()

            if render_2k:
                fx = res[0] * 0.8 * 2
                fy = res[1] * 0.8 * 2
                _cx = (res[0] * 0.5 - x_min) * 2
                _cy = (res[1] * 0.5 - y_min) * 2
                renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
                renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
                renderer.scene.cameras[1]._init()

                renderer.scene.render()
                camera = renderer.scene.cameras[0]
                camera_hr = renderer.scene.cameras[1]
                extrinsic = camera.export_extrinsic()
                intrinsic = camera.export_intrinsic()
                depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
                img = camera.img.to_numpy().swapaxes(0, 1)
                img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
                mask = camera.mask.to_numpy().swapaxes(0, 1)
                return extrinsic, intrinsic, depth_map, img, mask, img_hr

            renderer.scene.render()
            camera = renderer.scene.cameras[0]
            extrinsic = camera.export_extrinsic()
            intrinsic = camera.export_intrinsic()
            depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
            img = camera.img.to_numpy().swapaxes(0, 1)
            mask = camera.mask.to_numpy().swapaxes(0, 1)
            return extrinsic, intrinsic, depth_map, img, mask

        extr, intr, depth, img, mask = render(dis, angle, look_at_center, base_cam_pitch, renderer)
        save(pid, data_id, 0, save_path, extr, intr, depth, img, mask)
        extr, intr, depth, img, mask = render(dis, (angle + degree_interval) % 360, look_at_center, base_cam_pitch,
                                              renderer)
        save(pid, data_id, 1, save_path, extr, intr, depth, img, mask)

        # three novel viewpoints between source views
        angle1 = (angle + (np.random.uniform() * degree_interval / 2)) % 360
        angle2 = (angle + degree_interval / 2) % 360
        angle3 = (angle + degree_interval - (np.random.uniform() * degree_interval / 2)) % 360

        extr, intr, depth, img, mask, img_hr = render(dis, angle1, look_at_center, base_cam_pitch, renderer,
                                                      render_2k=True)
        save(pid, data_id, 2, save_path, extr, intr, depth, img, mask, img_hr)
        extr, intr, depth, img, mask, img_hr = render(dis, angle2, look_at_center, base_cam_pitch, renderer,
                                                      render_2k=True)
        save(pid, data_id, 3, save_path, extr, intr, depth, img, mask, img_hr)
        extr, intr, depth, img, mask, img_hr = render(dis, angle3, look_at_center, base_cam_pitch, renderer,
                                                      render_2k=True)
        save(pid, data_id, 4, save_path, extr, intr, depth, img, mask, img_hr)


if __name__ == '__main__':
    cam_nums = 16
    scene_radius = 2.0
    res = (1024, 1024)
    thuman_root = 'D:/dataset/thuman2'
    save_root = 'D:/dataset/thuman2/render'
    renderer = StaticRenderer()

    for phase in ['train', 'val']:
        thuman_list = sorted(os.listdir(os.path.join(thuman_root, phase)))
        save_path = os.path.join(save_root, phase)

        for data_id in tqdm(thuman_list):
            render_data(renderer, thuman_root, phase, data_id, save_path, cam_nums, res, dis=scene_radius,
                        is_thuman=True)
