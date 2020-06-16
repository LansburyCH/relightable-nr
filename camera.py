import numpy as np
import torch


def get_view_dir_map(img_size, proj_inv, R_inv):
    '''
    Get view direction map in camera space
    img_size: [2, ]
    proj_inv: [N, 3, 3]
    R_inv: [N, 3, 3]
    return0: [N, img_size[0], img_size[1], 3]
    return1: [N, img_size[0], img_size[1], 3]
    '''
    batch_size = proj_inv.shape[0]
    img_h = img_size[0]
    img_w = img_size[1]
    device = proj_inv.device

    v, u = torch.meshgrid([torch.arange(img_h, dtype = torch.float32, device = device) + 0.5, torch.arange(img_w, dtype = torch.float32, device = device) + 0.5]) # [H, W]
    uv = torch.stack((u, v, torch.ones_like(u)), dim = 0).flatten(start_dim = 1) # [3, H * W]

    view_dir_map_cam = torch.zeros((batch_size, img_h, img_w, 3), dtype = torch.float32, device = device)
    view_dir_map = torch.zeros_like(view_dir_map_cam)
    for i in range(batch_size):
        xyz = -torch.matmul(proj_inv[i, :], uv) # [3, H * W]
        xyz = torch.nn.functional.normalize(xyz, dim = 0) # [3, H * W]
        view_dir_map_cam[i, :] = xyz.reshape(3, img_h, img_w).permute((1, 2, 0))
        view_dir_map[i, :] = torch.matmul(R_inv[i, :], xyz).reshape(3, img_h, img_w).permute((1, 2, 0))

    view_dir_map = torch.nn.functional.normalize(view_dir_map, dim = -1)

    return view_dir_map, view_dir_map_cam


def get_reflect_dir(orig_dir, pivot_dir, dim = -1):
    '''
    Reflect directions according to pivot directions
    orig_dir: [..., 3, ...]
    pivot_dir: [..., 3, ...]
    dim: int, the dimension corresponding to direction
    return: [..., 3, ...]
    '''
    reflect_dir = torch.nn.functional.normalize((pivot_dir * orig_dir).sum(dim = dim, keepdim = True) * 2.0 * pivot_dir - orig_dir, dim = dim)

    return reflect_dir


def RT_from_pos_lookat(cam_pos, cam_lookat = np.array([0., 0., 0.]), cam_up = np.array([0., 1., 0.])):
    '''
    cam_pos: [3]
    cam_lookat: [3]
    cam_up: [3]
    return: [4, 4]
    '''
    cam_forward = cam_lookat - cam_pos
    cam_forward = cam_forward / np.linalg.norm(cam_forward)
    cam_right = np.cross(cam_forward, cam_up)
    cam_right = cam_right /np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_forward)

    R = np.eye(3).astype(cam_pos.dtype)
    R[0, :] = cam_right
    R[1, :] = -cam_up
    R[2, :] = cam_forward
    T = -R.dot(cam_pos[:, None])
    RT = np.hstack((R, T))
    RT = np.vstack((RT, np.array([0, 0, 0, 1])))

    return RT


def get_spiral(step_azi = -2, step_ele = 90.0 / 720):
    num_step = int(np.floor(90.0 / step_ele))
    cam_pos_azi = np.arange(0, step_azi * num_step, step = step_azi)
    cam_pos_ele = np.arange(0, step_ele * num_step, step = step_ele)
    return cam_pos_azi, cam_pos_ele