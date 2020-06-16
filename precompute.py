import argparse
import os

import torch
import numpy as np
import scipy.io
import cv2

from torch.utils.data import DataLoader

import dataio
import data_util

import network
import camera
import sph_harm
import render


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', required=True,
                    help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
parser.add_argument('--obj_fp', type=str, default='_/mesh.obj', required=False,
                    help='File name of obj mesh')
parser.add_argument('--calib_fp', type=str, default='_/calib.mat', required=False,
                    help='File name of calibration file')
parser.add_argument('--calib_format', type=str, default='convert', required=False,
                    help='Format of calibration file')
parser.add_argument('--img_dir', type=str, default='_/rgb0', required=False,
                    help='Path to image directory.')
parser.add_argument('--sampling_pattern', type=str, default='all', required=False)
parser.add_argument('--img_size', type=int, default=512,
                    help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='Cuda visible devices.')
parser.add_argument('--only_mesh_related', default=False, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether only compute necessary data related to mesh.')

opt = parser.parse_args()
if opt.obj_fp[:2] == '_/':
    opt.obj_fp = os.path.join(opt.data_root, opt.obj_fp[2:])
if opt.calib_fp[:2] == '_/':
    opt.calib_fp = os.path.join(opt.data_root, opt.calib_fp[2:])
if opt.img_dir[:2] == '_/':
    opt.img_dir = os.path.join(opt.data_root, opt.img_dir[2:])
obj_name = opt.obj_fp.split('/')[-1].split('.')[0]

print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda')

# load global_RT
if opt.calib_format == 'convert':
    global_RT = torch.from_numpy(scipy.io.loadmat(opt.calib_fp)['global_RT'].astype(np.float32))
else:
    global_RT = None

rasterizer = network.Rasterizer(obj_fp = opt.obj_fp, 
                                img_size = opt.img_size,
                                global_RT = global_RT)
rasterizer.to(device)
rasterizer.eval()


def main():
    # dataset loader for view data
    view_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                    img_dir = opt.img_dir,
                                    calib_path = opt.calib_fp,
                                    calib_format = opt.calib_format,
                                    img_size = [opt.img_size, opt.img_size],
                                    sampling_pattern = opt.sampling_pattern,
                                    ignore_dist_coeffs = True,
                                    load_precompute = False,
                                    )
    print('Start buffering view data...')
    view_dataset.buffer_all()
    view_dataloader = DataLoader(view_dataset, batch_size = 1, shuffle = False, num_workers = 8)

    # set up save directories
    save_dir_raster = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'raster')
    if not opt.only_mesh_related:
        save_dir_pose = os.path.join(opt.data_root, 'precomp_' + obj_name, 'pose')
        save_dir_proj = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'proj')
        save_dir_img_gt = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'img_gt')
        save_dir_uv_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'uv_map')
        save_dir_uv_map_preview = os.path.join(save_dir_uv_map, 'preview')
        save_dir_alpha_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'alpha_map')
        save_dir_position_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'position_map')
        save_dir_position_map_cam = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'position_map_cam')
        save_dir_normal_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'normal_map')
        save_dir_normal_map_preview = os.path.join(save_dir_normal_map, 'preview')
        save_dir_normal_map_cam = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'normal_map_cam')
        save_dir_normal_map_cam_preview = os.path.join(save_dir_normal_map_cam, 'preview')
        save_dir_view_dir_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'view_dir_map')
        save_dir_view_dir_map_preview = os.path.join(save_dir_view_dir_map, 'preview')
        save_dir_view_dir_map_cam = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'view_dir_map_cam')
        save_dir_view_dir_map_cam_preview = os.path.join(save_dir_view_dir_map_cam, 'preview')
        save_dir_view_dir_map_tangent = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'view_dir_map_tangent')
        save_dir_view_dir_map_tangent_preview = os.path.join(save_dir_view_dir_map_tangent, 'preview')
        save_dir_sh_basis_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'sh_basis_map')
        save_dir_reflect_dir_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'reflect_dir_map')
        save_dir_reflect_dir_map_preview = os.path.join(save_dir_reflect_dir_map, 'preview')
        save_dir_reflect_dir_map_cam = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'reflect_dir_map_cam')
        save_dir_reflect_dir_map_cam_preview = os.path.join(save_dir_reflect_dir_map_cam, 'preview')
        save_dir_TBN_map = os.path.join(opt.data_root, 'precomp_' + obj_name, 'resol_' + str(opt.img_size), 'TBN_map')
        save_dir_TBN_map_preview = os.path.join(save_dir_TBN_map, 'preview')

    data_util.cond_mkdir(save_dir_raster)
    if not opt.only_mesh_related:
        data_util.cond_mkdir(save_dir_pose)
        data_util.cond_mkdir(save_dir_proj)
        data_util.cond_mkdir(save_dir_img_gt)
        data_util.cond_mkdir(save_dir_uv_map)
        data_util.cond_mkdir(save_dir_uv_map_preview)
        data_util.cond_mkdir(save_dir_alpha_map)
        data_util.cond_mkdir(save_dir_position_map)
        data_util.cond_mkdir(save_dir_position_map_cam)
        data_util.cond_mkdir(save_dir_normal_map)
        data_util.cond_mkdir(save_dir_normal_map_preview)
        data_util.cond_mkdir(save_dir_normal_map_cam)
        data_util.cond_mkdir(save_dir_normal_map_cam_preview)
        data_util.cond_mkdir(save_dir_view_dir_map)
        data_util.cond_mkdir(save_dir_view_dir_map_preview)
        data_util.cond_mkdir(save_dir_view_dir_map_cam)
        data_util.cond_mkdir(save_dir_view_dir_map_cam_preview)
        data_util.cond_mkdir(save_dir_view_dir_map_tangent)
        data_util.cond_mkdir(save_dir_view_dir_map_tangent_preview)
        data_util.cond_mkdir(save_dir_sh_basis_map)
        data_util.cond_mkdir(save_dir_reflect_dir_map)
        data_util.cond_mkdir(save_dir_reflect_dir_map_preview)
        data_util.cond_mkdir(save_dir_reflect_dir_map_cam)
        data_util.cond_mkdir(save_dir_reflect_dir_map_cam_preview)
        data_util.cond_mkdir(save_dir_TBN_map)
        data_util.cond_mkdir(save_dir_TBN_map_preview)

    print('Precompute view-related data...')
    for view_trgt in view_dataloader:
        img_gt = view_trgt[0]['img_gt'][0, :].permute((1, 2, 0)).cpu().detach().numpy() * 255.0
        proj_orig = view_trgt[0]['proj_orig'].to(device)
        proj = view_trgt[0]['proj'].to(device)
        proj_inv = view_trgt[0]['proj_inv'].to(device)
        R_inv = view_trgt[0]['R_inv'].to(device)
        pose = view_trgt[0]['pose'].to(device)
        T = view_trgt[0]['pose'][:, :3, -1].to(device)
        img_fn = view_trgt[0]['img_fn'][0].split('.')[0]

        # rasterize
        uv_map, alpha_map, face_index_map, weight_map, faces_v_idx, normal_map, normal_map_cam, faces_v, faces_vt, position_map, position_map_cam, depth, v_uvz, v_front_mask = \
            rasterizer(proj = proj, 
                        pose = pose, 
                        dist_coeffs = view_trgt[0]['dist_coeffs'].to(device), 
                        offset = None,
                        scale = None,
                        )

        # save raster data
        scipy.io.savemat(os.path.join(save_dir_raster, img_fn + '.mat'),
                         {'face_index_map': face_index_map[0, :].cpu().detach().numpy(),
                          'weight_map': weight_map[0, :].cpu().detach().numpy(),
                          'faces_v_idx': faces_v_idx[0, :].cpu().detach().numpy(),
                          'v_uvz': v_uvz[0, :].cpu().detach().numpy(),
                          'v_front_mask': v_front_mask[0, :].cpu().detach().numpy()})

        if not opt.only_mesh_related:
            # save img_gt
            cv2.imwrite(os.path.join(save_dir_img_gt, img_fn + '.png'), img_gt[:, :, ::-1])
            
            # compute TBN_map
            TBN_map = render.get_TBN_map(normal_map, face_index_map, faces_v = faces_v[0, :], faces_texcoord = faces_vt[0, :], tangent = None)
            # save TBN_map
            scipy.io.savemat(os.path.join(save_dir_TBN_map, img_fn + '.mat'), {'TBN_map': TBN_map[0, :].cpu().detach().numpy()})
            # save preview
            cv2.imwrite(os.path.join(save_dir_TBN_map_preview, img_fn + '_0.png'), (TBN_map[0, ..., 0].cpu().detach().numpy()[:, :, ::-1] * 0.5 + 0.5) * 255)
            cv2.imwrite(os.path.join(save_dir_TBN_map_preview, img_fn + '_1.png'), (TBN_map[0, ..., 1].cpu().detach().numpy()[:, :, ::-1] * 0.5 + 0.5) * 255)
            cv2.imwrite(os.path.join(save_dir_TBN_map_preview, img_fn + '_2.png'), (TBN_map[0, ..., 2].cpu().detach().numpy()[:, :, ::-1] * 0.5 + 0.5) * 255)
            
            # removed padded regions
            alpha_map = alpha_map * torch.from_numpy(img_gt[:, :, 0] <= (2.0 * 255)).to(alpha_map.dtype).to(alpha_map.device)

            uv_map = uv_map.cpu().detach().numpy()
            alpha_map = alpha_map.cpu().detach().numpy()
            normal_map = normal_map.cpu().detach().numpy()
            normal_map_cam = normal_map_cam.cpu().detach().numpy()
            position_map = position_map.cpu().detach().numpy()
            position_map_cam = position_map_cam.cpu().detach().numpy()
            depth = depth.cpu().detach().numpy()

            # save pose, proj_orig
            scipy.io.savemat(os.path.join(save_dir_pose, img_fn + '.mat'), {'pose': pose[0, :].cpu().detach().numpy(), 'proj_orig': proj_orig[0, :].cpu().detach().numpy()})
            # save proj
            scipy.io.savemat(os.path.join(save_dir_proj, img_fn + '.mat'), {'proj': proj[0, :].cpu().detach().numpy()})

            # save uv_map
            scipy.io.savemat(os.path.join(save_dir_uv_map, img_fn + '.mat'), {'uv_map': uv_map[0, :]})
            # save uv_map preview
            uv_map_img = np.concatenate((uv_map[0, :, :, :], np.zeros((*uv_map.shape[1:3], 1))), axis = 2)
            cv2.imwrite(os.path.join(save_dir_uv_map_preview, img_fn + '.png'), uv_map_img[:, :, ::-1] * 255)
            # save alpha_map
            cv2.imwrite(os.path.join(save_dir_alpha_map, img_fn + '.png'), alpha_map[0, :] * 255)
            # save normal_map
            scipy.io.savemat(os.path.join(save_dir_normal_map, img_fn + '.mat'), {'normal_map': normal_map[0, :]})
            normal_map_cam = normal_map_cam * np.array([1, -1, -1], dtype = np.float32)[None, None, None, :] # change to z-out space
            scipy.io.savemat(os.path.join(save_dir_normal_map_cam, img_fn + '.mat'), {'normal_map_cam': normal_map_cam[0, :]})
            # save normal_map preview
            normal_map_img = (normal_map[0, :, :, :] + 1.0) / 2
            normal_map_cam_img = (normal_map_cam[0, :, :, :] + 1.0) / 2
            cv2.imwrite(os.path.join(save_dir_normal_map_preview, img_fn + '.png'), normal_map_img[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(save_dir_normal_map_cam_preview, img_fn + '.png'), normal_map_cam_img[:, :, ::-1] * 255)
            # save position map
            scipy.io.savemat(os.path.join(save_dir_position_map, img_fn + '.mat'), {'position_map': position_map[0, :]})
            scipy.io.savemat(os.path.join(save_dir_position_map_cam, img_fn + '.mat'), {'position_map_cam': position_map_cam[0, :]})

            # compute view_dir_map
            view_dir_map, view_dir_map_cam = camera.get_view_dir_map(img_gt.shape[:2], proj_inv, R_inv)
            view_dir_map_cam = view_dir_map_cam.cpu().detach().numpy()
            view_dir_map_cam = view_dir_map_cam * np.array([1, -1, -1], dtype = np.float32)[None, None, None, :] # change to z-out space
            # save view_dir_map
            scipy.io.savemat(os.path.join(save_dir_view_dir_map, img_fn + '.mat'), {'view_dir_map': view_dir_map.cpu().detach().numpy()[0, :]})
            scipy.io.savemat(os.path.join(save_dir_view_dir_map_cam, img_fn + '.mat'), {'view_dir_map_cam': view_dir_map_cam[0, :]})
            # save view_dir_map preview
            view_dir_map_img = (view_dir_map.cpu().detach().numpy()[0, :, :, :] + 1.0) / 2
            view_dir_map_cam_img = (view_dir_map_cam[0, :, :, :] + 1.0) / 2
            cv2.imwrite(os.path.join(save_dir_view_dir_map_preview, img_fn + '.png'), view_dir_map_img[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(save_dir_view_dir_map_cam_preview, img_fn + '.png'), view_dir_map_cam_img[:, :, ::-1] * 255)

            # compute view_dir_map in tangent space
            view_dir_map_tangent = torch.matmul(TBN_map.reshape((-1, 3, 3)).transpose(-2, -1), view_dir_map.reshape((-1, 3, 1)))[..., 0].reshape(view_dir_map.shape)
            view_dir_map_tangent = torch.nn.functional.normalize(view_dir_map_tangent, dim = -1)
            # save view_dir_map_tangent
            scipy.io.savemat(os.path.join(save_dir_view_dir_map_tangent, img_fn + '.mat'), {'view_dir_map_tangent': view_dir_map_tangent.cpu().detach().numpy()[0, :]})
            # save preview
            view_dir_map_tangent_img = (view_dir_map_tangent.cpu().detach().numpy()[0, :, :, :] + 1.0) / 2
            cv2.imwrite(os.path.join(save_dir_view_dir_map_tangent_preview, img_fn + '.png'), view_dir_map_tangent_img[:, :, ::-1] * 255)

            # SH basis value for view_dir_map
            sh_basis_map = sph_harm.evaluate_sh_basis(lmax = 2, directions = view_dir_map.reshape((-1, 3)).cpu().detach().numpy()).reshape((*(view_dir_map.shape[:3]), -1)).astype(np.float32) # [N, H, W, 9]
            # save
            scipy.io.savemat(os.path.join(save_dir_sh_basis_map, img_fn + '.mat'), {'sh_basis_map': sh_basis_map[0, :]})

            # compute reflect_dir_map
            reflect_dir_map = camera.get_reflect_dir(view_dir_map.to(device), torch.from_numpy(normal_map).to(device)).cpu().detach().numpy() * alpha_map[..., None]
            reflect_dir_map_cam = camera.get_reflect_dir(torch.from_numpy(view_dir_map_cam).to(device), torch.from_numpy(normal_map_cam).to(device)).cpu().detach().numpy() * alpha_map[..., None]
            # save reflect_dir_map
            scipy.io.savemat(os.path.join(save_dir_reflect_dir_map, img_fn + '.mat'), {'reflect_dir_map': reflect_dir_map[0, :]}) # [H, W, 3]
            scipy.io.savemat(os.path.join(save_dir_reflect_dir_map_cam, img_fn + '.mat'), {'reflect_dir_map_cam': reflect_dir_map_cam[0, :]})
            # save reflect_dir_map preview
            reflect_dir_map_img = (reflect_dir_map[0, :, :, :] + 1.0) / 2
            reflect_dir_map_cam_img = (reflect_dir_map_cam[0, :, :, :] + 1.0) / 2
            cv2.imwrite(os.path.join(save_dir_reflect_dir_map_preview, img_fn + '.png'), reflect_dir_map_img[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(save_dir_reflect_dir_map_cam_preview, img_fn + '.png'), reflect_dir_map_cam_img[:, :, ::-1] * 255)
        
        idx = view_trgt[0]['idx'].cpu().detach().numpy().item()
        if not idx % 10:
            print('View', idx)


if __name__ == '__main__':
    main()