import argparse
import os, time

import torch
from torch.utils.data import DataLoader

import numpy as np
import cv2
import scipy.io
from collections import OrderedDict

import dataio
import data_util
import util

import network
import render
import camera
import sph_harm


parser = argparse.ArgumentParser()

# lighting
parser.add_argument('--lp_dir', type=str, default='_/light_probe', required=False,
                    help='Path to directory that holds the light probe data.')
parser.add_argument('--lighting_type', type = str, default='SH', required=False,
                    help='Lighting type.')
parser.add_argument('--sh_lmax', type = int, default=10, required=False,
                    help='Maximum degrees of SH basis for lighting.')
parser.add_argument('--lighting_idx', default = 0, type = int,
                    help='Lighting index for inference.')
# inference sequence
parser.add_argument('--img_size', type=int, default=512,
                    help='Sidelength of generated images.')
parser.add_argument('--calib_dir', type=str, required=True,
                    help='Path of calibration file for inference sequence.')
parser.add_argument('--sampling_pattern', type=str, default='all', required=False)
# checkpoint
parser.add_argument('--checkpoint_dir', required=True,
                    help='Path to a checkpoint to load render_net weights from.')
parser.add_argument('--checkpoint_name', required=True,
                    help='Path to a checkpoint to load render_net weights from.')
# misc
parser.add_argument('--gpu_id', type=str, default='',
                    help='Cuda visible devices.')
parser.add_argument('--save_img_bg', default = True, type = lambda x: (str(x).lower() in ['true', '1']))
parser.add_argument('--force_recompute', default = False, type = lambda x: (str(x).lower() in ['true', '1']))

opt = parser.parse_args()
checkpoint_fp = os.path.join(opt.checkpoint_dir, opt.checkpoint_name)

# load params from checkpoint
params_fp = os.path.join(opt.checkpoint_dir, 'params.txt')
params_file = open(params_fp, "r")
params_lines = params_file.readlines()
params = {}
for line in params_lines:
    key = line.split(':')[0]
    val = line.split(':')[1]
    if len(val) > 1:
        val = val[1:-1]
    else:
        val = None
    params[key] = val
# general
opt.data_root = params['data_root']
# mesh
if not hasattr(opt, 'obj_high_fp'):
    opt.obj_high_fp = params['obj_high_fp']
opt.obj_low_fp = params['obj_low_fp']
# texture mapper
opt.texture_size = int(params['texture_size'])
opt.texture_num_ch = int(params['texture_num_ch'])
opt.mipmap_level = int(params['mipmap_level'])
opt.apply_sh = params['apply_sh'] in ['True', 'true', '1']
# lighting
opt.sphere_samples_fp = params['sphere_samples_fp']
# rendering net
opt.nf0 = int(params['nf0'])

if opt.calib_dir[:2] == '_/':
    opt.calib_dir = os.path.join(opt.data_root, opt.calib_dir[2:])
if opt.obj_high_fp[:2] == '_/':
    opt.obj_high_fp = os.path.join(opt.data_root, opt.obj_high_fp[2:])
if opt.obj_low_fp[:2] == '_/':
    opt.obj_low_fp = os.path.join(opt.data_root, opt.obj_low_fp[2:])
if opt.lp_dir[:2] == '_/':
    opt.lp_dir = os.path.join(opt.data_root, opt.lp_dir[2:])
if opt.sphere_samples_fp[:2] == '_/':
    opt.sphere_samples_fp = os.path.join(opt.data_root, opt.sphere_samples_fp[2:])
obj_high_name = opt.obj_high_fp.split('/')[-1].split('.')[0]

print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

if opt.gpu_id == '':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + opt.gpu_id)


# load global_RT
opt.calib_fp = os.path.join(opt.calib_dir, 'calib.mat')
global_RT = torch.from_numpy(scipy.io.loadmat(opt.calib_fp)['global_RT'].astype(np.float32))

num_channel = 3

# sample light directions on sphere
l_dir_np = scipy.io.loadmat(opt.sphere_samples_fp)['sphere_samples'].transpose() # [3, num_sample]
l_dir = torch.from_numpy(l_dir_np) # [3, num_sample]

# dataset for inference views
view_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                calib_path = opt.calib_fp,
                                calib_format = 'convert',
                                img_size = [opt.img_size, opt.img_size],
                                sampling_pattern = opt.sampling_pattern,
                                load_img = False,
                                load_precompute = False
                                )
num_view = len(view_dataset)

# dataset loader for light probes
lp_dataset = dataio.LightProbeDataset(data_dir = opt.lp_dir)
print('Start buffering light probe data...')
lp_dataset.buffer_all()
lp_dataloader = DataLoader(lp_dataset, batch_size = 1, shuffle = False, num_workers = 8)

# load mesh
mesh = network.Mesh(opt.obj_low_fp, global_RT = global_RT)
num_vertex = mesh.num_vertex

# interpolater
interpolater = network.Interpolater()

# texture mapper
texture_mapper = network.TextureMapper(texture_size = opt.texture_size,
                                        texture_num_ch = opt.texture_num_ch,
                                        mipmap_level = opt.mipmap_level,
                                        texture_init = None,
                                        fix_texture = True,
                                        apply_sh = opt.apply_sh)

# load checkpoint
checkpoint_dict = util.custom_load([texture_mapper], ['texture_mapper'], checkpoint_fp, strict = False)

# trained lighting model
new_state_dict = OrderedDict()
for k, v in checkpoint_dict['lighting_model'].items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
lighting_model_train = network.LightingSH(l_dir, lmax = int(params['sh_lmax']), num_lighting = 2, num_channel = num_channel, fix_params = True)
lighting_model_train.coeff.data = new_state_dict['coeff']
lighting_model_train.l_samples.data = new_state_dict['l_samples']

# lighting model lp
lighting_model_lp = network.LightingLP(l_dir, num_channel = num_channel, lp_dataloader = lp_dataloader, fix_params = True)
lighting_model_lp.fit_sh(lmax = opt.sh_lmax)

# lighting model sh
lighting_model_sh = network.LightingSH(l_dir, lmax = opt.sh_lmax, num_lighting = lighting_model_lp.num_lighting, num_channel = num_channel, init_coeff = lighting_model_lp.sh_coeff, fix_params = True)

# choose which lighting model to use
if opt.lighting_type == 'SH':
    lighting_model = lighting_model_sh
elif opt.lighting_type == 'train':
    lighting_model = lighting_model_train
else:
    raise ValueError('Unrecognized lighting type')

# ray sampler specular
new_state_dict = OrderedDict()
for k, v in checkpoint_dict['ray_sampler'].items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
ray_sampler = network.RaySampler(num_azi = new_state_dict['num_azi'].cpu().detach().numpy(), 
                                num_polar = new_state_dict['num_polar'].cpu().detach().numpy(), 
                                interval_polar = new_state_dict['interval_polar'].cpu().detach().numpy())
ray_sampler.load_state_dict(new_state_dict, strict = False)
num_ray = ray_sampler.num_ray

# ray sampler diffuse
new_state_dict = OrderedDict()
for k, v in checkpoint_dict['ray_sampler_diffuse'].items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
ray_sampler_diffuse = network.RaySampler(num_azi = new_state_dict['num_azi'].cpu().detach().numpy(), 
                                num_polar = new_state_dict['num_polar'].cpu().detach().numpy(), 
                                interval_polar = new_state_dict['interval_polar'].cpu().detach().numpy(),
                                mode = 'diffuse')
ray_sampler_diffuse.load_state_dict(new_state_dict, strict = False)
num_ray_diffuse = ray_sampler_diffuse.num_ray

num_ray_total = num_ray + num_ray_diffuse

# rendering net
new_state_dict = OrderedDict()
for k, v in checkpoint_dict['render_net'].items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
render_net = network.RenderingNet(nf0 = opt.nf0,
                                    in_channels = num_ray_total * 3 + 6 + opt.texture_num_ch,
                                    out_channels = 3 * num_ray_total,
                                    num_down_unet = 5,
                                    out_channels_gcn = int(params['out_channels_gcn'])
                                    )
render_net.load_state_dict(new_state_dict, strict = False)

# gcn output
v_feature = checkpoint_dict['v_feature']

# clear checkpoint content to save memory
checkpoint_dict = None

# ray renderer
ray_renderer = network.RayRenderer(lighting_model, interpolater)

# rasterizer
rasterizer = network.Rasterizer(obj_fp = opt.obj_high_fp, img_size = opt.img_size, global_RT = global_RT)

# move to device
interpolater.to(device)
texture_mapper.to(device)
mesh.to(device)
lighting_model.to(device)
ray_sampler.to(device)
ray_sampler_diffuse.to(device)
render_net.to(device)
v_feature = v_feature.to(device)
ray_renderer.to(device)
rasterizer.cuda(0) # currently, rasterizer can only be put on gpu 0

# set to inference mode
interpolater.eval()
texture_mapper.eval()
lighting_model.eval()
ray_sampler.eval()
ray_sampler_diffuse.eval()
render_net.eval()
ray_renderer.eval()
rasterizer.eval()

def set_bn_train(m):
    if type(m) == torch.nn.BatchNorm2d:
        m.train()

render_net.apply(set_bn_train)


def main():
    view_dataset.buffer_all()

    if opt.lighting_type == 'train':
        lighting_idx_all = [int(params['lighting_idx'])]
    else:
        lighting_idx_all = [opt.lighting_idx]
    
    log_dir = opt.checkpoint_dir.split('/')
    log_dir = os.path.join(opt.calib_dir, 'resol_' + str(opt.img_size), log_dir[-2], log_dir[-1].split('_')[0] + '_' + log_dir[-1].split('_')[1] + '_' + opt.checkpoint_name.split('-')[-1].split('.')[0])
    data_util.cond_mkdir(log_dir)

    # get estimated illumination
    lp_est = lighting_model_train.to(device)
    lp_est = lp_est(lighting_idx = int(params['lighting_idx']), is_lp = True)
    cv2.imwrite(log_dir + '/lp_est.png', lp_est.cpu().detach().numpy()[0, :, :, ::-1] * 255.0)

    save_dir_alpha_map = os.path.join(log_dir, 'alpha_map')
    data_util.cond_mkdir(save_dir_alpha_map)

    save_dir_sh_basis_map = os.path.join(opt.calib_dir, 'resol_' + str(opt.img_size), 'precomp', 'sh_basis_map')
    data_util.cond_mkdir(save_dir_sh_basis_map)

    # Save all command line arguments into a txt file in the logging directory for later reference.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Begin inference...')
    with torch.no_grad():
        for ithView in range(num_view):
            t_prep = time.time()

            # get view data
            view_trgt = view_dataset[ithView]
            proj = view_trgt[0]['proj'].to(device)
            pose = view_trgt[0]['pose'].to(device)
            proj_inv = view_trgt[0]['proj_inv'].to(device)
            R_inv = view_trgt[0]['R_inv'].to(device)

            proj = proj[None, :]
            pose = pose[None, :]
            proj_inv = proj_inv[None, :]
            R_inv = R_inv[None, :]

            t_prep = time.time() - t_prep

            t_raster = time.time()
            # rasterize
            uv_map, alpha_map, face_index_map, weight_map, faces_v_idx, normal_map, normal_map_cam, faces_v, faces_vt, position_map, position_map_cam, depth, v_uvz, v_front_mask = \
                rasterizer(proj = proj.cuda(0),
                            pose = pose.cuda(0),
                            dist_coeffs = None,
                            offset = None,
                            scale = None,
                            )
            uv_map = uv_map.to(device)
            alpha_map = alpha_map.to(device)
            face_index_map = face_index_map.to(device)
            normal_map = normal_map.to(device)
            faces_v = faces_v.to(device)
            faces_vt = faces_vt.to(device)
            t_raster = time.time() - t_raster

            # save alpha map
            cv2.imwrite(os.path.join(save_dir_alpha_map, str(ithView).zfill(5) + '.png'),
                        alpha_map[0, :, :, None].cpu().detach().numpy()[:, :, ::-1] * 255.)

            t_preproc = time.time()

            batch_size = alpha_map.shape[0]
            img_h = alpha_map.shape[1]
            img_w = alpha_map.shape[2]

            # compute TBN_map
            TBN_map = render.get_TBN_map(normal_map, face_index_map, faces_v = faces_v[0, :], faces_texcoord = faces_vt[0, :], tangent = None)
            # compute view_dir_map in world space
            view_dir_map, _ = camera.get_view_dir_map(uv_map.shape[1:3], proj_inv, R_inv)
            # compute view_dir_map in tangent space
            view_dir_map_tangent = torch.matmul(TBN_map.reshape((-1, 3, 3)).transpose(-2, -1), view_dir_map.reshape((-1, 3, 1)))[..., 0].reshape(view_dir_map.shape)
            view_dir_map_tangent = torch.nn.functional.normalize(view_dir_map_tangent, dim = -1)

            t_preproc = time.time() - t_preproc

            t_sh = time.time()
            # SH basis value for view_dir_map
            sh_basis_map_fp = os.path.join(save_dir_sh_basis_map, str(ithView).zfill(5) + '.mat')
            if opt.force_recompute or not os.path.isfile(sh_basis_map_fp):
                print('Compute sh_basis_map...')
                sh_basis_map = sph_harm.evaluate_sh_basis(lmax = 2, directions = view_dir_map.reshape((-1, 3)).cpu().detach().numpy()).reshape((*(view_dir_map.shape[:3]), -1)).astype(np.float32) # [N, H, W, 9]
                # save
                scipy.io.savemat(sh_basis_map_fp, {'sh_basis_map': sh_basis_map[0, :]})
            else:
                sh_basis_map = scipy.io.loadmat(sh_basis_map_fp)['sh_basis_map'][None, ...]
            sh_basis_map = torch.from_numpy(sh_basis_map).to(device)
            t_sh = time.time() - t_sh

            t_network = time.time()

            # sample texture
            neural_img = texture_mapper(uv_map, sh_basis_map, sh_start_ch = 6) # [N, C, H, W]
            albedo_diffuse = neural_img[:, :3, :, :]
            albedo_specular = neural_img[:, 3:6, :, :]

            # sample specular rays
            rays_dir, rays_uv, rays_dir_tangent = ray_sampler(TBN_map, view_dir_map_tangent, alpha_map[..., None]) # [N, H, W, 3, num_ray], [N, H, W, 2, num_ray], [N, H, W, 3, num_ray]
            num_ray = rays_uv.shape[-1]

            # sample diffuse rays
            rays_diffuse_dir, rays_diffuse_uv, _ = ray_sampler_diffuse(TBN_map, view_dir_map_tangent, alpha_map[..., None]) # [N, H, W, 3, num_ray], [N, H, W, 2, num_ray]
            num_ray_diffuse = rays_diffuse_uv.shape[-1]
            num_ray_total = num_ray + num_ray_diffuse

            # concat data
            rays_dir = torch.cat((rays_dir, rays_diffuse_dir), dim = -1)
            rays_uv = torch.cat((rays_uv, rays_diffuse_uv), dim = -1)

            # estimate light transport for rays
            render_net_input = torch.cat((rays_dir.permute((0, -1, -2, 1, 2)).reshape((batch_size, -1, img_h, img_w)),
                                            normal_map.permute((0, 3, 1, 2)),
                                            view_dir_map.permute((0, 3, 1, 2)),
                                            neural_img), dim = 1)
            rays_lt = render_net(render_net_input, v_feature).reshape((batch_size, num_ray_total, -1, img_h, img_w)) # [N, num_ray, C, H, W]
            lt_max_val = 2.0
            rays_lt = (rays_lt * 0.5 + 0.5) * lt_max_val # map to [0, lt_max_val]

            t_network = time.time() - t_network

            for lighting_idx in lighting_idx_all:
                print('Lighting', lighting_idx)

                save_dir_img_est = os.path.join(log_dir, 'img_est_' + opt.lighting_type + '_' + str(lighting_idx).zfill(3))
                data_util.cond_mkdir(save_dir_img_est)

                # render using ray_renderer
                t_render = time.time()
                outputs_final, _, _, _, _, _, lp = ray_renderer(albedo_specular, rays_uv, rays_lt, lighting_idx = lighting_idx, albedo_diffuse = albedo_diffuse, num_ray_diffuse = num_ray_diffuse, lp_scale_factor = 1, seperate_albedo = True)
                t_render = time.time() - t_render

                print('View:', ithView, t_prep, t_raster, t_preproc, t_sh, t_network, t_render)

                # save rendered image
                cv2.imwrite(os.path.join(save_dir_img_est, str(ithView).zfill(5) + '.png'), outputs_final[0, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)

                # get background image
                if opt.save_img_bg:
                    save_dir_img_bg = os.path.join(log_dir,
                                                   'img_bg_' + opt.lighting_type + '_' + str(lighting_idx).zfill(3))
                    data_util.cond_mkdir(save_dir_img_bg)

                    # get view uv on light probe
                    view_uv_map = render.spherical_mapping_batch(-view_dir_map.transpose(1, -1)).transpose(1, -1)  # [N, H, W, 2]

                    lp_sh = lp
                    lp_lp = lighting_model_lp(lighting_idx, is_lp = True).to(device) # [N, H, W, C]
                    img_bg_sh = interpolater(lp_sh, (view_uv_map[..., 0] * float(lp_sh.shape[2])).clamp(max = lp_sh.shape[2] - 1), (view_uv_map[..., 1] * float(lp_sh.shape[1])).clamp(max = lp_sh.shape[1] - 1)) # [N, H, W, C]
                    img_bg_lp = interpolater(lp_lp, (view_uv_map[..., 0] * float(lp_lp.shape[2])).clamp(max = lp_lp.shape[2] - 1), (view_uv_map[..., 1] * float(lp_lp.shape[1])).clamp(max = lp_lp.shape[1] - 1)) # [N, H, W, C]
                    cv2.imwrite(os.path.join(save_dir_img_bg, 'sh_' + str(ithView).zfill(5) + '.png'), img_bg_sh[0, :].cpu().detach().numpy()[:, :, ::-1] * 255.)
                    cv2.imwrite(os.path.join(save_dir_img_bg, 'lp_' + str(ithView).zfill(5) + '.png'), img_bg_lp[0, :].cpu().detach().numpy()[:, :, ::-1] * 255.)


if __name__ == '__main__':
    main()
