import argparse
import os, time, datetime

import torch
from torch import nn
import torchvision
import numpy as np
import cv2
import scipy.io

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch_geometric.data
import neural_renderer as nr

import dataio
import data_util
import util
import metric

import network
import render
import misc
import sph_harm


parser = argparse.ArgumentParser()

# general
parser.add_argument('--data_root', required=True,
                    help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
parser.add_argument('--logging_root', type=str, default=None, required=False,
                    help='Path to directory where to write tensorboard logs and checkpoints.')
# mesh
parser.add_argument('--calib_fp', type=str, default='_/calib.mat', required=False,
                    help='Path of calibration file.')
parser.add_argument('--calib_format', type=str, default='convert', required=False,
                    help='Format of calibration file')
parser.add_argument('--obj_high_fp', type=str, default='_/mesh.obj', required=False,
                    help='Path of high-resolution mesh obj.')
parser.add_argument('--obj_low_fp', type=str, default='_/mesh_7500v.obj', required=False,
                    help='Path of low-resolution mesh obj.')
parser.add_argument('--obj_gcn_fp', type=str, default='_/mesh_7500v.obj', required=False,
                    help='Path of mesh obj for gcn.')
parser.add_argument('--tex_fp', type=str, default='_/tex.png', required=False,
                    help='Path of texture.')
# view datasets
parser.add_argument('--img_size', type=int, default=512,
                    help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')
parser.add_argument('--img_gamma', type=float, default=1.0,
                    help='Image gamma.')
# texture mapper
parser.add_argument('--texture_size', type=int, default=512,
                    help='Sidelength of neural texture. Default 512.')
parser.add_argument('--texture_num_ch', type=int, default=24,
                    help='Number of channels for neural texture. Default 24.')
parser.add_argument('--mipmap_level', type=int, default=4, required=False,
                    help='Mipmap levels for neural texture. Default 4.')
parser.add_argument('--init_tex', default=False, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether initialize neural texture using reconstructed texture.')
parser.add_argument('--fix_tex', default=False, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether fix neural texture.')
parser.add_argument('--apply_sh', default=True, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether apply spherical harmonics to sampled feature maps. Default True.')
# lighting
parser.add_argument('--lp_dir', type=str, default=None, required=False,
                    help='Path to directory that holds the light probe data.')
parser.add_argument('--sphere_samples_fp', type = str, default='./sphere_samples_4096.mat', required=False,
                    help='Path to sphere samples.')
parser.add_argument('--sh_lmax', type = int, default=10, required=False,
                    help='Maximum degrees of SH basis for lighting.')
parser.add_argument('--fix_lighting', default = False, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether fix lighting params.')
parser.add_argument('--init_lighting', default=True, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether initialize lighting params.')
parser.add_argument('--lighting_idx', default = None, type = int,
                    help='Lighting index for training.')
parser.add_argument('--lighting_relight_idx', default = None, type = int,
                    help='Lighting index for relighting.')
# rendering net
parser.add_argument('--nf0', type=int, default=64,
                    help='Number of features in outermost layer of U-Net architectures.')
# gcn
parser.add_argument('--in_channels', default=6, type=int, help='the channel size of input point cloud')
parser.add_argument('--kernel_size', default=16, type=int, help='neighbor num (default:16)')
parser.add_argument('--block_type', default='res', type=str, help='graph backbone block type {res, dense}')
parser.add_argument('--conv_type', default='edge', type=str, help='graph conv layer {edge, mr}')
parser.add_argument('--act_type', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
parser.add_argument('--norm_type', default='batch', type=str, help='batch or instance normalization')
parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer, True or False')
parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
parser.add_argument('--n_blocks', default=20, type=int, help='number of basic blocks')
parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
parser.add_argument('--out_channels_gcn', default=512, type=int, help='the channel size of output features')
# losses
parser.add_argument('--loss_lighting_weight', type=float, default=1.0)
parser.add_argument('--loss_lighting_uncovered_weight', type=float, default=0.1)
parser.add_argument('--loss_rays_lt_chrom_weight', type=float, default=1.0)
parser.add_argument('--loss_alb_weight', type=float, default=1.0)
# training
parser.add_argument('--max_epoch', type=int, default=2000, help='Maximum number of epochs to train for.')
parser.add_argument('--max_iter', type=int, default=None, help='Maximum number of iterations to train for.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--sampling_pattern', type=str, default='all', required=False)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
# validation
parser.add_argument('--sampling_pattern_val', type=str, default='all', required=False)
parser.add_argument('--val_freq', type=int, default=100,
                    help='Test on validation data every X iterations.')
# misc
parser.add_argument('--exp_name', type=str, default='', help='(optional) Name for experiment.')
parser.add_argument('--gpu_id', type=str, default='', help='Cuda visible devices. First device for gcn, last device for the others.')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
parser.add_argument('--log_freq', type=int, default=100, help='Save tensorboard logs every X iterations.')
parser.add_argument('--ckp_freq', type=int, default=1000, help='Save checkpoint every X iterations.')


opt = parser.parse_args()
if opt.logging_root is None:
    opt.logging_root = os.path.join(opt.data_root, 'logs', 'rnr')
if opt.calib_fp[:2] == '_/':
    opt.calib_fp = os.path.join(opt.data_root, opt.calib_fp[2:])
if opt.obj_high_fp[:2] == '_/':
    opt.obj_high_fp = os.path.join(opt.data_root, opt.obj_high_fp[2:])
if opt.obj_low_fp[:2] == '_/':
    opt.obj_low_fp = os.path.join(opt.data_root, opt.obj_low_fp[2:])
if opt.obj_gcn_fp[:2] == '_/':
    opt.obj_gcn_fp = os.path.join(opt.data_root, opt.obj_gcn_fp[2:])
if opt.tex_fp[:2] == '_/':
    opt.tex_fp = os.path.join(opt.data_root, opt.tex_fp[2:])
if opt.lp_dir is not None and opt.lp_dir[:2] == '_/':
    opt.lp_dir = os.path.join(opt.data_root, opt.lp_dir[2:])
if opt.sphere_samples_fp[:2] == '_/':
    opt.sphere_samples_fp = os.path.join(opt.data_root, opt.sphere_samples_fp[2:])
obj_high_name = opt.obj_high_fp.split('/')[-1].split('.')[0]
obj_low_name = opt.obj_low_fp.split('/')[-1].split('.')[0]
opt.precomp_high_dir = os.path.join(opt.data_root, 'precomp_' + obj_high_name)
opt.precomp_low_dir = os.path.join(opt.data_root, 'precomp_' + obj_low_name)

print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

# device allocation
if opt.gpu_id == '':
    device_gcn = torch.device('cpu')
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device_gcn = torch.device('cuda:' + opt.gpu_id[0])
    device = torch.device('cuda:' + opt.gpu_id[-1])

# load global_RT
if opt.calib_format == 'convert':
    global_RT = torch.from_numpy(scipy.io.loadmat(opt.calib_fp)['global_RT'].astype(np.float32))
else:
    global_RT = None

# load texture of obj
texture_init = cv2.cvtColor(cv2.imread(opt.tex_fp), cv2.COLOR_BGR2RGB)
texture_init_resize = cv2.resize(texture_init, (opt.texture_size, opt.texture_size), interpolation = cv2.INTER_AREA).astype(np.float32) / 255.0
texture_init_use = None
if opt.init_tex is True:
    texture_init_use = torch.from_numpy(texture_init_resize)
num_channel = texture_init.shape[-1]

# sample light directions on sphere
l_dir_np = scipy.io.loadmat(opt.sphere_samples_fp)['sphere_samples'].transpose() # [3, num_sample]
l_dir = torch.from_numpy(l_dir_np) # [3, num_sample]
num_sample = l_dir.shape[1]

# handle lighting options
has_lighting_gt = True
if opt.lighting_idx is None:
    has_lighting_gt = False
    opt.lighting_idx = 0 # store estimated lighting as the first lighting
has_lighting_init = opt.init_lighting
has_lighting_relight = True
if opt.lighting_relight_idx is None:
    has_lighting_relight = False

# dataset for training views
if opt.lighting_idx is not None:
    img_dir = opt.data_root + '/rgb' + str(opt.lighting_idx) + '/'
else:
    img_dir = opt.data_root + '/rgb0/'
view_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                img_dir = img_dir,
                                calib_path = opt.calib_fp,
                                calib_format = opt.calib_format,
                                img_size = [opt.img_size, opt.img_size],
                                sampling_pattern = opt.sampling_pattern,
                                load_precompute = True,
                                precomp_high_dir = opt.precomp_high_dir,
                                precomp_low_dir = opt.precomp_low_dir,
                                img_gamma = opt.img_gamma,
                                )

# dataset for relighted training views
img_relight_dir = opt.data_root + '/rgb' + str(opt.lighting_relight_idx) + '/'
if os.path.isdir(img_relight_dir):
    view_dataset_relight = dataio.ViewDataset(root_dir = opt.data_root,
                                    img_dir = img_relight_dir,
                                    calib_path = opt.calib_fp,
                                    calib_format = opt.calib_format,
                                    img_size = [opt.img_size, opt.img_size],
                                    sampling_pattern = opt.sampling_pattern,
                                    img_gamma = opt.img_gamma,
                                    )
has_view_relight = has_lighting_relight and ('view_dataset_relight' in globals())

# dataset for validation views
view_val_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                img_dir = img_dir,
                                calib_path = opt.calib_fp,
                                calib_format = opt.calib_format,
                                img_size = [opt.img_size, opt.img_size],
                                sampling_pattern = opt.sampling_pattern_val,
                                load_precompute = True,
                                precomp_high_dir = opt.precomp_high_dir,
                                precomp_low_dir = opt.precomp_low_dir,
                                img_gamma = opt.img_gamma,
                                )
num_view_val = len(view_val_dataset)

# dataset for relighted validation views
if os.path.isdir(img_relight_dir):
    view_val_dataset_relight = dataio.ViewDataset(root_dir = opt.data_root,
                                    img_dir = img_relight_dir,
                                    calib_path = opt.calib_fp,
                                    calib_format = opt.calib_format,
                                    img_size = [opt.img_size, opt.img_size],
                                    sampling_pattern = opt.sampling_pattern_val,
                                    img_gamma = opt.img_gamma,
                                    )

# dataset loader for light probes
if opt.lp_dir is not None:
    lp_dataset = dataio.LightProbeDataset(data_dir = opt.lp_dir)
    print('Start buffering light probe data...')
    lp_dataset.buffer_all()
    lp_dataloader = DataLoader(lp_dataset, batch_size = 1, shuffle = False, num_workers = 8)
else:
    lp_dataloader = None

# interpolater
interpolater = network.Interpolater()

# texture mapper
texture_mapper = network.TextureMapper(texture_size = opt.texture_size,
                                        texture_num_ch = opt.texture_num_ch,
                                        mipmap_level = opt.mipmap_level,
                                        texture_init = texture_init_use,
                                        fix_texture = opt.fix_tex,
                                        apply_sh = opt.apply_sh)

# gcn input
v_attr, f_attr = nr.load_obj(opt.obj_gcn_fp, normalization = False, use_cuda = False)
gcn_input = torch_geometric.data.Data(pos = v_attr['v'], x = v_attr['v']).to(device_gcn)
opt.num_v_gcn = v_attr['v'].shape[0]

# deep_gcn
gcn = network.DenseDeepGCN(opt)

# lighting model lp
if lp_dataloader is not None:
    lighting_model_lp = network.LightingLP(l_dir, num_channel = num_channel, lp_dataloader = lp_dataloader, fix_params = opt.fix_lighting)
    lighting_model_lp.fit_sh(lmax = opt.sh_lmax)

# lighting model sh
if 'lighting_model_lp' in globals():
    lighting_model_sh = network.LightingSH(l_dir, lmax = opt.sh_lmax, num_lighting = lighting_model_lp.num_lighting, num_channel = num_channel, init_coeff = lighting_model_lp.sh_coeff, fix_params = opt.fix_lighting, lp_recon_h = 256, lp_recon_w = 512)
else:
    lighting_model_sh = network.LightingSH(l_dir, lmax = opt.sh_lmax, num_lighting = 1, num_channel = num_channel, fix_params =  opt.fix_lighting, lp_recon_h = 256, lp_recon_w = 512)

lighting_model = lighting_model_sh

#################### process lighting ####################
# load stitched light probes
if opt.lighting_idx is None:
    idx_use = 0
else:
    idx_use = opt.lighting_idx
lp_stitch_dir = os.path.join(opt.data_root, 'light_probe_stitch_' + opt.sampling_pattern)
if os.path.isfile(os.path.join(lp_stitch_dir, str(idx_use) + '.exr')):
    lp_stitch = cv2.imread(os.path.join(lp_stitch_dir, str(idx_use) + '.exr'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
else:
    lp_stitch = cv2.imread(os.path.join(lp_stitch_dir, str(idx_use) + '.png'), cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32) / 255.
lp_stitch[np.isnan(lp_stitch)] = 0
lp_stitch = cv2.cvtColor(lp_stitch, cv2.COLOR_BGR2RGB) ** opt.img_gamma
lp_stitch_mask = cv2.imread(os.path.join(lp_stitch_dir, 'mask', str(idx_use) + '.png')).astype(np.float32) / 255.0
lp_stitch_count = scipy.io.loadmat(os.path.join(lp_stitch_dir, 'count', str(idx_use) + '.mat'))
lp_stitch_count = lp_stitch_count['count'].astype(np.float32) / lp_stitch_count['num_view'].astype(np.float32)
# fill in missing regions
for ith_ch in range(num_channel):
    lp_stitch[lp_stitch_mask[:, :, ith_ch] == 0, ith_ch] = lp_stitch[lp_stitch_mask[:, :, ith_ch] == 1, ith_ch].mean()
# resize
lp_stitch_resize = cv2.resize(lp_stitch, (lighting_model_sh.lp_recon_w, lighting_model_sh.lp_recon_h), interpolation = cv2.INTER_AREA)
lp_stitch_mask_resize = cv2.resize(lp_stitch_mask, (lighting_model_sh.lp_recon_w, lighting_model_sh.lp_recon_h), interpolation = cv2.INTER_AREA)
lp_stitch_count_resize = cv2.resize(lp_stitch_count, (lighting_model_sh.lp_recon_w, lighting_model_sh.lp_recon_h), interpolation = cv2.INTER_AREA)
# convert to pytorch tensors
lp_stitch = torch.from_numpy(lp_stitch)
lp_stitch_mask = torch.from_numpy(lp_stitch_mask) == 1
lp_stitch_count = torch.from_numpy(lp_stitch_count)
lp_stitch_resize = torch.from_numpy(lp_stitch_resize).to(device)
lp_stitch_mask_resize = (torch.from_numpy(lp_stitch_mask_resize) == 1).to(device)
lp_stitch_count_resize = torch.from_numpy(lp_stitch_count_resize).to(device)
# fit sh to lp_stitch
l_samples_uv = render.spherical_mapping(l_dir) # [2, num_sample]
l_samples_lp_stitch = misc.interpolate_bilinear(lp_stitch, (l_samples_uv[None, 0, :] * float(lp_stitch.shape[1])).clamp(max = lp_stitch.shape[1] - 1), (l_samples_uv[None, 1, :] * float(lp_stitch.shape[0])).clamp(max = lp_stitch.shape[0] - 1))[0, :] # [num_sample, num_channel]
l_samples_mask = misc.interpolate_bilinear(lp_stitch_mask.to(torch.float32), (l_samples_uv[None, 0, :] * float(lp_stitch.shape[1])).clamp(max = lp_stitch.shape[1] - 1), (l_samples_uv[None, 1, :] * float(lp_stitch.shape[0])).clamp(max = lp_stitch.shape[0] - 1))[0, :, 0] == 1 # [num_sample]
lp_stitch_sh_coeff = sph_harm.fit_sh_coeff(samples = l_samples_lp_stitch, sh_basis_val = lighting_model_sh.basis_val) # [num_basis, num_channel]

# lighting gt (sh and reconstructed lp)
if has_lighting_gt:
    lighting_sh_coeff_gt = lighting_model_sh.coeff.data[opt.lighting_idx, :].clone().to(device) # [num_basis, num_channel]
    lp_gt = lighting_model_sh.reconstruct_lp(lighting_sh_coeff_gt.cpu()).to(device)
# lighting stitch (sh and reconstructed lp)
lighting_sh_coeff_stitch = lp_stitch_sh_coeff.to(device)
lp_stitch_sh_recon = lighting_model_sh.reconstruct_lp(lighting_sh_coeff_stitch.cpu()).to(device) # [H, W, C]

# initialize lighting
if has_lighting_init:
    lighting_sh_coeff_init = lighting_sh_coeff_stitch.clone() # [num_basis, num_channel]
    lighting_model_sh.coeff.data[opt.lighting_idx, :] = lighting_sh_coeff_init # initialize
    lp_init = lighting_model_sh.reconstruct_lp(lighting_sh_coeff_init.cpu()).to(device)
    l_samples_init = l_samples_lp_stitch.clone().to(device)
    l_samples_init_mask = l_samples_mask.clone().to(device)
else:
    lighting_model_sh.coeff.data[opt.lighting_idx, :] = 0.1 # reset lighting params, don't set to zero (normalize_factor will be nan)

# get lighting data for relight
if has_lighting_relight:
    l_samples_relight_lp = lighting_model_lp.l_samples.data[opt.lighting_relight_idx, :].to(device) # [num_sample, num_channel]
    lp_relight_lp = lighting_model_lp.lps[opt.lighting_relight_idx, :].to(device) # [H, W, C]
    l_samples_relight_sh = lighting_model_sh.l_samples.data[opt.lighting_relight_idx, :].to(device) # [num_sample, num_channel]
    lp_relight_sh = lighting_model_sh.reconstruct_lp(lighting_model_sh.coeff.data[opt.lighting_relight_idx, :]).to(device) # [H, W, C]

    l_samples_relight = l_samples_relight_sh
    lp_relight = lp_relight_sh

########################################

# ray sampler specular
opt.num_azi = 6
opt.num_polar = 2
opt.interval_polar = 5
ray_sampler = network.RaySampler(num_azi = opt.num_azi, num_polar = opt.num_polar, interval_polar = opt.interval_polar)
num_ray = ray_sampler.num_ray

# ray sampler diffuse
opt.num_azi = 6
opt.num_polar = 2
opt.interval_polar = 10
ray_sampler_diffuse = network.RaySampler(num_azi = opt.num_azi, num_polar = opt.num_polar, interval_polar = opt.interval_polar, mode = 'diffuse')
num_ray_diffuse = ray_sampler_diffuse.num_ray

num_ray_total = num_ray + num_ray_diffuse

# rendering net
render_net = network.RenderingNet(nf0 = opt.nf0,
                            in_channels = num_ray_total * 3 + 6 + opt.texture_num_ch,
                            out_channels = 3 * num_ray_total,
                            num_down_unet = 5,
                            out_channels_gcn = opt.out_channels_gcn)

# ray renderer
ray_renderer = network.RayRenderer(lighting_model, interpolater)

# L1 loss
criterionL1 = nn.L1Loss(reduction = 'mean').to(device)

# Chrom loss
criterion_rays_lt_chrom = network.RaysLTChromLoss().to(device)

# Optimizer
optimizerG = torch.optim.Adam(list(gcn.parameters()) + list(texture_mapper.parameters()) + list(lighting_model.parameters()) + list(render_net.parameters()), lr = opt.lr)
optimizerG.zero_grad()

# move to device
interpolater.to(device)
texture_mapper.to(device)
lighting_model.to(device)
ray_sampler.to(device)
ray_sampler_diffuse.to(device)
render_net.to(device)
ray_renderer.to(device)
gcn.to(device_gcn)

# get module
texture_mapper_module = texture_mapper
lighting_model_module = lighting_model
ray_sampler_module = ray_sampler
ray_sampler_diffuse_module = ray_sampler_diffuse
render_net_module = render_net
gcn_module = gcn

# set to training mode
interpolater.train()
texture_mapper.train()
lighting_model.train()
ray_sampler.train()
ray_sampler_diffuse.train()
render_net.train()
ray_renderer.train()
gcn.train()

# collect all networks and optimizers
part_list = [texture_mapper_module, lighting_model_module, ray_sampler_module, ray_sampler_diffuse_module, render_net_module, gcn_module, []]
part_name_list = ['texture_mapper', 'lighting_model', 'ray_sampler', 'ray_sampler_diffuse', 'render_net', 'gcn', 'v_feature']

print("*" * 100)
print("Number of parameters")
print("texture mapper:")
opt.num_params_texture_mapper = util.print_network(texture_mapper)
print("lighting model:")
opt.num_params_lighting_model = util.print_network(lighting_model)
print("render net:")
opt.num_params_render_net = util.print_network(render_net)
print("gcn:")
opt.num_params_gcn = util.print_network(gcn)
print("*" * 100)


def main():
    print('Start buffering data for training views...')
    view_dataset.buffer_all()
    view_dataloader = DataLoader(view_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = 8)

    if has_view_relight:
        print('Start buffering data for relighted training views...')
        view_dataset_relight.buffer_all()

    print('Start buffering data for validation views...')
    view_val_dataset.buffer_all()
    view_val_dataloader = DataLoader(view_val_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = 8)

    if has_view_relight:
        print('Start buffering data for relighted validation views...')
        view_val_dataset_relight.buffer_all()

    # directory name contains some info about hyperparameters.
    dir_name = os.path.join(datetime.datetime.now().strftime('%m-%d') + 
                            '_' + datetime.datetime.now().strftime('%H-%M-%S') +
                            '_' + opt.sampling_pattern +
                            '_' + opt.data_root.strip('/').split('/')[-1])
    if opt.exp_name is not '':
        dir_name += '_' + opt.exp_name

    # directory for logging
    log_dir = os.path.join(opt.logging_root, dir_name)
    data_util.cond_mkdir(log_dir)

    # directory for saving validation data on view synthesis
    val_out_dir = os.path.join(log_dir, 'val_out')
    val_gt_dir = os.path.join(log_dir, 'val_gt')
    val_err_dir = os.path.join(log_dir, 'val_err')
    data_util.cond_mkdir(val_out_dir)
    data_util.cond_mkdir(val_gt_dir)
    data_util.cond_mkdir(val_err_dir)

    # directory for saving validation data on relighting
    val_relight_out_dir = os.path.join(log_dir, 'val_relight_out')
    data_util.cond_mkdir(val_relight_out_dir)
    if has_view_relight:
        val_relight_gt_dir = os.path.join(log_dir, 'val_relight_gt')
        val_relight_err_dir = os.path.join(log_dir, 'val_relight_err')
        data_util.cond_mkdir(val_relight_gt_dir)
        data_util.cond_mkdir(val_relight_err_dir)

    # Save all command line arguments into a txt file in the logging directory for later reference.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # tensorboardX writer
    writer = SummaryWriter(log_dir)

    iter = opt.start_epoch * len(view_dataset)

    print('Begin training...')
    val_log_batch_id = 0
    first_val = True
    for epoch in range(opt.start_epoch, opt.max_epoch):
        for view_trgt in view_dataloader:
            if opt.max_iter is not None and iter >= opt.max_iter:
                return
            
            start = time.time()

            # gcn features
            v_feature = gcn(gcn_input).to(device)

            # get view data
            TBN_map = view_trgt[0]['TBN_map'].to(device) # [N, H, W, 3, 3]
            uv_map = view_trgt[0]['uv_map'].to(device) # [N, H, W, 2]
            sh_basis_map = view_trgt[0]['sh_basis_map'].to(device) # [N, H, W, 9]
            normal_map = view_trgt[0]['normal_map'].to(device) # [N, H, W, 3]
            view_dir_map = view_trgt[0]['view_dir_map'].to(device) # [N, H, W, 3]
            view_dir_map_tangent = view_trgt[0]['view_dir_map_tangent'].to(device) # [N, H, W, 3]
            alpha_map = view_trgt[0]['alpha_map'][:, None, :, :].to(device) # [N, 1, H, W]
            view_idx = view_trgt[0]['idx']

            batch_size = alpha_map.shape[0]
            img_h = alpha_map.shape[2]
            img_w = alpha_map.shape[3]

            num_view = len(view_trgt)
            img_gt = []
            for i in range(num_view):
                img_gt.append(view_trgt[i]['img_gt'].to(device)) # [N, C, H, W]

            # sample texture
            neural_img = texture_mapper(uv_map, sh_basis_map, sh_start_ch = 6) # [N, C, H, W]
            albedo_diffuse = neural_img[:, :3, :, :]
            albedo_specular = neural_img[:, 3:6, :, :]

            # sample specular rays
            rays_dir, rays_uv, rays_dir_tangent = ray_sampler(TBN_map, view_dir_map_tangent, alpha_map.permute((0, 2, 3, 1))) # [N, H, W, 3, num_ray], [N, H, W, 2, num_ray], [N, H, W, 3, num_ray]
            num_ray = rays_uv.shape[-1]

            # sample diffuse rays
            rays_diffuse_dir, rays_diffuse_uv, _ = ray_sampler_diffuse(TBN_map, view_dir_map_tangent, alpha_map.permute((0, 2, 3, 1))) # [N, H, W, 3, num_ray], [N, H, W, 2, num_ray]
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

            # render using ray_renderer
            outputs_final, _, _, _, _, _, _ = ray_renderer(albedo_specular, rays_uv, rays_lt, lighting_idx = opt.lighting_idx, albedo_diffuse = albedo_diffuse, num_ray_diffuse = num_ray_diffuse, seperate_albedo = True)
            outputs_final = [outputs_final] # [N, C, H, W]

            with torch.no_grad():
                # relight
                if has_lighting_relight:
                    # ray renderer
                    outputs_final_relight, _, _, _, _, _, _ = ray_renderer(albedo_specular, rays_uv, rays_lt, lp = lp_relight[None, :].expand(batch_size, -1, -1, -1), albedo_diffuse = albedo_diffuse, num_ray_diffuse = num_ray_diffuse, seperate_albedo = True)
                    outputs_final_relight = [outputs_final_relight] # [N, C, H, W]

            # relight gt
            if has_view_relight:
                img_relight_gt = []
                for i in range(batch_size):
                    img_relight_gt.append(view_dataset_relight.views_all[view_idx[i]]['img_gt'])
                img_relight_gt = torch.stack(img_relight_gt).to(device)
                img_relight_gt = [img_relight_gt]

            # get estimated lighting SH coeffs
            lighting_sh_coeff_est = lighting_model_module.get_lighting_params(opt.lighting_idx) # [num_basis, num_channel]
            # reconstruct light probe
            lp_est = lighting_model_module.reconstruct_lp(lighting_sh_coeff_est)
            # reconstruct light samples
            l_samples_est = sph_harm.reconstruct_sh(lighting_sh_coeff_est, lighting_model_module.basis_val)

            # We don't enforce a loss on the outermost 5 pixels to alleviate boundary errors, also weight loss by alpha
            alpha_map_central = alpha_map[:, :, 5:-5, 5:-5]
            img_gt_orig = img_gt[0].clone()
            for i in range(num_view):
                outputs_final[i] = outputs_final[i][:, :, 5:-5, 5:-5] * alpha_map_central
                img_gt[i] = img_gt[i][:, :, 5:-5, 5:-5] * alpha_map_central
                if has_lighting_relight:
                    outputs_final_relight[i] = outputs_final_relight[i][:, :, 5:-5, 5:-5] * alpha_map_central
                if has_view_relight:
                    img_relight_gt[i] = img_relight_gt[i][:, :, 5:-5, 5:-5] * alpha_map_central

            loss_lighting = 0
            if not opt.fix_lighting:
                # loss on estimated light samples
                loss_lighting = (l_samples_init[l_samples_init_mask, :] - l_samples_est[l_samples_init_mask, :]).abs().sum() / l_samples_init_mask.to(l_samples_est.dtype).sum() * opt.loss_lighting_weight
                loss_lighting = loss_lighting + (l_samples_init[(l_samples_init_mask != 1), :] - l_samples_est[(l_samples_init_mask != 1), :]).abs().sum() / (l_samples_init_mask != 1).to(l_samples_est.dtype).sum() * opt.loss_lighting_uncovered_weight

            # loss on final img
            loss_rn = list()
            for idx in range(num_view):
                loss_rn.append(criterionL1(outputs_final[idx].contiguous().view(-1).float(), img_gt[idx].view(-1).float()))
            loss_rn = torch.stack(loss_rn, dim = 0).mean()

            # loss on rays light transport chromaticity
            try:
                loss_rays_lt_chrom, rays_lt_chrom, rays_lt_chrom_mean, rays_lt_chrom_diff = criterion_rays_lt_chrom(rays_lt, alpha_map, img_gt_orig)
            except:
                loss_rays_lt_chrom, rays_lt_chrom, rays_lt_chrom_mean, rays_lt_chrom_diff = criterion_rays_lt_chrom.cpu()(rays_lt.cpu(), alpha_map.cpu(), img_gt_orig.cpu())
                loss_rays_lt_chrom = loss_rays_lt_chrom.to(device)
            loss_rays_lt_chrom = loss_rays_lt_chrom * opt.loss_rays_lt_chrom_weight

            # loss on albedo mean value
            albedo_specular_tex = texture_mapper_module.flatten_mipmap(start_ch = 3, end_ch = 6) # [1, H, W, C]
            albedo_diffuse_tex = texture_mapper_module.flatten_mipmap(start_ch = 0, end_ch = 3) # [1, H, W, C]
            mask_valid_tex_spec = (albedo_specular_tex != texture_mapper_module.tex_flatten_mipmap_init[..., 3:6]).any(dim = -1, keepdim = True).to(albedo_specular_tex.dtype)
            if mask_valid_tex_spec.sum(dim = (0, 1, 2)) == 0:
                loss_alb_spec = torch.zeros(1).to(device)
            else:
                loss_alb_spec = ((albedo_specular_tex * mask_valid_tex_spec).sum(dim = (0, 1, 2)) / mask_valid_tex_spec.sum(dim = (0, 1, 2)) - 0.5).abs().sum() / num_channel
            mask_valid_tex_diff = (albedo_diffuse_tex != texture_mapper_module.tex_flatten_mipmap_init[..., 0:3]).any(dim = -1, keepdim = True).to(albedo_diffuse_tex.dtype)
            if mask_valid_tex_diff.sum(dim = (0, 1, 2)) == 0:
                loss_alb_diff = torch.zeros(1).to(device)
            else:
                loss_alb_diff = ((albedo_diffuse_tex * mask_valid_tex_diff).sum(dim = (0, 1, 2)) / mask_valid_tex_diff.sum(dim = (0, 1, 2)) - 0.5).abs().sum() / num_channel
            loss_alb = (loss_alb_spec + loss_alb_diff) * opt.loss_alb_weight

            # total loss
            loss_g = loss_lighting + loss_rn + loss_rays_lt_chrom + loss_alb

            # compute gradients
            optimizer_step = True
            if not optimizer_step:
                loss_g.backward(retain_graph = True)
            else:
                loss_g.backward(retain_graph = False)

            # optimize
            if optimizer_step:
                optimizerG.step()
                optimizerG.zero_grad()

            # error metrics
            with torch.no_grad():
                err_metrics_batch_i_final = metric.compute_err_metrics_batch(outputs_final[0] * 255.0, img_gt[0] * 255.0, alpha_map_central, compute_ssim = False)
                if has_view_relight:
                    err_metrics_batch_i_final_relight = metric.compute_err_metrics_batch(outputs_final_relight[0] * 255.0, img_relight_gt[0] * 255.0, alpha_map_central, compute_ssim = False)

                if has_lighting_gt:
                    lighting_sh_coeff_mae = (lighting_sh_coeff_gt.to(lighting_sh_coeff_est.dtype) - lighting_sh_coeff_est).abs().sum()
                    err_metrics_batch_i_lp = metric.compute_err_metrics_batch(lp_est.permute((2, 0, 1))[None, :] * 255.0, lp_gt.to(lp_est.dtype).permute((2, 0, 1))[None, :] * 255.0, torch.ones_like(lp_est).permute((2, 0, 1))[None, :], compute_ssim = False)

            # tensorboard scalar logs of training data
            if optimizer_step:
                writer.add_scalar("loss_g", loss_g, iter)
                writer.add_scalar("loss_lighting", loss_lighting, iter)
                writer.add_scalar("loss_rn", loss_rn, iter)
                writer.add_scalar("loss_rays_lt_chrom", loss_rays_lt_chrom, iter)
                writer.add_scalar("loss_alb", loss_alb, iter)

                writer.add_scalar("final_mae_valid", err_metrics_batch_i_final['mae_valid_mean'], iter)
                writer.add_scalar("final_psnr_valid", err_metrics_batch_i_final['psnr_valid_mean'], iter)

                if has_view_relight:
                    writer.add_scalar("final_relight_mae_valid", err_metrics_batch_i_final_relight['mae_valid_mean'], iter)
                    writer.add_scalar("final_relight_psnr_valid", err_metrics_batch_i_final_relight['psnr_valid_mean'], iter)

                if has_lighting_gt:
                    writer.add_scalar("lighting_sh_coeff_mae", lighting_sh_coeff_mae, iter)
                    writer.add_scalar("lp_mae_valid", err_metrics_batch_i_lp['mae_valid_mean'], iter)
                    writer.add_scalar("lp_psnr_valid", err_metrics_batch_i_lp['psnr_valid_mean'], iter)

            end = time.time()
            print("Iter %07d   Epoch %03d   loss_g %0.4f   mae_valid %0.4f   psnr_valid %0.4f   t_total %0.4f" % (iter, epoch, loss_g, err_metrics_batch_i_final['mae_valid_mean'], err_metrics_batch_i_final['psnr_valid_mean'], end - start))
            
            # tensorboard figure logs of training data
            if not iter % opt.log_freq:
                output_final_vs_gt = []
                for i in range(num_view):
                    output_final_vs_gt.append(outputs_final[i].clamp(min = 0., max = 1.))
                    output_final_vs_gt.append(img_gt[i].clamp(min = 0., max = 1.))
                    output_final_vs_gt.append((outputs_final[i] - img_gt[i]).abs().clamp(min = 0., max = 1.))
                output_final_vs_gt = torch.cat(output_final_vs_gt, dim = 0)
                writer.add_image("output_final_vs_gt",
                                 torchvision.utils.make_grid(output_final_vs_gt,
                                                             nrow = batch_size,
                                                             range = (0, 1),
                                                             scale_each = False,
                                                             normalize = False).cpu().detach().numpy(),
                                 iter)

                lp_init_est_gt = []
                if has_lighting_init:
                    lp_init_est_gt.append(lp_init.to(lp_est.dtype).permute((2, 0, 1))[None, :].clamp(min = 0., max = 1.))
                lp_init_est_gt.append(lp_est.permute((2, 0, 1))[None, :].clamp(min = 0., max = 1.))
                if has_lighting_gt:
                    lp_init_est_gt.append(lp_gt.to(lp_est.dtype).permute((2, 0, 1))[None, :].clamp(min = 0., max = 1.))
                    lp_init_est_gt.append((lp_est - lp_gt.to(lp_est.dtype)).abs().permute((2, 0, 1))[None, :].clamp(min = 0., max = 1.))
                lp_init_est_gt = torch.cat(lp_init_est_gt, dim = 0)
                writer.add_image("lp_init_est_gt",
                                torchvision.utils.make_grid(lp_init_est_gt,
                                                            nrow = 1,
                                                            range = (0, 1),
                                                            scale_each = False,
                                                            normalize = False).cpu().detach().numpy(),
                                iter)
                                    
                if has_lighting_relight:
                    relight_final_est_gt = []
                    for i in range(num_view):
                        relight_final_est_gt.append(outputs_final_relight[i].clamp(min = 0., max = 1.))
                        if has_view_relight:
                            relight_final_est_gt.append(img_relight_gt[i].clamp(min = 0., max = 1.))
                            relight_final_est_gt.append((outputs_final_relight[i] - img_relight_gt[i]).abs().clamp(min = 0., max = 1.))
                    relight_final_est_gt = torch.cat(relight_final_est_gt, dim = 0)
                    writer.add_image("relight_final_est_gt",
                                    torchvision.utils.make_grid(relight_final_est_gt,
                                                                nrow = batch_size,
                                                                range = (0, 1),
                                                                scale_each = False,
                                                                normalize = False).cpu().detach().numpy(),
                                    iter)

            # validation
            if not iter % opt.val_freq:
                start_val = time.time()
                with torch.no_grad():
                    # error metrics
                    err_metrics_val = {}
                    err_metrics_val['mae_valid'] = []
                    err_metrics_val['mse_valid'] = []
                    err_metrics_val['psnr_valid'] = []
                    err_metrics_val['ssim_valid'] = []
                    err_metrics_val_relight = {}
                    err_metrics_val_relight['mae_valid'] = []
                    err_metrics_val_relight['mse_valid'] = []
                    err_metrics_val_relight['psnr_valid'] = []
                    err_metrics_val_relight['ssim_valid'] = []
                    # gcn features
                    v_feature = gcn(gcn_input).to(device)
                    # loop over batches
                    batch_id = 0
                    for view_val_trgt in view_val_dataloader:
                        start_val_i = time.time()

                        # get view data
                        TBN_map = view_val_trgt[0]['TBN_map'].to(device) # [N, H, W, 3, 3]
                        uv_map = view_val_trgt[0]['uv_map'].to(device) # [N, H, W, 2]
                        sh_basis_map = view_val_trgt[0]['sh_basis_map'].to(device) # [N, H, W, 9]
                        normal_map = view_val_trgt[0]['normal_map'].to(device) # [N, H, W, 3]
                        view_dir_map = view_val_trgt[0]['view_dir_map'].to(device) # [N, H, W, 3]
                        view_dir_map_tangent = view_val_trgt[0]['view_dir_map_tangent'].to(device) # [N, H, W, 3]
                        alpha_map = view_val_trgt[0]['alpha_map'][:, None, :, :].to(device) # [N, 1, H, W]
                        view_idx = view_val_trgt[0]['idx']

                        batch_size = alpha_map.shape[0]
                        img_h = alpha_map.shape[2]
                        img_w = alpha_map.shape[3]

                        num_view = len(view_val_trgt)
                        img_gt = []
                        for i in range(num_view):
                            img_gt.append(view_val_trgt[i]['img_gt'].to(device)) # [N, C, H, W]

                        # sample texture
                        neural_img = texture_mapper(uv_map, sh_basis_map, sh_start_ch = 6) # [N, C, H, W]
                        albedo_diffuse = neural_img[:, :3, :, :]
                        albedo_specular = neural_img[:, 3:6, :, :]

                        # sample specular rays
                        rays_dir, rays_uv, rays_dir_tangent = ray_sampler(TBN_map, view_dir_map_tangent, alpha_map.permute((0, 2, 3, 1))) # [N, H, W, 3, num_ray], [N, H, W, 2, num_ray], [N, H, W, 3, num_ray]
                        num_ray = rays_uv.shape[-1]

                        # sample diffuse rays
                        rays_diffuse_dir, rays_diffuse_uv, _ = ray_sampler_diffuse(TBN_map, view_dir_map_tangent, alpha_map.permute((0, 2, 3, 1))) # [N, H, W, 3, num_ray], [N, H, W, 2, num_ray]
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
                        rays_lt = (rays_lt * 0.5 + 0.5) * lt_max_val # map to [0, lt_max_val]

                        outputs_final, _, _, _, _, _, _ = ray_renderer(albedo_specular, rays_uv, rays_lt, lighting_idx = opt.lighting_idx, albedo_diffuse = albedo_diffuse, num_ray_diffuse = num_ray_diffuse, seperate_albedo = True)
                        outputs_final = [outputs_final] # [N, C, H, W]

                        # relight
                        if has_lighting_relight:
                            # ray renderer
                            outputs_final_relight, _, _, _, _, _, _ = ray_renderer(albedo_specular, rays_uv, rays_lt, lp = lp_relight[None, :].expand(batch_size, -1, -1, -1), albedo_diffuse = albedo_diffuse, num_ray_diffuse = num_ray_diffuse, seperate_albedo = True)
                            outputs_final_relight = [outputs_final_relight] # [N, C, H, W]

                        # relight gt
                        if has_view_relight:
                            img_relight_gt = []
                            for i in range(batch_size):
                                img_relight_gt.append(view_val_dataset_relight.views_all[view_idx[i]]['img_gt'])
                            img_relight_gt = torch.stack(img_relight_gt).to(device)
                            img_relight_gt = [img_relight_gt]

                        # apply alpha
                        for i in range(num_view):
                            outputs_final[i] = outputs_final[i] * alpha_map
                            img_gt[i] = img_gt[i] * alpha_map
                            if has_lighting_relight:
                                outputs_final_relight[i] = outputs_final_relight[i] * alpha_map
                            if has_view_relight:
                                img_relight_gt[i] = img_relight_gt[i] * alpha_map

                        # tensorboard figure logs of validation data
                        if batch_id == val_log_batch_id:
                            output_final_vs_gt = []
                            for i in range(num_view):
                                output_final_vs_gt.append(outputs_final[i].clamp(min = 0., max = 1.))
                                output_final_vs_gt.append(img_gt[i].clamp(min = 0., max = 1.))
                                output_final_vs_gt.append((outputs_final[i] - img_gt[i]).abs().clamp(min = 0., max = 1.))
                            output_final_vs_gt = torch.cat(output_final_vs_gt, dim = 0)
                            writer.add_image("output_final_vs_gt_val",
                                            torchvision.utils.make_grid(output_final_vs_gt,
                                                                        nrow = batch_size,
                                                                        range = (0, 1),
                                                                        scale_each = False,
                                                                        normalize = False).cpu().detach().numpy(),
                                            iter)

                            if has_lighting_relight:
                                relight_final_est_gt = []
                                for i in range(num_view):
                                    relight_final_est_gt.append(outputs_final_relight[i].clamp(min = 0., max = 1.))
                                    if has_view_relight:
                                        relight_final_est_gt.append(img_relight_gt[i].clamp(min = 0., max = 1.))
                                        relight_final_est_gt.append((outputs_final_relight[i] - img_relight_gt[i]).abs().clamp(min = 0., max = 1.))
                                relight_final_est_gt = torch.cat(relight_final_est_gt, dim = 0)
                                writer.add_image("relight_final_est_gt_val",
                                                torchvision.utils.make_grid(relight_final_est_gt,
                                                                            nrow = batch_size,
                                                                            range = (0, 1),
                                                                            scale_each = False,
                                                                            normalize = False).cpu().detach().numpy(),
                                                iter)

                        # error metrics
                        err_metrics_batch_i_final = metric.compute_err_metrics_batch(outputs_final[0] * 255.0, img_gt[0] * 255.0, alpha_map, compute_ssim = True)
                        if has_view_relight:
                            err_metrics_batch_i_final_relight = metric.compute_err_metrics_batch(outputs_final_relight[0] * 255.0, img_relight_gt[0] * 255.0, alpha_map, compute_ssim = True)
                    
                        for i in range(batch_size):
                            for key in list(err_metrics_val.keys()):
                                if key in err_metrics_batch_i_final.keys():
                                    err_metrics_val[key].append(err_metrics_batch_i_final[key][i])
                                    if has_view_relight:
                                        err_metrics_val_relight[key].append(err_metrics_batch_i_final_relight[key][i])

                        # save images
                        for i in range(batch_size):
                            cv2.imwrite(os.path.join(val_out_dir, str(iter).zfill(8) + '_' + str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'), outputs_final[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            cv2.imwrite(os.path.join(val_err_dir, str(iter).zfill(8) + '_' + str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'), (outputs_final[0] - img_gt[0]).abs().clamp(min = 0., max = 1.)[i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            if first_val:
                                cv2.imwrite(os.path.join(val_gt_dir, str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'), img_gt[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            cv2.imwrite(os.path.join(val_relight_out_dir, str(iter).zfill(8) + '_' + str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'), outputs_final_relight[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            if has_view_relight:
                                cv2.imwrite(os.path.join(val_relight_err_dir, str(iter).zfill(8) + '_' + str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'), (outputs_final_relight[0] - img_relight_gt[0]).abs().clamp(min = 0., max = 1.)[i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                                if first_val:
                                    cv2.imwrite(os.path.join(val_relight_gt_dir, str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'), img_relight_gt[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)

                        end_val_i = time.time()
                        print("Val   batch %03d   mae_valid %0.4f   psnr_valid %0.4f   ssim_valid %0.4f   t_total %0.4f" % (batch_id, err_metrics_batch_i_final['mae_valid_mean'], err_metrics_batch_i_final['psnr_valid_mean'], err_metrics_batch_i_final['ssim_valid_mean'], end_val_i - start_val_i))

                        batch_id += 1
                    
                    for key in list(err_metrics_val.keys()):
                        if err_metrics_val[key]:
                            err_metrics_val[key] = np.vstack(err_metrics_val[key])
                            err_metrics_val[key + '_mean'] = err_metrics_val[key].mean()
                        else:
                            err_metrics_val[key + '_mean'] = np.nan
                    if has_view_relight:
                        for key in list(err_metrics_val_relight.keys()):
                            if err_metrics_val_relight[key]:
                                err_metrics_val_relight[key] = np.vstack(err_metrics_val_relight[key])
                                err_metrics_val_relight[key + '_mean'] = err_metrics_val_relight[key][:num_view_val].mean()
                            else:
                                err_metrics_val_relight[key + '_mean'] = np.nan

                    # tensorboard scalar logs of validation data
                    writer.add_scalar("final_mae_valid_val", err_metrics_val['mae_valid_mean'], iter)
                    writer.add_scalar("final_psnr_valid_val", err_metrics_val['psnr_valid_mean'], iter)
                    writer.add_scalar("final_ssim_valid_val", err_metrics_val['ssim_valid_mean'], iter)
                    if has_view_relight:
                        writer.add_scalar("final_relight_mae_valid_val", err_metrics_val_relight['mae_valid_mean'], iter)
                        writer.add_scalar("final_relight_psnr_valid_val", err_metrics_val_relight['psnr_valid_mean'], iter)
                        writer.add_scalar("final_relight_ssim_valid_val", err_metrics_val_relight['ssim_valid_mean'], iter)

                    first_val = False
                    val_log_batch_id = (val_log_batch_id + 1) % batch_id

                    end_val = time.time()
                    print("Val   mae_valid %0.4f   psnr_valid %0.4f   ssim_valid %0.4f   t_total %0.4f" % (err_metrics_val['mae_valid_mean'], err_metrics_val['psnr_valid_mean'], err_metrics_val['ssim_valid_mean'], end_val - start_val))

            iter += 1

            if iter % opt.ckp_freq == 0:
                part_list[-1] = v_feature.cpu().detach()
                util.custom_save(os.path.join(log_dir, 'model_epoch-%d_iter-%s.pth' % (epoch, iter)), 
                                part_list, 
                                part_name_list)

    part_list[-1] = v_feature.cpu().detach()
    util.custom_save(os.path.join(log_dir, 'model_epoch-%d_iter-%s.pth' % (epoch, iter)), 
                                part_list, 
                                part_name_list)


if __name__ == '__main__':
    main()
