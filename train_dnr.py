import argparse
import os, time, datetime


parser = argparse.ArgumentParser()

# general
parser.add_argument('--data_root', required=True,
                    help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
parser.add_argument('--logging_root', type=str, default=None, required=False,
                    help='Path to directory where to write tensorboard logs and checkpoints.')
# mesh
parser.add_argument('--calib_fp', type=str, default='_/calib.mat', required=False,
                    help='File name of calibration file')
parser.add_argument('--calib_format', type=str, default='convert', required=False,
                    help='Format of calibration file')
parser.add_argument('--obj_fp', type=str, default='_/mesh.obj', required=False,
                    help='Path of high-resolution mesh obj.')
parser.add_argument('--tex_fp', type=str, default=None, required=False,
                    help='Path of texture.')
# view datasets
parser.add_argument('--img_dir', type=str, default='_/rgb0', required=False,
                    help='Path to directory that holds view images')
parser.add_argument('--img_size', type=int, default=512,
                    help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')
parser.add_argument('--img_gamma', type=float, default=1.0,
                    help='Image gamma.')
# texture mapper
parser.add_argument('--texture_size', type=int, default=512,
                    help='Sidelength of neural texture. Default 512.')
parser.add_argument('--texture_num_ch', type=int, default=30,
                    help='Number of channels for neural texture.')
parser.add_argument('--mipmap_level', type=int, default=4, required=False,
                    help='Mipmap levels for neural texture. Default 4.')
parser.add_argument('--apply_sh', default=True, type = lambda x: (str(x).lower() in ['true', '1']),
                    help='Whether apply spherical harmonics to sampled feature maps. Default False.')
# render net
parser.add_argument('--nf0', type=int, default=80,
                    help='Number of features in outermost layer of U-Net architectures.')
# training
parser.add_argument('--max_epoch', type=int, default=2000, help='Maximum number of epochs to train for.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--sampling_pattern', type=str, default='all', required=False)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
# validation
parser.add_argument('--sampling_pattern_val', type=str, default='all', required=False)
parser.add_argument('--val_freq', type=int, default=1000,
                    help='Test on validation data every X iterations.')
# misc
parser.add_argument('--exp_name', type=str, default='', help='(optional) Name for experiment.')
parser.add_argument('--checkpoint', default='',
                    help='Path to a checkpoint to load weights from.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Start epoch')
parser.add_argument('--gpu_id', type=str, default='',
                    help='Cuda visible devices.')
parser.add_argument('--log_freq', type=int, default=100,
                    help='Save tensorboard logs every X iterations.')
parser.add_argument('--ckp_freq', type=int, default=5000, help='Save checkpoint every X iterations.')

opt = parser.parse_args()
if opt.logging_root is None:
    opt.logging_root = os.path.join(opt.data_root, 'logs', 'dnr')
if opt.img_dir[:2] == '_/':
    opt.img_dir = os.path.join(opt.data_root, opt.img_dir[2:])
if opt.calib_fp[:2] == '_/':
    opt.calib_fp = os.path.join(opt.data_root, opt.calib_fp[2:])
if opt.obj_fp[:2] == '_/':
    opt.obj_fp = os.path.join(opt.data_root, opt.obj_fp[2:])
if opt.tex_fp is not None and opt.tex_fp[:2] == '_/':
    opt.tex_fp = os.path.join(opt.data_root, opt.tex_fp[2:])
obj_name = opt.obj_fp.split('/')[-1].split('.')[0]
opt.precomp_dir = os.path.join(opt.data_root, 'precomp_' + obj_name)

print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

# Set visible CUDA devices
if opt.gpu_id != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


import torch
from torch import nn
import torchvision
import numpy as np
import cv2

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import dataio
import data_util
import util
import metric

import network


# device allocation
if opt.gpu_id == '':
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

# load texture
if opt.tex_fp is not None:
    texture_init = cv2.cvtColor(cv2.imread(opt.tex_fp), cv2.COLOR_BGR2RGB)
    texture_init_resize = cv2.resize(texture_init, (opt.texture_size, opt.texture_size), interpolation = cv2.INTER_AREA).astype(np.float32) / 255.0
    texture_init_use = torch.from_numpy(texture_init_resize).to(device)

# dataset for training views
view_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                  img_dir = opt.img_dir,
                                  calib_path = opt.calib_fp,
                                  calib_format = opt.calib_format,
                                  img_size = [opt.img_size, opt.img_size],
                                  sampling_pattern = opt.sampling_pattern,
                                  load_precompute = True,
                                  precomp_high_dir = opt.precomp_dir,
                                  precomp_low_dir = opt.precomp_dir,
                                  img_gamma = opt.img_gamma,
                                  )

# dataset for validation views
view_val_dataset = dataio.ViewDataset(root_dir = opt.data_root,
                                img_dir = opt.img_dir,
                                calib_path = opt.calib_fp,
                                calib_format = opt.calib_format,
                                img_size = [opt.img_size, opt.img_size],
                                sampling_pattern = opt.sampling_pattern_val,
                                load_precompute = True,
                                precomp_high_dir = opt.precomp_dir,
                                precomp_low_dir = opt.precomp_dir,
                                img_gamma = opt.img_gamma,
                                )
num_view_val = len(view_val_dataset)

# texture mapper
texture_mapper = network.TextureMapper(texture_size = opt.texture_size,
                                        texture_num_ch = opt.texture_num_ch,
                                        mipmap_level = opt.mipmap_level,
                                        apply_sh = opt.apply_sh)

# render net
render_net = network.RenderingNet(nf0 = opt.nf0,
                            in_channels = opt.texture_num_ch,
                            out_channels = 3,
                            num_down_unet = 5,
                            use_gcn = False)

# interpolater
interpolater = network.Interpolater()

# L1 loss
criterionL1 = nn.L1Loss(reduction='mean').to(device)

# Optimizer
optimizerG = torch.optim.Adam(list(texture_mapper.parameters()) + list(render_net.parameters()), lr = opt.lr)

# load checkpoint
if opt.checkpoint:
    util.custom_load([texture_mapper, render_net], ['texture_mapper', 'render_net'], opt.checkpoint)

# move to device
texture_mapper.to(device)
render_net.to(device)
interpolater.to(device)

# get module
texture_mapper_module = texture_mapper
render_net_module = render_net

# use multi-GPU
if opt.gpu_id != '':
    texture_mapper = nn.DataParallel(texture_mapper)
    render_net = nn.DataParallel(render_net)
    interpolater = nn.DataParallel(interpolater)

# set to training mode
texture_mapper.train()
render_net.train()
interpolater.train()

# collect all networks
part_list = [texture_mapper_module, render_net_module]
part_name_list = ['texture_mapper', 'render_net']

print("*" * 100)
print("Number of parameters:")
print("texture mapper:")
opt.num_params_texture_mapper = util.print_network(texture_mapper)
print("render net:")
opt.num_params_render_net = util.print_network(render_net)
print("*" * 100)


def main():
    print('Start buffering data for training views...')
    view_dataset.buffer_all()
    view_dataloader = DataLoader(view_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = 8)

    print('Start buffering data for validation views...')
    view_val_dataset.buffer_all()
    view_val_dataloader = DataLoader(view_val_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = 8)

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

    # Save all command line arguments into a txt file in the logging directory for later reference.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    writer = SummaryWriter(log_dir)

    iter = opt.start_epoch * len(view_dataset)

    print('Begin training...')
    val_log_batch_id = 0
    first_val = True
    for epoch in range(opt.start_epoch, opt.max_epoch):
        for view_trgt in view_dataloader:
            start = time.time()
            # get view data
            uv_map = view_trgt[0]['uv_map'].to(device) # [N, H, W, 2]
            sh_basis_map = view_trgt[0]['sh_basis_map'].to(device) # [N, H, W, 9]
            alpha_map = view_trgt[0]['alpha_map'][:, None, :, :].to(device) # [N, 1, H, W]
            img_gt = []
            for i in range(len(view_trgt)):
                img_gt.append(view_trgt[i]['img_gt'].to(device))

            # sample texture
            neural_img = texture_mapper(uv_map, sh_basis_map)

            # rendering net
            outputs = render_net(neural_img, None)
            img_max_val = 2.0
            outputs = (outputs * 0.5 + 0.5) * img_max_val # map to [0, img_max_val]
            if type(outputs) is not list:
                outputs = [outputs]

            # We don't enforce a loss on the outermost 5 pixels to alleviate boundary errors, also weight loss by alpha
            alpha_map_central = alpha_map[:, :, 5:-5, 5:-5]
            for i in range(len(view_trgt)):
                outputs[i] = outputs[i][:, :, 5:-5, 5:-5] * alpha_map_central
                img_gt[i] = img_gt[i][:, :, 5:-5, 5:-5] * alpha_map_central

            # loss on final image
            loss_rn = list()
            for idx in range(len(view_trgt)):
                loss_rn.append(criterionL1(outputs[idx].contiguous().view(-1).float(), img_gt[idx].contiguous().view(-1).float()))
            loss_rn = torch.stack(loss_rn, dim = 0).mean()

            # total loss
            loss_g = loss_rn

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            # error metrics
            with torch.no_grad():
                err_metrics_batch_i = metric.compute_err_metrics_batch(outputs[0] * 255.0, img_gt[0] * 255.0, alpha_map_central, compute_ssim = False)

            # tensorboard scalar logs of training data
            writer.add_scalar("loss_g", loss_g, iter)
            writer.add_scalar("loss_rn", loss_rn, iter)
            writer.add_scalar("final_mae_valid", err_metrics_batch_i['mae_valid_mean'], iter)
            writer.add_scalar("final_psnr_valid", err_metrics_batch_i['psnr_valid_mean'], iter)

            end = time.time()
            print("Iter %07d   Epoch %03d   loss_g %0.4f   mae_valid %0.4f   psnr_valid %0.4f   t_total %0.4f" % (iter, epoch, loss_g, err_metrics_batch_i['mae_valid_mean'], err_metrics_batch_i['psnr_valid_mean'], end - start))

            # tensorboard figure logs of training data
            if not iter % opt.log_freq:
                output_final_vs_gt = []
                for i in range(len(view_trgt)):
                    output_final_vs_gt.append(outputs[i].clamp(min = 0., max = 1.))
                    output_final_vs_gt.append(img_gt[i].clamp(min = 0., max = 1.))
                    output_final_vs_gt.append((outputs[i] - img_gt[i]).abs().clamp(min = 0., max = 1.))
                output_final_vs_gt = torch.cat(output_final_vs_gt, dim = 0)
                writer.add_image("output_final_vs_gt",
                                torchvision.utils.make_grid(output_final_vs_gt,
                                                            nrow = outputs[0].shape[0],
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
                    # loop over batches
                    batch_id = 0
                    for view_val_trgt in view_val_dataloader:
                        start_val_i = time.time()

                        # get view data
                        uv_map = view_val_trgt[0]['uv_map'].to(device)  # [N, H, W, 2]
                        sh_basis_map = view_val_trgt[0]['sh_basis_map'].to(device)  # [N, H, W, 9]
                        alpha_map = view_val_trgt[0]['alpha_map'][:, None, :, :].to(device)  # [N, 1, H, W]
                        view_idx = view_val_trgt[0]['idx']

                        batch_size = alpha_map.shape[0]
                        img_h = alpha_map.shape[2]
                        img_w = alpha_map.shape[3]
                        num_view = len(view_val_trgt)
                        img_gt = []
                        for i in range(num_view):
                            img_gt.append(view_val_trgt[i]['img_gt'].to(device))

                        # sample texture
                        neural_img = texture_mapper(uv_map, sh_basis_map)

                        # rendering net
                        outputs = render_net(neural_img, None)
                        img_max_val = 2.0
                        outputs = (outputs * 0.5 + 0.5) * img_max_val  # map to [0, img_max_val]
                        if type(outputs) is not list:
                            outputs = [outputs]

                        # apply alpha
                        for i in range(num_view):
                            outputs[i] = outputs[i] * alpha_map
                            img_gt[i] = img_gt[i] * alpha_map

                        # tensorboard figure logs of validation data
                        if batch_id == val_log_batch_id:
                            output_final_vs_gt = []
                            for i in range(num_view):
                                output_final_vs_gt.append(outputs[i].clamp(min=0., max=1.))
                                output_final_vs_gt.append(img_gt[i].clamp(min=0., max=1.))
                                output_final_vs_gt.append(
                                    (outputs[i] - img_gt[i]).abs().clamp(min=0., max=1.))
                            output_final_vs_gt = torch.cat(output_final_vs_gt, dim=0)
                            writer.add_image("output_final_vs_gt_val",
                                             torchvision.utils.make_grid(output_final_vs_gt,
                                                                         nrow=batch_size,
                                                                         range=(0, 1),
                                                                         scale_each=False,
                                                                         normalize=False).cpu().detach().numpy(),
                                             iter)

                        # error metrics
                        err_metrics_batch_i_final = metric.compute_err_metrics_batch(outputs[0] * 255.0,
                                                                                     img_gt[0] * 255.0, alpha_map,
                                                                                     compute_ssim=True)

                        for i in range(batch_size):
                            for key in list(err_metrics_val.keys()):
                                if key in err_metrics_batch_i_final.keys():
                                    err_metrics_val[key].append(err_metrics_batch_i_final[key][i])

                        # save images
                        for i in range(batch_size):
                            cv2.imwrite(os.path.join(val_out_dir, str(iter).zfill(8) + '_' + str(
                                view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                                        outputs[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :,
                                        ::-1] * 255.)
                            cv2.imwrite(os.path.join(val_err_dir, str(iter).zfill(8) + '_' + str(
                                view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                                        (outputs[0] - img_gt[0]).abs().clamp(min=0., max=1.)[i, :].permute(
                                            (1, 2, 0)).cpu().detach().numpy()[:, :, ::-1] * 255.)
                            if first_val:
                                cv2.imwrite(os.path.join(val_gt_dir,
                                                         str(view_idx[i].cpu().detach().numpy()).zfill(5) + '.png'),
                                            img_gt[0][i, :].permute((1, 2, 0)).cpu().detach().numpy()[:, :,
                                            ::-1] * 255.)

                        end_val_i = time.time()
                        print("Val   batch %03d   mae_valid %0.4f   psnr_valid %0.4f   ssim_valid %0.4f   t_total %0.4f" % (
                            batch_id, err_metrics_batch_i_final['mae_valid_mean'],
                            err_metrics_batch_i_final['psnr_valid_mean'],
                            err_metrics_batch_i_final['ssim_valid_mean'], end_val_i - start_val_i))

                        batch_id += 1

                    for key in list(err_metrics_val.keys()):
                        if err_metrics_val[key]:
                            err_metrics_val[key] = np.vstack(err_metrics_val[key])
                            err_metrics_val[key + '_mean'] = err_metrics_val[key].mean()
                        else:
                            err_metrics_val[key + '_mean'] = np.nan

                    # tensorboard scalar logs of validation data
                    writer.add_scalar("final_mae_valid_val", err_metrics_val['mae_valid_mean'], iter)
                    writer.add_scalar("final_psnr_valid_val", err_metrics_val['psnr_valid_mean'], iter)
                    writer.add_scalar("final_ssim_valid_val", err_metrics_val['ssim_valid_mean'], iter)

                    first_val = False
                    val_log_batch_id = (val_log_batch_id + 1) % batch_id

                    end_val = time.time()
                    print("Val   mae_valid %0.4f   psnr_valid %0.4f   ssim_valid %0.4f   t_total %0.4f" % (
                    err_metrics_val['mae_valid_mean'], err_metrics_val['psnr_valid_mean'],
                    err_metrics_val['ssim_valid_mean'], end_val - start_val))

            iter += 1

            if iter % opt.ckp_freq == 0:
                util.custom_save(os.path.join(log_dir, 'model_epoch-%d_iter-%s.pth' % (epoch, iter)), 
                                part_list, 
                                part_name_list)

    util.custom_save(os.path.join(log_dir, 'model_epoch-%d_iter-%s.pth' % (epoch, iter)), 
                                part_list, 
                                part_name_list)


if __name__ == '__main__':
    main()
