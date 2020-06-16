import numpy as np
import math
import pytorch_msssim
import torch


def psnr(img1, img2, mask = None):
    if mask is None:
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    else:
        num_valid_ele = mask.sum(dtype = np.float64)
        mse = np.sum((img1 / 255. - img2 / 255.) ** 2 * mask) / num_valid_ele
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_err_metrics(img_est, img_gt, mask, compute_ssim = True):
    """
    :param img_est: numpy.ndarray or torch.Tensor, (H, W, 3)
    :param img_gt: numpy.ndarray or torch.Tensor, (H, W, 3)
    :param mask: numpy.ndarray or torch.Tensor, (H, W)
    :return: dict
    """
    # convert types
    if type(mask) is torch.Tensor:
        mask = mask.cpu().detach().numpy()
    mask = mask == 1

    img_est[mask == 0] = 0
    img_gt[mask == 0] = 0

    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis].repeat(3, axis = 2).astype(np.float32)
    if type(img_est) is torch.Tensor:
        img_est_torch = img_est[None, :].permute((0, 3, 1, 2))
        img_est = img_est.cpu().detach().numpy()
    else:
        img_est_torch = torch.FloatTensor(img_est)[None, :].permute((0, 3, 1, 2))
    if type(img_gt) is torch.Tensor:
        img_gt_torch = img_gt[None, :].permute((0, 3, 1, 2))
        img_gt = img_gt.cpu().detach().numpy()
    else:
        img_gt_torch = torch.FloatTensor(img_gt)[None, :].permute((0, 3, 1, 2))

    # get bounding box region
    suby, subx = (mask[:, :, 0] == 1).nonzero()
    bb_xmin = min(subx)
    bb_xmax = max(subx) + 1
    bb_ymin = min(suby)
    bb_ymax = max(suby) + 1
    img_est_bb = img_est[bb_ymin:bb_ymax, bb_xmin:bb_xmax, :]
    img_gt_bb = img_gt[bb_ymin:bb_ymax, bb_xmin:bb_xmax, :]
    img_est_bb_torch = torch.FloatTensor(img_est_bb)[None, :].permute((0, 3, 1, 2))
    img_gt_bb_torch = torch.FloatTensor(img_gt_bb)[None, :].permute((0, 3, 1, 2))

    # compute absolute difference
    img_diff = np.abs(img_est - img_gt)
    img_diff_bb = img_diff[bb_ymin:bb_ymax, bb_xmin:bb_xmax, :]

    # compute error metrics
    num_valid_ele = mask.sum(dtype = np.float64)
    err_metrics = {}
    err_metrics['mae'] = img_diff.mean(dtype = np.float64)
    err_metrics['mae_bb'] = img_diff_bb.mean(dtype = np.float64)
    err_metrics['mae_valid'] = (img_diff * mask).sum(dtype = np.float64) / num_valid_ele
    err_metrics['mse'] = (img_diff ** 2).mean(dtype = np.float64)
    err_metrics['mse_bb'] = (img_diff_bb ** 2).mean(dtype = np.float64)
    err_metrics['mse_valid'] = (img_diff ** 2 * mask).sum(dtype = np.float64) / num_valid_ele

    err_metrics['psnr'] = psnr(img_est, img_gt)
    err_metrics['psnr_bb'] = psnr(img_est_bb, img_gt_bb)
    err_metrics['psnr_valid'] = psnr(img_est, img_gt, mask = mask)

    if compute_ssim:
        err_metrics['ssim'] = pytorch_msssim.ssim(img_est_torch, img_gt_torch, data_range = 255, size_average = False).cpu().detach().numpy()[0]
        err_metrics['ssim_bb'] = pytorch_msssim.ssim(img_est_bb_torch, img_gt_bb_torch, data_range = 255, size_average = False).cpu().detach().numpy()[0]
        mask_bb_inverse = mask[bb_ymin:bb_ymax, bb_xmin:bb_xmax, 0] != 1
        img_est_bb_modify = img_est_bb
        img_est_bb_modify[mask_bb_inverse] = img_gt_bb[mask_bb_inverse]
        err_metrics['ssim_valid'] = pytorch_msssim.ssim(torch.FloatTensor(img_est_bb_modify[None, :].transpose((0, 3, 1, 2))), img_gt_bb_torch, data_range = 255, size_average = False).cpu().detach().numpy()[0]

    return err_metrics


def compute_err_metrics_batch(img_est, img_gt, mask, compute_ssim = True):
    """
    :param img_est: torch.Tensor, (N, 3, H, W)
    :param img_gt: torch.Tensor, (N, 3, H, W)
    :param mask: torch.Tensor, (N, 1, H, W)
    :return: dict
    """
    num_batch = img_est.shape[0]
    err_metrics = {}
    err_metrics['mae'] = []
    err_metrics['mae_bb'] = []
    err_metrics['mae_valid'] = []
    err_metrics['mse'] = []
    err_metrics['mse_bb'] = []
    err_metrics['mse_valid'] = []
    err_metrics['psnr'] = []
    err_metrics['psnr_bb'] = []
    err_metrics['psnr_valid'] = []
    err_metrics['ssim'] = []
    err_metrics['ssim_bb'] = []
    err_metrics['ssim_valid'] = []

    for i in range(num_batch):
        err_metrics_data_i = compute_err_metrics(img_est[i, :].permute((1, 2, 0)), img_gt[i, :].permute((1, 2, 0)), mask[i, 0, :], compute_ssim = compute_ssim)
        for key in err_metrics.keys():
            if key in err_metrics_data_i.keys():
                err_metrics[key].append(err_metrics_data_i[key])

    for key in list(err_metrics.keys()):
        if err_metrics[key]:
            err_metrics[key] = np.vstack(err_metrics[key])
            err_metrics[key + '_mean'] = err_metrics[key].mean()
        else:
            err_metrics[key + '_mean'] = np.nan

    return err_metrics