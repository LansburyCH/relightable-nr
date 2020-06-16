import numpy as np
import torch


def interpolate_bilinear(data, sub_x, sub_y):
    '''
    data: [H, W, C]
    sub_x: [...]
    sub_y: [...]
    return: [..., C]
    '''
    device = data.device

    mask_valid = ((sub_x >= 0) & (sub_x <= data.shape[1] - 1) & (sub_y >= 0) & (sub_y <= data.shape[0] - 1)).to(data.dtype).to(device)

    x0 = torch.floor(sub_x).long().to(device)
    x1 = x0 + 1
    
    y0 = torch.floor(sub_y).long().to(device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, data.shape[1] - 1)
    x1 = torch.clamp(x1, 0, data.shape[1] - 1)
    y0 = torch.clamp(y0, 0, data.shape[0] - 1)
    y1 = torch.clamp(y1, 0, data.shape[0] - 1)
    
    I00 = data[y0, x0, :] # [..., C]
    I10 = data[y1, x0, :]
    I01 = data[y0, x1, :]
    I11 = data[y1, x1, :]

    # right boundary
    x0 = x0 - (x0 == x1).to(x0.dtype)
    # bottom boundary
    y0 = y0 - (y0 == y1).to(y0.dtype)

    w00 = (x1.to(data.dtype) - sub_x) * (y1.to(data.dtype) - sub_y) * mask_valid # [...]
    w10 = (x1.to(data.dtype) - sub_x) * (sub_y - y0.to(data.dtype)) * mask_valid
    w01 = (sub_x - x0.to(data.dtype)) * (y1.to(data.dtype) - sub_y) * mask_valid
    w11 = (sub_x - x0.to(data.dtype)) * (sub_y - y0.to(data.dtype)) * mask_valid

    return I00 * w00.unsqueeze_(-1) + I10 * w10.unsqueeze_(-1) + I01 * w01.unsqueeze_(-1) + I11 * w11.unsqueeze_(-1)


def interpolate_bilinear_np(data, sub_x, sub_y):
    '''
    data: [H, W, C]
    sub_x: [...]
    sub_y: [...]
    return: [..., C]
    '''
    x0 = np.floor(sub_x).astype(np.int64)
    x1 = x0 + 1
    
    y0 = np.floor(sub_y).astype(np.int64)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, data.shape[1] - 1)
    x1 = np.clip(x1, 0, data.shape[1] - 1)
    y0 = np.clip(y0, 0, data.shape[0] - 1)
    y1 = np.clip(y1, 0, data.shape[0] - 1)
    
    I00 = data[y0, x0, :] # [..., C]
    I10 = data[y1, x0, :]
    I01 = data[y0, x1, :]
    I11 = data[y1, x1, :]

    w00 = (x1 - sub_x) * (y1 - sub_y) # [...]
    w10 = (x1 - sub_x) * (sub_y - y0)
    w01 = (sub_x - x0) * (y1 - sub_y)
    w11 = (sub_x - x0) * (sub_y - y0)

    return I00 * w00[..., None] + I10 * w10[..., None] + I01 * w01[..., None] + I11 * w11[..., None]
