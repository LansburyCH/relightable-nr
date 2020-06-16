import numpy as np
import torch
import pyshtools


def cart2sph(x, y, z):
    if type(x) is torch.Tensor:
        module = torch
        atan2_use = torch.atan2
    else:
        module = np
        atan2_use = np.arctan2

    azimuth = atan2_use(y, x)
    elevation = atan2_use(z, module.sqrt(x ** 2 + y ** 2))
    r = module.sqrt(x ** 2 + y ** 2 + z ** 2)

    return azimuth, elevation, r


def sph2cart(azimuth, elevation, r):
    if type(azimuth) is torch.Tensor:
        cos_azimuth = torch.cos(azimuth)
        sin_azimuth = torch.sin(azimuth)
    else:
        cos_azimuth = np.cos(azimuth)
        sin_azimuth = np.sin(azimuth)
    if type(elevation) is torch.Tensor:
        cos_elevation = torch.cos(elevation)
        sin_elevation = torch.sin(elevation)
    else:
        cos_elevation = np.cos(elevation)
        sin_elevation = np.sin(elevation)
    
    x = r * cos_elevation * cos_azimuth
    y = r * cos_elevation * sin_azimuth
    z = r * sin_elevation
    return x, y, z


def evaluate_sh_basis(lmax = 0, azi = None, pol = None, directions = None):
    '''
    Evaluate SH basis value at input directions
    lmax: int, maximum order of SH
    directions: np.ndarray, [num_sample, 3]
    azi: np.ndarray, [num_sample], in degrees
    pol: np.ndarray, [num_sample], in degrees
    return: np.ndarray, [num_sample, num_sh_basis]
    '''
    num_sh_basis = (lmax + 1) ** 2

    # convert cartesian directions to theta, phi
    if azi is None and pol is None:
        azi, ele, _ = cart2sph(directions[:, 0], directions[:, 1], directions[:, 2])
        pol = np.pi / 2.0 - ele
        azi *= 180 / np.pi
        pol *= 180 / np.pi
        num_samples = directions.shape[0]
    else:
        num_samples = azi.shape[0]

    out = np.zeros((num_samples, num_sh_basis))
    ith_sh_basis = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            sh_coeffs = pyshtools.SHCoeffs.from_zeros(l, csphase = 1, normalization = 'ortho')
            sh_coeffs.set_coeffs(1, l, m)
            out[..., ith_sh_basis] = sh_coeffs.expand(lon = azi, colat = pol)
            ith_sh_basis += 1

    return out


def fit_sh_coeff(samples, sh_basis_val):
    '''
    samples: np.ndarray or torch.Tensor, [num_sample, C] or [num_lighting, num_sample, C], should be uniformly sampled on unit sphere
    sh_basis_val: np.ndarray or torch.Tensor, [num_sample, num_basis]
    return: np.ndarray or torch.Tensor, [num_basis, C] or [num_lighting, num_basis, C]
    '''
    num_sample = samples.shape[-2]
    weight = 4. * np.pi / num_sample

    if len(samples.shape) == 2:
        sh_coeff = (samples[:, None, :] * sh_basis_val[:, :, None]).sum(-3) * weight # [num_basis, C]
    else:
        sh_coeff = (samples[:, :, None, :] * sh_basis_val[None, :, :, None]).sum(-3) * weight # [num_lighting, num_basis, C]

    return sh_coeff


def reconstruct_sh(sh_coeff, sh_basis_val):
    '''
    sh_coeff: np.ndarray or torch.Tensor, [num_basis, C] or [num_lighting, num_basis, C]
    sh_basis_val: np.ndarray or torch.Tensor, [num_sample, num_basis]
    return: np.ndarray or torch.Tensor, [num_sample, C] or [num_lighting, num_sample, C]
    '''
    if len(sh_coeff.shape) == 2:
        out = (sh_basis_val[..., None] * sh_coeff[None, :]).sum(-2) # [num_sample, C]
    else:
        out = (sh_basis_val[None, :, :, None] * sh_coeff[:, None, :, :]).sum(-2) # [num_lighting, num_sample, C]

    return out

