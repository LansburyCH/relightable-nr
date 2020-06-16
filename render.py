import numpy as np
import torch
import torch.nn.functional

import neural_renderer as nr

import misc
import sph_harm


def interp_vertex_attr(v_attr, faces_v_idx, face_index_map, weight_map):
    '''
    v_attr: [num_vertex, num_attr] or [N, num_vertex, num_attr]
    faces_v_idx: [N, num_face, 3]
    face_index_map: [N, H, W]
    weight_map: [N, H, W, 3, 1]
    return: [N, H, W, num_attr]
    '''
    num_attr = v_attr.shape[-1]
    batch_size = faces_v_idx.shape[0]
    if v_attr.ndimension() == 2:
        v_attr = v_attr[None, :].expand(batch_size, -1, -1)

    faces_v_attr = nr.vertex_attrs_to_faces(v_attr, faces_v_idx) # [1, num_face, 3, num_attr] or [N, num_face, 3, num_attr]
    attr_map = torch.stack([faces_v_attr[i, face_index_map[i, :].long()] for i in range(batch_size)]) # [N, H, W, 3, num_attr], before weighted combination
    attr_map = (attr_map * weight_map.to(attr_map.dtype)).sum(-2) # [N, H, W, num_attr], after weighted combination

    return attr_map


def texture_mapping(texture, uv_map):
    '''
    texture: torch.FloatTensor, [H, W, C]
    uv_map: torch.FloatTensor, [N, H, W, 2]
    return: torch.FloatTensor, [N, H, W, C]
    '''
    tex_h = texture.shape[0] * 1.0
    tex_w = texture.shape[1] * 1.0

    uv_map_unit_texel = uv_map
    uv_map_unit_texel[..., 0] = uv_map_unit_texel[..., 0] * (tex_w - 1)
    uv_map_unit_texel[..., 1] = uv_map_unit_texel[..., 1] * (tex_h - 1)
    uv_map_unit_texel[..., 1] = tex_h - 1 - uv_map_unit_texel[..., 1]
    img = misc.interpolate_bilinear(texture, uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1])

    return img


def lp_mapping(lp, dir_map, alpha_map):
    '''
    lp: torch.FloatTensor, [H, W, C]
    dir_map: torch.FloatTensor, [3, ...]
    alpha_map: torch.FloatTensor, [1, ...] or [3, ...]
    return: torch.FloatTensor, [..., C]
    '''
    uv_map = render.spherical_mapping(dir_map) # [2, ...]
    uv_map = uv_map * alpha_map - (alpha_map == 0).to(dir_map.dtype) # [2, ...], mask out unused regions
    sample_img = misc.interpolate_bilinear(lp, uv_map[0, :] * float(lp.shape[1] - 1), uv_map[1, :] * float(lp.shape[0] - 1))
    return sample_img


def sample_light_dir(azi_deg, pol_deg):
    '''
    azi_deg: torch.FloatTensor, [num_azi]
    pol_deg: torch.FloatTensor, [num_pol]
    return: torch.FloatTensor, [3, num_sample]
    '''
    # sample light directions on sphere in z-up space
    l_azi, l_pol = torch.meshgrid([azi_deg, pol_deg])
    l_azi = l_azi * np.pi / 180.0
    l_pol = l_pol * np.pi / 180.0
    l_ele = np.pi / 2.0 - l_pol
    # convert to cartesian coordinates
    l_dir_x, l_dir_y, l_dir_z = sph_harm.sph2cart(l_azi, l_ele, 1.0)
    l_dir = torch.stack((l_dir_x, l_dir_y, l_dir_z), dim = 0) # [3, num_azi, num_ele]
    l_dir = torch.nn.functional.normalize(l_dir, dim = 0)
    l_dir_zup = l_dir.clone()
    l_dir_zup = torch.flatten(l_dir_zup, start_dim = 1, end_dim = -1) # [3, num_sample]
    # transform to world space (z-out space)
    l_dir_temp = l_dir.clone()
    l_dir[1, :] = l_dir_temp[2, :]
    l_dir[2, :] = -l_dir_temp[1, :]
    l_dir = torch.flatten(l_dir, start_dim = 1, end_dim = -1) # [3, num_sample]
    return l_dir, l_dir_zup


def spherical_mapping(l_dir):
    '''
    l_dir: torch.FloatTensor, [3, ...]
    return: torch.FloatTensor, [2, ...]
    '''
    lp_samples_uv = torch.stack((torch.atan2(l_dir[2, :], l_dir[0, :]) * 0.5 / np.pi + 0.5, torch.acos(l_dir[1, :]) * 1.0 / np.pi), dim = 0) # [2, ...]
    return lp_samples_uv


def spherical_mapping_batch(l_dir):
    '''
    l_dir: torch.FloatTensor, [N, 3, ...]
    return: torch.FloatTensor, [N, 2, ...]
    '''
    lp_samples_uv = torch.stack((torch.atan2(l_dir[:, 2, :], l_dir[:, 0, :]) * 0.5 / np.pi + 0.5, torch.acos(l_dir[:, 1, :]) * 1.0 / np.pi), dim = 1) # [N, 2, ...]
    return lp_samples_uv


def spherical_mapping_inv(lp_samples_uv):
    '''
    lp_samples_uv: torch.FloatTensor, [2, num_sample]
    return: torch.FloatTensor, [3, num_sample]
    '''
    l_dir_y = torch.cos(lp_samples_uv[1, :] * np.pi)
    l_dir_xz_norm_sqrt = (1 - l_dir_y ** 2).sqrt()
    temp = lp_samples_uv[0, :] * 2 - 1
    l_dir_x = l_dir_xz_norm_sqrt * torch.cos(temp * np.pi)
    l_dir_z = l_dir_xz_norm_sqrt * torch.sin(temp * np.pi)
    # handle situations where pytorch produces nonzeros when should be zeros
    l_dir_z = l_dir_z * ((~(temp == 1.0)).to(l_dir_xz_norm_sqrt.dtype) * 2 - 1)
    l_dir_z = l_dir_z * ((~(temp == -1.0)).to(l_dir_xz_norm_sqrt.dtype) * 2 - 1)

    l_dir = torch.nn.functional.normalize(torch.stack((l_dir_x, l_dir_y, l_dir_z), dim = 0), dim = 0)

    return l_dir


def get_TBN_map(normal_map, face_index_map, faces_v = None, faces_texcoord = None, tangent = None):
    '''
    compute transformation matrix from tangent space to world space per pixel
    normal_map: [N, H, W, 3]
    face_index_map: [N, H, W]
    faces_v: [num_face, 3, 3]
    faces_texcoord: [num_face, 3, 2]
    tangent: [num_face, 3]
    return: [N, H, W, 3, 3]
    '''
    batch_size = face_index_map.shape[0]    

    if tangent is None:
        assert (faces_v is not None and faces_texcoord is not None)
        # get tangent vector per face
        edge1 = faces_v[:, 1, :] - faces_v[:, 0, :] # [num_face, 3]
        edge2 = faces_v[:, 2, :] - faces_v[:, 0, :]
        delta_uv1 = faces_texcoord[:, 1, :] - faces_texcoord[:, 0, :] # [num_face, 2]
        delta_uv2 = faces_texcoord[:, 2, :] - faces_texcoord[:, 0, :]
        f = 1.0 / (delta_uv1[:, 0] * delta_uv2[:, 1] - delta_uv2[:, 0] * delta_uv1[:, 1]).clamp(min = 1e-8) # [num_face]
        tangent = f[:, None] * (delta_uv2[:, 1:2] * edge1 - delta_uv1[:, 1:2] * edge2) # [num_face, 3]
        if torch.isnan(tangent).sum() > 0:
            raise ValueError('nan value detected')
    tangent = torch.nn.functional.normalize(tangent, dim = -1)
    if torch.isnan(tangent).sum() > 0:
        raise ValueError('nan value detected')

    # compute tangent map
    tangent_map = torch.stack([tangent[face_index_map[i, :].long()] for i in range(batch_size)]) # [N, H, W, 3]

    # compute bitangent map
    normal_map = torch.nn.functional.normalize(normal_map, dim = -1)
    bitangent_map = torch.cross(normal_map, tangent_map, dim = -1)
    bitangent_map = torch.nn.functional.normalize(bitangent_map, dim = -1)

    # recompute tangent map to be perpendicular to normal map
    tangent_map = torch.cross(bitangent_map, normal_map, dim = -1)
    tangent_map = torch.nn.functional.normalize(tangent_map, dim = -1)

    # form TBN matrix per pixel
    TBN_map = torch.stack((tangent_map, bitangent_map, normal_map), dim = 4) # [N, H, W, 3, 3]
    if torch.isnan(TBN_map).sum() > 0:
        raise ValueError('nan value detected')

    return TBN_map


def get_TBN_map_perpixel(normal_map, position_map, uv_map, alpha_map):
    '''
    compute transformation matrix from tangent space to world space per pixel
    normal_map: [N, H, W, 3]
    position_map: [N, H, W, 3]
    uv_map: [N, H, W, 2]
    alpha_map: [N, H, W, 1]
    return: [N, H, W, 3, 3]
    '''
    batch_size, img_h, img_w, _ = position_map.shape
    device = position_map.device

    data_map = torch.cat((position_map, uv_map), dim = -1)

    alpha_x0 = ((torch.cat((alpha_map[:, :, 1:, :], torch.zeros((batch_size, img_h, 1, 1), device = device)), dim = 2) * alpha_map) != 0).to(normal_map.dtype)
    alpha_x1 = ((alpha_x0 == 0) & (alpha_map != 0)).to(normal_map.dtype)
    alpha_y0 = ((torch.cat((alpha_map[:, 1:, :, :], torch.zeros((batch_size, 1, img_w, 1), device = device)), dim = 1) * alpha_map) != 0).to(normal_map.dtype)
    alpha_y1 = ((alpha_y0 == 0) & (alpha_map != 0)).to(normal_map.dtype)

    edge_x = data_map[:, :, 1:, :] - data_map[:, :, :-1, :]
    edge_x0 = torch.cat((edge_x, torch.zeros((batch_size, img_h, 1, 5), device = device)), dim = 2)
    edge_x1 = torch.cat((torch.zeros((batch_size, img_h, 1, 5), device = device), edge_x), dim = 2)
    edge_x = alpha_x0 * edge_x0 + alpha_x1 * edge_x1

    edge_y = data_map[:, 1:, :, :] - data_map[:, :-1, :, :]
    edge_y0 = torch.cat((edge_y, torch.zeros((batch_size, 1, img_w, 5), device = device)), dim = 1)
    edge_y1 = torch.cat((torch.zeros((batch_size, 1, img_w, 5), device = device), edge_y), dim = 1)
    edge_y = alpha_y0 * edge_y0 + alpha_y1 * edge_y1

    delta_pos1 = edge_x[:, :, :, :3]
    delta_uv1 = edge_x[:, :, :, 3:]
    delta_pos2 = edge_y[:, :, :, :3]
    delta_uv2 = edge_y[:, :, :, 3:]

    f = 1.0 / (delta_uv1[..., 0] * delta_uv2[..., 1] - delta_uv2[..., 0] * delta_uv1[..., 1]) # [N, H, W]

    # compute tangent map
    tangent_map = f[..., None] * (delta_uv2[..., 1:2] * delta_pos1 - delta_uv1[..., 1:2] * delta_pos2) # [N, H, W, 3]
    tangent_map = torch.nn.functional.normalize(tangent_map, dim = -1)

    # compute bitangent map
    bitangent_map = f[..., None] * (-delta_uv2[..., 0:1] * delta_pos1 + delta_uv1[..., 0:1] * delta_pos2) # [N, H, W, 3]
    bitangent_map = torch.nn.functional.normalize(bitangent_map, dim = -1)

    # TODO: force tangent, bitangent to be perpendicular to normal

    # form TBN matrix per pixel
    TBN_map = torch.stack((tangent_map, bitangent_map, normal_map), dim = 4) # [N, H, W, 3, 3]

    return TBN_map