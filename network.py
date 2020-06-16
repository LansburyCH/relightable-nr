import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import cv2

from gcn_lib.dense import BasicConv, GraphConv4D, ResDynBlock4D, DenseDynBlock4D, DenseDilatedKnnGraph
from torch.nn import Sequential as Seq

from pytorch_prototyping.pytorch_prototyping import *

import neural_renderer as nr
import sph_harm
import misc
import render
import camera
import data_util


class TextureMapper(nn.Module):
    def __init__(self,
                texture_size,
                texture_num_ch,
                mipmap_level, 
                texture_init = None,
                fix_texture = False,
                apply_sh = False):
        '''
        texture_size: [1]
        texture_num_ch: [1]
        mipmap_level: [1]
        texture_init: torch.FloatTensor, [H, W, C]
        apply_sh: bool, [1]
        '''
        super(TextureMapper, self).__init__()

        self.register_buffer('texture_size', torch.tensor(texture_size))
        self.register_buffer('texture_num_ch', torch.tensor(texture_num_ch))
        self.register_buffer('mipmap_level', torch.tensor(mipmap_level))
        self.register_buffer('apply_sh', torch.tensor(apply_sh))

        # create textures as images
        self.textures = nn.ParameterList([])
        self.textures_size = []
        for ithLevel in range(self.mipmap_level):
            texture_size_i = np.round(self.texture_size.numpy() / (2.0 ** ithLevel)).astype(np.int)
            texture_i = torch.ones(1, texture_size_i, texture_size_i, self.texture_num_ch, dtype = torch.float32)
            if ithLevel != 0:
                texture_i = texture_i * 0.01
            # initialize texture
            if texture_init is not None and ithLevel == 0:
                print('Initialize neural texture with reconstructed texture')
                texture_i[..., :texture_init.shape[-1]] = texture_init[None, :]
                texture_i[..., texture_init.shape[-1]:texture_init.shape[-1] * 2] = texture_init[None, :]
            self.textures_size.append(texture_size_i)
            self.textures.append(nn.Parameter(texture_i))

        tex_flatten_mipmap_init = self.flatten_mipmap(start_ch = 0, end_ch = 6)
        tex_flatten_mipmap_init = torch.nn.functional.relu(tex_flatten_mipmap_init)
        self.register_buffer('tex_flatten_mipmap_init', tex_flatten_mipmap_init)

        if fix_texture:
            print('Fix neural textures.')
            for i in range(self.mipmap_level):
                self.textures[i].requires_grad = False

    def forward(self, uv_map, sh_basis_map = None, sh_start_ch = 3):
        '''
        uv_map: [N, H, W, C]
        sh_basis_map: [N, H, W, 9]
        return: [N, C, H, W]
        '''
        for ithLevel in range(self.mipmap_level):
            texture_size_i = self.textures_size[ithLevel]
            texture_i = self.textures[ithLevel]

            # vertex texcoords map in unit of texel
            uv_map_unit_texel = (uv_map * (texture_size_i - 1))
            uv_map_unit_texel[..., -1] = texture_size_i - 1 - uv_map_unit_texel[..., -1]

            # sample from texture (bilinear)
            if ithLevel == 0:
                output = misc.interpolate_bilinear(texture_i[0, :], uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1]).permute((0, 3, 1, 2)) # [N, C, H, W]
            else:
                output = output + misc.interpolate_bilinear(texture_i[0, :], uv_map_unit_texel[..., 0], uv_map_unit_texel[..., 1]).permute((0, 3, 1, 2)) # [N, C, H, W]

        # apply spherical harmonics
        if self.apply_sh and sh_basis_map is not None:
            output[:, sh_start_ch:sh_start_ch + 9, :, :] = output[:, sh_start_ch:sh_start_ch + 9, :, :] * sh_basis_map.permute((0, 3, 1, 2))

        return output

    def flatten_mipmap(self, start_ch, end_ch):
        for ithLevel in range(self.mipmap_level):
            if ithLevel == 0:
                out = self.textures[ithLevel][..., start_ch:end_ch]
            else:
                out = out + torch.nn.functional.interpolate(self.textures[ithLevel][..., start_ch:end_ch].permute(0, 3, 1, 2), size = (self.textures_size[0], self.textures_size[0]), mode = 'bilinear').permute(0, 2, 3, 1)
        return out


class Rasterizer(nn.Module):
    def __init__(self, 
                obj_fp, 
                img_size,
                global_RT = None):
        super(Rasterizer, self).__init__()
        v_attr, f_attr = nr.load_obj(obj_fp, normalization = False)
        vertices = v_attr['v']
        faces = f_attr['f_v_idx']
        vertices_texcoords = v_attr['vt']
        faces_vt_idx = f_attr['f_vt_idx']
        vertices_normals = v_attr['vn']
        faces_vn_idx = f_attr['f_vn_idx']
        self.num_vertex = vertices.shape[0]
        self.num_face = faces.shape[0]
        print('vertices shape:', vertices.shape)
        print('faces shape:', faces.shape)
        print('vertices_texcoords shape:', vertices_texcoords.shape)
        print('faces_vt_idx shape:', faces_vt_idx.shape)
        print('vertices_normals shape:', vertices_normals.shape)
        print('faces_vn_idx shape:', faces_vn_idx.shape)
        self.img_size = img_size

        # apply global_RT
        if global_RT is not None:
            vertices = torch.matmul(global_RT.to(vertices.device), torch.cat((vertices, torch.ones(self.num_vertex, 1).to(vertices.device)), dim = 1).transpose(1, 0)).transpose(1, 0)[:, :3]
            vertices_normals = torch.nn.functional.normalize(torch.matmul(global_RT[:3, :3].to(vertices.device), vertices_normals.transpose(1, 0)).transpose(1, 0), dim = 1)

        self.register_buffer('vertices', vertices[None, :, :]) # [1, num_vertex, 3]
        self.register_buffer('faces', faces[None, :, :]) # [1, num_vertex, 3]
        self.register_buffer('vertices_texcoords', vertices_texcoords[None, :, :])
        self.register_buffer('faces_vt_idx', faces_vt_idx[None, :, :])
        self.register_buffer('vertices_normals', vertices_normals[None, :, :])
        self.register_buffer('faces_vn_idx', faces_vn_idx[None, :, :])

        self.mesh_span = (self.vertices[0, :].max(dim = 0)[0] - self.vertices[0, :].min(dim = 0)[0]).max()

        # create textures
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # setup renderer
        renderer = nr.Renderer(image_size = img_size, 
                                camera_mode = 'projection',
                                orig_size = img_size,
                                near = 0.0,
                                far = 1e5)
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        renderer.anti_aliasing = False
        renderer.fill_back = False
        self.renderer = renderer

    def forward(self, proj, pose, dist_coeffs, offset, scale):
        _, depth, alpha, face_index_map, weight_map, v_uvz, faces_v_uvz, faces_v_idx = self.renderer(self.vertices, 
                                                                                            self.faces, 
                                                                                            torch.tanh(self.textures), 
                                                                                            K = proj, 
                                                                                            R = pose[:, :3, :3], 
                                                                                            t = pose[:, :3, -1, None].permute(0, 2, 1),
                                                                                            dist_coeffs = dist_coeffs,
                                                                                            offset = offset,
                                                                                            scale = scale)
        batch_size = face_index_map.shape[0]
        image_size = face_index_map.shape[1]

        # find indices of vertices on frontal face
        v_uvz[..., 0] = (v_uvz[..., 0] * 0.5 + 0.5) * depth.shape[2] # [1, num_vertex]
        v_uvz[..., 1] = (1 - (v_uvz[..., 1] * 0.5 + 0.5)) * depth.shape[1] # [1, num_vertex]
        v_depth = misc.interpolate_bilinear(depth[0, :, :, None], v_uvz[..., 0], v_uvz[..., 1]) # [1, num_vertex, 1]
        v_front_mask = ((v_uvz[0, :, 2] - v_depth[0, :, 0]) < self.mesh_span * 5e-3)[None, :] # [1, num_vertex]

        # perspective correct weight
        faces_v_z_inv_map = torch.cuda.FloatTensor(batch_size, image_size, image_size, 3).fill_(0.0)
        for i in range(batch_size):
            faces_v_z_inv_map[i, ...] = 1 / faces_v_uvz[i, face_index_map[i, ...].long()][..., -1]
        weight_map = (faces_v_z_inv_map * weight_map) * depth.unsqueeze_(-1) # [batch_size, image_size, image_size, 3]
        weight_map = weight_map.unsqueeze_(-1) # [batch_size, image_size, image_size, 3, 1]

        # uv map
        if self.renderer.fill_back:
            faces_vt_idx = torch.cat((self.faces_vt_idx, self.faces_vt_idx[:, :, list(reversed(range(self.faces_vt_idx.shape[-1])))]), dim=1).detach()
        else:
            faces_vt_idx = self.faces_vt_idx.detach()
        faces_vt = nr.vertex_attrs_to_faces(self.vertices_texcoords, faces_vt_idx) # [1, num_face, 3, 2]
        uv_map = faces_vt[:, face_index_map.long()].squeeze_(0) # [batch_size, image_size, image_size, 3, 2], before weighted combination
        uv_map = (uv_map * weight_map).sum(-2) # [batch_size, image_size, image_size, 2], after weighted combination
        uv_map = uv_map - uv_map.floor() # handle uv_map wrapping, keep uv in [0, 1]

        # normal map in world space
        if self.renderer.fill_back:
            faces_vn_idx = torch.cat((self.faces_vn_idx, self.faces_vn_idx[:, :, list(reversed(range(self.faces_vn_idx.shape[-1])))]), dim=1).detach()
        else:
            faces_vn_idx = self.faces_vn_idx.detach()
        faces_vn = nr.vertex_attrs_to_faces(self.vertices_normals, faces_vn_idx) # [1, num_face, 3, 3]
        normal_map = faces_vn[:, face_index_map.long()].squeeze_(0) # [batch_size, image_size, image_size, 3, 3], before weighted combination
        normal_map = (normal_map * weight_map).sum(-2) # [batch_size, image_size, image_size, 3], after weighted combination
        normal_map = torch.nn.functional.normalize(normal_map, dim = -1)

        # normal_map in camera space
        normal_map_flat = normal_map.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1))
        normal_map_cam = pose[:, :3, :3].matmul(normal_map_flat).permute((0, 2, 1)).reshape(normal_map.shape)
        normal_map_cam = torch.nn.functional.normalize(normal_map_cam, dim = -1)

        # position_map in world space
        faces_v = nr.vertex_attrs_to_faces(self.vertices, faces_v_idx) # [1, num_face, 3, 3]
        position_map = faces_v[0, face_index_map.long()] # [batch_size, image_size, image_size, 3, 3], before weighted combination
        position_map = (position_map * weight_map).sum(-2) # [batch_size, image_size, image_size, 3], after weighted combination

        # position_map in camera space
        position_map_flat = position_map.flatten(start_dim = 1, end_dim = 2).permute((0, 2, 1))
        position_map_cam = pose[:, :3, :3].matmul(position_map_flat).permute((0, 2, 1)).reshape(position_map.shape) + pose[:, :3, -1][:, None, None, :]

        return uv_map, alpha, face_index_map, weight_map, faces_v_idx, normal_map, normal_map_cam, faces_v, faces_vt, position_map, position_map_cam, depth, v_uvz, v_front_mask


class RenderingNet(nn.Module):
    def __init__(self,
                 nf0,
                 in_channels,
                 out_channels,
                 num_down_unet = 5,
                 out_channels_gcn = 512,
                 use_gcn = True,
                 outermost_highway_mode = 'concat'):
        super().__init__()

        self.register_buffer('nf0', torch.tensor(nf0))
        self.register_buffer('in_channels', torch.tensor(in_channels))
        self.register_buffer('out_channels', torch.tensor(out_channels))
        self.register_buffer('num_down_unet', torch.tensor(num_down_unet))
        self.register_buffer('out_channels_gcn', torch.tensor(out_channels_gcn))

        self.net = Unet(in_channels = in_channels,
                 out_channels = out_channels,
                 outermost_linear = True,
                 use_dropout = True,
                 dropout_prob = 0.1,
                 nf0 = nf0,
                 norm = nn.BatchNorm2d,
                 max_channels = 8 * nf0,
                 num_down = num_down_unet,
                 out_channels_gcn = out_channels_gcn,
                 use_gcn = use_gcn,
                 outermost_highway_mode = outermost_highway_mode)

        self.tanh = nn.Tanh()

    def forward(self, input, v_fea):
        x = self.net(input, v_fea)
        return self.tanh(x)


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act_type
        norm = opt.norm_type
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv_type
        c_growth = channels
        self.n_blocks = opt.n_blocks
        num_v = opt.num_v_gcn
        out_channels = opt.out_channels_gcn

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv4D(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block_type.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock4D(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        elif opt.block_type.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock4D(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = BasicConv([channels+c_growth*(self.n_blocks-1), 1024], act, None, bias)
        self.prediction = Seq(*[BasicConv([1+channels+c_growth*(self.n_blocks-1), 512, 256], act, None, bias),
                                BasicConv([256, 64], act, None, bias)])
        self.linear = Seq(*[utils.spectral_norm(nn.Linear(num_v,2048)), utils.spectral_norm(nn.Linear(2048, out_channels))])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
            elif isinstance(m,torch.nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, inputs):
        data = torch.cat((inputs.pos,inputs.x),1).unsqueeze(0).unsqueeze(-1)
        feats = [self.head(data.transpose(2,1), self.knn(data[:,:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)

        fea = self.linear(fusion.view(-1)).unsqueeze(0)
        return fea


class Interpolater(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, sub_x, sub_y):
        '''
        data: [N, H, W, C] or [1, H, W, C]
        sub_x: [N, ...]
        sub_y: [N, ...]
        return: [N, ..., C]
        '''
        if data.shape[0] == 1:
            return misc.interpolate_bilinear(data[0, :], sub_x, sub_y) # [N, ..., C]
        elif data.shape[0] == sub_x.shape[0]:
            out = []
            for i in range(data.shape[0]):
                out.append(misc.interpolate_bilinear(data[i, :], sub_x[i, :], sub_y[i, :])) # [..., C]
            return torch.stack(out) # [N, ..., C]
        else:
            raise ValueError('data.shape[0] should be 1 or batch size')


class InterpolaterVertexAttr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v_attr, faces_v_idx, face_index_map, weight_map):
        '''
        v_attr: [N, num_vertex, num_attr] or [1, num_vertex, num_attr]
        faces_v_idx: [N, num_face, 3]
        face_index_map: [N, H, W]
        weight_map: [N, H, W, 3, 1]
        return: [N, H, W, num_attr]
        '''
        return render.interp_vertex_attr(v_attr, faces_v_idx, face_index_map, weight_map)


class Mesh(nn.Module):
    def __init__(self, obj_fp, global_RT = None):
        super().__init__()
        
        # load obj
        v_attr, f_attr = nr.load_obj(obj_fp, normalization = False)
        v = v_attr['v'].cpu() # [num_vertex, 3]
        vn = v_attr['vn'].cpu() # [num_vertex, 3]
        self.num_vertex = v.shape[0]

        # compute useful infomation
        self.v_orig = v.clone()
        self.vn_orig = vn.clone()
        self.span_orig = v.max(dim = 0)[0] - v.min(dim = 0)[0]
        self.span_max_orig = self.span_orig.max()
        self.center_orig = v.mean(dim = 0)

        # apply global_RT
        if global_RT is not None:
            v = torch.matmul(global_RT.to(v.device), torch.cat((v, torch.ones(self.num_vertex, 1).to(v.device)), dim = 1).transpose(1, 0)).transpose(1, 0)[:, :3]
            vn = torch.nn.functional.normalize(torch.matmul(global_RT[:3, :3].to(vn.device), vn.transpose(1, 0)).transpose(1, 0), dim = 1)
        
        self.register_buffer('v', v)
        self.register_buffer('vn', vn)
        print('v shape:', self.v.shape)
        print('vn shape:', self.vn.shape)

        # compute useful infomation
        self.span = v.max(dim = 0)[0] - v.min(dim = 0)[0]
        self.span_max = self.span.max()
        self.center = v.mean(dim = 0)

    def forward(self):
        pass


class RaysLTChromLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rays_lt, alpha_map, img = None):
        '''
        rays_lt: [N, num_ray, C, H, W]
        alpha_map: [N, 1, H, W]
        img: [N, C, H, W]
        return: [1]
        '''
        rays_lt_chrom = torch.nn.functional.normalize(rays_lt, dim = 2) # [N, num_ray, C, H, W]
        rays_lt_chrom_mean = rays_lt_chrom.mean(dim = 1)[:, None, :, :, :] # [N, 1, C, H, W]
        rays_lt_chrom_mean = torch.nn.functional.normalize(rays_lt_chrom_mean, dim = 2) # [N, 1, C, H, W]
        rays_lt_chrom_diff = (1 - (rays_lt_chrom * rays_lt_chrom_mean).sum(2)) * alpha_map # [N, num_ray, H, W]
        if img is not None:
            # weight by image intensity
            weight = (img.norm(dim = 1, keepdim = True) * 20).clamp(max = 1.0)
            rays_lt_chrom_diff = rays_lt_chrom_diff * weight # [N, num_ray, H, W]
        loss_rays_lt_chrom = rays_lt_chrom_diff.sum() / alpha_map.sum() / rays_lt_chrom_diff.shape[1]
        return loss_rays_lt_chrom, rays_lt_chrom, rays_lt_chrom_mean, rays_lt_chrom_diff


####################################################################################################################################
################################################## Modules for Ray Based Renderer ##################################################
####################################################################################################################################
class RaySampler(nn.Module):
    def __init__(self, num_azi, num_polar, interval_polar = 5, mode = 'reflect'):
        super().__init__()

        self.register_buffer('num_azi', torch.tensor(num_azi))
        self.register_buffer('num_polar', torch.tensor(num_polar))
        self.register_buffer('interval_polar', torch.tensor(interval_polar))
        self.mode = mode

        roty_rad = np.arange(1, num_polar + 1) * interval_polar * np.pi / 180.0
        rotz_rad = np.arange(num_azi) * 2 * np.pi / num_azi
        roty_rad, rotz_rad = np.meshgrid(roty_rad, rotz_rad, sparse = False)
        roty_rad = roty_rad.flatten()
        rotz_rad = rotz_rad.flatten()
        rotx_rad = np.zeros_like(roty_rad)
        self.rot_rad = np.vstack((rotx_rad, roty_rad, rotz_rad)) # [3, num_ray]
        self.num_ray = self.rot_rad.shape[1] + 1

        Rs = np.zeros((self.num_ray, 3, 3), dtype = np.float32)
        Rs[0, :, :] = np.eye(3)
        for i in range(self.num_ray - 1):
            Rs[i + 1, :, :] = data_util.euler_to_rot(self.rot_rad[:, i])
        self.register_buffer('Rs', torch.from_numpy(Rs)) # [num_ray, 3, 3]
        
        # pivots in tangent space
        pivots_dir = torch.matmul(self.Rs, torch.FloatTensor([0, 0, 1], device = self.Rs.device)[:, None])[..., 0].permute((1, 0)) # [3, num_ray]
        self.register_buffer('pivots_dir', pivots_dir)

    def forward(self, TBN_matrices, view_dir_map_tangent, alpha_map):
        '''
        TBN_matrices: [N, ..., 3, 3]
        view_dir_map_tangent: [N, ..., 3]
        alpha_map: [N, ..., 1]
        return0: [N, ..., 3, num_ray]
        return1: [N, ..., 2, num_ray]
        return2: [N, ..., 3, num_ray]
        '''
        if self.mode == 'reflect':
            # reflect view directions around pivots
            rays_dir_tangent = camera.get_reflect_dir(view_dir_map_tangent[..., None], self.pivots_dir, dim = -2) * alpha_map[..., None] # [N, ..., 3, num_ray]
            # transform to world space
            num_ray = rays_dir_tangent.shape[-1]
            rays_dir = torch.matmul(TBN_matrices.reshape((-1, 3, 3)), rays_dir_tangent.reshape((-1, 3, num_ray))).reshape((*(TBN_matrices.shape[:-1]), -1)) # [N, ..., 3, num_ray]
        else:
            rays_dir_tangent = self.pivots_dir # [3, num_ray]
            # transform to world space
            num_ray = rays_dir_tangent.shape[-1]
            rays_dir = torch.matmul(TBN_matrices.reshape((-1, 3, 3)), rays_dir_tangent).reshape((*(TBN_matrices.shape[:-1]), -1)) # [N, ..., 3, num_ray]
        
        rays_dir = torch.nn.functional.normalize(rays_dir, dim = -2)

        # get rays uv on light probe
        rays_uv = render.spherical_mapping_batch(rays_dir.transpose(1, -2)).transpose(1, -2) # [N, ..., 2, num_ray]
        rays_uv = rays_uv * alpha_map[..., None] - (alpha_map[..., None] == 0).to(rays_dir.dtype) # [N, ..., 2, num_ray]

        return rays_dir, rays_uv, rays_dir_tangent


class RayRenderer(nn.Module):
    def __init__(self, lighting_model, interpolater):
        super().__init__()
        self.lighting_model = lighting_model
        self.interpolater = interpolater

    def forward(self, albedo_specular, rays_uv, rays_lt, lighting_idx = None, lp = None, albedo_diffuse = None, num_ray_diffuse = 0, no_albedo = False, seperate_albedo = False, lp_scale_factor = 1):
        '''
        rays_uv: [N, H, W, 2, num_ray]
        rays_lt: [N, num_ray, C, H, W]
        albedo_specular: [N, C, H, W]
        albedo_diffuse: [N, C, H, W]
        return: [N, C, H, W]
        '''
        num_ray = rays_uv.shape[-1] - num_ray_diffuse

        # get light probe
        if lp is None:
            lp = self.lighting_model(lighting_idx, is_lp = True) # [N, H, W, C]
        lp = lp * lp_scale_factor

        # get rays color
        rays_color = self.interpolater(lp, (rays_uv[..., 0, :] * float(lp.shape[2])).clamp(max = lp.shape[2] - 1), (rays_uv[..., 1, :] * float(lp.shape[1])).clamp(max = lp.shape[1] - 1)).permute((0, -2, -1, 1, 2)) # [N, num_ray, C, H, W]

        # get specular light transport map
        ltt_specular_map = (rays_lt[:, :num_ray, ...] * rays_color[:, :num_ray, ...]).sum(1) / num_ray # [N, C, H, W]
        # get specular component
        if no_albedo:
            out_specular = ltt_specular_map
        else:
            out_specular = albedo_specular * ltt_specular_map

        if num_ray_diffuse > 0:
            # get diffuse light transport map
            ltt_diffuse_map = (rays_lt[:, num_ray:, ...] * rays_color[:, num_ray:, ...]).sum(1) / num_ray_diffuse # [N, C, H, W]
            # get diffuse component
            if no_albedo:
                out_diffuse = ltt_diffuse_map
            else:
                if seperate_albedo:
                    out_diffuse = albedo_diffuse * ltt_diffuse_map
                else:
                    out_diffuse = albedo_specular * ltt_diffuse_map
        else:
            ltt_diffuse_map = torch.zeros_like(ltt_specular_map)
            out_diffuse = torch.zeros_like(out_specular)

        if out_diffuse is not None:
            out = out_specular + out_diffuse
        else:
            out = out_specular

        return out, out_specular, out_diffuse, ltt_specular_map, ltt_diffuse_map, rays_color, lp


##########################################################################################################################
################################################## Modules for Lighting ##################################################
##########################################################################################################################
# Spherical Harmonics model
class LightingSH(nn.Module):
    def __init__(self, l_dir, lmax, num_lighting = 1, num_channel = 3, init_coeff = None, fix_params = False, lp_recon_h = 100, lp_recon_w = 200):
        '''
        l_dir: torch.Tensor, [3, num_sample], sampled light directions
        lmax: int, maximum SH degree
        num_lighting: int, number of lighting
        num_channel: int, number of color channels
        init_coeff: torch.Tensor, [num_lighting, num_basis, num_channel] or [num_basis, num_channel], initial coefficients
        fix_params: bool, whether fix parameters
        '''
        super().__init__()

        self.num_sample = l_dir.shape[1]
        self.lmax = lmax
        self.num_basis = (lmax + 1) ** 2
        self.num_lighting = num_lighting
        self.num_channel = num_channel
        self.fix_params = fix_params
        self.lp_recon_h = lp_recon_h
        self.lp_recon_w = lp_recon_w

        # get basis value on sampled directions
        print('LightingSH.__init__: Computing SH basis value on sampled directions...')
        basis_val = torch.from_numpy(sph_harm.evaluate_sh_basis(lmax = lmax, directions = l_dir.cpu().detach().numpy().transpose())).to(l_dir.dtype).to(l_dir.device)
        self.register_buffer('basis_val', basis_val) # [num_sample, num_basis]

        # basis coefficients as learnable parameters
        self.coeff = nn.Parameter(torch.zeros((num_lighting, self.num_basis, num_channel), dtype = torch.float32)) # [num_lighting, num_basis, num_channel]
        # initialize basis coeffients
        if init_coeff is not None:
            if init_coeff.dim == 2:
                init_coeff = init_coeff[None, :].repeat((num_lighting, 1, 1))
            self.coeff.data = init_coeff
        # change to non-learnable
        if self.fix_params:
            self.coeff.requires_grad_(False)
        
        # precompute light samples
        l_samples = sph_harm.reconstruct_sh(self.coeff.data, self.basis_val)
        self.register_buffer('l_samples', l_samples) # [num_lighting, num_sample, num_channel]

        # precompute SH basis value for reconstructing light probe
        lp_samples_recon_v, lp_samples_recon_u = torch.meshgrid([torch.arange(start = 0, end = self.lp_recon_h, step = 1, dtype = torch.float32) / (self.lp_recon_h - 1), 
                                                                torch.arange(start = 0, end = self.lp_recon_w, step = 1, dtype = torch.float32) / (self.lp_recon_w - 1)])
        lp_samples_recon_uv = torch.stack([lp_samples_recon_u, lp_samples_recon_v]).flatten(start_dim = 1, end_dim = -1)
        lp_samples_recon_dir = render.spherical_mapping_inv(lp_samples_recon_uv).permute((1, 0)).cpu().detach().numpy()

        basis_val_recon = torch.from_numpy(sph_harm.evaluate_sh_basis(lmax = self.lmax, directions = lp_samples_recon_dir)).to(l_dir.dtype).to(l_dir.device)
        self.register_buffer('basis_val_recon', basis_val_recon) # [num_lp_pixel, num_basis]

    def forward(self, lighting_idx = None, coeff = None, is_lp = None):
        '''
        coeff: torch.Tensor, [num_lighting, num_basis, num_channel]
        return: [1, num_lighting, num_sample, num_channel] or [1, num_sample, num_channel]
        '''
        if coeff is not None:
            if is_lp:
                out = self.reconstruct_lp(coeff)[None, :] # [1, num_lighting, H, W, C]
            else:
                out = sph_harm.reconstruct_sh(coeff, self.basis_val)[None, :]
        elif lighting_idx is not None:
            if is_lp:
                out = self.reconstruct_lp(self.coeff[lighting_idx, :])[None, :] # [1, H, W, C]
            else:
                if self.fix_params:
                    out = self.l_samples[lighting_idx, ...][None, :]
                else:
                    out = sph_harm.reconstruct_sh(self.coeff[lighting_idx, ...][None, :], self.basis_val)
        else:
            if is_lp:
                out = self.reconstruct_lp(self.coeff)[None, :] # [1, num_lighting, H, W, C]
            else:
                if self.fix_params:
                    out = self.l_samples[None, :]
                else:
                    out = sph_harm.reconstruct_sh(self.coeff, self.basis_val)[None, :]

        return out

    def get_lighting_params(self, lighting_idx):
        return self.coeff[lighting_idx, :] # [num_sample, num_channel]

    def normalize_lighting(self, lighting_ref_idx):
        lighting_ref_norm = self.coeff[lighting_ref_idx, :].norm('fro')
        norm_scale_factor = lighting_ref_norm / self.coeff.norm('fro', dim = [1, 2])
        norm_scale_factor[lighting_ref_idx] = 1.0
        self.coeff *= norm_scale_factor[:, None, None]

    def reconstruct_lp(self, coeff):
        '''
        coeff: [num_basis, C] or [num_lighting, num_basis, C]
        '''
        lp_recon = sph_harm.reconstruct_sh(coeff, self.basis_val_recon).reshape((int(self.lp_recon_h), int(self.lp_recon_w), -1)) # [H, W, C] or [num_lighting, H, W, C]
        return lp_recon


# Light Probe model
class LightingLP(nn.Module):
    def __init__(self, l_dir, num_lighting = 1, num_channel = 3, lp_dataloader = None, fix_params = False, lp_img_h = 1600, lp_img_w = 3200):
        '''
        l_dir: torch.FloatTensor, [3, num_sample], sampled light directions
        num_lighting: int, number of lighting
        num_channel: int, number of color channels
        lp_dataloader: dataloader for light probes (if not None, num_lighting is ignored)
        fix_params: bool, whether fix parameters
        '''
        super().__init__()

        self.register_buffer('l_dir', l_dir) # [3, num_sample]
        self.num_sample = l_dir.shape[1]
        self.num_lighting = num_lighting
        self.num_channel = num_channel
        self.fix_params = fix_params
        self.lp_img_h = lp_img_h
        self.lp_img_w = lp_img_w
        
        if lp_dataloader is not None:
            self.num_lighting = len(lp_dataloader)

        # spherical mapping to get light probe uv
        l_samples_uv = render.spherical_mapping(l_dir)
        self.register_buffer('l_samples_uv', l_samples_uv) # [2, num_sample]

        # light samples as learnable parameters
        self.l_samples = nn.Parameter(torch.zeros((self.num_lighting, self.num_sample, self.num_channel), dtype = torch.float32)) # [num_lighting, num_sample, num_channel]
        
        # initialize light samples from light probes
        if lp_dataloader is not None:
            self.num_lighting = len(lp_dataloader)
            lp_idx = 0
            lps = []
            for lp in lp_dataloader:
                lp_img = lp['lp_img'][0, :].permute((1, 2, 0))
                lps.append(torch.from_numpy(cv2.resize(lp_img.cpu().detach().numpy(), (lp_img_w, lp_img_h), interpolation = cv2.INTER_AREA))) # [H, W, C]
                lp_img = lps[-1]
                self.l_samples.data[lp_idx, :] = misc.interpolate_bilinear(lp_img.to(self.l_samples_uv.device), (self.l_samples_uv[None, 0, :] * float(lp_img.shape[1])).clamp(max = lp_img.shape[1] - 1), (self.l_samples_uv[None, 1, :] * float(lp_img.shape[0])).clamp(max = lp_img.shape[0] - 1))[0, :]
                lp_idx += 1

            lps = torch.stack(lps)
            self.register_buffer('lps', lps) # [num_lighting, H, W, C]

        # change to non-learnable
        if self.fix_params:
            self.l_samples.requires_grad_(False)

    def forward(self, lighting_idx = None, is_lp = False):
        '''
        return: [1, num_lighting, num_sample, num_channel] or [1, num_sample, num_channel]
        '''
        if is_lp:
            if lighting_idx is None:
                return self.lps[None, :]
            else:
                return self.lps[lighting_idx, :][None, :]
        else:
            if lighting_idx is None:
                return self.l_samples[None, :]
            else:
                return self.l_samples[lighting_idx, :][None, :]

    def fit_sh(self, lmax):
        print('LightingLP.fit_sh: Computing SH basis value on sampled directions...')
        basis_val = torch.from_numpy(sph_harm.evaluate_sh_basis(lmax = lmax, directions = self.l_dir.cpu().detach().numpy().transpose())).to(self.l_dir.dtype).to(self.l_dir.device) # [num_sample, num_basis]
        sh_coeff = sph_harm.fit_sh_coeff(samples = self.l_samples.to(self.l_dir.device), sh_basis_val = basis_val) # [num_lighting, num_basis, num_channel]
        self.register_buffer('sh_coeff', sh_coeff)
        return
