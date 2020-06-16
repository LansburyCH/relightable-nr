from __future__ import division
import os

import torch
import numpy as np
from skimage.io import imread

import neural_renderer.cuda.load_textures as load_textures_cuda

texture_wrapping_dict = {'REPEAT': 0, 'MIRRORED_REPEAT': 1,
                         'CLAMP_TO_EDGE': 2, 'CLAMP_TO_BORDER': 3}

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    return colors, texture_filenames


def load_textures(filename_obj, filename_mtl, texture_size, texture_wrapping='REPEAT', use_bilinear=True):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces]
    faces = torch.from_numpy(faces).cuda()

    colors, texture_filenames = load_mtl(filename_mtl)

    textures = torch.zeros(faces.shape[0], texture_size, texture_size, texture_size, 3, dtype=torch.float32) + 0.5
    textures = textures.cuda()

    #
    for material_name, color in colors.items():
        color = torch.from_numpy(color).cuda()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :, :, :] = color[None, None, None, :]

    for material_name, filename_texture in texture_filenames.items():
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = imread(filename_texture).astype(np.float32) / 255.

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3,-1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:,:,:3]

        # pytorch does not support negative slicing for the moment
        image = image[::-1, :, :]
        image = torch.from_numpy(image.copy()).cuda()
        is_update = (np.array(material_names) == material_name).astype(np.int32)
        is_update = torch.from_numpy(is_update).cuda()
        textures = load_textures_cuda.load_textures(image, faces, textures, is_update,
                                                    texture_wrapping_dict[texture_wrapping],
                                                    use_bilinear)
    return textures

def load_obj(filename_obj, normalization=True, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True, use_cuda = True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # read in all lines
    with open(filename_obj) as f:
        lines = f.readlines()

    # load vertices
    vertices = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(val) for val in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32))
    if use_cuda:
        vertices = vertices.cuda()

    # load vertex normal
    vertices_normal = []
    has_vn = False
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vn':
            vertices_normal.append([float(val) for val in line.split()[1:4]])
    if vertices_normal:
        vertices_normal = torch.from_numpy(np.vstack(vertices_normal).astype(np.float32))
        if use_cuda:
            vertices_normal = vertices_normal.cuda()
        has_vn = True

    # load vertex texture coordinates
    vertices_texcoords = []
    has_vt = False
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices_texcoords.append([float(val) for val in line.split()[1:3]])
    if vertices_texcoords:
        vertices_texcoords = torch.from_numpy(np.vstack(vertices_texcoords).astype(np.float32))
        if use_cuda:
            vertices_texcoords = vertices_texcoords.cuda()
        has_vt = True

    # load faces
    faces = []
    faces_vn_idx = []
    faces_vt_idx = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            face = [int(vs[i].split('/')[0]) for i in range(nv)]
            faces.append(face)
            if has_vt:
                face_vt_idx = [int(vs[i].split('/')[1]) for i in range(nv)]
                faces_vt_idx.append(face_vt_idx)
            if has_vn:
                face_vn_idx = [int(vs[i].split('/')[-1]) for i in range(nv)]
                faces_vn_idx.append(face_vn_idx)
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)) - 1
    faces_vn_idx = torch.from_numpy(np.vstack(faces_vn_idx).astype(np.int32)) - 1
    faces_vt_idx = torch.from_numpy(np.vstack(faces_vt_idx).astype(np.int32)) - 1
    if use_cuda:
        faces = faces.cuda()
        faces_vn_idx = faces_vn_idx.cuda()
        faces_vt_idx = faces_vt_idx.cuda()

    # load textures
    textures = None
    if load_texture:
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures = load_textures(filename_obj, filename_mtl, texture_size,
                                         texture_wrapping=texture_wrapping,
                                         use_bilinear=use_bilinear)
        if textures is None:
            raise Exception('Failed to load textures.')

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    v_attr = {'v' : vertices, 'vn' : vertices_normal, 'vt' : vertices_texcoords}
    f_attr = {'f_v_idx' : faces, 'f_vn_idx' : faces_vn_idx, 'f_vt_idx' : faces_vt_idx}

    if load_texture:
        return v_attr, f_attr, textures 
    else:
        return v_attr, f_attr
