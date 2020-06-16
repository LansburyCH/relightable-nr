import argparse
import os
import cv2
import trimesh
import numpy as np
import scipy.io

import data_util


def normalize(vectors, shape = 'n*3'):
    if shape == 'n*3':
        length = np.sqrt(np.sum(vectors * vectors, axis = 1))
        vectors /= np.repeat(length[:, np.newaxis], 3, 1)
    else:
        length = np.sqrt(np.sum(vectors * vectors, axis = 0))
        vectors /= np.repeat(length[np.newaxis], 3, 0)
    return vectors


def spherical_mapping(l_dir):
    lp_samples_uv = np.stack((np.arctan2(l_dir[2, :], l_dir[0, :]) * 0.5 / np.pi + 0.5, np.arccos(l_dir[1, :]) * 1.0 / np.pi), axis = 0)
    return lp_samples_uv


def camera2ray(extrinsic, intrinsic, w, h):
    y = np.repeat(np.array(range(h)), w).reshape((h, w))+0.5
    x = np.repeat(np.array(range(w)), h).reshape((w, h)).T+0.5
    position = np.stack((x, y, np.ones(x.shape)), axis = 0)
    position = np.dot(np.linalg.inv(intrinsic), position.reshape((3, -1)))
    position = np.dot(np.linalg.inv(extrinsic[:3, :3]), position)
    position = normalize(position, '3*n').reshape((3, h, w))
    return position
    
    
def ray2sphere(vectors):
    theta, phi = np.arccos(vectors[:, 2]), np.arctan(vectors[:, 1] / vectors[:, 0])
    
    mask = vectors[:, 0] < 0
    phi[mask] += np.pi
    
    mask = np.bitwise_and(vectors[:,0] > 0, vectors[:, 1] < 0)
    phi[mask] += 2 * np.pi
    return theta, phi


def drawMask(mask, face, vertex):
    for i in range(face.shape[0]):
        mask = cv2.fillPoly(mask, [vertex[:, face[i]].T], 255)
    return mask


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='./data/material_sphere', required=False,
                    help='Path to directory that holds the data.')
parser.add_argument('--calib_fp', type=str, default='_/calib.mat', required=False,
                    help='Path of calibration file.')
parser.add_argument('--obj_fp', type=str, default='_/mesh.obj', required=False,
                    help='Path of mesh obj.')
parser.add_argument('--lighting_idx', default = 0, type = int,
                    help='Lighting index.')
parser.add_argument('--sampling_pattern', type=str, default='skipinv_10', required=False)
parser.add_argument('--img_suffix', type=str, default='.exr', required=False, help = 'Suffix of image files.')
parser.add_argument('--lp_h', type=int, default=1600, required=False, help = 'Height of output light probe.')
parser.add_argument('--lp_w', type=int, default=3200, required=False, help = 'Width of output light probe.')

opt = parser.parse_args()
if opt.calib_fp[:2] == '_/':
    opt.calib_fp = os.path.join(opt.data_root, opt.calib_fp[2:])
if opt.obj_fp[:2] == '_/':
    opt.obj_fp = os.path.join(opt.data_root, opt.obj_fp[2:])
img_dir = os.path.join(opt.data_root, 'rgb' + str(opt.lighting_idx))

# save directories
save_dir_lp = os.path.join(opt.data_root, 'light_probe_stitch_' + opt.sampling_pattern)
save_dir_lp_mask = os.path.join(save_dir_lp, 'mask')
save_dir_lp_count = os.path.join(save_dir_lp, 'count')
data_util.cond_mkdir(save_dir_lp)
data_util.cond_mkdir(save_dir_lp_mask)
data_util.cond_mkdir(save_dir_lp_count)

# load calibration and mesh
calib = scipy.io.loadmat(opt.calib_fp)
poses = calib['poses']
projs = calib['projs']
img_hws = calib['img_hws']
global_RT = calib['global_RT']
global_RT_inv = np.linalg.inv(global_RT)
num_view = poses.shape[0]

mesh = trimesh.load(opt.obj_fp, process = False)
vertices = np.dot(global_RT, np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1)))).T)

# start to stitch
print('Starting to stitch...')
env = np.zeros((opt.lp_h, opt.lp_w, 3))
count = np.zeros((opt.lp_h, opt.lp_w, 3)).astype(np.float32)

for i in range(num_view):
    if opt.sampling_pattern == 'all':
        pass
    elif opt.sampling_pattern[:5] == 'skip_':
        skip_val = int(opt.sampling_pattern.split('_')[-1])
        if i % skip_val != 0:
            continue
    elif opt.sampling_pattern[:8] == 'skipinv_':
        skip_val = int(opt.sampling_pattern.split('_')[-1])
        if i % skip_val == 0:
            continue
    elif opt.sampling_pattern[:6] == 'first_':
        first_val = int(opt.sampling_pattern.split('_')[-1])
        if i >= first_val:
            continue

    print('View', i)
    img_w, img_h = img_hws[i, 1], img_hws[i, 0]
    pose = np.dot(poses[i, ...], global_RT_inv)
    proj = projs[i, ...]

    if opt.img_suffix == '.exr':
        img = cv2.imread(img_dir + ('/%03d' % i) + opt.img_suffix, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(img_dir + ('/%06d' % i) + opt.img_suffix, cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :, :3] / 255.
    
    vertice = np.dot(pose, vertices)
    vertice = np.dot(proj, vertice[:3])
    vertice[0] /= vertice[2] 
    vertice[1] /= vertice[2]
    vertice = vertice.astype('int32')
    vertice[vertice < 0] = 0
    vertice[0, vertice[0] > (int(img_w) - 1)] = int(img_w) - 1
    vertice[1, vertice[1] > (int(img_h) - 1)] = int(img_h) - 1

    mask = np.zeros((int(img_h), int(img_w)))
    mask = drawMask(mask, mesh.faces, vertice[:2])
    kernel = np.ones((17, 17), np.uint8)
    mask = cv2.resize(cv2.dilate(cv2.resize(mask, (512, 512)), kernel), (int(img_w), int(img_h)))

    mask = (mask == 0)
    
    vec = camera2ray(pose, proj, int(img_w), int(img_h))
    lp_samples_uv = spherical_mapping(vec[:, mask])

    lp_samples_uv[0] = (lp_samples_uv[0] * (opt.lp_w)).clip(max = opt.lp_w - 1.0)
    lp_samples_uv[1] = (lp_samples_uv[1] * (opt.lp_h)).clip(max = opt.lp_h - 1.0)
    lp_samples_uv = np.round(lp_samples_uv).astype('int')

    env[lp_samples_uv[1], lp_samples_uv[0]] += img[mask][:, :3]
    count[lp_samples_uv[1], lp_samples_uv[0]] += 1
    
mask = np.sum(count, axis = 2) > 0
env[mask] /= count[mask]

# save
cv2.imwrite(save_dir_lp + '/' + str(opt.lighting_idx) + '.png', (env * 255).astype('uint8'))
cv2.imwrite(save_dir_lp + '/' + str(opt.lighting_idx) + '.exr', env.astype(np.float32))
cv2.imwrite(save_dir_lp_mask + '/' + str(opt.lighting_idx) + '.png', (mask * 255).astype('uint8'))
cv2.imwrite(save_dir_lp_count + '/' + str(opt.lighting_idx) + '.png', (count / float(num_view) * 255.0).astype('uint8'))
scipy.io.savemat(save_dir_lp_count + '/' + str(opt.lighting_idx) + '.mat', {'count': count[:, :, 0].astype(np.int), 'num_view': num_view})