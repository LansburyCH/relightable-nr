import cv2
import numpy as np
from skimage import transform
from glob import glob
import os
import math
from scipy.linalg import logm, norm
import scipy.io


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    center_coord_new = np.array([min_dim // 2, min_dim // 2])

    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img, center_coord, center_coord_new


def load_img(filepath, target_size=None, anti_aliasing=True, downsampling_order=None, square_crop=False):
    if filepath[-4:] == '.mat':
        img = scipy.io.loadmat(filepath)['img'][:, :, ::-1]
    elif filepath[-4:] == '.exr' or filepath[-4:] == '.hdr':
        img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

    if img is None:
        print("Error: Path %s invalid" % filepath)
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if square_crop:
        img, center_coord, center_coord_new = square_crop_img(img)
    else:
        center_coord = np.array(img.shape[:2]) // 2
        center_coord_new = center_coord

    img_crop_size = img.shape

    if target_size is not None:
        if downsampling_order == 1:
            img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_AREA)
        else:
            img = transform.resize(img, target_size,
                                   order=downsampling_order,
                                   mode='reflect',
                                   clip=False, preserve_range=True,
                                   anti_aliasing=anti_aliasing)
                                   
    return img, center_coord, center_coord_new, img_crop_size


def glob_imgs(path, exts = ['*.png', '*.jpg', '*.JPEG', '*.bmp', '*.exr', '*.hdr', '*.mat']):
    imgs = []
    for ext in exts:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def get_archimedean_spiral(sphere_radius, origin = np.array([0., 0., 0.]), num_step = 1000):
    a = 300
    r = sphere_radius
    o = origin

    translations = []

    i = a / 2
    while i > 0.:
        x = r * np.cos(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        y = r * np.sin(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        z = r * - np.sin(-np.pi / 2 + i / a * np.pi)

        xyz = np.array((x,y,z)) + o

        translations.append(xyz)
        i -= a / (2.0 * num_step)

    return translations


def interpolate_views(pose_1, pose_2, num_steps=100):
    poses = []
    for i in np.linspace(0., 1., num_steps):
        pose_1_contrib = 1 - i
        pose_2_contrib = i

        # Interpolate the two matrices
        target_pose = pose_1_contrib * pose_1 + pose_2_contrib * pose_2

        # Renormalize the rotation matrix
        target_pose[:3,:3] /= np.linalg.norm(target_pose[:3,:3], axis=0, keepdims=True)
        poses.append(target_pose)

    return poses


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_nn_ranking(poses):
    # Calculate the ranking of nearest neigbors
    parsed_poses = np.stack([pose[:3,2] for pose in poses], axis=0)
    parsed_poses /= np.linalg.norm(parsed_poses, axis=1, ord=2, keepdims=True)
    cos_sim_mat = parsed_poses.dot(parsed_poses.T)
    np.fill_diagonal(cos_sim_mat, -1.)
    nn_idcs = cos_sim_mat.argsort(axis=1).astype(int)  # increasing order
    cos_sim_mat.sort(axis=1)

    return nn_idcs, cos_sim_mat


##################################################
##### Utility function for rotation matrices - from https://github.com/akar43/lsm/blob/b09292c6211b32b8b95043f7daf34785a26bce0a/utils.py #####
##################################################


def quat2rot(q):
    '''q = [w, x, y, z]
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion'''
    eps = 1e-5
    w, x, y, z = q
    n = np.linalg.norm(q)
    s = (0 if n < eps else 2.0 / n)
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array([[1 - (yy + zz), xy - wz,
                   xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                  [xz - wy, yz + wx, 1 - (xx + yy)]])
    return R


def rot2quat(M):
    if M.shape[0] < 4 or M.shape[1] < 4:
        newM = np.zeros((4, 4))
        newM[:3, :3] = M[:3, :3]
        newM[3, 3] = 1
        M = newM

    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def euler_to_rot(theta):
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def az_el_to_rot(az, el):
    corr_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    inv_corr_mat = np.linalg.inv(corr_mat)

    def R_x(theta):
        return np.array([[1, 0, 0], [0, math.cos(theta),
                                     math.sin(theta)],
                         [0, -math.sin(theta),
                          math.cos(theta)]])

    def R_y(theta):
        return np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0],
                         [math.sin(theta), 0,
                          math.cos(theta)]])

    def R_z(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    Rmat = np.matmul(R_x(-el * math.pi / 180), R_y(-az * math.pi / 180))
    return np.matmul(Rmat, inv_corr_mat)


def rand_euler_rotation_matrix(nmax=10):
    euler = (np.random.uniform(size=(3, )) - 0.5) * nmax * 2 * math.pi / 360.0
    Rmat = euler_to_rot(euler)
    return Rmat, euler * 180 / math.pi


def rot_mag(R):
    angle = (1.0 / math.sqrt(2)) * \
        norm(logm(R), 'fro') * 180 / (math.pi)
    return angle
