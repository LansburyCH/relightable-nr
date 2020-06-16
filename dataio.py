import os
import torch
import numpy as np
import numpy.linalg
import cv2
import scipy.io

import data_util


class ViewDataset():
    def __init__(self,
                 root_dir,
                 calib_path,
                 calib_format,
                 img_size,
                 sampling_pattern,
                 load_img = True,
                 img_dir = None,
                 ignore_dist_coeffs = True,
                 load_precompute = False,
                 precomp_high_dir = None,
                 precomp_low_dir = None,
                 img_gamma = 1.0):
        super().__init__()

        self.root_dir = root_dir
        self.calib_format = calib_format
        self.img_size = img_size
        self.ignore_dist_coeffs = ignore_dist_coeffs
        self.load_img = load_img
        self.load_precompute = load_precompute
        self.precomp_high_dir = precomp_high_dir
        self.precomp_low_dir = precomp_low_dir
        self.img_gamma = img_gamma

        if not os.path.isdir(root_dir):
            raise ValueError("Error! root dir is wrong")

        self.img_dir = img_dir
        if self.load_img and not os.path.isdir(self.img_dir):
            raise ValueError("Error! image dir is wrong")

        # load calibration data
        if calib_format == 'convert':
            if not os.path.isfile(calib_path):
                raise ValueError("Error! calib path is wrong")
            self.calib = scipy.io.loadmat(calib_path)
            self.global_RT = self.calib['global_RT']
            num_view = self.calib['poses'].shape[0]
        else:
            raise ValueError('Unknown calib format')
        self.global_RT_inv = np.linalg.inv(self.global_RT)

        # get path for all input images
        if self.load_img:
            self.img_fp_all = sorted(data_util.glob_imgs(self.img_dir))
        else:
            self.img_fp_all = ['x.x'] * num_view

        # get intrinsic/extrinsic of all input images
        self.poses_all = []
        img_fp_all_new = []
        for idx in range(len(self.img_fp_all)):
            img_fn = os.path.split(self.img_fp_all[idx])[-1]
            self.poses_all.append(self.calib['poses'][idx, :, :])
            img_fp_all_new.append(self.img_fp_all[idx])

        # remove views without calibration result
        self.img_fp_all = img_fp_all_new

        # Subsample data
        keep_idx = []
        if sampling_pattern == 'all':
            keep_idx = list(range(len(self.img_fp_all)))
        else:
            if sampling_pattern == 'filter':
                img_fp_all_new = []
                poses_all_new = []
                for idx in self.calib['keep_id'][0, :]:
                    img_fp_all_new.append(self.img_fp_all[idx])
                    poses_all_new.append(self.poses_all[idx])
                    keep_idx.append(idx)
                self.img_fp_all = img_fp_all_new
                self.poses_all = poses_all_new
            elif sampling_pattern.split('_')[0] == 'first':
                first_val = int(sampling_pattern.split('_')[-1])
                self.img_fp_all = self.img_fp_all[:first_val]
                self.poses_all = self.poses_all[:first_val]
                keep_idx = list(range(first_val))
            elif sampling_pattern.split('_')[0] == 'after':
                after_val = int(sampling_pattern.split('_')[-1])
                keep_idx = list(range(after_val, len(self.img_fp_all)))
                self.img_fp_all = self.img_fp_all[after_val:]
                self.poses_all = self.poses_all[after_val:]
            elif sampling_pattern.split('_')[0] == 'skip':
                skip_val = int(sampling_pattern.split('_')[-1])
                img_fp_all_new = []
                poses_all_new = []
                for idx in range(0, len(self.img_fp_all), skip_val):
                    img_fp_all_new.append(self.img_fp_all[idx])
                    poses_all_new.append(self.poses_all[idx])
                    keep_idx.append(idx)
                self.img_fp_all = img_fp_all_new
                self.poses_all = poses_all_new
            elif sampling_pattern.split('_')[0] == 'skipinv':
                skip_val = int(sampling_pattern.split('_')[-1])
                img_fp_all_new = []
                poses_all_new = []
                for idx in range(0, len(self.img_fp_all)):
                    if idx % skip_val == 0:
                        continue
                    img_fp_all_new.append(self.img_fp_all[idx])
                    poses_all_new.append(self.poses_all[idx])
                    keep_idx.append(idx)
                self.img_fp_all = img_fp_all_new
                self.poses_all = poses_all_new
            elif sampling_pattern.split('_')[0] == 'only':
                choose_idx = int(sampling_pattern.split('_')[-1])
                self.img_fp_all = [self.img_fp_all[choose_idx]]
                self.poses_all = [self.poses_all[choose_idx]]
                keep_idx.append(choose_idx)
            else:
                raise ValueError("Unknown sampling pattern!")

        if self.calib_format == 'convert':
            self.calib['img_hws'] = self.calib['img_hws'][keep_idx, ...]
            self.calib['projs'] = self.calib['projs'][keep_idx, ...]
            self.calib['poses'] = self.calib['poses'][keep_idx, ...]
            self.calib['dist_coeffs'] = self.calib['dist_coeffs'][keep_idx, ...]

        # get mapping from img_fn to idx and vice versa
        self.img_fn2idx = {}
        self.img_idx2fn = []
        for idx in range(len(self.img_fp_all)):
            img_fn = os.path.split(self.img_fp_all[idx])[-1]
            self.img_fn2idx[img_fn] = idx
            self.img_idx2fn.append(img_fn)

        print("*" * 100)
        print("Sampling pattern ", sampling_pattern)
        print("Image size ", self.img_size)
        print("*" * 100)


    def buffer_all(self):
        # Buffer files
        print("Buffering files...")
        self.views_all = []
        for i in range(self.__len__()):
            if not i % 50:
                print('Data', i)
            self.views_all.append(self.read_view(i))


    def buffer_one(self):
        self.views_all = []
        self.views_all.append(self.read_view(0))


    def read_view(self, idx):
        img_fp = self.img_fp_all[idx]
        img_fn = os.path.split(img_fp)[-1]

        # image size
        if self.calib_format == 'convert':
            img_hw = self.calib['img_hws'][idx, :]

        # get view image
        if self.load_img:
            img_gt, center_coord, center_coord_new, img_crop_size = data_util.load_img(img_fp, square_crop = True, downsampling_order = 1, target_size = self.img_size)
            img_gt = img_gt[:, :, :3]
            img_gt = img_gt.transpose(2,0,1)
            img_gt = img_gt ** self.img_gamma
        else:
            min_dim = np.amin(img_hw)
            center_coord = img_hw // 2
            center_coord_new = np.array([min_dim // 2, min_dim // 2])
            img_crop_size = np.array([min_dim, min_dim])

        # extrinsic
        pose = self.poses_all[idx]
        pose = np.dot(pose, self.global_RT_inv)

        # intrinsic
        proj = self.calib['projs'][idx, :, :]
        dist_coeffs = self.calib['dist_coeffs'][idx, :]
        if self.ignore_dist_coeffs:
            dist_coeffs[:] = 0.0

        proj_orig = proj.copy()
        offset = np.array([center_coord_new[0] - center_coord[0], center_coord_new[1] - center_coord[1]], dtype = np.float32)
        scale = np.array([self.img_size[0] * 1.0 / (img_crop_size[0] * 1.0), self.img_size[1] * 1.0 / (img_crop_size[1] * 1.0)], dtype = np.float32)
        proj[0, -1] = (proj[0, -1] + offset[1]) * scale[1]
        proj[1, -1] = (proj[1, -1] + offset[0]) * scale[0]
        proj[0, 0] *= scale[1]
        proj[1, 1] *= scale[0]
        view_dir = -pose[2, :3]

        proj_inv = numpy.linalg.inv(proj)
        R_inv = pose[:3, :3].transpose()

        view = {'proj_orig': torch.from_numpy(proj_orig.astype(np.float32)),
                'proj': torch.from_numpy(proj.astype(np.float32)),
                'pose': torch.from_numpy(pose.astype(np.float32)),
                'dist_coeffs': torch.from_numpy(dist_coeffs.astype(np.float32)),
                'offset': torch.from_numpy(offset),
                'scale': torch.from_numpy(scale),
                'view_dir': torch.from_numpy(view_dir.astype(np.float32)),
                'proj_inv': torch.from_numpy(proj_inv.astype(np.float32)),
                'R_inv': torch.from_numpy(R_inv.astype(np.float32)),
                'idx': idx,
                'img_fn': img_fn}

        if self.load_img:
            view['img_gt'] = torch.from_numpy(img_gt)

        # load precomputed data
        if self.load_precompute:
            # cannot share across meshes
            raster = scipy.io.loadmat(os.path.join(self.precomp_low_dir, 'resol_' + str(self.img_size[0]), 'raster', img_fn.split('.')[0] + '.mat'))
            view['face_index_map'] = torch.from_numpy(raster['face_index_map'])
            view['weight_map'] = torch.from_numpy(raster['weight_map'])
            view['faces_v_idx'] = torch.from_numpy(raster['faces_v_idx'])
            view['v_uvz'] = torch.from_numpy(raster['v_uvz'])
            view['v_front_mask'] = torch.from_numpy(raster['v_front_mask'])[0, :]

            # can share across meshes
            TBN_map = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'TBN_map', img_fn.split('.')[0] + '.mat'))['TBN_map']
            view['TBN_map'] = torch.from_numpy(TBN_map)
            uv_map = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'uv_map', img_fn.split('.')[0] + '.mat'))['uv_map']
            uv_map = uv_map - np.floor(uv_map) # keep uv in [0, 1]
            view['uv_map'] = torch.from_numpy(uv_map)
            normal_map = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'normal_map', img_fn.split('.')[0] + '.mat'))['normal_map']
            view['normal_map'] = torch.from_numpy(normal_map)
            view_dir_map = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'view_dir_map', img_fn.split('.')[0] + '.mat'))['view_dir_map']
            view['view_dir_map'] = torch.from_numpy(view_dir_map)
            view_dir_map_tangent = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'view_dir_map_tangent', img_fn.split('.')[0] + '.mat'))['view_dir_map_tangent']
            view['view_dir_map_tangent'] = torch.from_numpy(view_dir_map_tangent)
            sh_basis_map = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'sh_basis_map', img_fn.split('.')[0] + '.mat'))['sh_basis_map'].astype(np.float32)
            view['sh_basis_map'] = torch.from_numpy(sh_basis_map)
            reflect_dir_map = scipy.io.loadmat(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'reflect_dir_map', img_fn.split('.')[0] + '.mat'))['reflect_dir_map']
            view['reflect_dir_map'] = torch.from_numpy(reflect_dir_map)
            alpha_map = cv2.imread(os.path.join(self.precomp_high_dir, 'resol_' + str(self.img_size[0]), 'alpha_map', img_fn.split('.')[0] + '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            view['alpha_map'] = torch.from_numpy(alpha_map)

        return view


    def __len__(self):
        return len(self.img_fp_all)


    def __getitem__(self, idx):
        view_trgt = []

        # Read one target pose
        view_trgt.append(self.views_all[idx])

        return view_trgt


class LightProbeDataset():
    def __init__(self,
                 data_dir,
                 img_gamma = 1.0):
        super().__init__()

        self.data_dir = data_dir
        self.img_gamma = img_gamma

        if not os.path.isdir(data_dir):
            raise ValueError("Error! data dir is wrong")

        # get path for all light probes
        self.lp_fp_all = sorted(data_util.glob_imgs(self.data_dir))

        self.lp_all = [None] * len(self.lp_fp_all)


    def buffer_one(self, idx):
        if self.lp_all[idx] is not None:
            return
            
        # get light probe
        lp_fp = self.lp_fp_all[idx]
        print(lp_fp)
        if lp_fp[-4:] == '.exr' or lp_fp[-4:] == '.hdr':
            lp_img = cv2.imread(lp_fp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            lp_img = cv2.imread(lp_fp, cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32) / 255.0
        lp_img = cv2.cvtColor(lp_img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
        lp_img = lp_img ** self.img_gamma

        lp = {'lp_img': torch.from_numpy(lp_img)}

        self.lp_all[idx] = lp


    def buffer_all(self):
        for idx in range(len(self.lp_fp_all)):
            self.buffer_one(idx)


    def __len__(self):
        return len(self.lp_fp_all)


    def __getitem__(self, idx):
        self.buffer_one(idx)
        return self.lp_all[idx]
