# Basic libs
import os, glob, random, copy, torch
import numpy as np
# import open3d as o3d
from scipy.spatial.transform import Rotation
import h5py
# Dataset parent class
from torch.utils.data import Dataset

from kiss_icp.pybind import kiss_icp_pybind

def voxel_down_sample(points: np.ndarray, voxel_size: float):
    _points = kiss_icp_pybind._Vector3dVector(points)
    return np.asarray(kiss_icp_pybind._voxel_down_sample(_points, voxel_size))

def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''

    '''
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

class KittiDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    DATA_FILES = {
        'train': [0,1,2,3,4,5],
        'val': [6,7],
        'test': [8,9,10]
    }

    def __init__(self, config, phase, transforms, data_augmentation=True):
        super(KittiDataset, self).__init__()
        self.config = config
        self.root = os.path.join(config.root, 'dataset')
        self.icp_path = os.path.join(config.root, 'icp')
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        # self.augment_shift_range = config.augment_shift_range
        # self.augment_scale_max = config.augment_scale_max
        # self.augment_scale_min = config.augment_scale_min

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(phase)
        self.phase = phase

        # Initiate transforms
        self.transforms = transforms

        # Load precomputed overlapping points
        pairs_fname = f'{phase}_pairs_{self.config.overlap_radius}-overlapmask.h5'
        if os.path.exists(os.path.join(self.config.root, pairs_fname)):
            self.pairs_data = h5py.File(os.path.join(self.config.root, pairs_fname), 'r')

            # print(self.pairs_data.keys())
            # raise ValueError
        else:
            print('Overlapping regions not precomputed. Run data_processing/compute_overlap_3dmatch.py to speed up data loading')
            raise ValueError

    def prepare_kitti_ply(self, phase):
        assert phase in ['train', 'val', 'test']
        all_pairs = False
        subset_names = self.DATA_FILES[phase]
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))

            ######################################
            # D3Feat script to generate test pairs
            
            if phase=="test" and all_pairs:
                for i in range(len(inames)-1):
                    self.files.append((drive_id, inames[i], inames[i+1]))

            else:
                more_than_10 = pdist > 10
                curr_time = inames[0]
                while curr_time in inames:
                    # print(curr_time)
                    # raise ValueError
                    next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                    if len(next_time) == 0:
                        curr_time += 1
                    else:
                        next_time = next_time[0] + curr_time - 1

                    if next_time in inames:
                        self.files.append((drive_id, curr_time, next_time))
                        curr_time = next_time + 1

        # remove bad pairs
        if phase == 'test':
            try:
                self.files.remove((8, 15, 58))
            except:
                pass
            # self.files.remove((8, 0, 14))
        print(f'Num_{phase}: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxelize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = to_o3d_pcd(xyz0_t)
                pcd1 = to_o3d_pcd(xyz1)
                reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                           o3d.pipelines.registration.ICPConvergenceCriteria(
                                                               max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]

        # refined pose is denoted as trans
        # print(t0, t1)
        # print("positions: ", positions[0].shape, positions[1].shape)
        # raise ValueError
        # M2 = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
        #              @ np.linalg.inv(self.velo2cam)).T
        tsfm = M2
        rot = tsfm[:3, :3]
        trans = tsfm[:3, 3][:, None]

        # voxelize the point clouds here
        # pcd0 = to_o3d_pcd(xyz0)
        # pcd1 = to_o3d_pcd(xyz1)
        # pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        # pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        # src_pcd = np.array(pcd0.points)
        # tgt_pcd = np.array(pcd1.points)

        src_pcd_input = voxel_down_sample(xyz0, self.voxel_size)
        tgt_pcd_input = voxel_down_sample(xyz1, self.voxel_size)
        # src_pcd_input = xyz0
        # tgt_pcd_input = xyz1

        # if self.pairs_data is None:
        #     src_overlap_mask, tgt_overlap_mask, src_tgt_corr = compute_overlap(
        #         se3_transform(pose, src_pcd_input),
        #         tgt_pcd_input,
        #         self.search_voxel_size,
        #     )
        # else:
        #     src_overlap_mask = np.asarray(self.pairs_data[f'pair_{idx}/src_mask'])
        #     tgt_overlap_mask = np.asarray(self.pairs_data[f'pair_{idx}/tgt_mask'])
        #     src_tgt_corr = np.asarray(self.pairs_data[f'pair_{idx}/src_tgt_corr'])


        # crop the point cloud
        if self.config.crop_radius > 0:
            radius = np.sqrt(src_pcd_input[:, 0]**2 + src_pcd_input[:, 1]**2)
            src_pcd_input = src_pcd_input[radius <= self.config.crop_radius]

            radius = np.sqrt(tgt_pcd_input[:, 0]**2 + tgt_pcd_input[:, 1]**2)
            tgt_pcd_input = tgt_pcd_input[radius <= self.config.crop_radius]

        if self.config.remove_ground:
            src_pcd_input = src_pcd_input[src_pcd_input[:, 2] > -1]
            tgt_pcd_input = tgt_pcd_input[tgt_pcd_input[:, 2] > -1]


        # print("np array: ", src_overlap_mask.shape, tgt_overlap_mask.shape)
        data = {}
        data['src_xyz'] = torch.from_numpy(src_pcd_input.astype(np.float32))
        data['tgt_xyz'] = torch.from_numpy(tgt_pcd_input.astype(np.float32))
        # data['src_overlap'] = torch.from_numpy(src_overlap_mask)
        # data['tgt_overlap'] = torch.from_numpy(tgt_overlap_mask)
        data['pose'] = torch.from_numpy(tsfm.astype(np.float32))
        data['src_path'], data['tgt_path'] = None, None

        # print("torch tensor: ", data['src_overlap'].shape, data['tgt_overlap'].shape)
        if self.transforms is not None:
            data = self.transforms(data)  # Apply data augmentation

        return data

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)