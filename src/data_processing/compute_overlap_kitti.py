"""Precomputes the overlap regions for KITTI dataset,
used for computing the losses in RegTR.
"""
import argparse
import os
import pickle
import sys
sys.path.append(os.getcwd())
from easydict import EasyDict
from utils.misc import load_config
import h5py
import numpy as np
import torch
from tqdm import tqdm
from utils.pointcloud import compute_overlap
from utils.se3_numpy import se3_transform, se3_init
from matplotlib.pyplot import cm as colormap

import cvhelpers.visualization as cvv
import cvhelpers.colors as colors
from data_loaders.mkitti import KittiDataset

def process(phase, opt):
    cfg = EasyDict(load_config(opt.config))
    dataset = KittiDataset(config=cfg, phase=phase)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if opt.remove_ground:
        out_file = os.path.join(cfg.root, f'{phase}_pairs_{opt.overlap_radius}-overlapmask-no_ground.h5')
    else:
        out_file = os.path.join(cfg.root, f'{phase}_pairs_{opt.overlap_radius}-overlapmask.h5')
    print(f'Processing {phase}, output: {out_file}...')
    h5_fid = h5py.File(out_file, 'w')

    velo_2_cam0 = np.array([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
                            -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 
                            9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]).reshape(3,4)
    velo_2_cam0 = np.vstack((velo_2_cam0, np.array([0, 0, 0, 1])))
    cam0_2_velo = np.linalg.inv(velo_2_cam0)

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        src_xyz = batch['src_xyz'].cpu().numpy()[0]
        tgt_xyz = batch['tgt_xyz'].cpu().numpy()[0]
        
        pose = batch['pose'].cpu().numpy()[0]
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        pose = cam0_2_velo @ pose @ velo_2_cam0

        # Remove ground
        if opt.remove_ground:
            src_xyz = src_xyz[src_xyz[:, 2] > -1.4, :]
            tgt_xyz = tgt_xyz[tgt_xyz[:, 2] > -1.4, :]

        # print("type of src_xyz: ", type(src_xyz))
        # print("shape of src_xyz: ", src_xyz.shape)
        # print("type of tgt_xyz: ", type(tgt_xyz))
        # print("shape of tgt_xyz: ", tgt_xyz.shape)
        # print("type of pose: ", type(pose))
        # print("shape of pose: ", pose.shape)

        src_mask, tgt_mask, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_xyz),
            tgt_xyz,
            opt.overlap_radius,
        )

        h5_fid.create_dataset(f'/pair_{batch_idx}/src_mask', data=src_mask)
        h5_fid.create_dataset(f'/pair_{batch_idx}/tgt_mask', data=tgt_mask)
        h5_fid.create_dataset(f'/pair_{batch_idx}/src_tgt_corr', data=src_tgt_corr)

def check(phase, opt):
    cfg = EasyDict(load_config(opt.config))
    dataset = KittiDataset(config=cfg, phase=phase)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    velo_2_cam0 = np.array([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
                            -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 
                            9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]).reshape(3,4)
    velo_2_cam0 = np.vstack((velo_2_cam0, np.array([0, 0, 0, 1])))
    cam0_2_velo = np.linalg.inv(velo_2_cam0)

    print(velo_2_cam0)
    print(cam0_2_velo)
    min_n = 200000
    max_n = 0

    min_n_ng = 200000
    max_n_ng = 0

    min_n_35 = 200000
    max_n_35 = 0

    seq_cmp = 0
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        src_xyz = batch['src_xyz'].cpu().numpy()[0]
        tgt_xyz = batch['tgt_xyz'].cpu().numpy()[0]
        seq = batch['seq'][0]
        
        # print(src_xyz.shape)
        # print(tgt_xyz.shape)

        if src_xyz.shape[0] < min_n:
            min_n = src_xyz.shape[0]
        if src_xyz.shape[0] > max_n:
            max_n = src_xyz.shape[0]
        
        # Remove ground
        if opt.remove_ground:
            src_xyz_gnd = src_xyz[src_xyz[:, 2] > -1.4, :]
            tgt_xyz_gnd = tgt_xyz[tgt_xyz[:, 2] > -1.4, :]

        if src_xyz_gnd.shape[0] < min_n_ng:
            min_n_ng = src_xyz.shape[0]
        if src_xyz_gnd.shape[0] > max_n_ng:
            max_n_ng = src_xyz.shape[0]

        is_near_s = (src_xyz[:,0]**2 + src_xyz[:,1]**2)**0.5 < 35
        is_near_t = (tgt_xyz[:,0]**2 + tgt_xyz[:,1]**2)**0.5 < 35
        src_xyz = src_xyz[is_near_s, :]
        tgt_xyz = tgt_xyz[is_near_t, :]

        if src_xyz.shape[0] < min_n_35:
            min_n_35 = src_xyz.shape[0]
        if src_xyz.shape[0] > max_n_35:
            max_n_35 = src_xyz.shape[0]

        if seq == seq_cmp:
            continue
        else:
            print("\nmin_n: ", min_n)
            print("max_n: ", max_n)
            print("min_n_ng: ", min_n_ng)
            print("max_n_ng: ", max_n_ng)
            print("min_n_35: ", min_n_35)
            print("max_n_35: ", max_n_35)

            min_n = 200000
            max_n = 0
            min_n_ng = 200000
            max_n_ng = 0
            min_n_35 = 200000
            max_n_35 = 0
            seq_cmp = seq

        

        # pose = batch['pose'].cpu().numpy()[0]
        # pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        # pose = cam0_2_velo @ pose @ velo_2_cam0

        # src_mask, tgt_mask, src_tgt_corr = compute_overlap(
        #     se3_transform(pose, src_xyz),
        #     tgt_xyz,
        #     opt.overlap_radius,
        # )

        # visualize_result(src_xyz, tgt_xyz, np.expand_dims(src_mask, 1), np.expand_dims(tgt_mask, 1), pose)

        # raise ValueError('Check done.')

    


def visualize_result(src_xyz, tgt_xyz, src_overlap, tgt_overlap, pose, threshold: float = 0.5):
    print("src_xyz.shape", src_xyz.shape)
    print("tgt_xyz.shape", tgt_xyz.shape)
    print("src_overlap.shape", src_overlap.shape)
    print("tgt_overlap.shape", tgt_overlap.shape)
    print("pose.shape", pose.shape)

    large_pt_size = 4
    color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('coolwarm'))
    src_overlap_colors = (color_mapper.to_rgba(src_overlap[:, 0])[:, :3] * 255).astype(np.uint8)
    tgt_overlap_colors = (color_mapper.to_rgba(tgt_overlap[:, 0])[:, :3] * 255).astype(np.uint8)
    m_src = src_overlap[:, 0] > threshold
    m_tgt = tgt_overlap[:, 0] > threshold

    vis = cvv.Visualizer(
        win_size=(1600, 1000),
        num_renderers=4)

    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.GREEN),
        renderer_idx=0
    )
    vis.add_object(
        cvv.create_point_cloud(src_xyz[m_src, :], colors=src_overlap_colors[m_src, :], pt_size=large_pt_size),
        renderer_idx=0
    )

    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz[m_tgt, :], colors=tgt_overlap_colors[m_tgt, :], pt_size=large_pt_size),
        renderer_idx=1
    )

    # Before registration
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=2
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2
    )

    # After registration
    vis.add_object(
        cvv.create_point_cloud(se3_transform(pose, src_xyz), colors=colors.RED),
        renderer_idx=3
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3
    )

    vis.set_titles(['Source point cloud (with keypoints)',
                    'Target point cloud (with keypoints)',
                    'Before Registration',
                    'After Registration'])

    vis.reset_camera()
    vis.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--overlap_radius', type=float, default=1.5,
                        help='Overlap region will be sampled to this voxel size')
    parser.add_argument('--remove_ground', action='store_true', help='Remove ground points')
    opt = parser.parse_args()

    # process('train', opt)
    # process('val', opt)

    check("train", opt)