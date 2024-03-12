"""Generic model for registration"""

import os
from abc import ABC

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cvhelpers.torch_helpers import to_numpy
from models.generic_model import GenericModel
from models.scheduler.warmup import WarmUpScheduler
from benchmark.benchmark_predator import benchmark as benchmark_predator
from benchmark.benchmark_3dmatch import benchmark_dgr 
import benchmark.benchmark_modelnet as benchmark_modelnet
from utils.misc import StatsMeter, metrics_to_string
from utils.se3_torch import se3_compare, se3_transform

def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return:
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2)
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs


class GenericRegModel(GenericModel, ABC):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.loss_stats_meter = StatsMeter()  # For accumulating losses
        self.reg_success_thresh_rot = cfg.reg_success_thresh_rot
        self.reg_success_thresh_trans = cfg.reg_success_thresh_trans

    def configure_optimizers(self):  # override

        scheduler_type = self.cfg.get('scheduler', None)
        if scheduler_type is None or scheduler_type in ['none', 'step']:
            base_lr = self.cfg.base_lr
        elif scheduler_type == 'warmup':
            base_lr = 0.0  # start from 0
        else:
            raise NotImplementedError

        # Create optimizer
        if self.cfg.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr,
                                               weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=base_lr,
                                              weight_decay=self.cfg.weight_decay)
        else:
            raise NotImplementedError

        # Create scheduler
        if scheduler_type == 'warmup':
            # Warmup, then smooth exponential decay
            self.scheduler = WarmUpScheduler(self.optimizer, self.cfg.scheduler_param, self.cfg.base_lr)
        elif scheduler_type == 'step':
            # Step decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.scheduler_param[0],
                                                             self.cfg.scheduler_param[1])
        elif scheduler_type == 'none' or scheduler_type is None:
            # No decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, 1.0)
        else:
            raise AssertionError('Invalid scheduler')

        self.logger.info(f'Using optimizer {self.optimizer} with scheduler {self.scheduler}')

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)
        
        # Stores the losses for summary writing
        for k in losses:
            self.loss_stats_meter[k].update(losses[k])

        # visualize_registration(batch, pred)
        return pred, losses

    def train_summary_fn(self, writer: SummaryWriter, step: int,
                         data_batch, train_output, train_losses):

        losses_dict = {k: self.loss_stats_meter[k].avg for k in self.loss_stats_meter}
        self._generic_summary_function(writer, step, model=self, losses=losses_dict)
        self.loss_stats_meter.clear()

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)
        metrics = self._compute_metrics(pred, batch)

        # visualize_registration(batch, pred, metrics=metrics, iter_idx=5, b=2)

        val_outputs = (losses, metrics)

        return val_outputs

    def validation_epoch_end(self, validation_step_outputs):

        losses = [v[0] for v in validation_step_outputs]
        metrics = [v[1] for v in validation_step_outputs]

        loss_keys = set(losses[0].keys())
        losses_stacked = {k: torch.stack([l[k] for l in losses]) for k in loss_keys}

        # Computes the mean over all metrics
        avg_losses = {k: torch.mean(losses_stacked[k]) for k in loss_keys}
        avg_metrics = self._aggregate_metrics(metrics)
        return avg_metrics['reg_success_final'].item(), {'losses': avg_losses, 'metrics': avg_metrics}

    def validation_summary_fn(self, writer: SummaryWriter, step: int, val_outputs):
        """Logs data during validation. This function will be called after every
        validation run.
        The default implementation saves out the scalars from losses and metrics.

        Args:
            writer: validation writer
            step: The current step number
            val_outputs: Whatever that is returned from validation_epoch_end()

        """
        super().validation_summary_fn(writer, step, val_outputs)

        # Save histogram summaries
        metrics = val_outputs['metrics']
        for k in metrics:
            if k.endswith('hist'):
                writer.add_histogram(f'metrics/{k}', metrics[k], step)

    def compute_IR(self, src_points, tgt_points, pose):
        src_points_tf = se3_transform(pose, src_points)
        residuals = torch.linalg.norm(tgt_points - src_points_tf, dim=1)
        inlier_masks = torch.lt(residuals, self.cfg.acceptance_radius)

        return inlier_masks.float().sum()/src_points.shape[0]

    def compute_FMR(self, IR_list):
        # print(IR_list)
        mask = torch.gt(torch.Tensor(IR_list), 0.05)
        return mask.float().sum()/len(IR_list)

    def test_epoch_start(self):
        if self.cfg.dataset == '3dmatch':
            self.IR_list = []
        if self.cfg.dataset == 'modelnet':
            self.modelnet_metrics = []
            self.modelnet_poses = []

        if self.cfg.dataset == 'kitti':
            self.kitti_metrics_rot = []
            self.kitti_metrics_trans = []

    def test_step(self, batch, batch_idx):
        pred = self.forward(batch)
        # losses = self.compute_loss(pred, batch)
        losses = 0
        metrics = self._compute_metrics(pred, batch, batch_idx)

        # Dataset specific handling
        if self.cfg.dataset == '3dmatch':
            self._save_3DMatch_log(batch, pred)
            # self.IR_list.append(self.compute_IR(pred['src_corr'][0], pred['tgt_corr'][0], batch['pose'][0]))

        elif self.cfg.dataset == 'modelnet':
            if self.cfg.model in ["qk_regtr.RegTR", "qk_regtr_old.RegTR", "qk_regtr_modelnet_lowe.RegTR", "qk_regtr_overlap.RegTR", "qk_regtr_full.RegTR"]:
                modelnet_data = {
                    'points_src': torch.stack(batch['src_xyz']),
                    'points_ref': torch.stack(batch['tgt_xyz']),
                    'points_raw': torch.stack(batch['tgt_raw']),
                    'transform_gt': batch['pose'],
                }
                self.modelnet_metrics.append(
                    benchmark_modelnet.compute_metrics(modelnet_data, pred['pose'][-1])
                )
                self.modelnet_poses.append(
                    pred['pose'][-1]
                )
            elif self.cfg.model in ["qk_mink.RegTR", "qk_mink_2.RegTR", "qk_mink_3.RegTR"]:
                modelnet_data = {
                    'points_src': batch['src_xyz'],
                    'points_ref': batch['tgt_xyz'],
                    'points_raw': batch['tgt_raw'],
                    'transform_gt': batch['pose'],
                }
                self.modelnet_metrics.append(
                    benchmark_modelnet.compute_metrics(modelnet_data, pred['pose'][-1])
                )
                self.modelnet_poses.append(
                    pred['pose'][-1]
                )
            elif self.cfg.model in ["qk_revvit.RegTR", "qk_revvit_2.RegTR", "qk_ce.RegTR"]:
                modelnet_data = {
                    'points_src': torch.permute(batch['src_xyz'], (0, 2, 1)),
                    'points_ref': torch.permute(batch['tgt_xyz'], (0, 2, 1)),
                    'points_raw': torch.permute(batch['tgt_raw'], (0, 2, 1)),
                    'transform_gt': batch['pose'],
                }
                self.modelnet_metrics.append(
                    benchmark_modelnet.compute_metrics(modelnet_data, pred['pose'][-1])
                )
                self.modelnet_poses.append(
                    pred['pose'][-1]
                )

        elif self.cfg.dataset == 'kitti':

            if self.cfg.model=="regtr.RegTR":
                if metrics['rot_err_deg'][-1] < self.cfg.reg_success_thresh_rot and metrics['trans_err'][-1] < self.cfg.reg_success_thresh_trans:
                    self.kitti_metrics_rot.append(metrics['rot_err_deg'][-1]) 
                    self.kitti_metrics_trans.append(metrics['trans_err'][-1])
                else:
                    print('Failed registration')
                    print(metrics['rot_err_deg'][-1], metrics['trans_err'][-1])
            else:
                if metrics['rot_err_deg'] < self.cfg.reg_success_thresh_rot and metrics['trans_err'] < self.cfg.reg_success_thresh_trans:
                    self.kitti_metrics_rot.append(metrics['rot_err_deg']) 
                    self.kitti_metrics_trans.append(metrics['trans_err'])
                else:
                    # visualize_registration(batch, pred, metrics=metrics, iter_idx=5, b=2)
                    print('Failed registration')
                    print(metrics['rot_err_deg'], metrics['trans_err'])

        else:
            raise NotImplementedError

        test_outputs = (losses, metrics)
        return test_outputs

    def test_epoch_end(self, test_step_outputs):

        # losses = [v[0] for v in test_step_outputs]
        metrics = [v[1] for v in test_step_outputs]

        # loss_keys = losses[0].keys()
        # losses = {k: torch.stack([l[k] for l in losses]) for k in loss_keys}

        # Computes the mean over all metrics
        # avg_losses = {k: torch.mean(losses[k]) for k in loss_keys}
        avg_metrics = self._aggregate_metrics(metrics)

        log_str = 'Test ended:\n'
        # log_str += metrics_to_string(avg_losses, '[Losses]') + '\n'
        log_str += metrics_to_string(avg_metrics, '[Metrics]') + '\n'
        self.logger.info(log_str)

        if self.cfg.dataset == '3dmatch':
            # self.logger.info('\n'+"INLIER RATIO OF THE MODEL IS: "+str(torch.Tensor(self.IR_list).mean()))
            # fmr_str = "FEATURE MATCHING RECALL OF THE MODEL IS: "
            # fmr_str += str(self.compute_FMR(self.IR_list))
            # self.logger.info('\n' + fmr_str)

            # results_str, mean_precision = benchmark_dgr(
            #     os.path.join(self._log_path, self.cfg.benchmark),
            #     os.path.join('datasets', '3dmatch', 'benchmarks', self.cfg.benchmark))
            # self.logger.info('\n' + results_str)

            # Evaluate 3DMatch registration recall
            results_str, mean_precision = benchmark_predator(
                os.path.join(self._log_path, self.cfg.benchmark),
                os.path.join('datasets', '3dmatch', 'benchmarks', self.cfg.benchmark))
            self.logger.info('\n' + results_str)
            return mean_precision

        elif self.cfg.dataset == 'modelnet':
            metric_keys = self.modelnet_metrics[0].keys()
            metrics_cat = {k: np.concatenate([m[k] for m in self.modelnet_metrics])
                           for k in metric_keys}
            summary_metrics = benchmark_modelnet.summarize_metrics(metrics_cat)
            benchmark_modelnet.print_metrics(self.logger, summary_metrics)

            # Also save out the predicted poses, which can be evaluated using
            # RPMNet's eval.py
            poses_to_save = to_numpy(torch.stack(self.modelnet_poses, dim=0))
            np.save(os.path.join(self._log_path, 'pred_transforms.npy'), poses_to_save)

        elif self.cfg.dataset == 'kitti':
            benchmark_str = "KITTI BENCHMARK RESULTS: \n"
            benchmark_str += "ROTATION ERROR: " + str(torch.Tensor(self.kitti_metrics_rot).mean()) + "TRANSLATION ERROR: " + str(torch.Tensor(self.kitti_metrics_trans).mean())
            self.logger.info('\n' + benchmark_str)

    def _compute_metrics(self, pred, batch, idx=0):

        metrics = {}
        with torch.no_grad():
            
            pose_keys = [k for k in pred.keys() if k.startswith('pose')]
            for k in pose_keys:
                suffix = k[4:]
                # print(pred[k].shape)
                # print(batch['pose'].shape)
                # rot_est = pred[k][:, :3, :3].cpu().numpy()
                # rot_gt = batch['pose'][:, :3, :3].cpu().numpy()
                # r_deviation = get_angle_deviation(rot_est, rot_gt)
                # trans_est = pred[k][:, :3, 3].cpu().numpy()
                # trans_gt = batch['pose'][:, :3, 3].cpu().numpy()
                # translation_error = np.linalg.norm(trans_est - trans_gt, axis=-1)
                # metrics[f'rot_err_deg{suffix}'] = torch.Tensor(r_deviation).unsqueeze(0)
                # metrics[f'trans_err{suffix}'] = torch.Tensor(translation_error).unsqueeze(0)
                # print(translation_error)
                
                pose_err = se3_compare(pred[k], batch['pose'][None, :])
                metrics[f'rot_err_deg{suffix}'] = pose_err['rot_deg']
                metrics[f'trans_err{suffix}'] = pose_err['trans']
                # print(pose_err['trans'].shape)
                # print(pose_err['rot_deg'].shape)
                # print(torch.Tensor(r_deviation).shape)
                # raise ValueError
                
                # print("\nidx: ", idx, " rot_err is: ", pose_err['rot_deg'], " tras_err is: ", pose_err['trans'])
                # metrics[f'chamfer_distance{suffix}'] = pose_err['chamfer']

        return metrics

    def _aggregate_metrics(self, metrics):

        if len(metrics[0]) == 0:
            return {}

        batch_dim = 1  # dim=1 is batch dimension (0 is decoder layer)
        metrics_keys = set(metrics[0].keys())
        metrics_cat = {k: torch.cat([m[k] for m in metrics], dim=batch_dim) for k in metrics_keys}
        num_instances = next(iter(metrics_cat.values())).shape[batch_dim]
        self.logger.info(f'Aggregating metrics, total number of instances: {num_instances}')
        assert all([metrics_cat[k].shape[batch_dim] == num_instances for k in metrics_keys]), \
            'Dimensionality incorrect, check whether batch dimension is consistent'

        rot_err_keys = [k for k in metrics_cat.keys() if k.startswith('rot_err_deg')]
        if len(rot_err_keys) > 0:
            num_pred = metrics_cat[rot_err_keys[0]].shape[0]

        avg_metrics = {}
        for p in range(num_pred):
            suffix = f'{p}' if p < num_pred - 1 else 'final'

            for rk in rot_err_keys:
                pose_type_suffix = rk[11:]

                avg_metrics[f'rot_err_deg{pose_type_suffix}_{suffix}'] = torch.mean(metrics_cat[rk][p])
                avg_metrics[f'rot_err{pose_type_suffix}_{suffix}_hist'] = metrics_cat[rk][p]

                tk = 'trans_err' + pose_type_suffix
                avg_metrics[f'{tk}_{suffix}'] = torch.mean(metrics_cat[tk][p])
                avg_metrics[f'{tk}_{suffix}_hist'] = metrics_cat[tk][p]

                reg_success = torch.logical_and(metrics_cat[rk][p, :] < self.reg_success_thresh_rot,
                                                metrics_cat[tk][p, :] < self.reg_success_thresh_trans)
                avg_metrics[f'reg_success{pose_type_suffix}_{suffix}'] = reg_success.float().mean()

                rot_success = metrics_cat[rk][p, :] < self.reg_success_thresh_rot
                trans_success = metrics_cat[tk][p,:] < self.reg_success_thresh_trans

                avg_metrics[f'rot_success{pose_type_suffix}_{suffix}'] = rot_success.float().mean()
                avg_metrics[f'trans_success{pose_type_suffix}_{suffix}'] = trans_success.float().mean()

            if 'corr_err' in metrics_cat:
                avg_metrics[f'corr_err_{suffix}_hist'] = metrics_cat['corr_err'][p].flatten()
                avg_metrics[f'corr_err_{suffix}'] = torch.mean(metrics_cat['corr_err'][p])

        return avg_metrics

    @property
    def _log_path(self):
        return self.get_trainer().log_path


    """
    Dataset specific functions
    """
    def _save_3DMatch_log(self, batch, pred):
        B = len(batch['src_xyz'])

        for b in range(B):
            scene = batch['src_path'][b].split(os.path.sep)[1]
            src_idx = int(os.path.basename(batch['src_path'][b]).split('_')[-1].replace('.pth', ''))
            tgt_idx = int(os.path.basename(batch['tgt_path'][b]).split('_')[-1].replace('.pth', ''))

            pred_pose_np = to_numpy(pred['pose'][-1][b]) if pred['pose'].ndim == 4 else \
                to_numpy(pred['pose'][b])
            if pred_pose_np.shape[0] == 3:
                pred_pose_np = np.concatenate([pred_pose_np, [[0., 0., 0., 1.]]], axis=0)

            scene_folder = os.path.join(self._log_path, self.cfg.benchmark, scene)
            os.makedirs(scene_folder, exist_ok=True)
            est_log_path = os.path.join(scene_folder, 'est.log')
            with open(est_log_path, 'a') as fid:
                # We don't know the number of frames, so just put -1
                # This will be ignored by the benchmark function in any case
                fid.write('{}\t{}\t{}\n'.format(tgt_idx, src_idx, -1))
                for i in range(4):
                    fid.write('\t'.join(map('{0:.12f}'.format, pred_pose_np[i])) + '\n')
