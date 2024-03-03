"""
Define criterions.

Authors: Hongjie Fang.
"""
import mindspore.nn as mnn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from utils.functions import get_surface_normal_from_depth


class Criterion(mnn.Cell):
    """
    Various type of criterions.
    """
    def __init__(self, type, combined_smooth = False, **kwargs):
        super(Criterion, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.type = str.lower(type)
        if 'huber' in self.type:
            self.huber_k = kwargs.get('huber_k', 0.1)
        self.combined_smooth = combined_smooth
        if combined_smooth:
            self.combined_beta = kwargs.get('combined_beta', 0.005)
            self.combined_beta_decay = kwargs.get('combined_beta_decay', 0.1)
            self.combined_beta_decay_milestones = kwargs.get('combined_beta_decay_milestones', [])
            self.cur_epoch = kwargs.get('cur_epoch', 0)
            for milestone in self.combined_beta_decay_milestones:
                if milestone <= self.cur_epoch:
                    self.combined_beta = self.combined_beta * self.combined_beta_decay
        self.l2_loss = self.mse_loss
        self.masked_l2_loss = self.masked_mse_loss
        self.custom_masked_l2_loss = self.custom_masked_mse_loss
        self.main_loss = getattr(self, type)
        self._mse = self._l2
    
    def step(self):
        """
        Increase current epoch by 1.
        """
        if self.combined_smooth:
            self.cur_epoch += 1
            if self.cur_epoch in self.combined_beta_decay_milestones:
                self.combined_beta = self.combined_beta * self.combined_beta_decay
    
    def _l1(self, pred, gt):
        """
        L1 loss in pixel-wise representations for MindSpore.
        """
        return ops.abs(pred - gt)

    def _l2(self, pred, gt):
        """
        L2 loss in pixel-wise representations for MindSpore.
        """
        return ops.square(pred - gt)

    def _huber(self, pred, gt):
        """
        Huber loss in pixel-wise representations for MindSpore.
        """
        delta = ops.abs(pred - gt)
        condition = delta <= self.huber_k
        return ops.where(condition, ops.square(delta) / 2, self.huber_k * delta - ops.square(self.huber_k) / 2)

    def mse_loss(self, data_dict, *args, **kwargs):
        """
        MSE loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return ops.ReduceMean()(self._l2(pred, gt)[mask])

    def masked_mse_loss(self, data_dict, *args, **kwargs):
        """
        Masked MSE loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return ops.ReduceMean()(self._l2(pred, gt)[mask])

    def custom_masked_mse_loss(self, data_dict, *args, **kwargs):
        """
        Custom masked MSE loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return ops.ReduceMean()(self._l2(pred, gt)[mask])

    def l1_loss(self, data_dict, *args, **kwargs):
        """
        L1 loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return ops.ReduceMean()(self._l1(pred, gt)[mask])   
    def masked_l1_loss(self, data_dict, *args, **kwargs):
        """
        Masked L1 loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return ops.ReduceMean()(self._l1(pred, gt)[mask])

    def custom_masked_l1_loss(self, data_dict, *args, **kwargs):
        """
        Custom masked L1 loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        zero_mask = data_dict['zero_mask']
        loss = self._l1(pred, gt)
        return ops.ReduceMean()(loss[mask]) + 0.01 * ops.ReduceMean()(loss[zero_mask])

    def huber_loss(self, data_dict, *args, **kwargs):
        """
        Huber loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return ops.ReduceMean()(self._huber(pred, gt)[mask])
 
    def masked_huber_loss(self, data_dict, *args, **kwargs):
        """
        Masked Huber loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return ops.ReduceMean()(self._huber(pred, gt)[mask])

    def custom_masked_huber_loss(self, data_dict, *args, **kwargs):
        """
        Custom masked huber loss for MindSpore.
        """
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return ops.ReduceMean()(self._huber(pred, gt)[mask])   


    def smooth_loss(self, data_dict, *args, **kwargs):
        """
        Smooth loss for MindSpore: surface normal loss.
        """
        # Implement custom surface normal calculation and cosine similarity
        # MindSpore does not have a direct equivalent for F.cosine_similarity.
        # The get_surface_normal_from_depth function must be adapted for MindSpore.

        pred = data_dict['pred']
        fx, fy, cx, cy = data_dict['fx'], data_dict['fy'], data_dict['cx'], data_dict['cy']
        depth_gt_sn = data_dict['depth_gt_sn']
        _, original_h, original_w = data_dict['depth_original'].shape
        mask = data_dict['loss_mask_dilated']
        zero_mask = data_dict['zero_mask_dilated']

        pred_sn = get_surface_normal_from_depth(pred, fx, fy, cx, cy, original_size = (original_w, original_h))

        # Custom cosine similarity implementation
        pred_sn_norm = ops.sqrt(ops.reduce_sum(ops.square(pred_sn), axis=1))
        depth_gt_sn_norm = ops.sqrt(ops.reduce_sum(ops.square(depth_gt_sn), axis=1))
        sn_loss = 1 - ops.reduce_sum(pred_sn * depth_gt_sn, axis=1) / (pred_sn_norm * depth_gt_sn_norm + self.epsilon)

        return ops.ReduceMean()(sn_loss[mask]) + 0.01 * ops.ReduceMean()(sn_loss[zero_mask])

    def construct(self, data_dict):
        """
        Calculate criterion given data dict for MindSpore.
        """
        loss_dict = {
            self.type: self.main_loss(data_dict)
        }
        if self.combined_smooth:
            loss_dict['smooth'] = self.smooth_loss(data_dict)
            loss_dict['loss'] = loss_dict[self.type] + self.combined_beta * loss_dict['smooth']
        else:
            loss_dict['loss'] = loss_dict[self.type]
        return loss_dict