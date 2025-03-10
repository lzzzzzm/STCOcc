# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

from mmdet3d.core.hook.utils import is_parallel

__all__ = ['Loss_Hook']


@HOOKS.register_module()
class Loss_Hook(Hook):
    """Loss_Hook used in BEVDepth.
    """

    def __init__(self, update_iter=1000):
        super().__init__()
        self.update_iter = update_iter

    def after_train_iter(self, runner):
        scene_loss = runner.model.module.scene_loss
        group_weights = runner.data_loader.iter_loader._index_sampler.group_weights
        # utilize scene_loss to update group_weights
        curr_step = runner.iter
        if curr_step % self.update_iter == 0 and curr_step > 0:
            group_size = runner.data_loader.iter_loader._index_sampler.group_sizes
            # sort scene_loss by it key
            scene_loss = dict(sorted(scene_loss.items(), key=lambda x: x[0]))
            scene_loss_value = torch.tensor(list(scene_loss.values()))
            # normalize scene_loss_value
            scene_loss_value = scene_loss_value / group_size
            # update group_weights
            runner.data_loader.iter_loader._index_sampler.group_weights = scene_loss_value


        # curr_step = runner.iter
        # loss_weight = max(0.2, (1.0 - curr_step / self.total_iter))
        # model = runner.model.module
        # model.sem_scal_loss_weight = loss_weight
