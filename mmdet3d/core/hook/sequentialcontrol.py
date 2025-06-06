# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1, temporal_start_iter=-1):
        super().__init__()
        self.temporal_start_epoch = temporal_start_epoch
        self.temporal_start_iter = temporal_start_iter

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.forward_projection.with_prev = flag
        else:
            runner.model.module.forward_projection.with_prev = flag

    def set_temporal_flag_v2(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.solofusion.do_history = flag
        else:
            runner.model.module.solofusion.do_history = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)
        if self.temporal_start_iter > 0:
            self.set_temporal_flag_v2(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch and self.temporal_start_iter < 0:
            self.set_temporal_flag(runner, True)

    def after_train_iter(self, runner):

        curr_step = runner.iter
        if curr_step >= self.temporal_start_iter and self.temporal_start_iter >= 0:
            self.set_temporal_flag_v2(runner, True)