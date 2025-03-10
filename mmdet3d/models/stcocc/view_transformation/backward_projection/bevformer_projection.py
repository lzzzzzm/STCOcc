# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import numpy as np

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

from mmdet.models import HEADS
from mmdet.models.utils import build_transformer


@HEADS.register_module()
class BEVFormerBackwardProjection(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 transformer=None,
                 positional_encoding=None,
                 pc_range=None,
                 bev_h=100,
                 bev_w=100,
                 bev_z=8,
                 use_can_bus=False,
                 can_bus_norm=True,
                 **kwargs):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.grid_length_w = self.real_w / self.bev_w
        self.grid_length_h = self.real_h / self.bev_h
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the DeformDETR head."""
        if self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            if self.can_bus_norm:
                self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def obtain_shift_from_ego(self, bev_queries, img_metas):
        delta_x = np.array([each['can_bus'][0] for each in img_metas])
        delta_y = np.array([each['can_bus'][1] for each in img_metas])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in img_metas])
        grid_length_y = self.grid_length_h
        grid_length_x = self.grid_length_w
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
        shift = bev_queries.new_tensor(np.array([shift_x, shift_y])).permute(1, 0)  # xy, bs -> bs, xy
        return shift


    @force_fp32(apply_to=('mlvl_feats'))
    def forward(self,
                mlvl_feats,
                img_metas,
                voxel_feats=None,
                cam_params=None,
                pred_img_depth=None,
                bev_mask=None,
                last_occ_pred=None,
                prev_bev=None,
                prev_bev_aug=None,
                history_fusion_params=None,
                **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam = cam_params[0].shape[0], cam_params[0].shape[1]
        mlvl_feats[0] = mlvl_feats[0].reshape(bs, num_cam, *mlvl_feats[0].shape[-3:])
        pred_img_depth = pred_img_depth.reshape(bs, num_cam, *pred_img_depth.shape[-3:])

        dtype = mlvl_feats[0].dtype

        # voxel_feats shape: [bs, c, z, h, w] or bev_feats shape: [bs, c, h, w]
        if voxel_feats.dim() == 5:
            voxel_feats = voxel_feats.mean(2)
        lss_bev = voxel_feats.clone()
        lss_bev = lss_bev.flatten(2).permute(2, 0, 1)
        bev_queries = lss_bev

        # Deal with temporal fusion
        shift = self.obtain_shift_from_ego(bev_queries, img_metas)

        # check the same bev aug
        if prev_bev is not None:
            # shape: [bs, c, z, h, w]
            prev_bev = prev_bev.mean(2)
            curr_bev_aug = cam_params[-1]
            aug = torch.matmul(prev_bev_aug, curr_bev_aug)
            for i in range(bs):
                flip_dx, flip_dy = aug[i, 0, 0], aug[i, 1, 1]
                if flip_dx:
                    prev_bev[i] = torch.flip(prev_bev[i], [2])
                if flip_dy:
                    prev_bev[i] = torch.flip(prev_bev[i], [1])

            prev_bev = prev_bev.flatten(2).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]

        # check the start of sequence
        if history_fusion_params is not None and prev_bev is not None:
            if type(history_fusion_params['start_of_sequence']) is list:
                start_of_sequence = history_fusion_params['start_of_sequence'][0]
            else:
                start_of_sequence = history_fusion_params['start_of_sequence']

            for i in range(len(start_of_sequence)):
                if start_of_sequence[i] is True:
                    prev_bev[:, i] = bev_queries[:, i]

        # add can bus signals
        if self.use_can_bus:
            if type(img_metas) is not list:
                img_metas = [img_metas]
            if 'can_bus' in img_metas[0]:
                can_bus = bev_queries.new_tensor(
                    [each['can_bus'] for each in img_metas])  # [:, :]
                can_bus = self.can_bus_mlp(can_bus)[None, :, :]
                bev_queries = bev_queries + can_bus * self.use_can_bus

        if bev_mask is not None:
            bev_mask = bev_mask.reshape(bs, -1)

        if last_occ_pred is not None:
            # last_occ_pred: [bs, w, h, z, c]
            last_occ_pred = last_occ_pred.permute(0, 4, 3, 1, 2)    # [bs, c, z, w, h]
            last_occ_pred = F.interpolate(last_occ_pred, scale_factor=2, mode='trilinear', align_corners=False)
            last_occ_pred = last_occ_pred.permute(0, 3, 4, 2, 1)    # [bs, w, h, z, c]

        bev_pos = self.positional_encoding.forward_bda(bs, self.bev_h, self.bev_w, bev_queries.device, cam_params[-1]).to(dtype)

        bev, occ_pred = self.transformer(
            mlvl_feats=mlvl_feats,
            bev_queries=bev_queries,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
            cam_params=cam_params,
            pred_img_depth=pred_img_depth,
            bev_mask=bev_mask,
            last_occ_pred=last_occ_pred,
            shift=shift,
            prev_bev=prev_bev,
        )

        bev = bev.permute(0, 2, 1).view(bs, -1, self.bev_h, self.bev_w).contiguous()

        return bev, occ_pred

