import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS

from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.stcocc.modules.basic_block import BasicBlock3D

@HEADS.register_module()
class OccFlowHead(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            foreground_idx=None,
            num_classes=None,
            up_sample=True,
            bev_h=200,
            bev_w=200,
            bev_z=16,
            grid_length_h=0.4,
            grid_length_w=0.4,
    ):
        super(OccFlowHead, self).__init__()
        self.foreground_idx = foreground_idx
        self.num_classes = num_classes
        self.grid_length_h = grid_length_h
        self.grid_length_w = grid_length_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.semantic_guidance = nn.Sequential(
            nn.Linear(num_classes, out_channels//2),
            nn.Softplus(),
            nn.Linear(out_channels//2, out_channels//2)
        )
        self.voxel_conv = BasicBlock3D(
            channels_in=in_channels,
            channels_out=out_channels,
            stride=1,
        )

        self.predicter = nn.Sequential(
            nn.Linear(out_channels+out_channels//2, out_channels // 2),
            nn.Softplus(),
            nn.Linear(out_channels // 2, 2),
        )

        self.up_sample = up_sample

    @force_fp32()
    def forward(self, voxel_feats, pred_voxel_semantic):

        pred_voxel_semantic_cls = F.gumbel_softmax(pred_voxel_semantic)
        semantics_guide = self.semantic_guidance(pred_voxel_semantic_cls)

        pred_voxel_feats = self.voxel_conv(voxel_feats)
        if self.up_sample:
            pred_voxel_feats = F.interpolate(pred_voxel_feats, scale_factor=2, mode='trilinear', align_corners=False)

        pred_voxel_feats = pred_voxel_feats.permute(0, 4, 3, 2, 1)  # [bs, c, z, h, w] - > [bs, w, h, z, c]
        pred_voxel_feats = torch.cat([pred_voxel_feats, semantics_guide], dim=-1)
        pred_voxel_flow = self.predicter(pred_voxel_feats)

        pred_voxel_semantic = pred_voxel_semantic.softmax(-1).argmax(-1)
        foreground_mask = torch.zeros(pred_voxel_semantic.shape).to(pred_voxel_semantic.device)
        for idx in self.foreground_idx:
            foreground_mask[pred_voxel_semantic == idx] = 1

        return pred_voxel_flow, foreground_mask

