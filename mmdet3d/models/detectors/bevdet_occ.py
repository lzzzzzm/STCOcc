# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np

from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

from mmdet3d.models import builder

from mmdet3d.models.stcocc.losses.semkitti import sem_scal_loss, geo_scal_loss
from mmdet3d.models.stcocc.losses.lovasz_softmax import lovasz_softmax

import time

@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 in_channels=None,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=17,
                 empty_idx=16,
                 use_predicter=True,
                 class_wise=False,
                 class_weights=None,
                 sem_scal_loss_weight=1.0,
                 lovasz_softmax_weight=1.0,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.class_weights = torch.from_numpy(np.array(class_weights))
        self.empty_idx = empty_idx
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.sem_scal_loss_weight = sem_scal_loss_weight
        self.lovasz_softmax_weight = lovasz_softmax_weight

    def loss_single(self, target_voxels, output_voxels, tag):
        loss_dict = {}
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.sem_scal_loss_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.lovasz_softmax_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)
        return loss_dict

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        start_time = time.time()
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        end_time = time.time()
        print('extract_feat time:', end_time - start_time)

        pred_voxel_semantic = self.final_conv(img_feats[0])
        pred_voxel_semantic = F.interpolate(pred_voxel_semantic, scale_factor=2, mode='trilinear', align_corners=False)
        pred_voxel_semantic = pred_voxel_semantic.permute(0, 4, 3, 2, 1)  # [bs, c, z, h, w] - > [bs, w, h, z, c]
        if self.use_predicter:
            pred_voxel_semantic = self.predicter(pred_voxel_semantic)

        pred_voxel_flow = torch.zeros((1, 200, 200, 16, 2), dtype=torch.float32, device=pred_voxel_semantic.device)

        pred_voxel_semantic = pred_voxel_semantic.softmax(-1).argmax(-1)
        pred_voxel_semantic = pred_voxel_semantic.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        return_dict = dict()
        return_dict['occ_results'] = pred_voxel_semantic
        return_dict['flow_results'] = pred_voxel_flow

        return [return_dict]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        # occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        occ_pred = self.final_conv(img_feats[0])
        occ_pred = F.interpolate(occ_pred, scale_factor=2, mode='trilinear', align_corners=False)
        occ_pred = occ_pred.permute(0, 4, 3, 2, 1)  # [bs, c, z, h, w] - > [bs, w, h, z, c]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_pred = occ_pred.permute(0, 4, 1, 2, 3)  # [bs, w, h, z, c] - > [bs, c, w, h, z]

        voxel_semantics = kwargs['voxel_semantics']
        # mask_camera = kwargs['mask_camera']
        mask_camera = None
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, occ_pred, tag='c_0')
        losses.update(loss_occ)
        return losses
