import os
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

from mmdet3d.models.stcocc.losses.semkitti import geo_scal_loss, sem_scal_loss
from mmdet3d.models.stcocc.losses.lovasz_softmax import lovasz_softmax

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@DETECTORS.register_module()
class STCOcc(CenterPoint):

    def __init__(self,
                 # BEVDet-Series
                 forward_projection=None,
                 # BEVFormer-Series
                 backward_projection=None,
                 # Occupancy_Head
                 occupancy_head=None,
                 # Flow Head
                 flow_head=None,
                 # Option: Temporalfusion
                 temporal_fusion=None,
                 # Other setting
                 intermediate_pred_loss_weight=(0.5, 0.25),
                 backward_num_layer=(1, 2),
                 num_stage=2,
                 bev_w=None,
                 bev_h=None,
                 bev_z=None,
                 class_weights=None,
                 empty_idx=17,
                 class_weights_group=None,
                 history_frame_num=None,
                 foreground_idx=None,
                 background_idx=None,
                 train_flow=False,
                 use_ms_feats=False,
                 train_top_k=None,
                 val_top_k=None,
                 save_results=False,
                 **kwargs):
        super(STCOcc, self).__init__(**kwargs)
        # ---------------------- init params ------------------------------
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.bev_z = bev_z
        self.train_top_k = train_top_k
        self.val_top_k = val_top_k
        self.empty_idx = empty_idx
        self.num_stage = num_stage
        self.backward_num_layer = backward_num_layer
        self.intermediate_pred_loss_weight = intermediate_pred_loss_weight
        self.class_weights_group = class_weights_group
        self.history_frame_num = history_frame_num
        self.foreground_idx = foreground_idx
        self.background_idx = background_idx
        self.train_flow = train_flow
        self.use_ms_feats = use_ms_feats
        self.save_results = save_results
        self.scene_can_bus_info = dict()
        self.scene_loss = dict()
        # ---------------------- init loss ------------------------------
        self.class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32, device='cuda')
        self.flow_loss = builder.build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.focal_loss_dict = self._build_focal_loss(dict(type='CustomFocalLoss', bev_h=200, bev_w=200), num_stage)
        # ---------------------- build components ------------------------------
        # BEVDet-Series
        self.forward_projection = builder.build_neck(forward_projection)
        # BEVFormer-Series
        self._build_backward_projection(backward_projection, num_stage)
        # Temporal-Fsuion
        self._build_temporal_fusion(temporal_fusion, num_stage) if temporal_fusion else None
        # Simple Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head)
        # flow head
        self.flow_head = builder.build_head(flow_head) if flow_head else None


    def _build_focal_loss(self, focal_loss_config, num_stage):
        loss_dict = dict()
        focal_loss = builder.build_loss(focal_loss_config)
        loss_dict['num_stage_1_1'] = focal_loss
        for i in range(0, num_stage):
            focal_loss_config = copy.deepcopy(focal_loss_config)
            focal_loss_config['bev_h'] = int(focal_loss_config['bev_h'] / 2)
            focal_loss_config['bev_w'] = int(focal_loss_config['bev_w'] / 2)
            focal_loss = builder.build_loss(focal_loss_config)
            loss_dict['num_stage_1_{}'.format(2**(i+1))] = focal_loss
        return loss_dict

    def _build_backward_projection(self, backward_projection_config, num_stage):
        self.backward_projection_list = nn.ModuleList()
        backward_projection_config_dict = dict()
        backward_projection_config['transformer']['encoder']['num_layers'] = self.backward_num_layer[-1]
        backward_projection_config_dict['num_stage_0'] = copy.deepcopy(backward_projection_config)

        for index in range(num_stage):
            # first stage:
            if index == num_stage - 1:
                backward_projection_config_dict['num_stage_{}'.format(index)]['transformer']['encoder']['first_stage'] = True

            backward_projection = builder.build_neck(backward_projection_config_dict['num_stage_{}'.format(index)])
            self.backward_projection_list.append(backward_projection)
            if index != num_stage-1:
                # different stage, adjust backward params, copy avoid in-place operation
                backward_projection_config_dict['num_stage_{}'.format(index+1)] = copy.deepcopy(backward_projection_config_dict['num_stage_{}'.format(index)])

                # num_layer adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['num_layers'] = self.backward_num_layer[num_stage-index-2]

                # bev shape adjust or voxel shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_h'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_h'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_w'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_w'] / 2)
                if 'bev_z' in backward_projection_config_dict['num_stage_{}'.format(index + 1)]:
                    backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_z'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_z'] / 2)

                # grid config bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['x'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['x'][2] * 2
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['y'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['y'][2] * 2
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['z'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['z'][2] * 2

                # transformerlayers bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1][
                    'deformable_attention']['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1][
                        'deformable_attention']['num_points'] / 2)

                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                    'transformerlayers']['attn_cfgs'][0]['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                            'transformerlayers']['attn_cfgs'][0]['num_points'] / 2)

                if 'num_Z_anchors' in backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                    'transformerlayers']['attn_cfgs'][1]['deformable_attention']:
                    backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                        'transformerlayers']['attn_cfgs'][1]['deformable_attention']['num_Z_anchors'] = \
                        int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                                'transformerlayers']['attn_cfgs'][1]['deformable_attention']['num_Z_anchors'] / 2)

                # positional_encoding bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['row_num_embed'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['row_num_embed'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['col_num_embed'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['col_num_embed'] / 2)

    def _build_temporal_fusion(self, temporal_fusion_config, num_stage):
        self.temporal_fusion_list = nn.ModuleList()
        self.dx_list, self.bx_list, self.nx_list = [], [], []
        x_config = self.forward_projection.img_view_transformer.grid_config['x']
        y_config = self.forward_projection.img_view_transformer.grid_config['y']
        z_config = self.forward_projection.img_view_transformer.grid_config['z']
        for i in range(num_stage):
            temporal_fusion_config['history_num'] = self.history_frame_num[i]
            if self.train_top_k and self.training:
                temporal_fusion_config['top_k'] = self.train_top_k[i]
            elif self.val_top_k and not self.training:
                temporal_fusion_config['top_k'] = self.val_top_k[i]
            temporal_fusion = builder.build_head(temporal_fusion_config)
            dx, bx, nx = gen_dx_bx(x_config, y_config, z_config)

            dx = nn.Parameter(dx, requires_grad=False)
            bx = nn.Parameter(bx, requires_grad=False)
            nx = nn.Parameter(nx, requires_grad=False)
            self.temporal_fusion_list.append(temporal_fusion)
            self.dx_list.append(dx)
            self.bx_list.append(bx)
            self.nx_list.append(nx)

            x_config[2] = x_config[2] * 2
            y_config[2] = y_config[2] * 2
            z_config[2] = z_config[2] * 2
            temporal_fusion_config['bev_z'] = int(temporal_fusion_config['bev_z'] / 2)
            temporal_fusion_config['bev_h'] = int(temporal_fusion_config['bev_h'] / 2)
            temporal_fusion_config['bev_w'] = int(temporal_fusion_config['bev_w'] / 2)


    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def get_voxel_loss(self,
                       pred_voxel_semantic,
                       target_voxel_semantic,
                       loss_weight,
                       focal_loss=None,
                       tag='c_0',
                       ):
        # change pred_voxel_semantic from [bs, w, h, z, c] -> [bs, c, w, h, z]  !!!
        pred_voxel_semantic = pred_voxel_semantic.permute(0, 4, 1, 2, 3)
        loss_dict = {}

        loss_dict['loss_voxel_ce_{}'.format(tag)] = loss_weight * focal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            self.class_weights,
            ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = loss_weight * sem_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = loss_weight * geo_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255,
            empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = loss_weight * lovasz_softmax(
            torch.softmax(pred_voxel_semantic, dim=1),
            target_voxel_semantic,
            ignore=255,
        )

        return loss_dict

    def get_flow_loss(self, pred_flows, target_flows, target_sem, loss_weight):
        loss_dict = {}

        loss_flow = 0
        for i in range(target_flows.shape[0]):
            foreground_mask = torch.zeros(target_flows[i].shape[:-1])
            for idx in self.foreground_idx:
                foreground_mask[target_sem[i] == idx] = 1

            pred_flow = pred_flows[i][foreground_mask!=0]
            target_flow = target_flows[i][foreground_mask!=0]

            loss_flow = loss_flow + loss_weight * self.flow_loss(pred_flow, target_flow)
        loss_dict['loss_flow'] = loss_flow

        return loss_dict

    def obtain_feats_from_images(self, points, img, img_metas, **kwargs):
        # 0、Prepare
        if self.with_specific_component('temporal_fusion_list'):
            use_temporal = True
            sequence_group_idx = torch.stack(
                [torch.tensor(img_meta['sequence_group_idx'], device=img[0].device) for img_meta in img_metas])
            start_of_sequence = torch.stack(
                [torch.tensor(img_meta['start_of_sequence'], device=img[0].device) for img_meta in img_metas])
            curr_to_prev_ego_rt = torch.stack(
                [torch.tensor(np.array(img_meta['curr_to_prev_ego_rt']), device=img[0].device) for img_meta in img_metas])
            history_fusion_params = {
                'sequence_group_idx': sequence_group_idx,
                'start_of_sequence': start_of_sequence,
                'curr_to_prev_ego_rt': curr_to_prev_ego_rt
            }
            # process can_bus info
            if 'can_bus' in img_metas[0]:
                for index, start in enumerate(start_of_sequence):
                    if start:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = 0
                        can_bus[-1] = 0
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose':temp_pose,
                            'prev_angle':temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus
                    else:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = can_bus[:3] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_pose']
                        can_bus[-1] = can_bus[-1] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_angle']
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose': temp_pose,
                            'prev_angle': temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus

        else:
            use_temporal = False
            history_fusion_params = None

        # 1、Forward-Projection create coarse voxel features
        if 'sequential' not in kwargs or not kwargs['sequential']:
            voxel_feats, depth, tran_feats, ms_feats, cam_params = self.forward_projection.extract_feat(points, img=img, img_metas=img_metas, **kwargs)
        else:
            voxel_feats, depth, tran_feats, ms_feats, cam_params = kwargs['voxel_feats'], kwargs['depth'], kwargs['tran_feats'], kwargs['ms_feats'],kwargs['cam_params']

        # 2、Backward-Projection Refine
        last_voxel_feat = None
        last_occ_pred = None
        voxel_feats_index = len(voxel_feats) - 1
        intermediate_occ_pred_dict = {}
        intermediate_voxel_feat = []
        for i in range(len(self.backward_projection_list)-1, -1, -1):
            voxel_feat = voxel_feats[voxel_feats_index]

            if last_voxel_feat is not None:
                voxel_feat = last_voxel_feat + voxel_feat

            # voxel_feats shape: [bs, c, z, h, w]
            bev_feat, occ_pred = self.backward_projection_list[i](
                mlvl_feats=ms_feats if self.use_ms_feats else [tran_feats],
                img_metas=img_metas,
                voxel_feats=voxel_feat,  # mean in the z direction
                cam_params=cam_params,
                pred_img_depth=depth,
                last_occ_pred=last_occ_pred,
                prev_bev=self.temporal_fusion_list[i].history_last_bev if use_temporal else None,
                prev_bev_aug=self.temporal_fusion_list[i].history_forward_augs,
                history_fusion_params=history_fusion_params,
            )

            # save for loss
            intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2 ** (i + 1))] = occ_pred
            last_occ_pred = occ_pred.clone().detach()
            # bev_feats shape: [bs, c, h, w], recover to occupancy with occ weight
            if bev_feat.dim() == 4:
                bs, c, z, h, w = voxel_feat.shape
                bev_feat = bev_feat.unsqueeze(2).repeat(1, 1, z, 1, 1)
                nonempty_prob = 1 - last_occ_pred.softmax(-1)[..., -1].permute(0, 3, 2, 1)
                last_voxel_feat = voxel_feat + bev_feat * nonempty_prob.unsqueeze(1)
            else:
                nonempty_prob = 1 - last_occ_pred.softmax(-1)[..., -1].permute(0, 3, 2, 1)
                last_voxel_feat = voxel_feat + bev_feat * nonempty_prob.unsqueeze(1)

            # Option: temporal fusion
            if self.with_specific_component('temporal_fusion_list'):
                last_voxel_feat = self.temporal_fusion_list[i](
                    last_voxel_feat, cam_params, history_fusion_params, dx=self.dx_list[i], bx=self.bx_list[i],
                    history_last_bev=self.temporal_fusion_list[i+1].history_bev if i+1 < len(self.temporal_fusion_list)-1 else None,
                    last_occ_pred=last_occ_pred,
                    nonempty_prob=nonempty_prob,
                )

            # output stage don't need to upsample
            if i != 0:
                last_voxel_feat = F.interpolate(last_voxel_feat, scale_factor=2, align_corners=False, mode='trilinear')

            voxel_feats_index = voxel_feats_index - 1
            intermediate_voxel_feat.append(last_voxel_feat)

        return_dict = dict(
            voxel_feats=last_voxel_feat,
            last_occ_pred=last_occ_pred,
            depth=depth,
            tran_feats=tran_feats,
            cam_params=cam_params,
            intermediate_occ_pred_dict=intermediate_occ_pred_dict,
            history_fusion_params=history_fusion_params,
        )
        return return_dict

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    **kwargs):
        # ---------------------- obtain feats from images -----------------------------
        return_dict = self.obtain_feats_from_images(points, img=img_inputs[0], img_metas=img_metas, **kwargs)
        voxel_feat = return_dict['voxel_feats']
        last_occ_pred = return_dict['last_occ_pred']

        # ---------------------- forward ------------------------------
        pred_voxel_semantic, pred_voxel_feats = self.occupancy_head(voxel_feat, last_occ_pred=last_occ_pred)
        pred_voxel_semantic_cls = pred_voxel_semantic.softmax(-1).argmax(-1)
        if self.with_specific_component('flow_head'):
            pred_voxel_flows, foreground_mask = self.flow_head(voxel_feat, pred_voxel_semantic)
            return_pred_voxel_flows = torch.zeros_like(pred_voxel_flows)
            return_pred_voxel_flows[foreground_mask != 0] = pred_voxel_flows[foreground_mask != 0]
        else:
            return_pred_voxel_flows = torch.zeros(size=(pred_voxel_semantic_cls.shape + (2,)), device=pred_voxel_semantic_cls.device)

        return_dict = dict()
        return_dict['occ_results'] = pred_voxel_semantic_cls.cpu().numpy().astype(np.uint8)
        return_dict['flow_results'] = return_pred_voxel_flows.cpu().numpy().astype(np.float16)
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        if self.save_results:
            sample_idx = [img_meta['sample_idx'] for img_meta in img_metas]
            scene_name = [img_meta['scene_name'] for img_meta in img_metas]
            # check save_dir
            for name in scene_name:
                if not os.path.exists('results/{}'.format(name)):
                    os.makedirs('results/{}'.format(name))
            for i, idx in enumerate(sample_idx):
                np.savez('results/{}/{}.npz'.format(scene_name[i], idx),semantics=return_dict['occ_results'][i], flow=return_dict['flow_results'][i])
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

        # ---------------------- obtain feats from images -----------------------------
        return_dict = self.obtain_feats_from_images(points, img=img_inputs, img_metas=img_metas, **kwargs)

        voxel_feats = return_dict['voxel_feats']    # shape: [bs, c, z, h, w]
        last_occ_pred = return_dict['last_occ_pred']
        depth = return_dict['depth']
        intermediate_occ_pred_dict = return_dict['intermediate_occ_pred_dict']
        history_fusion_params = return_dict['history_fusion_params']

        # ---------------------- forward ------------------------------
        pred_voxel_semantic, pred_voxel_feats = self.occupancy_head(voxel_feats, last_occ_pred=last_occ_pred)
        intermediate_occ_pred_dict['pred_voxel_semantic_1_1'] = pred_voxel_semantic

        if self.with_specific_component('flow_head'):
            pred_voxel_flows, foreground_masks = self.flow_head(voxel_feats, pred_voxel_semantic)

        # ---------------------- calc loss ------------------------------
        losses = dict()

        gt_semantic_voxel_dict = dict()
        gt_semantic_voxel_dict['gt_semantic_voxel_1_1'] = kwargs['voxel_semantics']
        num_stage = self.num_stage
        for index in range(num_stage):
            gt_semantic_voxel_dict['gt_semantic_voxel_1_{}'.format(2**(index+1))] = kwargs['voxel_semantics_1_{}'.format(2**(index+1))]

        # calc forward-projection depth-loss
        loss_depth = self.forward_projection.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
        losses['loss_depth'] = loss_depth

        # calc voxel loss
        for index in range(num_stage+1):
            loss_occ = self.get_voxel_loss(
                intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2**index)],
                gt_semantic_voxel_dict['gt_semantic_voxel_1_{}'.format(2 **index)],
                self.intermediate_pred_loss_weight[index],
                focal_loss=self.focal_loss_dict['num_stage_1_{}'.format(2 **index)],
                tag='c_1_{}'.format(2**index),
            )
            losses.update(loss_occ)

        if self.with_specific_component('flow_head'):
            losses.update(
                self.get_flow_loss(
                    pred_voxel_flows,
                    kwargs['voxel_flows'],
                    gt_semantic_voxel_dict['gt_semantic_voxel_1_1'],
                    loss_weight=0.8
                )
            )

        return losses