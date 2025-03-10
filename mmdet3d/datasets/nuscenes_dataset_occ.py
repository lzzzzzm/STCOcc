# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from .nuscenes_ego_pose_loader import nuScenesDataset
from .ray_metrics_occ3d import main as ray_based_miou_occ3d
from .ray_metrics_openocc import main as ray_based_miou_openocc

occ3d_colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],   # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ]
)

openocc_colors_map = np.array(
    [
        [0, 150, 245],      # car                  blue         √
        [160, 32, 240],     # truck                purple       √
        [135, 60, 0],       # trailer              brown        √
        [255, 255, 0],      # bus                  yellow       √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 192, 203],    # bicycle              pink         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)



@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        if 'pts_semantic_mask_path' in self.data_infos[index]:
            input_dict['pts_semantic_mask_path'] = self.data_infos[index]['pts_semantic_mask_path']
        input_dict['seg_label_mapping'] = self.SegLabelMapping
        return input_dict

    def evaluate_rayioU(self, results, logger=None, dataset_name='openocc'):
        if self.eval_show:
            mmcv.mkdir_or_exist(self.work_dir)

        pred_sems, gt_sems = [], []
        pred_flows, gt_flows = [], []
        lidar_origins = []
        data_index = []

        print('\nStarting Evaluation...')
        processed_set = set()
        for index, result in enumerate(results):
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)

                pred_sem = result['occ_results'][i]

                if 'flow_results' not in result:
                    pred_flow = np.zeros(pred_sem.shape + (2, ))
                else:
                    pred_flow = result['flow_results'][i]

                data_index.append(id)
                pred_sems.append(pred_sem)
                pred_flows.append(pred_flow)

        nusc = NuScenes('v1.0-trainval', 'data/nuscenes/')
        nusdata = nuScenesDataset(nusc, 'val')

        for index in data_index:
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]

            occ_path = info['occ_path']
            if dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)

            gt_semantics = occ_gt['semantics'].astype(np.uint8)
            if dataset_name == 'occ3d':
                gt_flow = np.zeros((200, 200, 16, 2), dtype=np.float16)
            elif dataset_name == 'openocc':
                gt_flow = occ_gt['flow'].astype(np.float16)

            gt_sems.append(gt_semantics)
            gt_flows.append(gt_flow)

            # get lidar
            ref_sample_token, output_origin_tensor = nusdata.__getitem__(index)
            lidar_origins.append(output_origin_tensor.unsqueeze(0))

        # visualization
        # if self.eval_show:
        #     for index in range(len(data_index)):
        #         if index >= len(self.data_infos):
        #             break
        #         info = self.data_infos[data_index[index]]
        #         if dataset_name == 'openocc':
        #             occ_bev_vis = self.vis_occ(pred_sems[index], color_map=openocc_colors_map, empty_idx=16)
        #             occ_bev_vis_gt = self.vis_occ(gt_sems[index], color_map=openocc_colors_map, empty_idx=16)
        #         elif dataset_name == 'occ3d':
        #             occ_bev_vis = self.vis_occ(pred_sems[index], color_map=occ3d_colors_map, empty_idx=17)
        #             occ_bev_vis_gt = self.vis_occ(gt_sems[index], color_map=occ3d_colors_map, empty_idx=17)
        #         scene_token = info['token']
        #         occ_bev_vis = np.concatenate([occ_bev_vis, occ_bev_vis_gt], axis=1)
        #         cv2.imwrite(os.path.join(self.work_dir, f'{scene_token}.png'), occ_bev_vis)

        if dataset_name == 'openocc':
            miou, mave, occ_score = ray_based_miou_openocc(pred_sems, gt_sems, pred_flows, gt_flows, lidar_origins, logger=logger)
        elif dataset_name == 'occ3d':
            miou, mave, occ_score = ray_based_miou_occ3d(pred_sems, gt_sems, pred_flows, gt_flows, lidar_origins, logger=logger)

        eval_dict = {
            'miou':miou,
            'mave':mave
        }
        return eval_dict

    def evaluate_miou(self, results, logger=None, dataset_name='openocc'):
        pred_sems, gt_sems = [], []
        data_index = []

        num_classes = 17 if dataset_name == 'openocc' else 18
        use_image_mask = True if dataset_name == 'occ3d' else False
        self.miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=use_image_mask,
            logger=logger
        )

        print('\nStarting Evaluation...')
        processed_set = set()
        for result in results:
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)

                pred_sem = result['occ_results'][i]
                data_index.append(id)
                pred_sems.append(pred_sem)

        for index in tqdm(data_index):
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]

            occ_path = info['occ_path']
            if dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)

            gt_semantics = occ_gt['semantics']
            pr_semantics = pred_sems[data_index.index(index)]

            if dataset_name == 'occ3d':
                mask_camera = occ_gt['mask_camera'].astype(bool)
            else:
                mask_camera = None

            self.miou_metric.add_batch(pr_semantics, gt_semantics, None, mask_camera)

        _, miou, _, _, _ = self.miou_metric.count_miou()
        eval_dict = {
            'miou':miou,
        }
        return eval_dict

    def evaluate(self, occ_results, logger=None, runner=None, show_dir=None, **eval_kwargs):
        if self.eval_metric == 'rayiou':
            return self.evaluate_rayioU(occ_results, logger, dataset_name=self.dataset_name)
        elif self.eval_metric == 'miou':
            return self.evaluate_miou(occ_results, logger, dataset_name=self.dataset_name)


    def vis_occ(self, semantics, empty_idx, color_map=None):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == empty_idx)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = color_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 3)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (200, 200))

        occ_bev_vis = cv2.resize(occ_bev_vis, (600, 600))
        occ_bev_vis = cv2.cvtColor(occ_bev_vis, cv2.COLOR_BGR2RGB)
        return occ_bev_vis