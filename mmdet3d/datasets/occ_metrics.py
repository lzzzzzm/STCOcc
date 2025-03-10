import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce
from mmcv.utils import print_log
from mmdet3d.utils import get_root_logger
from terminaltables import AsciiTable

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=17,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 logger=None,
                 class_names=None,
                 foreground_idx=[2, 3, 4, 5, 6, 7, 9, 10],
                 ):
        if num_classes == 18:
            self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                                'driveable_surface', 'other_flat', 'sidewalk',
                                'terrain', 'manmade', 'vegetation','free']
        elif num_classes == 17:
            self.class_names = [
                        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
                        'driveable_surface', 'other_flat', 'sidewalk',
                        'terrain', 'manmade', 'vegetation', 'free'
                    ]
        else:
            self.class_names = class_names

        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.foreground_idx = foreground_idx
        if num_classes == 18:
            self.foreground_names = ['bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'trailer', 'truck']
        elif num_classes == 17:
            self.foreground_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian']

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        self.flow_error = np.zeros([len(self.foreground_idx)])
        self.ave_count = np.zeros([len(self.foreground_idx)])
        if logger is None:
            self.logger = get_root_logger()
        else:
            self.logger = logger

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def compute_mave(self, pred_flow, gt_flow, pred_sem, gt_sem):
        for index, cls in enumerate(self.foreground_idx):
            mask = gt_sem == cls
            if mask.sum() == 0:
                continue
            pred_flow_cls = pred_flow[mask]
            gt_flow_cls = gt_flow[mask]
            pred_sem_cls = pred_sem[mask]
            gt_sem_cls = gt_sem[mask]

            sem_cls_mask = pred_sem_cls == gt_sem_cls
            pred_flow_cls[sem_cls_mask] = gt_flow_cls[sem_cls_mask]
            pred_flow_cls[~sem_cls_mask] = 0

            flow_error = np.linalg.norm(pred_flow_cls - gt_flow_cls, axis=1)
            self.flow_error[index] += np.sum(flow_error)
            self.ave_count[index] += flow_error.shape[0]

    def compute_bev_mave(self, pred_flow, gt_flow, gt_sem):
        for index, cls in enumerate(self.foreground_idx):
            mask = gt_sem == cls
            if mask.sum() == 0:
                continue
            pred_flow_cls = pred_flow[mask]
            gt_flow_cls = gt_flow[mask]

            flow_error = np.linalg.norm(pred_flow_cls - gt_flow_cls, axis=1)
            self.flow_error[index] += np.sum(flow_error)
            self.ave_count[index] += flow_error.shape[0]

    def bev_projection(self, voxel):
        w, h, z = voxel.shape
        bev = np.zeros((w, h))
        for cls in self.foreground_idx:
            mask = (voxel == cls).any(axis=2)
            bev[mask] = cls
        return bev

    def add_batch(self,semantics_pred, semantics_gt, mask_lidar, mask_camera, flow_pred=None, flow_gt=None):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]

            if flow_pred is not None and flow_gt is not None:
                pass

        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self, count_flow=False):
        ret_dict = dict()
        header = ['classes']
        for i in range(self.num_classes):
            header.append(self.class_names[i])
        header.extend(['miou'])
        table_columns = [['results']]
        if count_flow:
            mave_list = self.flow_error / self.ave_count
            print_log('===> per class mAVE of {} samples:'.format(self.cnt), logger=self.logger)
            print_log('===> mAVE of {} samples: {}'.format(self.cnt, str(round(np.nanmean(mave_list), 2))), logger=self.logger)
            for index in range(len(self.foreground_idx)):
                print_log(f'===> {self.foreground_names[index]} - MAVE = ' + str(round(mave_list[index], 2)),logger=self.logger)

        mIoU = self.per_class_iu(self.hist)
        for i in range(self.num_classes):
            ret_dict[self.class_names[i]] = str(round(mIoU[i] * 100, 2))
            table_columns.append([f'{mIoU[i]:.4f}'])

        table_columns.append([f'{round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2):.4f}'])
        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        if self.num_classes == 2:
            for ind_class in range(self.num_classes):
                print_log(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)),
                          logger=self.logger)
            print_log(
                f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes]) * 100, 2)),
                logger=self.logger)
        else:
            # assert cnt == num_samples, 'some samples are not included in the miou calculation'
            print_log(f'===> per class IoU of {self.cnt} samples:', logger=self.logger)
            for ind_class in range(self.num_classes-1):
                print_log(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)), logger=self.logger)
                ret_dict[self.class_names[ind_class]] = str(round(mIoU[ind_class] * 100, 2))

            print_log('===> empty - IoU = {}'.format(str(round(np.nanmean(mIoU[self.num_classes-1]) * 100, 2))), logger=self.logger)
            ret_dict[self.class_names[self.num_classes-1]] = str(round(np.nanmean(mIoU[self.num_classes-1]) * 100, 2))

            print_log(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)), logger=self.logger)
            ret_dict['mIoU'] = str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2))

            return self.class_names, round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2), self.cnt, ret_dict, table.table


class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8



    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))


