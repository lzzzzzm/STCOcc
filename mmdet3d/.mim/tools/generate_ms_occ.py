import numpy as np
import torch
import mmcv
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate multi-scale occ')
    parser.add_argument('--dataset', type=str, help='occ3d or openocc')
    parser.add_argument('--pkl_path', type=str, help='path to the pkl file')
    args = parser.parse_args()
    return args

def squeeze_label(target_voxels, ratio, empty_idx=16):
    B, H, W, D = target_voxels.shape
    H = 50
    W = 50
    D = 4
    target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, H, W,
                                                                                                                D,
                                                                                                                ratio ** 3)
    empty_mask = target_voxels.sum(-1) == empty_idx
    target_voxels = target_voxels.to(torch.int64)
    occ_space = target_voxels[~empty_mask]
    occ_space[occ_space == 0] = -torch.arange(len(occ_space[occ_space == 0])).to(occ_space.device) - 1
    target_voxels[~empty_mask] = occ_space
    target_voxels = torch.mode(target_voxels, dim=-1)[0]
    target_voxels[target_voxels < 0] = 255
    target_voxels = target_voxels.long()
    return target_voxels[0]

def downsample_label(label, voxel_size=(200, 200, 16), downscale=2, empty_cls_idx=16):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == empty_cls_idx)).size

        zero_count = zero_count_0
        if zero_count > empty_t:
            label_downscale[x, y, z] = empty_cls_idx
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin >= 0, label_bin < empty_cls_idx))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale

def main(args):
    test_dataset = args.dataset
    train_pkl_path = args.pkl_path
    train_pkl = mmcv.load(train_pkl_path)

    for info in mmcv.track_iter_progress(train_pkl['infos']):
        occ_path = info['occ_path']
        if test_dataset == 'openocc':
            occ_path = info['occ_path'].replace('gts', 'openocc_v2')
            empty_cls_idx = 16
        else:
            empty_cls_idx = 17
        label_path = os.path.join(occ_path, 'labels.npz')
        save_path_1_2 = os.path.join(occ_path, 'labels_1_2.npz')
        save_path_1_4 = os.path.join(occ_path, 'labels_1_4.npz')
        save_path_1_8 = os.path.join(occ_path, 'labels_1_8.npz')

        labels = np.load(label_path)['semantics']
        labels = torch.from_numpy(labels)

        # process downsample 1/2, 1/4, 1/8
        labels_1_2 = downsample_label(labels, downscale=2, empty_cls_idx=empty_cls_idx)
        np.savez_compressed(save_path_1_2, semantics=labels_1_2, flow=labels_1_2)

        labels_1_4 = downsample_label(labels, downscale=4, empty_cls_idx=empty_cls_idx)
        np.savez_compressed(save_path_1_4, semantics=labels_1_4, flow=labels_1_4)

        labels_1_8 = downsample_label(labels, downscale=8, empty_cls_idx=empty_cls_idx)
        np.savez_compressed(save_path_1_8, semantics=labels_1_8, flow=labels_1_8)


if __name__ == "__main__":
    args = parse_args()
    main(args)