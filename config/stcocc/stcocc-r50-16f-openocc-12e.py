_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# 6019it [06:08, 16.32it/s]
# +----------------------+-------+-------+-------+-------+
# |     Class Names      | IoU@1 | IoU@2 | IoU@4 |  AVE  |
# +----------------------+-------+-------+-------+-------+
# |         car          | 0.523 | 0.595 | 0.622 | 0.410 |
# |        truck         | 0.416 | 0.501 | 0.540 | 0.335 |
# |       trailer        | 0.346 | 0.428 | 0.513 | 0.547 |
# |         bus          | 0.540 | 0.633 | 0.689 | 0.642 |
# | construction_vehicle | 0.265 | 0.363 | 0.383 | 0.138 |
# |       bicycle        | 0.229 | 0.258 | 0.262 | 0.498 |
# |      motorcycle      | 0.273 | 0.342 | 0.357 | 0.588 |
# |      pedestrian      | 0.354 | 0.400 | 0.414 | 0.371 |
# |     traffic_cone     | 0.313 | 0.330 | 0.340 |  nan  |
# |       barrier        | 0.429 | 0.470 | 0.491 |  nan  |
# |  driveable_surface   | 0.482 | 0.575 | 0.681 |  nan  |
# |      other_flat      | 0.244 | 0.288 | 0.322 |  nan  |
# |       sidewalk       | 0.263 | 0.310 | 0.354 |  nan  |
# |       terrain        | 0.267 | 0.338 | 0.397 |  nan  |
# |       manmade        | 0.404 | 0.483 | 0.535 |  nan  |
# |      vegetation      | 0.299 | 0.414 | 0.498 |  nan  |
# +----------------------+-------+-------+-------+-------+
# |         MEAN         | 0.353 | 0.420 | 0.462 | 0.441 |
# +----------------------+-------+-------+-------+-------+
# MIOU: 0.4119753149566574
# MAVE: 0.44101850333090886
# Occ score: 0.42667593312790075
# {'miou': 0.4119753149566574, 'mave': 0.44101850333090886}

# nuscenes val scene=150, recommend use 6 gpus, 5 batchsize

# Dataset Config
dataset_name = 'openocc'
eval_metric = 'rayiou'

class_weights = [0.0682, 0.0823, 0.0671, 0.0594, 0.0732, 0.0806, 0.0680, 0.0762, 0.0675, 0.0633, 0.0521, 0.0644, 0.0557,
                 0.0551, 0.0535, 0.0533, 0.0464]  # openocc

occ_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                   'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free']

foreground_idx = [occ_class_names.index('car'), occ_class_names.index('truck'), occ_class_names.index('trailer'),
                  occ_class_names.index('bus'), occ_class_names.index('construction_vehicle'),
                  occ_class_names.index('bicycle'),
                  occ_class_names.index('motorcycle'), occ_class_names.index('pedestrian')]

# train_top_k = [35000, 5600, 900]
# val_top_k = [28000, 4900, 800]

train_top_k = [12000, 2400, 450]
val_top_k = [12000, 2400, 450]

# DataLoader Config
data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)
# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# Running Config
num_gpus = 1
samples_per_gpu = 1
workers_per_gpu = 0
total_epoch = 12
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 4.554)  # total samples: 28130

# Model Config

# forward params
grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 45.0, 0.5],
}

downsample_rate = 16
multi_adj_frame_id_cfg = (1, 1 + 1, 1)
forward_numC_Trans = 80

# backward params
grid_config_bevformer = grid_config
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
bev_h_ = 100
bev_w_ = 100
bev_z = 8
backward_num_layer = [2, 2, 2]
backward_numC_Trans = 96
_dim_ = backward_numC_Trans * 2
_pos_dim_ = backward_numC_Trans // 2
_ffn_dim_ = backward_numC_Trans * 4
_num_levels_ = 1
num_stage = 3

# others params
num_classes = len(occ_class_names)  # 0-15->objects, 16->free

intermediate_pred_loss_weight = [1.0, 0.5, 0.25, 0.125]
history_frame_num = [16, 8, 4]

model = dict(
    type='STCOcc',
    num_stage=num_stage,
    bev_w=bev_h_,
    bev_h=bev_w_,
    bev_z=bev_z,
    train_top_k=train_top_k,
    val_top_k=val_top_k,
    class_weights=class_weights,
    foreground_idx=foreground_idx,
    history_frame_num=history_frame_num,
    backward_num_layer=backward_num_layer,
    empty_idx=occ_class_names.index('free'),
    intermediate_pred_loss_weight=intermediate_pred_loss_weight,
    forward_projection=dict(
        type='BEVDetStereoForwardProjection',
        align_after_view_transfromation=False,
        return_intermediate=True,
        num_adj=len(range(*multi_adj_frame_id_cfg)),
        adjust_channel=backward_numC_Trans,
        img_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            style='pytorch'),
        img_neck=dict(
            type='CustomFPN',
            in_channels=[1024, 2048],
            out_channels=256,
            num_outs=1,
            start_level=0,
            out_ids=[0]),
        img_view_transformer=dict(
            type='LSSVStereoForwardPorjection',
            grid_config=grid_config,
            input_size=data_config['input_size'],
            in_channels=256,
            out_channels=forward_numC_Trans,
            sid=False,
            collapse_z=False,
            loss_depth_weight=0.5,
            depthnet_cfg=dict(use_dcn=False,
                              aspp_mid_channels=96,
                              stereo=True,
                              bias=5.),
            downsample=downsample_rate),
        img_bev_encoder_backbone=dict(
            type='CustomResNet3D',
            numC_input=forward_numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
            num_layer=[1, 2, 4],
            with_cp=False,
            num_channels=[backward_numC_Trans, forward_numC_Trans * 2, forward_numC_Trans * 4],
            adjust_number_channel=backward_numC_Trans,
            stride=[1, 2, 2],
            backbone_output_ids=[0, 1, 2],
        ),
    ),
    backward_projection=dict(
        type='BEVFormerBackwardProjection',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=backward_numC_Trans,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                use_temporal=True,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                return_intermediate=False,
                predictor_in_channels=backward_numC_Trans,
                predictor_out_channels=backward_numC_Trans,
                predictor_num_calsses=num_classes,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='OA_TemporalAttention',
                            num_points=bev_z,
                            embed_dims=backward_numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='OA_SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=grid_config['depth'],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='OA_MSDeformableAttention3D',
                                embed_dims=backward_numC_Trans,
                                num_points=bev_z,
                                num_levels=_num_levels_),
                            embed_dims=backward_numC_Trans,
                        )
                    ],
                    conv_cfgs=dict(embed_dims=backward_numC_Trans),
                    operation_order=('predictor', 'self_attn', 'norm', 'cross_attn', 'norm', 'conv')
                )
            ),
        ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
    temporal_fusion=dict(
        type='SparseFusion',
        history_num=2,
        single_bev_num_channels=backward_numC_Trans,
        bev_w=bev_w_,
        bev_h=bev_h_,
        bev_z=bev_z,
    ),
    occupancy_head=dict(
        type='OccHead',
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        num_classes=num_classes,
        up_sample=True,
    ),
    flow_head=dict(
        type='OccFlowHead',
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        foreground_idx=foreground_idx,
        num_classes=num_classes,
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config=data_config, sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFileOpenOcc', scale_1_2=True, scale_1_4=True, scale_1_8=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D',
         keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'voxel_semantics_1_2', 'voxel_semantics_1_4',
               'voxel_semantics_1_8', 'voxel_flows'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names, is_train=False),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=occ_class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=occ_class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    use_sequence_group_flag=True,
    # Eval Config
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    work_dir='stcocc-r50-stage3-flow-8f-64dims-openocc',
    eval_show=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'nus-infos/bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nus-infos/bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=occ_class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        # Video Sequence
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        # Set BEV Augmentation for the same sequence
        # bda_aug_conf=bda_aug_conf,
    ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 1e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch * total_epoch, ])

checkpoint_epoch_interval = 1
runner = dict(type='IterBasedRunner', max_iters=total_epoch * num_iters_per_epoch)
checkpoint_config = dict(interval=checkpoint_epoch_interval * num_iters_per_epoch)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2 * num_iters_per_epoch,
    )
]
revise_keys = None
load_from = "pretrained/forward_projection-r50-4d-stereo-pretrained.pth"
