_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

num_gpus = 1
samples_per_gpu = 2
workers_per_gpu = 0

# class weights calc by frequencies
# class_weights = [0.0682, 0.0823, 0.0671, 0.0594, 0.0732, 0.0806, 0.0680, 0.0762, 0.0675,
#         0.0633, 0.0521, 0.0644, 0.0557, 0.0551, 0.0535, 0.0533, 0.0464]
class_weights = [0.0727, 0.0692, 0.0838, 0.0681, 0.0601, 0.0741, 0.0823, 0.0688, 0.0773,
        0.0681, 0.0641, 0.0527, 0.0655, 0.0563, 0.0558, 0.0541, 0.0538, 0.0468] # occ-3d
# Global
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# forward params
grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [2.0, 42.0, 0.5],
}

downsample_rate = 16
multi_adj_frame_id_cfg = (1, 1+4, 1)
numC_Trans = 80

# backward params
grid_config_bevformer={
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
}
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
bev_h_ = 100
bev_w_ = 100
bev_z = 8
backward_num_layer = [3, 3, 2]
backward_numC_Trans = numC_Trans
_dim_ = backward_numC_Trans*2
_pos_dim_ = backward_numC_Trans//2
_ffn_dim_ = backward_numC_Trans * 4
_num_levels_ = 1
num_stage = 3

# others params
num_classes = 18 # 0-15->objects, 16->free
empty_idx = 17
intermediate_pred_loss_weight=[0.5, 0.25, 0.125]

model = dict(
    type='CascadeOcc',
    backward_num_layer=backward_num_layer,
    num_stage=num_stage,
    bev_w=bev_h_,
    bev_h=bev_w_,
    bev_z=bev_z,
    num_classes=num_classes,
    class_weights=class_weights,
    empty_idx=empty_idx,
    intermediate_pred_loss_weight=intermediate_pred_loss_weight,
    forward_projection=dict(
        type='BEVDetStereoForwardProjection',
        align_after_view_transfromation=False,
        return_intermediate=True,
        num_adj=len(range(*multi_adj_frame_id_cfg)),
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
            out_channels=numC_Trans,
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
            numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
            num_layer=[1, 2, 4],
            with_cp=False,
            num_channels=[numC_Trans, numC_Trans*2, numC_Trans*4],
            stride=[1, 2, 2],
            backbone_output_ids=[0, 1, 2],
        ),
        img_bev_encoder_neck=dict(
            type='FPN',
            in_channels=[numC_Trans, numC_Trans*2, numC_Trans*4],
            out_channels=numC_Trans,
            start_level=0,
            num_outs=3,
            relu_before_extra_convs=True,
            add_extra_convs='on_output',
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
        ),
        pre_process=dict(
            type='PreProcessNet',
            real_h=80.0,
            bev_w=bev_w_,
            bev_h=bev_h_,
            bev_z=bev_z,
            numC_input=numC_Trans,
            with_cp=False,
            num_layer=[1,],
            num_channels=[numC_Trans,],
            stride=[1,],
            backbone_output_ids=[0,]),
        ),
    backward_projection=dict(
        type='SparseBEVFormerBackwardProjection',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseBEVFormer',
            use_cams_embeds=False,
            embed_dims=backward_numC_Trans,
            encoder=dict(
                type='CustomBEVFormerEncoder',
                num_layers=2,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                return_intermediate=False,
                predictor_in_channels=backward_numC_Trans,
                predictor_out_channels=backward_numC_Trans,
                predictor_num_calsses=num_classes,
                transformerlayers=dict(
                    type='CustomBEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='OA_SelfAttention',
                            num_points=bev_z,
                            embed_dims=backward_numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='OA_SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=[2.0, 42.0, 0.5],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='OA_MSDeformableAttention',
                                embed_dims=backward_numC_Trans,
                                num_points=bev_z,
                                num_Z_anchors=bev_z,
                                num_levels=_num_levels_),
                            embed_dims=backward_numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=backward_numC_Trans,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True), ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('predictor', 'self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
    occupancy_head=dict(
        type='OccupancyHead',
        in_channels=numC_Trans,
        out_channels=numC_Trans,
        num_classes=num_classes,
        empty_idx=empty_idx,
        class_weights=class_weights,
        up_sample=True,
    ),
    solofusion=dict(
        type='SoloFusion',
        history_num=16,
        single_bev_num_channels=numC_Trans
    )
)

# Data
# dataset_type = 'NuScenesDatasetFlowOccpancy'
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFileCVPR2023',
         scale_1_2=True,
         scale_1_4=True,
         scale_1_8=True,
         ignore_invisible=True,
         load_mask=True,
         ),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config, load_semantic=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'voxel_semantics_1_2', 'voxel_semantics_1_4', 'voxel_semantics_1_8'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
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
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'nus-infos/bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nus-infos/bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# load_from="pretrained/forward_projection-r50-4d-stereo-pretrained.pth"
load_from="ckpts/iter_5268.pth"
