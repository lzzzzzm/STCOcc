import numpy as np

import torch
import torch.nn as nn
import copy
import warnings

from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import ext_loader
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.models.backbones.resnet import BasicBlock

from .base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class OccPredictor(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            bev_h=200,
            bev_w=200,
            bev_z=16,
            num_classes=18,
            num_layer=2,
            first_stage=True,
            group_list=None,
    ):
        super(OccPredictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bev_z = bev_z
        self.bev_h = bev_h
        self.bev_w = bev_w
        if group_list is not None:
            self.num_classes = len(group_list)
        else:
            self.num_classes = num_classes
        self.bev_conv = BasicBlock(
            inplanes=in_channels,
            planes=out_channels,
        )
        self.predicter = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.Softplus(),
            nn.Linear(out_channels * 2, self.num_classes * bev_z),
        )
        self.first_stage = first_stage
        self.alpha_list = []
        if first_stage:
            for i in range(num_layer - 1):
                self.alpha_list.append(nn.Parameter(torch.tensor([0.5], device='cuda')))
        else:
            for i in range(num_layer):
                self.alpha_list.append(nn.Parameter(torch.tensor([0.5], device='cuda')))

    @force_fp32()
    def forward(self, bev_query_feats, last_occ_pred=None, layer_num=None):
        # required [bs, h*w, c] bev query
        # shape check
        bs, hw, c = bev_query_feats.shape
        assert hw == self.bev_h * self.bev_w
        # reshape bev_query_feats
        bev_feats = bev_query_feats.reshape(bs, self.bev_h, self.bev_w, c).permute(0, 3, 1, 2)  # [bs, h, w, c] -> [bs, c, h, w]
        # feature extraction
        occ_feat = self.bev_conv(bev_feats).permute(0, 3, 2, 1)  # [bs, c, h, w] -> [bs, w, h, c]
        # get the height of bev feature
        occ_pred = self.predicter(occ_feat).view(bs, self.bev_w, self.bev_h, self.bev_z, self.num_classes)  # [bs, w, h, z, c] for calc loss

        # self-recurrence
        if last_occ_pred is not None:
            if self.first_stage and layer_num != 0:
                occ_pred = occ_pred + last_occ_pred * self.alpha_list[layer_num - 1]
            else:
                occ_pred = occ_pred + last_occ_pred * self.alpha_list[layer_num]

        # parse occ_pred
        occ_logits = occ_pred.softmax(-1).clone().detach()
        empty_occ_logits = occ_logits[..., -1].unsqueeze(-1)
        nonempty_occ_logits = 1 - empty_occ_logits
        nonempty_occ_logits = nonempty_occ_logits.permute(0, 2, 1, 3, 4)  # [bs, w, h, z, c] -> [bs, h, w, z, c]

        return occ_pred, nonempty_occ_logits


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    def __init__(self,
                 *args,
                 pc_range=None,
                 grid_config=None,
                 data_config=None,
                 return_intermediate=False,
                 num_cam=6,
                 predictor_in_channels=None,
                 predictor_out_channels=None,
                 predictor_num_calsses=18,
                 first_stage=False,
                 group_list=None,
                 use_temporal=False,
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.depth_bound = grid_config['depth']
        self.bev_x = int((self.x_bound[1] - self.x_bound[0]) / self.x_bound[2])
        self.bev_y = int((self.y_bound[1] - self.y_bound[0]) / self.y_bound[2])
        self.bev_z = int((self.z_bound[1] - self.z_bound[0]) / self.z_bound[2])
        self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.num_cam = num_cam
        self.use_temporal = use_temporal
        self.fp16_enabled = False
        self.predictor_in_channels = predictor_in_channels
        self.predictor_out_channels = predictor_out_channels
        self.first_stage = first_stage
        self.occ_predictor = OccPredictor(
            in_channels=predictor_in_channels,
            out_channels=predictor_out_channels,
            num_classes=predictor_num_calsses,
            bev_w=self.bev_x,
            bev_h=self.bev_y,
            bev_z=self.bev_z,
            num_layer=self.num_layers,
            first_stage=self.first_stage,
            group_list=group_list,
        )


    def get_reference_points(self, H, W, Z=None, num_points_in_pillar =4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None):
        # prepare for point sampling
        lidar2img = []
        ego2lidar = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])     # lidar2img update the post aug in the loading pipeline
            ego2lidar.append(img_meta['ego2lidar'])
        lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)
        ego2lidar = torch.stack(ego2lidar, dim=0).to(reference_points.device)

        sensor2egos, ego2globals, intrins, post_augs, bda_mat = cam_params
        num_cam = sensor2egos.size(1)
        ogfH, ogfW = self.final_dim

        # reference_points defines in the bev space, [bs, D, hxw, 3]
        # change reference_points from bev-ego coordinate to ego coordinate
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        # prepare for point sampling
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)  # shape: (num_points_in_pillar,bs,h*w,4)
        D, B, num_query = reference_points.size()[:3]  # D=num_points_in_pillar , num_query=h*w
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)
        inverse_bda = bda_mat.view(1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1)

        # change reference_points from ego coordinate to img coordinate
        eps = 1e-5
        reference_points_cam = (lidar2img @ ego2lidar @ inverse_bda @ reference_points).squeeze(-1)   # [num_points_in_pillar, bs, num_cam, num_query=h*w, 4]
        reference_points_depth = reference_points_cam[..., 2:3]
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_depth, torch.ones_like(reference_points_depth) * eps)

        # Bug!!
        # Correct normalize is
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        # But for reproducing our results, we use the following normalization
        reference_points_cam[..., 0] /= ogfH
        reference_points_cam[..., 1] /= ogfW

        bev_mask = (reference_points_depth > eps)
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)                  # shape: (num_cam, bs,h*w, num_points_in_pillar, 2)
        reference_points_depth = reference_points_depth.permute(2, 1, 3, 0, 4)              # shape: (num_cam, bs,h*w, num_points_in_pillar, 1)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)                        # shape: (num_cam, bs,h*w, num_points_in_pillar)

        return reference_points_cam, reference_points_depth, bev_mask


    @force_fp32()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                bev_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                cam_params=None,
                pred_img_depth=None,
                last_occ_pred=None,
                shift=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        # vis img_depth

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2], self.bev_z, dim='3d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, reference_points_depth, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas=kwargs['img_metas'], cam_params=cam_params)

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        shift_ref_2d = ref_2d.clone()
        if self.use_temporal:
            shift_ref_2d += shift[:, None, None, :]
            if prev_bev is not None:
                prev_bev = prev_bev.permute(1, 0, 2)
                prev_bev = torch.stack(
                    [prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
                hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                    bs * 2, len_bev, num_bev_level, 2)
            else:
                hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                    bs * 2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = shift_ref_2d

        for lid, layer in enumerate(self.layers):
            output, occ_pred = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_mask=bev_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                reference_points_depth=reference_points_depth,
                pred_img_depth=pred_img_depth,
                occ_predictor=self.occ_predictor,
                last_occ_pred=last_occ_pred,
                prev_bev=prev_bev,
                layer_num=lid,
                depth_bound=self.depth_bound,
                **kwargs)

            bev_query = output
            last_occ_pred = occ_pred
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, last_occ_pred


@TRANSFORMER_LAYER.register_module()
class BEVFormerEncoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=512,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) in {2, 4, 6}

    @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                reference_points_depth=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                bev_mask=None,
                bev_query_depth=None,
                lidar_bev=None,
                pred_img_depth=None,
                occ_predictor=None,
                last_occ_pred=None,
                layer_num=None,
                nonempty_voxel_logits=None,
                depth_bound=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        conv_index = 0
        bs, num_query, dims = query.shape
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    reference_points=ref_2d,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    nonempty_voxel_logits=nonempty_voxel_logits,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'predictor':
                occ_pred, nonempty_voxel_logits = occ_predictor(query, last_occ_pred, layer_num=layer_num)

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    reference_points_depth=reference_points_depth,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_query_depth=bev_query_depth,
                    pred_img_depth=pred_img_depth,
                    nonempty_voxel_logits=nonempty_voxel_logits,
                    bev_mask=bev_mask,
                    depth_bound=depth_bound,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
            elif layer == 'conv':
                query = query.reshape(bs, bev_h, bev_w, dims).permute(0, 3, 1, 2)
                query = self.convs[conv_index](query)
                query = query.permute(0, 2, 3, 1).reshape(bs, -1, dims)
                conv_index += 1

        return query, occ_pred
