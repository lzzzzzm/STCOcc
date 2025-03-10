import warnings
import math

import torch
import torch.nn as nn

from mmcv.utils import ext_loader
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.transformer import build_attention

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class OA_SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 layer_scale=None,
                 dbound=None,
                 **kwargs
                 ):
        super(OA_SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dbound = dbound
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        if layer_scale is not None:
            self.layer_scale = nn.Parameter(
                layer_scale * torch.ones(embed_dims),
                requires_grad=True)
        else:
            self.layer_scale = None
        self.init_weight()
        self.count = 0

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points=None,
                reference_points_cam=None,
                reference_points_depth=None,
                level_start_index=None,
                bev_query_depth=None,
                pred_img_depth=None,
                depth_bound=None,
                nonempty_voxel_logits=None,
                bev_mask=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        # input check
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        # preprocess
        bs, num_query, _ = query.size()
        num_cam, D = reference_points_cam.size(0), reference_points_cam.size(3)         # reference_points_cam: [num_cam, bs, num_query, D, 2]
        nonempty_voxel_logits = nonempty_voxel_logits.reshape(bs, num_query, D, 1)
        pred_img_depth = pred_img_depth.view(bs * num_cam, -1, spatial_shapes[0][0], spatial_shapes[0][1])
        pred_img_depth = pred_img_depth.flatten(2).permute(0, 2, 1)                     # [bs, h*w, C]
        # Sampling
        if self.training:
            sampling_rate = nn.init._no_grad_trunc_normal_(torch.empty_like(nonempty_voxel_logits), mean=0.5, std=1, a=0, b=1)  # [bs, num_query, embed_dims]
        else:
            sampling_rate = torch.ones_like(nonempty_voxel_logits) * 0.5
        occlusion_mask = sampling_rate < nonempty_voxel_logits
        occlusion_mask = occlusion_mask[None].repeat(num_cam, 1, 1, 1, 1).squeeze(-1)
        if torch.sum(bev_mask & occlusion_mask) != 0:
            bev_mask = bev_mask & occlusion_mask

        # Use bev_mask to check if the reference_points_cam is valid
        # bev_mask: [num_cam, bs, h*w, num_points_in_pillar]
        indexes = [[] for _ in range(bs)]
        max_len = 0
        for j in range(bs):
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                if len(index_query_per_img) == 0:
                    index_query_per_img = bev_mask[i][j].sum(-1).nonzero().squeeze(-1)[0:1]
                indexes[j].append(index_query_per_img)
                # for batch operation, we need to pad the indexes to the same length
                max_len = max(max_len, len(index_query_per_img))

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_cam_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        reference_points_depth_rebatch = reference_points_depth.new_zeros([bs, self.num_cams, max_len, D, 1])
        reference_points_occlusion_rebatch = nonempty_voxel_logits.new_zeros([bs, max_len, D, 1])
        # reference_points_3d_rebatch = reference_points.new_zeros([bs, max_len, D, 3])

        # get non mask query and reference points
        for j in range(bs):
            for i, (reference_points_per_img, reference_points_depth_per_img) in enumerate(zip(reference_points_cam, reference_points_depth)):
                index_query_per_img = indexes[j][i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_depth_rebatch[j, i, :len(index_query_per_img)] = reference_points_depth_per_img[j, index_query_per_img]
                reference_points_occlusion_rebatch[j, :len(index_query_per_img)] = nonempty_voxel_logits[j, index_query_per_img]
        # repeat along the num_cams
        reference_points_occlusion_rebatch = reference_points_occlusion_rebatch.unsqueeze(1).repeat(1, self.num_cams, 1, 1, 1)

        num_cams, l, bs, embed_dims = key.shape
        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims),
            value=value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims),
            reference_points=reference_points_cam_rebatch.view(bs * self.num_cams, max_len, D, 2),
            reference_points_depth=reference_points_depth_rebatch.view(bs * self.num_cams, max_len, D),
            reference_points_occlusion=reference_points_occlusion_rebatch.view(bs * self.num_cams, max_len, D),
            # params
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            pred_img_depth=pred_img_depth,
            depth_bound=depth_bound,
        )
        queries = queries.view(bs, self.num_cams, max_len, self.embed_dims)

        # aug the query
        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[j][i]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # output
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

@ATTENTION.register_module()
class OA_MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 sigma=2,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sigma = sigma

        self.sampling_offsets = nn.Linear(embed_dims+1, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pred_img_depth=None,
                reference_points_depth=None,
                depth_bound=None,
                reference_points_occlusion=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        # obtain the sampling points offsets for deformable attention
        reference_points_occlusion_query = reference_points_occlusion.clone().mean(-1)[..., None]
        query_with_occlusion = torch.cat([query, reference_points_occlusion_query], -1)
        sampling_offsets = self.sampling_offsets(query_with_occlusion).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            bs, num_query, num_Z_anchors, xy = reference_points.shape

            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)

            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape

            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(f'Last dim of reference_points must be 'f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # obtain the attention weights for deformable attention
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # obtain the attention weights for Depth deformable attention
        depth_reference_points = reference_points.reshape(bs, num_query * num_Z_anchors, 1, 1, 1, 2).contiguous()
        depth_attention_weights = torch.ones_like(depth_reference_points[..., 0]).contiguous()
        pred_img_depth = pred_img_depth.unsqueeze(2).contiguous()

        # obtain sample point depth pred value
        depth_output = MultiScaleDeformableAttnFunction_fp32.apply(pred_img_depth, spatial_shapes,level_start_index,depth_reference_points,depth_attention_weights, self.im2col_step)
        depth_output = depth_output.reshape(bs, num_query, num_Z_anchors, -1)   # [bs*num_cam, num_query, num_Z_anchors, C]
        depth_output = depth_output.softmax(-1).argmax(-1) * depth_bound[2] + depth_bound[0]
        depth_weights = torch.exp(
            -torch.square(
                torch.min(torch.abs(reference_points_depth - (depth_output - depth_bound[2])), torch.abs(reference_points_depth - (depth_output + depth_bound[2])))
            ) / 2 * (self.sigma ** 2)
        )

        # obtain occlusion weights
        occlusion_weights = depth_weights * reference_points_occlusion
        attention_weights = attention_weights * occlusion_weights[:, :, None, None] # change to multi-head, do not softmax
        output = MultiScaleDeformableAttnFunction_fp32.apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output