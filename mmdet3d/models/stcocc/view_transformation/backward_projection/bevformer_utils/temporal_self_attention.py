import math
import warnings
import numpy as np
from typing import Optional, no_type_check

import torch
import torch.nn as nn

import mmcv
from mmcv.utils import ext_loader
from mmcv.runner.base_module import BaseModule
from mmcv.utils import deprecated_api_warning
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class OA_TemporalAttention(BaseModule):
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
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 num_bev_queue: int = 2,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 use_sampling=False,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmcv.ConfigDict] = None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.use_sampling = use_sampling

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
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(embed_dims*num_bev_queue+1, num_bev_queue*num_heads*num_levels*num_points*2)
        self.attention_weights = nn.Linear(embed_dims*num_bev_queue, num_bev_queue*num_heads*num_levels*num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()
        self.index = 0
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                nonempty_voxel_logits = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape
        _, num_value, embed_dims = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        prev_bev = value[0:1].clone()
        curr_bev = query.clone()

        query = torch.cat([value[:bs], query], -1)

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs*self.num_bev_queue, num_value, self.num_heads, -1)

        # sampling_offsets: [bs, num_query, num_head, num_bev_queue, num_levels, num_points, 2]
        nonempty_attention_weights = nonempty_voxel_logits.clone().reshape(bs, num_query, self.num_points)
        bev_nonempty_attention_weights = nonempty_attention_weights.clone().mean(-1)[..., None]
        query_with_weights = torch.cat([query, bev_nonempty_attention_weights], -1)
        sampling_offsets = self.sampling_offsets(query_with_weights).view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)

        # obtain attention weights
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)

        # obtain occlusion-aware attention weights
        attention_weights = attention_weights * nonempty_attention_weights[:, :, None, None, None, :]

        # reshape sampling_offsets and attention_weights
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            vis_sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5

        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step
        )

        # fuse history value and current value
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def sampling_forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                nonempty_voxel_logits=None,
                **kwargs) -> torch.Tensor:

        if identity is None:
            identity = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape

        # sampling based on nonempty_voxel_logits
        nonempty_bev_logits = nonempty_voxel_logits.clone().reshape(bs, num_query, self.num_points).mean(-1)
        if self.training:
            sampling_rate = nn.init._no_grad_trunc_normal_(torch.empty_like(nonempty_bev_logits), mean=0.5, std=1, a=0, b=1)
        else:
            sampling_rate = torch.ones_like(nonempty_bev_logits) * 0.5
        occlusion_mask = sampling_rate < nonempty_bev_logits
        occlusion_mask = occlusion_mask.unsqueeze(-1)

        indexes = [[] for _ in range(bs)]
        max_len = 0
        for i in range(bs):
            index_query_per_batch = occlusion_mask[i].sum(-1).nonzero().squeeze(-1)
            if len(index_query_per_batch) == 0:
                index_query_per_batch = occlusion_mask[i].sum(-1).nonzero().squeeze(-1)[0:1]
            indexes[i].append(index_query_per_batch)
            # for batch operation, we need to pad the indexes to the same length
            max_len = max(max_len, len(index_query_per_batch))

        queries_rebatch = query.new_zeros([bs, max_len, self.embed_dims])
        value_rebatch = query.new_zeros([bs*2, max_len, self.embed_dims])
        reference_points_rebatch = query.new_zeros([bs*2, max_len, 1, 2])
        nonempty_bev_logits_rebatch = query.new_zeros([bs, max_len, 1])
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        for i in range(bs):
            index_query_per_batch = indexes[i][0]
            queries_rebatch[i, :len(index_query_per_batch)] = query[i, index_query_per_batch]
            nonempty_bev_logits_rebatch[i, :len(index_query_per_batch)] = nonempty_bev_logits[i, index_query_per_batch].unsqueeze(-1)
            value_rebatch[:i*2, :len(index_query_per_batch)] = value[:i*2, index_query_per_batch]
            reference_points_rebatch[:i*2, :len(index_query_per_batch)] = reference_points[:i*2, index_query_per_batch]

        bs, num_query, embed_dims = query.shape
        _, num_value, embed_dims = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        queries_rebatch = torch.cat([value_rebatch[:bs], queries_rebatch], -1)

        value_rebatch = self.value_proj(value_rebatch)
        if key_padding_mask is not None:
            value_rebatch = value_rebatch.masked_fill(key_padding_mask[..., None], 0.0)
        value_rebatch = value_rebatch.view(bs*self.num_bev_queue, max_len, self.num_heads, -1)

        # sampling_offsets: [bs, num_query, num_head, num_bev_queue, num_levels, num_points, 2]
        query_with_weights = torch.cat([queries_rebatch, nonempty_bev_logits_rebatch], -1)
        sampling_offsets = self.sampling_offsets(query_with_weights).view(bs, max_len, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)

        # obtain attention weights
        attention_weights = self.attention_weights(queries_rebatch).view(bs, max_len, self.num_heads, self.num_bev_queue,self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, max_len, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)

        # obtain occlusion-aware attention weights
        attention_weights = attention_weights * nonempty_bev_logits_rebatch[:, :, None, None, None, :]

        # reshape sampling_offsets and attention_weights
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(bs * self.num_bev_queue, max_len, self.num_heads, self.num_levels,self.num_points, 2)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(bs * self.num_bev_queue, max_len, self.num_heads, self.num_levels,self.num_points).contiguous()

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points_rebatch[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points_rebatch[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5

        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value_rebatch,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step
        )

        # fuse history value and current value
        output = output.view(bs, self.num_bev_queue, max_len, self.embed_dims)
        output = output.mean(1)

        for j in range(bs):
            index_query_per_batch = indexes[j][0]
            slots[j, index_query_per_batch] += output[j, :len(index_query_per_batch)]

        # output
        slots = self.output_proj(slots)

        return self.dropout(slots) + identity

