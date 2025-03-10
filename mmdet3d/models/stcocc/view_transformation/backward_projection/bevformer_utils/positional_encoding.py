# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule
from mmcv.cnn import uniform_init


@POSITIONAL_ENCODING.register_module()
class CustormLearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(CustormLearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, bs, h, w, device):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)

        x_embed = self.col_embed(x).unsqueeze(0).repeat(h, 1, 1)
        y_embed = self.row_embed(y).unsqueeze(1).repeat(1, w, 1)

        pos = torch.cat([x_embed, y_embed],dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1) # [bs, c, h, w]

        return pos

    def forward_bda(self, bs, h, w, device, bda):
        x_embed_list, y_embed_list = [], []
        for i in range(bs):
            aug = bda[i]
            x_flip = False if aug[0][0] == 1 else True
            y_flip = False if aug[1][1] == 1 else True
            x = torch.arange(w, device=device)
            if x_flip:
                x = torch.flip(x, [0])

            y = torch.arange(h, device=device)
            if y_flip:
                y = torch.flip(y, [0])

            x_embed = self.col_embed(x).unsqueeze(0).repeat(h, 1, 1)
            y_embed = self.row_embed(y).unsqueeze(1).repeat(1, w, 1)

            x_embed_list.append(x_embed)
            y_embed_list.append(y_embed)

        x_embed_list = torch.stack(x_embed_list)
        y_embed_list = torch.stack(y_embed_list)

        pos = torch.cat([x_embed_list, y_embed_list], dim=-1).permute(0, 3, 1, 2)

        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str

@POSITIONAL_ENCODING.register_module()
class Learned3DPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, z_num_embed=8):
        super(Learned3DPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, 2 * num_feats)
        self.col_embed = nn.Embedding(col_num_embed, 2 * num_feats)
        self.z_embed = nn.Embedding(z_num_embed, 2 * num_feats)
        self.num_feats = 2 * num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.z_num_embed = z_num_embed
        self.init_weights()

    def init_weights(self):
        """Initialize the learnable weights."""
        uniform_init(self.row_embed)
        uniform_init(self.col_embed)
        uniform_init(self.z_embed)

    def forward(self, bs, h, w, z, device):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # h, w = mask.shape[-2:]
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        zz = torch.arange(z, device=device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        z_embed = self.z_embed(zz)
        pos3d = x_embed.view(h, 1, 1, self.num_feats).expand(h, w, z, self.num_feats) + y_embed.view(1, w, 1,
                                                                                                     self.num_feats).expand(
            h, w, z, self.num_feats) + z_embed.view(1, 1, z, self.num_feats).expand(h, w, z, self.num_feats)

        pos3d = pos3d.permute(3, 0, 1, 2).unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        return pos3d

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        repr_str += f'z_num_embed={self.z_num_embed})'
        return repr_str
