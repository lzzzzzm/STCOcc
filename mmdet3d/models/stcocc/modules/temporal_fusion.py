import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class SparseFusion(BaseModule):
    def __init__(
            self,
            history_num=8,
            single_bev_num_channels=None,
            history_cat_conv_out_channels=None,
            do_history=True,
            group_size=2,
            top_k=None,
            foreground_idx=None,
            num_classes=17,
            occ_embedims=32,
            **kwargs,
    ):
        super(SparseFusion, self).__init__()
        self.single_bev_num_channels = single_bev_num_channels
        self.history_bev = None
        self.history_last_bev = None
        self.history_forward_augs = None
        self.history_num = history_num
        self.history_seq_ids = None
        self.history_sweep_time = None
        self.do_history = do_history
        self.history_count = 0
        self.history_cam_sweep_freq = 0.5  # seconds between each frame
        self.group_size = group_size
        self.top_k = top_k
        self.foreground_idx = foreground_idx

        self.occ_embedding = nn.Sequential(
            nn.Linear(num_classes, occ_embedims),
            nn.Softplus(),
            nn.Linear(occ_embedims, occ_embedims),
        )

        self.history_fusion_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )
        self.history_fusion_bg_linear = nn.Sequential(
            nn.Linear(single_bev_num_channels * (history_num//2 + 1) + occ_embedims, single_bev_num_channels),
            nn.Softplus(),
            nn.Linear(single_bev_num_channels, single_bev_num_channels),
        )

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev

    @force_fp32()
    def forward(self, curr_bev, cam_params, history_fusion_params, dx, bx, history_last_bev=None, last_occ_pred=None, nonempty_prob=None):
        # curr_bev: [bs, c, z, h, w]
        # cam_params: dict, contain bda_mat

        # 0、check process voxel or bev features
        voxel_feat = True if len(curr_bev.shape) == 5 else False

        # 1、Get some history fusion information
        # Process test situation
        if type(history_fusion_params['sequence_group_idx']) is list:
            seq_ids = history_fusion_params['sequence_group_idx'][0]
        else:
            seq_ids = history_fusion_params['sequence_group_idx']
        if type(history_fusion_params['start_of_sequence']) is list:
            start_of_sequence = history_fusion_params['start_of_sequence'][0]
        else:
            start_of_sequence = history_fusion_params['start_of_sequence']
        if type(history_fusion_params['curr_to_prev_ego_rt']) is list:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt'][0]
        else:
            curr_to_prev_ego_rt = history_fusion_params['curr_to_prev_ego_rt']
        forward_augs = cam_params[-1]  # bda

        # check seq_ids > 0
        assert (seq_ids >= 0).all()

        # 2、Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_forward_augs = forward_augs.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_bev = curr_bev.repeat(1, self.history_num, 1, 1, 1)
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_num)

        self.history_bev = self.history_bev.detach()
        assert self.history_bev.dtype == torch.float32

        # 3、 Deal with the new sequences
        # Replace all the new sequences' positions in history with the curr_bev information
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        self.history_sweep_time += 1
        if start_of_sequence.sum() > 0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_num, 1, 1, 1)
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts

        # 4、Deal with the history fusion
        tmp_bev = self.history_bev
        bs, mc, z, h, w = tmp_bev.shape
        n, c_, z, h, w = curr_bev.shape

        # obtain the history fusion information
        grid = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid, dx, bx)

        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt @ torch.inverse(forward_augs) @ feat2bev)
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0   # grid order is x, y, z

        # sample the history bev
        tmp_bev = tmp_bev.reshape(bs, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),  align_corners=True, mode='bilinear')

        # determine the nonempty_voxel

        # top_k sampling, sampled foreground and background top_k
        last_occ_pred = last_occ_pred.permute(0, 3, 2, 1, 4).reshape(bs, h*w*z, -1)
        occ_embed = self.occ_embedding(last_occ_pred).permute(0, 2, 1)  # [bs, occ_embedims, h*w*z]
        nonempty_prob = nonempty_prob.reshape(bs, -1)
        total_number = nonempty_prob.shape[1]
        indices = torch.topk(nonempty_prob, self.top_k, dim=1)[1]                        # foreground indices
        bg_indices = torch.topk(1 - nonempty_prob, total_number - self.top_k, dim=1)[1]  # background indices
        sampled_history, sampled_current, sampled_occ_embd = [], [], []
        sampled_bg_history, sampled_bg_current, sampled_bg_occ_embed = [], [], []
        for i in range(bs):
            sampled_history_feature = sampled_history_bev[i].reshape(mc, -1)[:, indices[i]]
            sampled_history_bg_feature = sampled_history_bev[i, :mc//2].reshape(mc//2, -1)[:, bg_indices[i]]
            sampled_current_feature = curr_bev[i].reshape(c_, -1)[:, indices[i]]
            sampled_current_bg_feature = curr_bev[i].reshape(c_, -1)[:, bg_indices[i]]
            sampled_occ_embd_feature = occ_embed[i][:, indices[i]]
            sampled_bg_occ_embed_feature = occ_embed[i][:, bg_indices[i]]

            sampled_history.append(sampled_history_feature)
            sampled_current.append(sampled_current_feature)
            sampled_bg_history.append(sampled_history_bg_feature)
            sampled_bg_current.append(sampled_current_bg_feature)
            sampled_occ_embd.append(sampled_occ_embd_feature)
            sampled_bg_occ_embed.append(sampled_bg_occ_embed_feature)

        sampled_history = torch.stack(sampled_history, dim=0)
        sampled_current = torch.stack(sampled_current, dim=0)
        sampled_bg_history = torch.stack(sampled_bg_history, dim=0)
        sampled_bg_current = torch.stack(sampled_bg_current, dim=0)
        sampled_occ_embd = torch.stack(sampled_occ_embd, dim=0)
        sampled_bg_occ_embed = torch.stack(sampled_bg_occ_embed, dim=0)

        sampled_fusion = torch.cat([sampled_history, sampled_current, sampled_occ_embd], dim=1).permute(0, 2, 1)
        sampled_bg_fusion = torch.cat([sampled_bg_history, sampled_bg_current, sampled_bg_occ_embed], dim=1).permute(0, 2, 1)
        sampled_fusion = self.history_fusion_linear(sampled_fusion)
        sampled_bg_fusion = self.history_fusion_bg_linear(sampled_bg_fusion)
        sampled_fusion = sampled_fusion.permute(0, 2, 1)
        sampled_bg_fusion = sampled_bg_fusion.permute(0, 2, 1)

        # add the sampled fusion to the current bev
        curr_bev = curr_bev.reshape(n, c_, -1)
        for i in range(bs):
            curr_bev[i, :, indices[i]] += sampled_fusion[i]
            curr_bev[i, :, bg_indices[i]] += sampled_bg_fusion[i]

        sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
        curr_bev = curr_bev.reshape(n, c_, z, h, w)

        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1)  # B x (1 + T) * C x H x W or B x (1 + T) * C x Z x H x W
        feats_cat = feats_cat.reshape(feats_cat.shape[0], self.history_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:])  # B x (1 + T) x C x H x W

        # feats_to_return: shape [bs, 1 + T, c, z, h, w]
        feats_cat = feats_cat.reshape(feats_cat.shape[0], -1, *feats_cat.shape[3:])
        feats_to_return = curr_bev

        # update history information
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_last_bev = feats_to_return.detach().clone()
        self.history_forward_augs = forward_augs.clone()

        return feats_to_return.clone()



