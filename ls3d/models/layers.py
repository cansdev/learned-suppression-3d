"""Set abstraction (encoder) and UNet3+ feature propagation (decoder) stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import (canonical_mask, farthest_point_sample, index_points, interpolate_to_target,
                  square_distance)
from .point_transformer import PointTransformerBlock


def _group(xyz, points, npoint, nsample_list, use_dedup):
    """Sample centers by FPS and gather k nearest neighbors for each scale.

    The center itself is placed in the first neighbor slot. When a point has no
    input features, the relative coordinates are used as features.

    Returns:
        new_xyz: [B, S, 3]
        fps_idx: [B, S]
        groups: list of (group_idx [B,S,K], rel_xyz [B,S,K,3], neigh [B,S,K,D], center [B,S,D])
    """
    canonical = canonical_mask(xyz) if use_dedup else None
    fps_idx = farthest_point_sample(xyz, npoint, canonical=canonical)
    new_xyz = index_points(xyz, fps_idx)

    dists = square_distance(new_xyz, xyz)
    if canonical is not None:
        dists = dists + (~canonical).float().unsqueeze(1) * 1e10
    center_idx = dists.argmin(dim=-1)

    groups = []
    for K in nsample_list:
        _, group_idx = torch.topk(dists, k=K, dim=-1, largest=False)
        group_idx[:, :, 0] = center_idx
        rel_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
        if points is not None and points.shape[-1] > 0:
            neigh, center = index_points(points, group_idx), index_points(points, center_idx)
        else:
            neigh, center = rel_xyz, index_points(xyz, center_idx)
        groups.append((group_idx, rel_xyz, neigh, center))
    return new_xyz, fps_idx, groups


class SetAbstractionMsg(nn.Module):
    """Multi scale set abstraction: one Point Transformer block per kNN scale,
    outputs concatenated along channels (SA1 to SA3)."""

    def __init__(self, npoint, nsample_list, in_channels, out_channels_list, use_dedup=True,
                 share_planes=8):
        super().__init__()
        self.npoint = npoint
        self.nsample_list = nsample_list
        self.use_dedup = use_dedup
        in_ch = in_channels if in_channels > 0 else 3
        self.pt_blocks = nn.ModuleList(
            [PointTransformerBlock(in_ch, c, share_planes=share_planes) for c in out_channels_list])

    def forward(self, xyz, points):
        """xyz [B,3,N], points [B,D,N] or None -> (new_xyz [B,3,S], feats [B,C,S], fps_idx [B,S])."""
        xyz = xyz.transpose(1, 2)
        points = points.transpose(1, 2) if points is not None else None
        new_xyz, fps_idx, groups = _group(xyz, points, self.npoint, self.nsample_list, self.use_dedup)
        feats = [blk(rel, neigh, center).transpose(1, 2)
                 for blk, (_, rel, neigh, center) in zip(self.pt_blocks, groups)]
        return new_xyz.transpose(1, 2), torch.cat(feats, dim=1), fps_idx


class SetAbstractionPrior(nn.Module):
    """Single scale set abstraction with the training time log prior (SA4, Eq. 3).

    `labels` are the soft keypoint labels y of the input points of this stage,
    carried down the hierarchy through the FPS indices of earlier stages. They
    are gathered over each center's neighbors and added to the attention logits.
    Without labels (inference) the block is plain vector attention.
    """

    def __init__(self, npoint, nsample, in_channels, out_channels, use_dedup=True, share_planes=8):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.use_dedup = use_dedup
        in_ch = in_channels if in_channels > 0 else 3
        self.pt_block = PointTransformerBlock(in_ch, out_channels, share_planes=share_planes)

    def forward(self, xyz, points, labels=None):
        xyz = xyz.transpose(1, 2)
        points = points.transpose(1, 2) if points is not None else None
        new_xyz, fps_idx, groups = _group(xyz, points, self.npoint, [self.nsample], self.use_dedup)
        group_idx, rel, neigh, center = groups[0]
        prior = None
        if labels is not None:
            B, S, K = group_idx.shape
            prior = labels.gather(1, group_idx.reshape(B, -1)).view(B, S, K)
        out = self.pt_block(rel, neigh, center, prior)
        return new_xyz.transpose(1, 2), out.transpose(1, 2), fps_idx


class UNet3PlusFP(nn.Module):
    """Feature propagation with UNet3+ full scale skip connections.

    Inputs at the target resolution: the same scale encoder features and the
    preceding decoder features at full width, plus every other encoder or
    decoder output interpolated and projected to `proj_ch` channels.
    """

    def __init__(self, same_scale_ch, prev_decoder_ch, extra_ch_list, proj_ch, mlp):
        super().__init__()
        self.has_same_scale = same_scale_ch > 0
        self.proj_convs = nn.ModuleList([nn.Conv1d(ch, proj_ch, 1) for ch in extra_ch_list])
        self.proj_bns = nn.ModuleList([nn.BatchNorm1d(proj_ch) for _ in extra_ch_list])

        last = same_scale_ch + prev_decoder_ch + len(extra_ch_list) * proj_ch
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last = out_ch

    def forward(self, xyz_target, same_scale_feats, prev_xyz, prev_feats, extra_skips):
        parts = []
        if self.has_same_scale and same_scale_feats is not None:
            parts.append(same_scale_feats)
        parts.append(interpolate_to_target(xyz_target, prev_xyz, prev_feats))
        for conv, bn, (xyz_src, feats_src) in zip(self.proj_convs, self.proj_bns, extra_skips):
            parts.append(F.gelu(bn(conv(interpolate_to_target(xyz_target, xyz_src, feats_src)))))
        x = torch.cat(parts, dim=1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = F.gelu(bn(conv(x)))
        return x
