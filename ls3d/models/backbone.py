"""Graph-Transformer backbone (Sec. 3.1, 3.2, Fig. 2)."""

import torch.nn as nn
import torch.nn.functional as F

from .directional_gnn import DirectionalGNN
from .layers import SetAbstractionMsg, SetAbstractionPrior, UNet3PlusFP


class KeypointBackbone(nn.Module):
    """Hierarchical Point Transformer encoder, UNet3+ decoder and directional GNN.

    Encoder (kNN grouping, Point Transformer blocks with the spectral encoding):
        SA1  N   -> 640   K = 16, 32   ->  64 + 128 = 192
        SA2  640 -> 320   K = 16, 32   -> 128 + 256 = 384
        SA3  320 -> 160   K = 16, 32   -> 256 + 512 = 768
        SA4  160 -> 40    K = 32       -> 1024   (training time log prior)
    Decoder: FP4 to FP1 with UNet3+ skips back to N points and 128 channels,
    followed by the directional GNN. Two heads predict a keypoint logit and a
    3D offset per point.

    Inputs are [B, N, 3 + D]: XYZ followed by D per point features.
    """

    def __init__(self, input_channels, use_dedup=True, gnn_k=16):
        super().__init__()
        self.input_channels = input_channels
        feat_ch = max(0, input_channels - 3)
        self.feature_channels = feat_ch

        self.sa1 = SetAbstractionMsg(640, [16, 32], feat_ch, [64, 128], use_dedup)
        self.sa2 = SetAbstractionMsg(320, [16, 32], 192, [128, 256], use_dedup)
        self.sa3 = SetAbstractionMsg(160, [16, 32], 384, [256, 512], use_dedup)
        self.sa4 = SetAbstractionPrior(40, 32, 768, 1024, use_dedup)

        P = 64  # width of the projected (non direct) UNet3+ skips
        self.fp4 = UNet3PlusFP(768, 1024, [384, 192], P, [512, 512])
        self.fp3 = UNet3PlusFP(384, 512, [192, 768, 1024], P, [512, 256])
        self.fp2 = UNet3PlusFP(192, 256, [384, 768, 1024, 512], P, [256, 128])
        self.fp1 = UNet3PlusFP(feat_ch, 128, [192, 384, 768, 1024, 512, 256], P, [256, 128, 128])

        self.gnn_fp1 = DirectionalGNN(128, hidden_channels=128, k_neighbors=gnn_k, dropout=0.1,
                                      transform_channels=128, gate_init=0.3, use_dedup=use_dedup)

        # Classification head.
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 1, 1)
        # Offset regression head.
        self.reg_conv1 = nn.Conv1d(128, 64, 1)
        self.reg_bn1 = nn.BatchNorm1d(64)
        self.reg_drop1 = nn.Dropout(0.5)
        self.reg_conv2 = nn.Conv1d(64, 3, 1)

    def forward(self, x, labels=None, graph=None):
        """
        Args:
            x: [B, N, C] input points (XYZ first).
            labels: [B, N] soft keypoint labels y (training only) for the SA4 prior.
            graph: optional (knn_idx [B,N,K], knn_dist [B,N,K]) for the directional GNN.
        Returns:
            logits [B, N], offsets [B, N, 3] (in units of the label radius r),
            features [B, 128, N] (refined per point features).
        """
        l0_xyz = x[:, :, :3].transpose(1, 2).contiguous()
        l0_points = x[:, :, 3:].transpose(1, 2).contiguous() if self.feature_channels > 0 else None

        l1_xyz, l1, idx1 = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2, idx2 = self.sa2(l1_xyz, l1)
        l3_xyz, l3, idx3 = self.sa3(l2_xyz, l2)

        # Carry the soft labels down to the SA4 input points through the FPS indices.
        l3_labels = None
        if labels is not None:
            l3_labels = labels.gather(1, idx1).gather(1, idx2).gather(1, idx3)
        l4_xyz, l4, _ = self.sa4(l3_xyz, l3, l3_labels)

        fp4 = self.fp4(l3_xyz, l3, l4_xyz, l4, [(l2_xyz, l2), (l1_xyz, l1)])
        fp3 = self.fp3(l2_xyz, l2, l3_xyz, fp4, [(l1_xyz, l1), (l3_xyz, l3), (l4_xyz, l4)])
        fp2 = self.fp2(l1_xyz, l1, l2_xyz, fp3,
                       [(l2_xyz, l2), (l3_xyz, l3), (l4_xyz, l4), (l3_xyz, fp4)])
        fp1 = self.fp1(l0_xyz, l0_points, l1_xyz, fp2,
                       [(l1_xyz, l1), (l2_xyz, l2), (l3_xyz, l3), (l4_xyz, l4),
                        (l3_xyz, fp4), (l2_xyz, fp3)])

        knn_idx, knn_dist = graph if graph is not None else (None, None)
        feats = self.gnn_fp1(l0_xyz, fp1, knn_idx, knn_dist)

        h = self.drop1(F.relu(self.bn1(self.conv1(feats))))
        logits = self.conv2(h).squeeze(1)
        r = self.reg_drop1(F.relu(self.reg_bn1(self.reg_conv1(feats))))
        offsets = self.reg_conv2(r).transpose(1, 2)
        return logits, offsets, feats


def build_backbone(cfg):
    """Build the backbone from a model config dict (see configs/*.yaml)."""
    return KeypointBackbone(input_channels=cfg['input_channels'],
                            use_dedup=cfg.get('use_dedup', True),
                            gnn_k=cfg.get('gnn_k', 16))
