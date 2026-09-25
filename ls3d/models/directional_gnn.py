"""Directional GNN applied to the full resolution decoder features (Sec. 3.2)."""

import torch
import torch.nn as nn

from .ops import canonical_mask, index_points, square_distance


class DirectionalGNN(nn.Module):
    """Signed, direction aware message passing over a kNN graph (Eq. 4, 5).

        e_ij  = [f_i - f_j, dir_ij, dist_ij]
        a_ij  = softmax_j(MLP_a(e_ij) / sqrt(C))
        m_ij  = MLP_m(e_ij)                       (no final activation: signed)
        f_i'  = f_i + sigmoid(lambda) * MLP_o(sum_j a_ij m_ij)

    By default the graph is the Euclidean kNN graph of the points. A precomputed
    graph (for example geodesic neighbors on KeypointNet) can be passed as
    `(knn_idx, knn_dist)`; the edge length then comes from that graph while the
    edge direction stays the unit Euclidean offset.
    """

    def __init__(self, channels, hidden_channels=128, k_neighbors=16, dropout=0.1,
                 transform_channels=128, gate_init=0.3, use_dedup=True):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.use_dedup = use_dedup
        self.scale = channels ** 0.5

        # lambda in Eq. 5; the attribute name matches the released checkpoints.
        self.alpha = nn.Parameter(torch.tensor([gate_init]))

        edge_dim = channels + 4
        self.attention_mlp = nn.Sequential(          # MLP_a
            nn.Linear(edge_dim, hidden_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1))
        self.message_mlp = nn.Sequential(            # MLP_m
            nn.Linear(edge_dim, hidden_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels))
        self.transform = nn.Sequential(              # MLP_o
            nn.Linear(channels, transform_channels), nn.GELU(),
            nn.Linear(transform_channels, channels), nn.LayerNorm(channels))

    def knn_graph(self, xyz):
        """Euclidean kNN (self excluded, canonical points preferred). xyz [B,N,3] -> [B,N,K]."""
        N = xyz.shape[1]
        dists = square_distance(xyz, xyz)
        dists[:, torch.arange(N), torch.arange(N)] = 1e15
        if self.use_dedup:
            dists = dists + (~canonical_mask(xyz)).float().unsqueeze(1) * 1e10
        return torch.topk(dists, k=self.k_neighbors, dim=-1, largest=False)[1]

    def forward(self, xyz, features, knn_idx=None, knn_dist=None):
        """
        Args:
            xyz: [B, 3, N]
            features: [B, C, N]
            knn_idx: optional [B, N, K] precomputed neighbor indices
            knn_dist: optional [B, N, K] edge lengths for `knn_idx`
        Returns:
            [B, C, N]
        """
        B, C, N = features.shape
        xyz = xyz.transpose(1, 2).contiguous()
        f = features.transpose(1, 2).contiguous()

        if knn_idx is None:
            knn_idx = self.knn_graph(xyz)
        K = knn_idx.shape[-1]

        f_j = index_points(f, knn_idx)                              # [B,N,K,C]
        offset = index_points(xyz, knn_idx) - xyz.unsqueeze(2)      # [B,N,K,3]
        euclid = offset.norm(dim=-1, keepdim=True)
        direction = offset / (euclid + 1e-8)
        dist = euclid if knn_dist is None else knn_dist.unsqueeze(-1).to(f.dtype)

        edge = torch.cat([f.unsqueeze(2).expand(-1, -1, K, -1) - f_j, direction, dist], dim=-1)

        attn = torch.softmax(self.attention_mlp(edge).squeeze(-1) / self.scale, dim=-1)
        agg = (self.message_mlp(edge) * attn.unsqueeze(-1)).sum(dim=2)
        out = f + torch.sigmoid(self.alpha) * self.transform(agg)
        return out.transpose(1, 2).contiguous()
