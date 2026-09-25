"""Learned suppression module (Sec. 3.3, Algorithm 1)."""

import torch
import torch.nn as nn


class LearnedSuppression(nn.Module):
    """Graph network over keypoint candidates that keeps, suppresses and relocates.

    Algorithm 1 with the attribute names used in the released checkpoints:
        feat_net                   MLP_emb   shared embedding z_j -> h_j in R^128
        rel_proj, relation_nets    MLP_rel   per round, [h_i, h_j, d_ij] -> s_ij
        update_nets                MLP_upd   per round, h_j += MLP_upd([h_j, u_j])
        dxyz_heads                 MLP_mov   per round, c_j += MLP_mov(h_j)
        keep_head                  MLP_keep  shared, keep score

    For every round t = 1..T:
        s_ij = sigmoid(MLP_rel([h_i, h_j, d_ij]))  for d_ij < rho, i != j, else 0
        u_j  = sum_i s_ij h_i
        h_j  = h_j + MLP_upd([h_j, u_j])
        c_j  = c_j + MLP_mov(h_j)
    pi_j = sigmoid(MLP_keep(h_j)) * (1 - max_i s_ij)   (s from the final round)

    The relocation heads are zero initialized, so training starts from pure
    rescoring. `forward` returns the logit of pi_j and the accumulated
    displacement of every candidate.
    """

    def __init__(self, in_dim, hidden=128, rel_hidden=64, dropout=0.1, suppress_radius=0.08,
                 n_rounds=2):
        super().__init__()
        self.n_rounds = n_rounds
        self.suppress_radius = suppress_radius

        self.feat_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.rel_proj = nn.ModuleList([nn.Linear(hidden, rel_hidden) for _ in range(n_rounds)])
        self.relation_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * rel_hidden + 1, rel_hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(rel_hidden, 1))
            for _ in range(n_rounds)])
        self.update_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout))
            for _ in range(n_rounds)])
        self.dxyz_heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in range(n_rounds)])
        for head in self.dxyz_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.keep_head = nn.Linear(hidden, 1)

    def forward(self, z, centroids, valid=None):
        """
        Args:
            z: [B, K, D] or [K, D] candidate features.
            centroids: [B, K, 3] or [K, 3] candidate positions.
            valid: optional [B, K] bool mask for padded batches.
        Returns:
            keep_logit [B, K], displacement [B, K, 3] (unbatched inputs give unbatched outputs).
        """
        unbatched = z.dim() == 2
        if unbatched:
            z, centroids = z.unsqueeze(0), centroids.unsqueeze(0)
            valid = valid.unsqueeze(0) if valid is not None else None

        B, K, _ = z.shape
        h = self.feat_net(z)
        c = centroids.clone()
        disp = torch.zeros_like(c)

        not_self = ~torch.eye(K, device=z.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = (valid.unsqueeze(-1) & valid.unsqueeze(-2)) if valid is not None else not_self

        s = None
        for t in range(self.n_rounds):
            d = torch.cdist(c, c)
            active = (d < self.suppress_radius) & pair_valid & not_self
            hr = self.rel_proj[t](h)
            pair = torch.cat([hr.unsqueeze(2).expand(-1, -1, K, -1),
                              hr.unsqueeze(1).expand(-1, K, -1, -1),
                              d.unsqueeze(-1)], dim=-1)
            s = torch.sigmoid(self.relation_nets[t](pair).squeeze(-1)) * active.float()  # s[b,i,j]
            u = torch.einsum('bij,bif->bjf', s, h)
            h = h + self.update_nets[t](torch.cat([h, u], dim=-1))
            step = self.dxyz_heads[t](h)
            disp = disp + step
            c = c + step

        keep = torch.sigmoid(self.keep_head(h).squeeze(-1))
        pi = (keep * (1.0 - s.max(dim=1).values)).clamp(1e-7, 1.0 - 1e-7)
        logit = torch.log(pi) - torch.log1p(-pi)
        if unbatched:
            return logit.squeeze(0), disp.squeeze(0)
        return logit, disp


def build_suppression(in_dim, cfg):
    return LearnedSuppression(in_dim,
                              hidden=cfg.get('hidden', 128),
                              rel_hidden=cfg.get('rel_hidden', 64),
                              dropout=cfg.get('dropout', 0.1),
                              suppress_radius=cfg.get('suppress_radius', 0.08),
                              n_rounds=cfg.get('n_rounds', 2))
