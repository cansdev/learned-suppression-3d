"""Point Transformer block with the spectral positional encoding (Sec. 3.1).

Attribute names follow the checkpoints used in the paper so that trained
weights load without key remapping.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_lastdim(seq, x):
    """Apply a sequence of Linear/BatchNorm1d layers over the last dimension."""
    shape = x.shape
    out = x.reshape(-1, shape[-1])
    for layer in seq:
        out = layer(out)
    return out.view(*shape[:-1], -1)


class SpectralPositionalEncoding(nn.Module):
    """Geometry adaptive Fourier encoding of neighbor offsets (Eq. 1, Fig. 3b).

    For a center i with neighbor offsets dp_ij = p_j - p_i:
      1. A 64 dimensional neighborhood descriptor g_i: the scale normalized
         offsets are Fourier embedded, pooled by mean, std and max over the
         k neighbors, projected and layer normalized.
      2. mu_i = MLP(g_i) gives per axis multipliers for the octave bands
         omega_base = (pi, 2 pi, ..., 2^(L-1) pi); omega_i = omega_base * mu_i.
      3. Directional encoding gamma_dir evaluates [sin, cos] of dp_ij at
         omega_i per axis; radial encoding gamma_rad evaluates [sin, cos] of
         |dp_ij| at the fixed bands.
      4. psi maps [gamma_dir, gamma_rad, g_i] to an additive bias delta_ij and
         a sigmoid gate delta_gate_ij, both of dimension C.
    """

    def __init__(self, out_channels, num_bands=6, geo_hidden=32, n_freq=16, d=32,
                 freq_init_sigma=2.0):
        super().__init__()
        self.num_bands = num_bands
        geo_dim = 64

        self.register_buffer('base_freqs', math.pi * (2.0 ** torch.arange(num_bands, dtype=torch.float32)))

        # Neighborhood descriptor g_i.
        self.freqs = nn.Parameter(torch.randn(n_freq, 3) * freq_init_sigma)
        self.embed = nn.Linear(4 + 2 * n_freq, d, bias=False)
        self.to_geo = nn.Linear(3 * d, geo_dim)
        self.norm_geo = nn.LayerNorm(geo_dim)

        # Band multipliers mu_i, initialized to 1 (softplus(0.55) ~ 1).
        self.freq_net = nn.Sequential(
            nn.Linear(geo_dim, geo_hidden),
            nn.GELU(),
            nn.Linear(geo_hidden, num_bands * 3),
            nn.Softplus(),
        )
        nn.init.normal_(self.freq_net[2].weight, std=0.01)
        nn.init.constant_(self.freq_net[2].bias, 0.55)

        # psi: bias and gate heads.
        proj_in = num_bands * 8 + geo_dim  # 6L directional + 2L radial + descriptor
        self.proj = nn.Sequential(
            nn.Linear(proj_in, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(proj_in, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
        )

    def descriptor(self, rel_xyz):
        """rel_xyz [B, S, K, 3] -> g [B, S, 64]."""
        scale = rel_xyz.pow(2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()
        R = rel_xyz / scale[..., None, None]
        dist = R.norm(dim=-1, keepdim=True)
        proj = torch.einsum('bskd,nd->bskn', R, self.freqs)
        xe = self.embed(torch.cat([R, dist, proj.sin(), proj.cos()], dim=-1))
        pooled = torch.cat([xe.mean(2), xe.std(2, unbiased=False), xe.amax(2)], dim=-1)
        return self.norm_geo(self.to_geo(pooled))

    def forward(self, rel_xyz):
        """rel_xyz [B, S, K, 3] -> (delta, delta_gate), each [B, S, K, C]."""
        B, S, K, _ = rel_xyz.shape
        L = self.num_bands

        geo = self.descriptor(rel_xyz)
        eff_freqs = self.base_freqs.view(1, 1, L, 1) * self.freq_net(geo).view(B, S, L, 3)

        inner = rel_xyz.unsqueeze(3) * eff_freqs.unsqueeze(2)                   # [B,S,K,L,3]
        dir_feats = torch.cat([inner.sin(), inner.cos()], dim=-1).reshape(B, S, K, 6 * L)
        rad_inner = rel_xyz.norm(dim=-1, keepdim=True) * self.base_freqs.view(1, 1, 1, L)
        rad_feats = torch.cat([rad_inner.sin(), rad_inner.cos()], dim=-1)

        phi = torch.cat([dir_feats, rad_feats, geo.unsqueeze(2).expand(-1, -1, K, -1)], dim=-1)
        return _apply_lastdim(self.proj, phi), _apply_lastdim(self.gate_proj, phi)


class VectorAttention(nn.Module):
    """Grouped vector attention (Eq. 2) with the optional training prior (Eq. 3).

        w_ij  = phi(delta_gate_ij * (q_i - k_j) + delta_ij)
        w_ij' = w_ij + log(y_j + 0.01)        (SA4, training only)
        out_i = sum_j softmax_j(w_ij) * (v_j + delta_ij)

    phi outputs one weight per group of `share_planes` channels.
    """

    def __init__(self, channels, share_planes=8, num_bands=6):
        super().__init__()
        assert channels % share_planes == 0
        C, g = channels, share_planes
        self.channels = C
        self.share_planes = g

        self.proj_q = nn.Linear(C, C, bias=False)
        self.proj_k = nn.Linear(C, C, bias=False)
        self.proj_v = nn.Linear(C, C, bias=False)
        self.pe = SpectralPositionalEncoding(C, num_bands=num_bands)
        self.phi = nn.Sequential(
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.Conv1d(C, C // g, 1, bias=False),
            nn.BatchNorm1d(C // g),
            nn.ReLU(inplace=True),
            nn.Conv1d(C // g, C // g, 1, bias=False),
        )

    def forward(self, rel_xyz, neigh, center, prior=None):
        """
        Args:
            rel_xyz: [B, S, K, 3]
            neigh:   [B, S, K, C]
            center:  [B, S, C]
            prior:   [B, S, K] soft keypoint labels of the neighbors, or None
        Returns:
            [B, S, C]
        """
        B, S, K, _ = rel_xyz.shape
        C, g = self.channels, self.share_planes
        N = B * S

        q = self.proj_q(center.reshape(N, C))
        k = self.proj_k(neigh.reshape(N * K, C)).view(N, K, C)
        v = self.proj_v(neigh.reshape(N * K, C)).view(N, K, C)

        delta, gate = self.pe(rel_xyz)
        delta = delta.view(N, K, C)
        gate = gate.view(N, K, C)

        w = self.phi((gate * (q.unsqueeze(1) - k) + delta).transpose(1, 2)).transpose(1, 2)  # [N,K,C/g]
        if prior is not None:
            w = w + torch.log(prior.reshape(N, K, 1) + 0.01)
        w = torch.nan_to_num(torch.softmax(w, dim=1), nan=0.0, posinf=0.0, neginf=0.0)

        out = ((v + delta).view(N, K, g, C // g) * w.unsqueeze(2)).sum(dim=1)
        return out.reshape(B, S, C)


class PointTransformerBlock(nn.Module):
    """Shared input projection, vector attention, residual, feed forward, residual."""

    def __init__(self, in_channels, out_channels, share_planes=8, ffn_ratio=4.0, num_bands=6):
        super().__init__()
        self.out_channels = out_channels
        self.pre_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pre_bn = nn.BatchNorm1d(out_channels)
        self.va = VectorAttention(out_channels, share_planes=share_planes, num_bands=num_bands)
        hidden = int(out_channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels, bias=False),
        )

    def _project(self, x):
        x = self.pre_linear(x)
        x = self.pre_bn(x.reshape(-1, x.shape[-1])).view(x.shape)
        return F.relu(x, inplace=True)

    def forward(self, rel_xyz, neigh_feats, center_feats, prior=None):
        B, S, K, _ = rel_xyz.shape
        neigh = self._project(neigh_feats)
        center = self._project(center_feats)
        x = self.va(rel_xyz, neigh, center, prior) + center
        return x + self.ffn(x.reshape(-1, self.out_channels)).view(B, S, self.out_channels)
