"""Backbone objective (Sec. 3.4, Eq. 6, 7)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def variance_covariance(features):
    """VICReg variance and covariance terms over all points of the batch.

    Args:
        features: [B, C, N]
    Returns:
        (L_var, L_cov)
    """
    B, C, N = features.shape
    x = features.transpose(1, 2).reshape(-1, C)
    x = x - x.mean(dim=0)
    std = torch.sqrt(x.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    cov = (x.T @ x) / (x.shape[0] - 1)
    off_diag = cov - torch.diag(cov.diag())
    return var_loss, (off_diag ** 2).sum() / C


class KeypointLoss(nn.Module):
    """L = L_cls + L_off + w_var L_var + w_cov L_cov.

    L_cls: focal BCE on the soft labels, each point weighted by
           1 + beta exp(-d_i / r). The focal exponent gamma anneals linearly
           from `gamma_start` to `gamma_end` over `gamma_epochs` epochs.
    L_off: Smooth L1 on the offsets of the positive points (d_i < r), summed
           over xyz and averaged over the positives.
    """

    def __init__(self, label_radius, beta=2.0, gamma_start=2.0, gamma_end=1.05, gamma_epochs=100,
                 offset_weight=1.0, var_weight=0.1, cov_weight=0.01):
        super().__init__()
        self.radius = label_radius
        self.beta = beta
        self.gamma_start, self.gamma_end, self.gamma_epochs = gamma_start, gamma_end, gamma_epochs
        self.gamma = gamma_start
        self.offset_weight = offset_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def set_epoch(self, epoch):
        progress = min(epoch / float(self.gamma_epochs), 1.0)
        self.gamma = self.gamma_start - progress * (self.gamma_start - self.gamma_end)

    def forward(self, logits, offsets, features, batch):
        y, d = batch['labels'], batch['min_dist']
        prob = torch.sigmoid(logits)
        p_t = prob * y + (1 - prob) * (1 - y)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        weight = 1.0 + self.beta * torch.exp(-d / self.radius)
        cls_loss = (weight * (1 - p_t) ** self.gamma * bce).mean()

        pos = batch['positive']
        n_pos = pos.sum()
        if n_pos > 0:
            l1 = F.smooth_l1_loss(offsets, batch['offsets'], reduction='none').sum(-1)
            off_loss = (l1 * pos).sum() / n_pos
        else:
            off_loss = logits.new_zeros(())

        var_loss, cov_loss = variance_covariance(features)
        total = cls_loss + self.offset_weight * off_loss + self.var_weight * var_loss + self.cov_weight * cov_loss
        return total, {'cls': cls_loss.item(), 'offset': off_loss.item(), 'var': var_loss.item(),
                       'cov': cov_loss.item(), 'gamma': self.gamma}
