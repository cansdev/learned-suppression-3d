"""End to end keypoint prediction: backbone, candidates, post processing, TTA."""

import math

import numpy as np
import torch

from .candidates import generate_candidates
from .heuristics import consensus_nms, dbscan_keypoints


def rotz_np(angle_deg):
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotate_points(points, R, vector_channels=()):
    """Rotate XYZ and any per point vector channels of a [B, N, C] tensor."""
    Rt = torch.as_tensor(R, dtype=points.dtype, device=points.device).T
    out = points.clone()
    out[..., :3] = points[..., :3] @ Rt
    for a, b in vector_channels:
        out[..., a:b] = points[..., a:b] @ Rt
    return out


def graph_from_batch(batch, device):
    if 'graph_idx' not in batch:
        return None
    return batch['graph_idx'].to(device), batch['graph_dist'].to(device)


@torch.no_grad()
def run_backbone(backbone, points, graph=None):
    """Returns numpy xyz [B,N,3], prob [B,N], offsets [B,N,3], feats [B,N,128]."""
    logits, offsets, feats = backbone(points, None, graph)
    return (points[..., :3].cpu().numpy(), torch.sigmoid(logits).cpu().numpy(),
            offsets.cpu().numpy(), feats.transpose(1, 2).cpu().numpy())


class KeypointPredictor:
    """Predicts keypoints for a batch in normalized coordinates.

    postproc:
        'learned'  candidates -> learned suppression -> keep pi_j > tau, relocate
        'nms'      the candidate generation stage alone (greedy NMS on the votes)
        'dbscan'   DBSCAN on the offset shifted points above a threshold
    With several `tta_angles`, the cloud is also passed at those rotations about
    z and a keypoint is kept only if found in at least `min_votes` passes.
    """

    def __init__(self, backbone, cfg, suppression=None, postproc='learned', keep_threshold=None,
                 tta_angles=(0,), min_votes=2, vector_channels=(), dbscan_cfg=None, device='cuda'):
        self.backbone = backbone.eval()
        self.suppression = suppression.eval() if suppression is not None else None
        self.cfg = cfg
        self.postproc = postproc
        self.tau = keep_threshold if keep_threshold is not None else cfg['keep_threshold']
        self.angles = list(tta_angles)
        self.min_votes = min(min_votes, len(self.angles))
        self.vector_channels = vector_channels
        self.dbscan_cfg = dbscan_cfg or {}
        self.device = device
        if postproc == 'learned' and suppression is None:
            raise ValueError('postproc="learned" needs a trained suppression module')

    @torch.no_grad()
    def _single(self, xyz, prob, off, feats, R_back):
        if self.postproc == 'dbscan':
            p, s = dbscan_keypoints(xyz, prob, off, offset_scale=self.cfg['offset_scale'], **self.dbscan_cfg)
            return p @ R_back.T, s
        cand = generate_candidates(xyz, prob, off, feats, self.cfg, rotation=R_back)
        if cand is None:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
        if self.postproc == 'nms':
            return cand.positions, cand.scores
        z = torch.from_numpy(cand.features).to(self.device)
        c = torch.from_numpy(cand.positions).to(self.device)
        logit, disp = self.suppression(z, c)
        pi = torch.sigmoid(logit).cpu().numpy()
        keep = pi > self.tau
        return (cand.positions + disp.cpu().numpy())[keep], pi[keep]

    @torch.no_grad()
    def __call__(self, batch):
        points = batch['points'].to(self.device)
        graph = graph_from_batch(batch, self.device)
        per_angle = []
        for a in self.angles:
            R = rotz_np(a)
            outs = run_backbone(self.backbone, rotate_points(points, R, self.vector_channels) if a else points, graph)
            per_angle.append([self._single(*(o[b] for o in outs), rotz_np(-a)) for b in range(points.shape[0])])

        if len(self.angles) == 1 and self.postproc == 'dbscan':
            return per_angle[0]
        # With a single pass (min_votes = 1) this is a final NMS over the kept keypoints.
        return [consensus_nms([r[b][0] for r in per_angle], [r[b][1] for r in per_angle],
                              radius=self.cfg['nms_radius'], min_votes=self.min_votes)
                for b in range(points.shape[0])]
