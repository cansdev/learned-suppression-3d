"""Candidate generation for the learned suppression module (Sec. 3.3).

Every point whose keypoint probability exceeds the vote threshold casts a
Hough vote at p_i + r * offset_i. The votes are denoised by mean shift and
reduced to candidates by greedy NMS. Each candidate is described by
  * 10 statistics of its voters,
  * the probability weighted mean of its voters' refined backbone features,
  * the mean feature of its `k_nbr` nearest candidates,
which gives the candidate feature z_j (10 + 2 * 128 = 266 dimensions).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


@dataclass
class Candidates:
    positions: np.ndarray   # [K, 3] candidate positions c_j
    features: np.ndarray    # [K, 266] candidate features z_j
    scores: np.ndarray      # [K] backbone probability of the surviving vote


def cast_votes(xyz, prob, offsets, vote_threshold, offset_scale):
    """Votes of all points above the threshold. Returns (positions, probs, mask, offset norms)."""
    mask = prob > vote_threshold
    shift = offsets[mask] * offset_scale
    return xyz[mask] + shift, prob[mask], mask, np.linalg.norm(shift, axis=1)


def mean_shift(votes, weights, n_rounds, radius):
    """Probability weighted mean shift of the votes."""
    if n_rounds <= 0 or len(votes) <= 1:
        return votes
    pos = votes.copy()
    for _ in range(n_rounds):
        nbrs = cKDTree(pos).query_ball_point(pos, radius)
        new = pos.copy()
        for i, nb in enumerate(nbrs):
            if len(nb) > 1:
                w = weights[nb]
                new[i] = (pos[nb] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        pos = new
    return pos


def greedy_nms(votes, probs, radius):
    """Greedy NMS in descending probability. Returns survivor indices and their voters."""
    suppressed = np.zeros(len(votes), dtype=bool)
    survivors, voters = [], []
    for i in np.argsort(-probs):
        if suppressed[i]:
            continue
        group = (np.linalg.norm(votes - votes[i], axis=1) <= radius) & ~suppressed
        survivors.append(int(i))
        voters.append(np.where(group)[0])
        suppressed |= group
    return survivors, voters


def voter_statistics(votes, probs, feats, off_norm, survivors, voters):
    """10 voter statistics and the weighted mean voter feature per candidate."""
    K = len(survivors)
    stats = np.zeros((K, 10), dtype=np.float32)
    fmean = np.zeros((K, feats.shape[1]), dtype=np.float32)
    for j, (s, v) in enumerate(zip(survivors, voters)):
        p, cnt = probs[v], len(v)
        d = np.linalg.norm(votes[v] - votes[s], axis=1)
        fmean[j] = (feats[v] * (p / max(p.sum(), 1e-9))[:, None]).sum(0)
        stats[j] = [np.log1p(cnt), p.mean(), p.max(), p.std() if cnt > 1 else 0.0, probs[s],
                    np.percentile(p, 75) if cnt > 1 else p[0],
                    d.mean(), d.max(), d.std() if cnt > 1 else 0.0, off_norm[v].mean()]
    return stats, fmean


def neighbor_mean(positions, feats, k):
    """Mean feature of the k nearest other candidates."""
    K = len(positions)
    if K <= 1:
        return np.zeros_like(feats)
    d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, K - 1)
    idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    return feats[idx].mean(axis=1).astype(np.float32)


def generate_candidates(xyz, prob, offsets, feats, cfg, rotation=None):
    """Full candidate generation for one cloud.

    Args:
        xyz [N, 3], prob [N], offsets [N, 3] (units of r), feats [N, 128].
        cfg: suppression config (vote_threshold, offset_scale, refine_rounds,
             refine_radius, nms_radius, k_nbr).
        rotation: optional [3, 3] matrix applied to the votes (used to undo a
             test time rotation so candidates live in the original frame).
    Returns:
        Candidates, or None when no point passes the vote threshold.
    """
    votes, probs, mask, off_norm = cast_votes(xyz, prob, offsets, cfg['vote_threshold'], cfg['offset_scale'])
    if len(votes) == 0:
        return None
    if rotation is not None:
        votes = votes @ rotation.T
    votes = mean_shift(votes, probs, cfg['refine_rounds'], cfg['refine_radius'])
    survivors, voters = greedy_nms(votes, probs, cfg['nms_radius'])
    stats, fmean = voter_statistics(votes, probs, feats[mask], off_norm, survivors, voters)
    positions = votes[survivors].astype(np.float32)
    z = np.concatenate([stats, fmean, neighbor_mean(positions, fmean, cfg['k_nbr'])], axis=1)
    return Candidates(positions, z.astype(np.float32), probs[survivors].astype(np.float32))


def hungarian_targets(positions, keypoints, match_thresh):
    """Keep target 1 and displacement target for candidates matched to a keypoint."""
    keep = np.zeros(len(positions), dtype=np.float32)
    disp = np.zeros((len(positions), 3), dtype=np.float32)
    if len(positions) == 0 or len(keypoints) == 0:
        return keep, disp
    D = cdist(positions, keypoints)
    ci, ki = linear_sum_assignment(D)
    ok = D[ci, ki] <= match_thresh
    keep[ci[ok]] = 1.0
    disp[ci[ok]] = keypoints[ki[ok]] - positions[ci[ok]]
    return keep, disp
