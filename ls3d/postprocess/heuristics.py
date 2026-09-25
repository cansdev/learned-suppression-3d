"""Heuristic post processing baselines and test time augmentation consensus."""

import numpy as np
from sklearn.cluster import DBSCAN


def dbscan_keypoints(xyz, prob, offsets, threshold=0.3, eps=0.05, min_samples=1, offset_scale=0.05):
    """DBSCAN baseline: cluster the offset shifted points above `threshold`;
    each cluster's probability weighted centroid is a keypoint (noise points are
    kept as single keypoints)."""
    mask = prob > threshold
    if not mask.any():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    pts, p = xyz[mask] + offsets[mask] * offset_scale, prob[mask]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    out, scores = [], []
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        out.append(np.average(pts[m], axis=0, weights=p[m]))
        scores.append(p[m].max())
    noise = labels < 0
    out.extend(pts[noise])
    scores.extend(p[noise])
    return np.asarray(out, np.float32).reshape(-1, 3), np.asarray(scores, np.float32)


def consensus_nms(preds, scores, radius=0.05, min_votes=2):
    """Merge keypoint sets from several passes (e.g. rotations).

    Groups within `radius` are formed greedily in descending score; a group is
    kept only if at least `min_votes` distinct passes contribute to it, and it
    is represented by its highest scoring member.
    """
    pos, sc, src = [], [], []
    for r, (p, s) in enumerate(zip(preds, scores)):
        if len(p):
            pos.append(p)
            sc.append(s)
            src.append(np.full(len(p), r))
    if not pos:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    pos, sc, src = np.concatenate(pos), np.concatenate(sc), np.concatenate(src)

    suppressed = np.zeros(len(pos), dtype=bool)
    out_p, out_s = [], []
    for i in np.argsort(-sc):
        if suppressed[i]:
            continue
        group = (np.linalg.norm(pos - pos[i], axis=1) <= radius) & ~suppressed
        members = np.where(group)[0]
        if len(np.unique(src[members])) >= min_votes:
            best = members[np.argmax(sc[members])]
            out_p.append(pos[best])
            out_s.append(sc[best])
        suppressed |= group
    return np.asarray(out_p, np.float32).reshape(-1, 3), np.asarray(out_s, np.float32)
