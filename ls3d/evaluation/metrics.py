"""Benchmark metrics.

Building3D (corner detection): per scene one to one Hungarian matching of
predicted and ground truth corners in the normalized frame; a pair within
`dist_thresh` is a true positive. ACO is the mean distance of the true
positives, CP = TP / #pred, CR = TP / #gt, CF1 their harmonic mean. Values are
averaged over scenes. Official test numbers come from the Building3D online
evaluator; this local version scores the validation split.

KeypointNet (KeypointDETR protocol): per shape Hungarian IoU = TP / (|G| + |P|
- TP) at distance thresholds 0.00 to 0.10, and the symmetric Chamfer distance.
Averaged over all test shapes, and per category.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def building3d_scene_metrics(pred, gt, dist_thresh=0.1):
    tp, dist_sum = 0, 0.0
    if len(pred) and len(gt):
        D = cdist(pred, gt)
        pi, gi = linear_sum_assignment(D)
        ok = D[pi, gi] <= dist_thresh
        tp, dist_sum = int(ok.sum()), float(D[pi, gi][ok].sum())
    cp = tp / len(pred) if len(pred) else 0.0
    cr = tp / len(gt) if len(gt) else 0.0
    return {'tp': tp, 'n_pred': len(pred), 'n_gt': len(gt),
            'aco': dist_sum / tp if tp else 0.0, 'cp': cp, 'cr': cr,
            'cf1': 2 * cp * cr / (cp + cr) if cp + cr > 0 else 0.0}


def building3d_metrics(predictions, ground_truth, dist_thresh=0.1):
    """predictions, ground_truth: dict scan_id -> [M, 3] normalized corners."""
    per_scene = {k: building3d_scene_metrics(predictions.get(k, np.zeros((0, 3))), g, dist_thresh)
                 for k, g in ground_truth.items()}
    summary = {m: float(np.mean([s[m] for s in per_scene.values()])) for m in ('aco', 'cp', 'cr', 'cf1')}
    summary['num_scenes'] = len(per_scene)
    return summary, per_scene


def hungarian_iou(D, thresh):
    n_gt, n_pred = D.shape
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    gi, pi = linear_sum_assignment(D)
    tp = np.sum(D[gi, pi] <= thresh)
    return tp / (n_gt + n_pred - tp)


def chamfer(D):
    if 0 in D.shape:
        return float('inf')
    return D.min(axis=1).mean() + D.min(axis=0).mean()


KPN_THRESHOLDS = [round(0.01 * i, 2) for i in range(11)]


def keypointnet_metrics(predictions, ground_truth, categories):
    """predictions, ground_truth: dict id -> [M, 3]; categories: dict id -> name."""
    per_shape = {}
    for k, g in ground_truth.items():
        p = predictions.get(k, np.zeros((0, 3)))
        D = cdist(g, p) if len(g) and len(p) else np.zeros((len(g), len(p)))
        per_shape[k] = {'iou': {t: hungarian_iou(D, t) for t in KPN_THRESHOLDS}, 'cd': chamfer(D),
                        'category': categories[k]}

    def summarize(items):
        cds = [s['cd'] for s in items if np.isfinite(s['cd'])]
        return {'miou': {t: float(np.mean([s['iou'][t] for s in items])) for t in KPN_THRESHOLDS},
                'cd': float(np.mean(cds)) if cds else float('inf'), 'num_shapes': len(items)}

    by_cat = defaultdict(list)
    for s in per_shape.values():
        by_cat[s['category']].append(s)
    return summarize(list(per_shape.values())), {c: summarize(v) for c, v in sorted(by_cat.items())}, per_shape
