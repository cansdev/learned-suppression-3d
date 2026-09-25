"""Predict keypoints on a split and, when ground truth is available, score them.

    # Building3D validation with learned suppression and 4 rotation TTA
    python test.py --config configs/building3d_entry.yaml --backbone output/entry/last.pth \
        --suppression output/entry/suppression.pth --split validation

    # heuristic baselines on the same backbone
    python test.py ... --postproc nms
    python test.py ... --postproc dbscan
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ls3d.data import build_dataset, collate
from ls3d.evaluation import building3d_metrics, keypointnet_metrics
from ls3d.postprocess import KeypointPredictor
from ls3d.utils import load_backbone, load_config, load_suppression


def parse_args():
    p = argparse.ArgumentParser(description='Keypoint inference and evaluation.')
    p.add_argument('--config', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--suppression', default=None, help='required for --postproc learned')
    p.add_argument('--split', default=None, help='validation / test (Building3D), val / test (KeypointNet)')
    p.add_argument('--postproc', choices=['learned', 'nms', 'dbscan'], default='learned')
    p.add_argument('--keep_threshold', type=float, default=None, help='tau; default: value chosen in training')
    p.add_argument('--tta', type=str, default=None, help='rotation angles, e.g. "0,90,180,270" or "0"')
    p.add_argument('--min_votes', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir', default='output/predictions')
    return p.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when logging to a file
    args = parse_args()
    cfg = load_config(args.config)
    icfg = cfg['inference']
    name = cfg['data']['name']
    split = args.split or ('validation' if name == 'building3d' else 'test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone, _ = load_backbone(args.backbone, device)
    scfg, suppression, tau = dict(cfg['suppression']), None, args.keep_threshold
    if args.postproc == 'learned':
        suppression, ckpt = load_suppression(args.suppression, device)
        scfg.update(ckpt.get('config', {}))
        tau = tau if tau is not None else ckpt.get('best_threshold', scfg['keep_threshold'])
        print(f'Learned suppression, tau = {tau}')

    dataset = build_dataset(cfg['data'], split, augment=False)
    angles = [float(a) for a in args.tta.split(',')] if args.tta else icfg['tta_angles']
    predictor = KeypointPredictor(backbone, scfg, suppression, postproc=args.postproc, keep_threshold=tau,
                                  tta_angles=angles, min_votes=args.min_votes or icfg['min_votes'],
                                  vector_channels=dataset.vector_channels, dbscan_cfg=icfg['dbscan'],
                                  device=device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=args.num_workers)
    pred_dir = os.path.join(args.output_dir, f'{name}_{split}_{args.postproc}')
    os.makedirs(pred_dir, exist_ok=True)
    preds, gts, cats = {}, {}, {}
    for batch in tqdm(loader, desc=f'{args.postproc} on {split}'):
        for b, (kp, score) in enumerate(predictor(batch)):
            sid = int(batch['scan_id'][b])
            scale, centroid = float(batch['scale'][b]), batch['centroid'][b].numpy().astype(np.float64)
            preds[sid] = kp
            gts[sid] = batch['keypoints'][b].numpy()
            if 'category' in batch:
                cats[sid] = batch['category'][b]
            np.savez(os.path.join(pred_dir, f'{sid}.npz'), keypoints=kp, scores=score,
                     keypoints_world=kp.astype(np.float64) * scale + centroid)
    print(f'Saved {len(preds)} predictions to {pred_dir}')

    if not dataset.has_gt:
        print('No ground truth for this split; use make_submission.py for the online evaluator.')
        return

    if name == 'building3d':
        summary, _ = building3d_metrics(preds, gts, icfg['eval_dist_thresh'])
        print(f"\n{summary['num_scenes']} scenes  ACO {summary['aco']:.4f}  CP {100 * summary['cp']:.2f}  "
              f"CR {100 * summary['cr']:.2f}  CF1 {100 * summary['cf1']:.2f}")
    else:
        overall, per_cat, _ = keypointnet_metrics(preds, gts, cats)
        print(f"\n{'category':<12} {'mIoU@0.10':>10} {'CD':>8} {'N':>6}")
        for c, m in per_cat.items():
            print(f"{c:<12} {100 * m['miou'][0.1]:>10.2f} {m['cd']:>8.3f} {m['num_shapes']:>6}")
        print(f"{'all shapes':<12} {100 * overall['miou'][0.1]:>10.2f} {overall['cd']:>8.3f} "
              f"{overall['num_shapes']:>6}")
        summary = {'overall': overall, 'per_category': per_cat}
    with open(os.path.join(args.output_dir, f'{name}_{split}_{args.postproc}_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == '__main__':
    main()
