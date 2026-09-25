"""Stage 2: train the learned suppression module on top of the frozen backbone.

Phase 1 runs the frozen backbone over the training scenes at `n_aug` random
rotations about z and stores the candidates of every pass with their Hungarian
targets. Phase 2 trains the module on these candidates, and keeps the epoch
and keep threshold tau with the best mean per scene F1 on the validation split.

    python train_suppression.py --config configs/building3d_entry.yaml \
        --backbone output/entry/last.pth --output_dir output/entry
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from tqdm import tqdm

from ls3d.data import build_dataset, collate
from ls3d.models import build_suppression
from ls3d.postprocess import generate_candidates, graph_from_batch, hungarian_targets, rotate_points, rotz_np, run_backbone
from ls3d.utils import load_backbone, load_config, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description='Train the learned suppression module.')
    p.add_argument('--config', required=True)
    p.add_argument('--backbone', required=True, help='trained backbone checkpoint')
    p.add_argument('--output_dir', default='output/suppression')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def build_candidates(backbone, dataset, scfg, device, angles, num_workers, desc):
    """Candidates and Hungarian targets per scene, pooled over the given rotations (degrees)."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate, num_workers=num_workers)
    vec = getattr(getattr(dataset, 'dataset', dataset), 'vector_channels', [])
    scenes = {}
    n_cand = n_match = n_gt = 0
    for a in angles:
        R = rotz_np(a)
        for batch in tqdm(loader, desc=f'{desc} rotation {a:.0f} deg'):
            points = rotate_points(batch['points'].to(device), R, vec)
            outs = run_backbone(backbone, points, graph_from_batch(batch, device))
            for b in range(points.shape[0]):
                gt = batch['keypoints'][b].numpy() @ R.T
                sid = int(batch['scan_id'][b])
                n_gt += len(gt)
                cand = generate_candidates(*(o[b] for o in outs), scfg)
                if cand is None:
                    continue
                keep, disp = hungarian_targets(cand.positions, gt, scfg['match_thresh'])
                s = scenes.setdefault(sid, {'z': [], 'c': [], 'keep': [], 'disp': [], 'gt': gt})
                s['z'].append(cand.features)
                s['c'].append(cand.positions)
                s['keep'].append(keep)
                s['disp'].append(disp)
                n_cand += len(keep)
                n_match += int(keep.sum())
    print(f'[{desc}] {len(scenes)} scenes, {n_cand} candidates, {n_match} matched, '
          f'candidate recall {n_match / max(n_gt, 1):.4f}')
    return {k: {'z': np.concatenate(v['z']), 'c': np.concatenate(v['c']), 'keep': np.concatenate(v['keep']),
                'disp': np.concatenate(v['disp']), 'gt': v['gt']} for k, v in scenes.items()}


def pad_collate(items):
    B, K = len(items), max(len(s['z']) for s in items)
    D = items[0]['z'].shape[1]
    z, c, disp = torch.zeros(B, K, D), torch.zeros(B, K, 3), torch.zeros(B, K, 3)
    keep, valid = torch.zeros(B, K), torch.zeros(B, K, dtype=torch.bool)
    for i, s in enumerate(items):
        k = len(s['z'])
        z[i, :k], c[i, :k] = torch.from_numpy(s['z']), torch.from_numpy(s['c'])
        keep[i, :k], disp[i, :k] = torch.from_numpy(s['keep']), torch.from_numpy(s['disp'])
        valid[i, :k] = True
    return z, c, keep, disp, valid


@torch.no_grad()
def validate(module, scenes, device, thresholds, match_thresh):
    """Mean per scene F1 at every keep threshold."""
    module.eval()
    f1 = {t: [] for t in thresholds}
    for s in scenes.values():
        logit, disp = module(torch.from_numpy(s['z']).to(device), torch.from_numpy(s['c']).to(device))
        pi = torch.sigmoid(logit).cpu().numpy()
        pos = s['c'] + disp.cpu().numpy()
        for t in thresholds:
            pred, gt = pos[pi > t], s['gt']
            tp = 0
            if len(pred) and len(gt):
                D = cdist(pred, gt)
                i, j = linear_sum_assignment(D)
                tp = int((D[i, j] <= match_thresh).sum())
            p, r = (tp / len(pred) if len(pred) else 0.0), (tp / len(gt) if len(gt) else 0.0)
            f1[t].append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return {t: float(np.mean(v)) for t, v in f1.items()}


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when logging to a file
    args = parse_args()
    cfg = load_config(args.config)
    scfg = cfg['suppression']
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    backbone, _ = load_backbone(args.backbone, device)
    train_set = build_dataset(cfg['data'], 'train', augment=False)

    print('Phase 1: candidate generation on the frozen backbone')
    rng = np.random.default_rng(args.seed)
    angles = [math.degrees(rng.uniform(0, 2 * math.pi)) for _ in range(scfg['n_aug'])]
    train_scenes = build_candidates(backbone, train_set, scfg, device, angles, args.num_workers, 'train')
    if scfg.get('val_split'):
        val_set = build_dataset(cfg['data'], scfg['val_split'], augment=False)
        val_scenes = build_candidates(backbone, val_set, scfg, device, [0.0], args.num_workers, 'validation')
    else:  # hold out 20% of the training scenes
        ids = sorted(train_scenes)
        held = set(rng.choice(ids, size=max(1, len(ids) // 5), replace=False).tolist())
        subset = torch.utils.data.Subset(train_set, [i for i, s in enumerate(train_set.scan_ids) if s in held])
        val_scenes = build_candidates(backbone, subset, scfg, device, [0.0], args.num_workers, 'held out')
        train_scenes = {k: v for k, v in train_scenes.items() if k not in held}
    del backbone
    torch.cuda.empty_cache()

    in_dim = next(iter(train_scenes.values()))['z'].shape[1]
    module = build_suppression(in_dim, scfg).to(device)
    print(f'Phase 2: learned suppression, {sum(p.numel() for p in module.parameters()) / 1e6:.3f}M parameters, '
          f'input dim {in_dim}')
    optimizer = torch.optim.Adam(module.parameters(), lr=scfg['lr'], weight_decay=scfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scfg['epochs'])
    loader = DataLoader(list(train_scenes.values()), batch_size=scfg['batch_size'], shuffle=True,
                        collate_fn=pad_collate)

    best = {'f1': -1.0}
    for epoch in range(scfg['epochs']):
        module.train()
        losses = []
        for z, c, keep, disp, valid in loader:
            z, c, keep, disp, valid = (x.to(device) for x in (z, c, keep, disp, valid))
            logit, pred_disp = module(z, c, valid)
            bce = F.binary_cross_entropy_with_logits(logit[valid], keep[valid])
            pos = valid & (keep > 0.5)
            reg = (F.smooth_l1_loss(pred_disp[pos], disp[pos], beta=scfg['smooth_l1_beta'])
                   if pos.any() else logit.new_zeros(()))
            loss = bce + scfg['offset_weight'] * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        f1 = validate(module, val_scenes, device, scfg['keep_thresholds'], scfg['match_thresh'])
        tau = max(f1, key=f1.get)
        print(f'epoch {epoch:3d} loss {np.mean(losses):.4f}  best tau {tau:.2f}  validation F1 {f1[tau]:.4f}')
        if f1[tau] > best['f1']:
            best = {'f1': f1[tau], 'tau': tau, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}}

    config = {k: scfg[k] for k in ('vote_threshold', 'offset_scale', 'refine_rounds', 'refine_radius', 'nms_radius',
                                   'k_nbr', 'n_rounds', 'suppress_radius', 'hidden', 'rel_hidden', 'dropout',
                                   'match_thresh')}
    path = os.path.join(args.output_dir, 'suppression.pth')
    torch.save({'model_state_dict': best['state'], 'in_dim': in_dim, 'config': config,
                'best_threshold': best['tau'], 'best_epoch': best['epoch'], 'best_f1': best['f1']}, path)
    print(f'Best epoch {best["epoch"]}, tau {best["tau"]}, validation F1 {best["f1"]:.4f}. Saved {path}')


if __name__ == '__main__':
    main()
