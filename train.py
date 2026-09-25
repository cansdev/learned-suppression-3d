"""Stage 1: train the Graph-Transformer backbone.

    python train.py --config configs/building3d_entry.yaml --output_dir output/entry
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from ls3d.data import build_dataset, collate, input_channels
from ls3d.losses import KeypointLoss
from ls3d.models import build_backbone
from ls3d.postprocess import graph_from_batch
from ls3d.utils import load_config, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description='Train the keypoint backbone.')
    p.add_argument('--config', required=True)
    p.add_argument('--output_dir', default='output/backbone')
    p.add_argument('--resume', default=None, help='checkpoint to resume from')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb', action='store_true', help='log to Weights & Biases')
    return p.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when logging to a file
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg['train']
    epochs = args.epochs or tcfg['epochs']
    batch_size = args.batch_size or tcfg['batch_size']
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = build_dataset(cfg['data'], 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate,
                        num_workers=tcfg['num_workers'], pin_memory=True,
                        persistent_workers=tcfg['num_workers'] > 0)

    model_cfg = dict(cfg['model'], input_channels=input_channels(cfg['data']))
    model = build_backbone(model_cfg).to(device)
    print(f'Backbone: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, '
          f'{model_cfg["input_channels"]} input channels, {len(dataset)} training samples')

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['max_lr'], weight_decay=tcfg['weight_decay'])
    criterion = KeypointLoss(cfg['data']['label_radius'], **cfg['loss'])

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from {args.resume} (epoch {ckpt["epoch"]})')

    # On resume the one cycle schedule is rebuilt at the current step, so --epochs may change.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tcfg['max_lr'], epochs=epochs, steps_per_epoch=len(loader),
        pct_start=tcfg['pct_start'], anneal_strategy='cos', div_factor=tcfg['div_factor'],
        final_div_factor=tcfg['final_div_factor'], last_epoch=start_epoch * len(loader) - 1)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project='learned-suppression', config=cfg)

    def save(path, epoch):
        torch.save({'epoch': epoch, 'config': model_cfg, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, path)

    for epoch in range(start_epoch, epochs):
        model.train()
        criterion.set_epoch(epoch)
        t0, total = time.time(), 0.0
        for step, batch in enumerate(loader):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits, offsets, feats = model(batch['points'], batch['labels'], graph_from_batch(batch, device))
            loss, parts = criterion(logits, offsets, feats, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
            optimizer.step()
            scheduler.step()
            total += loss.item()

            if step % 20 == 0:
                print(f'epoch {epoch:3d} step {step:4d}/{len(loader)} loss {loss.item():.4f} '
                      + ' '.join(f'{k} {v:.4f}' for k, v in parts.items()))
                if run:
                    run.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0],
                             **{f'loss/{k}': v for k, v in parts.items()}})

        print(f'epoch {epoch:3d} done, mean loss {total / len(loader):.4f}, {time.time() - t0:.0f}s')
        save(os.path.join(args.output_dir, 'last.pth'), epoch)
        if (epoch + 1) % args.save_every == 0:
            save(os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth'), epoch)

    print(f'Finished. Backbone saved to {os.path.join(args.output_dir, "last.pth")}')
    if run:
        run.finish()


if __name__ == '__main__':
    main()
