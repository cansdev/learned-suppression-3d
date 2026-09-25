"""Config and checkpoint helpers."""

import random

import numpy as np
import torch
import yaml

from ..models import build_backbone, build_suppression


def load_config(path):
    """Read a YAML config; the GNN neighborhood size is shared with the data loader."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['data']['gnn_k'] = cfg['model'].get('gnn_k', 16)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone(path, device):
    """Load a backbone checkpoint; the architecture is rebuilt from the stored config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_backbone(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.eval(), ckpt


def load_suppression(path, device):
    """Load a learned suppression checkpoint. Returns (module, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_suppression(ckpt['in_dim'], ckpt.get('config', {})).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.eval(), ckpt
