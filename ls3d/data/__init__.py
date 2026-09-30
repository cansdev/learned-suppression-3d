from .building3d import Building3DDataset, load_wireframe
from .common import collate
from .keypointnet import CATEGORIES, KeypointNetDataset

DATASETS = {'building3d': Building3DDataset, 'keypointnet': KeypointNetDataset}


def build_dataset(data_cfg, split, **overrides):
    """Instantiate the dataset named by `data_cfg['name']` for `split`."""
    cfg = dict(data_cfg, **overrides)
    return DATASETS[cfg['name']](cfg, split)


def input_channels(data_cfg):
    return DATASETS[data_cfg['name']].input_channels(data_cfg)


__all__ = ['Building3DDataset', 'KeypointNetDataset', 'CATEGORIES', 'build_dataset', 'collate',
           'input_channels', 'load_wireframe']
