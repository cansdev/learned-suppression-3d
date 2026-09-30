"""Building3D roof corner dataset (Entry-Level and Tallinn)."""

import glob
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .common import augment, nearest_keypoint, normalize_unit_sphere, sample_indices, soft_targets


def load_wireframe(path):
    """Read the vertices and 0 indexed edges of a Building3D .obj wireframe."""
    vertices, edges = [], set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            if parts[0] == 'v':
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'l':
                edges.add(tuple(sorted(int(i) - 1 for i in parts[1:3])))
    return np.array(vertices, dtype=np.float64).reshape(-1, 3), np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _read_ids(path):
    if not path:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _preprocess(args):
    xyz_file, obj_file, cache_path = args
    if os.path.exists(cache_path):
        return None
    try:
        pc = np.loadtxt(xyz_file, dtype=np.float64, ndmin=2)
        n = len(pc)
        rgba = pc[:, 3:7] if pc.shape[1] >= 7 else np.zeros((n, 4))
        intensity = pc[:, 7:8] if pc.shape[1] >= 8 else np.zeros((n, 1))
        xyz, centroid, scale = normalize_unit_sphere(pc[:, :3])
        data = dict(xyz=xyz.astype(np.float32),
                    rgba=(rgba / 256.0).astype(np.float32),
                    intensity=(intensity / 65535.0 if intensity.max() > 0 else intensity).astype(np.float32),
                    centroid=centroid.astype(np.float64), scale=np.float64(scale))
        if obj_file is not None and os.path.exists(obj_file):
            corners = (load_wireframe(obj_file)[0] - centroid) / scale
            data['min_dist'], data['offsets'] = nearest_keypoint(xyz, corners)
            data['keypoints'] = corners.astype(np.float32)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **data)
        return None
    except Exception as e:  # report and continue with the other files
        return f'{xyz_file}: {e}'


class Building3DDataset(Dataset):
    """Building3D corners as 3D keypoints.

    Expected layout (the official release):
        root_dir/train/xyz/<id>.xyz   root_dir/train/wireframe/<id>.obj
        root_dir/test/xyz/<id>.xyz    (no public ground truth)

    `val_ids` names a list of training scan ids held out as the validation
    split; `train_ids` optionally restricts the training pool. Each scene is
    normalized to the unit sphere once and cached under `cache_dir`.

    Per point input channels: XYZ, RGBA, intensity and, with
    `use_scale_features`, four constant channels (log of the normalization
    scale and the three bounding box extents of the normalized scene).
    """

    def __init__(self, cfg, split):
        assert split in ('train', 'validation', 'test')
        self.cfg = cfg
        self.split = split
        self.num_points = cfg['num_points']
        self.radius = cfg['label_radius']
        self.augment = cfg.get('augment', True) and split == 'train'
        self.use_color = cfg.get('use_color', True)
        self.use_intensity = cfg.get('use_intensity', True)
        self.use_scale_features = cfg.get('use_scale_features', True)
        self.seed_offset = 0
        self.vector_channels = []

        folder = 'test' if split == 'test' else 'train'
        files = sorted(glob.glob(os.path.join(cfg['root_dir'], folder, 'xyz', '*.xyz')))
        if split != 'test':
            val_ids = _read_ids(cfg.get('val_ids'))
            train_ids = _read_ids(cfg.get('train_ids'))
            ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
            if split == 'validation':
                if val_ids is None:
                    raise ValueError('The validation split needs `val_ids` in the data config.')
                files = [f for f, i in zip(files, ids) if i in val_ids]
            else:
                files = [f for f, i in zip(files, ids)
                         if (val_ids is None or i not in val_ids) and (train_ids is None or i in train_ids)]
        if not files:
            raise FileNotFoundError(f'No .xyz files for split "{split}" under {cfg["root_dir"]}')
        self.files = files
        self.scan_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
        self.has_gt = split != 'test'
        self._cache_dir = os.path.join(cfg['cache_dir'], folder)
        self._preprocess_all()

    def cache_path(self, xyz_file):
        return os.path.join(self._cache_dir, os.path.splitext(os.path.basename(xyz_file))[0] + '.npz')

    def _preprocess_all(self):
        jobs = []
        for f in self.files:
            obj = f.replace(os.sep + 'xyz' + os.sep, os.sep + 'wireframe' + os.sep)[:-4] + '.obj'
            if not os.path.exists(self.cache_path(f)):
                jobs.append((f, obj if self.has_gt else None, self.cache_path(f)))
        if not jobs:
            return
        with Pool(max(1, cpu_count() - 1)) as pool:
            errors = [e for e in tqdm(pool.imap(_preprocess, jobs), total=len(jobs),
                                      desc=f'Preprocessing Building3D {self.split}') if e]
        if errors:
            raise RuntimeError('Preprocessing failed:\n' + '\n'.join(errors[:10]))

    @staticmethod
    def input_channels(cfg):
        return (3 + 4 * cfg.get('use_color', True) + cfg.get('use_intensity', True)
                + 4 * cfg.get('use_scale_features', True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        f = self.files[index]
        scan_id = int(os.path.splitext(os.path.basename(f))[0])
        c = np.load(self.cache_path(f))
        xyz = c['xyz']
        n = len(xyz)

        channels = [xyz]
        if self.use_color:
            channels.append(c['rgba'])
        if self.use_intensity:
            channels.append(c['intensity'])
        if self.use_scale_features:
            extent = (xyz.max(0) - xyz.min(0)) / 2.0
            scale_ch = np.concatenate([[np.log(float(c['scale']) + 1e-8)], extent]).astype(np.float32)
            channels.append(np.tile(scale_ch, (n, 1)))
        points = np.concatenate(channels, axis=1).astype(np.float32)

        if self.has_gt:
            min_dist, offsets, keypoints = c['min_dist'], c['offsets'], c['keypoints'].copy()
        else:
            min_dist, offsets = np.full(n, 1e3, np.float32), np.zeros((n, 3), np.float32)
            keypoints = np.zeros((0, 3), np.float32)

        # Random subset per epoch for training, a fixed subset per scene otherwise.
        rng = None if self.split == 'train' else np.random.RandomState(scan_id + self.seed_offset)
        idx = sample_indices(n, self.num_points, rng)
        points, min_dist, offsets = points[idx], min_dist[idx], offsets[idx].copy()

        if self.augment:
            points, keypoints, offsets = augment(points, keypoints, offsets)

        labels, positive, offsets = soft_targets(min_dist, offsets, self.radius)
        return dict(points=points, labels=labels, positive=positive, offsets=offsets, min_dist=min_dist,
                    keypoints=keypoints.astype(np.float32), scan_id=scan_id,
                    centroid=c['centroid'].astype(np.float64), scale=np.float64(c['scale']))
