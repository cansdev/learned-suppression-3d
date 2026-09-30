"""KeypointNet semantic keypoint dataset (16 ShapeNet categories)."""

import json
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from tqdm import tqdm

from .common import augment, nearest_keypoint, normalize_unit_sphere, sample_indices, soft_targets
from .geodesic import geodesic_knn, geodesic_to_keypoints

CATEGORIES = {
    '02691156': 'airplane', '02808440': 'bathtub', '02818832': 'bed', '02876657': 'bottle',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair', '03467517': 'guitar',
    '03513137': 'helmet', '03624134': 'knife', '03642806': 'laptop', '03790512': 'motorcycle',
    '03797390': 'mug', '04225987': 'skateboard', '04379243': 'table', '04530566': 'vessel',
}


def load_pcd(path):
    """ASCII PCD with fields `x y z rgb` (packed uint32 color)."""
    rows, started = [], False
    with open(path) as f:
        for line in f:
            if started:
                parts = line.split()
                if len(parts) >= 4:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3]))))
            elif line.strip().upper().startswith('DATA'):
                started = True
    arr = np.array(rows, dtype=np.float64)
    packed = arr[:, 3].astype(np.int64)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1)
    return arr[:, :3], rgb.astype(np.float32)


def pca_normals(xyz, k=16):
    """Unoriented normals: eigenvector of the smallest eigenvalue of the kNN covariance."""
    _, idx = cKDTree(xyz).query(xyz, k=min(k, len(xyz)))
    nb = xyz[idx] - xyz[idx].mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb) / idx.shape[1]
    return np.linalg.eigh(cov)[1][:, :, 0].astype(np.float32)


def _preprocess(args):
    pcd_file, keypoints, cache_path, cfg = args
    if os.path.exists(cache_path):
        return None
    try:
        xyz_raw, rgb = load_pcd(pcd_file)
        xyz, centroid, scale = normalize_unit_sphere(xyz_raw)
        kps = (np.asarray(keypoints, dtype=np.float64).reshape(-1, 3) - centroid) / scale
        if cfg.get('label_distance', 'geodesic') == 'geodesic' and len(kps):
            min_dist, nearest = geodesic_to_keypoints(xyz, kps, k=cfg.get('geodesic_graph_k', 10))
            offsets = (kps[nearest] - xyz).astype(np.float32)
        else:
            min_dist, offsets = nearest_keypoint(xyz, kps)
        data = dict(xyz=xyz.astype(np.float32), rgb=rgb / 255.0, normals=pca_normals(xyz, cfg.get('normal_k', 16)),
                    keypoints=kps.astype(np.float32), min_dist=min_dist, offsets=offsets,
                    centroid=centroid.astype(np.float64), scale=np.float64(scale))
        if cfg.get('gnn_graph', 'euclidean') == 'geodesic':
            data['graph_idx'], data['graph_dist'] = geodesic_knn(
                xyz, k=cfg.get('gnn_k', 16), graph_k=cfg.get('geodesic_graph_k', 10),
                limit=cfg.get('geodesic_limit', 0.3))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **data)
        return None
    except Exception as e:
        return f'{pcd_file}: {e}'


class KeypointNetDataset(Dataset):
    """KeypointNet shapes with human annotated keypoints.

    Expected layout:
        pcd_dir/<synset id>/<model id>.pcd
        annotation_dir/<category>.json     (e.g. airplane.json)
        split_dir/{train,val,test}.txt     (official lines `<synset id>-<model id>`)

    Per point input channels: XYZ, RGB and PCA normals (C = 9). Soft labels use
    the geodesic distance to the nearest keypoint by default. With
    `gnn_graph: geodesic` the dataset also returns the geodesic kNN graph used
    by the directional GNN.
    """

    def __init__(self, cfg, split):
        assert split in ('train', 'val', 'test')
        self.cfg = cfg
        self.split = split
        self.num_points = cfg['num_points']
        self.radius = cfg['label_radius']
        self.augment = cfg.get('augment', True) and split == 'train'
        self.use_color = cfg.get('use_color', True)
        self.use_normals = cfg.get('use_normals', True)
        self.use_graph = cfg.get('gnn_graph', 'euclidean') == 'geodesic'
        self.has_gt = True
        self.seed_offset = 0
        c = self.input_channels(cfg)
        self.vector_channels = [(c - 3, c)] if self.use_normals else []  # normals rotate with XYZ

        with open(os.path.join(cfg['split_dir'], f'{split}.txt')) as f:
            wanted = {tuple(l.strip().split('-', 1)) for l in f if l.strip()}
        categories = cfg.get('categories') or list(CATEGORIES)
        self.samples = []  # (pcd path, keypoints, category name)
        for synset in categories:
            with open(os.path.join(cfg['annotation_dir'], f'{CATEGORIES[synset]}.json')) as f:
                anns = json.load(f)
            for a in anns:
                if (a['class_id'], a['model_id']) not in wanted:
                    continue
                pcd = os.path.join(cfg['pcd_dir'], synset, a['model_id'] + '.pcd')
                if os.path.exists(pcd):
                    self.samples.append((pcd, [kp['xyz'] for kp in a['keypoints']], CATEGORIES[synset]))
        if not self.samples:
            raise FileNotFoundError(f'No KeypointNet samples for split "{split}"')
        self.scan_ids = list(range(len(self.samples)))
        self._preprocess_all()

    def cache_path(self, pcd_file):
        synset = os.path.basename(os.path.dirname(pcd_file))
        name = os.path.splitext(os.path.basename(pcd_file))[0]
        tag = f"{self.cfg.get('label_distance', 'geodesic')}_{self.cfg.get('gnn_graph', 'euclidean')}"
        return os.path.join(self.cfg['cache_dir'], tag, synset, name + '.npz')

    def _preprocess_all(self):
        jobs = [(p, k, self.cache_path(p), dict(self.cfg)) for p, k, _ in self.samples
                if not os.path.exists(self.cache_path(p))]
        if not jobs:
            return
        with Pool(max(1, cpu_count() - 1)) as pool:
            errors = [e for e in tqdm(pool.imap(_preprocess, jobs), total=len(jobs),
                                      desc=f'Preprocessing KeypointNet {self.split}') if e]
        if errors:
            raise RuntimeError('Preprocessing failed:\n' + '\n'.join(errors[:10]))

    @staticmethod
    def input_channels(cfg):
        return 3 + 3 * cfg.get('use_color', True) + 3 * cfg.get('use_normals', True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pcd, _, category = self.samples[index]
        c = np.load(self.cache_path(pcd))
        xyz = c['xyz']
        n = len(xyz)

        channels = [xyz]
        if self.use_color:
            channels.append(c['rgb'])
        if self.use_normals:
            channels.append(c['normals'])
        points = np.concatenate(channels, axis=1).astype(np.float32)
        normal_slice = [slice(a, b) for a, b in self.vector_channels]

        rng = None if self.split == 'train' else np.random.RandomState(index + self.seed_offset)
        idx = sample_indices(n, self.num_points, rng)
        points, min_dist, offsets = points[idx], c['min_dist'][idx], c['offsets'][idx].copy()
        keypoints = c['keypoints'].copy()

        out = {}
        if self.use_graph:
            if len(idx) == n and np.array_equal(idx, np.arange(n)):
                g_idx, g_dist = c['graph_idx'], c['graph_dist']
            else:
                g_idx, g_dist = geodesic_knn(points[:, :3].astype(np.float64), k=self.cfg.get('gnn_k', 16),
                                             graph_k=self.cfg.get('geodesic_graph_k', 10),
                                             limit=self.cfg.get('geodesic_limit', 0.3))
            out.update(graph_idx=g_idx, graph_dist=g_dist)

        if self.augment:
            points, keypoints, offsets = augment(points, keypoints, offsets, normal_slice,
                                                 jitter_std=self.cfg.get('jitter_std', 0.005))

        labels, positive, offsets = soft_targets(min_dist, offsets, self.radius)
        out.update(points=points, labels=labels, positive=positive, offsets=offsets, min_dist=min_dist,
                   keypoints=keypoints, scan_id=index, category=category,
                   centroid=c['centroid'].astype(np.float64), scale=np.float64(c['scale']))
        return out
