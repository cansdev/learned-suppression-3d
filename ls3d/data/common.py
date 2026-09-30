"""Shared dataset utilities: normalization, sampling, augmentation, collation."""

import numpy as np
import torch


def normalize_unit_sphere(xyz):
    """Center on the centroid and scale by the maximum radius."""
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return centered / scale, centroid, scale


def nearest_keypoint(xyz, keypoints):
    """Euclidean distance and offset from every point to its nearest keypoint."""
    if len(keypoints) == 0:
        n = len(xyz)
        return np.full(n, 1e3, np.float32), np.zeros((n, 3), np.float32)
    d = np.linalg.norm(xyz[:, None, :].astype(np.float64) - keypoints[None].astype(np.float64), axis=2)
    idx = d.argmin(axis=1)
    return d[np.arange(len(xyz)), idx].astype(np.float32), (keypoints[idx] - xyz).astype(np.float32)


def sample_indices(n, num_points, rng=None):
    """Random subset (or upsampling with replacement) to exactly `num_points`."""
    rng = rng or np.random
    if n == num_points:
        return np.arange(n)
    return rng.choice(n, num_points, replace=n < num_points)


def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def augment(points, keypoints, offsets, vector_slices=(), jitter_std=0.0):
    """Random flips about the YZ and XZ planes and a random rotation about z.

    Applied identically to the XYZ channels, the keypoints, the offset targets
    and any other per point 3D vectors (e.g. normals) given by `vector_slices`.
    """
    vecs = [points[:, 0:3], keypoints, offsets] + [points[:, s] for s in vector_slices]
    if jitter_std > 0:
        points[:, 0:3] += np.random.randn(len(points), 3).astype(np.float32) * jitter_std
    for axis in (0, 1):
        if np.random.random() > 0.5:
            for v in vecs:
                if len(v):
                    v[:, axis] = -v[:, axis]
    R = rotz(np.random.uniform(-np.pi, np.pi)).T
    points[:, 0:3] = points[:, 0:3] @ R
    if len(keypoints):
        keypoints[:] = keypoints @ R
    offsets[:] = offsets @ R
    for s in vector_slices:
        points[:, s] = points[:, s] @ R
    return points, keypoints, offsets


def soft_targets(min_dist, offsets, radius):
    """Soft labels y = clip(exp(-d/r), 0, 1), positive mask d < r, offsets / r."""
    labels = np.clip(np.exp(-min_dist / radius), 0.0, 1.0).astype(np.float32)
    positive = (min_dist < radius).astype(np.float32)
    return labels, positive, (offsets / radius).astype(np.float32)


STACK_KEYS = ('points', 'labels', 'offsets', 'positive', 'min_dist', 'graph_idx', 'graph_dist')


def collate(batch):
    """Stack fixed size arrays; keep ground truth keypoints as a list."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key in STACK_KEYS:
            out[key] = torch.from_numpy(np.stack(vals))
        elif key == 'keypoints':
            out[key] = [torch.from_numpy(v.astype(np.float32)) for v in vals]
        elif isinstance(vals[0], str):
            out[key] = vals
        else:
            out[key] = torch.from_numpy(np.asarray(vals))
    return out
