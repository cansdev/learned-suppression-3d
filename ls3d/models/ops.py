"""Point cloud primitives: distances, gathering, farthest point sampling, interpolation."""

import torch


def square_distance(src, dst):
    """Pairwise squared Euclidean distance.

    Args:
        src: [B, N, 3]
        dst: [B, M, 3]
    Returns:
        [B, N, M]
    """
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist


def index_points(points, idx):
    """Gather points along dim 1.

    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]
    Returns:
        [B, S, C] or [B, S, K, C]
    """
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_idx = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch_idx, idx, :]


def canonical_mask(xyz):
    """Mark the first occurrence of every distinct position.

    Building3D scenes with fewer points than the target size are upsampled with
    replacement, which creates exact duplicates. Sampling and grouping prefer
    canonical points so that duplicates do not waste centers or neighbor slots.

    Args:
        xyz: [B, N, 3]
    Returns:
        [B, N] bool, True for the first occurrence of each position.
    """
    B, N = xyz.shape[:2]
    if N == 0:
        return torch.zeros(B, 0, dtype=torch.bool, device=xyz.device)
    h = (xyz * torch.tensor([1.0, 31.0, 997.0], device=xyz.device, dtype=xyz.dtype)).sum(-1)
    sorted_h, order = h.sort(dim=-1, stable=True)
    is_first = torch.ones_like(sorted_h, dtype=torch.bool)
    is_first[:, 1:] = sorted_h[:, 1:] != sorted_h[:, :-1]
    mask = torch.zeros_like(is_first)
    mask.scatter_(1, order, is_first)
    return mask


@torch.no_grad()
def farthest_point_sample(xyz, npoint, canonical=None):
    """Farthest point sampling.

    With a canonical mask, canonical points are exhausted first and duplicates
    are only used when fewer than `npoint` distinct positions exist.

    Args:
        xyz: [B, N, 3]
        npoint: number of centers
        canonical: [B, N] bool or None
    Returns:
        [B, npoint] long
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    batch_idx = torch.arange(B, device=device)

    if canonical is None:
        penalty = None
        farthest = torch.randint(0, N, (B,), device=device)
    else:
        penalty = (~canonical).float() * 1e10
        distance.sub_(penalty)
        rand = torch.rand(B, N, device=device).masked_fill_(~canonical, -1.0)
        farthest = rand.argmax(dim=-1)

    for i in range(npoint):
        centroids[:, i] = farthest
        distance[batch_idx, farthest] = float('-inf')
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        if penalty is not None:
            dist.sub_(penalty)
        torch.minimum(distance, dist, out=distance)
        farthest = distance.argmax(dim=-1)
    return centroids


def interpolate_to_target(xyz_target, xyz_source, feats_source):
    """Inverse distance weighted three nearest neighbor interpolation.

    Args:
        xyz_target: [B, 3, N]
        xyz_source: [B, 3, S]
        feats_source: [B, C, S]
    Returns:
        [B, C, N]
    """
    B, C, S = feats_source.shape
    N = xyz_target.shape[2]
    t = xyz_target.transpose(1, 2)
    s = xyz_source.transpose(1, 2)
    f = feats_source.transpose(1, 2)
    if S == 1:
        return f.expand(B, N, C).transpose(1, 2)
    dists, idx = torch.topk(square_distance(t, s), k=min(3, S), dim=-1, largest=False, sorted=False)
    recip = 1.0 / (dists + 1e-8)
    weight = recip / recip.sum(dim=2, keepdim=True)
    out = (index_points(f, idx) * weight.unsqueeze(-1)).sum(dim=2)
    return out.transpose(1, 2)
