"""Approximate geodesic distances on point clouds (Dijkstra on a kNN graph)."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def knn_graph(xyz, k=10):
    """Symmetric kNN graph weighted by Euclidean edge length."""
    n = len(xyz)
    tree = cKDTree(xyz)
    dist, idx = tree.query(xyz, k=min(k + 1, n))
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].reshape(-1)
    w = np.maximum(dist[:, 1:].reshape(-1), 1e-9)  # keep duplicate points connected
    graph = csr_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                       shape=(n, n))
    return graph, tree


def geodesic_to_keypoints(xyz, keypoints, k=10):
    """Geodesic distance from every point to its nearest keypoint.

    Each keypoint enters the graph at its nearest cloud point.

    Returns:
        min_dist [N], nearest keypoint index [N]
    """
    graph, tree = knn_graph(xyz, k)
    _, sources = tree.query(keypoints, k=1)
    d = dijkstra(graph, directed=False, indices=np.atleast_1d(sources))  # [K, N]
    nearest = d.argmin(axis=0)
    return d[nearest, np.arange(len(xyz))].astype(np.float32), nearest


def geodesic_knn(xyz, k=16, graph_k=10, limit=0.3):
    """k nearest neighbors of every point under geodesic distance.

    Used as the graph of the directional GNN on KeypointNet. Points with fewer
    than k geodesic neighbors within `limit` are completed with their nearest
    Euclidean points, whose edge length is then the Euclidean distance.

    Returns:
        idx [N, k] int64, dist [N, k] float32
    """
    graph, _ = knn_graph(xyz, graph_k)
    geo = dijkstra(graph, directed=False, limit=limit)                    # [N, N], inf beyond limit
    euclid = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    reachable = np.isfinite(geo)
    dist = np.where(reachable, geo, euclid)
    rank = np.where(reachable, geo, 1e3 + euclid)
    np.fill_diagonal(rank, np.inf)
    idx = np.argpartition(rank, k, axis=1)[:, :k]
    order = np.take_along_axis(rank, idx, axis=1).argsort(axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    return idx.astype(np.int64), np.take_along_axis(dist, idx, axis=1).astype(np.float32)
