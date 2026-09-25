"""Package Building3D test predictions for the online evaluator.

The evaluator expects one wireframe .obj per test scene. We only predict
corners, so vertices are written in world coordinates and joined by a chain of
placeholder edges (the evaluator requires `l` lines); only the corner metrics
(ACO, CP, CR, CF1) are meaningful.

    python test.py --config configs/building3d_entry.yaml --split test ... --output_dir output/entry_test
    python make_submission.py --pred_dir output/entry_test/building3d_test_learned --zip submission.zip
"""

import argparse
import glob
import os
import zipfile

import numpy as np


def chain_edges(n, frac=0.5):
    if n < 2:
        return np.zeros((0, 2), dtype=int)
    chain = np.stack([np.arange(n), np.roll(np.arange(n), -1)], axis=1)
    k = max(1, int(round(n * frac)))
    return chain[np.unique(np.linspace(0, n - 1, k).round().astype(int))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', required=True)
    p.add_argument('--zip', default='submission.zip')
    p.add_argument('--edge_frac', type=float, default=0.5)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f'No predictions in {args.pred_dir}')
    with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            v = np.load(f)['keypoints_world']
            if len(v) == 0:
                v = np.zeros((1, 3))
            lines = [f'v {x} {y} {z}' for x, y, z in v]
            lines += [f'l {a + 1} {b + 1}' for a, b in chain_edges(len(v), args.edge_frac)]
            zf.writestr(os.path.splitext(os.path.basename(f))[0] + '.obj', '\n'.join(lines) + '\n')
    print(f'Wrote {len(files)} scenes to {args.zip}')


if __name__ == '__main__':
    main()
