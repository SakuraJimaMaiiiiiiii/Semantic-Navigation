"""Inspect a saved failure NPZ; geometry plausibility is not a calibration truth test."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data['metadata'].tobytes())
        depth = data['depth']
        k = data['intrinsics']
        pose = data['camera_pose']
        valid = np.isfinite(depth) & (depth > 0)
        rotation = pose[:3,:3]
        return {
            'source': str(path),
            'depth_shape': list(depth.shape),
            'valid_depth_fraction': float(valid.mean()),
            'depth_m_percentiles_1_50_99': np.percentile(depth[valid],[1,50,99]).tolist() if valid.any() else None,
            'intrinsics': k.tolist(),
            'intrinsics_finite_positive_focal_lengths': bool(np.isfinite(k).all() and k[0,0]>0 and k[1,1]>0),
            'camera_pose_finite': bool(np.isfinite(pose).all()),
            'rotation_orthogonality_error': float(np.linalg.norm(rotation.T@rotation-np.eye(3))),
            'rotation_determinant': float(np.linalg.det(rotation)),
            'rgb_depth_skew_seconds': abs(metadata['rgb_timestamp']-metadata['depth_timestamp']),
            'state_sync_error_seconds': metadata['state_sync_error'],
            'limitation': 'Cannot establish metric accuracy or axial/range convention without known scene geometry. No automatic calibration changes.',
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.snapshot), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(result, encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
