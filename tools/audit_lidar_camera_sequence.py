"""Audit repeated recorded LiDAR/RGB-D pairs without editing a run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from sensors.lidar_camera_alignment import project_lidar


def summary(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else dict(zip(
        ('median', 'p90', 'p95', 'max'), np.percentile(values, (50, 90, 95, 100)).tolist()))


def audit(run, max_pairs=200):
    run = Path(run)
    recordings = list((run / 'recordings').glob('flight_rgbd_*.h5'))
    if len(recordings) != 1:
        raise ValueError('Expected exactly one recorded RGB-D HDF5 file')
    with h5py.File(run / 'fastlio/inputs.h5', 'r') as lidar, h5py.File(recordings[0], 'r') as camera:
        keys = sorted(lidar['scans'])
        scan_times = np.array([lidar['scans'][k].attrs['shared_timestamp'] for k in keys])
        image_times = camera['timestamps/timestamp'][:]
        indices = np.unique(np.linspace(0, len(image_times)-1, min(max_pairs, len(image_times)), dtype=int))
        intrinsics, extrinsics = camera['camera/intrinsics'][:], camera['camera/extrinsics'][:]
        errors, ratios, skews, points_per_frame = [], [], [], []
        unmatched = 0
        for i in indices:
            t = image_times[i]
            j = int(np.searchsorted(scan_times, t))
            choices = [n for n in (j-1, j) if 0 <= n < len(keys)]
            if not choices:
                unmatched += 1
                continue
            j = min(choices, key=lambda n: abs(scan_times[n]-t))
            skew = float(scan_times[j]-t)
            if abs(skew) > .05:
                unmatched += 1
                continue
            scan = lidar['scans'][keys[j]]
            position = scan['px4_pose'][:]
            rot = Rotation.from_quat(position[[4, 5, 6, 3]]).as_matrix()
            world = scan['points_body'][:] @ rot.T + position[:3]
            q = camera['state/quaternion_wxyz'][i]
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            camera_pose[:3, 3] = camera['state/position_xyz'][i]
            camera_pose = camera_pose @ extrinsics
            aligned = project_lidar(world, camera_pose, intrinsics, camera['sensors/depth'][i], skew)
            available = np.isfinite(aligned.reference_depth) & (aligned.reference_depth > 0)
            if not available.any():
                unmatched += 1
                continue
            residual = np.abs(aligned.depth[available]-aligned.reference_depth[available])
            errors.extend(residual.tolist())
            ratios.append(float(aligned.consistent.sum()/available.sum()))
            points_per_frame.append(int(available.sum()))
            skews.append(abs(skew))
        intervals = np.diff(scan_times)
        return {
            'run': str(run.resolve()), 'sampled_rgbd_frames': len(indices),
            'matched_pairs': len(skews), 'unmatched_pairs': unmatched,
            'scan_interval_s': summary(intervals), 'pair_abs_skew_s': summary(skews),
            'projected_points_per_pair': summary(points_per_frame),
            'all_point_depth_residual_m': summary(errors), 'per_pair_consistent_fraction': summary(ratios),
            'scan_model_verified': False,
            'limitations': 'Shared timestamps are receive-aligned, not hardware synchronized; '
                           'camera and LiDAR share PX4 pose, no external ground truth. '
                           'Residuals include occlusions and moving objects; this cannot prove instantaneous scans.',
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--max-pairs', type=int, default=200)
    args = parser.parse_args()
    if args.max_pairs <= 0:
        parser.error('--max-pairs must be positive')
    print(json.dumps(audit(args.run, args.max_pairs), ensure_ascii=False, indent=2, allow_nan=False))
