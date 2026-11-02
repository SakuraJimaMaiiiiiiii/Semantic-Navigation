"""Read-only audit of recorded camera/LiDAR transforms and timing; no SDK required."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return dict(count=len(values), **dict(zip(
        ('median', 'p90', 'p95', 'p99', 'max'),
        np.percentile(values, [50, 90, 95, 99, 100]).tolist())))


def audit(run):
    with np.load(run / 'lidar_camera_alignment.npz') as data:
        pose = data['camera_pose_world']
        extrinsics = data['camera_extrinsics']
        world = data['lidar_points_world']
        optical = (world-pose[:3, 3]) @ pose[:3, :3]
        restored = optical @ pose[:3, :3].T + pose[:3, 3]
        depth, reference = data['lidar_optical_depth'], data['camera_depth']
        valid = np.isfinite(reference) & (reference > 0) & np.isfinite(depth)
        error = np.abs(depth-reference)
        report = {
            'run': str(run.resolve()),
            'scope': 'One saved paired cloud; HDF5 timing covers accepted recorded frames. No absolute ground truth.',
            'camera_extrinsics_body_from_optical': extrinsics.tolist(),
            'rotation_orthogonality_error': float(np.max(np.abs(pose[:3,:3].T@pose[:3,:3]-np.eye(3)))),
            'rotation_determinant': float(np.linalg.det(pose[:3,:3])),
            'world_optical_roundtrip_max_error_m': float(np.max(np.abs(world-restored))),
            'paired_lidar_minus_image_seconds': float(data['lidar_timestamp']-data['image_timestamp']),
            'projected_points': len(depth),
            'consistent_points': int(data['consistent'].sum()),
            'all_valid_depth_residual_m': distribution(error[valid]),
            'accepted_depth_residual_m': distribution(error[valid & data['consistent']]),
        }
        image_stamp = float(data['image_timestamp'])
        body_pose = pose @ np.linalg.inv(extrinsics)
    statistics = run / 'lidar_camera_alignment.json'
    if statistics.exists():
        report['runtime_alignment_statistics'] = json.loads(statistics.read_text(encoding='utf-8'))
    recordings = sorted((run / 'recordings').glob('flight_rgbd_*.h5'))
    if len(recordings) == 1:
        with h5py.File(recordings[0]) as recording:
            stamps = recording['timestamps/timestamp'][:]
            if len(stamps):
                nearest = int(np.argmin(np.abs(stamps-image_stamp)))
                report['nearest_recording_time_delta_s'] = float(stamps[nearest]-image_stamp)
                report['body_position_vs_nearest_recording_m'] = float(np.linalg.norm(
                    body_pose[:3,3]-recording['state/position_xyz'][nearest]))
            sync = recording['timestamps/state_sync_error'][:]
            report['recorded_pose_sync_error_s'] = distribution(sync)
            report['recorded_rgb_depth_skew_s'] = distribution(np.abs(recording['timestamps/rgb_depth_skew'][:]))
            speed = np.linalg.norm(recording['state/linear_velocity'][:], axis=1)
            angular_speed = np.linalg.norm(recording['state/angular_velocity'][:], axis=1)
            report['timing_sensitivity_at_10m_m_not_measured_error'] = distribution((speed+10*angular_speed)*sync)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    if not (args.run / 'lidar_camera_alignment.npz').is_file():
        parser.error('Run has no lidar_camera_alignment.npz paired sample')
    print(json.dumps(audit(args.run), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
