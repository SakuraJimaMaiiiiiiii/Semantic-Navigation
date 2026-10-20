"""Shared LiDAR pose-graph optimization and corrected-frame ray reconstruction.

OnlineLoopNavigationMap re-expresses corrected geometry in PX4 coordinates.
"""
from dataclasses import replace
import numpy as np
from .loop_closure import PointCloudLoopClosure, LoopKeyframe, PoseGraphEdge, _pose_vector
from .global_sparse_occupancy_map import GlobalSparseOccupancyMap


def optimize_frames(frames, constraints, config, check_cancelled=None):
    graph = PointCloudLoopClosure(None, config)
    for i, (origin, cloud, stamp) in enumerate(frames):
        if check_cancelled is not None:
            check_cancelled()
        pose = np.eye(4)
        pose[:3, 3] = origin
        graph.keyframes.append(LoopKeyframe(i, stamp, pose, pose.copy(),
            np.asarray(cloud)-origin, np.asarray(origin).copy(), np.asarray(origin).copy()))
        if i:
            relative = np.linalg.inv(graph.keyframes[i-1].raw_camera_pose) @ pose
            graph.edges.append(PoseGraphEdge(i-1, i, _pose_vector(relative), 1., False))
    for constraint in constraints:
        source, target = int(constraint['source']), int(constraint['target'])
        if not 0 <= target < source < len(frames):
            raise ValueError('Loop constraint indices do not match keyframe archive')
        transform = np.asarray(constraint['source_to_target_world_transform'], float)
        if transform.shape != (4,4) or not np.isfinite(transform).all():
            raise ValueError('Invalid loop transformation')
        relative = (np.linalg.inv(graph.keyframes[target].raw_camera_pose)
                    @ transform @ graph.keyframes[source].raw_camera_pose)
        graph.edges.append(PoseGraphEdge(target, source, _pose_vector(relative),
                                         config.loop_edge_weight, True))
    if constraints:
        graph._optimize_pose_graph(check_cancelled=check_cancelled)
    corrections = [f.corrected_camera_pose @ np.linalg.inv(f.raw_camera_pose) for f in graph.keyframes]
    for frame, correction in zip(graph.keyframes, corrections):
        displacement = np.linalg.norm(frame.corrected_drone_position-frame.raw_drone_position)
        angle = abs(np.arctan2(correction[1,0], correction[0,0]))
        if (not np.isfinite(correction).all() or displacement > config.loop_max_translation_correction
                or angle > np.radians(config.loop_max_yaw_correction_degrees)):
            raise ValueError('Pose graph correction exceeds configured bounds; map not exported')
    return graph.keyframes, np.asarray(corrections)


def rebuild(frames, config, corrected=True, check_cancelled=None):
    world = GlobalSparseOccupancyMap(replace(config, vehicle_free_radius=0., visualize=False))
    for frame in frames:
        if check_cancelled is not None:
            check_cancelled()
        pose = frame.corrected_camera_pose if corrected else frame.raw_camera_pose
        points = frame.points_camera @ pose[:3,:3].T + pose[:3,3]
        world.integrate_lidar_points(pose[:3,3], points, frame.timestamp, pose[:3,3])
    return world.snapshot()
