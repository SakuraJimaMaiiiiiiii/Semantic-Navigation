"""基于关键帧点云ICP和4自由度位姿图的回环校正。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LoopKeyframe:
    """保留回环重建所需的相机点云和校正前后位姿。"""

    index: int
    timestamp: float
    raw_camera_pose: np.ndarray
    corrected_camera_pose: np.ndarray
    points_camera: np.ndarray
    raw_drone_position: np.ndarray
    corrected_drone_position: np.ndarray


@dataclass(frozen=True)
class PoseGraphEdge:
    """i到j的相对位姿约束，平移在i坐标系中，角度为相对yaw。"""

    source_index: int
    target_index: int
    measurement: np.ndarray
    weight: float
    is_loop: bool


class PointCloudLoopClosure:
    """为全局稀疏地图创建关键帧、验证回环并校正历史轨迹。"""

    def __init__(self, global_map, config) -> None:
        self.global_map = global_map
        self.config = config
        self.keyframes: list[LoopKeyframe] = []
        self.edges: list[PoseGraphEdge] = []
        self.loop_count = 0
        self.rejected_loop_count = 0
        self._last_loop_keyframe = -10**9
        self._latest_correction = np.eye(4, dtype=np.float64)
        self._validate_config()

    def process(self, observation) -> bool:
        """处理一帧观测；只有满足运动阈值时才创建并融合关键帧。"""
        if not self.config.loop_closure_enabled:
            self.global_map.update(observation)
            return True
        if len(self.keyframes) >= int(self.config.max_keyframes):
            return False

        raw_camera_pose = np.asarray(
            observation.camera_pose_world,
            dtype=np.float64,
        )
        if not self._should_create_keyframe(
            raw_camera_pose,
            float(observation.sensor_frame.timestamp),
        ):
            return False

        frame = observation.sensor_frame
        points_camera = self.global_map.depth_to_camera_points(
            frame.depth,
            frame.camera_intrinsics,
        )
        if points_camera.shape[0] < 50:
            return False

        corrected_camera_pose = (
            self._latest_correction @ raw_camera_pose
        )
        raw_drone_position = np.asarray(
            observation.drone_state.position_xyz,
            dtype=np.float64,
        )
        corrected_drone_position = _transform_point(
            self._latest_correction,
            raw_drone_position,
        )
        keyframe = LoopKeyframe(
            index=len(self.keyframes),
            timestamp=float(frame.timestamp),
            raw_camera_pose=raw_camera_pose.copy(),
            corrected_camera_pose=corrected_camera_pose,
            points_camera=np.asarray(
                points_camera,
                dtype=np.float32,
            ),
            raw_drone_position=raw_drone_position.copy(),
            corrected_drone_position=corrected_drone_position,
        )

        if self.keyframes:
            previous = self.keyframes[-1]
            raw_relative = (
                np.linalg.inv(previous.raw_camera_pose)
                @ keyframe.raw_camera_pose
            )
            self.edges.append(PoseGraphEdge(
                source_index=previous.index,
                target_index=keyframe.index,
                measurement=_pose_vector(raw_relative),
                weight=1.0,
                is_loop=False,
            ))
        self.keyframes.append(keyframe)

        loop_result = self._detect_loop(keyframe)
        if loop_result is None:
            self.global_map.integrate_camera_points(
                points_camera=keyframe.points_camera,
                transform_world_camera=keyframe.corrected_camera_pose,
                timestamp=keyframe.timestamp,
                drone_position_ned=keyframe.corrected_drone_position,
            )
            return True

        candidate, transformation, fitness, rmse = loop_result
        self.edges.append(PoseGraphEdge(
            source_index=candidate.index,
            target_index=keyframe.index,
            measurement=_pose_vector(transformation),
            weight=float(self.config.loop_edge_weight),
            is_loop=True,
        ))
        self._optimize_pose_graph()
        self.global_map.rebuild_from_keyframes(self.keyframes)
        self.loop_count += 1
        self._last_loop_keyframe = keyframe.index
        print(
            "Loop closure accepted: "
            f"keyframe {keyframe.index} -> {candidate.index}, "
            f"fitness={fitness:.2f}, RMSE={rmse:.3f} m"
        )
        return True

    def write_hdf5_metadata(self, path) -> None:
        """把关键帧、校正轨迹和位姿图约束追加到全局地图文件。"""
        import h5py

        with h5py.File(path, mode="a") as h5_file:
            if "loop_closure" in h5_file:
                del h5_file["loop_closure"]
            group = h5_file.create_group("loop_closure")
            group.attrs["enabled"] = bool(
                self.config.loop_closure_enabled
            )
            group.attrs["accepted_loop_count"] = self.loop_count
            group.attrs["rejected_loop_count"] = self.rejected_loop_count
            group.attrs["keyframe_count"] = len(self.keyframes)
            group.create_dataset(
                "timestamps",
                data=np.asarray(
                    [frame.timestamp for frame in self.keyframes],
                    dtype=np.float64,
                ),
            )
            group.create_dataset(
                "raw_camera_poses",
                data=np.asarray(
                    [frame.raw_camera_pose for frame in self.keyframes],
                    dtype=np.float64,
                ),
            )
            group.create_dataset(
                "corrected_camera_poses",
                data=np.asarray(
                    [
                        frame.corrected_camera_pose
                        for frame in self.keyframes
                    ],
                    dtype=np.float64,
                ),
            )
            group.create_dataset(
                "raw_drone_positions",
                data=np.asarray(
                    [
                        frame.raw_drone_position
                        for frame in self.keyframes
                    ],
                    dtype=np.float64,
                ),
            )
            group.create_dataset(
                "corrected_drone_positions",
                data=np.asarray(
                    [
                        frame.corrected_drone_position
                        for frame in self.keyframes
                    ],
                    dtype=np.float64,
                ),
            )
            group.create_dataset(
                "edge_indices",
                data=np.asarray(
                    [
                        [edge.source_index, edge.target_index]
                        for edge in self.edges
                    ],
                    dtype=np.int32,
                ).reshape(-1, 2),
            )
            group.create_dataset(
                "edge_measurements_xyz_yaw",
                data=np.asarray(
                    [edge.measurement for edge in self.edges],
                    dtype=np.float64,
                ).reshape(-1, 4),
            )
            group.create_dataset(
                "edge_weights",
                data=np.asarray(
                    [edge.weight for edge in self.edges],
                    dtype=np.float64,
                ),
            )
            group.create_dataset(
                "edge_is_loop",
                data=np.asarray(
                    [edge.is_loop for edge in self.edges],
                    dtype=np.bool_,
                ),
            )

    def _should_create_keyframe(
        self,
        raw_camera_pose: np.ndarray,
        timestamp: float,
    ) -> bool:
        if not self.keyframes:
            return True
        previous = self.keyframes[-1].raw_camera_pose
        translation = float(np.linalg.norm(
            raw_camera_pose[:3, 3] - previous[:3, 3]
        ))
        relative = np.linalg.inv(previous) @ raw_camera_pose
        yaw_change = abs(_wrap_angle(_yaw_from_rotation(relative[:3, :3])))
        return (
            translation >= float(self.config.keyframe_translation)
            or yaw_change >= math.radians(
                float(self.config.keyframe_yaw_degrees)
            )
            or (
                float(timestamp) - float(self.keyframes[-1].timestamp)
                >= float(self.config.keyframe_max_interval)
            )
        )

    def _detect_loop(self, current: LoopKeyframe):
        separation = int(self.config.loop_min_separation)
        if current.index < separation:
            return None
        if current.index - self._last_loop_keyframe < 5:
            return None

        maximum_index = current.index - separation
        candidates = self.keyframes[:maximum_index + 1]
        current_position = current.corrected_camera_pose[:3, 3]
        ranked = []
        for candidate in candidates:
            distance = float(np.linalg.norm(
                candidate.corrected_camera_pose[:3, 3]
                - current_position
            ))
            if distance <= float(self.config.loop_search_radius):
                ranked.append((distance, candidate))
        ranked.sort(key=lambda item: item[0])

        best = None
        for _, candidate in ranked[
            :int(self.config.loop_max_candidates)
        ]:
            result = self._verify_with_icp(candidate, current)
            if result is None:
                self.rejected_loop_count += 1
                continue
            transformation, fitness, rmse = result
            if best is None or fitness > best[2]:
                best = (
                    candidate,
                    transformation,
                    fitness,
                    rmse,
                )
        return best

    def _verify_with_icp(
        self,
        candidate: LoopKeyframe,
        current: LoopKeyframe,
    ):
        import open3d as o3d

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(
            np.asarray(current.points_camera, dtype=np.float64)
        )
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(
            np.asarray(candidate.points_camera, dtype=np.float64)
        )
        voxel_size = float(self.config.icp_voxel_size)
        source = source.voxel_down_sample(voxel_size)
        target = target.voxel_down_sample(voxel_size)
        if len(source.points) < 50 or len(target.points) < 50:
            return None

        initial = (
            np.linalg.inv(candidate.corrected_camera_pose)
            @ current.corrected_camera_pose
        )
        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            float(self.config.icp_max_correspondence),
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=40
            ),
        )
        fitness = float(registration.fitness)
        rmse = float(registration.inlier_rmse)
        if (
            fitness < float(self.config.icp_min_fitness)
            or rmse > float(self.config.icp_max_rmse)
        ):
            return None

        correction = registration.transformation @ np.linalg.inv(initial)
        translation_correction = float(np.linalg.norm(
            correction[:3, 3]
        ))
        yaw_correction = abs(_wrap_angle(
            _yaw_from_rotation(correction[:3, :3])
        ))
        if (
            translation_correction
            > float(self.config.loop_max_translation_correction)
            or yaw_correction
            > math.radians(
                float(self.config.loop_max_yaw_correction_degrees)
            )
        ):
            return None
        return registration.transformation, fitness, rmse

    def _optimize_pose_graph(self) -> None:
        poses = np.asarray(
            [_pose_vector(frame.corrected_camera_pose)
             for frame in self.keyframes],
            dtype=np.float64,
        )
        if poses.shape[0] < 2:
            return

        for _ in range(int(self.config.pose_graph_iterations)):
            linearized, gradient, diagonal = self._linearize(poses)
            step = _pcg_solve(
                linearized,
                -gradient,
                diagonal,
                node_count=poses.shape[0],
            )
            poses[1:] += step[1:]
            poses[:, 3] = np.asarray(
                [_wrap_angle(yaw) for yaw in poses[:, 3]]
            )
            if float(np.max(np.abs(step[1:]))) < 1e-5:
                break

        for frame, optimized_pose in zip(self.keyframes, poses):
            optimized_matrix = _pose_matrix(optimized_pose)
            raw_planar = _pose_matrix(_pose_vector(frame.raw_camera_pose))
            correction = optimized_matrix @ np.linalg.inv(raw_planar)
            frame.corrected_camera_pose = (
                correction @ frame.raw_camera_pose
            )
            frame.corrected_drone_position = _transform_point(
                correction,
                frame.raw_drone_position,
            )
        latest = self.keyframes[-1]
        raw_planar = _pose_matrix(_pose_vector(latest.raw_camera_pose))
        optimized_planar = _pose_matrix(poses[-1])
        self._latest_correction = (
            optimized_planar @ np.linalg.inv(raw_planar)
        )

    def _linearize(self, poses: np.ndarray):
        node_count = poses.shape[0]
        gradient = np.zeros((node_count, 4), dtype=np.float64)
        diagonal = np.full((node_count, 4), 1e-3, dtype=np.float64)
        linearized = []
        epsilon = np.asarray([1e-4, 1e-4, 1e-4, 1e-5])

        for edge in self.edges:
            i, j = edge.source_index, edge.target_index
            residual = _edge_residual(
                poses[i],
                poses[j],
                edge.measurement,
            )
            jacobian_i = np.empty((4, 4), dtype=np.float64)
            jacobian_j = np.empty((4, 4), dtype=np.float64)
            for column in range(4):
                perturbed = poses[i].copy()
                perturbed[column] += epsilon[column]
                jacobian_i[:, column] = (
                    _edge_residual(
                        perturbed,
                        poses[j],
                        edge.measurement,
                    )
                    - residual
                ) / epsilon[column]

                perturbed = poses[j].copy()
                perturbed[column] += epsilon[column]
                jacobian_j[:, column] = (
                    _edge_residual(
                        poses[i],
                        perturbed,
                        edge.measurement,
                    )
                    - residual
                ) / epsilon[column]

            weight = float(edge.weight)
            linearized.append(
                (i, j, jacobian_i, jacobian_j, weight)
            )
            gradient[i] += weight * jacobian_i.T @ residual
            gradient[j] += weight * jacobian_j.T @ residual
            diagonal[i] += weight * np.sum(jacobian_i**2, axis=0)
            diagonal[j] += weight * np.sum(jacobian_j**2, axis=0)

        gradient[0] = 0.0
        diagonal[0] = 1.0
        return linearized, gradient, diagonal

    def _validate_config(self) -> None:
        config = self.config
        positive_values = {
            "keyframe_translation": config.keyframe_translation,
            "keyframe_yaw_degrees": config.keyframe_yaw_degrees,
            "keyframe_max_interval": config.keyframe_max_interval,
            "loop_search_radius": config.loop_search_radius,
            "icp_voxel_size": config.icp_voxel_size,
            "icp_max_correspondence": config.icp_max_correspondence,
            "icp_max_rmse": config.icp_max_rmse,
            "loop_max_translation_correction": (
                config.loop_max_translation_correction
            ),
            "loop_max_yaw_correction_degrees": (
                config.loop_max_yaw_correction_degrees
            ),
            "loop_edge_weight": config.loop_edge_weight,
        }
        for name, value in positive_values.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        if int(config.max_keyframes) < 2:
            raise ValueError("max_keyframes must be at least two.")
        if int(config.loop_min_separation) < 1:
            raise ValueError("loop_min_separation must be positive.")
        if int(config.loop_max_candidates) < 1:
            raise ValueError("loop_max_candidates must be positive.")
        if int(config.pose_graph_iterations) < 1:
            raise ValueError("pose_graph_iterations must be positive.")
        if not 0.0 <= float(config.icp_min_fitness) <= 1.0:
            raise ValueError("icp_min_fitness must be between zero and one.")


def _pcg_solve(
    linearized,
    right_hand_side: np.ndarray,
    diagonal: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """用预条件共轭梯度求高斯牛顿正规方程，避免依赖SciPy。"""
    solution = np.zeros((node_count, 4), dtype=np.float64)
    residual = right_hand_side.copy()
    residual[0] = 0.0
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    rz_old = float(np.sum(residual[1:] * preconditioned[1:]))
    if rz_old <= 1e-20:
        return solution

    for _ in range(min(120, max(20, 4 * node_count))):
        product = _normal_matrix_product(
            linearized,
            direction,
        )
        denominator = float(np.sum(direction[1:] * product[1:]))
        if abs(denominator) <= 1e-20:
            break
        alpha = rz_old / denominator
        solution += alpha * direction
        residual -= alpha * product
        residual[0] = 0.0
        if float(np.linalg.norm(residual[1:])) < 1e-6:
            break
        preconditioned = residual / diagonal
        rz_new = float(np.sum(residual[1:] * preconditioned[1:]))
        beta = rz_new / max(rz_old, 1e-20)
        direction = preconditioned + beta * direction
        direction[0] = 0.0
        rz_old = rz_new
    return solution


def _normal_matrix_product(
    linearized,
    vector: np.ndarray,
) -> np.ndarray:
    result = 1e-3 * vector
    for i, j, jacobian_i, jacobian_j, weight in linearized:
        projected = jacobian_i @ vector[i] + jacobian_j @ vector[j]
        result[i] += weight * jacobian_i.T @ projected
        result[j] += weight * jacobian_j.T @ projected
    result[0] = 0.0
    return result


def _edge_residual(
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    measurement: np.ndarray,
) -> np.ndarray:
    predicted = (
        np.linalg.inv(_pose_matrix(source_pose))
        @ _pose_matrix(target_pose)
    )
    predicted_vector = _pose_vector(predicted)
    residual = predicted_vector - measurement
    residual[3] = _wrap_angle(residual[3])
    return residual


def _pose_vector(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.asarray(
        [
            transform[0, 3],
            transform[1, 3],
            transform[2, 3],
            _yaw_from_rotation(transform[:3, :3]),
        ],
        dtype=np.float64,
    )


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    x, y, z, yaw = np.asarray(pose, dtype=np.float64)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = [x, y, z]
    return transform


def _yaw_from_rotation(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    return transform[:3, :3] @ point + transform[:3, 3]
