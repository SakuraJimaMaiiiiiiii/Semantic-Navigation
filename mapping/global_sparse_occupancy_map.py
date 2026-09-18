"""固定世界坐标系下、可持久化的全局稀疏三维占据地图。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from config import GlobalSparseMapConfig


@dataclass(frozen=True)
class GlobalSparseMapSnapshot:
    """供规划器、保存器和可视化线程使用的不可变地图快照。"""

    timestamp: float
    resolution: float
    indices: np.ndarray
    log_odds: np.ndarray
    occupied_threshold: float
    free_threshold: float
    drone_position_ned: np.ndarray
    dropped_new_voxels: int
    trajectory_ned: np.ndarray

    @property
    def occupied_points_ned(self) -> np.ndarray:
        return self._indices_to_points(
            self.indices[self.log_odds >= self.occupied_threshold]
        )

    @property
    def free_points_ned(self) -> np.ndarray:
        return self._indices_to_points(
            self.indices[self.log_odds <= self.free_threshold]
        )

    @property
    def voxel_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(
            self.log_odds >= self.occupied_threshold
        ))

    @property
    def bounds_ned(self) -> tuple[np.ndarray, np.ndarray]:
        if self.indices.shape[0] == 0:
            position = np.asarray(
                self.drone_position_ned,
                dtype=np.float64,
            )
            return position.copy(), position.copy()
        minimum = self.indices.min(axis=0).astype(np.float64) * self.resolution
        maximum = (
            self.indices.max(axis=0).astype(np.float64) + 1.0
        ) * self.resolution
        return minimum, maximum

    def _indices_to_points(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return (indices.astype(np.float64) + 0.5) * self.resolution


class GlobalSparseOccupancyMap:
    """用整数世界体素索引保存整个已观测区域，不进行滑动裁剪。"""

    FORMAT_VERSION = 1

    def __init__(
        self,
        config: GlobalSparseMapConfig | None = None,
    ) -> None:
        self.config = config or GlobalSparseMapConfig()
        self.resolution = float(self.config.resolution)
        self._log_odds: dict[tuple[int, int, int], float] = {}
        self._timestamp = 0.0
        self._drone_position_ned = np.zeros(3, dtype=np.float64)
        self._dropped_new_voxels = 0
        self._snapshot_cache = None
        self._trajectory_ned: list[np.ndarray] = []
        self._lock = threading.RLock()
        self._validate_config()

    def update(self, observation) -> None:
        """把一帧同步深度及对应位姿融合进固定世界地图。"""
        frame = observation.sensor_frame
        depth = np.asarray(frame.depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Depth must be a 2D array, got {depth.shape}.")

        points_camera = self.depth_to_camera_points(
            depth,
            frame.camera_intrinsics,
        )
        if points_camera.shape[0] == 0:
            return
        self.integrate_camera_points(
            points_camera=points_camera,
            transform_world_camera=observation.camera_pose_world,
            timestamp=float(frame.timestamp),
            drone_position_ned=observation.drone_state.position_xyz,
        )

    def integrate_camera_points(
        self,
        points_camera: np.ndarray,
        transform_world_camera: np.ndarray,
        timestamp: float,
        drone_position_ned: np.ndarray,
    ) -> None:
        """使用指定的已校正相机位姿融合一帧相机坐标点云。"""
        occupied_indices, free_indices = self._prepare_indices(
            points_camera,
            transform_world_camera,
        )
        with self._lock:
            self._apply_indices(
                free_indices,
                -float(self.config.miss_log_odds),
            )
            self._apply_indices(
                occupied_indices,
                float(self.config.hit_log_odds),
            )
            self._timestamp = float(timestamp)
            self._drone_position_ned = np.asarray(
                drone_position_ned,
                dtype=np.float64,
            ).copy()
            if (
                not self._trajectory_ned
                or np.linalg.norm(
                    self._drone_position_ned - self._trajectory_ned[-1]
                ) >= 0.5 * self.resolution
            ):
                self._trajectory_ned.append(
                    self._drone_position_ned.copy()
                )
            self._snapshot_cache = None

    def rebuild_from_keyframes(self, keyframes) -> None:
        """回环优化后，用全部关键帧校正位姿重新构建全局地图。"""
        prepared = []
        for keyframe in keyframes:
            occupied, free = self._prepare_indices(
                keyframe.points_camera,
                keyframe.corrected_camera_pose,
            )
            prepared.append((occupied, free))

        with self._lock:
            self._log_odds.clear()
            self._dropped_new_voxels = 0
            self._trajectory_ned = []
            for occupied_indices, free_indices in prepared:
                self._apply_indices(
                    free_indices,
                    -float(self.config.miss_log_odds),
                )
                self._apply_indices(
                    occupied_indices,
                    float(self.config.hit_log_odds),
                )
            if keyframes:
                latest = keyframes[-1]
                self._timestamp = float(latest.timestamp)
                self._drone_position_ned = np.asarray(
                    latest.corrected_drone_position,
                    dtype=np.float64,
                ).copy()
                self._trajectory_ned = [
                    np.asarray(
                        keyframe.corrected_drone_position,
                        dtype=np.float64,
                    ).copy()
                    for keyframe in keyframes
                ]
            self._snapshot_cache = None

    def snapshot(self) -> GlobalSparseMapSnapshot:
        """返回不会被后续地图更新修改的稀疏地图快照。"""
        with self._lock:
            if self._snapshot_cache is None:
                self._snapshot_cache = self._make_snapshot_unlocked()
            return self._snapshot_cache

    def save(self, path=None) -> Path:
        """把稀疏体素索引与log-odds保存为HDF5地图文件。"""
        import h5py

        snapshot = self.snapshot()
        output_path = (
            Path(path)
            if path is not None
            else self._make_output_path()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use HDF5's broadly compatible default format.  ``libver=latest``
        # produced unreadable dataset-layout messages in the Rfly runtime.
        with h5py.File(output_path, mode="w") as h5_file:
            h5_file.attrs["format_version"] = self.FORMAT_VERSION
            h5_file.attrs["created_at_unix"] = time.time()
            h5_file.attrs["world_frame"] = (
                "NED: x=north, y=east, z=down"
            )
            h5_file.attrs["map_type"] = "global_sparse_occupancy"
            h5_file.attrs["resolution"] = snapshot.resolution
            h5_file.attrs["timestamp"] = snapshot.timestamp
            h5_file.attrs["occupied_threshold"] = (
                snapshot.occupied_threshold
            )
            h5_file.attrs["free_threshold"] = snapshot.free_threshold
            h5_file.attrs["voxel_count"] = snapshot.voxel_count
            h5_file.attrs["occupied_count"] = snapshot.occupied_count
            h5_file.attrs["dropped_new_voxels"] = (
                snapshot.dropped_new_voxels
            )
            h5_file.create_dataset(
                "voxel_indices_ned",
                data=snapshot.indices,
                dtype=np.int32,
                compression="lzf",
                shuffle=True,
            )
            h5_file.create_dataset(
                "log_odds",
                data=snapshot.log_odds,
                dtype=np.float32,
                compression="lzf",
                shuffle=True,
            )
            h5_file.create_dataset(
                "drone_position_ned",
                data=snapshot.drone_position_ned,
                dtype=np.float64,
            )
        return output_path

    def export_occupied_ply(self, path) -> Path:
        """导出CloudCompare可读取的二进制PLY占据点云。

        内部地图使用世界NED坐标，导出时转换为N-E-Up：
        ``x=North, y=East, z=-Down``。HDF5中的原始NED数据不变。
        """
        snapshot = self.snapshot()
        occupied_mask = (
            snapshot.log_odds >= snapshot.occupied_threshold
        )
        occupied_indices = snapshot.indices[occupied_mask]
        occupied_log_odds = snapshot.log_odds[occupied_mask]
        points_ned = (
            occupied_indices.astype(np.float64) + 0.5
        ) * snapshot.resolution
        points_neu = points_ned.copy()
        if points_neu.size:
            points_neu[:, 2] *= -1.0

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vertices = np.empty(
            points_neu.shape[0],
            dtype=np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                    ("log_odds", "<f4"),
                ]
            ),
        )
        vertices["x"] = points_neu[:, 0]
        vertices["y"] = points_neu[:, 1]
        vertices["z"] = points_neu[:, 2]
        vertices["red"] = 235
        vertices["green"] = 45
        vertices["blue"] = 45
        vertices["log_odds"] = occupied_log_odds

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment Global sparse occupied voxels exported for CloudCompare\n"
            "comment Coordinates: x=North, y=East, z=Up (NED Down negated)\n"
            f"element vertex {vertices.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float log_odds\n"
            "end_header\n"
        ).encode("ascii")
        with output_path.open("wb") as ply_file:
            ply_file.write(header)
            vertices.tofile(ply_file)
        return output_path

    def export_dense_occupied_ply(
        self,
        path,
        subdivisions: int = 4,
        max_points: int = 1_500_000,
    ) -> Path:
        """Export dense occupied-voxel surfaces for visualization only.

        The expansion operates on a shutdown snapshot and never changes the
        sparse map used by mapping, collision checking, or A* planning.
        """
        subdivisions = max(2, int(subdivisions))
        max_points = max(1, int(max_points))
        snapshot = self.snapshot()
        occupied_mask = snapshot.log_odds >= snapshot.occupied_threshold
        occupied_indices = snapshot.indices[occupied_mask]
        occupied_log_odds = snapshot.log_odds[occupied_mask]

        axis = np.linspace(-0.5, 0.5, subdivisions, dtype=np.float64)
        ox, oy, oz = np.meshgrid(axis, axis, axis, indexing="ij")
        offsets = np.column_stack((ox.ravel(), oy.ravel(), oz.ravel()))
        offsets = offsets[
            np.any(np.isclose(np.abs(offsets), 0.5), axis=1)
        ] * snapshot.resolution

        occupied_count = occupied_indices.shape[0]
        samples_per_voxel = offsets.shape[0]
        if occupied_count:
            samples_per_voxel = min(
                samples_per_voxel,
                max(1, max_points // occupied_count),
            )
            if samples_per_voxel < offsets.shape[0]:
                selection = np.linspace(
                    0,
                    offsets.shape[0] - 1,
                    samples_per_voxel,
                    dtype=np.int64,
                )
                offsets = offsets[selection]

        vertex_count = occupied_count * samples_per_voxel
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("log_odds", "<f4"),
            ]
        )
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment Visualization-only occupied voxel surface samples\n"
            "comment Coordinates: x=North, y=East, z=Up (NED Down negated)\n"
            f"comment Source voxel resolution: {snapshot.resolution}\n"
            f"comment Surface samples per occupied voxel: {samples_per_voxel}\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float log_odds\n"
            "end_header\n"
        ).encode("ascii")

        centers_neu = (
            occupied_indices.astype(np.float64) + 0.5
        ) * snapshot.resolution
        if centers_neu.size:
            centers_neu[:, 2] *= -1.0
        offsets_neu = offsets.copy()
        offsets_neu[:, 2] *= -1.0

        with output_path.open("wb") as ply_file:
            ply_file.write(header)
            chunk_voxels = max(
                1,
                100_000 // max(1, samples_per_voxel),
            )
            for start in range(0, occupied_count, chunk_voxels):
                stop = min(start + chunk_voxels, occupied_count)
                points = (
                    centers_neu[start:stop, None, :]
                    + offsets_neu[None, :, :]
                ).reshape(-1, 3)
                vertices = np.empty(points.shape[0], dtype=vertex_dtype)
                vertices["x"] = points[:, 0]
                vertices["y"] = points[:, 1]
                vertices["z"] = points[:, 2]
                vertices["red"] = 235
                vertices["green"] = 45
                vertices["blue"] = 45
                vertices["log_odds"] = np.repeat(
                    occupied_log_odds[start:stop],
                    samples_per_voxel,
                )
                vertices.tofile(ply_file)
        return output_path

    def depth_to_camera_points(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        stride = int(self.config.depth_stride)
        rows = np.arange(0, depth.shape[0], stride)
        columns = np.arange(0, depth.shape[1], stride)
        uu, vv = np.meshgrid(columns, rows)
        sampled_depth = depth[np.ix_(rows, columns)]
        valid = np.isfinite(sampled_depth)
        valid &= sampled_depth >= float(self.config.min_depth)
        valid &= sampled_depth <= float(self.config.max_depth)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)

        k = np.asarray(intrinsics, dtype=np.float64)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")

        depth_values = sampled_depth[valid].astype(np.float64)
        normalized_x = (uu[valid].astype(np.float64) - cx) / fx
        normalized_y = (vv[valid].astype(np.float64) - cy) / fy
        rays = np.column_stack(
            (normalized_x, normalized_y, np.ones_like(normalized_x))
        )
        if self.config.depth_is_range:
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays * depth_values[:, None]

    def _prepare_indices(
        self,
        points_camera: np.ndarray,
        transform_world_camera: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points_camera = np.asarray(
            points_camera,
            dtype=np.float64,
        ).reshape(-1, 3)
        transform_world_camera = np.asarray(
            transform_world_camera,
            dtype=np.float64,
        )
        camera_origin = transform_world_camera[:3, 3]
        points_world = (
            transform_world_camera[:3, :3] @ points_camera.T
        ).T + camera_origin
        return self._ray_indices(camera_origin, points_world)

    def _ray_indices(
        self,
        camera_origin: np.ndarray,
        endpoints_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        occupied = self._world_to_indices(endpoints_world)
        occupied = np.unique(occupied, axis=0)

        directions = endpoints_world - camera_origin
        distances = np.linalg.norm(directions, axis=1)
        valid_distance = distances > float(self.config.ray_step)
        directions = directions[valid_distance]
        distances = distances[valid_distance]
        if directions.shape[0] == 0:
            return occupied, np.empty((0, 3), dtype=np.int64)
        directions /= distances[:, None]

        free_parts = []
        ray_step = float(self.config.ray_step)
        for distance_along_ray in np.arange(
            ray_step,
            float(np.max(distances)),
            ray_step,
        ):
            active = distances > distance_along_ray + 0.5 * ray_step
            if not np.any(active):
                continue
            samples = (
                camera_origin
                + directions[active] * distance_along_ray
            )
            free_parts.append(self._world_to_indices(samples))

        if not free_parts:
            return occupied, np.empty((0, 3), dtype=np.int64)
        free = np.unique(np.concatenate(free_parts), axis=0)
        occupied_rows = _row_keys(occupied)
        free_rows = _row_keys(free)
        free = free[~np.isin(free_rows, occupied_rows)]
        return occupied, free

    def _world_to_indices(self, points_world: np.ndarray) -> np.ndarray:
        return np.floor(
            np.asarray(points_world, dtype=np.float64) / self.resolution
        ).astype(np.int64)

    def _apply_indices(
        self,
        indices: np.ndarray,
        delta: float,
    ) -> None:
        minimum = float(self.config.min_log_odds)
        maximum = float(self.config.max_log_odds)
        max_voxels = int(self.config.max_voxels)
        for index in np.asarray(indices, dtype=np.int64):
            key = (int(index[0]), int(index[1]), int(index[2]))
            previous = self._log_odds.get(key)
            if previous is None:
                if len(self._log_odds) >= max_voxels:
                    self._dropped_new_voxels += 1
                    continue
                previous = 0.0
            self._log_odds[key] = float(np.clip(
                previous + delta,
                minimum,
                maximum,
            ))

    def _make_snapshot_unlocked(self) -> GlobalSparseMapSnapshot:
        if self._log_odds:
            keys = list(self._log_odds.keys())
            indices = np.asarray(keys, dtype=np.int32)
            values = np.fromiter(
                (self._log_odds[key] for key in keys),
                dtype=np.float32,
                count=len(keys),
            )
        else:
            indices = np.empty((0, 3), dtype=np.int32)
            values = np.empty(0, dtype=np.float32)
        drone_position = self._drone_position_ned.copy()
        trajectory = (
            np.asarray(self._trajectory_ned, dtype=np.float64).reshape(-1, 3)
            if self._trajectory_ned
            else np.empty((0, 3), dtype=np.float64)
        )
        indices.setflags(write=False)
        values.setflags(write=False)
        drone_position.setflags(write=False)
        trajectory.setflags(write=False)
        return GlobalSparseMapSnapshot(
            timestamp=self._timestamp,
            resolution=self.resolution,
            indices=indices,
            log_odds=values,
            occupied_threshold=float(self.config.occupied_threshold),
            free_threshold=float(self.config.free_threshold),
            drone_position_ned=drone_position,
            dropped_new_voxels=self._dropped_new_voxels,
            trajectory_ned=trajectory,
        )

    def _make_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.config.file_prefix}_{timestamp}.h5"
        return Path(self.config.output_dir) / filename

    def _validate_config(self) -> None:
        config = self.config
        if config.resolution <= 0.0:
            raise ValueError("resolution must be greater than zero.")
        if config.update_rate <= 0.0:
            raise ValueError("update_rate must be greater than zero.")
        if config.depth_stride <= 0:
            raise ValueError("depth_stride must be greater than zero.")
        if not 0.0 < config.min_depth < config.max_depth:
            raise ValueError("depth limits must satisfy 0 < min < max.")
        if config.ray_step <= 0.0:
            raise ValueError("ray_step must be greater than zero.")
        if config.max_voxels <= 0:
            raise ValueError("max_voxels must be greater than zero.")


def _row_keys(array: np.ndarray) -> np.ndarray:
    """把N×3整数坐标视为一维键，供向量化集合比较使用。"""
    array = np.ascontiguousarray(array)
    return array.view(
        np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    ).reshape(-1)
