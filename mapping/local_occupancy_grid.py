"""由同步深度和位姿构建世界NED下的滑动局部三维占据栅格。"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from config import OccupancyGridConfig
from .vehicle_footprint import vehicle_footprint_indices
from sensors import SynchronizedObservation

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)


@dataclass(frozen=True)
class OccupancyGridSnapshot:
    """可安全交给规划器或可视化线程的一份地图快照。"""

    timestamp: float
    origin_ned: np.ndarray
    resolution: float
    states: np.ndarray
    inflated_occupied: np.ndarray
    drone_position_ned: np.ndarray
    forward_clearance: float
    observed: np.ndarray | None = None


class LocalOccupancyGrid:
    """固定尺寸、随无人机平移的世界对齐NED三维占据栅格。"""

    def __init__(self, config: OccupancyGridConfig | None = None) -> None:
        self.config = config or OccupancyGridConfig()
        self.resolution = float(self.config.resolution)
        self.size_ned = np.asarray(self.config.size_ned, dtype=np.float64)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be greater than zero.")
        if self.size_ned.shape != (3,) or np.any(self.size_ned <= 0.0):
            raise ValueError("size_ned must contain three positive values.")
        if self.config.depth_stride <= 0:
            raise ValueError("depth_stride must be greater than zero.")
        if not 0.0 < self.config.min_depth < self.config.max_depth:
            raise ValueError("depth limits must satisfy 0 < min < max.")
        if self.config.inflation_radius < 0.0:
            raise ValueError("inflation_radius must not be negative.")
        for name in (
            "forward_safety_roi_width",
            "forward_safety_roi_height",
        ):
            value = float(getattr(self.config, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        if not 0.0 <= float(self.config.forward_safety_percentile) <= 100.0:
            raise ValueError("forward_safety_percentile must be in [0, 100].")

        self.shape = tuple(
            np.ceil(self.size_ned / self.resolution).astype(int).tolist()
        )
        self._log_odds = np.zeros(self.shape, dtype=np.float32)
        self._observed = np.zeros(self.shape, dtype=bool)
        self._origin_ned = None
        self._timestamp = 0.0
        self._last_observation_completed_at = 0.0
        self._drone_position_ned = np.zeros(3, dtype=np.float64)
        self._forward_clearance = math.inf
        self._inflation_offsets = self._make_inflation_offsets()
        self._lock = threading.RLock()

    def update(self, observation: SynchronizedObservation) -> None:
        """用一帧同步深度数据更新空闲射线与占据端点。"""
        frame = observation.sensor_frame
        depth = np.asarray(frame.depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Depth must be a 2D array, got {depth.shape}.")

        drone_position = np.asarray(
            observation.drone_state.position_xyz,
            dtype=np.float64,
        )
        transform_world_camera = np.asarray(
            observation.camera_pose_world,
            dtype=np.float64,
        )
        camera_origin = transform_world_camera[:3, 3]
        points_camera = self._depth_to_camera_points(
            depth,
            frame.camera_intrinsics,
        )
        forward_clearance = self._estimate_forward_clearance(depth)
        if points_camera.size == 0:
            with self._lock:
                self._recenter(drone_position)
                self._mark_vehicle_footprint(drone_position)
                self._timestamp = float(frame.timestamp)
                self._drone_position_ned = drone_position.copy()
                self._forward_clearance = forward_clearance
                self._last_observation_completed_at = time.monotonic()
            return
        points_world = (
            transform_world_camera[:3, :3] @ points_camera.T
        ).T + camera_origin

        with self._lock:
            self._recenter(drone_position)
            self._integrate_rays(camera_origin, points_world)
            self._mark_vehicle_footprint(drone_position)
            self._timestamp = float(frame.timestamp)
            self._drone_position_ned = drone_position.copy()
            self._forward_clearance = forward_clearance
            self._last_observation_completed_at = time.monotonic()

    def _mark_vehicle_footprint(self, position):
        indices = vehicle_footprint_indices(
            position,
            self._origin_ned,
            self.resolution,
            self.config.vehicle_free_radius,
            self.config.vehicle_free_half_height,
        )
        valid = np.all((indices >= 0) & (indices < np.asarray(self.shape)), axis=1)
        for index in indices[valid]:
            key = tuple(index)
            if not self._observed[key]:
                self._observed[key] = True
                self._log_odds[key] = float(self.config.free_threshold)

    @property
    def last_update_timestamp(self):
        with self._lock:
            return self._timestamp

    @property
    def last_observation_completed_at(self):
        """Monotonic heartbeat updated after every processed depth frame."""
        with self._lock:
            return self._last_observation_completed_at

    def snapshot(self) -> OccupancyGridSnapshot:
        """复制当前地图，供规划器等读取方无锁使用。"""
        with self._lock:
            states = self._states_unlocked()
            inflated = self._inflate_unlocked(states == OCCUPIED)
            origin = (
                np.zeros(3, dtype=np.float64)
                if self._origin_ned is None
                else self._origin_ned.copy()
            )
            return OccupancyGridSnapshot(
                timestamp=self._timestamp,
                origin_ned=origin,
                resolution=self.resolution,
                states=states,
                inflated_occupied=inflated,
                drone_position_ned=self._drone_position_ned.copy(),
                forward_clearance=float(self._forward_clearance),
                observed=self._observed.copy(),
            )

    def _estimate_forward_clearance(self, depth: np.ndarray) -> float:
        """返回相机中央视场的近距离深度分位数，作为独立急停依据。"""
        height, width = depth.shape
        roi_width = max(
            1,
            int(round(width * float(self.config.forward_safety_roi_width))),
        )
        roi_height = max(
            1,
            int(round(height * float(self.config.forward_safety_roi_height))),
        )
        left = (width - roi_width) // 2
        top = (height - roi_height) // 2
        roi = depth[top : top + roi_height, left : left + roi_width]
        valid = np.isfinite(roi)
        valid &= roi >= float(self.config.min_depth)
        valid &= roi <= float(self.config.max_depth)
        if not np.any(valid):
            return math.inf
        return float(
            np.percentile(
                roi[valid],
                float(self.config.forward_safety_percentile),
            )
        )

    def _depth_to_camera_points(
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

        z_or_range = sampled_depth[valid].astype(np.float64)
        normalized_x = (uu[valid].astype(np.float64) - cx) / fx
        normalized_y = (vv[valid].astype(np.float64) - cy) / fy
        rays = np.column_stack((normalized_x, normalized_y, np.ones_like(normalized_x)))
        if self.config.depth_is_range:
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            return rays * z_or_range[:, None]
        return rays * z_or_range[:, None]

    def _recenter(self, drone_position: np.ndarray) -> None:
        desired_origin = (
            np.floor((drone_position - 0.5 * self.size_ned) / self.resolution)
            * self.resolution
        )
        if self._origin_ned is None:
            self._origin_ned = desired_origin
            return

        shift = np.rint((desired_origin - self._origin_ned) / self.resolution).astype(
            int
        )
        if not np.any(shift):
            return

        new_log_odds = np.zeros_like(self._log_odds)
        new_observed = np.zeros_like(self._observed)
        source_slices = []
        destination_slices = []
        for axis, offset in enumerate(shift):
            axis_size = self.shape[axis]
            if abs(offset) >= axis_size:
                self._log_odds = new_log_odds
                self._observed = new_observed
                self._origin_ned = desired_origin
                return
            if offset >= 0:
                source_slices.append(slice(offset, axis_size))
                destination_slices.append(slice(0, axis_size - offset))
            else:
                source_slices.append(slice(0, axis_size + offset))
                destination_slices.append(slice(-offset, axis_size))

        new_log_odds[tuple(destination_slices)] = self._log_odds[tuple(source_slices)]
        new_observed[tuple(destination_slices)] = self._observed[tuple(source_slices)]
        self._log_odds = new_log_odds
        self._observed = new_observed
        self._origin_ned = desired_origin

    def _integrate_rays(
        self,
        camera_origin: np.ndarray,
        endpoints_world: np.ndarray,
    ) -> None:
        endpoints_indices, valid_endpoints = self._world_to_indices(endpoints_world)
        valid_endpoint_indices = endpoints_indices[valid_endpoints]
        occupied_flat = (
            np.unique(
                np.ravel_multi_index(
                    valid_endpoint_indices.T,
                    self.shape,
                )
            )
            if valid_endpoint_indices.size
            else np.empty(0, dtype=np.int64)
        )

        directions = endpoints_world - camera_origin
        distances = np.linalg.norm(directions, axis=1)
        valid_distance = distances > self.resolution
        directions = directions[valid_distance]
        distances = distances[valid_distance]
        if directions.size:
            directions /= distances[:, None]

        free_flat_parts = []
        max_distance = float(np.max(distances)) if distances.size else 0.0
        ray_step = self.resolution
        for distance_along_ray in np.arange(
            ray_step,
            max_distance,
            ray_step,
        ):
            active = distances > distance_along_ray + 0.5 * ray_step
            if not np.any(active):
                continue
            samples = camera_origin + directions[active] * distance_along_ray
            sample_indices, valid_samples = self._world_to_indices(samples)
            if np.any(valid_samples):
                free_flat_parts.append(
                    np.ravel_multi_index(
                        sample_indices[valid_samples].T,
                        self.shape,
                    )
                )

        if free_flat_parts:
            free_flat = np.unique(np.concatenate(free_flat_parts))
            free_flat = np.setdiff1d(
                free_flat,
                occupied_flat,
                assume_unique=True,
            )
            self._log_odds.flat[free_flat] -= float(self.config.miss_log_odds)
            self._observed.flat[free_flat] = True

        self._log_odds.flat[occupied_flat] += float(self.config.hit_log_odds)
        self._observed.flat[occupied_flat] = True
        np.clip(
            self._log_odds,
            float(self.config.min_log_odds),
            float(self.config.max_log_odds),
            out=self._log_odds,
        )

    def _world_to_indices(
        self,
        points_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.floor(
            (np.asarray(points_world) - self._origin_ned) / self.resolution
        ).astype(int)
        valid = np.all(indices >= 0, axis=1)
        valid &= np.all(indices < np.asarray(self.shape), axis=1)
        return indices, valid

    def _states_unlocked(self) -> np.ndarray:
        states = np.full(self.shape, UNKNOWN, dtype=np.int8)
        states[
            self._observed & (self._log_odds <= float(self.config.free_threshold))
        ] = FREE
        states[
            self._observed & (self._log_odds >= float(self.config.occupied_threshold))
        ] = OCCUPIED
        return states

    def _make_inflation_offsets(self) -> np.ndarray:
        radius_voxels = int(math.ceil(self.config.inflation_radius / self.resolution))
        if radius_voxels <= 0:
            return np.zeros((1, 3), dtype=int)
        coordinates = np.arange(-radius_voxels, radius_voxels + 1)
        xx, yy, zz = np.meshgrid(
            coordinates,
            coordinates,
            coordinates,
            indexing="ij",
        )
        offsets = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        distances = np.linalg.norm(
            offsets.astype(float) * self.resolution,
            axis=1,
        )
        return offsets[distances <= float(self.config.inflation_radius) + 1e-9]

    def _inflate_unlocked(self, occupied: np.ndarray) -> np.ndarray:
        occupied_indices = np.argwhere(occupied)
        inflated = np.zeros(self.shape, dtype=bool)
        if occupied_indices.size == 0:
            return inflated
        expanded = (
            occupied_indices[:, None, :] + self._inflation_offsets[None, :, :]
        ).reshape(-1, 3)
        valid = np.all(expanded >= 0, axis=1)
        valid &= np.all(expanded < np.asarray(self.shape), axis=1)
        expanded = expanded[valid]
        inflated[tuple(expanded.T)] = True
        return inflated
