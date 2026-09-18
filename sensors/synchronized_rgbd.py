"""RGB-D 图像与无人机状态的时间同步封装。"""

from __future__ import annotations

import bisect
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from vehicle import DroneState, RflyMultirotorInterface

from .rfly_rgbd_camera import RflyRGBDCamera
from .pose_uncertainty import camera_pose_covariance
from .empirical_pose_covariance import load_profile, combine_pose_covariance


@dataclass(frozen=True)
class SensorFrame:
    """一帧带标定参数的 RGB-D 数据。

    ``camera_extrinsics`` 是 ``T_body_camera``：把相机光学坐标
    （x右、y下、z前）转换到无人机机体 FRD 坐标。
    """

    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    camera_intrinsics: np.ndarray
    camera_extrinsics: np.ndarray
    rgb_timestamp: float
    depth_timestamp: float
    received_at: float

    @property
    def rgb_depth_skew(self) -> float:
        """RGB 与深度的时间差，单位 s。"""
        return abs(self.rgb_timestamp - self.depth_timestamp)


@dataclass(frozen=True)
class SynchronizedObservation:
    """按同一个图像时刻对齐的飞行状态和传感器数据。"""

    drone_state: DroneState
    sensor_frame: SensorFrame
    state_sync_error: float
    camera_pose_covariance: np.ndarray | None = None

    @property
    def camera_pose_world(self) -> np.ndarray:
        """返回 ``T_world_camera``，世界坐标系为 NED。"""
        transform_world_body = np.eye(4, dtype=np.float64)
        transform_world_body[:3, :3] = _quaternion_to_rotation(
            self.drone_state.quaternion
        )
        transform_world_body[:3, 3] = self.drone_state.position_xyz
        return transform_world_body @ self.sensor_frame.camera_extrinsics


class SynchronizedRGBDStream:
    """缓存飞行状态，并在图像时间上插值得到同步观测。

    RflySim 图像提供 Unix 秒时间戳；PX4MavCtrlV4 只把最新状态保存在
    数组中，因此本类用主机 Unix 时钟对这些数组做高频快照。返回数据前
    会检查 RGB/深度时间差以及图像到最近原始位姿样本的时间差。
    """

    def __init__(
        self,
        vehicle: RflyMultirotorInterface,
        camera: RflyRGBDCamera,
    ) -> None:
        self.vehicle = vehicle
        self.camera = camera
        config = camera.config
        self.sample_rate = float(config.state_sample_rate)
        self.history_seconds = float(config.state_history_seconds)
        self.max_sync_error = float(config.max_state_sync_error)
        position_std = np.asarray(config.pose_position_std_ned, dtype=float)
        rotation_std = np.deg2rad(np.asarray(config.pose_rotation_std_body_deg, dtype=float))
        for values in (position_std, rotation_std):
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("pose standard deviations must be three finite non-negative values")
        self._body_pose_covariance = np.diag(np.concatenate((position_std, rotation_std))**2)
        profile = getattr(config, "pose_covariance_profile", None)
        self._calibrated = profile is not None
        if self._calibrated:
            self._body_pose_covariance = load_profile(profile)
        self._pose_timestamp_std = float(config.pose_timestamp_std)
        if not np.isfinite(self._pose_timestamp_std) or self._pose_timestamp_std < 0:
            raise ValueError("pose_timestamp_std must be finite and non-negative")
        if self.sample_rate <= 0.0:
            raise ValueError("state_sample_rate must be greater than zero.")
        if self.history_seconds <= 0.0:
            raise ValueError("state_history_seconds must be greater than zero.")
        if self.max_sync_error <= 0.0:
            raise ValueError("max_state_sync_error must be greater than zero.")

        capacity = max(
            3,
            int(math.ceil(self.sample_rate * self.history_seconds)) + 2,
        )
        self._states: deque[DroneState] = deque(maxlen=capacity)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._last_frame_timestamp = 0.0
        self._sampler_error = None

    def start(self) -> None:
        """启动无人机状态历史采样线程。"""
        if not self.vehicle.connected:
            raise RuntimeError(
                "Connect the multirotor before starting synchronized data."
            )
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        mav = getattr(self.vehicle, "mav", None)
        if mav is not None:
            # 仅请求遥测，不修改飞控参数/固件。与 SDK 共用连接，不另开接收端口。
            mav.SendMavCmdLong(511, 331, 100000)  # ODOMETRY, 10 Hz
        self._sampler_error = None
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"DroneStateSampler-{self.camera.config.copter_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """停止状态采样；重复调用也是安全的。"""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        mav = getattr(self.vehicle, "mav", None)
        if thread is not None and mav is not None:
            mav.SendMavCmdLong(511, 331, 0)  # 恢复默认遥测速率

    def get_observation(self, timeout: float = 1.0) -> SynchronizedObservation:
        """返回一组通过时间戳对齐的 ``DroneState`` 和 ``SensorFrame``。

        位姿、线速度和角速度使用线性插值；四元数使用最短路径归一化
        插值。若最近的原始状态样本离图像时刻仍超过配置阈值，则拒绝该
        帧并等待下一帧，而不是返回可能造成三维标定误差的数据。
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Synchronized RGB-D stream is not started.")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout must be a positive finite number.")
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            self._raise_sampler_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                frame = self.camera.get_frame(
                    timeout=remaining,
                    after_timestamp=self._last_frame_timestamp,
                )
            except TimeoutError as error:
                last_error = error
                break
            self._last_frame_timestamp = frame.timestamp

            try:
                state, sync_error = self._state_at(
                    frame.timestamp,
                    deadline,
                )
            except TimeoutError as error:
                last_error = error
                # _state_at 仅在共享 deadline 已耗尽时超时。此时继续循环
                # 会拿约 1 ms 再调用 camera.get_frame，最终把真实的状态
                # 对齐失败错误伪装成 "RGB-D frame within 0.0 s"。
                break

            if sync_error > self.max_sync_error:
                last_error = TimeoutError(
                    "State/image synchronization error is "
                    f"{sync_error:.4f} s, exceeding "
                    f"{self.max_sync_error:.4f} s."
                )
                continue

            sensor_frame = SensorFrame(
                timestamp=frame.timestamp,
                rgb=frame.rgb_bgr,
                depth=frame.depth_m,
                camera_intrinsics=self.camera.camera_intrinsics,
                camera_extrinsics=self.camera.camera_extrinsics,
                rgb_timestamp=frame.rgb_timestamp,
                depth_timestamp=frame.depth_timestamp,
                received_at=frame.received_at,
            )
            return SynchronizedObservation(
                drone_state=state,
                sensor_frame=sensor_frame,
                state_sync_error=sync_error,
                camera_pose_covariance=camera_pose_covariance(
                    _quaternion_to_rotation(state.quaternion),
                    sensor_frame.camera_extrinsics,
                    combine_pose_covariance(
                        self._body_pose_covariance,
                        getattr(state, "position_covariance_ned", None),
                        self._calibrated,
                    ),
                    state.linear_velocity, state.angular_velocity,
                    self._pose_timestamp_std,
                ),
            )

        if last_error is not None:
            raise TimeoutError(
                f"No synchronized RGB-D observation within {timeout:.1f} s: "
                f"{last_error}"
            ) from last_error
        raise TimeoutError(
            f"No synchronized RGB-D observation within {timeout:.1f} s."
        )

    def _sample_loop(self) -> None:
        period = 1.0 / self.sample_rate
        next_sample = time.monotonic()
        last_px4_boot_timestamp = None
        try:
            while not self._stop.is_set():
                state = self.vehicle.get_drone_state()
                boot_timestamp = state.px4_boot_timestamp
                # MAVLink ATTITUDE 的 time_boot_ms 变化时才保存新状态，避免
                # 100 Hz轮询把同一条旧飞控数据伪装成多个“新”时间样本。
                is_new_px4_state = (
                    boot_timestamp <= 0.0
                    or last_px4_boot_timestamp is None
                    or boot_timestamp != last_px4_boot_timestamp
                )
                if is_new_px4_state:
                    last_px4_boot_timestamp = boot_timestamp
                    with self._condition:
                        self._states.append(state)
                        self._condition.notify_all()

                next_sample += period
                wait_time = next_sample - time.monotonic()
                if wait_time > 0.0:
                    self._stop.wait(wait_time)
                else:
                    next_sample = time.monotonic()
        except Exception as error:
            self._sampler_error = error
            with self._condition:
                self._condition.notify_all()

    def _state_at(
        self,
        timestamp: float,
        deadline: float,
    ) -> tuple[DroneState, float]:
        with self._condition:
            while True:
                self._raise_sampler_error()
                states = list(self._states)
                if states and states[-1].timestamp >= timestamp:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "No state sample arrived after the image timestamp."
                    )
                self._condition.wait(timeout=remaining)

        stamps = [state.timestamp for state in states]
        right_index = bisect.bisect_left(stamps, timestamp)
        if right_index == 0:
            state = states[0]
            return state, abs(state.timestamp - timestamp)
        if right_index >= len(states):
            state = states[-1]
            return state, abs(state.timestamp - timestamp)

        before = states[right_index - 1]
        after = states[right_index]
        interval = after.timestamp - before.timestamp
        if interval <= 1e-9:
            return before, abs(before.timestamp - timestamp)
        ratio = (timestamp - before.timestamp) / interval
        nearest_error = min(
            timestamp - before.timestamp,
            after.timestamp - timestamp,
        )
        return _interpolate_state(before, after, ratio, timestamp), nearest_error

    def _raise_sampler_error(self) -> None:
        if self._sampler_error is not None:
            raise RuntimeError(
                "Drone state sampler stopped unexpectedly."
            ) from self._sampler_error


def _interpolate_state(
    before: DroneState,
    after: DroneState,
    ratio: float,
    timestamp: float,
) -> DroneState:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    quaternion = _normalized_quaternion_lerp(
        before.quaternion,
        after.quaternion,
        ratio,
    )
    return DroneState(
        timestamp=float(timestamp),
        position_xyz=_lerp(before.position_xyz, after.position_xyz, ratio),
        quaternion=quaternion,
        linear_velocity=_lerp(
            before.linear_velocity,
            after.linear_velocity,
            ratio,
        ),
        angular_velocity=_lerp(
            before.angular_velocity,
            after.angular_velocity,
            ratio,
        ),
        euler_rpy=_quaternion_to_euler(quaternion),
        px4_boot_timestamp=float(
            before.px4_boot_timestamp
            + ratio
            * (after.px4_boot_timestamp - before.px4_boot_timestamp)
        ),
        # 两端都有有效值才插值；缺失/过期不会伪装成有效方差。
        position_covariance_ned=(
            _lerp(before.position_covariance_ned, after.position_covariance_ned, ratio)
            if before.position_covariance_ned is not None and after.position_covariance_ned is not None
            else None
        ),
    )


def _lerp(before, after, ratio: float) -> np.ndarray:
    before_array = np.asarray(before, dtype=float)
    after_array = np.asarray(after, dtype=float)
    return before_array + ratio * (after_array - before_array)


def _normalized_quaternion_lerp(before, after, ratio: float) -> np.ndarray:
    q0 = np.asarray(before, dtype=float)
    q1 = np.asarray(after, dtype=float)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    result = q0 + ratio * (q1 - q0)
    return result / np.linalg.norm(result)


def _quaternion_to_rotation(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-9:
        raise ValueError("Quaternion norm must be greater than zero.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z),
             2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),
             1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),
             2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_to_euler(quaternion) -> np.ndarray:
    rotation = _quaternion_to_rotation(quaternion)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return np.asarray([roll, pitch, yaw], dtype=float)
