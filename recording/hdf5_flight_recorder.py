"""将同步无人机状态和 RGB-D 帧持续写入单个 HDF5 文件。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from config import RecordingConfig
from sensors import SynchronizedObservation, SynchronizedRGBDStream


@dataclass(frozen=True)
class RecordedFrame:
    """从 HDF5 文件还原的一帧状态、RGB、深度和标定参数。"""

    state: dict
    rgb: np.ndarray
    depth: np.ndarray
    camera_intrinsics: np.ndarray
    camera_extrinsics: np.ndarray


class FlightDataRecorder:
    """在独立线程中读取同步数据，并追加写入 HDF5。"""

    FORMAT_VERSION = 2

    def __init__(
        self,
        data_stream: SynchronizedRGBDStream,
        config: RecordingConfig | None = None,
    ) -> None:
        self.data_stream = data_stream
        self.config = config or RecordingConfig()
        self.output_path = self._make_output_path()
        self.frame_count = 0
        self.dropped_timeouts = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None

        if self.config.record_rate <= 0.0:
            raise ValueError("record_rate must be greater than zero.")
        if self.config.observation_timeout <= 0.0:
            raise ValueError("observation_timeout must be greater than zero.")
        if self.config.flush_interval <= 0:
            raise ValueError("flush_interval must be greater than zero.")
        compression = str(self.config.hdf5_compression).strip().lower()
        if compression not in ("gzip", "lzf", "none"):
            raise ValueError(
                "hdf5_compression must be 'gzip', 'lzf', or 'none'."
            )
        if not 0 <= self.config.hdf5_compression_level <= 9:
            raise ValueError(
                "hdf5_compression_level must be between 0 and 9."
            )

    def start(self) -> None:
        """启动记录线程，飞行控制可以同时在主线程运行。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._record_loop,
            name="HDF5FlightDataRecorder",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """停止记录并等待最后一批数据刷新到磁盘。"""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(
                timeout=max(3.0, self.config.observation_timeout + 1.0)
            )

    def raise_if_failed(self) -> None:
        """记录线程异常时在主线程重新抛出，避免静默丢失数据。"""
        if self._error is not None:
            raise RuntimeError("HDF5 flight data recorder failed.") from (
                self._error
            )

    def _record_loop(self) -> None:
        h5_file = None
        try:
            h5_file = h5py.File(
                self.output_path,
                mode="w",
                libver="latest",
            )
            self._initialize_file(h5_file)
            period = 1.0 / self.config.record_rate
            next_record = time.monotonic()

            while not self._stop.is_set():
                try:
                    observation = self.data_stream.get_observation(
                        timeout=self.config.observation_timeout
                    )
                except TimeoutError:
                    self.dropped_timeouts += 1
                    continue

                self._append_observation(h5_file, observation)
                self.frame_count += 1
                if self.frame_count % self.config.flush_interval == 0:
                    self._flush(h5_file)

                next_record += period
                wait_time = next_record - time.monotonic()
                if wait_time > 0.0:
                    self._stop.wait(wait_time)
                else:
                    next_record = time.monotonic()

            self._write_runtime_statistics(h5_file)
            self._flush(h5_file)
        except Exception as error:
            self._error = error
        finally:
            if h5_file is not None:
                h5_file.close()

    def _initialize_file(self, h5_file: h5py.File) -> None:
        h5_file.attrs["format_version"] = self.FORMAT_VERSION
        h5_file.attrs["created_at_unix"] = time.time()
        h5_file.attrs["created_at_local"] = datetime.now().isoformat(
            timespec="seconds"
        )
        h5_file.attrs["world_frame"] = "NED: x=north, y=east, z=down"
        h5_file.attrs["body_frame"] = "FRD: x=forward, y=right, z=down"
        h5_file.attrs["quaternion_order"] = (
            "w,x,y,z; body FRD to world NED"
        )
        h5_file.attrs["camera_frame"] = (
            "optical: x=right, y=down, z=forward"
        )
        h5_file.attrs["camera_extrinsics"] = "T_body_camera"
        h5_file.attrs["rgb_encoding"] = "uint8 OpenCV BGR"
        h5_file.attrs["depth_encoding"] = "float32 meters"
        h5_file.attrs["committed_frame_count"] = 0
        h5_file.attrs["dropped_sync_timeouts"] = 0

        timestamps = h5_file.create_group("timestamps")
        state = h5_file.create_group("state")
        h5_file.create_group("sensors")
        h5_file.create_group("camera")

        for name in (
            "timestamp",
            "rgb_timestamp",
            "depth_timestamp",
            "received_at",
            "rgb_depth_skew",
            "state_sync_error",
            "px4_boot_timestamp",
        ):
            self._create_scalar_dataset(timestamps, name)

        self._create_vector_dataset(state, "position_xyz", 3)
        self._create_vector_dataset(state, "quaternion_wxyz", 4)
        self._create_vector_dataset(state, "euler_rpy", 3)
        self._create_vector_dataset(state, "linear_velocity", 3)
        self._create_vector_dataset(state, "angular_velocity", 3)

    def _append_observation(
        self,
        h5_file: h5py.File,
        observation: SynchronizedObservation,
    ) -> None:
        state = observation.drone_state
        frame = observation.sensor_frame
        rgb = np.ascontiguousarray(frame.rgb, dtype=np.uint8)
        depth = np.ascontiguousarray(frame.depth, dtype=np.float32)
        if rgb.ndim != 3:
            raise ValueError(f"RGB frame must be HxWxC, got {rgb.shape}.")
        if depth.ndim != 2:
            raise ValueError(f"Depth frame must be HxW, got {depth.shape}.")

        if "rgb" not in h5_file["sensors"]:
            self._create_sensor_datasets(h5_file, frame, rgb, depth)
        self._validate_sensor_shapes(h5_file, rgb, depth)

        index = self.frame_count
        scalar_values = {
            "timestamp": frame.timestamp,
            "rgb_timestamp": frame.rgb_timestamp,
            "depth_timestamp": frame.depth_timestamp,
            "received_at": frame.received_at,
            "rgb_depth_skew": frame.rgb_depth_skew,
            "state_sync_error": observation.state_sync_error,
            "px4_boot_timestamp": state.px4_boot_timestamp,
        }
        for name, value in scalar_values.items():
            self._append_dataset(
                h5_file[f"timestamps/{name}"],
                index,
                float(value),
            )

        vector_values = {
            "position_xyz": state.position_xyz,
            "quaternion_wxyz": state.quaternion,
            "euler_rpy": state.euler_rpy,
            "linear_velocity": state.linear_velocity,
            "angular_velocity": state.angular_velocity,
        }
        for name, value in vector_values.items():
            self._append_dataset(
                h5_file[f"state/{name}"],
                index,
                np.asarray(value, dtype=np.float64),
            )

        self._append_dataset(h5_file["sensors/rgb"], index, rgb)
        self._append_dataset(h5_file["sensors/depth"], index, depth)

    def _create_sensor_datasets(
        self,
        h5_file: h5py.File,
        frame,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        sensors = h5_file["sensors"]
        compression = self._compression_options()
        sensors.create_dataset(
            "rgb",
            shape=(0, *rgb.shape),
            maxshape=(None, *rgb.shape),
            chunks=(1, *rgb.shape),
            dtype=np.uint8,
            **compression,
        )
        sensors.create_dataset(
            "depth",
            shape=(0, *depth.shape),
            maxshape=(None, *depth.shape),
            chunks=(1, *depth.shape),
            dtype=np.float32,
            **compression,
        )
        camera = h5_file["camera"]
        camera.create_dataset(
            "intrinsics",
            data=np.asarray(frame.camera_intrinsics, dtype=np.float64),
        )
        camera.create_dataset(
            "extrinsics",
            data=np.asarray(frame.camera_extrinsics, dtype=np.float64),
        )

    @staticmethod
    def _validate_sensor_shapes(
        h5_file: h5py.File,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        expected_rgb = h5_file["sensors/rgb"].shape[1:]
        expected_depth = h5_file["sensors/depth"].shape[1:]
        if rgb.shape != expected_rgb:
            raise ValueError(
                f"RGB shape changed from {expected_rgb} to {rgb.shape}."
            )
        if depth.shape != expected_depth:
            raise ValueError(
                f"Depth shape changed from {expected_depth} to {depth.shape}."
            )

    @staticmethod
    def _append_dataset(dataset, index: int, value) -> None:
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    @staticmethod
    def _create_scalar_dataset(group, name: str) -> None:
        group.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=np.float64,
        )

    @staticmethod
    def _create_vector_dataset(group, name: str, width: int) -> None:
        group.create_dataset(
            name,
            shape=(0, width),
            maxshape=(None, width),
            chunks=(256, width),
            dtype=np.float64,
        )

    def _compression_options(self) -> dict:
        compression = str(self.config.hdf5_compression).strip().lower()
        if compression == "none":
            return {}
        options = {
            "compression": compression,
            "shuffle": True,
        }
        if compression == "gzip":
            options["compression_opts"] = int(
                self.config.hdf5_compression_level
            )
        return options

    def _flush(self, h5_file: h5py.File) -> None:
        h5_file.attrs["committed_frame_count"] = self.frame_count
        h5_file.attrs["dropped_sync_timeouts"] = self.dropped_timeouts
        h5_file.flush()

    def _write_runtime_statistics(self, h5_file: h5py.File) -> None:
        h5_file.attrs["closed_at_unix"] = time.time()
        h5_file.attrs["frame_count"] = self.frame_count
        h5_file.attrs["dropped_sync_timeouts"] = self.dropped_timeouts
        h5_file.attrs["dropped_source_frames"] = int(
            getattr(self.data_stream, "dropped_count", 0)
        )

    def _make_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.config.file_prefix}_{timestamp}.h5"
        return Path(self.config.output_dir) / filename


class FlightDataReader:
    """离线读取 HDF5 飞行数据，不参与实时飞行控制。"""

    def __init__(self, path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Flight data file not found: {self.path}")
        self.h5_file = h5py.File(self.path, mode="r")

    @property
    def frame_count(self) -> int:
        """返回最后一次成功刷新到磁盘的帧数。"""
        if (
            "sensors/rgb" not in self.h5_file
            or "sensors/depth" not in self.h5_file
        ):
            return 0
        committed = int(
            self.h5_file.attrs.get(
                "committed_frame_count",
                self.h5_file["timestamps/timestamp"].shape[0],
            )
        )
        available = min(
            self.h5_file["timestamps/timestamp"].shape[0],
            self.h5_file["state/position_xyz"].shape[0],
            self.h5_file["sensors/rgb"].shape[0],
            self.h5_file["sensors/depth"].shape[0],
        )
        return min(committed, available)

    def close(self) -> None:
        self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def read_state_rows(self) -> list[dict]:
        """以表格行形式读取全部状态；每个字典对应一帧。"""
        return [self._state_at(index) for index in range(self.frame_count)]

    def read_frame(self, frame_id: int) -> RecordedFrame:
        """按从0开始的 frame_id 还原一帧完整数据。"""
        index = int(frame_id)
        if not 0 <= index < self.frame_count:
            raise KeyError(f"Frame {frame_id} does not exist.")
        return RecordedFrame(
            state=self._state_at(index),
            rgb=self.h5_file["sensors/rgb"][index],
            depth=self.h5_file["sensors/depth"][index],
            camera_intrinsics=self.h5_file["camera/intrinsics"][:],
            camera_extrinsics=self.h5_file["camera/extrinsics"][:],
        )

    def read_batch(self, start: int, stop: int) -> dict[str, np.ndarray]:
        """批量切片读取训练数据，区间采用 Python 的 [start, stop)。"""
        start = max(0, int(start))
        stop = min(int(stop), self.frame_count)
        if stop < start:
            raise ValueError("stop must not be less than start.")
        selection = slice(start, stop)
        return {
            "timestamp": self.h5_file["timestamps/timestamp"][selection],
            "position_xyz": self.h5_file[
                "state/position_xyz"
            ][selection],
            "quaternion": self.h5_file[
                "state/quaternion_wxyz"
            ][selection],
            "linear_velocity": self.h5_file[
                "state/linear_velocity"
            ][selection],
            "angular_velocity": self.h5_file[
                "state/angular_velocity"
            ][selection],
            "rgb": self.h5_file["sensors/rgb"][selection],
            "depth": self.h5_file["sensors/depth"][selection],
        }

    def _state_at(self, index: int) -> dict:
        timestamps = self.h5_file["timestamps"]
        state = self.h5_file["state"]
        position = state["position_xyz"][index]
        quaternion = state["quaternion_wxyz"][index]
        euler = state["euler_rpy"][index]
        linear_velocity = state["linear_velocity"][index]
        angular_velocity = state["angular_velocity"][index]
        return {
            "frame_id": index,
            "timestamp": float(timestamps["timestamp"][index]),
            "rgb_timestamp": float(timestamps["rgb_timestamp"][index]),
            "depth_timestamp": float(
                timestamps["depth_timestamp"][index]
            ),
            "rgb_depth_skew": float(
                timestamps["rgb_depth_skew"][index]
            ),
            "state_sync_error": float(
                timestamps["state_sync_error"][index]
            ),
            "position_x": float(position[0]),
            "position_y": float(position[1]),
            "position_z": float(position[2]),
            "quaternion_w": float(quaternion[0]),
            "quaternion_x": float(quaternion[1]),
            "quaternion_y": float(quaternion[2]),
            "quaternion_z": float(quaternion[3]),
            "roll": float(euler[0]),
            "pitch": float(euler[1]),
            "yaw": float(euler[2]),
            "linear_velocity_x": float(linear_velocity[0]),
            "linear_velocity_y": float(linear_velocity[1]),
            "linear_velocity_z": float(linear_velocity[2]),
            "angular_velocity_x": float(angular_velocity[0]),
            "angular_velocity_y": float(angular_velocity[1]),
            "angular_velocity_z": float(angular_velocity[2]),
            "px4_boot_timestamp": float(
                timestamps["px4_boot_timestamp"][index]
            ),
        }
