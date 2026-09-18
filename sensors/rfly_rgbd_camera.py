"""可复用的 RflySim3D RGB-D 图像采集接口。"""

from __future__ import annotations

import importlib
import io
import json
import math
import sys
import threading
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.camera_config import CameraConfig


@dataclass(frozen=True)
class RGBDFrame:
    """一帧 RGB-D 观测，以及可选的 RflySim 真值分割图。"""

    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    rgb_timestamp: float
    depth_timestamp: float
    received_at: float
    segmentation_bgr: np.ndarray | None = None
    segmentation_timestamp: float = 0.0

    @property
    def timestamp(self) -> float:
        """返回有效 RGB 与深度时间戳的中间时刻。"""
        stamps = [s for s in (self.rgb_timestamp, self.depth_timestamp) if s > 0]
        return float(sum(stamps) / len(stamps)) if stamps else self.received_at

    @property
    def sync_skew(self) -> float:
        """返回 RGB 与深度时间戳的绝对差值，单位为秒。"""
        if self.rgb_timestamp <= 0 or self.depth_timestamp <= 0:
            return 0.0
        return abs(self.rgb_timestamp - self.depth_timestamp)


class RflyRGBDCamera:
    """启动 RflySim RGB-D 采集，并提供已复制、归一化的图像帧。"""

    PREVIEW_WINDOW_NAME = "RflySim Camera (RGB / Depth / TypeID=4)"
    PREVIEW_PANEL_WIDTH = 480

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture = None
        self._started = False
        self._preview_stop = threading.Event()
        self._preview_thread = None
        self._cv2 = None
        self._rgb_sensor = None
        self._timestamp_offset = None
        self._latest_frame_lock = threading.Lock()
        self._latest_frame = None
        self._detection_preview_provider = None
        self._detection_window_name = None
        self._validate_files()
        self._load_and_validate_sensor_json()

    def set_detection_preview_provider(self, provider, window_name: str) -> None:
        """设置由相机预览线程显示的独立实时检测画面。"""
        self._detection_preview_provider = provider
        self._detection_window_name = str(window_name)

    def start(self) -> RGBDFrame:
        """向 UE 请求 RGB/深度，并按显示开关请求 TypeID=4。"""
        if self._started:
            return self.get_frame(timeout=self.config.first_frame_timeout)

        output_context = (
            redirect_stdout(io.StringIO())
            if self.config.suppress_sdk_output
            else nullcontext()
        )
        with output_context:
            vision_module = self._import_vision_capture_api()
            self._capture = vision_module.VisionCaptureApi()
            if not self._capture.jsonLoad(str(self.config.sensor_json)):
                self._capture = None
                raise RuntimeError(
                    "Failed to load the RflySim vision sensor JSON."
                )
            self._configure_requested_sensors()
            # 清理 UE 中上一轮的 Capture 后重新请求本次传感器。
            self._capture.sendUE4Cmd(
                "RflyClearCapture",
                self.config.ue_window_id,
            )
            time.sleep(0.2)
            if not self._capture.sendReqToUE4(self.config.ue_window_id):
                self._capture = None
                raise RuntimeError(
                    "RflySim3D rejected the RGB-D sensor request."
                )

            self._capture.startImgCap(False)
            self._started = True
            self._timestamp_offset = None
            try:
                frame = self.get_frame(
                    timeout=float(self.config.first_frame_timeout)
                )
            except Exception:
                self.close()
                raise

        self._start_preview()
        return frame

    def get_frame(
        self,
        timeout: float = 1.0,
        after_timestamp: float | None = None,
    ) -> RGBDFrame:
        """返回最新的一帧完整 RGB-D 观测。

        RGB 与深度分别由不同的 SDK 接收线程更新。复制图像时持有对应的
        SDK 锁，防止感知代码读取数组期间接收线程同时写入。仅当两幅图的
        仿真时间戳差不超过 ``max_rgb_depth_skew`` 时才返回。
        """
        if not self._started or self._capture is None:
            raise RuntimeError("RGB-D camera is not started.")

        deadline = time.monotonic() + float(timeout)
        rgb_index = self._rgb_capture_index
        depth_index = self._depth_capture_index
        segmentation_index = self._segmentation_capture_index
        while time.monotonic() < deadline:
            if self._indices_have_data(rgb_index, depth_index):
                rgb, rgb_stamp = self._copy_sensor(rgb_index)
                depth_raw, depth_stamp = self._copy_sensor(depth_index)
                if self._valid_array(rgb) and self._valid_array(depth_raw):
                    if rgb_stamp <= 0.0 or depth_stamp <= 0.0:
                        time.sleep(float(self.config.poll_interval))
                        continue
                    segmentation = None
                    segmentation_stamp = 0.0
                    if (
                        segmentation_index is not None
                        and self._index_has_data(segmentation_index)
                    ):
                        candidate, candidate_stamp = self._copy_sensor(
                            segmentation_index
                        )
                        if self._valid_array(candidate):
                            segmentation = candidate
                            segmentation_stamp = candidate_stamp
                    received_at = time.time()
                    raw_midpoint = 0.5 * (rgb_stamp + depth_stamp)
                    timestamp_offset = getattr(
                        self,
                        "_timestamp_offset",
                        None,
                    )
                    if timestamp_offset is None:
                        timestamp_offset = received_at - raw_midpoint
                        self._timestamp_offset = timestamp_offset
                    rgb_stamp += timestamp_offset
                    depth_stamp += timestamp_offset
                    if segmentation_stamp > 0.0:
                        segmentation_stamp += timestamp_offset
                    frame = RGBDFrame(
                        rgb_bgr=rgb,
                        depth_m=self.normalize_depth_to_m(depth_raw),
                        rgb_timestamp=rgb_stamp,
                        depth_timestamp=depth_stamp,
                        received_at=received_at,
                        segmentation_bgr=segmentation,
                        segmentation_timestamp=segmentation_stamp,
                    )
                    is_new = (
                        after_timestamp is None
                        or frame.timestamp > float(after_timestamp) + 1e-9
                    )
                    timestamps_valid = (
                        frame.rgb_timestamp > 0.0
                        and frame.depth_timestamp > 0.0
                    )
                    if (
                        is_new
                        and timestamps_valid
                        and frame.sync_skew
                        <= float(self.config.max_rgb_depth_skew)
                    ):
                        self._cache_latest_frame(frame)
                        return frame
            time.sleep(float(self.config.poll_interval))

        raise TimeoutError(
            f"No complete RGB-D frame received within {timeout:.1f} s. "
            "Check RflySim3D and the configured UE window."
        )

    @property
    def camera_intrinsics(self) -> np.ndarray:
        """返回 RGB 相机针孔内参矩阵 K。

        JSON 的 ``CameraFOV`` 按水平视场角处理，并假设像素为正方形、
        主点位于图像中心；RflySim 针孔相机默认无畸变。
        """
        sensor = self._rgb_sensor
        width = float(sensor["DataWidth"])
        height = float(sensor["DataHeight"])
        horizontal_fov = math.radians(float(sensor["CameraFOV"]))
        focal = width / (2.0 * math.tan(horizontal_fov * 0.5))
        return np.asarray(
            [
                [focal, 0.0, (width - 1.0) * 0.5],
                [0.0, focal, (height - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def camera_extrinsics(self) -> np.ndarray:
        """返回相机光学坐标系到无人机机体 FRD 的齐次变换。

        返回矩阵记作 ``T_body_camera``，满足
        ``point_body = T_body_camera @ point_camera``。相机光学坐标为
        x 向右、y 向下、z 向前；机体 FRD 为 x 向前、y 向右、z 向下。
        """
        sensor = self._rgb_sensor
        translation = np.asarray(sensor["SensorPosXYZ"], dtype=float)
        roll, pitch, yaw = np.asarray(
            sensor["SensorAngEular"], dtype=float
        )
        rotation_mount = self._rotation_zyx(roll, pitch, yaw)
        optical_to_mount = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_mount @ optical_to_mount
        transform[:3, 3] = translation
        return transform

    def normalize_depth_to_m(self, depth_raw: np.ndarray) -> np.ndarray:
        """将 RflySim 原始深度转换成以米为单位的 float32 二维数组。"""
        depth = np.asarray(depth_raw)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")

        if depth.dtype == np.uint16:
            result = depth.astype(np.float32) * self.config.depth_unit_scale
        elif depth.dtype == np.uint8:
            result = depth.astype(np.float32) * (self.config.depth_max / 255.0)
        else:
            result = depth.astype(np.float32)
            finite = result[np.isfinite(result)]
            if finite.size and float(np.max(finite)) > 200.0:
                result *= self.config.depth_unit_scale

        invalid = (~np.isfinite(result)) | (result <= 0.0)
        invalid |= (result < self.config.depth_min) | (
            result > self.config.depth_max
        )
        result[invalid] = 0.0
        return result

    def close(self) -> None:
        """停止预览和 SDK 采集线程；重复调用也是安全的。"""
        self._stop_preview()
        capture = self._capture
        self._capture = None
        self._started = False
        self._timestamp_offset = None
        with self._latest_frame_lock:
            self._latest_frame = None
        if capture is not None:
            # RflySimSDK 4.12 的 stopRun() 无条件读取 t_menRec，
            # 而 UDP 图像模式不会创建该属性，会在清理时用
            # AttributeError 覆盖真正的启动异常。统一设置停止
            # 标志，仅在共享内存接收线程存在时等待它退出。
            capture.stopFlag = True
            memory_thread = getattr(capture, "t_menRec", None)
            if memory_thread is not None:
                memory_thread.join(timeout=2.0)

    def depth_to_colormap(self, depth_m: np.ndarray) -> np.ndarray:
        """根据配置将米制深度转换成伪彩色图或黑白图。"""
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth >= self.config.depth_min)
        valid &= depth <= self.config.depth_max

        gray = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            normalized = (
                (self.config.depth_max - depth[valid])
                / (self.config.depth_max - self.config.depth_min)
            )
            gray[valid] = np.clip(normalized * 255.0, 0, 255).astype(
                np.uint8
            )

        mode = str(self.config.depth_display_mode).strip().lower()
        if mode == "grayscale":
            return gray
        if mode != "color":
            raise ValueError(
                "depth_display_mode must be 'color' or 'grayscale'."
            )

        cv2 = self._require_cv2()
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        return color

    def _start_preview(self) -> None:
        if not (
            self.config.show_rgb
            or self.config.show_depth
            or self.config.show_segmentation
            or self._detection_preview_provider is not None
        ):
            return
        if self.config.show_depth and str(
            self.config.depth_display_mode
        ).strip().lower() not in ("color", "grayscale"):
            raise ValueError(
                "depth_display_mode must be 'color' or 'grayscale'."
            )
        self._require_cv2()
        if self.config.preview_fps <= 0:
            raise ValueError("preview_fps must be greater than zero.")
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return

        self._preview_stop.clear()
        self._preview_thread = threading.Thread(
            target=self._preview_loop,
            name=f"RflyRGBDPreview-{self.config.copter_id}",
            daemon=True,
        )
        self._preview_thread.start()

    def _preview_loop(self) -> None:
        cv2 = self._require_cv2()
        period = 1.0 / float(self.config.preview_fps)
        next_frame_time = time.monotonic()
        show_camera_preview = bool(
            self.config.show_rgb
            or self.config.show_depth
            or self.config.show_segmentation
        )
        try:
            if show_camera_preview:
                cv2.namedWindow(
                    self.PREVIEW_WINDOW_NAME,
                    cv2.WINDOW_AUTOSIZE,
                )
                cv2.moveWindow(self.PREVIEW_WINDOW_NAME, 20, 40)
            if self._detection_window_name is not None:
                cv2.namedWindow(
                    self._detection_window_name,
                    cv2.WINDOW_AUTOSIZE,
                )
                cv2.moveWindow(self._detection_window_name, 520, 40)
            while not self._preview_stop.is_set():
                frame = (
                    self._latest_frame_snapshot()
                    if show_camera_preview
                    else None
                )

                preview_images = []
                if frame is not None and self.config.show_rgb:
                    preview_images.append(frame.rgb_bgr)
                if frame is not None and self.config.show_depth:
                    depth_preview = self.depth_to_colormap(frame.depth_m)
                    if depth_preview.ndim == 2:
                        depth_preview = cv2.cvtColor(
                            depth_preview,
                            cv2.COLOR_GRAY2BGR,
                        )
                    preview_images.append(depth_preview)
                if frame is not None and self.config.show_segmentation:
                    segmentation_preview = frame.segmentation_bgr
                    if segmentation_preview is None:
                        segmentation_preview = np.zeros_like(frame.rgb_bgr)
                        cv2.putText(
                            segmentation_preview,
                            "TypeID=4: waiting for frame...",
                            (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        segmentation_preview = self._annotate_segmentation_status(
                            segmentation_preview
                        )
                    preview_images.append(segmentation_preview)

                if preview_images:
                    labels = []
                    if self.config.show_rgb:
                        labels.append("RGB (TypeID=1)")
                    if self.config.show_depth:
                        labels.append("Depth (TypeID=2)")
                    if self.config.show_segmentation:
                        labels.append("Segmentation (TypeID=4)")
                    combined_preview = self._combine_preview_panels(
                        preview_images,
                        labels,
                    )
                    cv2.imshow(
                        self.PREVIEW_WINDOW_NAME,
                        combined_preview,
                    )

                if self._detection_preview_provider is not None:
                    detection_preview = self._detection_preview_provider()
                    if detection_preview is not None:
                        cv2.imshow(
                            self._detection_window_name,
                            detection_preview,
                        )

                # OpenCV 需要通过 waitKey 处理窗口事件。按 q 只关闭预览，
                # 不会中断无人机飞行任务。
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._preview_stop.set()
                    break

                next_frame_time += period
                wait_time = next_frame_time - time.monotonic()
                if wait_time > 0:
                    self._preview_stop.wait(wait_time)
                else:
                    next_frame_time = time.monotonic()
        except cv2.error as error:
            print(f"--- WARNING: RGB-D preview stopped: {error} ----")
        finally:
            self._destroy_preview_windows()

    def _stop_preview(self) -> None:
        self._preview_stop.set()
        thread = self._preview_thread
        self._preview_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._destroy_preview_windows()

    def _destroy_preview_windows(self) -> None:
        if self._cv2 is None:
            return
        cv2 = self._cv2
        if (
            self.config.show_rgb
            or self.config.show_depth
            or self.config.show_segmentation
        ):
            try:
                cv2.destroyWindow(self.PREVIEW_WINDOW_NAME)
            except cv2.error:
                pass
        if self._detection_window_name is not None:
            try:
                cv2.destroyWindow(self._detection_window_name)
            except cv2.error:
                pass

    def _require_cv2(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            self._cv2 = importlib.import_module("cv2")
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required for RGB, depth, or detection previews. "
                "Activate the vision Conda environment or install opencv-python."
            ) from error
        return self._cv2

    def _indices_have_data(self, rgb_index: int, depth_index: int) -> bool:
        capture = self._capture
        required = max(rgb_index, depth_index)
        return bool(
            len(capture.Img) > required
            and len(capture.hasData) > required
            and capture.hasData[rgb_index]
            and capture.hasData[depth_index]
        )

    def _index_has_data(self, index: int) -> bool:
        capture = self._capture
        return bool(
            len(capture.Img) > index
            and len(capture.hasData) > index
            and capture.hasData[index]
        )

    def _combine_preview_panels(self, images, labels) -> np.ndarray:
        """把若干相机画面缩小后纵向合成一个窗口。"""
        cv2 = self._require_cv2()
        source_height, source_width = images[0].shape[:2]
        target_width = min(self.PREVIEW_PANEL_WIDTH, source_width)
        target_height = max(
            1,
            round(source_height * target_width / source_width),
        )
        panels = []
        for image, label in zip(images, labels):
            panel = np.asarray(image)
            if panel.ndim == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            if panel.shape[:2] != (target_height, target_width):
                interpolation = (
                    cv2.INTER_AREA
                    if label.startswith("RGB")
                    else cv2.INTER_NEAREST
                )
                panel = cv2.resize(
                    panel,
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            else:
                panel = panel.copy()
            cv2.rectangle(panel, (0, 0), (target_width, 34), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        return np.vstack(panels)

    def _annotate_segmentation_status(self, image) -> np.ndarray:
        """在黑图或近似单色图上提示 Stencil 标注状态。"""
        cv2 = self._require_cv2()
        output = np.asarray(image).copy()
        sample = output[::8, ::8]
        if sample.ndim == 2:
            colors = np.unique(sample)
            all_black = not np.any(sample)
        else:
            colors = np.unique(sample.reshape(-1, sample.shape[-1]), axis=0)
            all_black = not np.any(sample)
        if all_black:
            message = "BLACK: UE scene has no nonzero Stencil labels"
        elif len(colors) <= 2:
            message = "UE scene contains only one visible Stencil class"
        else:
            return output
        cv2.putText(
            output,
            message,
            (12, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def _copy_sensor(self, index: int):
        capture = self._capture
        with capture.Img_lock[index]:
            image = capture.Img[index]
            if isinstance(image, np.ndarray):
                image = image.copy()
            stamp = self._sensor_timestamp(index)
        return image, stamp

    def _cache_latest_frame(self, frame: RGBDFrame) -> None:
        """缓存 SDK 已复制的帧，供预览线程复用。"""
        with self._latest_frame_lock:
            self._latest_frame = frame

    def _latest_frame_snapshot(self) -> RGBDFrame | None:
        """返回最新不可变帧容器，不再向 SDK 重复取帧。"""
        with self._latest_frame_lock:
            return self._latest_frame

    def _configure_requested_sensors(self) -> None:
        """过滤 SDK 请求列表，隐藏 TypeID=4 时不请求该传感器。"""
        capture = self._capture
        if capture is None:
            raise RuntimeError("Vision capture API is not initialized.")

        segmentation_seq_id = int(self._segmentation_sensor["SeqID"])

        def requested(sensor) -> bool:
            is_segmentation = (
                int(sensor.SeqID) == segmentation_seq_id
                and int(sensor.TypeID) == 4
            )
            return bool(self.config.show_segmentation) or not is_segmentation

        capture.VisSensor = [
            sensor for sensor in capture.VisSensor if requested(sensor)
        ]
        capture.VisSensorNew = [
            sensor
            for sensor in capture.VisSensorNew
            if requested(sensor)
        ]

        def find_index(sensor_definition, type_id: int) -> int:
            seq_id = int(sensor_definition["SeqID"])
            for index, sensor in enumerate(capture.VisSensor):
                if (
                    int(sensor.SeqID) == seq_id
                    and int(sensor.TypeID) == type_id
                ):
                    return index
            raise RuntimeError(
                f"Requested TypeID={type_id} sensor is missing after filtering."
            )

        self._rgb_capture_index = find_index(self._rgb_sensor, 1)
        self._depth_capture_index = find_index(self._depth_sensor, 2)
        self._segmentation_capture_index = (
            find_index(self._segmentation_sensor, 4)
            if self.config.show_segmentation
            else None
        )

    def _sensor_timestamp(self, index: int) -> float:
        """返回随每次成功解码更新的图像时间戳。

        新版 VisionCaptureApi 的 UDP split 解码路径逐帧更新
        ``timeStmp``，而 ``imgStmp`` 只在解码统计分支中更新，因此使用
        SDK 的起始时间与逐帧相对时间实时合成。
        """
        capture = self._capture
        relative = float(np.asarray(capture.timeStmp[index]).reshape(-1)[0])
        start = float(np.asarray(capture.rflyStartStmp[index]).reshape(-1)[0])
        return start + relative if relative > 0.0 and start > 0.0 else 0.0

    @staticmethod
    def _valid_array(value) -> bool:
        return isinstance(value, np.ndarray) and value.size > 0

    def _validate_files(self) -> None:
        sdk_root = Path(self.config.sdk_root)
        if not sdk_root.is_dir():
            raise FileNotFoundError(f"RflySim SDK directory not found: {sdk_root}")
        if not Path(self.config.sensor_json).is_file():
            raise FileNotFoundError(
                f"RGB-D sensor configuration not found: {self.config.sensor_json}"
            )

    def _load_and_validate_sensor_json(self) -> None:
        with Path(self.config.sensor_json).open("r", encoding="utf-8") as file:
            sensors = json.load(file).get("VisionSensors", [])

        def sensor_with_type(type_id: int, name: str):
            matches = [
                sensor
                for sensor in sensors
                if int(sensor.get("TypeID", -1)) == type_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Sensor JSON must contain exactly one {name} "
                    f"sensor with TypeID={type_id}."
                )
            return matches[0]

        rgb = sensor_with_type(1, "RGB")
        depth = sensor_with_type(2, "depth")
        segmentation = sensor_with_type(4, "segmentation")
        self._rgb_sensor = rgb
        self._depth_sensor = depth
        self._segmentation_sensor = segmentation
        for name, sensor in (
            ("RGB", rgb),
            ("depth", depth),
            ("segmentation", segmentation),
        ):
            if int(sensor.get("TargetCopter", -1)) != self.config.copter_id:
                raise ValueError(
                    f"{name} TargetCopter does not match CameraConfig.copter_id."
                )
        for field in (
            "TargetMountType",
            "DataWidth",
            "DataHeight",
            "CameraFOV",
            "SensorPosXYZ",
            "SensorAngEular",
        ):
            for name, sensor in (
                ("depth", depth),
                ("segmentation", segmentation),
            ):
                if rgb.get(field) != sensor.get(field):
                    raise ValueError(
                        f"RGB and {name} sensors must share {field} "
                        "for alignment."
                    )

    @staticmethod
    def _rotation_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """由 roll、pitch、yaw（rad）生成 ZYX 旋转矩阵。"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.asarray(
            [
                [cy * cp, cy * sp * sr - sy * cr,
                 cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr,
                 sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    def _import_vision_capture_api(self):
        sdk_root = Path(self.config.sdk_root)
        for relative in ("vision", "ctrl", "ue"):
            path = str(sdk_root / relative)
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            return importlib.import_module("VisionCaptureApi")
        except ImportError as error:
            raise ImportError(
                f"Cannot import VisionCaptureApi from {sdk_root / 'vision'}"
            ) from error
