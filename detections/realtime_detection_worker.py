"""Run YOLO-World detection and SAM2 segmentation on the latest RGB frame."""

from navigation.perception import SemanticFrameStore

import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .object_detect import DetectorConfig, SAM2Segmenter, TiledVehicleDetector


class RealtimeDetectionWorker:
    """Independent latest-frame detector that never blocks navigation."""

    PREVIEW_PANEL_WIDTH = 480

    def __init__(self, observation_source, config=None) -> None:
        self.observation_source = observation_source
        self.config = config or DetectorConfig()
        self.detector = TiledVehicleDetector(self.config)
        self.segmenter = SAM2Segmenter(self.config)
        self.semantic_frames = SemanticFrameStore()
        self._semantic_history = None
        self.processed_frames = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._preview_lock = threading.Lock()
        self._latest_rendered = None
        self._last_completed_at = None
        self._display_fps = 0.0
        self._world_fps = 0.0
        self._segmentation_fps = 0.0
        self._world_video_writer = None
        self._sam_video_writer = None
        self._video_thread = None
        self._video_error = None
        self._semantic_map_error = None
        self._video_queue = queue.Queue(maxsize=3)
        self._recording_started_at = None
        self._recorded_frames = 0
        output_dir = Path(self.config.video_output_dir)
        self.world_video_path = output_dir / "yolo_world_detection.mp4"
        self.sam_video_path = output_dir / "sam2_segmentation.mp4"

    @property
    def video_output_paths(self) -> tuple[Path, ...]:
        """返回本次检测和分割视频的预期输出路径。"""
        if not self.config.save_videos:
            return ()
        return self.world_video_path, self.sam_video_path

    @property
    def semantic_map_output_path(self) -> Path | None:
        if not self.config.save_semantic_map:
            return None
        return Path(self.config.semantic_map_path)

    def semantic_snapshot(self):
        """Thread-safe complete frame for navigation; never exposes the live mapper."""
        return self.semantic_frames.snapshot()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._video_error = None
        self._semantic_map_error = None
        self._last_completed_at = None
        self._recording_started_at = None
        self._recorded_frames = 0
        self._video_queue = queue.Queue(maxsize=3)
        if self.config.save_videos:
            self._video_thread = threading.Thread(
                target=self._video_loop,
                name="YOLOWorldAndSAM2VideoWriter",
                daemon=True,
            )
            self._video_thread.start()
        self._thread = threading.Thread(
            target=self._detection_loop,
            name="RealtimeYOLOWorldAndSAM2Segmentation",
            daemon=True,
        )
        self._thread.start()
        precision = (
            "FP16"
            if self.config.prediction_quantization == 16
            else "FP32"
        )
        print(
            "--- Real-time YOLO-World + SAM2 enabled: combined window "
            f"'{self.config.window_name}' is active; {precision}, "
            f"target <= {self.config.inference_fps:.0f} FPS ----"
        )

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        self._stop_video_thread()
        if self._semantic_history is not None and (thread is None or not thread.is_alive()):
            self._semantic_history.close()
            self._semantic_history = None
        if self.config.save_semantic_map:
            try:
                self.segmenter.save_semantic_map()
            except Exception as error:  # pylint: disable=broad-except
                self._semantic_map_error = error
                print(f"--- WARNING: semantic map save failed: {error} ----")

    def raise_if_failed(self) -> None:
        error = self._error or self._video_error or self._semantic_map_error
        if error is not None:
            raise RuntimeError(
                "Real-time YOLO-World/SAM2 perception failed: "
                f"{error!r}"
            ) from error

    def get_latest_preview(self, fallback_image=None):
        """返回最新标注帧；可选原图仅用于首帧占位。"""
        with self._preview_lock:
            rendered = self._latest_rendered
        if rendered is not None:
            error = (
                self._error
                or self._video_error
                or self._semantic_map_error
            )
            if error is not None:
                rendered = rendered.copy()
                cv2.putText(
                    rendered,
                    f"PERCEPTION ERROR: {type(error).__name__}: {error}",
                    (12, rendered.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            return rendered
        if fallback_image is None:
            return None

        preview = fallback_image.copy()
        if self._error is None:
            message = "YOLO-World + SAM2: waiting for first inference..."
            color = (0, 255, 255)
        else:
            message = f"YOLO-WORLD ERROR: {type(self._error).__name__}"
            color = (0, 0, 255)
        cv2.putText(
            preview,
            message,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return self._combine_previews(preview, preview)

    def _detection_loop(self) -> None:
        period = 1.0 / float(self.config.inference_fps)
        try:
            while not self._stop.is_set():
                try:
                    observation = self.observation_source.get_observation(
                        timeout=0.5
                    )
                except TimeoutError:
                    continue
                except RuntimeError:
                    break

                started = time.monotonic()
                sensor_frame = observation.sensor_frame
                # 检测和分割只读输入；绘图函数各自创建输出，无需每帧预复制 RGB。
                rgb_bgr = sensor_frame.rgb
                world_started = time.monotonic()
                detections = self.detector.detect(rgb_bgr)
                world_elapsed = max(
                    time.monotonic() - world_started,
                    1e-6,
                )
                segmentation_started = time.monotonic()
                segmentation, segment_count = self.segmenter.segment(
                    rgb_bgr,
                    detections,
                    depth=sensor_frame.depth,
                    camera_intrinsics=sensor_frame.camera_intrinsics,
                    camera_pose_world=observation.camera_pose_world,
                    timestamp=sensor_frame.timestamp,
                    # 兼容不提供协方差的离线/自定义观测源。
                    camera_pose_covariance=getattr(
                        observation, "camera_pose_covariance", None
                    ),
                )
                segmentation_elapsed = max(
                    time.monotonic() - segmentation_started,
                    1e-6,
                )
                elapsed = max(time.monotonic() - started, 1e-6)
                records = self.segmenter.mapper.navigation_records(self.config.semantic_map_min_observations)
                self.semantic_frames.publish(sensor_frame.timestamp, records, elapsed)
                if self.config.save_semantic_map:
                    if self._semantic_history is None:
                        history_path = Path(self.config.semantic_map_path).with_name("semantic_observations.jsonl")
                        history_path.parent.mkdir(parents=True, exist_ok=True)
                        self._semantic_history = history_path.open("w", encoding="utf-8", buffering=1)
                    self._semantic_history.write(json.dumps({
                        "timestamp": float(sensor_frame.timestamp),
                        "inference_seconds": elapsed, "records": records,
                    }, ensure_ascii=False, allow_nan=False) + "\n")
                self.processed_frames += 1

                completed_at = time.monotonic()
                if self._last_completed_at is None:
                    measured_fps = min(
                        float(self.config.inference_fps),
                        1.0 / elapsed,
                    )
                else:
                    measured_fps = 1.0 / max(
                        completed_at - self._last_completed_at,
                        1e-6,
                    )
                self._last_completed_at = completed_at
                if self._display_fps <= 0.0:
                    self._display_fps = measured_fps
                else:
                    self._display_fps = (
                        0.8 * self._display_fps + 0.2 * measured_fps
                    )
                measured_world_fps = 1.0 / world_elapsed
                measured_segmentation_fps = 1.0 / segmentation_elapsed
                if self._world_fps <= 0.0:
                    self._world_fps = measured_world_fps
                    self._segmentation_fps = measured_segmentation_fps
                else:
                    self._world_fps = (
                        0.8 * self._world_fps + 0.2 * measured_world_fps
                    )
                    self._segmentation_fps = (
                        0.8 * self._segmentation_fps
                        + 0.2 * measured_segmentation_fps
                    )

                rendered = self.detector.draw(rgb_bgr, detections)
                cv2.putText(
                    rendered,
                    "YOLO-World "
                    f"{self._world_fps:.1f} infer FPS | "
                    f"{world_elapsed * 1000.0:.0f} ms | "
                    f"pipeline {self._display_fps:.1f} FPS | "
                    f"objects: {len(detections)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                memory_count = getattr(
                    self.segmenter,
                    "memory_count",
                    len(getattr(self.segmenter, "latest_memory", ())),
                )
                cv2.putText(
                    segmentation,
                    "SAM2 "
                    f"{self._segmentation_fps:.1f} infer FPS | "
                    f"{segmentation_elapsed * 1000.0:.0f} ms | "
                    f"pipeline {self._display_fps:.1f} FPS | "
                    f"instances: {segment_count} | "
                    f"3D memory: {memory_count}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                combined = self._combine_previews(rendered, segmentation)
                with self._preview_lock:
                    self._latest_rendered = combined
                # 先发布实时预览，再把录像交给独立线程。磁盘或编码器变慢
                # 时只丢弃旧录像帧，不能反向阻塞模型推理和窗口刷新。
                self._enqueue_video_frames(
                    rendered, segmentation, completed_at
                )

                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except Exception as error:
            if not self._stop.is_set():
                self._error = error
                print(
                    "--- WARNING: real-time YOLO-World/SAM2 stopped: "
                    f"{error} ----"
                )

        finally:
            if self._semantic_history is not None:
                self._semantic_history.close()
                self._semantic_history = None

    def _enqueue_video_frames(self, world_frame, sam_frame, timestamp) -> None:
        if not self.config.save_videos or self._video_error is not None:
            return
        item = (world_frame, sam_frame, float(timestamp))
        try:
            self._video_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        # 始终保留最新画面，录像线程落后时丢弃最旧画面。
        try:
            self._video_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._video_queue.put_nowait(item)
        except queue.Full:
            pass

    def _video_loop(self) -> None:
        try:
            while True:
                item = self._video_queue.get()
                if item is None:
                    break
                world_frame, sam_frame, timestamp = item
                self._record_video_frames(
                    world_frame, sam_frame, timestamp
                )
        except Exception as error:
            self._video_error = error
            print(
                "--- WARNING: detection video writer stopped: "
                f"{error} ----"
            )
        finally:
            self._release_video_writers()

    def _stop_video_thread(self) -> None:
        thread = self._video_thread
        self._video_thread = None
        if thread is None:
            return
        while thread.is_alive():
            try:
                self._video_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._video_queue.get_nowait()
                except queue.Empty:
                    pass
        if thread is not threading.current_thread():
            thread.join(timeout=10.0)

    def _record_video_frames(self, world_frame, sam_frame, timestamp) -> None:
        """按真实经过时间写入定帧率视频，慢推理时用当前帧补齐时长。"""
        if self._world_video_writer is None:
            output_dir = Path(self.config.video_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            height, width = world_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            frame_size = (int(width), int(height))
            self._world_video_writer = cv2.VideoWriter(
                str(self.world_video_path),
                fourcc,
                float(self.config.video_fps),
                frame_size,
            )
            self._sam_video_writer = cv2.VideoWriter(
                str(self.sam_video_path),
                fourcc,
                float(self.config.video_fps),
                frame_size,
            )
            if not (
                self._world_video_writer.isOpened()
                and self._sam_video_writer.isOpened()
            ):
                self._release_video_writers()
                raise RuntimeError(
                    "OpenCV 无法创建 MP4 视频，请检查 mp4v 编码器"
                )
            self._recording_started_at = float(timestamp)

        elapsed = max(0.0, float(timestamp) - self._recording_started_at)
        target_count = int(elapsed * float(self.config.video_fps)) + 1
        missing_frames = max(0, target_count - self._recorded_frames)
        # 长时间无推理结果后不能一次补写无限多帧。最多补一秒，随后
        # 跳到当前时间点，避免录像线程持续积压。
        frames_to_write = min(
            missing_frames,
            max(1, int(round(float(self.config.video_fps)))),
        )
        for _ in range(frames_to_write):
            self._world_video_writer.write(world_frame)
            self._sam_video_writer.write(sam_frame)
        self._recorded_frames = max(self._recorded_frames, target_count)

    def _release_video_writers(self) -> None:
        for writer_name in ("_world_video_writer", "_sam_video_writer"):
            writer = getattr(self, writer_name)
            if writer is not None:
                writer.release()
                setattr(self, writer_name, None)

    @staticmethod
    def _combine_previews(world_preview, segmentation_preview):
        """把检测与分割画面缩小后纵向组合。"""
        left = np.asarray(world_preview)
        right = np.asarray(segmentation_preview)
        source_height, source_width = left.shape[:2]
        target_width = min(
            RealtimeDetectionWorker.PREVIEW_PANEL_WIDTH,
            source_width,
        )
        target_height = max(
            1,
            round(source_height * target_width / source_width),
        )
        if left.shape[:2] != (target_height, target_width):
            left = cv2.resize(
                left,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        if right.shape[:2] != (target_height, target_width):
            right = cv2.resize(
                right,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.vstack((left, right))
