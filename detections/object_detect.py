"""YOLO-World 开放词汇检测、SAM2 分割与持久语义地图接口。"""

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2  # pylint: disable=import-error
import numpy as np
import torch  # pylint: disable=import-error
import ultralytics.nn.text_model as ultralytics_text_model
from ultralytics import SAM, YOLOWorld  # pylint: disable=import-error

from mapping.semantic_instance_mapper import (
    COLUMN_CLASS_ALIASES,
    SIGNBOARD_CLASS_ALIASES,
    FIRE_CABINET_CLASS_ALIASES,
    SemanticInstanceMapper,
    SemanticObservation3D,
    canonical_semantic_class_name as canonical_class_name,
    instance_class_name,
)
from mapping.vehicle_instance_tracker import VEHICLE_CLASSES
from .vehicle_color import vehicle_color_scores
from sensors.pose_uncertainty import world_point_covariance


BASE_DIR = Path(__file__).parent
CLIP_WEIGHTS_DIR = BASE_DIR / "weights"
DEFAULT_SEMANTIC_MAP_PATH = (
    BASE_DIR / "output" / "semantic_instances.json"
)
SEMANTIC_MAP_FORMAT_VERSION = 5

# OpenCV 使用 BGR。颜色按规范语义类别固定，不能依赖每帧检测排序。
SEMANTIC_CLASS_COLORS = {
    "floor": (128, 128, 128),
    "wall": (180, 120, 60),
    "ceiling": (210, 210, 210),
    "column": (0, 165, 255),
    "signboard": (255, 0, 255),
    "fire cabinet": (40, 40, 230),
    "car": (255, 80, 80),
    "truck": (180, 80, 255),
    "bus": (80, 200, 255),
    "van": (255, 160, 80),
    "person": (80, 80, 255),
    "parking barrier": (255, 255, 0),
}
DEFAULT_SEMANTIC_COLOR = (80, 255, 80)

def semantic_class_color(class_name: str) -> tuple[int, int, int]:
    """返回规范类别稳定的 OpenCV BGR颜色。"""
    canonical = canonical_class_name(class_name)
    return SEMANTIC_CLASS_COLORS.get(canonical, DEFAULT_SEMANTIC_COLOR)


# The fields intentionally mirror independent detector settings.
# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class DetectorConfig:
    """YOLO-World 与 SAM2 共用的实时感知配置。"""

    weights: str | Path = (
        BASE_DIR / "Yolomode" / "yolov8s-worldv2.pt"
    )
    sam_weights: str | Path = BASE_DIR / "Yolomode" / "sam2.1_t.pt"
    class_prompts: tuple[str, ...] = (
        "floor",
        "wall",
        "ceiling",
        *COLUMN_CLASS_ALIASES,
        *SIGNBOARD_CLASS_ALIASES,
        *FIRE_CABINET_CLASS_ALIASES,
        "car",
        "truck",
        "bus",
        "van",
        "person",
        "parking barrier",
    )
    tile_size: int = 640
    overlap: float = 0.25
    confidence: float = 0.12
    tile_iou: float = 0.50
    nms_iou: float = 0.45
    max_detections: int = 30
    max_sam_prompts: int = 12
    device: str | int = 0
    use_fp16: bool = True
    enabled: bool = True
    inference_fps: float = 30.0
    window_name: str = "YOLO-World Detection / SAM2 Segmentation"
    hidden_box_classes: tuple[str, ...] = ("floor", "ceiling", "wall")
    semantic_landmark_classes: tuple[str, ...] = ("column", *VEHICLE_CLASSES)
    instance_duplicate_merge_distance_xy: float = 1.5
    instance_mahalanobis_gate: float = 7.815
    instance_kalman_process_noise_std: float = 0.05
    instance_mask_pixel_noise_std: float = 3.0
    instance_depth_noise_std: float = 0.60
    column_min_vertical_extent: float = 1.5
    column_min_vertical_aspect_ratio: float = 1.2
    instance_depth_min: float = 0.3
    instance_depth_max: float = 15.0
    instance_depth_stride: int = 4
    instance_depth_min_points: int = 12
    vehicle_max_association_distance: float = 2.5
    vehicle_appearance_gate: float = 0.55
    vehicle_ambiguity_margin: float = 0.12
    vehicle_viewpoint_noise_std: float = 0.45
    vehicle_confirmation_observations: int = 3
    semantic_map_path: Path = DEFAULT_SEMANTIC_MAP_PATH
    semantic_map_min_observations: int = 3
    instance_display_min_observations: int = 10
    save_semantic_map: bool = True
    show_instance_ids: bool = True
    save_videos: bool = True
    video_output_dir: Path = BASE_DIR / "output"
    video_fps: float = 15.0
    video_codec: str = "mp4v"

    @property
    def prediction_quantization(self) -> int | None:
        """CUDA 使用 FP16，CPU/MPS 保持兼容的默认精度。"""
        device = str(self.device).strip().lower()
        uses_cuda = (
            isinstance(self.device, int)
            or device.isdigit()
            or device.startswith("cuda")
        )
        return 16 if self.use_fp16 and uses_cuda else None

    def __post_init__(self) -> None:
        if not self.class_prompts or any(
            not str(prompt).strip() for prompt in self.class_prompts
        ):
            raise ValueError("class_prompts 必须包含非空文本类别")
        if not str(self.sam_weights).strip():
            raise ValueError("sam_weights 不能为空")
        if self.tile_size <= 0:
            raise ValueError("tile_size 必须大于 0")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap 必须在 [0, 1) 范围内")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence 必须在 [0, 1] 范围内")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou 必须在 [0, 1] 范围内")
        if self.max_detections <= 0:
            raise ValueError("max_detections 必须大于 0")
        if self.max_sam_prompts <= 0:
            raise ValueError("max_sam_prompts 必须大于 0")
        if self.instance_duplicate_merge_distance_xy <= 0:
            raise ValueError(
                "instance_duplicate_merge_distance_xy 必须大于 0"
            )
        if self.instance_mahalanobis_gate <= 0:
            raise ValueError("instance_mahalanobis_gate 必须大于 0")
        if self.instance_kalman_process_noise_std < 0:
            raise ValueError(
                "instance_kalman_process_noise_std 不能小于 0"
            )
        if self.instance_mask_pixel_noise_std <= 0:
            raise ValueError("instance_mask_pixel_noise_std 必须大于 0")
        if self.instance_depth_noise_std <= 0:
            raise ValueError("instance_depth_noise_std 必须大于 0")
        if self.column_min_vertical_extent <= 0:
            raise ValueError("column_min_vertical_extent 必须大于 0")
        if self.column_min_vertical_aspect_ratio <= 0:
            raise ValueError("column_min_vertical_aspect_ratio 必须大于 0")
        if not 0 < self.instance_depth_min < self.instance_depth_max:
            raise ValueError("instance_depth_min/max 必须满足 0 < min < max")
        if self.instance_depth_stride <= 0:
            raise ValueError("instance_depth_stride 必须大于 0")
        if self.instance_depth_min_points <= 0:
            raise ValueError("instance_depth_min_points 必须大于 0")
        for name in (
            "vehicle_max_association_distance", "vehicle_appearance_gate",
            "vehicle_ambiguity_margin", "vehicle_viewpoint_noise_std",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.semantic_map_min_observations <= 0:
            raise ValueError("semantic_map_min_observations 必须大于 0")
        if (self.vehicle_confirmation_observations < 1
                or int(self.vehicle_confirmation_observations) != self.vehicle_confirmation_observations):
            raise ValueError("vehicle_confirmation_observations 必须为正整数")
        if self.instance_display_min_observations <= 0:
            raise ValueError("instance_display_min_observations 必须大于 0")
        if self.inference_fps <= 0:
            raise ValueError("inference_fps must be greater than zero")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be greater than zero")
        if len(self.video_codec) != 4:
            raise ValueError("video_codec must contain exactly four characters")


# pylint: enable=too-many-instance-attributes
@dataclass(frozen=True)
class Detection:
    """单个检测目标。"""

    box: tuple[int, int, int, int]
    confidence: float
    class_name: str


def instance_color(instance_id: int) -> tuple[int, int, int]:
    """根据实例ID生成稳定且分散的 OpenCV BGR颜色。"""
    hue = (int(instance_id) * 47) % 180
    hsv = np.asarray([[[hue, 210, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


class TiledVehicleDetector:
    """可复用的切片开放词汇检测器。

    模型仅在构造时加载一次，后续可以连续处理图片文件或 OpenCV 图像，
    适合接入相机循环、无人机任务类或 Web 服务。
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        weights = Path(self.config.weights).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"模型文件不存在: {weights}")
        self.model = YOLOWorld(str(weights))
        # Ultralytics 默认把 CLIP 文本编码器缓存到项目根目录
        # weights/clip。本项目的检测模型统一放在 detections
        # 下，因此只在构建文本特征前将该库的缓存根目录
        # 指向 detections/weights，不修改用户全局 Ultralytics 设置。
        ultralytics_text_model.WEIGHTS_DIR = CLIP_WEIGHTS_DIR
        self.model.set_classes(list(self.config.class_prompts))
        # 文本特征已经写入 WorldModel；运行期间提示词固定，因此无需让
        # CLIP 文本编码器继续占用内存或随检测网络迁移到 GPU。
        world_model = self.model.model
        if getattr(world_model, "clip_model", None) is not None:
            world_model.clip_model = None

    def detect(  # pylint: disable=too-many-locals
        self, image: np.ndarray
    ) -> list[Detection]:
        """检测 OpenCV BGR 图像并返回结构化结果。"""
        if image is None or image.size == 0:
            raise ValueError("输入图像不能为空")

        height, width = image.shape[:2]
        # 720x405 实时画面的总像素数小于 640x640 的模型
        # 输入预算。此时整帧 letterbox 一次即可；若仅因宽度
        # 多出 80 px 就做两次重叠切片，会几乎翻倍实时延迟。
        # 更大的离线图片仍保留切片路径，避免小目标因整帧
        # 缩小而丢失。
        if height * width <= self.config.tile_size ** 2:
            tiles = ((0, 0, image),)
        else:
            tiles = tuple(
                (
                    x1,
                    y1,
                    image[
                        y1 : min(y1 + self.config.tile_size, height),
                        x1 : min(x1 + self.config.tile_size, width),
                    ],
                )
                for y1 in self._tile_starts(height)
                for x1 in self._tile_starts(width)
            )
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_classes: list[torch.Tensor] = []

        for x1, y1, tile in tiles:
            result = self.model.predict(
                tile,
                imgsz=self.config.tile_size,
                conf=self.config.confidence,
                iou=self.config.tile_iou,
                max_det=self.config.max_detections,
                device=self.config.device,
                quantize=self.config.prediction_quantization,
                verbose=False,
            )[0]

            if len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.detach().cpu()
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            all_boxes.append(boxes)
            all_scores.append(result.boxes.conf.detach().cpu())
            all_classes.append(result.boxes.cls.detach().cpu().to(torch.int64))

        if not all_boxes:
            return []

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        classes = torch.cat(all_classes)
        canonical_classes = [
            canonical_class_name(self.model.names[int(class_id)])
            for class_id in classes.tolist()
        ]
        keep = self._nms(boxes, scores, canonical_classes)
        keep = keep[:self.config.max_detections]

        detections: list[Detection] = []
        for box, score, class_id in zip(boxes[keep], scores[keep], classes[keep]):
            cls = int(class_id)
            detections.append(
                Detection(
                    box=tuple(map(int, box.tolist())),
                    confidence=float(score),
                    class_name=canonical_class_name(self.model.names[cls]),
                )
            )
        return detections

    def draw(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
    ) -> np.ndarray:
        """绘制可见类别检测框；隐藏类别仍参与检测与 SAM2 分割。"""
        output = image.copy()
        hidden_classes = {
            str(name).strip().lower()
            for name in self.config.hidden_box_classes
        }
        for detection in detections:
            if detection.class_name.strip().lower() in hidden_classes:
                continue
            x1, y1, x2, y2 = detection.box
            color = semantic_class_color(detection.class_name)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return output

    def _tile_starts(self, length: int) -> list[int]:
        if length <= self.config.tile_size:
            return [0]
        stride = max(1, int(self.config.tile_size * (1 - self.config.overlap)))
        starts = list(range(0, length - self.config.tile_size + 1, stride))
        last = length - self.config.tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_names: Sequence[str],
    ) -> torch.Tensor:
        """先按规范类别 NMS，再去除不同车型的高度重合重复框。"""
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if len(class_names) != len(boxes):
            raise ValueError("class_names length must match boxes length")

        grouped_indices: dict[str, list[int]] = {}
        for index, class_name in enumerate(class_names):
            grouped_indices.setdefault(str(class_name), []).append(index)

        kept_indices: list[torch.Tensor] = []
        for indices in grouped_indices.values():
            group_indices = torch.as_tensor(indices, dtype=torch.long)
            group_keep = self._nms_one_class(
                boxes[group_indices], scores[group_indices]
            )
            kept_indices.extend(group_indices[group_keep])

        # 保持整体置信度降序，便于预览和后续 SAM2 提示顺序稳定。
        kept = torch.stack(kept_indices)
        # 同一辆车可能同时被 car/van 等文本框命中，只去除高度重合的重复框。
        vehicle_indices = [
            int(index) for index in kept
            if canonical_class_name(class_names[int(index)]) in VEHICLE_CLASSES
        ]
        if len(vehicle_indices) > 1:
            indices = torch.as_tensor(vehicle_indices, dtype=torch.long)
            unique = self._nms_one_class(
                boxes[indices], scores[indices], iou_threshold=0.85
            )
            surviving_vehicles = set(indices[unique].tolist())
            kept = torch.stack([
                index for index in kept
                if int(index) not in vehicle_indices or int(index) in surviving_vehicles
            ])
        return kept[scores[kept].argsort(descending=True)]

    def _nms_one_class(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float | None = None,
    ) -> torch.Tensor:
        """对单个规范类别执行 IoU 抑制。"""
        order = scores.argsort(descending=True)
        threshold = self.config.nms_iou if iou_threshold is None else iou_threshold
        keep: list[torch.Tensor] = []
        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(boxes[current, 0], boxes[rest, 0])
            yy1 = torch.maximum(boxes[current, 1], boxes[rest, 1])
            xx2 = torch.minimum(boxes[current, 2], boxes[rest, 2])
            yy2 = torch.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            current_area = (boxes[current, 2] - boxes[current, 0]) * (
                boxes[current, 3] - boxes[current, 1]
            )
            rest_area = (boxes[rest, 2] - boxes[rest, 0]) * (
                boxes[rest, 3] - boxes[rest, 1]
            )
            iou = intersection / (current_area + rest_area - intersection + 1e-7)
            order = rest[iou <= threshold]

        return torch.stack(keep)


class SAM2Segmenter:
    """用 YOLO-World 检测框提示 SAM2 逐帧生成实例掩膜。"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        weights = Path(self.config.sam_weights).expanduser()
        # 项目内显式模型路径必须存在，避免拼写错误时静默下载其他文件。
        if weights.parent != Path(".") and not weights.is_file():
            raise FileNotFoundError(f"SAM2 模型文件不存在: {weights}")
        self.model = SAM(str(weights))
        self.mapper = SemanticInstanceMapper(
            landmark_classes=self.config.semantic_landmark_classes,
            mahalanobis_gate=self.config.instance_mahalanobis_gate,
            process_noise_std=(
                self.config.instance_kalman_process_noise_std
            ),
            duplicate_merge_distance_xy=(
                self.config.instance_duplicate_merge_distance_xy
            ),
            column_min_vertical_extent=(
                self.config.column_min_vertical_extent
            ),
            column_min_vertical_aspect_ratio=(
                self.config.column_min_vertical_aspect_ratio
            ),
            vehicle_max_association_distance=self.config.vehicle_max_association_distance,
            vehicle_appearance_gate=self.config.vehicle_appearance_gate,
            vehicle_ambiguity_margin=self.config.vehicle_ambiguity_margin,
            vehicle_viewpoint_noise_std=self.config.vehicle_viewpoint_noise_std,
            vehicle_confirmation_observations=self.config.vehicle_confirmation_observations,
        )

    @property
    def memory_count(self) -> int:
        """当前地图实例数，不为实时计数构造完整不可变快照。"""
        return self.mapper.instance_count

    def segment(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        depth: np.ndarray | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_pose_world: np.ndarray | None = None,
        timestamp: float | None = None,
        camera_pose_covariance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """返回分割图；同步深度和位姿可启用持久三维实例关联。"""
        if image is None or image.size == 0:
            raise ValueError("输入图像不能为空")
        all_detections = list(detections)
        if not all_detections:
            rendered = image.copy()
            cv2.putText(
                rendered,
                "No YOLO-World boxes to prompt SAM2",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            return rendered, 0
        active_items = self._select_sam_detections(all_detections)

        result = self.model.predict(
            image,
            bboxes=[list(item[1].box) for item in active_items],
            imgsz=self.config.tile_size,
            device=self.config.device,
            quantize=self.config.prediction_quantization,
            verbose=False,
        )[0]
        masks = None if result.masks is None else result.masks.data.detach().cpu()
        height, width = image.shape[:2]
        prepared_masks: list[tuple[int, np.ndarray]] = []
        observations_3d: list[SemanticObservation3D | None] = (
            [None] * len(all_detections)
        )
        for (detection_index, detection), mask_tensor in zip(
            active_items,
            () if masks is None else masks,
        ):
            mask = mask_tensor.numpy()
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
            foreground = mask > 0.5
            if canonical_class_name(detection.class_name) in VEHICLE_CLASSES:
                # 提示框外的 SAM 泄漏区域不能进入车辆外观和深度测量。
                x1, y1, x2, y2 = detection.box
                bounded = np.zeros_like(foreground)
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                bounded[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
                foreground = bounded
            if not np.any(foreground):
                continue
            prepared_masks.append((detection_index, foreground))
            if self._is_semantic_landmark(detection.class_name):
                observations_3d[detection_index] = self._mask_world_geometry(
                    foreground,
                    depth,
                    camera_intrinsics,
                    camera_pose_world,
                    camera_pose_covariance,
                )
                geometry = observations_3d[detection_index]
                if geometry is not None and canonical_class_name(detection.class_name) in VEHICLE_CLASSES:
                    observations_3d[detection_index] = replace(
                        geometry, appearance=self._mask_appearance(image, foreground),
                        color_scores=vehicle_color_scores(image, foreground),
                    )

        # 所有检测都保持原始索引；纯三维模式只给定位成功的地标分配ID。
        observed_at = (
            datetime.now(timezone.utc).timestamp()
            if timestamp is None
            else float(timestamp)
        )
        instance_ids = self.mapper.update(
            [detection.class_name for detection in all_detections],
            [detection.confidence for detection in all_detections],
            observations_3d,
            timestamp=observed_at,
        )
        overlay = image.copy()
        displayed_instances: list[
            tuple[Detection, int | None, int | None]
        ] = []
        for detection_index, foreground in prepared_masks:
            detection = all_detections[detection_index]
            runtime_instance_id = instance_ids[detection_index]
            confirmed_instance_id = (
                None
                if runtime_instance_id is None
                else self.mapper.confirmed_instance_id(
                    detection.class_name,
                    runtime_instance_id,
                    self.config.instance_display_min_observations,
                )
            )
            color = (
                semantic_class_color(detection.class_name)
                if confirmed_instance_id is None
                else instance_color(confirmed_instance_id)
            )
            overlay[foreground] = color
            displayed_instances.append((
                detection,
                runtime_instance_id,
                confirmed_instance_id,
            ))
        count = len(displayed_instances)
        rendered = cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0)
        if self.config.show_instance_ids:
            for (
                detection,
                runtime_instance_id,
                confirmed_instance_id,
            ) in displayed_instances:
                if not self._is_semantic_landmark(detection.class_name):
                    continue
                identity_class = instance_class_name(detection.class_name)
                x1, y1, x2, y2 = detection.box
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                color = (
                    semantic_class_color(detection.class_name)
                    if confirmed_instance_id is None
                    else instance_color(confirmed_instance_id)
                )
                if runtime_instance_id is None:
                    label = f"{identity_class} [unmapped]"
                elif confirmed_instance_id is None:
                    label = f"{identity_class} [confirming]"
                else:
                    label = f"{identity_class}_{confirmed_instance_id:02d}"
                cv2.putText(
                    rendered,
                    label,
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        if not prepared_masks:
            cv2.putText(
                rendered,
                "SAM2 returned no mask for YOLO-World boxes",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        return rendered, count

    def _select_sam_detections(
        self, detections: Sequence[Detection]
    ) -> list[tuple[int, Detection]]:
        """柱子、车辆、标牌和消防箱轮流取得预算，其余目标补充。"""
        indexed = list(enumerate(detections))
        indexed.sort(
            key=lambda item: (
                self._is_semantic_landmark(item[1].class_name),
                item[1].confidence,
            ),
            reverse=True,
        )
        columns, vehicles, signs, cabinets, others = [], [], [], [], []
        for item in indexed:
            name = canonical_class_name(item[1].class_name)
            if self._is_semantic_landmark(name) and name == "column":
                columns.append(item)
            elif self._is_semantic_landmark(name) and name in VEHICLE_CLASSES:
                vehicles.append(item)
            elif name == "signboard":
                signs.append(item)
            elif name == "fire cabinet":
                cabinets.append(item)
            else:
                others.append(item)
        prioritized = []
        groups = (columns, vehicles, signs, cabinets)
        for index in range(max(map(len, groups))):
            for group in groups:
                if index < len(group):
                    prioritized.append(group[index])
        return (prioritized + others)[:self.config.max_sam_prompts]

    @staticmethod
    def _mask_appearance(image: np.ndarray, foreground: np.ndarray) -> np.ndarray | None:
        """掩膜内部颜色直方图作为轻量外观特征，不引入额外推理模型。"""
        pixels = image[::4, ::4][foreground[::4, ::4]]
        if len(pixels) < 4:
            return None
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        color = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        brightness = cv2.calcHist([hsv], [2], None, [16], [0, 256]).ravel()
        histogram = np.concatenate((color, brightness)).astype(np.float64)
        return histogram / histogram.sum()

    def _is_semantic_landmark(self, class_name: str) -> bool:
        return canonical_class_name(class_name) in self.mapper.landmark_classes

    def _mask_world_geometry(
        self,
        foreground: np.ndarray,
        depth: np.ndarray | None,
        camera_intrinsics: np.ndarray | None,
        camera_pose_world: np.ndarray | None,
        camera_pose_covariance: np.ndarray | None = None,
    ) -> SemanticObservation3D | None:
        """把稀疏掩膜深度反投影为稳健的 NED 中心和三维包围盒。"""
        if (
            depth is None
            or camera_intrinsics is None
            or camera_pose_world is None
        ):
            return None
        depth_image = np.asarray(depth, dtype=np.float32)
        if depth_image.ndim != 2 or depth_image.size == 0:
            return None
        mask = np.asarray(foreground)
        if mask.shape != depth_image.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth_image.shape[1], depth_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        stride = int(self.config.instance_depth_stride)
        rows, columns = np.nonzero(mask[::stride, ::stride])
        rows = rows * stride
        columns = columns * stride
        if rows.size < self.config.instance_depth_min_points:
            return None
        depth_values = depth_image[rows, columns]
        valid = np.isfinite(depth_values)
        valid &= depth_values >= float(self.config.instance_depth_min)
        valid &= depth_values <= float(self.config.instance_depth_max)
        if np.count_nonzero(valid) < self.config.instance_depth_min_points:
            return None
        rows = rows[valid]
        columns = columns[valid]
        depth_values = depth_values[valid].astype(np.float64)

        # 去掉掩膜边缘混入的前景/背景深度，保留主体深度簇。
        median_depth = float(np.median(depth_values))
        median_deviation = float(np.median(np.abs(depth_values - median_depth)))
        depth_gate = max(0.25, 3.0 * 1.4826 * median_deviation)
        inliers = np.abs(depth_values - median_depth) <= depth_gate
        if np.count_nonzero(inliers) >= self.config.instance_depth_min_points:
            rows = rows[inliers]
            columns = columns[inliers]
            depth_values = depth_values[inliers]
            median_depth = float(np.median(depth_values))

        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        pose = np.asarray(camera_pose_world, dtype=np.float64)
        if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
            return None
        if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(pose)):
            return None
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return None

        point_cloud_camera = np.column_stack((
            (columns.astype(np.float64) - cx) * depth_values / fx,
            (rows.astype(np.float64) - cy) * depth_values / fy,
            depth_values,
        ))
        point_cloud_world = (
            point_cloud_camera @ pose[:3, :3].T + pose[:3, 3]
        )
        if not np.all(np.isfinite(point_cloud_world)):
            return None

        # 百分位边界比单点极值更能抑制掩膜边缘残留的少量离群点。
        position_world = np.median(point_cloud_world, axis=0)
        bbox_3d_min = np.percentile(point_cloud_world, 2.5, axis=0)
        bbox_3d_max = np.percentile(point_cloud_world, 97.5, axis=0)
        # 对像素中心 (u, v) 和光轴深度 d 的噪声做反投影雅可比传播。
        center_u = float(np.median(columns))
        center_v = float(np.median(rows))
        pixel_std = float(self.config.instance_mask_pixel_noise_std)
        depth_std = float(self.config.instance_depth_noise_std) * (
            1.0
            + median_depth / float(self.config.instance_depth_max)
        )
        input_covariance = np.diag((
            pixel_std ** 2,
            pixel_std ** 2,
            depth_std ** 2,
        ))
        projection_jacobian = np.asarray((
            (
                median_depth / fx,
                0.0,
                (center_u - cx) / fx,
            ),
            (
                0.0,
                median_depth / fy,
                (center_v - cy) / fy,
            ),
            (0.0, 0.0, 1.0),
        ))
        covariance_camera = (
            projection_jacobian
            @ input_covariance
            @ projection_jacobian.T
        )
        # 代表像素的反投影点与上方投影雅可比使用同一线性化位置。
        representative_point = np.array([
            (center_u - cx) * median_depth / fx,
            (center_v - cy) * median_depth / fy,
            median_depth,
        ])
        measurement_covariance = world_point_covariance(
            pose[:3, :3], representative_point, covariance_camera,
            camera_pose_covariance,
        )
        return SemanticObservation3D(
            position_world=position_world,
            bbox_3d_min=bbox_3d_min,
            bbox_3d_max=bbox_3d_max,
            measurement_covariance=measurement_covariance,
        )

    def save_semantic_map(self) -> Path | None:
        """原子保存本次飞行已确认的三维语义地标。"""
        if not self.config.save_semantic_map:
            return None
        path = Path(self.config.semantic_map_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.mapper.semantic_map_records(
            min_observations=self.config.semantic_map_min_observations
        )
        payload = {
            "format_version": SEMANTIC_MAP_FORMAT_VERSION,
            "coordinate_frame": "NED",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "instance_count": len(records),
            "instances": records,
        }
        temporary_path = path.with_name(f"{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path
