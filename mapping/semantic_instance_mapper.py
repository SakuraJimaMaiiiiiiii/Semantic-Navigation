"""基于 NED 卡尔曼滤波和马氏距离关联的语义实例地图。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .vehicle_instance_tracker import VEHICLE_CLASSES, VehicleInstanceTracker


_Vector3 = tuple[float, float, float]
_Matrix3 = tuple[_Vector3, _Vector3, _Vector3]
_LandmarkKey = tuple[str, int]


STRUCTURAL_COLUMN_PROMPTS = (
    "parking garage structural column",
    "floor to ceiling concrete column",
    "concrete column supporting a ceiling beam",
)


COLUMN_CLASS_ALIASES = (
    "column",
    "concrete column",
    "concrete pillar",
    "parking garage pillar",
    "support column",
    "support pillar",
    "structural column",
    *STRUCTURAL_COLUMN_PROMPTS,
)

SIGNBOARD_CLASS_ALIASES = ("signboard", "wall mounted sign", "directional sign", "direction sign")
EXIT_SIGN_CLASS_ALIASES = ("exit sign", "exit signage", "emergency exit sign", "illuminated exit sign")
FIRE_CABINET_CLASS_ALIASES = ("fire cabinet", "fire hose cabinet", "fire equipment cabinet", "fire hydrant cabinet")

SEMANTIC_CLASS_PROPERTIES = {
    "exit sign": {"aliases": EXIT_SIGN_CLASS_ALIASES, "is_static": True, "is_obstacle": True},
    "signboard": {"aliases": SIGNBOARD_CLASS_ALIASES, "is_static": True, "is_obstacle": True},
    "fire cabinet": {"aliases": FIRE_CABINET_CLASS_ALIASES, "is_static": True, "is_obstacle": True},
    "column": {
        "aliases": tuple(
            alias for alias in COLUMN_CLASS_ALIASES if alias != "column"
        ),
        "is_static": True,
        "is_obstacle": True,
    },
}


def canonical_semantic_class_name(class_name: str) -> str:
    """将开放词汇别名规范为语义地图类别。"""
    normalized = str(class_name).strip().lower()
    if normalized in EXIT_SIGN_CLASS_ALIASES:
        return "exit sign"
    if normalized in SIGNBOARD_CLASS_ALIASES:
        return "signboard"
    if normalized in FIRE_CABINET_CLASS_ALIASES:
        return "fire cabinet"
    if normalized in COLUMN_CLASS_ALIASES:
        return "column"
    if normalized in ("sedan", "suv", "taxi", "police car", "automobile"):
        return "car"
    if normalized in ("pickup", "pickup truck", "lorry"):
        return "truck"
    if normalized in ("minivan", "delivery van"):
        return "van"
    return normalized


def instance_class_name(class_name: str) -> str:
    """车辆共用ID命名空间，车型标签波动不会改变身份。"""
    canonical = canonical_semantic_class_name(class_name)
    return "vehicle" if canonical in VEHICLE_CLASSES else canonical


@dataclass(frozen=True)
class SemanticObservation3D:
    """单帧掩膜反投影得到的 NED 几何测量及其协方差。"""

    position_world: np.ndarray
    bbox_3d_min: np.ndarray
    bbox_3d_max: np.ndarray
    measurement_covariance: np.ndarray
    appearance: np.ndarray | None = None
    color_scores: dict[str, float] | None = None


@dataclass(frozen=True)
class SemanticMemoryInstance:
    """当前飞行语义地图中可供导航读取的三维地标。"""

    instance_id: int
    class_name: str
    aliases: tuple[str, ...]
    position_world: _Vector3
    position_covariance: _Matrix3
    bbox_3d_min: _Vector3
    bbox_3d_max: _Vector3
    confidence: float
    observation_count: int
    last_seen: float
    is_static: bool
    is_obstacle: bool

    @property
    def label(self) -> str:
        return f"{instance_class_name(self.class_name)}_{self.instance_id:02d}"


@dataclass
class _SemanticLandmark:
    confidence: float
    position_world: np.ndarray
    position_covariance: np.ndarray
    bbox_3d_min: np.ndarray
    bbox_3d_max: np.ndarray
    observation_count: int
    last_seen: float


class SemanticInstanceMapper:
    """维护本次飞行的卡尔曼语义地标并完成马氏距离关联。"""

    def __init__(
        self,
        landmark_classes=("column", "signboard", "exit sign", "fire cabinet", *VEHICLE_CLASSES),
        mahalanobis_gate=7.815,                  #三维卡方分布95%置信度阈值
        process_noise_std=0.05,                  # 卡尔曼预测过程噪声标准差
        duplicate_merge_distance_xy=1.5,
        column_min_vertical_extent=2.2,
        column_min_vertical_aspect_ratio=1.2,
        vehicle_max_association_distance=2.5,
        vehicle_appearance_gate=0.55,
        vehicle_ambiguity_margin=0.12,
        vehicle_viewpoint_noise_std=0.45,
        vehicle_confirmation_observations=3,
    ) -> None:
        self.landmark_classes = {
            canonical_semantic_class_name(name)
            for name in landmark_classes
        }
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.process_noise_std = float(process_noise_std)
        self.duplicate_merge_distance_xy = float(
            duplicate_merge_distance_xy
        )
        self.column_min_vertical_extent = float(
            column_min_vertical_extent
        )
        self.column_min_vertical_aspect_ratio = float(
            column_min_vertical_aspect_ratio
        )
        self._validate_config()
        self._next_instance_id: dict[str, int] = {}
        self._landmarks: dict[_LandmarkKey, _SemanticLandmark] = {}
        self._next_display_instance_id: dict[str, int] = {}
        self._display_instance_ids: dict[_LandmarkKey, int] = {}
        self._last_timestamp = None
        self._vehicles = VehicleInstanceTracker(
            mahalanobis_gate=mahalanobis_gate,
            max_distance=vehicle_max_association_distance,
            appearance_gate=vehicle_appearance_gate,
            ambiguity_margin=vehicle_ambiguity_margin,
            process_noise_std=process_noise_std,
            viewpoint_noise_std=vehicle_viewpoint_noise_std,
            confirmation_observations=vehicle_confirmation_observations,
        )

    def update(
        self, 
        class_names: Sequence[str],
        confidences: Sequence[float],
        observations_3d: Sequence[SemanticObservation3D | None],
        timestamp: float,
    ) -> list[int | None]:
        """用三维马氏距离关联测量，并更新各实例的NED卡尔曼状态。"""
        item_count = len(class_names)
        if len(confidences) != item_count or len(observations_3d) != item_count:
            raise ValueError(
                "class_names, confidences and observations_3d must "
                "have equal lengths"
            )
        observed_at = float(timestamp)
        if not np.isfinite(observed_at):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and observed_at <= self._last_timestamp:
            return [None] * item_count
        self._last_timestamp = observed_at
        canonical_names = [
            canonical_semantic_class_name(name) for name in class_names
        ]
        observations = [
            self._valid_observation(item) for item in observations_3d
        ]
        vehicle_observations = [
            obs if name in VEHICLE_CLASSES and name in self.landmark_classes else None
            for name, obs in zip(canonical_names, observations)
        ]
        vehicle_ids = self._vehicles.update(
            canonical_names, confidences, vehicle_observations, observed_at
        )

        candidates: list[tuple[float, int, _LandmarkKey]] = []
        for observation_index, (class_name, observation) in enumerate(zip(
            canonical_names, observations
        )):
            if (class_name not in self.landmark_classes or observation is None
                    or class_name in VEHICLE_CLASSES):
                continue
            for landmark_key, landmark in self._landmarks.items():
                if landmark_key[0] != class_name:
                    continue
                distance_squared = self._innovation_distance_squared(
                    landmark, observation, observed_at
                )
                if distance_squared <= self.mahalanobis_gate:
                    candidates.append((
                        distance_squared,
                        observation_index,
                        landmark_key,
                    ))

        assignments: dict[int, _LandmarkKey] = {}
        used_landmarks: set[_LandmarkKey] = set()
        for _, observation_index, landmark_key in sorted(candidates):
            if (
                observation_index in assignments
                or landmark_key in used_landmarks
            ):
                continue
            assignments[observation_index] = landmark_key
            used_landmarks.add(landmark_key)

        runtime_ids: list[int | None] = []
        for index, (class_name, confidence, observation) in enumerate(zip(
            canonical_names, confidences, observations
        )):
            if (class_name not in self.landmark_classes or observation is None
                    or class_name in VEHICLE_CLASSES):
                runtime_ids.append(None)
                continue
            landmark_key = assignments.get(index)
            if landmark_key is None:
                instance_id = self._next_instance_id.get(class_name, 1)
                self._next_instance_id[class_name] = instance_id + 1
                landmark_key = (class_name, instance_id)
                position_world = observation.position_world.copy()
                position_covariance = (
                    observation.measurement_covariance.copy()
                )
                bbox_3d_min = observation.bbox_3d_min.copy()
                bbox_3d_max = observation.bbox_3d_max.copy()
                observation_count = 1
                mean_confidence = float(confidence)
            else:
                previous = self._landmarks[landmark_key]
                instance_id = landmark_key[1]
                position_world, position_covariance = self._kalman_update(
                    previous, observation, observed_at
                )
                bbox_3d_min = np.minimum(
                    previous.bbox_3d_min, observation.bbox_3d_min
                )
                bbox_3d_max = np.maximum(
                    previous.bbox_3d_max, observation.bbox_3d_max
                )
                observation_count = previous.observation_count + 1
                mean_confidence = (
                    previous.confidence * previous.observation_count
                    + float(confidence)
                ) / observation_count
            self._landmarks[landmark_key] = _SemanticLandmark(
                confidence=mean_confidence,
                position_world=position_world,
                position_covariance=position_covariance,
                bbox_3d_min=bbox_3d_min,
                bbox_3d_max=bbox_3d_max,
                observation_count=observation_count,
                last_seen=observed_at,
            )
            runtime_ids.append(instance_id)

        redirects = self._merge_duplicate_landmarks_by_ned()
        resolved_ids: list[int | None] = []
        for class_name, instance_id in zip(canonical_names, runtime_ids):
            if instance_id is None:
                resolved_ids.append(None)
                continue
            landmark_key = (class_name, instance_id)
            resolved_ids.append(
                redirects.get(landmark_key, landmark_key)[1]
            )
        return [
            vehicle_id if name in VEHICLE_CLASSES else instance_id
            for name, vehicle_id, instance_id in zip(
                canonical_names, vehicle_ids, resolved_ids
            )
        ]

    def confirmed_instance_id(
        self,
        class_name: str,
        instance_id: int,
        min_observations: int,
    ) -> int | None:
        """返回确认后的稳定视频ID，未确认时返回None。"""
        canonical = canonical_semantic_class_name(class_name)
        if canonical in VEHICLE_CLASSES:
            track = self._vehicles.tracks.get(int(instance_id))
            if track is None or track.observation_count < int(min_observations):
                return None
            # 车辆已分配的ID不进行显示重编号，视频与地图保持同一身份。
            self._display_instance_ids[("vehicle", int(instance_id))] = int(instance_id)
            return int(instance_id)
        landmark_key = (canonical, int(instance_id))
        landmark = self._landmarks.get(landmark_key)
        if landmark is None:
            return None
        if landmark.observation_count < int(min_observations):
            return None
        if not self._has_valid_landmark_geometry(
            canonical,
            landmark.bbox_3d_min,
            landmark.bbox_3d_max,
        ):
            return None
        display_instance_id = self._display_instance_ids.get(landmark_key)
        if display_instance_id is None:
            display_instance_id = self._next_display_instance_id.get(
                canonical, 1
            )
            self._next_display_instance_id[canonical] = (
                display_instance_id + 1
            )
            self._display_instance_ids[landmark_key] = display_instance_id
        return display_instance_id

    def semantic_map_records(self, min_observations=1) -> list[dict]:
        """导出过滤后的实例；柱子连续编号，车辆保留原始身份。"""
        records = []
        map_instance_counts: dict[str, int] = {}
        for memory in self.snapshot():
            if memory.observation_count < int(min_observations):
                continue
            if not self._has_valid_landmark_geometry(
                memory.class_name,
                memory.bbox_3d_min,
                memory.bbox_3d_max,
            ):
                continue
            identity_class = instance_class_name(memory.class_name)
            if identity_class == "vehicle":
                map_instance_id = memory.instance_id
            else:
                map_instance_id = map_instance_counts.get(identity_class, 0) + 1
                map_instance_counts[identity_class] = map_instance_id
            runtime_key = (identity_class, memory.instance_id)
            video_instance_id = self._display_instance_ids.get(runtime_key)
            records.append({
                "instance_id": (
                    f"{identity_class}_{map_instance_id:02d}"
                ),
                "runtime_instance_id": memory.label,
                "video_instance_id": (
                    None
                    if video_instance_id is None
                    else f"{identity_class}_{video_instance_id:02d}"
                ),
                "class_name": memory.class_name,
                "color": self.vehicle_color(memory.instance_id)[0] if identity_class == "vehicle" else "unknown",
                "color_confidence": self.vehicle_color(memory.instance_id)[1] if identity_class == "vehicle" else 0.0,
                "aliases": list(memory.aliases),
                "position_world": list(memory.position_world),
                "position_covariance": [
                    list(row) for row in memory.position_covariance
                ],
                "bbox_3d": {
                    "min": list(memory.bbox_3d_min),
                    "max": list(memory.bbox_3d_max),
                },
                "confidence": memory.confidence,
                "observation_count": memory.observation_count,
                "last_seen": memory.last_seen,
                "is_static": memory.is_static,
                "is_obstacle": memory.is_obstacle,
            })
        return records

    def vehicle_color(self, instance_id):
        track = self._vehicles.tracks.get(instance_id)
        return ("unknown", 0.0) if track is None else track.color_semantics

    def navigation_records(self, min_observations=3):
        """Stable online identities; export numbering remains backward compatible."""
        records = self.semantic_map_records(min_observations)
        for record in records:
            record["instance_id"] = record["runtime_instance_id"]
        return records

    def snapshot(self) -> tuple[SemanticMemoryInstance, ...]:
        """返回当前飞行的全部内部三维语义地标。"""
        snapshots = []
        for landmark_key, landmark in sorted(self._landmarks.items()):
            class_name, instance_id = landmark_key
            properties = SEMANTIC_CLASS_PROPERTIES.get(
                class_name,
                {"aliases": (), "is_static": False, "is_obstacle": False},
            )
            covariance = tuple(
                tuple(float(value) for value in row)
                for row in landmark.position_covariance
            )
            snapshots.append(SemanticMemoryInstance(
                instance_id=instance_id,
                class_name=class_name,
                aliases=tuple(properties["aliases"]),
                position_world=tuple(
                    float(value) for value in landmark.position_world
                ),
                position_covariance=covariance,
                bbox_3d_min=tuple(
                    float(value) for value in landmark.bbox_3d_min
                ),
                bbox_3d_max=tuple(
                    float(value) for value in landmark.bbox_3d_max
                ),
                confidence=landmark.confidence,
                observation_count=landmark.observation_count,
                last_seen=landmark.last_seen,
                is_static=bool(properties["is_static"]),
                is_obstacle=bool(properties["is_obstacle"]),
            ))
        for track in self._vehicles.tracks.values():
            snapshots.append(SemanticMemoryInstance(
                instance_id=track.instance_id,
                class_name=track.class_name,
                aliases=(),
                position_world=tuple(float(value) for value in track.state[:3]),
                position_covariance=tuple(
                    tuple(float(value) for value in row)
                    for row in track.covariance[:3, :3]
                ),
                bbox_3d_min=tuple(float(value) for value in track.bbox_3d_min),
                bbox_3d_max=tuple(float(value) for value in track.bbox_3d_max),
                confidence=track.confidence,
                observation_count=track.observation_count,
                last_seen=track.last_seen,
                is_static=True,
                is_obstacle=True,
            ))
        return tuple(snapshots)

    def _innovation_distance_squared(
        self,
        landmark: _SemanticLandmark,
        observation: SemanticObservation3D,
        timestamp: float,
    ) -> float:
        predicted_covariance = self._predicted_covariance(
            landmark, timestamp
        )
        innovation = observation.position_world - landmark.position_world
        innovation_covariance = (
            predicted_covariance + observation.measurement_covariance
        )
        inverse_covariance = np.linalg.pinv(
            innovation_covariance, hermitian=True
        )
        return max(0.0, float(
            innovation.T @ inverse_covariance @ innovation
        ))

    def _kalman_update(
        self,
        landmark: _SemanticLandmark,
        observation: SemanticObservation3D,
        timestamp: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted_position = landmark.position_world
        predicted_covariance = self._predicted_covariance(
            landmark, timestamp
        )
        measurement_covariance = observation.measurement_covariance
        innovation = observation.position_world - predicted_position
        innovation_covariance = (
            predicted_covariance + measurement_covariance
        )
        kalman_gain = predicted_covariance @ np.linalg.pinv(
            innovation_covariance, hermitian=True
        )
        updated_position = predicted_position + kalman_gain @ innovation
        identity = np.eye(3, dtype=np.float64)
        residual_transform = identity - kalman_gain
        # Joseph 形式能在浮点误差下继续保持协方差对称、半正定。
        updated_covariance = (
            residual_transform
            @ predicted_covariance
            @ residual_transform.T
            + kalman_gain
            @ measurement_covariance
            @ kalman_gain.T
        )
        updated_covariance = self._positive_semidefinite(
            updated_covariance
        )
        return updated_position, updated_covariance

    def _predicted_covariance(
        self,
        landmark: _SemanticLandmark,
        timestamp: float,
    ) -> np.ndarray:
        elapsed = max(0.0, min(float(timestamp) - landmark.last_seen, 5.0))
        process_variance = self.process_noise_std ** 2 * elapsed
        return landmark.position_covariance + np.eye(3) * process_variance

    def _merge_duplicate_landmarks_by_ned(
        self,
    ) -> dict[_LandmarkKey, _LandmarkKey]:
        """按 N/E 水平坐标合并残留重复地标，保留较小运行时ID。"""
        redirects: dict[_LandmarkKey, _LandmarkKey] = {}
        while True:
            duplicate_pair = None
            landmark_keys = sorted(self._landmarks)
            for first_index, first_key in enumerate(landmark_keys):
                if first_key[0] != "column":
                    continue
                first = self._landmarks[first_key]
                for second_key in landmark_keys[first_index + 1:]:
                    second = self._landmarks[second_key]
                    if first_key[0] != second_key[0]:
                        continue
                    distance_xy = float(np.linalg.norm(
                        first.position_world[:2]
                        - second.position_world[:2]
                    ))
                    if distance_xy <= self.duplicate_merge_distance_xy:
                        duplicate_pair = (first_key, second_key)
                        break
                if duplicate_pair is not None:
                    break
            if duplicate_pair is None:
                return redirects

            keep_key, drop_key = sorted(
                duplicate_pair, key=lambda item: item[1]
            )
            keep = self._landmarks[keep_key]
            drop = self._landmarks[drop_key]
            keep_weight = keep.observation_count
            drop_weight = drop.observation_count
            total_observations = keep_weight + drop_weight
            merged_position = (
                keep.position_world * keep_weight
                + drop.position_world * drop_weight
            ) / total_observations
            keep_offset = keep.position_world - merged_position
            drop_offset = drop.position_world - merged_position
            merged_covariance = (
                keep_weight
                * (
                    keep.position_covariance
                    + np.outer(keep_offset, keep_offset)
                )
                + drop_weight
                * (
                    drop.position_covariance
                    + np.outer(drop_offset, drop_offset)
                )
            ) / total_observations
            self._landmarks[keep_key] = _SemanticLandmark(
                confidence=(
                    keep.confidence * keep_weight
                    + drop.confidence * drop_weight
                ) / total_observations,
                position_world=merged_position,
                position_covariance=self._positive_semidefinite(
                    merged_covariance
                ),
                bbox_3d_min=np.minimum(
                    keep.bbox_3d_min, drop.bbox_3d_min
                ),
                bbox_3d_max=np.maximum(
                    keep.bbox_3d_max, drop.bbox_3d_max
                ),
                observation_count=total_observations,
                last_seen=max(keep.last_seen, drop.last_seen),
            )
            keep_display_id = self._display_instance_ids.get(keep_key)
            drop_display_id = self._display_instance_ids.pop(drop_key, None)
            if drop_display_id is not None:
                self._display_instance_ids[keep_key] = (
                    drop_display_id
                    if keep_display_id is None
                    else min(keep_display_id, drop_display_id)
                )
            del self._landmarks[drop_key]
            redirects[drop_key] = keep_key
            for source_key, target_key in tuple(redirects.items()):
                if target_key == drop_key:
                    redirects[source_key] = keep_key

    def _valid_observation(
        self,
        observation: SemanticObservation3D | None,
    ) -> SemanticObservation3D | None:
        if observation is None:
            return None
        position = np.asarray(
            observation.position_world, dtype=np.float64
        ).reshape(-1)
        bbox_min = np.asarray(
            observation.bbox_3d_min, dtype=np.float64
        ).reshape(-1)
        bbox_max = np.asarray(
            observation.bbox_3d_max, dtype=np.float64
        ).reshape(-1)
        if any(item.shape != (3,) for item in (position, bbox_min, bbox_max)):
            return None
        if not all(
            np.all(np.isfinite(item))
            for item in (position, bbox_min, bbox_max)
        ):
            return None
        if np.any(bbox_min > bbox_max):
            return None
        covariance = observation.measurement_covariance
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            return None
        appearance = observation.appearance
        if appearance is not None:
            appearance = np.asarray(appearance, dtype=np.float64)
            if (appearance.ndim != 1 or appearance.size == 0
                    or not np.all(np.isfinite(appearance))
                    or np.any(appearance < 0) or appearance.sum() <= 0):
                appearance = None
            else:
                appearance = appearance / appearance.sum()
        return SemanticObservation3D(
            position_world=position.copy(),
            bbox_3d_min=bbox_min.copy(),
            bbox_3d_max=bbox_max.copy(),
            measurement_covariance=self._positive_semidefinite(covariance),
            appearance=appearance,
            color_scores={str(k): float(v) for k, v in (observation.color_scores or {}).items() if np.isfinite(v) and v > 0},
        )

    def _has_valid_landmark_geometry(
        self,
        class_name: str,
        bbox_3d_min,
        bbox_3d_max,
    ) -> bool:
        if class_name != "column":
            return True
        extents = np.asarray(bbox_3d_max) - np.asarray(bbox_3d_min)
        vertical_extent = float(extents[2])
        horizontal_extent = float(max(extents[0], extents[1]))
        return (
            vertical_extent >= self.column_min_vertical_extent
            and vertical_extent
            >= self.column_min_vertical_aspect_ratio
            * max(horizontal_extent, 1e-6)
        )

    @staticmethod
    def _positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, 1e-9)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _validate_config(self) -> None:
        if not self.landmark_classes:
            raise ValueError("landmark_classes must not be empty")
        if self.mahalanobis_gate <= 0:
            raise ValueError("mahalanobis_gate must be greater than zero")
        if self.process_noise_std < 0:
            raise ValueError("process_noise_std must not be negative")
        if self.duplicate_merge_distance_xy <= 0:
            raise ValueError(
                "duplicate_merge_distance_xy must be greater than zero"
            )
        if self.column_min_vertical_extent <= 0:
            raise ValueError("column_min_vertical_extent must be positive")
        if self.column_min_vertical_aspect_ratio <= 0:
            raise ValueError(
                "column_min_vertical_aspect_ratio must be positive"
            )
