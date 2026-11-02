"""基于 NED 卡尔曼滤波和马氏距离关联的语义实例地图。"""

from itertools import combinations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .vehicle_instance_tracker import VEHICLE_CLASSES, VehicleInstanceTracker
from .instance_surface import InstanceSurface


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

# Detection prompts describe vehicle traffic signs, not generic wall signs,
# advertisements, room labels or pedestrian emergency-exit signage.
TRAFFIC_SIGN_PROMPTS = (
    "parking garage vehicle direction sign",
    "speed limit traffic sign",
    "no entry traffic sign",
)
# Keep the stored canonical category/IDs stable; do not prompt YOLO with it.
SIGNBOARD_CLASS_ALIASES = ("signboard", "traffic sign", "road traffic sign", *TRAFFIC_SIGN_PROMPTS)
EXIT_SIGN_PROMPTS = ("illuminated emergency exit sign", "emergency exit pictogram sign")
EXIT_SIGN_CLASS_ALIASES = ("exit sign", "exit signage", "emergency exit sign", "illuminated exit sign", *EXIT_SIGN_PROMPTS)
EXIT_DOOR_CLASS_ALIASES = ("exit door", "emergency exit door", "fire exit door", "emergency escape door")
FIRE_CABINET_CLASS_ALIASES = ("fire cabinet", "fire hose cabinet", "fire equipment cabinet", "fire hydrant cabinet")

FIRE_EXTINGUISHER_ALIASES = ("fire extinguisher", "portable fire extinguisher", "wall mounted fire extinguisher")

SEMANTIC_CLASS_PROPERTIES = {
    "fire extinguisher": {"aliases": FIRE_EXTINGUISHER_ALIASES, "is_static": True, "is_obstacle": True},
    "exit door": {"aliases": EXIT_DOOR_CLASS_ALIASES, "is_static": True, "is_obstacle": True},
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
    if normalized in EXIT_DOOR_CLASS_ALIASES:
        return "exit door"
    if normalized in EXIT_SIGN_CLASS_ALIASES:
        return "exit sign"
    if normalized in SIGNBOARD_CLASS_ALIASES:
        return "signboard"
    if normalized in FIRE_EXTINGUISHER_ALIASES:
        return "fire extinguisher"
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
    surface_points_world: np.ndarray | None = None


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
    surface: InstanceSurface = field(default_factory=InstanceSurface)


class SemanticInstanceMapper:
    """维护本次飞行的卡尔曼语义地标并完成马氏距离关联。"""

    def __init__(
        self,
        landmark_classes=("column", "signboard", "exit sign", "exit door", "fire cabinet", "fire extinguisher", *VEHICLE_CLASSES),
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
        sign_min_confidence=0.25,
        sign_min_observations=1,
        vehicle_map_min_confidence=0.25,
        safety_landmark_min_confidence=0.30,
        safety_landmark_min_observations=8,
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
        self.vehicle_map_min_confidence = float(vehicle_map_min_confidence)
        if not 0 < self.vehicle_map_min_confidence <= 1:
            raise ValueError("vehicle_map_min_confidence must be in (0, 1]")
        self.sign_min_observations = int(sign_min_observations)
        if self.sign_min_observations < 1:
            raise ValueError("sign_min_observations must be positive")
        self.sign_min_confidence = float(sign_min_confidence)
        self.safety_landmark_min_confidence = float(safety_landmark_min_confidence)
        self.safety_landmark_min_observations = int(safety_landmark_min_observations)
        if not 0 < self.safety_landmark_min_confidence <= 1 or self.safety_landmark_min_observations < 1:
            raise ValueError('Invalid safety landmark confirmation thresholds')
        if not 0 < self.sign_min_confidence <= 1:
            raise ValueError("sign_min_confidence must be in (0, 1]")
        self._validate_config()
        self._next_instance_id: dict[str, int] = {}
        self._landmarks: dict[_LandmarkKey, _SemanticLandmark] = {}
        self._displayed_instances: set[_LandmarkKey] = set()
        self._last_timestamp = None
        self.semantic_voxels = None
        self._distinct_landmark_pairs = set()
        self._landmark_redirects: dict[_LandmarkKey, _LandmarkKey] = {}
        self._category_blocked = set()
        self._category_aliases = {}
        self._category_evidence = {}
        self._vehicles = VehicleInstanceTracker(
            mahalanobis_gate=mahalanobis_gate,
            max_distance=vehicle_max_association_distance,
            appearance_gate=vehicle_appearance_gate,
            ambiguity_margin=vehicle_ambiguity_margin,
            process_noise_std=process_noise_std,
            viewpoint_noise_std=vehicle_viewpoint_noise_std,
            confirmation_observations=vehicle_confirmation_observations,
        )

    @property
    def vehicle_merge_diagnostics(self):
        return self._vehicles.merge_diagnostics

    def update(
        self, 
        class_names: Sequence[str],
        confidences: Sequence[float],
        observations_3d: Sequence[SemanticObservation3D | None],
        timestamp: float,
    ) -> list[int | None]:
        """用三维马氏距离关联测量，并更新各实例的NED卡尔曼状态。"""
        self.last_observation_batch = None
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
            (None if name in {"signboard", "exit sign"}
             and (not np.isfinite(score) or score < self.sign_min_confidence)
             else self._valid_observation(item))
            for name, score, item in zip(canonical_names, confidences, observations_3d)
        ]
        for index, name in enumerate(canonical_names):
            if name in {'exit door', 'fire cabinet', 'fire extinguisher'} and (
                    not np.isfinite(confidences[index])
                    or confidences[index] < self.safety_landmark_min_confidence):
                observations[index] = None
        # Suppress near-identical sign geometry before assigning IDs. Otherwise
        # duplicate proposals become permanently distinct co-observed landmarks.
        accepted_signs = []
        for index in sorted(range(item_count), key=lambda i: confidences[i], reverse=True):
            obs = observations[index]
            if obs is None or canonical_names[index] not in {"signboard", "exit sign"}:
                continue
            duplicate = False
            for other in accepted_signs:
                if canonical_names[index] != canonical_names[other]:
                    continue
                previous = observations[other]
                if np.linalg.norm(obs.position_world-previous.position_world) > .1:
                    continue
                overlap = np.minimum(obs.bbox_3d_max, previous.bbox_3d_max) - np.maximum(obs.bbox_3d_min, previous.bbox_3d_min)
                larger = np.maximum(obs.bbox_3d_max-obs.bbox_3d_min, previous.bbox_3d_max-previous.bbox_3d_min)
                if np.all(overlap >= .8 * larger):
                    duplicate = True
                    break
            if duplicate:
                observations[index] = None
            else:
                accepted_signs.append(index)
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
                surface_match = landmark.surface.supports(observation.surface_points_world)
                if not surface_match and not self._association_geometry_valid(class_name, landmark, observation):
                    continue
                distance_squared = self._innovation_distance_squared(
                    landmark, observation, observed_at
                )
                if surface_match or distance_squared <= self.mahalanobis_gate:
                    candidates.append((
                        min(distance_squared, 1.) if surface_match else distance_squared,
                        observation_index,
                        landmark_key,
                    ))

        assignments: dict[int, _LandmarkKey] = {}
        used_landmarks: set[_LandmarkKey] = set()
        for chosen_cost, observation_index, landmark_key in sorted(candidates):
            if (
                observation_index in assignments
                or landmark_key in used_landmarks
            ):
                continue
            # An ambiguous observation must not pull a landmark between two
            # objects, or create another identity beside an existing one.
            if any(cost <= chosen_cost * 1.25 + 1e-9
                   and ((index == observation_index and key != landmark_key)
                        or (key == landmark_key and index != observation_index))
                   for cost, index, key in candidates):
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
                if any(item[1] == index for item in candidates):
                    runtime_ids.append(None)
                    continue
                instance_id = self._next_instance_id.get(class_name, 1)
                self._next_instance_id[class_name] = instance_id + 1
                landmark_key = (class_name, instance_id)
                position_world = observation.position_world.copy()
                position_covariance = (
                    self._measurement_covariance(observation)
                )
                bbox_3d_min = observation.bbox_3d_min.copy()
                bbox_3d_max = observation.bbox_3d_max.copy()
                observation_count = 1
                mean_confidence = float(confidence)
                surface = InstanceSurface()
            else:
                previous = self._landmarks[landmark_key]
                surface = previous.surface
                instance_id = landmark_key[1]
                position_world, position_covariance = self._kalman_update(
                    previous, observation, observed_at
                )
                # Fuse object-relative extents rather than accumulating every
                # drifting world-space box into an ever-growing obstacle.
                bbox_3d_min = position_world + (
                    .8 * (previous.bbox_3d_min - previous.position_world)
                    + .2 * (observation.bbox_3d_min - observation.position_world)
                )
                bbox_3d_max = position_world + (
                    .8 * (previous.bbox_3d_max - previous.position_world)
                    + .2 * (observation.bbox_3d_max - observation.position_world)
                )
                if class_name == "column":
                    # Different views can observe upper/lower parts of a pillar.
                    # Retain actual vertical evidence for the structural filter.
                    bbox_3d_min[2] = min(previous.bbox_3d_min[2], observation.bbox_3d_min[2])
                    bbox_3d_max[2] = max(previous.bbox_3d_max[2], observation.bbox_3d_max[2])
                observation_count = previous.observation_count + 1
                mean_confidence = (
                    previous.confidence * previous.observation_count
                    + float(confidence)
                ) / observation_count
            surface.update(observation.surface_points_world)
            bounds = surface.bounds()
            if bounds is not None:
                bbox_3d_min, bbox_3d_max = bounds
            self._landmarks[landmark_key] = _SemanticLandmark(
                confidence=mean_confidence,
                position_world=position_world,
                position_covariance=position_covariance,
                bbox_3d_min=bbox_3d_min,
                bbox_3d_max=bbox_3d_max,
                observation_count=observation_count,
                last_seen=observed_at,
                surface=surface,
            )
            runtime_ids.append(instance_id)

        seen_keys = [(name, identity) for name, identity in zip(canonical_names, runtime_ids)
                     if identity is not None]
        self._distinct_landmark_pairs.update(frozenset(pair) for pair in combinations(seen_keys, 2))
        self._merge_duplicate_landmarks_by_ned()
        self._resolve_door_cabinet_conflicts()
        resolved_ids: list[int | None] = []
        for class_name, instance_id in zip(canonical_names, runtime_ids):
            if instance_id is None:
                resolved_ids.append(None)
                continue
            landmark_key = (class_name, instance_id)
            resolved_ids.append(
                self._canonical_landmark_key(landmark_key)[1]
            )
        identities = [
            vehicle_id if name in VEHICLE_CLASSES else instance_id
            for name, vehicle_id, instance_id in zip(
                canonical_names, vehicle_ids, resolved_ids
            )
        ]
        if self.semantic_voxels is not None:
            self.semantic_voxels.update([
                (f"{instance_class_name(name)}_{identity:02d}", observation.surface_points_world)
                for name, identity, observation in zip(canonical_names, identities, observations)
                if identity is not None and observation is not None
            ], observed_at)
        self.last_observation_batch = (canonical_names, list(confidences), observations, observed_at, identities)
        return [None if (name, identity) in self._category_blocked else identity
                for name, identity in zip(canonical_names, identities)]

    def _resolve_door_cabinet_conflicts(self):
        """Publish one supported category for coincident door/cabinet proposals.

        Keep both hypotheses internally so later observations can correct the
        category. Do not redirect numeric IDs across class namespaces.
        """
        pairs = []
        for a, first in self._landmarks.items():
            if a[0] != 'exit door':
                continue
            for b, second in self._landmarks.items():
                if b[0] != 'fire cabinet':
                    continue
                centers = [(x.bbox_3d_min+x.bbox_3d_max)*.5 for x in (first, second)]
                if np.linalg.norm(centers[0]-centers[1]) > .35:
                    continue
                # Thin wall surfaces need a voxel-sized depth tolerance, but
                # retain measured widths/heights to avoid swallowing a cabinet
                # mounted on a much larger door or a neighboring cabinet.
                sizes = [np.maximum(x.bbox_3d_max-x.bbox_3d_min, InstanceSurface.resolution)
                         for x in (first, second)]
                overlap = np.maximum(0., np.minimum(centers[0]+sizes[0]/2, centers[1]+sizes[1]/2)
                                     - np.maximum(centers[0]-sizes[0]/2, centers[1]-sizes[1]/2))
                volumes = [float(np.prod(size)) for size in sizes]
                intersection = float(np.prod(overlap))
                if intersection/min(volumes) < .8 or intersection/max(volumes) < .45:
                    continue
                pairs.append((a, b))
        self._category_blocked = set()
        self._category_aliases = {}
        active = set(pairs)
        self._category_evidence = {p: v for p, v in self._category_evidence.items() if p in active}
        for pair in pairs:
            self._category_blocked.update(pair)
            if any(other != pair and set(pair) & set(other) for other in pairs):
                continue  # Multiple competing objects: do not guess a merge.
            scores = {key: self._landmarks[key].confidence*self._landmarks[key].observation_count
                      for key in pair}
            winner = max(scores, key=scores.get)
            loser = next(key for key in pair if key != winner)
            if (self._landmarks[winner].observation_count < 3
                    or scores[winner] < 1.25*scores[loser]):
                self._category_evidence.pop(pair, None)
                continue
            stamp = max(self._landmarks[key].last_seen for key in pair)
            old_winner, count, old_stamp = self._category_evidence.get(pair, (None, 0, None))
            count = (count if old_winner == winner else 0) + int(stamp != old_stamp)
            self._category_evidence[pair] = (winner, count, stamp)
            if count >= 3:
                self._category_blocked.remove(winner)
                self._category_aliases[winner] = loser

    def confirmed_instance_id(
        self,
        class_name: str,
        instance_id: int,
        min_observations: int,
    ) -> int | None:
        """返回确认后的稳定视频ID，未确认时返回None。"""
        canonical = canonical_semantic_class_name(class_name)
        if canonical in VEHICLE_CLASSES:
            instance_id = self._vehicles.canonical_id(int(instance_id))
            track = self._vehicles.tracks.get(instance_id)
            if (track is None or track.observation_count < int(min_observations)
                    or track.confidence < self.vehicle_map_min_confidence):
                return None
            # 车辆已分配的ID不进行显示重编号，视频与地图保持同一身份。
            self._displayed_instances.add(("vehicle", instance_id))
            return instance_id
        landmark_key = self._canonical_landmark_key((canonical, int(instance_id)))
        if landmark_key in self._category_blocked:
            return None
        landmark = self._landmarks.get(landmark_key)
        if landmark is None:
            return None
        required = max(int(min_observations), self.sign_min_observations) if canonical in {"signboard", "exit sign"} else int(min_observations)
        if canonical in {'exit door', 'fire cabinet', 'fire extinguisher'}:
            required = max(required, self.safety_landmark_min_observations)
        if landmark.observation_count < required:
            return None
        if not self._has_valid_landmark_geometry(
            canonical,
            landmark.bbox_3d_min,
            landmark.bbox_3d_max,
        ):
            return None
        self._displayed_instances.add(landmark_key)
        return landmark_key[1]

    def semantic_map_records(self, min_observations=1) -> list[dict]:
        """导出统一身份；过滤或合并不会给剩余物体重新编号。"""
        records = []
        for memory in self.snapshot():
            if memory.class_name in VEHICLE_CLASSES and memory.confidence < self.vehicle_map_min_confidence:
                continue
            required = (max(int(min_observations), self.sign_min_observations)
                        if memory.class_name in {"signboard", "exit sign"} else int(min_observations))
            if memory.class_name in {'exit door', 'fire cabinet', 'fire extinguisher'}:
                required = max(required, self.safety_landmark_min_observations)
            if memory.observation_count < required:
                continue
            if not self._has_valid_landmark_geometry(
                memory.class_name,
                memory.bbox_3d_min,
                memory.bbox_3d_max,
            ):
                continue
            identity_class = instance_class_name(memory.class_name)
            runtime_key = (identity_class, memory.instance_id)
            color, color_confidence = (
                self.vehicle_color(memory.instance_id) if identity_class == "vehicle"
                else ("unknown", 0.0)
            )
            records.append({
                "instance_id": memory.label,
                "runtime_instance_id": memory.label,
                "video_instance_id": memory.label if runtime_key in self._displayed_instances else None,
                "class_name": memory.class_name,
                "merged_from": ([f"vehicle_{old:02d}" for old in sorted(self._vehicles.redirects)
                                 if self._vehicles.canonical_id(old) == memory.instance_id]
                                if identity_class == "vehicle" else [
                                    f"{old[0]}_{old[1]:02d}" for old in sorted(self._landmark_redirects)
                                    if self._canonical_landmark_key(old) == runtime_key
                                ]),
                "color": color,
                "color_confidence": color_confidence,
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
            if identity_class == "vehicle":
                track = self._vehicles.tracks[memory.instance_id]
                records[-1].update(
                    **track.surface.vehicle_geometry(),
                    geometry_representation="observed_surface_union",
                    anchor_position_world=track.state.tolist(),
                    supported_surface_voxels=len(track.surface.supported_points()),
                    connected_fragment_count=track.connected_fragment_count,
                )
            elif runtime_key in self._category_aliases:
                other = self._category_aliases[runtime_key]
                records[-1]['merged_from'].append(f'{other[0]}_{other[1]:02d}')
                records[-1]['classification_status'] = 'cross_category_resolved'
                records[-1]['category_evidence'] = {
                    key[0]: dict(observations=self._landmarks[key].observation_count,
                                 mean_confidence=self._landmarks[key].confidence)
                    for key in (runtime_key, other)}
        return records

    def vehicle_color(self, instance_id):
        track = self._vehicles.tracks.get(instance_id)
        return ("unknown", 0.0) if track is None else track.color_semantics

    def object_inspection_records(self, published_records):
        """Confirmed geometric hypotheses for re-observation, NOT map objects.

        Low-confidence tracks can request a better view without lowering the
        publication threshold or labelling occupancy voxels as a vehicle.
        """
        records = []
        connections = self._vehicles.connection_hypotheses()
        for identity, track in self._vehicles.tracks.items():
            if track.observation_count < self._vehicles.confirmation_observations:
                continue
            records.append({
                "runtime_instance_id": f"vehicle_{identity:02d}",
                "class_name": track.class_name,
                "position_world": track.map_position.tolist(),
                "bbox_3d": {"min": track.bbox_3d_min.tolist(), "max": track.bbox_3d_max.tolist()},
                "last_seen": track.last_seen, "observation_count": track.observation_count,
                "confidence": track.confidence,
                "connection_targets": connections.get(identity, []),
                "connected_fragment_count": getattr(track, "connected_fragment_count", 0),
                "merged_from": [f"vehicle_{old:02d}" for old in self._vehicles.redirects
                                if self._vehicles.canonical_id(old) == identity],
            })
        records.extend(r for r in published_records
                       if r["class_name"] in {"fire extinguisher", "fire cabinet", "exit door", "exit sign", "signboard"})
        return records

    def navigation_records(self, min_observations=3):
        """Navigation, video, JSON and HTML share the same canonical IDs."""
        return self.semantic_map_records(min_observations)

    def snapshot(self) -> tuple[SemanticMemoryInstance, ...]:
        """返回当前飞行的全部内部三维语义地标。"""
        snapshots = []
        for landmark_key, landmark in sorted(self._landmarks.items()):
            if landmark_key in self._category_blocked:
                continue
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
                    float(value) for value in (
                        (landmark.bbox_3d_min+landmark.bbox_3d_max)*.5
                        if landmark.surface.bounds() is not None else landmark.position_world)
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
                position_world=tuple(float(value) for value in track.map_position),
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

    def _canonical_landmark_key(self, key):
        while key in self._landmark_redirects:
            key = self._landmark_redirects[key]
        return key

    def _association_geometry_valid(self, class_name, landmark, observation):
        delta = observation.position_world - landmark.position_world
        if class_name == "column":
            if np.linalg.norm(delta[:2]) > self.duplicate_merge_distance_xy:
                return False
            # A different visible height on a continuous pillar is legitimate.
            return (min(landmark.bbox_3d_max[2], observation.bbox_3d_max[2])
                    >= max(landmark.bbox_3d_min[2], observation.bbox_3d_min[2]))
        distance = np.linalg.norm(delta)
        if distance <= min(.5, self.duplicate_merge_distance_xy):
            return True
        if distance > self.duplicate_merge_distance_xy:
            return False
        # Larger visible-anchor changes need substantial common 3-D support.
        overlap = np.minimum(landmark.bbox_3d_max, observation.bbox_3d_max) - np.maximum(
            landmark.bbox_3d_min, observation.bbox_3d_min)
        smaller = np.minimum(landmark.bbox_3d_max-landmark.bbox_3d_min,
                             observation.bbox_3d_max-observation.bbox_3d_min)
        return bool(np.all(overlap >= 0) and np.all(overlap >= .5*smaller))

    @staticmethod
    def _measurement_covariance(observation):
        # Pixel/depth precision does not measure the uncertainty of a visible
        # surface anchor: it changes with viewpoint and partial occlusion.
        extent = observation.bbox_3d_max - observation.bbox_3d_min
        return observation.measurement_covariance + np.diag(extent * extent / 12.)

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
            predicted_covariance + self._measurement_covariance(observation)
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
        measurement_covariance = self._measurement_covariance(observation)
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
    ) -> None:
        """合并有共同三维支持且未同时观测为不同物体的残留地标。"""
        while True:
            duplicate_pair = None
            landmark_keys = sorted(self._landmarks)
            for first_index, first_key in enumerate(landmark_keys):
                if first_key[0] not in {"column", "signboard", "exit sign", "exit door", "fire cabinet", "fire extinguisher"}:
                    continue
                first = self._landmarks[first_key]
                for second_key in landmark_keys[first_index + 1:]:
                    second = self._landmarks[second_key]
                    if first_key[0] != second_key[0]:
                        continue
                    if frozenset((first_key, second_key)) in self._distinct_landmark_pairs:
                        continue
                    # Nearby columns are distinct objects too. A duplicate must
                    # have overlapping measured support, not just nearby XY.
                    if first_key[0] != "column" and np.linalg.norm(
                            first.position_world-second.position_world) > min(.5, self.duplicate_merge_distance_xy):
                        continue
                    overlap = np.minimum(first.bbox_3d_max, second.bbox_3d_max) - np.maximum(first.bbox_3d_min, second.bbox_3d_min)
                    smaller = np.minimum(first.bbox_3d_max-first.bbox_3d_min,
                                         second.bbox_3d_max-second.bbox_3d_min)
                    if np.any(overlap < 0) or np.any(overlap < .5*smaller):
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
                return

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
            bbox_min = merged_position + (
                keep_weight * (keep.bbox_3d_min - keep.position_world)
                + drop_weight * (drop.bbox_3d_min - drop.position_world)
            ) / total_observations
            bbox_max = merged_position + (
                keep_weight * (keep.bbox_3d_max - keep.position_world)
                + drop_weight * (drop.bbox_3d_max - drop.position_world)
            ) / total_observations
            if keep_key[0] == "column":
                bbox_min[2] = min(keep.bbox_3d_min[2], drop.bbox_3d_min[2])
                bbox_max[2] = max(keep.bbox_3d_max[2], drop.bbox_3d_max[2])
            keep.surface.merge(drop.surface)
            bounds = keep.surface.bounds()
            if bounds is not None:
                bbox_min, bbox_max = bounds
            self._landmarks[keep_key] = _SemanticLandmark(
                confidence=(
                    keep.confidence * keep_weight
                    + drop.confidence * drop_weight
                ) / total_observations,
                position_world=merged_position,
                position_covariance=self._positive_semidefinite(
                    merged_covariance
                ),
                bbox_3d_min=bbox_min,
                bbox_3d_max=bbox_max,
                surface=keep.surface,
                observation_count=total_observations,
                last_seen=max(keep.last_seen, drop.last_seen),
            )
            if drop_key in self._displayed_instances:
                self._displayed_instances.remove(drop_key)
                self._displayed_instances.add(keep_key)
            self._distinct_landmark_pairs = {
                frozenset(keep_key if key == drop_key else key for key in pair)
                for pair in self._distinct_landmark_pairs
            }
            del self._landmarks[drop_key]
            self._landmark_redirects[drop_key] = keep_key

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
            surface_points_world=observation.surface_points_world,
        )

    def _has_valid_landmark_geometry(
        self,
        class_name: str,
        bbox_3d_min,
        bbox_3d_max,
    ) -> bool:
        extents = np.asarray(bbox_3d_max) - np.asarray(bbox_3d_min)
        vertical_extent = float(extents[2])
        horizontal_extent = float(max(extents[0], extents[1]))
        if class_name == 'exit door':
            return 1.5 <= vertical_extent <= 3.2 and .55 <= horizontal_extent <= 2.5
        if class_name == 'fire cabinet':
            return .4 <= vertical_extent <= 2.0 and .3 <= horizontal_extent <= 1.8
        if class_name != "column":
            return True
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
