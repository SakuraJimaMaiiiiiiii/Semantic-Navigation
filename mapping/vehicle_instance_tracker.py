"""静止车辆的 NED 位置滤波、外观辅助关联；不按距离合并独立车辆。"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from .instance_surface import InstanceSurface


VEHICLE_CLASSES = ("car", "truck", "bus", "van")


@dataclass
class VehicleTrack:
    instance_id: int
    class_name: str
    state: np.ndarray
    covariance: np.ndarray
    bbox_3d_min: np.ndarray
    bbox_3d_max: np.ndarray
    appearance: np.ndarray | None
    confidence: float
    observation_count: int
    last_seen: float
    appearance_gallery: list[np.ndarray] = field(default_factory=list)
    color_votes: dict[str, float] = field(default_factory=dict)
    color_observations: int = 0
    surface: InstanceSurface = field(default_factory=InstanceSurface)
    # Unpublished partial surfaces: parent hypotheses are not identity evidence.
    possible_parents: set[int] = field(default_factory=set)
    bridge_counts: dict[int, int] = field(default_factory=dict)
    separate_counts: dict[int, int] = field(default_factory=dict)
    connected_fragment_count: int = 0

    @property
    def map_position(self):
        geometry = self.surface.vehicle_geometry()
        if geometry["position_status"] == "multi_surface_estimate":
            return np.asarray(geometry["observed_oriented_box"]["center"])
        if self.surface.bounds() is not None:
            return (self.bbox_3d_min+self.bbox_3d_max)*.5
        return self.state

    @property
    def color_semantics(self):
        if self.color_observations < 3 or not self.color_votes:
            return "unknown", 0.0
        name = max(self.color_votes, key=self.color_votes.get)
        confidence = self.color_votes[name] / sum(self.color_votes.values())
        return (name, float(confidence)) if confidence >= 0.6 else ("unknown", 0.0)


class VehicleInstanceTracker:
    """一辆车一个全程不复用的ID；不确定的观测不更新历史身份。"""

    _merge_window_seconds = 2.0

    def __init__(self, mahalanobis_gate=7.815, max_distance=2.5,
                 appearance_gate=0.55, ambiguity_margin=0.12,
                 process_noise_std=0.05, viewpoint_noise_std=0.45,
                 confirmation_observations=3):
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.max_distance = float(max_distance)
        self.appearance_gate = float(appearance_gate)
        self.ambiguity_margin = float(ambiguity_margin)
        process_noise_std = float(process_noise_std)
        viewpoint_noise_std = float(viewpoint_noise_std)
        if int(confirmation_observations) != confirmation_observations or confirmation_observations < 1:
            raise ValueError("confirmation_observations must be a positive integer")
        self.confirmation_observations = int(confirmation_observations)
        if not all(np.isfinite(value) and value > 0 for value in (
            self.mahalanobis_gate, self.max_distance, self.appearance_gate,
            self.ambiguity_margin,
        )):
            raise ValueError("vehicle association thresholds must be finite and positive")
        if not np.isfinite(process_noise_std) or process_noise_std < 0:
            raise ValueError("process_noise_std must be finite and non-negative")
        if not np.isfinite(viewpoint_noise_std) or viewpoint_noise_std < 0:
            raise ValueError("viewpoint_noise_std must be finite and non-negative")
        self.tracks: dict[int, VehicleTrack] = {}
        self._pending: list[VehicleTrack] = []
        self.redirects = {}
        self._distinct_pairs = set()
        self._merge_evidence = {}
        self.merge_diagnostics = {}
        self.replay_source_votes = {}
        self._replay_supported_pairs = set()
        self._next_id = 1
        self._last_timestamp = None
        self._last_vehicle_timestamp = None
        self._confirmation_window = 2.0
        self._identity = np.eye(3)
        self._viewpoint_covariance = self._identity * viewpoint_noise_std**2
        self._process_variance_rate = process_noise_std**2

    def update(self, class_names, confidences, observations, timestamp):
        ids = [None] * len(class_names)
        # 同一同步帧重复消费不能增加确认次数，乱序帧不能倒写轨迹。
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return ids
        self._last_timestamp = timestamp
        indices = [i for i, obs in enumerate(observations)
                   if obs is not None and self._usable_vehicle_surface(obs)]
        # Adapt to slow positive observations, not to missed/invalid detections.
        if indices:
            interval = (0.0 if self._last_vehicle_timestamp is None
                        else timestamp-self._last_vehicle_timestamp)
            self._last_vehicle_timestamp = timestamp
            self._confirmation_window = max(2.0, min(15.0, 3.0*interval))
        confirmation_window = self._confirmation_window
        self._merge_window_seconds = confirmation_window
        self._expire_merge_evidence(timestamp)
        self._pending = [track for track in self._pending
                         if timestamp-track.last_seen <= (max(90., confirmation_window)
                             if track.observation_count >= self.confirmation_observations else confirmation_window)]
        for track in self._pending:
            track.possible_parents = {self.canonical_id(i) for i in track.possible_parents}
        if not indices:
            # A missed detection adds no evidence; recent positive evidence may
            # survive briefly instead of demanding consecutive video frames.
            for track in self._pending:
                track.bridge_counts.clear()
                track.separate_counts.clear()
            return ids
        tracks = list(self.tracks.values()) + self._pending
        predictions = [self._predict(track, timestamp) for track in tracks]
        measurement_covariances = {
            index: self._measurement_covariance(observations[index])
            for index in indices
        }
        costs = np.full((len(indices), len(tracks)), np.inf)
        geometric_costs = np.full_like(costs, np.inf)
        close_geometry = np.zeros(costs.shape, dtype=bool)
        shared_views = np.zeros(costs.shape, dtype=bool)
        for row, index in enumerate(indices):
            obs = observations[index]
            for col, (track, prediction) in enumerate(zip(tracks, predictions)):
                state, covariance = prediction
                delta = obs.position_world - state
                distance = float(np.linalg.norm(delta))
                extent_distance = float(np.linalg.norm(
                    (obs.bbox_3d_min+obs.bbox_3d_max)*.5-track.map_position))
                innovation_covariance = covariance + measurement_covariances[index]
                mahalanobis = float(delta @ np.linalg.pinv(
                    innovation_covariance, hermitian=True
                ) @ delta)
                low = np.maximum(track.bbox_3d_min, obs.bbox_3d_min)
                high = np.minimum(track.bbox_3d_max, obs.bbox_3d_max)
                footprint_iou, containment = self._shared_footprint(track, obs)
                containment = max(containment, self._partial_view_support(track, obs))
                surface_match = track.surface.supports(obs.surface_points_world)
                if (distance > self.max_distance and extent_distance > self.max_distance
                        and containment < 0.75 and not surface_match):
                    continue
                appearance_distance = 0.0
                has_appearance = track.appearance is not None and obs.appearance is not None
                if has_appearance:
                    appearance_distance = min(self._appearance_distance(sample, obs.appearance)
                                              for sample in track.appearance_gallery or [track.appearance])
                # A connecting view may contain mostly new bodywork. Associate
                # it through a measured old face plus matching appearance, rather
                # than demanding that 70% of the new surface was already mapped.
                connecting_view = (has_appearance
                                   and appearance_distance <= self.appearance_gate*0.5
                                   and track.surface.observed_patch(obs.surface_points_world))
                viewpoint_match = surface_match or connecting_view or (has_appearance
                                   and appearance_distance <= self.appearance_gate*0.5
                                   and (footprint_iou >= 0.3 or containment >= 0.75))
                ordinary_match = (distance <= self.max_distance
                                  and mahalanobis <= self.mahalanobis_gate)
                # The centroid of a bumper is not the centroid of the car.
                # A contained partial surface is a plausible existing identity,
                # even when appearance is missing/conflicting: defer it instead
                # of minting another ID. Assignment needs appearance or measured
                # overlap with a repeatedly observed surface in the same frame.
                if not ordinary_match and not viewpoint_match and containment < 0.75:
                    continue
                geometric_costs[row, col] = (
                    0.5 * min(mahalanobis / self.mahalanobis_gate, 1.)
                    + 0.2 * min((extent_distance if viewpoint_match else distance) / self.max_distance, 1.)
                )
                if containment >= 0.75:
                    # Compare measured extent support, not two different visible
                    # surface anchors. Competing contained views remain ambiguous.
                    geometric_costs[row, col] = min(geometric_costs[row, col],
                                                   0.2 * (1. - containment))
                shared_views[row, col] = viewpoint_match
                # 外观突变仅在位置接近且可见包围盒明显重叠时允许恢复。
                overlap = np.maximum(0.0, high-low)
                smaller_volume = min(
                    float(np.prod(track.bbox_3d_max - track.bbox_3d_min)),
                    float(np.prod(obs.bbox_3d_max - obs.bbox_3d_min)),
                )
                close_geometry[row, col] = (
                    ordinary_match and distance <= 0.75 and smaller_volume > 1e-9
                    and float(np.prod(overlap)) / smaller_volume >= 0.5
                )
                if not ordinary_match and not viewpoint_match:
                    continue
                if has_appearance and appearance_distance > self.appearance_gate and not surface_match:
                    continue
                costs[row, col] = (
                    geometric_costs[row, col]
                    + (0. if surface_match else 0.3 * appearance_distance / self.appearance_gate)
                )

        # 唯一的近距离几何对应可跨越颜色变化；邻车或重复框竞争时禁用。
        geometric_candidates = np.isfinite(geometric_costs)
        for row, col in zip(*np.nonzero(close_geometry & ~np.isfinite(costs))):
            if (np.count_nonzero(geometric_candidates[row]) == 1
                    and np.count_nonzero(geometric_candidates[:, col]) == 1):
                costs[row, col] = geometric_costs[row, col] + 0.3

        # A fuller connecting view may explain a published car and its deferred
        # fragment. Keep assigning the published identity; require repeated
        # connecting views before importing any fragment geometry.
        bridges = {}
        for row in range(len(indices)):
            parents = [col for col, track in enumerate(tracks)
                       if track.instance_id and np.isfinite(costs[row, col])]
            if len(parents) != 1:
                continue
            parent_col = parents[0]
            parent = tracks[parent_col]
            for col, track in enumerate(tracks):
                if (not track.instance_id
                        and track.separate_counts.get(parent.instance_id, 0) < self.confirmation_observations
                        and shared_views[row, col] and shared_views[row, parent_col]
                        and track.surface.observed_patch(observations[indices[row]].surface_points_world)
                        and parent.surface.observed_patch(observations[indices[row]].surface_points_world)
                        and np.isfinite(costs[row, col])):
                    costs[row, col] = np.inf
                    bridges.setdefault(row, []).append(track)
        assignments = self._unambiguous_assignments(costs)
        connected = {id(fragment) for row in assignments for fragment in bridges.get(row, [])}
        for track in self._pending:
            if id(track) not in connected:
                track.bridge_counts.clear()
        assigned_tracks = {}
        for row, index in enumerate(indices):
            obs = observations[index]
            col = assignments.get(row)
            if col is not None:
                track = tracks[col]
                assigned_tracks[index] = track
                state, covariance = predictions[col]
                measurement_covariance = measurement_covariances[index]
                gain = covariance @ np.linalg.pinv(
                    covariance + measurement_covariance,
                    hermitian=True,
                )
                # A small face must not pull the anchor of a fuller model.
                # Fuller observations may still improve the initial partial anchor.
                old_area = np.prod(track.bbox_3d_max[:2]-track.bbox_3d_min[:2])
                new_area = np.prod(obs.bbox_3d_max[:2]-obs.bbox_3d_min[:2])
                if (shared_views[row, col] and new_area < .5*old_area
                        and np.linalg.norm(obs.position_world-state) > 1.5*InstanceSurface.resolution):
                    gain = np.zeros_like(gain)
                    covariance = track.covariance
                track.state = state + gain @ (obs.position_world - state)
                residual = self._identity - gain
                track.covariance = (
                    residual @ covariance @ residual.T
                    + gain @ measurement_covariance @ gain.T
                )
                track.covariance = (track.covariance + track.covariance.T) * 0.5
                if shared_views[row, col]:
                    # Keep the fuller measured extent until repeated point-cloud
                    # evidence updates the union. Never translate a new bumper
                    # box to the filtered center of an older side view.
                    if new_area > old_area:
                        track.bbox_3d_min = obs.bbox_3d_min.copy()
                        track.bbox_3d_max = obs.bbox_3d_max.copy()
                else:
                    offset = track.state - obs.position_world
                    track.bbox_3d_min = obs.bbox_3d_min + offset
                    track.bbox_3d_max = obs.bbox_3d_max + offset
                track.surface.update(obs.surface_points_world)
                bounds = track.surface.bounds()
                if bounds is not None:
                    track.bbox_3d_min, track.bbox_3d_max = bounds
                self._remember_visuals(track, obs)
                track.confidence = (
                    track.confidence * track.observation_count + confidences[index]
                ) / (track.observation_count + 1)
                track.observation_count += 1
                track.last_seen = timestamp
                if not track.instance_id:
                    track.possible_parents.update(i for i in self._partial_parent_candidates(obs)
                        if track.separate_counts.get(i, 0) < self.confirmation_observations)
                ids[index] = self._confirm(track)
                for fragment in bridges.get(row, []):
                    parent_id = track.instance_id
                    fragment.last_seen = timestamp
                    fragment.bridge_counts[parent_id] = fragment.bridge_counts.get(parent_id, 0)+1
                    if fragment.bridge_counts[parent_id] >= self.confirmation_observations:
                        track.surface.merge(fragment.surface)
                        track.connected_fragment_count += 1
                        bounds = track.surface.bounds()
                        if bounds is not None:
                            track.bbox_3d_min, track.bbox_3d_max = bounds
                        self._pending = [item for item in self._pending if item is not fragment]
            elif not np.any(geometric_candidates[row]):
                track = VehicleTrack(
                    instance_id=0,
                    class_name=class_names[index],
                    state=obs.position_world.copy(),
                    covariance=measurement_covariances[index].copy(),
                    bbox_3d_min=obs.bbox_3d_min.copy(),
                    bbox_3d_max=obs.bbox_3d_max.copy(),
                    appearance=None if obs.appearance is None else obs.appearance.copy(),
                    confidence=float(confidences[index]),
                    observation_count=1,
                    last_seen=timestamp,
                )
                track.surface.update(obs.surface_points_world)
                self._remember_visuals(track, obs)
                track.possible_parents = self._partial_parent_candidates(obs)
                self._pending.append(track)
                assigned_tracks[index] = track
                ids[index] = self._confirm(track)
            # 有候选但分配不唯一时保持 None，不造新ID也不污染已有轨迹。
        # Repeated separate, one-to-one detections provide independence evidence.
        # Merely revisiting an isolated face must never release its hypothesis.
        observed_ids = {track.instance_id for track in assigned_tracks.values() if track.instance_id}
        for index, track in assigned_tracks.items():
            if track.instance_id or not track.possible_parents:
                continue
            for parent in list(track.possible_parents):
                track.separate_counts[parent] = (track.separate_counts.get(parent, 0)+1
                                                  if parent in observed_ids else 0)
                if track.separate_counts[parent] >= self.confirmation_observations:
                    track.possible_parents.remove(parent)
            ids[index] = self._confirm(track)
        self._distinct_pairs.update(combinations(sorted(set(i for i in ids if i is not None)), 2))
        # Ambiguous observations do not update tracks, but can confirm that two
        # previously split IDs repeatedly explain the same measured surface.
        observed_pairs = set()
        for row in range(len(indices)):
            candidates = sorted(track.instance_id for col, track in enumerate(tracks)
                                if track.instance_id and shared_views[row, col])
            observed_pairs.update(combinations(candidates, 2))
        self._merge_duplicates(timestamp, observed_pairs)
        return [None if identity is None else self.canonical_id(identity) for identity in ids]

    def canonical_id(self, identity):
        while identity in self.redirects:
            identity = self.redirects[identity]
        return identity

    def _partial_parent_candidates(self, observation):
        """Defer plausible partial faces/corners; never merge by this test.

        Separation along the thin axis can represent the other end of a vehicle.
        Require lateral and vertical overlap so ordinary side-by-side cars are
        not all held just because they have the same paint.
        """
        result = set()
        size = observation.bbox_3d_max-observation.bbox_3d_min
        axis = int(np.argmin(size[:2]))
        lateral = 1-axis
        thin_face = size[axis] <= .35*max(size[lateral], InstanceSurface.resolution)
        for identity, track in self.tracks.items():
            if track.appearance is None or observation.appearance is None:
                continue
            appearance_distance = min(self._appearance_distance(a, observation.appearance)
                                      for a in track.appearance_gallery or [track.appearance])
            if appearance_distance > self.appearance_gate:
                continue
            # An oblique bumper/corner is not necessarily a thin plane. If it
            # partially overlaps a much fuller measured car, keep its identity
            # unresolved until a connecting view or separate detections arrive.
            # This is a deferral gate, not permission to fuse their point clouds.
            _, containment = self._shared_footprint(track, observation)
            parent_size = track.bbox_3d_max-track.bbox_3d_min
            combined_size = (np.maximum(track.bbox_3d_max, observation.bbox_3d_max)
                             - np.minimum(track.bbox_3d_min, observation.bbox_3d_min))
            if (containment >= .25
                    and np.prod(size[:2]) <= .5*np.prod(parent_size[:2])
                    and max(size[:2]) <= 2.5
                    and max(combined_size[:2]) <= 6.0
                    and min(combined_size[:2]) <= 3.0):
                result.add(identity)
                continue
            if not thin_face or appearance_distance > self.appearance_gate*.5:
                continue
            low = np.maximum(track.bbox_3d_min, observation.bbox_3d_min)
            high = np.minimum(track.bbox_3d_max, observation.bbox_3d_max)
            if high[lateral] <= low[lateral] or high[2] <= low[2]:
                continue
            gap = max(0., low[axis]-high[axis])
            if gap <= 2*self.max_distance:
                result.add(identity)
        return result

    def connection_hypotheses(self):
        """Suggest inspection only: proximity does not authorize identity fusion."""
        result = {}
        for fragment in self._pending:
            if fragment.observation_count < self.confirmation_observations:
                continue
            parents = []
            for identity, track in self.tracks.items():
                low = np.minimum(track.bbox_3d_min, fragment.bbox_3d_min)
                high = np.maximum(track.bbox_3d_max, fragment.bbox_3d_max)
                gap = np.maximum(0., np.maximum(track.bbox_3d_min, fragment.bbox_3d_min)
                                 - np.minimum(track.bbox_3d_max, fragment.bbox_3d_max))
                if (np.linalg.norm(gap[:2]) <= self.max_distance and gap[2] == 0
                        and max((high-low)[:2]) <= 6.0 and min((high-low)[:2]) <= 3.0
                        and fragment.separate_counts.get(identity, 0) < self.confirmation_observations):
                    parents.append(identity)
            if len(parents) == 1:
                result.setdefault(parents[0], []).append({
                    "min": fragment.bbox_3d_min.tolist(), "max": fragment.bbox_3d_max.tolist(),
                    "last_seen": fragment.last_seen})
        return result

    @staticmethod
    def _usable_vehicle_surface(observation):
        # A broad planar side/bumper is valid even with almost zero thickness.
        # A tiny edge/floor strip must neither mint an ID nor move an existing car.
        size = np.asarray(observation.bbox_3d_max)-observation.bbox_3d_min
        return bool(np.isfinite(size).all() and (size >= 0).all()
                    and size[2] >= 2*InstanceSurface.resolution
                    and max(size[:2]) >= 4*InstanceSurface.resolution)

    @staticmethod
    def _partial_view_support(first, second):
        """Contain a partial face in an already measured fuller extent.

        Allow one surface matching tolerance at its border, not a car-sized
        expansion. Require actual XY intersection and vertical overlap; two
        separated parked cars cannot qualify by color or proximity alone.
        """
        items = (first, second)
        sizes = [np.maximum(item.bbox_3d_max[:2]-item.bbox_3d_min[:2],
                            InstanceSurface.resolution) for item in items]
        areas = [float(np.prod(size)) for size in sizes]
        small, large = (0, 1) if areas[0] <= areas[1] else (1, 0)
        if areas[large] < 2*areas[small]:
            return 0.
        a, b = items[small], items[large]
        low, high = np.maximum(a.bbox_3d_min, b.bbox_3d_min), np.minimum(a.bbox_3d_max, b.bbox_3d_max)
        min_height = min(a.bbox_3d_max[2]-a.bbox_3d_min[2], b.bbox_3d_max[2]-b.bbox_3d_min[2])
        if np.any(high[:2] < low[:2]) or high[2]-low[2] < .5*min_height or min_height <= 0:
            return 0.
        tolerance = 1.5*InstanceSurface.resolution
        center = (a.bbox_3d_min[:2]+a.bbox_3d_max[:2])*.5
        if np.any(center < b.bbox_3d_min[:2]) or np.any(center > b.bbox_3d_max[:2]):
            return 0.
        small_low, small_high = center-sizes[small]*.5, center+sizes[small]*.5
        overlap = np.maximum(0., np.minimum(small_high, b.bbox_3d_max[:2]+tolerance)
                             - np.maximum(small_low, b.bbox_3d_min[:2]-tolerance))
        fraction = float(np.prod(overlap)/areas[small])
        return fraction if fraction >= .85 else 0.

    @staticmethod
    def _shared_footprint(first, second):
        """IoU and partial/full containment of *measured* vehicle extents.

        One surface voxel is the minimum footprint thickness for planar scans;
        boxes are not expanded by pose covariance or vehicle association radius.
        Containment is reserved for different-sized views, not adjacent full cars.
        """
        low = np.maximum(first.bbox_3d_min, second.bbox_3d_min)
        high = np.minimum(first.bbox_3d_max, second.bbox_3d_max)
        z_size = min(first.bbox_3d_max[2]-first.bbox_3d_min[2],
                     second.bbox_3d_max[2]-second.bbox_3d_min[2])
        if high[2]-low[2] <= 0 or high[2]-low[2] < 0.5*z_size:
            return 0., 0.
        sizes, centers = [], []
        for item in (first, second):
            sizes.append(np.maximum(item.bbox_3d_max[:2]-item.bbox_3d_min[:2],
                                    InstanceSurface.resolution))
            centers.append((item.bbox_3d_min[:2]+item.bbox_3d_max[:2])*0.5)
        overlap = np.maximum(0., np.minimum(centers[0]+sizes[0]*0.5, centers[1]+sizes[1]*0.5)
                             - np.maximum(centers[0]-sizes[0]*0.5, centers[1]-sizes[1]*0.5))
        intersection = float(np.prod(overlap))
        areas = [float(np.prod(size)) for size in sizes]
        iou = intersection / (sum(areas)-intersection)
        smaller, larger = min(areas), max(areas)
        containment = intersection/smaller if larger >= 2.*smaller else 0.
        return iou, containment

    def _merge_duplicates(self, timestamp, observed_pairs=()):
        self._expire_merge_evidence(timestamp)
        evidence = dict(self._merge_evidence)
        candidates = []
        for first, second in combinations(sorted(self.tracks), 2):
            a, b = self.tracks[first], self.tracks[second]
            pair = (first, second)
            if max(a.last_seen, b.last_seen) != timestamp and pair not in observed_pairs:
                continue
            iou, containment = self._shared_footprint(a, b)
            partial_support = self._partial_view_support(a, b)
            containment = max(containment, partial_support)
            # Compare the same measured-surface anchors exported to map/HTML.
            # Filtered observation centroids can sit at opposite ends of a car.
            distance = float(np.linalg.norm(a.map_position-b.map_position))
            # Far, disjoint objects are not plausible duplicates. This broad
            # phase is only a query bound, never evidence for merging.
            if distance > self.max_distance and iou == 0. and containment == 0.:
                evidence.pop(pair, None)
                continue
            common_surface = (a.surface.supports(b.surface.supported_points())
                              or b.surface.supports(a.surface.supported_points()))
            difference = (min(self._appearance_distance(x, y)
                              for x in a.appearance_gallery for y in b.appearance_gallery)
                          if a.appearance_gallery and b.appearance_gallery else None)
            common_appearance = difference is not None and difference <= self.appearance_gate*.5
            shared_patch = (common_appearance and iou > 0.
                            and distance <= self.max_distance and a.surface.shares_patch(b.surface))
            overturn_split = (shared_patch and a.surface.shares_patch(b.surface, minimum_fraction=.5)
                              and not (a.last_seen == timestamp and b.last_seen == timestamp))
            # A split mask can create two IDs in one frame. Only repeated fresh
            # views with >=50% reciprocal stable patch support can overturn it;
            # a new same-frame independent assignment still vetoes merging.
            geometry_match = containment >= .75 or (iou >= .5 and distance <= self.max_distance)
            # A replay must not discard repeated identity evidence recorded by
            # the live tracker. It supplements geometry, never overrides a
            # same-frame distinct pair or joins disjoint/similar parked cars.
            source_a = self.replay_source_votes.get(first, {})
            source_b = self.replay_source_votes.get(second, {})
            same_source = (len(source_a) == len(source_b) == 1
                           and source_a.keys() == source_b.keys()
                           and min(*source_a.values(), *source_b.values()) >= self.confirmation_observations)
            overlap_xy = np.maximum(0., np.minimum(a.bbox_3d_max[:2], b.bbox_3d_max[:2])
                                    - np.maximum(a.bbox_3d_min[:2], b.bbox_3d_min[:2]))
            smaller_area = min(np.prod(a.bbox_3d_max[:2]-a.bbox_3d_min[:2]),
                               np.prod(b.bbox_3d_max[:2]-b.bbox_3d_min[:2]))
            source_overlap = float(np.prod(overlap_xy)/smaller_area) if smaller_area > 0. else 0.
            source_supported = (same_source and common_appearance and source_overlap >= .5
                                and iou > 0. and distance <= self.max_distance
                                and min(a.bbox_3d_max[2], b.bbox_3d_max[2])
                                > max(a.bbox_3d_min[2], b.bbox_3d_min[2]))
            if source_supported:
                self._replay_supported_pairs.add(pair)
            elif pair in self._replay_supported_pairs:
                self._replay_supported_pairs.discard(pair)
                evidence.pop(pair, None)
            geometry_match = geometry_match or source_supported
            # A historical split can also come from complementary proposals.
            # Repeated source identity plus overlapping geometry may repair it,
            # but two independently assigned detections in this frame veto it.
            overturn_split = overturn_split or (source_supported and not (
                a.last_seen == timestamp and b.last_seen == timestamp))
            if pair in self._distinct_pairs and not overturn_split:
                self._replay_supported_pairs.discard(pair)
                evidence.pop(pair, None)
                self._record_merge_decision(pair, timestamp, 'coobserved_distinct')
                continue
            # A bumper/side view can have a poor box IoU with a whole-car view.
            # Override that gate only when stable world surfaces AND a fresh
            # shared observation support the same two identities. No ICP fit.
            surface_override = (common_surface and pair in observed_pairs) or shared_patch
            metrics = dict(footprint_iou=iou, containment=containment,
                           partial_view_support=partial_support,
                           center_distance_m=distance, common_surface=bool(common_surface),
                           filtered_center_distance_m=float(np.linalg.norm(a.state-b.state)),
                           shared_current_view=pair in observed_pairs, appearance_distance=difference)
            metrics['shared_surface_patch'] = bool(shared_patch)
            metrics['replay_source_supported'] = bool(source_supported)
            if not geometry_match and not surface_override:
                evidence.pop(pair, None)
                self._record_merge_decision(pair, timestamp, 'geometry_without_shared_surface', **metrics)
                continue
            if not common_surface and not common_appearance:
                evidence.pop(pair, None)
                self._record_merge_decision(pair, timestamp, 'appearance_and_surface_unconfirmed', **metrics)
                continue
            stamps = evidence.get(pair, ())
            if not stamps or timestamp > stamps[-1]:
                stamps = (*stamps, timestamp)[-self.confirmation_observations:]
            evidence[pair] = stamps
            self._record_merge_decision(pair, timestamp, 'confirming',
                                       supporting_frames=len(stamps),
                                       evidence_span_seconds=float(stamps[-1]-stamps[0]), **metrics)
            candidates.append(pair)
        # A track must have exactly one plausible duplicate. Do not collapse a
        # row of similar parked cars via transitive proximity.
        counts = {i: sum(i in pair for pair in candidates) for i in self.tracks}
        for first, second in candidates:
            if counts[first] != 1 or counts[second] != 1:
                evidence.pop((first, second), None)
                self._record_merge_decision((first, second), timestamp, 'multiple_partners')
                continue
            if len(evidence[(first,second)]) < self.confirmation_observations:
                continue
            self._record_merge_decision((first, second), timestamp, 'merged')
            a, b = self.tracks[first], self.tracks.pop(second)
            total = a.observation_count+b.observation_count
            a.confidence = (a.confidence*a.observation_count+b.confidence*b.observation_count)/total
            # Keep the fuller extent for a partial/full pair, even if the bumper
            # has been observed more often. Do not average two surface anchors
            # into an invented box between them.
            _, containment = self._shared_footprint(a, b)
            containment = max(containment, self._partial_view_support(a, b))
            prefer_b_extent = (
                np.prod(b.bbox_3d_max[:2]-b.bbox_3d_min[:2])
                > np.prod(a.bbox_3d_max[:2]-a.bbox_3d_min[:2])
            ) if containment >= .75 else b.observation_count > a.observation_count
            if b.observation_count > a.observation_count:
                a.state, a.covariance = b.state.copy(), b.covariance.copy()
            if prefer_b_extent:
                a.bbox_3d_min, a.bbox_3d_max = b.bbox_3d_min.copy(), b.bbox_3d_max.copy()
            a.surface.merge(b.surface)
            a.connected_fragment_count += b.connected_fragment_count
            bounds = a.surface.bounds()
            if bounds is not None:
                a.bbox_3d_min, a.bbox_3d_max = bounds
            a.observation_count = total
            a.last_seen = max(a.last_seen,b.last_seen)
            a.appearance_gallery = (a.appearance_gallery+b.appearance_gallery)[-8:]
            for color, value in b.color_votes.items():
                a.color_votes[color] = a.color_votes.get(color,0.)+value
            a.color_observations += b.color_observations
            self.redirects[second] = first
            self._distinct_pairs = {tuple(sorted((self.canonical_id(x), self.canonical_id(y))))
                                    for x,y in self._distinct_pairs if self.canonical_id(x) != self.canonical_id(y)}
        self._merge_evidence = {pair: stamps for pair, stamps in evidence.items()
                                if all(identity in self.tracks for identity in pair)}
        self._replay_supported_pairs.intersection_update(self._merge_evidence)

    def _expire_merge_evidence(self, timestamp):
        for pair, stamps in list(self._merge_evidence.items()):
            # Revisit evidence for a recorded static identity survives gaps in
            # observation; contradictory geometry/source or co-observation
            # clears it in _merge_duplicates. Ordinary live evidence still ages.
            recent = tuple(stamp for stamp in stamps if 0 <= timestamp-stamp and
                           (pair in self._replay_supported_pairs or timestamp-stamp <= self._merge_window_seconds))
            if recent and all(identity in self.tracks for identity in pair):
                self._merge_evidence[pair] = recent
            else:
                self._merge_evidence.pop(pair)
                self._record_merge_decision(pair, timestamp, 'confirmation_expired')

    def _record_merge_decision(self, pair, timestamp, reason, **metrics):
        # Aggregate per pair/reason, not an unbounded per-frame point-cloud log.
        key = '|'.join(f'vehicle_{identity:02d}' for identity in pair)
        entry = self.merge_diagnostics.setdefault(key, {})
        previous = entry.get(reason, {})
        entry[reason] = dict(count=previous.get('count', 0)+1,
                             last_timestamp=float(timestamp), **metrics)

    def _confirm(self, track):
        if track.instance_id == 0:
            if track.possible_parents or track.observation_count < self.confirmation_observations:
                return None
            # Repeated sightings of one small bumper patch are not independent
            # evidence of a whole new car. Keep it pending until a broader face
            # is observed; existing identities can still consume partial views.
            size = track.bbox_3d_max - track.bbox_3d_min
            # A floor/road patch can span several metres while its vertical
            # extent is only one occupancy voxel (~0.3 m). Horizontal width
            # alone must not promote that patch to a new vehicle identity.
            # Keep it pending: a later body view may establish the car. Existing
            # identities may still accept supported partial observations.
            if size[2] < 4*InstanceSurface.resolution:
                return None
            if max(size[:2]) < 1.5 and size[2] < .8:
                return None
            track.instance_id = self._next_id
            self._next_id += 1
            self.tracks[track.instance_id] = track
            self._pending = [item for item in self._pending if item is not track]
        return track.instance_id

    def _remember_visuals(self, track, observation):
        appearance = observation.appearance
        if appearance is not None:
            track.appearance = appearance.copy()
            if not track.appearance_gallery or min(
                self._appearance_distance(sample, appearance)
                for sample in track.appearance_gallery
            ) > 0.15:
                track.appearance_gallery.append(appearance.copy())
                # 有界多视角记忆，保留首次视角与最近的不同视角。
                if len(track.appearance_gallery) > 8:
                    del track.appearance_gallery[1]
        scores = observation.color_scores
        if scores and max(scores.values()) >= 0.55:
            for name, score in scores.items():
                track.color_votes[name] = track.color_votes.get(name, 0.0) + score
            track.color_observations += 1

    def _measurement_covariance(self, observation):
        # 雅可比仅描述像素/深度误差；可见车面中心随视角改变是额外误差。
        # 每次测量都加入该项，不能因历史滤波收敛就忽略它。
        return observation.measurement_covariance + self._viewpoint_covariance

    def _predict(self, track, timestamp):
        # 场景车辆静止；NED 观测已在上游补偿无人机位姿。
        # F=H=I3，不从可见表面偏移估计车速。长时间离开仍保留原位置。
        elapsed = max(0.0, min(timestamp - track.last_seen, 5.0))
        covariance = track.covariance + self._identity * self._process_variance_rate * elapsed
        return track.state, covariance

    @staticmethod
    def _appearance_distance(first, second):
        # Hellinger距离；输入为掩膜内归一化颜色直方图。
        if first.shape != second.shape:
            return 1.0
        first = first / first.sum()
        second = second / second.sum()
        return float(np.sqrt(max(0.0, 1.0 - np.sqrt(first * second).sum())))

    def _unambiguous_assignments(self, costs):
        if costs.shape[1] == 0:
            return {}
        # 每行配置虚拟未匹配列，确保一对一全局分配始终可解。
        padded = np.column_stack((
            np.where(np.isfinite(costs), costs, 1e6),
            np.full((costs.shape[0], costs.shape[0]), 2.0),
        ))
        assignments = {}
        for row, col in enumerate(_minimum_cost_assignment(padded)):
            if col >= costs.shape[1] or not np.isfinite(costs[row, col]):
                continue
            row_alternatives = np.delete(costs[row], col)
            col_alternatives = np.delete(costs[:, col], row)
            chosen = costs[row, col]
            if (
                np.any(row_alternatives - chosen < self.ambiguity_margin)
                or np.any(col_alternatives - chosen < self.ambiguity_margin)
            ):
                continue
            assignments[row] = col
        return assignments


def _minimum_cost_assignment(costs):
    """矩形匈牙利分配（行数 <= 列数），仅依赖 NumPy。"""
    rows, cols = costs.shape
    u, v = np.zeros(rows + 1), np.zeros(cols + 1)
    owners, previous = np.zeros(cols + 1, dtype=int), np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        owners[0] = row
        col = 0
        minimum = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[col] = True
            current_row = owners[col]
            delta, next_col = np.inf, 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                value = costs[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if value < minimum[candidate]:
                    minimum[candidate], previous[candidate] = value, col
                if minimum[candidate] < delta:
                    delta, next_col = minimum[candidate], candidate
            for candidate in range(cols + 1):
                if used[candidate]:
                    u[owners[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            col = next_col
            if owners[col] == 0:
                break
        while col:
            source = previous[col]
            owners[col] = owners[source]
            col = source
    assignment = np.empty(rows, dtype=int)
    for col in range(1, cols + 1):
        if owners[col]:
            assignment[owners[col] - 1] = col - 1
    return assignment
