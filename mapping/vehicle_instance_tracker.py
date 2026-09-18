"""静止车辆的 NED 位置滤波、外观辅助关联；不按距离合并独立车辆。"""

from dataclasses import dataclass, field

import numpy as np


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

    @property
    def color_semantics(self):
        if self.color_observations < 3 or not self.color_votes:
            return "unknown", 0.0
        name = max(self.color_votes, key=self.color_votes.get)
        confidence = self.color_votes[name] / sum(self.color_votes.values())
        return (name, float(confidence)) if confidence >= 0.6 else ("unknown", 0.0)


class VehicleInstanceTracker:
    """一辆车一个全程不复用的ID；不确定的观测不更新历史身份。"""

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
        self._next_id = 1
        self._last_timestamp = None
        self._identity = np.eye(3)
        self._viewpoint_covariance = self._identity * viewpoint_noise_std**2
        self._process_variance_rate = process_noise_std**2

    def update(self, class_names, confidences, observations, timestamp):
        ids = [None] * len(class_names)
        # 同一同步帧重复消费不能增加确认次数，乱序帧不能倒写轨迹。
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return ids
        self._last_timestamp = timestamp
        # 待确认目标只保留近期连续证据；正式地图在本次运行中长期保留。
        self._pending = [track for track in self._pending if timestamp - track.last_seen <= 2.0]
        indices = [i for i, obs in enumerate(observations) if obs is not None]
        if not indices:
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
        for row, index in enumerate(indices):
            obs = observations[index]
            for col, (track, prediction) in enumerate(zip(tracks, predictions)):
                state, covariance = prediction
                delta = obs.position_world - state
                distance = float(np.linalg.norm(delta))
                if distance > self.max_distance:
                    continue
                innovation_covariance = covariance + measurement_covariances[index]
                mahalanobis = float(delta @ np.linalg.pinv(
                    innovation_covariance, hermitian=True
                ) @ delta)
                if mahalanobis > self.mahalanobis_gate:
                    continue
                geometric_costs[row, col] = (
                    0.5 * mahalanobis / self.mahalanobis_gate
                    + 0.2 * distance / self.max_distance
                )
                # 外观突变仅在位置接近且可见包围盒明显重叠时允许恢复。
                overlap = np.maximum(0.0, np.minimum(track.bbox_3d_max, obs.bbox_3d_max)
                                     - np.maximum(track.bbox_3d_min, obs.bbox_3d_min))
                smaller_volume = min(
                    float(np.prod(track.bbox_3d_max - track.bbox_3d_min)),
                    float(np.prod(obs.bbox_3d_max - obs.bbox_3d_min)),
                )
                close_geometry[row, col] = (
                    distance <= 0.75 and smaller_volume > 1e-9
                    and float(np.prod(overlap)) / smaller_volume >= 0.5
                )
                appearance_distance = 0.0
                if track.appearance is not None and obs.appearance is not None:
                    appearance_distance = min(self._appearance_distance(sample, obs.appearance)
                                              for sample in track.appearance_gallery or [track.appearance])
                    if appearance_distance > self.appearance_gate:
                        continue
                costs[row, col] = (
                    geometric_costs[row, col]
                    + 0.3 * appearance_distance / self.appearance_gate
                )

        # 唯一的近距离几何对应可跨越颜色变化；邻车或重复框竞争时禁用。
        geometric_candidates = np.isfinite(geometric_costs)
        for row, col in zip(*np.nonzero(close_geometry & ~np.isfinite(costs))):
            if (np.count_nonzero(geometric_candidates[row]) == 1
                    and np.count_nonzero(geometric_candidates[:, col]) == 1):
                costs[row, col] = geometric_costs[row, col] + 0.3

        assignments = self._unambiguous_assignments(costs)
        for row, index in enumerate(indices):
            obs = observations[index]
            col = assignments.get(row)
            if col is not None:
                track = tracks[col]
                state, covariance = predictions[col]
                measurement_covariance = measurement_covariances[index]
                gain = covariance @ np.linalg.pinv(
                    covariance + measurement_covariance,
                    hermitian=True,
                )
                track.state = state + gain @ (obs.position_world - state)
                residual = self._identity - gain
                track.covariance = (
                    residual @ covariance @ residual.T
                    + gain @ measurement_covariance @ gain.T
                )
                track.covariance = (track.covariance + track.covariance.T) * 0.5
                # 将本帧可见包围盒对齐融合位置，避免累积视角误差造成膨胀。
                offset = track.state - obs.position_world
                track.bbox_3d_min = obs.bbox_3d_min + offset
                track.bbox_3d_max = obs.bbox_3d_max + offset
                self._remember_visuals(track, obs)
                track.confidence = (
                    track.confidence * track.observation_count + confidences[index]
                ) / (track.observation_count + 1)
                track.observation_count += 1
                track.last_seen = timestamp
                ids[index] = self._confirm(track)
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
                self._remember_visuals(track, obs)
                self._pending.append(track)
                ids[index] = self._confirm(track)
            # 有候选但分配不唯一时保持 None，不造新ID也不污染已有轨迹。
        return ids

    def _confirm(self, track):
        if track.instance_id == 0:
            if track.observation_count < self.confirmation_observations:
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
