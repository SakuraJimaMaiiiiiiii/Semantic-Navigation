"""Frontier exploration with bounded, clearance-certified height recovery."""

from dataclasses import dataclass, fields, field
from .frontier_decision_config import FrontierDecisionConfig
import math


@dataclass(frozen=True)
class FrontierExplorationConfig:
    decision: FrontierDecisionConfig = field(default_factory=FrontierDecisionConfig)
    timeout: float = 3600.0
    max_viewpoints: int = 600
    max_radius: float = 100.0
    flight_clearance: float = 0.40  # 水平规划净空，m；探索与返航共用
    height_step: float = 0.4  # 候选高度检查间隔，m；选中的完整上升通道必须有净空证据
    max_height_gain: float = 1.0  # 相对本次起飞巡航高度的最大额外上升，m
    object_inspection_views: int = 4  # 每个物体最多尝试的补充观察航点数
    object_inspection_timeout: float = 90.0  # 单物体补看总预算，秒
    viewpoint_spacing: float = 3.0
    information_radius: float = 3.0
    camera_fov_degrees: float = 90.0
    candidate_radius_min: float = 1.5
    candidate_radius_max: float = 2.5
    recovery_observation_wait: float = 0.3
    rolling_replan_period: float = 1.5
    progress_timeout: float = 8.0
    progress_distance: float = 0.15
    perception_recovery_timeout: float = 20.0  # 语义超时后悬停等待新帧的最长秒数
    frontier_retry_cooldown: float = 30.0
    max_frontier_attempts: int = 3
    frontier_empty_confirmations: int = 3
    frontier_empty_min_duration: float = 10.0
    recovery_observations: int = 2
    completion_frontier_area: float = 1.0
    gain_window_viewpoints: int = 5
    coverage_plateau_seconds: float = 120.0  # 地图增长滑动窗口秒数
    coverage_relative_gain: float = 0.01  # 每窗口新增不足已知面积的 1% 时复查
    coverage_review_attempts: int = 3  # 低增长期间最多再尝试的探索目标数
    minimum_map_gain_area: float = 1.0
    free_space_recovery_timeout: float = 5.0
    return_time_reserve: float = 120.0
    relation_radius: float = 6.0
    near_distance: float = 3.0
    home_tolerance: float = 0.6
    server_port: int = 8877

    def __post_init__(self):
        if self.height_step < .1:
            raise ValueError('height_step must be at least 0.1 m')
        if not isinstance(self.decision, FrontierDecisionConfig):
            raise ValueError("decision must be FrontierDecisionConfig")
        for field in fields(self):
            if field.name == "decision":
                continue
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0 or (value == 0 and field.name != "max_height_gain"):
                raise ValueError(f"{field.name} must be positive and finite (max_height_gain may be zero)")
        if (
            not isinstance(self.object_inspection_views, int)
            or isinstance(self.object_inspection_views, bool)
            or not isinstance(self.max_viewpoints, int)
            or not isinstance(self.server_port, int)
            or not isinstance(self.max_frontier_attempts, int)
            or not isinstance(self.frontier_empty_confirmations, int)
            or not isinstance(self.recovery_observations, int)
            or not isinstance(self.gain_window_viewpoints, int)
        ):
            raise ValueError(
                "viewpoint, retry, confirmation, recovery, gain-window, "
                "and port values must be integers"
            )
        if not 0 < self.coverage_relative_gain <= 1:
            raise ValueError("coverage_relative_gain must be in (0, 1]")
        if not isinstance(self.coverage_review_attempts, int) or isinstance(self.coverage_review_attempts, bool):
            raise ValueError("coverage_review_attempts must be an integer")
        if self.server_port > 65535:
            raise ValueError("Invalid server port")
        if self.camera_fov_degrees > 180:
            raise ValueError("camera_fov_degrees must not exceed 180")
        if self.candidate_radius_max < self.candidate_radius_min:
            raise ValueError(
                "candidate_radius_max must not be below candidate_radius_min"
            )
