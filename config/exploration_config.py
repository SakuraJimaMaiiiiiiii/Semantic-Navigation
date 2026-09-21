"""Information-gain frontier exploration budgets at a fixed flight height."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class FrontierExplorationConfig:
    timeout: float = 3600.0
    max_viewpoints: int = 600
    max_radius: float = 100.0
    viewpoint_spacing: float = 3.0
    information_radius: float = 3.0
    information_weight: float = 1.0
    path_cost_weight: float = 0.35
    heading_cost_weight: float = 0.2
    camera_fov_degrees: float = 90.0
    tour_frontiers: int = 5
    tour_cost_weight: float = 0.15
    scan_wait: float = 0.8
    arrival_observation_wait: float = 1.0
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
    minimum_map_gain_area: float = 1.0
    free_space_recovery_timeout: float = 5.0
    return_time_reserve: float = 120.0
    relation_radius: float = 6.0
    near_distance: float = 3.0
    home_tolerance: float = 0.6
    server_port: int = 8877

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be positive and finite")
        if (
            not isinstance(self.max_viewpoints, int)
            or not isinstance(self.server_port, int)
            or not isinstance(self.max_frontier_attempts, int)
            or not isinstance(self.frontier_empty_confirmations, int)
            or not isinstance(self.recovery_observations, int)
            or not isinstance(self.gain_window_viewpoints, int)
        ):
            raise ValueError(
                "viewpoint, retry, confirmation, recovery, gain-window and port values must be integers"
            )
        if self.server_port > 65535:
            raise ValueError("Invalid server port")
        if not isinstance(self.tour_frontiers, int) or self.tour_frontiers > 10:
            raise ValueError("tour_frontiers must be an integer from 1 to 10")
        if self.camera_fov_degrees > 180:
            raise ValueError("camera_fov_degrees must not exceed 180")


# Backward-compatible name for callers and archived tests created before the
# strategy changed from depth-first traversal to utility-ranked frontiers.
DFSExplorationConfig = FrontierExplorationConfig
