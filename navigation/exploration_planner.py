"""Decision-only waypoint interface for Utility exploration."""
from dataclasses import dataclass, field
import math
from typing import Protocol, TypedDict
import numpy as np


class ExplorationWaypoint(TypedDict):
    position: list[float]  # NED metres
    yaw: float  # suggested observation yaw in radians, not a flight command
    score: float
    source: str


@dataclass
class ExplorationPose:
    position: np.ndarray
    yaw: float
    travel_heading: float | None = None


@dataclass
class ExplorationHistory:
    origin: np.ndarray
    excluded: set = field(default_factory=set)


class ExplorationPlanner(Protocol):
    def select_next_waypoint(self, occupancy_map, current_pose: ExplorationPose,
                             exploration_history: ExplorationHistory) -> ExplorationWaypoint | None:
        ...


class FrontierExplorationPlanner:
    """Keep backend visit memory; expose one waypoint without motion side effects."""
    def __init__(self, manager):
        self.manager = manager
        self.selected_key = None
        self.search_limited = self.outside_radius = False

    def prepare_candidates(self, grid, pose, history):
        choices, self.search_limited, self.outside_radius = self.manager.update(
            grid, pose.position, history.origin, pose.yaw, history.excluded)
        decision = self.manager
        heading = pose.yaw if pose.travel_heading is None else pose.travel_heading
        forward = np.array([math.cos(heading), math.sin(heading)])
        cone_cosine = math.cos(math.radians(decision.options.forward_cone_degrees))
        preferred = []
        for index, (_, point) in enumerate(choices):
            delta = np.asarray(point)[:2] - np.asarray(pose.position)[:2]
            distance = np.linalg.norm(delta)
            projection = float(delta @ forward)
            if (distance > grid.config.arrival_tolerance and projection >= cone_cosine*distance
                    and grid.segment_free(pose.position, point)):
                lateral = abs(forward[0]*delta[1]-forward[1]*delta[0])
                visits = decision.scores.get(choices[index][0], {}).get('visit_count', 0)
                preferred.append((-visits, projection, -lateral, -index))
        if preferred:
            *_, negative_index = max(preferred)
            index = -negative_index
            choices = [choices[index], *choices[:index], *choices[index+1:]]
            self.manager.visit_order = [key for key, _ in choices]
        decision.diagnostics["selection_policy"] = "forward_corridor" if preferred else "utility_turn"
        return choices

    def select_next_waypoint(self, occupancy_map, current_pose, exploration_history, *, candidates=None):
        # The runtime may reuse this cycle's coverage-reviewed candidates, avoiding
        # duplicate A* scoring. The public three-argument call prepares them itself.
        choices = (self.prepare_candidates(occupancy_map, current_pose, exploration_history)
                   if candidates is None else candidates)
        self.selected_key = None
        if not choices:
            return None  # completion still requires the coverage/recovery checks
        view = self.manager.select(choices)
        position = np.asarray(view.point, dtype=float)
        if (position.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite([view.yaw, view.score]).all()):
            raise ValueError('Invalid exploration waypoint')
        self.selected_key = view.key
        return ExplorationWaypoint(position=position.tolist(), yaw=float(view.yaw),
                                   score=float(view.score), source='utility_frontier')
