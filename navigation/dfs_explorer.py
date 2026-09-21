"""Information-gain frontier exploration with A*/EGO motion execution.

The historical module and ``DFSExplorer`` name remain for entry-point
compatibility. Exploration itself uses utility-ranked frontier regions.
"""

from dataclasses import replace
import math
import sys

import numpy as np

from .mission import SemanticNavigator
from .route import NavigationGrid, next_leg
from .scene_memory import write_json
from .frontier_manager import FrontierManager, visible_unknown


def _angle_difference(first, second):
    return abs((first - second + math.pi) % (2 * math.pi) - math.pi)


class FrontierExplorer(SemanticNavigator):
    """FUEL-inspired region/viewpoint tour planning with EGO execution."""

    def __init__(self, *args, exploration_config, memory, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_config = exploration_config
        self.config = replace(
            self.config,
            mission_timeout=exploration_config.timeout,
            scan_wait=exploration_config.scan_wait,
            max_search_radius=exploration_config.max_radius,
        )
        self.memory = memory
        self.checkpoint = checkpoint
        self.current_place = None
        self.recovery_yaws = {}
        self._empty_frontier_since = None
        self._empty_frontier_confirmations = 0
        self._coverage_samples = []
        self._last_frontier_area = None
        self.blocked = 0
        self.failed_frontiers = {}
        self._motion_progress = None
        self.frontiers = FrontierManager(exploration_config)
        self.active_viewpoint = None

    def _check_health(self):
        super()._check_health()
        if self._motion_progress is not None:
            position, changed = self._motion_progress
            current = self.vehicle.get_position()
            if (
                np.linalg.norm(current[:2] - position[:2])
                >= self.exploration_config.progress_distance
            ):
                self._motion_progress = current.copy(), self.clock()
            elif self.clock() - changed >= self.exploration_config.progress_timeout:
                self.vehicle.hover()
                raise TimeoutError("exploration_no_motion_progress")

    def _frame(self):
        try:
            frame = super()._frame()
        except RuntimeError as error:
            if str(error) != "semantic_perception_stale":
                raise
            frame = self._recover_perception()
        self.memory.observe(frame, self.current_place)
        return frame

    def _recover_perception(self):
        self.vehicle.hover()
        started = self.clock()
        end = min(
            self.deadline, started + self.exploration_config.perception_recovery_timeout
        )
        last = self.perception.semantic_snapshot()
        self._event(
            "PERCEPTION_WAIT",
            frame_age_seconds=(
                None if last is None else max(0.0, started - last.completed_at)
            ),
            inference_seconds=None if last is None else last.inference_seconds,
        )
        while self.clock() < end:
            self.perception.raise_if_failed()
            self._check_maps()
            try:
                frame = super()._frame()
            except RuntimeError as error:
                if str(error) != "semantic_perception_stale":
                    raise
            else:
                if self._motion_progress is not None:
                    self._motion_progress = (
                        self.vehicle.get_position().copy(),
                        self.clock(),
                    )
                self._event(
                    "PERCEPTION_RECOVERED", waited_seconds=self.clock() - started
                )
                return frame
            self.vehicle.wait(min(0.2, end - self.clock()))
        self._event(
            "PERCEPTION_RECOVERY_TIMEOUT", waited_seconds=self.clock() - started
        )
        if self.clock() >= self.deadline:
            raise TimeoutError("semantic_mission_timeout")
        raise RuntimeError("semantic_perception_stale")

    def _checkpoint(self):
        if self.result is None:
            self.memory.exploration = {
                "strategy": "fuel_inspired_frontier",
                "motion_policy": "global_route_direct_ego_on_obstacle",
                "local_planner": "ego_avoidance_only",
                "height_policy": "fixed_takeoff_plane",
                "state": "exploring",
                "viewpoint_count": len(self.memory.places),
                "blocked_frontiers": self.blocked,
                "execution_events": list(self.events),
                "returned_home": False,
            }
        if self.checkpoint is not None:
            self.checkpoint(self)

    def _full_scan(self, reason):
        print(f"--- 前沿探索 360° 扫描：{self.current_place}，原因={reason} ----")
        initial = float(self.vehicle.get_euler()[2])
        for yaw in (
            initial,
            initial + math.pi / 2,
            initial + math.pi,
            initial + 3 * math.pi / 2,
        ):
            self._align(yaw)
            self._wait(self.config.scan_wait)
        self._frame()
        self.memory.places[self.current_place]["scanned"] = True
        self._checkpoint()

    def _recovery_observation(self, grid, bin_key):
        """Look once toward the best unknown sector instead of rotating 360 degrees."""
        cell = grid.cell(self.vehicle.get_position())
        if cell not in grid.free:
            return False
        current_yaw = float(self.vehicle.get_euler()[2])
        excluded = self.recovery_yaws.setdefault(bin_key, set())
        options = []
        for index in range(8):
            if index in excluded:
                continue
            yaw = index * math.pi / 4
            visible = visible_unknown(
                grid,
                cell,
                yaw,
                self.exploration_config.information_radius,
                self.exploration_config.camera_fov_degrees,
            )
            turn = _angle_difference(yaw, current_yaw)
            options.append((len(visible), -turn, index, yaw))
        if not options:
            return False
        gain, _, index, yaw = max(options)
        if gain <= 0:
            return False
        excluded.add(index)
        print(f"--- 前沿恢复观察：预计未知格={gain}，仅转向一次 ----")
        self._align(yaw)
        self._wait(self.exploration_config.arrival_observation_wait)
        self._frame()
        self._record_coverage(self._grid())
        self._checkpoint()
        return True

    def _record_coverage(self, grid):
        known = getattr(grid, "known", None)
        resolution = getattr(grid, "resolution", None)
        if known is None or resolution is None:
            return
        self._coverage_samples.append(len(known) * float(resolution) ** 2)

    def _recent_map_gain(self):
        window = self.exploration_config.gain_window_viewpoints
        if len(self._coverage_samples) < window + 1:
            return None
        samples = self._coverage_samples[-(window + 1) :]
        return max(0.0, samples[-1] - samples[0])

    def _remaining_frontier_area(self, grid):
        return (
            sum(len(region) for region in self.frontiers.regions) * grid.resolution**2
        )

    def _reset_empty_frontier_evidence(self):
        self._empty_frontier_since = None
        self._empty_frontier_confirmations = 0

    def _confirm_empty_frontiers(self):
        now = self.clock()
        if self._empty_frontier_since is None:
            self._empty_frontier_since = now
            self._empty_frontier_confirmations = 1
        else:
            self._empty_frontier_confirmations += 1
        elapsed = now - self._empty_frontier_since
        enough = (
            self._empty_frontier_confirmations
            >= self.exploration_config.frontier_empty_confirmations
            and elapsed >= self.exploration_config.frontier_empty_min_duration
        )
        return enough, elapsed

    def _recover_free_space(self):
        self.vehicle.hover()
        end = min(
            self.deadline,
            self.clock() + self.exploration_config.free_space_recovery_timeout,
        )
        while self.clock() < end:
            self._wait(min(0.5, end - self.clock()))
            grid = self._grid()
            if grid.cell(self.vehicle.get_position()) in grid.free:
                self._event("FREE_SPACE_EVIDENCE_RECOVERED")
                return grid
        return None

    def _observe_arrival(self):
        # Motion already faces the selected frontier. Collect a stable forward
        # observation without forcing another full rotation.
        if (
            self.active_viewpoint is not None
            and np.linalg.norm(
                self.vehicle.get_position()[:2] - self.active_viewpoint.point[:2]
            )
            <= self.config.arrival_tolerance
        ):
            self._align(self.active_viewpoint.yaw)
            self.frontiers.observe_view(self.active_viewpoint)
        self._wait(self.exploration_config.arrival_observation_wait)
        self._frame()
        self._record_coverage(self._grid())
        self.memory.places[self.current_place]["scanned"] = False
        self._checkpoint()

    def _grid(self):
        return NavigationGrid.from_global(
            self.global_map.snapshot(), self.altitude, self.config
        )

    def _travel(self, goal):
        self.current_place = None
        while (
            np.linalg.norm(self.vehicle.get_position() - goal)
            > self.config.arrival_tolerance
        ):
            self._check_health()
            grid = self._grid()
            grid.free = {
                cell
                for cell in grid.free
                if np.linalg.norm(grid.point(cell)[:2] - self.exploration_origin[:2])
                <= self.exploration_config.max_radius
            }
            path = grid.path(self.vehicle.get_position(), goal)
            if path is None:
                self.vehicle.hover()
                return False
            try:
                # Velocity guidance turns toward the checked route continuously;
                # do not stop and fully align at every replanning leg.
                self._motion_progress = self.vehicle.get_position().copy(), self.clock()
                self._move(path)
            except TimeoutError as error:
                if self.clock() >= self.deadline:
                    raise
                self.vehicle.hover()
                self._event(
                    "FRONTIER_MOTION_BLOCKED",
                    reason=str(error),
                    local_status=self._guard.last_status,
                )
                return False
            finally:
                self._motion_progress = None
        return True

    def _move(self, path):
        self._check_health()
        goal = next_leg(path, max(2.4, self.config.leg_length))
        intermediate = np.linalg.norm(goal - path[-1]) > self.config.arrival_tolerance
        self.vehicle.move_velocity_guided(
            goal,
            face_direction=True,
            local_planner=self._guard,
            pass_through=bool(intermediate),
            timeout=max(
                0.01, min(self.config.leg_timeout, self.deadline - self.clock())
            ),
        )
        self._sample_position()

    def _choices(self, grid, current, origin):
        yaw = float(self.vehicle.get_euler()[2])
        excluded = {
            key
            for key, (count, until) in self.failed_frontiers.items()
            if count >= self.exploration_config.max_frontier_attempts
            or self.clock() < until
        }
        return self.frontiers.update(grid, current, origin, yaw, excluded)

    def run(self):
        print(
            "--- 探索策略：FUEL 思路前沿区域 / 观察点 / 访问顺序 + EGO 定高局部避障 ----"
        )
        self.started = self.clock()
        self.return_time_reserve = min(
            self.exploration_config.return_time_reserve,
            self.config.mission_timeout * 0.25,
        )
        exploration_budget = self.config.mission_timeout - self.return_time_reserve
        self.deadline = self.started + exploration_budget
        origin = self._sample_position()
        self.exploration_origin = origin.copy()
        self.altitude = float(origin[2])
        self.memory.flight_altitude = self.altitude
        self.memory.add_place("place_0000", origin)
        self.current_place = "place_0000"
        visit_trace = [("observe", "place_0000")]
        reason, completed, truncated = "initializing", False, False
        radius_limited = False
        try:
            warmup = min(self.deadline, self.clock() + self.config.perception_timeout)
            while self.perception.semantic_snapshot() is None and self.clock() < warmup:
                self.perception.raise_if_failed()
                self.vehicle.wait(0.2)
            self._full_scan("initial_map")
            self._record_coverage(self._grid())
            while self.clock() < self.deadline:
                self._check_health()
                current = self.vehicle.get_position()
                grid = self._grid()
                choices, limited, outside_radius = self._choices(grid, current, origin)
                truncated |= limited
                radius_limited |= outside_radius
                start_is_free = grid.cell(current) in grid.free
                print(
                    f"--- 前沿探索 可通行格={len(grid.free)}，候选={len(choices)}，"
                    f"当前位置可通行={start_is_free} ----"
                )
                if not start_is_free:
                    grid = self._recover_free_space()
                    if grid is None:
                        reason = "insufficient_free_space_evidence"
                        break
                    current = self.vehicle.get_position()
                    choices, limited, outside_radius = self._choices(
                        grid, current, origin
                    )
                    truncated |= limited
                    radius_limited |= outside_radius
                if len(self.memory.places) >= self.exploration_config.max_viewpoints:
                    reason = "viewpoint_budget_exhausted"
                    break

                if not choices:
                    bin_key = tuple(
                        np.floor(
                            (np.asarray(current)[:2] - origin[:2])
                            / self.exploration_config.viewpoint_spacing
                        ).astype(int)
                    )
                    used_yaws = self.recovery_yaws.get(bin_key, set())
                    if len(used_yaws) < self.exploration_config.recovery_observations:
                        if self._recovery_observation(grid, bin_key):
                            continue
                    pending = [
                        until
                        for count, until in self.failed_frontiers.values()
                        if count < self.exploration_config.max_frontier_attempts
                        and until > self.clock()
                    ]
                    if pending:
                        self._reset_empty_frontier_evidence()
                        self._wait(min(1.0, max(0.05, min(pending) - self.clock())))
                        continue
                    stable, empty_elapsed = self._confirm_empty_frontiers()
                    if not stable:
                        remaining = (
                            self.exploration_config.frontier_empty_min_duration
                            - empty_elapsed
                        )
                        self._wait(min(1.0, max(0.05, remaining)))
                        continue
                    frontier_area = self._remaining_frontier_area(grid)
                    self._last_frontier_area = frontier_area
                    recent_gain = self._recent_map_gain()
                    if truncated:
                        reason = "frontier_search_limit"
                    elif radius_limited:
                        reason = "radius_limit"
                    elif self.failed_frontiers:
                        reason = "frontiers_blocked"
                    else:
                        reason = "no_new_reachable_frontiers"
                    completed = reason == "no_new_reachable_frontiers"
                    if completed and self.frontiers.regions:
                        diminishing = (
                            frontier_area
                            <= self.exploration_config.completion_frontier_area
                            and recent_gain is not None
                            and recent_gain
                            <= self.exploration_config.minimum_map_gain_area
                        )
                        if diminishing:
                            reason = "diminishing_information_gain"
                        else:
                            reason, completed = "unobserved_frontiers", False
                    break

                key, goal = choices[0]
                self._reset_empty_frontier_evidence()
                self.active_viewpoint = self.frontiers.candidates.get(key)
                previous = self.current_place
                print(
                    f"--- 选择信息增益前沿：bin={key}，"
                    f"目标距离={np.linalg.norm(goal[:2] - current[:2]):.1f}m ----"
                )
                if not self._travel(goal):
                    self.blocked += 1
                    count = self.failed_frontiers.get(key, (0, 0))[0] + 1
                    self.failed_frontiers[key] = (
                        count,
                        self.clock() + self.exploration_config.frontier_retry_cooldown,
                    )
                    visit_trace.append(("blocked", self.frontiers.identity(key)))
                    # Replan from the current safe position rather than flying
                    # backwards merely to preserve a DFS tree.
                    stopped = self.vehicle.get_position()
                    previous_position = np.asarray(
                        self.memory.places[previous]["position"]
                    )
                    if (
                        np.linalg.norm(stopped[:2] - previous_position[:2])
                        > self.config.arrival_tolerance
                    ):
                        identity = f"place_{len(self.memory.places):04d}"
                        self.memory.add_place(identity, stopped, previous)
                        self.current_place = identity
                        visit_trace.append(("replan", identity))
                        self._observe_arrival()
                    else:
                        self.current_place = previous
                    self._checkpoint()
                    continue
                self.failed_frontiers.pop(key, None)
                identity = f"place_{len(self.memory.places):04d}"
                self.memory.add_place(identity, self.vehicle.get_position(), previous)
                self.current_place = identity
                visit_trace.append(("visit", identity))
                self._observe_arrival()
            else:
                reason = "time_budget_exhausted"
        except TimeoutError:
            if self.clock() < self.deadline:
                reason = "scan_or_motion_timeout"
                raise
            reason = "time_budget_exhausted"
        except Exception as error:
            reason = type(error).__name__
            raise
        except KeyboardInterrupt:
            reason = "interrupted"
            raise
        finally:
            self.result = {
                "strategy": "fuel_inspired_frontier",
                "frontiers_exhausted": completed,
                "reason": reason,
                "viewpoint_count": len(self.memory.places),
                "blocked_frontiers": self.blocked,
                "elapsed_seconds": self.clock() - self.started,
                "max_radius_m": self.exploration_config.max_radius,
                "return_time_reserve_seconds": self.return_time_reserve,
                "coverage_guaranteed": False,
                "visit_trace": visit_trace,
                "execution_events": self.events,
                "remaining_frontier_regions": len(self.frontiers.regions),
                "remaining_frontier_area_m2": self._last_frontier_area,
                "recent_map_gain_area_m2": self._recent_map_gain(),
                "planned_visit_order": [
                    self.frontiers.identity(k) for k in self.frontiers.visit_order
                ],
                "height_policy": "fixed_takeoff_plane",
                "motion_policy": "global_route_direct_ego_on_obstacle",
                "local_planner": "ego_avoidance_only",
                "returned_home": False,
            }
            self.memory.exploration = self.result.copy()
            print(f"--- 前沿探索结束：{reason}，观察点={len(self.memory.places)} ----")
            active_error = sys.exc_info()[0] is not None
            try:
                write_json(self.output_path, self.result)
                self._checkpoint()
            except Exception as error:
                if not active_error:
                    raise
                print(f"--- 探索报告保存失败：{error}；保留原始异常 ----")
        self.vehicle.hover()
        return completed


# Compatibility for older callers; the active implementation does not use DFS.
DFSExplorer = FrontierExplorer
