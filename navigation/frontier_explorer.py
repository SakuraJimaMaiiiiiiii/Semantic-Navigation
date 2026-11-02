"""Utility frontier decisions with existing guarded trajectory execution."""

from dataclasses import replace, asdict
import math
import sys

import numpy as np

from .mission import SemanticNavigator
from .coverage_progress import CoverageProgress
from .exploration_planner import FrontierExplorationPlanner, ExplorationPose, ExplorationHistory
from .route import NavigationGrid
from .waypoint_cursor import WaypointCursor
from .scene_memory import write_json
from .frontier_geometry import visible_unknown
from .utility_frontier_manager import UtilityFrontierManager


def _angle_difference(first, second):
    return abs((first - second + math.pi) % (2 * math.pi) - math.pi)


class FrontierExplorer(SemanticNavigator):
    """Utility waypoint selection with A* navigation and EGO local avoidance."""

    def __init__(self, *args, exploration_config, memory, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_config = exploration_config
        self.config = replace(
            self.config,
            mission_timeout=exploration_config.timeout,
            max_search_radius=exploration_config.max_radius,
            clearance_radius=max(
                self.config.clearance_radius,
                exploration_config.flight_clearance,
            ),
        )
        self.memory = memory
        self.motion_backend = "ego_avoidance_only"
        self.phase = "exploring"
        self.checkpoint = checkpoint
        self.current_place = None
        self.recovery_yaws = {}
        self._empty_frontier_since = None
        self._empty_frontier_confirmations = 0
        self._coverage_samples = []
        self.coverage_progress = CoverageProgress(
            exploration_config.minimum_map_gain_area,
            exploration_config.coverage_plateau_seconds,
            exploration_config.coverage_relative_gain,
        )
        self._coverage_review_attempts = 0
        self._last_frontier_area = None
        self.blocked = 0
        self.failed_frontiers = {}
        self._motion_progress = None
        self.frontiers = UtilityFrontierManager(exploration_config)
        self.active_viewpoint = None
        self.exploration_waypoints = []
        self.active_route = None
        self._inspected_vehicle_ids = set()
        self._selection_reason = "initial"
        self._decision_trace = []
        self._selection_trace = []
        self._arrival_frontiers = []
        self._travel_heading = None

    def _check_maps(self):
        try:
            return super()._check_maps()
        except RuntimeError as error:
            if str(error) not in {'semantic_local_map_stale', 'semantic_global_map_stale'}:
                raise
            # A transient mapping delay must not trigger immediate landing.
            # Hold once, keep all freshness thresholds, and bound the recovery.
            self.vehicle.hover()
            self._event('MAP_UPDATE_WAIT', reason=str(error))
            end = self.clock() + self.config.map_timeout
            if getattr(self, 'phase', 'exploring') == 'exploring':
                end = min(self.deadline, end)
            while self.clock() < end:
                self.vehicle.wait(min(.2, end-self.clock()))
                self._last_health_map_check = -float('inf')
                try:
                    super()._check_maps()
                except RuntimeError as pending:
                    if str(pending) not in {'semantic_local_map_stale', 'semantic_global_map_stale'}:
                        raise
                    continue
                self._event('MAP_UPDATE_RESUMED')
                if self._motion_progress is not None:
                    self._motion_progress = self.vehicle.get_position().copy(), self.clock()
                return
            raise error

    def vehicle_observation_speed_limit(self):
        if self.clock() <= getattr(self, '_vehicle_observation_until', -float('inf')):
            return self.local_planner.config.cruise_max_speed * 0.6
        return float('inf')

    def _check_health(self):
        # Semantic recognition may take seconds; transit safety comes from
        # the independently refreshed LiDAR/depth maps and local planner.
        super()._check_health(require_semantics=False)
        frame = self.perception.semantic_snapshot()
        if frame is not None and self.clock() - frame.completed_at <= self.config.perception_timeout:
            self._frame()
            if getattr(self, '_semantic_deferred', False):
                self._semantic_deferred = False
                self._event('SEMANTIC_RESUMED')
        elif not getattr(self, '_semantic_deferred', False):
            self._semantic_deferred = True
            self._event('SEMANTIC_DEFERRED', reason='recognition_not_fresh')
        if self._motion_progress is not None:
            position, changed = self._motion_progress
            current = self.vehicle.get_position()
            displacement = np.linalg.norm(current[:2] - position[:2])
            control = getattr(self.vehicle, 'waypoint_control_state', None)
            now = self.clock()
            if getattr(self, '_progress_pause_anchor', None) != changed:
                self._progress_pause_anchor = changed
                self._progress_pause_credit = 0.
                self._progress_last_check = changed
            if isinstance(control, dict):
                waiting = (control.get('plan_status', control.get('status')) == 'waiting_for_plan'
                           or (control.get('status') == 'holding_for_stable_clearance'
                               and control.get('speed_limit', 0.) > 0.))
                if waiting:
                    self._progress_pause_credit = min(self.exploration_config.progress_timeout,
                        self._progress_pause_credit + max(0., now-self._progress_last_check))
            self._progress_last_check = now
            advancing = False
            if isinstance(control, dict) and control.get('status') in ('route_clear', 'ego_avoid', 'ego_clear'):
                target = np.asarray(control.get('target_ned'), dtype=float)
                if target.shape == (3,) and np.isfinite(target).all():
                    direction = target[:2] - current[:2]
                    distance = np.linalg.norm(direction)
                    displacement_vector = current[:2] - position[:2]
                    # Measured progress toward a clear local goal counts even
                    # if a noisy velocity sample happens to be near zero.
                    advancing = (displacement >= self.exploration_config.progress_distance / 3
                                 and distance > 1e-6
                                 and np.dot(displacement_vector, direction / distance)
                                 >= self.exploration_config.progress_distance / 3)
                    if distance <= min(.1, self.config.arrival_tolerance):
                        waypoint = tuple(target)
                        if waypoint != getattr(self, '_progress_completed_waypoint', None):
                            self._progress_completed_waypoint = waypoint
                            advancing = True
            if displacement >= self.exploration_config.progress_distance or advancing:
                self._motion_progress = current.copy(), self.clock()
            elif self.clock() - changed - self._progress_pause_credit >= self.exploration_config.progress_timeout:
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
        self._vehicle_observation_until = (
            frame.completed_at + self.config.perception_timeout
            if getattr(frame, 'vehicle_observation_pending', False) else -float('inf'))
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
                "strategy": "cluster_utility",
                "exploration_planner": self.frontiers.backend_name,
                "motion_policy": "committed_waypoints_astar_ego",
                "local_planner": self.motion_backend,
                "height_policy": "bounded_clearance_checked_ascent",
                "height_gain_m": float(self._nominal_altitude-self.altitude),
                "state": "exploring",
                "viewpoint_count": len(self.memory.places),
                "blocked_frontiers": self.blocked,
                "execution_events": list(self.events),
                "coverage_progress": dict(self.coverage_progress.metrics),
                "decision_config": asdict(self.exploration_config.decision),
                "decision_trace": list(self._decision_trace),
                "returned_home": False,
            }
        if self.checkpoint is not None:
            self.checkpoint(self)

    def _initial_observation(self):
        print(f"--- Utility 初始前向观测：{self.current_place} ----")
        self._frame()
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
        previous_progress = self._motion_progress
        self._motion_progress = None
        try:
            self._align(yaw)
            self._wait(self.exploration_config.recovery_observation_wait)
            self._frame()
            self._record_coverage(self._grid())
            self._checkpoint()
        finally:
            self._motion_progress = ((self.vehicle.get_position().copy(), self.clock())
                                     if previous_progress is not None else None)
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

    def _update_coverage(self, grid):
        visits = sum(w["status"] == "arrived" for w in self.exploration_waypoints)
        old_stage = self.coverage_progress.metrics.get("stage")
        plateau = self.coverage_progress.update(grid, self.altitude, self.clock(), visits)
        progress = self.coverage_progress.metrics
        changed = old_stage != progress["stage"]
        if changed:
            self._event("COVERAGE_STATE", **progress)
        from .terminal_progress import shared_progress
        display = getattr(self.vehicle, '_motion_progress_display', None) or shared_progress()
        display.update_coverage(progress)
        return plateau

    def _coverage_review(self, grid, current, origin):
        """At rest, search deferred batches for genuinely unvisited targets."""
        arrived = [np.asarray(w["position"]) for w in self.exploration_waypoints
                   if w["status"] == "arrived"]
        spacing = self.exploration_config.viewpoint_spacing
        limited = outside = False
        decision = self.frontiers
        decision._cycle_token = None
        choices, limited, outside = self._choices(grid, current, origin)
        while True:
            self._check_health()
            novel = [(key, goal) for key, goal in choices
                     if all(np.linalg.norm(np.asarray(goal)[:2]-p[:2]) > spacing for p in arrived)]
            if novel:
                # Review spends its finite attempts on actual unseen area,
                # not tiny one-cell gaps preferred by heading continuity.
                gains = {key: len(decision.candidates[key].visible_unknown) for key, _ in novel}
                best_gain = max(gains.values())
                novel = [(key, goal) for key, goal in novel
                         if gains[key] >= max(1, .5 * best_gain)]
            if novel:
                attempts = getattr(self, "_coverage_review_attempts", 0)
                if attempts < self.exploration_config.coverage_review_attempts:
                    self._coverage_review_attempts = attempts + 1
                    self._event("COVERAGE_REVIEW_CONTINUE", unvisited_candidates=len(novel),
                                attempt=self._coverage_review_attempts)
                    return novel, False, limited, outside
                # Candidate jitter cannot buy unlimited extra flights. This is
                # stagnation, not a declaration that remaining targets are covered.
                self._event("COVERAGE_REVIEW_BUDGET_EXHAUSTED", attempts=attempts)
                return [], True, limited, outside
            if not decision.batch_pending:
                break
            choices, batch_limited, batch_outside = self._choices(grid, current, origin)
            limited |= batch_limited
            outside |= batch_outside
        self._event("COVERAGE_REVIEW_EXHAUSTED", **self.coverage_progress.metrics)
        return [], True, limited, outside

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

    def _local_recovery_path(self, global_grid, snapshot, goal=None):
        """Use fresh local evidence only; never erase a global occupied voxel."""
        local = NavigationGrid.from_local(snapshot, self.altitude, self.config)
        current = self.vehicle.get_position()
        if local.cell(current) not in local.free:
            return None
        if goal is not None:
            return local.path(current, goal) if global_grid.cell(goal) in global_grid.free else None
        candidates = []
        seen = set()
        for cell in local.free:
            global_cell = global_grid.cell(local.point(cell))
            if global_cell in seen or global_cell not in global_grid.free:
                continue
            seen.add(global_cell)
            # Aim at the global cell center, not a local cell near its edge.
            # Otherwise even a small arrival tolerance can stop outside it.
            point = global_grid.point(global_cell)
            if local.cell(point) not in local.free:
                continue
            distance = float(np.linalg.norm(point[:2]-current[:2]))
            if (max(.8, 2*self.config.arrival_tolerance) <= distance <= 2.4
                    and global_grid.cell(point) in global_grid.free
                    and np.linalg.norm(point[:2]-self.exploration_origin[:2]) <= self.exploration_config.max_radius):
                candidates.append((distance, point))
        for _, point in sorted(candidates, key=lambda item:item[0])[:8]:
            path = local.path(current,point)
            if path is not None and sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:])) <= 4.8:
                return path
        return None

    def _recover_free_space(self):
        """Separate evidence waiting from a bounded, continuously guarded escape."""
        self.vehicle.hover()
        end = min(self.deadline, self.clock() + self.exploration_config.free_space_recovery_timeout)
        grid = self._grid()
        if grid.cell(self.vehicle.get_position()) in grid.free:
            return grid
        self._guard._diagnostic_context = None
        self._guard.capture_failure("global_start_not_free")
        clear_frames, last_stamp, recovery_goal = 0, None, None
        previous_place = getattr(self, "current_place", None)
        previous_progress = getattr(self, "_motion_progress", None)
        self._event("FREE_SPACE_RECOVERY_WAIT")
        try:
            while self.clock() < end:
                self._wait(min(0.2, end-self.clock()))
                self._frame()
                grid = self._grid()
                if grid.cell(self.vehicle.get_position()) in grid.free:
                    self._event("FREE_SPACE_EVIDENCE_RECOVERED",
                                via="local_certified_path" if recovery_goal is not None else "map_update")
                    return grid
                snapshot = self.local_map.snapshot()
                if snapshot.timestamp == last_stamp:
                    continue
                last_stamp = snapshot.timestamp
                path = self._local_recovery_path(grid, snapshot, recovery_goal)
                clear_frames = clear_frames+1 if path is not None else 0
                if path is None and recovery_goal is not None:
                    self._event("LOCAL_RECOVERY_BLOCKED", reason="local_path_no_longer_clear")
                    return None
                if clear_frames < 2:
                    continue
                if recovery_goal is None:
                    recovery_goal = np.asarray(path[-1]).copy()
                    # A 5 s map wait is not a flight budget. Use the existing
                    # leg budget once a safe route has been certified twice.
                    end = min(self.deadline, self.clock() + self.config.leg_timeout)
                    self._motion_progress = self.vehicle.get_position().copy(), self.clock()
                    self._event("LOCAL_RECOVERY_REPOSITION", distance_m=float(
                        np.linalg.norm(recovery_goal[:2]-self.vehicle.get_position()[:2])))
                self.current_place = None
                self._move(path, recovery=True)
            self._event("LOCAL_RECOVERY_TIMEOUT",
                        phase="motion" if recovery_goal is not None else "observation")
            return None
        except TimeoutError as error:
            if self.clock() >= self.deadline:
                raise
            self._event("LOCAL_RECOVERY_BLOCKED", reason=str(error))
            return None
        finally:
            self.current_place = previous_place
            self._motion_progress = previous_progress
            self.vehicle.hover()

    def _observe_arrival(self):
        # Semantics are acquired throughout flight. A routine visit must not
        # stop and rotate toward a sampled view before choosing the next leg.
        self._frame()
        self._record_coverage(self._grid())
        self._checkpoint()
        self._inspect_nearby_vehicle()

    def _inspect_nearby_vehicle(self):
        from .vehicle_inspection import needs_inspection, observation_route
        frame = self._frame()
        current = self.vehicle.get_position()
        records = [r for r in frame.object_inspection_candidates
                   if r['runtime_instance_id'] not in self._inspected_vehicle_ids
                   and not self._inspected_vehicle_ids.intersection(r.get('merged_from', []))
                   and frame.timestamp-r['last_seen'] <= self.config.perception_timeout
                   and needs_inspection(r, self.config.clearance_radius)
                   and np.linalg.norm(np.asarray(r['position_world'])[:2]-current[:2])
                   <= self.exploration_config.information_radius + self.exploration_config.candidate_radius_max]
        if not records:
            return
        record = min(records, key=lambda r: np.linalg.norm(np.asarray(r['position_world'])-current))
        identity = record['runtime_instance_id']
        self._inspected_vehicle_ids.add(identity)
        original_deadline, previous_progress = self.deadline, self._motion_progress
        self.deadline = min(original_deadline, self.clock()+self.exploration_config.vehicle_inspection_timeout)
        self._motion_progress = None
        used_yaws = []
        waypoint = None
        try:
            for _ in range(self.exploration_config.vehicle_inspection_views):
                self._check_health()
                current = self.vehicle.get_position()
                route = observation_route(self._grid(), current, record, self.altitude,
                                          self.config.standoff_distance, used_yaws)
                if route is None:
                    break
                _, goal, path, angle = route
                if np.linalg.norm(goal[:2]-self.exploration_origin[:2]) > self.exploration_config.max_radius:
                    break
                used_yaws.append(angle)
                waypoint = dict(number=len(self.exploration_waypoints)+1, frontier=None,
                                position=goal.tolist(), yaw=angle+math.pi, score=0.,
                                source='vehicle_inspection', target_id=identity, status='active')
                self.exploration_waypoints.append(waypoint)
                self.active_route = path
                self._waypoint_cursor = WaypointCursor()
                self._checkpoint()
                while not self._goal_reached(goal):
                    self._check_health()
                    path = self._grid().path(self.vehicle.get_position(), goal)
                    if path is None:
                        waypoint['status'] = 'blocked'
                        return
                    self._move(path)
                waypoint['status'] = 'arrived'
                place = f"place_{len(self.memory.places):04d}"
                self.memory.add_place(place, self.vehicle.get_position(), self.current_place)
                self.current_place = place
                self._align(math.atan2(record['position_world'][1]-goal[1], record['position_world'][0]-goal[0]))
                baseline = self._frame().timestamp
                observation_end = min(self.deadline, self.clock()+self.config.perception_timeout)
                while self.clock() < observation_end:
                    self._wait(.2)
                    frame = self._frame()
                    if frame.timestamp > baseline:
                        break
                updated = next((r for r in frame.object_inspection_candidates
                                if r['runtime_instance_id'] == identity or identity in r.get('merged_from', [])), None)
                if updated is not None:
                    record = updated
                    self._inspected_vehicle_ids.add(record['runtime_instance_id'])
                    if record['last_seen'] > baseline and not needs_inspection(record, self.config.clearance_radius):
                        break
                self._checkpoint()
        except TimeoutError:
            if waypoint is not None and waypoint['status'] == 'active':
                waypoint['status'] = 'inspection_timeout'
        finally:
            self.deadline = original_deadline
            self._motion_progress = previous_progress
            self.active_route = None
            self._waypoint_cursor = WaypointCursor()
            self.vehicle.hover()
            self._checkpoint()

    def _grid(self):
        self._check_maps()
        return NavigationGrid.from_global(
            self.global_map.snapshot(), self.altitude, self.config,
            local_snapshot=self.local_map.snapshot(),
            current=self.vehicle.get_position(),
        )

    def _goal_reached(self, goal):
        delta = np.asarray(goal) - self.vehicle.get_position()
        # Do not interrupt the controller's smooth arrival with hover() as
        # soon as a moving vehicle enters the larger exploration radius.
        return (np.linalg.norm(delta[:2]) <= min(.1, self.config.arrival_tolerance)
                and abs(delta[2]) <= min(.1, self.config.arrival_tolerance)
                and np.linalg.norm(self.vehicle.get_velocity()) <= .05)

    def _height_unavailable(self, reason, current):
        key = (reason, *np.floor(np.asarray(current)[:2]).astype(int))
        if getattr(self, "_height_rejection", None) != key:
            self._height_rejection = key
            self._event("HEIGHT_RECOVERY_UNAVAILABLE", reason=reason)
        return False

    def _try_raise_for_obstacle(self, goal):
        from .height_clearance import vertical_corridor_free
        current = self.vehicle.get_position()
        source = self.local_map.snapshot()
        occupied = np.argwhere(source.states == 1)
        if not len(occupied):
            return False
        points = source.origin_ned + (occupied+.5)*source.resolution
        direction = np.asarray(goal)[:2] - current[:2]
        distance = np.linalg.norm(direction)
        if distance < self.config.arrival_tolerance:
            return False
        direction = direction / distance
        delta = points[:, :2] - current[:2]
        forward = delta @ direction
        lateral = np.abs(delta[:, 0]*direction[1] - delta[:, 1]*direction[0])
        corridor = ((forward > 0) & (forward < min(distance, self.config.clearance_radius + .8))
                    & (lateral <= self.config.clearance_radius + source.resolution*.5))
        half_voxel = source.resolution*.5
        body = ((points[:, 2]+half_voxel >= self.altitude-self.config.vertical_half_extent)
                & (points[:, 2]-half_voxel <= self.altitude+self.config.vertical_half_extent))
        blocking = corridor & body
        if not blocking.any():
            return False
        lower_bound = self._nominal_altitude-self.exploration_config.max_height_gain
        proposed = max(lower_bound,self.altitude-self.exploration_config.height_step)
        if self.altitude-proposed < .1:
            return self._height_unavailable("height_limit", current)
        # Only climb over low obstacles. Walls and overhead beams must not
        # trigger repeated ascents toward the ceiling (NED z increases down).
        if not np.all(points[blocking, 2]-half_voxel > proposed+self.config.vertical_half_extent):
            return self._height_unavailable("obstacle_not_below_upper_band", current)
        target = current.copy()
        target[2] = proposed
        if not vertical_corridor_free(source,current,target,self.config.clearance_radius,self.config.vertical_half_extent):
            return self._height_unavailable("vertical_corridor_unknown_or_blocked", current)
        elevated = NavigationGrid.from_global(self.global_map.snapshot(),proposed,self.config)
        raised_goal = np.asarray(goal).copy()
        raised_goal[2] = proposed
        if elevated.path(target,raised_goal) is None:
            return self._height_unavailable("upper_route_unavailable", current)
        self.vehicle.hover()
        self._event("HEIGHT_RECOVERY_START", rise_m=float(self.altitude-proposed))
        end = min(self.deadline,self.clock()+self.config.leg_timeout)
        moved = False
        previous_progress = self._motion_progress
        self._motion_progress = None
        try:
            # Do not combine an ascent with residual horizontal flight.
            while np.linalg.norm(self.vehicle.get_velocity()[:2]) > .05:
                self._check_health()
                if self.clock() >= end:
                    return False
                self._wait(.1)
            while self.clock() < end:
                self._check_health()
                position = self.vehicle.get_position()
                if abs(position[2]-proposed) <= .05:
                    break
                # Drifted position gets a new complete vertical-corridor check.
                target[:2] = position[:2]
                if not vertical_corridor_free(self.local_map.snapshot(),position,target,
                                              self.config.clearance_radius,self.config.vertical_half_extent):
                    self._event("HEIGHT_RECOVERY_BLOCKED")
                    break
                self.vehicle.raise_if_collision(include_ground=True)
                self.vehicle.send_velocity([0.,0.,float(np.clip(proposed-position[2],-.2,.2))])
                moved = True
                self.vehicle.wait(.05)
        finally:
            self.vehicle.hover()
            self._motion_progress = (
                (self.vehicle.get_position().copy(),self.clock())
                if previous_progress is not None else None
            )
            if moved:
                # Even an interrupted ascent must plan from the actual level.
                self.altitude = float(self.vehicle.get_position()[2])
                self.memory.flight_altitude = self.altitude
                self._guard._grid_stamp = None
                self._event("HEIGHT_RECOVERY_FINISHED", rise_m=float(self._nominal_altitude-self.altitude))
        return moved

    def _nearby_valid_viewpoint(self, grid, current, goal):
        """Repair one invalid endpoint without abandoning its observed frontier."""
        view = getattr(self, "active_viewpoint", None)
        if view is None or not view.visible_unknown or grid.cell(current) not in grid.free:
            return None
        radius = min(self.exploration_config.viewpoint_spacing, 2*grid.resolution)
        candidates = sorted(
            (float(np.linalg.norm(grid.point(cell)[:2]-goal[:2])), cell)
            for cell in grid.free
            if np.linalg.norm(grid.point(cell)[:2]-goal[:2]) <= radius+1e-9
        )
        for _, cell in candidates[:8]:
            point = grid.point(cell)
            if np.linalg.norm(point[:2]-self.exploration_origin[:2]) > self.exploration_config.max_radius:
                continue
            visible = visible_unknown(grid, cell, view.yaw,
                                      self.exploration_config.information_radius,
                                      self.exploration_config.camera_fov_degrees)
            if not (visible & view.visible_unknown):
                continue
            path = grid.path(current, point)
            if path is not None:
                return point, path, visible
        return None

    def _travel(self, goal):
        self._last_travel_failure = None
        start_recovery_used = False
        endpoint_repaired = False
        map_revision = getattr(self.global_map, "revision", 0)
        self.current_place = None
        self._motion_progress = self.vehicle.get_position().copy(), self.clock()
        self._progress_completed_waypoint = None
        path = None
        try:
            while not self._goal_reached(goal):
                self._check_health()
                grid = self._grid()
                revision = getattr(self.global_map, "revision", 0)
                if revision != map_revision:
                    path = None
                    self._waypoint_cursor = WaypointCursor()
                    map_revision = revision
                    self._event("LOOP_MAP_REPLAN", revision=revision)
                if getattr(self, "coverage_progress", None) is not None:
                    self._update_coverage(grid)
                    # Known-space transit can outlast the growth window.
                    # Review coverage at arrival, not halfway to a new frontier;
                    # mission and motion timeouts still bound this flight.
                grid.free = {
                    cell
                    for cell in grid.free
                    if np.linalg.norm(
                        grid.point(cell)[:2] - self.exploration_origin[:2]
                    )
                    <= self.exploration_config.max_radius
                }
                current = self.vehicle.get_position()
                # Decision strategies only supply committed exploration goals.
                # Map churn/centroid yaw must not redirect a moving vehicle.
                # Keep a valid route across rolling horizons instead of letting
                # equal-cost A* alternatives alternate left/right every cycle.
                if path is not None:
                    while (len(path)>2 and np.linalg.norm(current[:2]-path[1][:2]) <= self.config.arrival_tolerance
                           and abs(current[2]-path[1][2]) <= self.config.arrival_tolerance
                           and grid.segment_free(current,path[2])):
                        path = [current,*path[2:]]
                    path = [current,*path[1:]]
                    if not all(grid.segment_free(a,b) for a,b in zip(path,path[1:])):
                        path = None
                if path is None:
                    self._hold_for_planning()
                    current = self.vehicle.get_position()
                    grid = self._grid()
                    path = grid.path(current, goal)
                    if path is None and grid.cell(goal) not in grid.free and not endpoint_repaired:
                        repaired = self._nearby_valid_viewpoint(grid, current, goal)
                        if repaired is not None:
                            endpoint_repaired = True
                            original_goal = goal.copy()
                            point, path, visible = repaired
                            goal[:] = point
                            self.active_viewpoint = replace(self.active_viewpoint, point=point.copy(),
                                                            visible_unknown=frozenset(visible))
                            if self.exploration_waypoints:
                                record = self.exploration_waypoints[-1]
                                record.setdefault("original_position", original_goal.tolist())
                                record["position"] = point.tolist()
                                record["adjustment_reason"] = "goal_clearance_changed"
                            self._event("FRONTIER_ENDPOINT_ADJUSTED", original_ned=original_goal.tolist(),
                                        goal_ned=point.tolist())
                    self.active_route = path
                    if path is not None:
                        print(f"--- A* 探索路线已生成：{len(path)-1} 个航点，按顺序执行；EGO负责局部避障 ----")
                    if getattr(self, "checkpoint", None) is not None:
                        self.checkpoint(self)
                if path is None:
                    if grid.cell(current) not in grid.free and not start_recovery_used:
                        start_recovery_used = True
                        recovered = self._recover_free_space()
                        if recovered is not None:
                            self._motion_progress = self.vehicle.get_position().copy(), self.clock()
                            # A temporary invalid start is not a failed destination.
                            # Retry this committed waypoint after certified recovery.
                            continue
                        current = self.vehicle.get_position()
                        grid = self._grid()
                    # A blocked straight line to the final goal does not imply
                    # a blocked horizontal route. Try altitude recovery only
                    # after the current-height A* search has actually failed.
                    if grid.cell(current) in grid.free and self._try_raise_for_obstacle(goal):
                        goal = np.asarray(goal).copy()
                        goal[2] = self.altitude
                        self._motion_progress = self.vehicle.get_position().copy(), self.clock()
                        continue
                    self.vehicle.hover()
                    self._last_travel_failure = {
                        "reason": ("start_not_free" if grid.cell(current) not in grid.free
                                   else "goal_not_free" if grid.cell(goal) not in grid.free
                                   else "astar_no_route"),
                        "current_ned": np.asarray(current).tolist(),
                        "goal_ned": np.asarray(goal).tolist(),
                    }
                    self._event("FRONTIER_PATH_BLOCKED", **self._last_travel_failure)
                    from .mission import _FlightGuard
                    if isinstance(self._guard, _FlightGuard):
                        self._guard._diagnostic_context = (
                            self.local_map.snapshot(), self.global_map.snapshot(), current, goal)
                        self._guard.capture_failure("frontier_path_blocked", self._last_travel_failure)
                    return False
                continuous = len(path) > 2 and grid.segment_free(current, path[2])
                self._move(path, continuous=continuous)

        except TimeoutError as error:
            if self.clock() >= self.deadline:
                raise
            self.vehicle.hover()
            current = self.vehicle.get_position()
            progress = self._motion_progress
            self._last_travel_failure = {
                "reason": str(error), "local_status": self._guard.last_status,
                "current_ned": current.tolist(), "goal_ned": np.asarray(goal).tolist(),
                "controller": getattr(self.vehicle, "waypoint_control_state", None),
                "displacement_since_progress_m": None if progress is None else float(np.linalg.norm(current[:2]-progress[0][:2])),
                "seconds_since_progress": None if progress is None else float(self.clock()-progress[1]),
            }
            self._event(
                "FRONTIER_MOTION_BLOCKED",
                reason=str(error),
                local_status=self._guard.last_status,
            )
            return False
        finally:
            self._motion_progress = None
        self._hold_for_planning()
        return True

    def _hold_for_planning(self):
        """Stop old velocity/yaw commands before synchronous, expensive planning."""
        self.vehicle.hover()
        previous = self._motion_progress
        self._motion_progress = None
        end = min(self.deadline,self.clock()+self.config.leg_timeout)
        try:
            while np.linalg.norm(self.vehicle.get_velocity()[:2]) > .05:
                self._check_health()
                if self.clock() >= end:
                    raise TimeoutError("waypoint_braking_timeout")
                self.vehicle.wait(.05)
        finally:
            self._motion_progress = ((self.vehicle.get_position().copy(),self.clock())
                                     if previous is not None else None)
        return self.vehicle.get_position()

    def _move(self, path, *, recovery=False, continuous=False):
        self._check_health()
        current = self.vehicle.get_position()
        # Follow the first A* segment, never a chord spanning several corners.
        points = [np.asarray(p) for p in path[1:] if np.linalg.norm(np.asarray(p)-current)>1e-6]
        if not points:
            return
        endpoint = points[0]
        if not hasattr(self, "_waypoint_cursor"):
            self._waypoint_cursor = WaypointCursor()
        tolerance = min(0.05 if recovery else 0.1, self.config.arrival_tolerance)
        goal, travel_yaw = self._waypoint_cursor.select(
            current, endpoint, max(2.4, self.config.leg_length), tolerance,
            pass_radius=None if recovery else self.config.arrival_tolerance,
            z_tolerance=getattr(getattr(self.vehicle, "config", None), "guided_z_tolerance", tolerance)
        )
        current_yaw = float(self.vehicle.get_euler()[2])
        held_yaw = getattr(self,"_travel_heading",None)
        if held_yaw is None:
            held_yaw = current_yaw
        if recovery and _angle_difference(travel_yaw,held_yaw)>math.radians(15):
            self._hold_for_planning()
            # Rotation is allowed only at a currently certified body footprint.
            local = NavigationGrid.from_local(self.local_map.snapshot(),self.altitude,self.config,
                                             self.global_map.snapshot())
            if local.cell(self.vehicle.get_position()) not in local.free:
                raise TimeoutError("waypoint_turn_without_clearance")
            previous_progress = self._motion_progress
            self._motion_progress = None
            try:
                self._align(travel_yaw)
            finally:
                self._motion_progress = (
                    (self.vehicle.get_position().copy(), self.clock())
                    if previous_progress is not None else None
                )
            held_yaw = travel_yaw
        if not recovery:
            held_yaw = travel_yaw
        self._travel_heading = held_yaw
        pass_through = not recovery and (self._waypoint_cursor.intermediate or continuous)
        self.vehicle.move_waypoint_guided(
            goal, yaw=held_yaw, face_direction=False, local_planner=self._guard,
            # Slow into bends instead of cruising through a corner and trying
            # to brake only after changing the next waypoint's direction.
            pass_through=pass_through,
            xy_tolerance=(self.config.arrival_tolerance
                          if pass_through else tolerance),
            # A short local segment is continuously revalidated by the
            # controller; avoid rebuilding the global grid every 1.5 s while
            # that segment is still being executed.
            control_horizon=max(self.exploration_config.rolling_replan_period,
                                self.config.leg_timeout / 3),
            timeout=max(0.01,min(self.config.leg_timeout,self.deadline-self.clock())),
        )
        self._sample_position()

    def _planner_context(self, current, origin):
        excluded = {key for key, (count, until) in self.failed_frontiers.items()
                    if count >= self.exploration_config.max_frontier_attempts or self.clock() < until}
        pose = ExplorationPose(np.asarray(current), float(self.vehicle.get_euler()[2]),
                               travel_heading=getattr(self, "_travel_heading", None))
        return pose, ExplorationHistory(np.asarray(origin), excluded)

    def _decision_planner(self):
        if (not hasattr(self, "exploration_planner")
                or self.exploration_planner.manager is not self.frontiers):
            self.exploration_planner = FrontierExplorationPlanner(self.frontiers)
        return self.exploration_planner

    def _choices(self, grid, current, origin):
        held = self._hold_for_planning()
        if held is not None:
            current = held
        pose, history = self._planner_context(current, origin)
        planner = self._decision_planner()
        choices = planner.prepare_candidates(grid, pose, history)
        decision = self.frontiers
        if not choices and self.clock()-getattr(self, "_last_empty_diagnostics", -float("inf")) >= 10.:
            self._last_empty_diagnostics=self.clock()
            print(f"--- 空候选诊断：{decision.diagnostics} ----")
        live_ids = {cluster.id for cluster in decision.clusters}
        self.failed_frontiers = {key:value for key,value in self.failed_frontiers.items() if key[0] in live_ids}
        return choices, planner.search_limited, planner.outside_radius

    def _log_decision(self, reason, view, old_key=None):
        decision = self.frontiers
        entry = dict(elapsed=self.clock()-self.started, reason=reason,
                     selected=None if view is None else self.frontiers.identity(view.key),
                     switched=bool(old_key is not None and (view is None or old_key != view.key)),
                     **decision.diagnostics)
        if view is not None:
            entry.update(decision.scores[view.key])
        self._decision_trace.append(entry)
        self._event("FRONTIER_DECISION", **{k:v for k,v in entry.items() if k != "elapsed"})

    def run(self):
        print("--- 探索决策：前方安全走廊优先推进，无前向候选时按Utility评分换向；探索航点锁定执行；A*导航+EGO局部避障；"
              f"轨迹后端={self.motion_backend}；安全检查与控制器保持原实现 ----")
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
        self._nominal_altitude = self.altitude
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
            self._initial_observation()
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
                    current = self._sample_position()
                    previous_position = np.asarray(self.memory.places[self.current_place]["position"])
                    if np.linalg.norm(current[:2]-previous_position[:2]) > self.config.arrival_tolerance:
                        identity = f"place_{len(self.memory.places):04d}"
                        self.memory.add_place(identity,current,self.current_place)
                        self.current_place = identity
                        visit_trace.append(("recovery",identity))
                        self._checkpoint()
                    choices, limited, outside_radius = self._choices(
                        grid, current, origin
                    )
                    truncated |= limited
                    radius_limited |= outside_radius
                if len(self.memory.places) >= self.exploration_config.max_viewpoints:
                    reason = "viewpoint_budget_exhausted"
                    break

                # Coverage stopping also applies while residual/repeated candidates exist.
                visits = sum(w["status"] == "arrived" for w in self.exploration_waypoints)
                plateau = self._update_coverage(grid)
                # Low growth requests a frontier review. Waiting or traversing
                # already-known space must not bypass remaining review attempts.
                if (self.coverage_progress.metrics["stop_due"] and not choices
                        and not self.frontiers.batch_pending):
                    reason = "map_expansion_stalled"
                    break
                if choices:
                    progress = self.coverage_progress.metrics
                    print(f"--- 地图扩展：已知={progress['observed_area_m2']:.1f}m²，"
                          f"窗口={progress['window_seconds']:.0f}s，"
                          f"折算新增={progress['normalized_window_gain_m2']:.2f}m²，"
                          f"低增长门槛={progress['low_gain_threshold_m2']:.2f}m²，"
                          f"成功到达={visits}，低增长复查={plateau} ----")
                if not plateau:
                    self._coverage_review_attempts = 0
                if choices and plateau:
                    choices, exhausted, limited, outside = self._coverage_review(
                        grid, self.vehicle.get_position(), origin)
                    truncated |= limited
                    radius_limited |= outside
                    if exhausted:
                        bin_key = tuple(np.floor((current[:2]-origin[:2]) /
                                                 self.exploration_config.viewpoint_spacing).astype(int))
                        if (len(self.recovery_yaws.get(bin_key, set())) < self.exploration_config.recovery_observations
                                and self._recovery_observation(grid, bin_key)):
                            continue
                        self._last_frontier_area = self._remaining_frontier_area(grid)
                        # Unfinished or blocked boundaries are never called complete.
                        completed = (self._coverage_review_attempts < self.exploration_config.coverage_review_attempts
                                     and self._last_frontier_area <= self.exploration_config.completion_frontier_area
                                     and not truncated and not radius_limited and not self.failed_frontiers)
                        reason = "map_coverage_saturated" if completed else "map_expansion_stalled"
                        break

                if not choices:
                    decision = self.frontiers
                    if decision.batch_pending:
                        # A batch with no candidate is not an exhausted map.
                        self._wait(0.05)
                        continue
                    bin_key = tuple(
                        np.floor(
                            (np.asarray(current)[:2] - origin[:2])
                            / self.exploration_config.viewpoint_spacing
                        ).astype(int)
                    )
                    used_yaws = self.recovery_yaws.get(bin_key, set())
                    if (len(used_yaws) < self.exploration_config.recovery_observations
                            and self._recovery_observation(grid, bin_key)):
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

                decision = self.frontiers
                planner = self._decision_planner()
                pose, history = self._planner_context(self.vehicle.get_position(), origin)
                selected_waypoint = planner.select_next_waypoint(grid, pose, history, candidates=choices)
                if selected_waypoint is None:
                    continue
                key = planner.selected_key
                goal = np.asarray(selected_waypoint["position"], dtype=float)
                self._log_decision(self._selection_reason, self.frontiers.candidates[key])
                self._reset_empty_frontier_evidence()
                self.active_viewpoint = self.frontiers.candidates.get(key)
                delta = np.asarray(goal)[:2]-current[:2]
                self._selection_trace.append(dict(
                    elapsed=self.clock()-self.started, frontier=self.frontiers.identity(key),
                    travel_heading=float(math.atan2(delta[1],delta[0])),
                    yaw_change=(None if self.active_viewpoint is None else _angle_difference(
                        self.active_viewpoint.yaw,float(self.vehicle.get_euler()[2])))))
                previous = self.current_place
                source = selected_waypoint["source"]
                print(
                    f"--- 选择探索目标：来源={source}，bin={key}，"
                    f"目标距离={np.linalg.norm(goal[:2] - current[:2]):.1f}m ----"
                )
                waypoint = {
                    "number": len(self.exploration_waypoints) + 1,
                    "frontier": self.frontiers.identity(key),
                    **selected_waypoint,
                    "status": "active",
                }
                self.exploration_waypoints.append(waypoint)
                self.active_route = None
                self._checkpoint()
                try:
                    outcome = self._travel(goal)
                except BaseException:
                    waypoint["status"] = "interrupted"
                    self.active_route = None
                    raise
                waypoint["status"] = "arrived" if outcome else "blocked"
                self.active_route = None
                if not outcome:
                    waypoint["failure"] = getattr(self, "_last_travel_failure", None)
                    if ((waypoint['failure'] or {}).get('reason') == 'exploration_no_motion_progress'
                            and self.coverage_progress.metrics.get('stage') in ('review', 'review_due')):
                        # No observation was completed: this is an execution
                        # failure, not evidence that another region has no gain.
                        self._coverage_review_attempts = max(0, self._coverage_review_attempts - 1)
                    self._guard._diagnostic_context = (
                        self.local_map.snapshot(), self.global_map.snapshot(),
                        self.vehicle.get_position(), np.asarray(goal))
                    self._guard.capture_failure(
                        "waypoint_failed", {"waypoint_number": waypoint["number"], **(waypoint["failure"] or {})},
                        waypoint_number=waypoint["number"],
                    )
                    self._selection_reason = "path_invalid"
                    self.blocked += 1
                    count = self.failed_frontiers.get(key, (0, 0))[0] + 1
                    self.failed_frontiers[key] = (
                        count,
                        self.clock() + self.exploration_config.frontier_retry_cooldown,
                    )
                    visit_trace.append(("blocked", self.frontiers.identity(key)))
                    # Replan from the current safe position after a blocked target.
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
                    else:
                        self.current_place = previous
                    self._checkpoint()
                    continue
                decision.record_visit(self.active_viewpoint)
                decision.active_key = None
                self._arrival_frontiers.append(self.frontiers.identity(key))
                self._selection_reason = "arrival"
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
            try:
                self._guard.capture_failure("exploration_exception", {
                    "reason": reason, "message": str(error),
                    "flight_context": getattr(error, "flight_context", None),
                }, terminal=True)
                self._guard.close_diagnostics()
            except Exception as diagnostic_error:
                print(f"--- 异常诊断保存失败：{diagnostic_error} ----")
            raise
        except KeyboardInterrupt:
            reason = "interrupted"
            raise
        finally:
            # Reporting/checkpoint IO must never prolong a previous velocity command.
            ending_with_error = sys.exc_info()[0] is not None
            try:
                self.vehicle.hover()
            except Exception as hold_error:
                if not ending_with_error:
                    raise
                print(f"--- 探索结束位置保持失败：{hold_error} ----")
            self.result = {
                "strategy": "cluster_utility",
                "exploration_planner": self.frontiers.backend_name,
                "frontiers_exhausted": completed,
                "reason": reason,
                "viewpoint_count": len(self.memory.places),
                "blocked_frontiers": self.blocked,
                "elapsed_seconds": self.clock() - self.started,
                "max_radius_m": self.exploration_config.max_radius,
                "flight_clearance_m": self.config.clearance_radius,
                "rolling_replan_period_seconds": (
                    self.exploration_config.rolling_replan_period
                ),
                "return_time_reserve_seconds": self.return_time_reserve,
                "coverage_guaranteed": False,
                "coverage_progress": dict(self.coverage_progress.metrics),
                "visit_trace": visit_trace,
                "decision_config": asdict(self.exploration_config.decision),
                "decision_trace": self._decision_trace,
                "selection_trace": self._selection_trace,
                "arrived_frontiers": self._arrival_frontiers,
                "travel_distance_m": float(sum(np.linalg.norm(b-a) for a,b in zip(self.trajectory,self.trajectory[1:]))),
                "observed_area_m2": float(max(self._coverage_samples, default=0.)),
                "execution_events": self.events,
                "remaining_frontier_regions": len(self.frontiers.regions),
                "remaining_frontier_area_m2": self._last_frontier_area,
                "recent_map_gain_area_m2": self._recent_map_gain(),
                "planned_visit_order": [
                    self.frontiers.identity(k) for k in self.frontiers.visit_order
                ],
                "height_policy": "bounded_clearance_checked_ascent",
                "height_gain_m": float(self._nominal_altitude-self.altitude),
                "motion_policy": "committed_waypoints_astar_ego",
                "local_planner": self.motion_backend,
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
        if not completed:
            try:
                self._guard.capture_failure("exploration_terminated", {"reason": reason}, terminal=True)
            except Exception as error:
                print(f"--- 终止诊断采集失败：{error}；继续安全返航 ----")
        self._guard.close_diagnostics()
        return completed
