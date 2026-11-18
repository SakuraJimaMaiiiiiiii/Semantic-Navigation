"""WildOS saved-target navigation. Mission flow adapted from navigation/mission.py.

Only graph routing and vehicle exclusions differ; bounded retries, visual
verification, local execution guard and result semantics remain explicit here.
"""
from dataclasses import asdict
import json
import math
import numpy as np
from navigation.exploration_session import SavedMapNavigator, TargetUnavailableError
from navigation.route import NavigationGrid
from navigation.scene_memory import write_json
from .target_graph import TargetGraphGrid
from .map_adapter import MapAdapter
from .vehicle_constraints import constrain_adapter
from .vehicle_motion import VehiclePlanVeto, VehiclePlanningMap


class GraphTargetNavigator(SavedMapNavigator):
    def __init__(self, *args, graph_provider, objects_provider, frame_validator, live_map=None,
                 graph_altitude=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_provider, self.objects_provider = graph_provider, objects_provider
        self.frame_validator, self.live_map = frame_validator, live_map
        self.graph_routes = []
        self.graph_altitude = graph_altitude
        self._last_target_grid = None
        self._next_live_pose = 0.
        self._navigation_state = dict(phase='等待图路径', route=None, target_id=None,
                                      active_segment=None, trace_ned=[])
        self._guard = VehiclePlanVeto(self._guard, self.objects_provider, self.config)
        self.global_map = VehiclePlanningMap(self.global_map, self.objects_provider, self)

    def _check_health(self):
        super()._check_health()
        self._validated_graph()
        if hasattr(self, 'altitude') and self.clock() >= self._next_live_pose:
            self._next_live_pose = self.clock()+.5
            self._publish_navigation()
            if self.live_map is None:
                return
            current = self.vehicle.get_position()
            grid = self._last_target_grid
            if grid is None:
                return
            attachment = dict(position_ned=current.tolist(), node_id=None,
                              node_stable_id=None, status='no_certified_connection')
            ordered = sorted(grid.graph['nodes'], key=lambda n: np.linalg.norm(np.asarray(n['position_ned'])-current))
            for node in ordered:
                if (np.linalg.norm(np.asarray(node['position_ned'])-current) <= grid.connector_radius
                        and grid.segment_free(current, node['position_ned'])):
                    attachment.update(node_id=node['id'], node_stable_id=node['stable_id'], status='attached',
                                      evidence='last_graph_snapshot_display_only')
                    break
            self.live_map.robot(attachment)

    def _publish_navigation(self):
        state = self._navigation_state
        current = np.asarray(self.vehicle.get_position(), dtype=float)
        if current.shape == (3,) and np.isfinite(current).all():
            trace = state['trace_ned']
            if not trace or np.linalg.norm(current-np.asarray(trace[-1])) >= .02:
                trace.append(current.tolist())
                del trace[:-4000]
        control = getattr(self.vehicle, 'waypoint_control_state', None)
        state['control_status'] = control.get('status','') if isinstance(control,dict) else ''
        if self.live_map is not None:
            self.live_map.navigation(state)

    def _validated_graph(self):
        try:
            self.frame_validator()
            return self.graph_provider()
        except TargetUnavailableError as error:
            # Once this navigator is running, a frame failure is an in-flight
            # fault, not the session's recoverable preflight lookup failure.
            raise RuntimeError('wildos_target_graph_frame_invalid: '+str(error)) from error

    def _navigation_grid(self, current):
        graph = self._validated_graph()
        grid = NavigationGrid.from_global(self.global_map.snapshot(), self.altitude, self.config,
                                          local_snapshot=self.local_map.snapshot(), current=current)
        constrained, _ = constrain_adapter(MapAdapter.from_grid(grid), self.objects_provider())
        self._last_target_grid = TargetGraphGrid(constrained.grid, graph)
        return self._last_target_grid

    def _restore_graph_height(self):
        if self.graph_altitude is None:
            return
        from .height_recovery import (HeightRecoveryGuard, MAX_RECOVERY_METRES, height_detail,
                                      vertical_rejection, RECOVERY_START_METRES, RECOVERY_ARRIVAL_METRES)
        current = self.vehicle.get_position().copy()
        tolerance = min(RECOVERY_START_METRES,self.vehicle.config.guided_z_tolerance)
        if abs(current[2]-self.altitude) <= tolerance:
            return
        detail = height_detail(current,self.altitude,tolerance)
        target = current.copy();target[2] = self.altitude
        reason = ('高度偏差超过有界恢复上限' if abs(current[2]-self.altitude) > MAX_RECOVERY_METRES
                  else vertical_rejection(self.local_map.execution_snapshot,current,target,
                                          self.config,self.objects_provider()))
        if reason is not None:
            raise RuntimeError('图高度恢复不可执行；'+detail+'；'+reason)
        self.vehicle.hover()
        stop_deadline = min(self.deadline,self.clock()+self.config.leg_timeout)
        while np.linalg.norm(self.vehicle.get_velocity()) > .05:
            self._check_health()
            if self.clock() >= stop_deadline:
                raise TimeoutError('图高度恢复前未能停稳')
            self.vehicle.wait(.05)
        current = self.vehicle.get_position().copy()
        target = current.copy();target[2] = self.altitude
        reason = ('高度偏差超过有界恢复上限' if abs(current[2]-self.altitude) > MAX_RECOVERY_METRES
                  else vertical_rejection(self.local_map.execution_snapshot,current,target,
                                          self.config,self.objects_provider()))
        if reason is not None:
            raise RuntimeError('停稳后图高度恢复通道失效；'+reason)
        self._navigation_state.update(phase='三维净空已检查，恢复图高度',active_segment=[current.tolist(),target.tolist()])
        self._publish_navigation()
        self._event('WILDOS_GRAPH_HEIGHT_RECOVERY',current_ned=current.tolist(),target_ned=target.tolist())
        try:
            self.vehicle.move_waypoint_guided(target,face_direction=False,
                local_planner=HeightRecoveryGuard(self,target),
                xy_tolerance=.05,
                z_tolerance=min(tolerance,RECOVERY_ARRIVAL_METRES),
                timeout=max(.01,min(self.config.leg_timeout,self.deadline-self.clock())))
            self._check_health()
            if abs(self.vehicle.get_position()[2]-self.altitude) > min(tolerance,RECOVERY_ARRIVAL_METRES):
                raise RuntimeError('图高度恢复后仍超出容差')
        finally:
            self.vehicle.hover()
            self._navigation_state['active_segment'] = None

    def _on_choice(self, record, goal, path):
        route = self._last_target_grid.routes[tuple(goal)]
        route = dict(route, target_id=record['runtime_instance_id'])
        # Avoid chasing a very close initial attachment back toward a wall.
        # This changes only the certified entry connector, never cuts a graph corner.
        if (len(route['node_path']) > 1 and 1e-6 < np.linalg.norm(np.asarray(path[1])-path[0])
                <= self.config.arrival_tolerance):
            if self._last_target_grid.segment_free(path[0],path[2]):
                route['skipped_start_node'] = route['node_path'][0]
                route['node_path'] = route['node_path'][1:]
                path.pop(1)
        route['points_ned'] = [np.asarray(p).tolist() for p in path]
        route['length_m'] = float(np.linalg.norm(np.diff(np.asarray(path),axis=0),axis=1).sum())
        route['execution_policy'] = 'graph_nodes_in_order_no_corner_rounding'
        self.graph_routes.append(route)
        self._navigation_state.update(route=route,target_id=record['runtime_instance_id'],
                                      phase='图路径已规划',active_segment=None)
        self._publish_navigation()
        write_json(self.output_path.with_suffix('.graph_route.json'), route)
        self._event('WILDOS_TARGET_GRAPH_ROUTE', **route)
        super()._on_choice(record, goal, path)

    def _move(self, path):
        """Execute graph vertices in order; retain local avoidance and live checks."""
        self._check_health()
        grid = self._navigation_grid(self.vehicle.get_position())
        revision = getattr(self.global_map, 'revision', 0)
        finished = False
        try:
            for index, goal in enumerate(path[1:]):
                goal = np.asarray(goal)
                self._check_health()
                # Replanning after a map correction must use the new frame.
                if getattr(self.global_map, 'revision', 0) != revision:
                    return
                current = self.vehicle.get_position()
                grid = self._navigation_grid(self.vehicle.get_position())
                if not grid.segment_free(current, goal):
                    return
                self._motion_speed_limit = None
                self._navigation_state.update(phase='沿图路径执行',segment_index=index,
                    segment_kind=('connector' if index in (0,len(path)-2) else 'graph_edge'),
                    active_segment=[np.asarray(path[index]).tolist(),goal.tolist()])
                self._publish_navigation()
                # Stop at graph vertices. Subdivision inside a straight edge is
                # still handled by the existing certified movement controller.
                while self._move_segment([current, goal], flowing=False):
                    self._check_health()
                    current = self.vehicle.get_position()
                    grid = self._navigation_grid(self.vehicle.get_position())
                    if (getattr(self.global_map, 'revision', 0) != revision
                            or not grid.segment_free(current, goal)):
                        return
            finished = True
        finally:
            self._motion_speed_limit = None
            self._navigation_state.update(phase='路径执行完成' if finished else '路径中断，等待重新规划',
                                          active_segment=None)
            self._publish_navigation()
            write_json(self.output_path.with_suffix('.navigation.json'),self._navigation_state)

    def run(self):
        self.started = self.clock()
        self.deadline = self.started + self.config.mission_timeout
        origin = self._sample_position()
        self.altitude = float(origin[2] if self.graph_altitude is None else self.graph_altitude)
        searches, attempts, scanned = 0, 0, False
        active_id = None
        self._event("QUERY", request=asdict(self.request))
        success, reason, selected = False, "not_started", None
        try:
            # Give the detector a bounded first-frame warm-up after takeoff.
            warmup = min(self.deadline, self.clock() + self.config.perception_timeout)
            while self.clock() < warmup:
                self.perception.raise_if_failed()
                if (
                    self.perception.semantic_snapshot() is not None
                    and self.local_map.snapshot().timestamp > 0
                    and self.global_map.snapshot().timestamp > 0
                ):
                    break
                self.vehicle.wait(0.2)
            while self.clock() < self.deadline:
                self._check_health()
                self._restore_graph_height()
                current, frame = self._sample_position(), self._frame()
                grid = self._navigation_grid(current)
                candidates = self._query(frame, current)
                if active_id is not None:
                    active = next(
                        (
                            r
                            for r in candidates
                            if r["runtime_instance_id"] == active_id
                        ),
                        None,
                    )
                    if active is None:
                        self._event("TARGET_LOST", target_id=active_id)
                        active_id = None
                    else:
                        candidates = [active] + [
                            r for r in candidates if r is not active
                        ]
                choice = None
                for record in candidates:
                    if self.clock() < self.failed_targets.get(
                        record["runtime_instance_id"], -math.inf
                    ):
                        continue
                    approach = grid.approach(record, current)
                    if approach is not None:
                        choice = record, approach
                        break
                    self.failed_targets[record["runtime_instance_id"]] = (
                        self.clock() + self.config.retry_cooldown
                    )
                    self._event("UNREACHABLE", target_id=record["runtime_instance_id"])
                if choice is not None:
                    record, (_, goal, path) = choice
                    selected = record["runtime_instance_id"]
                    active_id = selected
                    self._on_choice(record, goal, path)
                    if self.state != "APPROACH":
                        self._event(
                            "APPROACH", target_id=selected, goal_ned=goal.tolist()
                        )
                    if np.linalg.norm(current - goal) > self.config.arrival_tolerance:
                        try:
                            self._move(path)
                        except TimeoutError:
                            self.vehicle.hover()
                            self.failed_targets[selected] = (
                                self.clock() + self.config.retry_cooldown
                            )
                            active_id = None
                            attempts += 1
                            self._event(
                                "RETRY", target_id=selected, reason="leg_timeout"
                            )
                        if attempts >= self.config.max_target_attempts:
                            reason = "approach_attempts_exhausted"
                            break
                        continue
                    try:
                        confirmed = self._verify(record, goal)
                    except RuntimeError as error:
                        if self.require_visual_confirmation or str(error) != 'semantic_perception_stale':
                            raise
                        confirmed = None
                        self._event('VISUAL_CONFIRMATION_UNAVAILABLE', reason=str(error))
                    if confirmed is not None:
                        success, reason = True, "visually_confirmed"
                        self._event(
                            "SUCCEEDED",
                            target_id=selected,
                            target_position_ned=confirmed["position_world"],
                        )
                        break
                    if not self.require_visual_confirmation:
                        self._check_health()
                        # Recheck position after observation; reaching the goal
                        # before a drift or interruption is not sufficient.
                        if np.linalg.norm(self.vehicle.get_position() - goal) <= self.config.arrival_tolerance:
                            success, reason = True, 'arrived_visual_unconfirmed'
                            self._event('SUCCEEDED', target_id=selected,
                                        navigation_arrived=True, visual_confirmed=False,
                                        goal_ned=goal.tolist())
                            break
                    attempts += 1
                    active_id = None
                    self.failed_targets[selected] = (
                        self.clock() + self.config.retry_cooldown
                    )
                    self._event(
                        "RETRY", target_id=selected, reason="target_not_reobserved"
                    )
                    if attempts >= self.config.max_target_attempts:
                        reason = "verification_attempts_exhausted"
                        break
                    continue
                if not self.allow_search:
                    # A failed verification/leg temporarily excludes its target;
                    # it does not make the saved target unreachable. Honour the
                    # existing retry budget and cooldown while holding position.
                    cooling = [self.failed_targets.get(r['runtime_instance_id'], -math.inf)
                               for r in candidates]
                    if attempts and any(until > self.clock() for until in cooling):
                        self._wait(min(.2, max(0., self.deadline-self.clock())))
                        continue
                    reason = "no_reachable_saved_target"
                    break
                if not scanned:
                    self._scan()
                    scanned = True
                    continue
                if searches >= self.config.max_search_steps:
                    reason = "search_budget_exhausted"
                    break
                frontier = grid.frontier(current, origin, self.visited)
                if frontier is None:
                    reason = "no_reachable_frontier_or_target"
                    break
                goal, path = frontier
                self._event("SEARCH_MOVE", goal_ned=goal.tolist())
                try:
                    self._move(path)
                except TimeoutError:
                    self.vehicle.hover()
                    self._event("SEARCH_BLOCKED", goal_ned=goal.tolist())
                self.visited.append(self.vehicle.get_position().copy())
                searches += 1
                scanned = False
            else:
                reason = "mission_timeout"
            if not success:
                self._event("FAILED", reason=reason)
            return success
        except Exception as error:
            reason = str(error)
            self._event("FAILED", reason=reason, exception_type=type(error).__name__)
            raise
        finally:
            self._sample_position()
            self._navigation_state.update(phase='已到达，悬停' if success else '导航结束：'+reason,
                                          active_segment=None)
            self._publish_navigation()
            write_json(self.output_path.with_suffix('.navigation.json'),self._navigation_state)
            self.result = {
                "version": 1,
                "planning_backend": "wildos_graph_dijkstra",
                "graph_routes": self.graph_routes,
                "success": success,
                "reason": reason,
                "navigation_arrived": success,
                "visual_confirmed": success and reason == 'visually_confirmed',
                "request": asdict(self.request),
                "selected_target_id": selected,
                "elapsed_seconds": self.clock() - self.started,
                "path_length_m": (
                    float(
                        np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1).sum()
                    )
                    if len(self.trajectory) > 1
                    else 0.0
                ),
                "trajectory_ned": [p.tolist() for p in self.trajectory],
                "search_steps": searches,
                "failed_attempts": attempts,
                "collision": any(
                    e.get("exception_type") == "CollisionDetectedError"
                    for e in self.events
                ),
                "inference_seconds_mean": (
                    float(np.mean(list(self._latencies.values())))
                    if self._latencies
                    else None
                ),
                "events": self.events,
            }
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.result, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(self.output_path)
