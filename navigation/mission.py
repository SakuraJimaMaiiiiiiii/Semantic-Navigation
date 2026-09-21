"""Bounded search -> approach -> fresh visual verification task executor."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
import numpy as np

from .target import select_targets
from .route import NavigationGrid, next_leg
from planner.local_avoidance_planner import LocalPlan


class _FlightGuard:
    """Guard EGO execution against unknown-space and off-plane proposals."""

    def __init__(self, navigator):
        self.navigator = navigator
        self.inner = navigator.local_planner
        self.config = self.inner.config
        self._grid_stamp = None
        self._grid = None
        self.last_status = None

    def _report(self, plan):
        if self.last_status != plan.status:
            self.last_status = plan.status
            self.navigator._event("LOCAL_EXECUTION", status=plan.status)
        return plan

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def tracking_target(self, now=None):
        # Advance the EGO spline only while its current target is certified.
        if self.last_status not in {"ego_clear", "ego_avoid"} or self._grid is None:
            return None
        getter = getattr(self.inner, "tracking_target", None)
        target = getter(now=now) if callable(getter) else None
        if target is not None and self._grid.segment_free(self.navigator.vehicle.get_position(), target):
            return target
        return None

    def plan(self, current, target):
        nav = self.navigator
        nav._check_health()
        nav._sample_position(current)
        snapshot = nav.local_map.snapshot()
        global_source = getattr(nav, "global_map", None)
        global_snapshot = (
            global_source.snapshot() if global_source is not None else None
        )
        stamp = (
            snapshot.timestamp,
            None if global_snapshot is None else global_snapshot.timestamp,
        )
        if stamp != self._grid_stamp:
            self._grid = NavigationGrid.from_local(
                snapshot, nav.altitude, nav.config, global_snapshot
            )
            self._grid_stamp = stamp
        grid = self._grid
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        forward_clearance = float(getattr(snapshot, "forward_clearance", math.inf))
        emergency_distance = float(
            getattr(self.config, "emergency_stop_distance", 0.6)
        )
        # The frontier/global planner already supplies a known-free route. Follow
        # a clear segment directly; reserve EGO spline generation for an actual
        # local obstruction or loss of corridor evidence.
        if (
            forward_clearance >= emergency_distance
            and grid.segment_free(current, target)
        ):
            return self._report(LocalPlan(
                status="route_clear",
                local_target_ned=target.copy(),
                path_ned=np.vstack((current, target)),
                speed_limit=float(getattr(self.config, "cruise_max_speed", 0.6)),
                obstacle_distance=forward_clearance,
            ))

        configure = getattr(self.inner, "set_navigation_plane", None)
        if callable(configure):
            configure(nav.altitude, nav.config, nav.global_map)
        plan = self.inner.plan(current, target)
        if not plan.can_move:
            return self._report(plan)
        # Once the direct certified route is unavailable, every executable EGO
        # result is an avoidance-mode result, even if the final spline segment
        # itself happens to be straight.
        if plan.status == "ego_clear":
            plan = replace(plan, status="ego_avoid")
        points = [np.asarray(current), np.asarray(plan.local_target_ned)]
        valid = grid.segment_free(*points)
        if len(plan.path_ned):
            # Check the entire proposed trajectory; a spline may leave its chord.
            valid &= all(
                grid.segment_free(a, b)
                for a, b in zip(plan.path_ned[:-1], plan.path_ned[1:])
            )
        if not valid:
            return self._report(
                replace(
                    plan,
                    status="semantic_unknown_or_blocked",
                    local_target_ned=None,
                    speed_limit=0.0,
                )
            )
        return self._report(plan)


class SemanticNavigator:
    def __init__(
        self,
        vehicle,
        perception,
        local_map,
        global_map,
        local_planner,
        request,
        config,
        output_path,
        clock=time.monotonic,
    ):
        self.vehicle, self.perception = vehicle, perception
        self.local_map, self.global_map = local_map, global_map
        self.local_planner, self.request = local_planner, request
        self.config, self.output_path, self.clock = config, Path(output_path), clock
        self.events, self.trajectory = [], []
        self.failed_targets, self.visited = {}, []
        self._map_stamps = {}
        self._last_health_map_check = -math.inf
        self.result = None
        self.state = None
        self._latencies = {}
        self._guard = _FlightGuard(self)
        self.allow_search = True

    def _event(self, state, **details):
        self.state = state
        self.events.append(
            {"elapsed": self.clock() - self.started, "state": state, **details}
        )
        print(f"--- Semantic navigation: {state} {details} ----")

    def _sample_position(self, position=None):
        p = np.asarray(
            self.vehicle.get_position() if position is None else position, dtype=float
        )
        if not self.trajectory or np.linalg.norm(p - self.trajectory[-1]) > 0.01:
            self.trajectory.append(p.copy())
        return p

    def _frame(self):
        self.perception.raise_if_failed()
        frame = self.perception.semantic_snapshot()
        if (
            frame is None
            or self.clock() - frame.completed_at > self.config.perception_timeout
        ):
            raise RuntimeError("semantic_perception_stale")
        self._latencies[frame.timestamp] = frame.inference_seconds
        return frame

    def _check_health(self):
        if self.clock() >= self.deadline:
            raise TimeoutError("semantic_mission_timeout")
        self._frame()
        self._check_maps()

    def _check_maps(self):
        """Check mapping independently while semantic inference recovers."""
        if self.clock() - self._last_health_map_check < 0.2:
            return
        self._last_health_map_check = self.clock()
        for name, source in (("local", self.local_map), ("global", self.global_map)):
            heartbeat = getattr(source, "last_observation_completed_at", None)
            if heartbeat is None:
                heartbeat = getattr(source, "last_update_timestamp", None)
            if heartbeat is None:
                heartbeat = source.snapshot().timestamp
            previous, changed = self._map_stamps.get(name, (None, self.clock()))
            if heartbeat != previous:
                changed = self.clock()
            self._map_stamps[name] = (heartbeat, changed)
            if heartbeat <= 0 or self.clock() - changed > self.config.map_timeout:
                raise RuntimeError(f"semantic_{name}_map_stale")

    def _wait(self, duration):
        end = min(self.deadline, self.clock() + duration)
        while self.clock() < end:
            self._check_health()
            remaining = end - self.clock()
            if remaining > 0:
                self.vehicle.wait(min(0.2, remaining))

    def _align(self, yaw):
        self._check_health()
        self.vehicle.align_yaw(
            yaw,
            timeout=max(
                0.01, min(self.config.leg_timeout, self.deadline - self.clock())
            ),
            health_check=self._check_health,
        )

    def _scan(self):
        self._event("SEARCH_SCAN")
        for angle in (0, math.pi / 2, math.pi, -math.pi / 2):
            self._check_health()
            self._align(angle)
            self._wait(self.config.scan_wait)
            frame = self._frame()
            if select_targets(
                frame.records,
                self.request,
                self.vehicle.get_position(),
                frame.timestamp,
            ):
                return

    def _move(self, path):
        self._check_health()
        goal = next_leg(path, self.config.leg_length)
        self.vehicle.move_velocity_guided(
            goal,
            face_direction=True,
            local_planner=self._guard,
            timeout=max(
                0.01, min(self.config.leg_timeout, self.deadline - self.clock())
            ),
        )
        self._sample_position()

    def _verify(self, record, goal):
        self._event("VERIFY", target_id=record["runtime_instance_id"])
        delta = np.array(record["position_world"]) - self.vehicle.get_position()
        self._align(math.atan2(delta[1], delta[0]))
        # Require new synchronized observations after arrival AND yaw alignment.
        baseline = self._frame().timestamp
        seen, latest, deadline = (
            set(),
            record,
            min(self.deadline, self.clock() + self.config.verification_timeout),
        )
        while self.clock() < deadline:
            self._check_health()
            frame = self._frame()
            matches = self._verification_candidates(frame, self.vehicle.get_position())
            match = next(
                (
                    r
                    for r in matches
                    if r["runtime_instance_id"] == record["runtime_instance_id"]
                ),
                None,
            )
            if (
                match is not None
                and match["last_seen"] > baseline
                and frame.timestamp - match["last_seen"] < 1e-6
            ):
                current = self.vehicle.get_position()
                low, high = np.array(match["bbox_3d"]["min"]), np.array(
                    match["bbox_3d"]["max"]
                )
                distance = np.linalg.norm(
                    np.maximum(
                        np.maximum(low[:2] - current[:2], current[:2] - high[:2]), 0
                    )
                )
                if (
                    np.linalg.norm(current - goal) <= self.config.arrival_tolerance
                    and self.config.standoff_distance - self.config.arrival_tolerance
                    <= distance
                    <= self.config.standoff_distance + self.config.arrival_tolerance
                ):
                    seen.add(match["last_seen"])
                    latest = match
                    if len(seen) >= self.config.verification_frames:
                        return latest
            self._wait(0.2)
        return None

    def _verification_candidates(self, frame, current):
        return select_targets(frame.records, self.request, current, frame.timestamp)

    def _query(self, frame, current):
        return select_targets(frame.records, self.request, current, frame.timestamp)

    def _on_choice(self, record, goal, path):
        """Optional map annotation hook after a reachable goal has been selected."""

    def run(self):
        self.started = self.clock()
        self.deadline = self.started + self.config.mission_timeout
        origin = self._sample_position()
        self.altitude = float(origin[2])
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
                current, frame = self._sample_position(), self._frame()
                grid = NavigationGrid.from_global(
                    self.global_map.snapshot(), self.altitude, self.config
                )
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
                    confirmed = self._verify(record, goal)
                    if confirmed is not None:
                        success, reason = True, "visually_confirmed"
                        self._event(
                            "SUCCEEDED",
                            target_id=selected,
                            target_position_ned=confirmed["position_world"],
                        )
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
            self.result = {
                "version": 1,
                "success": success,
                "reason": reason,
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
