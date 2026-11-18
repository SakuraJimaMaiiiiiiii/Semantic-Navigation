"""Bounded search -> approach -> fresh visual verification task executor."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np

from .target import select_targets
from .route import NavigationGrid, execution_segment_free, observed_body_segment_rejection
from .path_smoothing import rounded_path
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
        self._failure_recorder = None
        self._diagnostic_context = None
        self.last_plan_detail = None
        self.last_plan_timings = {}
        self._execution_context = None
        self._execution_plan_detail = None
        self._execution_rejection = None
        self._latest_execution_snapshot = None
        self._recovering_height = False
        self._front_depth_cache = {}

    def _report(self, plan, rejection=None):
        motion_limit = getattr(self.navigator, '_motion_speed_limit', None)
        if plan.can_move and motion_limit is not None:
            plan = replace(plan, speed_limit=min(plan.speed_limit, motion_limit))
        attention_limit = getattr(self.navigator, 'vehicle_observation_speed_limit', None)
        if plan.can_move and callable(attention_limit):
            plan = replace(plan, speed_limit=min(plan.speed_limit, attention_limit()))
        self.last_plan_detail = {
            "status": plan.status, "speed_limit": float(plan.speed_limit),
            "local_target_ned": None if plan.local_target_ned is None else np.asarray(plan.local_target_ned).tolist(),
            "path_ned": np.asarray(plan.path_ned).tolist(),
            "rejection": rejection or getattr(self.inner, "last_rejection", None),
            "timings": dict(self.last_plan_timings),
            "ego_timings": (dict(getattr(self.inner, 'last_plan_timings', {}))
                            if 'local_planner_seconds' in self.last_plan_timings else {}),
        }
        return plan

    def notify_execution(self, plan, *, certificate=None):
        self._execution_context = self._diagnostic_context if certificate is None else certificate[2]
        if self._execution_context is not None and self._latest_execution_snapshot is not None:
            _, world, _, target = self._execution_context
            self._execution_context = (self._latest_execution_snapshot, world,
                self.navigator.vehicle.get_position(),
                target if plan.local_target_ned is None else plan.local_target_ned)
        source = self.last_plan_detail if certificate is None else certificate[3]
        self._execution_plan_detail = dict(source or {})
        self._execution_plan_detail.update(status=plan.status, speed_limit=float(plan.speed_limit),
            local_target_ned=None if plan.local_target_ned is None else np.asarray(plan.local_target_ned).tolist())
        if self._execution_rejection is not None:
            self._execution_plan_detail['rejection'] = self._execution_rejection
        if self.last_status != plan.status:
            self.last_status = plan.status
            detail = self._execution_plan_detail.get('rejection') if not plan.can_move else None
            reason = detail.get("reason") if detail else None
            event_details = dict(status=plan.status,rejection=reason)
            if plan.status in {'height_recovery','height_recovery_blocked'}:
                reader = getattr(self.navigator.vehicle,'height_tracking_diagnostics',None)
                try:
                    evidence = reader() if callable(reader) else None
                except Exception as error:
                    evidence = None
                    event_details['height_tracking_error'] = str(error)
                if isinstance(evidence,dict):
                    event_details['height_tracking'] = evidence
                    if not plan.can_move:
                        detail = dict(detail or {},height_tracking=evidence)
            self.navigator._event("LOCAL_EXECUTION", **event_details)
            # An asynchronous solve has not produced a certificate yet. This
            # is a scheduling wait, not a rejected route; capturing it with
            # the previous certificate mislabels clear corridors as failures
            # and exhausts the mission's diagnostic snapshot budget.
            if (not plan.can_move and plan.status != 'waiting_for_plan'
                    and self._execution_context is not None):
                self.capture_failure(plan.status, detail)
        return plan

    def capture_failure(self, status, detail=None, terminal=False, waypoint_number=None):
        nav = self.navigator
        if not hasattr(nav, 'output_path'):
            return
        from .planning_diagnostics import FailureRecorder
        if self._failure_recorder is None:
            phase = "returning" if getattr(nav, "phase", None) == "returning" else "exploring"
            self._failure_recorder = FailureRecorder(
                (nav.output_path.parent if nav.output_path.parent.name == 'private'
                 else nav.output_path.parent / 'private')
                / 'diagnostics' / f"{phase}_{uuid.uuid4().hex[:12]}"
            )
        if terminal or self._execution_context is None:
            current = nav.vehicle.get_position()
            context = (nav.local_map.snapshot(), nav.global_map.snapshot(), current, current)
        else:
            context = self._execution_context
        detail = dict(detail or {})
        detail["planner"] = self._execution_plan_detail
        detail["controller"] = getattr(nav.vehicle, "waypoint_control_state", None)
        get_velocity = getattr(nav.vehicle, "get_velocity", None)
        if callable(get_velocity):
            detail["measured_velocity_ned"] = np.asarray(get_velocity()).tolist()
        self._failure_recorder.capture(status, *context, nav.altitude, nav.config, detail, terminal=terminal, waypoint_number=waypoint_number)

    def close_diagnostics(self):
        if self._failure_recorder is not None:
            self._failure_recorder.close()

    def __getattr__(self, name):
        return getattr(self.inner, name)


    def _braking_evidence(self, current):
        vehicle = getattr(self.navigator, "vehicle", None)
        get_velocity = getattr(vehicle, "get_velocity", None)
        if not callable(get_velocity):
            return None
        velocity = np.asarray(get_velocity(), dtype=float)
        speed = float(np.linalg.norm(velocity[:2]))
        if not np.isfinite(velocity).all():
            return None
        vehicle_config = getattr(vehicle, "config", None)
        deceleration = float(getattr(vehicle_config, "guided_max_horizontal_acceleration", 0.3))
        control_rate = float(getattr(vehicle_config, "guided_update_rate", 20.0))
        map_config = getattr(self.navigator.local_map, "config", None)
        map_rate = float(getattr(map_config, "update_rate", 5.0))
        latency = 1.0 / float(getattr(self.config, "replan_rate", 5.0)) + 1.0/control_rate + 1.0/map_rate
        distance = speed*latency + speed*speed/(2.0*deceleration)
        stop = np.asarray(current, dtype=float).copy()
        if speed > 1e-9:
            stop[:2] += velocity[:2] * (distance/speed)
        stop[2] += velocity[2]*(latency+speed/(2.*deceleration))
        return dict(speed_mps=speed,braking_distance_m=distance,
                    deceleration_mps2=deceleration,braking_endpoint_ned=stop.tolist())

    def _braking_rejection(self, grid, current):
        evidence = self._braking_evidence(current)
        if evidence is not None and evidence['speed_mps'] >= .05 and not grid.segment_free(
                current,evidence['braking_endpoint_ned']):
            return {"reason": "insufficient_braking_corridor",
                    **evidence}
        return None

    def _forward_clearance(self, snapshot, current, target):
        from planner.directional_clearance import motion_forward_clearance
        if snapshot is None:
            return float(self.navigator.local_map.forward_clearance)
        velocity = self.navigator.vehicle.get_velocity()
        clearance = motion_forward_clearance(snapshot,current,target,velocity)
        if clearance < float(self.config.emergency_stop_distance):
            evidence = self._braking_evidence(current)
            if evidence is not None:
                from planner.certified_depth import refine_front_clearance
                clearance = refine_front_clearance(snapshot,current,target,evidence['braking_endpoint_ned'],
                    clearance,self.navigator.config,self.config,getattr(self.navigator.local_map,'config',None),
                    cache=self._front_depth_cache)
        return clearance

    def revalidate_plan(self, plan, current, *, certificate=None):
        """Validate a published result on the flight thread without running a solver.

        Retain its matching grid/revision, then inspect the commanded body volume
        on the newest completed local geometry and read current front depth.
        """
        self._execution_rejection = None
        self._latest_execution_snapshot = None
        if not plan.can_move:
            return plan
        self.navigator._check_maps()
        revision = getattr(self.navigator.global_map, 'revision', 0)
        grid, planned_revision, context = ((self._grid, getattr(self, '_planned_map_revision', revision),
                                          self._diagnostic_context) if certificate is None else certificate[:3])
        if revision != planned_revision:
            self._execution_rejection = {'reason': 'map_revision_changed'}
            return LocalPlan('map_revision_changed', None, np.empty((0, 3)), 0.)
        current = np.asarray(current, dtype=float)
        if grid is not None and (plan.status == 'height_recovery'
                                 or self._height_needs_recovery(current, grid)):
            return self._height_recovery(current, self.navigator.local_map.execution_snapshot)
        target = np.asarray(plan.local_target_ned, dtype=float)
        rejection = None
        if grid is None or not grid.segment_free(current, target):
            rejection = {"reason": "post_planning_tracking_chord_not_certified",
                         "current_ned": current.tolist(), "target_ned": target.tolist()}
            status = "semantic_unknown_or_blocked"
        else:
            rejection = self._braking_rejection(grid, current)
            status = "braking_corridor_blocked"
        if rejection is None and certificate is not None:
            local = self.navigator.local_map.execution_snapshot
            self._latest_execution_snapshot = local
            if local is None:
                return LocalPlan('waiting_for_map', None, np.empty((0, 3)), 0.)
            if not execution_segment_free(local, current, target, self.navigator.config, grid):
                rejection = {"reason": "latest_local_chord_blocked"}
                status = "semantic_unknown_or_blocked"
        depth_snapshot = self.navigator.local_map.execution_snapshot
        forward_clearance = self._forward_clearance(depth_snapshot,current,target)
        # EGO detours are validated by their curve; the front scalar applies
        # only to a direct route segment.
        if (rejection is None and plan.status == "route_clear"
                and forward_clearance < float(self.config.emergency_stop_distance)):
            rejection = {"reason": "post_planning_front_depth_blocked",
                         "forward_clearance": forward_clearance}
            status = "transient_no_path"
        if rejection is not None:
            self._execution_rejection = rejection
            return replace(plan, status=status, local_target_ned=None, speed_limit=0.)
        if forward_clearance == float(self.config.emergency_stop_distance):
            plan = replace(plan,speed_limit=min(plan.speed_limit,.1))
        return plan

    def check_execution_health(self):
        self.navigator._check_health()

    def _height_needs_recovery(self, current, grid):
        # Correct before a full voxel of drift consumes the certified body band.
        threshold = min(.03, .15*grid.resolution)
        if self._recovering_height:
            threshold = min(.02, .1*grid.resolution)
        return abs(current[2]-grid.altitude) > threshold

    def _height_recovery(self, current, snapshot):
        self._recovering_height = True
        self._latest_execution_snapshot = snapshot
        target = np.asarray(current, dtype=float).copy()
        target[2] = self.navigator.altitude
        rejection = observed_body_segment_rejection(snapshot, current, target, self.navigator.config)
        if rejection is not None:
            self._execution_rejection = dict(reason=rejection, current_ned=np.asarray(current).tolist(),
                                             target_ned=target.tolist())
            return LocalPlan('height_recovery_blocked', None, np.empty((0, 3)), 0.)
        return LocalPlan('height_recovery', target, np.vstack((current, target)),
                         min(.15, float(self.config.cruise_max_speed)))

    def retarget_clear_plan(self, plan, current, target, certificate):
        """Reuse a certified clear grid for a new short waypoint; revalidation still follows."""
        if certificate is None or plan.status != 'route_clear' or not plan.can_move:
            return None
        grid, revision = certificate[:2]
        if (grid is None or revision != getattr(self.navigator.global_map, 'revision', 0)
                or not grid.segment_free(current, target)):
            return None
        return replace(plan, local_target_ned=np.asarray(target).copy(),
                       path_ned=np.vstack((current, target)))

    def execution_certificate(self):
        return self._grid, self._planned_map_revision, self._diagnostic_context, self.last_plan_detail

    def plan(self, current, target):
        self.check_execution_health()
        self._execution_rejection = None
        return self.notify_execution(self.compute_plan(current, target))

    def compute_plan(self, current, target):
        nav = self.navigator
        self.last_plan_timings = {}
        nav._sample_position(current)
        started = time.monotonic()
        snapshot = nav.local_map.snapshot()
        global_source = getattr(nav, "global_map", None)
        if hasattr(global_source, 'snapshot_with_revision'):
            global_snapshot, revision = global_source.snapshot_with_revision()
        else:
            global_snapshot = global_source.snapshot() if global_source is not None else None
            revision = 0
        self.last_plan_timings['snapshots_seconds'] = time.monotonic()-started
        if revision != getattr(self, '_planned_map_revision', revision):
            discard = getattr(self.inner, 'discard_trajectory', None)
            if callable(discard):
                discard()
        self._planned_map_revision = revision
        self._diagnostic_context = (snapshot, global_snapshot, current, target)
        stamp = (
            float(nav.altitude),
            revision,
            snapshot.timestamp,
            None if global_snapshot is None else global_snapshot.timestamp,
        )
        started = time.monotonic()
        if stamp != self._grid_stamp:
            self._grid = NavigationGrid.from_local(
                snapshot, nav.altitude, nav.config, global_snapshot
            )
            self._grid_stamp = stamp
        self.last_plan_timings['grid_projection_seconds'] = time.monotonic()-started
        grid = self._grid
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        if self._height_needs_recovery(current, grid):
            plan = self._height_recovery(current, snapshot)
            return self._report(plan, rejection=self._execution_rejection)
        self._recovering_height = False
        braking = self._braking_rejection(grid, current)
        if braking is not None:
            return self._report(
                LocalPlan("braking_corridor_blocked", None, np.empty((0, 3)), 0.0),
                rejection=braking,
            )
        forward_clearance = self._forward_clearance(snapshot,current,target)
        emergency_distance = float(getattr(self.config, "emergency_stop_distance", 0.6))
        # The frontier/global planner already supplies a known-free route. Follow
        # a clear segment directly; reserve EGO spline generation for an actual
        # local obstruction or loss of corridor evidence.
        if forward_clearance >= emergency_distance and grid.segment_free(
            current, target
        ):
            discard = getattr(self.inner, "discard_trajectory", None)
            if callable(discard):
                discard()
            speed_limit = float(getattr(self.config, "cruise_max_speed", 0.6))
            limiter = getattr(self.inner, "_clear_speed_limit", None)
            if callable(limiter):
                speed_limit = float(limiter(forward_clearance))
            if forward_clearance == emergency_distance:
                speed_limit = min(speed_limit,.1)
            return self._report(
                LocalPlan(
                    status="route_clear",
                    local_target_ned=target.copy(),
                    path_ned=np.vstack((current, target)),
                    speed_limit=speed_limit,
                )
            )

        configure = getattr(self.inner, "set_navigation_plane", None)
        if callable(configure):
            configure(nav.altitude, nav.config, nav.global_map)
        started = time.monotonic()
        plan = self.inner.plan(current, target)
        self.last_plan_timings['local_planner_seconds'] = time.monotonic()-started
        if not plan.can_move:
            return self._report(plan)
        # Once the direct certified route is unavailable, every executable EGO
        # result is an avoidance-mode result, even if the final spline segment
        # itself happens to be straight.
        if plan.status == "ego_clear":
            plan = replace(plan, status="ego_avoid")
        segments = [(np.asarray(current), np.asarray(plan.local_target_ned))]
        segments.extend(zip(plan.path_ned[:-1], plan.path_ned[1:]))
        started = time.monotonic()
        for index, (start, end) in enumerate(segments):
            if not grid.segment_free(start, end):
                return self._report(
                    replace(plan, status="semantic_unknown_or_blocked",
                            local_target_ned=None, speed_limit=0.0),
                    rejection={
                        "reason": "tracking_chord_not_certified" if index == 0 else "trajectory_segment_not_certified",
                        "segment_start_ned": np.asarray(start).tolist(),
                        "segment_end_ned": np.asarray(end).tolist(),
                    },
                )
        self.last_plan_timings['semantic_path_validation_seconds'] = time.monotonic()-started
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
        self.require_visual_confirmation = True

    def _event(self, state, **details):
        self.state = state
        self.events.append(
            {"elapsed": self.clock() - self.started, "state": state, **details}
        )
        if state in {"FRONTIER_DECISION", "COVERAGE_STATE", "LOCAL_EXECUTION", "SEMANTIC_DEFERRED", "SEMANTIC_RESUMED"}:
            return  # Keep diagnostics in the report without interrupting the bar.
        transient = {
            'MAP_UPDATE_WAIT': 'map_wait', 'MAP_UPDATE_RESUMED': 'map_resumed',
            'PERCEPTION_WAIT': 'perception_wait', 'PERCEPTION_RECOVERED': 'perception_resumed',
        }
        if state in transient:
            from .terminal_progress import shared_progress
            progress = getattr(self.vehicle, '_motion_progress_display', None) or shared_progress()
            reason = details.get('reason')
            detail = {'semantic_local_map_stale': '局部地图',
                      'semantic_global_map_stale': '全局地图'}.get(reason)
            progress.update(transient[state], moving=False, detail=detail)
            return
        progress = getattr(self.vehicle, '_motion_progress_display', None)
        if progress is not None:
            progress.finish()
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

    def _check_health(self, *, require_semantics=True):
        if self.clock() >= self.deadline:
            raise TimeoutError("semantic_mission_timeout")
        if require_semantics:
            self._frame()
        else:
            self.perception.raise_if_failed()
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

    def _align(self, yaw, *, deadline=None):
        self._check_health()
        self.vehicle.align_yaw(
            yaw,
            timeout=max(
                0.01, min(self.config.leg_timeout,
                          min(self.deadline,self.deadline if deadline is None else deadline) - self.clock())
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
        """Commit a certified rounded route, with live checks before each chord."""
        self._check_health()
        grid = NavigationGrid.from_global(self.global_map.snapshot(), self.altitude, self.config)
        route = rounded_path(path, grid.segment_free, self.config.corner_rounding_distance)
        revision = getattr(self.global_map, 'revision', 0)
        try:
            for goal, flowing, turning in route:
                self._check_health()
                # Replanning after a map correction must use the new frame.
                if getattr(self.global_map, 'revision', 0) != revision:
                    return
                current = self.vehicle.get_position()
                grid = NavigationGrid.from_global(self.global_map.snapshot(), self.altitude, self.config)
                if not grid.segment_free(current, goal):
                    return
                self._motion_speed_limit = (0.6*self.vehicle.config.guided_max_forward_speed
                                            if turning else None)
                while self._move_segment([current, goal], flowing=flowing):
                    self._check_health()
                    current = self.vehicle.get_position()
                    grid = NavigationGrid.from_global(self.global_map.snapshot(), self.altitude, self.config)
                    if (getattr(self.global_map, 'revision', 0) != revision
                            or not grid.segment_free(current, goal)):
                        return
        finally:
            self._motion_speed_limit = None

    def _move_segment(self, path, *, flowing=False):
        self._check_health()
        current = np.asarray(self.vehicle.get_position())
        points = [np.asarray(point) for point in path[1:]
                  if np.linalg.norm(np.asarray(point)-current) > 1e-6]
        if not points:
            return
        # A distance along the complete polyline can land AFTER a bend. Sending
        # that point directly cuts across obstacles and needlessly invokes EGO.
        endpoint = points[0]
        distance = float(np.linalg.norm(endpoint-current))
        intermediate = distance > self.config.leg_length
        goal = (current + (endpoint-current)*(self.config.leg_length/distance)
                if intermediate else endpoint.copy())
        self.vehicle.move_waypoint_guided(
            goal,
            face_direction=True,
            local_planner=self._guard,
            pass_through=intermediate or flowing,
            xy_tolerance=(min(self.config.arrival_tolerance, .25*self.config.leg_length)
                          if intermediate else min(.25 if flowing else .05, self.config.arrival_tolerance)),
            timeout=max(
                0.01, min(self.config.leg_timeout, self.deadline - self.clock())
            ),
        )
        self._sample_position()
        # A long straight chord is subdivided, but its interior points do not
        # stop the vehicle or trigger a new equally-priced A* route.
        return intermediate

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
        observed_frames, matched_frames = set(), set()
        last_distance = None
        while self.clock() < deadline:
            self._check_health()
            frame = self._frame()
            observed_frames.add(frame.timestamp)
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
                matched_frames.add(frame.timestamp)
                current = self.vehicle.get_position()
                low, high = np.array(match["bbox_3d"]["min"]), np.array(
                    match["bbox_3d"]["max"]
                )
                distance = np.linalg.norm(
                    np.maximum(
                        np.maximum(low[:2] - current[:2], current[:2] - high[:2]), 0
                    )
                )
                last_distance = float(distance)
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
        self._event('VERIFICATION_INCOMPLETE', target_id=record['runtime_instance_id'],
                    observed_frames=len(observed_frames), matched_frames=len(matched_frames),
                    accepted_frames=len(seen), required_frames=self.config.verification_frames,
                    last_surface_distance_m=last_distance,
                    reason=('target_not_reobserved' if not matched_frames
                            else 'verification_geometry_or_frame_count'))
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
            self.result = {
                "version": 1,
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
