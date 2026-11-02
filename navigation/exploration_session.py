"""One SDK owner, two terminal workflow, and persistent semantic-map navigation."""

from dataclasses import replace
import math
from pathlib import Path
import time
import uuid
import numpy as np

from .frontier_explorer import FrontierExplorer
from .map_view import render_map
from .mission import SemanticNavigator
from .route import NavigationGrid
from .scene_memory import SceneMemory, load_bundle, write_json
from .target import select_targets


class TargetUnavailableError(ValueError):
    """A preflight target lookup failure; no vehicle motion has been requested."""


class NavigationNotReadyError(RuntimeError):
    """Live sensors are temporarily unavailable before submitting motion."""


class ContinuityWatch:
    def __init__(self, vehicle, clock=time.monotonic):
        self.vehicle, self.clock = vehicle, clock
        self.last_boot = None
        self.last_change = clock()
        self.invalid = False
        self.position = np.asarray(vehicle.get_position()).copy()
        self.checked_at = clock()

    def check(self):
        if self.invalid:
            raise RuntimeError(
                "Control session frame is invalid; explore again after restarting"
            )
        now = self.clock()
        boot = float(getattr(self.vehicle.mav, "uavTimeStmp", 0))
        position = np.asarray(self.vehicle.get_position())
        failed = not np.all(np.isfinite(position)) or not math.isfinite(boot)
        if boot > 0:
            if self.last_boot is not None and boot < self.last_boot:
                failed = True
            if boot != self.last_boot:
                self.last_change = now
            if now - self.last_change > 3.0:
                failed = True
            self.last_boot = boot
        if np.linalg.norm(position - self.position) > 2.0 + 3.0 * (
            now - self.checked_at
        ):
            failed = True
        self.position, self.checked_at = position.copy(), now
        self.invalid |= failed
        if failed:
            raise RuntimeError(
                "PX4 reset, stale telemetry or position jump; saved map cannot be reused"
            )


class WatchedPerception:
    def __init__(self, worker, watch):
        self.worker, self.watch = worker, watch

    def semantic_snapshot(self):
        return self.worker.semantic_snapshot()

    def raise_if_failed(self):
        self.watch.check()
        self.worker.raise_if_failed()


class SavedMapNavigator(SemanticNavigator):
    def __init__(self, *args, records, annotate, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.annotate = records, annotate
        self.allow_search = False
        self.require_visual_confirmation = False

    def _check_health(self):
        # Saved targets do not require detector refresh on every control tick.
        # Live mapping and perception worker failures are still checked.
        super()._check_health(require_semantics=False)

    def _frame(self):
        end = min(self.deadline, self.clock() + self.config.perception_timeout)
        waiting = False
        while True:
            self._check_health()
            frame = self.perception.semantic_snapshot()
            if frame is not None and self.clock() - frame.completed_at <= self.config.perception_timeout:
                self._latencies[frame.timestamp] = frame.inference_seconds
                if waiting:
                    self._event('PERCEPTION_RECOVERED')
                return frame
            if self.clock() >= end:
                raise RuntimeError('semantic_perception_stale')
            if not waiting:
                self.vehicle.hover()
                self._event('PERCEPTION_WAIT', reason='saved_target_requires_fresh_observation')
                waiting = True
            self.vehicle.wait(min(.2, end - self.clock()))

    def _verification_candidates(self, frame, current):
        # Relative conditions were resolved against the saved graph. Confirmation
        # needs a new observation of the pinned object, not every old anchor.
        request = replace(self.request, near_instance_id=None)
        return select_targets(frame.records, request, current, frame.timestamp)

    def _query(self, frame, current):
        matches = select_targets(
            self.records, self.request, current, frame.timestamp, check_age=False
        )
        fresh = {
            r["runtime_instance_id"]: r
            for r in self._verification_candidates(frame, current)
        }
        result = []
        for record in matches:
            live = fresh.get(record["runtime_instance_id"])
            if (
                live is not None
                and np.linalg.norm(
                    np.asarray(live["position_world"]) - record["position_world"]
                )
                <= self.config.standoff_distance
            ):
                # Refine a static object's visible-surface estimate during approach.
                result.append(
                    {
                        **record,
                        **{key: live[key] for key in ("position_world", "bbox_3d")},
                    }
                )
            else:
                result.append(record)
        return result

    def _on_choice(self, record, goal, path):
        self.annotate(record["runtime_instance_id"], goal, path, "approaching")


class ExplorationSession:
    navigator_type = SavedMapNavigator

    def navigation_archive(self):
        return load_bundle(self.directory, self.session_id)

    def __init__(self, runtime, config, session_id=None, clock=time.monotonic):
        self.runtime, self.config = runtime, config
        self.clock = clock
        self.session_id = session_id or uuid.uuid4().hex
        self.directory = Path(runtime.run_output_dir) / "exploration"
        self.map_path = self.directory / "map.html"
        self.home = np.asarray(runtime.vehicle.get_position()).copy()
        self.home_yaw = float(runtime.vehicle.get_euler()[2])
        if not np.all(np.isfinite(self.home)) or not math.isfinite(self.home_yaw):
            raise ValueError("Initial pose is invalid")
        self.memory = SceneMemory(self.session_id, self.home, self.home_yaw, config)
        self.watch = ContinuityWatch(runtime.vehicle, clock)
        self.perception = WatchedPerception(runtime.detection_worker, self.watch)
        self.selected, self.approach, self.route = None, None, None
        self.airborne_navigation = False
        self.exploration_waypoints = []
        self.show_place_links = True

    def return_home(self):
        if self.airborne_navigation:
            self.watch.check()
            # Verification's mission deadline is not the return-flight budget.
            nav = self.runtime.controller.semantic_navigator
            nav.deadline = self.clock() + self.runtime.semantic_navigation_config.mission_timeout
            self.runtime.controller.return_and_land(force_return=True)
            self._confirm_home()
            self.airborne_navigation = False
        self.annotate(state="returned home")
        return {"success": True, "returned_home": True, "map_path": str(self.map_path.resolve())}

    def _arguments(self, request, output):
        runtime = self.runtime
        return dict(
            vehicle=runtime.vehicle,
            perception=self.perception,
            local_map=runtime.occupancy_grid,
            global_map=getattr(runtime, "navigation_map", runtime.global_sparse_map),
            local_planner=runtime.local_planner,
            request=request,
            config=runtime.semantic_navigation_config,
            output_path=output,
            clock=self.clock,
        )

    def annotate(self, selected=None, approach=None, route=None, state="exploring"):
        self.selected, self.approach, self.route = selected, approach, route
        live_map = getattr(self.runtime, 'live_loop_mapping', None)
        if live_map is not None:
            live_map.update_display(state=state, vehicle=self.runtime.vehicle.get_position(),
                                    route=route, approach=approach,
                                    waypoints=self.exploration_waypoints,
                                    show_place_links=self.show_place_links,
                                    timestamp=self.runtime.global_sparse_map.snapshot().timestamp)
            return
        render_map(
            self.map_path,
            self.runtime.global_sparse_map.snapshot(),
            self.memory,
            selected=selected,
            approach=approach,
            route=route,
            vehicle=self.runtime.vehicle.get_position(),
            state=state,
            navigation_config=self.runtime.semantic_navigation_config,
            exploration_waypoints=self.exploration_waypoints,
            show_place_links=self.show_place_links,
        )

    def checkpoint(self, explorer):
        self.memory.checkpoint(self.directory)
        self.exploration_waypoints = list(explorer.exploration_waypoints)
        # Geometry is private navigation data, not part of the semantic graph.
        write_json(self.directory / "private" / "exploration_waypoints.json", self.exploration_waypoints)
        self.annotate(route=explorer.active_route,
                      state=f"{explorer.phase}: {explorer.motion_backend}")

    def _confirm_home(self):
        try:
            self.runtime.vehicle.finish_flight_session()
        except TimeoutError as error:
            write_json(self.directory / "private" / "landing_failure.json", {
                "confirmed": False, "error": str(error),
                "home_ned": np.asarray(self.home).tolist(),
                "landing": getattr(self.runtime.vehicle, "landing_confirmation", None),
            })
            raise
        self.watch.check()
        current = np.asarray(self.runtime.vehicle.get_position(), dtype=float)
        delta = current-self.home
        finite = bool(np.isfinite(current).all())
        distance = float(np.linalg.norm(delta)) if finite else None
        detail = {
            "confirmed": finite and distance <= self.config.home_tolerance,
            "home_ned": self.home.tolist(),
            "current_ned": current.tolist() if finite else None,
            "horizontal_error_m": float(np.linalg.norm(delta[:2])) if finite else None,
            "vertical_error_m": abs(float(delta[2])) if finite else None,
            "distance_3d_m": distance, "tolerance_m": self.config.home_tolerance,
            "landing": getattr(self.runtime.vehicle, "landing_confirmation", None),
        }
        self.memory.exploration["home_confirmation"] = detail
        write_json(self.directory / "private" / "home_confirmation.json", detail)
        if not detail["confirmed"]:
            raise RuntimeError(
                "Return-to-home position not confirmed; saved-map navigation is disabled; "
                f"home={detail['home_ned']}, current={detail['current_ned']}, "
                f"horizontal_error={detail['horizontal_error_m']} m, "
                f"vertical_error={detail['vertical_error_m']} m, "
                f"tolerance_3d={self.config.home_tolerance} m"
            )

    def explore(self):
        lidar = getattr(self.runtime, 'lidar_worker', None)
        detector = getattr(lidar, 'loop_detector', None)
        if detector is not None and getattr(self.runtime, 'live_loop_mapping', None) is None:
            from mapping.live_loop_mapping import LiveLoopMapping
            from mapping.explore_v3.adapters import SemanticInstanceMapper, rebuild
            self.runtime.live_loop_mapping = LiveLoopMapping(
                detector, self.runtime.run_output_dir, self.home, self.home_yaw,
                navigation_map=getattr(self.runtime, "navigation_map", None),
                mapper_type=SemanticInstanceMapper, rebuild_map=rebuild)
            self.runtime.live_loop_mapping.start()
        explorer = FrontierExplorer(
            **self._arguments(None, self.directory / "exploration_report.json"),
            exploration_config=self.config,
            memory=self.memory,
            checkpoint=self.checkpoint,
        )
        self.runtime.controller.semantic_navigator = explorer
        try:
            self.runtime.run_mission()
            self._confirm_home()
            self.runtime.finish_exploration_recording()
            frame = self.perception.semantic_snapshot()
            if frame is not None:
                self.memory.observe(frame, "place_0000")
            numbering = self.memory.number_final_instances()
            write_json(self.directory / "instance_id_mapping.json", {
                "source_namespace": "live_detection", "target_namespace": "final_map",
                "old_to_new": numbering,
            })
            self.memory.exploration["returned_home"] = True
            explorer.result["returned_home"] = True
            write_json(
                self.directory / "exploration_report.json", self.memory.exploration
            )
            self.memory.save_bundle(
                self.directory, self.runtime.global_sparse_map.snapshot(), explorer.config
            )
            self.annotate(state="READY: returned home")
        except BaseException as error:
            self.memory.exploration["returned_home"] = False
            self.memory.exploration["failure_phase"] = explorer.phase
            self.memory.exploration["error"] = str(error)
            try:
                self.memory.checkpoint(self.directory)
                self.memory.save_bundle(self.directory, self.runtime.global_sparse_map.snapshot(), explorer.config)
                write_json(self.directory / "exploration_report.json", self.memory.exploration)
                self.annotate(state=f"未确认返航完成：{explorer.phase} / {error}")
            except Exception as error:
                print(f"--- 无法保存异常检查点：{error}；保留原始异常 ----")
            raise
        return explorer.result

    def navigate(self, request, job_id):
        self.show_place_links = False
        self.watch.check()
        if not self.airborne_navigation and (
            np.linalg.norm(self.runtime.vehicle.get_position() - self.home)
            > self.config.home_tolerance
        ):
            raise RuntimeError("Vehicle moved away from the saved home frame")
        self.perception.raise_if_failed()
        frame = self.perception.semantic_snapshot()
        if (
            frame is None
            or self.clock() - frame.completed_at
            > self.runtime.semantic_navigation_config.perception_timeout
        ):
            raise NavigationNotReadyError("感知暂未更新；未执行新目标，请稍后重新输入")
        for name, source in (("local", self.runtime.occupancy_grid),
                             ("global", self.runtime.global_sparse_map)):
            # Compare completion age in the monotonic clock, not map capture
            # time against a semantic image captured before slow inference.
            stamp = source.last_observation_completed_at
            age = self.clock() - stamp
            if stamp <= 0 or age > self.runtime.semantic_navigation_config.map_timeout:
                raise NavigationNotReadyError(
                    f"{name} 地图暂未更新（距完成 {age:.2f}s）；未执行新目标，请稍后重新输入")
        archive = self.navigation_archive()
        if not np.allclose(archive["home"], self.home):
            raise RuntimeError("Saved home does not match active control session")
        start = self.runtime.vehicle.get_position().copy()
        if not self.airborne_navigation:
            start[2] -= self.runtime.mission_config.takeoff_height
        matches = select_targets(archive["records"], request, start, 0, check_age=False)
        if not matches:
            self.annotate(state="target not found: no flight started")
            raise TargetUnavailableError(
                "No matching object in the saved semantic map; no flight started"
            )
        grid = NavigationGrid.from_global(
            self.runtime.global_sparse_map.snapshot(),
            start[2],
            self.runtime.semantic_navigation_config,
        )
        chosen = None
        for candidate in matches:
            approach = grid.approach(candidate, start)
            if approach is not None:
                chosen = candidate, approach
                break
        if chosen is None:
            self.annotate(
                matches[0]["runtime_instance_id"],
                state="unreachable: no flight started",
            )
            raise TargetUnavailableError(
                "Saved target has no known-free approach route; no flight started"
            )
        record, (_, goal, path) = chosen
        pinned = replace(request, instance_id=record["runtime_instance_id"])
        self.annotate(record["runtime_instance_id"], goal, path, "selected")
        navigator = self.navigator_type(
            **self._arguments(
                pinned, self.directory / "private" / f"navigation_{job_id}.json"
            ),
            records=archive["records"],
            annotate=self.annotate,
        )
        self.runtime.controller.semantic_navigator = navigator
        self.runtime.controller.run_target(resume=self.airborne_navigation)
        self.airborne_navigation = True
        result = {
            "success": bool(navigator.result["success"]),
            "reason": navigator.result["reason"],
            "navigation_arrived": bool(navigator.result["success"]),
            "visual_confirmed": bool(navigator.result["success"] and
                                     navigator.result["reason"] == "visually_confirmed"),
            "target_id": record["runtime_instance_id"],
            "map_path": str(self.map_path.resolve()),
            "returned_home": False,
            "holding_at_target": bool(navigator.result["success"]),
            "waiting_for_next_target": True,
        }
        write_json(self.directory / f"target_result_{job_id}.json", result)
        self.annotate(
            record["runtime_instance_id"],
            self.approach,
            self.route,
            (
                ("已到达；视觉已确认；悬停等待下一目标" if result["visual_confirmed"]
                 else "已到达地图接近点；视觉未确认；悬停等待下一目标")
                if result["navigation_arrived"]
                else "导航未完成；等待指令"
            ),
        )
        return result


def serve_target_commands(session, server):
    from .terminal_commands import TerminalCommands
    server.set_state('READY')
    console = TerminalCommands(server)
    print('--- 已就绪：目标导航后悬停，可直接在本终端输入下一个目标或 home ----')
    console.start()
    try:
        return _serve_target_commands(session, server)
    finally:
        console.close()


def _serve_target_commands(session, server):
    """Keep a single SDK owner alive between sequential target requests."""
    import queue
    while True:
        try:
            command, identity, request = server.commands.get(timeout=.25)
        except queue.Empty:
            if server.state == 'READY':
                try:
                    session.watch.check()
                    if session.airborne_navigation:
                        session.runtime.vehicle._require_offboard()
                        if not session.runtime.vehicle.mav.isArmed:
                            raise RuntimeError('Vehicle disarmed while waiting for next target')
                        session.runtime.vehicle.raise_if_collision()
                except Exception as error:
                    server.set_state('ERROR')
                    print(f'--- 等待目标期间状态异常：{error} ----')
                    session.runtime.land_if_armed(hover_first=True)
            continue
        try:
            if command == 'shutdown':
                session.return_home()
                return
            result = (session.return_home() if command == 'return_home'
                      else session.navigate(request, identity))
            server.finish(identity, result)
        except (TargetUnavailableError, NavigationNotReadyError) as error:
            server.finish(identity, {'success': False, 'reason': str(error),
                                     'flight_started': session.airborne_navigation,
                                     'waiting_for_next_target': True})
        except Exception as error:
            server.finish(identity, {'success': False, 'reason': str(error)}, fatal=True)
            session.runtime.land_if_armed(hover_first=True)
            session.annotate(session.selected, session.approach, session.route, 'ERROR: '+str(error))
            if command == 'shutdown':
                return
