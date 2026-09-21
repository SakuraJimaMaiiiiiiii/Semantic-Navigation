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

    def _arguments(self, request, output):
        runtime = self.runtime
        return dict(
            vehicle=runtime.vehicle,
            perception=self.perception,
            local_map=runtime.occupancy_grid,
            global_map=runtime.global_sparse_map,
            local_planner=runtime.local_planner,
            request=request,
            config=runtime.semantic_navigation_config,
            output_path=output,
            clock=self.clock,
        )

    def annotate(self, selected=None, approach=None, route=None, state="exploring"):
        self.selected, self.approach, self.route = selected, approach, route
        render_map(
            self.map_path,
            self.runtime.global_sparse_map.snapshot(),
            self.memory,
            selected=selected,
            approach=approach,
            route=route,
            vehicle=self.runtime.vehicle.get_position(),
            state=state,
        )

    def checkpoint(self, explorer):
        self.memory.checkpoint(self.directory)
        route = None
        if explorer.frontiers.visit_order:
            route = [self.runtime.vehicle.get_position()]
            grid = explorer._grid()
            for key in explorer.frontiers.visit_order:
                segment = grid.path(route[-1], explorer.frontiers.candidates[key].point)
                if segment is None:
                    break
                route.extend(segment[1:])
        self.annotate(route=route, state="exploring: frontier regions / fixed-height EGO")

    def _confirm_home(self):
        self.runtime.vehicle.finish_flight_session()
        self.watch.check()
        if (
            np.linalg.norm(self.runtime.vehicle.get_position() - self.home)
            > self.config.home_tolerance
        ):
            raise RuntimeError(
                "Return-to-home position not confirmed; saved-map navigation is disabled"
            )

    def explore(self):
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
            frame = self.perception.semantic_snapshot()
            if frame is not None:
                self.memory.observe(frame, "place_0000")
            self.memory.exploration["returned_home"] = True
            explorer.result["returned_home"] = True
            write_json(
                self.directory / "exploration_report.json", self.memory.exploration
            )
            self.memory.save_bundle(
                self.directory, self.runtime.global_sparse_map.snapshot()
            )
            self.annotate(state="READY: returned home")
        except BaseException:
            self.memory.exploration["returned_home"] = False
            try:
                self.memory.checkpoint(self.directory)
            except Exception as error:
                print(f"--- 无法保存异常检查点：{error}；保留原始异常 ----")
            raise
        return explorer.result

    def navigate(self, request, job_id):
        self.watch.check()
        if (
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
            raise RuntimeError("Perception is stale; no flight started")
        for source in (self.runtime.occupancy_grid, self.runtime.global_sparse_map):
            stamp = source.snapshot().timestamp
            if (
                stamp <= 0
                or abs(stamp - frame.timestamp)
                > self.runtime.semantic_navigation_config.map_timeout
            ):
                raise RuntimeError("Mapping is stale; no flight started")
        archive = load_bundle(self.directory, self.session_id)
        if not np.allclose(archive["home"], self.home):
            raise RuntimeError("Saved home does not match active control session")
        start = self.runtime.vehicle.get_position().copy()
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
        navigator = SavedMapNavigator(
            **self._arguments(
                pinned, self.directory / "private" / f"navigation_{job_id}.json"
            ),
            records=archive["records"],
            annotate=self.annotate,
        )
        self.runtime.controller.semantic_navigator = navigator
        self.runtime.run_mission()
        self._confirm_home()
        result = {
            "success": bool(navigator.result["success"]),
            "reason": navigator.result["reason"],
            "target_id": record["runtime_instance_id"],
            "map_path": str(self.map_path.resolve()),
            "returned_home": True,
        }
        write_json(self.directory / f"target_result_{job_id}.json", result)
        self.annotate(
            record["runtime_instance_id"],
            self.approach,
            self.route,
            (
                "verified and returned home"
                if result["success"]
                else "failed and returned home"
            ),
        )
        return result
