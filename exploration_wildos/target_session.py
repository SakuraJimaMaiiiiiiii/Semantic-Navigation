"""Preflight graph validation and target session flow for WildOS only."""
from dataclasses import replace
from types import SimpleNamespace
from .vehicle_motion import VehiclePlanningMap
import numpy as np
from navigation.exploration_session import NavigationNotReadyError, TargetUnavailableError
from navigation.target import select_targets
from navigation.perception_mode import active_perception
from navigation.route import NavigationGrid
from navigation.scene_memory import write_json
from .map_adapter import MapAdapter
from .target_graph import TargetGraphGrid
from .vehicle_constraints import constrain_adapter
from .semantic_graph import attach_objects


class GraphTargetSessionMixin:
    @active_perception
    def return_home(self):
        # Return still uses the existing certified return planner. Never leave
        # the outbound graph route highlighted as if it were the return route.
        state = dict(phase='返航：现有返航规划',route=None,target_id=None,
                     active_segment=None,trace_ned=[])
        navigator = getattr(self.runtime.controller,'semantic_navigator',None)
        if hasattr(navigator,'_navigation_state'):
            navigator._navigation_state = state
        viewer = getattr(self,'live_map',None)
        if viewer is not None:
            viewer.navigation(state)
        return super().return_home()

    def navigation_archive(self):
        from .identity_overrides import corrected_records
        archive = super().navigation_archive()
        archive['records'] = corrected_records(archive['records'], self.directory, self.session_id)
        return archive

    def _validate_target_frame(self):
        explorer = getattr(self, 'graph_explorer', None)
        if explorer is None or explorer.frontiers.graph is None:
            raise TargetUnavailableError('No active exploration graph; run WildOS exploration first')
        expected = f"{self.memory.session_id}:PX4-raw:{getattr(self.runtime.global_sparse_map, 'revision', 0)}"
        if explorer.frontiers.epoch is None or explorer.frontiers.epoch[0] != expected:
            raise TargetUnavailableError('Exploration graph frame changed; graph navigation withheld')
        return explorer.frontiers.graph

    def _target_objects(self):
        temporary = {'nodes': []}
        from .identity_overrides import corrected_records
        records = corrected_records(self.memory.records.values(), self.directory, self.memory.session_id)
        attach_objects(temporary, {'objects': records}, {},
                       self.runtime.semantic_navigation_config.clearance_radius, max_links=0)
        return temporary['semantic_nodes']

    def _target_grid(self, altitude, current):
        graph = self._validate_target_frame()
        source = VehiclePlanningMap(self.runtime.global_sparse_map, self._target_objects,
                                    SimpleNamespace(altitude=altitude, config=self.runtime.semantic_navigation_config))
        grid = NavigationGrid.from_global(source.snapshot(), altitude,
            self.runtime.semantic_navigation_config, local_snapshot=self.runtime.occupancy_grid.snapshot(),
            current=current)
        adapter, _ = constrain_adapter(MapAdapter.from_grid(grid), self._target_objects())
        return TargetGraphGrid(adapter.grid, graph)

    @active_perception
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
        grid = self._target_grid(start[2], start)
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
                "Saved target has no certified WildOS graph approach route; no flight started"
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
            "planning_backend": "wildos_graph_dijkstra",
            "graph_route": navigator.graph_routes[-1] if navigator.graph_routes else None,
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
