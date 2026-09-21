"""DFS order, semantic archives, loopback commands and two-stage navigation."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import math

import numpy as np
import pytest

from config.exploration_config import DFSExplorationConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from config.mission_config import MissionConfig
from navigation.dfs_explorer import DFSExplorer
from navigation.scene_memory import SceneMemory, load_bundle
from navigation.map_view import render_map
from navigation.perception import SemanticFrame
from navigation.session_server import SessionServer, send_command
from navigation.target import SemanticTargetRequest
from navigation.exploration_session import (
    ExplorationSession,
    ContinuityWatch,
    TargetUnavailableError,
)
from mission.rfly_mission_controller import RflyMissionController


def obj(identity="vehicle_01", position=(5, 0, -1.5), stamp=10, kind="car"):
    p = np.asarray(position, dtype=float)
    return {
        "instance_id": identity,
        "runtime_instance_id": identity,
        "class_name": kind,
        "position_world": p.tolist(),
        "bbox_3d": {"min": (p - 0.5).tolist(), "max": (p + 0.5).tolist()},
        "confidence": 0.9,
        "observation_count": 10,
        "last_seen": stamp,
        "color": "red",
        "color_confidence": 0.9,
    }


def snapshot(stamp=10):
    indices = np.array(
        [(i, j, k) for i in range(-10, 21) for j in range(-10, 21) for k in (-4, -3)],
        dtype=np.int32,
    )
    return SimpleNamespace(
        timestamp=stamp,
        indices=indices,
        log_odds=np.full(len(indices), -1.0),
        resolution=0.5,
        occupied_threshold=0.7,
        free_threshold=-0.2,
    )


def memory():
    result = SceneMemory("session-1", [0, 0, 0], 0, DFSExplorationConfig())
    result.add_place("place_0000", [0, 0, -1.5])
    result.add_place("place_0001", [1, 0, -1.5], "place_0000")
    result.observe(
        SemanticFrame(
            10, 10, 0.1, (obj(), obj("column_01", (5, 2, -1.5), kind="column"))
        ),
        "place_0001",
    )
    return result


def test_scene_graph_has_semantics_relations_and_no_absolute_geometry():
    graph = memory().graph()
    assert len(graph["objects"]) == 2
    assert {r["relation"] for r in graph["relations"]} >= {"near", "co_observed"}
    assert graph["connections"][0]["relation"] == "traversed_to"
    assert graph["objects"][0]["observed_at"] == ["place_0001"]

    def check(value):
        if isinstance(value, dict):
            assert not (
                {
                    "position_world",
                    "position_ned",
                    "bbox_3d",
                    "position",
                    "home",
                    "initial_yaw",
                }
                & set(value)
            )
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(graph)


def test_direction_reference_is_initial_heading():
    scene = memory()
    scene.initial_yaw = math.pi / 2
    # column is east of the vehicle, hence forward in an initially east-facing frame.
    relations = scene.graph()["relations"]
    assert any(
        r["source"] == "column_01"
        and r["relation"] == "in_front_of"
        and r["target"] == "vehicle_01"
        for r in relations
    )


def test_coobservation_counts_distinct_fresh_frames_only():
    scene = memory()
    scene.observe(
        SemanticFrame(10, 11, 0.1, tuple(scene.records.values())), "place_0001"
    )
    scene.observe(
        SemanticFrame(11, 11, 0.1, tuple(scene.records.values())), "place_0000"
    )
    assert (
        next(r for r in scene.graph()["relations"] if r["relation"] == "co_observed")[
            "frame_count"
        ]
        == 1
    )


def test_bundle_roundtrip_requires_same_session_return_and_checksum(tmp_path):
    scene = memory()
    scene.save_bundle(tmp_path, snapshot())
    with pytest.raises(ValueError, match="return and landing"):
        load_bundle(tmp_path, "session-1")
    scene.exploration["returned_home"] = True
    scene.save_bundle(tmp_path, snapshot())
    restored = load_bundle(tmp_path, "session-1")
    assert restored["records"][0]["runtime_instance_id"] in scene.records
    np.testing.assert_allclose(restored["home"], [0, 0, 0])
    with pytest.raises(ValueError, match="another control session"):
        load_bundle(tmp_path, "wrong")
    path = tmp_path / "private" / "navigation_cache.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path, "session-1")


def test_html_marks_selected_target_goal_and_route(tmp_path):
    path = tmp_path / "map.html"
    render_map(
        path,
        snapshot(),
        memory(),
        selected="vehicle_01",
        approach=[3.3, 0, -1.5],
        route=[[0, 0, -1.5], [3.3, 0, -1.5]],
        state="selected",
    )
    text = path.read_text()
    assert (
        "vehicle_01" in text
        and "#e53935" in text
        and "polyline" in text
        and "GOAL" in text
    )
    assert "position_world" not in text
    empty = SceneMemory("empty", [0, 0, 0], 0, DFSExplorationConfig())
    render_map(path, snapshot(), empty)
    assert "<svg" in path.read_text()


def make_explorer(tmp_path, monkeypatch, max_viewpoints=10, fail_goal=False):
    position = np.array([0.0, 0.0, -1.5])
    vehicle = SimpleNamespace(
        get_position=lambda: position.copy(),
        get_euler=lambda: np.zeros(3),
        hover=lambda: None,
        wait=lambda _seconds: None,
    )
    perception = SimpleNamespace(
        semantic_snapshot=lambda: SemanticFrame(10, 10, 0, ()),
        raise_if_failed=lambda: None,
    )
    scene = SceneMemory("dfs", [0, 0, 0], 0, DFSExplorationConfig())
    explorer = DFSExplorer(
        vehicle,
        perception,
        None,
        None,
        SimpleNamespace(config=None),
        None,
        SemanticNavigationConfig(),
        tmp_path / "dfs.json",
        exploration_config=replace(
            DFSExplorationConfig(),
            max_viewpoints=max_viewpoints,
            max_frontier_attempts=1,
            frontier_empty_confirmations=1,
            frontier_empty_min_duration=0.001,
        ),
        memory=scene,
    )
    fake_grid = SimpleNamespace(
        cell=lambda p: (0, 0),
        free={(0, 0)},
        known={(0, 0)},
        resolution=0.5,
    )
    explorer._grid = lambda: fake_grid
    explorer._check_health = lambda: None

    scans = []

    def full_scan(reason):
        scans.append((explorer.current_place, reason))
        scene.places[explorer.current_place]["scanned"] = True

    def observe_arrival():
        scene.places[explorer.current_place]["scanned"] = False

    explorer._full_scan = full_scan

    def recovery_observation(_grid, bin_key):
        scans.append((explorer.current_place, "best_unknown_sector"))
        used = explorer.recovery_yaws.setdefault(bin_key, set())
        used.add(len(used))
        return True

    explorer._recovery_observation = recovery_observation
    explorer._observe_arrival = observe_arrival
    moves = []
    completed_goals = set()

    def travel(goal):
        if fail_goal and tuple(goal[:2]) == (4.0, 0.0):
            return False
        moves.append(tuple(goal[:2]))
        completed_goals.add(tuple(goal[:2]))
        position[:] = goal
        return True

    explorer._travel = travel
    goals = [
        ((2, 0), np.array([2.0, 0.0, -1.5])),
        ((4, 0), np.array([4.0, 0.0, -1.5])),
        ((0, 10), np.array([0.0, 10.0, -1.5])),
    ]

    def choices(current):
        remaining = [
            goal
            for goal in goals
            if tuple(goal[1][:2]) not in completed_goals
            and goal[0] not in explorer.failed_frontiers
        ]
        if tuple(current[:2]) == (0.0, 0.0):
            remaining = [g for g in remaining if g[0] != (4, 0)]
        return remaining, False, False

    monkeypatch.setattr(
        explorer, "_choices", lambda grid, current, origin: choices(current)
    )
    return explorer, moves, scans


def test_frontier_explorer_moves_between_gain_targets_without_scanning_each_stop(
    tmp_path, monkeypatch
):
    explorer, moves, scans = make_explorer(tmp_path, monkeypatch)
    assert explorer.run()
    assert moves == [(2.0, 0.0), (4.0, 0.0), (0.0, 10.0)]
    assert scans == [
        ("place_0000", "initial_map"),
        ("place_0003", "best_unknown_sector"),
        ("place_0003", "best_unknown_sector"),
    ]
    assert explorer.result["viewpoint_count"] == 4
    assert explorer.result["strategy"] == "fuel_inspired_frontier"
    assert explorer.result["coverage_guaranteed"] is False
    assert (
        explorer.result["returned_home"] is False
    )  # Only the flight session can confirm landing.


def test_frontier_explorer_budget_and_blocked_goal_remain_partial(
    tmp_path, monkeypatch
):
    explorer, _, _ = make_explorer(tmp_path, monkeypatch, max_viewpoints=2)
    assert explorer.run() is False
    assert explorer.result["reason"] == "viewpoint_budget_exhausted"
    explorer, _, _ = make_explorer(tmp_path, monkeypatch, fail_goal=True)
    assert explorer.run() is False
    assert explorer.result["blocked_frontiers"] == 1


def test_loopback_server_rejects_early_or_duplicate_flights_and_preserves_results():
    server = SessionServer("test-session", 0)
    try:
        descriptor = server.descriptor()
        assert send_command(descriptor, "status")["state"] == "EXPLORING"
        request = SemanticTargetRequest(class_name="car")
        with pytest.raises(RuntimeError, match="EXPLORING"):
            send_command(descriptor, "navigate", request)
        server.set_state("READY")
        first = send_command(descriptor, "navigate", request)["job_id"]
        with pytest.raises(RuntimeError, match="NAVIGATING"):
            send_command(descriptor, "navigate", request)
        assert server.commands.get_nowait()[1] == first
        server.finish(first, {"success": True})
        second = send_command(descriptor, "navigate", request)["job_id"]
        assert (
            send_command(descriptor, "status", job_id=first)["job"]["state"]
            == "succeeded"
        )
        assert second != first
        with pytest.raises(RuntimeError, match="credentials"):
            send_command({**descriptor, "token": "wrong"}, "status")
    finally:
        server.close()


def test_failed_session_allows_shutdown_but_not_navigation():
    server = SessionServer("test-session", 0)
    try:
        server.set_state("ERROR")
        with pytest.raises(RuntimeError):
            send_command(
                server.descriptor(), "navigate", SemanticTargetRequest(class_name="car")
            )
        assert send_command(server.descriptor(), "shutdown")["ok"]
    finally:
        server.close()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Vehicle:
    def __init__(self, clock):
        self.clock = clock
        self.position = np.zeros(3)
        self.mav = SimpleNamespace(uavTimeStmp=0, isArmed=False)
        self.config = SimpleNamespace()
        self.moves = []
        self.landings = 0

    def get_position(self):
        return self.position.copy()

    def get_euler(self):
        return np.zeros(3)

    def wait(self, seconds):
        self.clock.now += seconds

    def align_yaw(self, *args, **kwargs):
        self.clock.now += 0.1

    def enable_offboard(self, arm=True):
        self.mav.isArmed = arm

    def takeoff(self, height, **kwargs):
        self.position[2] -= height

    def move_velocity_guided(self, point, **kwargs):
        self.position = np.array(point)
        self.moves.append(self.position.copy())
        self.clock.now += 0.5

    def hover(self):
        pass

    def land(self):
        self.position[2] = 0
        self.mav.isArmed = False
        self.landings += 1

    def finish_flight_session(self):
        assert not self.mav.isArmed


def test_saved_map_target_flight_confirms_then_returns_without_scene_reset(
    tmp_path, monkeypatch
):
    clock = Clock()
    vehicle = Vehicle(clock)
    runtime = SimpleNamespace(
        run_output_dir=tmp_path,
        vehicle=vehicle,
        mission_config=replace(MissionConfig(), takeoff_wait=0.1),
        semantic_navigation_config=SemanticNavigationConfig(),
        local_planner=SimpleNamespace(config=None),
    )
    runtime.global_sparse_map = SimpleNamespace(snapshot=lambda: snapshot(clock()))
    runtime.occupancy_grid = runtime.global_sparse_map
    runtime.detection_worker = SimpleNamespace(
        semantic_snapshot=lambda: SemanticFrame(
            clock(), clock(), 0.01, (obj(stamp=clock()),)
        ),
        raise_if_failed=lambda: None,
    )
    runtime.controller = RflyMissionController(
        vehicle, runtime.mission_config, local_planner=runtime.local_planner
    )
    runtime.run_mission = runtime.controller.run
    session = ExplorationSession(
        runtime, DFSExplorationConfig(), session_id="same-live-session", clock=clock
    )
    session.memory.add_place("place_0000", [0, 0, -1.5])
    session.memory.records = {
        "vehicle_01": obj(stamp=1),
        "column_01": obj("column_01", (5, 2, -1.5), stamp=1, kind="column"),
    }
    session.memory.exploration["returned_home"] = True
    session.memory.save_bundle(session.directory, snapshot())
    result = session.navigate(
        SemanticTargetRequest(
            instance_id="vehicle_01", near_instance_id="column_01", max_age=2
        ),
        "job-1",
    )
    assert result["success"] and result["returned_home"]
    np.testing.assert_allclose(vehicle.get_position(), [0, 0, 0])
    assert vehicle.landings == 1
    assert "vehicle_01" in session.map_path.read_text()
    assert (session.directory / "target_result_job-1.json").exists()
    with pytest.raises(TargetUnavailableError):
        session.navigate(SemanticTargetRequest(instance_id="missing"), "job-2")
    assert vehicle.landings == 1


def test_continuity_detects_reboot_and_position_jump():
    clock = Clock()
    vehicle = Vehicle(clock)
    watch = ContinuityWatch(vehicle, clock)
    vehicle.mav.uavTimeStmp = 10
    watch.check()
    vehicle.mav.uavTimeStmp = 5
    with pytest.raises(RuntimeError, match="reset"):
        watch.check()
    vehicle.mav.uavTimeStmp = 20
    with pytest.raises(RuntimeError, match="invalid"):
        watch.check()
    watch = ContinuityWatch(vehicle, clock)
    vehicle.position[0] = 10
    with pytest.raises(RuntimeError, match="position jump"):
        watch.check()


def test_finish_flight_session_keeps_connections_and_resets_offboard():
    from vehicle.rfly_multirotor import RflyMultirotorInterface

    vehicle = RflyMultirotorInterface.__new__(RflyMultirotorInterface)
    vehicle._require_connection = lambda: None
    vehicle.wait_until_landed = lambda: True
    vehicle.mav = MagicMock()
    vehicle.mav.isArmed = False
    vehicle.offboard_enabled = True
    vehicle.armed = True
    vehicle._landing_commanded = True
    vehicle.finish_flight_session()
    vehicle.mav.endOffboard.assert_called_once()
    vehicle.mav.stopRun.assert_not_called()
    assert not vehicle.offboard_enabled and not vehicle.armed
    vehicle.wait_until_landed = lambda: False
    with pytest.raises(TimeoutError):
        vehicle.finish_flight_session()


def test_exploration_runtime_uses_coordinate_free_output_and_global_map(tmp_path):
    from application.demo_runtime import DemoRuntime

    runtime = DemoRuntime.__new__(DemoRuntime)
    runtime.semantic_target = None
    runtime.exploration_mode = True
    runtime.semantic_navigation_config = SemanticNavigationConfig()
    runtime.run_output_dir = tmp_path
    runtime._create_configs()
    assert (
        runtime.global_map_config.enabled
        and not runtime.global_map_config.loop_closure_enabled
    )
    assert not runtime.detector_config.save_semantic_map
    assert not runtime.global_map_config.save_on_close
    assert runtime.mission_config.return_to_origin


def test_session_exploration_saves_only_after_landing_and_home_confirmation(
    tmp_path, monkeypatch
):
    clock = Clock()
    vehicle = Vehicle(clock)
    runtime = SimpleNamespace(
        run_output_dir=tmp_path,
        vehicle=vehicle,
        mission_config=replace(MissionConfig(), takeoff_wait=0.1),
        semantic_navigation_config=SemanticNavigationConfig(),
        local_planner=SimpleNamespace(config=None),
    )
    runtime.global_sparse_map = SimpleNamespace(snapshot=lambda: snapshot(clock()))
    runtime.occupancy_grid = runtime.global_sparse_map
    runtime.detection_worker = SimpleNamespace(
        semantic_snapshot=lambda: SemanticFrame(
            clock(), clock(), 0.01, (obj(stamp=clock()),)
        ),
        raise_if_failed=lambda: None,
    )
    runtime.controller = RflyMissionController(
        vehicle, runtime.mission_config, local_planner=runtime.local_planner
    )
    runtime.run_mission = runtime.controller.run

    class Explorer:
        def __init__(self, **kwargs):
            self.memory = kwargs["memory"]
            self.result = {
                "reason": "no_new_reachable_frontiers",
                "returned_home": False,
            }

        def run(self):
            self.memory.add_place("place_0000", vehicle.get_position())
            self.memory.exploration = self.result.copy()
            vehicle.position[0] = 2
            clock.now += 3
            return True

    monkeypatch.setattr("navigation.exploration_session.FrontierExplorer", Explorer)
    session = ExplorationSession(
        runtime, DFSExplorationConfig(), session_id="explore-session", clock=clock
    )
    result = session.explore()
    assert result["returned_home"] and vehicle.landings == 1
    graph = json.loads((session.directory / "semantic_graph.json").read_text())
    assert graph["exploration"]["returned_home"]
    assert load_bundle(session.directory, "explore-session")["records"]


def test_stale_telemetry_blocks_future_tasks():
    clock = Clock()
    vehicle = Vehicle(clock)
    vehicle.mav.uavTimeStmp = 50
    watch = ContinuityWatch(vehicle, clock)
    watch.check()
    clock.now += 4
    with pytest.raises(RuntimeError, match="stale telemetry"):
        watch.check()


def test_second_control_listener_cannot_bind_same_endpoint():
    server = SessionServer("one", 0)
    try:
        with pytest.raises(OSError):
            SessionServer("two", server.descriptor()["port"])
    finally:
        server.close()


def test_radius_limit_is_not_reported_as_map_completion(tmp_path, monkeypatch):
    explorer, _, _ = make_explorer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        explorer,
        "_choices",
        lambda *args, **kwargs: ([], False, True),
    )
    assert explorer.run() is False
    assert explorer.result["reason"] == "radius_limit"
    assert not explorer.result["frontiers_exhausted"]
