"""Semantic navigation contracts and simulated closed-loop regression tests."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Only simulator transports are replaced; map, association and navigation are real.
for name in ("PX4MavCtrlV4", "ReqCopterSim", "UE4CtrlAPI"):
    sys.modules.setdefault(name, ModuleType(name))

from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.target import SemanticTargetRequest, load_target_request, select_targets
from navigation.perception import SemanticFrameStore, SemanticFrame
from navigation.route import NavigationGrid
from navigation.mission import SemanticNavigator, _FlightGuard
from navigation.evaluation import evaluate_map, evaluate_identity_frames, summarize_runs
from mapping.semantic_instance_mapper import (
    SemanticInstanceMapper,
    SemanticObservation3D,
    canonical_semantic_class_name,
)
from planner.local_avoidance_planner import LocalPlan


def record(identity="vehicle_01", position=(5.0, 0.0, -1.5), stamp=10.0, **fields):
    center = np.array(position)
    return {
        "instance_id": identity,
        "runtime_instance_id": identity,
        "class_name": "car",
        "position_world": center.tolist(),
        "bbox_3d": {
            "min": (center - [0.5, 0.5, 0.5]).tolist(),
            "max": (center + [0.5, 0.5, 0.5]).tolist(),
        },
        "confidence": 0.9,
        "observation_count": 10,
        "last_seen": stamp,
        "color": "red",
        "color_confidence": 0.9,
        **fields,
    }


def test_request_validation_and_empty_interface(tmp_path):
    path = tmp_path / "target.json"
    path.write_text('{"target": null}')
    assert load_target_request(path) is None
    for kwargs in (
        {},
        {"class_name": ""},
        {"class_name": "car", "max_age": float("nan")},
        {"class_name": "car", "min_observations": 1.5},
    ):
        with pytest.raises(ValueError):
            SemanticTargetRequest(**kwargs)
    path.write_text('{"target": {"instance_id": "vehicle_01"}}')
    assert load_target_request(path).instance_id == "vehicle_01"


def test_target_request_supports_standalone_comments(tmp_path):
    path = tmp_path / "target.json"
    path.write_text(
        '// 配置说明\n{\n  // 被注释的示例不生效\n'
        '  // "target": {"class_name": "car"}\n'
        '  "target": {"instance_id": "vehicle_02"}\n}\n',
        encoding="utf-8",
    )
    assert load_target_request(path).instance_id == "vehicle_02"
    path.write_text('{"target": {"class_name": "a//b"}}')
    assert load_target_request(path).class_name == "a//b"
    default = Path(__file__).resolve().parents[1] / "config" / "semantic_target.json"
    assert load_target_request(default) is None


def test_query_filters_identity_color_age_and_relation():
    records = [
        record(),
        record("vehicle_02", (3, 0, -1.5), color="blue"),
        record("column_01", (5, 1, -1.5), class_name="column"),
    ]
    req = SemanticTargetRequest(
        class_name="vehicle", color="red", near_instance_id="column_01", near_distance=2
    )
    assert [
        r["instance_id"] for r in select_targets(records, req, [0, 0, -1.5], 11)
    ] == ["vehicle_01"]
    assert not select_targets(records, req, [0, 0, -1.5], 200)
    assert not select_targets(
        records, replace(req, near_instance_id="missing"), [0, 0, -1.5], 11
    )
    assert not select_targets(
        records, SemanticTargetRequest(instance_id="vehicle_99"), [0, 0, -1.5], 11
    )


def test_frame_store_copy_and_out_of_order():
    store = SemanticFrameStore(clock=lambda: 100)
    records = [record()]
    store.publish(10, records, 0.2)
    records[0]["class_name"] = "wrong"
    frame = store.snapshot()
    frame.records[0]["class_name"] = "wrong"
    store.publish(9, [], 0.1)
    assert store.snapshot().records[0]["class_name"] == "car"
    assert store.snapshot().completed_at == 100


def grid_with_wall(gap=True):
    config = replace(
        SemanticNavigationConfig(), clearance_radius=0.1, standoff_distance=1.2
    )
    free = {(i, j) for i in range(-5, 15) for j in range(-8, 9)}
    occupied = {(3, j) for j in range(-8, 9) if not gap or j != 5}
    return NavigationGrid(free - occupied, occupied, 0.5, (0, 0), -1.5, config)


def test_astar_goes_through_gap_and_never_unknown():
    grid = grid_with_wall()
    path = grid.path([0, 0, -1.5], [4, 0, -1.5])
    assert path is not None
    assert max(path[:, 1]) >= 2.5
    assert all(grid.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))
    assert grid.path([0, 0, -1.5], [100, 0, -1.5]) is None
    assert grid_with_wall(False).path([0, 0, -1.5], [4, 0, -1.5]) is None


def test_approach_outside_box_and_reachable():
    grid = grid_with_wall()
    obj = record(position=(4, 0, -1.5))
    choice = grid.approach(obj, [0, 0, -1.5])
    assert choice is not None
    _, goal, path = choice
    low, high = np.array(obj["bbox_3d"]["min"]), np.array(obj["bbox_3d"]["max"])
    distance = np.linalg.norm(
        np.maximum(np.maximum(low[:2] - goal[:2], goal[:2] - high[:2]), 0)
    )
    assert distance == pytest.approx(grid.config.standoff_distance)
    assert all(grid.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))


def test_frontier_is_reachable_and_excludes_visited():
    grid = grid_with_wall()
    frontier = grid.frontier([0, 0, -1.5], [0, 0, -1.5], [])
    assert frontier is not None
    point, path = frontier
    assert grid.cell(point) in grid.free
    assert grid.path([0, 0, -1.5], point) is not None
    assert grid.frontier([0, 0, -1.5], [1000, 1000, -1.5], []) is None


def test_global_requires_full_vertical_clearance():
    config = replace(SemanticNavigationConfig(), clearance_radius=0.01)
    snapshot = SimpleNamespace(
        resolution=0.5,
        indices=np.array([[0, 0, -4], [0, 0, -3], [0, 0, -2]]),
        log_odds=np.array([-1.0, -1.0, 1.0]),
        occupied_threshold=0.7,
        free_threshold=-0.2,
    )
    grid = NavigationGrid.from_global(snapshot, -1.5, config)
    assert (0, 0) in grid.free
    snapshot.log_odds[1] = 1
    assert (0, 0) not in NavigationGrid.from_global(snapshot, -1.5, config).free


def test_column_height_and_extended_landmarks():
    mapper = SemanticInstanceMapper()
    short = SemanticObservation3D(
        np.array([0, 0, -1.0]),
        np.array([-0.2, -0.2, -2.0]),
        np.array([0.2, 0.2, 0.0]),
        np.eye(3) * 0.1,
    )
    tall = replace(
        short,
        position_world=np.array([5, 0, -1.5]),
        bbox_3d_min=np.array([4.8, -0.2, -3.0]),
        bbox_3d_max=np.array([5.2, 0.2, 0]),
    )
    sign = replace(
        short,
        position_world=np.array([10, 0, -1.0]),
        bbox_3d_min=np.array([9.8, -0.2, -1.2]),
        bbox_3d_max=np.array([10.2, 0.2, -0.8]),
    )
    for stamp in (1, 2, 3):
        mapper.update(
            ["column", "column", "exit sign"], [0.9] * 3, [short, tall, sign], stamp
        )
    records = mapper.navigation_records()
    assert {r["class_name"] for r in records} == {"column", "exit sign"}
    assert (
        next(r for r in records if r["class_name"] == "column")["instance_id"]
        == "column_02"
    )
    assert canonical_semantic_class_name("fire hose cabinet") == "fire cabinet"


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class World:
    def __init__(self, clock, target=True, fresh=True):
        self.clock, self.target, self.fresh = clock, target, fresh
        self.position = np.array([0.0, 0.0, -1.5])
        self.moves = []
        self.yaws = []

    def semantic_snapshot(self):
        return SemanticFrame(
            self.clock(),
            self.clock(),
            0.05,
            tuple(
                [record(stamp=self.clock() if self.fresh else 10)]
                if self.target
                else []
            ),
        )

    def raise_if_failed(self):
        pass

    def get_position(self):
        return self.position.copy()

    def wait(self, seconds):
        self.clock.now += seconds

    def align_yaw(self, yaw, **kwargs):
        self.yaws.append(yaw)
        self.clock.now += 0.1

    def hover(self):
        pass

    def move_velocity_guided(self, point, **kwargs):
        self.moves.append(np.asarray(point).copy())
        self.position = np.array(point)
        self.clock.now += 0.5

    def snapshot(self):
        indices = np.array(
            [
                (i, j, k)
                for i in range(-10, 30)
                for j in range(-15, 16)
                for k in (-4, -3)
            ]
        )
        return SimpleNamespace(
            timestamp=self.clock(),
            resolution=0.5,
            indices=indices,
            log_odds=np.full(len(indices), -1.0),
            occupied_threshold=0.7,
            free_threshold=-0.2,
        )


def navigator(tmp_path, world, **options):
    config = replace(
        SemanticNavigationConfig(),
        mission_timeout=40,
        verification_timeout=1,
        max_target_attempts=1,
        max_search_steps=2,
        **options
    )
    return SemanticNavigator(
        world,
        world,
        world,
        world,
        SimpleNamespace(config=SimpleNamespace()),
        SemanticTargetRequest(class_name="car"),
        config,
        tmp_path / "result.json",
        clock=world.clock,
    )


def test_closed_loop_requires_fresh_confirmation_and_saves_result(tmp_path):
    world = World(FakeClock())
    nav = navigator(tmp_path, world)
    assert nav.run() is True
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["success"] and result["selected_target_id"] == "vehicle_01"
    assert result["path_length_m"] > 0
    assert "VERIFY" in [e["state"] for e in result["events"]]
    assert result["events"][-1]["state"] == "SUCCEEDED"


def test_old_memory_cannot_confirm_arrival(tmp_path):
    world = World(FakeClock(), fresh=False)
    nav = navigator(tmp_path, world)
    assert nav.run() is False
    assert nav.result["reason"] == "verification_attempts_exhausted"


def test_search_without_target_has_finite_budget(tmp_path):
    world = World(FakeClock(), target=False)
    nav = navigator(tmp_path, world)
    assert nav.run() is False
    assert nav.result["search_steps"] <= 2
    assert "SEARCH_SCAN" in [e["state"] for e in nav.events]


def test_stale_perception_fails_and_records(tmp_path):
    world = World(FakeClock())
    world.semantic_snapshot = lambda: SemanticFrame(1, 1, 0.1, ())
    nav = navigator(tmp_path, world)
    with pytest.raises(RuntimeError, match="perception_stale"):
        nav.run()
    assert nav.result["success"] is False
    assert not world.moves


def test_collision_propagates_and_is_recorded(tmp_path):
    world = World(FakeClock())

    class CollisionDetectedError(RuntimeError):
        pass

    def crash(*args, **kwargs):
        raise CollisionDetectedError("collision")

    world.move_velocity_guided = crash
    nav = navigator(tmp_path, world)
    with pytest.raises(CollisionDetectedError):
        nav.run()
    assert nav.result["collision"]


def test_evaluation_duplicate_error_switches_and_missing_truth():
    truth = [
        {"instance_id": "gt1", "class_name": "car", "position_world": [5, 0, -1.5]}
    ]
    metrics = evaluate_map([record(), record("vehicle_02")], truth)
    assert metrics["duplicate_count"] == 1 and metrics["position_rmse_m"] == 0
    frames = [
        {"timestamp": 10, "records": [record()]},
        {"timestamp": 11, "records": [record("vehicle_02", stamp=11)]},
    ]
    assert evaluate_identity_frames(frames, truth)["id_switches"] == 1
    assert summarize_runs([])["success_rate"] is None


def test_guard_blocks_unknown_trajectory_and_spline_bypass(tmp_path):
    clock = FakeClock()
    world = World(clock)
    nav = navigator(tmp_path, world)
    nav.altitude = -1.5
    nav.started = clock()
    nav.deadline = clock() + 20
    nav._check_health = lambda: None
    states = np.zeros((20, 20, 6), dtype=np.int8)
    snapshot = SimpleNamespace(
        timestamp=1,
        resolution=0.5,
        origin_ned=np.array([-5.0, -5.0, -3.0]),
        states=states,
    )
    nav.local_map = SimpleNamespace(snapshot=lambda: snapshot)
    plan = LocalPlan(
        "ego_clear",
        np.array([1.0, 0.0, -1.5]),
        np.array([[0, 0, -1.5], [1, 0, -1.5]]),
        0.5,
        10,
    )
    nav.local_planner = SimpleNamespace(
        config=SimpleNamespace(),
        plan=lambda *args: plan,
        tracking_target=lambda **kwargs: np.array([100, 100, 100]),
    )
    guard = _FlightGuard(nav)
    assert guard.plan([0, 0, -1.5], [1, 0, -1.5]).can_move
    assert guard.tracking_target(now=10) is None
    states[11:14, :, :] = -1
    snapshot.timestamp = 2
    assert not guard.plan([0, 0, -1.5], [1, 0, -1.5]).can_move


def test_corner_cutting_is_rejected():
    config = replace(SemanticNavigationConfig(), clearance_radius=0.01)
    grid = NavigationGrid({(0, 0), (1, 1)}, set(), 1.0, (0, 0), -1.5, config)
    assert not grid.segment_free([0.5, 0.5, -1.5], [1.5, 1.5, -1.5])


def test_map_stream_stall_aborts_before_motion(tmp_path):
    world = World(FakeClock())
    nav = navigator(tmp_path, world)
    nav.started = 10
    nav.deadline = 100
    world.last_update_timestamp = 1
    nav._check_health()
    world.clock.now += nav.config.map_timeout + 0.1
    with pytest.raises(RuntimeError, match="map_stale"):
        nav._check_health()


def test_mapping_heartbeat_prevents_false_stale_when_map_content_is_unchanged(tmp_path):
    world = World(FakeClock())
    world.last_update_timestamp = 1
    world.last_observation_completed_at = 1
    nav = navigator(tmp_path, world)
    nav.started = 10
    nav.deadline = 100
    nav._check_health()

    world.clock.now += nav.config.map_timeout + 0.1
    world.last_observation_completed_at = 2
    nav._check_health()

    world.clock.now += nav.config.map_timeout + 0.1
    with pytest.raises(RuntimeError, match="map_stale"):
        nav._check_health()


def test_mission_executes_semantic_task_instead_of_waypoints():
    from mission.rfly_mission_controller import RflyMissionController
    from config.mission_config import MissionConfig
    from unittest.mock import MagicMock

    vehicle = MagicMock()
    vehicle.get_position.return_value = np.zeros(3)
    vehicle.get_euler.return_value = np.zeros(3)
    task = MagicMock()
    controller = RflyMissionController(
        vehicle,
        replace(MissionConfig(), return_to_origin=False),
        semantic_navigator=task,
    )
    controller.run()
    task.run.assert_called_once()
    vehicle.move_velocity_guided.assert_not_called()
    vehicle.land.assert_called_once()


def test_found_target_during_search_switches_to_approach(tmp_path):
    world = World(FakeClock(), target=False)
    original = world.move_velocity_guided

    def discover(*args, **kwargs):
        original(*args, **kwargs)
        world.target = True

    world.move_velocity_guided = discover
    nav = navigator(tmp_path, world)
    assert nav.run()
    states = [e["state"] for e in nav.events]
    assert (
        states.index("SEARCH_MOVE")
        < states.index("APPROACH")
        < states.index("SUCCEEDED")
    )


def test_task_timeout_does_not_become_success(tmp_path):
    world = World(FakeClock())

    def stall(*args, **kwargs):
        world.clock.now += 40
        raise TimeoutError("leg_timeout")

    world.move_velocity_guided = stall
    nav = navigator(tmp_path, world)
    assert nav.run() is False
    assert not nav.result["success"]


def test_exit_sign_is_not_any_generic_sign():
    assert canonical_semantic_class_name("exit signage") == "exit sign"
    assert canonical_semantic_class_name("directional sign") == "signboard"
    mapper = SemanticInstanceMapper()
    assert "exit sign" in mapper.landmark_classes


def test_new_landmark_types_do_not_merge_by_column_distance():
    mapper = SemanticInstanceMapper()
    a = SemanticObservation3D(
        np.array([0.0, 0.0, -1.0]),
        np.array([-0.1, -0.1, -1.1]),
        np.array([0.1, 0.1, -0.9]),
        np.eye(3) * 0.001,
    )
    b = replace(
        a,
        position_world=np.array([1.0, 0.0, -1.0]),
        bbox_3d_min=np.array([0.9, -0.1, -1.1]),
        bbox_3d_max=np.array([1.1, 0.1, -0.9]),
    )
    for stamp in (1, 2, 3):
        mapper.update(["signboard", "signboard"], [0.9, 0.9], [a, b], stamp)
    assert len(mapper.navigation_records()) == 2


def test_column_confirmation_does_not_count_duplicate_timestamps():
    mapper = SemanticInstanceMapper()
    obs = SemanticObservation3D(
        np.array([0.0, 0.0, -1.5]),
        np.array([-0.2, -0.2, -3.0]),
        np.array([0.2, 0.2, 0.0]),
        np.eye(3) * 0.1,
    )
    for _ in range(5):
        mapper.update(["column"], [0.9], [obs], 10)
    assert not mapper.navigation_records(min_observations=3)


def test_evaluation_cli_emits_report(tmp_path):
    import subprocess

    directory = tmp_path / "run"
    (directory / "evaluation").mkdir(parents=True)
    (directory / "evaluation" / "semantic_navigation.json").write_text(
        json.dumps(
            {
                "success": True,
                "collision": False,
                "elapsed_seconds": 20,
                "path_length_m": 5,
                "reason": "visually_confirmed",
            }
        )
    )
    report = tmp_path / "report.json"
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "tools"
            / "evaluate_semantic_navigation.py"
        ),
        str(directory),
        "--output",
        str(report),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(report.read_text())["navigation"]["success_rate"] == 1


def test_yaw_health_check_aborts_before_sending_commands():
    from config.drone_config import Config
    from vehicle.rfly_multirotor import RflyMultirotorInterface
    from unittest.mock import MagicMock

    vehicle = RflyMultirotorInterface.__new__(RflyMultirotorInterface)
    vehicle.config = Config()
    vehicle._require_offboard = lambda: None
    vehicle.get_euler = lambda: np.zeros(3)
    vehicle.send_body_velocity = MagicMock()
    check = MagicMock(side_effect=RuntimeError("perception failed"))
    with pytest.raises(RuntimeError, match="perception failed"):
        vehicle.align_yaw(1.0, timeout=2.0, health_check=check)
    check.assert_called_once()
    vehicle.send_body_velocity.assert_not_called()
