"""Runtime regressions: Windows NumPy keys, guarded fallback and stalled motion."""

import json
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from config.exploration_config import FrontierExplorationConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.scene_memory import write_json
from navigation.dfs_explorer import DFSExplorer
from navigation.mission import SemanticNavigator, _FlightGuard
from planner.local_avoidance_planner import LocalPlan


def test_numpy_failure_trace_roundtrips_and_nonfinite_preserves_last_checkpoint(
    tmp_path,
):
    path = tmp_path / "graph.json"
    value = {
        "exploration": {
            "visit_trace": [("blocked", [np.int32(-2), np.int64(3)])],
            "elapsed": np.float32(2.5),
            "done": np.bool_(False),
        }
    }
    write_json(path, value)
    assert json.loads(path.read_text())["exploration"]["visit_trace"] == [
        ["blocked", [-2, 3]]
    ]
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"invalid": np.float32(np.nan)})
    assert path.read_bytes() == previous


def test_guard_uses_direct_route_and_rejects_off_plane_ego_when_blocked():
    config = SemanticNavigationConfig()
    snap = SimpleNamespace(
        timestamp=1,
        resolution=0.5,
        origin_ned=np.array([-5.0, -5.0, -3.0]),
        states=np.zeros((20, 20, 6), dtype=np.int8),
    )
    current, target = np.array([0.0, 0.0, -1.5]), np.array([2.0, 0.0, -1.5])
    plan = LocalPlan(
        "ego_clear", target, np.array([current, [1.0, 0.0, -0.5], target]), 0.6, 10.0
    )
    inner = SimpleNamespace(
        config=SimpleNamespace(avoidance_max_speed=0.35), plan=lambda *args: plan
    )
    events = []
    nav = SimpleNamespace(
        local_planner=inner,
        local_map=SimpleNamespace(snapshot=lambda: snap),
        config=config,
        altitude=-1.5,
        _check_health=lambda: None,
        _sample_position=lambda *args: None,
        _event=lambda state, **kw: events.append(kw),
    )
    guard = _FlightGuard(nav)
    actual = guard.plan(current, target)
    assert actual.status == "route_clear" and actual.can_move
    snap.states[12, :, :] = -1
    snap.timestamp += 1
    actual = guard.plan(current, target)
    assert actual.status == "semantic_unknown_or_blocked" and not actual.can_move
    plan = replace(
        plan, status="emergency_stop", local_target_ned=None, speed_limit=0.0
    )
    assert not guard.plan(current, target).can_move
    assert events[-1]["status"] == "emergency_stop"
    plan = replace(plan, status="ego_clear", local_target_ned=target, speed_limit=0.6)
    assert not guard.plan(current, target).can_move


def test_stall_timeout_runs_inside_planner_health_checks(monkeypatch):
    monkeypatch.setattr(SemanticNavigator, "_check_health", lambda self: None)
    explorer = DFSExplorer.__new__(DFSExplorer)
    explorer.exploration_config = FrontierExplorationConfig()
    position = np.array([0.0, 0.0, -1.5])
    now = [0.0]
    hover = []
    explorer.clock = lambda: now[0]
    explorer.vehicle = SimpleNamespace(
        get_position=lambda: position.copy(), hover=lambda: hover.append(True)
    )
    explorer._motion_progress = position.copy(), 0.0
    now[0] = 7.0
    explorer._check_health()
    position[0] += 0.2
    explorer._check_health()
    now[0] = 14.0
    explorer._check_health()
    now[0] = 16.0
    with pytest.raises(TimeoutError, match="no_motion_progress"):
        explorer._check_health()
    assert hover


def test_failed_frontier_recovers_after_cooldown_and_short_frontiers_are_not_lost():
    from navigation.route import NavigationGrid

    config = replace(SemanticNavigationConfig(), clearance_radius=0.1)
    grid = NavigationGrid(
        {(x, y) for x in range(-3, 4) for y in range(-3, 4)},
        set(),
        0.5,
        (0, 0),
        -1.5,
        config,
    )
    origin = np.array([0.0, 0.0, -1.5])
    explorer = DFSExplorer.__new__(DFSExplorer)
    explorer.config = config
    explorer.exploration_config = FrontierExplorationConfig()
    explorer.vehicle = SimpleNamespace(get_euler=lambda: np.zeros(3))
    explorer.failed_frontiers = {}
    from navigation.frontier_manager import FrontierManager

    explorer.frontiers = FrontierManager(explorer.exploration_config)
    now = [0.0]
    explorer.clock = lambda: now[0]
    choices = explorer._choices(grid, origin, origin)[0]
    assert choices and all(type(v) is int for key, _ in choices for v in key)
    explorer.failed_frontiers = {key: (1, 30.0) for key, _ in choices}
    assert all(
        key not in explorer.failed_frontiers
        for key, _ in explorer._choices(grid, origin, origin)[0]
    )
    now[0] = 31.0
    assert explorer._choices(grid, origin, origin)[0]


def test_completion_requires_stable_empty_frontiers_and_low_recent_gain():
    explorer = DFSExplorer.__new__(DFSExplorer)
    explorer.exploration_config = replace(
        FrontierExplorationConfig(),
        frontier_empty_confirmations=3,
        frontier_empty_min_duration=10.0,
        gain_window_viewpoints=2,
    )
    now = [0.0]
    explorer.clock = lambda: now[0]
    explorer._empty_frontier_since = None
    explorer._empty_frontier_confirmations = 0
    explorer._coverage_samples = [10.0, 10.4, 10.8]

    assert explorer._confirm_empty_frontiers() == (False, 0.0)
    now[0] = 5.0
    assert explorer._confirm_empty_frontiers() == (False, 5.0)
    now[0] = 10.0
    assert explorer._confirm_empty_frontiers() == (True, 10.0)
    assert explorer._recent_map_gain() == pytest.approx(0.8)


def test_empty_frontier_evidence_resets_when_a_candidate_reappears():
    explorer = DFSExplorer.__new__(DFSExplorer)
    explorer._empty_frontier_since = 1.0
    explorer._empty_frontier_confirmations = 4
    explorer._reset_empty_frontier_evidence()
    assert explorer._empty_frontier_since is None
    assert explorer._empty_frontier_confirmations == 0


def test_global_free_evidence_fills_only_unobserved_local_voxels():
    from navigation.route import NavigationGrid

    states = np.full((20, 20, 6), -1, dtype=np.int8)
    observed = np.zeros_like(states, dtype=bool)
    snap = SimpleNamespace(
        resolution=0.5,
        origin_ned=np.array([-5.0, -5.0, -3.0]),
        states=states,
        observed=observed,
    )
    indices = np.array(
        [
            (x, y, z)
            for x in range(-10, 10)
            for y in range(-10, 10)
            for z in range(-6, 0)
        ]
    )
    world = SimpleNamespace(
        resolution=0.5,
        indices=indices,
        log_odds=np.full(len(indices), -2.0),
        free_threshold=-0.2,
    )
    cfg = SemanticNavigationConfig()
    start, goal = [0.0, 0.0, -1.5], [2.0, 0.0, -1.5]
    assert NavigationGrid.from_local(snap, -1.5, cfg).path(start, goal) is None
    assert (
        NavigationGrid.from_local(snap, -1.5, cfg, world).path(start, goal) is not None
    )
    # A whole strip with actual contradictory/uncertain observations must not
    # be filled from historical global data, even though global says free.
    observed[12, :, :] = True
    assert NavigationGrid.from_local(snap, -1.5, cfg, world).path(start, goal) is None
    states[12, :, :] = 1
    assert NavigationGrid.from_local(snap, -1.5, cfg, world).path(start, goal) is None


def test_real_explorer_blocked_numpy_key_still_writes_report(tmp_path, monkeypatch):
    from test_dfs_workflow import make_explorer

    explorer, _, _ = make_explorer(tmp_path, monkeypatch)
    key = (np.int32(2), np.int32(-1))

    def choices(grid, current, origin, spacing, radius, attempted, **kw):
        return (
            ([] if key in attempted else [(key, np.array([3.0, -1.0, -1.5]))]),
            False,
            False,
        )

    monkeypatch.setattr(
        explorer,
        "_choices",
        lambda grid, current, origin: choices(
            grid, current, origin, 3.0, 100.0, set(explorer.failed_frontiers)
        ),
    )
    explorer._travel = lambda goal: False
    assert not explorer.run()
    result = json.loads(explorer.output_path.read_text())
    assert result["reason"] == "frontiers_blocked"
    assert ["blocked", "frontier_0000"] in result["visit_trace"]
    explorer.memory.checkpoint(tmp_path)
