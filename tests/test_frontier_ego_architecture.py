"""Exploration-layer geometry and EGO-only fixed-plane execution contracts."""

from dataclasses import replace
import numpy as np

from config.exploration_config import FrontierExplorationConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.frontier_manager import FrontierManager, visible_unknown
from navigation.route import NavigationGrid
from planner import EgoLocalPlanner, NativeEgoLocalPlanner


def test_viewpoint_gain_does_not_see_through_known_wall():
    cfg = replace(SemanticNavigationConfig(), clearance_radius=0.1)
    free = {(x, y) for x in range(-4, 1) for y in range(-10, 11)}
    wall = {(1, y) for y in range(-10, 11)}
    grid = NavigationGrid(free, wall, 0.5, (0, 0), -1.5, cfg)
    seen = visible_unknown(grid, (0, 0), 0.0, 3.0, 90.0)
    assert not any(x > 1 for x, y in seen)


def test_connected_regions_update_when_unknown_becomes_observed():
    cfg = replace(SemanticNavigationConfig(), clearance_radius=0.1)
    free = {(x, y) for x in range(-7, 8) for y in range(-7, 8)}
    grid = NavigationGrid(free, set(), 0.5, (0, 0), -1.5, cfg)
    manager = FrontierManager(FrontierExplorationConfig())
    origin = np.array([0.0, 0.0, -1.5])
    choices, truncated, _ = manager.update(grid, origin, origin, 0.0, set())
    assert choices and manager.regions and not truncated
    candidate = manager.candidates[choices[0][0]]
    manager.observe_view(candidate)
    newer, _, _ = manager.update(grid, origin, origin, 0.0, set())
    assert all(k != candidate.key for k, _ in newer)
    # Marking a view reached must not delete its still-unknown region.
    assert manager.regions
    wall = {
        (x, y) for x in range(-8, 9) for y in range(-8, 9) if abs(x) == 8 or abs(y) == 8
    }
    enclosed = NavigationGrid(free, wall, 0.5, (0, 0), -1.5, cfg)
    choices, _, _ = manager.update(enclosed, origin, origin, 0.0, set())
    assert not choices and not manager.regions


def test_frontier_tour_uses_only_reachable_regions():
    cfg = replace(SemanticNavigationConfig(), clearance_radius=0.1)
    free = {(x, y) for x in range(-7, 8) for y in range(-7, 8)} | {(50, 50), (50, 51)}
    grid = NavigationGrid(free, set(), 0.5, (0, 0), -1.5, cfg)
    origin = np.array([0.0, 0.0, -1.5])
    manager = FrontierManager(FrontierExplorationConfig())
    choices, _, _ = manager.update(grid, origin, origin, 0.0, set())
    assert manager.visit_order
    assert all(grid.path(origin, goal) is not None for _, goal in choices)


def test_python_ego_generates_fixed_height_splines_for_successive_targets():
    from test_ego_local_planner import _Grid, _snapshot, _config

    source = _Grid(_snapshot())
    planner = EgoLocalPlanner(source, _config())
    planner.set_navigation_plane(0.0, SemanticNavigationConfig())
    current = np.zeros(3)
    for i, target in enumerate(([1.5, 0.0, 0.0], [1.5, 1.5, 0.0], [0.0, 1.5, 0.0])):
        source.value = replace(source.value, timestamp=float(i + 1))
        plan = planner.plan(current, np.array(target), now=10.0 + i)
        assert plan.can_move and plan.status in {"ego_clear", "ego_avoid"}
        assert np.allclose(plan.path_ned[:, 2], 0.0)
        assert np.linalg.norm(plan.local_target_ned[:2] - current[:2]) > 0.05
        current = np.array(target)


def test_ego_will_not_climb_over_a_wall_in_fixed_height_mode():
    from test_ego_local_planner import _Grid, _snapshot, _config

    snap = _snapshot()
    # A thin wall only at flight height. A 3-D planner could climb over it;
    # fixed-plane exploration must stop rather than issue an ascent.
    snap.states[19, :, 5:10] = 1
    snap.inflated_occupied[18:21, :, 5:10] = True
    planner = EgoLocalPlanner(_Grid(snap), _config())
    planner.set_navigation_plane(0.0, SemanticNavigationConfig())
    result = planner.plan(np.zeros(3), np.array([2.0, 0.0, 0.0]), now=10.0)
    if result.can_move:
        assert np.allclose(result.path_ned[:, 2], 0.0)
        assert np.max(result.path_ned[:, 0]) < 0.5


def test_native_ego_plane_projects_then_revalidates_trajectory():
    from test_native_ego_local_planner import _Grid, _snapshot, _config, _Backend

    planner = NativeEgoLocalPlanner(
        _Grid(_snapshot()), _config(), native_backend=_Backend()
    )
    planner.set_navigation_plane(0.0, SemanticNavigationConfig())
    result = planner.plan(np.zeros(3), np.array([1.5, 0.0, 0.5]), now=10.0)
    assert result.can_move
    assert np.allclose(result.path_ned[:, 2], 0.0)
