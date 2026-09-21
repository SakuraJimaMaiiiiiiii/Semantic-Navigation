"""A camera blind spot must not disconnect the vehicle from observed free space."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from config import GlobalSparseMapConfig, OccupancyGridConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from mapping.global_sparse_occupancy_map import GlobalSparseOccupancyMap
from mapping.local_occupancy_grid import LocalOccupancyGrid, FREE, UNKNOWN
from mapping.vehicle_footprint import vehicle_footprint_indices
from navigation.route import NavigationGrid
from config.exploration_config import FrontierExplorationConfig
from navigation.frontier_manager import FrontierManager


def test_global_body_blind_spot_connects_to_observed_frontiers():
    config = replace(GlobalSparseMapConfig(), vehicle_free_radius=0.35)
    world = GlobalSparseOccupancyMap(config)
    position = np.array([-0.087, 0.068, -1.499])
    footprint = set(
        map(tuple, vehicle_footprint_indices(position, np.zeros(3), 0.3, 0.35, 0.25))
    )
    world._log_odds = {
        (x, y, z): -2.0
        for x in range(-15, 16)
        for y in range(-15, 16)
        for z in (-6, -5)
        if (x, y, z) not in footprint
    }
    nav_config = SemanticNavigationConfig()
    grid = NavigationGrid.from_global(world.snapshot(), position[2], nav_config)
    assert grid.cell(position) not in grid.free
    # The actual frame integration must supply the pose footprint.
    world.integrate_camera_points(np.array([[6.0, 0.0, 0.0]]), np.eye(4), 1.0, position)
    grid = NavigationGrid.from_global(world.snapshot(), position[2], nav_config)
    manager = FrontierManager(
        replace(FrontierExplorationConfig(), viewpoint_spacing=1.2)
    )
    candidates, _, _ = manager.update(grid, position, position, 0.0, set())
    assert candidates
    assert grid.path(position, candidates[0][1]) is not None
    assert grid.path(position, [50, 0, position[2]]) is None


def test_pose_footprint_does_not_erase_global_obstacle_or_clear_unknown_corridor():
    world = GlobalSparseOccupancyMap(
        replace(GlobalSparseMapConfig(), vehicle_free_radius=0.35)
    )
    position = np.array([0.0, 0.0, -1.5])
    world._log_odds[(0, 0, -5)] = 3.5
    world._mark_vehicle_footprint(position)
    assert world._log_odds[(0, 0, -5)] == 3.5
    grid = NavigationGrid.from_global(
        world.snapshot(), -1.5, SemanticNavigationConfig()
    )
    assert grid.cell(position) not in grid.free
    assert (5, 0, -5) not in world._log_odds
    disabled = GlobalSparseOccupancyMap(GlobalSparseMapConfig())
    disabled._mark_vehicle_footprint(position)
    assert not disabled._log_odds


def test_local_update_supplies_body_evidence_but_preserves_obstacle_hits():
    local = LocalOccupancyGrid(replace(OccupancyGridConfig(), vehicle_free_radius=0.35))
    position = np.array([0.0, 0.0, -1.5])
    transform = np.eye(4)
    transform[:3, 3] = position
    transform[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    frame = SimpleNamespace(
        depth=np.full((12, 16), 2.0, dtype=np.float32),
        camera_intrinsics=np.array([[10, 0, 7.5], [0, 10, 5.5], [0, 0, 1]]),
        timestamp=1.0,
    )
    obs = SimpleNamespace(
        sensor_frame=frame,
        drone_state=SimpleNamespace(position_xyz=position),
        camera_pose_world=transform,
    )
    local.update(obs)
    snap = local.snapshot()
    center = tuple(np.floor((position - snap.origin_ned) / snap.resolution).astype(int))
    assert snap.states[center] == FREE
    grid = NavigationGrid.from_local(snap, position[2], SemanticNavigationConfig())
    assert grid.cell(position) in grid.free
    assert snap.states[0, 0, 0] == UNKNOWN
    local._log_odds[center] = 3.5
    local._observed[center] = True
    local._mark_vehicle_footprint(position)
    assert local._log_odds[center] == 3.5
