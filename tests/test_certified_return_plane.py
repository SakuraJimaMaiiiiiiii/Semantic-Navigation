"""Centimetre Z drift must not choose a new obstacle layer at return handoff."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from config.mission_config import MissionConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from mission.rfly_mission_controller import RflyMissionController
from navigation.route import NavigationGrid


class CertifiedReturnPlaneTests(unittest.TestCase):
    def scene(self):
        plane = -1.4623817205429077
        current = np.array([0., 0., -1.439095139503479])
        home = np.array([1.2, 0., 0.])
        indices = np.array([(x,y,z) for x in range(-8,9) for y in range(-8,9)
                            for z in (-6,-5,-4)])
        odds = np.full(len(indices), -1.)
        odds[np.all(indices == [0,0,-4],axis=1)] = 3.
        world = SimpleNamespace(indices=indices, log_odds=odds, resolution=.3,
                                timestamp=1., free_threshold=-.2, occupied_threshold=1.2)
        states = np.zeros((20,20,10),dtype=np.int8)
        states[10,10,4] = 1  # [-1.2,-1.0] overlaps measured band, not task band.
        local = SimpleNamespace(states=states, observed=np.ones_like(states,dtype=bool),
                                resolution=.2, origin_ned=np.array([-2.,-2.,-2.]), timestamp=1.)
        config = SemanticNavigationConfig(clearance_radius=.4, map_timeout=.05, leg_timeout=.3)
        return plane, current, home, world, local, config

    def test_return_reuses_task_plane_and_reaches_home(self):
        plane, current, home, world, local, config = self.scene()
        actual = NavigationGrid.from_global(world, current[2], config, local, current)
        planned = NavigationGrid.from_global(world, plane, config, local, current)
        self.assertNotIn(actual.cell(current), actual.free)
        self.assertIn(planned.cell(current), planned.free)
        self.assertIsNotNone(planned.path(current, np.array([*home[:2], plane])))
        position = current.copy()
        vehicle = SimpleNamespace(get_position=lambda:position.copy(), get_velocity=lambda:np.zeros(3),
                                  hover=Mock(), align_yaw=Mock(), wait=Mock(),
                                  config=SimpleNamespace(guided_z_tolerance=.15))
        def move(goal, **kwargs):
            position[:] = goal
        vehicle.move_waypoint_guided = Mock(side_effect=move)
        nav = SimpleNamespace(altitude=plane, config=config, _check_maps=Mock(),
            local_map=SimpleNamespace(snapshot=lambda:local),
            global_map=SimpleNamespace(snapshot=lambda:world,revision=0))
        planner = SimpleNamespace(set_navigation_plane=Mock())
        controller = RflyMissionController(vehicle, MissionConfig(), return_safety_planner=planner,
                                          semantic_navigator=nav)
        controller._return_origin_ned = home
        with patch('navigation.mission._FlightGuard') as guard:
            controller._return_on_certified_path()
        self.assertEqual(guard.call_args.args[0].altitude, plane)
        planner.set_navigation_plane.assert_called_once_with(plane, config, nav.global_map)
        self.assertTrue(vehicle.move_waypoint_guided.called)
        for call in vehicle.move_waypoint_guided.call_args_list:
            self.assertEqual(call.args[0][2], plane)
        np.testing.assert_allclose(position[:2], home[:2])

    def test_obstacle_inside_task_band_still_blocks(self):
        plane, current, _, world, local, config = self.scene()
        local.states[10,10,3] = 1
        grid = NavigationGrid.from_global(world, plane, config, local, current)
        self.assertNotIn(grid.cell(current), grid.free)

    def test_global_projection_preserves_free_and_occupied_evidence(self):
        plane, _, _, world, _, config = self.scene()
        grid = NavigationGrid.from_global(world, plane, config)
        from planner.vertical_projection import height_band_indices
        zmin, zmax = height_band_indices(plane, config.vertical_half_extent, world.resolution)
        counts, occupied = {}, set()
        for index, odds in zip(world.indices, world.log_odds):
            if zmin <= index[2] <= zmax:
                cell = tuple(map(int, index[:2]))
                if odds >= world.occupied_threshold:
                    occupied.add(cell)
                elif odds <= world.free_threshold:
                    counts[cell] = counts.get(cell, 0) + 1
        free = {cell for cell, count in counts.items() if count == zmax-zmin+1}
        expected = NavigationGrid(free, occupied, world.resolution, (0, 0), plane, config)
        self.assertEqual(grid.occupied, expected.occupied)
        self.assertEqual(grid.free, expected.free)
