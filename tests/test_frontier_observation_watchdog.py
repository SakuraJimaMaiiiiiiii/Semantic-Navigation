from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from navigation.frontier_explorer import FrontierExplorer
from navigation.mission import SemanticNavigator


class ObservationWatchdogTests(unittest.TestCase):
    def test_planning_wait_has_bounded_separate_credit(self):
        nav = self.explorer(moving=True)
        nav.clock.return_value = 7.
        nav.vehicle.waypoint_control_state = dict(status='holding_for_stable_clearance',
                                                plan_status='waiting_for_plan', speed_limit=0.)
        with patch.object(SemanticNavigator, '_check_health'):
            nav._check_health()
            nav.clock.return_value = 9.
            nav.vehicle.waypoint_control_state = dict(status='route_clear', target_ned=[2.,0.,0.])
            nav._check_health()
            nav.clock.return_value = 16.
            with self.assertRaisesRegex(TimeoutError, 'exploration_no_motion_progress'):
                nav._check_health()


    def explorer(self, moving=False):
        nav = object.__new__(FrontierExplorer)
        nav._motion_progress = (np.zeros(3), 0.) if moving else None
        nav.clock = Mock(return_value=20.)
        nav.deadline = 100.
        nav.config = SimpleNamespace(map_timeout=5., perception_timeout=5., arrival_tolerance=.4)
        nav.perception = Mock()
        nav.perception.semantic_snapshot.return_value = None
        nav.memory = Mock()
        nav.current_place = None
        nav._latencies = {}
        nav.exploration_config = SimpleNamespace(
            information_radius=3., camera_fov_degrees=90.,
            recovery_observation_wait=.3, progress_distance=.15, progress_timeout=8.)
        nav.vehicle = Mock()
        nav.vehicle.get_position.return_value = np.zeros(3)
        nav.vehicle.get_euler.return_value = np.zeros(3)
        nav._event = Mock()
        return nav

    def test_stale_semantics_does_not_block_certified_motion_health(self):
        nav = self.explorer()
        nav.perception.semantic_snapshot.return_value = SimpleNamespace(completed_at=0.)
        nav._check_maps = Mock()
        nav._frame = Mock()
        nav._check_health()
        nav._check_maps.assert_called_once()
        nav._frame.assert_not_called()
        nav.vehicle.hover.assert_not_called()
        nav._event.assert_called_once_with('SEMANTIC_DEFERRED', reason='recognition_not_fresh')

    def test_stale_map_still_blocks_when_semantics_is_delayed(self):
        nav = self.explorer()
        nav.perception.semantic_snapshot.return_value = None
        nav._check_maps = Mock(side_effect=RuntimeError('semantic_local_map_stale'))
        with self.assertRaisesRegex(RuntimeError, 'semantic_local_map_stale'):
            nav._check_health()

    def recover_map(self, nav):
        with patch.object(SemanticNavigator, '_check_maps',
                          side_effect=[RuntimeError('semantic_local_map_stale'), None]):
            nav._check_maps()

    def test_map_recovery_does_not_arm_stationary_watchdog(self):
        nav = self.explorer()
        self.recover_map(nav)
        self.assertIsNone(nav._motion_progress)
        with patch.object(SemanticNavigator, '_check_health'):
            nav._check_health()

    def test_map_recovery_resets_active_watchdog(self):
        nav = self.explorer(moving=True)
        self.recover_map(nav)
        self.assertEqual(nav._motion_progress[1], 20.)

    def test_return_map_recovery_does_not_use_expired_exploration_deadline(self):
        nav = self.explorer()
        nav.phase = 'returning'
        nav.deadline = 10.
        self.recover_map(nav)
        nav.vehicle.hover.assert_called_once()
        nav._event.assert_any_call('MAP_UPDATE_RESUMED')

    def test_stale_map_still_fails_after_return_recovery_timeout(self):
        nav = self.explorer()
        nav.phase = 'returning'
        nav.deadline = 10.
        clock = [20.]
        nav.clock = lambda: clock[0]
        nav.vehicle.wait.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        with patch.object(SemanticNavigator, '_check_maps', side_effect=RuntimeError('semantic_global_map_stale')):
            with self.assertRaisesRegex(RuntimeError, 'semantic_global_map_stale'):
                nav._check_maps()
        self.assertGreaterEqual(clock[0], 25.)

    def test_stationary_observation_survives_map_recovery_and_restores_on_error(self):
        for moving in (False, True):
            for fail in (False, True):
                with self.subTest(moving=moving, fail=fail):
                    nav = self.explorer(moving)
                    nav.recovery_yaws = {}
                    grid = Mock(free={(0,0)})
                    grid.cell.return_value = (0,0)
                    def align(yaw):
                        self.assertIsNone(nav._motion_progress)
                        self.recover_map(nav)
                        self.assertIsNone(nav._motion_progress)
                        if fail:
                            raise TimeoutError('yaw_timeout')
                    nav._align = align
                    for name in ('_wait', '_frame', '_record_coverage', '_checkpoint'):
                        setattr(nav, name, Mock())
                    nav._grid = Mock(return_value=grid)
                    with patch('navigation.frontier_explorer.visible_unknown', return_value={(1,1)}):
                        if fail:
                            with self.assertRaisesRegex(TimeoutError, 'yaw_timeout'):
                                nav._recovery_observation(grid,(0,0))
                        else:
                            self.assertTrue(nav._recovery_observation(grid,(0,0)))
                    if moving:
                        self.assertEqual(nav._motion_progress[1],20.)
                    else:
                        self.assertIsNone(nav._motion_progress)

    def test_actual_translation_stall_is_still_detected(self):
        nav = self.explorer(moving=True)
        with patch.object(SemanticNavigator, '_check_health'):
            with self.assertRaisesRegex(TimeoutError,'exploration_no_motion_progress'):
                nav._check_health()
        nav.vehicle.hover.assert_called_once()

    def test_slow_forward_progress_on_clear_route_is_not_marked_blocked(self):
        nav = self.explorer(moving=True)
        nav.vehicle.get_position.return_value = np.array([.126, 0., 0.])
        nav.vehicle.waypoint_control_state = dict(status='route_clear', target_ned=[2.,0.,0.],
                                                  velocity_ned=[.072,0.,0.])
        with patch.object(SemanticNavigator, '_check_health'):
            nav._check_health()
        self.assertEqual(nav._motion_progress[1], 20.)
        nav.vehicle.hover.assert_not_called()

    def test_slow_measured_displacement_counts_despite_low_instantaneous_speed(self):
        nav = self.explorer(moving=True)
        nav.vehicle.get_position.return_value = np.array([.071, 0., 0.])
        nav.vehicle.waypoint_control_state = dict(status='route_clear', target_ned=[2.,0.,0.],
                                                  velocity_ned=[.012,0.,0.])
        with patch.object(SemanticNavigator, '_check_health'):
            nav._check_health()
        self.assertEqual(nav._motion_progress[1], 20.)

    def test_movement_away_from_goal_does_not_clear_watchdog(self):
        nav = self.explorer(moving=True)
        nav.vehicle.get_position.return_value = np.array([-.071, 0., 0.])
        nav.vehicle.waypoint_control_state = dict(status='route_clear', target_ned=[2.,0.,0.],
                                                  velocity_ned=[-.012,0.,0.])
        with patch.object(SemanticNavigator, '_check_health'):
            with self.assertRaisesRegex(TimeoutError, 'exploration_no_motion_progress'):
                nav._check_health()

    def test_reached_intermediate_waypoint_gets_one_chance_to_advance(self):
        nav = self.explorer(moving=True)
        nav.vehicle.get_position.return_value = np.array([.14, 0., 0.])
        nav.vehicle.waypoint_control_state = dict(status='route_clear', target_ned=[.14,0.,0.],
                                                  velocity_ned=[0.,0.,0.])
        with patch.object(SemanticNavigator, '_check_health'):
            nav._check_health()
            nav.clock.return_value = 29.
            with self.assertRaisesRegex(TimeoutError, 'exploration_no_motion_progress'):
                nav._check_health()

    def test_slow_motion_toward_obstacle_is_not_treated_as_clear_progress(self):
        nav = self.explorer(moving=True)
        nav.vehicle.get_position.return_value = np.array([.126, 0., 0.])
        nav.vehicle.waypoint_control_state = dict(status='emergency_stop', target_ned=[2.,0.,0.],
                                                  velocity_ned=[.072,0.,0.])
        with patch.object(SemanticNavigator, '_check_health'):
            with self.assertRaisesRegex(TimeoutError, 'exploration_no_motion_progress'):
                nav._check_health()
