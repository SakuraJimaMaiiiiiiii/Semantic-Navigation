from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from navigation.frontier_explorer import FrontierExplorer
from navigation.mission import SemanticNavigator


class ObservationWatchdogTests(unittest.TestCase):


    def explorer(self, moving=False):
        nav = object.__new__(FrontierExplorer)
        nav._motion_progress = (np.zeros(3), 0.) if moving else None
        nav.clock = Mock(return_value=20.)
        nav.deadline = 100.
        nav.config = SimpleNamespace(map_timeout=5.)
        nav.exploration_config = SimpleNamespace(
            information_radius=3., camera_fov_degrees=90.,
            recovery_observation_wait=.3, progress_distance=.15, progress_timeout=8.)
        nav.vehicle = Mock()
        nav.vehicle.get_position.return_value = np.zeros(3)
        nav.vehicle.get_euler.return_value = np.zeros(3)
        nav._event = Mock()
        return nav

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
