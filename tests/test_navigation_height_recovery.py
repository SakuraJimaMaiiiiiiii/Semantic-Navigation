import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np

from navigation.mission import _FlightGuard
from navigation.route import observed_body_segment_free
from planner.local_avoidance_planner import LocalPlan


class HeightRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = NS(resolution=.2, origin_ned=np.array([-2., -2., -3.]),
                           states=np.zeros((20, 20, 20), dtype=np.int8),
                           observed=np.ones((20, 20, 20), dtype=bool), timestamp=1.)
        self.config = NS(clearance_radius=.35, vertical_half_extent=.25)
        self.current = np.array([0., 0., -1.72])
        self.target = np.array([0., 0., -1.5])
        nav = NS(altitude=-1.5, config=self.config, global_map=None,
                 local_map=NS(execution_snapshot=self.snapshot, snapshot=lambda:self.snapshot),
                 local_planner=NS(config=NS(cruise_max_speed=.3)),
                 _check_maps=Mock(), _sample_position=Mock())
        self.guard = _FlightGuard(nav)
        self.guard._report = lambda plan: plan

    def test_off_plane_correction_is_vertical_and_hysteretic(self):
        first = self.guard.compute_plan(self.current, [0., 1., -1.5])
        self.assertEqual(first.status, 'height_recovery')
        np.testing.assert_allclose(first.local_target_ned, self.target)
        # Do not alternate route/recovery around the 20 cm plane boundary.
        second = self.guard.compute_plan([0., 0., -1.69], [0., 1., -1.5])
        self.assertEqual(second.status, 'height_recovery')

    def test_new_hit_or_unknown_in_actual_body_volume_prevents_recovery(self):
        for state in (1, -1):
            self.snapshot.states[10, 10, 7] = state
            self.assertFalse(observed_body_segment_free(self.snapshot, self.current, self.target, self.config))
            self.assertFalse(self.guard._height_recovery(self.current, self.snapshot).can_move)
        self.snapshot.states.fill(0)
        self.assertFalse(observed_body_segment_free(self.snapshot, self.current, [0., 0., -4.], self.config))

    def test_height_drift_after_planning_recovers_instead_of_stalling(self):
        grid = NS(altitude=-1.5, resolution=.2)
        plan = LocalPlan('route_clear', np.array([0., 1., -1.5]),
                         np.array([self.current, [0., 1., -1.5]]), .3)
        recovered = self.guard.revalidate_plan(plan, self.current, certificate=(grid, 0, None))
        self.assertEqual(recovered.status, 'height_recovery')
        np.testing.assert_allclose(recovered.local_target_ned, self.target)
