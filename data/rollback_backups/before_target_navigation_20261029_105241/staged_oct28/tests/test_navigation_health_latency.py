"""Health reads must not build maps or block a valid route behind a writer."""
import math
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from config.drone_config import Config
from config.planner_config import PlannerConfig
from mapping.global_sparse_occupancy_map import GlobalSparseOccupancyMap
from mapping.local_occupancy_grid import LocalOccupancyGrid
from mapping.online_loop_navigation import OnlineLoopNavigationMap
from navigation.mission import SemanticNavigator, _FlightGuard
from planner.local_avoidance_planner import LocalPlan


class NavigationHealthLatencyTests(unittest.TestCase):
    def test_online_map_health_uses_heartbeat_and_still_detects_staleness(self):
        raw = NS(config=None, last_observation_completed_at=5.,
                 snapshot=Mock(side_effect=AssertionError('health must not build geometry')))
        world = OnlineLoopNavigationMap(raw)
        clock = [10.]
        nav = NS(clock=lambda:clock[0], _last_health_map_check=-math.inf,
                 local_map=NS(last_observation_completed_at=5.), global_map=world,
                 _map_stamps={}, config=NS(map_timeout=3.))
        SemanticNavigator._check_maps(nav)
        clock[0] += 4.
        nav.local_map.last_observation_completed_at = 6.
        with self.assertRaisesRegex(RuntimeError, 'semantic_global_map_stale'):
            SemanticNavigator._check_maps(nav)
        raw.snapshot.assert_not_called()

    def test_health_scalar_reads_do_not_wait_for_integration_lock(self):
        for grid in (LocalOccupancyGrid(), GlobalSparseOccupancyMap()):
            grid._last_observation_completed_at = 7.
            held, release, done = threading.Event(), threading.Event(), threading.Event()
            values = []
            def writer():
                with grid._lock:
                    held.set(); release.wait(2.)
            def reader():
                values.append(grid.last_observation_completed_at); done.set()
            write = threading.Thread(target=writer)
            read = threading.Thread(target=reader)
            write.start()
            try:
                self.assertTrue(held.wait(1.))
                read.start()
                self.assertTrue(done.wait(.5))
                self.assertEqual(values, [7.])
            finally:
                release.set(); write.join(2.); read.join(2.)

    def test_direct_plan_revalidation_uses_new_depth_without_map_snapshot(self):
        local = NS(forward_clearance=float('inf'),
                   snapshot=Mock(side_effect=AssertionError('duplicate geometry snapshot')))
        nav = NS(local_map=local, global_map=NS(revision=0), _check_maps=Mock(),
                 local_planner=NS(config=PlannerConfig()))
        guard = _FlightGuard(nav)
        guard._grid = NS(segment_free=lambda a,b:True)
        guard._braking_rejection = lambda *a:None
        guard._report = lambda plan, **kwargs:plan
        plan = LocalPlan('route_clear', np.array([2.,0.,0.]), np.array([[0.,0.,0.],[2.,0.,0.]]), .3)
        self.assertIs(guard.revalidate_plan(plan,np.zeros(3)),plan)
        local.forward_clearance = .2
        self.assertFalse(guard.revalidate_plan(plan,np.zeros(3)).can_move)
        local.snapshot.assert_not_called()

    def test_yaw_reference_uses_elapsed_time_under_slow_health_checks(self):
        import vehicle.rfly_multirotor as module
        clock = [0.]
        sent = [0.]
        vehicle = module.RflyMultirotorInterface.__new__(module.RflyMultirotorInterface)
        vehicle.config = Config()
        vehicle._require_offboard = lambda:None
        vehicle.get_position = lambda:np.zeros(3)
        vehicle.get_euler = lambda:np.array([0.,0.,sent[-1]])
        vehicle.raise_if_collision = lambda:None
        vehicle.send_position = lambda point,yaw:sent.append(yaw)
        vehicle.hover = lambda:None
        def health(): clock[0] += 1.
        def sleep(dt): clock[0] += max(0.,dt)
        target = math.pi/2
        with patch.object(module,'time',NS(monotonic=lambda:clock[0],sleep=sleep)):
            elapsed = vehicle.align_yaw(target, timeout=15.,health_check=health)
        self.assertLess(elapsed,15.)
        self.assertLess(abs(sent[-1]-target),vehicle.config.align_yaw_tolerance)
        self.assertTrue(all(0 <= yaw <= target for yaw in sent))
