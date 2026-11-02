"""Fresh local maps and certified curves must survive slow global work."""
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

from config.planner_config import PlannerConfig
from mapping.lidar_mapping_worker import LidarMappingWorker
from planner.ego_local_planner import EgoLocalPlanner, UniformCubicBSpline


class LocalExecutionTimingTests(unittest.TestCase):
    def test_slow_global_integration_does_not_block_local_updates(self):
        entered, release, local_done = threading.Event(), threading.Event(), threading.Event()
        global_map = NS(integrate_lidar_points=lambda *a: (entered.set(), release.wait(2.)))
        local_map = Mock()
        local_map.integrate_lidar_points.side_effect = lambda *a: local_done.set()
        with patch('mapping.lidar_mapping_worker.RflyLidarReader'):
            worker = LidarMappingWorker(None, None, local_map, global_map,
                                       NS(update_rate=100., stale_timeout=1.))
        workers = [threading.Thread(target=worker._integrate_map, args=(name, target))
                   for name, target in [('global',global_map),('local',local_map)]]
        cloud = (np.zeros(3), np.zeros((2,3)), 1.)
        worker._pending_cloud = {'local':cloud, 'global':cloud}
        for thread in workers: thread.start()
        try:
            self.assertTrue(entered.wait(1.))
            self.assertTrue(local_done.wait(1.))
            local_done.clear()
            with worker._cloud_condition:
                worker._pending_cloud['local'] = (*cloud[:2], 2.)
                worker._cloud_condition.notify_all()
            self.assertTrue(local_done.wait(1.))
            self.assertEqual(local_map.integrate_lidar_points.call_count, 2)
            self.assertFalse(release.is_set())
        finally:
            worker._stop.set(); release.set()
            with worker._cloud_condition: worker._cloud_condition.notify_all()
            for thread in workers: thread.join(2.)
        self.assertIsNone(worker._error)

    def make_planner(self):
        states = np.zeros((50,50,8), dtype=np.int8)
        states[30,25,:] = 1
        snapshot = NS(timestamp=1., origin_ned=np.array([-5.,-5.,-.5]), resolution=.2,
                      states=states, inflated_occupied=states==1, forward_clearance=float('inf'))
        planner = EgoLocalPlanner(NS(snapshot=lambda:snapshot), PlannerConfig())
        controls = np.array([[0.,0.,0.]]*3+[[.4,1.,0.],[1.5,1.,0.],[2.5,1.,0.]]+[[3.,0.,0.]]*3)
        planner._build_trajectory = Mock(return_value=UniformCubicBSpline(controls, 1.))
        return planner, snapshot

    def test_fresh_map_reuses_valid_curve_without_solving_again(self):
        planner, snapshot = self.make_planner()
        current, target = np.zeros(3), np.array([3.,0.,0.])
        first = planner.plan(current, target, now=1.)
        self.assertTrue(first.can_move)
        snapshot.timestamp = 2.
        second = planner.plan(current, target, now=1.3)
        self.assertTrue(second.can_move)
        self.assertEqual(planner._build_trajectory.call_count, 1)
        self.assertTrue(planner.last_plan_timings['trajectory_reused'])

    def test_new_obstacle_rejects_old_curve_and_solves_again(self):
        planner, snapshot = self.make_planner()
        current, target = np.zeros(3), np.array([3.,0.,0.])
        planner.plan(current, target, now=1.)
        # Block a section of the old detour, not the UAV's own footprint.
        snapshot.states[32,29,:] = 1
        snapshot.inflated_occupied = snapshot.states==1
        snapshot.timestamp = 2.
        planner._build_trajectory.return_value = None
        result = planner.plan(current, target, now=1.3)
        self.assertFalse(result.can_move)
        self.assertEqual(planner._build_trajectory.call_count, 2)

    def test_changed_waypoint_cannot_reuse_previous_goal_curve(self):
        planner, snapshot = self.make_planner()
        planner.plan(np.zeros(3), np.array([3.,0.,0.]), now=1.)
        snapshot.timestamp = 2.
        planner.plan(np.zeros(3), np.array([3.,.1,0.]), now=1.3)
        self.assertEqual(planner._build_trajectory.call_count, 2)
        self.assertFalse(planner.last_plan_timings['trajectory_reused'])

    def test_snapshot_cache_keeps_camera_updates_without_reinflating(self):
        from mapping.local_occupancy_grid import LocalOccupancyGrid
        grid = LocalOccupancyGrid()
        grid.integrate_lidar_points(np.zeros(3), np.array([[2.,0.,0.]]), 1., np.zeros(3))
        first = grid.snapshot()
        with patch.object(grid, '_inflate_unlocked', wraps=grid._inflate_unlocked) as inflate:
            grid.update_depth_observation(NS(sensor_frame=NS(depth=np.ones((20,20))*2.)))
            second = grid.snapshot()
            self.assertEqual(inflate.call_count, 0)
        self.assertIs(first.states, second.states)
        self.assertIsNotNone(second.observation)
        self.assertFalse(second.states.flags.writeable)
        grid.integrate_lidar_points(np.zeros(3), np.array([[1.,0.,0.]]), 2., np.zeros(3))
        self.assertIsNot(first.states, grid.snapshot().states)

    def test_pause_recovery_counts_time_since_completed_clear_plan(self):
        from config.drone_config import Config
        import vehicle.rfly_multirotor as module
        from planner.local_avoidance_planner import LocalPlan
        clock = [0.]
        first_clear = []
        commands = []
        calls = [0]
        def plan(current, target):
            calls[0] += 1
            if calls[0] == 1:
                clock[0] += .1
                return LocalPlan('transient_no_path',None,np.empty((0,3)),0.)
            clock[0] += 2. if calls[0] == 2 else .1
            if not first_clear: first_clear.append(clock[0])
            return LocalPlan('route_clear',np.array([2.,0.,0.]),np.array([[0.,0.,0.],[2.,0.,0.]]),.3)
        vehicle = module.RflyMultirotorInterface.__new__(module.RflyMultirotorInterface)
        vehicle.config = Config()
        vehicle._require_offboard = lambda:None
        vehicle.raise_if_collision = lambda **kw:None
        vehicle.get_position = lambda:np.zeros(3)
        vehicle.get_velocity = lambda:np.zeros(3)
        vehicle.get_euler = lambda:np.zeros(3)
        vehicle._flight_status = lambda:{}
        vehicle.send_position = lambda point,yaw:commands.append((clock[0],np.array(point)))
        vehicle.hover = lambda:None
        def sleep(dt): clock[0] += max(0.,dt)
        # Isolate recovery-time arithmetic; real thread scheduling is exercised
        # separately by test_async_waypoint_reference with a blocked solver.
        class ImmediateWorker:
            def __init__(self, planner): self.planner=planner
            def poll(self, current, target, velocity):
                return NS(plan=plan(current,target),certificate=None,duration=0.)
            def tracking_plan(self, update, current): return update.plan
        with patch.object(module, 'time', NS(monotonic=lambda:clock[0],sleep=sleep)), patch(
                'planner.waypoint_planning_worker.WaypointPlanningWorker',ImmediateWorker):
            vehicle.move_waypoint_guided([2.,0.,0.],local_planner=NS(plan=plan),control_horizon=3.5)
        forward = [stamp for stamp,point in commands if point[0] > 1e-7]
        self.assertTrue(forward)
        self.assertGreaterEqual(min(forward)-first_clear[0], vehicle.config.guided_stable_time-1e-8)

    def test_first_clear_plan_does_not_wait_for_obstacle_recovery_timer(self):
        from config.drone_config import Config
        from planner.local_avoidance_planner import LocalPlan
        import vehicle.rfly_multirotor as module
        clock = [0.]
        commands = []
        calls = [0]
        class ImmediateWorker:
            def __init__(self, planner): self.planner=planner
            def poll(self, current, target, velocity):
                calls[0] += 1
                if calls[0] == 1:
                    return None
                plan = LocalPlan('route_clear', target.copy(), np.array([current,target]), .3)
                return NS(plan=plan, certificate=None, duration=0.)
            def tracking_plan(self, update, current): return update.plan
        vehicle = module.RflyMultirotorInterface.__new__(module.RflyMultirotorInterface)
        vehicle.config=Config()
        vehicle._require_offboard=lambda:None
        vehicle.raise_if_collision=lambda **kw:None
        vehicle.get_position=lambda:np.zeros(3)
        vehicle.get_velocity=lambda:np.zeros(3)
        vehicle.get_euler=lambda:np.zeros(3)
        vehicle._flight_status=lambda:{}
        vehicle.send_position=lambda point,yaw:commands.append((clock[0], np.array(point)))
        def sleep(dt): clock[0] += max(0., dt)
        with patch.object(module, 'time', NS(monotonic=lambda:clock[0], sleep=sleep)), patch(
                'planner.waypoint_planning_worker.WaypointPlanningWorker', ImmediateWorker):
            vehicle.move_waypoint_guided([2.,0.,0.], local_planner=NS(plan=Mock()), control_horizon=.4)
        self.assertTrue(any(point[0] > 0. for _,point in commands))
        self.assertLess(min(stamp for stamp,point in commands if point[0] > 0.), vehicle.config.guided_stable_time)

    def test_new_solve_starts_from_measured_hold_velocity(self):
        planner, snapshot = self.make_planner()
        planner.plan(np.zeros(3), np.array([3.,0.,0.]), now=1.)
        planner.set_motion_state(np.zeros(3))
        snapshot.timestamp = 2.
        planner.plan(np.zeros(3), np.array([3.,.1,0.]), now=2.3)
        np.testing.assert_array_equal(planner._planning_start_velocity, np.zeros(3))

    def test_solver_time_does_not_advance_unexecuted_new_curve(self):
        import planner.ego_local_planner as module
        planner, _ = self.make_planner()
        curve = planner._build_trajectory.return_value
        clock = [10.]
        def build(*args):
            clock[0] += 2.
            return curve
        planner._build_trajectory.side_effect = build
        with patch.object(module, 'time', NS(monotonic=lambda:clock[0])):
            result = planner.plan(np.zeros(3), np.array([3.,0.,0.]), now=1.)
        self.assertTrue(result.can_move)
        self.assertEqual(planner._trajectory_started_at, 3.)
