"""Control must continue while a solver is busy, and reject superseded results."""
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from config.drone_config import Config
from config.planner_config import PlannerConfig
from planner.local_avoidance_planner import LocalPlan
from planner.waypoint_planning_worker import WaypointPlanningWorker, WaypointPlanUpdate
from vehicle.rfly_multirotor import RflyMultirotorInterface


def clear_plan(current, target):
    return LocalPlan('route_clear',target.copy(),np.array([current,target]),.3)


class AsyncWaypointReferenceTests(unittest.TestCase):

    def test_initial_async_wait_does_not_consume_failure_evidence_budget(self):
        from navigation.mission import _FlightGuard
        nav = NS(local_planner=NS(config=PlannerConfig()), _event=Mock())
        guard = _FlightGuard(nav)
        guard._diagnostic_context = ('old local', 'old world', np.zeros(3), np.ones(3))
        guard.last_plan_detail = dict(status='route_clear')
        guard.capture_failure = Mock()
        wait = LocalPlan('waiting_for_plan', None, np.empty((0,3)), 0.)
        guard.notify_execution(wait)
        guard.capture_failure.assert_not_called()
        nav._event.assert_called_once_with('LOCAL_EXECUTION', status='waiting_for_plan', rejection=None)
        # A real blocked result must still preserve its diagnostic evidence.
        guard.notify_execution(LocalPlan('semantic_unknown_or_blocked', None, np.empty((0,3)), 0.))
        guard.capture_failure.assert_called_once_with('semantic_unknown_or_blocked', None)

    def test_control_reference_continues_during_blocked_replan(self):
        blocked, release = threading.Event(), threading.Event()
        calls, planner_threads, commands = [], [], []
        owner = threading.get_ident()
        def plan(current,target):
            planner_threads.append(threading.get_ident())
            calls.append(1)
            if len(calls)>1:
                blocked.set(); release.wait(3.)
            return clear_plan(current,target)
        planner = NS(config=PlannerConfig(),plan=plan)
        vehicle = RflyMultirotorInterface.__new__(RflyMultirotorInterface)
        vehicle.config=Config()
        vehicle._require_offboard=lambda:None
        vehicle.raise_if_collision=lambda **kw:None
        vehicle.get_position=lambda:np.zeros(3)
        vehicle.get_velocity=lambda:np.zeros(3)
        vehicle.get_euler=lambda:np.zeros(3)
        vehicle._flight_status=lambda:{}
        vehicle.hover=lambda:None
        vehicle.send_position=lambda point,yaw:commands.append((time.monotonic(),np.array(point),threading.get_ident()))
        try:
            vehicle.move_waypoint_guided([2.,0.,0.],local_planner=planner,control_horizon=1.5)
            self.assertTrue(blocked.is_set())
            moving=[c for c in commands if c[1][0]>1e-5]
            self.assertGreaterEqual(len(moving),10)
            self.assertLess(np.percentile(np.diff([c[0] for c in commands]),95),.15)
            self.assertTrue(all(c[2]==owner for c in commands))
            self.assertTrue(all(thread!=owner for thread in planner_threads))
            intervals=np.diff([c[0] for c in commands])
            self.control_metrics=dict(commands=len(commands),moving_commands=len(moving),
                control_interval_median_s=float(np.median(intervals)),
                control_interval_p95_s=float(np.percentile(intervals,95)),
                control_interval_max_s=float(np.max(intervals)),
                solver_was_blocked=blocked.is_set())
        finally:
            release.set()
            vehicle.stop_waypoint_planning()

    def test_superseded_or_hover_cancelled_result_is_not_published(self):
        entered, release = threading.Event(),threading.Event()
        def plan(current,target):
            entered.set();release.wait(2.);return clear_plan(current,target)
        worker=WaypointPlanningWorker(NS(config=PlannerConfig(),plan=plan))
        try:
            self.assertIsNone(worker.poll(np.zeros(3),np.array([1.,0.,0.]),np.zeros(3)))
            self.assertTrue(entered.wait(1.))
            worker.poll(np.zeros(3),np.array([2.,0.,0.]),np.zeros(3))
            worker.invalidate()
            release.set();time.sleep(.05)
            self.assertIsNone(worker._latest)
        finally:
            release.set();worker.close()

    def test_background_exception_reaches_flight_thread(self):
        error=ValueError('native solve failed')
        worker=WaypointPlanningWorker(NS(config=PlannerConfig(),plan=Mock(side_effect=error)))
        try:
            worker.poll(np.zeros(3),np.ones(3),np.zeros(3))
            end=time.monotonic()+1.
            with worker._condition:
                while worker._error is None and time.monotonic()<end:
                    worker._condition.wait(.01)
            with self.assertRaises(RuntimeError) as caught:
                worker.poll(np.zeros(3),np.ones(3),np.zeros(3))
            self.assertIs(caught.exception.__cause__,error)
        finally: worker.close()

    def test_lookahead_advances_on_frozen_curve_without_new_solve(self):
        worker=WaypointPlanningWorker(NS(config=PlannerConfig(),plan=Mock()))
        try:
            path=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
            update=WaypointPlanUpdate(LocalPlan('ego_avoid',path[1],path,.18),None,.8,path[-1])
            first=worker.tracking_plan(update,np.zeros(3))
            second=worker.tracking_plan(update,np.array([.5,0.,0.]))
            self.assertGreater(second.local_target_ned[0],first.local_target_ned[0])
            worker.planner.plan.assert_not_called()
        finally: worker.close()

    def test_latest_obstacle_blocks_a_previously_certified_segment(self):
        from navigation.route import NavigationGrid, execution_segment_free
        from config.semantic_navigation_config import SemanticNavigationConfig
        cfg=SemanticNavigationConfig(clearance_radius=.4)
        grid=NavigationGrid({(x,y) for x in range(20) for y in range(20)},set(),
                            .2,(-2.,-2.),0.,cfg)
        states=np.zeros((20,20,12),dtype=np.int8)
        observed=np.ones_like(states,dtype=bool)
        snapshot=NS(states=states,observed=observed,resolution=.2,origin_ned=np.array([-2.,-2.,-1.2]))
        start,end=np.zeros(3),np.array([1.,0.,0.])
        self.assertTrue(execution_segment_free(snapshot,start,end,cfg,grid))
        states[14,10,6]=1
        self.assertTrue(grid.segment_free(start,end))
        self.assertFalse(execution_segment_free(snapshot,start,end,cfg,grid))
        states[14,10,6]=-1
        self.assertFalse(execution_segment_free(snapshot,start,end,cfg,grid))
        observed[14,10,6]=False
        self.assertTrue(execution_segment_free(snapshot,start,end,cfg,grid))

    def test_horizon_handoff_retains_completed_certificate_until_new_plan(self):
        ready,release=threading.Event(),threading.Event()
        calls=[0]
        def solve(current,target):
            calls[0]+=1
            if calls[0]>1:
                ready.set();release.wait(2.)
            return clear_plan(current,target)
        worker=WaypointPlanningWorker(NS(config=PlannerConfig(),plan=solve))
        try:
            target=np.array([1.,0.,0.]); end=time.monotonic()+1.
            update=None
            while update is None and time.monotonic()<end:
                update=worker.poll(np.zeros(3),target,np.zeros(3));time.sleep(.01)
            self.assertIsNotNone(update)
            handoff=worker.poll(np.zeros(3),np.array([2.,0.,0.]),np.zeros(3))
            self.assertIs(handoff,update)
            self.assertTrue(ready.wait(1.))
            self.assertIs(worker.poll(np.zeros(3),np.array([2.,0.,0.]),np.zeros(3)),update)
        finally:
            release.set();worker.close()

    def test_loop_revision_change_rejects_old_published_plan(self):
        from navigation.mission import _FlightGuard
        nav=NS(local_planner=NS(config=PlannerConfig()),global_map=NS(revision=2),
               _check_maps=lambda:None)
        guard=_FlightGuard(nav)
        plan=clear_plan(np.zeros(3),np.ones(3))
        result=guard.revalidate_plan(plan,np.zeros(3),certificate=(None,1,None,{}))
        self.assertFalse(result.can_move)
        self.assertEqual(result.status,'map_revision_changed')
