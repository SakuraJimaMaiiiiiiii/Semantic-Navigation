from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pytest

from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.route import NavigationGrid
from planner.local_avoidance_planner import LocalPlan
from mission.rfly_mission_controller import RflyMissionController
from exploration_wildos.vehicle_motion import (VehiclePlanningMap, VehiclePlanVeto,
    VehicleMotionProxy, VehicleAwareMissionController)


def fixtures():
    config=SemanticNavigationConfig(clearance_radius=.4)
    objects=[dict(id='car',bbox_3d={'min':[-.6,-.6,-.7],'max':[.6,.6,0]})]
    raw=SimpleNamespace(timestamp=1.,resolution=.3,occupied_threshold=.7,free_threshold=-.2,
        indices=np.array([(x,y,z) for x in range(-15,16) for y in range(-15,16) for z in (-11,-10)]),
        log_odds=np.full(31*31*2,-1.))
    source=SimpleNamespace(snapshot=lambda:raw,revision=0,last_observation_completed_at=1.)
    nav=SimpleNamespace(altitude=-3.,config=config,global_map=source)
    return raw,source,nav,objects


def test_return_map_routes_around_low_vehicle_without_mutating_sensor_map():
    raw,source,nav,objects=fixtures()
    before=raw.indices.copy(),raw.log_odds.copy()
    view=VehiclePlanningMap(source,lambda:objects,nav)
    snapshot=view.snapshot()
    assert snapshot is view.snapshot()
    np.testing.assert_array_equal(raw.indices,before[0]);np.testing.assert_array_equal(raw.log_odds,before[1])
    grid=NavigationGrid.from_global(snapshot,-3.,nav.config)
    start,end=np.array([-3.,0.,-3.]),np.array([3.,0.,-3.])
    assert not grid.segment_free(start,end)
    path=grid.path(start,end)
    assert path is not None and len(path)>2
    assert all(grid.segment_free(a,b) for a,b in zip(path,path[1:]))
    revision=view.revision
    objects[0]['bbox_3d']['max'][0]=.9
    assert view.revision!=revision and view.snapshot() is not snapshot


def test_execution_veto_rejects_above_car_after_existing_checks():
    _,_,nav,objects=fixtures()
    plan=LocalPlan('ego_avoid',np.array([3.,0.,-8.]),np.array([[-3.,0.,-8.],[3.,0.,-8.]]),.5)
    inner=SimpleNamespace(revalidate_plan=Mock(return_value=plan))
    veto=VehiclePlanVeto(inner,lambda:objects,nav.config)
    assert not veto.revalidate_plan(plan,np.array([-3.,0.,-8.])).can_move
    inner.revalidate_plan.assert_called_once()
    assert inner._execution_rejection['vehicle']=='car'


@pytest.mark.parametrize('fail',[False,True])
def test_return_controller_installs_constraints_and_restores_original_context(fail):
    _,source,nav,objects=fixtures()
    vehicle=Mock();nav.vehicle=vehicle
    old=RflyMissionController(vehicle,Mock(),semantic_navigator=nav)
    controller=VehicleAwareMissionController.wrap(old,lambda:objects)
    def returning(self,**kwargs):
        assert isinstance(self.vehicle,VehicleMotionProxy)
        assert isinstance(nav.global_map,VehiclePlanningMap)
        assert nav.vehicle is self.vehicle
        self.vehicle.move_waypoint_guided([1,2,-3],local_planner=Mock())
        assert isinstance(vehicle.move_waypoint_guided.call_args.kwargs['local_planner'],VehiclePlanVeto)
        if fail:raise RuntimeError('original return fault')
    with patch.object(RflyMissionController,'return_and_land',returning):
        if fail:
            with pytest.raises(RuntimeError,match='original return fault'):controller.return_and_land()
        else:controller.return_and_land()
    assert nav.global_map is source and nav.vehicle is vehicle and controller.vehicle is vehicle


def test_return_horizons_reuse_guard_identity_without_bypassing_checks():
    _,_,nav,objects=fixtures()
    vehicle=Mock();proxy=VehicleMotionProxy(vehicle,lambda:objects,nav.config)
    planner=SimpleNamespace(revalidate_plan=Mock(return_value=LocalPlan(
        'route_clear',np.array([3.,0.,-3.]),np.empty((0,3)),.3)))
    for _ in range(5):
        proxy.move_waypoint_guided([3,0,-3],local_planner=planner,control_horizon=1.)
    guards=[call.kwargs['local_planner'] for call in vehicle.move_waypoint_guided.call_args_list]
    assert len({id(g) for g in guards})==1
    assert not guards[0].revalidate_plan(None,np.array([-3.,0.,-3.])).can_move
    planner.revalidate_plan.assert_called_once()
    another=Mock()
    proxy.move_waypoint_guided([3,0,-3],local_planner=another)
    assert vehicle.move_waypoint_guided.call_args.kwargs['local_planner'] is not guards[0]


@pytest.mark.parametrize('reuse_guard', [False, True])
def test_real_control_loop_recovers_clearance_across_return_horizons(reuse_guard):
    """Reproduce a .72 s solve within 1 s horizons after an obstacle hold.

    Use the real position controller, a deterministic clock and a delayed
    worker double. The legacy rewrap case must remain held; reuse must resume
    commands only after the ordinary continuous-clearance gate has passed.
    No flight SDK or vehicle connection is created.
    """
    from config.drone_config import Config
    from vehicle.rfly_multirotor import RflyMultirotorInterface

    now = [10.]
    workers, commands = [], []
    current = np.array([3.045, 31.329, -1.496])
    target = np.array([1.817, 29.279, -1.494])
    plan = LocalPlan('route_clear', target, np.array([current, target]), .3)
    planner = SimpleNamespace(revalidate_plan=Mock(side_effect=lambda p, *a, **k: p))

    class DelayedWorker:
        def __init__(self, guard):
            self.planner = guard
            self.ready_at = now[0] + .72
            self.closed = False
            workers.append(self)

        def poll(self, *args):
            return (SimpleNamespace(plan=plan, certificate=None, duration=.72)
                    if now[0] >= self.ready_at else None)

        def tracking_plan(self, update, position):
            return update.plan

        def close(self):
            self.closed = True

    vehicle = RflyMultirotorInterface.__new__(RflyMultirotorInterface)
    vehicle.config = Config()
    vehicle._require_offboard = lambda: None
    vehicle.raise_if_collision = lambda **kw: None
    vehicle.get_position = lambda: current.copy()
    vehicle.get_velocity = lambda: np.zeros(3)
    vehicle.get_euler = lambda: np.zeros(3)
    vehicle._flight_status = lambda: {}
    vehicle.hover = Mock()
    vehicle._motion_progress_display = Mock()
    vehicle._waypoint_hold_target = current.copy()
    vehicle._waypoint_hold_reason = 'emergency_stop'
    vehicle.send_position = lambda point, yaw: commands.append((now[0], point.copy()))
    proxy = VehicleMotionProxy(vehicle, lambda: [], SemanticNavigationConfig())
    clock = SimpleNamespace(monotonic=lambda: now[0],
                            sleep=lambda dt: now.__setitem__(0, now[0] + dt))
    with patch('vehicle.rfly_multirotor.time', clock), patch(
            'planner.waypoint_planning_worker.WaypointPlanningWorker', DelayedWorker):
        for _ in range(5):
            if reuse_guard:
                proxy.move_waypoint_guided(target, local_planner=planner, control_horizon=1.)
            else:
                # The previous wrapper behavior, retained only as a test control.
                vehicle.move_waypoint_guided(target, local_planner=VehiclePlanVeto(
                    planner, lambda: [], SemanticNavigationConfig()), control_horizon=1.)

    moving = [(t, p) for t, p in commands if np.linalg.norm(p-current) > 1e-6]
    assert len(workers) == (1 if reuse_guard else 5)
    if reuse_guard:
        assert moving
        assert moving[0][0] >= 10. + .72 + vehicle.config.guided_stable_time
        assert vehicle._waypoint_hold_target is None
        assert not workers[0].closed
    else:
        assert not moving
        assert vehicle.waypoint_control_state['status'] == 'holding_for_stable_clearance'
        assert all(worker.closed for worker in workers[:-1])
    assert planner.revalidate_plan.call_count > 0
