import importlib.util
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
import pytest

missing={n:ModuleType(n) for n in ('PX4MavCtrlV4','ReqCopterSim','UE4CtrlAPI')
         if n not in sys.modules and importlib.util.find_spec(n) is None}
sys.modules.update(missing)
try:
    from exploration_wildos.target_navigation import GraphTargetNavigator
    from exploration_wildos.standalone_navigation import IndependentGraphSession
finally:
    for n in missing:sys.modules.pop(n,None)
from exploration_wildos.height_recovery import HeightRecoveryGuard, vertical_clear, vertical_rejection
from exploration_wildos.target_session import GraphTargetSessionMixin, TargetUnavailableError
from config.semantic_navigation_config import SemanticNavigationConfig


def fixture():
    local=NS(states=np.zeros((20,20,30),dtype=np.int8),observed=np.ones((20,20,30),dtype=bool),
             resolution=.1,origin_ned=np.array([-1.,-1.,-2.5]))
    cfg=SemanticNavigationConfig(clearance_radius=.1,vertical_half_extent=.1)
    current=np.array([0.,0.,-1.778]);target=np.array([0.,0.,-1.491])
    return local,cfg,current,target


@pytest.mark.parametrize('state',[1,-1])
def test_vertical_recovery_rejects_obstacle_and_unknown(state):
    local,cfg,current,target=fixture()
    assert vertical_clear(local,current,target,cfg,[])
    local.states[10,10,9]=state
    assert not vertical_clear(local,current,target,cfg,[])
    assert ('占用体素' if state == 1 else '完整观测') in vertical_rejection(local,current,target,cfg,[])


def test_recovery_guard_rechecks_new_obstacle_vehicle_and_distance():
    local,cfg,current,target=fixture()
    objects=[]
    nav=NS(config=cfg,local_map=NS(execution_snapshot=local),objects_provider=lambda:objects,
           local_planner=NS(config=NS(cruise_max_speed=.3)),_check_health=Mock())
    guard=HeightRecoveryGuard(nav,target)
    plan=guard.plan(current,target)
    assert plan.can_move and plan.speed_limit<=.15
    local.states[10,10,9]=1
    assert not guard.revalidate_plan(plan,current).can_move
    local.states[10,10,9]=0
    objects.append(dict(id='car',bbox_3d=dict(min=[-.2,-.2,-.8],max=[.2,.2,0.])))
    assert not guard.revalidate_plan(plan,current).can_move
    objects.clear()
    assert not guard.plan(current+[1.,0.,0.],target).can_move
    drifted=current+np.array([.02,.01,0.])
    corrected=guard.plan(drifted,target)
    assert corrected.can_move
    np.testing.assert_array_equal(corrected.local_target_ned[:2],drifted[:2])
    assert not guard.plan(current+[.06,0.,0.],target).can_move
    guard.check_execution_health();nav._check_health.assert_called_once()


def test_airborne_preflight_certifies_recovery_but_keeps_graph_plane():
    local,cfg,current,target=fixture()
    session=object.__new__(IndependentGraphSession)
    session._restored_altitude=target[2];session.airborne_navigation=True
    session.runtime=NS(vehicle=NS(config=NS(guided_z_tolerance=.15)),
        occupancy_grid=NS(execution_snapshot=local),semantic_navigation_config=cfg)
    session._target_objects=lambda:[]
    with patch.object(GraphTargetSessionMixin,'_target_grid',return_value='graph') as make:
        assert session._target_grid(current[2],current)=='graph'
        assert make.call_args.args[0]==target[2]
        np.testing.assert_array_equal(current,[0.,0.,-1.778])
    local.states[10,10,9]=-1
    with pytest.raises(TargetUnavailableError,match='偏差=0.287'):
        session._target_grid(current[2],current)
    session.airborne_navigation=False
    with pytest.raises(TargetUnavailableError,match='未执行新目标'):
        session._target_grid(current[2],current)


def test_height_recovery_executes_only_certified_vertical_leg():
    local,cfg,current,target=fixture()
    nav=object.__new__(GraphTargetNavigator)
    nav.graph_altitude=nav.altitude=target[2]
    nav.config=cfg;nav.local_map=NS(execution_snapshot=local)
    nav.objects_provider=lambda:[];nav.local_planner=NS(config=NS(cruise_max_speed=.3))
    nav.vehicle=Mock(config=NS(guided_z_tolerance=.15))
    nav.vehicle.get_position.side_effect=lambda:current.copy()
    nav.vehicle.get_velocity.return_value=np.zeros(3)
    def move(point,**kwargs):
        assert kwargs['local_planner'].plan(current,point).can_move
        assert kwargs['z_tolerance'] == .02
        assert kwargs['xy_tolerance'] == .05
        np.testing.assert_array_equal(point[:2],current[:2])
        current[:]=point
    nav.vehicle.move_waypoint_guided.side_effect=move
    nav._check_health=Mock();nav._publish_navigation=Mock();nav._event=Mock()
    nav._navigation_state={};nav.clock=lambda:1.;nav.deadline=100.
    nav._restore_graph_height()
    np.testing.assert_allclose(current,target)
    nav.vehicle.move_waypoint_guided.assert_called_once()
    nav._restore_graph_height()  # Already on the fixed graph plane: no second motion.
    nav.vehicle.move_waypoint_guided.assert_called_once()
    current[2] = target[2]-.06  # Below ordinary 15 cm arrival tolerance, still recover.
    nav._restore_graph_height()
    assert nav.vehicle.move_waypoint_guided.call_count == 2
    current[2]=-2.3
    with pytest.raises(RuntimeError,match='不可执行'):
        nav._restore_graph_height()
    assert nav.vehicle.move_waypoint_guided.call_count == 2
