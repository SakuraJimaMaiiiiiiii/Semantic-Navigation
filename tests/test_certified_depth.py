from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from config.semantic_navigation_config import SemanticNavigationConfig
from config.planner_config import PlannerConfig
from planner.certified_depth import refine_front_clearance


def fixture():
    pose=np.eye(4)
    pose[:3,:3]=[[0,0,1],[1,0,0],[0,1,0]]  # optical Z faces NED X
    pose[0,3]=-.1
    depth=np.full((5,5),4.)
    depth[2,2]=.55
    frame=NS(depth=depth,depth_timestamp=1.,camera_intrinsics=np.array([[10.,0,2],[0,10.,2],[0,0,1.]]))
    snapshot=NS(states=np.zeros((30,30,20),dtype=np.int8),resolution=.1,
                origin_ned=np.array([-1.,-1.,-1.]),timestamp=1.,
                observation=NS(sensor_frame=frame,camera_pose_world=pose))
    body=SemanticNavigationConfig(clearance_radius=.2,vertical_half_extent=.1)
    return snapshot,body,PlannerConfig()


def check(snapshot,body,planner,target=(0.,.5,0.),stopping=(0.,0.,0.),clearance=.55):
    return refine_front_clearance(snapshot,np.zeros(3),target,stopping,clearance,body,planner)


def test_side_pillar_allows_only_clear_lateral_corridor():
    snapshot,body,planner=fixture()
    assert check(snapshot,body,planner)==.6
    assert check(snapshot,body,planner,target=(.5,0.,0.))==.55


def test_residual_velocity_requires_its_own_clear_stopping_corridor():
    snapshot,body,planner=fixture()
    assert check(snapshot,body,planner,stopping=(.5,0.,0.))==.55


@pytest.mark.parametrize('state',[-1,1])
def test_unknown_or_occupied_corridor_preserves_stop(state):
    snapshot,body,planner=fixture()
    snapshot.states[10,13,10]=state
    assert check(snapshot,body,planner)==.55


def test_missing_calibration_and_near_hard_stop_are_not_bypassed():
    snapshot,body,planner=fixture()
    assert check(snapshot,body,planner,clearance=.3)==.3
    snapshot.observation.sensor_frame.camera_intrinsics[0,0]=0.
    assert check(snapshot,body,planner)==.55
    snapshot.observation=None
    assert check(snapshot,body,planner)==.55


def test_stale_camera_cannot_release_lidar_certified_route():
    snapshot,body,planner=fixture()
    snapshot.observation.sensor_frame.depth_timestamp=0.
    assert check(snapshot,body,planner)==.55


def test_one_pixel_obstacle_overrides_free_map():
    snapshot,body,planner=fixture()
    # A single extra hit on the requested lateral chord must veto the release.
    snapshot.observation.sensor_frame.depth[2,4]=.3
    assert check(snapshot,body,planner)==.55


def test_flight_guard_applies_refinement_and_rechecks_new_raw_depth_hit():
    from navigation.mission import _FlightGuard
    from config.drone_config import Config
    snapshot,body,planner=fixture()
    snapshot.timestamp=1.
    snapshot.forward_clearance=.55
    vehicle=NS(config=Config(),get_velocity=lambda:np.zeros(3))
    local=NS(snapshot=lambda:snapshot,execution_snapshot=snapshot,
             config=NS(update_rate=5.,min_depth=.3,depth_is_range=False))
    inner=NS(config=planner,_clear_speed_limit=lambda distance:.3,discard_trajectory=Mock(),
             plan=Mock(side_effect=AssertionError('clear corridor should not invoke avoidance')))
    nav=NS(altitude=0.,config=body,local_map=local,global_map=None,vehicle=vehicle,
           local_planner=inner,_sample_position=Mock(),_check_maps=Mock())
    guard=_FlightGuard(nav)
    current=np.zeros(3)
    plan=guard.compute_plan(current,np.array([0.,.5,0.]))
    assert plan.can_move and plan.speed_limit <= .1
    certificate=guard.execution_certificate()
    assert guard.revalidate_plan(plan,current,certificate=certificate).can_move
    old=snapshot.observation
    new_depth=old.sensor_frame.depth.copy()
    new_depth[2,4]=.3
    snapshot.observation=NS(sensor_frame=NS(depth=new_depth,depth_timestamp=1.,
                                          camera_intrinsics=old.sensor_frame.camera_intrinsics),
                            camera_pose_world=old.camera_pose_world)
    assert not guard.revalidate_plan(plan,current,certificate=certificate).can_move
    assert guard._execution_rejection['reason']=='post_planning_front_depth_blocked'
