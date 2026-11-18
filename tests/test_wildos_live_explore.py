"""Offline integration checks: SDK imports are inert; no simulator/vehicle opens."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Existing SceneMemory imports the vehicle package even for offline graph tests.
# Stub only missing vendor modules, never flight code or the planner under test.
missing = {name: ModuleType(name) for name in ('PX4MavCtrlV4', 'ReqCopterSim', 'UE4CtrlAPI')
           if name not in sys.modules and importlib.util.find_spec(name) is None}
sys.modules.update(missing)
try:
    from exploration_wildos.explore import (GraphFrontiers, GraphPlanner, GraphRouteGrid,
        GraphExplorer, GraphExplorationSession)
    from navigation.frontier_explorer import FrontierExplorer
    from navigation.scene_memory import SceneMemory
finally:
    for name in missing:
        sys.modules.pop(name, None)

from config.exploration_config import FrontierExplorationConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from exploration_wildos.static_graph import GraphConfig
from exploration_wildos.map_adapter import MapAdapter
from exploration_wildos.vehicle_constraints import constrain_adapter
from navigation.exploration_planner import ExplorationPose, ExplorationHistory
from navigation.route import NavigationGrid


def grid(size=6, occupied=()):
    return NavigationGrid({(x,y) for x in range(-size,size+1) for y in range(-size,size+1)},
        set(occupied), 1., (0.,0.), -1.5, SemanticNavigationConfig(clearance_radius=.4))


def manager():
    return GraphFrontiers(FrontierExplorationConfig(), options=GraphConfig(
        paper_aligned=True, max_samples=45, max_nodes=300, free_radius_cap=4.,
        edge_radius=8., prune_redundant=False))


def prepared():
    m, g, current = manager(), grid(), np.array([0.,0.,-1.5])
    m.refresh(g, current, frame_key='test-px4')
    choices, _, _ = m.update(g, current, current, 0., set())
    assert choices
    return m, g, current, choices


def test_candidates_are_frontier_nodes_and_route_uses_actual_edges():
    m, g, current, choices = prepared()
    planner = GraphPlanner(m)
    waypoint = planner.select_next_waypoint(g, ExplorationPose(current, 0.),
        ExplorationHistory(current), candidates=choices)
    assert waypoint['source'] == 'wildos_frontier_node'
    stable = f'nav:{planner.selected_key[0]}'
    node = next(n for n in m.graph['nodes'] if n['stable_id'] == stable)
    assert node['is_frontier'] and node['frontier_ids']
    route = m.route(g, current, waypoint['position'], stable)
    assert route is not None
    edges = {frozenset((e['source_stable_id'], e['target_stable_id'])) for e in m.graph['edges']}
    chain = m.last_route['node_path']
    assert all(frozenset((a,b)) in edges for a,b in zip(chain,chain[1:]))
    assert all(g.segment_free(a,b) for a,b in zip(route,route[1:]))


def test_new_obstacle_rejects_old_goal_without_astar_fallback():
    m,g,current,choices = prepared()
    key, goal = choices[0]
    blocked = grid(occupied=[g.cell(goal)])
    with patch.object(NavigationGrid, 'path', side_effect=AssertionError('A* fallback forbidden')):
        assert m.route(blocked,current,goal,f'nav:{key[0]}') is None
        m.refresh(blocked,current,frame_key='test-px4')
        assert m.route(blocked,current,goal,f'nav:{key[0]}') is None


def test_visits_do_not_mean_boundary_coverage_or_completion():
    m,g,current,choices = prepared()
    for key,_ in choices:
        m.record_visit(m.candidates[key])
    remaining,limited,_ = m.update(g,current,current,0.,set())
    assert remaining == [] and not limited
    assert m.regions and sum(map(len,m.regions)) > 1
    assert not m.batch_pending


def test_geometry_cache_and_epoch_reset():
    m,g,current,_ = prepared()
    count, state = m.update_count, json.dumps(m.state, sort_keys=True)
    assert not m.refresh(g,current+[.1,0,0],frame_key='test-px4')
    assert m.update_count == count and json.dumps(m.state,sort_keys=True) == state
    m.visited['nav:1'] = frozenset({(7,0)})
    m.refresh(g,current,frame_key='test-px4-new')
    assert m.epoch_number == 1 and not m.visited


def test_unattached_robot_keeps_unknown_regions():
    m,g,current,_ = prepared()
    unsafe = np.array([100.,100.,-1.5])
    choices,_,_ = m.update(g,unsafe,current,0.,set())
    assert not choices and m.regions
    assert m.diagnostics['reachable_nodes'] == 0


def test_vehicle_geometry_excludes_nodes_and_routes():
    g, m, current = grid(), manager(), np.array([-4.,0.,-1.5])
    objects=[dict(id='object:car',bbox_3d={'min':[-1.,-1.,-2.], 'max':[1.,1.,0.]})]
    adapter,_ = constrain_adapter(MapAdapter.from_grid(g),objects)
    m.refresh(adapter.grid,current,frame_key='test-px4',objects=objects)
    assert m.validation['vehicles']['passed']
    assert all(adapter.grid.cell(n['position_ned']) not in {(0,0),(-1,0)} for n in m.graph['nodes'])


def explorer(tmp_path):
    e = GraphExplorer(vehicle=Mock(), perception=Mock(), local_map=Mock(),
        global_map=SimpleNamespace(revision=0), local_planner=Mock(config=Mock()),
        request=None, config=SemanticNavigationConfig(), output_path=tmp_path/'report.json',
        exploration_config=FrontierExplorationConfig(),
        memory=SceneMemory('test', [0,0,0], 0., FrontierExplorationConfig()))
    e.frontiers = manager()
    e.exploration_planner = GraphPlanner(e.frontiers)
    e.altitude = e._nominal_altitude = -1.5
    e.exploration_origin = np.array([0.,0.,-1.5])
    e.started = 0.
    e.clock = lambda: 1.
    e.deadline = 100.
    e._check_health = Mock()
    e._check_maps = Mock()
    e._hold_for_planning = Mock(return_value=None)
    e.vehicle.get_position.return_value = np.array([0.,0.,-1.5])
    e.vehicle.get_euler.return_value = np.zeros(3)
    e.vehicle.config = SimpleNamespace(guided_z_tolerance=.15)
    return e


def test_inherited_travel_consumes_graph_waypoints(tmp_path):
    e = explorer(tmp_path)
    g, position = grid(), np.array([0.,0.,-1.5])
    e.vehicle.get_position.side_effect = lambda:position.copy()
    e.vehicle.get_velocity.return_value = np.zeros(3)
    e.frontiers.refresh(g,position,frame_key='test:PX4-raw:0')
    choices,_,_ = e.frontiers.update(g,position,position,0.,set())
    e.active_viewpoint = e.frontiers.candidates[choices[0][0]]
    goal = e.active_viewpoint.point.copy()
    e._grid = lambda:GraphRouteGrid(g,e._graph_path) if e._graph_travel else g
    e._update_coverage = Mock()
    e._checkpoint = Mock()
    executed=[]
    def move(path, **kwargs):
        assert all(g.segment_free(a,b) for a,b in zip(path,path[1:]))
        executed.append([p.copy() for p in path])
        position[:] = path[1]  # deterministic mock vehicle, one graph segment per call
    e._move = move
    with patch.object(NavigationGrid,'path',side_effect=AssertionError('No goal A*')):
        assert e._travel(goal)
    assert executed and np.allclose(position,goal)
    assert e.frontiers.last_route['target'] == f'nav:{e.active_viewpoint.key[0]}'
    assert not e._graph_travel


def test_execution_rechecks_fresh_map_and_keeps_existing_guard(tmp_path):
    e = explorer(tmp_path)
    e._graph_travel = True
    path=[np.array([0.,0.,-1.5]),np.array([2.,0.,-1.5])]
    e._grid = lambda:grid(occupied=[(1,0)])
    with patch.object(FrontierExplorer,'_move') as inherited:
        with pytest.raises(TimeoutError,match='graph_route_changed'):
            e._move(path)
        inherited.assert_not_called()
        e.vehicle.hover.assert_called()
    e._grid = grid
    with patch.object(FrontierExplorer,'_move') as inherited:
        e._move(path)
        inherited.assert_called_once_with(path,recovery=False,continuous=False)


def test_new_route_skips_reached_connector_and_turns_toward_departure(tmp_path):
    e = explorer(tmp_path)
    e._graph_travel = True
    e._grid = grid
    e._turn_for_graph_departure = Mock()
    path = [np.array([0.,0.,-1.5]), np.array([.14,.12,-1.5]), np.array([-2.,0.,-1.5])]
    with patch.object(FrontierExplorer, '_move') as inherited:
        e._move(path)
    assert len(path) == 2
    np.testing.assert_allclose(path[1], [-2.,0.,-1.5])
    assert abs(e._turn_for_graph_departure.call_args.args[0]) == pytest.approx(np.pi)
    inherited.assert_called_once_with(path, recovery=False, continuous=False)


def test_prefix_skip_never_cuts_corner_or_drops_terminal():
    from exploration_wildos.departure import trim_reached_prefix
    p = [np.array([0.,0.,-1.5]), np.array([.2,0.,-1.5]), np.array([1.,1.,-1.5])]
    denied = SimpleNamespace(segment_free=lambda a,b:False)
    safe = SimpleNamespace(segment_free=lambda a,b:True)
    assert not trim_reached_prefix(p,p[0],denied,.4,.15)[1]
    assert not trim_reached_prefix(p[:2],p[0],safe,.4,.15)[1]
    high = [p[0], np.array([.2,0.,-1.]), p[2]]
    assert not trim_reached_prefix(high,p[0],safe,.4,.15)[1]


@pytest.mark.parametrize('safe', [True, False])
def test_departure_turn_checks_body_and_keeps_health_without_front_stop_bypass(tmp_path, safe):
    e = explorer(tmp_path)
    e._grid = grid
    e._motion_progress = (np.zeros(3), 0.)
    e.local_map.forward_clearance = .48  # translation blocked, rotation independently certified
    e.local_map.execution_snapshot = object()
    def align(yaw, **kwargs):
        assert e._motion_progress is None
        assert e._foreground_graph
        kwargs['health_check']()
    e.vehicle.align_yaw.side_effect = align
    with patch('navigation.route.execution_segment_free', return_value=safe) as check:
        if safe:
            e._turn_for_graph_departure(np.pi)
            assert check.call_count == 3
            e.vehicle.align_yaw.assert_called_once()
        else:
            with pytest.raises(TimeoutError, match='graph_departure_without_clearance'):
                e._turn_for_graph_departure(np.pi)
            e.vehicle.align_yaw.assert_not_called()
    assert e._motion_progress is not None and not e._foreground_graph
    e.vehicle.move_waypoint_guided.assert_not_called()


def test_map_change_during_turn_cancels_departure(tmp_path):
    e = explorer(tmp_path)
    e._graph_travel = True
    states = iter([grid(),grid(),grid(occupied=[(1,0)])])
    e._grid = lambda: next(states)
    e.vehicle.get_euler.return_value = np.array([0.,0.,np.pi])
    e.local_map.execution_snapshot = object()
    with patch('navigation.route.execution_segment_free', return_value=True), patch.object(FrontierExplorer, '_move') as move:
        with pytest.raises(TimeoutError, match='graph_route_changed_before_execution'):
            e._move([np.array([0.,0.,-1.5]),np.array([2.,0.,-1.5])])
        move.assert_not_called()


def test_route_revalidation_retries_once_and_preserves_failure_history(tmp_path):
    e = explorer(tmp_path)
    e.active_viewpoint = SimpleNamespace(key=(198,0))
    e.exploration_waypoints = [dict(number=21,status='active')]
    e._grid = grid
    def failed(goal):
        e._last_travel_failure = dict(reason='graph_route_changed_before_execution')
        return False
    with patch.object(FrontierExplorer, '_travel', side_effect=failed) as travel:
        assert not e._travel(np.array([3.,0.,-1.5]))
        assert travel.call_count == 2
    assert len(e.exploration_waypoints[-1]['replans']) == 1
    assert e._last_travel_failure['scope'] == 'route_revalidation'
    assert e._last_travel_failure['target_invalidated'] is False
    assert e._transient_target_failure == ((198,0),0)


@pytest.mark.parametrize('retry_success', [True, False])
def test_replanned_route_success_or_disconnection_does_not_invalidate_target(tmp_path, retry_success):
    e = explorer(tmp_path)
    e.active_viewpoint = SimpleNamespace(key=(208,0))
    e.exploration_waypoints = [dict(number=21,status='active')]
    calls = []
    def travel(goal):
        calls.append(1)
        if len(calls) == 1:
            e._last_travel_failure = dict(reason='graph_route_changed_before_execution')
            return False
        e._last_travel_failure = None if retry_success else dict(reason='astar_no_route')
        return retry_success
    with patch.object(FrontierExplorer, '_travel', side_effect=travel):
        assert e._travel(np.array([3.,0.,-1.5])) == retry_success
    assert len(calls) == 2
    assert len(e.exploration_waypoints[-1]['replans']) == 1
    if retry_success:
        assert e._last_travel_failure is None
        assert not hasattr(e, '_transient_target_failure')
    else:
        assert e._last_travel_failure['reason'] == 'graph_no_certified_route'
        assert e._last_travel_failure['scope'] == 'route_revalidation'
        assert e._last_travel_failure['target_invalidated'] is False


@pytest.mark.parametrize('clearance', [1., float('inf')])
def test_departure_failure_does_not_burn_target_and_blocks_other_targets_until_change(tmp_path, clearance):
    e = explorer(tmp_path)
    key = (198,0)
    e.active_viewpoint = SimpleNamespace(key=key)
    e.exploration_waypoints = [dict(number=15,status='active')]
    e._grid = grid
    e.local_map.forward_clearance = .48
    e.local_planner.config.emergency_stop_distance = .6
    def failed(goal):
        e._last_travel_failure = dict(reason='exploration_no_motion_progress', local_status='emergency_stop')
        return False
    with patch.object(FrontierExplorer, '_travel', side_effect=failed):
        assert not e._travel(np.array([3.,0.,-1.5]))
    record = e.exploration_waypoints[-1]
    record.update(status='blocked', failure=e._last_travel_failure)
    e.failed_frontiers[key] = (1, 99.)  # inherited run's accounting
    e._checkpoint()
    assert e.failed_frontiers[key][0] == 0
    assert record['status'] == 'blocked' and record['failure']['scope'] == 'departure_blocked'
    e._refresh = Mock()
    assert e._choices(grid(),np.array([0.,0.,-1.5]),np.zeros(3)) == ([],False,False)
    e._refresh.assert_not_called()
    e.local_map.forward_clearance = clearance
    with patch.object(FrontierExplorer, '_choices', return_value=([],False,False)) as choices:
        e._choices(grid(),np.array([0.,0.,-1.5]),np.zeros(3))
        choices.assert_called_once()
    assert e._departure_block is None and e.failed_frontiers[key] == (0,0.)


def test_failure_after_transit_is_not_misattributed_to_departure():
    from exploration_wildos.departure import transient_failure
    failure = dict(reason='exploration_no_motion_progress', local_status='emergency_stop')
    assert transient_failure(failure,np.zeros(3),np.array([2.,0.,0.]),.4) is None


def test_epoch_change_cancels_committed_target(tmp_path):
    e=explorer(tmp_path)
    g,current=grid(),np.array([0.,0.,-1.5])
    e.vehicle.get_position.return_value=current
    e.frontiers.refresh(g,current,frame_key='old')
    e._committed_epoch=e.frontiers.epoch_number
    e._committed_node=e.frontiers.graph['nodes'][0]['stable_id']
    assert e._graph_path(g,current,e.frontiers.graph['nodes'][0]['position_ned']) is None


def test_checkpoint_saves_state_and_graph_under_run(tmp_path):
    e=explorer(tmp_path)
    g,current=grid(),np.array([0.,0.,-1.5])
    e.frontiers.refresh(g,current,frame_key='test')
    e._checkpoint()
    output=tmp_path/'navigation_graph'/'update_000001'
    assert (output/'graph.html').exists()
    assert (output/'incremental_state.json').exists()
    e._checkpoint()  # unchanged graph does not recreate existing output directory
    assert len(list((tmp_path/'navigation_graph').glob('update_*'))) == 1


def test_checkpoint_keeps_two_complete_graph_updates(tmp_path):
    e = explorer(tmp_path)
    e.frontiers.refresh(grid(), np.array([0., 0., -1.5]), frame_key='test')
    for number in range(1, 6):
        e.frontiers.update_count = number
        e._checkpoint()
    assert sorted(p.name for p in e.graph_directory.glob('update_*')) == ['update_000004', 'update_000005']
    with np.load(e.latest_graph_directory / 'planning_grid.npz') as data:
        assert data['free_indices'].shape[1] == 2
    assert (e.latest_graph_directory / 'incremental_state.json').is_file()


def test_failed_graph_checkpoint_cleans_temporary_files_and_allows_retry(tmp_path):
    e = explorer(tmp_path)
    e.frontiers.refresh(grid(), np.array([0., 0., -1.5]), frame_key='test')
    e._checkpoint()
    previous = e.latest_graph_directory
    e.frontiers.update_count += 1
    with patch('exploration_wildos.explore.np.savez_compressed', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            e._checkpoint()
    assert list(e.graph_directory.iterdir()) == [previous]
    assert e.latest_graph_directory == previous
    e._checkpoint()
    assert e.latest_graph_directory.name == 'update_000002'


def test_help_needs_no_sdk_or_simulator():
    result=subprocess.run([sys.executable,'-m','exploration_wildos.explore','--help'],
                          capture_output=True,text=True)
    assert result.returncode == 0
    assert '--sam2' in result.stdout


def test_real_exploration_loop_does_not_complete_unobserved_boundary(tmp_path):
    e=explorer(tmp_path)
    g,current=grid(),np.array([0.,0.,-1.5])
    e._sample_position=lambda:current.copy()
    e.vehicle.get_position.return_value=current.copy()
    e.vehicle.get_euler.return_value=np.zeros(3)
    e._grid=lambda:g
    e._initial_observation=Mock()
    e._record_coverage=Mock()
    e._update_coverage=lambda grid:False
    e.coverage_progress.metrics["stop_due"]=False
    e._recent_map_gain=lambda:0.
    e._recovery_observation=lambda *args:False
    e._confirm_empty_frontiers=lambda:(True,20.)
    e._guard=Mock()
    e.frontiers.refresh(g,current,frame_key='test:PX4-raw:0')
    choices,_,_=e.frontiers.update(g,current,current,0.,set())
    for key,_ in choices:
        e.frontiers.record_visit(e.frontiers.candidates[key])
    assert not e.run()
    assert not e.result['frontiers_exhausted']
    assert e.result['reason']=='unobserved_frontiers'
    assert e.result['strategy']=='wildos_frontier_graph'
    assert e.result['returned_home'] is False


def test_session_failure_retains_original_error_and_never_confirms_home(tmp_path):
    s=object.__new__(GraphExplorationSession)
    s.home, s.home_yaw = np.zeros(3), 0.
    s.directory=tmp_path
    s.config=FrontierExplorationConfig()
    s.memory=Mock(records={},exploration={})
    raw=Mock()
    s.runtime=SimpleNamespace(global_sparse_map=raw, controller=SimpleNamespace(),
        vehicle=Mock(get_position=Mock(return_value=np.zeros(3))),
        run_mission=Mock(side_effect=RuntimeError('original mission fault')))
    s._arguments=lambda request,path:dict(vehicle=Mock(),perception=Mock(),local_map=Mock(),
        global_map=Mock(),local_planner=Mock(config=Mock()),request=None,
        config=SemanticNavigationConfig(),output_path=path)
    s.annotate=Mock()
    s._confirm_home=Mock()
    with pytest.raises(RuntimeError,match='original mission fault'):
        s.explore()
    assert s.graph_explorer.global_map is raw
    s._confirm_home.assert_not_called()
    assert not s.memory.exploration['returned_home']


def test_health_failure_during_background_build_cannot_publish(tmp_path):
    import multiprocessing
    e=explorer(tmp_path)
    e._check_health=Mock(side_effect=TimeoutError('mission deadline'))
    before={p.pid for p in multiprocessing.active_children()}
    with pytest.raises(TimeoutError,match='mission deadline'):
        e._refresh(grid(),np.array([0.,0.,-1.5]))
    assert e.frontiers.graph is None
    assert e.frontiers.update_count == 0
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_local_detour_cannot_bypass_vehicle_geometry(tmp_path):
    from exploration_wildos.explore import VehicleAwareGuard
    from navigation.mission import _FlightGuard
    from planner.local_avoidance_planner import LocalPlan
    e=explorer(tmp_path)
    e._objects=lambda:[dict(id='car',bbox_3d={'min':[1.,-1.,-2.], 'max':[2.,1.,0.]})]
    plan=LocalPlan('ego_avoid',np.array([3.,0.,-1.5]),np.array([[0,0,-1.5],[3,0,-1.5]]),.5)
    with patch.object(_FlightGuard,'revalidate_plan',return_value=plan) as inherited:
        result=e._guard.revalidate_plan(plan,np.array([0.,0.,-1.5]))
        inherited.assert_called_once()
        assert not result.can_move
        assert e._guard._execution_rejection['vehicle']=='car'
    stopped=LocalPlan('waiting_for_map',None,np.empty((0,3)),0.)
    with patch.object(_FlightGuard,'revalidate_plan',return_value=stopped):
        assert e._guard.revalidate_plan(plan,np.array([0.,0.,-1.5])) is stopped


def test_combined_page_uses_same_raw_frame_and_two_panels(tmp_path):
    import exploration_wildos.explore as module
    e=explorer(tmp_path)
    e.frontiers.refresh(grid(),np.array([0.,0.,-1.5]),frame_key='test')
    e._checkpoint()
    session=object.__new__(GraphExplorationSession)
    snapshot=object()
    session.runtime=SimpleNamespace(global_sparse_map=SimpleNamespace(snapshot=lambda:snapshot),
        vehicle=SimpleNamespace(get_position=lambda:np.array([0,0,-1.5])),
        semantic_navigation_config=SemanticNavigationConfig())
    session.map_path=tmp_path/'map.html'
    session.memory=e.memory
    session.home=np.array([0.,0.,0.]);session.home_yaw=0.
    session.exploration_waypoints=[];session.show_place_links=True
    session.graph_explorer=e
    def render(path,source,memory,**kwargs):
        assert source is snapshot
        path.write_text('<html><head></head><body><svg viewBox="-10 -10 20 20"></svg></body></html>')
    with patch.object(module,'render_map',side_effect=render):
        session.annotate()
        session.annotate()  # rebuild from raw top, never recursively nest panels
    content=session.map_path.read_text()
    assert content.count('id="exploration-panel"') == 1
    assert content.count('id="graph-panel"') == 1
    assert 'graph-frontiers' in content and 'data-map-action="in"' in content


def test_new_cli_keeps_existing_configuration_options():
    from exploration_wildos.explore import parse_config
    from run_explore import parse_config as original
    args=['--sam2','--timeout','600','--max-radius','30','--return-reserve','60']
    config,parsed=parse_config(args,include_perception=True)
    assert config==original(args)
    assert parsed.perception_backend=='yolo_sam2'


def test_process_builder_returns_graph_and_leaves_no_child(tmp_path):
    import multiprocessing
    e=explorer(tmp_path)
    e.vehicle.get_position.return_value=np.array([0.,0.,-1.5])
    before={p.pid for p in multiprocessing.active_children()}
    e._refresh(grid(),np.array([0.,0.,-1.5]))
    assert e.frontiers.validation['passed']
    assert e.frontiers.graph['nodes']
    assert e._check_health.call_count >= 2
    assert {p.pid for p in multiprocessing.active_children()} == before
    assert any(event['state']=='WILDOS_GRAPH_BUILD_FINISHED' for event in e.events)


def test_worker_error_is_reported_and_process_reaped(tmp_path):
    import multiprocessing
    e=explorer(tmp_path)
    before={p.pid for p in multiprocessing.active_children()}
    with pytest.raises(RuntimeError,match='Graph worker failed'):
        e._build_in_process({'options':{'max_samples':1}})  # invalid snapshot, worker must fail
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_worker_time_budget_cancels_process(tmp_path):
    import multiprocessing
    e=explorer(tmp_path)
    before={p.pid for p in multiprocessing.active_children()}
    with patch('exploration_wildos.explore.time',
               SimpleNamespace(monotonic=Mock(side_effect=[0.,31.]))):
        with pytest.raises(TimeoutError,match='graph_build_time_budget_exceeded'):
            e._build_in_process({'options':{'max_samples':1}})
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_later_target_navigation_uses_same_vehicle_policy(tmp_path):
    import copy
    from exploration_wildos.explore import VehicleTargetNavigator, VehicleAwareGuard
    from exploration_wildos.vehicle_motion import VehiclePlanningMap
    records=[dict(runtime_instance_id='vehicle_01',instance_id='vehicle_01',class_name='car',
                  position_world=[1.,2.,-.4],bbox_3d={'min':[0.,1.,-.7],'max':[2.,3.,0.]})]
    before=copy.deepcopy(records)
    source=Mock()
    nav=VehicleTargetNavigator(vehicle=Mock(),perception=Mock(),local_map=Mock(),
        global_map=source,local_planner=Mock(config=Mock()),request=None,
        config=SemanticNavigationConfig(),output_path=tmp_path/'target.json',
        records=records,annotate=Mock())
    assert isinstance(nav.global_map,VehiclePlanningMap)
    assert isinstance(nav._guard,VehicleAwareGuard)
    assert nav.global_map.source is source
    assert nav._objects()[0]['position_ned']==[1.,2.,-.4]
    assert records==before


def test_moving_robot_changes_attachment_without_moving_nodes():
    m, g, current, _ = prepared()
    before = {n['stable_id']: list(n['position_ned']) for n in m.graph['nodes']}
    for node in m.graph['nodes'][:5]:
        m.update_robot_attachment(g, np.array(node['position_ned']))
        assert m.graph['robot_attachment']['node_stable_id'] == node['stable_id']
    assert before == {n['stable_id']: list(n['position_ned']) for n in m.graph['nodes']}
    m.update_robot_attachment(g, np.array([1000., 1000., -1.5]))
    assert m.graph['robot_attachment']['status'] == 'no_certified_connection'


def test_background_incremental_graph_updates_without_vehicle_api():
    from exploration_wildos.moving_graph import MovingGraph
    import copy
    m, g, current, _ = prepared()
    pending = copy.deepcopy(m)
    larger = grid(size=8)
    worker = MovingGraph()
    try:
        worker.submit(pending, larger, current, 'test-px4', [])
        changed, result = worker.future.result(timeout=30)
        assert changed
        assert result.update_count == m.update_count + 1
        assert result.validation['passed']
        assert m.update_count == 1
        assert result.epoch == m.epoch
    finally:
        worker.close()


@pytest.mark.parametrize('stale', [False, True])
def test_moving_result_commit_rejects_stale_graph_version(tmp_path, stale):
    import copy
    import time
    from concurrent.futures import Future
    m, g, current, _ = prepared()
    frame = 'session:PX4-raw:0'
    m.epoch = (frame, *m.epoch[1:])
    pending = copy.deepcopy(m)
    pending.update_count += 1
    future = Future()
    future.set_result((True, pending))
    explorer = object.__new__(GraphExplorer)
    explorer.frontiers = m
    explorer._next_moving_tick = 0
    explorer._next_moving_build = time.monotonic()+100
    explorer._moving_graph = SimpleNamespace(future=future)
    explorer._moving_base = (m.update_count - int(stale), frame)
    explorer._grid = lambda: g
    explorer.vehicle = SimpleNamespace(get_position=lambda: current)
    explorer.global_map = SimpleNamespace(revision=0)
    explorer.memory = SimpleNamespace(session_id='session')
    explorer.graph_directory = tmp_path
    explorer._event = Mock()
    explorer._moving_graph_tick()
    assert m.update_count == (1 if stale else 2)
    assert json.loads((tmp_path/'robot_attachment_live.json').read_text())['status'] == 'attached'


def test_cleanup_same_batch_rejects_points_inside_new_node_circles():
    from dataclasses import replace
    m = manager()
    m.options = replace(m.options, incremental_cleanup=True)
    g = grid(size=8)
    m.refresh(g, np.array([0., 0., -1.5]), frame_key='cleanup')
    nodes = m.state['nodes']
    for i, node in enumerate(nodes):
        for prior in nodes[:i]:
            assert np.linalg.norm(np.array(node['position_ned'])-prior['position_ned']) > prior['free_radius']
    assert m.validation['passed']


def test_retirement_preserves_explored_memory_and_does_not_reactivate_deleted_ids():
    from exploration_wildos.node_lifecycle import retire_compacted_nodes
    from exploration_wildos.incremental_graph import update_graph
    import copy
    m, g, current, _ = prepared()
    old = copy.deepcopy(m.state)
    state = retire_compacted_nodes(m.state, m.graph)
    kept = {n['stable_id'] for n in m.graph['nodes']}
    retired = {n['stable_id'] for n in old['nodes']} - kept
    assert retired
    assert {n['stable_id'] for n in state['nodes']} == kept
    assert {n['stable_id'] for n in state['explored_memory']} == retired
    assert m.state == old
    adapter = MapAdapter.from_grid(g, dict(m.adapter.metadata, timestamp=2., snapshot_id='next'))
    graph, next_state = update_graph(adapter, frame_key='test-px4', state=state,
                                     options=m.options, start=current.tolist())
    assert not retired & {n['stable_id'] for n in graph['nodes']}
    assert next_state['explored_memory'] == state['explored_memory']


def test_live_background_growth_cleans_nodes_each_update_without_resurrection():
    """Exercise production defaults and the spawned worker across growing maps."""
    import copy
    from exploration_wildos.moving_graph import MovingGraph
    m = GraphFrontiers(FrontierExplorationConfig())
    assert m.options.incremental_cleanup
    current = np.array([0., 0., -1.5])
    retired = set()
    worker = MovingGraph()
    try:
        for size in (6, 9, 12, 15):
            worker.submit(copy.deepcopy(m), grid(size=size), current, 'moving-test', [])
            changed, m = worker.future.result(timeout=30)
            assert changed and m.validation['passed']
            ids = {n['stable_id'] for n in m.graph['nodes']}
            assert not ids & retired
            assert {n['stable_id'] for n in m.state['nodes'] if n['active']} == ids
            report = m.graph['node_compaction']
            assert report['covered_cells_before'] == report['covered_cells_after']
            assert report['components_preserved']
            retired.update(m.state['retirement']['retired_ids'])
            assert retired <= {n['stable_id'] for n in m.state['explored_memory']}
            for frontier in m.graph['frontiers']:
                assert frontier['node_stable_id'] in ids
        assert retired, 'Fixture must actually retire nodes, not just exercise a no-op'
        assert m.update_count == 4
    finally:
        worker.close()


def test_wildos_target_session_rejects_unreachable_before_takeoff(tmp_path):
    from exploration_wildos.target_session import GraphTargetSessionMixin, TargetUnavailableError
    from navigation.target import SemanticTargetRequest
    session=object.__new__(GraphExplorationSession)
    session.directory=tmp_path
    session.home=np.zeros(3)
    session.clock=lambda:10.
    session.config=FrontierExplorationConfig()
    session.watch=Mock()
    session.airborne_navigation=False
    session.perception=Mock()
    session.perception.semantic_snapshot.return_value=SimpleNamespace(completed_at=10.)
    source=SimpleNamespace(last_observation_completed_at=10.)
    session.runtime=SimpleNamespace(vehicle=Mock(),controller=Mock(),occupancy_grid=source,
        global_sparse_map=source,mission_config=SimpleNamespace(takeoff_height=1.5),
        semantic_navigation_config=SemanticNavigationConfig())
    session.runtime.vehicle.get_position.return_value=np.zeros(3)
    session.navigation_archive=lambda:dict(home=np.zeros(3),records=[])
    session.annotate=Mock()
    session._target_grid=Mock(return_value=SimpleNamespace(approach=lambda record,start:None))
    with patch('exploration_wildos.target_session.select_targets',return_value=[dict(runtime_instance_id='vehicle_02')]):
        with pytest.raises(TargetUnavailableError,match='WildOS graph'):
            session.navigate(SemanticTargetRequest(instance_id='vehicle_02'),'test')
    session.runtime.controller.run_target.assert_not_called()
    session.runtime.vehicle.enable_offboard.assert_not_called()


def test_wildos_target_navigator_logs_graph_route_and_keeps_arrival_semantics(tmp_path):
    from exploration_wildos.target_navigation import GraphTargetNavigator
    from exploration_wildos.target_graph import TargetGraphGrid
    from navigation.target import SemanticTargetRequest
    g=grid()
    points=[g.point((x,0)) for x in (0,2,4)]
    graph=dict(nodes=[dict(id=i,stable_id=f'nav:{i}',position_ned=p.tolist()) for i,p in enumerate(points)],
               edges=[dict(source=0,target=1),dict(source=1,target=2)],options=dict(edge_radius=3.))
    current=points[0].copy()
    vehicle=Mock();vehicle._motion_progress_display=None
    vehicle.get_position.side_effect=lambda:current.copy()
    source=SimpleNamespace(snapshot=lambda:SimpleNamespace(timestamp=1.))
    nav=GraphTargetNavigator(vehicle=vehicle,perception=Mock(),local_map=source,global_map=source,
        local_planner=Mock(),request=SemanticTargetRequest(instance_id='vehicle_02'),
        config=SemanticNavigationConfig(),output_path=tmp_path/'navigation_test.json',clock=lambda:10.,
        records=[],annotate=Mock(),graph_provider=lambda:graph,objects_provider=lambda:[],frame_validator=lambda:None)
    nav.global_map=source
    nav._check_health=Mock()
    nav._frame=Mock(return_value=SimpleNamespace(timestamp=1.))
    nav._query=Mock(return_value=[dict(runtime_instance_id='vehicle_02')])
    nav._verify=Mock(return_value=None)
    def make_grid(position):
        routing=TargetGraphGrid(g,graph)
        def approach(record,start):
            path=routing.path(start,points[-1])
            return 4.,points[-1].copy(),path
        routing.approach=approach
        nav._last_target_grid=routing
        return routing
    nav._navigation_grid=make_grid
    def move(path):
        nonlocal current
        assert nav.graph_routes[-1]['node_path']==['nav:0','nav:1','nav:2']
        current=np.array(path[-1])
    nav._move=move
    assert nav.run()
    assert nav.result['reason']=='arrived_visual_unconfirmed'
    assert nav.result['planning_backend']=='wildos_graph_dijkstra'
    assert (tmp_path/'navigation_test.graph_route.json').exists()
    assert nav.result['graph_routes'][0]['node_path']==['nav:0','nav:1','nav:2']


def test_target_frame_loss_during_flight_is_not_a_preflight_rejection():
    from exploration_wildos.target_navigation import GraphTargetNavigator
    from navigation.exploration_session import TargetUnavailableError
    nav=object.__new__(GraphTargetNavigator)
    nav.frame_validator=Mock(side_effect=TargetUnavailableError('frame changed'))
    nav.graph_provider=Mock()
    with pytest.raises(RuntimeError,match='wildos_target_graph_frame_invalid'):
        nav._validated_graph()
    nav.graph_provider.assert_not_called()


def test_target_executes_graph_corners_and_publishes_measured_trace(tmp_path):
    import copy
    from exploration_wildos.target_navigation import GraphTargetNavigator
    nav=object.__new__(GraphTargetNavigator)
    path=[np.array([0.,0.,-1.5]),np.array([2.,0.,-1.5]),np.array([2.,2.,-1.5])]
    current=path[0].copy()
    nav.vehicle=SimpleNamespace(get_position=lambda:current.copy(),waypoint_control_state={'status':'route_clear'})
    nav.global_map=SimpleNamespace(revision=0)
    nav._check_health=Mock()
    nav._navigation_grid=Mock(return_value=SimpleNamespace(segment_free=lambda a,b:True))
    nav._navigation_state=dict(route={'points_ned':[p.tolist() for p in path]},trace_ned=[],active_segment=None)
    states=[]
    nav.live_map=SimpleNamespace(navigation=lambda state:states.append(copy.deepcopy(state)))
    nav.output_path=tmp_path/'target.json'
    executed=[]
    def move(segment, *, flowing):
        assert flowing is False  # no rounded shortcut past a graph vertex
        executed.append([p.copy() for p in segment])
        current[:]=segment[-1]
        return False
    nav._move_segment=move
    with patch('navigation.path_smoothing.rounded_path',side_effect=AssertionError('No corner smoothing')):
        nav._move(path)
    np.testing.assert_allclose([s[-1] for s in executed],path[1:])
    assert [s['segment_index'] for s in states[:-1]]==[0,1]
    assert states[-1]['active_segment'] is None
    np.testing.assert_allclose(states[-1]['trace_ned'],path)
    assert json.loads((tmp_path/'target.navigation.json').read_text())['phase']=='路径执行完成'


def test_target_execution_still_rejects_newly_blocked_edge(tmp_path):
    from exploration_wildos.target_navigation import GraphTargetNavigator
    nav=object.__new__(GraphTargetNavigator)
    nav.vehicle=SimpleNamespace(get_position=lambda:np.array([0.,0.,-1.5]))
    nav.global_map=SimpleNamespace(revision=0)
    nav._check_health=Mock()
    nav._navigation_grid=Mock(return_value=SimpleNamespace(segment_free=lambda a,b:False))
    nav._navigation_state=dict(route=None,trace_ned=[],active_segment=None)
    nav.live_map=None;nav.output_path=tmp_path/'target.json'
    nav._move_segment=Mock()
    nav._move([np.array([0.,0.,-1.5]),np.array([2.,0.,-1.5])])
    nav._move_segment.assert_not_called()
    assert nav._navigation_state['phase']=='路径中断，等待重新规划'


def test_wildos_target_client_handshake_uses_same_control_session(tmp_path):
    from exploration_wildos.session_service import WildOSSessionServer
    from navigation.session_server import send_command
    from navigation.target import SemanticTargetRequest
    server=WildOSSessionServer('graph-session',port=0)
    try:
        server.live_map_url='http://127.0.0.1:12345/map.html'
        server.set_state('READY')
        descriptor=server.descriptor()
        assert descriptor['navigation_backend']=='wildos_graph_dijkstra'
        status=send_command(descriptor,'status')
        assert status['navigation_backend']=='wildos_graph_dijkstra'
        assert status['live_map_url']==server.live_map_url
        response=send_command(descriptor,'navigate',SemanticTargetRequest(instance_id='vehicle_02'))
        name,identity,request=server.commands.get(timeout=1)
        assert name=='navigate' and identity==response['job_id']
        assert request.instance_id=='vehicle_02'
    finally:
        server.close()


def test_motion_health_tick_does_not_project_live_maps(tmp_path):
    import time
    m,g,current,_=prepared()
    frame='session:PX4-raw:0';m.epoch=(frame,*m.epoch[1:])
    e=object.__new__(GraphExplorer)
    e.frontiers=m;e._next_moving_tick=0;e._next_moving_build=time.monotonic()+100
    e._moving_graph=None;e._grid=Mock(side_effect=AssertionError('Blocking projection in flight callback'))
    e.vehicle=SimpleNamespace(get_position=lambda:current)
    e.global_map=SimpleNamespace(revision=0,snapshot=Mock(side_effect=AssertionError('Snapshot in callback')))
    e.memory=SimpleNamespace(session_id='session');e.graph_directory=tmp_path;e._event=Mock()
    e._moving_graph_tick()
    e._grid.assert_not_called();e.global_map.snapshot.assert_not_called()
    assert m.graph['robot_attachment']['evidence']=='last_graph_snapshot_display_only'


def test_map_preparation_runs_in_background_not_submitter():
    import threading,copy
    from exploration_wildos.moving_graph import MovingGraph
    m,g,current,_=prepared()
    entered=threading.Event();release=threading.Event();threads=[]
    def snapshot():
        threads.append(threading.current_thread().name);entered.set()
        assert release.wait(10)
        return object()
    worker=MovingGraph()
    try:
        with patch.object(NavigationGrid,'from_global',return_value=g):
            worker.submit_from_maps(copy.deepcopy(m),SimpleNamespace(snapshot=snapshot),
                SimpleNamespace(snapshot=snapshot),current,'test-px4',[],g.altitude,g.config,None,100.)
            assert entered.wait(2)
            assert not worker.future.done()
            assert all(t.startswith('WildOSGraphWait') for t in threads)
            release.set()
            _,result=worker.future.result(timeout=30)
            assert result.graph is not None
            assert result.preparation_seconds>=0
    finally:
        release.set();worker.close()


def test_emergency_evidence_is_json_safe_without_disabling_stop(tmp_path):
    from planner.local_avoidance_planner import LocalPlan
    e=explorer(tmp_path)
    guard=e._guard
    guard.config=SimpleNamespace(emergency_stop_distance=.6,map_emergency_stop_distance=.4)
    guard.inner=SimpleNamespace(last_rejection=None)
    guard._diagnostic_context=(SimpleNamespace(timestamp=10.,forward_clearance=float('inf')),None,
                               np.array([0.,0.,-1.5]),np.array([1.,0.,-1.5]))
    plan=LocalPlan('emergency_stop',None,np.empty((0,3)),0.)
    assert guard._report(plan) is plan
    assert guard.last_plan_detail['emergency_evidence']['forward_clearance_m'] is None
    json.dumps(guard.last_plan_detail,allow_nan=False)
