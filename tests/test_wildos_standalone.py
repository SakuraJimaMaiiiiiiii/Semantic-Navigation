"""Independent startup and archive gates; all flight resources are doubles."""
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest

missing = {name:ModuleType(name) for name in ('PX4MavCtrlV4','ReqCopterSim','UE4CtrlAPI')
           if name not in sys.modules and importlib.util.find_spec(name) is None}
sys.modules.update(missing)
try:
    from exploration_wildos import standalone_navigation as standalone, saved_graph
finally:
    for name in missing: sys.modules.pop(name,None)
from config.semantic_navigation_config import SemanticNavigationConfig
from config.exploration_config import FrontierExplorationConfig
from navigation.exploration_session import TargetUnavailableError
import navigate_wildos_target as client


def saved_fixture(tmp_path, monkeypatch):
    directory = tmp_path/'exploration'
    update = directory/'navigation_graph/update_000001'
    update.mkdir(parents=True)
    (directory/'bundle.json').write_text('{}')
    config = SemanticNavigationConfig(clearance_radius=.1)
    graph = dict(metadata=dict(frame='NED'),incremental=dict(frame_key='old:PX4-raw:0'),
        nodes=[dict(id=i,stable_id=f'nav:{i}',position_ned=[i+.5,.5,-1.5]) for i in range(2)],
        edges=[dict(id=0,source=0,target=1,length=1.)],options=dict(edge_radius=8.))
    (update/'graph.json').write_text(json.dumps(graph))
    identity=dict(frame_key='old:PX4-raw:0',resolution=1.,origin=[0.,0.],altitude=-1.5,
                  navigation_config=asdict(config))
    (update/'incremental_state.json').write_text(json.dumps(dict(identity=identity)))
    cells=np.array([(x,y) for x in range(-3,5) for y in range(-3,5)])
    np.savez(update/'planning_grid.npz',known_indices=cells,free_indices=cells,
             occupied_indices=np.empty((0,2)),resolution=1.,origin=[0.,0.],altitude_ned=-1.5)
    archive=dict(session_id='old',home=np.zeros(3),initial_yaw=0.,records=[],
                 occupancy_map=dict(resolution=1.))
    monkeypatch.setattr(saved_graph,'read_archive',lambda p:(directory,archive))
    return directory,archive,update


@pytest.mark.parametrize('corruption',['frame','edge','occupied','height'])
def test_archive_rejects_mismatched_or_unsafe_graph(tmp_path,monkeypatch,corruption):
    source,archive,update=saved_fixture(tmp_path,monkeypatch)
    assert saved_graph.read_graph_archive(source)[2]['validation']['passed']
    graph=json.loads((update/'graph.json').read_text())
    if corruption=='frame':graph['incremental']['frame_key']='other:PX4-raw:0'
    if corruption=='edge':graph['edges'][0]['target']=100
    if corruption=='occupied':graph['nodes'][0]['position_ned']=[100.,100.,-1.5]
    if corruption=='height':graph['nodes'][0]['position_ned'][2]=-3.
    (update/'graph.json').write_text(json.dumps(graph))
    with pytest.raises(ValueError):saved_graph.read_graph_archive(source)


def test_default_cli_uses_saved_graph_without_old_session_file(tmp_path,monkeypatch):
    run=Mock(return_value=dict(success=True))
    monkeypatch.setattr(standalone,'latest_graph_archive',lambda:Path('saved'))
    monkeypatch.setattr(standalone,'run_standalone',run)
    monkeypatch.setattr(client,'command',Mock(side_effect=AssertionError('No old server')))
    monkeypatch.setattr(client,'DEFAULT_SESSION_FILE',tmp_path/'missing_session.json')
    client.main(['--no-open-map'])
    assert run.call_args.args==(Path('saved'),None)


def test_explicit_map_and_target_use_independent_mode(monkeypatch):
    run=Mock(return_value=dict(success=True));monkeypatch.setattr(standalone,'run_standalone',run)
    client.main(['--map','saved','--instance-id','vehicle_02','--no-open-map'])
    assert run.call_args.args[0]==Path('saved')
    assert run.call_args.args[1].instance_id=='vehicle_02'


def test_session_binds_verified_graph_and_refuses_revision_or_height_change(tmp_path,monkeypatch):
    source,archive,update=saved_fixture(tmp_path,monkeypatch)
    _,archive,saved=saved_graph.read_graph_archive(source)
    vehicle=Mock(config=NS(guided_z_tolerance=.15))
    vehicle.get_position.return_value=np.zeros(3);vehicle.get_euler.return_value=np.zeros(3)
    runtime=NS(vehicle=vehicle,run_output_dir=tmp_path/'new',detection_worker=Mock(),
               global_sparse_map=NS(revision=0),controller=Mock())
    with pytest.raises(ValueError,match='起点一致性'):
        standalone.IndependentGraphSession(runtime,FrontierExplorationConfig(),archive,saved,
                                           session_id='new',evidence={})
    session=standalone.IndependentGraphSession(runtime,FrontierExplorationConfig(),archive,saved,
        session_id='new',evidence=dict(same_origin_consistency_passed=True))
    assert session._validate_target_frame()['nodes'][0]['stable_id']=='nav:0'
    assert session.navigation_archive()['session_id']=='old'
    assert session.memory.session_id=='new'
    assert (session.directory/'navigation_graph/restored/graph.json').exists()
    with pytest.raises(TargetUnavailableError,match='高度'):
        session._target_grid(-3.,np.zeros(3))
    runtime.global_sparse_map.revision=1
    with pytest.raises(TargetUnavailableError,match='坐标系'):
        session._validate_target_frame()


def runner_fixture(tmp_path,monkeypatch):
    archive=dict(records=[],home=np.zeros(3),occupancy_map={})
    monkeypatch.setattr(standalone,'read_graph_archive',lambda p:(tmp_path,archive,{}))
    import navigation.target_navigation_config as settings
    monkeypatch.setattr(settings,'parse_config',lambda *a,**k:(NS(server_port=8877),NS(
        perception_backend='yolo_sam2',sam3_url='url',object_prompts=None)))
    server=Mock(session_id='new');monkeypatch.setattr(standalone,'WildOSSessionServer',Mock(return_value=server))
    runtime=Mock()
    runtime.vehicle.get_position.return_value=np.zeros(3)
    runtime.vehicle.get_euler.return_value=np.zeros(3)
    runtime.detection_worker.semantic_snapshot.side_effect=lambda:NS(completed_at=standalone.time.monotonic(),timestamp=1.)
    runtime.global_sparse_map.snapshot.return_value=NS(timestamp=1.)
    runtime.semantic_navigation_config=SemanticNavigationConfig()
    app=ModuleType('application');app.DemoRuntime=Mock(return_value=runtime)
    app.WebSocketRequestSender=Mock()
    monkeypatch.setitem(sys.modules,'application',app)
    @dataclass
    class Detector:
        backend:str='yolo_sam2'
        sam3_url:str='url'
    detection=ModuleType('detections');detection.DetectorConfig=Detector
    monkeypatch.setitem(sys.modules,'detections',detection)
    session=Mock(directory=tmp_path/'new',map_path=tmp_path/'new/map.html')
    construct=Mock(return_value=session);monkeypatch.setattr(standalone,'IndependentGraphSession',construct)
    serve=Mock();monkeypatch.setattr(standalone,'serve_target_commands',serve)
    import exploration_wildos.live_map as live
    viewer=Mock(url='http://localhost/map.html')
    monkeypatch.setattr(live,'LiveMapServer',Mock(return_value=viewer))
    return runtime,server,construct,serve,app


def test_startup_checks_frame_before_seeding_then_serves_without_exploration(tmp_path,monkeypatch):
    runtime,server,construct,serve,app=runner_fixture(tmp_path,monkeypatch)
    def verify(*args):
        runtime.global_sparse_map.seed_saved_map.assert_not_called()
        construct.assert_not_called()
        return dict(same_origin_consistency_passed=True)
    monkeypatch.setattr(standalone,'check_same_frame',verify)
    standalone.run_standalone(tmp_path,None,config_path=tmp_path/'config',open_map=False)
    runtime.global_sparse_map.seed_saved_map.assert_called_once()
    serve.assert_called_once_with(construct.return_value,server)
    construct.return_value.explore.assert_not_called()
    construct.return_value.navigate.assert_not_called()
    runtime.controller.run.assert_not_called()
    runtime.close.assert_called_once();server.close.assert_called_once()


def test_failed_frame_check_never_seeds_or_starts_target_session(tmp_path,monkeypatch):
    runtime,server,construct,serve,app=runner_fixture(tmp_path,monkeypatch)
    clock=[0.]
    monkeypatch.setattr(standalone.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(standalone.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+20.))
    monkeypatch.setattr(standalone,'check_same_frame',Mock(side_effect=ValueError('wrong scene')))
    with pytest.raises(RuntimeError,match='起飞前核对失败'):
        standalone.run_standalone(tmp_path,None,config_path=tmp_path/'config')
    runtime.global_sparse_map.seed_saved_map.assert_not_called()
    construct.assert_not_called();serve.assert_not_called();runtime.close.assert_called_once()


def test_busy_control_port_never_initializes_runtime(tmp_path,monkeypatch):
    import errno
    runtime,server,construct,serve,app=runner_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(standalone,'WildOSSessionServer',Mock(side_effect=OSError(errno.EADDRINUSE,'busy')))
    with pytest.raises(standalone.NavigationPortInUse):
        standalone.run_standalone(tmp_path,None,config_path=tmp_path/'config')
    app.DemoRuntime.assert_not_called();app.WebSocketRequestSender.assert_not_called()


@pytest.mark.parametrize('bad_archive',[False,True])
def test_invalid_archive_or_missing_target_stops_before_connection(tmp_path,monkeypatch,bad_archive):
    from navigation.target import SemanticTargetRequest
    runtime,server,construct,serve,app=runner_fixture(tmp_path,monkeypatch)
    if bad_archive:
        monkeypatch.setattr(standalone,'read_graph_archive',Mock(side_effect=ValueError('cache checksum mismatch')))
    with pytest.raises(ValueError):
        standalone.run_standalone(tmp_path,SemanticTargetRequest(instance_id='vehicle_99'),
                                  config_path=tmp_path/'config')
    standalone.WildOSSessionServer.assert_not_called()
    app.DemoRuntime.assert_not_called();app.WebSocketRequestSender.assert_not_called()
