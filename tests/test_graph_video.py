import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from exploration_wildos.graph_video import GraphVideoRecorder


def drawing():
    grid = SimpleNamespace(known={(x,y) for x in range(5) for y in range(5)},
        free={(x,y) for x in range(1,4) for y in range(1,4)}, resolution=1., origin=[0.,0.])
    graph = dict(nodes=[dict(id=0, position_ned=[1.,1.,-1.5], is_frontier=False),
                        dict(id=1, position_ned=[3.,3.,-1.5], is_frontier=True)],
        edges=[dict(source=0,target=1)],frontiers=[dict(position_ned=[4.,3.,-1.5])],semantic_nodes=[])
    return SimpleNamespace(grid=grid), graph


def test_real_mp4_graph_updates_and_robot_motion_stop_writing_after_close(tmp_path):
    pose = np.array([1.,1.,0.])
    video = GraphVideoRecorder(tmp_path/'graph_exploration.mp4',lambda:pose.copy(),[0,0,0],0.,fps=10.,size=(640,480))
    video.start()
    a, g = drawing()
    video.publish(a,g,1)
    time.sleep(.25)
    pose[:] = [3.,3.,-1.5]
    g['nodes'].append(dict(id=2,position_ned=[2.,3.,-1.5],is_frontier=True))
    video.publish(a,g,2)
    time.sleep(.25)
    pose[2] = 0.
    video.close()
    assert video.error is None and not video.thread.is_alive()
    cap = cv2.VideoCapture(str(video.path))
    frames=[]
    while True:
        ok, frame=cap.read()
        if not ok:break
        frames.append(frame)
    cap.release()
    assert len(frames)>=4 and frames[0].shape==(480,640,3)
    assert not np.array_equal(frames[0][40:],frames[-1][40:])
    before=video.path.stat()
    video.publish(a,g,3)
    video.close()
    time.sleep(.2)
    after=video.path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert [p.name for p in tmp_path.iterdir()]==['graph_exploration.mp4']


def test_existing_video_cannot_be_overwritten(tmp_path):
    path=tmp_path/'graph_exploration.mp4';path.write_bytes(b'existing')
    video=GraphVideoRecorder(path,lambda:np.zeros(3),[0,0,0],0.)
    with pytest.raises(FileExistsError):video.start()
    assert path.read_bytes()==b'existing'
    video.close()


@pytest.mark.parametrize('failure', [False, True])
def test_session_video_starts_before_takeoff_and_closes_at_confirmed_landing(tmp_path,monkeypatch,failure):
    from test_wildos_live_explore import GraphExplorationSession
    import exploration_wildos.explore as module
    import exploration_wildos.graph_video as recording
    from config.exploration_config import FrontierExplorationConfig
    events=[]
    video=Mock()
    video.start.side_effect=lambda:events.append('video_start')
    video.close.side_effect=lambda:events.append('video_close')
    factory=Mock(return_value=video);monkeypatch.setattr(recording,'GraphVideoRecorder',factory)
    explorer=Mock(result={},phase='exploring')
    monkeypatch.setattr(module,'GraphExplorer',Mock(return_value=explorer))
    session=object.__new__(GraphExplorationSession)
    session.directory=tmp_path;session.home=np.zeros(3);session.home_yaw=0.
    session.config=FrontierExplorationConfig();session.memory=Mock(exploration={})
    session.runtime=SimpleNamespace(controller=SimpleNamespace(),global_sparse_map=Mock(),vehicle=Mock(),
        run_mission=Mock(),finish_exploration_recording=Mock(side_effect=lambda:events.append('finalize')))
    def flight():
        events.append('takeoff_explore_return_land')
        if failure:raise RuntimeError('flight failed')
    session.runtime.run_mission.side_effect=flight
    session._confirm_home=Mock(side_effect=lambda:events.append('land_confirmed'))
    session.perception=Mock();session.perception.semantic_snapshot.return_value=None
    session.memory.number_final_instances.return_value={}
    session._arguments=lambda *a:{};session.checkpoint=Mock();session.annotate=Mock()
    if failure:
        with pytest.raises(RuntimeError,match='flight failed'):session.explore()
        assert events==['video_start','takeoff_explore_return_land','video_close']
        session._confirm_home.assert_not_called()
    else:
        session.explore()
        assert events[:5]==['video_start','takeoff_explore_return_land','land_confirmed','video_close','finalize']
    assert factory.call_args.args[0]==tmp_path/'graph_exploration.mp4'
    assert explorer.graph_video is video
