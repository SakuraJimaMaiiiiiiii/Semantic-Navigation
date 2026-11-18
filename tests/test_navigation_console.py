import io
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from navigation.console_output import quiet_navigation, navigation_result_text
from navigation.mission import SemanticNavigator
from navigation.perception_mode import active_perception
from navigation.terminal_commands import TerminalCommands
from navigation.terminal_progress import MotionProgress


def test_route_and_approach_are_recorded_without_console_output(capsys):
    stream = io.StringIO()
    progress = MotionProgress(stream)
    vehicle = SimpleNamespace(_motion_progress_display=progress)
    navigator = SimpleNamespace(vehicle=vehicle, events=[], clock=lambda:2., started=0., state=None)
    route = dict(target_id='vehicle_05', node_path=['nav:157', 'nav:161'],
                 points_ned=[[-8.,36.,-1.5], [4.,29.,-1.5]])
    class Session:
        runtime = SimpleNamespace(vehicle=vehicle)
        @active_perception
        def navigate(self):
            SemanticNavigator._event(navigator, 'WILDOS_TARGET_GRAPH_ROUTE', **route)
            SemanticNavigator._event(navigator, 'APPROACH', target_id='vehicle_05', goal_ned=[4.,29.,-1.5])
            progress.update('route_clear')
            return {'success':True}
    assert Session().navigate()['success']
    assert capsys.readouterr().out == ''
    assert stream.getvalue() == ''
    report = json.loads(json.dumps(navigator.events))
    assert report[0]['node_path'] == route['node_path']
    assert report[0]['points_ned'] == route['points_ned']
    assert report[1]['state'] == 'APPROACH'
    assert progress.enabled


def test_quiet_scope_preserves_background_messages_and_stderr(capsys):
    original = sys.stdout
    with quiet_navigation():
        print('foreground details')
        thread = threading.Thread(target=lambda: print('background warning'))
        thread.start(); thread.join()
        print('critical error', file=sys.stderr)
    assert sys.stdout is original
    print('normal output')
    captured = capsys.readouterr()
    assert captured.out == 'background warning\nnormal output\n'
    assert captured.err == 'critical error\n'


def test_quiet_scope_restores_stream_and_progress_after_failure(capsys):
    stream = io.StringIO()
    progress = MotionProgress(stream)
    original = sys.stdout
    with pytest.raises(RuntimeError):
        with quiet_navigation(SimpleNamespace(_motion_progress_display=progress)):
            print('internal details')
            raise RuntimeError('navigation failed')
    assert sys.stdout is original and progress.enabled
    progress.update('route_clear')
    assert '任务进行中' in stream.getvalue()
    assert capsys.readouterr().out == ''


def test_progress_stream_created_in_quiet_scope_works_afterwards(capsys):
    with quiet_navigation():
        progress = MotionProgress()
        progress.update('route_clear')
    progress.update('ego_clear')
    assert '任务进行中' in capsys.readouterr().out


@pytest.mark.parametrize('success', [True, False])
def test_interactive_targets_emit_only_one_result_per_attempt(success):
    class Server:
        def __init__(self): self.calls=[]
        def descriptor(self): return {}
        def dispatch(self, command):
            self.calls.append(command)
            return {'job_id':str(len(self.calls))}
        def status(self, identity):
            return {'job':dict(id=identity,state='succeeded' if success else 'failed',result=dict(
                success=success,reason='unreachable',graph_routes=[{'points_ned':[[1,2,3]]}]))}
    lines = iter(['vehicle_05 vehicle_06'])
    def reader(prompt):
        try:return next(lines)
        except StopIteration:raise EOFError
    server, output = Server(), []
    console = TerminalCommands(server,reader=reader,writer=output.append)
    console.stop.wait = lambda seconds:False
    console.run()
    expected = ['vehicle_05：导航成功','vehicle_06：导航成功'] if success else ['vehicle_05：导航失败（unreachable）']
    assert output == expected
    assert len(server.calls) == len(expected)


def test_result_summary_does_not_dump_paths_or_multiline_errors():
    result = dict(success=False,reason='failed\nwith details',points_ned=[[1,2,3]],node_path=['nav:157'])
    assert navigation_result_text(result,'vehicle_05') == 'vehicle_05：导航失败（failed with details）'
    assert navigation_result_text({'success':True},command='return_home') == '返航成功'
