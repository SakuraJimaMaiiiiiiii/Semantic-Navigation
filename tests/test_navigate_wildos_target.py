import json
from unittest.mock import Mock
import pytest
import navigate_wildos_target as client


def descriptor(tmp_path):
    path=tmp_path/'session.json'
    path.write_text(json.dumps(dict(navigation_backend='wildos_graph_dijkstra',host='127.0.0.1',port=8877)))
    return path


def test_target_client_pins_id_and_waits_for_result(tmp_path,monkeypatch,capsys):
    path=descriptor(tmp_path)
    send=Mock(side_effect=[dict(navigation_backend='wildos_graph_dijkstra',state='READY'),
                          dict(job_id='job'),dict(job=dict(id='job',state='succeeded',result=dict(success=True)))])
    monkeypatch.setattr(client,'command',send)
    client.main(['--session',str(path),'--instance-id','vehicle_02','--no-open-map'])
    assert send.call_args_list[1].args[1]=='navigate'
    assert send.call_args_list[1].args[2].instance_id=='vehicle_02'
    assert send.call_args_list[2].kwargs['job_id']=='job'
    assert capsys.readouterr().out == 'vehicle_02：导航成功\n'


def test_failed_target_client_prints_one_result_line(tmp_path,monkeypatch,capsys):
    send=Mock(side_effect=[dict(navigation_backend='wildos_graph_dijkstra',state='READY'),
        dict(job_id='job'),dict(job=dict(id='job',state='failed',result=dict(
            success=False,reason='no_reachable_saved_target',graph_routes=[{'node_path':['nav:157','nav:161']}])) )])
    monkeypatch.setattr(client,'command',send)
    with pytest.raises(SystemExit) as caught:
        client.main(['--session',str(descriptor(tmp_path)),'--instance-id','vehicle_05','--no-open-map'])
    assert caught.value.code == 1
    assert capsys.readouterr().out == 'vehicle_05：导航失败（no_reachable_saved_target）\n'


def test_non_wildos_session_cannot_submit(tmp_path,monkeypatch):
    path=tmp_path/'session.json';path.write_text('{}')
    send=Mock();monkeypatch.setattr(client,'command',send)
    with pytest.raises(SystemExit,match='不是 WildOS'):
        client.main(['--session',str(path),'--instance-id','vehicle_02'])
    send.assert_not_called()


def test_return_home_uses_existing_session(tmp_path,monkeypatch):
    path=descriptor(tmp_path)
    send=Mock(side_effect=[dict(navigation_backend='wildos_graph_dijkstra'),dict(job_id='home'),
                          dict(job=dict(id='home',state='succeeded',result=dict(success=True)))])
    monkeypatch.setattr(client,'command',send)
    client.main(['--session',str(path),'--return-home','--no-open-map'])
    assert send.call_args_list[1].args[1:]==('return_home',None)


def test_no_target_opens_interactive_client_after_backend_check(tmp_path, monkeypatch):
    from exploration_wildos import terminal_client
    send = Mock(return_value=dict(navigation_backend='wildos_graph_dijkstra',state='READY'))
    terminal = Mock()
    monkeypatch.setattr(client,'command',send)
    monkeypatch.setattr(terminal_client,'run_terminal',terminal)
    client.main(['--session',str(descriptor(tmp_path)),'--no-open-map'])
    assert send.call_count == 1 and send.call_args.args[1] == 'status'
    terminal.assert_called_once()
    assert terminal.call_args.args[1] is send


@pytest.mark.parametrize('failed', [False, True])
def test_interactive_targets_are_sequential_and_failure_stops_remaining(failed):
    from exploration_wildos.terminal_client import RemoteTerminalSession
    from navigation.terminal_commands import TerminalCommands
    calls = []
    def send(descriptor, name, target=None, job_id=None):
        calls.append((name,target.instance_id if target else None))
        if name == 'navigate':
            return dict(job_id=target.instance_id)
        return dict(job=dict(id=job_id,state='failed' if failed else 'succeeded',
                            result=dict(success=not failed)))
    lines = iter(['vehicle_03 vehicle_01'])
    def reader(prompt):
        try: return next(lines)
        except StopIteration: raise EOFError
    terminal = TerminalCommands(RemoteTerminalSession({},send,10),reader=reader,writer=Mock())
    terminal.stop.wait = lambda seconds: False
    terminal.run()
    assert calls == ([('navigate','vehicle_03'),('status',None)] if failed else
                     [('navigate','vehicle_03'),('status',None),('navigate','vehicle_01'),('status',None)])


def test_remote_interactive_timeout_does_not_submit_more_targets(monkeypatch):
    from exploration_wildos import terminal_client
    now = [0.]
    monkeypatch.setattr(terminal_client.time,'monotonic',lambda:now[0])
    send = Mock(side_effect=[dict(job_id='job'),dict(job=dict(id='job',state='running'))])
    session = terminal_client.RemoteTerminalSession({},send,1.)
    session.dispatch(dict(command='navigate',target=dict(instance_id='vehicle_02')))
    now[0] = 2.
    with pytest.raises(SystemExit,match='控制端任务可能仍在执行'):
        session.status('job')
    assert send.call_count == 2


def test_interactive_status_home_quit_use_existing_session():
    from exploration_wildos.terminal_client import RemoteTerminalSession
    from navigation.terminal_commands import TerminalCommands
    send = Mock(side_effect=[dict(state='READY'),dict(job_id='home'),
        dict(job=dict(id='home',state='succeeded',result=dict(success=True))),dict(job_id='quit')])
    lines = iter(['status','home','quit'])
    terminal = TerminalCommands(RemoteTerminalSession({},send,10),
                                reader=lambda prompt:next(lines),writer=Mock())
    terminal.stop.wait = lambda seconds:False
    terminal.run()
    assert [c.args[1] for c in send.call_args_list] == ['status','return_home','status','shutdown']
