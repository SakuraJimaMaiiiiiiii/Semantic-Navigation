from unittest.mock import Mock

import pytest

from navigation.terminal_commands import TerminalCommands, parse_line


IDS = ['vehicle_01','vehicle_06','vehicle_05','vehicle_03','vehicle_04','vehicle_02']


@pytest.mark.parametrize('suffix', ['home', 'HOME', 'return', '返航'])
def test_user_sequence_supports_a_final_return(suffix):
    commands = parse_line(' '.join([*IDS, suffix]))
    assert [c['target']['instance_id'] for c in commands[:-1]] == IDS
    assert commands[-1] == {'command':'return_home'}


@pytest.mark.parametrize('line', ['home vehicle_01', 'vehicle_01 home vehicle_02',
    'vehicle_01 home home', 'vehicle_01 quit', 'vehicle_01 status home', 'vehicle_01 help'])
def test_invalid_mixed_commands_are_rejected_before_dispatch(line):
    with pytest.raises(ValueError):
        parse_line(line)


def terminal_for(server, line, output):
    lines = iter([line])
    def read(prompt):
        try:return next(lines)
        except StopIteration:raise EOFError
    terminal = TerminalCommands(server,reader=read,writer=output.append)
    terminal.stop.wait = lambda seconds:False
    return terminal


@pytest.mark.parametrize('failure', [None, 'vehicle_06'])
def test_batch_runs_in_order_and_preserves_requested_return_after_known_failure(failure):
    calls, output, jobs = [], [], {}
    class Server:
        def descriptor(self):return {}
        def dispatch(self, command):
            target = command.get('target',{}).get('instance_id')
            calls.append((command['command'],target))
            success = target != failure if target is not None else True
            identity = str(len(calls))
            jobs[identity] = dict(id=identity,state='succeeded' if success else 'failed',
                                 result=dict(success=success,reason='unreachable'))
            return {'job_id':identity}
        def status(self, identity):return {'job':jobs[identity]}
    terminal_for(Server(),' '.join([*IDS,'home']),output).run()
    expected = IDS if failure is None else IDS[:2]
    assert calls == [('navigate',identity) for identity in expected] + [('return_home',None)]
    assert len(output) == len(expected)+1
    assert output[-1] == '返航成功'
    if failure:
        assert output[-2] == 'vehicle_06：导航失败（unreachable）'


def test_unknown_in_flight_state_never_dispatches_return():
    server = Mock()
    server.descriptor.return_value = {}
    server.dispatch.return_value = {'job_id':'running'}
    server.status.side_effect = SystemExit('task may still be executing')
    with pytest.raises(SystemExit):
        terminal_for(server,'vehicle_01 vehicle_06 home',[]).run()
    server.dispatch.assert_called_once_with({'command':'navigate','target':{'instance_id':'vehicle_01'}})


@pytest.mark.parametrize('failure', [None, 'vehicle_06'])
def test_remote_client_sends_targets_then_return_to_existing_controller(failure):
    from exploration_wildos.terminal_client import RemoteTerminalSession
    calls, jobs, output = [], {}, []
    def send(descriptor, command, target=None, job_id=None):
        calls.append((command, target.instance_id if target else None))
        if command == 'status':
            return {'job':jobs[job_id]}
        identity = target.instance_id if target else 'home'
        success = identity != failure
        jobs[identity] = dict(id=identity,state='succeeded' if success else 'failed',
                             result=dict(success=success,reason='unreachable'))
        return {'job_id':identity}
    remote = RemoteTerminalSession({},send,10.)
    terminal_for(remote,' '.join([*IDS,'home']),output).run()
    dispatched = [(name,target) for name,target in calls if name != 'status']
    expected = IDS if failure is None else IDS[:2]
    assert dispatched == [('navigate',identity) for identity in expected] + [('return_home',None)]
    assert output[-1] == '返航成功'


def test_rejected_return_does_not_kill_terminal_or_force_a_move():
    calls, output = [], []
    class Server:
        def descriptor(self):return {}
        def dispatch(self, command):
            calls.append(command['command'])
            if command['command'] == 'return_home':
                raise RuntimeError('Session is ERROR; vehicle already recovering')
            return {'job_id':'failed'}
        def status(self, identity):
            return {'job':dict(id=identity,state='failed',result=dict(success=False,reason='flight_fault'))}
    terminal_for(Server(),'vehicle_01 vehicle_06 home',output).run()
    assert calls == ['navigate','return_home']
    assert 'Session is ERROR' in output[-1]
