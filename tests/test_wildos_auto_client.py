"""Automatic client selection uses a read-only, authenticated status probe."""
import socket
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

import navigate_wildos_target as client
from exploration_wildos import terminal_client
from exploration_wildos.session_service import WildOSSessionServer
from navigation import session_server


@pytest.fixture
def standalone_stub(monkeypatch):
    module = ModuleType('exploration_wildos.standalone_navigation')
    module.latest_graph_archive = Mock(return_value=Path('saved'))
    module.run_standalone = Mock(return_value={'success': True})
    module.SavedMapNotFound = FileNotFoundError
    module.NavigationPortInUse = RuntimeError
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def live_session(tmp_path, monkeypatch):
    server = WildOSSessionServer('auto-client-test', port=0)
    path = tmp_path / 'session.json'
    server.publish_descriptor(path)
    monkeypatch.setattr(client, 'DEFAULT_SESSION_FILE', path)
    server.set_state('READY')
    try:
        yield server
    finally:
        server.close()


@pytest.mark.parametrize('state', ['READY', 'EXPLORING', 'NAVIGATING'])
def test_default_connects_to_live_session_without_enqueuing_movement(
        live_session, standalone_stub, monkeypatch, state):
    live_session.set_state(state)
    terminal = Mock()
    monkeypatch.setattr(terminal_client, 'run_terminal', terminal)
    client.main(['--no-open-map'])
    terminal.assert_called_once_with(live_session.descriptor(), client.command, 900.)
    standalone_stub.run_standalone.assert_not_called()
    assert live_session.commands.empty()
    assert live_session.state == state


def test_target_without_mode_uses_existing_controller(
        live_session, standalone_stub, monkeypatch, capsys):
    send = Mock(side_effect=[{'job_id': 'job'}, {'job': {
        'id': 'job', 'state': 'succeeded', 'result': {'success': True},
    }}])
    monkeypatch.setattr(client, 'command', send)
    client.main(['--instance-id', 'vehicle_02', '--no-open-map'])
    assert send.call_args_list[0].args[1] == 'navigate'
    assert send.call_args_list[0].args[2].instance_id == 'vehicle_02'
    assert send.call_args_list[1].kwargs['job_id'] == 'job'
    standalone_stub.run_standalone.assert_not_called()
    assert capsys.readouterr().out == 'vehicle_02：导航成功\n'


def test_missing_session_starts_saved_map_mode(tmp_path, standalone_stub, monkeypatch):
    monkeypatch.setattr(client, 'DEFAULT_SESSION_FILE', tmp_path / 'missing.json')
    send = Mock()
    monkeypatch.setattr(session_server, 'send_command', send)
    client.main(['--no-open-map'])
    send.assert_not_called()
    assert standalone_stub.run_standalone.call_args.args == (Path('saved'), None)


def test_stopped_session_starts_saved_map_mode(live_session, standalone_stub, monkeypatch):
    send = Mock(side_effect=ConnectionRefusedError('stopped'))
    monkeypatch.setattr(session_server, 'send_command', send)
    client.main(['--no-open-map'])
    assert send.call_args.args[1] == 'status'
    assert standalone_stub.run_standalone.call_args.args == (Path('saved'), None)


@pytest.mark.parametrize('error', [socket.timeout('timeout'), ConnectionResetError('reset'),
                                 RuntimeError('Invalid session credentials'), ValueError('bad response')])
def test_uncertain_status_never_starts_second_controller(
        live_session, standalone_stub, monkeypatch, error):
    monkeypatch.setattr(session_server, 'send_command', Mock(side_effect=error))
    with pytest.raises(SystemExit) as caught:
        client.main(['--no-open-map'])
    assert caught.value.code == 2
    standalone_stub.run_standalone.assert_not_called()
    assert live_session.commands.empty()


def test_mismatched_response_session_is_rejected(live_session, standalone_stub, monkeypatch):
    monkeypatch.setattr(session_server, 'send_command', Mock(return_value={
        'navigation_backend': 'wildos_graph_dijkstra', 'session_id': 'other',
    }))
    with pytest.raises(SystemExit) as caught:
        client.main(['--no-open-map'])
    assert caught.value.code == 2
    standalone_stub.run_standalone.assert_not_called()


def test_invalid_session_file_never_starts_second_controller(tmp_path, standalone_stub, monkeypatch):
    path = tmp_path / 'invalid.json'
    path.write_text('{', encoding='utf-8')
    monkeypatch.setattr(client, 'DEFAULT_SESSION_FILE', path)
    with pytest.raises(SystemExit) as caught:
        client.main(['--no-open-map'])
    assert caught.value.code == 2
    standalone_stub.run_standalone.assert_not_called()


@pytest.mark.parametrize('options', [['--standalone'], ['--map', 'selected']])
def test_explicit_saved_map_mode_does_not_auto_connect(options, standalone_stub, monkeypatch):
    probe = Mock(side_effect=AssertionError('Explicit saved map mode'))
    monkeypatch.setattr(client, 'active_default_session', probe)
    client.main([*options, '--no-open-map'])
    probe.assert_not_called()
    expected = Path('selected') if '--map' in options else Path('saved')
    assert standalone_stub.run_standalone.call_args.args == (expected, None)


def test_legacy_client_flag_still_connects(live_session, standalone_stub, monkeypatch):
    terminal = Mock()
    monkeypatch.setattr(terminal_client, 'run_terminal', terminal)
    client.main(['--client', '--no-open-map'])
    terminal.assert_called_once()
    standalone_stub.run_standalone.assert_not_called()


def test_help_hides_client_flag(capsys):
    with pytest.raises(SystemExit) as caught:
        client.main(['--help'])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert '--client' not in output
    assert '--standalone' in output
