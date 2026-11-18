"""A completed exploration releases its runtime rather than silently waiting forever."""
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('keep', [False, True])
def test_wildos_landed_exploration_closes_owned_service(tmp_path, monkeypatch, keep):
    import exploration_wildos.explore as module
    import exploration_wildos.session_service as services
    import exploration_wildos.live_map as viewers
    import application
    import detections.sam3_process as sam3
    from config.exploration_config import FrontierExplorationConfig

    events = []
    server = Mock(session_id='test')
    server.close.side_effect = lambda: events.append('control_port_released')
    server.commands = Mock()
    if keep:
        commands = queue.Queue()
        commands.put(('shutdown', None, None))
        server.commands = commands
    else:
        server.commands.get.side_effect = AssertionError('Completed exploration must not wait for commands')
    runtime, service, viewer = Mock(), Mock(), Mock(url='http://127.0.0.1/map.html')
    runtime.close.side_effect = lambda: events.append('runtime_closed')
    service.close.side_effect = lambda: events.append('sam3_closed')
    session = Mock(directory=tmp_path, map_path=tmp_path/'map.html', airborne_navigation=False)
    session.explore.side_effect = lambda: events.append('landed_and_map_saved')
    monkeypatch.setattr(module, '_CLI', (FrontierExplorationConfig(), SimpleNamespace(
        perception_backend='sam3', fastlio=False, keep_session=keep)))
    monkeypatch.setattr(services, 'WildOSSessionServer', Mock(return_value=server))
    monkeypatch.setattr(application, 'DemoRuntime', Mock(return_value=runtime))
    monkeypatch.setattr(application, 'WebSocketRequestSender', Mock())
    monkeypatch.setattr(sam3, 'SAM3ServiceProcess', Mock(return_value=service))
    monkeypatch.setattr(module, 'GraphExplorationSession', Mock(return_value=session))
    monkeypatch.setattr(viewers, 'LiveMapServer', Mock(return_value=viewer))
    monkeypatch.setattr(module, 'open_map_page', Mock())
    module.main()
    assert events == ['landed_and_map_saved', 'runtime_closed', 'sam3_closed', 'control_port_released']
    assert runtime.keep_session is keep
    runtime.close.assert_called_once()
    service.close.assert_called_once()
    viewer.close.assert_called_once()
    if not keep:
        server.commands.get.assert_not_called()
