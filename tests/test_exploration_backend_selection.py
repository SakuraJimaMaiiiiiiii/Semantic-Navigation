import contextlib
import io
import queue
import unittest
from unittest.mock import Mock, patch

import run_explore


class ExplorationBackendSelectionTests(unittest.TestCase):
    def test_default_and_explicit_backends(self):
        for argv, expected in (([], 'sam3'), (['--sam3'], 'sam3'), (['--sam2'], 'yolo_sam2')):
            with self.subTest(argv=argv):
                _, args = run_explore.parse_config(argv, include_perception=True)
                self.assertEqual(args.perception_backend, expected)
                self.assertFalse(args.fastlio)
        self.assertTrue(run_explore.parse_config(['--fastlio'], include_perception=True)[1].fastlio)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_explore.parse_config(['--sam2', '--sam3'])

    def test_selected_backend_is_started_and_cleaned_up(self):
        for argv, backend in (([], 'sam3'), (['--sam3'], 'sam3'), (['--sam2'], 'yolo_sam2'),
                              (['--sam3', '--fastlio'], 'sam3')):
            with self.subTest(argv=argv):
                server = Mock()
                server.commands = queue.Queue()
                server.commands.put(('shutdown', None, None))
                runtime, session, service = Mock(), Mock(), Mock()
                session.map_path.resolve.return_value.as_uri.return_value = 'file:///map.html'
                session.directory = __import__('pathlib').Path('exploration')
                session.airborne_navigation = False
                with patch('sys.argv', ['run_explore.py', *argv]), \
                     patch('navigation.session_server.SessionServer', return_value=server), \
                     patch('navigation.exploration_session.ExplorationSession', return_value=session), \
                     patch('application.DemoRuntime', return_value=runtime) as runtime_factory, \
                     patch('application.WebSocketRequestSender'), \
                     patch('detections.sam3_process.SAM3ServiceProcess', return_value=service), \
                     patch('webbrowser.open', return_value=True):
                    run_explore.main()
                self.assertEqual(runtime_factory.call_args.kwargs['detector_config'].backend, backend)
                self.assertEqual(runtime_factory.call_args.kwargs['fastlio_mode'],
                                 'live' if '--fastlio' in argv else None)
                self.assertEqual(service.start.call_count, int(backend == 'sam3'))
                self.assertEqual(service.close.call_count, int(backend == 'sam3'))
                session.explore.assert_called_once()
                runtime.close.assert_called_once()

    def test_failed_sam3_start_releases_control_port(self):
        server, service = Mock(), Mock()
        service.start.side_effect = RuntimeError('model failed')
        with patch('sys.argv', ['run_explore.py']), \
             patch('navigation.session_server.SessionServer', return_value=server), \
             patch('detections.sam3_process.SAM3ServiceProcess', return_value=service), \
             patch('application.WebSocketRequestSender') as sender:
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                run_explore.main()
        server.close.assert_called_once()
        service.close.assert_called_once()
        sender.assert_not_called()


if __name__ == '__main__':
    unittest.main()
