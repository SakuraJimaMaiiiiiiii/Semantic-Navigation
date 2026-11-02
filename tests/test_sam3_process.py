"""Service ownership and failed-start cleanup without loading GPU models."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import sys
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'sam3_process', Path(__file__).resolve().parents[1] / 'detections/sam3_process.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServiceTests(unittest.TestCase):
    def test_external_service_is_not_stopped(self):
        service = module.SAM3ServiceProcess('http://127.0.0.1:8878')
        with patch.object(service, 'ready', return_value=True), patch.object(module.subprocess, 'Popen') as spawn:
            service.start()
            service.close()
            spawn.assert_not_called()

    def test_owned_service_uses_current_python_and_stops(self):
        service = module.SAM3ServiceProcess('http://127.0.0.1:8879')
        process = Mock()
        process.poll.return_value = None
        with patch.object(module, 'ensure_checkpoint', return_value=Path('models/sam3.pt')), patch.object(service, 'ready', side_effect=[False, True]), patch.object(
                module.subprocess, 'Popen', return_value=process) as spawn:
            service.start()
            self.assertEqual(spawn.call_args.args[0][0], module.sys.executable)
            self.assertEqual(spawn.call_args.args[0][-1], '8879')
            command = spawn.call_args.args[0]
            self.assertEqual(command[command.index('--checkpoint') + 1], str(Path('models/sam3.pt')))
            service.close()
            service.close()
            process.terminate.assert_called_once()
            process.wait.assert_called_once()

    def test_failed_start_does_not_wait_for_timeout(self):
        service = module.SAM3ServiceProcess('http://127.0.0.1:8878')
        process = Mock()
        process.poll.return_value = 1
        with patch.object(module, 'ensure_checkpoint'), patch.object(service, 'ready', return_value=False), patch.object(
                module.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                service.start()
        self.assertIsNone(service.process)

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            module.SAM3ServiceProcess('http://example.com:8878')

    def test_download_once_into_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'detections/Yolomode/sam3.pt'
            def download(**kwargs):
                self.assertEqual(kwargs['repo_id'], 'facebook/sam3')
                self.assertEqual(Path(kwargs['local_dir']), target.parent)
                target.parent.mkdir(parents=True)
                target.write_bytes(b'checkpoint')
            hub = Mock()
            hub.hf_hub_download.side_effect = download
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                self.assertEqual(module.ensure_checkpoint(root), target)
                self.assertEqual(module.ensure_checkpoint(root), target)
            hub.hf_hub_download.assert_called_once()

    def test_download_failure_explains_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Mock()
            hub.hf_hub_download.side_effect = OSError('network unavailable')
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                with self.assertRaisesRegex(RuntimeError, 'hf auth login'):
                    module.ensure_checkpoint(Path(directory))
