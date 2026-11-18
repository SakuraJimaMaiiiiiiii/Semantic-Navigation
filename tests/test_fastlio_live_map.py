import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from localization.fastlio_live_map import FastLioLiveMap
from localization.fastlio_recording import FastLioRecorder


class LiveMapTests(unittest.TestCase):
    def test_publisher_only_renders_requested_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = FastLioLiveMap(tmp, 'snapshot_assumed')
            published = threading.Event()
            calls = []
            def publish():
                calls.append(1)
                published.set()
            mapping.publish = publish
            thread = threading.Thread(target=mapping._publish_loop)
            thread.start()
            try:
                mapping._publish_requested.set()
                self.assertTrue(published.wait(2.))
                time.sleep(.5)
                self.assertEqual(len(calls), 1)
            finally:
                mapping._publisher_stopped.set()
                mapping._publish_requested.set()
                thread.join(2.)
                self.assertFalse(thread.is_alive())

    def test_locked_html_retries_without_losing_tracking_or_final_map(self):
        from navigation.atomic_file import atomic_write_text
        with tempfile.TemporaryDirectory() as tmp:
            mapping = FastLioLiveMap(tmp, 'snapshot_assumed')
            mapping.alignment = np.eye(4)
            mapping.path = [[0., 0., 0.]]
            mapping.cells = {(1, 2, 3)}
            mapping.timestamp = 1.
            attempts = []
            def replace(path, text):
                attempts.append(path)
                if len(attempts) == 1:
                    raise PermissionError('sharing violation')
                atomic_write_text(path, text)
            with patch('navigation.atomic_file.atomic_write_text', side_effect=replace):
                mapping.publish()
                self.assertFalse(mapping._html_current)
                mapping.publish()
                mapping.close()
            self.assertTrue((Path(tmp)/'map.html').exists())
            self.assertTrue((Path(tmp)/'live_map.npz').exists())
            metadata = json.loads((Path(tmp)/'live_map.json').read_text())
            self.assertTrue(metadata['html_current'])
            self.assertEqual(metadata['html_publish_failures'], 1)

    def test_persistently_locked_html_preserves_final_data_and_reports_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = FastLioLiveMap(tmp, 'snapshot_assumed')
            mapping.alignment = np.eye(4)
            mapping.path = [[0., 0., 0.]]
            mapping.cells = {(1, 2, 3)}
            mapping.timestamp = 1.
            with patch('navigation.atomic_file.atomic_write_text', side_effect=PermissionError('locked')):
                mapping.close()
            self.assertTrue((Path(tmp)/'semantic_map.json').exists())
            self.assertTrue((Path(tmp)/'live_map.npz').exists())
            metadata = json.loads((Path(tmp)/'live_map.json').read_text())
            self.assertFalse(metadata['html_current'])
            self.assertEqual(metadata['html_publish_failures'], 1)

    def test_slow_html_publication_does_not_block_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = FastLioLiveMap(tmp, 'snapshot_assumed')
            entered, release = threading.Event(), threading.Event()
            original = mapping.publish
            def slow_publish():
                entered.set()
                if not release.wait(5.):
                    raise TimeoutError('publisher was not released')
                original()
            mapping.publish = slow_publish
            result = dict(status='tracking', timestamp=1., pose=np.eye(4), points_world=np.array([[1.,0.,0.]]))
            px4 = np.array([0.,0.,-1.,1.,0.,0.,0.])
            try:
                mapping.update(result, px4)
                self.assertTrue(entered.wait(5.))
                result['timestamp'] = 2.
                self.assertTrue(mapping.update(result, px4))
                self.assertEqual(len(mapping.path), 2)
            finally:
                release.set()
                mapping.close()

    def test_alignment_applied_once_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = FastLioLiveMap(tmp, 'snapshot_assumed')
            q = np.sqrt(.5)
            px4 = np.array([10., 20., -1., q, 0., 0., q])
            result = dict(status='tracking', timestamp=1., pose=np.eye(4),
                          points_world=np.array([[1., 0., 0.]]))
            self.assertTrue(mapping.update(result, px4))
            np.testing.assert_allclose(mapping.path[0], px4[:3])
            result['pose'][0, 3] = 1.
            result['timestamp'] = 2.
            px4[0] = 999.  # Never re-anchor to later PX4 drift.
            mapping.update(result, px4)
            np.testing.assert_allclose(mapping.path[-1], [10., 21., -1.])
            mapping.close()
            with np.load(Path(tmp)/'live_map.npz') as data:
                self.assertEqual(data['trajectory'].shape, (2, 3))
            metadata = json.loads((Path(tmp)/'live_map.json').read_text())
            self.assertFalse(metadata['controls_navigation'])
            self.assertEqual(metadata['scan_model'], 'snapshot_assumed')
            self.assertTrue((Path(tmp)/'map.html').exists())

    def test_uninitialized_output_not_mapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = FastLioLiveMap(tmp, 'snapshot')
            self.assertFalse(mapping.update({'status': 'initializing'}, None))
            mapping.close()
            self.assertFalse((Path(tmp)/'live_map.npz').exists())

    def test_initialization_failure_exposed(self):
        recorder = FastLioRecorder('unused', live_map=True, scan_model='snapshot_assumed')
        recorder.estimator_error = 'Bad IMU'
        with self.assertRaisesRegex(RuntimeError, 'Bad IMU'):
            recorder.wait_ready(.5)

    def test_saved_map_cli_rejects_unused_fastlio_options(self):
        from navigation.target_navigation_config import parse_config
        with self.assertRaises(SystemExit):
            parse_config(['--fastlio', 'live', '--lidar-scan-model', 'snapshot_assumed'], include_perception=True)
