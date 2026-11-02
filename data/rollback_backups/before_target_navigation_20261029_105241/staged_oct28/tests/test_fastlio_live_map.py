import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from localization.fastlio_live_map import FastLioLiveMap
from localization.fastlio_recording import FastLioRecorder


class LiveMapTests(unittest.TestCase):
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

    def test_cli_live_assumption_is_explicit(self):
        from run_explore import parse_config
        _, args = parse_config(['--fastlio', 'live', '--lidar-scan-model',
                                'snapshot_assumed'], include_perception=True)
        self.assertEqual(args.fastlio, 'live')
        with self.assertRaises(SystemExit):
            parse_config(['--fastlio', 'live', '--lidar-scan-model', 'unverified'], include_perception=True)
