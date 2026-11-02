import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from navigation.standalone_navigation import check_same_frame, associate_live_records, read_archive
from mapping.global_sparse_occupancy_map import GlobalSparseOccupancyMap


class IndependentNavigationTests(unittest.TestCase):
    def test_standalone_does_not_read_old_session_descriptor(self):
        import navigate_target
        with patch('sys.argv', ['navigate_target.py', '--standalone', '--map', 'saved',
                                '--instance-id', 'vehicle_01', '--session', 'nonexistent.json']), \
             patch('navigation.standalone_navigation.run_standalone', return_value={'success': True}) as run:
            navigate_target.main()
        self.assertEqual(run.call_args.args[1].instance_id, 'vehicle_01')

    def test_no_arguments_starts_interactive_latest_map(self):
        import navigate_target
        from pathlib import Path
        with patch('sys.argv', ['navigate_target.py']), \
             patch('navigation.standalone_navigation.latest_archive', return_value=Path('saved')) as latest, \
             patch('navigation.standalone_navigation.run_standalone', return_value={'success': True}) as run:
            navigate_target.main()
        latest.assert_called_once()
        self.assertEqual(run.call_args.args, (Path('saved'), None))
        self.assertTrue(run.call_args.kwargs['open_map'])

    def test_latest_skips_invalid_and_navigation_only_runs(self):
        import tempfile
        from pathlib import Path
        from navigation.standalone_navigation import latest_archive
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['run_20261026_100000', 'run_20261026_110000', 'run_20261026_120000']:
                (root/name/'exploration').mkdir(parents=True)
            for name in ['run_20261026_100000', 'run_20261026_110000']:
                (root/name/'exploration'/'bundle.json').write_text('{}')
            def load(directory):
                if '110000' in str(directory): raise ValueError('checksum mismatch')
                return directory, {'records':[{}]}
            with patch('navigation.standalone_navigation.read_archive', side_effect=load):
                self.assertEqual(latest_archive(root), (root/'run_20261026_100000'/'exploration').resolve())
            with patch('navigation.standalone_navigation.read_archive', side_effect=ValueError('unfinished')):
                with self.assertRaises(FileNotFoundError): latest_archive(root)

    def record(self, identity, x=0):
        return {'runtime_instance_id': identity, 'class_name': 'car',
                'position_world': [x, 0, 0],
                'bbox_3d': {'min': [x-1, -1, -1], 'max': [x+1, 1, 1]}}

    def test_new_id_is_matched_by_geometry(self):
        saved = [self.record('vehicle_01'), self.record('vehicle_02', 6)]
        result = associate_live_records(saved, [self.record('vehicle_02', .1)])
        self.assertEqual(result[0]['runtime_instance_id'], 'vehicle_01')
        self.assertEqual(result[0]['live_instance_id'], 'vehicle_02')

    def test_matching_number_does_not_override_location(self):
        self.assertEqual(associate_live_records([self.record('vehicle_01')],
                                               [self.record('vehicle_01', 8)]), [])

    def test_ambiguous_and_fragmented_observations_rejected(self):
        self.assertEqual(associate_live_records([self.record('a'), self.record('b', .2)],
                                               [self.record('new', .1)]), [])
        self.assertEqual(associate_live_records([self.record('a')],
                                               [self.record('n1', .1), self.record('n2', -.1)]), [])

    def test_geometry_gate_and_wrong_origin(self):
        indices = np.array([[x, y, 0] for x in range(15) for y in range(15)])
        saved = {'indices': indices, 'log_odds': np.ones(len(indices)),
                 'occupied_threshold': .5, 'resolution': .3}
        archive = {'home': np.zeros(3), 'initial_yaw': 0., 'occupancy_map': saved}
        points = (indices + .5) * .3
        live = SimpleNamespace(occupied_points_ned=points)
        self.assertEqual(check_same_frame(archive, np.zeros(3), 0, live)['matched_fraction'], 1)
        with self.assertRaises(ValueError):
            check_same_frame(archive, np.ones(3), 0, live)
        with self.assertRaises(ValueError):
            check_same_frame(archive, np.zeros(3), 0, SimpleNamespace(occupied_points_ned=points+10))

    def test_seed_preserves_live_evidence_and_timestamp(self):
        world = GlobalSparseOccupancyMap()
        with world._lock:
            world._log_odds[(1, 2, 3)] = -2.
            world._timestamp = 123.
        cfg = world.config
        saved = {'indices': np.array([[1, 2, 3], [4, 5, 6]]), 'log_odds': np.array([2., 2.]),
                 'resolution': world.resolution, 'occupied_threshold': cfg.occupied_threshold,
                 'free_threshold': cfg.free_threshold}
        world.seed_saved_map(saved)
        snap = world.snapshot()
        values = dict(zip(map(tuple, snap.indices), snap.log_odds))
        self.assertEqual(values[(1, 2, 3)], -2)
        self.assertEqual(values[(4, 5, 6)], 2)
        self.assertEqual(snap.timestamp, 123)
        with self.assertRaises(ValueError):
            world.seed_saved_map({**saved, 'resolution': world.resolution*2})


if __name__ == '__main__':
    unittest.main()
