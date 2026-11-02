import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from run_explore import parse_config
import numpy as np
from mapping.semantic_instance_mapper import SemanticInstanceMapper, SemanticObservation3D


class StartupConfigTests(unittest.TestCase):
    def test_default_explore_columns_publish_only_structural_geometry(self):
        _, args = parse_config([], include_perception=True)
        self.assertIn('parking garage structural column', args.object_prompts)
        mapper = SemanticInstanceMapper(landmark_classes=tuple(args.object_prompts))
        names = ['parking garage structural column'] * 2
        observations = []
        for x, height in [(0., 2.6), (5., .8)]:
            low = np.array([x, 0., -height])
            high = np.array([x+.5, .5, 0.])
            observations.append(SemanticObservation3D((low+high)/2, low, high, np.eye(3)*.01))
        for stamp in (1., 2., 3.):
            mapper.update(names, [.8, .8], observations, stamp)
        records = mapper.semantic_map_records(min_observations=3)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['class_name'], 'column')

    def parse_file(self, values, extra=()):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'explore.json'
            path.write_text(json.dumps(values), encoding='utf-8')
            return parse_config(['--config', str(path), *extra], include_perception=True)

    def test_file_and_cli_precedence(self):
        config, args = self.parse_file(
            dict(perception_backend='sam3', fastlio='live', lidar_scan_model='snapshot_assumed',
                 max_radius=42, object_prompts=['car', 'fire hose cabinet']),
            ['--sam2', '--fastlio', 'off', '--max-radius', '12'])
        self.assertEqual(config.max_radius, 12)
        self.assertEqual(args.perception_backend, 'yolo_sam2')
        self.assertIsNone(args.fastlio)
        self.assertEqual(args.object_prompts, ['car', 'fire hose cabinet'])

    def test_empty_file_uses_original_defaults(self):
        config, args = self.parse_file({})
        self.assertIsNone(args.fastlio)
        self.assertEqual(args.perception_backend, 'yolo_sam2')
        self.assertEqual(config.server_port, 8877)

    def test_bad_values_fail_before_start(self):
        for value in [dict(fastlio='wrong'), dict(max_radius=True),
                      dict(port=1.2), dict(object_prompts='car'), dict(typo=1),
                      dict(perception_backend='sam4'), dict(timeout=None)]:
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse_file(value)

    def test_missing_file_does_not_silently_fall_back(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_config(['--config', '/missing/explore.json'])
