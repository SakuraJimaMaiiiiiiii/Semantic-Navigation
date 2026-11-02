import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import DEFAULT, Mock, patch

import numpy as np

from application.demo_runtime import DemoRuntime
from config.occupancy_grid_config import OccupancyGridConfig
from config.global_map_config import GlobalSparseMapConfig
from config.planner_config import PlannerConfig
from config.lidar_config import LidarConfig
from detections.object_detect import DetectorConfig, SemanticMaskProcessor
from mapping.explore_v3 import adapters as v3
from mapping.local_occupancy_grid import LocalOccupancyGrid
from mapping.global_sparse_occupancy_map import GlobalSparseOccupancyMap
from mapping.semantic_instance_mapper import SemanticInstanceMapper
from mapping.vehicle_instance_tracker import VehicleInstanceTracker


class ExploreV3BackendTests(unittest.TestCase):
    def test_algorithms_match_manifest(self):
        root = Path('mapping/explore_v3')
        manifest = json.loads((root/'source_manifest.json').read_text())
        self.assertEqual(manifest['commit'], '599499a')
        for name, original in manifest['sha256'].items():
            digest = manifest.get('modified_sha256', {}).get(name, original)
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), digest, name)

    def test_sam3_processor_accepts_restored_vehicle_mapper_without_loading_a_model(self):
        config = DetectorConfig(backend='sam3')
        processor = SemanticMaskProcessor(config, mapper_type=v3.SemanticInstanceMapper)
        self.assertEqual(processor.config.backend, 'sam3')
        self.assertIs(type(processor.mapper._vehicles), VehicleInstanceTracker)
        self.assertIs(type(processor.mapper), v3.SemanticInstanceMapper)
        current = SemanticMaskProcessor(config)
        self.assertIs(type(current.mapper), SemanticInstanceMapper)

    def test_map_adapter_keeps_v3_raycast_results(self):
        from mapping.explore_v3.local_occupancy_grid import LocalOccupancyGrid as Original
        config = OccupancyGridConfig(size_ned=(4.,4.,3.))
        original, adapted = Original(config), v3.LocalOccupancyGrid(config)
        origin = np.array([0.,0.,-1.])
        points = np.array([[1.,0.,-1.], [1.,.3,-1.], [0.,1.,-1.]])
        for timestamp in (1.,2.):
            for grid in (original, adapted):
                grid.integrate_lidar_points(origin, points, timestamp, origin)
        np.testing.assert_array_equal(original.snapshot().states, adapted.execution_snapshot.states)
        np.testing.assert_array_equal(original.snapshot().observed, adapted.execution_snapshot.observed)
        self.assertEqual(adapted.execution_snapshot.timestamp, 2.)


    def test_exploration_alone_selects_legacy_maps_and_vehicle_adapter(self):
        for exploring in (False, True):
            with self.subTest(exploring=exploring), tempfile.TemporaryDirectory() as folder:
                runtime = object.__new__(DemoRuntime)
                runtime.exploration_mode = exploring
                runtime.recording_config = SimpleNamespace(enabled=False)
                runtime.mapping_config = OccupancyGridConfig(size_ned=(4.,4.,3.))
                runtime.global_map_config = GlobalSparseMapConfig(enabled=True, visualize=False)
                runtime.planner_config = PlannerConfig()
                runtime.drone_config = SimpleNamespace(guided_max_horizontal_acceleration=.3)
                runtime.astar_visualization_config = Mock()
                runtime.detector_config = DetectorConfig(backend='sam3')
                runtime.lidar_config = LidarConfig()
                runtime.run_output_dir = Path(folder)
                runtime.semantic_target = None
                runtime.mission_config = Mock()
                runtime.vehicle = Mock()
                runtime.camera = Mock()
                runtime.data_stream = Mock()
                for name in ('recorder_source','mapping_source','global_mapping_source','detection_source'):
                    setattr(runtime,name,Mock())
                runtime._local_planner_type = Mock(return_value=Mock())
                runtime._create_global_visualizer = Mock(return_value=None)
                with patch.multiple('application.demo_runtime', AStar2DVisualizer=DEFAULT,
                                    GlobalReturnPlanner=DEFAULT, RflyMissionController=DEFAULT,
                                    LocalMappingWorker=DEFAULT, GlobalMappingWorker=DEFAULT,
                                    RealtimeDetectionWorker=DEFAULT) as factories:
                    runtime._create_processing()
                self.assertIs(type(runtime.occupancy_grid), v3.LocalOccupancyGrid if exploring else LocalOccupancyGrid)
                self.assertIs(type(runtime.global_sparse_map), v3.GlobalSparseOccupancyMap if exploring else GlobalSparseOccupancyMap)
                options = factories['RealtimeDetectionWorker'].call_args.kwargs
                if exploring:
                    self.assertIs(options['mapper_type'], v3.SemanticInstanceMapper)
                    self.assertEqual(type(runtime.lidar_worker).__module__, 'mapping.explore_v3.lidar_mapping_worker')
                else:
                    self.assertNotIn('mapper_type', options)
                    self.assertEqual(type(runtime.lidar_worker).__module__, 'mapping.lidar_mapping_worker')

    def test_fastlio_selection_for_target_navigation_is_unchanged(self):
        names = ('_create_run_output_directory','_create_configs','_create_core',
                 '_create_subscribers','_create_processing','_reset_started_flags')
        with patch.multiple(DemoRuntime, **{name: DEFAULT for name in names}):
            explore = DemoRuntime(exploration=True, fastlio_mode='live')
            target = DemoRuntime(fastlio_mode='live')
        self.assertEqual(explore.fastlio_mode, 'live')
        self.assertEqual(target.fastlio_mode, 'live')
