import tempfile
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from mapping.live_loop_mapping import LiveLoopMapping
from mapping.loop_closure import PointCloudLoopClosure, PoseGraphEdge, _edge_residual, _pose_matrix, _pose_vector


class LoopMapTests(unittest.TestCase):
    def test_scalar_voxel_updates_preserve_clipping_order_and_capacity(self):
        from config.global_map_config import GlobalSparseMapConfig
        from mapping.global_sparse_occupancy_map import GlobalSparseOccupancyMap
        from mapping.explore_v3.global_sparse_occupancy_map import GlobalSparseOccupancyMap as V3
        from dataclasses import replace
        config = replace(GlobalSparseMapConfig(), max_voxels=2, visualize=False)
        indices = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]])
        for cls in (GlobalSparseOccupancyMap, V3):
            world = cls(config)
            expected = {}
            dropped = 0
            for delta in (-.4, 2., 20., -30., .3):
                for index in indices:
                    key = tuple(index)
                    if key not in expected and len(expected) >= 2:
                        dropped += 1
                        continue
                    expected[key] = float(np.clip(expected.get(key, 0.)+delta,
                                                  config.min_log_odds, config.max_log_odds))
                world._apply_indices(indices, delta)
                self.assertEqual(world._log_odds, expected)
                self.assertEqual(world._dropped_new_voxels, dropped)

    def test_analytic_edges_match_matrix_residual_and_numerical_derivatives(self):
        rng = np.random.default_rng(13)
        graph = object.__new__(PointCloudLoopClosure)
        for _ in range(20):
            poses = rng.normal(size=(2, 4))
            measurement = rng.normal(size=4)
            expected = _pose_vector(np.linalg.inv(_pose_matrix(poses[0])) @ _pose_matrix(poses[1]))-measurement
            expected[3] = (expected[3]+np.pi) % (2*np.pi)-np.pi
            np.testing.assert_allclose(_edge_residual(*poses, measurement), expected, atol=1e-12)
            graph.edges = [PoseGraphEdge(0, 1, measurement, 1., False)]
            edges, _, _ = graph._linearize(poses)
            for node, jac in enumerate(edges[0][2:4]):
                for axis in range(4):
                    plus, minus = poses.copy(), poses.copy()
                    plus[node, axis] += 1e-6
                    minus[node, axis] -= 1e-6
                    derivative = (_edge_residual(*plus, measurement)-_edge_residual(*minus, measurement))/2e-6
                    np.testing.assert_allclose(jac[:, axis], derivative, atol=1e-8)

    def test_geometry_reused_for_new_semantics_and_final_numbering(self):
        with tempfile.TemporaryDirectory() as root:
            history = Path(root)/'maps/semantic_observations.jsonl'
            history.parent.mkdir()
            history.write_text('{}\n')
            frames = [(np.zeros(3), np.zeros((1, 3)), 1.)]
            detector = SimpleNamespace(mapping_snapshot=(frames, []), config=Mock())
            rebuild = Mock(return_value=Mock())
            mapper = LiveLoopMapping(detector, root, [0, 0, 0], 0, rebuild_map=rebuild)
            with patch('mapping.live_loop_mapping.optimize_frames', return_value=([], [np.eye(4)])) as optimize, patch('mapping.live_loop_mapping.export_bundle', return_value=(Path('revision_test'), 1)) as export:
                mapper.update()
                history.write_text('{}\n{}\n')
                mapper.update()
                mapper.finalize()
                mapper.finalize()
                self.assertEqual(optimize.call_count, 1)
                self.assertEqual(rebuild.call_count, 1)
                self.assertEqual(export.call_count, 3)
                self.assertTrue(export.call_args.kwargs['number_ids'])

    def test_shutdown_finalizes_after_inputs_close_even_if_vehicle_close_fails(self):
        # Exercise the actual shutdown method without loading GPU perception
        # libraries or connecting simulator dependencies in this unit test.
        source = Path(__file__).resolve().parents[1]/'application/demo_runtime.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DemoRuntime')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        Runtime = type('Runtime', (), {'close': namespace['close']})
        runtime = Runtime()
        events = []
        runtime.vehicle = Mock()
        runtime.vehicle.close.side_effect = RuntimeError('landing failed')
        runtime.live_loop_mapping = Mock()
        runtime.live_loop_mapping.finalize.side_effect = lambda: events.append('final')
        runtime.lidar_worker = Mock()
        runtime.lidar_worker.close.side_effect = lambda: events.append('lidar')
        runtime.fastlio_recorder = None
        runtime.detection_started = True
        runtime.detection_worker = Mock()
        runtime.detection_worker.close.side_effect = lambda: events.append('semantics')
        runtime.global_mapping_started = runtime.mapping_started = runtime.recorder_started = runtime.hub_started = False
        runtime.data_stream = runtime.camera = runtime.astar_visualizer = Mock()
        runtime._save_alignment_report = runtime._report_results = Mock()
        with self.assertRaisesRegex(RuntimeError, 'landing failed'):
            runtime.close()
        self.assertEqual(events, ['lidar', 'semantics', 'final'])
