from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from mapping.semantic_numbering import number_exported_instances
from mapping.semantic_voxels import SemanticVoxelSnapshot, project_semantic_voxels
from mapping.live_loop_mapping import LiveLoopMapping
from navigation.scene_memory import SceneMemory


class SemanticNumberingTests(unittest.TestCase):
    def records(self):
        return [dict(instance_id=identity, runtime_instance_id=identity, video_instance_id=identity,
                     merged_from=['vehicle_02'] if identity == 'vehicle_04' else [],
                     class_name=category, position_world=[i, 0., -1.], aliases=[],
                     confidence=.8, observation_count=10, last_seen=1.)
                for i, (identity, category) in enumerate([
                    ('vehicle_04', 'car'), ('column_03', 'column'),
                    ('vehicle_09', 'car'), ('column_12', 'column'), ('exit door_02', 'exit door')])]

    def evidence(self):
        return SemanticVoxelSnapshot(.3, np.array([[i, 0, -4] for i in range(5)]),
            np.array([r['runtime_instance_id'] for r in self.records()]),
            np.full(5, 3), np.ones(5))

    def test_separate_continuous_categories_and_voxel_namespace(self):
        original = self.records()
        evidence = self.evidence()
        records, numbered, mapping = number_exported_instances(original, evidence)
        expected = ['vehicle_01', 'column_01', 'vehicle_02', 'column_02', 'exit door_02']
        self.assertEqual([r['runtime_instance_id'] for r in records], expected)
        self.assertEqual(numbered.instance_ids.tolist(), expected)
        self.assertEqual(original[0]['runtime_instance_id'], 'vehicle_04')
        self.assertEqual(evidence.instance_ids[0], 'vehicle_04')
        self.assertEqual(records[0]['source_merged_from'], ['vehicle_02'])
        self.assertEqual(records[0]['merged_from'], [])
        self.assertEqual(records[0]['source_video_instance_id'], 'vehicle_04')
        self.assertIsNone(records[0]['video_instance_id'])
        snapshot = SimpleNamespace(indices=evidence.indices, resolution=.3,
                                   log_odds=np.ones(5)*2, occupied_threshold=1.)
        labels = project_semantic_voxels(snapshot, numbered, records)
        self.assertEqual(labels['semantic_instance_ids'][labels['semantic_labels']].tolist(), expected)
        self.assertEqual(mapping['vehicle_09'], 'vehicle_02')

    def test_scene_graph_relations_follow_final_ids(self):
        scene = SceneMemory('test', np.zeros(3), 0., SimpleNamespace(relation_radius=6., near_distance=3.))
        scene.records = {r['runtime_instance_id']: r for r in self.records()}
        scene.semantic_voxels = self.evidence()
        scene.add_place('place_0000', np.zeros(3))
        scene.observed_at['vehicle_09'].add('place_0000')
        scene.coobserved[('column_12', 'vehicle_09')] = 7
        scene.number_final_instances()
        graph = scene.graph()
        self.assertEqual({n['id'] for n in graph['nodes']}, {
            'vehicle_01', 'vehicle_02', 'column_01', 'column_02', 'exit door_02', 'place_0000'})
        self.assertTrue(any(e['relation'] == 'co_observed' and e['source'] == 'column_02'
                            and e['target'] == 'vehicle_02' and e['frame_count'] == 7
                            for e in graph['edges']))
        self.assertTrue(any(e['relation'] == 'observed_at' and e['source'] == 'vehicle_02'
                            for e in graph['edges']))

    def test_final_loop_publication_requests_numbering_even_after_live_update(self):
        mapping = object.__new__(LiveLoopMapping)
        mapping.close = Mock()
        mapping.update = Mock()
        mapping._stop = Mock()
        mapping.finalize()
        mapping.update.assert_called_once_with(final=True)
        mapping.close.assert_called_once()
        mapping._stop.set.assert_called_once()
