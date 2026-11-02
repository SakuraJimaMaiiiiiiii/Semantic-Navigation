import unittest
from types import SimpleNamespace
import numpy as np
from localization.scene_graph_adapter import SceneGraphAdapter


class SceneGraphTests(unittest.TestCase):
    def make_graph(self, identity=2):
        graph = SceneGraphAdapter()
        sample = dict(class_name='car',confidence=.9,source_instance_id=8)
        measurement = SimpleNamespace(position_world=np.array([1.,2.,3.]),
                                      measurement_covariance=np.eye(3),surface_points_world=np.zeros((5,3)))
        graph.observe(10.,[sample],[measurement],[identity],np.eye(4),lambda _: 'vehicle')
        return graph

    def test_merge_redirect_no_dangling_edges(self):
        graph=self.make_graph()
        records=[dict(instance_id='vehicle_01',merged_from=['vehicle_02'],position_world=[1,2,3])]
        snapshot=graph.snapshot(records,'run')
        self.assertEqual(snapshot['edges'][0]['target'],'vehicle_01')
        ids={n['id'] for n in snapshot['nodes']}
        self.assertTrue(all(e['source'] in ids and e['target'] in ids for e in snapshot['edges']))
        observation=snapshot['nodes'][1]['attributes']
        self.assertEqual(observation['source_instance_id'],'vehicle_08')
        self.assertEqual(observation['assigned_instance_id'],'vehicle_02')
        self.assertEqual(observation['canonical_instance_id'],'vehicle_01')
        snapshot['nodes'][0]['attributes']['position_world'][0]=99
        self.assertEqual(records[0]['position_world'][0],1)

    def test_filtered_instance_keeps_evidence_without_edge(self):
        snapshot=self.make_graph().snapshot([],'run')
        self.assertEqual(snapshot['edges'],[])
        self.assertEqual(snapshot['nodes'][0]['attributes']['association_status'],'not_exported')

    def test_unassigned_measurement_does_not_invent_instance(self):
        snapshot=self.make_graph(None).snapshot([],'run')
        self.assertEqual(snapshot['edges'],[])
        self.assertEqual(snapshot['nodes'][0]['attributes']['association_status'],'unassigned')

    def test_ambiguous_alias_rejected(self):
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            self.make_graph().snapshot([dict(instance_id='vehicle_01',merged_from=['vehicle_02']),
                                        dict(instance_id='vehicle_03',merged_from=['vehicle_02'])],'run')
