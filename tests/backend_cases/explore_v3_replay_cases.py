import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from mapping.explore_v3 import adapters as v3


class V3ReplayTests(unittest.TestCase):
    def test_vehicle_configuration_and_landmark_backend_are_preserved(self):
        from mapping.explore_v3.semantic_instance_mapper import SemanticInstanceMapper as Original
        mapper = v3.SemanticInstanceMapper(vehicle_max_association_distance=1.7,
            vehicle_appearance_gate=.4, vehicle_confirmation_observations=4,
            vehicle_viewpoint_noise_std=.2, process_noise_std=.03)
        self.assertEqual(mapper._vehicles.max_distance, 1.7)
        self.assertEqual(mapper._vehicles.appearance_gate, .4)
        self.assertEqual(mapper._vehicles.confirmation_observations, 4)
        np.testing.assert_allclose(mapper._vehicles._viewpoint_covariance, np.eye(3)*.04)
        self.assertIs(type(mapper)._has_valid_landmark_geometry, Original._has_valid_landmark_geometry)

    def test_live_and_loop_replay_share_vehicle_association(self):
        from mapping.loop_semantic_replay import replay, transform_measurement
        side = np.array([[x,-1.,z] for x in np.linspace(-2,2,41)
                         for z in np.linspace(-1,0,11)])
        end = np.array([[2.,y,z] for y in np.linspace(-1,1,21)
                        for z in np.linspace(-1,0,11)])
        corner = np.vstack([side[side[:,0] >= .8], end])
        packets = []
        mapper = v3.SemanticInstanceMapper()
        identities = []
        for i, points in enumerate([side]*3+[corner]*3+[np.vstack([side,end])]*3, 1):
            data = dict(position_world=points.mean(0).tolist(), bbox_3d_min=points.min(0).tolist(),
                bbox_3d_max=points.max(0).tolist(), measurement_covariance=(np.eye(3)*.001).tolist(),
                appearance=[1.,0.], surface_points_world=points.tolist())
            ids = mapper.update(['car'], [.9], [transform_measurement(data, np.eye(4))], float(i))
            identities.extend(x for x in ids if x is not None)
            packets.append(dict(timestamp=float(i), samples=[dict(class_name='car', confidence=.9,
                source_instance_id=ids[0], measurement=data)]))
        self.assertEqual(set(identities), {1})
        records = mapper.navigation_records(3)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['position_status'], 'multi_surface_estimate')
        np.testing.assert_allclose(records[0]['position_world'], [0.,0.,-.5], atol=.15)
        inspections = mapper.object_inspection_records(records)
        self.assertEqual(inspections[0]['runtime_instance_id'], 'vehicle_01')
        self.assertIn('connection_targets', inspections[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'measurements.jsonl'
            path.write_text(''.join(json.dumps(p)+'\n' for p in packets))
            replayed, _, _ = replay(path, [1.,9.], [np.eye(4),np.eye(4)], .3, 2000,
                                    mapper_type=v3.SemanticInstanceMapper)
        self.assertEqual([r['runtime_instance_id'] for r in replayed],
                         [r['runtime_instance_id'] for r in records])
        for actual, expected in zip(replayed, records):
            np.testing.assert_allclose(actual['position_world'], expected['position_world'])
            self.assertEqual(actual['bbox_3d'], expected['bbox_3d'])
            self.assertEqual(actual['observed_oriented_box'], expected['observed_oriented_box'])

    def test_loop_replay_uses_supplied_v3_mapper(self):
        from mapping.loop_semantic_replay import replay
        measurement = dict(position_world=[0.,0.,-1.3], bbox_3d_min=[-.3,-.3,-2.6],
                           bbox_3d_max=[.3,.3,0.], measurement_covariance=np.eye(3).tolist(),
                           surface_points_world=None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'observations.jsonl'
            path.write_text(''.join(json.dumps(dict(timestamp=float(i), samples=[dict(
                class_name='parking garage structural column', confidence=.9,
                measurement=measurement, source_instance_id=1)]))+'\n' for i in (1,2,3)))
            factory = Mock(side_effect=v3.SemanticInstanceMapper)
            records, evidence, count = replay(path, [1.,3.], [np.eye(4),np.eye(4)], .3, 100,
                                               mapper_type=factory)
        factory.assert_called_once_with(sign_min_observations=10)
        self.assertEqual(count, 3)
        self.assertEqual(len(records), 0)  # A box without measured faces cannot confirm a column.

