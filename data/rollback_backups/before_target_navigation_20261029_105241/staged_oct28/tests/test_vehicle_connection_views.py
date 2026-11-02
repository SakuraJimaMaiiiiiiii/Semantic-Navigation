import unittest
from types import SimpleNamespace
import numpy as np
from test_vehicle_split_evidence import module
from test_vehicle_inspection import car, grid
from navigation.object_inspection import ObjectInspectionMemory


class ConnectionTests(unittest.TestCase):
    def test_oblique_corner_waits_for_parent_evidence(self):
        tracker = module.VehicleInstanceTracker()
        parent = SimpleNamespace(
            bbox_3d_min=np.array([5.978,28.996,-1.455]),
            bbox_3d_max=np.array([10.187,31.825,-.054]),
            appearance=np.array([1.,0.]), appearance_gallery=[])
        tracker.tracks[3] = parent
        corner = SimpleNamespace(
            bbox_3d_min=np.array([9.7,29.21,-1.47]),
            bbox_3d_max=np.array([11.29,30.6,-.24]),
            appearance=np.array([.75,.25]))
        self.assertEqual(tracker._partial_parent_candidates(corner), {3})
        # A disjoint neighboring car must not be held against this parent.
        corner.bbox_3d_min += [0.,4.,0.]
        corner.bbox_3d_max += [0.,4.,0.]
        self.assertEqual(tracker._partial_parent_candidates(corner), set())
        # A full-sized overlapping observation is not a small corner either.
        corner.bbox_3d_min = parent.bbox_3d_min.copy()
        corner.bbox_3d_max = parent.bbox_3d_max.copy()
        self.assertEqual(tracker._partial_parent_candidates(corner), set())

    def test_surface_position_waits_for_width_and_height_support(self):
        surface = module.InstanceSurface()
        side = np.array([[x, -1., z] for x in np.linspace(-2,2,41) for z in np.linspace(-1.,0.,11)])
        for _ in range(2):
            surface.update(side)
        self.assertEqual(surface.vehicle_geometry()['position_status'], 'partial_surface')
        other = side.copy(); other[:,1] = 1.
        for _ in range(2):
            surface.update(other)
        model = surface.vehicle_geometry()
        self.assertEqual(model['position_status'], 'multi_surface_estimate')
        center = model['observed_oriented_box']['center']
        np.testing.assert_allclose(center, [0.,0.,-.5], atol=.1)
        surface.update(side)
        np.testing.assert_allclose(surface.vehicle_geometry()['observed_oriented_box']['center'], center, atol=.01)

    def test_connecting_measurement_must_cover_both_surfaces(self):
        a,b = module.InstanceSurface(),module.InstanceSurface()
        left = np.array([[0.,y,z] for y in np.linspace(0,1,11) for z in np.linspace(-1,0,11)])
        right = left + [2.,0.,0.]
        for _ in range(2):
            a.update(left);b.update(right)
        self.assertTrue(a.observed_patch(left))
        self.assertFalse(b.observed_patch(left))
        both = np.vstack([left,right])
        self.assertTrue(a.observed_patch(both))
        self.assertTrue(b.observed_patch(both))

    def test_three_connecting_views_import_fragment_into_existing_id(self):
        tracker = module.VehicleInstanceTracker()
        side = np.array([[x,-1.,z] for x in np.linspace(-2,2,41) for z in np.linspace(-1,0,11)])
        end = np.array([[2.,y,z] for y in np.linspace(-1,1,21) for z in np.linspace(-1,0,11)])
        tracks = []
        for identity, points in [(1,side),(0,end)]:
            track = module.VehicleTrack(identity, 'car', points.mean(0), np.eye(3),
                points.min(0), points.max(0), np.array([1.,0.]), .6, 5, 1.,
                appearance_gallery=[np.array([1.,0.])])
            track.surface.update(points);track.surface.update(points)
            tracks.append(track)
        tracker.tracks[1] = tracks[0]
        tracker._pending = [tracks[1]]
        tracker._next_id = 2
        points = np.vstack([side,end])
        observation = SimpleNamespace(position_world=points.mean(0), bbox_3d_min=points.min(0),
            bbox_3d_max=points.max(0), measurement_covariance=np.eye(3),
            appearance=np.array([1.,0.]), color_scores={}, surface_points_world=points)
        for stamp in (2.,2.5,3.):
            identities = tracker.update(['car'],[.6],[observation],stamp)
            self.assertEqual(identities,[1])
        self.assertFalse(tracker._pending)
        self.assertEqual(list(tracker.tracks),[1])
        self.assertTrue(tracker.tracks[1].surface.observed_patch(end))

    def test_ambiguous_neighbors_do_not_create_a_connection_request(self):
        tracker = module.VehicleInstanceTracker()
        parent = SimpleNamespace(bbox_3d_min=np.array([-2.,-1.,-1.]),bbox_3d_max=np.array([2.,1.,0.]))
        fragment = SimpleNamespace(observation_count=5, bbox_3d_min=np.array([2.2,-.5,-.8]),
                                   bbox_3d_max=np.array([2.4,.5,-.2]), separate_counts={}, last_seen=10.)
        tracker.tracks={1:parent}; tracker._pending=[fragment]
        self.assertIn(1,tracker.connection_hypotheses())
        tracker.tracks[2]=parent
        self.assertEqual(tracker.connection_hypotheses(),{})

    def test_connection_review_once_and_joint_visibility(self):
        memory = ObjectInspectionMemory()
        record = car()
        record['connection_targets'] = [{'min':[2.2,-.5,-1.], 'max':[2.5,.5,0.], 'last_seen':12.}]
        records = {'vehicle_01':record}
        current=np.array([5.,3.,-1.5])
        memory.sessions['vehicle_01']={'status':'done', 'connection_review':False}
        self.assertIsNotNone(memory.eligible(records,12.,current))
        point,_ = memory.candidate(grid(),record,current,[],current,20.)
        self.assertTrue(memory._visible(grid(),point,record['bbox_3d']))
        self.assertTrue(memory._visible(grid(),point,record['connection_targets'][0]))
        memory.sessions['vehicle_01']['connection_review']=True
        self.assertIsNone(memory.eligible(records,12.,current))
