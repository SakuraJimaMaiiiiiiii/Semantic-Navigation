"""Vehicle geometry must follow accumulated body surfaces, not view centroids."""
from types import SimpleNamespace
import unittest
import numpy as np
from test_vehicle_split_evidence import module


class MultiViewGeometryTests(unittest.TestCase):
    def surfaces(self):
        side = np.array([[x, -1., z] for x in np.linspace(-2, 2, 41)
                         for z in np.linspace(-1, 0, 11)])
        end = np.array([[2., y, z] for y in np.linspace(-1, 1, 41)
                        for z in np.linspace(-1, 0, 21)])
        return side, end

    def test_rotated_corner_estimates_box_not_pca_density_axis(self):
        side, end = self.surfaces()
        yaw = np.deg2rad(31.)
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        surface = module.InstanceSurface()
        for face in (side, end):
            face = face.copy()
            face[:, :2] = face[:, :2] @ rotation.T + [8., 3.]
            surface.update(face)
            surface.update(face)
        geometry = surface.vehicle_geometry()
        self.assertEqual(geometry['position_status'], 'multi_surface_estimate')
        box = geometry['observed_oriented_box']
        np.testing.assert_allclose(box['center'], [8., 3., -.5], atol=.15)
        self.assertLess(abs((box['yaw']-yaw+np.pi/2) % np.pi-np.pi/2), np.deg2rad(3))
        # More observations of only one face must not pull the model toward it.
        face = side.copy()
        face[:, :2] = face[:, :2] @ rotation.T + [8., 3.]
        for _ in range(10):
            surface.update(face)
        np.testing.assert_allclose(surface.vehicle_geometry()['observed_oriented_box']['center'],
                                   box['center'], atol=.01)

    def test_single_face_and_single_frame_outlier_do_not_complete_car(self):
        side, _ = self.surfaces()
        surface = module.InstanceSurface()
        surface.update(side); surface.update(side)
        surface.update(side + [0., 8., 0.])
        self.assertEqual(surface.vehicle_geometry()['position_status'], 'partial_surface')
        np.testing.assert_allclose(surface.bounds()[:, 1], [-1., -1.])

    def test_connecting_view_keeps_id_despite_changed_surface_center(self):
        side, end = self.surfaces()
        tracker = module.VehicleInstanceTracker()
        track = module.VehicleTrack(1, 'car', side.mean(0), np.eye(3)*.001,
            side.min(0), side.max(0), np.array([1., 0.]), .9, 10, 1.)
        track.surface.update(side); track.surface.update(side)
        tracker.tracks[1] = track
        tracker._next_id = 2
        points = np.vstack([side[side[:, 0] >= .8], end])
        self.assertFalse(track.surface.supports(points))
        self.assertTrue(track.surface.observed_patch(points))
        observation = SimpleNamespace(position_world=points.mean(0),
            bbox_3d_min=points.min(0), bbox_3d_max=points.max(0),
            measurement_covariance=np.eye(3)*.001, appearance=np.array([1., 0.]),
            color_scores={}, surface_points_world=points)
        for stamp in (2., 3., 4.):
            self.assertEqual(tracker.update(['car'], [.9], [observation], stamp), [1])
        self.assertEqual(list(tracker.tracks), [1])
        self.assertFalse(tracker._pending)

    def test_disjoint_same_color_car_has_no_surface_bridge(self):
        side, end = self.surfaces()
        surface = module.InstanceSurface()
        surface.update(side); surface.update(side)
        self.assertFalse(surface.observed_patch(np.vstack([side, end]) + [0., 4., 0.]))
