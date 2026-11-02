"""Tracker regression tests without loading simulator-dependent mapping workers."""
import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np

pkg = types.ModuleType('_vehicle_test_mapping')
pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'mapping')]
sys.modules[pkg.__name__] = pkg
module = importlib.import_module('_vehicle_test_mapping.vehicle_instance_tracker')


class SplitEvidenceTests(unittest.TestCase):
    def test_replay_identity_requires_overlapping_geometry_and_repeated_support(self):
        for same_source, separate, distinct in [(True, False, False),
                                                (False, False, False),
                                                (True, True, False),
                                                (True, False, True)]:
            tracker = module.VehicleInstanceTracker()
            tracker._merge_window_seconds = 10.
            for i, x in [(1, 0.), (2, 1.5 if not separate else 6.)]:
                low, high = np.array([x, 0., -1.]), np.array([x+3., 2., 0.])
                tracker.tracks[i] = module.VehicleTrack(i, 'car', (low+high)/2, np.eye(3),
                    low, high, np.array([1., 0.]), .9, 5, 0.,
                    appearance_gallery=[np.array([1., 0.])])
            tracker.replay_source_votes = {1: {'vehicle_02': 5},
                                          2: {'vehicle_02' if same_source else 'vehicle_03': 5}}
            if distinct:
                tracker._distinct_pairs.add((1, 2))
            for stamp in (1., 40., 80.):
                tracker.tracks[2].last_seen = stamp
                if distinct:
                    tracker.tracks[1].last_seen = stamp
                tracker._merge_duplicates(stamp)
            self.assertEqual(len(tracker.tracks), 1 if same_source and not separate and not distinct else 2)

    def test_broad_ground_patch_does_not_create_vehicle_id(self):
        tracker = module.VehicleInstanceTracker()
        # Bounds observed for vehicle_07 in run_20261022_133756_361281.
        low = np.array([3.258, 30.051, -.136])
        high = np.array([4.895, 32.422, .168])
        track = module.VehicleTrack(0, 'car', (low+high)/2, np.eye(3),
                                    low, high, None, .26, 11, 10.)
        tracker._pending.append(track)
        self.assertIsNone(tracker._confirm(track))
        self.assertEqual(tracker._next_id, 1)
        self.assertFalse(tracker.tracks)
        # Confirmation becomes possible when a real vertical body is observed.
        track.bbox_3d_min[2] = -1.0
        self.assertEqual(tracker._confirm(track), 1)
        # The new-ID gate does not renumber an already established vehicle.
        track.bbox_3d_min[2] = -.136
        self.assertEqual(tracker._confirm(track), 1)

    def tracker(self, separated=False):
        tracker = module.VehicleInstanceTracker()
        points = np.array([[x,y,-.5] for x in np.arange(0,1,.1) for y in np.arange(0,1,.1)])
        for i in (1,2):
            shift = np.array([2.,0.,0.]) if separated and i == 2 else np.zeros(3)
            t = module.VehicleTrack(i, 'car', np.array([.5,.5,-.5])+shift, np.eye(3),
                np.array([0.,0.,-.6])+shift, np.array([1.,1.,-.4])+shift,
                np.array([1.,0.]), .6, 5, 0., appearance_gallery=[np.array([1.,0.])])
            t.surface.update(points+shift)
            t.surface.update(points+shift)
            tracker.tracks[i] = t
        tracker._distinct_pairs.add((1,2))
        return tracker

    def test_strong_surface_evidence_corrects_old_split_after_three_frames(self):
        tracker = self.tracker()
        for stamp in (1.,1.5):
            tracker.tracks[2].last_seen = stamp
            tracker._merge_duplicates(stamp)
            self.assertEqual(len(tracker.tracks), 2)
        tracker.tracks[2].last_seen = 2.
        tracker._merge_duplicates(2.)
        self.assertEqual(tracker.redirects, {2:1})

    def test_simultaneously_separate_detections_keep_veto(self):
        tracker = self.tracker()
        for stamp in (1.,1.5,2.,2.5):
            for t in tracker.tracks.values():
                t.last_seen = stamp
            tracker._merge_duplicates(stamp)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertFalse(tracker.redirects)

    def test_neighboring_same_color_cars_not_merged(self):
        tracker = self.tracker(separated=True)
        for stamp in (1.,1.5,2.,2.5):
            tracker.tracks[2].last_seen = stamp
            tracker._merge_duplicates(stamp)
        self.assertEqual(len(tracker.tracks), 2)

    def test_repeated_small_face_cannot_publish_an_independent_car(self):
        tracker = module.VehicleInstanceTracker()
        track = module.VehicleTrack(0, 'car', np.array([0.,0.,-.4]), np.eye(3),
                                    np.array([0.,0.,-.7]), np.array([.36,1.16,-.18]),
                                    None, .6, 20, 10.)
        tracker._pending.append(track)
        self.assertIsNone(tracker._confirm(track))
        self.assertEqual(tracker._next_id, 1)
        self.assertFalse(tracker.tracks)
        # A broader follow-up face may establish identity normally.
        track.bbox_3d_max = np.array([.4,2.,-.1])
        self.assertEqual(tracker._confirm(track), 1)
        # Losing the full view later must not revoke or re-number the car.
        track.bbox_3d_max = np.array([.36,1.16,-.18])
        self.assertEqual(tracker._confirm(track), 1)

    def test_tall_front_face_can_establish_identity(self):
        tracker = module.VehicleInstanceTracker()
        track = module.VehicleTrack(0, 'car', np.zeros(3), np.eye(3),
                                    np.array([0.,0.,-1.2]), np.array([.2,1.4,0.]),
                                    None, .6, 5, 10.)
        tracker._pending.append(track)
        self.assertEqual(tracker._confirm(track), 1)

    def test_missed_confirmation_expires(self):
        tracker = self.tracker()
        for stamp in (1.,5.,9.):
            tracker.tracks[2].last_seen = stamp
            tracker._merge_duplicates(stamp)
        self.assertEqual(len(tracker.tracks), 2)
