from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from navigation.coverage_progress import CoverageProgress
from navigation.frontier_explorer import FrontierExplorer


class CoverageReviewTests(unittest.TestCase):
    def test_known_space_transit_does_not_cancel_unvisited_review(self):
        coverage = CoverageProgress(1., 120.)
        grid = SimpleNamespace(known=set(range(12000)), resolution=.3)
        for now in (0., 120., 250.):
            coverage.update(grid, -1.5, now, 1)
        self.assertTrue(coverage.metrics['review_due'])
        self.assertEqual(coverage.metrics['stage'], 'review_due')
        nav = object.__new__(FrontierExplorer)
        nav.coverage_progress = coverage
        nav.exploration_waypoints = [dict(status='arrived', position=[0., 0., -1.5])]
        nav.exploration_config = SimpleNamespace(viewpoint_spacing=3., coverage_review_attempts=3)
        nav._coverage_review_attempts = 0
        nav.frontiers = SimpleNamespace(batch_pending=False)
        nav._check_health = Mock()
        nav._event = Mock()
        # Repeated nearby viewpoints are excluded; a distant frontier survives
        # even though transit there takes longer than the low-growth window.
        choices = [('repeat', [1., 0., -1.5]), ('unvisited', [25., 0., -1.5])]
        nav._choices = Mock(return_value=(choices, False, False))
        for _ in range(3):
            selected, exhausted, _, _ = nav._coverage_review(grid, np.zeros(3), np.zeros(3))
            self.assertEqual([key for key, _ in selected], ['unvisited'])
            self.assertFalse(exhausted)
        selected, exhausted, _, _ = nav._coverage_review(grid, np.zeros(3), np.zeros(3))
        self.assertEqual(selected, [])
        self.assertTrue(exhausted)

    def test_empty_first_batch_does_not_hide_unvisited_later_batch(self):
        nav = object.__new__(FrontierExplorer)
        nav.exploration_waypoints = []
        nav.exploration_config = SimpleNamespace(viewpoint_spacing=3., coverage_review_attempts=3)
        nav._coverage_review_attempts = 0
        nav.frontiers = SimpleNamespace(batch_pending=True)
        nav._check_health = Mock()
        nav._event = Mock()
        def choices(*args):
            if nav._choices.call_count == 1:
                return [], False, False
            nav.frontiers.batch_pending = False
            return [('later', [10., 0., -1.5])], False, False
        nav._choices = Mock(side_effect=choices)
        selected, exhausted, _, _ = nav._coverage_review(None, np.zeros(3), np.zeros(3))
        self.assertEqual(selected[0][0], 'later')
        self.assertFalse(exhausted)
