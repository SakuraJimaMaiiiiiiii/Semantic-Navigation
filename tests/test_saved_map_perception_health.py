import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.exploration_session import SavedMapNavigator
from navigation.target import SemanticTargetRequest


class SavedMapPerceptionHealthTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.vehicle = Mock()
        self.vehicle._motion_progress_display = None
        self.vehicle.wait.side_effect = self.advance
        self.perception = Mock()
        self.perception.semantic_snapshot.return_value = SimpleNamespace(
            completed_at=0., timestamp=1., inference_seconds=3.9)
        self.world = SimpleNamespace(last_observation_completed_at=10.)
        self.nav = SavedMapNavigator(
            self.vehicle, self.perception, self.world, self.world, Mock(),
            SemanticTargetRequest(instance_id='vehicle_03'), SemanticNavigationConfig(),
            Path('unused.json'), clock=lambda: self.now, records=[], annotate=Mock())
        self.nav.started = 10.
        self.nav.deadline = 100.
        self.nav._event = Mock()

    def advance(self, seconds):
        self.now += seconds
        self.world.last_observation_completed_at = self.now

    def test_transit_allows_slow_recognition(self):
        self.nav._check_health()
        self.perception.raise_if_failed.assert_called_once()
        self.vehicle.hover.assert_not_called()

    def test_query_waits_for_fresh_frame(self):
        fresh = SimpleNamespace(completed_at=10.2, timestamp=2., inference_seconds=3.9)
        self.perception.semantic_snapshot.side_effect = [None, fresh]
        self.assertIs(self.nav._frame(), fresh)
        self.vehicle.hover.assert_called_once()

    def test_missing_frames_still_timeout(self):
        with self.assertRaisesRegex(RuntimeError, 'semantic_perception_stale'):
            self.nav._frame()
        self.assertAlmostEqual(self.now, 15.)

    def test_detector_failure_is_not_ignored(self):
        self.perception.raise_if_failed.side_effect = RuntimeError('detector failed')
        with self.assertRaisesRegex(RuntimeError, 'detector failed'):
            self.nav._check_health()

    def test_map_timeout_is_not_ignored(self):
        self.nav._check_health()
        self.now += 4.
        with self.assertRaisesRegex(RuntimeError, 'semantic_local_map_stale'):
            self.nav._check_health()
