from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from config.exploration_config import FrontierExplorationConfig
from config.semantic_navigation_config import SemanticNavigationConfig
from navigation.exploration_session import ExplorationSession
from navigation.frontier_explorer import FrontierExplorer
from mission.rfly_mission_controller import RflyMissionController


class ExplorerSessionCompatibilityTests(unittest.TestCase):
    def make_session(self, directory):
        session = object.__new__(ExplorationSession)
        session.directory = Path(directory)
        session.memory = Mock()
        session.memory.exploration = {}
        session.memory.places = {}
        session.config = FrontierExplorationConfig()
        session.runtime = SimpleNamespace(lidar_worker=None, controller=SimpleNamespace(),
            global_sparse_map=Mock(), run_mission=Mock())
        session.runtime.global_sparse_map.snapshot.return_value = Mock()
        session.annotate = Mock()
        session._arguments = lambda request, output: dict(
            vehicle=Mock(), perception=Mock(), local_map=Mock(), global_map=Mock(),
            local_planner=SimpleNamespace(config=Mock()), request=request,
            config=SemanticNavigationConfig(), output_path=output, clock=lambda: 10.)
        return session

    def test_initial_observation_records_exploration_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(directory)
            explorer = FrontierExplorer(**session._arguments(None, Path(directory)/'report.json'),
                exploration_config=session.config, memory=session.memory, checkpoint=session.checkpoint)
            explorer._nominal_altitude = explorer.altitude = -1.5
            explorer.current_place = 'place_0000'
            explorer._frame = Mock()
            explorer._initial_observation()
            self.assertEqual(session.annotate.call_args.kwargs['state'], 'exploring: ego_avoidance_only')
            session.memory.checkpoint.assert_called_once_with(session.directory)
            self.assertTrue((session.directory/'private'/'exploration_waypoints.json').exists())
            RflyMissionController._set_navigation_phase(SimpleNamespace(semantic_navigator=explorer), 'returning')
            self.assertEqual(session.annotate.call_args.kwargs['state'], 'returning: ego_avoidance_only')

    def test_mission_exception_preserves_original_error_and_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(directory)
            session.runtime.run_mission.side_effect = RuntimeError('original mission failure')
            with self.assertRaisesRegex(RuntimeError, 'original mission failure'):
                session.explore()
            self.assertEqual(session.memory.exploration['failure_phase'], 'exploring')
            self.assertEqual(session.memory.exploration['error'], 'original mission failure')


if __name__ == '__main__':
    unittest.main()
