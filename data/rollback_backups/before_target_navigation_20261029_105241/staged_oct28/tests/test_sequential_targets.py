from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import unittest
import numpy as np
from mission.rfly_mission_controller import RflyMissionController
from navigation.exploration_session import ExplorationSession
from navigation.session_server import SessionServer
from navigation.target import SemanticTargetRequest


class SequentialTargetTests(unittest.TestCase):
    def controller(self):
        vehicle = Mock()
        vehicle.get_position.return_value = np.array([1.,2.,0.])
        vehicle.get_euler.return_value = np.array([0.,0.,.4])
        vehicle.mav.isArmed = True
        mission = SimpleNamespace(initial_yaw=None,takeoff_height=1.5,takeoff_wait=.1)
        return RflyMissionController(vehicle,mission,semantic_navigator=Mock())

    def test_two_targets_take_off_once_and_keep_home(self):
        controller = self.controller()
        controller.run_target()
        home = controller._return_origin_ned.copy()
        controller.vehicle.get_position.return_value = np.array([10.,20.,-1.5])
        controller.run_target(resume=True)
        controller.vehicle.takeoff.assert_called_once()
        controller.vehicle.enable_offboard.assert_called_once()
        controller.vehicle.land.assert_not_called()
        self.assertEqual(controller.semantic_navigator.run.call_count,2)
        self.assertEqual(controller.vehicle.hover.call_count,2)
        np.testing.assert_array_equal(controller._return_origin_ned,home)
        self.assertEqual(controller._return_yaw,.4)

    def test_disarmed_resume_is_rejected(self):
        controller = self.controller()
        controller.vehicle.mav.isArmed = False
        with self.assertRaises(RuntimeError): controller.run_target(resume=True)
        controller.semantic_navigator.run.assert_not_called()

    def test_return_command_resets_airborne_only_after_confirmation(self):
        session = ExplorationSession.__new__(ExplorationSession)
        session.airborne_navigation = True
        session.watch = Mock()
        session.clock = lambda:100
        session.runtime = SimpleNamespace(controller=Mock(),semantic_navigation_config=SimpleNamespace(mission_timeout=300))
        session._confirm_home = Mock()
        session.annotate = Mock()
        session.map_path = Path('map.html')
        result = session.return_home()
        session.runtime.controller.return_and_land.assert_called_once_with(force_return=True)
        session._confirm_home.assert_called_once()
        self.assertFalse(session.airborne_navigation)
        self.assertTrue(result['returned_home'])

    def test_airborne_next_target_keeps_current_altitude(self):
        session = ExplorationSession.__new__(ExplorationSession)
        session.airborne_navigation=True
        session.watch=Mock()
        session.clock=lambda:10
        session.home=np.zeros(3)
        current=np.array([10.,5.,-1.5])
        config=SimpleNamespace(home_tolerance=.6)
        session.config=config
        session.perception=Mock()
        session.perception.semantic_snapshot.return_value=SimpleNamespace(completed_at=10,timestamp=10)
        world=Mock()
        world.snapshot.return_value=SimpleNamespace(timestamp=10)
        session.runtime=SimpleNamespace(vehicle=Mock(),mission_config=SimpleNamespace(takeoff_height=1.5),
            semantic_navigation_config=SimpleNamespace(perception_timeout=5,map_timeout=3),
            occupancy_grid=world,global_sparse_map=world,controller=Mock())
        session.runtime.vehicle.get_position.return_value=current
        record={'runtime_instance_id':'vehicle_01'}
        session.navigation_archive=lambda:{'home':np.zeros(3),'records':[record]}
        session.annotate=Mock()
        session._arguments=lambda *args:{}
        session.navigator_type=Mock(return_value=SimpleNamespace(result={'success':True,'reason':'visually_confirmed'}))
        session.approach=session.route=None
        with tempfile.TemporaryDirectory() as directory:
            session.directory=Path(directory)
            session.map_path=session.directory/'map.html'
            grid=Mock()
            grid.approach.return_value=(0,current+1,[current,current+1])
            with patch('navigation.exploration_session.select_targets',return_value=[record]), \
                 patch('navigation.exploration_session.NavigationGrid.from_global',return_value=grid) as make_grid:
                result=session.navigate(SemanticTargetRequest(instance_id='vehicle_01'),'test')
            self.assertEqual(make_grid.call_args.args[1],-1.5)
            session.runtime.controller.run_target.assert_called_once_with(resume=True)
            self.assertFalse(result['returned_home'])
            self.assertTrue(result['waiting_for_next_target'])

    def test_server_serializes_targets_and_return(self):
        server=SessionServer('test',0)
        try:
            message={**server.descriptor(),'command':'return_home'}
            server.set_state('READY')
            reply=server.dispatch(message)
            self.assertEqual(server.state,'RETURNING')
            with self.assertRaises(ValueError): server.dispatch(message)
            server.commands.get_nowait()
            server.finish(reply['job_id'],{'success':True})
            message.update(command='navigate',target={'instance_id':'vehicle_02'})
            server.dispatch(message)
            self.assertEqual(server.commands.get_nowait()[2].instance_id,'vehicle_02')
        finally: server.close()


if __name__ == '__main__': unittest.main()
