import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np


def method(name, namespace, path='vehicle/rfly_multirotor.py'):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<landing>', 'exec'), namespace)
    return namespace[name]


class LandingTests(unittest.TestCase):
    def test_ue_freshness_uses_received_packet_not_constant_sim_clock(self):
        clock=[0.]
        packet=SimpleNamespace(runnedTime=3., PosE=[0.,0.,-1.], VelE=[0.,0.,0.])
        nav=SimpleNamespace(ue=SimpleNamespace(getUE4Data=lambda _:packet),config=SimpleNamespace(copter_id=1))
        execute=method('_ue_landing_diagnostics', {'np':np,'math':math,
                       'time':SimpleNamespace(monotonic=lambda:clock[0])})
        self.assertEqual(execute(nav)['observed_age_seconds'], 0.)
        clock[0]=10.
        self.assertEqual(execute(nav)['observed_age_seconds'], 10.)
        packet.PosE=list(packet.PosE)  # SDK CopyData received another packet.
        self.assertEqual(execute(nav)['observed_age_seconds'], 0.)

    def test_return_hands_directly_to_auto_land_before_checkpoint(self):
        events = []
        vehicle = Mock()
        vehicle.land.side_effect = lambda: events.append('land')
        vehicle.align_yaw.side_effect = lambda yaw: events.append('yaw')
        nav = SimpleNamespace(vehicle=vehicle, mission=SimpleNamespace(
            return_to_origin=True, final_hover_time=0., restore_initial_yaw=True),
            _return_yaw=1.2, global_return_planner=object(),
            _return_on_global_path=lambda: events.append('return'),
            _confirm_return_position=lambda: events.append('confirm_xy'),
            _set_navigation_phase=lambda phase: events.append(phase))
        execute = method('return_and_land', {'math': math}, 'mission/rfly_mission_controller.py')
        execute(nav)
        self.assertEqual(events, ['returning', 'return', 'confirm_xy', 'yaw',
                                  'confirm_xy', 'land', 'landing'])
        vehicle.approach_home_ground.assert_not_called()
        vehicle.land.assert_called_once()
