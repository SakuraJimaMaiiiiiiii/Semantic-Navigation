import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
from navigation.mission import SemanticNavigator
from config.drone_config import Config


class TargetMotionTests(unittest.TestCase):
    def move(self, path):
        vehicle = SimpleNamespace(get_position=lambda: np.zeros(3), move_waypoint_guided=Mock())
        nav = SimpleNamespace(vehicle=vehicle, _check_health=lambda:None, _sample_position=lambda:None,
                              config=SimpleNamespace(leg_length=1.2, arrival_tolerance=.4, leg_timeout=30),
                              deadline=100, clock=lambda:0, _guard=object())
        SemanticNavigator._move_segment(nav, np.asarray(path, dtype=float))
        return vehicle.move_waypoint_guided.call_args

    def test_short_first_segment_does_not_cut_corner(self):
        call = self.move([[0,0,0], [.5,0,0], [.5,3,0]])
        np.testing.assert_allclose(call.args[0], [.5,0,0])
        self.assertFalse(call.kwargs['pass_through'])

    def test_straight_subgoal_does_not_stop(self):
        call = self.move([[0,0,0], [4,0,0]])
        np.testing.assert_allclose(call.args[0], [1.2,0,0])
        self.assertTrue(call.kwargs['pass_through'])

    def test_final_goal_stops(self):
        call = self.move([[0,0,0], [1,0,0]])
        self.assertFalse(call.kwargs['pass_through'])

    def test_yaw_commands_slew_on_short_arc(self):
        tree = ast.parse(Path('vehicle/rfly_multirotor.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='move_waypoint_guided')
        clock = [0.]
        def sleep(dt): clock[0] += dt
        scope = {'np':np, 'math':math, 'time':SimpleNamespace(monotonic=lambda:clock[0], sleep=sleep),
                 'ROUTE_SOURCE_LABELS':{}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<guided>', 'exec'), scope)
        commands = []
        initial_yaw = math.radians(179)
        vehicle = SimpleNamespace(config=Config(), _require_offboard=lambda:None,
            _vector3=lambda value,name:np.asarray(value,dtype=float),
            get_position=lambda:np.zeros(3), get_velocity=lambda:np.zeros(3),
            get_euler=lambda:np.array([0,0,initial_yaw]), _flight_status=lambda:{},
            raise_if_collision=lambda **kwargs:None, hover=lambda:None,
            send_position=lambda point,yaw:commands.append(yaw))
        scope['move_waypoint_guided'](vehicle, [2,0,0], yaw=math.radians(-179), control_horizon=.5)
        increments = np.diff([initial_yaw,*commands])
        self.assertTrue(np.all(increments >= -1e-12))
        self.assertLessEqual(max(increments), vehicle.config.guided_max_yaw_rate/vehicle.config.guided_update_rate+1e-10)
        self.assertAlmostEqual(commands[-1], math.radians(181))


if __name__ == '__main__': unittest.main()
