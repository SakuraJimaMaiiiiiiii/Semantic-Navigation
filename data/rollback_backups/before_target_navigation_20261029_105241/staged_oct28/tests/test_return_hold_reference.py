"""Exercise real return-loop retries with simulated estimator drift, no SDK."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class ReturnHoldTests(unittest.TestCase):
    def test_blocked_route_does_not_recapture_hold_position_each_retry(self):
        tree = ast.parse(Path('mission/rfly_mission_controller.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_return_on_certified_path')
        method.body = [n for n in method.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        now = [0.]
        position = np.array([-5.7,13.6,-1.82])
        holds, events = [], []
        class Collision(RuntimeError):
            flight_context = {'last_position_command': 'retained'}
        class Guard:
            def __init__(self, context):
                pass
            def capture_failure(self, name, detail, **kwargs):
                events.append((name, detail))
            def close_diagnostics(self):
                pass
        grid = SimpleNamespace(free=set(), cell=lambda p:(0,0), path=lambda a,b:None)
        namespace = dict(np=np, math=math, SimpleNamespace=SimpleNamespace,
                         time=SimpleNamespace(monotonic=lambda:now[0]),
                         NavigationGrid=SimpleNamespace(from_global=lambda *a,**kw:grid),
                         WaypointCursor=lambda:None, _FlightGuard=Guard)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<return>', 'exec'),namespace)
        def wait(duration):
            now[0] += duration
            position[2] -= .04
            # Model the recorded collision while still inside the wait budget.
            if now[0] >= .6:
                raise Collision('pipe collision')
        vehicle = SimpleNamespace(get_position=lambda:position.copy(),
                                  hover=lambda:holds.append((now[0],position.copy())), wait=wait)
        nav = SimpleNamespace(config=SimpleNamespace(arrival_tolerance=.4, leg_timeout=30., map_timeout=3.),
                              local_map=SimpleNamespace(snapshot=lambda:None),
                              global_map=SimpleNamespace(snapshot=lambda:None), _check_maps=lambda:None)
        controller = SimpleNamespace(vehicle=vehicle, semantic_navigator=nav,
                                     return_safety_planner=object(), _return_origin=lambda:np.zeros(3),
                                     _restore_return_height=lambda *a:None)
        with self.assertRaisesRegex(Collision,'pipe collision'):
            namespace['_return_on_certified_path'](controller)
        # Only one hold before the collision; subsequent holds are cleanup.
        before_collision = [p for t,p in holds if t < .6]
        self.assertEqual(len(before_collision),1)
        np.testing.assert_allclose(before_collision[0],[-5.7,13.6,-1.82])
        self.assertEqual(events[-1][0], 'return_exception')
        self.assertEqual(events[-1][1]['flight_context'], Collision.flight_context)
