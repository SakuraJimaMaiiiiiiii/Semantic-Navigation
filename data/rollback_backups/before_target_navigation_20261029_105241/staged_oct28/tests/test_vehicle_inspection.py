"""CPU-only tests; no simulator, flight hardware or model imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import math
import unittest

import numpy as np

from navigation.object_inspection import ObjectInspectionMemory
from navigation.route import NavigationGrid
from navigation.path_smoothing import rounded_path, advance_route
from navigation.waypoint_cursor import WaypointCursor


def car(identity='vehicle_01', **updates):
    result = dict(runtime_instance_id=identity, class_name='car', last_seen=10.,
                  position_world=[0., 0., -1.5], observation_count=10,
                  bbox_3d={'min': [-2., -1., -2.], 'max': [2., 1., 0.]})
    result.update(updates)
    return result


def grid():
    return NavigationGrid({(x,y) for x in range(-20,21) for y in range(-20,21)},
                          set(), .5, (0,0), -1.5,
                          SimpleNamespace(clearance_radius=.4, max_plan_nodes=10000))


class InspectionTests(unittest.TestCase):
    def test_fresh_and_alias_memory(self):
        memory = ObjectInspectionMemory()
        records = {'vehicle_01': car()}
        self.assertIsNotNone(memory.eligible(records, 10., np.array([4.,0.,-1.5])))
        # A sighting survives leaving the camera's FOV during transit.
        self.assertIsNotNone(memory.eligible(records, 12., np.array([4.,0.,-1.5])))
        self.assertIsNone(ObjectInspectionMemory().eligible(records, 12., np.array([4.,0.,-1.5])))
        memory.sessions['vehicle_01'] = {'status': 'view_budget_reached'}
        records = {'vehicle_06': car('vehicle_06', merged_from=['vehicle_01'])}
        self.assertIsNone(memory.eligible(records, 10., np.array([4.,0.,-1.5])))
        self.assertEqual(memory.resolve(records, 'vehicle_01')['runtime_instance_id'], 'vehicle_06')

    def test_low_confidence_hypotheses_can_request_inspection(self):
        tree = ast.parse(Path('mapping/semantic_instance_mapper.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SemanticInstanceMapper')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'object_inspection_records')
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<mapper>', 'exec'), namespace)
        track = SimpleNamespace(observation_count=10, class_name='car', map_position=np.array([0.,0.,-1.]),
                                bbox_3d_min=np.array([-2.,-1.,-2.]), bbox_3d_max=np.array([2.,1.,0.]),
                                last_seen=10., confidence=.23)
        mapper = SimpleNamespace(semantic_map_records=lambda **kw: [], _vehicles=SimpleNamespace(tracks={5:track}, confirmation_observations=3,
                                                          redirects={}, canonical_id=lambda i:i, connection_hypotheses=lambda:{}))
        records = namespace['object_inspection_records'](mapper, [])
        self.assertEqual(records[0]['confidence'], .23)
        memory = ObjectInspectionMemory()
        self.assertIsNotNone(memory.eligible({r['runtime_instance_id']:r for r in records}, 10., np.array([4.,0.,-1.5])))
        track.observation_count = 1
        self.assertEqual(namespace['object_inspection_records'](mapper, []), [])

    def test_views_safe_separated_fixed_height(self):
        g = grid()
        current = np.array([5., 0., -1.5])
        angles = []
        for _ in range(4):
            point, angle = ObjectInspectionMemory.candidate(g, car(), current, angles, current, 20.)
            self.assertEqual(point[2], -1.5)
            self.assertIn(g.cell(point), g.free)
            self.assertIsNotNone(g.path(current, point))
            for old in angles:
                self.assertGreaterEqual(abs((angle-old+math.pi) % (2*math.pi)-math.pi), math.pi/3)
            angles.append(angle)
            current = point

    def test_unreachable_and_radius_rejected(self):
        g = grid()
        current = np.array([5.,0.,-1.5])
        self.assertIsNone(ObjectInspectionMemory.candidate(g, car(), current, [], current, .1))
        g.free.clear()
        self.assertIsNone(ObjectInspectionMemory.candidate(g, car(), current, [], current, 20.))

    def test_committed_transit_is_not_preempted(self):
        tree = ast.parse(Path('navigation/frontier_explorer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in {'_travel'}]
        namespace = {'np': np, 'rounded_path': rounded_path, 'advance_route': advance_route, 'WaypointCursor': WaypointCursor}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), '<flight>', 'exec'), namespace)
        for inspecting in (False, True):
            with self.subTest(inspecting=inspecting):
                position = np.array([5.,0.,-1.5])
                goal = np.array([5.,4.,-1.5])
                holds = []
                records = {'vehicle_01': car()}
                nav = SimpleNamespace(
                    _inspecting_object=inspecting, memory=SimpleNamespace(places={}, _last_frame=10.),
                    object_inspections=ObjectInspectionMemory(), global_map=SimpleNamespace(revision=0),
                    clock=lambda:0., deadline=100., exploration_origin=position.copy(), checkpoint=None,
                    exploration_config=SimpleNamespace(max_viewpoints=10, max_radius=20.),
                    config=SimpleNamespace(arrival_tolerance=.4, corner_rounding_distance=1.),
                    vehicle=SimpleNamespace(get_position=lambda:position.copy(), get_velocity=lambda:np.zeros(3),
                                            config=SimpleNamespace(guided_z_tolerance=.2)),
                    _grid=grid, _check_health=lambda:None,
                    _goal_reached=lambda p:np.linalg.norm(p-position)<.1,
                    _hold_for_planning=lambda:holds.append(position.copy()),
                    _inspection_records=lambda:records, _event=lambda *a,**kw:None)
                def move(path, *, flowing, turning):
                    position[1] = min(position[1]+1., goal[1])
                nav._move = move
                outcome = namespace['_travel'](nav, goal)
                self.assertIs(outcome, True)
                self.assertIsNone(nav._last_travel_failure)
                self.assertIsNone(nav._motion_progress)
                self.assertGreaterEqual(len(holds), 2)
                self.assertEqual(position[1], goal[1])

    def test_fresh_observation_and_execution_budget(self):
        # Execute the real orchestration method with a deterministic flight adapter.
        tree = ast.parse(Path('navigation/frontier_explorer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_inspect_object')
        namespace = {'np': np, 'math': math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<inspection>', 'exec'), namespace)
        for fresh in (False, True, "timeout", "blocked"):
            with self.subTest(fresh=fresh):
                record = car()
                position = np.array([5.,0.,-1.5])
                clock = [0.]
                nav = SimpleNamespace(
                    memory=SimpleNamespace(records={'vehicle_01': record}, places={}),
                    object_inspections=ObjectInspectionMemory(), deadline=100., clock=lambda: clock[0],
                    config=SimpleNamespace(verification_frames=3, verification_timeout=.5),
                    exploration_config=SimpleNamespace(object_inspection_views=2, object_inspection_timeout=90.,
                                                       max_viewpoints=10, max_radius=20.),
                    exploration_origin=position.copy(), current_place='start', exploration_waypoints=[],
                    _checkpoint=lambda: None, _frame=lambda: None, _grid=grid,
                    _align=lambda yaw: None)
                nav._inspection_records = lambda: nav.memory.records
                nav.memory.add_place=lambda name, pos, parent=None: nav.memory.places.update({name: pos.copy()})
                nav.vehicle=SimpleNamespace(get_position=lambda: position.copy(), hover=lambda: None)
                def travel(goal):
                    if fresh == "timeout":
                        clock[0] = nav.deadline
                        raise TimeoutError("semantic_mission_timeout")
                    if fresh == "blocked":
                        nav._last_travel_failure = {"reason": "no_path"}
                        return False
                    position[:] = goal
                    nav.current_place = None
                    return True
                def wait(duration):
                    clock[0] += duration
                    if fresh is True:
                        record['last_seen'] += 1
                        record['observation_count'] += 1
                nav._travel, nav._wait = travel, wait
                namespace['_inspect_object'](nav, record)
                state = nav.object_inspections.sessions['vehicle_01']
                self.assertEqual(len(state['views']), 2 if fresh is True else 0)
                self.assertEqual(state['status'], 'time_budget_reached' if fresh == 'timeout' else
                                 'navigation_failed' if fresh == 'blocked' else
                                 'view_budget_reached' if fresh is True else 'no_fresh_observation')
                self.assertEqual(nav.deadline, 100.)
                self.assertFalse(nav._inspecting_object)
                self.assertEqual(len(nav.exploration_waypoints), 2 if fresh is True else 1)
                if isinstance(fresh, bool):
                    self.assertTrue(all(w['observation_verified'] == fresh for w in nav.exploration_waypoints))
                else:
                    self.assertTrue(all(w['status'] == ("interrupted" if fresh == "timeout" else "blocked")
                                        for w in nav.exploration_waypoints))
                self.assertTrue(all('frontier' in w for w in nav.exploration_waypoints))


if __name__ == '__main__':
    unittest.main()
