import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from navigation.exploration_failures import filter_failed_routes
from navigation.object_inspection import ObjectInspectionMemory
from test_vehicle_inspection import car


class ExplorationRegressionTests(unittest.TestCase):
    def test_failed_local_segment_excludes_other_frontiers_on_same_route(self):
        current = np.array([4.93, 30.36, -1.5])
        blocked = dict(status='blocked', source='utility_frontier', failed_at=100.,
                       position=[6.75, 39.45, -1.5], failure=dict(local_status='no_path',
                       current_ned=current.tolist(), controller=dict(target_ned=[5.25,32.25,-1.5])))
        choices = [('same', [8.55,38.55,-1.5]), ('other', [0.,30.,-1.5])]
        paths = {'same': [current, [5.25,32.25,-1.5], choices[0][1]],
                 'other': [current, choices[1][1]]}
        result, count = filter_failed_routes(choices, paths, [blocked], current, 101., 3., 30.)
        self.assertEqual([key for key, _ in result], ['other'])
        self.assertEqual(count, 1)
        self.assertEqual(filter_failed_routes(choices, paths, [blocked], current, 131., 3., 30.)[1], 0)
        blocked['failure']['local_status'] = 'stale_map'
        self.assertEqual(filter_failed_routes(choices, paths, [blocked], current, 101., 3., 30.)[1], 0)

    def test_supplementary_views_do_not_unlock_more_supplementary_sessions(self):
        tree = ast.parse(Path('navigation/frontier_explorer.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_inspection_due')
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<inspection>', 'exec'), ns)
        nav = SimpleNamespace(exploration_waypoints=[dict(status='arrived', source='object_inspection')])
        self.assertFalse(ns['_inspection_due'](nav))
        nav.exploration_waypoints.append(dict(status='arrived', source='utility_frontier'))
        self.assertTrue(ns['_inspection_due'](nav))
        nav._inspection_after_arrivals = 1
        self.assertFalse(ns['_inspection_due'](nav))

    def test_unproductive_inspection_cannot_retry_as_connection_review(self):
        memory = ObjectInspectionMemory()
        record = car(connection_targets=[dict(last_seen=10.)])
        memory.sessions['vehicle_01'] = dict(status='no_fresh_observation', connection_review=False)
        self.assertIsNone(memory.eligible({'vehicle_01':record}, 10., np.array([4.,0.,-1.5])))
