import ast
import math
from pathlib import Path
import unittest

import numpy as np
from navigation.object_inspection import ObjectInspectionMemory, LANDMARKS
from test_vehicle_inspection import car, grid


class LandmarkInspectionTests(unittest.TestCase):
    def test_safety_landmarks_are_remembered_without_vehicle_labels(self):
        for category in LANDMARKS:
            record = car(category.replace(' ', '_')+'_01', class_name=category)
            memory = ObjectInspectionMemory()
            records = {record['runtime_instance_id']: record}
            self.assertIsNotNone(memory.eligible(records, 10., np.array([4.,0.,-1.5])))
            self.assertIsNotNone(memory.eligible(records, 15., np.array([4.,0.,-1.5])))
            memory.sessions[record['runtime_instance_id']] = {'status':'done'}
            self.assertIsNone(memory.eligible(records, 15., np.array([4.,0.,-1.5])))

    def test_landmark_views_stay_on_visible_side(self):
        record = car('exit_sign_01', class_name='exit sign',
                     bbox_3d={'min':[-.1,-.3,-2.4], 'max':[.1,.3,-2.1]})
        current = np.array([4.,0.,-1.5])
        origin = current.copy()
        g = grid()
        angles = []
        for _ in range(4):
            point, angle = ObjectInspectionMemory.candidate(g, record, current, angles, origin, 20.)
            self.assertGreater(point[0], 0.)
            self.assertLessEqual(abs(angle), math.pi/3 + 1e-8)
            self.assertEqual(point[2], -1.5)
            self.assertIsNotNone(g.path(current, point))
            angles.append(angle)
            current = point

    def test_extinguisher_alias_is_not_cabinet(self):
        tree = ast.parse(Path('mapping/semantic_instance_mapper.py').read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'canonical_semantic_class_name')
        aliases = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.endswith('ALIASES'):
                    try:
                        aliases[name] = ast.literal_eval(node.value)
                    except ValueError:
                        aliases[name] = ()
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])), '<canonical>', 'exec'), aliases)
        canonical = aliases['canonical_semantic_class_name']
        self.assertEqual(canonical('portable fire extinguisher'), 'fire extinguisher')
        self.assertEqual(canonical('wall mounted fire extinguisher'), 'fire extinguisher')
        self.assertEqual(canonical('fire hose cabinet'), 'fire cabinet')
