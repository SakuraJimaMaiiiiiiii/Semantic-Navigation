import unittest

import numpy as np
from navigation.object_inspection import ObjectInspectionMemory
from test_vehicle_inspection import car


class ExplorationRegressionTests(unittest.TestCase):
    def test_unproductive_inspection_cannot_retry_as_connection_review(self):
        memory = ObjectInspectionMemory()
        record = car(connection_targets=[dict(last_seen=10.)])
        memory.sessions['vehicle_01'] = dict(status='no_fresh_observation', connection_review=False)
        self.assertIsNone(memory.eligible({'vehicle_01':record}, 10., np.array([4.,0.,-1.5])))
