"""地下车库环绕任务配置测试。"""

import unittest

import numpy as np

from config import MissionConfig


class GarageMissionConfigTest(unittest.TestCase):
    def test_waypoints_form_clockwise_inset_garage_loop(self):
        mission = MissionConfig()
        points = np.asarray([
            waypoint.position_ned for waypoint in mission.waypoints
        ])

        expected = np.asarray([
            [6.0, 2.0, -1.5],
            [6.0, 28.0, -1.5],
            [-6.0, 28.0, -1.5],
            [-6.0, 2.0, -1.5],
        ])
        np.testing.assert_allclose(points, expected)
        self.assertTrue(mission.return_to_origin)
        self.assertTrue(mission.face_path)


if __name__ == "__main__":
    unittest.main()
