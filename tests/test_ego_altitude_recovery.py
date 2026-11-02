"""Small altitude tracking errors must not hit an artificial one-voxel plane wall."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
import numpy as np
from config.semantic_navigation_config import SemanticNavigationConfig
from planner.ego_local_planner import EgoLocalPlanner


@dataclass
class Snapshot:
    timestamp: float
    origin_ned: np.ndarray
    resolution: float
    states: np.ndarray
    inflated_occupied: np.ndarray


class AltitudeRecoveryTests(unittest.TestCase):
    current = np.array([-1.55, 3.23, -1.65])
    target = np.array([-1.451, .262, -1.496])

    def setup_case(self, obstacle=None):
        states=np.zeros((40,40,30),dtype=np.int8)
        if obstacle is not None:
            states[:,:,9]=obstacle
        snapshot=Snapshot(1.,np.array([-4.,-2.,-4.]),.2,states,states==1)
        planner=EgoLocalPlanner(SimpleNamespace(snapshot=lambda:snapshot))
        planner.set_navigation_plane(self.target[2],SemanticNavigationConfig())
        return planner,snapshot

    def test_recorded_start_can_rejoin_nominal_plane(self):
        planner,raw=self.setup_case()
        projected=planner._plane_snapshot(raw)
        cell=tuple(planner.world_to_grid(projected,self.current))
        self.assertEqual(projected.states[cell],0)
        self.assertTrue(planner._line_is_free_3d(projected,projected.inflated_occupied,
                                               self.current,self.target))
        # This regression fails with the previous single-free-layer mask.
        nominal=int(np.floor((self.target[2]-raw.origin_ned[2])/raw.resolution))
        self.assertNotEqual(cell[2],nominal)
        self.assertTrue(np.all(projected.inflated_occupied[:,:,:nominal-1]))
        np.testing.assert_array_equal(raw.states,np.zeros_like(raw.states))
        self.assertEqual(planner.navigation_altitude,self.target[2])

    def test_ceiling_or_unknown_body_band_remains_blocked(self):
        for obstacle in (1,-1):
            with self.subTest(obstacle=obstacle):
                planner,raw=self.setup_case(obstacle)
                projected=planner._plane_snapshot(raw)
                cell=tuple(planner.world_to_grid(projected,self.current))
                self.assertTrue(projected.inflated_occupied[cell])
                self.assertFalse(planner._line_is_free_3d(projected,projected.inflated_occupied,
                                                        self.current,self.target))

    def test_horizontal_wall_still_blocks_recovery(self):
        planner,raw=self.setup_case()
        raw.states[:,15,:]=1
        projected=planner._plane_snapshot(raw)
        self.assertFalse(planner._line_is_free_3d(projected,projected.inflated_occupied,
                                                self.current,self.target))
