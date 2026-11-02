import unittest
import numpy as np
from navigation.path_smoothing import rounded_path, tracking_points


class CornerTests(unittest.TestCase):
    def test_dense_samples_do_not_become_dense_control_targets(self):
        dense = [(np.array([x,0.,0.]), True, True) for x in np.arange(.1,1.31,.1)]
        dense.append((np.array([2.,0.,0.]), False, False))
        route = tracking_points(np.zeros(3), dense, lambda a,b:True)
        self.assertLess(len(route), len(dense)/2)
        self.assertTrue(all(np.linalg.norm(b[0]-a[0]) >= .3-1e-9
                            for a,b in zip(route,route[1:])))
        np.testing.assert_allclose(route[-1][0], [2,0,0])
        self.assertFalse(route[-1][1])

    def test_spacing_never_overrides_chord_safety(self):
        dense = [(np.array([x,0.,0.]), True, True) for x in [.1,.2,.3,.4]]
        route = tracking_points(np.zeros(3), dense,
                                lambda a,b:np.linalg.norm(b-a) <= .11)
        self.assertEqual(len(route),len(dense))

    def test_open_corner_is_rounded_with_preserved_destination(self):
        points = np.array([[0.,0.,-1.5], [4.,0.,-1.5], [4.,4.,-1.5]])
        route = rounded_path(points, lambda a,b:True)
        self.assertGreater(len(route), 2)
        self.assertTrue(all(flow and turn for _,flow,turn in route[:-1]))
        np.testing.assert_allclose(route[-1][0], points[-1])
        self.assertFalse(route[-1][1])
        self.assertTrue(all(p[2] == -1.5 for p,_,_ in route))
        self.assertFalse(any(np.allclose(p,points[1]) for p,_,_ in route))

    def test_obstacle_on_rounded_chord_preserves_stopping_corner(self):
        calls = []
        def free(a,b):
            calls.append((a,b))
            return not (b[0] < 4 and b[1] > 0)
        route = rounded_path([[0,0,0],[4,0,0],[4,4,0]], free)
        self.assertEqual(len(route),2)
        np.testing.assert_allclose(route[0][0], [4,0,0])
        self.assertFalse(route[0][1])
        self.assertGreater(len(calls),1)

    def test_uturn_still_stops(self):
        route = rounded_path([[0,0,0],[4,0,0],[0,0,0]], lambda a,b:True)
        self.assertEqual(len(route),2)
        self.assertFalse(route[0][1])

    def test_straight_waypoint_flows(self):
        route = rounded_path([[0,0,0],[4,0,0],[8,0,0]], lambda a,b:True)
        self.assertTrue(route[0][1])
        self.assertFalse(route[-1][1])


if __name__ == '__main__': unittest.main()
