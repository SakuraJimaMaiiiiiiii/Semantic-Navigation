import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from navigation.map_view import render_map


class MultiViewMapTests(unittest.TestCase):
    def test_html_uses_same_center_and_orientation_as_semantic_record(self):
        record = dict(runtime_instance_id='vehicle_01', class_name='car',
                      position_world=[8., 3., -.5], position_status='multi_surface_estimate',
                      observed_oriented_box=dict(center=[8., 3., -.5], size=[4., 2., 1.], yaw=np.pi/4),
                      bbox_3d=dict(min=[5., 0., -1.], max=[11., 6., 0.]))
        memory = SimpleNamespace(initial_yaw=0., home=np.zeros(3), records={'vehicle_01':record},
                                 places={}, session_id='test')
        snapshot = SimpleNamespace(indices=np.empty((0,3), dtype=int), log_odds=np.empty(0),
                                   resolution=.2, free_threshold=-1., occupied_threshold=1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'map.html'
            render_map(path, snapshot, memory)
            text = path.read_text(encoding='utf-8')
        match = re.search(r'<polygon class="semantic-footprint"[^>]*points="([^"]+)"', text)
        self.assertIsNotNone(match)
        corners = np.array([[float(v) for v in pair.split(',')] for pair in match[1].split()])
        np.testing.assert_allclose(corners.mean(0), [3., -8.])
        np.testing.assert_allclose(np.linalg.norm(np.roll(corners,-1,axis=0)-corners, axis=1), [4.,2.,4.,2.])
