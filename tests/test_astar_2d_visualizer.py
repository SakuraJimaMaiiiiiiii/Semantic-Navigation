"""A*二维规划图保存测试。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Visual import AStar2DVisualizer, AStarVisualizationConfig


class AStar2DVisualizerTest(unittest.TestCase):
    def test_saves_interval_local_plots_and_global_plot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = AStar2DVisualizer(
                AStarVisualizationConfig(
                    local_save_interval=1.0,
                    image_dpi=60,
                    output_root=temporary_directory,
                ),
                run_timestamp="test_run",
            )
            states = np.zeros((10, 10, 7), dtype=np.int8)
            states[2, 2, 1] = 1  # 飞行平面上方0.4m，应从鸟瞰图排除
            states[4, 5, 5] = 1  # 飞行平面下方0.4m，补全柱体投影
            inflated = np.zeros_like(states, dtype=bool)
            inflated[1:4, 1:4, 1] = True
            inflated[3:6, 4:7, 5] = True
            local_snapshot = SimpleNamespace(
                timestamp=1.0,
                origin_ned=np.asarray([-1.0, -1.0, -0.7]),
                resolution=0.2,
                states=states,
                inflated_occupied=inflated,
            )
            plan = SimpleNamespace(
                status="avoid",
                speed_limit=0.5,
                local_target_ned=np.asarray([0.8, 0.4, 0.0]),
                path_ned=np.asarray([
                    [0.0, 0.0, 0.0],
                    [0.4, 0.3, 0.0],
                    [0.8, 0.4, 0.0],
                ]),
            )

            self.assertTrue(visualizer.submit_local(
                local_snapshot,
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                plan,
                inflation_radius=0.3,
                now=0.0,
            ))
            self.assertFalse(visualizer.submit_local(
                local_snapshot,
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                plan,
                inflation_radius=0.3,
                now=0.5,
            ))
            self.assertTrue(visualizer.submit_local(
                local_snapshot,
                [0.2, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                plan,
                inflation_radius=0.3,
                now=1.1,
            ))

            occupied, unknown, projected_inflation = (
                visualizer._project_local_grid(
                    local_snapshot,
                    center_down=0.0,
                    above_height=0.25,
                    below_height=0.9,
                    inflation_radius=0.3,
                )
            )
            self.assertTrue(occupied[4, 5])
            self.assertFalse(occupied[2, 2])
            self.assertFalse(unknown[4, 5])
            self.assertTrue(projected_inflation[3, 4])
            self.assertFalse(projected_inflation[4, 5])
            self.assertFalse(projected_inflation[2, 1])

            global_snapshot = SimpleNamespace(
                resolution=0.5,
                indices=np.asarray([
                    [0, 0, -4],
                    [1, 0, -4],
                    [2, 0, -4],
                ]),
                log_odds=np.asarray([-1.0, -1.0, 1.0]),
                occupied_threshold=0.7,
                free_threshold=-0.2,
                trajectory_ned=np.asarray([
                    [0.25, 0.25, -1.75],
                    [1.25, 0.25, -1.75],
                ]),
            )
            self.assertTrue(visualizer.submit_global(
                global_snapshot,
                [1.25, 0.25, -1.75],
                [0.25, 0.25, -1.75],
                [
                    [1.25, 0.25, -1.75],
                    [0.25, 0.25, -1.75],
                ],
                above_height=0.25,
                below_height=0.9,
            ))
            visualizer.close()
            visualizer.raise_if_failed()

            output_directory = Path(temporary_directory) / "test_run"
            images = sorted(output_directory.glob("*.png"))
            self.assertEqual(len(images), 3)
            self.assertEqual(
                len(list(output_directory.glob("local_*.png"))),
                2,
            )
            self.assertEqual(
                len(list(output_directory.glob("global_*.png"))),
                1,
            )
            self.assertTrue(all(image.stat().st_size > 0 for image in images))


if __name__ == "__main__":
    unittest.main()
