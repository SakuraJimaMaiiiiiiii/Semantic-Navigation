"""粗粒度颜色提取：背景隔离、低质量数据与主要色系。"""

import unittest
import numpy as np
from detections.vehicle_color import vehicle_color_scores


class VehicleColorTest(unittest.TestCase):
    def test_body_colors_ignore_background(self):
        for name, bgr in (("red", (20, 20, 200)), ("blue", (200, 20, 20)),
                          ("white", (220, 220, 220)), ("black", (35, 35, 35)),
                          ("gray", (120, 120, 120))):
            with self.subTest(name=name):
                image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
                mask = np.zeros((64, 64), dtype=bool)
                mask[8:56, 8:56] = True
                image[mask] = bgr
                scores = vehicle_color_scores(image, mask)
                self.assertEqual(max(scores, key=scores.get), name)
                self.assertAlmostEqual(sum(scores.values()), 1.0)

    def test_empty_dark_and_overexposed_masks_have_no_color_evidence(self):
        for value in (0, 255):
            image = np.full((32, 32, 3), value, dtype=np.uint8)
            self.assertEqual(vehicle_color_scores(image, np.ones((32, 32), bool)), {})
            self.assertEqual(vehicle_color_scores(image, np.zeros((32, 32), bool)), {})


if __name__ == "__main__":
    unittest.main()
