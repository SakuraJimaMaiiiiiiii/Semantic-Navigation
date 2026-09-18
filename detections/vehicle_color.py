"""掩膜内部的粗粒度车漆颜色投票；不宣称分离出真实车身部件。"""

import cv2
import numpy as np


def vehicle_color_scores(image, foreground):
    mask = np.asarray(foreground, dtype=np.uint8)
    if mask.shape != image.shape[:2]:
        return {}
    rows, _ = np.nonzero(mask)
    if rows.size < 32:
        return {}
    # 去掩膜边缘和顶部/底部，降低背景、轮胎、车顶高光的占比。
    mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
    lower, upper = int(rows.min()), int(rows.max()) + 1
    margin = max(1, int((upper - lower) * 0.15))
    mask[:lower + margin] = 0
    mask[upper - margin:] = 0
    pixels = image[::2, ::2][mask[::2, ::2].astype(bool)]
    if len(pixels) < 16:
        return {}
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, saturation, value = hsv.T
    # 剔除极暗无信息像素及接近过曝的白色高光。
    valid = (value >= 20) & ~((value >= 250) & (saturation < 25))
    if np.count_nonzero(valid) < 16:
        return {}
    hue, saturation, value = hue[valid], saturation[valid], value[valid]
    labels = np.full(len(hue), "gray", dtype="<U8")
    chromatic = (saturation >= 60) & (value >= 55)
    labels[value < 55] = "black"
    labels[(saturation < 60) & (value >= 190)] = "white"
    for name, selected in (
        ("red", (hue < 10) | (hue >= 170)),
        ("orange", (hue >= 10) & (hue < 23)),
        ("yellow", (hue >= 23) & (hue < 35)),
        ("green", (hue >= 35) & (hue < 85)),
        ("blue", (hue >= 85) & (hue < 130)),
        ("purple", (hue >= 130) & (hue < 170)),
    ):
        labels[chromatic & selected] = name
    labels[chromatic & (hue >= 10) & (hue < 30) & (value < 150)] = "brown"
    names, counts = np.unique(labels, return_counts=True)
    return {str(name): float(count / len(labels)) for name, count in zip(names, counts)}
