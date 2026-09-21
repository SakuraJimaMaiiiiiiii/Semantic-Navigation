"""Standalone top-down map with semantic labels, selected goal and planned route."""

import html
import math
from pathlib import Path
import numpy as np


def render_map(
    path,
    snapshot,
    memory,
    selected=None,
    approach=None,
    route=None,
    vehicle=None,
    state="exploring",
):
    c, s = math.cos(memory.initial_yaw), math.sin(memory.initial_yaw)

    def local(points):
        delta = np.asarray(points, dtype=float).reshape(-1, 3) - memory.home
        return np.column_stack(
            (-s * delta[:, 0] + c * delta[:, 1], -(c * delta[:, 0] + s * delta[:, 1]))
        )

    records = list(memory.records.values())
    occupied = snapshot.indices[snapshot.log_odds >= snapshot.occupied_threshold]
    # Project only the navigation-height band, avoiding roof/floor filling the image.
    altitude = getattr(memory, "flight_altitude", memory.home[2] - 1.5)
    points = (occupied + 0.5) * snapshot.resolution
    points = points[np.abs(points[:, 2] - altitude) <= 0.8]
    points = points[:: max(1, int(math.ceil(len(points) / 15000)))]
    obstacle_xy = local(points)
    free_points = (
        snapshot.indices[snapshot.log_odds <= snapshot.free_threshold] + 0.5
    ) * snapshot.resolution
    free_points = free_points[
        np.abs(free_points[:, 2] - altitude) <= snapshot.resolution / 2
    ]
    free_points = free_points[:: max(1, int(math.ceil(len(free_points) / 20000)))]
    free_xy = local(free_points)
    object_xy = local([r["position_world"] for r in records])
    route_xy = local([] if route is None else route)
    place_xy = local([node["position"] for node in memory.places.values()])
    extents = np.vstack(
        (np.zeros((1, 2)), free_xy, obstacle_xy, object_xy, route_xy, place_xy)
    )
    low, high = extents.min(axis=0) - 2, extents.max(axis=0) + 2
    span = np.maximum(high - low, 4)
    stroke = max(0.025, float(max(span)) / 0.8 / 1000)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{low[0]} {low[1]} {span[0]} {span[1]}" role="img" aria-label="Semantic map">',
        f'<rect x="{low[0]}" y="{low[1]}" width="{span[0]}" height="{span[1]}" fill="#e3e6e9"/>',
    ]
    size = float(snapshot.resolution)
    for x, y in free_xy:
        parts.append(
            f'<rect x="{x-size/2}" y="{y-size/2}" width="{size}" height="{size}" transform="rotate({-math.degrees(memory.initial_yaw)} {x} {y})" fill="#fff"/>'
        )
    for x, y in obstacle_xy:
        parts.append(
            f'<rect x="{x-size/2}" y="{y-size/2}" width="{size}" height="{size}" transform="rotate({-math.degrees(memory.initial_yaw)} {x} {y})" fill="#707980"/>'
        )
    for identity, node in memory.places.items():
        parent = node["parent"]
        if parent is not None:
            a, b = local([node["position"], memory.places[parent]["position"]])
            parts.append(
                f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" stroke="#b8c3cc" stroke-width="{stroke}"/>'
            )
    if len(route_xy):
        coordinates = " ".join(f"{x},{y}" for x, y in route_xy)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="#0077cc" stroke-width="{stroke*3}"/>'
        )
    for record, (x, y) in zip(records, object_xy):
        identity = record["runtime_instance_id"]
        chosen = identity == selected
        color = "#e53935" if chosen else "#00856b"
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{.4 if chosen else .18}" fill="{color}"><title>{html.escape(identity+" / "+record["class_name"])}</title></circle>'
        )
        parts.append(
            f'<text x="{x+.25}" y="{y-.25}" font-size=".32" fill="{color}">{html.escape(identity)}</text>'
        )
    for position, color, label in (
        (memory.home, "#222", "HOME"),
        (approach, "#f59e0b", "GOAL"),
        (vehicle, "#7e22ce", "DRONE"),
    ):
        if position is not None:
            x, y = local([position])[0]
            parts.append(
                f'<circle cx="{x}" cy="{y}" r=".25" fill="{color}"/><text x="{x+.3}" y="{y}" font-size=".35">{label}</text>'
            )
    parts.append("</svg>")
    status = html.escape(state)
    selection = html.escape(selected or "未指定")
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>语义探索地图</title><style>body{font-family:system-ui;margin:20px;background:#fafafa}svg{width:100%;height:80vh;border:1px solid #ddd}b{color:#d22}</style>'
    page += (
        f"<h2>语义探索地图</h2><p>状态：{status}　目标：<b>{selection}</b>　上方＝初始朝向，右方＝初始朝向的右侧。浅灰＝未知；白色＝已观测自由区；深灰＝障碍。红色＝目标；黄色＝接近点；蓝线＝规划路径。每 5 秒刷新。</p>"
        + "".join(parts)
        + "</html>"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(path)
