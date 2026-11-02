"""Standalone top-down map with semantic labels, selected goal and planned route."""

import html
import math
from .atomic_file import atomic_write_text
import numpy as np
from mapping.semantic_voxels import project_semantic_voxels, semantic_height_mask


def render_map(
    path,
    snapshot,
    memory,
    selected=None,
    approach=None,
    route=None,
    vehicle=None,
    state="exploring",
    navigation_config=None,
    exploration_waypoints=None,
    show_place_links=True,
):
    c, s = math.cos(memory.initial_yaw), math.sin(memory.initial_yaw)

    def local(points):
        delta = np.asarray(points, dtype=float).reshape(-1, 3) - memory.home
        return np.column_stack(
            (-s * delta[:, 0] + c * delta[:, 1], -(c * delta[:, 0] + s * delta[:, 1]))
        )

    records = list(memory.records.values())
    # A target selected before an instance merge still refers to the survivor.
    if selected not in memory.records:
        selected = next((record["runtime_instance_id"] for record in records
                         if selected in record.get("merged_from", [])), selected)
    from types import SimpleNamespace
    from config.semantic_navigation_config import SemanticNavigationConfig
    from .route import NavigationGrid

    config = navigation_config or SemanticNavigationConfig()
    altitude = getattr(memory, "flight_altitude", memory.home[2] - 1.5)
    # Same voxel-inclusive vertical band as routing. Show raw occupancy;
    # horizontal vehicle clearance remains a planner constraint.
    grid = NavigationGrid.from_global(
        snapshot, altitude, SimpleNamespace(vertical_half_extent=config.vertical_half_extent, clearance_radius=0.0)
    )
    def cells_xy(cells):
        points = np.array([(x + .5, y + .5, 0) for x, y in sorted(cells)], dtype=float).reshape(-1, 3)
        return local(points * snapshot.resolution)

    obstacle_xy = cells_xy(grid.occupied)
    free_xy = cells_xy(grid.free)
    semantic = project_semantic_voxels(snapshot, getattr(memory, "semantic_voxels", None), records)
    in_band = semantic_height_mask(snapshot, semantic["semantic_labels"], altitude,
                                   config.vertical_half_extent)
    # Use the export lookup table itself: integer voxel labels are snapshot-local
    # codes; the strings they resolve to are also semantic graph node IDs.
    instance_rows = []
    identities = semantic["semantic_instance_ids"]
    voxel_counts = np.bincount(semantic["semantic_labels"], minlength=len(identities))
    band_counts = np.bincount(semantic["semantic_labels"][in_band], minlength=len(identities))
    for label, identity in enumerate(identities[1:], start=1):
        escaped_id = html.escape(str(identity), quote=True)
        category = html.escape(str(semantic["semantic_class_names"][label]))
        note = ("尚无已确认的占据体素标签" if voxel_counts[label] == 0 else
                "仅在当前飞行高度带外" if band_counts[label] == 0 else "")
        instance_rows.append(
            f'<tr data-instance-id="{escaped_id}" data-semantic-label="{label}">'
            f'<td><button type="button" class="instance-select" data-instance-id="{escaped_id}">{escaped_id}</button></td>'
            f'<td>{category}</td><td>{label}</td><td>{voxel_counts[label]}</td>'
            f'<td>{band_counts[label]}</td><td>{note}</td></tr>'
        )
    semantic_cells = {}
    for row in np.flatnonzero(semantic["semantic_labels"]):
        cell = snapshot.indices[row]
        semantic_cells.setdefault((int(cell[0]), int(cell[1]), bool(in_band[row])), []).append(row)
    semantic_xy = local([((n+.5)*snapshot.resolution, (e+.5)*snapshot.resolution, 0.)
                         for n, e, _ in semantic_cells])
    object_xy = local([r["position_world"] for r in records])
    footprints = []
    for record in records:
        if record.get("position_status") == "multi_surface_estimate":
            box = record["observed_oriented_box"]
            c_box, s_box = math.cos(box["yaw"]), math.sin(box["yaw"])
            rotation = np.array([[c_box, -s_box], [s_box, c_box]])
            offsets = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
            corners = offsets*np.asarray(box["size"][:2])*.5 @ rotation.T + box["center"][:2]
            footprints.append(local(np.column_stack((corners, np.zeros(4)))))
            continue
        bounds = record.get("bbox_3d")
        if bounds is None:
            footprints.append(np.empty((0, 2)))
            continue
        low_corner, high_corner = bounds["min"], bounds["max"]
        footprints.append(local([
            (low_corner[0], low_corner[1], 0),
            (high_corner[0], low_corner[1], 0),
            (high_corner[0], high_corner[1], 0),
            (low_corner[0], high_corner[1], 0),
        ]))
    route_xy = local([] if route is None else route)
    waypoints = list(exploration_waypoints or [])
    waypoint_xy = local([w["position"] for w in waypoints])
    place_xy = local([node["position"] for node in memory.places.values()])
    extents = np.vstack(
        (np.zeros((1, 2)), free_xy, obstacle_xy, semantic_xy, object_xy, route_xy, place_xy, waypoint_xy,
         *footprints)
    )
    low, high = extents.min(axis=0) - 2, extents.max(axis=0) + 2
    span = np.maximum(high - low, 4)
    stroke = max(0.025, float(max(span)) / 0.8 / 1000)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{low[0]} {low[1]} {span[0]} {span[1]}" role="img" aria-label="Semantic map">',
        f'<rect x="{low[0]}" y="{low[1]}" width="{span[0]}" height="{span[1]}" fill="#e3e6e9"/>',
    ]
    size = float(snapshot.resolution)
    for coordinates, fill in ((free_xy, "#fff"), (obstacle_xy, "#707980")):
        for x, y in coordinates:
            parts.append(
                f'<rect x="{x-size/2}" y="{y-size/2}" width="{size}" height="{size}" transform="rotate({-math.degrees(memory.initial_yaw)} {x} {y})" fill="{fill}"/>'
            )
    category_colors = {"car": "#3182ce", "truck": "#3182ce", "bus": "#3182ce",
                       "van": "#3182ce", "column": "#9a6700", "signboard": "#be185d"}
    for ((north, east, inside), rows), (x, y) in zip(semantic_cells.items(), semantic_xy):
        labels = np.unique(semantic["semantic_labels"][rows])
        identities = semantic["semantic_instance_ids"][labels].tolist()
        categories = semantic["semantic_class_names"][labels].tolist()
        color = category_colors.get(categories[0], "#6d28d9") if len(set(categories)) == 1 else "#6d28d9"
        if selected in identities:
            color = "#e53935"
        layers = snapshot.indices[rows, 2]
        z_min = float(layers.min() * size)
        z_max = float((layers.max() + 1) * size)
        band = "当前高度带内" if inside else "高度带外，仅俯视投影"
        title = html.escape(
            f"语义占据体素 · {band} · " + "; ".join(f"{i} / {k}" for i, k in zip(identities, categories))
            + f" · XY格=[{north}, {east}] · NED Z范围=[{z_min:.2f}, {z_max:.2f}] m"
            + f" · 体素数={len(rows)} · 最少支持帧={int(np.min(semantic['semantic_support'][rows]))}"
        )
        extra_class = "" if inside else " outside-band"
        fill = color if inside else "none"
        dash = "" if inside else ' stroke-dasharray=".08 .06"'
        parts.append(
            f'<rect class="semantic-voxel{extra_class}" data-instance-ids="{html.escape(" ".join(identities), quote=True)}" '
            f'data-height-band="{"inside" if inside else "outside"}" '
            f'x="{x-size/2}" y="{y-size/2}" width="{size}" height="{size}" '
            f'transform="rotate({-math.degrees(memory.initial_yaw)} {x} {y})" '
            f'fill="{fill}" fill-opacity=".85" stroke="{color}" stroke-width="{stroke}"{dash}>'
            f'<title>{title}</title></rect>'
        )
    for identity, node in (memory.places.items() if show_place_links else ()):
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
    # P labels are route corners, distinct from W exploration destinations.
    for index, (x, y) in enumerate(route_xy[1:], start=1):
        point = np.asarray(route[index], dtype=float)
        title = html.escape(f"P{index} · A* 路线航点 · NED={point.round(3).tolist()}")
        parts.append(
            f'<g class="astar-waypoint"><title>{title}</title>'
            f'<circle cx="{x}" cy="{y}" r=".14" fill="#0077cc"/>'
            f'<text x="{x+.18}" y="{y-.18}" font-size=".3" fill="#0077cc">P{index}</text></g>'
        )
    for record, (x, y), footprint in zip(records, object_xy, footprints):
        identity = record["runtime_instance_id"]
        chosen = identity == selected
        color = "#e53935" if chosen else "#00856b"
        position = np.asarray(record["position_world"], dtype=float)
        height = float(memory.home[2] - position[2])
        band_note = ""
        bounds = record.get("bbox_3d")
        if bounds is not None:
            if bounds["max"][2] < altitude - config.vertical_half_extent:
                band_note = " · 观测几何在当前飞行高度带上方"
            elif bounds["min"][2] > altitude + config.vertical_half_extent:
                band_note = " · 观测几何在当前飞行高度带下方"
            else:
                band_note = " · 观测几何与当前飞行高度带相交"
        title = html.escape(
            f"{identity} / {record['class_name']} · 估计锚点 NED={position.round(3).tolist()} m"
            f" · 距起点高度={height:.2f} m · 观测次数={record.get('observation_count', 0)}{band_note}"
            + (" · 位置待完善（局部车面）" if record.get('position_status') == 'partial_surface'
               else " · 多表面位置估计" if record.get('position_status') == 'multi_surface_estimate' else "")
            + (" · 原始识别ID=" + ", ".join(record['source_runtime_instance_ids'])
               if record.get('source_runtime_instance_ids') else "")
            + (" · 包围框为已观测车面范围，非完整车体尺寸"
               if record.get('geometry_representation') == 'observed_surface_union' else "")
            + (" · 门/消防柜分类冲突已按多帧证据归一"
               if record.get('classification_status') == 'cross_category_resolved' else "")
        )
        if len(footprint):
            coordinates = " ".join(f"{px},{py}" for px, py in footprint)
            is_vehicle = record['class_name'] in {'car', 'truck', 'bus', 'van'}
            footprint_style = (f'fill="#3b82f6" fill-opacity=".55" stroke="{"#e53935" if chosen else "#2563eb"}"'
                               if is_vehicle else
                               f'fill="none" stroke="{color}" stroke-opacity=".4" stroke-dasharray=".15 .12"')
            parts.append(
                f'<polygon class="semantic-footprint" data-instance-id="{html.escape(identity, quote=True)}" '
                f'points="{coordinates}" {footprint_style} '
                f'stroke-width="{stroke}"><title>{title}</title></polygon>'
            )
        parts.append(
            f'<circle class="semantic-object" data-instance-id="{html.escape(identity, quote=True)}" '
            f'cx="{x}" cy="{y}" r="{.4 if chosen else .18}" fill="{color}"><title>{title}</title></circle>'
        )
        parts.append(
            f'<text class="semantic-label" data-instance-id="{html.escape(identity, quote=True)}" '
            f'x="{x+.25}" y="{y-.25}" font-size=".32" fill="{color}">{html.escape(identity)}</text>'
        )
    waypoint_colors = {"active": "#f97316", "arrived": "#2563eb", "deferred": "#8b5cf6",
                       "blocked": "#dc2626", "interrupted": "#6b7280", "stopped": "#6b7280"}
    waypoint_status = {"active": "当前目标", "arrived": "已到达", "deferred": "优先观测物体，暂缓",
                       "blocked": "受阻", "interrupted": "中断", "stopped": "低增长结束"}
    for waypoint, (x, y) in zip(waypoints, waypoint_xy):
        number = html.escape(str(waypoint["number"]))
        status_name = waypoint_status.get(waypoint["status"], "未知")
        color = waypoint_colors.get(waypoint["status"], "#6b7280")
        failure = waypoint.get("failure") or {}
        failure_text = f" · 原因：{failure['reason']}" if failure.get("reason") else ""
        title = html.escape(f"W{waypoint['number']} · {status_name} · {waypoint['frontier']}{failure_text}")
        parts.append(
            f'<g class="exploration-waypoint"><title>{title}</title>'
            f'<circle cx="{x}" cy="{y}" r=".32" fill="{color}" stroke="white" stroke-width=".06"/>'
            f'<text x="{x+.4}" y="{y-.35}" font-size=".4" font-weight="bold" '
            f'fill="{color}" stroke="white" stroke-width=".07" paint-order="stroke">W{number}</text></g>'
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
    page = (f'<!doctype html><html lang="zh" data-map-id="{html.escape(memory.session_id, quote=True)}">'
            '<meta charset="utf-8"><meta http-equiv="refresh" content="2"><title>语义探索地图</title>'
            '<style>body{font-family:system-ui;margin:20px;background:#fafafa}svg{width:100%;height:80vh;border:1px solid #ddd}'
            'b{color:#d22}.outside-band{display:none}.show-outside-band .outside-band{display:inline}'
            '.semantic-object,.semantic-label,.semantic-footprint,.semantic-voxel,.instance-select{cursor:pointer}'
            'svg .instance-focus{stroke:#f59e0b;stroke-width:3px;vector-effect:non-scaling-stroke}'
            'svg text.instance-focus{stroke-width:0;fill:#b45309;font-weight:bold}'
            'table{border-collapse:collapse}th,td{padding:6px 12px;border:1px solid #ddd;text-align:left}'
            'tr.instance-focus{background:#fef3c7}.instance-select{font:inherit}#identity-status{font-weight:bold}</style>')
    progress = getattr(memory, "exploration", {}).get("coverage_progress", {})
    progress_text = ""
    if progress:
        progress_text = (f"<p>当前高度累计已知面积：{float(progress['observed_area_m2']):.1f} m²；"
                         f"地图增长评估窗口：{float(progress['window_seconds']):.0f} 秒；"
                         f"窗口内已到达航点：{int(progress['arrivals_without_meaningful_expansion'])}。"
                         f"窗口新增：{float(progress['window_new_area_m2']):.2f} m²；"
                         f"低增长门槛：{float(progress.get('low_gain_threshold_m2', 0)):.2f} m² / 配置窗口。"
                         "未提供全图边界，不将此指标当作整图覆盖率。</p>")
    if progress:
        stage_names = {"collecting": "收集窗口数据", "expanding": "地图仍在扩展",
                       "review": "低增长复查", "review_due": "低增长计时到期：检查剩余未访问区域，补充复查次数有限", "return_required": "旧版低增长复查到期"}
        remaining = progress.get("review_remaining_seconds")
        progress_text += f"<p>结束判定：{stage_names.get(progress.get('stage'), '评估中')}"
        if remaining is not None:
            progress_text += f"；复查剩余 {remaining:.0f} 秒"
        elif progress.get("stage") == "collecting":
            progress_text += f"；首次评估还需 {progress.get('window_remaining_seconds', 0):.0f} 秒"
        if progress.get("stage") == "review":
            progress_text += (f"；本轮复查新增 {progress.get('review_gain_m2', 0):.2f} m²，"
                              f"达到 {progress.get('review_resume_gain_m2', 0):.2f} m² 才恢复正常探索")
        progress_text += "。判据：完整窗口低增长即复查；复查到期仍无明显扩展，完成当前航点后结束。残余候选不延长倒计时。</p>"
    reason = getattr(memory, "exploration", {}).get("reason")
    if reason:
        reason_names = {"map_expansion_stalled": "地图增长停滞，探索已结束",
                        "map_coverage_saturated": "可探索区域趋于饱和，探索已结束",
                        "frontiers_exhausted": "可达前沿已耗尽，探索已结束"}
        progress_text += f"<p><strong>结束原因：{html.escape(reason_names.get(reason, str(reason)))}</strong>；返航及落地结果见上方状态。</p>"
    page += (
        f"<h2>语义探索地图</h2><p>状态：{status}　目标：<b>{selection}</b>　上方＝初始朝向，右方＝初始朝向的右侧。显示高度＝起点上方 {memory.home[2]-altitude:.2f} m（机身上下各 {config.vertical_half_extent:.2f} m）；浅灰＝未知或净空证据不足；白色＝该高度带已观测空闲（未计水平安全距离）；深灰＝该高度带障碍。物体圆点使用地图中的稳定 ID，表示多帧可见表面估计锚点的俯视投影；蓝色色块为车辆已观测包围框的填充投影，其他物体使用虚线框；均不代表完整物体轮廓或当前高度障碍栅格。悬停显示世界 NED 坐标（北、东、下），物体可能不在当前障碍显示高度带内。红色＝目标；黄色＝接近点；蓝线＝当前 A* 规划路径；P1、P2…＝该路线依次执行的航点（重新规划后重新编号）。W1、W2…＝探索目标航点及选择顺序（橙色＝当前目标，蓝色＝已到达，红色＝受阻，灰色＝中断）；鼠标悬停查看详情。每 2 秒刷新，地图数据随探索检查点更新。</p>"
        + progress_text
        + '<p>彩色占据格：蓝＝车辆，棕＝柱子，粉＝交通指示牌，紫＝其他或多类别；深灰障碍格尚无确定物体归属。'
          '悬停彩色格查看稳定 ID。彩色占据格只标注有多帧表面观测支持的占据体素；车辆蓝色填充仅用于显示，不写入导航占据数据。'
          '<label><input type="checkbox" id="show-outside">显示高度带外语义投影（彩色空心虚线，不代表当前高度障碍）</label></p>'
        + '<p id="identity-status" aria-live="polite">点击物体 ID 或彩色占据格，查看同一实例的对应标注。</p>'
          '<button type="button" id="clear-instance">取消高亮</button>'
        + "".join(parts)
        + '<details open><summary>物体 ID 对应表（HTML / 语义图 / 占据地图）</summary>'
          '<p>物体 ID 等于语义图 object 节点的 id，也等于占据文件 semantic_instance_ids 中的字符串。'
          '数字标签码只在当前快照内有效，通过查表映射到物体 ID；0 表示未分类。'
          '三维体素数统计已确认归属的占据体素，当前高度带数量可以为 0。'
          '点击只高亮查看，不会给无人机下发目标。W/P 航点编号不属于物体 ID。</p>'
          '<table id="instance-index"><thead><tr><th>物体 ID / 图节点 ID</th><th>类别</th>'
          '<th>当前体素标签码</th><th>三维占据体素数</th><th>当前高度带体素数</th><th>说明</th>'
          '</tr></thead><tbody>' + ''.join(instance_rows) + '</tbody></table></details>'
        + "<details open><summary>探索航点记录（按选择顺序）</summary><ol>"
        + "".join(f"<li>W{html.escape(str(w['number']))}：{waypoint_status.get(w['status'], '未知')} — {html.escape(str(w['frontier']))}</li>" for w in waypoints)
        + "</ol></details>"
        + '<script>const toggle=document.getElementById("show-outside");'
          'try{toggle.checked=localStorage.getItem("semantic-outside-band")==="1";}catch(e){}'
          'function showOutside(){document.documentElement.classList.toggle("show-outside-band",toggle.checked);}'
          'toggle.addEventListener("change",()=>{showOutside();try{localStorage.setItem("semantic-outside-band",toggle.checked?"1":"0");}catch(e){}});'
          'showOutside();</script>'
        + '''<script>
const instanceRows = Array.from(document.querySelectorAll('#instance-index tbody tr'));
const currentIds = new Set(instanceRows.map(row => row.dataset.instanceId));
const identityMarks = Array.from(document.querySelectorAll('[data-instance-id],[data-instance-ids]'));
const focusKey = 'semantic-instance:' + document.documentElement.dataset.mapId;
const statusLine = document.getElementById('identity-status');
let focusedId = '';
function instanceIds(element) {
    return element.dataset.instanceId ? [element.dataset.instanceId] :
        (element.dataset.instanceIds || '').split(' ').filter(Boolean);
}
function focusInstance(identity) {
    focusedId = currentIds.has(identity) ? identity : '';
    identityMarks.forEach(element => element.classList.toggle('instance-focus',
        Boolean(focusedId) && instanceIds(element).includes(focusedId)));
    statusLine.textContent = focusedId ? '查看：' + focusedId +
        '（同名语义图节点、物体标记与占据标签；高度带外的格子需勾选上方显示选项）' :
        '点击物体 ID 或彩色占据格，查看同一实例的对应标注。';
    try { localStorage.setItem(focusKey, focusedId); } catch(e) {}
}
document.addEventListener('click', event => {
    const element = event.target.closest('.instance-select,.semantic-object,.semantic-label,.semantic-footprint,.semantic-voxel');
    if (!element) return;
    const ids = instanceIds(element).filter(id => currentIds.has(id));
    // A projected cell can contain several instances at different heights.
    // Repeated clicks cycle through its IDs; the table selects an exact one.
    const next = ids[(ids.indexOf(focusedId) + 1) % ids.length];
    focusInstance(next);
});
document.getElementById('clear-instance').addEventListener('click', () => focusInstance(''));
try { focusInstance(localStorage.getItem(focusKey)); } catch(e) {}
</script></html>'''
    )
    atomic_write_text(path, page)
