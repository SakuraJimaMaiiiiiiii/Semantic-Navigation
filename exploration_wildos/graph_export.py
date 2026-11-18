"""Portable JSON and self-contained SVG/HTML; no network or plotting dependency."""
import html
import json
from pathlib import Path
from .graph_controls import CONTROLS, SCRIPT


def export_graph(directory, adapter, graph, validation):
    directory = Path(directory)
    # Refuse any existing destination, including partially generated earlier runs.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'graph.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (directory / 'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False), encoding='utf-8')
    (directory / 'graph.html').write_text(render_graph_html(adapter, graph, validation), encoding='utf-8')


def render_graph_html(adapter, graph, validation):
    """Render an immutable snapshot without filesystem side effects."""
    grid = adapter.grid
    r = grid.resolution
    cells = sorted(grid.known)
    if cells:
        low = [min(c[i] for c in cells)*r+grid.origin[i]-r for i in range(2)]
        high = [(max(c[i] for c in cells)+1)*r+grid.origin[i]+r for i in range(2)]
    else:
        low, high = [-1., -1.], [1., 1.]
    layers = []
    hidden_layers = {'semantic-edges', 'uncovered'}

    owner_by_cell = {tuple(row['cell']): row['owners'] for row in graph.get('vehicle_constraints', {}).get('cell_owners', [])}

    def rectangles(name, selected, fill):
        visibility = ' style="display:none"' if name in hidden_layers else ''
        parts = [f'<g id="{name}" fill="{fill}"{visibility}>']
        for x, y in sorted(selected):
            owners = owner_by_cell.get((x,y), []) if name == 'vehicle-constraints' else []
            owner_text = html.escape(', '.join(o['vehicle_id'] + ':' + o['zone'] for o in owners), quote=True)
            attrs = f' data-vehicle-owners="{owner_text}"' if owners else ''
            parts.append(f'<rect{attrs} x="{x*r+grid.origin[0]:.6f}" y="{y*r+grid.origin[1]:.6f}" width="{r:.6f}" height="{r:.6f}"/>')
        parts.append('</g>')
        layers.append(''.join(parts))

    rectangles('raw', adapter.raw_free-grid.free, '#707980')
    rectangles('safe', grid.free, '#ffffff')
    rectangles('obstacles', grid.occupied, '#707980')
    rectangles('vehicle-constraints', graph.get('vehicle_constraints', {}).get('occupied_cells', graph.get('vehicle_constraints', {}).get('excluded_safe_cells', [])), '#707980')
    rectangles('uncovered', graph['diagnostics']['uncovered_safe_cells'], '#fcd34d')
    parts = ['<g id="radii" fill="none" stroke="#14b8a6" stroke-opacity=".65" stroke-width=".025" pointer-events="none">']
    for node in graph['nodes']:
        x, y, _ = node['position_ned']
        parts.append(f'<circle data-node-radius="{node["id"]}" data-free-radius="{node["free_radius"]}" data-explored-radius="{node["explored_radius"]}" style="display:none" cx="{x}" cy="{y}" r="{node["free_radius"]}"/>')
    layers.append(''.join(parts)+'</g>')
    parts = ['<g id="edges" stroke="#2563eb" stroke-width=".025" pointer-events="none">']
    for edge in graph['edges']:
        a, b = (graph['nodes'][edge[k]]['position_ned'] for k in ('source', 'target'))
        parts.append(f'<line data-edge-source="{edge["source"]}" data-edge-target="{edge["target"]}" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>')
    layers.append(''.join(parts)+'</g>')
    colors = ['#059669', '#e11d48', '#9333ea', '#ea580c', '#0891b2']
    parts = ['<g id="nodes">']
    for component, ids in enumerate(graph['components']):
        for identity in ids:
            node = graph['nodes'][identity]
            x, y, _ = node['position_ned']
            parts.append(f'<circle data-node-id="{identity}" data-stable-id="{html.escape(str(node.get("stable_id", identity)), quote=True)}" data-explored-radius="{node["explored_radius"]}" data-frontier-count="{len(node.get("frontier_ids", []))}" data-above-vehicles="{html.escape(", ".join(node.get("above_vehicle_ids", [])), quote=True)}" data-free-radius="{node["free_radius"]}" cx="{x}" cy="{y}" r=".10" fill="{colors[component%len(colors)]}"><title>node {identity}; component {component}; free radius {node["free_radius"]:.3f} m</title></circle>')
    layers.append(''.join(parts)+'</g>')
    parts = ['<g id="frontiers" fill="#a855f7" pointer-events="none">']
    for frontier in graph.get('frontiers', []):
        x,y,_ = frontier['position_ned']
        parts.append(f'<rect x="{x-.06}" y="{y-.06}" width=".12" height=".12"/>')
    for node in graph['nodes']:
        if node.get('is_frontier'):
            x,y,_=node['position_ned']
            parts.append(f'<circle cx="{x}" cy="{y}" r=".16" fill="none" stroke="#a855f7" stroke-width=".035"/>')
    layers.append(''.join(parts)+'</g>')
    parts = ['<g id="semantic-edges" style="display:none" stroke="#f59e0b" stroke-width=".045" stroke-dasharray=".16 .12" pointer-events="none">']
    semantic = {node['id']:node for node in graph.get('semantic_nodes', [])}
    for edge in graph.get('semantic_edges', []):
        a = semantic[edge['source']]['position_ned']
        b = graph['nodes'][edge['target']]['position_ned']
        identity = html.escape(edge['source'], quote=True)
        color = '#d97706' if edge['ray_status']=='clear_to_surface' else '#94a3b8'
        parts.append(f'<line data-object-edge="{identity}" data-nav-target="{edge["target"]}" stroke="{color}" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"><title>{edge["ray_status"]}; 非通行边，未验证相机可见性</title></line>')
    layers.append(''.join(parts)+'</g>')
    parts = ['<g id="vehicles">']
    for node in semantic.values():
        x,y,_ = node['position_ned']
        low, high = node['bbox_3d']['min'], node['bbox_3d']['max']
        identity = html.escape(node['id'], quote=True)
        details = html.escape(f"{node['instance_id']} · {node['class_name']} · 颜色 {node['color']} · 置信度 {node['confidence']} · 关联 {node['association_count']} 个导航节点 · 观测位置 {node['position_ned']} · 已观测包围盒 {node['bbox_3d']} · 非通行节点；相机可见性未验证", quote=True)
        parts.append(f'<rect data-object-outline="{identity}" x="{low[0]}" y="{low[1]}" width="{high[0]-low[0]}" height="{high[1]-low[1]}" fill="#fdba74" fill-opacity="1" stroke="#ea580c" stroke-width=".045" pointer-events="none"/>')
        parts.append(f'<rect data-object-id="{identity}" data-object-info="{details}" x="{x-.22}" y="{y-.22}" width=".44" height=".44" fill="#f97316" stroke="#7c2d12" stroke-width=".04"><title>{details}</title></rect>')
    layers.append(''.join(parts)+'</g>')
    attachment = graph['robot_attachment']
    from .navigation_overlay import navigation_svg
    layers.append(navigation_svg(graph.get('navigation')))
    if attachment.get('position_ned'):
        x, y, _ = attachment['position_ned']
        layers.append(f'<circle data-snapshot-robot="" cx="{x}" cy="{y}" r=".16" fill="#111827"><title>Requested robot start</title></circle>')
        if attachment['node_id'] is not None:
            p = graph['nodes'][attachment['node_id']]['position_ned']
            layers.append(f'<line data-snapshot-robot="" x1="{x}" y1="{y}" x2="{p[0]}" y2="{p[1]}" stroke="#111827" stroke-width=".05" stroke-dasharray=".1 .1"/>')
    labels = [('raw', '净空不足区（深灰）'), ('safe', '安全自由区域'), ('obstacles', '障碍（深灰）'),
              ('frontiers', '前沿点 / 前沿节点紫环'), ('vehicle-constraints', '车辆安全约束排除区'), ('uncovered', '未被半径覆盖'), ('edges', '安全边'), ('nodes', '节点 / 连通分量'), ('vehicles', '车辆（橙色方形及观测轮廓）'), ('semantic-edges', '车辆关联（虚线，非通行边）')]
    controls = ' '.join(f'<label><input type="checkbox" {"" if key in hidden_layers else "checked"} data-layer="{key}">{value}</label>' for key, value in labels)
    stats = graph['diagnostics']
    summary = f"节点 {len(graph['nodes'])} · 边 {len(graph['edges'])} · 车辆 {len(graph.get('semantic_nodes', []))} · 连通分量 {len(graph['components'])} · 未覆盖安全格 {len(stats['uncovered_safe_cells'])}"
    source = html.escape(str(adapter.metadata.get('source', 'synthetic fixture')))
    dataset_label = html.escape(adapter.metadata.get('dataset_label', 'Saved map archive; provenance in graph.json'))
    stage_note = '增量快照图：紫点为未知侧前沿，紫环为关联节点；尚未接入实时探索或飞行。' if graph.get('incremental') else '静态快照图；尚未关联前沿，也未接入飞行。'
    if graph.get('navigation'):
        state=graph['navigation']
        chain=' → '.join((state.get('route') or {}).get('node_path',[]))
        stage_note=html.escape(f"图导航 {state.get('target_id') or ''} · {state.get('phase','')} · {chain}。紫色：图路径；橙色虚线：接入段；青色：当前航段；红色：实际轨迹。")
    content = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>WildOS 离线导航图</title>
<style>body{{font-family:system-ui;margin:24px;color:#172033;background:#f8fafc}}label{{display:inline-block;margin:6px}}#viewport{{overflow:auto;border:1px solid #94a3b8;height:72vh}}svg{{width:100%;height:100%;background:#b0b0b0}}small{{overflow-wrap:anywhere}}</style>
<h1>WildOS 单高度离线导航图</h1><p>{summary}</p><p>{dataset_label}</p><small>来源：{source}</small>
<p>高度 NED z={grid.altitude:g} m；分辨率 {r:g} m。横轴 North，纵轴 East（向下）。白色为安全区域，深灰色为障碍及净空不足区域，灰色为未知。橙色为车辆观测轮廓的规划禁入投影，所有高度均禁止穿越；框外安全余量保持深灰，归属车辆但不作为定位证据。车辆关联、黄色覆盖诊断默认隐藏；节点颜色区分连通分量。</p>
<p>{stage_note}安全检查：{'通过' if validation['passed'] else '失败'}。</p>
<div>{controls}</div>{CONTROLS}<label>缩放 <input id="zoom" type="range" min="1" max="5" step=".25" value="1"></label>
<div id="viewport"><svg id="map" viewBox="{low[0]} {low[1]} {high[0]-low[0]} {high[1]-low[1]}">{''.join(layers)}</svg></div>
<script>{SCRIPT}
wireGraphInspector(document.getElementById('map'),document);
document.querySelectorAll('[data-layer]').forEach(x=>x.onchange=()=>document.getElementById(x.dataset.layer).style.display=x.checked?'':'none');document.getElementById('zoom').oninput=e=>{{const s=document.getElementById('map');s.style.width=(e.target.value*100)+'%';s.style.height=(e.target.value*100)+'%';}};</script></html>'''
    return content
