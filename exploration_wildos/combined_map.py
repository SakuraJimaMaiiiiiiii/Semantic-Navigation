"""Compose the saved semantic map and static navigation graph into one page."""
import html
import math
import re
from .graph_controls import CONTROLS, SCRIPT as INSPECTOR_SCRIPT


STYLE = '''<style id="wildos-combined-style">
.map-panel{margin:24px 0;background:#fff;border:1px solid #cbd5e1;border-radius:12px;overflow:hidden}
.map-panel header{padding:16px 20px;border-bottom:1px solid #e2e8f0;background:#f8fafc}
.map-panel h2{margin:0 0 8px;font-size:21px}.map-panel p{margin:6px 0;color:#475569}
.map-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}
.map-toolbar button{padding:7px 14px;border:1px solid #cbd5e1;border-radius:6px;background:white;cursor:pointer;font:inherit}
.map-toolbar button:hover{background:#e0f2fe}.map-toolbar output{min-width:55px;text-align:center;font-variant-numeric:tabular-nums}
.map-canvas{height:clamp(420px,72vh,900px);background:#e3e6e9;overflow:hidden}
#graph-panel .map-canvas,#graph-panel .map-canvas svg{background:#b0b0b0}
.map-panel .map-canvas svg{display:block;width:100%;height:100%;border:0;cursor:grab;touch-action:none;user-select:none}
.map-panel .map-canvas svg.dragging{cursor:grabbing}.map-layers{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:14px}
.map-layers label{cursor:pointer}.map-hint{font-size:14px}body{max-width:1800px;margin:20px auto;padding:0 18px}
.graph-inspector{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:12px;font-size:14px}.graph-inspector select,.graph-inspector button{padding:6px;border:1px solid #cbd5e1;border-radius:5px;background:white;font:inherit}
@media(max-width:640px){.map-panel header{padding:12px}.map-canvas{height:60vh;min-height:350px}}
</style>'''

SCRIPT = '<script id="wildos-combined-controls">'+INSPECTOR_SCRIPT+'''
(() => {
  document.querySelectorAll('.map-panel').forEach(panel => {
    const svg = panel.querySelector('svg');
    const original = svg.getAttribute('viewBox').trim().split(/[ ,]+/).map(Number);
    let box = original.slice(), drag = null, moved = false, userChanged = false;
    const readout = panel.querySelector('output');
    function render(){svg.setAttribute('viewBox',box.join(' '));readout.value=Math.round(original[2]/box[2]*100)+'%';}
    function refreshGraphBounds(){
      if(panel.id!=='graph-panel')return;
      // The lower panel has its own geometry. The upper map's old viewBox
      // cannot be the zoom limit after graph growth or a frame transform.
      const bounds=svg.getBBox();
      if(![bounds.x,bounds.y,bounds.width,bounds.height].every(Number.isFinite)||bounds.width<=0||bounds.height<=0)return;
      const pad=Math.max(.5,.04*Math.max(bounds.width,bounds.height));
      original.splice(0,4,bounds.x-pad,bounds.y-pad,bounds.width+2*pad,bounds.height+2*pad);
    }
    function fitAll(){userChanged=false;refreshGraphBounds();box=original.slice();render();}
    svg._wildosUpdateBounds=values=>{
      if(values.length!==4||!values.every(Number.isFinite)||values[2]<=0||values[3]<=0)return;
      original.splice(0,4,...values);refreshGraphBounds();if(!userChanged)box=original.slice();render();
    };
    function local(x,y){return new DOMPoint(x,y).matrixTransform(svg.getScreenCTM().inverse());}
    function zoom(factor,anchor){
      userChanged=true;
      const next=Math.max(.02,Math.min(32,original[2]/box[2]*factor));
      const scale=original[2]/next/box[2];
      const point=anchor||{x:box[0]+box[2]/2,y:box[1]+box[3]/2};
      box=[point.x+(box[0]-point.x)*scale,point.y+(box[1]-point.y)*scale,box[2]*scale,box[3]*scale];render();
    }
    panel.querySelectorAll('[data-map-action]').forEach(button=>button.onclick=()=>{
      if(button.dataset.mapAction==='reset'){fitAll();}
      else zoom(button.dataset.mapAction==='in'?1.25:1/1.25);
    });
    svg.addEventListener('wheel',event=>{event.preventDefault();zoom(Math.exp(-Math.max(-100,Math.min(100,event.deltaY))*.003),local(event.clientX,event.clientY));},{passive:false});
    svg.addEventListener('dblclick',event=>{event.preventDefault();fitAll();});
    svg.addEventListener('pointerdown',event=>{
      if(event.button!==0||drag)return;
      drag={id:event.pointerId,x:event.clientX,y:event.clientY,box:box.slice(),inverse:svg.getScreenCTM().inverse()};moved=false;
    });
    svg.addEventListener('pointermove',event=>{
      if(!drag||event.pointerId!==drag.id)return;
      if(!moved&&Math.hypot(event.clientX-drag.x,event.clientY-drag.y)<4)return;
      if(!moved){userChanged=true;moved=true;svg.setPointerCapture(event.pointerId);svg.classList.add('dragging');}
      const a=new DOMPoint(drag.x,drag.y).matrixTransform(drag.inverse);
      const b=new DOMPoint(event.clientX,event.clientY).matrixTransform(drag.inverse);
      box=[drag.box[0]-(b.x-a.x),drag.box[1]-(b.y-a.y),drag.box[2],drag.box[3]];render();
    });
    const release=()=>{drag=null;svg.classList.remove('dragging');};
    window.addEventListener('pointerup',release);window.addEventListener('pointercancel',release);
    svg.addEventListener('lostpointercapture',release);
    svg.addEventListener('click',event=>{if(moved){event.stopImmediatePropagation();event.preventDefault();moved=false;}},true);
    panel.querySelectorAll('[data-graph-layer]').forEach(input=>input.onchange=()=>{
      svg.querySelector('#graph-'+input.dataset.graphLayer).style.display=input.checked?'':'none';
    });
    wireGraphInspector(svg,panel);
    fitAll();
  });
})();</script>'''


def compose_map_html(exploration_html, graph_html, graph, home, initial_yaw):
    """Preserve original semantic controls, transform graph to the same display frame."""
    if 'id="wildos-combined-style"' in exploration_html:
        raise ValueError('Already combined; use the saved original exploration HTML')
    # Recolor saved vehicle footprints in their ORIGINAL display frame. Never
    # replace corrected upper-panel coordinates with raw graph coordinates.
    vehicle_ids = {str(n['instance_id']) for n in graph.get('semantic_nodes', [])}
    def recolor(match):
        tag = match.group()
        identity = re.search(r'data-instance-id="([^"]+)"', tag)
        if 'semantic-footprint' not in tag or identity is None or html.unescape(identity[1]) not in vehicle_ids:
            return tag
        tag = re.sub(r'(?<![-\w])fill="[^"]*"', 'fill="#f97316"', tag)
        tag = re.sub(r'fill-opacity="[^"]*"', 'fill-opacity=".45"', tag)
        return tag
    exploration_html = re.sub(r'<polygon\b[^>]*>', recolor, exploration_html)
    top = re.search(r'<svg\b[^>]*>.*?</svg>', exploration_html, re.S | re.I)
    bottom = re.search(r'<svg\b([^>]*)>(.*?)</svg>', graph_html, re.S | re.I)
    if top is None or bottom is None:
        raise ValueError('Both source pages must contain a map SVG')
    view = re.search(r'viewBox="([^"]+)"', top.group())
    if not view:
        raise ValueError('Exploration SVG requires a viewBox')
    values = [float(v) for v in view[1].split()]
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or min(values[2:]) <= 0:
        raise ValueError('Invalid exploration viewBox')
    if len(home) != 3 or not all(math.isfinite(v) for v in [*home, initial_yaw]):
        raise ValueError('Invalid map display pose')
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    tx, ty = s*home[0]-c*home[1], c*home[0]+s*home[1]
    # Same home/initial-heading transform as navigation.map_view.local().
    contents = re.sub(r'(?<![\w:-])id="([^"]+)"', lambda m: f'id="graph-{html.escape(m[1], quote=True)}"', bottom[2])
    graph_svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view[1]}" role="img" aria-label="导航图结构">'
                 f'<g transform="matrix({-s} {-c} {c} {-s} {tx} {ty})">{contents}</g></svg>')

    def panel(identity, title, description, svg, layers=''):
        return f'''<section class="map-panel" id="{identity}"><header><h2>{title}</h2><p>{description}</p>
<div class="map-toolbar"><button type="button" data-map-action="out" aria-label="缩小">− 缩小</button>
<output aria-label="缩放比例">100%</output><button type="button" data-map-action="in" aria-label="放大">＋ 放大</button>
<button type="button" data-map-action="reset">适应全图</button><span class="map-hint">滚轮缩放 · 拖动平移 · 双击复位</span></div>{layers}</header>
<div class="map-canvas">{svg}</div></section>'''

    labels = [('raw','净空不足区（深灰）'),('safe','安全自由区域'),('obstacles','障碍（深灰）'),('uncovered','未覆盖区域'),
              ('frontiers','前沿点 / 前沿节点紫环'),('vehicle-constraints','车辆安全约束排除区'),('edges','连接边'),('nodes','节点'),('vehicles','车辆（橙色方形及观测轮廓）'),('semantic-edges','车辆关联（虚线，非通行边）')]
    layers = '<div class="map-layers">'+''.join(
        f'<label><input type="checkbox" data-graph-layer="{key}" {"" if key in {"semantic-edges", "uncovered"} else "checked"}>{label}</label>' for key,label in labels)+'</div>'+CONTROLS
    navigation = graph.get('navigation') or {}
    chain = ' → '.join((navigation.get('route') or {}).get('node_path',[]))
    layers += '<p data-navigation-status="">'+html.escape(
        f"图导航 {navigation.get('target_id') or ''} · {navigation.get('phase','等待目标')} · {chain}；紫色：图路径，橙色虚线：接入段，青色：当前航段，红色：实际轨迹。")+'</p>'
    sections = panel('exploration-panel','探索地图','查看探索航点、物体标注和占据地图。',top.group())
    profile_label = '论文对齐模式 · ' if graph.get('algorithm_profile')=='paper_aligned' else ''
    if graph.get('edge_sparsification'):
        profile_label = '论文采样 + 稀疏连边 · '
    if graph.get('node_compaction'):
        profile_label = '紧凑导航图 · '
    sections += panel('graph-panel','导航图结构',
                      f'{profile_label}{len(graph["nodes"])} 个导航节点 · {len(graph.get("semantic_nodes", []))} 辆车辆 · {len(graph["edges"])} 条边 · {len(graph["components"])} 个连通分量 · {len(graph.get("frontiers", []))} 个前沿点（紫点），紫环为前沿节点。底图：白色安全、深灰色障碍及净空不足、灰色未知。橙色为车辆观测轮廓的规划禁入投影，所有高度均禁止穿越；框外余量为深灰。规划约束不会回写车辆定位。车辆关联可单独开启。与上图采用相同朝向和显示范围；两图独立缩放。',graph_svg,layers)
    result = exploration_html[:top.start()]+sections+exploration_html[top.end():]
    # The archived map auto-refreshed every 2 seconds, resetting zoom/pan.
    result = re.sub(r'<meta\b(?=[^>]*http-equiv\s*=\s*[\"\']?refresh\b)[^>]*>', '', result, flags=re.I)
    result = result.replace('<title>语义探索地图</title>','<title>探索地图与导航图结构</title>')
    # Original generator omits explicit head/body; style and script still belong to this single document.
    return result.replace('</html>', STYLE+SCRIPT+'</html>')
