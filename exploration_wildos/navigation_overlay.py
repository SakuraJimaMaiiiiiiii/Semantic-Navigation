"""Render planned graph segments separately from measured flight positions."""
import html


def navigation_svg(state):
    if not state:
        return ''
    route = state.get('route') or {}
    points = route.get('points_ned', [])
    parts = ['<g data-navigation-overlay="" pointer-events="none">']
    for i,(a,b) in enumerate(zip(points,points[1:])):
        connector = i == 0 or i == len(points)-2
        color = '#d97706' if connector else '#c026d3'
        dash = ' stroke-dasharray=".18 .12"' if connector else ''
        parts.append(f'<line data-planned-segment="{i}" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" stroke-width=".10"{dash}/>')
    for identity,p in zip(route.get('node_path',[]),points[1:-1]):
        parts.append(f'<circle cx="{p[0]}" cy="{p[1]}" r=".18" fill="none" stroke="#c026d3" stroke-width=".05"><title>{html.escape(identity)}</title></circle>')
    trace = state.get('trace_ned',[])
    if trace:
        coords=' '.join(f'{p[0]},{p[1]}' for p in trace)
        parts.append(f'<polyline data-measured-trace="" points="{coords}" fill="none" stroke="#ef4444" stroke-width=".055"/>')
    active = state.get('active_segment')
    if active:
        a,b=active
        parts.append(f'<line data-active-segment="" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#06b6d4" stroke-width=".12"/>')
    parts.append('</g>')
    return ''.join(parts)


CLIENT = r'''
function drawNavigation(group,state){
  group.querySelectorAll('[data-navigation-overlay]').forEach(e=>e.remove());
  if(!state)return;
  const ns='http://www.w3.org/2000/svg', layer=document.createElementNS(ns,'g');
  layer.setAttribute('data-navigation-overlay','');layer.setAttribute('pointer-events','none');
  function add(tag,attrs){const e=document.createElementNS(ns,tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));layer.append(e);return e;}
  function line(a,b,color,width,extra={}){return add('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':width,...extra});}
  const route=state.route||{}, points=route.points_ned||[];
  for(let i=0;i+1<points.length;i++){
    const connector=i===0||i===points.length-2;
    line(points[i],points[i+1],connector?'#d97706':'#c026d3','.10',
      {'data-planned-segment':i,...(connector?{'stroke-dasharray':'.18 .12'}:{})});
  }
  for(let i=1;i+1<points.length;i++)add('circle',{cx:points[i][0],cy:points[i][1],r:'.18',fill:'none',stroke:'#c026d3','stroke-width':'.05'});
  const trace=state.trace_ned||[];
  if(trace.length)add('polyline',{'data-measured-trace':'',points:trace.map(p=>p[0]+','+p[1]).join(' '),fill:'none',stroke:'#ef4444','stroke-width':'.055'});
  if(state.active_segment)line(...state.active_segment,'#06b6d4','.12',{'data-active-segment':''});
  group.insertBefore(layer,group.querySelector('[data-live-robot]'));
}
'''
