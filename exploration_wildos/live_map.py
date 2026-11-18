"""Read-only loopback map viewer; rendering never accesses the flight runtime."""
import copy
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from .map_adapter import MapAdapter
from urllib.parse import urlsplit, parse_qs
from .graph_export import render_graph_html
from .navigation_overlay import CLIENT as NAVIGATION_CLIENT


class LiveMapServer:
    def __init__(self, map_path):
        self.map_path = Path(map_path)
        self.lock = threading.Lock()
        self.page = self.map_path.read_text(encoding="utf-8") if self.map_path.exists() else None
        self.payload = dict(version=-1, robot=None, status='等待首次构图')
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='WildOSMapRender')
        self.future = None
        self.submitted = -1
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlsplit(self.path).path
                if path == '/live.json':
                    with owner.lock:
                        payload = dict(owner.payload)
                        since = parse_qs(urlsplit(self.path).query).get('since', [''])[0]
                        if since == str(payload['version']):
                            payload.pop('svg', None)
                            payload.pop('positions', None)
                        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                    kind = 'application/json; charset=utf-8'
                elif path in ('/', '/map.html'):
                    with owner.lock:
                        page = owner.page
                    if page is None:
                        self.send_error(503)
                        return
                    # No full-page refresh: only panel contents are patched.
                    page = re.sub(r'<meta\b(?=[^>]*http-equiv\s*=\s*[\"\']?refresh\b)[^>]*>', '', page, flags=re.I)
                    page = page.replace('</html>', '<script>'+CLIENT+'</script></html>')
                    body, kind = page.encode(), 'text/html; charset=utf-8'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='WildOSMapHTTP')
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/map.html'

    def set_page(self, page):
        """Serve committed in-memory HTML; HTTP readers never hold map.html open."""
        with self.lock:
            self.page = page

    def publish(self, adapter, graph, validation, version):
        if version == self.submitted or (self.future is not None and not self.future.done()):
            return
        # GraphRouteGrid holds a bound flight-planner method. Copy only render
        # data, never deepcopy runtime handles/locks through that method.
        grid = adapter.grid
        detached = SimpleNamespace(known=set(grid.known), free=set(grid.free),
            occupied=set(grid.occupied), origin=list(grid.origin),
            resolution=grid.resolution, altitude=grid.altitude)
        snapshot = (MapAdapter(detached, copy.deepcopy(adapter.metadata), frozenset(adapter.raw_free)),
                    copy.deepcopy(graph), copy.deepcopy(validation))
        self.submitted = version

        def render():
            try:
                a, g, v = snapshot
                content = render_graph_html(a, g, v)
                svg = re.search(r'<svg\b[^>]*>(.*?)</svg>', content, re.S)[1]
                bounds = [float(x) for x in re.search(r'<svg[^>]*viewBox="([^"]+)"', content)[1].split()]
                result = dict(version=version, svg=svg, bounds=bounds, nodes=len(g['nodes']), edges=len(g['edges']),
                    retired=len(g.get('node_retirement', {}).get('retired_ids', [])),
                    timestamp=time.time(), status='图已更新',
                    positions={n['stable_id']: n['position_ned'] for n in g['nodes']})
                path = self.map_path.parent/'navigation_graph/live_graph.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
                temporary.replace(path)
                with self.lock:
                    result['robot'] = self.payload.get('robot')
                    result['heartbeat'] = self.payload.get('heartbeat')
                    result['navigation'] = self.payload.get('navigation')
                    self.payload = result
            except Exception as error:
                with self.lock:
                    self.payload['status'] = '图显示更新失败：'+str(error)
                self.submitted = -1  # Retry from the next main-thread publish.
        self.future = self.executor.submit(render)

    def robot(self, attachment):
        with self.lock:
            self.payload['robot'] = copy.deepcopy(attachment)
            self.payload['heartbeat'] = time.time()

    def navigation(self, state):
        with self.lock:
            self.payload['navigation'] = copy.deepcopy(state)

    def navigation_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.payload.get('navigation'))

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.executor.shutdown(wait=True, cancel_futures=True)


CLIENT = NAVIGATION_CLIENT + r'''
(() => {
  if (location.protocol === 'file:') return;
  const status=document.createElement('p');status.id='wildos-live-status';
  status.textContent='连接实时导航图…';document.body.prepend(status);
  let version=-1, failures=0, topMarkup='', polling=false, positions={}, nextTop=0;
  const parser=new DOMParser();
  async function tick(){
    if(polling)return;polling=true;
    try{
      const response=await fetch('/live.json?since='+version,{cache:'no-store',signal:AbortSignal.timeout(4000)});
      if(!response.ok)throw Error('HTTP '+response.status);
      const data=await response.json();
      const panel=document.getElementById('graph-panel');
      if(panel&&data.version!==version&&data.svg){
        const svg=panel.querySelector('svg'), group=svg.firstElementChild;
        if(svg._disposeInspector)svg._disposeInspector();
        const parsed=parser.parseFromString('<svg xmlns="http://www.w3.org/2000/svg">'+data.svg+'</svg>','image/svg+xml');
        if(parsed.querySelector('parsererror'))throw Error('图数据无效');
        parsed.querySelectorAll('[id]').forEach(e=>e.id='graph-'+e.id);
        group.replaceChildren(...Array.from(parsed.documentElement.childNodes).map(n=>document.importNode(n,true)));
        panel.querySelectorAll('[data-graph-layer]').forEach(input=>{
          const layer=svg.querySelector('#graph-'+input.dataset.graphLayer);
          if(layer)layer.style.display=input.checked?'':'none';
        });
        wireGraphInspector(svg,panel);
        panel.querySelector('header p').textContent=`第 ${data.version} 次更新 · ${data.nodes} 个节点 · ${data.edges} 条边 · 本轮退休 ${data.retired} 个节点`;
        positions=data.positions||{};
        if(data.bounds&&svg._wildosUpdateBounds){
          const [x,y,w,h]=data.bounds, matrix=group.transform.baseVal.consolidate().matrix;
          const corners=[[x,y],[x+w,y],[x,y+h],[x+w,y+h]].map(p=>new DOMPoint(...p).matrixTransform(matrix));
          const xs=corners.map(p=>p.x),ys=corners.map(p=>p.y);
          svg._wildosUpdateBounds([Math.min(...xs),Math.min(...ys),Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys)]);
        }
        version=data.version;
      }
      if(panel&&data.navigation){
        drawNavigation(panel.querySelector('svg').firstElementChild,data.navigation);
        let caption=panel.querySelector('[data-navigation-status]');
        if(!caption){caption=document.createElement('p');caption.setAttribute('data-navigation-status','');panel.querySelector('header').append(caption);}
        const n=data.navigation, chain=(n.route||{}).node_path||[];
        caption.textContent=`图导航 ${n.target_id||''} · ${n.phase||''} · ${n.control_status||''} · ${chain.join(' → ')}；紫色：图路径，橙色虚线：接入段，青色：当前航段，红色：实际轨迹`;
      }
      if(panel&&data.robot&&data.robot.position_ned){
        const group=panel.querySelector('svg').firstElementChild;
        let marker=group.querySelector('[data-live-robot]');
        if(!marker){marker=document.createElementNS('http://www.w3.org/2000/svg','g');marker.setAttribute('data-live-robot','');group.append(marker);}
        group.querySelectorAll('[data-snapshot-robot]').forEach(n=>n.style.display='none');
        const p=data.robot.position_ned, q=positions[data.robot.node_stable_id];
        marker.replaceChildren();
        if(q&&!(data.navigation&&data.navigation.route)){const line=document.createElementNS(marker.namespaceURI,'line');
          Object.entries({x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:'#f97316','stroke-width':'.07'}).forEach(([k,v])=>line.setAttribute(k,v));marker.append(line);}
        const circle=document.createElementNS(marker.namespaceURI,'circle');
        Object.entries({cx:p[0],cy:p[1],r:'.18',fill:'#0f172a',stroke:'white','stroke-width':'.05'}).forEach(([k,v])=>circle.setAttribute(k,v));marker.append(circle);
      }
      // Upper map is refreshed when a new saved checkpoint is available.
      if(Date.now()>=nextTop){nextTop=Date.now()+5000;
      const page=await fetch('/map.html',{cache:'no-store',signal:AbortSignal.timeout(4000)});
      if(page.ok){const doc=parser.parseFromString(await page.text(),'text/html');
        const fresh=doc.querySelector('#exploration-panel svg'), old=document.querySelector('#exploration-panel svg');
        if(fresh&&old&&fresh.innerHTML!==topMarkup){old.innerHTML=fresh.innerHTML;topMarkup=fresh.innerHTML;
          if(old._wildosUpdateBounds)old._wildosUpdateBounds(fresh.getAttribute('viewBox').split(/\s+/).map(Number));}}}
      failures=0;
      const stale=data.heartbeat&&Date.now()/1000-data.heartbeat>10;
      status.textContent=`${stale?'暂无新观测':'实时显示'} · ${data.status} · 图版本 ${data.version} · 保留缩放与图层设置`;
    }catch(error){failures++;status.textContent='实时连接中断，保留最后画面，正在重连…';}
    finally{polling=false;setTimeout(tick,failures?2000:1000);}
  }
  tick();
})();
'''
