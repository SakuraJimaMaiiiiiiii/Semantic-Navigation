"""Retain local nearest neighbors and minimum-length connectivity repairs."""
import math
import heapq
from .static_graph import _components


def sparsify_edges(count, edges, *, neighbors=4, local_radius=3., max_stretch=1.5):
    if type(neighbors) is not int or neighbors<1 or not math.isfinite(local_radius) or local_radius<=0:
        raise ValueError('neighbors and local_radius must be positive')
    if not math.isfinite(max_stretch) or max_stretch<1:
        raise ValueError('max_stretch must be finite and >= 1')
    ranked=sorted(range(len(edges)),key=lambda i:(edges[i]['length'],edges[i]['source'],edges[i]['target']))
    adjacency=[[] for _ in range(count)]
    for i in ranked:
        e=edges[i]
        if e['length']<=local_radius:
            adjacency[e['source']].append(i);adjacency[e['target']].append(i)
    selected={i for incident in adjacency for i in incident[:neighbors]}
    local=set(selected)
    parent=list(range(count))
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):
        a,b=root(a),root(b)
        if a==b:return False
        parent[b]=a
        return True
    for i in sorted(selected):
        join(edges[i]['source'],edges[i]['target'])
    for i in ranked:
        e=edges[i]
        if join(e['source'],e['target']):selected.add(i)
    bridges=selected-local
    paths=[{} for _ in range(count)]
    def add(i):
        e=edges[i];a,b=e['source'],e['target']
        paths[a][b]=paths[b][a]=e['length']
    for i in selected:add(i)
    shortcuts=set()
    def within(a,b,limit):
        distances={a:0.};queue=[(0.,a)]
        while queue:
            distance,current=heapq.heappop(queue)
            if distance!=distances[current]:continue
            if current==b:return True
            for nxt,length in paths[current].items():
                cost=distance+length
                if cost<=limit+1e-9 and cost<distances.get(nxt,math.inf):
                    distances[nxt]=cost;heapq.heappush(queue,(cost,nxt))
        return False
    # Guarantee a replacement path <= t times every original edge. Therefore
    # every original shortest path also has a replacement <= t times its length.
    for i in ranked:
        if i in selected:continue
        e=edges[i]
        if not within(e['source'],e['target'],max_stretch*e['length']):
            selected.add(i);shortcuts.add(i);add(i)
    result=[dict(edges[i],id=j,sparse_reason='local_neighbor' if i in local else 'detour_shortcut' if i in shortcuts else 'connectivity_bridge')
            for j,i in enumerate(sorted(selected,key=lambda i:(edges[i]['source'],edges[i]['target'])))]
    before=_components(count,edges);after=_components(count,result)
    if before!=after:raise ValueError('Edge sparsification changed connected components')
    return result,dict(enabled=True,method='local k-nearest safe edges, shortest component bridges, and bounded-detour shortcuts',
        neighbors_per_node=neighbors,local_radius_m=local_radius,before_edges=len(edges),after_edges=len(result),
        local_edges=len(local),bridge_edges=len(bridges),detour_shortcut_edges=len(shortcuts),
        max_path_stretch=max_stretch,components_preserved=True,
        long_bridge_edges=sum(edges[i]['length']>local_radius for i in bridges),
        long_edges=sum(e['length']>local_radius for e in result),
        max_length_m=max((e['length'] for e in result),default=0.),
        note='Only existing safe edges retained. Degree may exceed k. Each dense edge has a sparse replacement within max_path_stretch.')


def apply_sparsification(graph, dense_edges, *, enabled=True, max_stretch=1.5):
    if enabled:
        graph['edges'],graph['edge_sparsification']=sparsify_edges(len(graph['nodes']),dense_edges,max_stretch=max_stretch)
    else:
        graph['edges']=[dict(e) for e in dense_edges]
        graph.pop('edge_sparsification',None)
    graph['diagnostics']['edge_count']=len(graph['edges'])
    return graph
