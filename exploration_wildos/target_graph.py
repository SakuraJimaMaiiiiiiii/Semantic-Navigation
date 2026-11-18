"""Dijkstra on the explored graph, with certified start/approach connectors."""
import heapq
import math
import numpy as np
from navigation.route import NavigationGrid


class TargetGraphGrid(NavigationGrid):
    def __init__(self, grid, graph):
        self.__dict__.update(grid.__dict__)
        self.graph = graph
        self.routes = {}
        self._start = None
        self._paths = {}
        self.connector_radius = graph.get('options', {}).get('edge_radius', 8.)

    def _search(self, start):
        key = tuple(np.asarray(start, dtype=float))
        if self._start == key:
            return
        self._start, self._paths = key, {}
        nodes = self.graph['nodes']
        if self.cell(start) not in self.free:
            return
        attachments = sorted((float(np.linalg.norm(np.asarray(n['position_ned'])-start)), n['id']) for n in nodes)
        first = next(((d,i) for d,i in attachments if d <= self.connector_radius
                      and self.segment_free(start,nodes[i]['position_ned'])), None)
        if first is None:
            return
        links = {n['id']: [] for n in nodes}
        for edge in self.graph['edges']:
            a,b = edge['source'],edge['target']
            pa,pb = nodes[a]['position_ned'],nodes[b]['position_ned']
            if self.segment_free(pa,pb):
                length=float(np.linalg.norm(np.asarray(pa)-pb))
                links[a].append((b,length));links[b].append((a,length))
        distance,root = first
        distances,parents,queue = {root:distance},{root:None},[(distance,root)]
        while queue:
            distance,i=heapq.heappop(queue)
            if distance != distances[i]:
                continue
            for j,length in links[i]:
                candidate=distance+length
                if candidate < distances.get(j,math.inf):
                    distances[j]=candidate;parents[j]=i
                    heapq.heappush(queue,(candidate,j))
        for i,distance in distances.items():
            chain=[];j=i
            while j is not None:
                chain.append(j);j=parents[j]
            self._paths[i]=(distance,list(reversed(chain)))

    def path(self, start, goal):
        start,goal=np.asarray(start),np.asarray(goal)
        if self.cell(goal) not in self.free:
            return None
        self._search(start)
        nodes=self.graph['nodes']
        endings=[]
        for i,(distance,chain) in self._paths.items():
            connector=float(np.linalg.norm(np.asarray(nodes[i]['position_ned'])-goal))
            if connector <= self.connector_radius and self.segment_free(nodes[i]['position_ned'],goal):
                endings.append((connector,distance+connector,i,chain))
        if not endings:
            return None  # No fallback to a grid route or semantic association edge.
        _,length,_,chain=min(endings,key=lambda item:(item[0],item[1],item[2]))
        points=[start.copy(),*[np.asarray(nodes[i]['position_ned']).copy() for i in chain],goal.copy()]
        self.routes[tuple(goal)]=dict(backend='wildos_graph_dijkstra',
            node_path=[nodes[i]['stable_id'] for i in chain], length_m=length,
            points_ned=[p.tolist() for p in points], goal_kind='safe_vehicle_approach')
        return points
