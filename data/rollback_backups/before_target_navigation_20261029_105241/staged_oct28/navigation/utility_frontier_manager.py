"""Clustered, single-viewpoint receding-horizon exploration (no frontier tour).

Uses the existing NavigationGrid.path A* and visible_unknown ray probes.
Raw free cells define frontiers; clearance-eroded cells define viewpoints.
"""
from collections import deque
from dataclasses import dataclass
import math
import time
import numpy as np

from .frontier_geometry import Viewpoint, visible_unknown

NEIGHBORS8 = tuple((x,y) for x in (-1,0,1) for y in (-1,0,1) if x or y)


@dataclass
class FrontierCluster:
    id: int
    cells: frozenset
    centroid: np.ndarray


def is_frontier(cell, known, occupied):
    return cell in known and cell not in occupied and any(
        (cell[0]+dx,cell[1]+dy) not in known for dx,dy in NEIGHBORS8)


def detect_clusters(grid, minimum, max_span):
    remaining = {c for c in grid.known-grid.occupied if is_frontier(c,grid.known,grid.occupied)}
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cells, queue = {seed}, deque([seed])
        low, high = list(seed), list(seed)
        while queue:
            x,y = queue.popleft()
            for dx,dy in NEIGHBORS8:
                nxt=(x+dx,y+dy)
                if nxt not in remaining:
                    continue
                lo=[min(low[i],nxt[i]) for i in range(2)]
                hi=[max(high[i],nxt[i]) for i in range(2)]
                if max(hi[i]-lo[i] for i in range(2))*grid.resolution > max_span:
                    continue
                low,high=lo,hi
                remaining.remove(nxt);cells.add(nxt);queue.append(nxt)
        if len(cells)>=minimum:
            result.append(frozenset(cells))
    return result


class UtilityFrontierManager:
    backend_name = 'cluster_utility_astar'

    def __init__(self, config):
        self.config=config
        self.options=config.decision
        self.clusters=[];self.regions=[];self.candidates={};self.visit_order=[]
        self.paths={};self.scores={}
        self._history={};self._next_id=0;self._visits=[]
        self.active_key=None
        self.batch_pending=False
        self._cycle_token=None
        self._cycle_seen=set()
        self._cycle_ids=set()
        self._excluded=set()
        self.diagnostics={}

    def identity(self,key):
        return f'cluster_{int(key[0]):04d}'

    def record_visit(self,candidate):
        if candidate is not None:
            self._visits.append(np.asarray(candidate.point[:2]).copy())

    def _visit_count(self,point):
        return sum(np.linalg.norm(p-point[:2]) <= self.options.revisit_radius for p in self._visits)

    def _clusters(self,grid):
        parts=detect_clusters(grid,self.options.min_frontier_size,self.options.cluster_max_span)
        available=dict(self._history)
        clusters=[]
        for cells in parts:
            centroid=np.mean([grid.point(c) for c in cells],axis=0)
            matches=[(len(cells & old.cells),-np.linalg.norm(centroid[:2]-old.centroid[:2]),-key,key)
                     for key,old in available.items() if cells & old.cells]
            if matches:
                identity=max(matches)[-1];available.pop(identity)
            else:
                identity=self._next_id;self._next_id+=1
            clusters.append(FrontierCluster(identity,cells,centroid))
        self.clusters=clusters
        # Only previous live shapes are needed for identity association. Visits
        # are spatial and survive disappearing/splitting/merging frontiers.
        self._history={c.id:c for c in clusters}
        self.regions=[c.cells for c in clusters]

    def _candidate_cells(self,grid,cluster):
        # Circle samples around a bounded cluster's centroid. Resolve each to a
        # safe grid center, deduplicate, then retain spatially diverse samples.
        pool=set()
        for radius in np.linspace(self.config.candidate_radius_min,self.config.candidate_radius_max,3):
            for angle in np.linspace(-math.pi,math.pi,24,endpoint=False):
                p=cluster.centroid.copy();p[:2]+=radius*np.array([math.cos(angle),math.sin(angle)])
                cell=grid.cell(p)
                if cell not in grid.free or (cluster.id,*cell) in self._excluded:
                    continue
                distance=np.linalg.norm(grid.point(cell)[:2]-cluster.centroid[:2])
                if self.config.candidate_radius_min <= distance <= self.config.candidate_radius_max:
                    pool.add(cell)
        # Retain the previous candidate for comparison with newly sampled views.
        chosen=[]
        if self.active_key and self.active_key[0]==cluster.id:
            cell=tuple(self.active_key[1:])
            distance=np.linalg.norm(grid.point(cell)[:2]-cluster.centroid[:2])
            if cell in grid.free and self.active_key not in self._excluded and self.config.candidate_radius_min <= distance <= self.config.candidate_radius_max:
                chosen.append(cell);pool.discard(cell)
        while pool and len(chosen)<self.options.candidates_per_cluster:
            cell=min(pool) if not chosen else max(sorted(pool), key=lambda c:min(
                (c[0]-p[0])**2+(c[1]-p[1])**2 for p in chosen))
            chosen.append(cell);pool.remove(cell)
        return chosen

    def score_candidates(self, rows):
        """All four terms in [0,1]; coefficients are decision configuration."""
        max_gain=max((len(v.visible_unknown) for v,visits in rows),default=1)
        self.scores={}
        for view,visits in rows:
            terms=dict(gain=len(view.visible_unknown)/max(1,max_gain),
                       path_cost=min(1.,view.cost/self.options.path_normalization_m),
                       yaw_cost=self._yaw_costs[view.key]/math.pi,
                       revisit_penalty=visits/(1.+visits))
            view.score=(self.options.gain_weight*terms['gain']
                        -self.options.path_weight*terms['path_cost']
                        -self.options.yaw_weight*terms['yaw_cost']
                        -self.options.revisit_weight*terms['revisit_penalty'])
            self.scores[view.key]=dict(**terms,utility=view.score,path_length_m=view.cost,
                                      unknown_count=len(view.visible_unknown),visit_count=visits)

    def update(self,grid,current,origin,yaw,excluded):
        started=time.perf_counter()
        self._clusters(grid)
        self.paths={};self._yaw_costs={};rows=[]
        self._excluded=set(excluded)
        # Bounded round-robin batches avoid monopolizing the control thread or
        # permanently discarding distant clusters. The active cluster is kept.
        clusters=sorted(self.clusters,key=lambda c:c.id)
        all_ids={c.id for c in clusters}
        token=frozenset(excluded)
        if token != self._cycle_token or not (self._cycle_ids-self._cycle_seen):
            self._cycle_seen=set()
            self._cycle_ids=set(all_ids)
            self._cycle_token=token
        # Freeze this pass's IDs. Newly created frontiers belong to the next
        # pass; disappearing IDs count as resolved, rather than restarting it.
        self._cycle_ids.intersection_update(all_ids)
        pending=self._cycle_ids-self._cycle_seen
        limited=len(clusters)>self.options.max_clusters_per_update
        count=self.options.max_clusters_per_update
        batch=[c for c in clusters if c.id in pending][:count]
        self._cycle_seen.update(c.id for c in batch)
        active=next((c for c in clusters if self.active_key and c.id==self.active_key[0]),None)
        if active is not None and all(c.id!=active.id for c in batch):
            batch.append(active)
        clusters=batch
        self.batch_pending=bool(self._cycle_ids-self._cycle_seen)
        outside=False;attempted=0;unreachable=0
        sampled=0;no_gain=0;near_or_excluded=0;outside_count=0
        if grid.cell(current) in grid.free:
            for cluster in clusters:
                for cell in self._candidate_cells(grid,cluster):
                    sampled+=1
                    key=(cluster.id,*cell);point=grid.point(cell)
                    if key in excluded or np.linalg.norm(point[:2]-current[:2])<=grid.config.arrival_tolerance:
                        near_or_excluded+=1
                        continue
                    if np.linalg.norm(point[:2]-np.asarray(origin)[:2])>self.config.max_radius:
                        outside=True;outside_count+=1;continue
                    view_yaw=math.atan2(cluster.centroid[1]-point[1],cluster.centroid[0]-point[0])
                    visible=visible_unknown(grid,cell,view_yaw,self.config.information_radius,self.config.camera_fov_degrees)
                    if not visible:
                        no_gain+=1
                        continue
                    attempted+=1
                    path=grid.path(current,point)  # Existing A*, never Euclidean cost.
                    if path is None:
                        unreachable+=1;continue
                    cost=float(sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:])))
                    view=Viewpoint(key,point,view_yaw,cost,visible,0.)
                    self.paths[key]=path
                    # Observation yaw is not the commanded travel heading.
                    # Penalize the actual first non-zero A* leg, including turns
                    # needed to route around walls before reaching the viewpoint.
                    departure = next((np.asarray(b)[:2]-np.asarray(a)[:2]
                                      for a,b in zip(path,path[1:])
                                      if np.linalg.norm(np.asarray(b)[:2]-np.asarray(a)[:2]) > 1e-6), None)
                    travel_yaw = yaw if departure is None else math.atan2(departure[1], departure[0])
                    self._yaw_costs[key]=abs((travel_yaw-yaw+math.pi)%(2*math.pi)-math.pi)
                    rows.append((view,self._visit_count(point)))
        self.score_candidates(rows)
        ordered=sorted((v for v,_ in rows),key=lambda v:(-v.score,v.key))
        self.candidates={v.key:v for v in ordered}
        self.visit_order=[ordered[0].key] if ordered else []
        self.diagnostics=dict(cluster_count=len(self.clusters),evaluated_clusters=len(clusters),
                              sampled_viewpoints=sampled, no_gain_candidates=no_gain,
                              near_or_excluded=near_or_excluded, outside_radius_candidates=outside_count,
                              candidate_count=len(rows),astar_calls=attempted,unreachable_candidates=unreachable,
                              batched=limited, batch_pending=self.batch_pending,
                              decision_seconds=time.perf_counter()-started)
        # Batching is deferred work, not a permanent A* search-limit failure.
        return [(v.key,v.point) for v in ordered],False,outside

    def select(self,choices):
        if not choices:
            self.visit_order=[]
            return None
        key=choices[0][0]
        self.active_key=key
        self.visit_order=[key]
        return self.candidates[key]
