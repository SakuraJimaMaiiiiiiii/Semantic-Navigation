"""Independent live graph exploration: python -m exploration_wildos.explore.

The existing explorer owns execution, recovery and mission limits. This module
supplies frontier-node goals and graph routes, never velocity commands. Graph
updates also run in a bounded background worker during motion. No saved/corrected graph is loaded into flight.
"""
from dataclasses import replace, asdict
import copy
import argparse
import multiprocessing
import time
import tempfile
import hashlib
import heapq
import math
import os
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import uuid
import webbrowser

import numpy as np

# Parse CLI before importing flight SDK dependencies, so --help is offline-safe.
from run_explore import shutdown_session, DEFAULT_SESSION_FILE
from config.exploration_config import FrontierExplorationConfig


def parse_config(argv=None, *, include_perception=False):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = FrontierExplorationConfig()
    flags = {
        'port': ('server_port', int), 'timeout': ('timeout', float),
        'max-radius': ('max_radius', float), 'max-viewpoints': ('max_viewpoints', int),
        'viewpoint-spacing': ('viewpoint_spacing', float),
        'information-radius': ('information_radius', float),
        'frontier-empty-seconds': ('frontier_empty_min_duration', float),
        'completion-frontier-area': ('completion_frontier_area', float),
        'coverage-plateau-seconds': ('coverage_plateau_seconds', float),
        'minimum-map-gain-area': ('minimum_map_gain_area', float),
        'return-reserve': ('return_time_reserve', float),
    }
    for flag, (field, kind) in flags.items():
        parser.add_argument('--'+flag, type=kind, default=getattr(defaults, field))
    perception = parser.add_mutually_exclusive_group()
    perception.add_argument('--sam3', dest='perception_backend', action='store_const', const='sam3')
    perception.add_argument('--sam2', dest='perception_backend', action='store_const', const='yolo_sam2')
    parser.set_defaults(perception_backend='sam3')
    parser.add_argument('--fastlio', action='store_true', help='Experimental mapping alongside PX4 graph navigation')
    parser.add_argument('--keep-session', action='store_true',
                        help='Keep the live target server and sensors after landing (default: save and exit)')
    args = parser.parse_args(argv)
    config = replace(defaults, **{field: getattr(args, flag.replace('-', '_'))
                                 for flag, (field, _) in flags.items()})
    return (config, args) if include_perception else config


_CLI = parse_config(include_perception=True) if __name__ == '__main__' else None

from navigation.exploration_session import ExplorationSession, SavedMapNavigator, TargetUnavailableError
from navigation.exploration_planner import FrontierExplorationPlanner
from navigation.frontier_explorer import FrontierExplorer
from navigation.mission import _FlightGuard
from planner.local_avoidance_planner import LocalPlan
from navigation.frontier_geometry import Viewpoint
from .departure import (trim_reached_prefix, heading_difference, transient_failure,
                        local_geometry_signature)
from navigation.map_view import render_map
from navigation.route import NavigationGrid
from navigation.scene_memory import write_json
from .map_adapter import MapAdapter
from .graph_worker import build_live_graph, graph_process
from .static_graph import GraphConfig
from .semantic_graph import attach_objects
from .vehicle_constraints import constrain_adapter, expanded_boxes, segment_hits_vehicle, VEHICLE_POLICY
from .vehicle_motion import VehicleAwareMissionController, VehiclePlanningMap
from .graph_export import export_graph
from .combined_map import compose_map_html


class GraphFrontiers:
    """Persistent sampling memory, compact routing graph and frontier visit memory."""
    backend_name = 'wildos_frontier_graph_dijkstra'

    def __init__(self, config, *, options=None):
        from .frontier_memory import FrontierMemory
        self.config = config
        self.travel_memory = FrontierMemory(config.viewpoint_spacing)
        self.options = options or GraphConfig(paper_aligned=True, incremental_cleanup=True, max_samples=128,
            max_nodes=1500, free_radius_cap=4., edge_radius=8., prune_redundant=False)
        self.state = self.graph = self.adapter = None
        self.signature = self.epoch = None
        self.epoch_number = -1
        self.clusters, self.regions, self.visit_order = [], [], []
        self.candidates, self.paths, self.scores, self.visited = {}, {}, {}, {}
        self.active_key = None
        self.batch_pending = False
        self._cycle_token = None
        self.diagnostics = {}
        self.update_count = 0
        self.last_route = None

    def identity(self, key):
        return f'epoch:{self.epoch_number}/nav:{key[0]}'

    def refresh(self, grid, current, *, frame_key, objects=(), builder=build_live_graph):
        epoch = (frame_key, grid.altitude, grid.resolution, tuple(grid.origin))
        if epoch != self.epoch:
            from .frontier_memory import FrontierMemory
            self.travel_memory = FrontierMemory(self.config.viewpoint_spacing)
            self.epoch, self.epoch_number = epoch, self.epoch_number + 1
            self.state = self.signature = None
            self.visited.clear()
            self.active_key = None
        signature = hashlib.sha256(repr((VEHICLE_POLICY, epoch, sorted(grid.known), sorted(grid.free),
            sorted(grid.occupied), [(o['id'], o['bbox_3d']) for o in objects])).encode()).hexdigest()
        self.adapter = MapAdapter.from_grid(grid, dict(snapshot_id=signature,
            timestamp=float(self.update_count + 1), timestamp_basis='graph_update_sequence',
            source='active PX4 occupancy snapshot', dataset_label='实时探索规划快照',
            frame='NED', offline_only=False, vehicle_constraint_policy=VEHICLE_POLICY,
            frame_evidence='active PX4 raw map; no loop-corrected map imported'))
        if signature == self.signature:
            return False
        start = np.asarray(current).copy()
        start[2] = grid.altitude
        payload = dict(grid=dict(known=list(grid.known), free=list(grid.free),
            occupied=list(grid.occupied), origin=grid.origin.tolist(),
            resolution=grid.resolution, altitude=grid.altitude, config=asdict(grid.config)),
            metadata=self.adapter.metadata, frame_key=frame_key, state=self.state,
            options=asdict(self.options), start=start.tolist(), objects=list(objects),
            vehicle_constraints=getattr(grid, 'vehicle_constraint_report', None))
        graph, state, validation = builder(payload)
        self.state, self.graph, self.validation = state, graph, validation
        self.signature = signature
        self.update_count += 1
        return True

    def update_robot_attachment(self, grid, current):
        """Update the robot association, never move an environment node."""
        if self.graph is None:
            return
        current = np.asarray(current)
        attachment = dict(position_ned=current.tolist(), status='no_certified_connection',
                          node_id=None, node_stable_id=None, component_id=None)
        if grid.cell(current) in grid.free:
            nodes = sorted(self.graph['nodes'], key=lambda n:
                           (np.linalg.norm(np.asarray(n['position_ned'])-current), n['id']))
            for node in nodes:
                if (np.linalg.norm(np.asarray(node['position_ned'])-current) <= self.options.edge_radius
                        and grid.segment_free(current, node['position_ned'])):
                    attachment.update(status='attached', node_id=node['id'],
                                      node_stable_id=node['stable_id'], component_id=next(
                        (i for i,c in enumerate(self.graph['components']) if node['id'] in c), None))
                    break
        self.graph['robot_attachment'] = attachment

    def routes(self, grid, current):
        """Single certified robot connector followed exclusively by graph edges."""
        if self.graph is None or grid.cell(current) not in grid.free:
            return {}
        self.update_robot_attachment(grid, current)
        nodes = self.graph['nodes']
        links = [[] for _ in nodes]
        for edge in self.graph['edges']:
            a, b = edge['source'], edge['target']
            # Recheck even unchanged graphs against current local/global evidence.
            if grid.segment_free(nodes[a]['position_ned'], nodes[b]['position_ned']):
                repeated = self.travel_memory.reused_distance([nodes[a]['position_ned'], nodes[b]['position_ned']])
                cost = edge['length']+self.travel_memory.repeat_weight*repeated
                links[a].append((b, edge['length'], cost))
                links[b].append((a, edge['length'], cost))
        attachments = sorted((float(np.linalg.norm(np.asarray(n['position_ned'])-current)), n['id'])
                             for n in nodes)
        root = next(((d, i) for d, i in attachments if d <= self.options.edge_radius
                     and grid.segment_free(current, nodes[i]['position_ned'])), None)
        if root is None:
            return {}
        connector, first = root
        connector_repeat = self.travel_memory.reused_distance([current, nodes[first]['position_ned']])
        def search(prefer_new):
            start_cost = connector+(self.travel_memory.repeat_weight*connector_repeat if prefer_new else 0.)
            distances, lengths, parents = {first: start_cost}, {first: connector}, {first: None}
            pending = [(start_cost, first)]
            while pending:
                distance, i = heapq.heappop(pending)
                if distance != distances[i]:
                    continue
                for j, length, cost in links[i]:
                    candidate = distance+(cost if prefer_new else length)
                    if candidate < distances.get(j, math.inf):
                        distances[j], parents[j] = candidate, i
                        lengths[j] = lengths[i]+length
                        heapq.heappush(pending, (candidate, j))
            return lengths, parents
        shortest, short_parents = search(False)
        lengths, parents = search(True) if self.travel_memory.traversed else (shortest, short_parents)
        result = {}
        for i, distance in lengths.items():
            # Avoid replacing a short necessary retrace with a large loop.
            selected_parents = parents
            if distance > 1.5*shortest[i]+1e-9:
                distance, selected_parents = shortest[i], short_parents
            chain, j = [], i
            while j is not None:
                chain.append(j)
                j = selected_parents[j]
            chain.reverse()
            points = [np.asarray(current).copy(), *[np.array(nodes[j]['position_ned']) for j in chain]]
            result[nodes[i]['stable_id']] = (distance, points, [nodes[j]['stable_id'] for j in chain])
        return result

    def update(self, grid, current, origin, yaw, excluded):
        self.travel_memory.record_position(current)
        routes = self.routes(grid, current)
        frontier_by_id = {f['id']: f for f in self.graph['frontiers']}
        # Count ALL unknown boundary cells, including those with no graph owner.
        boundary = {(x+dx, y+dy) for x, y in grid.known-grid.occupied
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))
                    if (x+dx, y+dy) not in grid.known}
        self.regions = [boundary] if boundary else []
        self.clusters, self.candidates, self.paths, self.scores = [], {}, {}, {}
        outside = any(np.linalg.norm(grid.point(c)[:2]-origin[:2]) > self.config.max_radius
                      for c in boundary)
        for node in self.graph['nodes']:
            if not node['frontier_ids']:
                continue
            point = np.array(node['position_ned'])
            cells = frozenset(tuple(frontier_by_id[i]['cell']) for i in node['frontier_ids']
                              if tuple(frontier_by_id[i]['cell']) not in grid.known)
            if not cells:
                continue
            key = (int(node['stable_id'].split(':')[1]), *grid.cell(point))
            self.clusters.append(SimpleNamespace(id=key[0]))
            if np.linalg.norm(point[:2]-origin[:2]) > self.config.max_radius:
                outside = True
                continue
            if key in excluded or node['stable_id'] not in routes:
                continue
            # Revisit only if the associated unknown region gains new cells.
            if cells <= self.visited.get(node['stable_id'], frozenset()):
                continue
            distance, path, _ = routes[node['stable_id']]
            target = np.mean([grid.point(c) for c in cells], axis=0)
            heading = math.atan2(target[1]-point[1], target[0]-point[0])
            novel = self.travel_memory.novel_cells(point, heading, cells)
            if not novel:
                continue
            target = np.mean([grid.point(c) for c in novel], axis=0)
            heading = math.atan2(target[1]-point[1], target[0]-point[0])
            repeated = self.travel_memory.reused_distance(path)
            turn_cost = self.travel_memory.turn_cost(path, yaw)
            cost = distance+self.travel_memory.repeat_weight*repeated+turn_cost
            score = len(novel)*grid.resolution**2/(1. + cost)
            self.candidates[key] = Viewpoint(key, point, heading, distance, cells, score)
            self.paths[key] = path
            self.scores[key] = dict(score=score, graph_distance_m=distance,
                frontier_cells=len(cells), novel_frontier_cells=len(novel),
                repeated_distance_m=repeated, turn_cost_m=turn_cost, selection_cost_m=cost,
                visit_count=int(node['stable_id'] in self.visited))
        self.visit_order = sorted(self.candidates, key=lambda k: (-self.candidates[k].score, k))
        self.diagnostics = dict(selection_policy='novel_frontier_gain_over_travel_repeat_and_turn_cost',
            graph_nodes=len(self.graph['nodes']), graph_edges=len(self.graph['edges']),
            reachable_nodes=len(routes), candidate_count=len(self.candidates),
            frontier_cells=len(boundary), graph_epoch=self.epoch_number)
        limited = self.graph['diagnostics']['node_budget_reached']
        return [(k, self.candidates[k].point) for k in self.visit_order], limited, outside

    def select(self, choices):
        self.active_key = choices[0][0]
        return self.candidates[self.active_key]

    def record_visit(self, view):
        if view is not None:
            self.travel_memory.record_visit(view)
            stable = f'nav:{view.key[0]}'
            self.visited[stable] = self.visited.get(stable, frozenset()) | view.visible_unknown

    def route(self, grid, current, goal, stable_id):
        self.last_route = None
        result = self.routes(grid, current).get(stable_id)
        if result is None or not np.allclose(result[1][-1], goal, rtol=0, atol=1e-7):
            return None  # never repair a deleted goal into an unrelated A* target
        distance, points, chain = result
        self.last_route = dict(target=stable_id, node_path=chain, length_m=distance,
                               points_ned=[p.tolist() for p in points], epoch=self.epoch_number)
        return points


class GraphPlanner(FrontierExplorationPlanner):
    def prepare_candidates(self, grid, pose, history):
        choices, self.search_limited, self.outside_radius = self.manager.update(
            grid, pose.position, history.origin, pose.yaw, history.excluded)
        return choices

    def select_next_waypoint(self, *args, **kwargs):
        waypoint = super().select_next_waypoint(*args, **kwargs)
        if waypoint is not None:
            waypoint['source'] = 'wildos_frontier_node'
        return waypoint


class GraphRouteGrid(NavigationGrid):
    """Per-instance adapter; original NavigationGrid/A* are never modified."""
    def __init__(self, grid, planner):
        self.__dict__.update(grid.__dict__)
        self.graph_path = planner

    def path(self, start, goal):
        return self.graph_path(self, start, goal)


class VehicleAwareGuard(_FlightGuard):
    """Keep every existing execution check, then veto observed vehicle geometry."""
    def _report(self, plan, rejection=None):
        reported = super()._report(plan, rejection)
        context = getattr(self, '_diagnostic_context', None)
        if plan.status == 'emergency_stop' and context is not None:
            snapshot, _, current, target = context
            value = getattr(snapshot, 'forward_clearance', None)
            self.last_plan_detail['emergency_evidence'] = dict(
                source='guard_input_snapshot; inner planner may have a newer snapshot',
                timestamp=getattr(snapshot, 'timestamp', None),
                forward_clearance_m=(float(value) if value is not None and math.isfinite(float(value)) else None),
                front_stop_threshold_m=float(self.config.emergency_stop_distance),
                map_stop_threshold_m=float(getattr(self.config, 'map_emergency_stop_distance', .4)),
                current_ned=np.asarray(current).tolist(), target_ned=np.asarray(target).tolist())
        return reported

    def revalidate_plan(self, plan, current, *, certificate=None):
        checked = super().revalidate_plan(plan, current, certificate=certificate)
        if checked.can_move:
            for identity, low, high in expanded_boxes(self.navigator._objects(), self.navigator.config):
                if segment_hits_vehicle(current, checked.local_target_ned, low, high):
                    self._execution_rejection = dict(reason='observed_vehicle_geometry', vehicle=identity)
                    return LocalPlan('semantic_vehicle_blocked', None, np.empty((0, 3)), 0.)
        return checked


class GraphExplorer(FrontierExplorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = VehicleAwareGuard(self)
        self.frontiers = GraphFrontiers(self.exploration_config)
        self.exploration_planner = GraphPlanner(self.frontiers)
        self._graph_travel = False
        self.graph_directory = Path(self.output_path).parent / 'navigation_graph'
        self.latest_graph_directory = None
        self._saved_updates = 0
        self._moving_graph = None
        self._next_moving_tick = 0.
        self._next_moving_build = 0.
        self._foreground_graph = False

    def run(self):
        try:
            return super().run()
        finally:
            self._stop_moving_graph()

    def _stop_moving_graph(self):
        worker = getattr(self, '_moving_graph', None)
        if worker is not None:
            worker.close()
            self._moving_graph = None

    def _check_health(self):
        super()._check_health()
        epoch = self.frontiers.epoch
        frame = f'{self.memory.session_id}:PX4-raw:{getattr(self.global_map, "revision", 0)}'
        if epoch is not None and epoch[0] == frame:
            self.frontiers.travel_memory.record_position(self.vehicle.get_position())
        if (getattr(self, '_graph_travel', False)
                and not getattr(self, '_foreground_graph', False)):
            self._moving_graph_tick()

    def _publish_live_map(self):
        video = getattr(self, 'graph_video', None)
        if video is not None and self.frontiers.graph is not None:
            video.publish(self.frontiers.adapter, self.frontiers.graph, self.frontiers.update_count)
        viewer = getattr(self, 'live_map', None)
        if viewer is not None and self.frontiers.graph is not None:
            viewer.robot(self.frontiers.graph['robot_attachment'])
            viewer.publish(self.frontiers.adapter, self.frontiers.graph,
                           self.frontiers.validation, self.frontiers.update_count)

    def _moving_graph_tick(self):
        now = time.monotonic()
        if now < self._next_moving_tick:
            return
        self._next_moving_tick = now + .5
        current = self.vehicle.get_position().copy()
        # This is a display association on the last certified graph snapshot.
        # Live movement remains checked by the original execution guard.
        grid = self.frontiers.adapter.grid if self.frontiers.adapter is not None else None
        revision = getattr(self.global_map, 'revision', 0)
        frame = f'{self.memory.session_id}:PX4-raw:{revision}'
        worker = self._moving_graph
        if worker is not None and worker.future is not None and worker.future.done():
            future, worker.future = worker.future, None
            try:
                changed, pending = future.result()
                if (changed and self._moving_base == (self.frontiers.update_count, frame)):
                    for name in ('state', 'graph', 'validation', 'signature', 'adapter', 'update_count'):
                        setattr(self.frontiers, name, getattr(pending, name))
                    self._event('WILDOS_GRAPH_UPDATED_IN_MOTION', update=self.frontiers.update_count,
                                nodes=len(self.frontiers.graph['nodes']),
                                retired_nodes=len(self.frontiers.state.get('retirement', {}).get('retired_ids', [])),
                                archived_regions=len(self.frontiers.state.get('explored_memory', [])),
                                preparation_seconds=getattr(pending, 'preparation_seconds', None))
                    grid = self.frontiers.adapter.grid
            except Exception as error:
                self._event('WILDOS_MOVING_GRAPH_FAILED', reason=str(error))
        if grid is None:
            return
        self.frontiers.update_robot_attachment(grid, current)
        self.frontiers.graph['robot_attachment']['evidence'] = 'last_graph_snapshot_display_only'
        self._publish_live_map()
        # A lightweight live association record; graph.html remains a snapshot.
        if self.frontiers.graph is not None:
            write_json(self.graph_directory / 'robot_attachment_live.json',
                       dict(timestamp=time.time(), update=self.frontiers.update_count,
                            **self.frontiers.graph['robot_attachment']))
        if now < self._next_moving_build or (worker is not None and worker.future is not None):
            return
        # Frame changes are handled by held replanning, never by a moving swap.
        if self.frontiers.epoch is None or self.frontiers.epoch[0] != frame:
            return
        from .moving_graph import MovingGraph
        if worker is None:
            worker = self._moving_graph = MovingGraph()
        pending = copy.copy(self.frontiers)
        pending.state = copy.deepcopy(self.frontiers.state)
        pending.visited = dict(self.frontiers.visited)
        self._moving_base = (self.frontiers.update_count, frame)
        origin = getattr(self, 'exploration_origin', None)
        worker.submit_from_maps(pending, self.global_map, self.local_map, current, frame,
            copy.deepcopy(self._objects()), self.altitude, self.config,
            None if origin is None else np.asarray(origin).copy(), self.exploration_config.max_radius)
        self._next_moving_build = now + 2.

    def _initial_observation(self):
        print('--- WildOS 当前策略：新增前沿优先 / 减少重复通道与掉头 / 有界加权图路径 / 原执行检查与 EGO 避障 ----')
        self._frame()
        self._checkpoint()

    def _objects(self):
        temporary = {'nodes': []}
        from .identity_overrides import corrected_records
        records = corrected_records(self.memory.records.values(), Path(self.output_path).parent, self.memory.session_id)
        attach_objects(temporary, {'objects': records}, {},
                       self.config.clearance_radius, max_links=0)
        return temporary['semantic_nodes']

    def _grid(self):
        grid = super()._grid()
        constrained, _ = constrain_adapter(MapAdapter.from_grid(grid), self._objects())
        grid = constrained.grid
        if hasattr(self, 'exploration_origin'):
            grid.free = {c for c in grid.free if np.linalg.norm(
                grid.point(c)[:2]-self.exploration_origin[:2]) <= self.exploration_config.max_radius}
        return GraphRouteGrid(grid, self._graph_path) if self._graph_travel else grid

    def _build_in_process(self, payload):
        context = multiprocessing.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=graph_process, args=(sender, payload),
                                  name='WildOSGraph', daemon=True)
        started = time.monotonic()
        next_notice = started + 2.
        try:
            process.start()
            sender.close()
            self._event('WILDOS_GRAPH_BUILD_STARTED', samples=payload['options']['max_samples'],
                        worker_pid=process.pid)
            print(f"--- WildOS 独立进程构图：本轮 {payload['options']['max_samples']} 次采样 ----")
            while True:
                self._check_health()
                self._check_maps()
                elapsed = time.monotonic() - started
                if elapsed > 30.:
                    raise TimeoutError('graph_build_time_budget_exceeded')
                if receiver.poll(.05):
                    status, result = receiver.recv()
                    if status != 'ok':
                        raise RuntimeError('Graph worker failed: '+result)
                    self._event('WILDOS_GRAPH_BUILD_FINISHED', seconds=elapsed,
                                nodes=len(result[0]['nodes']), edges=len(result[0]['edges']))
                    print(f"--- WildOS 图已就绪：{len(result[0]['nodes'])} 个节点，"
                          f"{len(result[0]['edges'])} 条边，用时 {elapsed:.2f}s ----")
                    return result
                if not process.is_alive():
                    raise RuntimeError(f'Graph worker exited without a result: {process.exitcode}')
                if time.monotonic() >= next_notice:
                    print(f'--- WildOS 构图中：{elapsed:.1f}s；保持悬停，地图健康检查持续运行 ----')
                    next_notice = time.monotonic() + 2.
        finally:
            # Cancel computation BEFORE sensor/runtime cleanup on every exit.
            # Never leave a CPU-heavy daemon thread competing with LiDAR shutdown.
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2.)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.)
                process.close()
            receiver.close()
            sender.close()

    def _refresh(self, grid, current):
        self._foreground_graph = True
        try:
            self._stop_moving_graph()
            return self._refresh_held(grid, current)
        finally:
            self._foreground_graph = False

    def _refresh_held(self, grid, current):
        revision = getattr(self.global_map, 'revision', 0)
        frame = f'{self.memory.session_id}:PX4-raw:{revision}'
        old_epoch = self.frontiers.epoch_number
        pending = copy.copy(self.frontiers)
        pending.visited = dict(self.frontiers.visited)
        objects = copy.deepcopy(self._objects())
        previous_progress = self._motion_progress
        self._motion_progress = None
        try:
            changed = pending.refresh(grid, current, frame_key=frame, objects=objects,
                                      builder=self._build_in_process)
            self._check_health()
            self._check_maps()
            if getattr(self.global_map, 'revision', 0) != revision:
                raise TimeoutError('graph_frame_changed_during_planning')
            self.frontiers.__dict__.update(pending.__dict__)
            if old_epoch != self.frontiers.epoch_number:
                self.failed_frontiers.clear()
            if changed:
                self._event('WILDOS_GRAPH_UPDATED', epoch=self.frontiers.epoch_number,
                            update=self.frontiers.update_count, nodes=len(self.frontiers.graph['nodes']),
                            edges=len(self.frontiers.graph['edges']),
                            retired_nodes=len(self.frontiers.state.get('retirement', {}).get('retired_ids', [])),
                            archived_regions=len(self.frontiers.state.get('explored_memory', [])))
            self.frontiers.update_robot_attachment(grid, current)
            self._publish_live_map()
        finally:
            self._motion_progress = ((self.vehicle.get_position().copy(), self.clock())
                                     if previous_progress is not None else None)

    def _choices(self, grid, current, origin):
        self._hold_for_planning()
        grid, current = self._grid(), self.vehicle.get_position()
        block = getattr(self, '_departure_block', None)
        if block is not None:
            clearance = self.local_map.forward_clearance
            changed = (np.linalg.norm(current-block['position']) > self.config.arrival_tolerance
                       or heading_difference(float(self.vehicle.get_euler()[2]), block['yaw']) > math.radians(15)
                       or local_geometry_signature(grid, block['position']) != block['geometry']
                       or (clearance is not None and not np.isnan(clearance)
                           and clearance > self.local_planner.config.emergency_stop_distance))
            if not changed:
                # Let the original bounded observation/wait and mission budget
                # run, without burning a new distant target at the same hold.
                count, _ = self.failed_frontiers.get(block['key'], (0, 0))
                self.failed_frontiers[block['key']] = (count, self.clock()+self.exploration_config.frontier_retry_cooldown)
                return [], False, False
            self._departure_block = None
            self.failed_frontiers[block['key']] = (block['previous_count'], 0.)
            self._event('WILDOS_DEPARTURE_RECHECK', reason='local_conditions_changed')
        self._refresh(grid, current)
        return super()._choices(self._grid(), self.vehicle.get_position(), origin)

    def _graph_path(self, grid, current, goal):
        self._hold_for_planning()
        previous = self._motion_progress
        self._motion_progress = None
        try:
            self._refresh(grid, current)
            if self.frontiers.epoch_number != self._committed_epoch:
                return None
            path = self.frontiers.route(self._grid(), self.vehicle.get_position(), goal, self._committed_node)
            if path is not None and self.exploration_waypoints:
                self.exploration_waypoints[-1]['graph_route'] = self.frontiers.last_route
            return path
        finally:
            self._motion_progress = ((self.vehicle.get_position().copy(), self.clock())
                                     if previous is not None else None)

    def _travel(self, goal):
        self._graph_travel = True
        self._committed_node = f'nav:{self.active_viewpoint.key[0]}'
        self._committed_epoch = self.frontiers.epoch_number
        start = self.vehicle.get_position().copy()
        key = self.active_viewpoint.key
        previous_count = self.failed_frontiers.get(key, (0, 0))[0]
        replanned = False
        try:
            result = super()._travel(goal)
            failure = getattr(self, '_last_travel_failure', None)
            if not result and failure and failure.get('reason') == 'graph_route_changed_before_execution':
                replanned = True
                if self.exploration_waypoints:
                    self.exploration_waypoints[-1].setdefault('replans', []).append(copy.deepcopy(failure))
                self._event('WILDOS_ROUTE_REPLAN', reason=failure['reason'], attempt=1)
                # Bounded retry: rebuild the graph route, never execute the old
                # invalid chord or silently fall back to direct flight.
                result = super()._travel(goal)
                failure = getattr(self, '_last_travel_failure', None)
            if failure and failure.get('reason') == 'astar_no_route':
                failure['reason'] = 'graph_no_certified_route'
            if not result and failure:
                scope = transient_failure(failure, start, self.vehicle.get_position(), self.config.arrival_tolerance)
                if replanned and failure.get('reason') == 'graph_no_certified_route':
                    scope = 'route_revalidation'
                if scope:
                    failure.update(scope=scope, retryable=True, target_invalidated=False)
                    self._transient_target_failure = (key, previous_count)
                    if scope == 'departure_blocked':
                        position = self.vehicle.get_position().copy()
                        self._departure_block = dict(key=key, previous_count=previous_count,
                            position=position, yaw=float(self.vehicle.get_euler()[2]),
                            geometry=local_geometry_signature(self._grid(), position))
            return result
        finally:
            self._graph_travel = False

    def _nearby_valid_viewpoint(self, grid, current, goal):
        return None  # select another frontier node on the next decision

    def _try_raise_for_obstacle(self, goal):
        return False  # this graph epoch represents one fixed flight-height band

    def _move(self, path, *, recovery=False, continuous=False):
        if self._graph_travel and not recovery:
            fresh = self._grid()
            current = self.vehicle.get_position()
            trimmed, skipped = trim_reached_prefix(path, current, fresh,
                self.config.arrival_tolerance,
                getattr(self.vehicle.config, 'guided_z_tolerance', self.config.arrival_tolerance))
            path[:] = trimmed
            if skipped:
                self._event('WILDOS_REACHED_CONNECTOR_SKIPPED', points_ned=skipped)
            if not all(fresh.segment_free(a, b) for a, b in zip(path, path[1:])):
                self.vehicle.hover()
                raise TimeoutError('graph_route_changed_before_execution')
            if len(path) > 1:
                delta = path[1]-current
                yaw = math.atan2(float(delta[1]), float(delta[0]))
                if (np.linalg.norm(delta[:2]) > .01
                        and heading_difference(yaw, float(self.vehicle.get_euler()[2])) > math.radians(15)):
                    self._turn_for_graph_departure(yaw)
                    # Turning may reveal a new obstacle or change map evidence.
                    fresh = self._grid()
                    path[0] = self.vehicle.get_position().copy()
                    if not all(fresh.segment_free(a, b) for a, b in zip(path, path[1:])):
                        raise TimeoutError('graph_route_changed_before_execution')
            continuous = len(path) > 2 and fresh.segment_free(path[0], path[2])
        return super()._move(path, recovery=recovery, continuous=continuous)

    def _turn_for_graph_departure(self, yaw):
        from navigation.route import execution_segment_free
        self._hold_for_planning()
        grid = self._grid()
        previous = self._motion_progress
        foreground = self._foreground_graph
        self._motion_progress = None
        self._foreground_graph = True
        revision = getattr(self.global_map, 'revision', 0)
        anchor = self.vehicle.get_position().copy()
        def check_turn():
            self._check_health()
            current = self.vehicle.get_position()
            snapshot = self.local_map.execution_snapshot
            if (getattr(self.global_map, 'revision', 0) != revision
                    or snapshot is None
                    or not execution_segment_free(snapshot, current, anchor, self.config, grid)):
                raise TimeoutError('graph_departure_without_clearance')
        try:
            check_turn()
            self._event('WILDOS_DEPARTURE_TURN', yaw=yaw, position_ned=anchor.tolist())
            self.vehicle.align_yaw(yaw, health_check=check_turn,
                timeout=max(.01, min(self.config.leg_timeout, self.deadline-self.clock())))
            check_turn()
        except TimeoutError as error:
            self.vehicle.hover()
            if str(error) == 'graph_departure_without_clearance':
                raise
            raise TimeoutError('graph_departure_turn_timeout') from error
        finally:
            self._foreground_graph = foreground
            self._motion_progress = ((self.vehicle.get_position().copy(), self.clock())
                                     if previous is not None else None)

    def _observe_arrival(self):
        self._hold_for_planning()
        grid = self._grid()
        if self.active_viewpoint is not None and grid.cell(self.vehicle.get_position()) in grid.free:
            local = NavigationGrid.from_local(self.local_map.snapshot(), self.altitude,
                                              self.config, self.global_map.snapshot())
            if local.cell(self.vehicle.get_position()) not in local.free:
                raise TimeoutError('frontier_observation_without_local_clearance')
            self._align(self.active_viewpoint.yaw)
        super()._observe_arrival()

    def _checkpoint(self):
        transient = getattr(self, '_transient_target_failure', None)
        if transient is not None:
            key, previous_count = transient
            # The inherited loop has recorded a failed W and incremented its
            # target counter. Keep the history, undo only this misattribution.
            self.failed_frontiers[key] = (previous_count,
                self.clock()+self.exploration_config.frontier_retry_cooldown)
            self._transient_target_failure = None
        # Preserve the original checkpoint protocol, but label actual policy.
        if self.result is not None:
            self.result.update(strategy='wildos_frontier_graph',
                motion_policy='graph_route_existing_guarded_waypoints_ego',
                height_policy='fixed_graph_height', graph_updates=self.frontiers.update_count)
            self.memory.exploration = self.result.copy()
            write_json(self.output_path, self.result)
        if self.frontiers.graph is not None and self._saved_updates != self.frontiers.update_count:
            destination = self.graph_directory / f'update_{self.frontiers.update_count:06d}'
            self.graph_directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.wildos-graph-', dir=self.graph_directory) as temporary:
                stage = Path(temporary) / destination.name
                export_graph(stage, self.frontiers.adapter, self.frontiers.graph, self.frontiers.validation)
                write_json(stage / 'incremental_state.json', self.frontiers.state)
                grid = self.frontiers.adapter.grid
                # Save exact constrained planning evidence, including radius clipping.
                np.savez_compressed(stage / 'planning_grid.npz',
                    known_indices=np.asarray(sorted(grid.known), dtype=np.int64).reshape(-1, 2),
                    free_indices=np.asarray(sorted(grid.free), dtype=np.int64).reshape(-1, 2),
                    occupied_indices=np.asarray(sorted(grid.occupied), dtype=np.int64).reshape(-1, 2),
                    origin=grid.origin, resolution=grid.resolution, altitude_ned=grid.altitude)
                graph_page = stage / 'graph.html'
                content = graph_page.read_text(encoding='utf-8').replace('离线导航图', '探索规划图')
                content = content.replace('尚未接入实时探索或飞行。', '已接入前沿选点与图路径规划；此页为当时快照。')
                graph_page.write_text(content, encoding='utf-8')
                stage.rename(destination)
            self.latest_graph_directory = destination
            self._saved_updates = self.frontiers.update_count
            from .snapshot_retention import prune_graph_snapshots
            prune_graph_snapshots(self.graph_directory, destination)
        if self.frontiers.last_route is not None:
            write_json(self.graph_directory / 'last_route.json', self.frontiers.last_route)
        if self.result is None:
            self.memory.exploration = dict(strategy='wildos_frontier_graph',
                exploration_planner=self.frontiers.backend_name, state='exploring',
                motion_policy='graph_route_existing_guarded_waypoints_ego',
                height_policy='fixed_graph_height', returned_home=False,
                graph_updates=self.frontiers.update_count, execution_events=list(self.events))
        if self.checkpoint is not None:
            self.checkpoint(self)


class VehicleTargetNavigator(SavedMapNavigator):
    """Keep the same no-overflight policy for this session's later target trips."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_map = VehiclePlanningMap(self.global_map, self._objects, self)
        self._guard = VehicleAwareGuard(self)

    def _objects(self):
        temporary = {'nodes': []}
        attach_objects(temporary, {'objects': list(self.records)}, {}, 1., max_links=0)
        return temporary['semantic_nodes']


from .target_navigation import GraphTargetNavigator
from .target_session import GraphTargetSessionMixin


class GraphExplorationSession(GraphTargetSessionMixin, ExplorationSession):
    navigator_type = GraphTargetNavigator

    def _arguments(self, request, output):
        arguments = super()._arguments(request, output)
        arguments['global_map'] = self.runtime.global_sparse_map
        if request is not None:
            arguments.update(graph_provider=lambda: self._validate_target_frame(),
                frame_validator=self._validate_target_frame, objects_provider=self._target_objects,
                live_map=getattr(self, 'live_map', None))
        return arguments

    def annotate(self, selected=None, approach=None, route=None, state='exploring'):
        self.selected, self.approach, self.route = selected, approach, route
        # Render privately: never replace the public page with a half-composed
        # upper-only map, and never share a staging file with HTTP readers.
        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.wildos-map-', dir=self.map_path.parent) as directory:
            raw_path = Path(directory) / 'map.html'
            from .identity_overrides import corrected_records
            display_memory = copy.copy(self.memory)
            display_memory.records = {r['runtime_instance_id']:r for r in corrected_records(
                self.memory.records.values(), self.map_path.parent, self.memory.session_id)}
            render_map(raw_path, self.runtime.global_sparse_map.snapshot(), display_memory,
                selected=selected, approach=approach, route=route,
                vehicle=self.runtime.vehicle.get_position(), state=state,
                navigation_config=self.runtime.semantic_navigation_config,
                exploration_waypoints=self.exploration_waypoints, show_place_links=self.show_place_links)
            exploration_html = raw_path.read_text(encoding='utf-8')
        explorer = getattr(self, 'graph_explorer', None)
        if explorer is not None and explorer.frontiers.graph is not None:
            from .graph_export import render_graph_html
            graph = dict(explorer.frontiers.graph)
            viewer = getattr(self, 'live_map', None)
            navigation = viewer.navigation_snapshot() if viewer is not None else None
            if navigation is None:
                navigator = getattr(getattr(self.runtime,'controller',None),'semantic_navigator',None)
                navigation = getattr(navigator,'_navigation_state',None)
            if isinstance(navigation,dict):
                graph['navigation'] = copy.deepcopy(navigation)
                graph['robot_attachment'] = dict(position_ned=self.runtime.vehicle.get_position().tolist(),
                    node_id=None,node_stable_id=None,component_id=None,status='current_pose')
            graph_html = render_graph_html(explorer.frontiers.adapter, graph, explorer.frontiers.validation)
        else:
            graph = dict(nodes=[], edges=[], components=[], frontiers=[])
            graph_html = '<svg viewBox="-1 -1 2 2"></svg>'
        combined = compose_map_html(exploration_html, graph_html,
                                    graph, self.home, self.home_yaw)
        from .map_publication import publish_map_page
        publish_map_page(self, combined)

    def explore(self):
        arguments = self._arguments(None, self.directory / 'exploration_report.json')
        arguments['global_map'] = self.runtime.global_sparse_map
        explorer = GraphExplorer(
            **arguments,
            exploration_config=self.config,
            memory=self.memory,
            checkpoint=self.checkpoint,
        )
        self.graph_explorer = explorer
        explorer.live_map = getattr(self, 'live_map', None)
        self.runtime.controller = VehicleAwareMissionController.wrap(self.runtime.controller, explorer._objects)
        self.runtime.controller.semantic_navigator = explorer
        from .graph_video import GraphVideoRecorder
        video = GraphVideoRecorder(self.directory/'graph_exploration.mp4', self.runtime.vehicle.get_position,
                                   self.home, self.home_yaw)
        explorer.graph_video = video
        try:
            video.start()
            self.runtime.run_mission()
            self._confirm_home()
            video.close()
            self.runtime.finish_exploration_recording()
            frame = self.perception.semantic_snapshot()
            if frame is not None:
                self.memory.observe(frame, "place_0000")
            numbering = self.memory.number_final_instances()
            write_json(self.directory / "instance_id_mapping.json", {
                "source_namespace": "live_detection", "target_namespace": "final_map",
                "old_to_new": numbering,
            })
            self.memory.exploration["returned_home"] = True
            explorer.result["returned_home"] = True
            write_json(
                self.directory / "exploration_report.json", self.memory.exploration
            )
            self.memory.save_bundle(
                self.directory, self.runtime.global_sparse_map.snapshot(), explorer.config
            )
            self.annotate(state="READY: returned home")
        except BaseException as error:
            self.memory.exploration["returned_home"] = False
            self.memory.exploration["failure_phase"] = explorer.phase
            self.memory.exploration["error"] = str(error)
            try:
                self.memory.checkpoint(self.directory)
                self.memory.save_bundle(self.directory, self.runtime.global_sparse_map.snapshot(), explorer.config)
                write_json(self.directory / "exploration_report.json", self.memory.exploration)
                self.annotate(state=f"未确认返航完成：{explorer.phase} / {error}")
            except Exception as error:
                print(f"--- 无法保存异常检查点：{error}；保留原始异常 ----")
            raise
        finally:
            video.close()
        return explorer.result


def open_map_page(path, url=None):
    """Open the generated map once, before takeoff, using the Windows file handler."""
    path = Path(path).resolve()
    print(f"--- 本次探索地图：{path} ----")
    if not path.is_file():
        print("--- 地图页面尚未生成，暂未打开浏览器 ----")
        return False
    if url is not None:
        print(f"--- 实时地图地址：{url} ----")
        try:
            return bool(webbrowser.open(url, new=2))
        except Exception as error:
            print(f"--- 自动打开失败：{error}；请手动打开实时地图地址 ----")
            return False
    if os.name == 'nt':
        try:
            os.startfile(str(path))
            print("--- 已请求系统默认程序打开地图 ----")
            return True
        except OSError as error:
            print(f"--- 系统打开地图失败：{error}；尝试浏览器方式 ----")
    try:
        if webbrowser.open(path.as_uri(), new=2):
            print("--- 已请求浏览器打开地图 ----")
            return True
    except Exception as error:
        print(f"--- 打开地图失败：{error} ----")
    print(f"--- 请手动打开地图：{path} ----")
    return False


def main():
    config, args = _CLI or parse_config(include_perception=True)
    backend = args.perception_backend
    from .session_service import WildOSSessionServer as SessionServer

    # Acquire the single local control endpoint BEFORE sending any scene command.
    server = SessionServer(uuid.uuid4().hex, config.server_port)
    runtime = sam3_service = live_map = None
    try:
        from application import DemoRuntime, WebSocketRequestSender
        from detections import DetectorConfig

        detector = replace(DetectorConfig(), backend=backend)
        if backend == "sam3":
            from detections.sam3_process import SAM3ServiceProcess
            detector = replace(detector, window_name="SAM3 Detection / Semantic Segmentation")
            sam3_service = SAM3ServiceProcess(detector.sam3_url)
            sam3_service.start()

        WebSocketRequestSender().send_when_ue_ready()
        if args.fastlio:
            print("--- FAST-LIO 实验建图：假定雷达单帧瞬时采集；探索及控制仍使用 PX4 地图 ----")
        runtime = DemoRuntime(exploration=True, exploration_config=config, detector_config=detector,
                              fastlio_mode='live' if args.fastlio else None,
                              lidar_scan_model='snapshot_assumed' if args.fastlio else 'unverified')
        runtime.keep_session = args.keep_session
        runtime.start()
        session = GraphExplorationSession(runtime, config, session_id=server.session_id)
        server.map_path = str(session.map_path.resolve())
        server.publish_descriptor(session.directory / "session.json")
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        print(f"--- WildOS graph exploration output: {session.directory} ----")
        session.annotate(state="准备起飞：雷达建图 / WildOS 前沿节点探索")
        from .live_map import LiveMapServer
        live_map = LiveMapServer(session.map_path)
        session.live_map = live_map
        server.live_map_url = live_map.url
        from navigate_wildos_target import DEFAULT_SESSION_FILE as WILDOS_SESSION_FILE
        server.publish_descriptor(WILDOS_SESSION_FILE)
        server.publish_descriptor(session.directory / "session.json")
        server.publish_descriptor(DEFAULT_SESSION_FILE)
        open_map_page(session.map_path, live_map.url)
        session.explore()
        if not args.keep_session:
            print(f'--- 已确认落地并保存探索地图：{session.map_path} ----')
            print('--- 正在关闭感知、SAM3 和建图后台，释放本次运行资源 ----')
            print('--- 后续图导航可独立加载保存地图：python navigate_wildos_target.py ----')
            return
        server.set_state("READY")
        print("--- 探索结果已保存，后台录像、语义日志和回环重建已停止写入；等待目标导航 ----")
        print(f"--- 已返航并确认落地。保持本终端运行。地图：{session.map_path} ----")
        print("--- 另开终端执行：python navigate_wildos_target.py --instance-id vehicle_01 ----")
        while True:
            try:
                command, identity, request = server.commands.get(timeout=0.25)
            except queue.Empty:
                if server.state == "READY":
                    try:
                        session.watch.check()
                    except RuntimeError as error:
                        server.set_state("ERROR")
                        print(f"--- Session continuity lost: {error} ----")
                continue
            if command == "shutdown":
                if session.airborne_navigation:
                    session.return_home()
                break
            try:
                result = (session.return_home() if command == "return_home"
                          else session.navigate(request, identity))
                server.finish(identity, result)
            except TargetUnavailableError as error:
                server.finish(
                    identity,
                    {"success": False, "reason": str(error), "flight_started": False},
                )
            except Exception as error:
                # A flight/runtime fault requires recovery before another command.
                server.finish(
                    identity, {"success": False, "reason": str(error)}, fatal=True
                )
                runtime.land_if_armed(hover_first=True)
                session.annotate(
                    session.selected,
                    session.approach,
                    session.route,
                    "ERROR: " + str(error),
                )
    except KeyboardInterrupt:
        print("--- Stopping exploration session ----")
    finally:
        try:
            shutdown_session(server, runtime, primary_error=sys.exc_info()[1], sam3_service=sam3_service)
        finally:
            if live_map is not None:
                live_map.close()


if __name__ == "__main__":
    main()
