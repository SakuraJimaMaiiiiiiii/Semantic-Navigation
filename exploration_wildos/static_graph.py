"""Bounded, deterministic single-height sampling and certified graph edges.

Radii are conservative 2-D distances to cell volumes, not 3-D visibility or
vehicle-clearance certificates. Safety uses the existing eroded NavigationGrid.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from .node_reduction import covered_cells, reduce_nodes


@dataclass(frozen=True)
class GraphConfig:
    paper_aligned: bool = False
    incremental_cleanup: bool = False
    traversability_clearance: float = 0.5
    seed: int = 0
    max_samples: int = 2000
    max_nodes: int = 1000
    coverage_samples: int = 2000
    free_radius_cap: float = 2.0
    edge_radius: float = 3.0
    max_dense_cells: int = 4_000_000
    prune_redundant: bool = True
    prune_max_detour: float = 1.15

    def __post_init__(self):
        if type(self.incremental_cleanup) is not bool:
            raise ValueError('incremental_cleanup must be boolean')
        if type(self.paper_aligned) is not bool:
            raise ValueError('paper_aligned must be boolean')
        if type(self.prune_redundant) is not bool:
            raise ValueError('prune_redundant must be boolean')
        if not math.isfinite(self.prune_max_detour) or self.prune_max_detour < 1:
            raise ValueError('prune_max_detour must be finite and at least 1')
        for name in ('max_samples', 'max_nodes', 'max_dense_cells', 'coverage_samples'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for name in ('free_radius_cap', 'edge_radius', 'traversability_clearance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')


class DistanceField:
    def __init__(self, adapter, max_cells):
        grid = adapter.grid
        self.grid = grid
        self.minimum = np.zeros(2, dtype=np.int64)
        self.obstacle = self.unknown = None
        if not grid.known:
            return
        points = np.asarray(sorted(grid.known), dtype=np.int64)
        self.minimum = points.min(axis=0) - 1
        shape = points.max(axis=0) - self.minimum + 2
        if math.prod(map(int, shape)) > max_cells:
            raise ValueError('Distance-field bounding box exceeds max_dense_cells; use a smaller snapshot/window')
        known = np.zeros(tuple(shape), dtype=bool)
        indices = points - self.minimum
        known[indices[:, 0], indices[:, 1]] = True
        # Outer padding is explicitly UNKNOWN, even for an all-free input.
        self.unknown = self._conservative(distance_transform_edt(known))
        obstacle = np.zeros(tuple(shape), dtype=bool)
        if grid.occupied:
            indices = np.asarray(sorted(grid.occupied)) - self.minimum
            obstacle[indices[:, 0], indices[:, 1]] = True
            self.obstacle = self._conservative(distance_transform_edt(~obstacle))

    def _conservative(self, distances):
        # Subtract cell circumradius: lower bound to the union of occupied/unknown squares.
        return np.maximum(0., distances * self.grid.resolution - self.grid.resolution / math.sqrt(2))

    def at_point(self, point):
        cell = self.grid.cell(point)
        obstacle, unknown = self.at(cell)
        offset = float(np.linalg.norm(np.asarray(point)[:2]-self.grid.point(cell)[:2]))
        return max(0., obstacle-offset), max(0., unknown-offset)

    def at(self, cell):
        if self.unknown is None:
            return 0., 0.
        index = np.asarray(cell) - self.minimum
        if np.any(index < 0) or np.any(index >= self.unknown.shape):
            return 0., 0.
        index = tuple(index)
        return (math.inf if self.obstacle is None else float(self.obstacle[index]),
                float(self.unknown[index]))


def _components(count, edges):
    neighbors = [set() for _ in range(count)]
    for edge in edges:
        a, b = edge['source'], edge['target']
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(range(count))
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        pending, group = [seed], []
        while pending:
            node = pending.pop()
            group.append(node)
            fresh = neighbors[node] & remaining
            remaining.difference_update(fresh)
            pending.extend(sorted(fresh, reverse=True))
        result.append(sorted(group))
    return result


def build_graph(adapter, options=None, start=None):
    started = time.perf_counter()
    options = options or GraphConfig()
    grid = adapter.grid
    field = DistanceField(adapter, options.max_dense_cells)
    nodes, positions, radii = [], [], []
    rejected = Counter()
    cells = sorted(grid.free)
    rng = np.random.default_rng(options.seed)
    order = rng.permutation(len(cells))[:options.max_samples]
    sampled = set()

    def sample(cell):
        sampled.add(cell)
        point = grid.point(cell)
        if positions and np.any(np.linalg.norm(np.asarray(positions) - point, axis=1) <= radii):
            rejected['covered_by_existing_or_new_node'] += 1
            return
        obstacle, unknown = field.at(cell)
        radius = min(obstacle, unknown, options.free_radius_cap)
        nodes.append(dict(id=len(nodes), position_ned=point.tolist(), cell=list(cell),
                          free_radius=radius, explored_radius=unknown, valid=True,
                          frontier_ids=[], is_frontier=False,
                          validated_snapshot=adapter.metadata.get('snapshot_id')))
        positions.append(point)
        radii.append(radius)

    for index in order:
        if len(nodes) >= options.max_nodes:
            break
        sample(cells[int(index)])
    # Bounded second pass prioritizes low-clearance cells, including narrow passages.
    remaining = sorted((c for c in cells if c not in sampled), key=lambda c: (min(field.at(c)), c))
    for cell in remaining[:options.coverage_samples]:
        if len(nodes) >= options.max_nodes:
            break
        sample(cell)
    edges = []
    pairs = sorted(cKDTree(positions).query_pairs(options.edge_radius)) if positions else []
    for a, b in pairs:
        if grid.segment_free(positions[a], positions[b]):
            edges.append(dict(id=len(edges), source=a, target=b,
                              length=float(np.linalg.norm(positions[a] - positions[b])), valid=True,
                              validated_snapshot=adapter.metadata.get('snapshot_id')))
        else:
            rejected['unsafe_edge'] += 1
    protected = []
    if start is not None:
        start = np.asarray(start, dtype=float)
        if start.shape != (3,) or not np.isfinite(start).all():
            raise ValueError('start must be finite NED xyz')
        if abs(start[2]-grid.altitude) <= 1e-8 and grid.cell(start) in grid.free:
            for i in sorted(range(len(nodes)), key=lambda i: (np.linalg.norm(positions[i]-start), i)):
                if np.linalg.norm(positions[i]-start) <= options.edge_radius and grid.segment_free(start, positions[i]):
                    protected = [i]
                    break
    reduction = dict(enabled=False, before_nodes=len(nodes), after_nodes=len(nodes), removed_sample_ids=[])
    if options.prune_redundant:
        nodes, edges, reduction = reduce_nodes(nodes, edges, grid, protected, options.prune_max_detour)
        positions = [np.asarray(node['position_ned']) for node in nodes]
    components = _components(len(nodes), edges)
    attachment = dict(status='not_requested', node_id=None, component_id=None)
    if start is not None:
        start = np.asarray(start, dtype=float)
        if start.shape != (3,) or not np.isfinite(start).all():
            raise ValueError('start must be finite NED xyz')
        attachment['position_ned'] = start.tolist()
        if abs(start[2] - grid.altitude) > 1e-8:
            attachment['status'] = 'wrong_altitude'
        elif grid.cell(start) not in grid.free:
            attachment['status'] = 'unsafe_start'
        else:
            attachment['status'] = 'no_certified_connection'
            for i in sorted(range(len(nodes)), key=lambda i: (np.linalg.norm(positions[i]-start), i)):
                if np.linalg.norm(positions[i]-start) <= options.edge_radius and grid.segment_free(start, positions[i]):
                    attachment.update(status='attached', node_id=i,
                                      component_id=next(k for k, group in enumerate(components) if i in group))
                    break
    # Coverage uses the conservative raw-space balls, not a claim of camera visibility.
    covered = set()
    for node in nodes:
        covered.update(covered_cells(grid, node))
    return dict(schema_version=1, implementation_stage='A_static_single_height',
                metadata=adapter.metadata, options=asdict(options), nodes=nodes, edges=edges,
                components=components, robot_attachment=attachment, reduction=reduction,
                limitations=['No frontier association yet', 'No temporal or loop-closure update yet',
                             'Single-height projection; no 3-D flight certificate',
                             'No live local-map fusion in archive reader'],
                diagnostics=dict(raw_free_cells=len(adapter.raw_free), safe_free_cells=len(cells),
                                 occupied_cells=len(grid.occupied), node_count=len(nodes), edge_count=len(edges),
                                 component_count=len(components), covered_safe_cells=len(covered),
                                 uncovered_safe_cells=[list(c) for c in cells if c not in covered],
                                 sampled_cells=len(sampled), untested_safe_cells=len(cells)-len(sampled),
                                 node_budget_reached=reduction['before_nodes'] >= options.max_nodes,
                                 candidate_edge_pairs=len(pairs), rejected=dict(rejected),
                                 build_seconds=time.perf_counter()-started))


def validate_graph(adapter, graph):
    """Recheck every exported node and edge against the existing navigation grid."""
    grid = adapter.grid
    nodes = graph['nodes']
    invalid_nodes = [n['id'] for n in nodes if grid.cell(n['position_ned']) not in grid.free
                     or abs(n['position_ned'][2] - grid.altitude) > 1e-8]
    invalid_edges = [e['id'] for e in graph['edges']
                     if not grid.segment_free(nodes[e['source']]['position_ned'], nodes[e['target']]['position_ned'])]
    return dict(passed=not invalid_nodes and not invalid_edges, checked_nodes=len(nodes),
                checked_edges=len(graph['edges']), invalid_nodes=invalid_nodes, invalid_edges=invalid_edges,
                certificate='Existing NavigationGrid same-height free-space checks; not real-flight validation')
