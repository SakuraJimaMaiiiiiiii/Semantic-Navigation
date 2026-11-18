"""Remove redundant static nodes while retaining sampled-cell coverage and routes."""
from collections import Counter
import heapq
import math


def covered_cells(grid, node):
    x, y = node['cell']
    radius = node['free_radius']
    offset = (0., 0.)
    if hasattr(grid, 'point') and 'position_ned' in node:
        center = grid.point((x,y))
        offset = (center[0]-node['position_ned'][0], center[1]-node['position_ned'][1])
    extent = math.ceil(radius / grid.resolution)+1
    return {(x+dx, y+dy) for dx in range(-extent, extent+1)
            for dy in range(-extent, extent+1)
            if (x+dx, y+dy) in grid.free
            and math.hypot(dx*grid.resolution+offset[0],dy*grid.resolution+offset[1]) <= radius+1e-9}


def reduce_nodes(nodes, edges, grid, protected=(), max_detour=1.15):
    """Greedy deletion; no radius changes, node moves, or new uncertified edges.

    Coverage means safe cell centers. The detour limit applies to each removal,
    not a global shortest-path stretch guarantee after all removals.
    """
    covers = [covered_cells(grid, node) for node in nodes]
    counts = Counter(c for cells in covers for c in cells)
    neighbors = {n['id']: {} for n in nodes}
    for edge in edges:
        a, b = edge['source'], edge['target']
        neighbors[a][b] = neighbors[b][a] = edge['length']
    protected = set(protected) | {n['id'] for n in nodes if n.get('frontier_ids') or n.get('is_frontier')}
    removed, reasons = [], Counter()

    def detour_available(node):
        adjacent = sorted(neighbors[node])
        if not adjacent:
            return False  # Preserve the last representative of every component.
        for index, start in enumerate(adjacent[:-1]):
            targets = {other: max_detour*(neighbors[node][start]+neighbors[node][other])
                       for other in adjacent[index+1:]}
            cutoff = max(targets.values())
            distance, queue = {start: 0.}, [(0., start)]
            while queue:
                cost, current = heapq.heappop(queue)
                if cost != distance[current]:
                    continue
                if current in targets:
                    if cost > targets[current]+1e-9:
                        return False
                    del targets[current]
                    if not targets:
                        break
                for nxt, length in neighbors[current].items():
                    trial = cost+length
                    if nxt != node and trial <= cutoff+1e-9 and trial < distance.get(nxt, math.inf):
                        distance[nxt] = trial
                        heapq.heappush(queue, (trial, nxt))
            if targets:
                return False
        return True

    # Small disks first: remove locally redundant boundary samples before broad coverage.
    for identity in sorted(neighbors, key=lambda i: (len(covers[i]), i)):
        if identity in protected:
            reasons['protected'] += 1
        elif any(counts[cell] <= 1 for cell in covers[identity]):
            reasons['unique_coverage'] += 1
        elif not detour_available(identity):
            reasons['connectivity_or_detour'] += 1
        else:
            counts.subtract(covers[identity])
            for other in list(neighbors[identity]):
                del neighbors[other][identity]
            del neighbors[identity]
            removed.append(identity)
    mapping = {old: new for new, old in enumerate(sorted(neighbors))}
    retained = [dict(nodes[old], id=new, sample_id=old) for old, new in mapping.items()]
    retained_edges = [dict(edge, source=mapping[edge['source']], target=mapping[edge['target']])
                      for edge in edges if edge['source'] in mapping and edge['target'] in mapping]
    for i, edge in enumerate(retained_edges):
        edge['id'] = i
    return retained, retained_edges, dict(
        enabled=True, before_nodes=len(nodes), after_nodes=len(retained), before_edges=len(edges),
        after_edges=len(retained_edges), removed_sample_ids=removed, retained_reasons=dict(reasons),
        covered_cells_before=len(counts), covered_cells_after=sum(value > 0 for value in counts.values()),
        max_detour_per_removal=max_detour,
        note='Safe cell-center coverage preserved; detour bound is per deletion, not global')
