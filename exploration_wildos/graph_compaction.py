"""Compact planning nodes while preserving covered safe cells and components.

Sampling/history nodes are stored separately by the workflow, so this does not
remove the incremental exploration memory or change retained stable identities.
"""
from collections import Counter
import copy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .node_reduction import covered_cells
from .static_graph import _components


def compact_graph(graph, grid, path_stretch=1.2):
    graph=copy.deepcopy(graph)
    nodes,edges=graph['nodes'],graph['edges']
    covers=[covered_cells(grid,n) for n in nodes]
    counts=Counter(c for cells in covers for c in cells)
    adjacency={n['id']:set() for n in nodes}
    for e in edges:
        adjacency[e['source']].add(e['target']);adjacency[e['target']].add(e['source'])
    anchor=graph['robot_attachment'].get('node_id')
    protected={n['id'] for n in nodes if n.get('is_frontier') or n.get('frontier_ids')}
    if anchor is not None:protected.add(anchor)
    removed=[]
    for i in sorted(adjacency,key=lambda i:(len(covers[i]),i)):
        if i in protected or any(counts[c]<=1 for c in covers[i]):continue
        neighbors=adjacency[i]
        if not neighbors:continue
        start=min(neighbors);seen={start};pending=[start];targets=neighbors-{start}
        while pending and targets:
            current=pending.pop()
            fresh=adjacency[current]-seen-{i}
            seen.update(fresh);targets.difference_update(fresh);pending.extend(fresh)
        if targets:continue
        counts.subtract(covers[i])
        for neighbor in neighbors:adjacency[neighbor].remove(i)
        del adjacency[i];removed.append(i)
    # Restore original shortest-path intermediates until every retained pair
    # has a near-original route before the subsequent edge sparsification.
    initially_removed=set(removed)
    if len(nodes)>2000:
        # Avoid an unbounded all-pairs matrix for large future snapshots.
        kept=set(range(len(nodes)))
    else:
        def matrix(indices):
            ids={old:new for new,old in enumerate(indices)}
            rows=[];cols=[];values=[]
            for e in edges:
                if e['source'] in ids and e['target'] in ids:
                    a,b=ids[e['source']],ids[e['target']]
                    rows.extend((a,b));cols.extend((b,a));values.extend((e['length'],e['length']))
            return csr_matrix((values,(rows,cols)),shape=(len(indices),len(indices)))
        baseline,predecessors=shortest_path(matrix(list(range(len(nodes)))),directed=False,return_predecessors=True)
        kept=set(adjacency)
        while True:
            indices=sorted(kept)
            current=shortest_path(matrix(indices),directed=False)
            target=baseline[np.ix_(indices,indices)]
            bad=np.argwhere(np.triu(np.isfinite(target)&(current>path_stretch*target+1e-9),1))
            if not len(bad):break
            added=set()
            for i,j in bad:
                source,destination=indices[i],indices[j]
                cursor=destination
                while cursor!=source:
                    added.add(cursor);cursor=int(predecessors[source,cursor])
                    if cursor<0:raise ValueError('Missing original path during compaction')
                added.add(source)
            if not added-kept:raise ValueError('Unable to repair compaction path stretch')
            kept.update(added)
    # Path repair may restore several redundant alternatives at once. Recheck
    # these against the ORIGINAL distance matrix, never a cumulative local bound.
    repair_pruned=[]
    if len(nodes)<=2000:
        final_counts=Counter(c for i in kept for c in covers[i])
        for i in sorted(kept, key=lambda j: (len(covers[j]), j)):
            if i in protected or any(final_counts[c]<=1 for c in covers[i]):
                continue
            trial=sorted(kept-{i})
            if not trial:
                continue
            current=shortest_path(matrix(trial),directed=False)
            target=baseline[np.ix_(trial,trial)]
            if np.any(np.isfinite(target)&(current>path_stretch*target+1e-9)):
                continue
            # Retain a representative even for an isolated component.
            if not any(i in c and any(j in kept-{i} for j in c) for c in graph['components']):
                continue
            kept.remove(i);final_counts.subtract(covers[i]);repair_pruned.append(i)
    removed=sorted(set(range(len(nodes)))-kept)
    mapping={old:new for new,old in enumerate(sorted(kept))}
    graph['nodes']=[dict(nodes[old],id=new) for old,new in mapping.items()]
    graph['edges']=[dict(e,source=mapping[e['source']],target=mapping[e['target']])
                    for e in edges if e['source'] in mapping and e['target'] in mapping]
    for i,e in enumerate(graph['edges']):e['id']=i
    graph['components']=_components(len(graph['nodes']),graph['edges'])
    if len(graph['components'])!=len(_components(len(nodes),edges)):
        raise ValueError('Compaction changed components')
    if anchor is not None:
        graph['robot_attachment']['node_id']=mapping[anchor]
        graph['robot_attachment']['component_id']=next(i for i,c in enumerate(graph['components']) if mapping[anchor] in c)
    for f in graph.get('frontiers',[]):f['node_id']=mapping[f['node_id']]
    graph.pop('edge_sparsification',None)
    graph['diagnostics'].update(node_count=len(graph['nodes']),edge_count=len(graph['edges']),component_count=len(graph['components']))
    graph['node_compaction']=dict(before_nodes=len(nodes),after_nodes=len(graph['nodes']),
        removed_stable_ids=[nodes[i].get('stable_id',str(i)) for i in removed],
        covered_cells_before=len(counts),covered_cells_after=len(set().union(*(covers[i] for i in kept))) if kept else 0,
        protected_nodes=len(protected),components_preserved=True,
        path_repair_restored_nodes=len(initially_removed-set(removed)),
        max_path_stretch_before_edge_sparsification=path_stretch,
        repair_redundancy_removed=[nodes[i].get('stable_id', str(i)) for i in repair_pruned],
        bounded_matrix_fallback=len(nodes)>2000,
        note='Zero incremental safe-cell-center coverage only; frontier/anchor protected. Path stretch checked separately.')
    return graph
