"""Retire redundant routing nodes while retaining their explored-region memory."""
import copy


def retire_compacted_nodes(state, graph):
    state = copy.deepcopy(state)
    retained = {n['stable_id'] for n in graph['nodes']}
    retired = [n for n in state['nodes'] if n.get('active', True) and n['stable_id'] not in retained]
    memory = {m['stable_id']: m for m in state.get('explored_memory', [])}
    for node in retired:
        memory[node['stable_id']] = dict(stable_id=node['stable_id'],
            position_ned=list(node['position_ned']), explored_radius=node['explored_radius'])
    state['nodes'] = [n for n in state['nodes'] if not n.get('active', True) or n['stable_id'] in retained]
    state['explored_memory'] = list(memory.values())
    state['graph'] = copy.deepcopy(graph)
    state['retirement'] = dict(retired_ids=[n['stable_id'] for n in retired],
                              retained_sampling_nodes=len(state['nodes']),
                              archived_explored_regions=len(memory))
    return state
