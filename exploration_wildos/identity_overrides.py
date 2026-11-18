"""Session-scoped user-confirmed identity corrections, separate from localization."""
import copy
import json
from pathlib import Path
import numpy as np


def merge_identities(records, spec, session_id):
    if spec.get('session_id') != session_id:
        raise ValueError('Vehicle identity overrides belong to a different session')
    result={r['runtime_instance_id']:copy.deepcopy(r) for r in records}
    source_namespace = spec.get('source_namespace', 'final_map')
    if source_namespace not in ('final_map', 'live_detection'):
        raise ValueError('Unknown vehicle identity namespace: '+str(source_namespace))
    source_ids = {}
    numbered = any('source_runtime_instance_id' in r for r in result.values())
    if source_namespace == 'live_detection':
        for record in result.values():
            for part in (record, *record.get('identity_geometry_parts', [])):
                source = part.get('source_runtime_instance_id', part['runtime_instance_id'])
                target = part['runtime_instance_id']
                if source in source_ids and source_ids[source] != target:
                    raise ValueError('Ambiguous live vehicle identity: '+source)
                source_ids[source] = target
    for old_source, canonical_source in spec.get('redirects',{}).items():
        old, canonical = old_source, canonical_source
        if source_namespace == 'live_detection':
            old = source_ids.get(old_source, None if numbered else old_source)
            canonical = source_ids.get(canonical_source, None if numbered else canonical_source)
        if old == canonical or canonical not in result:
            raise ValueError('Invalid canonical vehicle identity: '+str(canonical))
        keep=result[canonical]
        fragment=result.pop(old,None)
        keep['merged_from']=sorted(set(keep.get('merged_from',[]))|({old} if old is not None else set())|
                                   set((fragment or {}).get('merged_from',[])))
        if source_namespace == 'live_detection':
            keep['source_merged_from'] = sorted(set(keep.get('source_merged_from', []))|{old_source})
        parts=keep.setdefault('identity_geometry_parts',[])
        if fragment is not None:
            parts.append({key:copy.deepcopy(fragment[key]) for key in
                          ('runtime_instance_id','position_world','bbox_3d','source_runtime_instance_id')
                          if key in fragment})
        boxes=[keep['bbox_3d']]+[p['bbox_3d'] for p in parts]
        keep['planning_bbox_3d']=dict(min=np.min([b['min'] for b in boxes],axis=0).tolist(),
                                      max=np.max([b['max'] for b in boxes],axis=0).tolist())
        keep['identity_evidence']=('user_confirmed_same_column' if keep.get('class_name') == 'column'
                                   else 'user_confirmed_same_vehicle')
    return list(result.values())


def corrected_records(records, directory, session_id):
    result = list(records)
    for name in ('vehicle_identity_overrides.json', 'semantic_identity_overrides.json'):
        path = Path(directory)/name
        if path.exists():
            result = merge_identities(result, json.loads(path.read_text(encoding='utf-8')), session_id)
    return result
