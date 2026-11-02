"""Number finalized instances without changing tracker identities during flight."""
from copy import deepcopy
from dataclasses import replace

import numpy as np


def number_exported_instances(records, evidence):
    """Compact each requested category and remap its measured voxel labels."""
    records = deepcopy(list(records))
    mapping = {}
    for category in ('vehicle', 'column'):
        members = sorted((r for r in records if r['runtime_instance_id'].startswith(category + '_')),
                         key=lambda r: int(r['runtime_instance_id'].rsplit('_', 1)[1]))
        mapping.update((r['runtime_instance_id'], f'{category}_{i:02d}')
                       for i, r in enumerate(members, 1))
    for record in records:
        old = record['runtime_instance_id']
        if old not in mapping:
            continue
        record['source_runtime_instance_id'] = old
        record['source_merged_from'] = record.get('merged_from', [])
        record['source_video_instance_id'] = record.get('video_instance_id')
        # Source IDs are provenance, never aliases in the new namespace:
        # a removed source column_02 may now name a different exported column.
        record['merged_from'] = []
        record['instance_id'] = record['runtime_instance_id'] = mapping[old]
        record['video_instance_id'] = None
    if evidence is not None:
        evidence = replace(evidence, instance_ids=np.asarray(
            [mapping.get(str(identity), str(identity)) for identity in evidence.instance_ids], dtype=str))
    return records, evidence, mapping
