"""Replay raw observations with no manual overrides or recorded ID decisions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping.semantic_instance_mapper import SemanticInstanceMapper, instance_class_name
from mapping.explore_v3.adapters import SemanticInstanceMapper as V3Mapper
from mapping.loop_semantic_replay import transform_measurement
from navigation.scene_memory import write_json


def audit(history, mapper_type=SemanticInstanceMapper):
    mapper = mapper_type()
    links = defaultdict(Counter)
    packets = measurements = 0
    started = time.monotonic()
    previous = None
    with Path(history).open(encoding='utf-8') as stream:
        for line in stream:
            packet = json.loads(line)
            if packet == previous:
                continue
            previous = packet
            samples = packet['samples']
            identities = mapper.update([s['class_name'] for s in samples], [s['confidence'] for s in samples],
                [transform_measurement(s['measurement'], np.eye(4)) for s in samples], packet['timestamp'])
            for sample, identity in zip(samples, identities):
                # Evaluation labels are read only AFTER association has finished.
                source = sample.get('source_instance_id')
                if identity is not None and source is not None:
                    category = instance_class_name(sample['class_name'])
                    links[f'{category}_{identity:02d}'][f'{category}_{source:02d}'] += 1
            packets += 1
            measurements += len(samples)
    records = mapper.navigation_records(3)
    for record in records:
        votes = Counter()
        for alias in [record['runtime_instance_id'], *record.get('merged_from', [])]:
            votes.update(links[alias])
        record['source_observations'] = dict(votes)
    vehicles = mapper._vehicles
    return dict(history=str(Path(history).resolve()), backend=mapper_type.__module__,
        manual_overrides_used=False, recorded_ids_used_for_association=False,
        packets=packets, measurements=measurements,
        category_counts=dict(Counter(instance_class_name(r['class_name']) for r in records)),
        records=records, deferred_column_connections=[dict(parent=r['runtime_instance_id'],
            targets=[t['target_id'] for t in r.get('connection_targets', [])])
            for r in records if r['class_name'] == 'column' and r.get('connection_targets')],
        deferred_vehicle_fragments=[dict(observation_count=t.observation_count,
            possible_parents=[f'vehicle_{i:02d}' for i in sorted(t.possible_parents)])
            for t in vehicles._pending if t.observation_count >= vehicles.confirmation_observations],
        duration_seconds=round(time.monotonic()-started, 3))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path)
    parser.add_argument('--backend', choices=['current', 'v3', 'both'], default='both')
    parser.add_argument('--output', type=Path, help='One compact JSON; existing report is replaced')
    args = parser.parse_args(argv)
    backends = {'current': SemanticInstanceMapper, 'v3': V3Mapper}
    report = {name: audit(args.history, mapper) for name, mapper in backends.items()
              if args.backend in (name, 'both')}
    if args.output:
        write_json(args.output, report)
    print(json.dumps({name: {k:r[k] for k in ('packets', 'measurements', 'category_counts', 'duration_seconds')}
                      for name, r in report.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
