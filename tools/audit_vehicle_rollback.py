"""Compare vehicle association backends on identical recorded PX4 measurements."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping.semantic_instance_mapper import SemanticInstanceMapper, VEHICLE_CLASSES
from mapping.explore_v3.semantic_instance_mapper import SemanticInstanceMapper as V3Mapper
from mapping.explore_v3.adapters import SemanticInstanceMapper as ExploreMapper
from mapping.loop_semantic_replay import transform_measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packets = [json.loads(line) for line in
               (args.run/'maps/semantic_observations.jsonl').read_text(encoding='utf-8').splitlines()]
    report = {'run': str(args.run), 'coordinate_frame': 'recorded PX4 world; identical input, no loop correction',
              'note': 'ID counts and position differences are not ground-truth localization accuracy.',
              'backends': {}}
    for name, factory in [('v3', V3Mapper), ('current', SemanticInstanceMapper), ('explore_restored', ExploreMapper)]:
        mapper = factory(sign_min_observations=10)
        links = defaultdict(Counter)
        last_packet = None
        count = 0
        for packet in packets:
            if packet == last_packet:
                continue
            if last_packet is not None and packet['timestamp'] <= last_packet['timestamp']:
                raise ValueError('Conflicting or non-increasing observation timestamps')
            last_packet = packet
            samples = [s for s in packet['samples'] if s['class_name'] in VEHICLE_CLASSES]
            observations = [transform_measurement(s['measurement'], np.eye(4)) for s in samples]
            identities = mapper.update([s['class_name'] for s in samples],
                                       [s['confidence'] for s in samples], observations, packet['timestamp'])
            for sample, identity in zip(samples, identities):
                if identity is not None:
                    links[identity][str(sample.get('source_instance_id'))] += 1
            count += len(samples)
        records = mapper.navigation_records(3)
        for record in records:
            identity = int(record['runtime_instance_id'].split('_')[-1])
            sources = Counter()
            for old, counts in links.items():
                if mapper._vehicles.canonical_id(old) == identity:
                    sources.update(counts)
            record['input_source_id_counts'] = dict(sources)
        report['backends'][name] = {
            'vehicle_measurements': count, 'published_vehicles': len(records),
            'pending_fragments': len(mapper._vehicles._pending),
            'redirects': dict(mapper._vehicles.redirects), 'vehicles': records,
            'merge_diagnostics': mapper.vehicle_merge_diagnostics,
            'confirmed_tracks': [{'id': identity, 'observations': track.observation_count,
                'confidence': float(track.confidence),
                'size_m': (track.bbox_3d_max-track.bbox_3d_min).tolist(),
                'source_id_counts': dict(links[identity])}
                for identity, track in mapper._vehicles.tracks.items()]}
        print(f'{name}: measurements={count}, vehicles={len(records)}, pending={len(mapper._vehicles._pending)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
