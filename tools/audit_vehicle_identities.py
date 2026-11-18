"""Replay raw vehicle observations through the live tracker without flying or overrides."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mapping.vehicle_instance_tracker import VEHICLE_CLASSES, VehicleInstanceTracker
from mapping.loop_semantic_replay import transform_measurement
from navigation.scene_memory import write_json


def audit(history):
    tracker = VehicleInstanceTracker()
    links = defaultdict(Counter)
    packets = measurements = 0
    started = time.monotonic()
    last_packet = None
    with Path(history).open(encoding='utf-8') as stream:
        for line in stream:
            packet = json.loads(line)
            if packet == last_packet:
                continue
            last_packet = packet
            samples = [sample for sample in packet['samples'] if sample['class_name'] in VEHICLE_CLASSES]
            identities = tracker.update([s['class_name'] for s in samples], [s['confidence'] for s in samples],
                [transform_measurement(s['measurement'], np.eye(4)) for s in samples], packet['timestamp'])
            for sample, identity in zip(samples, identities):
                # Recorded IDs are evaluation labels only; never fed into association.
                source = sample.get('source_instance_id')
                if identity is not None and source is not None:
                    links[identity][f'vehicle_{source:02d}'] += 1
            packets += 1
            measurements += len(samples)
    connections = tracker.connection_hypotheses()
    records = []
    for identity, track in sorted(tracker.tracks.items()):
        sources = Counter()
        for old, counts in links.items():
            if tracker.canonical_id(old) == identity:
                sources.update(counts)
        records.append(dict(runtime_instance_id=f'vehicle_{identity:02d}', source_observations=dict(sources),
            position_ned=track.map_position.tolist(), position_status=track.surface.vehicle_geometry()['position_status'],
            identity_status='needs_connection_view' if connections.get(identity) else 'observed',
            connection_targets=connections.get(identity, [])))
    return dict(history=str(Path(history).resolve()), coordinate_frame='raw_world_NED',
        manual_overrides_used=False, recorded_ids_used_for_association=False,
        packets=packets, measurements=measurements, published_vehicle_count=len(records), vehicles=records,
        deferred_fragments=[dict(observation_count=track.observation_count,
            possible_parents=[f'vehicle_{i:02d}' for i in sorted(track.possible_parents)],
            bbox_3d=dict(min=track.bbox_3d_min.tolist(), max=track.bbox_3d_max.tolist()))
            for track in tracker._pending if track.observation_count >= tracker.confirmation_observations],
        automatic_merges=tracker.redirects, duration_seconds=round(time.monotonic()-started, 3))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path, help='maps/semantic_observations.jsonl from an exploration run')
    parser.add_argument('--output', type=Path, help='Overwrite one compact diagnostic JSON; no map or sensor copy')
    args = parser.parse_args(argv)
    report = audit(args.history)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
