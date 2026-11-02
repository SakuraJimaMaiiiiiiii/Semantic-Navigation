"""Compare identical recorded scans under PX4 and live FAST-LIO poses, offline."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_fastlio_replay import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-scans', type=int, default=300)
    args = parser.parse_args()
    if args.max_scans < 2:
        parser.error('--max-scans must be at least 2')
    records = []
    with (args.run/'fastlio/shadow_poses.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if 'pose' in row:
                records.append({key: row[key] for key in ('timestamp', 'pose', 'px4_pose')})
    if len(records) < 2:
        parser.error('Not enough estimated poses')
    selected = np.unique(np.linspace(0, len(records)-1, min(args.max_scans, len(records)), dtype=int))
    source = args.run/'fastlio/inputs.h5'
    with h5py.File(source, 'r') as data:
        keys = sorted(data['scans'])
        stamps = np.array([data['scans'][key].attrs['source_timestamp'] for key in keys])
        rows = []
        for i in selected:
            row = records[i]
            index = int(np.argmin(np.abs(stamps-row['timestamp'])))
            if abs(stamps[index]-row['timestamp']) > 1e-8:
                raise ValueError('Estimated pose has no exactly matching recorded scan')
            rows.append({**row, 'scan_key': keys[index]})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'trajectory.jsonl').write_text(
        ''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    (args.output/'source.json').write_text(json.dumps({
        'run': str(args.run.resolve()), 'available_poses': len(records),
        'sampled_scans': len(rows), 'original_data_modified': False,
        'note': 'Identical uniformly sampled scans; PX4 is not ground truth.'}, indent=2), encoding='utf-8')
    compare(source, args.output, args.run/'exploration/semantic_map.json')
    print(f'Comparison saved: {args.output}')


if __name__ == '__main__':
    main()
