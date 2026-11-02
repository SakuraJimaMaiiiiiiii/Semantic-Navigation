"""Check IMU source timing and stationary-data eligibility before noise calibration."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization.imu_noise import analyze_stationary


def resolve_directory(path=None, root=None):
    if path is None:
        root = root or Path(__file__).resolve().parents[1]/'data/imu_calibration'
        candidates = sorted(p for p in root.glob('*')
                            if (p/'imu.csv').is_file() and (p/'capture.json').is_file())
        if not candidates:
            raise ValueError('No completed IMU capture found. Run tools/record_stationary_imu.py first.')
        path = candidates[-1]
    directory = path/'vins' if (path/'vins').is_dir() else path
    if not (directory/'imu.csv').is_file():
        raise ValueError(f'IMU CSV not found: {directory / "imu.csv"}. '
                         'Use a real capture directory, or omit the path to select the latest completed capture.')
    return path, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, nargs='?',
                        help='Run/capture directory; default: latest completed stationary capture')
    args = parser.parse_args()
    try:
        path, directory = resolve_directory(args.path)
    except ValueError as error:
        parser.error(str(error))
    print(f'Input: {directory.resolve()}', flush=True)
    try:
        raw = np.loadtxt(directory/'imu.csv', delimiter=',', skiprows=1, ndmin=2)
    except (OSError, ValueError) as error:
        parser.error(f'Cannot read IMU CSV: {error}')
    if raw.shape[1] != 8 or len(raw) < 20:
        parser.error('IMU CSV requires at least 20 samples and 8 columns; check capture.json for recording errors.')
    samples = raw[:, 1:]
    checked = analyze_stationary(samples)
    timing_keys = ('samples', 'duration_s', 'effective_rate_hz', 'interval_median_s',
                   'interval_p95_s', 'max_gap_s', 'relative_interval_jitter_p95')
    timing = {key: checked[key] for key in timing_keys}
    milliseconds, counts = np.unique(np.rint(np.diff(samples[:, 0])*1000).astype(int), return_counts=True)
    timing['interval_histogram_ms'] = {str(int(ms)): int(count) for ms, count in zip(milliseconds, counts)}
    timing['gaps_over_50ms'] = int(np.sum(np.diff(samples[:, 0]) > .05))
    timing['shared_minus_source_span_s'] = float(np.ptp(raw[:, 0]-raw[:, 1]))
    timing['note'] = 'Timing diagnostics only; stationarity is evaluated separately below. '
    report = {'source_timing': timing, 'static_selection': None}
    metadata = directory/'capture.json'
    if metadata.exists():
        capture = json.loads(metadata.read_text(encoding='utf-8'))
        report['static_selection'] = analyze_stationary(samples,
            confirmed_stationary=capture.get('stationary_checks_passed') is True)
        report['static_selection']['stationarity_confirmed_by_capture'] = capture.get('stationary_checks_passed') is True
        report['packet_diagnostics'] = capture.get('packet_diagnostics', {
            'available': False, 'reason': 'Older capture did not retain pre-filter packet timestamps or update masks; '
                                           'source scheduling, transport loss and filtering cannot be distinguished.'})
    else:
        import h5py
        recordings = sorted((path/'recordings').glob('*.h5'))
        if len(recordings) == 1:
            with h5py.File(recordings[0], 'r') as f:
                t = f['timestamps/timestamp'][:]
                speed = np.linalg.norm(f['state/linear_velocity'][:], axis=1)
                spin = np.linalg.norm(f['state/angular_velocity'][:], axis=1)
                valid = (t >= raw[0, 0]) & (t <= raw[-1, 0]) & (speed < .02) & (spin < .01)
                indices = np.flatnonzero(valid)
                blocks = np.split(indices, np.flatnonzero((np.diff(indices)>1) | (np.diff(t[indices])>.3))+1)
                blocks = [b for b in blocks if len(b)>1]
                if blocks:
                    best = max(blocks, key=lambda b: t[b[-1]]-t[b[0]])
                    a, b = t[best[0]], t[best[-1]]
                    segment = samples[(raw[:, 0]>=a)&(raw[:, 0]<=b)]
                    if len(segment) >= 20:
                        report['static_selection'] = analyze_stationary(segment)
                        report['static_selection']['selection'] = 'Longest low-speed/low-angular-rate telemetry interval; not proven stationary'
                        report['static_selection']['start_shared_timestamp'] = float(a)
    output = directory/'imu_noise_audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
