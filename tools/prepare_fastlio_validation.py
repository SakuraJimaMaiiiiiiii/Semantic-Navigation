"""Copy a capture, trimming ONLY a trailing all-zero IMU suffix; never edit the source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import h5py
import numpy as np


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(f)


def _digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        result.update(block)
    return result.hexdigest()


def prepare(source, output):
    source, output = Path(source), Path(output)
    if source.is_dir():
        source = source/'fastlio/inputs.h5'
    with h5py.File(source, 'r') as f:
        imu = f['imu'][:]
    end = len(imu)
    while end and np.all(imu[end-1] == 0):
        end -= 1
    if end < 2 or not np.isfinite(imu[:end]).all() or np.any(np.diff(imu[:end, 0]) <= 0):
        raise ValueError('Remaining IMU is invalid; refusing automatic repair')
    original_hash = digest(source)
    output.mkdir(parents=True, exist_ok=False)
    target = output/'inputs.h5'
    shutil.copy2(source, target)
    with h5py.File(target, 'r+') as f:
        f['imu'].resize(end, axis=0)
        f.attrs['validation_copy'] = True
        f.attrs['source_sha256'] = original_hash
        f.attrs['removed_trailing_zero_imu_rows'] = len(imu)-end
    if digest(source) != original_hash:
        raise RuntimeError('Source changed during preparation; do not use this validation copy')
    report = dict(source=str(source.resolve()), source_sha256=original_hash,
                  copy_sha256=digest(target), removed_trailing_zero_imu_rows=len(imu)-end,
                  retained_imu_rows=end, original_capture_complete=False,
                  scope='usable recorded prefix only; no claim of lossless capture')
    (output/'recovery_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
