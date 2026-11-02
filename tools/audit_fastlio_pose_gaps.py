"""Leave-one-out check of interpolation across longer recorded LIO correction gaps."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def audit(path):
    rows = [json.loads(line) for line in Path(path).open(encoding='utf-8')]
    rows = [r for r in rows if r.get('status') == 'tracking' and 'px4_to_lio_map' in r]
    t = np.array([r['shared_timestamp'] for r in rows])
    matrices = np.array([r['px4_to_lio_map'] for r in rows])
    rotations = Rotation.from_matrix(matrices[:, :3, :3])
    errors, gaps = [], []
    for i in range(0, len(t)-2, 2):
        j = int(np.searchsorted(t, t[i]+.9))
        if j >= len(t) or j <= i+1 or t[j]-t[i] > 1.5:
            continue
        translation_change = np.linalg.norm(matrices[j, :3, 3]-matrices[i, :3, 3])
        rotation_change = (rotations[i].inv()*rotations[j]).magnitude()
        if translation_change > .1 or rotation_change > np.deg2rad(1):
            continue
        middle = (i+j)//2
        fraction = (t[middle]-t[i])/(t[j]-t[i])
        translation = (1-fraction)*matrices[i,:3,3]+fraction*matrices[j,:3,3]
        rotation = Slerp([0.,1.], rotations[[i,j]])([fraction])[0]
        # Upper bound on the displacement of a point up to 10 m away.
        error = np.linalg.norm(translation-matrices[middle,:3,3]) + 10*(rotation.inv()*rotations[middle]).magnitude()
        errors.append(float(error))
        gaps.append(float(t[j]-t[i]))
    return dict(tracking_poses=len(rows), tested_gaps=len(errors),
                max_gap_s=max(gaps) if gaps else None,
                interpolation_displacement_bound_at_10m_m=(
                    dict(zip(('median','p95','p99','max'), np.percentile(errors,[50,95,99,100]).tolist()))
                    if errors else None),
                limitation='Holdout checks agreement with recorded FAST-LIO/PX4 correction, not ground truth; '
                           'only spans 0.9–1.5 s with endpoint change <= 0.1 m and 1 degree are tested.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run/'fastlio/shadow_poses.jsonl'), indent=2, ensure_ascii=False, allow_nan=False))
