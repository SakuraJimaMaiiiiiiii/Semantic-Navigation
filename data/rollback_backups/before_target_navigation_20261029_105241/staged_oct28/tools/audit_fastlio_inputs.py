"""Read-only timing/static-start audit for recorded FAST-LIO inputs."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def audit(path):
    source=path/'fastlio/inputs.h5' if path.is_dir() else path
    if not source.is_file():
        raise ValueError(f'Missing raw capture: {source}')
    with h5py.File(source,'r') as f:
        imu=f['imu'][:]
        stamps=np.asarray([f['scans'][k].attrs['source_timestamp'] for k in sorted(f['scans'])])
        reasons=[]
        if len(imu)<2 or len(stamps)<2:
            raise ValueError('At least two scans and two raw IMU samples are required')
        dt=np.diff(imu[:,0]);finite=np.isfinite(imu).all()
        monotonic=bool(finite and np.all(dt>0))
        if not monotonic:reasons.append('invalid_or_nonmonotonic_imu')
        if np.max(dt)>.1:reasons.append('imu_gap_over_100ms')
        if not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):reasons.append('invalid_scan_clock')
        static=imu[imu[:,0]<=imu[0,0]+1.]
        static_ok=bool(len(static)>10 and static[-1,0]-static[0,0]>=.95
                       and np.max(np.linalg.norm(static[:,4:7],axis=1))<=.1
                       and np.max(np.std(static[:,1:4],axis=0))<=.2
                       and abs(np.linalg.norm(static[:,1:4].mean(0))-9.81)<=1.)
        if not static_ok:reasons.append('static_initialization_not_verified')
        coverage=None
        if monotonic:
            index=np.searchsorted(imu[:,0],stamps,side='right')-1
            covered=index>=0
            covered[covered] &= stamps[covered]-imu[index[covered],0]<=.03
            coverage=float(covered.mean())
            if coverage < .95:
                reasons.append('insufficient_imu_scan_time_coverage')
        model=f.attrs.get('scan_model','unverified')
        if model != 'snapshot':reasons.append('lidar_scan_model_unverified')
        report_file=source.parent/'capture_report.json'
        capture=json.loads(report_file.read_text()) if report_file.exists() else {}
        if not capture or capture.get('completed') is False or capture.get('error') or capture.get('overflow'):
            reasons.append('capture_incomplete')
        return dict(source=str(source),imu_samples=len(imu),scans=len(stamps),
                    median_imu_interval_s=float(np.median(dt)),max_imu_gap_s=float(np.max(dt)),
                    p95_imu_interval_s=float(np.percentile(dt,95)),
                    static_start_verified=static_ok,scan_end_imu_coverage_fraction=coverage,
                    scan_model=model,clock_alignment=f.attrs['clock_alignment'],capture=capture,
                    issues=reasons,online_navigation_validated=False,
                    note='Timestamp coverage is not proof of physical synchronization or geometric accuracy.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    try:print(json.dumps(audit(args.run),indent=2,ensure_ascii=False))
    except ValueError as error:parser.error(str(error))


if __name__=='__main__':main()
