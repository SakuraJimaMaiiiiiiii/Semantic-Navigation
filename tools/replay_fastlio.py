"""Replay recorded LiDAR/raw IMU through native FAST-LIO; never controls a vehicle."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localization.fastlio_backend import FastLioBackend


def replay(source, output, scan_model, *, assume_snapshot=False, max_scans=None):
    from localization import _native_fastlio
    if source.is_dir():
        source=source/'fastlio'/'inputs.h5'
    if not source.is_file():
        raise ValueError(f'Missing {source}; run exploration with --fastlio record first')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    with h5py.File(source,'r') as data:
        model=scan_model or data.attrs.get('scan_model','unverified')
        timing_verified = model == 'snapshot'
        if assume_snapshot:
            model = 'snapshot'
        if model!='snapshot':
            raise ValueError('Scan timing is unverified. Verify Type20 snapshot timing before using --scan-model snapshot')
        imu=data['imu'][:,:7]
        if len(imu)<2 or not np.isfinite(imu).all() or np.any(np.diff(imu[:,0])<=0):
            raise ValueError('IMU source timestamps must be finite and strictly increasing')
        report_path=source.parent/'capture_report.json'
        if report_path.exists():
            capture=json.loads(report_path.read_text())
            if capture.get('completed') is False or capture.get('overflow') or capture.get('error'):
                raise ValueError(f'Incomplete input capture: {capture}')
        output.mkdir(parents=True)
        backend=FastLioBackend(instantaneous=True)
        cursor=0;results=[];clouds=[];px4=[];timings=[];skipped=0
        with (output/'trajectory.jsonl').open('w',encoding='utf-8') as f:
            for scan_index, key in enumerate(sorted(data['scans'])):
                if max_scans is not None and scan_index >= max_scans:
                    break
                group=data['scans'][key];stamp=float(group.attrs['source_timestamp'])
                end=int(np.searchsorted(imu[:,0],stamp,side='right'))
                if end-cursor<2 or stamp-imu[end-1,0]>.03:
                    skipped+=1;continue
                points=group['points_body'][:]
                points=np.column_stack((points,np.zeros(len(points))))
                started=time.perf_counter()
                result=backend.process(stamp,points,imu[cursor:end]);cursor=end
                timings.append(time.perf_counter()-started)
                cloud=result.pop('points_world',None)
                if cloud is not None and result['status'] in {'map_initialized','tracking'}:
                    clouds.append(cloud);px4.append(group['px4_pose'][:])
                result['px4_pose'] = group['px4_pose'][:]
                result['scan_key'] = key
                result={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
                results.append(result);f.write(json.dumps(result,allow_nan=False)+'\n')
                if (scan_index+1) % 100 == 0:
                    f.flush()
                    if clouds:
                        merged = np.vstack(clouds)
                        _, keep = np.unique(np.floor(merged/.1).astype(np.int64), axis=0, return_index=True)
                        clouds = [merged[keep]]
                    print(f'FAST-LIO replay: {scan_index+1} scans, status={result["status"]}', flush=True)
        # Export downsampled map and comparable trajectory, in explicitly separate frames.
        if clouds:
            cloud=np.vstack(clouds);_,keep=np.unique(np.floor(cloud/.1).astype(np.int64),axis=0,return_index=True)
            np.savez_compressed(output/'map.npz',points_lio=cloud[keep],px4_pose=np.asarray(px4))
        counts={s:sum(r['status']==s for r in results) for s in {r['status'] for r in results}}
        report=dict(upstream_commit=_native_fastlio.UPSTREAM_COMMIT,status_counts=counts,
                    skipped_scans=skipped,imu_samples=len(imu),scan_model=model,
                    scan_timing_verified=timing_verified,assumed_snapshot=assume_snapshot,
                    max_scans=max_scans,
                    source_clock_gap_max_s=float(np.max(np.diff(imu[:,0]))),
                    processing_median_ms=float(np.median(timings)*1000) if timings else None,
                    coordinate_frame='LIO initial IMU frame; NOT PX4 NED',
                    synchronization=data.attrs['clock_alignment'],controls_navigation=False,
                    validation='experimental replay; no ground-truth accuracy claim')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scan-model',choices=['snapshot'])
    parser.add_argument('--assume-snapshot', action='store_true', help='Offline hypothesis only; does not verify scan timing')
    parser.add_argument('--max-scans', type=int, help='Limit diagnostic replay to a prefix')
    args=parser.parse_args()
    try:
        replay(args.run,args.output,args.scan_model, assume_snapshot=args.assume_snapshot, max_scans=args.max_scans)
    except (ValueError,ImportError) as error:
        parser.error(str(error))


if __name__=='__main__':
    main()
