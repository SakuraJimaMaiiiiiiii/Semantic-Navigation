import sys, json, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from mapping.local_occupancy_grid import LocalOccupancyGrid
run=Path('data/runs/run_20261027_191835_850615')
with h5py.File(run/'fastlio/inputs.h5') as f:
    groups=f['scans'];lo,hi=0,len(groups)
    stamp=1793101532.8487744
    while lo<hi:
        mid=(lo+hi)//2
        if groups[f'{mid:08d}'].attrs['shared_timestamp']<stamp: lo=mid+1
        else: hi=mid
    g=groups[f'{lo:08d}'];pose=g['px4_pose'][:];q=pose[3:]
    points=g['points_body'][:]@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T+pose[:3]
grid=LocalOccupancyGrid();start=time.perf_counter()
grid.integrate_lidar_points(pose[:3],points,stamp,pose[:3]);integration=time.perf_counter()-start
start=time.perf_counter();one=grid.snapshot();first=time.perf_counter()-start
start=time.perf_counter()
for _ in range(20): two=grid.snapshot()
repeat=(time.perf_counter()-start)/20
assert one.states is two.states
report=dict(lidar_points=len(points),integration_seconds=integration,first_snapshot_seconds=first,cached_snapshot_seconds=repeat,scope='Single recorded scan, no UE or competing inference/loop reconstruction; not a live latency guarantee.')
Path('data/audits/local_execution_20261028/local_benchmark.json').write_text(json.dumps(report,indent=2))
print(report)
