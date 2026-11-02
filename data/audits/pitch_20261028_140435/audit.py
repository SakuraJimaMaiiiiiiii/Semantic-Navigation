from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
run=Path('data/runs/run_20261028_140435_896686')
rows=[json.loads(line) for line in (run/'fastlio/shadow_poses.jsonl').read_text().splitlines() if line.endswith('}') and 'px4_pose' in line]
t=np.array([r['shared_timestamp'] for r in rows]);p=np.array([r['px4_pose'] for r in rows]);dt=np.diff(t)
keep=np.r_[True,dt>0];t,p=t[keep],p[keep];t=t-t[0]
rpy=Rotation.from_quat(p[:,[4,5,6,3]]).as_euler('xyz',degrees=True)
# Differentiate only at uniformly interpolated times; this is a PX4 position
# based speed estimate, not the separately recorded FAST-LIO velocity.
grid=np.arange(t[0],t[-1],.2);position=np.column_stack([np.interp(grid,t,p[:,k]) for k in range(3)])
euler=np.column_stack([np.interp(grid,t,rpy[:,k]) for k in range(3)])
velocity=np.gradient(position,grid,axis=0);speed=np.linalg.norm(velocity[:,:2],axis=1)
height=p[0,2]-position[:,2];air=height>.7;moving=air&(speed>.08)
pitch=euler[:,1];peaks,_=find_peaks(np.abs(pitch),height=1.,distance=10)
peaks=[i for i in peaks if air[i]]
raw_air = p[0,2]-p[:,2] > .7
straight = (grid>=100)&(grid<=140)
yaw = np.rad2deg(np.unwrap(np.deg2rad(euler[:,2])))
summary=dict(run=run.name,source='px4_pose in FAST-LIO shadow JSONL; NOT FAST-LIO estimated attitude',
    samples=len(t),duration_s=float(t[-1]),interval_median_s=float(np.median(np.diff(t))),interval_p95_s=float(np.percentile(np.diff(t),95)),
    pitch_air_abs_deg=dict(median=float(np.median(np.abs(rpy[raw_air,1]))),p95=float(np.percentile(np.abs(rpy[raw_air,1]),95)),max=float(np.max(np.abs(rpy[raw_air,1])))),
    pitch_moving_abs_deg=dict(median=float(np.median(np.abs(pitch[moving]))),p95=float(np.percentile(np.abs(pitch[moving]),95)),max=float(np.max(np.abs(pitch[moving])))),
    straight_100_140s=dict(yaw_range_deg=float(np.ptp(yaw[straight])),pitch_peak_to_peak_deg=float(np.ptp(pitch[straight])),speed_p10_mps=float(np.percentile(speed[straight],10)),speed_p90_mps=float(np.percentile(speed[straight],90))),
    peaks=[dict(elapsed_s=float(grid[i]),pitch_deg=float(pitch[i]),speed_mps=float(speed[i]),position_ned=position[i].tolist()) for i in peaks],
    note='Partial 253-second prefix; interpolation cannot recover motion above the source sampling frequency; no per-command stream recorded.')
out=Path('data/audits/pitch_20261028_140435');(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
np.savez(out/'series.npz',time=grid,position=position,rpy=euler,speed=speed,air=air)
print(json.dumps({k:v for k,v in summary.items() if k != 'peaks'},ensure_ascii=False,indent=2))
