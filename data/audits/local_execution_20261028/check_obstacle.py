import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
run=Path('data/runs/run_20261027_191835_850615')
p=run/'exploration/private/diagnostics/exploring_82a36802b4f5/waypoint_W020.npz'
with np.load(p) as a:
    metadata=json.loads(a['metadata'].tobytes())
    points=a['origin']+(np.argwhere(a['states']==1)+.5)*a['resolution']
    nearby=points[(np.linalg.norm(points[:,:2]-[4.79,16.53],axis=1)<.65)&(np.abs(points[:,2]-metadata['altitude'])<.4)]
    stamp=metadata['local_timestamp']
with h5py.File(run/'fastlio/inputs.h5') as f:
    scans=f['scans']; lo,hi=0,len(scans)
    while lo<hi:
        mid=(lo+hi)//2
        if scans[f'{mid:08d}'].attrs['shared_timestamp']<stamp: lo=mid+1
        else: hi=mid
    hits=[]; clouds=[]
    for index in range(max(0,lo-50),min(len(scans),lo+30)):
        group=scans[f'{index:08d}']; t=float(group.attrs['shared_timestamp'])
        if abs(t-stamp)>8: continue
        pose=group['px4_pose'][:]; q=pose[3:]
        world=group['points_body'][:]@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T+pose[:3]
        if len(nearby):
            distances=np.linalg.norm(world[:,None,:]-nearby[None,:,:],axis=2)
            count=int(np.count_nonzero(distances.min(axis=1)<.3))
        else: count=0
        hits.append({'timestamp':t,'nearby_returns':count})
        clouds.append(world[(np.linalg.norm(world[:,:2]-[4.79,16.53],axis=1)<2)&(np.abs(world[:,2]-metadata['altitude'])<.5)])
result={'occupied_centers_ned':nearby.tolist(),'scans_checked':len(hits),'scans_with_returns':sum(h['nearby_returns']>0 for h in hits),'scan_hits':hits,'note':'PX4-frame raw LiDAR evidence; repeated returns support a measured surface, not object identity or absolute calibration.'}
out=Path('data/audits/local_execution_20261028');(out/'obstacle_evidence.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(out/'obstacle_clouds.npz',points=np.concatenate(clouds) if clouds else np.empty((0,3)),occupied=nearby)
print({k:v for k,v in result.items() if k not in {'scan_hits','occupied_centers_ned'}})
print('occupied centers',nearby.tolist())
