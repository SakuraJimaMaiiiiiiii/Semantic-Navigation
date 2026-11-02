import cv2,h5py,json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from sensors.semantic_support import reliable_surface_mask
from types import SimpleNamespace
from navigation.standalone_navigation import read_archive
run=Path('data/runs/run_20261026_201454_804958');out=Path('data/audits/vehicle03_arrival')
f=h5py.File(next((run/'recordings').glob('*.h5')),'r');t=f['timestamps/timestamp'][:];pos=f['state/position_xyz'][:];goal=np.array([5.18766,1.14566,-1.49614]);first=np.flatnonzero(np.linalg.norm(pos-goal,axis=1)<.4)[0];i=int(np.argmin(abs(t-(t[first]+5))))
rgb=f['sensors/rgb'][i];depth=f['sensors/depth'][i];K=f['camera/intrinsics'][:];ext=f['camera/extrinsics'][:];q=f['state/quaternion_wxyz'][i];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();pose[:3,3]=pos[i];pose=pose@ext
_,a=read_archive('data/runs/run_20261026_123724_906919');saved=next(x for x in a['records'] if x['runtime_instance_id']=='vehicle_03');cam=(np.array(saved['position_world'])-pose[:3,3])@pose[:3,:3];uv=K@cam;uv=uv[:2]/uv[2];print('h5 frame',i,'time',t[i], 'camera target',cam,'pixel',uv,'depthshape',depth.shape)
cv2.circle(rgb,tuple(uv.astype(int)),12,(0,0,255),3);cv2.putText(rgb,'OLD vehicle_03 anchor projection',(20,40),0,.8,(0,0,255),2);cv2.imwrite(str(out/'arrival_old_target_projection.jpg'),rgb)
cap=cv2.VideoCapture(str(run/'detections/sam3_segmentation.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,833);ok,seg=cap.read();cap.release();seg=seg.astype(float);mask=(seg[:,:,0]>seg[:,:,1]+20.)&(seg[:,:,1]>seg[:,:,2]+5.)&(seg[:,:,0]>100);mask[:150]=False;mask[:,:100]=False
mask=cv2.resize(mask.astype('uint8'),(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)>0
cfg=SimpleNamespace(instance_depth_min=.3,instance_depth_max=15.,instance_max_incidence_degrees=75.,instance_depth_min_points=12);mask=cv2.erode(mask.astype('uint8'),np.array([[0,1,0],[1,1,1],[0,1,0]],dtype='uint8')).astype(bool);mask &= np.isfinite(depth)&(depth>=cfg.instance_depth_min)&(depth<=cfg.instance_depth_max)
print('approx overlay mask valid pixels',mask.sum());mask=reliable_surface_mask(depth,mask,K,cfg.instance_max_incidence_degrees);values=np.sort(depth[mask]);print('incidence supported',mask.sum(),'minpoints',cfg.instance_depth_min_points)
gaps=np.flatnonzero(np.diff(values)>.5+.02*values[:-1])+1;clusters=np.split(values,gaps);print('clusters',[(len(x),float(x[0]),float(x[-1])) for x in clusters if len(x)>=cfg.instance_depth_min_points])
rows=[json.loads(x) for x in (run/'maps/semantic_observations.jsonl').read_text().splitlines()];window=[x for x in rows if t[first]<=x['timestamp']<=t[first]+15];print('arrival15s packets',len(window),'measurements',sum(len(x['samples']) for x in window));print('all',len(rows),'nonempty',sum(bool(x['samples']) for x in rows))
