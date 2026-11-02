import cv2,numpy as np,h5py,json
from pathlib import Path
run=Path('data/runs/run_20261026_201454_804958');out=Path('data/audits/vehicle03_arrival')
for name in ['sam3_detection','sam3_segmentation']:
 cap=cv2.VideoCapture(str(run/'detections'/f'{name}.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);print(name,count,fps)
 tiles=[]
 for i in np.linspace(0,count-1,24).astype(int):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,img=cap.read()
  if not ok:continue
  tile=cv2.resize(img,(400,225));cv2.putText(tile,f'frame={i} video={i/fps:.1f}s',(5,219),0,.5,(0,0,255),1);tiles.append(tile)
 sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
 cv2.imwrite(str(out/f'{name}_sheet.jpg'),sheet);cap.release()
f=h5py.File(next((run/'recordings').glob('*.h5')),'r');p=f['state/position_xyz'][:];e=f['state/euler_rpy'][:];t=f['timestamps/timestamp'][:];goal=np.array([5.18766,1.14566,-1.49614]);mask=np.linalg.norm(p-goal,axis=1)<.4;inds=np.where(mask)[0];print('h5 arrival',len(inds),'firstlast',t[inds[[0,-1]]].tolist(),'offsets',(t[inds[[0,-1]]]-t[0]).tolist());print('arrival yaw range',np.rad2deg(e[inds,2]).min(),np.rad2deg(e[inds,2]).max());print('datasets',list(f['camera']),list(f['sensors']));print('time',t[0],t[-1])
