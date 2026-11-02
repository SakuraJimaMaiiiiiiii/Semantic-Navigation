import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from config.global_map_config import GlobalSparseMapConfig
from mapping.lidar_loop_map import optimize_frames
from mapping.loop_semantic_replay import replay
from mapping.explore_v3.adapters import SemanticInstanceMapper
from mapping.vehicle_instance_tracker import VehicleInstanceTracker
p=Path('data/runs/run_20261102_153130_966238');d=np.load(p/'maps/lidar_loop_keyframes.npz');o=d['offsets'];frames=[(v,d['points_world'][o[i]:o[i+1]],float(t)) for i,(v,t) in enumerate(zip(d['origins'],d['timestamps']))]
revision=p/'maps/live_loop_corrected/revision_11a982559ae3'
cutoff=float(np.load(revision/'private/navigation_cache.npz')['timestamp'])
frames=[f for f in frames if f[2]<=cutoff]
measurement_limit=json.loads((revision/'semantic_map.json').read_text(encoding='utf-8'))['observation_replay_count']
lines=[];count=0
for line in (p/'maps/semantic_observations.jsonl').read_text(encoding='utf-8').splitlines():
 packet=json.loads(line);lines.append(line);count+=len(packet['samples'])
 if count>=measurement_limit:break
history=Path('reports/vehicle_replay_20261102/prefix.jsonl');history.write_text('\n'.join(lines)+'\n',encoding='utf-8')
c=GlobalSparseMapConfig(**json.loads(str(d['config_json'])));_,cor=optimize_frames(frames,[e for e in json.loads(str(d['constraints_json'])) if e['source']<len(frames)],c)
original=VehicleInstanceTracker._merge_duplicates
results={}
for mode in ['after']:
 def disabled(self,*args,**kwargs):
  self.replay_source_votes={}
  return original(self,*args,**kwargs)
 VehicleInstanceTracker._merge_duplicates=disabled if mode=='before' else original
 holder=[]
 def factory(**kwargs):
  instance=SemanticInstanceMapper(**kwargs);holder.append(instance);return instance
 records,_,count=replay(history,[f[2] for f in frames],cor,.3,500000,mapper_type=factory)
 results[mode]=[dict(id=r['runtime_instance_id'],sources=r['source_identity_observations'],position=r['position_world']) for r in records]
 print(mode,results[mode],flush=True)
 print('DIAGNOSTICS',holder[0]._vehicles.merge_diagnostics,flush=True)
 print('VOTES',holder[0]._vehicles.replay_source_votes,flush=True)
Path('reports/vehicle_replay_20261102/comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
