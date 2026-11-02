import sys,json,time,cProfile,pstats
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from config.global_map_config import GlobalSparseMapConfig
from mapping.live_loop_mapping import LiveLoopMapping
from mapping.explore_v3.adapters import SemanticInstanceMapper,rebuild
run=Path('data/runs/run_20261102_102916_137270')
d=np.load(run/'maps/lidar_loop_keyframes.npz'); offsets=d['offsets']
frames=[(p,d['points_world'][offsets[i]:offsets[i+1]],float(t)) for i,(p,t) in enumerate(zip(d['origins'],d['timestamps']))]
detector=SimpleNamespace(mapping_snapshot=(frames,json.loads(str(d['constraints_json']))),config=GlobalSparseMapConfig(**json.loads(str(d['config_json']))))
source=json.loads((run/'exploration/semantic_map.json').read_text(encoding='utf-8'))
mapper=LiveLoopMapping(detector,run,source['home_ned'],source['initial_yaw_rad'],mapper_type=SemanticInstanceMapper,rebuild_map=rebuild)
# Preserve the previous display for comparison. Never overwrite source archives.
backup=Path('reports/loop_map_20261102/map_before.html')
if not backup.exists(): backup.write_bytes((run/'exploration/map.html').read_bytes())
mapper.update_display(state='离线完成最终回环保存；本次落地未确认',
    waypoints=json.loads((run/'exploration/private/exploration_waypoints.json').read_text(encoding='utf-8')))
pr=cProfile.Profile();pr.enable();mapper.finalize();pr.disable()
pr.dump_stats('reports/loop_map_20261102/finalize.prof')
with open('reports/loop_map_20261102/profile.txt','w',encoding='utf-8') as stream:pstats.Stats(pr,stream=stream).sort_stats('cumtime').print_stats(30)
print((run/'maps/live_loop_corrected/status.json').read_text(encoding='utf-8'),flush=True)
