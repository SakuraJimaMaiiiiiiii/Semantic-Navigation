from pathlib import Path
import ast,json,subprocess
backup=Path(__file__).resolve().parent
stage=Path(json.loads((backup/'rollback_stage.json').read_text())['stage'])
changes=[]
def edit(name, old, new):
 p=stage/name;s=p.read_text()
 if s.count(old)!=1:raise RuntimeError(f'{name}: expected one match, found {s.count(old)}: {old[:90]}')
 p.write_text(s.replace(old,new));changes.append(name)
def remove_method(name, method):
 p=stage/name;s=p.read_text();tree=ast.parse(s);nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)and n.name==method]
 if len(nodes)!=1:raise RuntimeError(f'{name}: method {method} missing')
 n=nodes[0];lines=s.splitlines(keepends=True);p.write_text(''.join(lines[:n.lineno-1]+lines[n.end_lineno:]));changes.append(name)
edit('planner/waypoint_planning_worker.py', '''                    previous = self._latest
                    # A failed solve does not invalidate an already published
                    # route. Keep it for THIS target only; the flight thread
                    # still revalidates its chord on the newest geometry before
                    # every command. Hard stop statuses must replace it.
                    reuse = (error is None and previous is not None
                             and result.plan.status in {'transient_no_path', 'no_path'}
                             and previous.plan.can_move
                             and np.array_equal(previous.target, target)
                             and callable(getattr(self.planner, 'revalidate_plan', None)))
                    self._latest, self._error = (previous if reuse else result), error''', '                    self._latest, self._error = result, error')
remove_method('tests/test_async_waypoint_reference.py','test_failed_resolve_keeps_same_target_route_but_hard_stop_replaces_it')
edit('navigation/mission.py','    def _frame(self, *, max_age=None):','    def _frame(self):')
edit('navigation/mission.py','or self.clock() - frame.completed_at > (self.config.perception_timeout if max_age is None else max_age)','or self.clock() - frame.completed_at > self.config.perception_timeout')
edit('navigation/frontier_explorer.py', '''            # Semantic inference does not certify collision clearance. During
            # transit, tolerate its existing bounded recovery interval while
            # LiDAR map freshness and body clearance remain checked normally.
            max_age = (max(self.config.perception_timeout, self.exploration_config.perception_recovery_timeout)
                       if self._motion_progress is not None else self.config.perception_timeout)
            frame = super()._frame(max_age=max_age)''','            frame = super()._frame()')
edit('navigation/frontier_explorer.py', '''            # At a coarse-grid boundary the nearest certified center can be
            # only 20cm away. Requiring an 80cm move skips that valid escape
            # and sends every new frontier through the same invalid start.
            if (.5*global_grid.resolution <= distance <= 2.4''','            if (max(.8, 2*self.config.arrival_tolerance) <= distance <= 2.4')
edit('navigation/frontier_explorer.py', '''                position = self.vehicle.get_position()
                if (grid.cell(position) in grid.free
                        and (recovery_goal is None or np.linalg.norm(position[:2]-recovery_goal[:2])
                             <= min(.05, self.config.arrival_tolerance))):''','                if grid.cell(self.vehicle.get_position()) in grid.free:')
for old in [ '        column_min_horizontal_extent=0.25,\n','        self.column_min_horizontal_extent = float(column_min_horizontal_extent)\n', '''        if canonical == 'column':
            required = max(required, self.safety_landmark_min_observations)
            if landmark.confidence < self.safety_landmark_min_confidence:
                return None
''', '''            if memory.class_name == 'column' and (
                    memory.confidence < self.safety_landmark_min_confidence
                    or memory.observation_count < self.safety_landmark_min_observations):
                continue
''', '            and min(extents[:2]) >= self.column_min_horizontal_extent\n', '''        if self.column_min_horizontal_extent <= 0:
            raise ValueError("column_min_horizontal_extent must be positive")
''']:
 edit('mapping/semantic_instance_mapper.py',old,'')
for old in ['    column_min_horizontal_extent: float = 0.25  # 多视角横向厚度下限；不把细长墙边或管线发布为柱子\n', '''        if self.column_min_horizontal_extent <= 0:
            raise ValueError("column_min_horizontal_extent 必须大于 0")
''','            column_min_horizontal_extent=self.config.column_min_horizontal_extent,\n', '''            if name == 'column' and item[1].confidence < self.config.safety_landmark_min_confidence:
                continue
''']:
 edit('detections/object_detect.py',old,'')
edit('detections/object_detect.py',"            if (canonical_class_name(detection.class_name) in {'column', 'exit door', 'fire cabinet', 'fire extinguisher'}", "            if (canonical_class_name(detection.class_name) in {'exit door', 'fire cabinet', 'fire extinguisher'}")
for method in ['test_slow_semantics_do_not_stop_lidar_guarded_transit','test_grid_boundary_recovery_can_use_nearby_certified_center']:
 remove_method('tests/test_frontier_observation_watchdog.py',method)
edit('tests/test_explore_startup_config.py',"names = ['parking garage structural column'] * 4", "names = ['parking garage structural column'] * 2")
edit('tests/test_explore_startup_config.py','for x, height in [(0., 2.6), (5., .8), (10., 2.6), (15., 2.6)]:','for x, height in [(0., 2.6), (5., .8)]:')
edit('tests/test_explore_startup_config.py', '''            if x == 10.:
                high[0] = x + .03  # Tall wall/pipe fragment, not a supported column.
''','')
edit('tests/test_explore_startup_config.py','for stamp in range(1, 9):','for stamp in (1., 2., 3.):')
edit('tests/test_explore_startup_config.py','mapper.update(names, [.8, .8, .8, .15], observations, stamp)','mapper.update(names, [.8, .8], observations, stamp)')
edit('config/explore.json', '启用车辆、消防/安全出口门及车库承重柱。parking garage structural column 统一记为 column；柱子要求可观测高度至少2.2m、竖直长宽比至少1.2、横向两个方向至少0.25m，并复用安全地标的0.30置信度与8帧确认门槛，排除矮柱和细长片段。SAM3和YOLO-World+SAM2共用此列表。', '启用车辆、消防/安全出口门及车库承重柱。parking garage structural column 统一记为 column；柱子保留多帧高度至少2.2m、竖直长宽比至少1.2的过滤，排除矮柱。SAM3和YOLO-World+SAM2共用此列表。')
# Revert the other local session's successful patches in reverse chronological order.
log=next(Path('/home/qyd/.codex/sessions/2026/10/29').glob('*.jsonl'))
events=[]
for line in log.open():
 d=json.loads(line);v=d.get('payload',{})
 if v.get('type')=='patch_apply_end' and v.get('success'):events.append(v)
for event in reversed(events):
 for absolute,ch in reversed(list(event.get('changes',{}).items())):
  prefix='/mnt/e/files/Semantic Navigation/'
  if not absolute.startswith(prefix):raise RuntimeError('Unexpected patch path '+absolute)
  name=absolute[len(prefix):];p=stage/name
  if ch['type']=='update':
   if ch.get('move_path'):raise RuntimeError('Unexpected move')
   patch=f'diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n'+ch['unified_diff']
   result=subprocess.run(['git','apply','--reverse','--unidiff-zero','--whitespace=nowarn','-'],input=patch.encode(),cwd=stage,capture_output=True)
   if result.returncode:raise RuntimeError(f'Reverse failed {name}: '+result.stderr.decode())
  elif ch['type']=='add':
   if p.read_text().rstrip('\n')!=ch['content'].rstrip('\n'):raise RuntimeError('Added file differs '+name)
   p.unlink()
  else:raise RuntimeError('Unhandled patch type '+ch['type'])
  changes.append(name)
(backup/'rollback_files.json').write_text(json.dumps(sorted(set(changes)),ensure_ascii=False,indent=2))
print(json.dumps(dict(stage=str(stage),files=sorted(set(changes))),ensure_ascii=False,indent=2))
