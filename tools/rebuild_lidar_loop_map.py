"""Rebuild a separate loop-corrected map after exploration has stopped."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def resolve_run(run=None, runs_root=None):
    root = Path(runs_root) if runs_root is not None else Path(__file__).resolve().parents[1]/'data/runs'
    if run is not None:
        run = Path(run)
        if any(word in str(run) for word in ('实际运行目录', '本次运行目录')):
            raise ValueError('路径中包含示例占位文字。请直接不带路径运行，或填写真实的 run_时间 目录。')
        candidates = [run]
    else:
        candidates = sorted(root.glob('run_*'), reverse=True)
    for candidate in candidates:
        archive = candidate/'maps/lidar_loop_keyframes.npz'
        history = candidate/'maps/semantic_observations.jsonl'
        if archive.is_file() and (not (candidate/'exploration/semantic_map.json').exists()
                                  or history.is_file() and history.stat().st_size > 0):
            return candidate
    if run is not None:
        raise ValueError(f'{run} 缺少雷达关键帧或逐帧语义记录。请使用新版代码采集并正常关闭探索终端后再重建。')
    raise ValueError(f'{root} 中没有数据齐全的实验。请先运行 python run_explore.py，完成探索后正常关闭该终端（Ctrl+C）以保存关键帧，再运行本脚本。仅有回环约束 JSON 不足以重建。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, nargs='?', help='实际实验目录；省略则查找最新数据齐全的实验')
    args = parser.parse_args()
    try:
        args.run = resolve_run(args.run)
    except ValueError as error:
        parser.error(str(error))
    print(f'重建输入目录：{args.run.resolve()}')
    archive = args.run / 'maps/lidar_loop_keyframes.npz'
    if not archive.exists():
        parser.error(f'Missing {archive}. Run exploration with keyframe recording enabled, then close it normally.')
    if (args.run/'exploration/semantic_map.json').exists() and not (args.run/'maps/semantic_observations.jsonl').exists():
        parser.error('Missing per-observation semantic history; start a new experiment with the updated recorder.')
    from config.global_map_config import GlobalSparseMapConfig
    from mapping.lidar_loop_map import optimize_frames, rebuild
    with np.load(archive, allow_pickle=False) as data:
        offsets = data['offsets']
        frames = [(p, data['points_world'][offsets[i]:offsets[i+1]], float(t))
                  for i,(p,t) in enumerate(zip(data['origins'], data['timestamps']))]
        constraints = json.loads(str(data['constraints_json']))
        config = GlobalSparseMapConfig(**json.loads(str(data['config_json'])))
    if not constraints:
        parser.error('No verified loop constraints; no corrected map generated.')
    optimized, corrections = optimize_frames(frames, constraints, config)
    out = args.run / 'maps/loop_corrected'
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, corrected in [('raw_keyframes', False), ('corrected', True)]:
        snapshot = rebuild(optimized, config, corrected)
        np.savez_compressed(out / f'{name}.npz', indices=snapshot.indices,
            log_odds=snapshot.log_odds, resolution=snapshot.resolution,
            occupied_threshold=snapshot.occupied_threshold, free_threshold=snapshot.free_threshold,
            trajectory=snapshot.trajectory_ned)
        counts[name] = snapshot.voxel_count
        print(f'{name}: {snapshot.voxel_count} voxels')
    semantic_bundle = None
    if (args.run/'exploration/semantic_map.json').exists():
        from mapping.loop_semantic_replay import export_bundle
        semantic_bundle, observation_count = export_bundle(args.run, out, snapshot,
            [f.timestamp for f in optimized], corrections)
        print(f'Synchronized semantic map: {semantic_bundle} ({observation_count} measurements replayed)')
    np.savez_compressed(out/'keyframe_corrections.npz',
        timestamps=[f.timestamp for f in optimized], raw_to_corrected=corrections)
    report = dict(coordinate_frame='loop_corrected_ned_first_keyframe_fixed',
        live_navigation_map_modified=False, semantic_objects_corrected=semantic_bundle is not None,
        semantic_bundle=None if semantic_bundle is None else semantic_bundle.name,
        keyframes=len(frames), loop_constraints=len(constraints), voxel_counts=counts,
        max_keyframe_displacement_m=max(float(np.linalg.norm(f.corrected_drone_position-f.raw_drone_position)) for f in optimized),
        limitations=['4-DoF optimization; roll/pitch not optimized',
                    'Rebuild uses sampled keyframe clouds, not every original LiDAR scan',
                    'Geometric consistency does not establish absolute accuracy',
                    'Do not use this map with uncorrected semantic labels or PX4 waypoints'])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved: {out}; live PX4 navigation unchanged')


if __name__ == '__main__':
    main()
