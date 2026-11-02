"""Experimental LIO point map; never supplied to the flight planner.

The first tracking pose is aligned once to simultaneous PX4 NED. Subsequent
poses retain LIO drift; repeatedly anchoring to PX4 would hide that drift.
"""
import json
from html import escape
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class FastLioLiveMap:
    def __init__(self, directory, scan_model):
        self.directory = Path(directory)
        self.scan_model = scan_model
        self.alignment = None
        self.cells = set()
        self.path = []
        self.last_publish = 0.
        self.resolution = .15
        self.timestamp = None
        from localization.fastlio_semantics import FastLioSemantics
        self.semantics = FastLioSemantics()

    def update(self, result, px4_pose, shared_timestamp=None):
        if result.get('status') != 'tracking':
            return False
        pose = np.asarray(result['pose'], dtype=float)
        points = np.asarray(result['points_world'], dtype=float)
        px4 = np.asarray(px4_pose, dtype=float)
        if (pose.shape != (4, 4) or points.ndim != 2 or points.shape[1] != 3
                or px4.shape != (7,) or not all(np.isfinite(x).all() for x in (pose, points, px4))):
            raise ValueError('Invalid FAST-LIO map pose/points')
        if self.alignment is None:
            anchor = np.eye(4)
            anchor[:3, :3] = Rotation.from_quat(px4[[4, 5, 6, 3]]).as_matrix()
            anchor[:3, 3] = px4[:3]
            self.alignment = anchor @ np.linalg.inv(pose)
        mapped = points @ self.alignment[:3, :3].T + self.alignment[:3, 3]
        cells = np.unique(np.floor(mapped / self.resolution).astype(np.int64), axis=0)
        self.cells.update(map(tuple, cells.tolist()))
        current = self.alignment @ pose
        if shared_timestamp is not None:
            px4_transform = np.eye(4)
            px4_transform[:3, :3] = Rotation.from_quat(px4[[4, 5, 6, 3]]).as_matrix()
            px4_transform[:3, 3] = px4[:3]
            self.semantics.add_pose(float(shared_timestamp), current @ np.linalg.inv(px4_transform))
        self.path.append(current[:3, 3].tolist())
        self.timestamp = float(result['timestamp'])
        if time.monotonic() - self.last_publish >= 2.:
            self.publish()
        return True

    def publish(self):
        if not self.path:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        cells = np.asarray(sorted(self.cells), dtype=np.int64).reshape(-1, 3)
        path = np.asarray(self.path)
        # This is an XY projection of returns at ALL heights, not a free-space grid.
        xy = np.unique(cells[:, :2], axis=0) * self.resolution
        extent = np.vstack((xy, path[:, :2]))
        low = extent.min(axis=0) - 1
        span = np.maximum(extent.max(axis=0) - low + 1, 2)
        rects = ''.join(f'<rect x="{x:.2f}" y="{y:.2f}" width=".15" height=".15"/>' for x, y in xy)
        line = ' '.join(f'{x:.3f},{y:.3f}' for x, y in path[:, :2])
        labels = ''
        for obj in self.semantics.records:
            center = obj.get('position', obj.get('position_world'))
            if center is None:
                continue
            identity = escape(str(obj['instance_id']))
            labels += (f'<circle cx="{center[0]}" cy="{center[1]}" r=".18" fill="#0077ff"/>'
                       f'<text x="{center[0]}" y="{center[1]}" font-size=".35" fill="#0050bb">{identity}</text>')
        from navigation.scene_memory import write_json
        write_json(self.directory / 'semantic_map.json', dict(
            frame='lio_aligned_to_initial_px4_ned', controls_navigation=False,
            identity_scope='independent_lio_instances', objects=self.semantics.records,
            synchronization=dict(self.semantics.counts)))
        write_json(self.directory / 'scene_graph.json',
                   self.semantics.scene_graph.snapshot(self.semantics.records, self.directory.parent.name))
        html = (f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="3">'
                f'<title>FAST-LIO experimental map</title><p>FAST-LIO 实时点云地图（非导航占据地图）'
                f'；扫描模型：{self.scan_model}；时间：{self.timestamp:.3f}；帧数：{len(path)}。'
                '全部高度的回波投影；灰色不代表当前飞行高度不可通行。蓝色标签为 LIO 独立语义实例（不与旧地图 ID 通用）。PX4 继续控制飞行。<a href="scene_graph.json">物体与观测图 JSON</a></p>'
                f'<svg width="95vw" height="85vh" viewBox="{low[0]} {low[1]} {span[0]} {span[1]}">'
                f'<g fill="#78858b">{rects}</g><polyline points="{line}" fill="none" stroke="#1677ff" '
                f'stroke-width=".07"/><circle cx="{path[-1,0]}" cy="{path[-1,1]}" r=".2" fill="red"/>{labels}</svg>')
        temporary = self.directory / 'map.html.tmp'
        temporary.write_text(html, encoding='utf-8')
        temporary.replace(self.directory / 'map.html')
        self.last_publish = time.monotonic()

    def close(self):
        self.semantics.drain(final=True)
        self.publish()
        if self.alignment is None:
            return
        np.savez_compressed(self.directory / 'live_map.npz',
                            voxel_indices=np.asarray(sorted(self.cells)), resolution=self.resolution,
                            trajectory=np.asarray(self.path), alignment=self.alignment)
        (self.directory / 'live_map.json').write_text(json.dumps(dict(
            frame='lio_aligned_to_initial_px4_ned', scan_model=self.scan_model,
            controls_navigation=False, representation='point_hits_not_free_space',
            tracking_frames=len(self.path), last_timestamp=self.timestamp), indent=2), encoding='utf-8')
