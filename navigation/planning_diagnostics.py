"""Bounded, asynchronous failure evidence kept outside the public semantic graph."""

import json
import math
from pathlib import Path
import threading

import numpy as np


def footprint_summary(snapshot, position, altitude, config):
    """Count raw evidence inside exactly the routing clearance footprint."""
    resolution = float(snapshot.resolution)
    origin = np.asarray(getattr(snapshot, 'origin_ned', np.zeros(3)))
    center = np.floor((np.asarray(position)-origin)/resolution).astype(int)
    radius = math.ceil(config.clearance_radius/resolution)
    low = math.floor((altitude-config.vertical_half_extent-origin[2])/resolution)
    high = math.floor((altitude+config.vertical_half_extent-origin[2])/resolution)
    counts = dict(free=0, occupied=0, unknown=0, outside=0)
    dense = hasattr(snapshot, 'states')
    lookup = None
    if not dense:
        lower = np.array([center[0]-radius, center[1]-radius, low])
        upper = np.array([center[0]+radius, center[1]+radius, high])
        inside = np.all((snapshot.indices>=lower)&(snapshot.indices<=upper),axis=1)
        lookup = {tuple(i): float(v) for i,v in zip(snapshot.indices[inside],snapshot.log_odds[inside])}
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            if math.hypot(dx,dy)*resolution > config.clearance_radius+resolution*.5:
                continue
            for z in range(low, high+1):
                cell = (int(center[0])+dx,int(center[1])+dy,z)
                if dense:
                    if any(i<0 or i>=n for i,n in zip(cell,snapshot.states.shape)):
                        counts['outside'] += 1
                        continue
                    state = snapshot.states[cell]
                    key = 'free' if state==0 else ('occupied' if state==1 else 'unknown')
                else:
                    odds = lookup.get(cell)
                    key = ('unknown' if odds is None else 'occupied' if odds>=snapshot.occupied_threshold
                           else 'free' if odds<=snapshot.free_threshold else 'unknown')
                counts[key] += 1
    return counts


class FailureRecorder:
    """At most 16 snapshots per mission, one pending write; never block flight IO."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._thread = None
        self._count = 0
        self._terminal_thread = None
        self._waypoint_threads = {}

    def capture(self, status, local, global_map, current, target, altitude, config, detail=None, terminal=False, waypoint_number=None):
        if waypoint_number is not None:
            number = int(waypoint_number)
            if number < 1 or number in self._waypoint_threads:
                return None
            path = self.directory / f'waypoint_W{number:03d}.npz'
            thread = threading.Thread(target=self._write, args=(
                path, status, local, global_map, np.array(current, copy=True),
                np.array(target, copy=True), altitude, config, dict(detail or {})), daemon=True)
            self._waypoint_threads[number] = thread
            thread.start()
            return str(path)
        if terminal:
            if self._terminal_thread is not None:
                return None
            path = self.directory / "termination.npz"
            self._terminal_thread = threading.Thread(target=self._write, args=(
                path, status, local, global_map, np.array(current, copy=True),
                np.array(target, copy=True), altitude, config, dict(detail or {})), daemon=True)
            self._terminal_thread.start()
            return str(path)
        if self._count >= 16 or (self._thread is not None and self._thread.is_alive()):
            return None
        self._count += 1
        path = self.directory / f'failure_{self._count:03d}.npz'
        # Snapshots are detached copies (or immutable global snapshots); keep
        # their references while the background worker serializes the evidence.
        self._thread = threading.Thread(target=self._write, args=(
            path, status, local, global_map, np.array(current,copy=True),
            np.array(target,copy=True), altitude, config, dict(detail or {})), daemon=True)
        self._thread.start()
        return str(path)

    @staticmethod
    def _write(path, status, local, global_map, current, target, altitude, config, detail):
        temp = path.with_suffix('.tmp.npz')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(status=status, detail=detail, altitude=float(altitude),
                            clearance_radius=config.clearance_radius,
                            vertical_half_extent=config.vertical_half_extent,
                            local_timestamp=float(local.timestamp),
                            local_footprint=footprint_summary(local,current,altitude,config))
            arrays = dict(current=current, target=target, origin=local.origin_ned,
                          resolution=local.resolution, states=local.states)
            if hasattr(local, 'observed'):
                arrays['observed'] = local.observed
            if hasattr(local, 'inflated_occupied'):
                arrays['inflated_occupied'] = local.inflated_occupied
            if global_map is not None:
                metadata['global_timestamp'] = float(global_map.timestamp)
                metadata['global_footprint'] = footprint_summary(global_map,current,altitude,config)
                # Store only the same local window, not the entire garage.
                points = (global_map.indices+.5)*global_map.resolution
                high = local.origin_ned + np.asarray(local.states.shape)*local.resolution
                target_margin = config.clearance_radius + global_map.resolution
                target_low = np.asarray(target) - target_margin
                target_high = np.asarray(target) + target_margin
                inside = (np.all((points>=local.origin_ned)&(points<high),axis=1)
                          | np.all((points>=target_low)&(points<=target_high),axis=1))
                arrays.update(global_indices=global_map.indices[inside],
                              global_log_odds=global_map.log_odds[inside],
                              global_resolution=global_map.resolution,
                              occupied_threshold=global_map.occupied_threshold,
                              free_threshold=global_map.free_threshold)
            observation = getattr(local,'observation',None)
            if observation is not None:
                frame = observation.sensor_frame
                arrays.update(depth=frame.depth, rgb=frame.rgb, intrinsics=frame.camera_intrinsics,
                              camera_pose=observation.camera_pose_world,
                              mapped_position=observation.drone_state.position_xyz)
                metadata.update(depth_timestamp=float(frame.depth_timestamp),
                                rgb_timestamp=float(frame.rgb_timestamp),
                                state_sync_error=float(observation.state_sync_error))
            arrays['metadata'] = np.frombuffer(json.dumps(metadata,allow_nan=False).encode(),dtype=np.uint8)
            np.savez_compressed(temp, **arrays)
            temp.replace(path)
            summary = path.with_suffix('.json')
            summary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            print(f'--- 规划失败证据已保存：{path} ----')
        except Exception as error:
            temp.unlink(missing_ok=True)
            print(f'--- 规划诊断保存失败：{error} ----')

    def close(self):
        for thread in (self._thread, self._terminal_thread, *self._waypoint_threads.values()):
            if thread is not None:
                thread.join(timeout=5.)
