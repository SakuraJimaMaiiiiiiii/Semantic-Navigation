"""Synchronized LiDAR fusion into existing local/global occupancy maps."""
from collections import deque
import threading
import time
import numpy as np
from sensors.synchronized_rgbd import _quaternion_to_rotation
from sensors.rfly_lidar import RflyLidarReader
from sensors.lidar_camera_alignment import project_lidar


class LidarMappingWorker:
    def __init__(self, camera, state_stream, local_map, global_map, config, loop_detector=None):
        self.loop_detector = loop_detector
        self.reader = RflyLidarReader(camera, config)
        self.stream, self.local_map, self.global_map, self.config = state_stream, local_map, global_map, config
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._acquisition_thread = None
        self._error = None
        self.last_update = 0.
        self.update_count = 0
        self._cloud_condition = threading.Condition(threading.Lock())
        self._cloud_history = deque(maxlen=30)
        self._pending_cloud = None
        self._acquisition_done = False
        self.alignment_statistics = {
            "matched_frames": 0, "unmatched_frames": 0, "rejected_frames": 0,
            "pose_timeout_frames": 0, "pose_rejected_frames": 0,
        }
        self.alignment_preview_path = None
        self._preview_saved = False
        self._alignment_sample_saved = False

    def start(self, timeout=10.):
        if self.loop_detector is not None:
            self.loop_detector.start()
        self._thread = threading.Thread(target=self._run, name='LidarMapping', daemon=True)
        self._thread.start()
        try:
            if not self._ready.wait(timeout):
                self.raise_if_failed()
                raise TimeoutError('No synchronized LiDAR map frame received before startup timeout')
            self.raise_if_failed()
        except Exception:
            self.close()
            raise

    def close(self):
        self._stop.set()
        with self._cloud_condition:
            self._cloud_condition.notify_all()
        deadline = time.monotonic()+self.config.stale_timeout+2.
        for thread in (self._thread, self._acquisition_thread):
            if thread is None:
                continue
            thread.join(timeout=max(0., deadline-time.monotonic()))
            if thread.is_alive():
                raise RuntimeError('LiDAR worker did not stop')
        if self.loop_detector is not None:
            self.loop_detector.close()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError('LiDAR mapping failed') from self._error
        if self.last_update and time.monotonic()-self.last_update > self.config.stale_timeout:
            raise RuntimeError('LiDAR mapping stale')

    def align_to_camera(self, observation):
        frame = observation.sensor_frame
        with self._cloud_condition:
            # A slightly later LiDAR sample can be the closest one. Bound the
            # wait in the perception thread; occupancy/control never wait here.
            deadline = time.monotonic() + 2 * self.config.camera_max_skew
            while (self._acquisition_thread is not None and self._acquisition_thread.is_alive()
                   and not self._stop.is_set() and not self._acquisition_done
                   and (not self._cloud_history or self._cloud_history[-1][0] < frame.timestamp)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cloud_condition.wait(remaining)
            nearest = min(self._cloud_history, key=lambda item: abs(item[0]-frame.timestamp), default=None)
        if nearest is None or abs(nearest[0]-frame.timestamp)>self.config.camera_max_skew:
            self.alignment_statistics["unmatched_frames"] += 1
            return None
        aligned = project_lidar(nearest[1], observation.camera_pose_world,
                                frame.camera_intrinsics, frame.depth,
                                nearest[0]-frame.timestamp, self.config.camera_max_skew,
                                self.config.semantic_min_points,
                                self.config.alignment_absolute_gate, self.config.alignment_relative_gate)
        if aligned is None:
            return None
        stats = self.alignment_statistics
        stats["matched_frames"] += 1
        stats["rejected_frames"] += int(not aligned.usable)
        valid = np.isfinite(aligned.reference_depth) & (aligned.reference_depth>0)
        stats["last_time_skew_seconds"] = aligned.time_skew
        stats["last_projected_points"] = len(aligned.depth)
        stats["last_consistent_points"] = int(aligned.consistent.sum())
        stats["last_median_absolute_error_m"] = (float(np.median(np.abs(
            aligned.depth[valid]-aligned.reference_depth[valid]))) if valid.any() else None)
        # Retain actual paired geometry, not only a scalar error or overlay.
        # This permits offline verification of axis, scale, pose and depth conventions.
        if (not self._alignment_sample_saved and self.alignment_preview_path is not None
                and len(aligned.depth)):
            np.savez_compressed(
                self.alignment_preview_path.with_suffix(".npz"),
                lidar_points_world=nearest[1],
                camera_pose_world=np.asarray(observation.camera_pose_world),
                camera_intrinsics=frame.camera_intrinsics,
                camera_extrinsics=frame.camera_extrinsics,
                depth=frame.depth,
                rgb_timestamp=np.asarray(frame.rgb_timestamp),
                depth_timestamp=np.asarray(frame.depth_timestamp),
                image_timestamp=np.asarray(frame.timestamp),
                lidar_timestamp=np.asarray(nearest[0]),
                projected_rows=aligned.rows, projected_columns=aligned.columns,
                lidar_optical_depth=aligned.depth, camera_depth=aligned.reference_depth,
                consistent=aligned.consistent,
            )
            self._alignment_sample_saved = True
            if not aligned.usable:
                print("--- 雷达/相机几何不一致：本帧不参与语义融合；已保存 lidar_camera_alignment.npz 用于坐标与深度核对 ----")
        if not self._preview_saved and self.alignment_preview_path is not None and len(aligned.depth):
            import cv2
            preview = frame.rgb.copy()
            for row,col,agree in zip(aligned.rows,aligned.columns,aligned.consistent):
                cv2.circle(preview,(int(col),int(row)),2,(0,220,0) if agree else (0,140,255),-1)
            cv2.putText(preview,"LiDAR projection: green=depth agreement; orange=rejected",
                        (10,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            self._preview_saved = bool(cv2.imwrite(str(self.alignment_preview_path),preview))
        return aligned

    def _fail(self, error):
        with self._cloud_condition:
            if self._error is None:
                self._error = error
            self._stop.set()
            self._ready.set()
            self._cloud_condition.notify_all()

    def _acquire(self):
        """Keep synchronized history at sensor cadence, independent of map cost."""
        try:
            while not self._stop.is_set():
                frame = self.reader.read()
                if frame is None:
                    self._stop.wait(.01)
                    continue
                self.alignment_statistics['last_received_lidar_timestamp'] = float(frame.timestamp)
                try:
                    state, error = self.stream._state_at(frame.timestamp, time.monotonic()+.2)
                except TimeoutError:
                    self.alignment_statistics["pose_timeout_frames"] += 1
                    continue
                self.alignment_statistics["last_pose_sync_error_seconds"] = float(error)
                if error > self.stream.max_sync_error:
                    self.alignment_statistics["pose_rejected_frames"] += 1
                    continue
                origin = np.asarray(state.position_xyz, dtype=float)
                rotation = _quaternion_to_rotation(state.quaternion)
                if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
                    raise ValueError("Non-finite synchronized LiDAR pose")
                points = frame.points_body @ rotation.T + origin
                self.alignment_statistics['last_accepted_lidar_timestamp'] = float(frame.timestamp)
                with self._cloud_condition:
                    self._cloud_history.append((frame.timestamp, points))
                    # Map fusion consumes only the latest complete sample. A
                    # slow integration must neither stall history nor queue work.
                    self._pending_cloud = (origin, points, frame.timestamp)
                    self._cloud_condition.notify_all()
        except Exception as error:
            self._fail(error)
        finally:
            with self._cloud_condition:
                self._acquisition_done = True
                self._cloud_condition.notify_all()

    def _run(self):
        self._acquisition_thread = threading.Thread(
            target=self._acquire, name='LidarAcquisition', daemon=True)
        self._acquisition_thread.start()
        try:
            next_update = 0.
            while True:
                with self._cloud_condition:
                    while self._pending_cloud is None and not self._acquisition_done:
                        self._cloud_condition.wait()
                    if self._error is not None or self._pending_cloud is None:
                        break
                    delay = next_update-time.monotonic()
                    if delay > 0 and not self._stop.is_set():
                        self._cloud_condition.wait(delay)
                        continue
                    origin, points, timestamp = self._pending_cloud
                    self._pending_cloud = None
                for name, target in (('local', self.local_map), ('global', self.global_map)):
                    if target is not None:
                        started = time.monotonic()
                        self.alignment_statistics['integration_stage'] = name
                        target.integrate_lidar_points(origin, points, timestamp, origin)
                        duration = time.monotonic()-started
                        self.alignment_statistics[name+'_integration_last_seconds'] = duration
                        key = name+'_integration_max_seconds'
                        self.alignment_statistics[key] = max(duration, self.alignment_statistics.get(key, 0.))
                self.alignment_statistics['integration_stage'] = 'waiting_for_cloud'
                self.alignment_statistics['last_integrated_lidar_timestamp'] = float(timestamp)
                self.last_update = time.monotonic()
                if self.loop_detector is not None:
                    self.loop_detector.submit(origin, points, timestamp)
                self.update_count += 1
                self._ready.set()
                if self._stop.is_set():
                    break
                next_update = self.last_update+1./self.config.update_rate
        except Exception as error:
            self._fail(error)
        finally:
            self._stop.set()
            self._acquisition_thread.join(timeout=self.config.stale_timeout+2.)
            if self._acquisition_thread.is_alive():
                self._fail(RuntimeError('LiDAR acquisition worker did not stop'))
            with self._cloud_condition:
                self._pending_cloud = None
