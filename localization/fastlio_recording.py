"""Loss-audited raw LIO inputs, captured on the existing MAVLink receiver.

No second UDP owner; the receive hook only queues copies. Flight/navigation
poses are not changed by this recorder or the optional shadow estimator.
"""
import json
import bisect
from pathlib import Path
import queue
import socket
import threading
import time

import numpy as np


class FastLioRecorder:
    def __init__(self, directory, *, scan_model='unverified', shadow=False, live_map=False):
        if scan_model not in {'unverified', 'snapshot', 'snapshot_assumed'}:
            raise ValueError('Unsupported LiDAR scan model')
        if (shadow or live_map) and scan_model not in {'snapshot', 'snapshot_assumed'}:
            raise ValueError('FAST-LIO shadow needs explicitly verified snapshot LiDAR')
        self.directory = Path(directory)
        self.scan_model, self.shadow = scan_model, shadow or live_map
        self.live_map = None
        if live_map:
            from localization.fastlio_live_map import FastLioLiveMap
            self.live_map = FastLioLiveMap(self.directory, scan_model)
        self.ready = threading.Event()
        self.estimator_error = None
        self.queue = queue.Queue(maxsize=50000)
        self.shadow_queue = queue.Queue(maxsize=50000) if self.shadow else None
        self._shadow_stop = threading.Event()
        self._shadow_thread = None
        self._shadow_accepting = self.shadow
        self.shadow_overflow = 0
        self.estimator_scans = 0
        self.shadow_skipped_scans = 0
        self.shadow_reordered_imu = 0
        self.accepting = False
        self.connection = self.mav = self.thread = None
        self.error = None
        self.counts = dict(imu=0, scans=0, overflow=0, partial_imu=0, other_sensor=0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._hook = self._receive
        self._progress = 0
        self.receive_diagnostics = {'imu_gap_count': 0, 'max_imu_gap_s': 0.,
                                    'gap_examples': [], 'udp_receive_buffer_bytes': None}
        self._last_imu_timing = None

    def start(self, mav):
        connection = getattr(mav, 'the_connection', None)
        if connection is None or not hasattr(connection, 'message_hooks'):
            raise RuntimeError('FAST-LIO requires the existing pymavlink message_hooks receiver')
        self.directory.mkdir(parents=True, exist_ok=False)
        # Catch missing libraries/native extension before entering flight.
        import h5py  # noqa: F401
        if self.shadow:
            from localization import _native_fastlio  # noqa: F401
        self.mav, self.connection = mav, connection
        port = getattr(connection, 'port', None)
        if isinstance(port, socket.socket) and port.type & socket.SOCK_DGRAM:
            # Allow short scheduling stalls without overflowing the shared UDP
            # receiver; this cannot recover samples never sent by PX4.
            before = port.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            port.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, max(before, 4 * 1024 * 1024))
            self.receive_diagnostics['udp_receive_buffer_bytes'] = port.getsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.accepting = True
        if self.shadow:
            self._shadow_thread = threading.Thread(target=self._estimate, name='FastLioEstimator', daemon=True)
            self._shadow_thread.start()
        self.thread = threading.Thread(target=self._write, name='FastLioInputs', daemon=True)
        self.thread.start()
        connection.message_hooks.append(self._hook)
        mav.SendMavCmdLong(511, 105, 5000)  # HIGHRES_IMU, requested 200 Hz; audit actual source intervals.

    def _enqueue(self, kind, value):
        with self._lock:
            if not self.accepting:
                return
            try:
                self.queue.put_nowait((kind, value))
            except queue.Full:
                self.counts['overflow'] += 1
                self.error = 'FAST-LIO input queue overflow; replay data is incomplete'
                self.accepting = False

    def _receive(self, connection, message):
        if message.get_type() != 'HIGHRES_IMU' or not self.accepting:
            return
        try:
            sensor_id = int(getattr(message, 'id', 0))
            if sensor_id != 0:
                self.counts['other_sensor'] += 1
                return
            mask = int(message.fields_updated)
            if mask & 63 != 63:
                self.counts['partial_imu'] += 1
                return
            row = [float(message.time_usec)*1e-6, message.xacc, message.yacc, message.zacc,
                   message.xgyro, message.ygyro, message.zgyro,
                   float(getattr(message, '_timestamp', time.time())), mask, sensor_id]
            previous = self._last_imu_timing
            if previous is not None:
                gap = row[0] - previous[0]
                self.receive_diagnostics['max_imu_gap_s'] = max(
                    self.receive_diagnostics['max_imu_gap_s'], gap)
                if gap > .1:
                    self.receive_diagnostics['imu_gap_count'] += 1
                    examples = self.receive_diagnostics['gap_examples']
                    if len(examples) < 20:
                        examples.append({'source_before': previous[0], 'source_after': row[0],
                                         'source_gap_s': gap, 'receive_gap_s': row[7] - previous[1]})
            self._last_imu_timing = (row[0], row[7])
            self._enqueue('imu', row)
        except Exception as error:
            self.error = f'Invalid HIGHRES_IMU: {error}'
            self.accepting = False

    def submit_scan(self, frame, state, sync_error):
        if self.accepting:
            pose = np.r_[state.position_xyz, state.quaternion]
            self._enqueue('scan', (float(state.px4_boot_timestamp), float(frame.timestamp),
                                   float(sync_error), frame.points_body.copy(), pose))

    def _write(self):
        import h5py
        try:
            with h5py.File(self.directory/'inputs.h5', 'w') as output:
                output.attrs['schema_version'] = 1
                output.attrs['point_frame'] = 'body_FRD_m'
                output.attrs['imu_frame'] = 'body_FRD_specific_force_mps2_gyro_radps'
                output.attrs['scan_model'] = self.scan_model
                output.attrs['clock_alignment'] = 'PX4 boot time interpolated at LiDAR receive-aligned time; NOT hardware synchronized'
                output.attrs['imu_columns'] = 'source_s,ax,ay,az,gx,gy,gz,received_epoch_s,fields_updated,sensor_id'
                ds = output.create_dataset('imu', (0,10), maxshape=(None,10), dtype='f8', chunks=True)
                scans = output.create_group('scans')
                imu_buffer = []

                def flush_imu():
                    if not imu_buffer:
                        return
                    start = len(ds)
                    ds.resize(start + len(imu_buffer), axis=0)
                    ds[start:] = np.asarray(imu_buffer)
                    self.counts['imu'] += len(imu_buffer)
                    self._progress += len(imu_buffer)
                    imu_buffer.clear()

                while not self._stop.is_set() or not self.queue.empty():
                    try:
                        kind, value = self.queue.get(timeout=.1)
                    except queue.Empty:
                        flush_imu()
                        continue
                    if self._shadow_accepting:
                        try:
                            self.shadow_queue.put_nowait((kind, value))
                        except queue.Full:
                            self.shadow_overflow += 1
                            self.estimator_error = 'FAST-LIO estimator queue overflow; offline replay required'
                            self._shadow_accepting = False
                    if kind == 'imu':
                        imu_buffer.append(value)
                        if len(imu_buffer) >= 256:
                            flush_imu()
                        continue
                    flush_imu()
                    stamp, received, sync_error, points, pose = value
                    group=scans.create_group(f'{self.counts["scans"]:08d}')
                    group.attrs['source_timestamp']=stamp;group.attrs['shared_timestamp']=received
                    group.attrs['pose_sync_error']=sync_error
                    group.create_dataset('points_body',data=points,compression='lzf')
                    group.create_dataset('px4_pose',data=pose)
                    self.counts['scans']+=1
                    self._progress += 1
                    if self.counts['scans'] % 10 == 0:
                        output.flush()
                flush_imu()
        except Exception as error:
            self.error=f'{type(error).__name__}: {error}'
            self.accepting=False
        finally:
            self._shadow_stop.set()
            if self._shadow_thread is not None:
                self._shadow_thread.join()
            summary=dict(self.counts,receive_diagnostics=self.receive_diagnostics,
                         error=self.error,shadow_error=self.estimator_error,
                         completed=self.error is None and self._stop.is_set() and self.queue.empty(),
                         shadow_completed=(not self.shadow or (self.estimator_error is None and self.shadow_queue.empty())),
                         shadow_overflow=self.shadow_overflow,
                         estimator_scans=self.estimator_scans,
                         shadow_skipped_scans=self.shadow_skipped_scans,
                         shadow_reordered_imu=self.shadow_reordered_imu,
                         scan_model=self.scan_model,shadow=self.shadow,controls_navigation=False)
            (self.directory/'capture_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            if self.error or self.estimator_error:
                print(f'--- FAST-LIO recording/shadow diagnostic: {self.error or self.estimator_error} ----',flush=True)

    def _estimate(self):
        from localization.fastlio_backend import FastLioBackend
        pending_imu = []
        last_source = last_processed = -float('inf')
        try:
            backend = FastLioBackend(instantaneous=True)
            with (self.directory/'shadow_poses.jsonl').open('w', encoding='utf-8') as trajectory:
                while not self._shadow_stop.is_set() or not self.shadow_queue.empty():
                    try:
                        kind, value = self.shadow_queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    self._progress += 1
                    if self.estimator_error is not None:
                        continue
                    if kind == 'imu':
                        if value[0] <= last_source:
                            if last_source - value[0] > .05 or value[0] <= last_processed:
                                raise ValueError('IMU source time duplicated or moved backwards beyond pending window')
                            index = bisect.bisect_left([row[0] for row in pending_imu], value[0])
                            if index < len(pending_imu) and pending_imu[index][0] == value[0]:
                                raise ValueError('IMU source time duplicated')
                            pending_imu.insert(index, value[:7])
                            self.shadow_reordered_imu += 1
                        else:
                            pending_imu.append(value[:7])
                            last_source = value[0]
                        continue
                    stamp, received, _, points, pose = value
                    batch = [r for r in pending_imu if last_processed < r[0] <= stamp]
                    if len(batch) < 2 or stamp-batch[-1][0] > .03:
                        self.shadow_skipped_scans += 1
                        continue
                    result = backend.process(stamp, np.column_stack((points,np.zeros(len(points)))), np.asarray(batch))
                    self.estimator_scans += 1
                    last_processed = stamp
                    pending_imu = [r for r in pending_imu if r[0] > stamp]
                    if self.live_map is not None:
                        self.live_map.update(result, pose, received)
                        result['shared_timestamp'] = received
                        if result.get('status') == 'tracking':
                            result['px4_to_lio_map'] = self.live_map.semantics.poses[-1][1]
                    if result.get('status') == 'tracking':
                        self.ready.set()
                    result['px4_pose'] = pose
                    result.pop('points_world',None)
                    result = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
                    trajectory.write(json.dumps(result,allow_nan=False)+'\n')
        except Exception as error:
            self.estimator_error = f'{type(error).__name__}: {error}'
            self._shadow_accepting = False

    def wait_ready(self, timeout=45.):
        deadline = time.monotonic() + timeout
        while not self.ready.wait(.1):
            if self.error or self.estimator_error:
                raise RuntimeError(self.error or self.estimator_error)
            if time.monotonic() >= deadline:
                raise TimeoutError("FAST-LIO did not initialize while stationary; flight not started")

    def close(self):
        with self._lock:
            self.accepting=False
        if self.connection is not None and self._hook in self.connection.message_hooks:
            self.connection.message_hooks.remove(self._hook)
            self.mav.SendMavCmdLong(511,105,0)
        self._stop.set()
        if self.thread is not None:
            started = last_progress = last_notice = time.monotonic()
            from navigation.terminal_progress import shared_progress
            display = shared_progress()
            progress = self._progress
            while self.thread.is_alive():
                self.thread.join(timeout=.2)
                now = time.monotonic()
                if self._progress != progress:
                    progress, last_progress = self._progress, now
                if not self.thread.is_alive():
                    break
                if now - last_progress >= 30. or now - started >= 300.:
                    display.finish()
                    raise TimeoutError(
                        f'FAST-LIO recorder did not finish: queued={self.queue.qsize()}, '
                        f'written_imu={self.counts["imu"]}, written_scans={self.counts["scans"]}, '
                        f'no_progress_seconds={now-last_progress:.1f}')
                if now - last_notice >= 5.:
                    display.update('fastlio_save', detail=f'待处理 {self.queue.qsize()} | IMU {self.counts["imu"]} | 雷达 {self.counts["scans"]}')
                    last_notice = now
            display.finish()
            self.thread=None
            if self.live_map is not None:
                self.live_map.close()
