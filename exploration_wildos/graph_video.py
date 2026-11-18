"""Bounded graph-only video; the encoder is owned and released by one thread."""
from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np


class GraphVideoRecorder:
    def __init__(self, path, position, home, yaw, *, fps=10., size=(1280, 960)):
        self.path, self.position = Path(path), position
        self.home = np.asarray(home, dtype=float)
        self.yaw, self.fps, self.size = float(yaw), float(fps), size
        self.lock = threading.Lock()
        self.stop, self.ready = threading.Event(), threading.Event()
        self.thread = None
        self.snapshot = None
        self.version = None
        self.error = None
        self.closed = False
        self.reported = False
        self.trail = deque(maxlen=4000)

    def publish(self, adapter, graph, version):
        if self.closed or version == self.version:
            return
        grid = adapter.grid
        # Copy only drawing data. No SDK, route-grid methods or frame queue.
        data = dict(version=version, resolution=float(grid.resolution), origin=np.asarray(grid.origin).copy(),
            known=np.asarray(list(grid.known), dtype=float).reshape(-1, 2),
            free=np.asarray(list(grid.free), dtype=float).reshape(-1, 2),
            nodes=np.asarray([n['position_ned'][:2] for n in graph['nodes']], dtype=float).reshape(-1, 2),
            edges=[(e['source'], e['target']) for e in graph['edges']],
            frontier_nodes=[n['id'] for n in graph['nodes'] if n.get('is_frontier')],
            frontiers=np.asarray([f['position_ned'][:2] for f in graph.get('frontiers', [])], dtype=float).reshape(-1, 2),
            vehicles=[(list(n['bbox_3d']['min'][:2]), list(n['bbox_3d']['max'][:2]))
                      for n in graph.get('semantic_nodes', [])])
        with self.lock:
            if not self.closed:
                self.snapshot, self.version = data, version

    def start(self):
        if self.thread is not None or self.closed:
            raise RuntimeError('Graph video recorder cannot be restarted')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to truncate an existing recording in this run.
        with self.path.open('xb'):
            pass
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name='WildOSGraphVideo', daemon=True)
        self.thread.start()
        if not self.ready.wait(10.):
            self.close()
            raise RuntimeError('Graph video encoder startup timed out')
        if self.error:
            self.close()
            raise RuntimeError('Graph video encoder could not start: '+str(self.error))

    def _local(self, points):
        xy = np.asarray(points, dtype=float).reshape(-1, 2)-self.home[:2]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return xy @ np.array([[-s, -c], [c, -s]])

    def _base(self, data):
        w, h = self.size
        canvas = np.full((h, w, 3), 176, np.uint8)
        world = self.home[:2].reshape(1, 2) if data is None else np.vstack([
            self.home[:2], data['known']*data['resolution']+data['origin']+data['resolution']*.5])
        xy = self._local(world)
        low, high = xy.min(0)-2., xy.max(0)+2.
        scale = min((w-60)/max(high[0]-low[0], 4.), (h-100)/max(high[1]-low[1], 4.))
        middle = (high+low)*.5
        def pixel(points):
            return np.rint((self._local(points)-middle)*scale+[w/2, (h+40)/2]).astype(np.int32)
        if data is not None:
            r, origin = data['resolution'], data['origin']
            for cells, color in ((data['known'], (128, 121, 112)), (data['free'], (255, 255, 255))):
                for x, y in cells*r+origin:
                    box = pixel([[x, y], [x+r, y], [x+r, y+r], [x, y+r]])
                    cv2.fillConvexPoly(canvas, box, color)
            nodes = pixel(data['nodes'])
            for a, b in data['edges']:
                cv2.line(canvas, tuple(nodes[a]), tuple(nodes[b]), (245, 222, 207), 1, cv2.LINE_AA)
            for low_box, high_box in data['vehicles']:
                x, y = low_box; xx, yy = high_box
                box = pixel([[x, y], [xx, y], [xx, yy], [x, yy]])
                cv2.fillConvexPoly(canvas, box, (116, 186, 253))
                cv2.polylines(canvas, [box], True, (12, 88, 210), 2)
            for point in nodes:
                cv2.circle(canvas, tuple(point), 2, (110, 155, 25), -1, cv2.LINE_AA)
            for identity in data['frontier_nodes']:
                cv2.circle(canvas, tuple(nodes[identity]), 4, (234, 85, 168), 1, cv2.LINE_AA)
            for point in pixel(data['frontiers']):
                cv2.circle(canvas, tuple(point), 1, (234, 85, 168), -1)
        return canvas, pixel

    def _run(self):
        writer = None
        last_frame = None
        try:
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
            if not writer.isOpened():
                raise RuntimeError('MP4 encoder unavailable')
            self.ready.set()
            version, base, pixel = object(), None, None
            deadline = self.started
            while not self.stop.is_set():
                with self.lock:
                    data = self.snapshot
                current_version = None if data is None else data['version']
                if base is None or version != current_version:
                    base, pixel = self._base(data)
                    version = current_version
                frame = base.copy()
                height = None
                try:
                    point = np.asarray(self.position(), dtype=float)
                    if point.shape == (3,) and np.isfinite(point).all():
                        height = self.home[2]-point[2]
                        if not self.trail or np.linalg.norm(point[:2]-self.trail[-1]) >= .1:
                            self.trail.append(point[:2].copy())
                        if len(self.trail) > 1:
                            cv2.polylines(frame, [pixel(self.trail)], False, (80, 80, 225), 1, cv2.LINE_AA)
                        cv2.circle(frame, tuple(pixel([point[:2]])[0]), 6, (15, 15, 15), -1, cv2.LINE_AA)
                        cv2.circle(frame, tuple(pixel([self.home[:2]])[0]), 7, (190, 100, 20), 1, cv2.LINE_AA)
                except (RuntimeError, OSError):
                    pass  # Keep the last graph during a telemetry interruption.
                elapsed = time.monotonic()-self.started
                cv2.rectangle(frame, (0, 0), (self.size[0], 38), (245, 245, 245), -1)
                text = f'WildOS graph exploration | {elapsed:7.1f}s | update {current_version if data else "waiting"}'
                if height is not None:
                    text += f' | Height {height:.2f}m'
                cv2.putText(frame, text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (35, 35, 35), 1, cv2.LINE_AA)
                writer.write(frame)
                last_frame = frame
                deadline += 1./self.fps
                self.stop.wait(max(0., deadline-time.monotonic()))
            if last_frame is not None:
                writer.write(last_frame)
        except Exception as error:
            self.error = error
        finally:
            self.ready.set()
            if writer is not None:
                writer.release()

    def close(self):
        self.closed = True
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.error and not self.reported:
            print(f'--- 图探索视频录制失败，已关闭写入：{self.error} ----')
            self.reported = True
