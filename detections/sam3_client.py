"""SAM3 process boundary; bypass system proxies for the local model service."""
import io
import http.client
import json
import socket
import select
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

import numpy as np
import torch
from .object_detect import Detection, TiledVehicleDetector, canonical_class_name
from .sam3_transport import SAM3TransportError


class SAM3Detector(TiledVehicleDetector):
    def __init__(self, config):
        # Reuse only class-aware NMS and drawing; never load YOLO or SAM2.
        self.config = config
        if len(config.class_prompts) > 32 or any(len(p) > 128 for p in config.class_prompts):
            raise ValueError("SAM3 accepts up to 32 prompts, each up to 128 characters")
        url = config.sam3_url.rstrip('/')
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('SAM3 must use a local HTTP endpoint')
        self.url = url
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with self.opener.open(url+'/health', timeout=5.) as response:
                self.metadata = json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f'Cannot connect to SAM3 service at {url}') from error
        if self.metadata.get('api') != 'Sam3Processor/image':
            raise ValueError('Unexpected SAM3 server API')
        self.masks = []
        self.last_inference_seconds = None
        self.last_transport_timings = {}
        self._connection = None
        self._active_socket = None
        self._connection_generation = 0

    def close(self):
        self._connection_generation = getattr(self,'_connection_generation',0)+1
        connection = getattr(self, '_connection', None)
        self._connection = None
        active_socket = getattr(self,'_active_socket',None)
        self._active_socket = None
        if active_socket is None and connection is not None:
            active_socket = getattr(connection,'sock',None)
        if active_socket is not None:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection is not None:
            connection.close()

    def cancel_pending(self):
        """Interrupt blocked loopback reads so shutdown need not await inference."""
        self.close()

    def _wait_response(self, active_socket, deadline, generation):
        # On Windows, closing a socket in another thread need not wake an
        # already-blocked recv. Poll readiness before entering HTTP/body reads.
        while True:
            if generation != getattr(self,'_connection_generation',0):
                raise SAM3TransportError('SAM3 request cancelled',phase='read')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('SAM3 response deadline exceeded')
            ready,_,_ = select.select([active_socket],[],[],min(.1,remaining))
            if ready:
                return

    def _read_response(self, response, active_socket, deadline, generation, timeout):
        if response.chunked or response.length is None:
            return response.read()  # Compatibility with nonstandard external services.
        remaining = response.length
        reader = response.fp
        if reader is None and remaining:
            raise http.client.IncompleteRead(b'',remaining)
        chunks = []
        active_socket.setblocking(False)
        try:
            while remaining:
                if generation != getattr(self,'_connection_generation',0):
                    raise SAM3TransportError('SAM3 request cancelled',phase='read')
                if time.monotonic() >= deadline:
                    raise TimeoutError('SAM3 response deadline exceeded')
                chunk = reader.read1(min(65536,remaining))
                if not chunk:
                    self._wait_response(active_socket,deadline,generation)
                    chunk = reader.read1(min(65536,remaining))
                    if not chunk:
                        raise http.client.IncompleteRead(b''.join(chunks),remaining)
                chunks.append(chunk)
                remaining -= len(chunk)
            response.length = 0
            response.close()
            return b''.join(chunks)
        finally:
            try:
                active_socket.settimeout(timeout)
            except OSError:
                pass

    def _predict_request(self, payload):
        """Reuse a direct loopback connection; retry belongs to the frame worker."""
        started = time.monotonic()
        timings = {'reused_connection': getattr(self, '_connection', None) is not None}
        phase = 'connect'
        generation = getattr(self,'_connection_generation',0)
        try:
            parsed = urlparse(self.url)
            timeout = float(getattr(self.config,'sam3_request_timeout',30.))
            connection = getattr(self,'_connection',None)
            if connection is None:
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
                    timeout=min(5.,timeout))
                self._connection = connection
            if not timings['reused_connection'] or connection.sock is None:
                connection.connect()
            # Explicit connect avoids silently reconnecting after close() has
            # cancelled this request in another thread.
            connection.auto_open = 0
            if generation != getattr(self,'_connection_generation',0):
                connection.close()
                raise SAM3TransportError('SAM3 request cancelled',phase='connect')
            active_socket = connection.sock
            self._active_socket = active_socket
            timings['connect_ms'] = (time.monotonic()-started)*1000.
            phase, mark = 'send', time.monotonic()
            if active_socket is not None:
                active_socket.settimeout(min(8.,timeout))
            connection.request('POST', parsed.path.rstrip('/')+'/predict', body=payload,
                                     headers={'Content-Type':'application/octet-stream'})
            timings['send_ms'] = (time.monotonic()-mark)*1000.
            phase, mark = 'read', time.monotonic()
            if active_socket is not None:
                active_socket.settimeout(timeout)
                self._wait_response(active_socket,mark+timeout,generation)
            response = connection.getresponse()
            body = self._read_response(response,active_socket,mark+timeout,generation,timeout)
            timings['read_ms'] = (time.monotonic()-mark)*1000.
            if response.will_close:
                self.close()
            if response.status >= 400:
                self.close()
                if response.status == 503:
                    raise SAM3TransportError(f'SAM3 service is busy at {self.url}', phase='busy')
                raise urllib.error.HTTPError(self.url+'/predict', response.status,
                                             response.reason, response.headers, None)
            return body
        except SAM3TransportError:
            self.close()
            raise
        except urllib.error.HTTPError:
            self.close()
            raise
        except ValueError as error:
            if generation != getattr(self,'_connection_generation',0):
                self.close()
                raise SAM3TransportError('SAM3 request cancelled',phase=phase) from error
            raise
        except (OSError, http.client.HTTPException) as error:
            self.close()
            timings['failure_phase'] = phase
            raise SAM3TransportError(f'SAM3 {phase} failed at {self.url}: {error}', phase=phase) from error
        finally:
            timings['request_ms'] = (time.monotonic()-started)*1000.
            self.last_transport_timings = timings

    def detect(self, image):
        self.masks = []
        self.last_inference_seconds = None
        payload = io.BytesIO()
        np.savez_compressed(payload, rgb=np.ascontiguousarray(image[:,:,::-1]),
                            prompts=np.asarray(self.config.class_prompts, dtype='U128'),
                            confidence=np.asarray(self.config.confidence))
        body = self._predict_request(payload.getvalue())
        with np.load(io.BytesIO(body), allow_pickle=False) as result:
            boxes, scores = result['boxes'].copy(), result['scores'].copy()
            masks, labels = result['masks'].copy(), result['labels'].tolist()
            self.last_inference_seconds = float(result['inference_seconds'])
        n = len(scores)
        h, w = image.shape[:2]
        if boxes.shape != (n,4) or masks.shape != (n,h,w) or len(labels) != n:
            raise ValueError('SAM3 response shape mismatch')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite SAM3 predictions')
        labels = [canonical_class_name(label) for label in labels]
        keep = self._nms(torch.from_numpy(boxes), torch.from_numpy(scores), labels)
        detections = []
        for index in keep.tolist()[:self.config.max_detections]:
            x1,y1 = np.floor(boxes[index,:2]).astype(int)
            x2,y2 = np.ceil(boxes[index,2:]).astype(int)
            box = (max(0,min(w,x1)), max(0,min(h,y1)), max(0,min(w,x2)), max(0,min(h,y2)))
            if box[0] >= box[2] or box[1] >= box[3] or not masks[index].any():
                continue
            detections.append(Detection(box, float(scores[index]), labels[index]))
            self.masks.append(masks[index])
        return detections
