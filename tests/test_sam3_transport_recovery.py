"""Transient local transport faults must not publish stale semantic frames."""
from dataclasses import replace
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

import numpy as np
import pytest

from detections.object_detect import DetectorConfig
from detections.sam3_client import SAM3Detector
from detections.sam3_transport import SAM3TransportError
from detections.realtime_detection_worker import RealtimeDetectionWorker
from tools.sam3_service import SAM3HTTPServer, handler_for


@pytest.mark.parametrize('failure', [urllib.error.URLError(TimeoutError('timed out')),
    ConnectionResetError('reset'), TimeoutError('send timed out')])
def test_transport_errors_are_recoverable_and_clear_previous_masks(failure):
    client = object.__new__(SAM3Detector)
    client._wait_response = Mock()
    client._read_response = lambda response,*args:response.read()
    client.config = DetectorConfig(sam3_request_timeout=2.)
    client.url = 'http://127.0.0.1:8878'
    connection = Mock()
    connection.request.side_effect = failure
    client.masks = [np.ones((4,4), bool)]
    client.last_inference_seconds = 1.
    with patch('detections.sam3_client.http.client.HTTPConnection', return_value=connection) as factory:
        with pytest.raises(SAM3TransportError) as caught:
            client.detect(np.zeros((4,4,3), np.uint8))
        factory.assert_called_once_with('127.0.0.1', 8878, timeout=2.)
    assert caught.value.phase == 'send'
    assert client.masks == [] and client.last_inference_seconds is None
    assert client._connection is None
    connection.close.assert_called_once()


def test_model_http_errors_remain_fatal():
    client = object.__new__(SAM3Detector)
    client._wait_response = Mock()
    client._read_response = lambda response,*args:response.read()
    client.config = DetectorConfig()
    client.url = 'http://127.0.0.1:8878'
    connection = Mock()
    connection.getresponse.return_value = SimpleNamespace(status=500, reason='model failed',
        headers={}, will_close=True, read=lambda: b'model failed')
    with patch('detections.sam3_client.http.client.HTTPConnection', return_value=connection):
        with pytest.raises(urllib.error.HTTPError):
            client.detect(np.zeros((4,4,3), np.uint8))


@pytest.mark.parametrize('phase, method', [('connect','connect'), ('read','getresponse')])
def test_failure_phase_is_reported_and_broken_connection_is_discarded(phase, method):
    client = object.__new__(SAM3Detector)
    client._wait_response = Mock()
    client._read_response = lambda response,*args:response.read()
    client.config = DetectorConfig()
    client.url = 'http://127.0.0.1:8878'
    connection = Mock()
    getattr(connection, method).side_effect = TimeoutError('timed out')
    with patch('detections.sam3_client.http.client.HTTPConnection', return_value=connection):
        with pytest.raises(SAM3TransportError) as caught:
            client.detect(np.zeros((4,4,3), np.uint8))
    assert caught.value.phase == phase
    assert client.last_transport_timings['failure_phase'] == phase
    assert client._connection is None


def test_busy_response_is_recoverable_and_connection_is_closed():
    client = object.__new__(SAM3Detector)
    client._wait_response = Mock()
    client._read_response = lambda response,*args:response.read()
    client.config = DetectorConfig()
    client.url = 'http://127.0.0.1:8878'
    connection = Mock()
    connection.getresponse.return_value = SimpleNamespace(status=503, reason='Busy',
        headers={}, will_close=True, read=lambda: b'')
    with patch('detections.sam3_client.http.client.HTTPConnection', return_value=connection):
        with pytest.raises(SAM3TransportError) as caught:
            client.detect(np.zeros((4,4,3), np.uint8))
    assert caught.value.phase == 'busy'
    assert client._connection is None


@pytest.mark.parametrize('protocol', ['HTTP/1.1', 'HTTP/1.0'])
def test_connection_reuse_and_legacy_server_compatibility(protocol):
    class Engine:
        metadata = {'api':'Sam3Processor/image'}
        def predict(self, image, prompts, confidence):
            h,w = image.shape[:2]
            return dict(boxes=np.empty((0,4)), scores=np.empty(0), labels=np.array([], dtype='U128'),
                        masks=np.empty((0,h,w), bool), inference_seconds=np.asarray(.01))
    class Server(SAM3HTTPServer):
        accepted = 0
        def get_request(self):
            connection = super().get_request()
            self.accepted += 1
            return connection
    handler = handler_for(Engine())
    handler.protocol_version = protocol
    server = Server(('127.0.0.1',0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = None
    try:
        client = SAM3Detector(DetectorConfig(sam3_url=f'http://127.0.0.1:{server.server_port}'))
        for _ in range(3):
            assert client.detect(np.zeros((4,4,3),np.uint8)) == []
        assert server.accepted == (2 if protocol == 'HTTP/1.1' else 4)
        assert client.last_transport_timings['reused_connection'] == (protocol == 'HTTP/1.1')
    finally:
        if client is not None:
            client.close()
        server.shutdown(); server.server_close(); thread.join(2.)


def worker_for(outcomes):
    image = np.zeros((8,8,3), np.uint8)
    observations = [SimpleNamespace(sensor_frame=SimpleNamespace(rgb=image.copy(),
        depth=np.ones((8,8)), camera_intrinsics=np.eye(3), timestamp=float(i)),
        camera_pose_world=np.eye(4)) for i in range(1, len(outcomes)+1)]
    source = Mock()
    source.get_observation.side_effect = [*observations, RuntimeError('source closed')]
    detector = Mock(masks=[], last_inference_seconds=.01)
    detector.detect.side_effect = outcomes
    detector.draw.side_effect = lambda rgb, detections: rgb.copy()
    mapper = SimpleNamespace(navigation_records=Mock(return_value=[]), semantic_voxels=None,
        last_observation_batch=None, object_inspection_records=Mock(return_value=[]))
    segmenter = SimpleNamespace(mapper=mapper, memory_count=0, latest_memory=(),
        segment=Mock(return_value=(image.copy(), 0)))
    config = DetectorConfig(backend='sam3', save_videos=False, save_semantic_map=False,
                            sam3_retry_delay=0., inference_fps=1000.)
    with patch('detections.sam3_client.SAM3Detector', return_value=detector), \
         patch('detections.realtime_detection_worker.SemanticMaskProcessor', return_value=segmenter):
        worker = RealtimeDetectionWorker(source, config)
    worker._recording_enabled = False
    return worker, detector, segmenter


def test_retry_reads_new_frame_and_does_not_publish_failed_observation():
    worker, detector, segmenter = worker_for([SAM3TransportError('timeout'), []])
    worker._detection_loop()
    worker.raise_if_failed()
    assert detector.detect.call_count == 2
    assert segmenter.segment.call_count == worker.processed_frames == 1
    assert worker.semantic_snapshot().timestamp == 2.
    assert worker._sam3_transport_failures == 1
    assert worker._sam3_consecutive_failures == 0


def test_continuous_unavailability_stops_after_bounded_attempts():
    worker, detector, segmenter = worker_for([SAM3TransportError('timeout') for _ in range(4)])
    worker._detection_loop()
    assert detector.detect.call_count == 3
    segmenter.segment.assert_not_called()
    assert worker.semantic_snapshot() is None
    with pytest.raises(RuntimeError, match='3 consecutive requests'):
        worker.raise_if_failed()


def test_success_resets_consecutive_failure_budget():
    worker, _, _ = worker_for([SAM3TransportError('timeout'), [], SAM3TransportError('reset'), []])
    worker._detection_loop()
    worker.raise_if_failed()
    assert worker.processed_frames == 2
    assert worker.semantic_snapshot().timestamp == 4.
    assert worker._sam3_transport_failures == 2 and worker._sam3_consecutive_failures == 0


def test_invalid_predictions_are_not_retried():
    worker, detector, segmenter = worker_for([ValueError('SAM3 response shape mismatch'), []])
    worker._detection_loop()
    assert detector.detect.call_count == 1
    segmenter.segment.assert_not_called()
    with pytest.raises(RuntimeError, match='shape mismatch'):
        worker.raise_if_failed()


def test_recording_boundary_stops_perception_without_restarting():
    worker, _, _ = worker_for([])
    worker._recording_enabled = True
    with patch.object(worker, 'close'), patch.object(worker, 'start'):
        worker.finish_recording()
        worker.finish_recording()
        worker.close.assert_called_once()
        worker.start.assert_not_called()
    assert not worker._recording_enabled


def test_explicit_persistent_session_can_resume_without_recording():
    worker, _, _ = worker_for([])
    worker._recording_enabled = True
    with patch.object(worker, 'close'), patch.object(worker, 'start'):
        worker.finish_recording(resume=True)
        worker.close.assert_called_once()
        worker.start.assert_called_once()
    assert worker.target_inference_fps == .5
    assert not worker._recording_enabled


@pytest.mark.parametrize('fail', [False, True])
def test_navigation_restores_idle_even_when_a_target_request_fails(fail):
    from navigation.perception_mode import active_perception
    worker = Mock()
    class Session:
        runtime = SimpleNamespace(detection_worker=worker)
        @active_perception
        def navigate(self):
            assert worker.set_idle_mode.call_args.args == (False,)
            if fail:
                raise ValueError('no target')
            return 'arrived'
    if fail:
        with pytest.raises(ValueError):
            Session().navigate()
    else:
        assert Session().navigate() == 'arrived'
    assert [call.args for call in worker.set_idle_mode.call_args_list] == [(False,), (True,)]


def test_health_remains_responsive_and_model_jobs_do_not_queue():
    started, release = threading.Event(), threading.Event()
    class Engine:
        metadata = {'api':'Sam3Processor/image'}
        calls = 0
        def predict(self, image, prompts, confidence):
            self.calls += 1
            started.set()
            assert release.wait(5.)
            h,w = image.shape[:2]
            return dict(boxes=np.empty((0,4)), scores=np.empty(0), labels=np.array([], dtype='U128'),
                        masks=np.empty((0,h,w), bool), inference_seconds=np.asarray(.01))
    engine = Engine()
    server = SAM3HTTPServer(('127.0.0.1', 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = io.BytesIO()
    np.savez_compressed(payload, rgb=np.zeros((4,4,3), np.uint8), prompts=np.array(['car']), confidence=.5)
    results = []
    def predict():
        try:
            with opener.open(url+'/predict', data=payload.getvalue(), timeout=5.) as response:
                results.append(response.status)
        except Exception as error:
            results.append(error)
    first = threading.Thread(target=predict, daemon=True)
    first.start()
    try:
        assert started.wait(2.)
        with opener.open(url+'/health', timeout=1.) as response:
            assert json.load(response)['api'] == 'Sam3Processor/image'
        with pytest.raises(urllib.error.HTTPError) as busy:
            opener.open(url+'/predict', data=payload.getvalue(), timeout=1.)
        assert busy.value.code == 503
        busy.value.close()
        assert engine.calls == 1
    finally:
        release.set()
        first.join(3.)
        server.shutdown()
        server.server_close()
        thread.join(2.)
    assert results == [200]


@pytest.mark.parametrize('overrides', [dict(sam3_request_timeout=0.), dict(sam3_request_timeout=float('nan')),
    dict(sam3_retry_delay=-1.), dict(sam3_max_retries=-1), dict(sam3_max_retries=True),
    dict(idle_inference_fps=0.)])
def test_invalid_recovery_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        replace(DetectorConfig(), **overrides)
