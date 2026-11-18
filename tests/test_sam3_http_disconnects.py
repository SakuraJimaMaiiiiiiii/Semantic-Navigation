import io
import json
import threading
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import urllib.error
import urllib.request
import numpy as np
import pytest
from detections.object_detect import DetectorConfig
from detections.sam3_client import SAM3Detector
from detections.sam3_transport import SAM3TransportError
from tools.sam3_service import SAM3HTTPServer,handler_for


def payload():
    stream=io.BytesIO()
    np.savez_compressed(stream,rgb=np.zeros((4,4,3),np.uint8),prompts=np.array(['car']),confidence=.5)
    return stream.getvalue()


class Engine:
    metadata={'api':'Sam3Processor/image'}
    def predict(self,image,prompts,confidence):
        h,w=image.shape[:2]
        return dict(boxes=np.empty((0,4)),scores=np.empty(0),labels=np.array([],dtype='U128'),
                    masks=np.empty((0,h,w),bool),inference_seconds=np.asarray(.01))


@pytest.mark.parametrize('error',[ConnectionAbortedError('软件中止连接'),ConnectionResetError('强迫关闭连接')])
def test_disconnected_result_is_not_answered_with_a_second_500(error):
    handler_type=handler_for(Engine())
    handler=object.__new__(handler_type)
    data=payload()
    handler.path='/predict'
    handler.headers={'Content-Length':str(len(data))}
    handler.rfile=io.BytesIO(data)
    handler.wfile=Mock(write=Mock(side_effect=error))
    handler.connection=Mock()
    handler.send_response=Mock()
    handler.send_header=Mock()
    handler.end_headers=Mock()
    handler.do_POST()
    handler.send_response.assert_called_once_with(200)
    assert handler.close_connection


@pytest.mark.parametrize('error',[ConnectionResetError('closed idle socket'),ConnectionAbortedError('aborted idle socket')])
def test_idle_keep_alive_disconnect_has_no_handler_traceback(error):
    handler=object.__new__(handler_for(Engine()))
    with patch.object(BaseHTTPRequestHandler,'handle',side_effect=error):
        handler.handle()
    assert handler.close_connection


@pytest.mark.parametrize('error',[RuntimeError('模型推理失败：显存不足'),ConnectionAbortedError('模型内部连接失败')])
def test_unicode_model_error_is_a_real_500_with_utf8_json(error):
    engine=Engine()
    engine.predict=Mock(side_effect=error)
    server=SAM3HTTPServer(('127.0.0.1',0),handler_for(engine))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            opener.open(url+'/predict',data=payload(),timeout=2.)
        with caught.value as response:
            assert response.code==500
            assert json.load(response)['message']==str(error)
        with opener.open(url+'/health',timeout=1.) as response:
            assert response.status==200
        engine.predict=Engine().predict
        with opener.open(url+'/predict',data=payload(),timeout=2.) as response:
            assert response.status==200  # Failed job released the single-model lock.
    finally:
        server.shutdown();server.server_close();thread.join(2.)


def test_read_budget_is_longer_than_connect_and_send_budgets():
    client=object.__new__(SAM3Detector)
    client._wait_response=Mock()
    client._read_response=lambda response,*args:response.read()
    client.config=DetectorConfig()
    client.url='http://127.0.0.1:8878'
    connection=Mock()
    connection.getresponse.return_value=NS(status=200,will_close=False,read=lambda:b'result')
    with patch('detections.sam3_client.http.client.HTTPConnection',return_value=connection) as factory:
        assert client._predict_request(b'request')==b'result'
    factory.assert_called_once_with('127.0.0.1',8878,timeout=5.)
    assert [c.args for c in connection.sock.settimeout.call_args_list]==[(8.,),(30.,)]
    client.close()


def test_shutdown_interrupts_pending_read_without_waiting_for_model():
    entered,release=threading.Event(),threading.Event()
    class SlowEngine(Engine):
        def predict(self,*args):
            entered.set()
            assert release.wait(5.)
            return super().predict(*args)
    server=SAM3HTTPServer(('127.0.0.1',0),handler_for(SlowEngine()))
    serve=threading.Thread(target=server.serve_forever,daemon=True);serve.start()
    client=SAM3Detector(DetectorConfig(sam3_url=f'http://127.0.0.1:{server.server_port}'))
    results=[]
    def predict():
        try:client.detect(np.zeros((4,4,3),np.uint8))
        except Exception as error:results.append(error)
    request=threading.Thread(target=predict,daemon=True)
    request.start()
    try:
        assert entered.wait(2.)
        client.cancel_pending()
        request.join(1.)
        assert not request.is_alive()
        assert len(results)==1 and isinstance(results[0],SAM3TransportError)
        assert not release.is_set()
    finally:
        release.set();request.join(2.);client.close()
        server.shutdown();server.server_close();serve.join(2.)


def test_worker_cancels_before_joining_perception_thread():
    from detections.realtime_detection_worker import RealtimeDetectionWorker
    worker=object.__new__(RealtimeDetectionWorker)
    worker._stop=threading.Event()
    order=[]
    worker.detector=NS(cancel_pending=lambda:order.append('cancel'),close=lambda:order.append('close'))
    worker._thread=NS(join=lambda **kwargs:order.append('join'),is_alive=lambda:False)
    worker._stop_video_thread=Mock()
    worker._semantic_history=None
    worker.config=NS(save_semantic_map=False)
    worker.close()
    assert order==['cancel','join','close']


def test_client_timeout_discards_frame_and_server_can_serve_next_request(capsys):
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    class SlowEngine(Engine):
        calls=0
        def predict(self,*args):
            self.calls += 1
            if self.calls==1:
                entered.set()
                assert release.wait(5.)
            return super().predict(*args)
    class Handler(handler_for(SlowEngine())):
        def do_POST(self):
            try:super().do_POST()
            finally:finished.set()
    server=SAM3HTTPServer(('127.0.0.1',0),Handler)
    serve=threading.Thread(target=server.serve_forever,daemon=True);serve.start()
    client=SAM3Detector(DetectorConfig(sam3_url=f'http://127.0.0.1:{server.server_port}',sam3_request_timeout=.1))
    try:
        with pytest.raises(SAM3TransportError) as caught:
            client.detect(np.zeros((4,4,3),np.uint8))
        assert entered.is_set() and caught.value.phase=='read'
        assert client.masks==[] and client.last_inference_seconds is None
        release.set()
        assert finished.wait(2.)
        assert client.detect(np.zeros((4,4,3),np.uint8))==[]
    finally:
        release.set();client.close()
        server.shutdown();server.server_close();serve.join(2.)
    assert 'Traceback' not in capsys.readouterr().err


@pytest.mark.parametrize('cancel',[True,False])
def test_partial_response_body_can_be_cancelled_or_time_out(cancel):
    started,release=threading.Event(),threading.Event()
    class Handler(handler_for(Engine())):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Length','1000')
            self.end_headers()
            self.wfile.write(b'x')
            self.wfile.flush()
            self.close_connection=True
            started.set()
            release.wait(5.)
    server=SAM3HTTPServer(('127.0.0.1',0),Handler)
    serve=threading.Thread(target=server.serve_forever,daemon=True);serve.start()
    client=SAM3Detector(DetectorConfig(sam3_url=f'http://127.0.0.1:{server.server_port}',
                                     sam3_request_timeout=30. if cancel else .1))
    results=[]
    def predict():
        try:client.detect(np.zeros((4,4,3),np.uint8))
        except Exception as error:results.append(error)
    request=threading.Thread(target=predict,daemon=True);request.start()
    try:
        assert started.wait(2.)
        if cancel:
            client.cancel_pending()
        request.join(1.)
        assert not request.is_alive()
        assert len(results)==1 and isinstance(results[0],SAM3TransportError)
    finally:
        release.set();request.join(2.);client.close()
        server.shutdown();server.server_close();serve.join(2.)
