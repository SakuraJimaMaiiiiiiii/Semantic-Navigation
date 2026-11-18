"""API/transport and shared mapping tests without downloading model weights."""
import ast
from contextlib import nullcontext
from dataclasses import dataclass
import io
import http.client
import json
import socket
import select
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.request
from urllib.parse import urlparse
from http.server import HTTPServer

import cv2
import numpy as np
from tools.sam3_service import OfficialSAM3, handler_for, check_environment
from unittest.mock import patch


def extract(path, name, namespace):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<test>', 'exec'), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, value): self.value=np.asarray(value)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


@dataclass
class Detection:
    box: tuple
    confidence: float
    class_name: str


class FakeDetector:
    def _nms(self, boxes, scores, labels):
        return np.argsort(-scores)


class SAM3Tests(unittest.TestCase):
    def test_official_image_api_resets_prompts_and_returns_original_resolution(self):
        calls=[]
        class Processor:
            def set_confidence_threshold(self,t): calls.append(('threshold',t))
            def set_image(self,image):
                self.size=image.size
                calls.append(('image', image.mode))
                return {}
            def reset_all_prompts(self,state): calls.append(('reset',))
            def set_text_prompt(self,prompt,state):
                calls.append(('text',prompt))
                w,h=self.size
                state.update(boxes=Tensor([[0,0,w,h]]),scores=Tensor([.9]),masks=Tensor(np.ones((1,1,h,w),bool)))
                return state
        engine=OfficialSAM3.__new__(OfficialSAM3)
        engine.processor=Processor()
        engine.device="cpu"
        engine.torch=SimpleNamespace(inference_mode=nullcontext)
        result=engine.predict(np.zeros((4,7,3),np.uint8),['car','fire hose cabinet'],.5)
        self.assertEqual(result['masks'].shape,(2,4,7))
        self.assertEqual(result['labels'].tolist(),['car','fire hose cabinet'])
        self.assertEqual(calls.count(('image','RGB')),1)
        self.assertEqual(calls.count(('reset',)),2)

    def test_local_service_client_preserves_rgb_and_mask_score_order(self):
        received=[]
        class Engine:
            metadata={'api':'Sam3Processor/image'}
            def predict(self,image,prompts,confidence):
                received.append((image.copy(),prompts,confidence))
                h,w=image.shape[:2]
                masks=np.zeros((2,h,w),bool);masks[0,:,0]=True;masks[1,:,-1]=True
                return dict(boxes=np.array([[0,0,w,h],[0,0,w,h]],np.float32),
                    scores=np.array([.6,.9],np.float32),masks=masks,labels=np.array(['car','fire hose cabinet']),
                    inference_seconds=np.asarray(.1))
        server=HTTPServer(('127.0.0.1',0),handler_for(Engine()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            ns=dict(io=io,json=json,np=np,urllib=urllib,urlparse=urlparse,http=http,time=time,socket=socket,select=select,
                    torch=SimpleNamespace(from_numpy=lambda x:x),Detection=Detection,
                    TiledVehicleDetector=FakeDetector,canonical_class_name=lambda x:x)
            cls=extract('detections/sam3_client.py','SAM3Detector',ns)
            cfg=SimpleNamespace(sam3_url=f'http://127.0.0.1:{server.server_port}',
                class_prompts=('car','fire hose cabinet'),confidence=.12,max_detections=30)
            client=cls(cfg)
            image=np.zeros((4,6,3),np.uint8);image[:]=[1,2,3]
            detections=client.detect(image)
            self.assertEqual(received[0][0][0,0].tolist(),[3,2,1])
            self.assertEqual(detections[0].class_name,'fire hose cabinet')
            self.assertTrue(client.masks[0][:,-1].all())
            self.assertTrue(client.masks[1][:,0].all())
            cfg.sam3_url='https://example.com'
            with self.assertRaises(ValueError): cls(cfg)
        finally:
            if 'client' in locals():
                client.close()
            server.shutdown();thread.join();server.server_close()

    def test_shared_segment_uses_supplied_masks_without_sam2_inference(self):
        ns=dict(np=np,cv2=cv2,Sequence=list,Detection=Detection,SemanticObservation3D=object,
                semantic_class_color=lambda _: (0,255,0), canonical_class_name=lambda x:x)
        fn=extract('detections/object_detect.py','segment',ns)
        seen=[]
        mapper=SimpleNamespace(update=lambda names,scores,obs,**kw:seen.append(names) or [None],
                               confirmed_instance_id=lambda *a:None)
        nav=SimpleNamespace(mapper=mapper,config=SimpleNamespace(show_instance_ids=False),
                            _is_semantic_landmark=lambda _:False)
        image=np.zeros((4,6,3),np.uint8)
        rendered,count=fn(nav,image,[Detection((0,0,6,4),.9,'car')],timestamp=1.,masks=[np.ones((4,6),bool)])
        self.assertEqual(count,1)
        self.assertEqual(seen,[['car']])
        self.assertGreater(rendered.sum(),0)
        with self.assertRaises(ValueError):
            fn(nav,image,[Detection((0,0,6,4),.9,'car')],timestamp=1.,masks=[])

    def test_original_sam2_box_prompt_call_is_preserved(self):
        calls=[]
        class Base:
            def __init__(self, config): self.config=config
        class Model:
            def __init__(self, weights): calls.append(weights)
            def predict(self,image,**kwargs):
                calls.append(kwargs)
                return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(np.ones((1,4,6)))))]
        cls=extract('detections/object_detect.py','SAM2Segmenter',
                    dict(SemanticMaskProcessor=Base,Path=Path,SAM=Model))
        cfg=SimpleNamespace(sam_weights='sam2.pt',tile_size=640,device=0,prediction_quantization=16)
        segmenter=cls(cfg)
        result=segmenter._predict_masks(np.zeros((4,6,3)),[(2,Detection((1,1,5,3),.8,'car'))])
        self.assertEqual(result.shape,(1,4,6))
        self.assertEqual(calls[0],'sam2.pt')
        self.assertEqual(calls[1]['bboxes'],[[1,1,5,3]])
        self.assertEqual(calls[1]['quantize'],16)

    def test_service_startup_explains_environment_and_checkpoint(self):
        with patch('tools.sam3_service.sys.version_info', (3,9)):
            with self.assertRaisesRegex(RuntimeError, 'Python >=3.10'):
                check_environment(None)
        with patch('tools.sam3_service.sys.version_info', (3,12)):
            with patch('tools.sam3_service.importlib.util.find_spec', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'Missing torch'):
                    check_environment(None)
            with patch('tools.sam3_service.importlib.util.find_spec', return_value=object()):
                with patch('tools.sam3_service.Path.is_file', return_value=False):
                    with self.assertRaisesRegex(RuntimeError, 'checkpoint does not exist'):
                        check_environment('missing.pt')

    def test_launch_defaults_use_startup_file(self):
        from navigation.target_navigation_config import parse_config
        _,args=parse_config([],include_perception=True)
        self.assertEqual(args.perception_backend,'sam3')
        _,args=parse_config(['--sam2'],include_perception=True)
        self.assertEqual(args.perception_backend,'yolo_sam2')
        _,args=parse_config(['--sam3','--object-prompts','car','fire hose cabinet'],include_perception=True)
        self.assertEqual(args.perception_backend,'sam3')
        self.assertEqual(args.object_prompts,['car','fire hose cabinet'])
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            parse_config(['--sam2','--sam3'],include_perception=True)
        self.assertEqual(error.exception.code,2)
