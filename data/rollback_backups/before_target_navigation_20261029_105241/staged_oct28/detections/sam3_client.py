"""SAM3 process boundary; bypass system proxies for the local model service."""
import io
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

import numpy as np
import torch
from .object_detect import Detection, TiledVehicleDetector, canonical_class_name


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

    def detect(self, image):
        self.masks = []
        payload = io.BytesIO()
        np.savez_compressed(payload, rgb=np.ascontiguousarray(image[:,:,::-1]),
                            prompts=np.asarray(self.config.class_prompts, dtype='U128'),
                            confidence=np.asarray(self.config.confidence))
        request = urllib.request.Request(self.url+'/predict', data=payload.getvalue(),
                                         headers={'Content-Type':'application/octet-stream'})
        with self.opener.open(request, timeout=8.) as response:
            body = response.read()
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
