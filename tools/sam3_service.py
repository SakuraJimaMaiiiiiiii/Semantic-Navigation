"""Official SAM3 image-only API, compatible with the existing navigation environment."""
import argparse
from contextlib import nullcontext
import io
import importlib.util
import sys
from pathlib import Path
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


def check_environment(checkpoint):
    """Fail before model loading with instructions for the service environment."""
    executable = sys.executable
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Image compatibility backend requires Python >=3.10: {executable}")
    source = Path(__file__).resolve().parents[1] / '.tools' / 'sam3'
    if (source / 'image_only_compat.json').is_file():
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    for package in ("torch", "sam3"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(
                f"Missing {package} in {executable}. Run python tools/setup_sam3_image.py "
                "in the existing navigation environment. See readme/SAM3可选后端与对比.md")
    if checkpoint is not None and not Path(checkpoint).expanduser().is_file():
        raise RuntimeError(
            f"SAM3 checkpoint does not exist: {checkpoint}. Supply the actual downloaded sam3.pt; "
            "place it in detections/Yolomode/sam3.pt or specify --checkpoint.")


class OfficialSAM3:
    def __init__(self, checkpoint, device):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.torch = torch
        self.device = device
        model = build_sam3_image_model(checkpoint_path=checkpoint, device=device,
                                      load_from_HF=checkpoint is None)
        self.processor = Sam3Processor(model, device=device)
        self.metadata = dict(backend='sam3', api='Sam3Processor/image',
                             checkpoint=checkpoint or 'facebook/sam3', device=device,
                             torch_version=torch.__version__,
                             precision='bf16_autocast' if device == 'cuda' else 'float32',
                             compatibility='pinned_image_only_cpu_mask_nms')

    def predict(self, image, prompts, confidence):
        started = time.perf_counter()
        height, width = image.shape[:2]
        boxes, scores, labels, masks = [], [], [], []
        # PIL RGB avoids the official NumPy input's CHW shape convention.
        precision = (self.torch.autocast('cuda', dtype=self.torch.bfloat16)
                     if self.device == 'cuda' else nullcontext())
        with self.torch.inference_mode(), precision:
            self.processor.set_confidence_threshold(confidence)
            state = self.processor.set_image(Image.fromarray(image, mode='RGB'))
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(prompt=prompt, state=state)
                b = result['boxes'].detach().float().cpu().numpy()
                s = result['scores'].detach().float().cpu().numpy()
                m = result['masks'].detach().cpu().numpy().reshape(-1, height, width)
                boxes.extend(b)
                scores.extend(s)
                masks.extend(m)
                labels.extend([prompt]*len(s))
        return dict(boxes=np.asarray(boxes, dtype=np.float32).reshape(-1,4),
                    scores=np.asarray(scores, dtype=np.float32), labels=np.asarray(labels, dtype='U128'),
                    masks=np.asarray(masks, dtype=bool).reshape(-1,height,width),
                    inference_seconds=np.asarray(time.perf_counter()-started))


class SAM3HTTPServer(ThreadingHTTPServer):
    """Health requests stay responsive; the handler admits only one model job."""
    request_queue_size = 16

    def handle_error(self, request, client_address):
        # A local caller can cancel a timed-out inference or close an idle
        # keep-alive stream. Only transport disconnects are routine here.
        if not isinstance(sys.exc_info()[1], (BrokenPipeError,ConnectionResetError,ConnectionAbortedError)):
            super().handle_error(request,client_address)


def handler_for(engine):
    inference_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(30.)

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                self.close_connection = True

        def reply(self, status, body, content_type, *, close=False):
            if close:
                self.close_connection = True
            try:
                # Exception text belongs in a UTF-8 body, never in the ASCII
                # HTTP status line (Windows socket messages may be Chinese).
                self.send_response(status)
                self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(body)))
                if close:
                    self.send_header('Connection','close')
                if status == 503:
                    self.send_header('Retry-After','1')
                self.end_headers()
                self.wfile.write(body)
                if not close:
                    self.connection.settimeout(30.)
            except OSError:
                # Headers/results may already be partially sent. Never try a
                # second 500 response on a cancelled or broken connection.
                self.close_connection = True

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != '/health':
                self.send_error(404)
                return
            body = json.dumps(engine.metadata).encode()
            self.reply(200,body,'application/json')

        def do_POST(self):
            if self.path != '/predict':
                self.close_connection = True
                self.send_error(404)
                return
            acquired = False
            inference_started = False
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 32*1024*1024:
                    raise ValueError('Invalid request size')
                acquired = inference_lock.acquire(blocking=False)
                if not acquired:
                    # The POST body is unread; never reuse this byte stream.
                    self.reply(503,b'','application/octet-stream',close=True)
                    return
                self.connection.settimeout(8.)
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as data:
                    image = data['rgb']
                    prompts = data['prompts'].tolist()
                    confidence = float(data['confidence'])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError('Expected uint8 HWC RGB')
                if not isinstance(prompts, list) or not 1 <= len(prompts) <= 32 or any(
                        not isinstance(p, str) or not p.strip() or len(p) > 128 for p in prompts):
                    raise ValueError('Invalid text prompts')
                if not 0 <= confidence <= 1:
                    raise ValueError('Invalid confidence')
                inference_started = True
                result = engine.predict(image, prompts, confidence)
                out = io.BytesIO()
                np.savez_compressed(out, **result)
                body = out.getvalue()
                self.reply(200,body,'application/octet-stream')
            except Exception as error:
                if not inference_started and isinstance(error,(BrokenPipeError,ConnectionResetError,ConnectionAbortedError)):
                    self.close_connection = True
                else:
                    body = json.dumps(dict(error=type(error).__name__,message=str(error)),ensure_ascii=False).encode('utf-8')
                    self.reply(500,body,'application/json; charset=utf-8',close=True)
            finally:
                if acquired:
                    inference_lock.release()
    return Handler


def check_image_backend(device):
    import torch
    import ftfy
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.perflib.nms import generic_nms
    from sam3.model.tokenizer_ve import SimpleTokenizer
    source = Path(__file__).resolve().parents[1] / '.tools' / 'sam3'
    tokenizer = SimpleTokenizer(str(source/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
    tokens = tokenizer(['car', 'fire hose cabinet', 'fire extinguisher'])
    ious = torch.tensor([[1.,.9,0.],[.9,1.,0.],[0.,0.,1.]],device=device)
    scores = torch.tensor([.9,.8,.7],device=device)
    keep = generic_nms(ious,scores,.5)
    assert keep.cpu().tolist() == [0,2]
    assert not any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
    print(json.dumps(dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        ftfy=ftfy.__version__, device=device, tokens_shape=list(tokens.shape), nms=keep.cpu().tolist(),
        builder=build_sam3_image_model.__name__, processor=Sam3Processor.__name__,
        triton_imported=False, weights_tested=False),indent=2))


def smoke_image_backend(device):
    """Exercise official model construction/forward, explicitly WITHOUT learned weights."""
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    engine = OfficialSAM3.__new__(OfficialSAM3)
    engine.torch, engine.device = torch, device
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    print('Building untrained official model for compatibility test; no download, no server.', flush=True)
    model = build_sam3_image_model(device=device, load_from_HF=False)
    engine.processor = Sam3Processor(model, device=device)
    result = engine.predict(np.zeros((405,720,3),dtype=np.uint8),
                            ['car','fire hose cabinet','fire extinguisher'], 0.)
    print(json.dumps(dict(smoke_test_passed=True, weights_tested=False,
        inference_seconds=float(result['inference_seconds']), masks_shape=list(result['masks'].shape),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else None,
        triton_imported='triton' in sys.modules),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke-test', action='store_true', help='Untrained model forward test; never serves predictions')
    mode.add_argument('--check-only', action='store_true', help='Validate imports and small GPU ops; no checkpoint needed')
    parser.add_argument('--checkpoint',
                        default=str(Path(__file__).resolve().parents[1] / 'detections' / 'Yolomode' / 'sam3.pt'),
                        help='Local official weights (default: project detections/Yolomode/sam3.pt)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--port', type=int, default=8878)
    parser.add_argument('--warmup', action='store_true',
                        help='Run one blank-image inference before serving (disabled by default)')
    args = parser.parse_args()
    try:
        check_environment(None if args.check_only or args.smoke_test else args.checkpoint)
    except RuntimeError as error:
        parser.error(str(error))
    if args.smoke_test:
        smoke_image_backend(args.device)
        return
    if args.check_only:
        check_image_backend(args.device)
        return
    engine = OfficialSAM3(args.checkpoint, args.device)
    if args.warmup:
        print('Warming up SAM3 before accepting flight requests...', flush=True)
        engine.predict(np.zeros((405,720,3),dtype=np.uint8), ['car'], .5)
    else:
        print('SAM3 warmup disabled; inference starts with the first image.', flush=True)
    server = SAM3HTTPServer(('127.0.0.1', args.port), handler_for(engine))
    print(f'SAM3 ready: http://127.0.0.1:{args.port}; official image API', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
