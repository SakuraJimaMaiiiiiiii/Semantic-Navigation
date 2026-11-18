"""Own only the SAM3 service started by this exploration session."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


def ensure_checkpoint(root):
    """Download official weights to the model directory, using HF's partial-file handling."""
    directory = root / 'detections' / 'Yolomode'
    checkpoint = directory / 'sam3.pt'
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return checkpoint
    print('--- 未找到 SAM3 权重，正在下载 facebook/sam3 到 detections/Yolomode ----', flush=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', local_dir=str(directory))
    except Exception as error:
        raise RuntimeError(
            'SAM3 权重下载失败。请在 https://huggingface.co/facebook/sam3 申请并获得访问权限，'
            '然后在当前环境执行 hf auth login，再重新运行 python run_explore.py --sam3。'
            f'也可手动把官方 sam3.pt 放到 {checkpoint}。错误类型：{type(error).__name__}'
        ) from error
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f'SAM3 download did not produce a checkpoint: {checkpoint}')
    return checkpoint


class SAM3ServiceProcess:
    def __init__(self, url):
        self.url = url.rstrip('/')
        parsed = urlparse(self.url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1'}
                or parsed.path or parsed.query or parsed.fragment or parsed.username):
            raise ValueError('Automatic SAM3 startup requires http://127.0.0.1:PORT')
        self.port = parsed.port or 80
        self.process = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def ready(self):
        try:
            with self.opener.open(self.url + '/health', timeout=1.) as response:
                metadata = json.load(response)
        except urllib.error.HTTPError:
            raise RuntimeError(f'SAM3 port is occupied by an incompatible service: {self.url}')
        except (urllib.error.URLError, TimeoutError):
            return False
        if metadata.get('api') != 'Sam3Processor/image':
            raise RuntimeError(f'Unexpected SAM3 API at {self.url}')
        return True

    def start(self):
        if self.ready():
            print('--- 复用已启动的 SAM3 服务 ----', flush=True)
            return
        root = Path(__file__).resolve().parents[1]
        # Download before the model startup timer; large weights may take minutes.
        checkpoint = ensure_checkpoint(root)
        print(f'--- SAM3 使用本地权重：{checkpoint} ----', flush=True)
        print('--- 正在自动启动 SAM3，等待模型加载（默认不预热） ----', flush=True)
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(root / 'tools' / 'sam3_service.py'),
             '--checkpoint', str(checkpoint),
             '--port', str(self.port)], cwd=str(root))
        try:
            deadline = time.monotonic() + 180.
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('SAM3 startup failed; see service error above')
                if self.ready():
                    return
                time.sleep(.25)
            raise TimeoutError('SAM3 model startup timed out after 180 seconds')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10.)
        self.process = None
