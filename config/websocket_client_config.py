"""连接外部 Node.js WebSocket 服务的客户端配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class WebSocketClientConfig:
    """等待 UE 连接并发送 JSON 请求所需的参数。"""

    host: str = "127.0.0.1"
    port: int = 8766
    request_file: Path = PROJECT_ROOT / "config" / "websocket_request.json"
    connect_timeout: float = 2.0
    ue_wait_timeout: float = 120.0
    status_poll_interval: float = 0.25
    command_timeout: float = 2.0
