"""等待 UE 连接已有 WebSocket 服务，然后发送 JSON 请求。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config import WebSocketClientConfig


class WebSocketRequestSender:
    """只作为 WS 客户端工作，不启动或关闭 Node.js 与 UE。"""

    STATUS_REQUEST = {"type": "get_connection_status"}

    def __init__(self, config: WebSocketClientConfig | None = None) -> None:
        self.config = config or WebSocketClientConfig()
        if self.config.ue_wait_timeout <= 0.0:
            raise ValueError("ue_wait_timeout must be greater than zero.")
        if self.config.status_poll_interval <= 0.0:
            raise ValueError("status_poll_interval must be greater than zero.")

    def send_when_ue_ready(self) -> dict[str, Any]:
        """确认 UE 已连接后，发送请求文件并返回服务端确认。"""
        request_path, request = self._load_request(self.config.request_file)
        try:
            from websockets.sync.client import connect
        except ImportError as error:
            raise RuntimeError(
                "Python package 'websockets' is required. Install it with: "
                "python -m pip install websockets"
            ) from error

        uri = (
            f"ws://{self.config.host}:{int(self.config.port)}"
            "/?role=controller"
        )
        try:
            websocket_context = connect(
                uri,
                open_timeout=float(self.config.connect_timeout),
                close_timeout=float(self.config.command_timeout),
            )
        except (ConnectionError, OSError, TimeoutError) as error:
            raise RuntimeError(
                "Cannot connect to the external WebSocket server at "
                f"{uri}. Start it first in another terminal with: npm start"
            ) from error

        with websocket_context as websocket:
            print(f"--- Connected to external WebSocket server: {uri} ----")
            self._wait_for_ue(websocket)
            websocket.send(self._encode(request))
            acknowledgment = self._receive_object(websocket)

        self._validate_acknowledgment(request, acknowledgment)
        print(
            f"--- WebSocket request sent after UE connection: "
            f"{request_path.name}, type={request['type']!r}, "
            f"receivers={int(acknowledgment['receivers'])} ----"
        )
        return acknowledgment

    def _wait_for_ue(self, websocket) -> None:
        deadline = time.monotonic() + float(self.config.ue_wait_timeout)
        waiting_message_printed = False
        while True:
            websocket.send(self._encode(self.STATUS_REQUEST))
            status = self._receive_object(websocket)
            if (
                status.get("type") != self.STATUS_REQUEST["type"]
                or status.get("success") is not True
            ):
                raise RuntimeError(
                    f"WebSocket server returned invalid status: {status!r}"
                )
            ue_clients = int(status.get("ue_clients", 0))
            if ue_clients > 0:
                print(f"--- UE WebSocket connected: clients={ue_clients} ----")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "UE did not connect to the WebSocket server within "
                    f"{self.config.ue_wait_timeout:.1f} s. Start UE after WS "
                    "and verify that UE connects to port "
                    f"{int(self.config.port)}."
                )
            if not waiting_message_printed:
                print("--- Waiting for UE to connect to WebSocket... ----")
                waiting_message_printed = True
            time.sleep(float(self.config.status_poll_interval))

    def _receive_object(self, websocket) -> dict[str, Any]:
        response = websocket.recv(timeout=float(self.config.command_timeout))
        try:
            value = json.loads(response)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"WebSocket server returned invalid JSON: {response!r}"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(
                f"WebSocket server response must be a JSON object: {value!r}"
            )
        return value

    @staticmethod
    def _encode(value: dict[str, Any]) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _load_request(
        request_file: str | Path,
    ) -> tuple[Path, dict[str, Any]]:
        request_path = Path(request_file).resolve()
        if not request_path.is_file():
            raise FileNotFoundError(
                f"WebSocket request file not found: {request_path}"
            )
        try:
            request = json.loads(request_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid JSON in WebSocket request file "
                f"{request_path} at line {error.lineno}, column {error.colno}: "
                f"{error.msg}"
            ) from error
        if not isinstance(request, dict):
            raise ValueError("WebSocket request JSON root must be an object.")

        request_type = request.get("type")
        if not isinstance(request_type, str) or not request_type.strip():
            raise ValueError(
                "WebSocket request JSON requires a non-empty string 'type'."
            )
        if request_type == "set_scene_mode":
            mode = request.get("mode")
            if not isinstance(mode, str) or not mode.strip():
                raise ValueError(
                    "set_scene_mode request requires a non-empty string 'mode'."
                )
        return request_path, request

    @staticmethod
    def _validate_acknowledgment(
        request: dict[str, Any],
        acknowledgment: dict[str, Any],
    ) -> None:
        request_type = request["type"]
        if (
            acknowledgment.get("type") != request_type
            or acknowledgment.get("success") is not True
        ):
            raise RuntimeError(
                f"WebSocket server rejected {request_type}: "
                f"{acknowledgment!r}"
            )
        if int(acknowledgment.get("receivers", 0)) < 1:
            raise RuntimeError(
                "WebSocket server did not deliver the request to any UE client."
            )
        if (
            request_type == "set_scene_mode"
            and acknowledgment.get("mode") != request["mode"].strip()
        ):
            raise RuntimeError(
                "WebSocket server acknowledged a different scene mode: "
                f"{acknowledgment!r}"
            )
