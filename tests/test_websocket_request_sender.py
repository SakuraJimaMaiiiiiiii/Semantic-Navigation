"""外部 WebSocket/UE 就绪顺序的回归测试。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from application.websocket_request_sender import WebSocketRequestSender
from config import WebSocketClientConfig


class WebSocketRequestSenderTest(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            WebSocketClientConfig(),
            ue_wait_timeout=0.1,
            status_poll_interval=0.001,
        )

    def _request_file(self, directory: str) -> Path:
        request_file = Path(directory) / "request.json"
        request_file.write_text(
            json.dumps({"type": "set_scene_mode", "mode": "fixed_1"}),
            encoding="utf-8",
        )
        return request_file

    def test_waits_for_ue_before_sending_request_file(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.recv.side_effect = [
            json.dumps({
                "type": "get_connection_status",
                "success": True,
                "ue_clients": 0,
            }),
            json.dumps({
                "type": "get_connection_status",
                "success": True,
                "ue_clients": 1,
            }),
            json.dumps({
                "type": "set_scene_mode",
                "success": True,
                "mode": "fixed_1",
                "receivers": 1,
            }),
        ]

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                request_file=self._request_file(directory),
            )
            sender = WebSocketRequestSender(config)
            with patch(
                "websockets.sync.client.connect",
                return_value=connection,
            ), patch("application.websocket_request_sender.time.sleep"):
                acknowledgment = sender.send_when_ue_ready()

        sent = [json.loads(call.args[0]) for call in connection.send.call_args_list]
        self.assertEqual(sent[0], {"type": "get_connection_status"})
        self.assertEqual(sent[1], {"type": "get_connection_status"})
        self.assertEqual(
            sent[2],
            {"type": "set_scene_mode", "mode": "fixed_1"},
        )
        self.assertEqual(acknowledgment["receivers"], 1)

    def test_reports_external_server_connection_failure(self):
        sender = WebSocketRequestSender(self.config)
        with patch(
            "websockets.sync.client.connect",
            side_effect=ConnectionRefusedError,
        ):
            with self.assertRaisesRegex(RuntimeError, "npm start"):
                sender.send_when_ue_ready()

    def test_rejects_invalid_request_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            request_file = Path(directory) / "request.json"
            request_file.write_text(
                json.dumps({"type": "set_scene_mode"}),
                encoding="utf-8",
            )
            sender = WebSocketRequestSender(
                replace(self.config, request_file=request_file)
            )
            with self.assertRaisesRegex(ValueError, "requires.*'mode'"):
                sender.send_when_ue_ready()

    def test_rejects_delivery_without_ue_receiver(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.recv.side_effect = [
            json.dumps({
                "type": "get_connection_status",
                "success": True,
                "ue_clients": 1,
            }),
            json.dumps({
                "type": "set_scene_mode",
                "success": False,
                "mode": "fixed_1",
                "receivers": 0,
            }),
        ]
        sender = WebSocketRequestSender(self.config)
        with patch(
            "websockets.sync.client.connect",
            return_value=connection,
        ):
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                sender.send_when_ue_ready()


if __name__ == "__main__":
    unittest.main()
