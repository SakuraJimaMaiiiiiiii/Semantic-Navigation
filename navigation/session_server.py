"""Loopback command queue: only the main session thread can control the vehicle."""

from dataclasses import asdict
import json
import queue
import secrets
import socket
import socketserver
import threading
import uuid

from .scene_memory import write_json
from .target import SemanticTargetRequest

MAX_MESSAGE = 65536


class SessionServer:
    def __init__(self, session_id, port=8877):
        self.session_id = session_id
        self.token = secrets.token_urlsafe(32)
        self.commands = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.state = "EXPLORING"
        self.job = None
        self.completed_jobs = {}
        self.map_path = None
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(5)
                try:
                    line = self.rfile.readline(MAX_MESSAGE + 1)
                    if len(line) > MAX_MESSAGE:
                        raise ValueError("Command too large")
                    response = owner.dispatch(json.loads(line))
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                self.wfile.write(
                    (json.dumps(response, allow_nan=False) + "\n").encode()
                )

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = False
            daemon_threads = True

        self.server = Server(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
            name="SemanticSessionCommands",
        )
        self.thread.start()

    def descriptor(self):
        return {
            "version": 1,
            "host": "127.0.0.1",
            "port": self.server.server_address[1],
            "session_id": self.session_id,
            "token": self.token,
        }

    def publish_descriptor(self, path):
        write_json(path, self.descriptor())

    def status(self, job_id=None):
        with self._lock:
            return {
                "ok": True,
                "state": self.state,
                "job": self.completed_jobs.get(job_id, self.job),
                "map_path": self.map_path,
                "session_id": self.session_id,
            }

    def dispatch(self, message):
        if (
            not isinstance(message, dict)
            or message.get("session_id") != self.session_id
            or not secrets.compare_digest(str(message.get("token", "")), self.token)
        ):
            raise ValueError("Invalid session credentials")
        command = message.get("command")
        if command == "status":
            return self.status(message.get("job_id"))
        if command not in ("navigate", "shutdown"):
            raise ValueError("Unknown command")
        request = (
            SemanticTargetRequest(**message["target"])
            if command == "navigate"
            else None
        )
        with self._lock:
            if self.state != "READY" and not (
                command == "shutdown" and self.state == "ERROR"
            ):
                raise ValueError(
                    f"Session is {self.state}; commands require READY after landing"
                )
            identity = uuid.uuid4().hex
            self.state = "NAVIGATING" if command == "navigate" else "STOPPING"
            self.job = {
                "id": identity,
                "state": "accepted",
                "target": None if request is None else asdict(request),
            }
            self.commands.put_nowait((command, identity, request))
        return {"ok": True, "job_id": identity}

    def finish(self, identity, result, *, fatal=False):
        with self._lock:
            self.job = {
                "id": identity,
                "state": "succeeded" if result.get("success") else "failed",
                "result": result,
            }
            self.completed_jobs[identity] = self.job
            if len(self.completed_jobs) > 32:
                self.completed_jobs.pop(next(iter(self.completed_jobs)))
            self.state = "ERROR" if fatal else "READY"

    def set_state(self, state):
        with self._lock:
            self.state = state

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def send_command(descriptor, command, target=None, job_id=None):
    if descriptor.get("host") != "127.0.0.1":
        raise ValueError("Only the local control session is supported")
    payload = {
        "session_id": descriptor["session_id"],
        "token": descriptor["token"],
        "command": command,
    }
    if job_id is not None:
        payload["job_id"] = job_id
    if target is not None:
        payload["target"] = asdict(target)
    data = (json.dumps(payload, allow_nan=False) + "\n").encode()
    with socket.create_connection(
        ("127.0.0.1", int(descriptor["port"])), timeout=5
    ) as connection:
        connection.sendall(data)
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_MESSAGE + 1)
            if len(line) > MAX_MESSAGE:
                raise ValueError("Session response too large")
            response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Session command failed"))
    return response
