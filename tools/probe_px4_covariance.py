"""Read-only flight-state probe: requests telemetry, never arms or changes mode."""

import json
import math
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pymavlink.dialects.v20 import common as mavlink


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sock.bind(("0.0.0.0", 20101))
    sock.settimeout(0.5)
    parser = mavlink.MAVLink(None, srcSystem=255, srcComponent=190)
    counts = Counter()
    latest = {}
    odometry_samples = []
    target = None
    requested = False
    started = time.monotonic()

    def command(code, p1, p2=0):
        message = parser.command_long_encode(target[0], target[1], code, 0,
                                             p1, p2, 0, 0, 0, 0, 0)
        sock.sendto(message.pack(parser), ("127.0.0.1", 20100))

    try:
        while time.monotonic() - started < 18:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            for msg in parser.parse_buffer(data) or []:
                kind = msg.get_type()
                counts[kind] += 1
                if kind in ("HEARTBEAT", "AUTOPILOT_VERSION", "ODOMETRY",
                            "LOCAL_POSITION_NED_COV", "ESTIMATOR_STATUS", "COMMAND_ACK"):
                    latest[kind] = msg.to_dict()
                    if kind == "ODOMETRY":
                        odometry_samples.append(msg.to_dict())
                if kind == "HEARTBEAT" and msg.autopilot == mavlink.MAV_AUTOPILOT_PX4:
                    target = (msg.get_srcSystem(), msg.get_srcComponent())
                    if not requested:
                        print(f"PX4 target={target}, armed={bool(msg.base_mode & 128)}", flush=True)
                        command(mavlink.MAV_CMD_REQUEST_MESSAGE, 148)
                        for message_id in (331, 64, 230):
                            command(mavlink.MAV_CMD_REQUEST_MESSAGE, message_id)
                        command(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 331, 100000)
                        requested = True
    finally:
        if requested:
            command(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 331, 0)
        sock.close()

    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    version = latest.get("AUTOPILOT_VERSION", {}).get("flight_sw_version")
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts), "latest": latest,
        "odometry_sample_count": len(odometry_samples),
        "firmware_version_bytes": None if version is None else
            [(version >> shift) & 255 for shift in (24, 16, 8, 0)],
        "note": "Estimator-reported variances, not empirical ground-truth error covariance. "
                "Zero attitude covariance in the inspected local firmware is unimplemented, not perfect accuracy.",
    }
    if odometry_samples:
        covariance = odometry_samples[-1]["pose_covariance"]
        diagonal = [covariance[i] for i in (0, 6, 11, 15, 18, 20)]
        summary["last_pose_diagonal"] = diagonal
        summary["last_position_std_m"] = [math.sqrt(v) if math.isfinite(v) and v >= 0
                                          else None for v in diagonal[:3]]
        summary["all_attitude_blocks_zero"] = all(
            all(sample["pose_covariance"][i] == 0 for i in (15, 16, 17, 18, 19, 20))
            for sample in odometry_samples
        )
    report = clean({**summary, "odometry_samples": odometry_samples})
    output_dir = Path(__file__).resolve().parents[1] / "data" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"px4_covariance_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean(summary), indent=2, allow_nan=False))
    print(f"REPORT={output_path}")


if __name__ == "__main__":
    main()
