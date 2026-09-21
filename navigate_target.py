"""Terminal 2: read a target JSON and submit it to the active exploration session."""

import argparse
import json
from pathlib import Path
import time
import webbrowser

from config.semantic_navigation_config import DEFAULT_TARGET_FILE
from navigation.target import load_target_request, SemanticTargetRequest
from navigation.session_server import send_command

DEFAULT_SESSION_FILE = (
    Path(__file__).resolve().parent / "data" / "exploration_session.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_FILE)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--class-name", help="本次按类别选择，例如 vehicle 或 column；覆盖 target 文件"
    )
    selector.add_argument(
        "--instance-id",
        help="本次按 semantic_graph.json 中的对象 ID 选择；覆盖 target 文件",
    )
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION_FILE)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true")
    action.add_argument("--shutdown", action="store_true")
    parser.add_argument("--open-map", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=900)
    args = parser.parse_args()
    if not 0 < args.wait_timeout < float("inf"):
        parser.error("--wait-timeout must be finite and positive")
    descriptor = json.loads(args.session.read_text(encoding="utf-8"))
    if args.status or args.shutdown:
        response = send_command(descriptor, "status" if args.status else "shutdown")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        if args.open_map and response.get("map_path"):
            webbrowser.open(Path(response["map_path"]).as_uri())
        return
    if args.class_name:
        target = SemanticTargetRequest(class_name=args.class_name)
    elif args.instance_id:
        target = SemanticTargetRequest(instance_id=args.instance_id)
    else:
        target = load_target_request(args.target)
    if target is None:
        parser.error(
            f"尚未指定目标：{args.target} 中 target 为 null。"
            '将其改为 "target": {"class_name": "vehicle"}，'
            "或运行 python navigate_target.py --class-name vehicle --open-map；"
            "指定对象可用 --instance-id vehicle_01（以本次 semantic_graph.json 为准）。"
        )
    response = send_command(descriptor, "navigate", target)
    job_id = response["job_id"]
    print(f"任务已提交：{job_id}")
    deadline, opened = time.monotonic() + args.wait_timeout, False
    while time.monotonic() < deadline:
        status = send_command(descriptor, "status", job_id=job_id)
        if status.get("map_path") and not opened:
            print("目标地图：" + status["map_path"])
            if args.open_map:
                webbrowser.open(Path(status["map_path"]).as_uri())
            opened = True
        job = status.get("job")
        if job and job["id"] == job_id and job["state"] in ("succeeded", "failed"):
            print(json.dumps(job["result"], ensure_ascii=False, indent=2))
            if job["state"] == "failed":
                raise SystemExit(1)
            return
        time.sleep(0.5)
    raise TimeoutError(
        "Client wait timed out; the control terminal continues the accepted task. Use --status."
    )


if __name__ == "__main__":
    main()
