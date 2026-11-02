"""Submit a target to a live session, or navigate independently using a saved map."""

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


def session_command(descriptor, command, target=None, job_id=None):
    """Report a stopped control server without starting another flight process."""
    try:
        return send_command(descriptor, command, target, job_id)
    except ConnectionRefusedError:
        raise SystemExit(
            f"无法连接探索会话服务 127.0.0.1:{descriptor['port']}。\n"
            "探索终端可能已经退出，或会话文件记录的是旧端口。\n"
            "此端口不是 server.js 的 WebSocket 端口；仅启动 npm start 不会恢复探索会话。\n"
            "若探索终端仍在运行，请检查其中的报错及当前会话端口；不要重复启动控制进程。\n"
            "若旧进程已退出且已保存地图，可使用 --standalone --map <exploration目录> "
            "直接导航（同场景、同起点）；也可重新探索后保持终端运行。\n"
            "地图文件存在不代表控制服务正在运行。若此前已有导航任务被接受，"
            "本错误也不代表飞机已停止，请查看控制终端。"
        ) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="目标 JSON；普通客户端默认读取 config/semantic_target.json")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--class-name", help="本次按类别选择，例如 vehicle 或 column；覆盖 target 文件"
    )
    selector.add_argument(
        "--instance-id",
        help="本次按 semantic_graph.json 中的对象 ID 选择；覆盖 target 文件",
    )
    parser.add_argument("--session", type=Path, help="已有会话描述文件；指定后使用客户端模式")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--standalone", action="store_true", help="独立交互导航（默认模式）")
    mode.add_argument("--client", action="store_true", help="向已运行的控制终端发送命令")
    parser.add_argument("--map", type=Path, help="保存的 exploration 目录或 run 目录")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "explore.json",
                        help="独立模式复用感知配置及控制端口；FAST-LIO 不接管导航")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true")
    action.add_argument("--shutdown", action="store_true")
    action.add_argument("--return-home", action="store_true", help="结束当前悬停，返回起飞点并降落")
    parser.add_argument("--open-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-timeout", type=float, default=900)
    args = parser.parse_args()
    if not 0 < args.wait_timeout < float("inf"):
        parser.error("--wait-timeout must be finite and positive")
    command_only = args.status or args.shutdown or args.return_home
    if args.standalone and command_only:
        parser.error("--standalone 不能与 --status/--shutdown/--return-home 一起使用")
    args.standalone = args.standalone or not (args.client or command_only or args.session is not None)
    args.session = args.session or DEFAULT_SESSION_FILE
    if args.map is not None and not args.standalone:
        parser.error("--map 需配合 --standalone")
    if args.status or args.shutdown:
        descriptor = json.loads(args.session.read_text(encoding="utf-8"))
        response = session_command(descriptor, "status" if args.status else "shutdown")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        if args.open_map and response.get("map_path"):
            webbrowser.open(Path(response["map_path"]).as_uri())
        return
    if args.return_home:
        target = None
    elif args.class_name:
        target = SemanticTargetRequest(class_name=args.class_name)
    elif args.instance_id:
        target = SemanticTargetRequest(instance_id=args.instance_id)
    elif args.standalone and args.target is None:
        target = None
    else:
        target = load_target_request(args.target or DEFAULT_TARGET_FILE)
    if target is None and not args.return_home and not args.standalone:
        parser.error(
            f"尚未指定目标：{args.target or DEFAULT_TARGET_FILE} 中 target 为 null。"
            '将其改为 "target": {"class_name": "vehicle"}，'
            "或运行 python navigate_target.py --class-name vehicle --open-map；"
            "指定对象可用 --instance-id vehicle_01（以本次 semantic_graph.json 为准）。"
        )
    if args.standalone:
        from navigation.standalone_navigation import run_standalone, latest_archive
        if args.map is None:
            try:
                args.map = latest_archive()
            except FileNotFoundError as error:
                parser.error(str(error))
        print(f"--- 导航地图：{args.map.resolve()} ----")
        result = run_standalone(args.map, target, config_path=args.config, open_map=args.open_map)
        if not result["success"]:
            raise SystemExit(1)
        return
    descriptor = json.loads(args.session.read_text(encoding="utf-8"))
    response = session_command(descriptor, "return_home" if args.return_home else "navigate", target)
    job_id = response["job_id"]
    print(f"任务已提交：{job_id}")
    deadline, opened = time.monotonic() + args.wait_timeout, False
    while time.monotonic() < deadline:
        status = session_command(descriptor, "status", job_id=job_id)
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
