"""自动连接运行中的 WildOS 控制端；无运行中会话时加载保存地图导航。"""
import argparse
import json
import math
from pathlib import Path
import time
import webbrowser
from navigation.console_output import navigation_result_text

DEFAULT_SESSION_FILE = Path(__file__).resolve().parent/'data/wildos_exploration_session.json'


def command(descriptor, name, target=None, job_id=None):
    from navigation.session_server import send_command
    try:
        return send_command(descriptor, name, target, job_id)
    except (ConnectionError, OSError) as error:
        raise SystemExit('无法连接 WildOS 控制终端。请确认探索或独立导航控制端仍运行，'
                         '并确认 --session 指向本次会话；客户端不会另起飞控进程。'
                         '若控制端已退出，可直接运行 python navigate_wildos_target.py，加载保存地图启动导航。'
                         '若任务已提交，连接中断不代表飞行已停止。\n'+str(error)) from None


def require_wildos(data):
    if not isinstance(data, dict) or data.get('navigation_backend') != 'wildos_graph_dijkstra':
        raise SystemExit('该会话不是 WildOS 图导航会话，请运行新版 run_wildos_explore.py。')


def active_default_session():
    """Probe only status; ambiguous failures must not start another controller."""
    from navigation.session_server import send_command
    try:
        descriptor = json.loads(DEFAULT_SESSION_FILE.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    require_wildos(descriptor)
    try:
        status = send_command(descriptor, 'status')
    except ConnectionRefusedError:
        # An old descriptor remains after a completed controller exits.
        return None
    require_wildos(status)
    if not descriptor.get('session_id') or status.get('session_id') != descriptor['session_id']:
        raise ValueError('控制端返回的会话 ID 与本地会话记录不一致')
    return descriptor, status


def open_map(status):
    url=status.get('live_map_url')
    if not url and status.get('map_path'):
        url=Path(status['map_path']).resolve().as_uri()
    if url:
        print('地图：'+url)
        webbrowser.open(url,new=2)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    selector=parser.add_mutually_exclusive_group()
    selector.add_argument('--instance-id', help='本次探索最终车辆编号，例如 vehicle_02')
    selector.add_argument('--class-name', help='例如 vehicle，按保存的目标选择规则匹配')
    selector.add_argument('--target',type=Path,help='目标 JSON 文件；不指定目标参数时进入终端交互')
    parser.add_argument('--session',type=Path,help='现有控制会话文件（使用客户端模式）')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--standalone',action='store_true',help='明确要求独立加载已保存图导航')
    modes.add_argument('--client',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--map',type=Path,help='保存的 WildOS run 或 exploration 目录；默认最新有效地图')
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'config/explore.json',
                        help='独立模式感知与连接配置，与 navigate_target 相同')
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--status',action='store_true')
    actions.add_argument('--return-home',action='store_true')
    actions.add_argument('--shutdown',action='store_true')
    parser.add_argument('--open-map',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--wait-timeout',type=float,default=900.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.wait_timeout) or args.wait_timeout<=0:
        parser.error('--wait-timeout 必须为有限正数')
    if (args.status or args.return_home or args.shutdown) and (args.instance_id or args.class_name or args.target):
        parser.error('状态/返航/关闭命令不能同时指定目标')
    command_only = args.status or args.return_home or args.shutdown
    if args.standalone and (command_only or args.session is not None):
        parser.error('--standalone 不能与控制命令或 --session 同时使用')
    standalone = args.standalone or not (args.client or command_only or args.session is not None)
    if args.map is not None and not standalone:
        parser.error('--map 仅用于独立导航模式')
    automatic = not (args.standalone or args.client or command_only or
                     args.session is not None or args.map is not None)
    target=None
    interactive = not (args.status or args.return_home or args.shutdown or
                       args.instance_id or args.class_name or args.target)
    if not interactive and not (args.status or args.return_home or args.shutdown):
        from navigation.target import SemanticTargetRequest, load_target_request
        if args.instance_id:
            target=SemanticTargetRequest(instance_id=args.instance_id)
        elif args.class_name:
            target=SemanticTargetRequest(class_name=args.class_name)
        else:
            target=load_target_request(args.target)
        if target is None:
            parser.error('请指定 --instance-id vehicle_02、--class-name vehicle 或有效的 --target 文件')
    descriptor = status = None
    if automatic:
        try:
            active = active_default_session()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            parser.error(f'无法确认已有 WildOS 控制端状态，请检查控制终端或会话文件：{error}')
        if active is not None:
            descriptor, status = active
            standalone = False
    if standalone:
        from exploration_wildos.standalone_navigation import (
            run_standalone, latest_graph_archive, SavedMapNotFound, NavigationPortInUse)
        try:
            directory = args.map or latest_graph_archive()
            print(f'--- WildOS 保存地图：{directory.resolve()} ----')
            result = run_standalone(directory,target,config_path=args.config,open_map=args.open_map)
        except (SavedMapNotFound, NavigationPortInUse, ValueError) as error:
            parser.error(str(error))
        if not result['success']:
            raise SystemExit(1)
        return
    if descriptor is None:
        args.session = args.session or DEFAULT_SESSION_FILE
        try:
            descriptor=json.loads(args.session.read_text(encoding='utf-8'))
        except (OSError,ValueError) as error:
            parser.error(f'无法读取 WildOS 会话：{args.session}；无运行中控制端时可直接运行本程序加载保存地图。{error}')
        require_wildos(descriptor)
        status=command(descriptor,'status')
        require_wildos(status)
    if args.open_map:
        open_map(status)
    if interactive:
        from exploration_wildos.terminal_client import run_terminal
        run_terminal(descriptor, command, args.wait_timeout)
        return
    if args.status:
        print(json.dumps(status,ensure_ascii=False,indent=2));return
    action='shutdown' if args.shutdown else 'return_home' if args.return_home else 'navigate'
    response=command(descriptor,action,target)
    job=response['job_id']
    if args.shutdown:
        print('关闭请求已接受；请在控制终端确认返航和退出。');return
    deadline=time.monotonic()+args.wait_timeout
    while time.monotonic()<deadline:
        status=command(descriptor,'status',job_id=job)
        current=status.get('job')
        if current and current['id']==job and current['state'] in ('succeeded','failed'):
            print(navigation_result_text(current['result'], getattr(target,'instance_id',None), command=action))
            if current['state']=='failed':
                raise SystemExit(1)
            return
        time.sleep(.5)
    raise SystemExit('等待超时，控制端任务可能仍在执行；请用 --status 查看，勿重复提交。')


if __name__=='__main__':
    main()
