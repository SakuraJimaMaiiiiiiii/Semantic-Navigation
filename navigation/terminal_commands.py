"""Terminal input submits commands; only the session main thread flies."""
import json
import re
import threading
from .console_output import navigation_result_text

HELP = ('输入一个或多个物体 ID，空格分隔；可在末尾添加 home，例如 vehicle_03 vehicle_01 home。\n'
        'home / 返航：返回起点并降落；quit / 退出：返航降落并退出；status：状态；help：帮助。')


def parse_line(line):
    words = line.strip().split()
    if not words:
        return []
    commands = {'home':'return_home', 'return':'return_home', '返航':'return_home',
                'quit':'shutdown', 'exit':'shutdown', '退出':'shutdown',
                'status':'status', '状态':'status', 'help':'help', '帮助':'help'}
    if len(words) == 1 and words[0].lower() in commands:
        return [{'command':commands[words[0].lower()]}]
    return_home = words[-1].lower() in {'home', 'return', '返航'}
    if return_home:
        words = words[:-1]
    if any(word.lower() in commands or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', word) for word in words):
        raise ValueError('请输入物体 ID，以空格分隔；home/返航可单独输入或放在末尾，quit 等其他命令需单独输入。')
    result = [{'command':'navigate', 'target':{'instance_id':word}} for word in words]
    if return_home:
        result.append({'command':'return_home'})
    return result


class TerminalCommands:
    def __init__(self, server, reader=input, writer=print):
        self.server, self.reader, self.writer = server, reader, writer
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name='NavigationTerminalInput')

    def start(self):
        self.writer(HELP)
        self.thread.start()

    def close(self):
        # input() may still be blocked. A daemon reader owns no SDK resources,
        # and checks this event before submitting anything after shutdown.
        self.stop.set()

    def run(self):
        while not self.stop.is_set():
            try:
                line = self.reader('\n请输入导航点（多个 ID 空格分隔，可在末尾加 home 返航，quit 退出）：')
            except EOFError:
                return
            if self.stop.is_set():
                return
            try:
                commands = parse_line(line)
                skip_targets = False
                for command in commands:
                    if self.stop.is_set():
                        return
                    if skip_targets and command['command'] == 'navigate':
                        continue
                    if command['command'] == 'help':
                        self.writer(HELP)
                        continue
                    response = self.server.dispatch({**self.server.descriptor(), **command})
                    if command['command'] == 'status':
                        self.writer(json.dumps(response, ensure_ascii=False, indent=2))
                        continue
                    if command['command'] == 'shutdown':
                        self.writer('已请求返航降落并退出，请等待控制终端完成清理。')
                        return
                    identity = response['job_id']
                    failed = False
                    while not self.stop.wait(.2):
                        job = self.server.status(identity).get('job')
                        if job and job['id'] == identity and job['state'] in ('succeeded','failed'):
                            self.writer(navigation_result_text(job['result'],
                                command.get('target', {}).get('instance_id'), command=command['command']))
                            if job['state'] == 'failed':
                                failed = True
                                break
                            break
                    if failed:
                        if command['command'] == 'navigate' and commands[-1]['command'] == 'return_home':
                            # Only a completed failure reaches here. A timeout,
                            # stop or unknown task state never triggers return.
                            skip_targets = True
                        else:
                            break
            except (ValueError, RuntimeError) as error:
                self.writer(str(error))
