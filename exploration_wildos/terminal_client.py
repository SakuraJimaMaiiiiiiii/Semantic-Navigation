"""Reuse the navigation terminal grammar against the existing graph session."""
import time

from navigation.target import SemanticTargetRequest
from navigation.terminal_commands import TerminalCommands


class RemoteTerminalSession:
    def __init__(self, descriptor, send, wait_timeout):
        self._descriptor, self.send = descriptor, send
        self.wait_timeout = wait_timeout
        self.deadlines = {}

    def descriptor(self):
        return dict(self._descriptor)

    def dispatch(self, message):
        name = message['command']
        target = SemanticTargetRequest(**message['target']) if name == 'navigate' else None
        response = self.send(self._descriptor, name, target)
        if name in {'navigate', 'return_home'}:
            self.deadlines[response['job_id']] = time.monotonic()+self.wait_timeout
        return response

    def status(self, job_id):
        response = self.send(self._descriptor, 'status', job_id=job_id)
        job = response.get('job')
        if job and job.get('id') == job_id and job.get('state') in {'succeeded','failed'}:
            self.deadlines.pop(job_id, None)
        elif time.monotonic() >= self.deadlines.get(job_id, float('inf')):
            raise SystemExit('等待超时，已停止本行剩余目标；控制端任务可能仍在执行。'
                             '请用 --status 查看，勿重复提交。')
        return response


def run_terminal(descriptor, send, wait_timeout):
    terminal = TerminalCommands(RemoteTerminalSession(descriptor, send, wait_timeout))
    print('WildOS 图导航交互模式：物体编号以本次地图为准。')
    from navigation.terminal_commands import HELP
    print(HELP)
    print('关闭本客户端或 Ctrl+C 不会取消控制端任务；需要返航请输入 home。')
    try:
        terminal.run()
    except KeyboardInterrupt:
        print('\n客户端输入已结束；控制端任务状态请用 --status 查看。')
    finally:
        terminal.close()
