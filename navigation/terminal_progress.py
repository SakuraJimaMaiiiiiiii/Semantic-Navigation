"""Compact single-line activity bar; exploration has no known total distance."""
import sys
import shutil
import threading
import time
import unicodedata


class MotionProgress:
    def __init__(self, stream=None, clock=time.monotonic):
        self.stream = sys.stdout if stream is None else stream
        self.clock = clock
        self.interactive = bool(getattr(self.stream, 'isatty', lambda: False)())
        self.last_update = -float('inf')
        self.last_status = None
        self.last_moving = None
        self.active = False
        self.frame = 0
        self.coverage_text = ""
        self.line_columns = 0
        self._lock = threading.RLock()
        self._redirected_announced = False

    def update(self, status, *, moving=True, detail=None):
        with self._lock:
            self._update(status, moving=moving, detail=detail)

    def _update(self, status, *, moving=True, detail=None):
        changed = status != self.last_status or moving != self.last_moving
        now = self.clock()
        if not changed and (not self.interactive or now-self.last_update < .25):
            return
        self.last_update, self.last_status, self.last_moving = now, status, moving
        label = {'route_clear':'A*跟随', 'ego_clear':'EGO避障', 'ego_avoid':'EGO避障',
                 'height_recovery':'恢复巡航高度', 'height_recovery_blocked':'高度恢复通路受阻',
                 'position_waypoint':'航点跟随', 'holding_for_stable_clearance':'等待通路稳定',
                 'braking_without_pullback':'减速停车', 'waiting_for_plan':'等待规划',
                 'return_route':'返航路线已生成', 'fastlio_save':'保存 FAST-LIO',
                 'coverage':'探索监测',
                 'map_wait':'等待地图更新', 'map_resumed':'地图更新已恢复',
                 'perception_wait':'等待感知更新', 'perception_resumed':'感知更新已恢复'}.get(status, '等待可行路径')
        if self.interactive:
            # A short bar avoids wrapping in narrow Windows terminals. No
            # distance-based percentage: rolling waypoints change continuously.
            width, block = 20, 4
            offset = self.frame % (width-block+1)
            bar = ('-'*offset + '='*block + '-'*(width-block-offset)
                   if moving else '-'*width)
            if moving:
                self.frame += 1
            line = f'[{bar}] {label}{" | " + detail if detail else ""}{self.coverage_text if status not in {"fastlio_save", "return_route"} else ""}'
            # Leave the last terminal column unused: wrapping moves the cursor
            # to another row, where carriage return cannot erase the old bar.
            limit = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns-1)
            columns = 0
            visible = []
            for character in line:
                width = (0 if unicodedata.combining(character) else
                         2 if unicodedata.east_asian_width(character) in {'W', 'F'} else 1)
                if columns+width > limit:
                    break
                visible.append(character)
                columns += width
            self.stream.write('\r'+''.join(visible)+' '*max(0, min(self.line_columns, limit)-columns)+'\r')
            self.line_columns = columns
            self.stream.flush()
            self.active = True
        elif not self._redirected_announced:
            self.stream.write('--- 任务进行中，详细状态记录于运行报告 ----\n')
            self.stream.flush()
            self._redirected_announced = True

    def update_coverage(self, metrics):
        with self._lock:
            stage = {'collecting':'采集', 'expanding':'扩展', 'review':'复查', 'review_due':'补充复查',
                     'return_required':'返航'}.get(metrics['stage'], metrics['stage'])
            self.coverage_text = (f" | {stage} {metrics['normalized_window_gain_m2']:.1f}/"
                                  f"{metrics['low_gain_threshold_m2']:.1f}m²")
            remaining = metrics['review_remaining_seconds']
            if remaining is not None:
                self.coverage_text += f" {remaining:.0f}s"
            if self.interactive:
                self._update(self.last_status or 'coverage',
                             moving=False if self.last_moving is None else self.last_moving)

    def finish(self):
        with self._lock:
            if self.active:
                limit = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns-1)
                self.stream.write('\r' + ' ' * min(self.line_columns, limit) + '\r')
                self.stream.flush()
                self.active = False
                self.line_columns = 0


_shared_progress = None
_shared_lock = threading.Lock()


def shared_progress():
    global _shared_progress
    with _shared_lock:
        if _shared_progress is None:
            _shared_progress = MotionProgress()
        return _shared_progress
