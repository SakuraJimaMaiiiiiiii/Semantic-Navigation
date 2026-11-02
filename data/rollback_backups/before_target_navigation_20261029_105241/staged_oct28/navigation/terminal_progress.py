"""Compact single-line activity bar; exploration has no known total distance."""
import sys
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

    def update(self, status, *, moving=True):
        changed = status != self.last_status or moving != self.last_moving
        now = self.clock()
        if not changed and (not self.interactive or now-self.last_update < .25):
            return
        self.last_update, self.last_status, self.last_moving = now, status, moving
        label = {'route_clear':'A*跟随', 'ego_clear':'EGO避障', 'ego_avoid':'EGO避障',
                 'position_waypoint':'航点跟随', 'holding_for_stable_clearance':'等待通路稳定',
                 'braking_without_pullback':'减速停车'}.get(status, '等待可行路径')
        if self.interactive:
            # A short bar avoids wrapping in narrow Windows terminals. No
            # distance-based percentage: rolling waypoints change continuously.
            width, block = 20, 4
            offset = self.frame % (width-block+1)
            bar = ('-'*offset + '='*block + '-'*(width-block-offset)
                   if moving else '-'*width)
            if moving:
                self.frame += 1
            line = f'[{bar}] {label if moving else "暂停"}{self.coverage_text}'
            columns = sum(2 if unicodedata.east_asian_width(c) in {'W', 'F'} else 1 for c in line)
            self.stream.write('\r'+line+' '*max(0, self.line_columns-columns)+'\r')
            self.line_columns = columns
            self.stream.flush()
            self.active = True
        elif changed:
            self.stream.write(f'--- 飞行状态：{label} ({status}) ----\n')
            self.stream.flush()

    def update_coverage(self, metrics):
        stage = {'collecting':'采集', 'expanding':'扩展', 'review':'复查', 'review_due':'补充复查',
                 'return_required':'返航'}.get(metrics['stage'], metrics['stage'])
        self.coverage_text = (f" | {stage} {metrics['normalized_window_gain_m2']:.1f}/"
                              f"{metrics['low_gain_threshold_m2']:.1f}m²")
        remaining = metrics['review_remaining_seconds']
        if remaining is not None:
            self.coverage_text += f" {remaining:.0f}s"
        if self.interactive:
            self.update(self.last_status or 'position_waypoint',
                        moving=True if self.last_moving is None else self.last_moving)

    def finish(self):
        if self.active:
            self.stream.write('\n')
            self.stream.flush()
            self.active = False
