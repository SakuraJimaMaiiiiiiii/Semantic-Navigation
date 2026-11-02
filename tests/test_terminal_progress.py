import io
import os
import unicodedata
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from navigation.terminal_progress import MotionProgress
from navigation.mission import SemanticNavigator


class TTY(io.StringIO):
    def isatty(self): return True


class ProgressTests(unittest.TestCase):
    def test_narrow_terminal_never_wraps_progress_or_padding(self):
        stream = TTY()
        progress = MotionProgress(stream)
        with patch('navigation.terminal_progress.shutil.get_terminal_size', return_value=os.terminal_size((42, 24))):
            progress.coverage_text = ' | 补充复查 6.0/11.7m²'
            progress.update('route_clear')
            progress.update('fastlio_save', detail='待处理 7167 | 雷达 5995')
            progress.finish()
        for line in stream.getvalue().split('\r'):
            columns = sum(2 if unicodedata.east_asian_width(c) in {'W', 'F'} else 1 for c in line)
            self.assertLessEqual(columns, 41)
        self.assertNotIn('\n', stream.getvalue())

    def test_waypoints_refresh_same_line_at_bounded_rate(self):
        stream=TTY()
        clock=[0.]
        progress=MotionProgress(stream,clock=lambda:clock[0])
        for i in range(100):
            clock[0]=i*.01
            progress.update('route_clear')
        self.assertEqual(stream.getvalue().count('\n'),0)
        self.assertEqual(stream.getvalue().count('A*跟随'),4)
        self.assertNotIn('航点距离', stream.getvalue())
        self.assertNotIn('限速', stream.getvalue())
        self.assertIn('[====----------------]', stream.getvalue())
        progress.finish()
        progress.finish()
        self.assertEqual(stream.getvalue().count('\n'),0)

    def test_redirected_output_only_reports_state_changes(self):
        stream=io.StringIO()
        progress=MotionProgress(stream)
        for i in range(100): progress.update('route_clear')
        progress.update('ego_avoid')
        self.assertEqual(len(stream.getvalue().splitlines()),1)
        self.assertNotIn('\r',stream.getvalue())

    def test_stop_is_not_throttled(self):
        stream=TTY()
        progress=MotionProgress(stream,clock=lambda:0.)
        progress.update('route_clear')
        progress.update('braking_without_pullback',moving=False)
        self.assertIn('[--------------------] 减速停车',stream.getvalue())

    def test_saving_and_return_status_replace_same_line(self):
        stream=TTY()
        clock=[0.]
        progress=MotionProgress(stream,clock=lambda:clock[0])
        progress.update('return_route',detail='3 个航点')
        clock[0]=1.
        progress.update('fastlio_save',detail='待处理 7167 | IMU 213181 | 雷达 5995')
        clock[0]=2.
        progress.update('route_clear')
        self.assertIn('返航路线已生成 | 3 个航点',stream.getvalue())
        self.assertIn('保存 FAST-LIO | 待处理 7167',stream.getvalue())
        self.assertEqual(stream.getvalue().count('\n'),0)

    def test_transient_events_update_bar_and_remain_in_report(self):
        stream=TTY()
        progress=MotionProgress(stream,clock=lambda:0.)
        navigator=SimpleNamespace(vehicle=SimpleNamespace(_motion_progress_display=progress),
                                  events=[], clock=lambda:2., started=0., state=None)
        with patch('builtins.print') as output:
            SemanticNavigator._event(navigator, 'MAP_UPDATE_WAIT', reason='semantic_global_map_stale')
            SemanticNavigator._event(navigator, 'MAP_UPDATE_RESUMED')
            SemanticNavigator._event(navigator, 'PERCEPTION_WAIT', frame_age_seconds=5.)
            SemanticNavigator._event(navigator, 'PERCEPTION_RECOVERED', waited_seconds=2.)
        output.assert_not_called()
        self.assertEqual([event['state'] for event in navigator.events],
                         ['MAP_UPDATE_WAIT','MAP_UPDATE_RESUMED','PERCEPTION_WAIT','PERCEPTION_RECOVERED'])
        self.assertIn('等待地图更新 | 全局地图',stream.getvalue())
        self.assertIn('感知更新已恢复',stream.getvalue())
        self.assertEqual(stream.getvalue().count('\n'),0)

    def test_coverage_updates_same_line_with_stage_and_remaining_time(self):
        stream=TTY()
        clock=[0.]
        progress=MotionProgress(stream,clock=lambda:clock[0])
        metrics=dict(stage='collecting', normalized_window_gain_m2=293.89,
                     low_gain_threshold_m2=1.84, review_remaining_seconds=None)
        progress.update_coverage(metrics)
        clock[0]=1.
        progress.update_coverage(dict(metrics, stage='review', review_remaining_seconds=45.))
        self.assertIn('采集 293.9/1.8m²',stream.getvalue())
        self.assertIn('复查 293.9/1.8m² 45s',stream.getvalue())
        self.assertEqual(stream.getvalue().count('\n'),0)


if __name__ == '__main__': unittest.main()
