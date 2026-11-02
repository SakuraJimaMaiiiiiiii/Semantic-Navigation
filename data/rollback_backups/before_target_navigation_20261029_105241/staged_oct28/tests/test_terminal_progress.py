import io
import unittest
from navigation.terminal_progress import MotionProgress


class TTY(io.StringIO):
    def isatty(self): return True


class ProgressTests(unittest.TestCase):
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
        self.assertEqual(stream.getvalue().count('\n'),1)

    def test_redirected_output_only_reports_state_changes(self):
        stream=io.StringIO()
        progress=MotionProgress(stream)
        for i in range(100): progress.update('route_clear')
        progress.update('ego_avoid')
        self.assertEqual(len(stream.getvalue().splitlines()),2)
        self.assertNotIn('\r',stream.getvalue())

    def test_stop_is_not_throttled(self):
        stream=TTY()
        progress=MotionProgress(stream,clock=lambda:0.)
        progress.update('route_clear')
        progress.update('braking_without_pullback',moving=False)
        self.assertIn('[--------------------] 暂停',stream.getvalue())


if __name__ == '__main__': unittest.main()
