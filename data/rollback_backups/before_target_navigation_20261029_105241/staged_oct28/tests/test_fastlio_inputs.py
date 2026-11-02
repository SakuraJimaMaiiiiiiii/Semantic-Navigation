"""Raw capture must not steal MAVLink or continue writing after exploration."""
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import h5py
from localization.fastlio_recording import FastLioRecorder


class InputTests(unittest.TestCase):
    def test_existing_receiver_source_timestamps_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'fastlio'
            connection=SimpleNamespace(message_hooks=[])
            mav=SimpleNamespace(the_connection=connection,SendMavCmdLong=Mock())
            recorder=FastLioRecorder(folder)
            recorder.start(mav)
            hook=connection.message_hooks[0]
            msg=SimpleNamespace(get_type=lambda:'HIGHRES_IMU',time_usec=1234567,
                                fields_updated=63,xacc=0.,yacc=0.,zacc=-9.81,
                                xgyro=0.,ygyro=0.,zgyro=0.,_timestamp=1000.)
            hook(connection,msg)
            frame=SimpleNamespace(timestamp=1000.,points_body=np.ones((15,3)))
            state=SimpleNamespace(px4_boot_timestamp=1.234567,position_xyz=np.zeros(3),
                                  quaternion=np.array([1.,0.,0.,0.]))
            recorder.submit_scan(frame,state,.01)
            recorder.close();recorder.close()
            self.assertEqual(connection.message_hooks,[])
            before=(folder/'inputs.h5').stat().st_size
            hook(connection,msg);recorder.submit_scan(frame,state,.01)
            self.assertEqual((folder/'inputs.h5').stat().st_size,before)
            with h5py.File(folder/'inputs.h5') as f:
                self.assertEqual(f['imu'].shape,(1,10))
                self.assertAlmostEqual(f['imu'][0,0],1.234567)
                self.assertEqual(f.attrs['scan_model'],'unverified')
                self.assertEqual(len(f['scans']),1)
                np.testing.assert_array_equal(f['scans/00000000/points_body'][:],frame.points_body)

    def test_backlog_drains_in_order_and_flushes_final_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'fastlio'
            recorder = FastLioRecorder(folder)
            mav = SimpleNamespace(the_connection=SimpleNamespace(message_hooks=[]),
                                  SendMavCmdLong=Mock())
            release = threading.Event()
            write = recorder._write

            def delayed_write():
                release.wait()
                write()

            recorder._write = delayed_write
            recorder.start(mav)
            try:
                for i in range(4097):
                    recorder._enqueue('imu', [i*.005, 0, 0, -9.81, 0, 0, 0, 1000+i*.005, 63, 0])
                    if i % 200 == 0:
                        recorder._enqueue('scan', (i*.005, 1000+i*.005, .01,
                                                  np.ones((15, 3)), np.zeros(7)))
            finally:
                release.set()
                recorder.close()
            recorder.close()
            self.assertTrue(recorder.queue.empty())
            self.assertIsNone(recorder.thread)
            with h5py.File(folder/'inputs.h5') as f:
                np.testing.assert_allclose(f['imu'][:, 0], np.arange(4097)*.005)
                self.assertEqual(len(f['scans']), 21)
            report = json.loads((folder/'capture_report.json').read_text())
            self.assertTrue(report['completed'])
            self.assertEqual(report['imu'], 4097)
            self.assertIsNone(report['error'])

    def test_close_allows_progress_beyond_old_ten_second_deadline(self):
        recorder = FastLioRecorder('unused')
        clock = [0.]
        worker = Mock()
        worker.is_alive.side_effect = lambda: clock[0] < 24.

        def join(timeout):
            clock[0] += timeout
            recorder._progress += 1

        worker.join.side_effect = join
        recorder.thread = worker
        with patch('localization.fastlio_recording.time.monotonic', side_effect=lambda: clock[0]), \
             patch('builtins.print'):
            recorder.close()
        self.assertGreater(clock[0], 10.)
        self.assertIsNone(recorder.thread)

    def test_close_reports_stalled_writer(self):
        recorder = FastLioRecorder('unused')
        clock = [0.]
        worker = Mock()
        worker.is_alive.return_value = True
        worker.join.side_effect = lambda timeout: clock.__setitem__(0, clock[0]+timeout)
        recorder.thread = worker
        with patch('localization.fastlio_recording.time.monotonic', side_effect=lambda: clock[0]), \
             patch('builtins.print'):
            with self.assertRaisesRegex(TimeoutError, 'queued=0.*no_progress_seconds='):
                recorder.close()
        self.assertIs(recorder.thread, worker)

    def test_shadow_requires_explicit_scan_model(self):
        with self.assertRaisesRegex(ValueError,'verified snapshot'):
            FastLioRecorder('unused',shadow=True)

    def test_cli_uses_startup_file_and_allows_override(self):
        from run_explore import parse_config
        _,args=parse_config([],include_perception=True)
        self.assertEqual(args.fastlio, 'live')
        _,args=parse_config(['--fastlio','record','--sam3'],include_perception=True)
        self.assertEqual(args.fastlio,'record')
        self.assertEqual(args.perception_backend,'sam3')
