"""Short calibration flight must close geometrically and record THROUGH landing."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
from run_fastlio_test import (loop_waypoints, parse_args, record_static_start,
                             CaptureCheckedPlanner, fly_and_finish_recording)


class ShortFlightTests(unittest.TestCase):
    def test_rectangle_uses_initial_heading_and_ground_offset(self):
        home=np.array([10.,20.,.4])
        points=loop_waypoints(home,np.pi/2,3.,1.5,1.5)
        np.testing.assert_allclose([p.position_ned for p in points],
                                   [[10,23,-1.1],[8.5,23,-1.1],[8.5,20,-1.1],[10,20,-1.1]])
        self.assertAlmostEqual(points[0].yaw,np.pi/2)
        self.assertAlmostEqual(points[-1].yaw,0.)

    def test_invalid_dimensions_before_sdk_import(self):
        with patch('sys.stderr'):
            for args in [['--length','nan'],['--width','0'],['--height','-1'],['--length','100']]:
                with self.assertRaises(SystemExit):parse_args(args)

    def test_moving_aircraft_rejected_before_arming(self):
        state=SimpleNamespace(position_xyz=np.zeros(3),linear_velocity=np.array([.3,0,0]),
                              angular_velocity=np.zeros(3),timestamp=time.time())
        vehicle=SimpleNamespace(get_position=lambda:np.zeros(3),get_drone_state=lambda:state,
                                mav=SimpleNamespace(isArmed=False),wait=Mock())
        recorder=SimpleNamespace(error=None,thread=SimpleNamespace(is_alive=lambda:True))
        with self.assertRaisesRegex(RuntimeError,'stationary'):
            record_static_start(SimpleNamespace(vehicle=vehicle,fastlio_recorder=recorder))
        vehicle.wait.assert_not_called()

    def test_failed_capture_stops_planning(self):
        planner=Mock()
        wrapped=CaptureCheckedPlanner(SimpleNamespace(fastlio_recorder=SimpleNamespace(error='overflow')),planner)
        with self.assertRaisesRegex(RuntimeError,'overflow'):wrapped.plan(np.zeros(3),np.ones(3))
        planner.plan.assert_not_called()

    def test_landing_confirmed_before_capture_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fastlio').mkdir()
            events=[]
            def close():
                events.append('close')
                (root/'fastlio/capture_report.json').write_text(json.dumps({'completed':True}))
            vehicle=SimpleNamespace(finish_flight_session=lambda:events.append('land_confirmed'),
                wait=lambda seconds:events.append('ground_recording'),get_position=lambda:np.zeros(3),
                landing_confirmation={'confirmed':True})
            runtime=SimpleNamespace(vehicle=vehicle,run_mission=lambda:events.append('mission'),
                run_output_dir=root,fastlio_recorder=SimpleNamespace(close=close))
            report={}
            fly_and_finish_recording(runtime,report)
            self.assertEqual(events,['mission','land_confirmed','ground_recording','close'])
            self.assertTrue(report['capture_completed'])
            self.assertTrue(report['flight_completed'])

    def test_failed_landing_does_not_report_success_or_stop_capture_early(self):
        runtime=SimpleNamespace(run_mission=Mock(),vehicle=SimpleNamespace(
            finish_flight_session=Mock(side_effect=TimeoutError('not landed'))),fastlio_recorder=Mock())
        report={'flight_completed':False}
        with self.assertRaises(TimeoutError):fly_and_finish_recording(runtime,report)
        self.assertFalse(report['flight_completed'])
        runtime.fastlio_recorder.close.assert_not_called()

    def test_short_capture_comparison_needs_no_semantic_map(self):
        import h5py
        from tools.compare_fastlio_replay import compare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'inputs.h5';records=[]
            with h5py.File(source,'w') as f:
                scans=f.create_group('scans')
                for i,x in enumerate([0.,1.,0.]):
                    key=f'{i:08d}'
                    scans.create_group(key).create_dataset('points_body',data=np.array(
                        [[1.,y,z] for y in np.linspace(-1,1,8) for z in np.linspace(-1,0,8)]))
                    pose=np.eye(4);pose[0,3]=x
                    records.append(dict(timestamp=float(i),pose=pose.tolist(),scan_key=key,
                                        px4_pose=[10+x,20.,0.,1.,0.,0.,0.]))
            (root/'trajectory.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            with patch('builtins.print'):
                compare(source,root)
            result=json.loads((root/'comparison.json').read_text())
            self.assertAlmostEqual(result['disagreement_rmse_m'],0.)
            self.assertAlmostEqual(result['endpoint_displacement_lio_m'],0.)
            self.assertEqual(result['vehicle_roi_surface_proxy'],{})
            self.assertTrue((root/'map_comparison.png').exists())
