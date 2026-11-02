import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


def method(name, namespace):
    tree = ast.parse(Path('vehicle/rfly_multirotor.py').read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<landing>', 'exec'), namespace)
    return namespace[name]


class LandingTests(unittest.TestCase):
    def test_velocity_descent_then_handoff_without_disarming(self):
        clock = [100.]
        local = SimpleNamespace(x=0., y=0., z=-1.5, _timestamp=100.)
        heartbeat = SimpleNamespace(base_mode=128, custom_mode=6 << 16, _timestamp=100.)
        commands = []
        def sleep(dt):
            clock[0] += dt
            local._timestamp = heartbeat._timestamp = clock[0]
        fake_time = SimpleNamespace(monotonic=lambda:clock[0], time=lambda:clock[0], sleep=sleep)
        nav = SimpleNamespace(_require_offboard=lambda:None, _vector3=lambda x,n:np.asarray(x),
            get_euler=lambda:np.array([0.,0.,1.2]), get_position=lambda:np.array([local.x,local.y,local.z]),
            config=SimpleNamespace(landed_height_tolerance=.35, guided_max_z_speed=.12, landing_timeout=30., guided_max_horizontal_acceleration=.12),
            mav=SimpleNamespace(isArmed=True, the_connection=SimpleNamespace(messages={
                'LOCAL_POSITION_NED':local,'HEARTBEAT':heartbeat})),
            raise_if_collision=lambda:None, hover=lambda:None)
        follow = [True]
        def send(**command):
            commands.append(command)
            if follow[0]:
                local.z += command['vel'][2]*.05
        nav.mav.SendOffAll = send
        sdk = SimpleNamespace(PosTypeMask=lambda ignore_all:SimpleNamespace())
        execute = method('approach_home_ground', {'np':np,'time':fake_time,'PX4MavCtrl':sdk})
        self.assertTrue(execute(nav, [0.,0.,0.], .6))
        self.assertGreaterEqual(local.z, -.40)
        self.assertTrue(nav.mav.isArmed)
        self.assertTrue(all(c['yaw'] == 1.2 for c in commands))
        self.assertTrue(all(np.isnan(c['pos'][2]) and np.isnan(c['vel'][0]) for c in commands))
        self.assertTrue(all(c['pos'][:2] == [0.,0.] for c in commands))
        self.assertTrue(all(0 <= c['vel'][2] <= .12 for c in commands))
        self.assertLessEqual(max(np.diff([0.]+[c['vel'][2] for c in commands])), .12*.05+1e-8)
        self.assertEqual(nav._landing_approach_evidence['status'],'near_ground')
        local.z = -.98
        follow[0] = False
        started = clock[0]
        self.assertFalse(execute(nav, [0.,0.,0.], .6))
        self.assertGreaterEqual(clock[0]-started, 30.)
        self.assertLess(clock[0]-started, 30.1)
        self.assertTrue(nav.mav.isArmed)
        self.assertEqual(nav._landing_approach_evidence['status'],'approach_timeout')
        self.assertEqual(nav._landing_approach_evidence['descent_m'],0.)
        local.z=-1.5
        heartbeat._timestamp = 0.
        with self.assertRaisesRegex(RuntimeError, 'telemetry stale'):
            execute(nav, [0.,0.,0.], .6)
        heartbeat._timestamp=clock[0]
        nav.mav.isArmed=False
        with self.assertRaisesRegex(RuntimeError, 'disarmed'):
            execute(nav, [0.,0.,0.], .6)

    def test_ue_freshness_uses_received_packet_not_constant_sim_clock(self):
        clock=[0.]
        packet=SimpleNamespace(runnedTime=3., PosE=[0.,0.,-1.], VelE=[0.,0.,0.])
        nav=SimpleNamespace(ue=SimpleNamespace(getUE4Data=lambda _:packet),config=SimpleNamespace(copter_id=1))
        execute=method('_ue_landing_diagnostics', {'np':np,'math':math,
                       'time':SimpleNamespace(monotonic=lambda:clock[0])})
        self.assertEqual(execute(nav)['observed_age_seconds'], 0.)
        clock[0]=10.
        self.assertEqual(execute(nav)['observed_age_seconds'], 10.)
        packet.PosE=list(packet.PosE)  # SDK CopyData received another packet.
        self.assertEqual(execute(nav)['observed_age_seconds'], 0.)
