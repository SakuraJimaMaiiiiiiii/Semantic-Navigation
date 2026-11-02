import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np


def methods():
    tree=ast.parse(Path('vehicle/rfly_multirotor.py').read_text(encoding='utf-8'))
    names={'enable_offboard','_prepare_offboard_sender'}
    namespace=dict(np=np, math=__import__('math'), threading=threading,
                   time=SimpleNamespace(sleep=lambda _:None))
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<offboard>','exec'),namespace)
    return namespace


class OffboardRestartTests(unittest.TestCase):
    def vehicle(self):
        ns=methods()
        mav=SimpleNamespace(isArmed=False, isInOffboard=False, SendPosNED=Mock(),
                            the_connection=object(), OffboardSendMode=Mock())
        mav.t2=threading.Thread(target=mav.OffboardSendMode)
        nav=SimpleNamespace(mav=mav,offboard_enabled=False,armed=False,
                            _require_connection=lambda:None,
                            get_position=lambda:np.array([1.,2.,-1.5]),
                            get_euler=lambda:np.array([0.,0.,1.]),arm=Mock())
        nav._prepare_offboard_sender=lambda:ns['_prepare_offboard_sender'](nav)
        def init():
            mav.isInOffboard=True
            mav.t2.start()
            mav.isArmed=True
        mav.initOffboard=Mock(side_effect=init)
        return nav,ns

    def test_three_flights_reuse_connection_but_not_completed_thread(self):
        nav,ns=self.vehicle()
        connection=nav.mav.the_connection
        threads=[]
        for _ in range(3):
            ns['enable_offboard'](nav)
            threads.append(nav.mav.t2)
            nav.mav.t2.join(timeout=1)
            self.assertFalse(nav.mav.t2.is_alive())
            self.assertIs(nav.mav.the_connection,connection)
            nav.offboard_enabled=False
            nav.mav.isInOffboard=False
            nav.mav.isArmed=False
            nav.armed=False
        self.assertEqual(len({id(t) for t in threads}),3)
        self.assertEqual(nav.mav.OffboardSendMode.call_count,3)

    def test_fresh_thread_retained(self):
        nav,ns=self.vehicle()
        original=nav.mav.t2
        nav._prepare_offboard_sender()
        self.assertIs(nav.mav.t2,original)

    def test_active_sender_rejected_before_commands(self):
        nav,ns=self.vehicle()
        stop=threading.Event()
        nav.mav.t2=threading.Thread(target=stop.wait)
        nav.mav.t2.start()
        try:
            with self.assertRaisesRegex(RuntimeError,'still running'):
                ns['enable_offboard'](nav)
            nav.mav.SendPosNED.assert_not_called()
            nav.mav.initOffboard.assert_not_called()
        finally:
            stop.set();nav.mav.t2.join()

    def test_already_offboard_does_not_restart(self):
        nav,ns=self.vehicle()
        nav.offboard_enabled=True;nav.armed=True
        ns['enable_offboard'](nav)
        nav.mav.initOffboard.assert_not_called()
