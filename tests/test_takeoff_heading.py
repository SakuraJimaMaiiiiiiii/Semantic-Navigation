"""Preserve measured heading throughout Offboard startup and takeoff."""

from unittest.mock import MagicMock, call

import numpy as np

from config.drone_config import Config
from vehicle.rfly_multirotor import RflyMultirotorInterface


def test_offboard_seeds_heading_before_starting_sdk(monkeypatch):
    vehicle = RflyMultirotorInterface(Config())
    vehicle.connected = True
    vehicle.mav = MagicMock()
    vehicle.mav.isArmed = True
    vehicle.get_position = lambda: np.array([2.0, 3.0, 0.0])
    vehicle.get_euler = lambda: np.array([0.0, 0.0, -0.7])
    monkeypatch.setattr("vehicle.rfly_multirotor.time.sleep", lambda _: None)

    vehicle.enable_offboard()
    vehicle.takeoff(1.5)

    assert vehicle.mav.method_calls == [
        call.SendPosNED(2.0, 3.0, 0.0, -0.7),
        call.initOffboard(),
        call.SendPosNED(2.0, 3.0, 0.0, -0.7),
        call.SendPosNED(2.0, 3.0, -1.5, -0.7),
    ]
