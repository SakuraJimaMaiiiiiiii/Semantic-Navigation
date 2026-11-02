"""Standalone RflySim reset helper for reinforcement-learning episodes.

This module is intentionally not imported by ``run_demo.py`` or the vehicle
package. Merely importing it does not send any command to PX4, CopterSim, or
RflySim3D. A reset happens only when ``RLEpisodeResetter.reset()`` is called.

The reset is destructive to the current simulated flight and must never be
used on a real vehicle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EpisodeResetConfig:
    """Parameters for one simulated episode reset.

    ``home_xy_yaw`` is ``(north_m, east_m, yaw_deg)``. CopterSim determines
    the ground height automatically when ``sendReSimXYyaw`` is used.
    """

    home_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reboot_delay: int = -1
    reboot_wait: float = 6.0
    pose_wait: float = 3.0


class RLEpisodeResetter:
    """Reset a connected ``RflyMultirotorInterface`` between RL episodes.

    RflySim3D collision reception is kept alive. The MAVLink/CopterSim side is
    stopped, rebooted, moved to the configured home pose, and reconnected.
    The caller decides whether to arm and enter Offboard after this method.
    """

    def __init__(self, config: EpisodeResetConfig | None = None):
        self.config = config or EpisodeResetConfig()

    def reset(self, vehicle):
        """Force-reset one simulated vehicle and return it reconnected.

        This method intentionally does not call ``vehicle.close()`` because
        ``close()`` also stops the persistent UE collision receiver and, for
        safety, refuses to disarm a vehicle that has not confirmed landing.
        An RL collision reset instead deliberately terminates the simulated
        flight immediately.
        """

        self._validate_vehicle(vehicle)
        home_xy_yaw = self._vector3(self.config.home_xy_yaw)
        copter_id = int(vehicle.config.copter_id)
        old_mav = vehicle.mav

        print(f"Resetting RL episode for CopterID={copter_id} ...")

        # Stop the setpoint stream before restarting the simulation instance.
        if vehicle.offboard_enabled:
            old_mav.endOffboard()
            vehicle.offboard_enabled = False

        # This is a simulation-only emergency reset. Do not wait for landing.
        if bool(vehicle.armed or old_mav.isArmed):
            old_mav.SendMavArm(False)
            vehicle.armed = False

        # SDK implementation sends the reboot packet to the UDP port belonging
        # to this CopterID. It restarts the aircraft simulation, not RflySim3D.
        old_mav.sendRebootPix(copter_id, int(self.config.reboot_delay))
        old_mav.stopRun()

        vehicle.connected = False
        vehicle.mav = None

        time.sleep(float(self.config.reboot_wait))

        # Create a short-lived request interface after reboot and move the
        # simulated aircraft horizontally to the episode origin. CopterSim
        # fits its Z position to the terrain for this command.
        import ReqCopterSim

        reset_request = ReqCopterSim.ReqCopterSim()
        reset_request.sendReSimXYyaw(copter_id, home_xy_yaw)
        time.sleep(float(self.config.pose_wait))

        # connect() creates fresh MAVLink receivers and waits for PX4 EKF.
        vehicle.connect()
        self._clear_collision_state(vehicle)

        print(
            f"RL episode reset complete: CopterID={copter_id}, "
            f"home_xy_yaw={home_xy_yaw}"
        )
        return vehicle

    @staticmethod
    def _validate_vehicle(vehicle):
        if not bool(getattr(vehicle, "connected", False)):
            raise RuntimeError("The vehicle must be connected before reset().")
        if getattr(vehicle, "mav", None) is None:
            raise RuntimeError("The vehicle has no active PX4MavCtrl interface.")

    @staticmethod
    def _vector3(value: Sequence[float]) -> list[float]:
        result = [float(item) for item in value]
        if len(result) != 3:
            raise ValueError("home_xy_yaw must contain north, east, and yaw.")
        return result

    @staticmethod
    def _clear_collision_state(vehicle):
        ue = getattr(vehicle, "ue", None)
        if ue is not None:
            ue.isVehicleCrash = False
            ue.isVehicleCrashID = -10

        mav = getattr(vehicle, "mav", None)
        if mav is not None:
            mav.isVehicleCrash = False
            mav.isVehicleCrashID = -10


if __name__ == "__main__":
    print(
        "This file only defines the RL episode reset helper. "
        "Import RLEpisodeResetter and call reset(vehicle) explicitly."
    )
