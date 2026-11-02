"""飞行器控制接口。"""

from .rfly_multirotor import (
    CollisionDetectedError,
    DroneState,
    RflyMultirotorInterface,
)

__all__ = [
    "CollisionDetectedError",
    "DroneState",
    "RflyMultirotorInterface",
]
