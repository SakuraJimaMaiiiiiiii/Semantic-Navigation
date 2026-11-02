"""让 RflySim 无人机随机飞行，并记录碰撞消息中的原始名称格式。

运行前请先启动 RflySim3D、CopterSim 和 PX4：

    python test_collision_format.py

脚本使用本地 NED 坐标系，以起飞点为中心随机生成航点。检测到碰撞后会
把 CrashedName 的 Python 类型、repr、原始十六进制及多种解码结果打印到
终端，同时追加写入 collision_samples.jsonl，然后命令无人机自动降落。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from vehicle import RflyMultirotorInterface


def raw_name_bytes(value: Any) -> bytes | None:
    """尽量保留 SDK 返回的 CrashedName 原始字节。"""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and 0 <= item <= 255 for item in value
    ):
        return bytes(value)
    return None


def inspect_name(value: Any) -> dict[str, Any]:
    """生成可写入 JSON 的名称格式诊断信息。"""
    result: dict[str, Any] = {
        "python_type": f"{type(value).__module__}.{type(value).__name__}",
        "repr": repr(value),
    }
    raw = raw_name_bytes(value)
    if raw is None:
        text = str(value).split("\0", 1)[0]
        result["text"] = text
        result["note"] = "SDK 已返回文本，无法从该值恢复原始编码字节"
        return result

    # char[20] 是定长缓冲区；第一个 NUL 后是填充，不属于名称。
    payload = raw.split(b"\0", 1)[0]
    result.update(
        raw_length=len(raw),
        raw_hex=raw.hex(" "),
        payload_length=len(payload),
        payload_hex=payload.hex(" "),
    )
    decoded: dict[str, str] = {}
    for encoding in ("utf-8", "gbk", "ascii"):
        try:
            decoded[encoding] = payload.decode(encoding)
        except UnicodeDecodeError as error:
            decoded[encoding] = f"<解码失败: {error}>"
    result["decoded"] = decoded
    return result


def collision_sample(ue_data: Any, position: np.ndarray) -> dict[str, Any]:
    raw_name = getattr(ue_data, "CrashedName", None)
    return {
        "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "crash_type": int(getattr(ue_data, "CrashType", 0)),
        "copter_id": int(getattr(ue_data, "copterID", -1)),
        "vehicle_type": int(getattr(ue_data, "vehicleType", -1)),
        "position_ned": np.asarray(position, dtype=float).round(6).tolist(),
        "crashed_name": inspect_name(raw_name),
    }


def append_sample(path: Path, sample: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(sample, stream, ensure_ascii=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copter-id", type=int, default=1)
    parser.add_argument("--height", type=float, default=1.5,
                        help="相对起飞点高度，单位 m")
    parser.add_argument("--radius", type=float, default=8.0,
                        help="随机航点相对起飞点的水平半径，单位 m")
    parser.add_argument("--ram-heading", type=float, default=None,
                        help="主动撞击方向，NED 航向角（度）；设置后不再随机飞")
    parser.add_argument("--ram-distance", type=float, default=50.0,
                        help="主动撞击目标距离，单位 m")
    parser.add_argument("--target-timeout", type=float, default=15.0,
                        help="每个随机航点的最长飞行时间，单位 s")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="测试总时长，单位 s")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-ground", action="store_true",
                        help="也把 CrashType=-2 的地面接触记为碰撞")
    parser.add_argument("--output", type=Path,
                        default=Path("collision_samples.jsonl"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "height", "radius", "ram_distance", "target_timeout", "duration"
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须是正数")
    if args.copter_id <= 0:
        raise ValueError("--copter-id 必须大于 0")


def run(args: argparse.Namespace) -> bool:
    rng = random.Random(args.seed)
    vehicle = RflyMultirotorInterface(Config(copter_id=args.copter_id))
    collision_found = False
    landing_commanded = False

    try:
        vehicle.setup_ue()
        vehicle.connect()
        origin = vehicle.get_position()
        target_down = float(origin[2] - args.height)

        vehicle.enable_offboard(arm=True)
        vehicle.takeoff(args.height)
        print("--- 等待起飞；此阶段忽略地面接触消息 ----")
        takeoff_deadline = time.monotonic() + 12.0
        while time.monotonic() < takeoff_deadline:
            if abs(float(vehicle.get_position()[2]) - target_down) <= 0.25:
                break
            time.sleep(0.1)

        print(
            f"--- 开始随机飞行：中心={origin.round(3).tolist()}，"
            f"半径={args.radius:.1f} m，总时长={args.duration:.1f} s ----"
        )
        test_deadline = time.monotonic() + args.duration
        target_number = 0
        next_status_time = 0.0

        while time.monotonic() < test_deadline:
            if args.ram_heading is None:
                angle = rng.uniform(-math.pi, math.pi)
                # sqrt 使航点在圆形区域内近似均匀分布。
                distance = args.radius * math.sqrt(rng.random())
            else:
                angle = math.radians(args.ram_heading)
                distance = args.ram_distance
            target = np.asarray([
                origin[0] + distance * math.cos(angle),
                origin[1] + distance * math.sin(angle),
                target_down,
            ])
            target_number += 1
            mode = "主动撞击目标" if args.ram_heading is not None else "随机航点"
            print(f"--- {mode} {target_number}: {target.round(3).tolist()} ----")
            vehicle.send_position(target, yaw=angle)
            if args.ram_heading is None:
                target_deadline = min(
                    test_deadline, time.monotonic() + args.target_timeout
                )
            else:
                target_deadline = test_deadline

            while time.monotonic() < target_deadline:
                raw_ue_data = vehicle.ue.getUE4Data(
                    int(vehicle.config.copter_id)
                )
                crash_type = 0
                if raw_ue_data != 0:
                    crash_type = int(getattr(raw_ue_data, "CrashType", 0))

                now = time.monotonic()
                if now >= next_status_time:
                    packet_status = "无 UE 数据" if raw_ue_data == 0 else "有 UE 数据"
                    print(
                        f"状态：{packet_status}, CrashType={crash_type}, "
                        f"position_ned={vehicle.get_position().round(3).tolist()}"
                    )
                    next_status_time = now + 2.0

                is_collision = crash_type != 0 and (
                    args.include_ground or crash_type != -2
                )
                if is_collision:
                    ue_data = raw_ue_data
                    sample = collision_sample(ue_data, vehicle.get_position())
                    print("\n========== 捕获到碰撞原始格式 ==========")
                    print(json.dumps(sample, ensure_ascii=False, indent=2))
                    print("==========================================\n")
                    append_sample(args.output, sample)
                    print(f"碰撞样本已追加保存到：{args.output.resolve()}")
                    collision_found = True
                    return True

                position = vehicle.get_position()
                if np.linalg.norm(position - target) <= 0.35:
                    break
                time.sleep(0.05)

            if args.ram_heading is not None:
                break

        print("--- 测试时限内未检测到目标碰撞 ----")
        return False
    finally:
        if vehicle.connected and vehicle.mav is not None:
            if bool(vehicle.mav.isArmed):
                print("--- 测试结束，命令无人机自动降落 ----")
                vehicle.land()
                landing_commanded = True
            try:
                vehicle.close(disarm=landing_commanded)
            except Exception as error:
                print(f"WARNING: 飞控接口安全关闭未完成：{error}")
        elif vehicle.ue is not None:
            vehicle.close()
        if collision_found:
            print("--- 已获得碰撞名称格式样本 ----")


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n--- 用户中止测试 ----")


if __name__ == "__main__":
    main()
