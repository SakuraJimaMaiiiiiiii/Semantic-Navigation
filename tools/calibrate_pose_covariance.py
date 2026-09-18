"""被动接收 CopterSim 真值和 PX4 遥测，估计位姿误差统计；不会解锁/起飞。"""

import argparse
import json
import select
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pymavlink.dialects.v20 import common as mavlink

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors.empirical_pose_covariance import estimate_profile, pose_error


def decode_truth(data, copter_id):
    """仅接受本地 SDK 明确定义的 200/168 字节 CopterSim 真值包。"""
    if len(data) == 200 and struct.unpack_from('<i', data, 4)[0] == 152:
        v = struct.unpack_from('<2id27f3d', data, 8)
        if v[0] == copter_id:
            return v[2], np.array(v[6:9]), np.array(v[12:16])
    elif len(data) == 168:
        v = struct.unpack('<4i24f7d', data)
        if v[0] == 123456789 and v[1] == copter_id:
            return v[28], np.array(v[29:32]), np.array(v[10:14])
    return None


def attitude_quaternion(msg):
    if msg.get_type() == 'ATTITUDE_QUATERNION':
        return np.array([msg.q1, msg.q2, msg.q3, msg.q4])
    cr, cp, cy = np.cos(np.array([msg.roll, msg.pitch, msg.yaw]) / 2)
    sr, sp, sy = np.sin(np.array([msg.roll, msg.pitch, msg.yaw]) / 2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--copter-id', type=int, default=1)
    parser.add_argument('--truth-origin-ned', type=float, nargs=3, required=True,
                        metavar=('N', 'E', 'D'),
                        help='PX4 本地原点在仿真真值 NED 中的位置（米）；两个坐标系轴向必须一致')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.copter_id < 1:
        parser.error('seconds 必须为正数，copter-id 必须为正整数')
    origin = np.array(args.truth_origin_ned)
    if not np.isfinite(origin).all():
        parser.error('原点偏移必须为有限数值')
    base_port = 20100 + (args.copter_id - 1) * 2
    sockets = []
    samples, errors = [], []
    latest = {}
    last_pair = None
    last_truth_stamp = None
    decoder = mavlink.MAVLink(None)
    output = Path(__file__).resolve().parents[1] / 'data' / 'diagnostics' / f'pose_calibration_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    print('被动标定，不控制飞机。请停止 run_demo.py 及其他占用遥测/真值端口的程序。', flush=True)
    try:
        for port in (base_port + 1, base_port + 10001):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets.append(sock)
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(('0.0.0.0', port))
        deadline = time.monotonic() + args.seconds
        report_at = time.monotonic() + 5
        while time.monotonic() < deadline:
            for sock in select.select(sockets, [], [], min(.2, max(0, deadline-time.monotonic())))[0]:
                data = sock.recvfrom(65535)[0]
                received = time.monotonic()
                if sock is sockets[0]:
                    for msg in decoder.parse_buffer(data) or []:
                        if msg.get_srcSystem() != args.copter_id or msg.get_srcComponent() != 1:
                            continue
                        kind = msg.get_type()
                        if kind == 'LOCAL_POSITION_NED':
                            latest['position'] = received, msg
                        elif kind in ('ATTITUDE', 'ATTITUDE_QUATERNION'):
                            latest['attitude'] = received, msg
                    continue
                truth = decode_truth(data, args.copter_id)
                if truth is None or 'position' not in latest or 'attitude' not in latest:
                    continue
                stamp, truth_position, truth_q = truth
                if last_truth_stamp is not None and stamp < last_truth_stamp:
                    raise RuntimeError('仿真时钟重置，请重新进行独立标定')
                if stamp == last_truth_stamp:
                    continue
                last_truth_stamp = stamp
                pt, pos = latest['position']
                qt, att = latest['attitude']
                pair = (pos.time_boot_ms, att.time_boot_ms)
                # 不重复统计缓存遥测；接收时间配对只是一种有界近似，不声称硬件同步。
                if (last_pair is not None and (pair[0] == last_pair[0] or pair[1] == last_pair[1])):
                    continue
                skew = max(received-pt, received-qt)
                if skew > .02 or abs(pair[0]-pair[1]) > 20:
                    continue
                position = np.array([pos.x, pos.y, pos.z])
                quaternion = attitude_quaternion(att)
                try:
                    error = pose_error(position, quaternion, truth_position-origin, truth_q)
                except ValueError:
                    continue
                last_pair = pair
                errors.append(error)
                samples.append({'truth_time': stamp, 'px4_time_boot_ms': pair,
                                'receive_skew_s': skew, 'position': position.tolist(),
                                'quaternion_wxyz': quaternion.tolist(),
                                'truth_position_ned': truth_position.tolist(),
                                'truth_quaternion_wxyz': truth_q.tolist(), 'error': error.tolist()})
            if time.monotonic() >= report_at:
                print(f'有效配对数：{len(errors)}', flush=True)
                report_at = time.monotonic()+5
    finally:
        for sock in sockets:
            sock.close()
        (output / 'samples.json').write_text(json.dumps(samples, indent=2, allow_nan=False), encoding='utf-8')
    profile = estimate_profile(errors)
    profile.update(truth_origin_ned=origin.tolist(),
                   timestamp_alignment='host_receive_time_within_20ms_not_hardware_synchronized',
                   note='Empirical operating-condition estimate, not PX4 posterior covariance. Includes bias and residual timing error.')
    (output / 'profile.json').write_text(json.dumps(profile, indent=2, allow_nan=False), encoding='utf-8')
    print(f'标定文件：{output / "profile.json"}')
    print('有效标准差 [m,m,m,rad,rad,rad]：', np.sqrt(np.diag(profile['effective_covariance'])))


if __name__ == '__main__':
    main()
