"""Record a disarmed RflySim IMU without launching exploration, VIO or flight control."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=1800.)
    parser.add_argument('--copter-id', type=int, default=1)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 10 or args.copter_id < 1:
        parser.error('Use at least 10 seconds and a positive copter ID; noise calibration needs >=600 seconds')
    from pymavlink import mavutil
    directory = Path('data/imu_calibration')/time.strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    # RflySim SDK SIL convention: command port 20100+2*(id-1), receive port +1.
    port = 20100+2*(args.copter_id-1)
    rx = mavutil.mavlink_connection(f'udpin:0.0.0.0:{port+1}')
    tx = mavutil.mavlink_connection(f'udpout:{args.host}:{port}')
    report = {'stationary_checks_passed': False, 'received_samples': 0, 'error': None}
    packet_stats = {'available': True, 'highres_packets': 0, 'other_sensor_packets': 0,
                    'partial_update_packets': 0, 'duplicate_source_packets': 0,
                    'fields_updated_counts': {}}
    report['packet_diagnostics'] = packet_stats
    requested = []
    target = None
    try:
        heartbeat = rx.wait_heartbeat(timeout=10.)
        if heartbeat is None:
            raise TimeoutError('No PX4 heartbeat; run CopterSim and close other control terminals')
        target = heartbeat.get_srcSystem(), heartbeat.get_srcComponent()
        if heartbeat.base_mode & 128:
            raise RuntimeError('Aircraft is armed; stationary recording requires disarmed aircraft')
        for message_id, rate in ((105, 200), (32, 20)):
            tx.mav.command_long_send(*target, 511, 0, message_id, 1_000_000/rate, 0, 0, 0, 0, 0)
            requested.append(message_id)
        print(f'Recording {args.seconds:g}s without arming. Keep the simulator stationary. Output: {directory}', flush=True)
        start = last_heartbeat = time.monotonic()
        last_local = last_imu = start
        origin = None
        last_stamp = None
        offset = None
        next_print = start+30
        with (directory/'imu.csv').open('w', newline='', encoding='utf-8') as f, \
                (directory/'imu_packets.csv').open('w', newline='', encoding='utf-8') as packet_file:
            writer = csv.writer(f)
            writer.writerow(['shared_timestamp','source_timestamp','ax','ay','az','gx','gy','gz'])
            packet_writer = csv.writer(packet_file)
            packet_writer.writerow(['received_monotonic', 'source_time_usec', 'sensor_id', 'fields_updated'])
            while time.monotonic()-start < args.seconds:
                message = rx.recv_match(blocking=True, timeout=.2)
                now = time.monotonic()
                if now-last_heartbeat > 3 or now-last_local > 3 or now-last_imu > 3:
                    raise TimeoutError('Required heartbeat, position or IMU telemetry stopped')
                if message is None:
                    continue
                if (message.get_srcSystem(), message.get_srcComponent()) != target:
                    continue
                kind = message.get_type()
                if kind == 'HEARTBEAT':
                    last_heartbeat = now
                    if message.base_mode & 128:
                        raise RuntimeError('Arming detected; recording aborted without issuing flight commands')
                elif kind == 'LOCAL_POSITION_NED':
                    last_local = now
                    position = np.array([message.x,message.y,message.z])
                    speed = np.linalg.norm([message.vx,message.vy,message.vz])
                    if origin is None: origin = position
                    if not np.isfinite(position).all() or not np.isfinite(speed) or speed > .05 or np.linalg.norm(position-origin)>.1:
                        raise RuntimeError('Stationary position/velocity check failed')
                elif kind == 'HIGHRES_IMU':
                    sensor_id, mask = int(getattr(message, 'id', 0)), int(message.fields_updated)
                    packet_writer.writerow([now, message.time_usec, sensor_id, mask])
                    packet_stats['highres_packets'] += 1
                    key = str(mask)
                    packet_stats['fields_updated_counts'][key] = packet_stats['fields_updated_counts'].get(key, 0)+1
                    if sensor_id != 0:
                        packet_stats['other_sensor_packets'] += 1
                        continue
                    if mask & 63 != 63:
                        packet_stats['partial_update_packets'] += 1
                        continue
                    stamp = message.time_usec*1e-6
                    values = [message.xacc,message.yacc,message.zacc,message.xgyro,message.ygyro,message.zgyro]
                    if not np.isfinite([stamp,*values]).all() or stamp <= 0:
                        raise ValueError('Non-finite or invalid IMU packet')
                    if last_stamp is not None and stamp <= last_stamp:
                        if stamp == last_stamp:
                            packet_stats['duplicate_source_packets'] += 1
                            continue
                        raise RuntimeError('IMU clock regressed')
                    if np.linalg.norm(values[3:]) > .03:
                        raise RuntimeError('Rotation detected; not a stationary calibration capture')
                    if offset is None: offset = time.time()-stamp if stamp < 1e9 else 0.
                    writer.writerow([stamp+offset,stamp,*values])
                    last_stamp, last_imu = stamp, now
                    report['received_samples'] += 1
                if now >= next_print:
                    f.flush()
                    packet_file.flush()
                    print(f'{now-start:.0f}s: {report["received_samples"]} IMU samples', flush=True)
                    next_print = now+30
        report['stationary_checks_passed'] = origin is not None and report['received_samples'] > 0
        report['elapsed_seconds'] = time.monotonic()-start
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            for message_id in requested:
                tx.mav.command_long_send(*target, 511, 0, message_id, 0, 0, 0, 0, 0, 0)
        finally:
            rx.close()
            tx.close()
            (directory/'capture.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Next: python tools/audit_imu_noise.py "{directory}"')


if __name__ == '__main__':
    main()
