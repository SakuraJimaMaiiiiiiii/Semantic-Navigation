# FAST-LIO 时序与坐标核查

探索启动时按状态采样频率（最高 50 Hz）请求 MAVLink ATTITUDE（30）和 LOCAL_POSITION_NED（32），并在同步线程关闭时恢复默认请求。PX4 不保证接受指定频率；实际同步通过运行结果验证，不将请求频率当作实测频率。

最近一次可用录制为 `data/runs/run_20261029_185557_986217`。日志显示 2,986 帧雷达因位姿同步误差被拒绝；输入文件保存的 2,477 帧全部进入 FAST-LIO 求解。旧结果无法证明调整遥测频率后丢帧会减少。新运行的 `lidar_camera_alignment.json` 新增 `received_lidar_frames` 与 `accepted_lidar_frames`，可与 `pose_rejected_frames`、`pose_timeout_frames` 一起比较。

多帧坐标与时间审计（只读，不改旧地图）：

```powershell
python tools/audit_lidar_camera_sequence.py "data/runs/实际运行目录"
python tools/audit_camera_lidar_geometry.py "data/runs/实际运行目录"
python tools/audit_fastlio_inputs.py "data/runs/实际运行目录"
python tools/audit_fastlio_pose_gaps.py "data/runs/实际运行目录"
```

多帧审计将原始雷达、机体位姿、相机外参与深度图转换到同一相机光学坐标，统计 50 ms 内配对的深度差。遮挡和错配点也保留在全部点残差中，不能只看过滤后的数据。相机与雷达使用共同 PX4 位姿，且当前没有独立地面真值，因此这些指标不能证明绝对定位精度。

间隔遮挡验证使用已录制的中间 FAST-LIO 修正位姿作对照。旧实验中 779 段 0.9–1.5 s 间隔，在端点修正变化较小的情况下，10 m 范围内的误差界限 p95 约 0.098 m、最大约 1.19 m。中间位姿也不是地面真值；这些数据只能说明放宽到 1.5 s 并不可靠，不能据此正式接管导航。现有 0.5 s 上限不变。

本机 RflySim Type20 SDK 输出帧时间戳与 XYZ；第四通道是 Stencil/零，不是逐点采样时间。`DataCheckFreq` 不是扫描曝光模型。要确认 `snapshot_assumed`，必须检查 UE 端传感器实现或供应商对一次点云生成时刻的说明；若是逐束扫描，还需从源头取得真实逐点时间并进行运动去畸变。仅靠帧间隔、单帧对齐、平面平整度或飞行轨迹不能证明这一点。

在时序和坐标未验证前，FAST-LIO 继续用于独立实验地图；主占据地图和飞行控制仍使用 PX4 链路。建议新实验覆盖静止、平移、旋转和同一路径往返，并提供已知场景几何或 UE 物体真值，分别检查墙面重影、位置误差和同车跨视角一致性。
