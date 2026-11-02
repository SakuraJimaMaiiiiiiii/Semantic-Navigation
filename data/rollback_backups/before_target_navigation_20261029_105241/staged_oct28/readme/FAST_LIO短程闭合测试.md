# FAST-LIO 短程闭合录制

在项目根目录、原有 Conda 环境运行。开启 UE、CopterSim/PX4 和项目 WebSocket 服务；正常退出旧的探索控制终端。不要同时启动 `run_explore.py`。

```powershell
python run_fastlio_test.py --dry-run
python run_fastlio_test.py
```

第一条只打印相对航点，不连接传感器、不起飞、不创建实验目录。第二条会实际解锁起飞。

默认过程：

1. 地面保持未解锁，静止录制雷达与原始 IMU 10 秒。
2. 保持原始航向起飞至相对起飞点 1.5 m，等待 10 秒。
3. 相对初始朝向前进 3 m、右移 1.5 m、后退 3 m、左移 1.5 m，完成闭合矩形。每个顶点停留 2 秒，飞行中沿用已有平滑位置航点控制和局部避障。
4. 在起点上方悬停 10 秒，按原任务控制器恢复起飞前航向、近地接近并交给 PX4 降落。
5. 确认落地后再录制 3 秒，排空录制队列，生成输入审计并自动退出。不会停留在目标导航服务中。

前方和右侧应有足够空间及高度净空；未知或被占据区域仍由现有规划器拦截。遇到阻断、超时或录制故障时终止测试，尝试当前位置降落并记录错误，不把未完成任务标成闭合成功。Ctrl+C 使用同一清理流程；落地失败仍会报告，不强制关桨。

可缩短范围：

```powershell
python run_fastlio_test.py --length 2 --width 1 --height 1.5
```

`length/width` 是相对起飞朝向的前/右距离，不是世界 N/E 坐标。高度默认保持 1.5 m，水平参考速度不超过现有配置或 0.3 m/s 中的较小值。脚本不启动 YOLO/SAM，也不录制 RGB-D 视频；相机、雷达、占据地图和避障照常运行。FAST-LIO 只录制输入，不接管定位或飞行控制。

## 输出

终端打印真实目录 `data/runs/run_...`，其中：

- `fastlio_test.json`：预定航点、实际起点、初始航向、末位置、落地证据及完成/异常状态。
- `fastlio/inputs.h5`：包含地面初始化、起飞、四段移动、返航、降落的雷达/IMU。
- `fastlio/capture_report.json`：必须检查 `completed=true`、`error=null`、`overflow=0`。
- `fastlio/input_audit.json`：自动生成的输入检查；扫描时序尚未确认，`lidar_scan_model_unverified` 仍属预期待核查项，不能把它当作已验证。

成功落地但录制失败仍会报错。两个完成标记都通过也只说明试飞/录制完成，不代表 FAST-LIO 精度已通过。

## 离线验证

把下方 `run_你的实际目录` 替换为终端打印的目录。正常完成的新录制无需运行修复工具。

```powershell
python tools/audit_fastlio_inputs.py "data/runs/run_你的实际目录"
python tools/replay_fastlio.py "data/runs/run_你的实际目录" --output "data/fastlio_validation/short_loop_01" --assume-snapshot
python tools/compare_fastlio_replay.py "data/runs/run_你的实际目录/fastlio/inputs.h5" "data/fastlio_validation/short_loop_01"
```

`--assume-snapshot` 是离线试验假设，不是对传感器时序的确认。比较工具不再强制要求语义地图：短程测试输出轨迹/几何地图及墙面条带指标；只有另行指定 `--semantic-map` 才分析车辆 ROI。重复回放须指定新的输出目录。

## 第三段 `no_path` 的高度层修复

`run_20261023_130021_090922` 的第三段目标 Down=-1.496 m，停滞期间实际 Down 约 -1.65 m。
旧定高投影仅开放目标所在单层体素，相邻层全部标为障碍，约 0.15 m 的高度偏差跨过 0.2 m 体素边界后便不能回到目标层；水平规划图仍可能看起来畅通。

现在 `EgoLocalPlanner._plane_snapshot` 对目标层上下相邻层分别执行真实机身净空检查；只有已观测自由、包含整层高度变化对应机身体积的格子才开放。目标高度与未知/障碍判定未放宽。此 Python 逻辑也由原生 EGO 包装器使用，无须重新编译 C++。

Windows 下高度恢复、天花板/未知阻挡、横向墙体阻挡三项回归测试，以及短程脚本七项测试通过。仍需在 UE 中重跑确认实际飞行表现。

## 闭合后 `status=clear` 仍超时的修复

`run_20261023_190120_712028` 已完成矩形、回到起点附近，但旧局部返航以 1e-6 m 判断是否需要二次微调，并将实测 Down=-1.519 m 传给定高规划器（设定 Down=-1.5167359 m）。规划器输出高度与控制器原目标不完全一致，无法满足最终参考点完成判断。

现在局部返航使用与降落前确认一致的起点容差；已经回到起点就进入既有悬停、恢复航向及降落步骤。若确实需要继续返航，目标 Z 使用规划器的固定高度。没有放宽障碍检查或控制器的轨迹终点检查。

该次录制的 `capture_report.json` 为 completed=true，24,884 条 IMU、1,438 帧雷达，静止初始化与 IMU 时间覆盖均通过。虽然测试任务因额外返航微调超时而失败，但录制可用于后续离线验证。脚本现将“任务成功”和“录制完整”分别更新，异常清理成功排空的数据不再被误标为未完成。
