# FAST-LIO2 原生接入与验证

## 当前阶段（2026-10-22）

已实现 Windows Python 3.10 原生 CMake 扩展、同一 MAVLink 接收器上的原始 IMU 采集、雷达同步录制、离线回放、在线旁路输出。
**尚未让 LIO 接管占据地图、语义坐标或飞行控制。** 必须先用实际同步数据验证时间模型和估计效果，不能把合成测试当成定位精度验收。

原有 PX4 建图、回环、A*、EGO、控制器保持原行为。不创建 Python 环境，不安装 ROS，不启用 VIO。

## 代码来源与移植范围

- FAST_LIO：`7cc4175de6f8ba2edf34bab02a42195b141027e9`。
- ikd-Tree：`e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4`。
- 官方仓库：https://github.com/hku-mars/FAST_LIO
- 固定源码保留在 `localization/native_fastlio/third_party/FAST_LIO`，由 `prepare_source.py` 拉取；保留原 LICENSE。
- 直接使用官方 IKFoM 状态模型、滤波器、IMU 初始化/积分/去畸变代码及 ikd-Tree。
- `bindings.cpp` 适配官方点面残差与雅可比，固定雷达/IMU 外参，4 次迭代；不是普通 ICP。
- ROS 消息换成字段相同的纯 C++ 数据容器，PCL 点云容器换成向量，PCL VoxelGrid 换成体素质心采样，pthread 在 Windows 映射到标准线程/互斥锁。
- 官方 IMU 模块的首帧扫描结束时刻补充初始化；去掉内部调试文件写入，日志由会话管理。
- 当前版本不是完整官方 ROS 节点：无 ROS 发布、无驱动、无自动外参标定、无官方滑动地图裁剪。ikd-Tree 以体素降采样增量维护，超大场景仍需增加局部窗口；先用于本车库回放与旁路验证。
- 原始 IMU 噪声暂用官方示例量级（gyro/acc covariance 0.1，bias covariance 0.0001），不是该仿真传感器的标定结果。

## 编译与检查

在项目根目录、现有 Conda 环境执行：

```powershell
python localization/native_fastlio/build.py
python tools/test_fastlio_native.py
```

编译使用现有 `.tools/vcpkg`（或环境变量 `VCPKG_ROOT`）、MSVC、Eigen、Boost 和 pybind11。依赖在模块自己的 build-windows/vcpkg_installed 内，不更改 Python 科学计算包。
产物：`localization/_native_fastlio.cp310-win_amd64.pyd`。

## 第一步：采集真实输入

开启 UE / CopterSim 后：

```powershell
python run_explore.py --sam3 --fastlio record
```

也可以把 `--sam3` 换成 `--sam2`。起飞前保持机体静止，便于获得至少 1 秒静态初始化数据。

输出到本次运行的 `fastlio/`：

- `inputs.h5`：原始 IMU、每帧机体 FRD 雷达点、扫描时刻的 PX4 位姿。
- `capture_report.json`：采样数、队列溢出、部分字段更新包等诊断。
- `shadow_poses.jsonl`：只有 shadow 模式有姿态输出，record 模式为空。

探索结束会关闭录制，不会在等待后续目标导航时继续写入。
接收来自已有 pymavlink `message_hooks`，不新建飞控接收端口。HIGHRES_IMU 请求 200 Hz，实际频率由 source timestamps 决定；不通过轮询最新缓存伪造 200 Hz 数据。首版只接受 sensor_id=0 且加速度/角速度六字段完整更新的样本，其他包统计但不填补。

录制完成后，先审计实际目录：

```powershell
python tools/audit_fastlio_inputs.py "data/runs/实际新运行目录"
```

审计只读文件，检查源时间单调性、采样间隔、初始静止段和 IMU 对扫描时刻的覆盖情况；不自动宣称同步或定位精度合格。

## 第二步：核实扫描与时间模型

Type20 目前只有整帧 XYZ 和时间戳。尚未得到 UE 端逐束扫描时间定义的证据。

- 若全帧是同一时刻的瞬时快照，可选择 snapshot，采用官方 MARSIM 分支，**不伪造逐点时间**。
- 若为随时间逐束采样，必须先让驱动提供真实点内偏移；不能用数组顺序凭空生成时间。

采集文件的雷达源时刻由现有 RGBD 位姿插值得到 PX4 boot time，保存的 `clock_alignment` 明确标为接收时间对齐，**不是硬件同步**。IMU 使用真实 HIGHRES_IMU time_usec。PX4 与 UE 的时间偏差仍需通过运动数据审计。
Type20 当前零安装变换且统一 FRD；默认 IMU 与机体共点同轴。真实非零安装位置必须标定并传入原生 Estimator 外参，不能直接沿用默认值。

## 第三步：离线回放

确认传感器确实为瞬时快照后，用上一步真实运行目录：

```powershell
python tools/replay_fastlio.py "data/runs/实际新运行目录" --scan-model snapshot --output "data/fastlio_comparison/第一次回放"
```

`实际新运行目录` 必须替换为真正存在且包含 fastlio/inputs.h5 的目录；旧录像没有原始 IMU，不能直接回放 LIO。

输出：`trajectory.jsonl`、`map.npz`、`report.json`。坐标明确为 **LIO 初始化 IMU 坐标系，非 PX4 NED**，不能直接把这些点送给控制器。报告显示跟踪/初始化/配准不足数量与耗时，不把相对 PX4 差异冒充绝对精度。

## 可选：在线旁路

只在扫描模型核实之后：

```powershell
python run_explore.py --sam3 --fastlio shadow --lidar-scan-model snapshot
```

同样记录原始输入，后台运行原生滤波，把结果写到 shadow_poses.jsonl。缺少可靠输入时在 capture_report 中报告 shadow_error，不用 PX4 位姿伪装成 LIO 成功输出。计算阶段释放 Python GIL；导航仍读 PX4 链路。

## 在线地图接管的验收条件

1. 静态初始化通过；原始 IMU 时间单调、采样间隔合理，无未解释缺包/重启。
2. 扫描模式、FRD/FLU、加速度单位、重力方向及雷达/IMU外参确认。
3. 同一数据回放，检查墙面厚度、重影、轨迹闭合误差；以 UE 真值/人工几何为基准，而非仅比较 PX4。
4. 建立 `map_lio ↔ odom_px4` 变换与历史位姿插值，再同时适配占据建图、相机语义定位、规划航点到控制系的转换。
5. 回环更新只校正 map 变换，避免校正突跳直接传给控制器。

本阶段没有实现第 4/5 项在线接管，默认运行 `python run_explore.py --sam3` 仍完全不启用 FAST-LIO。

## 2026-10-23：异常录制的独立验证副本

不修改原始 HDF5。以下工具只允许删除 IMU 末尾连续全零行，遇到内部时间倒退、重复或非有限数会拒绝恢复；同时保存原文件和副本 SHA-256。它不宣称恢复了尚未写盘的数据。

```powershell
python tools/prepare_fastlio_validation.py "data/runs/run_20261022_190623_019668" --output "data/fastlio_validation/run_20261022_190623_019668/input"
python tools/audit_fastlio_inputs.py "data/fastlio_validation/run_20261022_190623_019668/input/inputs.h5"
```

只有离线回放提供 `--assume-snapshot`：这是明确标记的扫描时序假设，不会把录制文件改成已验证的 snapshot，也不允许据此启动在线接管。当前 Type20 的 SDK 点云第四列是 Stencil/零值，不是采样时间；过滤后的录制数据不能恢复逐点时序。若最终确认扫描跨时间累积，需要从源头保存真实逐点时间并去畸变，不能按发布频率或过滤后点序伪造时间。

```powershell
python tools/replay_fastlio.py "data/fastlio_validation/run_20261022_190623_019668/input/inputs.h5" --output "data/fastlio_validation/run_20261022_190623_019668/replay_full" --assume-snapshot
python tools/compare_fastlio_replay.py "data/fastlio_validation/run_20261022_190623_019668/input/inputs.h5" "data/fastlio_validation/run_20261022_190623_019668/replay_full" --semantic-map "data/runs/run_20261022_190623_019668/exploration/semantic_map.json"
```

上述输出目录本轮已生成，重跑回放时应换一个新目录。对比仅用首个匹配位姿做刚体坐标对齐，不拟合尺度、不用全轨迹拟合掩盖漂移。PX4 只是参考估计；起终点距离仅在物理位置确实相同时才可解释为闭合误差。地图局部平面残差属于粗糙度代理指标，不能单独证明车辆几何准确或重影消失。

## 实时点云建图试验（2026-10-23）

在原环境启动 UE、PX4 和 WebSocket 后运行：

```powershell
python run_explore.py --sam3 --fastlio live --lidar-scan-model snapshot_assumed
```

也可以将 `--sam3` 换成 `--sam2`。已编译过原生扩展的环境不需重新编译。
`snapshot_assumed` 明确表示“假定瞬时扫描”，并非已经确认雷达时序；原始 H5
也保留该标记。只有确认传感器一次生成整帧点云时才使用 `snapshot`。

起飞前等待 FAST-LIO 静止初始化及 tracking，失败则不开始任务。运行中打开本次目录的
`fastlio/map.html`，每三秒刷新；结束保存 `fastlio/live_map.npz`（0.15 m
体素化点云、轨迹、初始对齐矩阵）和 `live_map.json`。位姿另存
`shadow_poses.jsonl`，其中包含同帧 PX4 位姿。该 HTML 是全部高度点云的二维投影，
不包含自由空间证据，不能把所有投影点直接当作飞行平面的障碍。

这是独立的实时建图验证模式：FAST-LIO 首次 tracking 与同期 PX4 位姿做一次刚体对齐，
随后不重新锚定 PX4。探索用的 `exploration/map.html`、语义位置、A*/EGO 和控制器
仍使用既有链路；本次没有让未经验证的 LIO 坐标直接接管导航。
数据采集结束后地图停止更新。在线计算错误会保存在 `capture_report.json`，
不能把停止更新的试验地图视为导航地图。

不使用 FAST-LIO：启动时省略 `--fastlio`。只录原始数据：`--fastlio record`。

## LIO 语义坐标链路（2026-10-24）

当前 `--fastlio live` 自动接收感知模块融合前的逐帧语义测量：
相机反投影及现有外参先给出 PX4 世界测量，再按同一观测时刻施加
`T_lio_map_from_px4_world(t) = T_lio_map_from_body(t) * inverse(T_px4_world_from_body(t))`。
FAST-LIO 的 IMU boot 时间不直接与相机比较，使用雷达帧已记录的 shared_timestamp
匹配相机时间轴。首个 tracking 的初始对齐与点云完全共用。

旋转采用三维四元数插值，平移线性插值；禁止时间外推、禁止跨超过 0.5 秒的位姿空档插值。
物体中心、表面点、包围框八角点、测量协方差同时转换后才进行实例关联。
这是坐标统一，不是重新标定外参，也不能消除已有相机/雷达时间误差。

输出：
- `fastlio/map.html`：LIO 点云及蓝色语义标签。
- `fastlio/semantic_map.json`：同坐标系的物体记录和同步统计。
- `fastlio/shadow_poses.jsonl`：新增 shared_timestamp 和 px4_to_lio_map，便于审计。

LIO 实例使用独立 ID 空间；不应将其 vehicle_05 当作旧 exploration 地图的 vehicle_05。
原探索语义、占据图、A*/EGO 和控制器不变。地图仍是点云回波图，不是导航可通行占据图。

离线验证已有实验（输出文件必须不存在）：

```powershell
python tools/validate_fastlio_semantics.py data/runs/run_20261024_144818_484894 --output data/fastlio_validation/semantic_check.json
```

该工具在原 Windows Python 环境执行，需现有 RflySim SDK 导入路径；不启动 UE、不发飞行命令。
