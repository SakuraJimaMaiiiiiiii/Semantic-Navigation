# FAST-LIO 原生接入审查

> 以下为早期接入审查，2026-10-22 已新增原生扩展、原始输入录制与旁路验证。当前状态见 [FAST_LIO原生接入与验证.md](FAST_LIO原生接入与验证.md)。在线地图接管仍未启用。

参考：https://github.com/hku-mars/FAST_LIO
已读取提交 7cc4175de6f8ba2edf34bab02a42195b141027e9 的 CMakeLists.txt、
use-ikfom.hpp、IMU_Processing.hpp、laserMapping.cpp 与 ikd-Tree 代码。
官方代码在 /tmp/semantic-fastlio-reference 作只读参考，不是本工程运行依赖。

## 已落地

修正本机雷达样本的轴向转换；运行时改为雷达单独更新几何占据地图，相机负责语义关联。
现有 A* / EGO / 控制仍使用 NED；这仍是 PX4 位姿驱动的建图，不称为 FAST-LIO。

## 原生 Windows 接入尚未完成的条件

1. 官方 CMake 依赖 catkin/ROS 消息、PCL、Eigen；ikd-Tree 使用 pthread。需抽离通信、移植线程层，不能原样当 MSVC 子目录编译。
2. 本机 PX4MavCtrlV4 的 HIGHRES_IMU 接收代码只覆盖 uavAccB/uavGyro，没有保存每条 IMU 的 time_usec、fields_updated 和消息队列。
   不能按应用线程读取时刻伪造原始采样时刻，也不能用速度差分冒充原始加速度。
3. 当前 Type20 XYZ 输出只记录整帧时间。需要核实 UE 为瞬时快照还是逐束扫描；若逐束扫描，必须取得点内时间偏移才能正确去畸变。
4. LIO 地图系、控制器 NED 和相机位姿需统一变换。相机语义位置必须使用图像时刻的 LIO 位姿；
   不能用 LIO 建图却继续直接用 PX4 位姿放置语义目标。
5. 需要雷达+原始 IMU 的录制数据作离线回放验证，再接控制。当前已有 NPZ 没有原始 IMU 序列，无法验收 LIO。

没有添加空 CMake 包装、伪造 FAST-LIO 后端或用普通 ICP 代替官方 LIO。
后续工作应是原始 IMU 接入/录制→ROS-free 官方核心移植及编译→回放验证→LIO 位姿接管地图和语义坐标。
