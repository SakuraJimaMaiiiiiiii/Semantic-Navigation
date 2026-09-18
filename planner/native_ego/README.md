# Native EGO planner

这个目录提供无 ROS 的 C++17 EGO-Planner 核心适配，并通过 pybind11 暴露给
现有 Python 工程。它保留项目的 NED 三维占据栅格与 `LocalPlan` 上层接口，
底层规划按官方 EGO 的阶段组织：

- `rebound_replan(...)`：接收种子、地图及起始速度/加速度，返回均匀三次
  B 样条控制点和节点间隔；
- 首次规划用五次边界多项式采样，后续规划复用在执行轨迹的剩余部分；
- 使用官方形式的 B 样条位置/速度/加速度约束方程生成控制点；
- 只对碰撞控制点区段运行三维 A*，由路径与控制点法平面的交点生成
  base point/direction；
- rebound 主优化使用 L-BFGS，并在碰撞复核失败后重新生成约束、提高碰撞权重；
- 动力学不可行时按速度/加速度比例拉伸时间，再执行 fitness refine。

ROS `NodeHandle`、topic、消息、FSM 和官方 `GridMap` 没有引入；它们分别由普通
配置、pybind 调用、现有任务循环和本项目占据栅格适配。地图时效检查、深度急停、
速度限制、语义任务和飞控接口继续由 Python 层负责，因此上层任务接口不变。

本实现以官方仓库的 `planner_manager.cpp`、`bspline_optimizer.cpp`、
`dyn_a_star.cpp` 和 `uniform_bspline.cpp` 为算法对照，但不是逐行复制。

## Windows 构建

应当在项目根目录、运行 RflySim 的同一个 Conda 环境中执行。Visual Studio
2026 对应生成器为 `Visual Studio 18 2026`：

```powershell
python -m pip install pybind11 cmake ninja
$pythonExe = (Get-Command python).Source
$pybind11Dir = (python -m pybind11 --cmakedir).Trim()
cmake -S "planner/native_ego" -B "planner/native_ego/build-vs2026" `
  -G "Visual Studio 18 2026" -A x64 `
  "-Dpybind11_DIR=$pybind11Dir" `
  "-DPython_EXECUTABLE=$pythonExe" `
  "-DPython_ROOT_DIR=$env:CONDA_PREFIX"
cmake --build "planner/native_ego/build-vs2026" --config Release
```

编译产物会放入 `planner/`。随后运行：

```powershell
python -c "from planner import _native_ego; print(_native_ego.__doc__)"
```

未编译扩展时，`ego_native` 会按配置自动回退到 Python EGO 实现，飞行安全检查
不会被绕过。
