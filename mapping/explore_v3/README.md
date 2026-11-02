# 探索专用的 2026-10-20 建图与定位

来源为 Git 提交 `599499a`（v3）。`source_manifest.json` 记录原始模块的 SHA-256；
雷达工作线程新增可选 FAST-LIO 同步扫描记录钩子，原有占据地图融合不变；其实际文件哈希单独记录。
未变化的射线、机身覆盖、回环重建等辅助模块通过包搜索路径复用。

仅 `DemoRuntime(exploration=True)` 选用本后端：

- v3 局部/全局占据地图，原有 LiDAR→PX4 NED 融合与顺序地图更新；
- 当前车辆跨视角关联、碎片确认和实测表面几何；非车辆地标仍使用 v3 逻辑；
- v3 在线导航地图的回环重建；实时与回环语义回放共用 adapters.SemanticInstanceMapper。

SAM3 的服务、模型、请求格式和掩膜生成不变。当前掩膜反投影/雷达匹配的核心计算
与 v3 一致，定位差异来自后续实例关联和表面拟合；当前处理器注入适配后的 mapper。
YOLO-World+SAM2 也可通过原有入口使用该 mapper。

接口适配只提供当前执行器需要的 execution_snapshot、前向净空和地图心跳接口；
v3 每次融合完成后发布快照，避免控制线程反复构建完整地图。
车辆关联使用当前 tracker，并向感知提供关联候选；当前 v3 探索器本身不执行主动多视角补拍。

探索可使用 `--fastlio` 启动独立的 FAST-LIO 实验点云和语义地图；不指定时关闭。
`navigate_target.py`、独立地图加载、多目标输入、目标验证、飞行控制和 SAM3 服务文件
均未回退。探索前沿决策已回到 GitHub v3，目标导航的 A*/EGO 和控制器保持现有实现。
最终地图的连续编号与展示功能保留。

运行方式仍为 `python run_explore.py`，识别后端沿用 `config/explore.json`，不需要 CMake。
探索使用 v3 占据建图和实例关联。

回退前的共享模块备份位于 `data/rollback_backups/before_explore_v3_20261029/`。
此修改不重写已有地图，新探索和回环语义重放使用相同的车辆关联。
