# Semantic Navigation

RflySim 地下车库 RGB-D 感知、语义实例地图和无人机避障导航。

- [运行开关与启动顺序](readme/运行开关与模式说明.md)
- [P0–P1 语义导航目标接口与验收](readme/语义导航任务接口与验收.md)

按顺序启动 `npm start`、UE/RflySim，再运行 `python run_demo.py`。

`config/semantic_target.json` 默认 `{"target": null}`，执行原有航点任务。指定类别或运行时实例 ID 后，启用语义目标查询、安全接近、搜索和到达复核。请求格式、Python 接口和评估命令见接口文档。

新的两阶段流程：`python run_explore.py` 以前沿信息增益探索并返航后保持待命，在另一终端执行 `python navigate_target.py --open-map`。详见 [前沿探索与两终端导航](readme/深度优先探索与两终端导航.md)。
