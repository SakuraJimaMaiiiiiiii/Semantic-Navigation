# Semantic Navigation

RflySim 地下车库 RGB-D 感知、语义实例地图和无人机避障导航。

- [运行开关与启动顺序](readme/运行开关与模式说明.md)
- [P0–P1 语义导航目标接口与验收](readme/语义导航任务接口与验收.md)
- [雷达回环检测](readme/雷达回环检测.md)
- [车辆身份关联与连接视角补拍](readme/车辆身份关联验证.md)

按顺序启动 `npm start`、UE/RflySim，再运行 `python run_demo.py`。

`config/semantic_target.json` 默认 `{"target": null}`，执行原有航点任务。指定类别或运行时实例 ID 后，启用语义目标查询、安全接近、搜索和到达复核。请求格式、Python 接口和评估命令见接口文档。

新的两阶段流程：`python run_explore.py` 探索、确认返航落地并保存后自动关闭后台进程；随后可运行 `python navigate_target.py --open-map` 独立加载保存地图导航。WildOS 对应 `run_wildos_explore.py` 与 `navigate_wildos_target.py`。需要旧的长期待命控制会话时，探索命令显式添加 `--keep-session`。详见 [前沿探索与两终端导航](readme/前沿探索与两终端导航.md)。

运行数据保存在 `data/runs/<本次运行>/`。默认关闭原始 RGB-D HDF5 录制、检测视频及周期性规划 PNG；需要调试时，分别在 `RecordingConfig.enabled`、`DetectorConfig.save_videos`、`AStarVisualizationConfig.enabled` 中显式开启。实时感知和最终导航地图仍正常生成。

WildOS 的 `navigation_graph/update_*` 和在线回环地图的 `live_loop_corrected/revision_*` 各只保留最近两份完整快照，成功发布新版本后清理旧版本。相同内容的 JSON/HTML 跳过重写；正常保存异常或取消时清理临时目录。含自添文件的快照、离线导出、不同运行的最终地图以及 `data` 中已有备份均保留。
