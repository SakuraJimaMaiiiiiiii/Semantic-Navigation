# SPIN 场景图代码调研与接入方案

核查日期：2026-10-24。本文为代码调研与接入设计，尚未安装 Spark-DSG 或切换运行链路。

## 官方代码与范围

没有找到可直接安装、完整复现 2020 年 SPIN 系统的独立官方仓库；这不等于证明代码从未公开。
官方 Kimera 索引引用 SPIN 论文，但公开模块列表是 VIO、RPGO、Mesher 和 Semantics。
不能把 Spark-DSG 当作 SPIN 完整算法。

- SPIN 论文：https://arxiv.org/abs/2002.06289
- 官方模块索引：https://github.com/MIT-SPARK/Kimera
- 图结构库：https://github.com/MIT-SPARK/Spark-DSG
- 在线构图系统：https://github.com/MIT-SPARK/Hydra

本次 GitHub API 返回的 main 提交：
- Spark-DSG：3c40997e4ae739c56ed3cdd6ef5e882d403d8b64
- Hydra：2e58a35baea629eee8838409771876acff015daf
接入应锁定提交并重新检查接口，不能默认旧示例与最新绑定完全一致。

## 实际查阅入口

Spark-DSG：
- include/spark_dsg/dynamic_scene_graph.h：图层、节点、层内/层间边。
- include/spark_dsg/node_attributes.h：节点属性入口。
- examples/python_api.py：读图、节点访问、图层遍历、可视化示例。
- CMakeLists.txt：CMake >=3.22、C++20、Eigen3、nlohmann_json；
  Python 与 ZMQ 可选。禁用 ZMQ 后可避开该路径的 pkg-config 依赖。
  核心 CMake 未要求 ROS；Windows/MSVC 实际编译尚未验证。

Hydra：
- include/hydra/places/graph_extractor.h 与 src/places/graph_extractor.cpp：
  基于 GVD 结构的稀疏地点图提取、压缩及连边。
- include/hydra/rooms/room_finder.h 与 src/rooms/room_finder.cpp：
  findRooms(places) 和 addRoomPlaceEdges(graph)。
这些不是接收任意点云就能独立工作的函数，依赖 Hydra 的地图类型、GVD 和配置体系。

## 对接现有代码

1. localization/fastlio_semantics.py
   在逐帧测量完成时间匹配、坐标转换后提供观测事件。
   必须保留 mapper.update 返回的实例关联结果、source_instance_id、观测位姿/时间、
   部分表面几何及关联状态，而不能只留下最终 records。
   新增观测证据应引用现有测量存储，避免向图 JSON 重复塞入每帧全部点云。

2. localization/fastlio_live_map.py
   当前 cells 只保存回波体素，没有 FREE/UNKNOWN 区分。
   先在 LIO 坐标下用同帧传感器原点和点云射线更新独立占据地图，
   复用 mapping/global_sparse_occupancy_map.py 的积分接口。
   全高度二维回波投影不能直接作为飞行高度的占据图。
   这一阶段不替换导航当前使用的地图。

3. navigation/scene_memory.py
   已有 nodes/edges、物体关系、历史地点。
   可将图表示抽出为独立层，接收统一帧的 records、观测事件与地图快照。
   历史 traversed_to 保留历史含义；只有当前地图/A*验证成功才生成 traversable_to。
   不混用旧 PX4 的 place 坐标与新 LIO 的 object 坐标。

4. mapping/semantic_voxels.py
   将物体表面标签和实例 ID 接到同一 LIO 占据版本。
   图节点、体素标签与 HTML 共享实例 ID，保留 merged_from 的重定向。

5. 图适配器（拟新增 mapping/scene_graph_adapter.py）
   第一版建立 object/place 层及观测证据引用，区域层随后加入。
   Spark-DSG 作为可选输出后端；本项目 JSON 不宣称兼容其原生序列化格式。
   Spark-DSG export/load round-trip 要检查节点/边/属性，不能直接改 JSON 字段冒充兼容。
   关联证据不足的实例不能因距离近自动合并。

## 推荐实现次序与验收

A. 固定现有 LIO 语义输入，建立物体图与观测引用。
   检查 HTML/JSON ID 一致，实例合并后无悬空边。
B. 补齐 LIO FREE/OCCUPIED/UNKNOWN 地图，再生成地点层。
   使用同一坐标系中的现有 A* 计算路径边、路径代价及净空；
   空间相近不自动连边，未知区域不作为可通行。
C. Spark-DSG Windows 构建小规模验证：静态核心、Python 绑定、关闭 ZMQ/示例；
   用真实的小图验证保存与重载，再考虑纳入默认依赖。
D. 地点连通图上划分区域，车库不强制套用办公室房间分类。
E. 最后单独接入导航坐标转换。回环 revision 改变时重新更新几何及边验证状态。

建议图元数据：schema_version、map_id、frame、map_revision、timestamp。
节点：稳定 id、layer/type、position/covariance、attributes、geometry_ref。
边：source、target、relation、证据引用、validated_revision、validity。
地图位姿统一修正，不等于回环已接入所有层；需要明确图版本。
车辆模型仍是表面证据，不能凭场景图自动补全遮挡车体。

## 本轮状态

完成资料和现有接口核查；仅新增本文档。
未下载整仓、未安装依赖、未修改飞控/定位/语义关联运行逻辑、未进行 Windows 构建。

## 阶段 A 已接入（2026-10-24）

新增 localization/scene_graph_adapter.py，保持纯图导出，不修改 mapper 的关联结果。
FAST-LIO live 自动新增 fastlio/scene_graph.json，并在 fastlio/map.html 提供链接。

图节点为 object 与 observation；measurement_of 边指向已有 LIO 实例。
保留原始测量文件的时间戳/样本序号引用、源 PX4 ID（单独命名空间）、
当时分配的 LIO ID、当前 canonical ID、协方差和坐标校正矩阵。
不复制所有表面点。未分配的观测不会被强行关联，已过滤的实例不会产生悬空边。
图 snapshot 的对象记录与现有 semantic_map.json、HTML 使用同一 records。
这里的 graph_revision 仅表示图观测更新次数，不宣称是几何回环版本。
目前没有可通行边，没有冒充 Spark-DSG 原生文件格式。

验证：Windows 17 项相关测试通过。
对 run_20261024_144818_484894 离线回放得到 6 个 object、457 个 observation、
260 条 measurement_of 边，197 条观测保持 unassigned，0 条悬空边。
新旧回放的 objects 内容完全相同，图中对象 ID 与语义地图一致。
结果在 data/fastlio_validation/run_20261024_144818_484894/scene_graph_stage1/。
未启动飞行；旧实验数据未覆盖。后续运行保持 python run_explore.py，无需编译。

阶段 B（LIO 自由空间与地点图）、阶段 C（Spark-DSG 构建）及区域/导航集成尚未完成。
先保留此阶段可单独验证的输出，再接入下一层。
