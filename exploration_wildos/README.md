# WildOS 导航图：静态构图与增量快照更新

已实现静态构图、车辆语义与几何约束，以及批次 B 的增量快照更新、历史探索半径和前沿生命周期。尚未接入实时探索回调。

**已完成合成场景验证，以及 `run_20261103_134810_164283` 的真实探索保存地图静态验证。** 真实地图结果见 `data/runs/run_20261103_134810_164283/exploration/navigation_graph/README.md`。当前模块没有接入 `run_explore.py`，不会启动仿真、PX4、感知模型或发送飞行指令。


当前默认采用论文对齐采样 + 节点压缩 + 实际稀疏连边，详见 [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)。下文历史覆盖适配模式的全覆盖和同轮去重描述不适用于 paper_aligned 模式；当前真实运行指标以该次 navigation_graph/README.md 为准。

## 统一入口与目录组织

所有 WildOS 功能代码和命令入口集中在本目录，`tools/` 中的旧转发脚本已移除。统一使用 `python -m exploration_wildos` 或本包的子模块入口；测试仍遵循项目约定放在 `tests/`。

每次 explore 完成后执行：

```bash
python3 -m exploration_wildos --exploration-dir data/runs/run_20261103_134810_164283/exploration
```

新运行时替换路径。默认从现有安全图初始化或恢复 `navigation_graph/incremental_state.json`，处理同一坐标系的下一份完整快照，再更新同一个 `map.html`。同一快照重复执行不会新增节点或推进版本；过期时间、坐标系或高度配置变化会拒绝。没有增量状态的旧图若与源地图不匹配，仍拒绝混合。

```text
exploration_wildos/                 # 功能实现
  __main__.py                      # 一条命令处理一次运行
  build_snapshot_graph.py          # 离线构图入口
  combine_map.py                   # 合并页面及来源验证
  combined_map.py                  # 上下布局、对齐、缩放与平移
  map_adapter.py
  static_graph.py
  incremental_graph.py             # 稳定身份、历史半径、前沿生命周期
  replay_graph.py                  # 同坐标系连续快照回放
  node_reduction.py                # 保覆盖与连通性的节点去冗余
  graph_controls.py                # 共享节点选择与半径轮廓控件
  graph_export.py
  make_demo_map.py
  audit_real_map.py

data/runs/<run_id>/exploration/     # 该次运行的地图与图结构
  map.html                         # 上：探索地图；下：导航图结构
  map.exploration.original.html     # 合并前的探索页面备份
  occupancy_2d.npz
  semantic_map.json
  private/navigation_cache.npz
  navigation_graph/
    graph.json
    incremental_state.json         # 可恢复的增量状态
    graph.html                     # 单独图页面，备用
    validation.json
    audit.json                     # 实际执行核查后生成
    graph_preview.png              # 本次预览
    default_budget/                # 本次参数对照结果
    completion_audit_*/             # 本次完成情况复核
    ui_verification/               # 页面浏览器检查与截图
```

两张图使用相同的起点朝向和显示范围，各自支持按钮放大/缩小、滚轮缩放、拖动平移、双击复位及“适应全图”。导航图支持图层开关，探索地图保留物体点击高亮。

合并的是已保存结果，因此移除了原页面每两秒自动刷新的标记，避免缩放被重置。若现有运行程序随后重新导出 `map.html`，再次执行统一入口即可恢复上下视图；当前尚未接入在线图更新。

合成示例统一存放在 `data/wildos_examples/`；开发基线备份统一存放在 `data/wildos_backups/`。旧报告目录只保留迁移说明，不再保存本次地图和图数据。

## 已实现内容

- 读取 `private/navigation_cache.npz`、全局稀疏占据 NPZ 和 `occupancy_2d.npz`。
- 复用当前 `NavigationGrid.from_global()` 的高度带投影；未知体素和阈值中间值不能作为自由证据。
- 二维导出使用原始状态重新计算安全区域，避免重复膨胀，不直接信任 `safe_indices`。
- 有界障碍/未知距离场；扣除网格单元外接圆半径形成保守距离下界。
- 固定随机种子的节点采样，同轮去重，以及优先低净空位置的有界补充采样。
- 使用 `NavigationGrid.segment_free()` 验证每条边，并计算连通分量。
- 可选起点接入；无法安全连接时返回明确状态。
- 输出节点、边、半径、覆盖不足位置、参数、输入 SHA-256、验证结果以及离线 HTML。

静态入口 `build_snapshot_graph` 的前沿字段仍为空；默认统一入口会调用增量模块填充 `frontier_ids`、`is_frontier` 和历史 `explored_radius`。跨快照引用使用 `stable_id`；`id` 只是当前导出中的连续索引。

## 依赖

使用能够导入本项目 `navigation.route` 的 Python 环境，新增模块直接依赖 NumPy、SciPy；测试使用 pytest。HTML 不依赖网络和 Plotly。

在仓库根目录执行下列命令。Linux 使用 `python3`，Windows 对应环境通常使用 `python`。

## 立即查看已生成结果

- `data/runs/run_20261103_134810_164283/exploration/navigation_graph/graph.html`：真实探索地图，论文对齐模式：222 个规划节点、647 条实际边（原采样图为 798 个节点、43246 条候选边），10032 / 10367 个受约束安全格被自由圆覆盖；20 个前沿点关联到 16 个节点。
- `data/wildos_examples/synthetic_stage_a_20261103/door/graph.html`：门洞、车辆形状障碍和柱子。
- `data/wildos_examples/synthetic_stage_a_20261103/wall/graph.html`：墙体隔开的两个房间。
- `data/wildos_examples/synthetic_stage_a_20261103/dead_end/graph.html`：可进入并退出的死胡同几何。

浏览器直接打开即可。勾选框控制各图层，缩放后可以滚动查看；鼠标悬停节点显示 ID、连通分量和自由半径。

这些是合成场景。死胡同示例仅验证静态连通结构，不表示已经实现自主退出或历史探索记忆。

## 重新生成一个独立示例

输出路径必须尚不存在，工具会拒绝覆盖已有文件/目录。

```bash
python3 -m exploration_wildos.make_demo_map --scenario door --output data/wildos_examples/my_demo/inputs/map.npz
python3 -m exploration_wildos.build_snapshot_graph data/wildos_examples/my_demo/inputs/map.npz --altitude -1.5 --start-ned -5 0 -1.5 --output data/wildos_examples/my_demo/graph
```

`--scenario` 还支持 `wall`、`dead_end`。默认单高度图参数：自由半径上限 2 m、连边半径 3 m、随机采样预算 2000、补充采样预算 2000、节点上限 1000。这些是实验初值，不代表论文参数或已标定值。

## 有探索结果后如何读取

推荐先读取探索目录中的 `occupancy_2d.npz`，它记录了导出高度和安全配置。以下 `<探索目录>` 为占位符，需要替换为实际路径：

```bash
python3 -m exploration_wildos.build_snapshot_graph "<探索目录>/occupancy_2d.npz" --output "<探索目录>/navigation_graph"
```

三维归档需要明确提供分析高度，例如：

```bash
python3 -m exploration_wildos.build_snapshot_graph "<探索目录>/private/navigation_cache.npz" --altitude -1.5 --clearance 0.40 --vertical-half-extent 0.25 --output "<探索目录>/navigation_graph_3d"
```

上例的高度、水平净空和垂直范围必须替换成该次运行的实际值。`-1.5` 是 NED z，不是无条件适用于任意起点的离地高度。三维输入省略安全参数时使用 0.40 m / 0.25 m，并将实际值写入输出。

`--start-ned x y z` 是可选参数，必须提供同一地图坐标系下的位置；省略时不评估机器人接入，不会把返航位置或原点当作起点。

二维投影不能重指定其他高度或高度带。三维归档按项目约定解释为 NED；工具不进行重定位，也不赋予旧地图在线导航权限。

## 输出文件与判读

| 文件 | 内容 |
|---|---|
| `graph.json` | 图、来源、参数、组件、机器人接入和覆盖统计 |
| `validation.json` | 全部节点与边的同高度安全复核 |
| `graph.html` | 自包含交互可视化 |

未知背景为灰蓝色，原始自由区域为浅色，安全自由区域为白色，障碍为深灰；黄色表示安全区域中未被节点自由半径覆盖的格子。节点颜色表示连通分量。

`validation.passed` 只说明输出图中的节点/边通过现有网格检查，不能证明图覆盖充分、真实飞行安全或探索任务完成。空图的检查也可能通过，需同时查看节点数、自由格数与覆盖统计。

覆盖统计衡量几何半径覆盖，不能代替相机可见性、真实观测覆盖率或目标发现率。

## 测试

```bash
python3 -m pytest -q tests/test_wildos_static_graph.py
```

测试包括：缺失高度证据、模糊占据值、障碍否决、负坐标、投影高度限制、重复体素拒绝、距离场边界、超大窗口拒绝、墙/未知带阻断、门洞连通、起点接入、确定性、JSON 往返和输出防覆盖。

## 当前限制和下一步

1. 已验证一份真实探索保存地图的坐标投影、覆盖、连通性和静态运行开销；尚需更多场景及地图前缀验证，不能推广为在线导航或完整探索验收。
2. 当前读取的是静态全局归档，未融合在线局部快照；不能声称与实时混合地图完全一致。
3. 尚未实现前沿关联、历史 ID、旧边更新与回环重建处理；下一批处理这些生命周期。
4. 大范围稀疏地图若包围框超过默认 400 万格会明确拒绝；尚未实现分块距离场。
5. 图的连通性受采样预算和连边半径影响；出现多个分量时需用现有 A* 分析几何隔断还是采样不足。
6. 尚未实现在线旁路、探索策略切换、语义关联及多高度飞行。

## 基线与停用

基线保存在 `data/wildos_backups/baseline_20261103_133606/`：包含 HEAD 信息、原有已跟踪改动补丁、暂存区补丁、未忽略的未跟踪文件副本与关键代码哈希。被忽略的传感器归档不在备份范围内。

所有实现都是新增文件；现有运行入口和配置未接入该模块。停用时停止运行两个离线脚本即可。恢复旧工作区时应先检查备份补丁范围，不应自动将整个补丁覆盖到已有新改动上。

## 节点去冗余与半径显示（2026-11-03）

静态构图默认启用保守节点去冗余：保持安全格中心覆盖、连接关系和起点接入；逐次删除检查邻居路径的绕行代价。节点位置与半径不变，允许圆圈重叠。使用 `--keep-redundant-nodes` 可关闭该步骤做对照。

实际地图节点395→348、边1426→1077，全部10530个安全格仍被覆盖。保留节点对路径平均增加约0.074%，最大增加约11.50%；局部删除限制不能解释为全局最短路不变。

显示默认仅展示悬停/选中节点的自由半径；支持全部轮廓、隐藏圆圈、清除选择和相邻连接突出显示。半径没有填充，不再遮挡地图结构。

当前结果在本次 `exploration/navigation_graph/`；优化前结果归档至其 `history/before_node_reduction_20261103/`。`sample_id` 记录保留节点在原采样列表中的编号，当前 `id` 重新连续编号；这仍不是跨地图稳定 ID。

### 车辆语义图层

运行统一命令 `python -m exploration_wildos --exploration-dir <探索目录>` 会创建或复用导航图，
再从同次探索的 `semantic_map.json` 添加车辆语义节点，重新输出合并的 `map.html`。
旧图及页面保存在该次 `navigation_graph/history/before_semantics_<hash>/`。

- `nodes/edges` 仍是可通行导航图；`semantic_nodes/semantic_edges` 单独保存非通行物体与关联。
- 每个车辆实例一个语义节点，保留实例 ID、类别、颜色、置信度、观测位置和已观测包围盒。
- 在观测锚点 6 m 内选择最多 3 个关联导航节点，优先选三维射线通过已知空闲体素的候选。
  射线检查排除终点表面所在体素；遇障碍则排除，经过未知体素则标为 unknown。
  即使 `clear_to_surface` 也没有验证相机朝向、视场或完整车辆可见性，`visibility_confirmed` 始终为 false。
  没有合适候选时车辆保留但没有关联，不强行跨越障碍连接。
- 橙色方形表示车辆，矩形轮廓表示已观测轴对齐包围盒，不代表完整车辆外形或新增禁入区。
  虚线是非通行关联：橙色表示射线路径已知空闲，灰色表示包含未知区域。
  点击车辆显示属性并高亮关联导航节点；图层可独立开关。
- 本功能为离线语义标注，不修改占据栅格、导航覆盖、通行连通性或飞行控制。

### 车辆禁越顶约束与定位隔离

当前策略为 `no_overflight_v1`：车辆已观测 AABB 的整个水平投影，在所有飞行高度均视为规划障碍；水平额外余量仍为 0.2 m（不超过配置净空）。高于车顶不再是允许通过的理由。车辆框覆盖格及余量格全部禁止生成导航节点或被导航边穿越。

`graph.json.vehicle_constraints.cell_owners` 为每个车辆禁入格记录车辆 ID，并区分 `observed_footprint` 与 `clearance_buffer`。重叠区域保留多个归属。这些是规划归属，不是修改后的传感器语义标签。原始点云/占据数组、车辆中心、三维观测框、实例关联和定位算法均不回写、不扩张。

下方图用实色浅橙表示车辆投影，框外余量为深灰，可查看车辆归属。上方保留原始观测地图；黑/灰色测量证据不被人为删除。新增区域不代表已经测得完整的隐藏车身，物理车辆外形仍以感知后续观测为准。

该策略用于新 WildOS 入口的探索图、局部执行复查、车辆补充观察、返航以及后续目标导航。返航使用原控制器的路线和执行逻辑，但输入规划副本增加车辆立柱障碍，并用执行检查拒绝越顶。原地图和定位记录保持独立。

当前最新运行的离线预览保存在 `navigation_graph/no_overflight_preview/`；历史飞行图和航迹不覆盖。原 `run_explore.py` 的行为没有更改。

## 增量更新、历史探索半径与前沿生命周期

默认统一入口现在启用增量快照更新，输入是同一坐标系、同一高度配置的完整地图快照。
首次可沿用当前已精简的安全图；后续保留节点位置与 `stable_id`，只在未覆盖的安全区域新增节点。
当前快照中的边全部复核，仍有效的端点对保留同一个边 `stable_id`，不能把旧边直接当作仍然安全。
增量阶段暂不再次做冗余精简，以免删除历史记忆或导致身份抖动；静态初始化仍可先精简。

- 当前自由半径：`min(障碍距离, 未知距离, free_radius_cap)`，可增可减。
- 历史探索半径：`max(旧 explored_radius, 当前未知距离)`，存活节点只增不减。
- 节点变为未知：暂时 dormant，不导出为可通行节点，保留历史；重新安全时恢复同一身份。
- 节点在已知区域中占据或失去净空：永久退役，相关边消失；后续新节点不会复用该稳定 ID。
- 前沿采用空闲/未知边界的未知侧单元。转为已知后移除；边界消失或无可关联节点时清理。
  新前沿落入已保留节点的历史探索圆时跳过，否则关联最近且有空闲几何视线的节点。
  所属节点变化记录 reassigned，节点无前沿点时取消 `is_frontier`。
- 前沿关联的视线终点在边界的已知空闲侧，按原始空闲格检查；不是飞到未知点的通行边，
  没有验证机体到达边界的净空，也没有验证相机视场。只把关联的安全导航节点作为图节点。
- 紫点表示前沿，紫环表示关联节点。半径类型可切换到历史探索半径。

显式放弃当前增量状态并重新静态构图（旧状态归档）：

```bash
python3 -m exploration_wildos --exploration-dir <探索目录> --static-rebuild
```

连续快照回放：

```bash
python3 -m exploration_wildos.replay_graph --manifest <清单.json> --output <新输出目录>
```

清单示例（路径相对清单文件；全局三维 NPZ 需提供 altitude）：

```json
{
  "frame_key": "same-session-uncorrected-NED",
  "options": {"max_samples": 12000, "coverage_samples": 12000},
  "snapshots": [
    {"map": "step_001/occupancy_2d.npz", "semantic_map": "step_001/semantic_map.json"},
    {"map": "step_002/occupancy_2d.npz", "semantic_map": "step_002/semantic_map.json"}
  ]
}
```

每步导出图与验证，最后写入可恢复状态及事件汇总；`--initial-state` 可继续已有回放。
省略 semantic_map 时不应用车辆语义约束，应仅用于纯几何输入。frame_key 是调用方对齐声明，
不会自动完成地图配准；回环修正地图必须先正确对齐，不能仅填相同字符串就视为同坐标系。
当前真实地图只有第一份增量状态，不从最终地图虚构过去探索历史。真实回环修正版本没有自动串联。
同次运行 `navigation_graph/replay_validation_synthetic` 是明确标记的合成回放验证，不是实际飞行录像回放。

## 当前默认：论文采样 + 实际稀疏连边

根据新的连边密度要求，统一入口保留论文采样/历史半径/前沿规则，但输出实际稀疏图。
首先从已验证的候选边中，为每个节点选择 3 m 内最近 4 条边（取无向并集，所以节点度数可能大于 4）；
再按长度补齐维持原连通分量所需的边；最后补回少量捷径，使每条原候选边都有不超过其长度 1.5 倍的替代路径。
因此任意原可达节点对仍可达，最短路长度不超过原密集图的 1.5 倍。少量边允许超过 3 m，但仍不超过原候选范围。
节点位置、半径、车辆和前沿关联不因连边稀疏化改变。

`graph.json.edges` 就是实际稀疏后的边；完整已认证候选边保存在 `incremental_state.json.candidate_edges`，
便于后续快照更新或恢复密集模式，不作为当前规划图的通行边。
`edge_sparsification` 保存参数和计数，`sparsification_comparison.json` 保存本次全节点对路径对比。
增量边变更事件按实际输出边集合计算，同一快照反复处理保持结果不变。

```bash
# 默认稀疏图
python3 -m exploration_wildos --exploration-dir <探索目录>
# 明确恢复论文范围内的全部安全候选边
python3 -m exploration_wildos --exploration-dir <探索目录> --dense-edges
```

这是对论文 Algorithm 5 全邻域连边的明确适配，目的为实际图结构稀疏，不仅是隐藏显示。

## 当前默认的节点压缩

生成规划图前，按自由圆覆盖贡献检查节点。只有该节点覆盖的所有安全格仍被其他节点覆盖，且删除不会拆开连通分量时，才允许删除；起点锚点和前沿所属节点受到保护。
删除后从原安全候选图的最短路径中补回必要中间节点，使保留节点对之间的路径长度不超过原采样候选图的 1.2 倍。
再对边采用 1.2 倍绕行限制，两阶段组合上界为 1.44 倍。实际路径变化记录在 compaction_comparison.json。
全节点对矩阵仅在不超过 2000 个采样节点时执行；超过时保留节点并报告 bounded_matrix_fallback，避免无界内存消耗。

这次输出从 798 节点/2173 边降为 222 节点/647 边，覆盖的 10032 个安全格与原来相同。
压缩只改变规划图，原始稳定 ID、位置和探索半径不被篡改；完整采样节点及历史单独保留在增量状态。
`incremental_state.json.uncompacted_graph` 保存压缩前采样图，`candidate_edges` 保存当前规划节点之间的完整候选边。
当前实际规划结构始终是 `graph.json.nodes/edges`。

```bash
# 默认紧凑图
python3 -m exploration_wildos --exploration-dir <探索目录>
# 保留所有采样节点，但仍精简边
python3 -m exploration_wildos --exploration-dir <探索目录> --keep-all-nodes
# 恢复所有采样节点及完整安全候选边
python3 -m exploration_wildos --exploration-dir <探索目录> --keep-all-nodes --dense-edges
```

保留节点的车辆关联会重新计算，前沿归属和起点通过稳定 ID 保留并重映射当前索引。
这属于用户要求的实际图结构优化，偏离论文不精简采样节点的几何构图伪代码。

## 导航图底图配色

下方导航图和独立 graph.html 默认采用三色底图：白色为当前安全自由格，黑色为占据障碍或安全净空不足区域（包含车辆安全约束排除区），灰色为未知区域。
车辆语义方块、观测轮廓、关联虚线以及黄色未覆盖诊断默认隐藏，可通过各自图层开关显示。
导航节点、通行边与前沿仍以彩色叠加显示。该调整只改显示，不修改占据档案、车辆高度判定或规划图。
上方原始探索地图保留原记录的语义显示方式。
