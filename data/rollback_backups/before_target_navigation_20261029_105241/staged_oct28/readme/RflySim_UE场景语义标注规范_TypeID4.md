# RflySim UE 场景语义标注规范（TypeID=4）

## 1. 目的与交付边界

本文档用于指导 Unreal Engine 场景技术人员为本项目的简单地下停车场制作 RflySim `TypeID=4` 像素级任务标签。地面、墙、柱和其他设施采用固定语义 ID，只有每一辆车采用独立实例 ID；白名单之外的对象不要求分配语义类别。

TypeID=4 不运行识别模型。它读取每个可渲染组件的 `CustomDepthStencilValue`，再由 RflySim 的分割材质把不同 Stencil ID 渲染为不同颜色。因此，真值质量完全取决于 UE 场景标注质量。

本项目采用“语义分割 + 车辆实例分割”的混合约定：除车辆外，同类物体共享语义 ID，例如所有墙使用同一个 ID、所有柱使用同一个 ID；每一辆车使用不同的 Stencil ID。车辆实例 ID 的具体含义通过 Actor Tags 和交付映射文件记录。

满足以下交付目标：

- 第 3 节白名单中所有能被RGB相机看到的目标物体均已分类。
- 白名单之外的对象、天空、场景外部空域、不可见辅助体和视觉特效统一使用 ID 0，不再扩展类别。
- 固定语义类别在所有关卡、子关卡、LOD 和蓝图实例中保持统一；同一辆车的所有组件、LOD 和动态状态始终使用该车的唯一实例 ID。
- TypeID=1、TypeID=2、TypeID=4 的分辨率、FOV、安装位置和姿态完全一致。
- 打包后的 RflySim3D 场景仍能输出正确分割图，不能只在 UE Editor 中有效。

## 2. UE 项目全局设置
供参考

在 UE Editor 中打开：

```text
Edit
→ Project Settings
→ Engine
→ Rendering
→ Postprocessing
→ Custom Depth-Stencil Pass
→ Enabled with Stencil
```

修改后重启 Editor，并重新检查设置没有恢复为 `Disabled` 或 `Enabled`。

所有参与标注的 `PrimitiveComponent` 必须设置：

```text
Render CustomDepth Pass = True
CustomDepth Stencil Write Mask = Default / 255
CustomDepth Stencil Value = 本文类别表中的 ID
```

`StaticMeshComponent`、`SkeletalMeshComponent`、`InstancedStaticMeshComponent`、`HierarchicalInstancedStaticMeshComponent` 和蓝图中的子 Mesh 都必须检查，不能只设置最外层 Actor。

## 3. 地下停车场类别与 Stencil ID

ID 0 用于白名单之外的对象、天空、场景外部空域、不可见辅助体和视觉特效。ID 1～17 是固定语义 ID，只有车辆使用 32～199 的实例 ID 池。

### 3.1 固定语义 ID  参考

| ID | 标准类别名 | 地下停车场对象示例 |
|---:|---|---|
| 0 | background | 白名单之外的对象、天空、场景外空域、辅助体和特效 |
| 1 | floor | 所有地面 |
| 2 | ceiling | 所有天花板 |
| 3 | wall | 所有外墙和墙段 |
| 4 | column | 所有柱子 |
| 5 | beam | 横梁、结构梁、门楣、支撑梁 |
| 6 | curb_wheel_stop | 路缘、挡车器、限位墩 |
| 7 | parking_marking | 停车位线、停车框线、编号底色 |
| 8 | door_gate | 门、卷帘门、防火门、检修门、普通门扇 |
| 9 | guardrail | 车辆防撞护栏、金属防撞栏 |
| 10 | pipe | 消防管、给排水管、燃气管和其他管道 |
| 11 | duct | 通风管、风道、排烟管道 |
| 12 | fire_equipment | 灭火器、消火栓箱、水带箱、消防器材 |
| 13 | signboard | 指示牌、广告牌、楼层牌、区域牌、悬挂牌 |
| 14 | traffic_sign | 限速牌、禁行牌、停车标志、交通标牌 |
| 15 | barrier_arm | 道闸杆、升降杆、横向拦截杆 |
| 16 | speed_bump | 减速带、减速垄 |
| 17 | person | 行人、工作人员、驾驶员、骨骼人物 |

### 3.2 车辆实例 ID

- ID 32～199 为车辆实例池，最多直接容纳 168 辆车。
- 每一辆车必须使用一个未占用的 ID；即使两辆车属于同一车型，也不能共享 ID。
- 同一辆车的车身、车门、车窗、车轮、全部 Mesh Component 和全部 LOD 使用相同实例 ID。
- UE 开发人员可以按照关卡顺序、停车位顺序或资产清单顺序分配车辆 ID，不强制指定某辆车必须使用哪个具体编号。
- 车辆类型写入 `VehicleType`，可取 `car`、`suv`、`van` 或 `truck`。
- 车辆被移动、隐藏再显示或重新加载时必须保持原 ID。运行时新生成车辆使用预先登记的空闲 ID，销毁后本次运行内不得立即复用。

车辆映射示例：

| Stencil ID | VehicleInstance | VehicleType | Actor 示例 |
|---:|---|---|---|
| 32 | vehicle_001 | car | 轿车、出租车或警车 |
| 33 | vehicle_002 | suv | SUV、越野车或皮卡 |
| 34 | vehicle_003 | van | 面包车、厢式客车或小型厢式货车 |
| 35 | vehicle_004 | truck | 卡车、货车或工程运输车 |


### 3.3 预留区与开发自由度

| ID 范围 | 用途 | 开发人员权限 |
|---:|---|---|
| 1～17 | 固定语义 | 不得改变已有含义，不得分配给车辆实例 |
| 18～31 | 固定语义扩展 | 场景确有必要时可增加少量类别，但必须写入映射和变更记录 |
| 32～199 | 车辆实例 | 可自由分配给具体车辆，只需保证唯一、稳定并登记车型 |
| 200～239 | 项目扩展 | 可用于新增实例或特殊对象；必须说明用途并确认不与现有 ID 冲突 |
| 240～254 | 临时调试 | 可在开发期用于漏标检查，正式交付前必须清零 |
| 255 | 保留 | 不分配，避免与全位掩码或后处理约定混淆 |

开发人员可以决定车辆 ID 的具体分配顺序，也可以在预留区增加确有价值的类别，不必为每次小调整重新排列全部 ID。但必须满足三条底线：已有 ID 不改义、同一车辆 ID 不重复表示不同车辆、所有扩展都进入机器可读映射和变更记录。

灯具、普通玻璃、车道线、箭头、楼梯、排水设施、电缆、风机、摄像头、家具、杂物、植被及其他未列对象默认保持 ID 0。若开发人员认为其中某类对任务有价值，可使用 18～31 或 200～239 扩展，而不是改动固定 ID 或占用车辆实例池。

## 4. 白名单对象标注规则

### 4.1 结构与组合网格

- 所有地面统一使用 `floor=1`，所有墙统一使用 `wall=3`，所有柱统一使用 `column=4`；天花板使用 `ceiling=2`，梁使用 `beam=5`。
- 如果一个 `StaticMeshComponent` 同时包含地面和墙，Stencil 无法在组件内部区分，必须在 DCC 或 UE 中拆成不同 Mesh Component。
- 同类墙或同类柱可以合并并共享 ID；不同语义类别不能合并到只能写一个 Stencil 值的组件中。
- 合并网格、HLOD、Level Instance 和运行时生成的代理网格必须保持原类别，不能因为性能合批而退回 ID 0。
- 所有 LOD 使用同一类别 ID，切换 LOD 时分割颜色不得变化或消失。

### 4.2 蓝图与动态物体

- 蓝图 Actor 的每个可见子 `PrimitiveComponent` 都要写入 Stencil，不能只标 Blueprint 根节点。
- 一辆车内部的车身、车门、车窗和车轮统一使用该车的实例 ID；不同车辆不得共享实例 ID。
- 人物的身体、衣服、头发和随身骨骼附件默认统一使用 `person=17`；本项目暂不区分人物实例。
- 生成、复制、换装或对象池复用后必须重新保证 Stencil 属性有效。
- 门、道闸、车辆和人物在运动、动画和物理模拟期间不得丢失标注。

### 4.3 车位线与特殊材质

- 车位线、停车框线和编号底色统一使用 `parking_marking=7`。
- Decal 通常不能独立写入 Custom Depth。上述停车标线应改为薄 Mesh，或使用能够稳定写入 Custom Depth/Stencil 的实现。
- 门上的玻璃仍随门使用 `door_gate=8`，车辆玻璃仍随所属车辆使用相同实例 ID，不拆分玻璃类别。
- 灯具、普通玻璃、镜子、粒子、雾、光晕和纯后处理效果不属于白名单，保持 ID 0。

### 4.4 实例化与重复物体

- ISM/HISM/Foliage 中同一组件的实例只能稳定共享类别时才可合并。
- 重复柱、挡车器、消防设施和标牌可以共享固定语义 ID，但白名单对象的每个组件都必须启用 Custom Depth。
- 不同车辆不能合并到只能共享一个 Stencil 值的 ISM/HISM 组件中；必须保证每辆车可以独立设置实例 ID。
- 白名单之外的实例化资产不需要启用 Custom Depth，保持 ID 0。

### 4.5 白名单外对象和不可见对象

下列对象不需要为了消除黑色而开启 Custom Depth 或分配语义类：

- 第 3 节白名单以外的所有可见场景对象。
- Collision Volume、Blocking Volume、Trigger Volume。
- NavMesh、导航链接、样条控制点和编辑器辅助图标。
- 相机、PlayerStart、灯光 Actor 本身的不可见控制对象。
- 调试线、坐标轴、Bounds 和编辑器 Gizmo。
- 以上对象、天空和场景外空域统一保留 ID 0。


## 5. 推荐实施流程

1. 冻结当前关卡版本，记录 UE、RflySim 和场景提交版本。
2. 导出 World Outliner 中所有 Actor，并建立“Actor/组件—类别或车辆实例—Stencil ID”资产清单；先锁定固定类别和已使用 ID。
3. 标注结构：同类地面、墙、柱、天花板和梁分别使用各自固定语义 ID。
4. 标注固定设施：护栏、管道、风道、消防器材和标牌。
5. 标注路缘/挡车器、停车位线、道闸和减速带；再为每辆车从 32～199 分配唯一 ID，并填写 `VehicleInstance` 和 `VehicleType`。
6. 标注人员，并测试车辆、门、道闸和人物的运行时生成与动画状态；确认车辆运动前后 ID 不变。
7. 处理停车标线 Decal、实例化组件、合并网格和蓝图子组件等特殊项。
8. 在 Editor 和打包后的 RflySim3D 中逐区域检查 TypeID=4，确认打包没有剥离 Custom Depth/Stencil 设置。

对于大型场景，建议制作 Editor Utility Widget 批处理：读取 Actor Tags，遍历其全部 `PrimitiveComponent`，统一设置 `Render CustomDepth Pass=True` 和对应 `CustomDepthStencilValue`。地面、墙、柱等固定类别可以按类别批量赋值；车辆必须逐实例分配唯一 ID，不能把全部车辆一次设置成相同值。批处理完成后仍必须人工检查，因为名称和 Tags 可能错误。

