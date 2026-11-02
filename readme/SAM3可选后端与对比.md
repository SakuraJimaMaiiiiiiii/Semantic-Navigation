# SAM3 可选后端与对比

原 YOLO-World＋SAM2 保留且仍是默认值。本次新增的是官方 **SAM3 图像文字提示接口**，不是 Ultralytics SAM3 封装，也不是视频跟踪或 SAM3.1 接口。逐帧输出框、掩码和置信度，继续使用原深度／雷达定位、颜色估计、跨视角实例关联、占据标签和 HTML。SAM3 不直接分配永久地图 ID。

官方接口核对日期：2026-10-21。

- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py
- https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py

调用链为 `build_sam3_image_model` → `Sam3Processor` → 每帧一次 `set_image(PIL RGB)` → 每类 `reset_all_prompts` / `set_text_prompt`。框使用官方输出的原图像素 XYXY，掩码恢复到原始图像分辨率；BGR→RGB 只在客户端转换一次。对同类及重叠车型复用现有 NMS，索引同步筛选掩码。

## 使用原 Python 3.10 环境（图像接口兼容适配）

不需要创建环境，也不升级 Python、Torch、torchvision、NumPy、OpenCV、ftfy 或 Ultralytics。探索会在同一 Conda 环境自动启动后台 SAM3 服务，不需要另开终端。

直接运行 `python run_explore.py --sam3`。缺少权重时会从 `facebook/sam3` 自动下载 `sam3.pt` 到 `detections/Yolomode`；首次需在官方模型页面获得访问权限并在当前环境执行 `hf auth login`。下载耗时不计入模型启动的 180 秒限制，本地已有权重时不重复下载。程序等待模型加载和预热后才继续启动探索，退出时关闭本次启动的服务。已有服务则直接复用，退出探索不会关闭外部启动的服务。原 YOLO-World＋SAM2 启动方式不变。

```powershell
conda activate Segm-Navi-python3.10
python tools/setup_sam3_image.py
python tools/sam3_service.py --check-only
```

本次已在该环境执行安装与检查；无需重复安装。setup 脚本固定官方 commit `7f473cfd983d5eb43040ff608f66bff0d7bdcb53`，源码存于 `.tools/sam3`。它只对缺失依赖执行 `pip install --no-deps`，保留已有包，并记录核心依赖前后版本。不执行官方包的 `pip install -e`，避免其 NumPy `<2` 元数据触发自动降级；服务入口显式加载适配后的源码。

兼容改动仅涉及：

- `model_builder.py`：视频跟踪导入推迟到视频构建函数内；交互分割导入放在启用交互功能的分支内，文字提示图像模式不触发。
- `sam3_image.py`：仅供类型注解使用的交互预测器不在运行时导入；数据类型从原始定义模块导入，避免连带加载训练数据集。
- `perflib/nms.py`：复用官方已有的 CPU 贪心掩码 NMS，索引返回原设备，不依赖 Triton。此分支可能比定制 CUDA 核函数慢。

没有改模型结构、权重、检测评分或掩码计算。默认图像模型不启用交互分割或编译；CUDA 推理使用 BF16 autocast，记录在服务元数据。视频/交互模式不属于本次适配支持范围。

补装依赖：huggingface-hub、iopath、einops、portalocker、timm、safetensors、hf-xet，以及 Windows 上的 pywin32。现有 NumPy 实际导入为 2.2.5；环境原本留有 1.26.4 与 2.2.5 两份 metadata，pip check 可能因此报告 OpenCV 冲突。本轮未擅自删除旧 metadata，setup 报告会保留两份版本记录。

官方 README 的 Python >=3.12 与其 pyproject 的 >=3.8 不一致。本项目兼容入口接受 Python >=3.10；“语法可解析、导入成功”不等于所有官方功能可运行。

可额外运行不下载权重的完整模型构建／前向检查：

```powershell
python tools/sam3_service.py --smoke-test
```

该模式使用未训练参数，仅验证运算链路，运行完退出，绝不会启动预测服务或给导航输出随机识别结果。

正式识别需要官方 `sam3.pt` 权重，放在与 SAM2 相同的目录：

```text
E:\files\Semantic Navigation\detections\Yolomode\sam3.pt
```

默认路径根据项目目录解析，移动项目后仍然有效。启动时无需再填写权重路径：

```powershell
python tools/sam3_service.py
```

权重路径必须真实存在，不要使用 SAM2 或 SAM3.1 multiplex 权重。如果尚未下载，先在 https://huggingface.co/facebook/sam3 申请访问权限，下载官方 `sam3.pt` 并放入上述目录。其他存放位置仍可使用 `--checkpoint "实际路径\sam3.pt"` 指定。`--check-only` 和 `--smoke-test` 不要求权重文件存在。

上述手动启动方式适用于独立服务或离线对比；看到 `SAM3 ready: http://127.0.0.1:8878` 后即可在另一终端进行离线对比。在线探索无需这一步，会自动启动服务。服务仅监听本机，客户端绕过系统代理。

## 先做同帧离线对比，不需要 UE

在原来的导航环境执行。两次使用同一个 H5、同样的 `--step`、`--frames` 和提示词，输出目录必须不存在：

```powershell
python tools/compare_perception.py "data/runs/run_20261021_145419_673636/recordings/flight_rgbd_20261021_145419_680897.h5" --backend yolo_sam2 --object-prompts car "fire hose cabinet" "fire extinguisher" --output "data/comparison/yolo_sam2"
python tools/compare_perception.py "data/runs/run_20261021_145419_673636/recordings/flight_rgbd_20261021_145419_680897.h5" --backend sam3 --object-prompts car "fire hose cabinet" "fire extinguisher" --output "data/comparison/sam3"
```

上面的 H5 文件名请替换为实际录像路径；也支持 MP4。默认每 100 帧抽取一帧，共最多 30 帧。该工具不创建无人机连接、不发送场景命令、不修改原实验数据。

输出：相同编号的预览 JPG、掩码 NPZ、`report.json`（框、类别、分数、每帧耗时、平均/P95、去掉首帧的平均耗时、配置）。这是前端视觉对比，不是离线语义地图回放。没有标注真值时，不能把检测数量当准确率；重点人工比较漏检、误检、掩码是否泄漏到墙面以及相邻车辆是否粘连。

YOLO＋SAM2 保留原来的分割预算（默认每帧最多 6 个提示）；SAM3 返回已分割的检测结果，不再重复调用 SAM2。两者的模型分数也不保证同等校准。不要直接将分数大小、右侧“掩码＋建图”耗时当作模型速度或准确率的公平结论。比较总耗时，并在测试 YOLO 时关闭 SAM3 服务以避免显存占用干扰。

## 在线探索切换

原流程（默认只识别车辆）：

```powershell
python run_explore.py
```

车辆、消防箱和灭火器对比，使用相同提示词：

```powershell
python run_explore.py --sam2 --object-prompts car "fire hose cabinet" "fire extinguisher"
python run_explore.py --sam3 --object-prompts car "fire hose cabinet" "fire extinguisher"
```

两次分别运行，不能同时控制同一架飞机。SAM3 服务需提前启动。两个后端都可用 `--object-prompts` 修改类别；提示词在 mapper 内规范化，如 `fire hose cabinet` → `fire cabinet`。消防箱和灭火器继续经过原有多帧确认门槛，不会看到一次就正式标到地图。

每次运行 `detections/perception_backend.json` 保存后端、提示词、完整配置及服务模型信息；视频分别命名为 `yolo_world_detection.mp4` / `sam2_segmentation.mp4` 或 `sam3_detection.mp4` / `sam3_segmentation.mp4`。`perception_performance.json` 保存后端和总处理耗时；SAM3 的 `detector_ms` 包含 HTTP 往返及检测分割，后面的 `mask_geometry_association_ms` 是共同定位关联流程。

SAM3 请求有 8 秒超时，不会悄悄回退到 YOLO，也不会把上一次掩码套到新图像。先离线测量速度；若明显慢于原感知容许的新鲜度，暂勿用于飞行，不要靠放宽地图或感知超时来掩盖性能不足。

## 文件与验证

- `tools/sam3_service.py`：同环境可运行的官方图像模型服务。
- `tools/setup_sam3_image.py`：固定源码版本、可复现兼容补丁及缺失依赖安装。
- `detections/sam3_client.py`：本机通信、框／掩码同步过滤。
- `detections/object_detect.py`：提取公共 `SemanticMaskProcessor`，保留原 `SAM2Segmenter`。
- `detections/realtime_detection_worker.py`：后端选择及输出标识。
- `application/demo_runtime.py`、`run_explore.py`：配置注入与启动参数。
- `tools/compare_perception.py`：同帧离线视觉对比。

已运行 CPU 单元测试与本机 HTTP 集成测试；未下载授权权重，未进行有训练权重的识别质量评估或 UE 实飞。因此当前验证的是接口接线、数据形状、颜色通道、掩码索引及原后端兼容性，不是识别质量或实时性。

## 本次原环境实测结果

在 Windows `Segm-Navi-python3.10` 中完成：

- Python 3.10.20、Torch 2.11.0+cu128、实际 NumPy 2.2.5、ftfy 6.3.1。
- 官方图像 builder / processor 导入、三类文本 tokenizer、CUDA 输入的官方 CPU NMS 测试通过。
- 未训练官方完整模型在 RTX 5060 上完成三提示词前向；以 0 阈值强制走非空输出路径，得到形状 `(600,405,720)` 的掩码，峰值 CUDA 分配约 3.95 GiB；未加载 Triton。这里只验证运算兼容性，600 个随机候选不是识别结果，也不是性能基准。
- 34 项单元／接口测试通过。
- 此前原环境已有两份 NumPy dist-info，安装脚本核对并保留两份；所有受保护包的安装记录前后相同。

真实 SAM3 权重尚未提供或下载，因此车辆/消防箱准确率、实际显存峰值与在线时延仍待后续同帧对比验证。
