# 工作日志

## 2026-09-21

### 研究与资料

- 解压完整工程资料包和研究资料包。
- 确认项目范围为数字域 YOLO-Seg 掩码选择性退化。
- 以 `CODE_IMPLEMENTATION.md` 和技术路线文档作为实现依据。

### 环境

- 创建 Conda 环境 `yolo-mask-selective`。
- 固定 Python 3.11、PyTorch 2.5.1、TorchVision 0.20.1 和 CUDA 12.1。
- 固定 Ultralytics 8.4.157，并使用参考仓库 commit `00be778`。
- 修复 Pillow/libtiff 二进制兼容问题。
- 在 Tesla V100-SXM2-16GB 上完成 CUDA 张量运算验证。

### 工程

- 建立 `src/yolo_mask_attack` Python 包。
- 增加阶段 1–4 配置草案。
- 实现基础掩码生成、soft/hard IoU、框 IoU、约束、L∞ 投影、PGD、增广拉格朗日状态和五类评测。
- 增加冻结参照集的防覆盖写入。

### 验证

- `pytest`：6 项通过。
- `ruff check src tests`：通过。
- `pip check`：无破损依赖。

### 重要决策

- raw head 解码暂不猜测实现，必须等模型版本和权重冻结后与官方输出做回归验证。
- 优化阶段使用可微 pre-NMS 代理；正式指标必须来自完整推理流程。

### 主模型权重

- 通过 Ultralytics 官方接口自动下载 `yolov8n-seg.pt`。
- 模型类型：`SegmentationModel`，任务类型：`segment`。
- SHA-256：`a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`。
- 权重路径：`weights/yolov8n-seg.pt`。

### 阶段 1A：官方等价验证

- 标准化 Segment raw 输出：DFL 分布、类别 logits、mask coefficients、三层特征与 prototypes。
- 实现 8400 个候选的 anchor 网格、DFL 解码及 xywh/xyxy 坐标输出。
- 实现干净参照候选匹配和固定 anchor 邻域代理选择。
- raw decoded tensor 与 Ultralytics head 输出在 `1e-5` 容差内一致。
- 自有 decode + NMS 与官方 NMS 一致。
- 真实图片完整 `predict()` 的框输出在 `1e-4` 容差内一致。
- 自有实例掩码与官方 `process_mask` 逐像素一致。
- 测试总数由 6 项增加至 9 项，全部通过。

### 阶段 1B：COCO 数据准备

- 下载并校验官方 COCO 2017 validation 图片与 Train/Val 实例标注。
- `val2017/`：5000 张图片，解压后约 788 MB。
- `instances_val2017.json`：约 20 MB，SHA-256 为
  `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
- COCO API 校验结果：2693 张图片含 person，共 11004 个人物标注实例。
- 两个 ZIP 通过完整性检查后已移入系统回收站。

### 阶段 1B：参照集冻结

- 新增 COCO person 数据适配器、确定性抽样、GT 框/掩码读取和全局贪心一对一匹配。
- 20 张图片冒烟集产生 45 个参照实例，验证完整构建流程。
- 全量处理 2693 张含 person 的图片，冻结 4681 个高质量参照实例，覆盖 2156 张图片。
- 最低 GT 框 IoU：`0.507365`；最低 GT 掩码 IoU：`0.75`；平均置信度：`0.750285`。
- 正式参照集 SHA-256：`56d52414997d303f100ee148cd94e7514ff957295adf10c9d2aaf67a764d5b1c`。
- 唯一性、阈值、代理 level/grid 和前 100 个 RLE 解码检查全部通过。

### 阶段 1B：干净 COCO 指标复现

- 在全部 5000 张 COCO val2017 图片上完成官方 `model.val()` 与独立 COCO API 双重评估。
- 官方 COCOeval：bbox AP `0.367`、mask AP `0.306`；发布参考值为 `0.367/0.305`，
  差异为 `0.0/0.1` 个百分点，通过 `<1 AP` 门槛。
- 独立 square-batch COCO API：全类别 bbox/mask AP `0.36056/0.29484`；person
  bbox/mask AP `0.52810/0.39214`。
- 固化本地验证数据配置 `configs/coco_seg_eval.yaml`，避免全局 Ultralytics 数据目录设置影响复现。
- 官方预测 JSON SHA-256：
  `cd91a77a84c3a9963e802c57b4d12a16dcf6b385f5d4db139d046bf2f9be676b`。
- 增加并固定官方评估依赖 `faster-coco-eval==1.8.0`。

### 阶段 2：单实例攻击链路

- 新增冻结 feature level/grid 代理定位与邻域跟踪，连接 raw head、prototype/coefficient
  掩码和可微 box/mask IoU。
- 新增 `scripts/run_stage2_smoke.py`，统一运行 `mask_only` 与 `joint` PGD，并在攻击后调用
  官方完整推理进行类别、置信度、框和硬掩码复核。
- V100 上 10 步、L∞ `8/255` 冒烟结果：`mask_only` 的官方 mask IoU 为 `0.35171`，
  但类别改变；`joint` 的官方 mask IoU 为 `0.30125`，但目标漏检。两者均未构成选择性退化。
- 排除运行时间后，同一固定种子的重复运行规范化 SHA-256 一致：
  `c29b319e4142bcd69ef5f2ac8eb569089c9e242537f25734e1f8892b9075f009`。
- 新增 2 项代理/目标函数测试，测试总数增至 14 项。
