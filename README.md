# YOLO Mask Selective Attack

在类别、置信度和检测框基本不变的前提下，对 YOLO 实例分割掩码进行选择性退化。

当前仓库是阶段 1–2 的代码骨架，包含统一配置、可微掩码与几何算子、约束定义、
增广拉格朗日攻击循环、评测分类和可复现工具。`repos/` 中的第三方仓库仅作参考，
不作为本项目源码的一部分。

## 快速开始

```bash
conda env create -f environment.yml
conda activate yolo-mask-selective
yolo-mask-attack doctor --config configs/base.yaml
pytest
```

环境固定为 Python 3.11、PyTorch 2.5.1、TorchVision 0.20.1 和 CUDA 12.1；
Ultralytics 使用 `repos/models/ultralytics` 的固定浅克隆（8.4.157，commit `00be778`）。

首次真实模型实验还需要下载官方分割权重，并把路径与 SHA-256 写入配置。优化阶段走
可微 raw head，最终评测阶段必须走完整 Ultralytics 推理流程。

## 目录

- `configs/`：实验协议与各阶段配置。
- `src/yolo_mask_attack/models/`：YOLO 包装、解码、掩码生成。
- `src/yolo_mask_attack/attack/`：约束、目标、PGD 与增广拉格朗日。
- `src/yolo_mask_attack/eval/`：参照集、匹配、互斥分类与指标。
- `tests/`：不变量与数值回归测试。
- `docs/progress/`：当前进度、阶段路线图和工作日志。
- `CODE_IMPLEMENTATION.md`：完整实现规格。

## 当前边界

代码骨架不声称已经完成 Ultralytics 官方输出等价验证。开始实验前，必须先让
`test_decode_matches_official.py` 与 `test_mask_equiv_ultralytics.py` 在锁定版本和权重上通过。
