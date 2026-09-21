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

阶段 2 单实例冒烟实验：

```bash
python scripts/run_stage2_smoke.py --steps 10
```

该命令从冻结参照集中选择一个实例，运行五种攻击基线，将扰动图片与
代理/官方完整推理指标写入 `outputs/stage2_smoke/`。

环境固定为 Python 3.11、PyTorch 2.5.1、TorchVision 0.20.1 和 CUDA 12.1；
Ultralytics 使用 `repos/models/ultralytics` 的固定浅克隆（8.4.157，commit `00be778`）。

官方分割权重、SHA-256、COCO val2017 干净指标和冻结参照集均已验证。优化阶段走
可微 raw head，最终评测阶段走完整 Ultralytics 推理流程。

## 目录

- `configs/`：实验协议与各阶段配置。
- `src/yolo_mask_attack/models/`：YOLO 包装、解码、掩码生成。
- `src/yolo_mask_attack/attack/`：约束、目标、PGD 与增广拉格朗日。
- `src/yolo_mask_attack/eval/`：参照集、匹配、互斥分类与指标。
- `tests/`：不变量与数值回归测试。
- `docs/progress/`：当前进度、阶段路线图和工作日志。
- `CODE_IMPLEMENTATION.md`：完整实现规格。

## 当前边界

阶段 1 的输出等价、COCO 指标和冻结参照集已经完成。阶段 2 已打通单实例
`mask_only`、`joint`、`fixed_weight`、`dynamic_weight` 和 `constrained` 攻击链路；
批量运行、对等调参与自适应攻击仍待实现。攻击用于构建 YOLO 防御研究的压力测试。
