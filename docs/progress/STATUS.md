# 当前进度

> 最后更新：2026-09-21
> 当前阶段：阶段 2 攻击基线已打通，开始探索局部纹理攻击及其防御

## 总览

| 工作流 | 状态 | 说明 |
| --- | --- | --- |
| 研究范围与技术路线 | 完成 | 聚焦数字域 YOLO-Seg 掩码选择性退化 |
| 参考仓库与资料整理 | 完成 | 参考源码和研究资料均已解压 |
| Conda GPU 环境 | 完成 | V100、PyTorch 2.5.1、CUDA 12.1 验证通过 |
| Python 工程骨架 | 完成 | 配置、模型、攻击、评测和工具模块已建立 |
| 基础数值与协议测试 | 完成 | 16 项测试通过，Ruff 与依赖检查通过 |
| 模型权重冻结 | 完成 | 官方 `yolov8n-seg.pt` 已下载并记录 SHA-256 |
| raw head 解码等价验证 | 完成 | DFL、anchor、NMS 和完整 `predict()` 回归通过 |
| 官方掩码生成等价验证 | 完成 | 自有掩码与官方 `process_mask` 逐像素一致 |
| COCO 数据准备 | 完成 | val2017、实例标注、数量与 SHA-256 均已验证 |
| COCO 参照集 | 完成 | 4681 个实例已冻结，唯一性、阈值与 RLE 校验通过 |
| COCO 干净指标 | 完成 | 官方 36.7/30.6；person 52.81/39.21 box/mask AP |
| 攻击与正式实验 | 进行中 | 批量 pilot 通过，dynamic-weight 暂时最稳 |
| 局部纹理攻击与防御 | 进行中 | 0.171% 局部贴片攻击成功，净化防御完成单图探测 |

## 已完成产物

- `environment.yml`：稳定、可复现的 Conda 环境声明。
- `configs/`：基础配置和阶段 1–4 配置草案。
- `src/yolo_mask_attack/models/`：包装器、张量类型、掩码及几何算子。
- `src/yolo_mask_attack/attack/`：投影、约束、PGD 和增广拉格朗日骨架。
- `src/yolo_mask_attack/eval/`：冻结参照集、五类互斥结果和 SDR/DPR。
- `tests/`：配置、协议、参照集与张量算子的基础测试。

## 当前验证基线

```text
Python          3.11.16
PyTorch         2.5.1
TorchVision     0.20.1
CUDA runtime    12.1
GPU             Tesla V100-SXM2-16GB
Ultralytics     8.4.157 (reference commit 00be778)
NumPy           1.26.4
OpenCV          4.11.0
Tests           19 passed
Ruff            passed
pip check       no broken requirements
```

## 下一批任务

- [x] 选择并下载主模型权重 `yolov8n-seg.pt`。
- [x] 记录权重 SHA-256、Ultralytics commit、输入尺寸和推理阈值。
- [x] 实现固定版本的 raw head 输出归一化与 DFL 框解码。
- [x] 增加 `test_decode_matches_official.py`，误差目标 `< 1e-3`。
- [x] 增加 `test_mask_equiv_ultralytics.py`，验证裁剪、上采样和阈值口径。
- [x] 下载并验证 COCO val2017 及实例标注。
- [x] 生成 20 张样本的干净预测并检查坐标和掩码对齐。
- [x] 实现 COCO person 数据适配器与预测—GT 一对一匹配。
- [x] 构建带模型、数据和协议哈希的冻结参照实例集。
- [x] 在全部 5000 张 val2017 图片上复现官方和 person box/mask AP。
- [x] 建立阶段 2 的单实例 clean/attack 成对实验运行器。
- [x] 运行 mask-only 与 joint attack 单实例冒烟基线。
- [x] 实现并验证 fixed-weight、dynamic-weight 和 constrained-AL 单实例攻击。
- [x] 将扰动限制到有效图像区域，并增加零扰动完整推理自检。
- [x] 扩展为冻结参照集批量运行、断点续跑和聚合统计。
- [ ] 进行对等超参数搜索，并用强攻击结果作为防御研究基线。
- [ ] 将 pilot 扩展到更大样本，并修正 constrained-AL 的框保持稳定性。
- [x] 实现目标框内局部纹理贴片并完成单图攻击可行性验证。
- [ ] 加入可学习透明度、EOT 与多视角共享纹理。
- [ ] 在自适应攻击下评估输入净化、贴片定位和时序一致性防御。

## 当前阻塞项

- 无。阶段 1B 的最终门槛已经通过。
