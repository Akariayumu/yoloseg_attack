# 项目接续上下文

> 供新会话或重新启动研究时优先读取。最后更新：2026-09-22。

## 项目定位

本项目研究 YOLO 实例分割的防御。攻击代码用于构造足够强的防御压力测试，核心目标是在保持人物类别、置信度和检测框的同时，分析并防御实例掩码退化。

仓库路径：`/home/akunai/claw_work/Research/yoloseg/YOLO-Mask-Selective`

当前阶段：**阶段 2 进行中；基础攻击链路完成，正在构建隐蔽局部服装纹理与自适应防御评测。**

## 启动时先读

1. [`docs/progress/STATUS.md`](docs/progress/STATUS.md)：当前事实和待办。
2. [`docs/progress/ROADMAP.md`](docs/progress/ROADMAP.md)：总阶段与通过判据。
3. [`docs/research/stealth_local_texture_defense/REPORT.md`](docs/research/stealth_local_texture_defense/REPORT.md)：文献结论与仓库审查。
4. [`docs/research/stealth_local_texture_defense/THREAT_MODEL.md`](docs/research/stealth_local_texture_defense/THREAT_MODEL.md)：防御评测威胁模型。
5. [`docs/research/stealth_local_texture_defense/IMPLEMENTATION_PLAN.md`](docs/research/stealth_local_texture_defense/IMPLEMENTATION_PLAN.md)：接下来 M0–M6 的执行顺序。

## 已完成且验证通过

- 稳定环境：Python 3.11、PyTorch 2.5.1、TorchVision 0.20.1、CUDA 12.1、Ultralytics 8.4.157。
- GPU：Tesla V100-SXM2 16 GB。
- `yolov8n-seg.pt`、Ultralytics commit 和权重 SHA-256 已冻结。
- raw head、DFL、anchor、NMS 和 mask prototype 解码与官方实现等价。
- COCO val2017 和标注已验证；冻结参照集包含 4681 个实例。
- 官方 COCOeval：bbox AP `36.7`、mask AP `30.6`；person bbox/mask AP `52.81/39.21`。
- 五种攻击链路已实现：mask-only、joint、fixed-weight、dynamic-weight、constrained-AL。
- 批量运行、断点恢复和聚合统计已实现。
- 局部纹理贴片已完成单图可行性验证。
- 位置搜索、服装纹理先验、两阶段透明度、EOT 及防御的文献与代码审查已完成。
- 当前回归状态：`19 passed`，Ruff 通过。

## 当前实验结果及边界

### 常规像素攻击

- 单实例 constrained-AL：confidence `0.753`、box IoU `0.906`、mask IoU `0.0059`。
- 5 实例 pilot：dynamic-weight SDR/DPR `0.8/0.8`；fixed-weight `0.6/0.8`；constrained-AL `0.2/0.2`。
- constrained-AL 掩码破坏最强，但 4/5 出现框保持失败；当前 dynamic-weight 更稳定。

这些结果是链路验证或小样本 pilot，不是总体统计结论。

### 局部纹理攻击

- 贴片面积占整图约 `0.1709%`。
- 100 步 dynamic-weight：confidence `0.623`、box IoU `0.927`、mask IoU `0.314`。
- 非自适应 JPEG Q95 后 mask IoU `0.523`；Gaussian sigma=1 后 `0.689`，且目标检测保持。

JPEG/Gaussian 仅为单图、非自适应诊断，**不能表述为防御已经有效**。正式防御结论必须在攻击已知防御的 BPDA/EOT 条件下成立。

## 已确定的技术方案

- 位置：IAP 式服装可行域候选搜索。
- 纹理：先做 TV/有限调色板和 FashionAdv 式内容/风格约束，再消融 AdvCaT 式 Voronoi + Gumbel-softmax。
- 透明度：先优化 RGB，再冻结 RGB 优化 alpha，并保持面积与平均 alpha 等预算。
- 变换：先实现 2D EOT；TPS、打印色域、相机响应后置。
- 防御：先评测 JPEG/blur/随机化和局部定位，再做 PBCAT 式训练防御。
- 不把 Mitsuba、PyTorch3D、旧 YOLACT/MMDetection 环境直接装进当前稳定环境；按论文洁净重写最小模块。
- 无明确许可证的第三方源码不复制进主工程。

## 下一次启动后的首要任务

从 M0/M1 开始，不直接跳到服装先验或 3D：

1. 冻结局部纹理训练集、位置选择集和最终测试集，保证实例不重叠。
2. 把面积上限、检测保持阈值和 EOT 范围写入正式 YAML。
3. 增加区域外像素不变、alpha 范围和随机种子复现测试。
4. 实现服装可行域候选网格与 mask-head 梯度敏感度评分。
5. 实现 batch 2D EOT 和逐样本变换日志。
6. 比较固定位置、随机位置、仅敏感度和完整位置评分。
7. 用独立 EOT 种子及 BPDA 重新评价 JPEG/Gaussian 防御。

M1 通过条件：位置搜索相对固定位置提高冻结 pilot 的 SDR，且不降低 box IoU/置信度，并能在独立 EOT 种子上保持。

## 常用命令

```bash
cd /home/akunai/claw_work/Research/yoloseg/YOLO-Mask-Selective
conda activate yolo-mask-selective
pytest
ruff check src tests scripts
python scripts/run_stage2_smoke.py --steps 10
python scripts/run_stage2_batch.py --start 0 --limit 20 --steps 10
```

若非交互 shell 找不到 `conda`，可直接使用：

```bash
/home/akunai/miniconda3/envs/yolo-mask-selective/bin/python -m pytest
/home/akunai/miniconda3/envs/yolo-mask-selective/bin/python -m ruff check src tests scripts
```

## 仓库状态

- 主分支：`main`
- 最近完成的调查提交：`ae8574d Add defense-oriented stealth texture survey`
- `repos/papers/` 下的论文仓库仅作本地审计并被 `.gitignore` 排除。
- 开始工作前先运行 `git status --short`，不要覆盖用户已有修改。
