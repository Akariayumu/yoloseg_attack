# 隐蔽局部纹理攻击与防御调查

本目录记录面向 **YOLO 实例分割防御研究** 的文献调查和工程决策。攻击方法只作为防御的压力测试器，目标是避免用过弱、非自适应攻击高估防御效果。

## 文件索引

- [REPORT.md](REPORT.md)：论文权威性、方法证据、仓库可行性和最终建议。
- [THREAT_MODEL.md](THREAT_MODEL.md)：位置搜索、服装纹理先验、两阶段透明度、EOT 的统一威胁模型及防御评测协议。
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)：按风险排序的实现和实验路线。
- [evidence/papers.csv](evidence/papers.csv)：论文证据表。
- [evidence/repositories.csv](evidence/repositories.csv)：官方仓库审计表。
- [evidence/SOURCES.md](evidence/SOURCES.md)：检索范围、来源与可复核说明。

## 当前结论

不直接移植某一个外部项目，而是在现有 `torch 2.5.1 + ultralytics 8.4.157` 环境中组合并重写四个最小模块：

1. IAP 式候选位置评分；
2. FashionAdv 式内容/风格约束，随后再做 AdvCaT 式离散服装调色板；
3. 先纹理、后透明度的两阶段优化；
4. 可微 EOT，以及对 JPEG 等不可微预处理的 BPDA 自适应评测。

第一阶段不引入 Mitsuba、Detectron2、PyTorch3D 或扩散模型。这样能保留关键威胁因素，同时控制变量并避免破坏稳定 Conda 环境。

调查冻结日期：2026-09-22。
