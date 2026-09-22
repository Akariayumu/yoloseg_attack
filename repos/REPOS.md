# repos/ — 本项目克隆的参考仓库索引

> 克隆日期：2026-09-21。全部为浅克隆（`--depth 1`）。用途：复现基线、借用实现、以及跨架构迁移目标。
> 更新请重跑 `scripts/clone_repos.sh`（幂等，已存在则跳过）。

## papers/ — 论文官方代码

| 目录 | 仓库 | commit | 对应文献 | 用途 |
| --- | --- | --- | --- | --- |
| `alma_prox_segmentation` | jeromerony/alma_prox_segmentation | 1ce75cb | R2 Rony CVPR 2023 | **核心方法参考**：语义分割的增广拉格朗日 + proximal splitting + adaptive masking；本项目约束优化建模直接借鉴 |
| `alma_iccv2021` | jeromerony/augmented_lagrangian_adversarial_attacks | e1881d2 | ALMA ICCV 2021 | 分类任务的增广拉格朗日攻击原型；乘子/罚参数更新实现的参考 |
| `jeromerony_adversarial_library` | jeromerony/adversarial-library | 9dec220 | — | 作者通用对抗库（C&W、PGD、距离/投影工具），复刻基线用 |
| `physical_attacks_embodied_nav` | chen37058/Physical-Attacks-in-Embodied-Nav | 8991780 | Chen et al., IROS 2025 | 可学习纹理/透明度、对象感知多视角采样和物理贴片参考 |

## models/ — 目标模型与迁移目标

| 目录 | 仓库 | commit | 用途 |
| --- | --- | --- | --- |
| `ultralytics` | ultralytics/ultralytics | 00be778 | **主模型** YOLOv8/v11-seg；wrapper/解码/mask 生成一切以它为准（注意锁版本 + 权重 sha256） |
| `yolact` | dbolya/yolact | 902073d | R1 的攻击对象 + 阶段 4 迁移目标（同属原型-系数范式） |
| `detectron2` | facebookresearch/detectron2 | a2f4a87 | 阶段 4 迁移目标 Mask R-CNN；**Boundary IoU（R4）官方实现在此**（`detectron2/evaluation/boundary_iou_evaluation.py`） |

## tools/ — 评测与基线工具

| 目录 | 仓库 | commit | 用途 |
| --- | --- | --- | --- |
| `cocoapi` | cocodataset/cocoapi | 8c9bcc3 | pycocotools：box/mask mAP、参照集匹配（也可 pip 装） |
| `adversarial-attacks-pytorch` | Harry24k/adversarial-attacks-pytorch | 23620a6 | torchattacks：PGD / MI-FGSM / C&W 等基线一键调用 |
| `seg-robustness-cvpr2018` | anuragarnab/adversarial-attacks | 937f7df | CVPR2018 语义分割鲁棒性基线（DAG 等密集预测攻击），相关工作效率对照 |
| `mmsegmentation` | open-mmlab/mmsegmentation | b040e14 | 语义分割模型与 pipeline；SMTA²（Guo 2024）风格的多任务/密集预测对照 |
| `mmdetection` | open-mmlab/mmdetection | cfd5d3a | 备选迁移目标（Mask R-CNN / 实例分割变体）与评测工具 |

## 未找到 / 需另找

- **R1 Zhang 2022（YOLACT 攻击）**：Elsevier 付费，未发现公开代码；按论文描述自行复刻（`attack/objectives.py` 的 mask-only baseline）。
- **R3 Guo 2024 SMTA²**：arXiv 无官方代码链接；动态加权基线按论文描述复刻。
- **R5 Zhe 2024（Hidden Tasks MTL）**：未发现公开仓库。
- **R6 实例分割鲁棒性 benchmark**：暂用 detectron2 自带指标。
- **R10 Tokiwagi IEICE 2026**：全文未取，阶段 0 待确认。
