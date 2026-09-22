# 位置搜索、服装纹理先验、两阶段透明度与 EOT：防御导向调查报告

## 1. 执行摘要

本轮调查支持继续构建这一组合攻击，但应把它定义为 **防御红队基线**，而不是新攻击结论。四个组件分别有可靠文献支持，却没有一个官方仓库能直接覆盖 YOLO-Seg、局部 mask 退化和当前运行环境。

最合理的工程选择是：在现有攻击器上洁净实现 IAP 式位置评分、轻量服装风格约束、两阶段 alpha 和 EOT；用 APDE 的统一评测原则、PBCAT 的复合对抗训练思路以及 NAPGuard 的自然纹理检测基线来检验防御。首版暂缓 3D 渲染、扩散先验和完整 AdvCaT 复现。

关键判断如下：

- **位置搜索**：IAP 是最直接的近期证据。它把类定位图与敏感度图结合，在容易影响模型但不显眼的位置放置贴片，并展示了对若干贴片防御的规避能力。这意味着只依赖“高显著、高频异常”的防御可能存在盲区。
- **日常服装纹理先验**：AdvCaT 的主会论文、实物服装和多角度实验最完整；FashionAdv 虽是 workshop 论文，却是候选中与“人物实例分割 + 服装区域 + 风格约束”最贴近的工作。
- **两阶段透明度**：Physical-Attacks-in-Embodied-Nav 明确先优化纹理、再微调透明度，且官方代码暴露纹理与 opacity 参数。证据来自 3D 导航而非 YOLO-Seg，因此应把它视为可检验假设，不能直接宣称对本任务有效。
- **EOT**：Athalye 等人的 EOT 给出基础定义；AdvTexture、AdvCaT 和 embodied-navigation 工作进一步说明视角、形变和多视图采样对服装/物理纹理不可缺少。
- **防御评价**：ICCV 2025 的 APDE 统一评测指出，贴片定位精度与最终防御效果并不等价；所以必须同时报告攻击区域定位、受害目标恢复、干净精度和时延，不能只看一个 mIoU 或 ASR。

## 2. 调查问题和方法

调查围绕五个问题：

1. 四个组件是否有同行评审证据支持？
2. 证据是否直接覆盖人物检测或实例分割、物理场景和自然性？
3. 官方代码是否存在、可许可复用、能否进入当前稳定环境？
4. 这些威胁对 JPEG、模糊、贴片定位和对抗训练等防御意味着什么？
5. 怎样设计自适应评测，避免梯度遮蔽或只对已知贴片有效？

只用论文官方页面、论文 PDF、作者官方 GitHub 和本地检出的源码作为核心证据。星数只记录社区采用度，不参与论文权威性评级。详细证据见 `evidence/`。

### 权威性口径

- **A**：CVPR、ICCV、ICML、USENIX Security 等严格同行评审主会；结论仍受其任务和威胁模型边界约束。
- **A-**：IROS 等领域主会或证据完整但任务迁移较大的论文。
- **B**：CVPR workshop 等同行评审论文，适合作为直接任务证据，但不与主会证据等量看待。
- **代码成熟度**单独评级；论文权威不等于仓库可运行。

## 3. 四个攻击组件的证据

### 3.1 位置搜索

[IAP（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Dutta_IAP_Invisible_Adversarial_Patch_Attack_through_Perceptibility-Aware_Localization_and_Perturbation_ICCV_2025_paper.html)同时考虑类别定位与模型敏感度，在模型脆弱性和人类视觉敏感度之间选择位置，并加入感知正则与颜色恒常更新。其贡献与本项目的“搜索人物服装内部的小区域”高度相关，权威性为 **A**。

局限是原任务主要为图像分类，并非检测或实例分割；官方 [IAP 仓库](https://github.com/subratkishoredutta/IAP) 只有一个主程序，依赖清单来自大型 CUDA 环境，没有单元测试。因此应移植算法思想，不应直接依赖整个环境。

建议的位置评分为：候选区域内 mask-head 目标梯度或分割代理敏感度，减去检测保持风险、人体轮廓越界惩罚与显著性代价。位置只在冻结的服装可行域内搜索，禁止在人物外部或任意背景中寻找“捷径”。

### 3.2 日常服装纹理先验

[AdvCaT（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Physically_Realizable_Natural-Looking_Clothing_Textures_Evade_Person_Detectors_via_3D_CVPR_2023_paper.html)用 Voronoi 图和 Gumbel-softmax 参数化迷彩纹理，并以 TopoProj + TPS 建模衣服形变，最终制作 T 恤和裤子并进行多视角实拍。它同时覆盖日常服装先验、非刚体形变和物理实验，是本组件最强的主证据，权威性为 **A**。

[FashionAdv（CVPRW 2021）](https://openaccess.thecvf.com/content/CVPR2021W/WMF/papers/Treu_Fashion-Guided_Adversarial_Attack_on_Person_Segmentation_CVPRW_2021_paper.pdf)直接面向 YOLACT 人物实例分割，只修改衣服掩码，并用 VGG 内容/风格损失约束纹理；还评价了 JPEG 和滤波。它的任务适配度最高，但属于 workshop、没有物理服装实验，权威性为 **B**。

[AdvTexture（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Adversarial_Texture_for_Fooling_Person_Detectors_in_the_Physical_World_CVPR_2022_paper.html)通过可重复、可扩展纹理覆盖任意衣服形状，并在 T 恤、裙子、连衣裙上做多角度物理实验，证明平铺一致性和跨视角优化很重要，权威性为 **A**。

因此先实现 FashionAdv 风格的轻量约束以验证机制，再增加 AdvCaT 式离散调色板/Voronoi 参数化。生成式或扩散式自然图像先验暂不进入首版：它们增加依赖和调参维度，却不会优先回答现阶段的防御问题。

### 3.3 两阶段透明度

[Physical Attacks in Embodied Navigation（IROS 2025 / arXiv）](https://arxiv.org/abs/2409.10071)联合研究可学习纹理、opacity 和对象感知多视角采样；其 README 与源码明确支持先纹理、后 opacity 的优化。论文报告导航成功率平均下降 22.39%，并提供自然性实验，权威性为 **A-**。

该方法的官方[仓库](https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav)是完整但很重的 3D 流水线：Mitsuba/DrJit、Detectron2、PEANUT、Blender/场景资产、`torch 1.10 + CUDA 11.1`。它不适合直接并入当前 `torch 2.5.1 + CUDA 12.1` 环境，但两阶段参数调度可以在当前 2D 合成器中独立复现。

防御研究中必须限制透明度优化：固定最大面积、alpha 上下界和平均 alpha 预算，并分别报告可见贴片和低透明贴片；否则“更隐蔽”可能只是把扰动预算定义得不可比较。

### 3.4 EOT 和服装形变

[Synthesizing Robust Adversarial Examples（ICML 2018）](https://proceedings.mlr.press/v80/athalye18b.html)给出了对变换分布求期望的基础方法，权威性为 **A**。对本项目，EOT 至少应包含缩放、平移、旋转、轻度透视、亮度/对比度、噪声和模糊；TPS 应作为第二层消融，用于模拟布料非刚性形变。

JPEG 属于不可微或梯度不可靠的预处理。若防御包含 JPEG、量化、硬阈值或贴片遮挡，攻击端必须使用 BPDA/STE 或可微近似并在真实防御前向上复核。否则结果只能说明梯度被截断，不能说明防御稳健。

## 4. 自然性证据不能只靠 SSIM

[PAN（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Towards_Benchmarking_and_Assessing_Visual_Naturalness_of_Physical_World_Adversarial_CVPR_2023_paper.html)指出既有“自然/隐蔽”主张经常缺少统一的人类实验，并提供带人类评分和注视信息的自然性基准。权威性为 **A**。

本项目应同时记录：面积比、平均/最大 alpha、LPIPS、SSIM、CIEDE2000、总变分、调色板大小和服装区域越界率。它们只是代理指标；进入物理结论前，还需要预注册的小规模双盲人评或 PAN 式评分，且必须把普通服装纹理、随机纹理和攻击纹理混合展示。

## 5. 防御工作及其约束

### 5.1 自然贴片检测

[NAPGuard（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_NAPGuard_Towards_Detecting_Naturalistic_Adversarial_Patches_CVPR_2024_paper.html)针对自然贴片检测，发布 GAP 数据集：25 类贴片、9266 张图像，并用特征调制增强定位和泛化。论文权威性 **A**；[官方仓库](https://github.com/wsynuiag/NAPGaurd)提供训练/验证脚本和权重链接，但基于内嵌 YOLOv5，根目录未见许可证，也没有项目测试。

适合把它作为外部防御基线单独运行，不适合复制代码。我们的隐蔽位置 + 服装先验攻击应作为 GAP 之外的 OOD 测试，检验其是否把普通服装纹理误报为贴片。

### 5.2 统一检测器防御评测

[APDE（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Revisiting_Adversarial_Patch_Defenses_on_Object_Detectors_Unified_Evaluation_Large-Scale_ICCV_2025_paper.html)统一比较多种攻击和九类防御，强调补丁定位准确率与最终目标恢复可能不一致；还报告干净样本影响、效率和未见攻击。权威性 **A**。

[官方 APDE 仓库](https://github.com/Gandolfczjh/APDE)目前已有防御代码和再训练权重，但 README 的评测框架/数据加载仍列为待办，且仓库未声明许可证。它最适合作为指标与攻击/防御覆盖清单，不应成为当前主工程的硬依赖。

### 5.3 对抗训练

[PBCAT（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Li_PBCAT_Patch-Based_Composite_Adversarial_Training_against_Physically_Realizable_Attacks_on_ICCV_2025_paper.html)用梯度引导的小区域贴片和全局受限扰动进行复合对抗训练，并评价未见的物理纹理攻击。它是本项目后续训练型防御的最直接证据，权威性 **A**。

[官方代码](https://github.com/LixiaoTHU/oddefense-PatchAT)是 MIT，提供 Faster R-CNN、FCOS、DN-DETR 配置和权重，但锁定 `torch 1.13.1 / mmcv 1.7 / mmdet 2.28`，不含 YOLO-Seg。因此建议在我们的训练接口中重写“位置引导贴片 + 全局小扰动”的 batch 生成器，不迁移整套 MMDetection。

### 5.4 可证明防御的边界

[PatchGuard（USENIX Security 2021）](https://www.usenix.org/conference/usenixsecurity21/presentation/xiang)利用小感受野和鲁棒聚合，对给定单块局部贴片威胁模型提供可证明鲁棒性，权威性 **A**。官方 MIT 仓库基于 `torch 1.7` 且主要是图像分类。

它提供了重要的上界思维，但其证书不能自动外推到大面积服装纹理、多个不连通区域、透明混合或实例分割。报告可把其作为理论参照，不能把分类证书表述成 YOLO-Seg 的保证。

## 6. 官方仓库可行性审查

| 仓库 | 许可/环境 | 直接复用 | 结论 |
|---|---|---:|---|
| IAP | MIT；Python 3.10；庞大 CUDA 依赖 | 中低 | 重写位置评分；不安装整份 requirements |
| Physical-Attacks-in-Embodied-Nav | MIT；torch 1.10/CUDA 11.1；3D 全栈 | 低 | 借鉴两阶段 alpha 和采样逻辑 |
| AdvCaT | 根目录未见许可证；torch 1.10/PyTorch3D 0.6.2 | 低 | 只依据论文重写调色板/Voronoi 先验 |
| FashionAdv | 根目录未见许可证；旧 YOLACT/Kornia | 中低 | 依据论文重写内容/风格损失 |
| NAPGuard | 根目录未见许可证；内嵌 YOLOv5 | 中 | 隔离环境作外部防御基线 |
| APDE | 未见许可证；大规模整合仓库、仍有 roadmap 项 | 中低 | 采用协议和指标，不设为依赖 |
| PBCAT | MIT；torch 1.13/mmcv 1.7/mmdet 2.28 | 中低 | 在 YOLO-Seg 中重写训练样本生成 |
| PatchGuard | MIT；torch 1.7；分类任务 | 低 | 理论参照或独立分类基线 |

审查快照、commit 和 GitHub 元数据见 `evidence/repositories.csv`。所有外部仓库放在被 `.gitignore` 排除的 `repos/papers/` 下，仅供审计，不会随主项目分发。

## 7. 推荐的最小可行栈

### 第一层：可控 2D 防御压力测试

1. 从目标人物实例的服装可行域生成固定大小候选网格。
2. 用 mask 退化梯度、检测保持风险、显著性和越界率给候选排序。
3. 纹理阶段优化 RGB，加入内容/风格、TV、调色板和非目标保持约束。
4. 冻结 RGB 后优化 alpha logits，保持面积和平均 alpha 预算。
5. 每步采样 EOT；每个 checkpoint 用完整 Ultralytics 后处理和真实防御前向复核。

### 第二层：防御适配

- 非训练型：JPEG、Gaussian/median、位深压缩、随机 resize/pad。
- 定位型：服装区域内局部特征异常检测 + 局部修复；评价定位 mIoU 与恢复 AP/SDR 两条轴。
- 训练型：已知攻击混合训练 + PBCAT 式复合扰动；保留未见纹理、未见位置和未见 EOT 分布。
- 时序型：框稳定而 mask 面积、轮廓或 embedding 突变时触发时序融合/告警。

## 8. 必须避免的错误结论

- 单图 JPEG/模糊恢复不能证明防御有效；攻击未对防御适配。
- 位置检测 mIoU 高不能替代受害目标恢复指标。
- 对一种纹理训练后的提升不能称为对自然纹理攻击的泛化。
- 分类贴片的可证明鲁棒性不能直接外推到实例分割和服装全纹理。
- 数字 EOT 成功不能直接称为物理可实现；至少需要打印-拍摄闭环。
- SSIM/LPIPS 高不能单独支持“人类不可察觉”。

## 9. 决策

进入实现阶段，但按 `IMPLEMENTATION_PLAN.md` 的闸门推进。首个里程碑是完成 **位置搜索 + 2D EOT** 并做无防御/非自适应/自适应三组评测；只有在批量冻结集上保持检测且稳定降低 mask IoU 后，才加入服装先验和 alpha 第二阶段。这样能把每个组件的真实贡献和防御失效原因分开。
