# YOLO-Seg 掩码选择性对抗攻击 — 代码实现说明

> 版本 2026-09-21。基于 `yolo-mask-selective-research` 资料包（README / 02_technical_roadmap / 03_feasibility_and_novelty / references）与 ref1 构想整理。
> 范围：只做数字域、公开模型、公开数据集。机器人导航/SLAM/防御/物理贴片全部移出。
> 本文是**实现规格**（repo 结构 / 模块接口 / 核心算法 / 伪码 / 评测口径 / 陷阱），可直接交给 Codex 或成员按文件落地。

---

## 0. 一句话定义

> 在**类别、置信度、检测框基本不变**的前提下，让 YOLO-Seg（Ultralytics v8/v11-seg）的**实例掩码**显著退化，并把这个问题写成**约束优化**而不是加权 loss 求和。

形式化：

```
min_δ  ||δ||∞
s.t.   C_mask(δ) ≤ 0      # 掩码退化达标（IoU ≤ τ_m）
       C_cls(δ)  ≤ 0      # 类别保持 + 置信度 ≥ κ_c·c_clean
       C_box(δ)  ≤ 0      # 框 IoU ≥ τ_b
       x+δ ∈ [0,1]^d
```

核心不是"把 detector 打崩"，而是"检测照常、空间感知损坏"。因此所有指标都以**冻结的参照实例集**为分母。

---

## 1. 目标模型的结构事实（决定一切实现方式）

Ultralytics YOLOv8-seg / v11-seg 头（`ultralytics/nn/modules/head.py`）：

- `nc` 类数，`reg_max=16`，`nm=32`（原型数）。
- 每个尺度输出拼成 `[B, no, H, W]`，`no = 4*reg_max + nc + nm`：
  - `cv2` → 分布式框回归（`4*reg_max`）
  - `cv3` → 类别分数（`nc`，**sigmoid 多标签，不是 softmax**）
  - `cv4`（Segment 独有）→ 掩码系数（`nm=32`）
- **原型** `Proto` 模块：取 P3/P4/P5 特征 → 32 通道 → 上采样到 **输入的 1/4 分辨率**（640 → 160×160）。
- 实例掩码生成：

```
mask_logits[n] = coeffs[n] @ proto.reshape(nm, -1)      # (32) · (32, H*W)
mask[n]        = sigmoid(mask_logits[n])                # (H, W) @ 1/4 分辨率
mask_cropped[n]= crop_mask(mask[n], bbox[n])            # 用预测框裁剪
```

**三个直接推论**（写进论文的"为什么容易"）：

1. 框保持不变 ⇒ `crop_mask` 之后掩码**不可能长出框外**，退化只能是框内的收缩 / 空洞 / 碎裂。
2. 梯度有两条通路到 δ：**共享原型 proto**（全局）与**实例系数 coeffs**（逐实例）。两者都依赖输入图像 ⇒ 单实例选择性攻击在数字白盒下天然可分。
3. 原型分辨率 1/4 是**局部区域控制（阶段 3）的物理上限**，必须用输入尺寸消融来量化。

---

## 2. Repo 结构

```
yolo-mask-attack/
├─ configs/
│  ├─ base.yaml                 # ε、迭代数、seed、阈值、模型哈希
│  ├─ stage1_protocol.yaml      # 参照集、匹配规则、判定阈值（冻结后不改）
│  ├─ stage2_methods.yaml       # 五种方法 + 调参预算
│  ├─ stage3_region.yaml        # 区域定义、区域选择性阈值
│  └─ stage4_transfer.yaml      # 迁移矩阵、预处理扰动
├─ src/
│  ├─ models/
│  │  ├─ wrapper.py             # YoloSegWrapper：可微 raw 输出（无 NMS）
│  │  ├─ decode.py              # 头解码 + anchor 匹配（代理索引）
│  │  └─ masks.py               # coeff@proto / crop_mask / 上采样 / 二值化
│  ├─ attack/
│  │  ├─ base.py                # AttackBase：δ 初始化、投影、单步
│  │  ├─ objectives.py          # 目标组合器（5 种方法统一插件）
│  │  ├─ constraints.py         # C_mask / C_cls / C_box / 区域约束
│  │  ├─ lagrangian.py          # Augmented Lagrangian 优化器（本项目方法）
│  │  └─ pgd.py                 # PGD / C&W / 动态加权基线
│  ├─ eval/
│  │  ├─ reference_set.py       # 构建并冻结参照实例集（带版本号）
│  │  ├─ match.py               # 一对一匹配 + 五类互斥归类
│  │  ├─ metrics.py             # SDR/DPR/IoU/BoundaryIoU/COCO mAP
│  │  └─ run_eval.py            # 完整推理评测入口
│  ├─ analysis/
│  │  ├─ component_swap.py      # 原型↔系数 组件替换实验
│  │  ├─ morphology.py          # 收缩/空洞/连通域/位置统计
│  │  └─ region_selectivity.py  # 区域内外 IoU 比
│  ├─ data/
│  │  ├─ coco_person.py         # COCO val2017 人物子集 + 标注
│  │  └─ transforms_natural.py  # 噪声/模糊/亮度/JPEG/缩放
│  └─ utils/{seed.py, io.py, viz.py, logger.py, hash.py}
├─ scripts/
│  ├─ run_stage1_protocol.py    # 生成冻结参照集 + 干净指标复现
│  ├─ run_attack.py             # 单方法单配置跑攻击
│  ├─ run_eval.py              # 批评测 → CSV/JSON
│  ├─ run_mechanism.py          # 组件替换 + 形态统计
│  └─ run_transfer.py           # 迁移矩阵 + 预处理稳定性
├─ tests/
│  ├─ test_decode_matches_official.py   # raw 解码 == ultralytics.predict
│  ├─ test_mask_equiv_ultralytics.py    # 我们的 mask 生成 == 官方
│  ├─ test_lagrangian_feasible.py       # AL 收敛到可行
│  └─ test_protocol_determinism.py      # 同种子同输入 → 同输出
└─ requirements.lock.txt        # ultralytics==8.3.x, torch==2.x, pycocotools
```

**版本锁**：`requirements.lock.txt` 固定 ultralytics 与 torch 版本；`configs/base.yaml` 记录权重文件 **sha256**。所有结果必须能复现，权重哈希写进论文。

---

## 3. 模块接口与伪码

### 3.1 `models/wrapper.py` —— 可微前向（绕过 NMS）

NMS 不可微，所以优化时**绝不走 `model.predict()`**，而是直接调用内部模块拿 raw 头输出。

```python
class YoloSegWrapper(nn.Module):
    def __init__(self, weights, device, nc, reg_max=16, nm=32, version="v8"):
        super().__init__()
        self.model = YOLO(weights)          # ultralytics
        self.net   = self.model.model       # DetectionModel
        self.head  = self.net[-1]           # Segment head
        self.net.eval()                     # BN/backbone 保持 eval
        self.strides = self.head.stride
        self.nc, self.reg_max, self.nm = nc, reg_max, nm
        # 用 hook 抓 cv4/proto，避免依赖 ultralytics 返回元组的格式漂移
        self._hooks = self._register_capture_hooks()

    def forward(self, x):                   # x: [B,3,H,W] in [0,1]
        # 返回 RawOut(preds, protos, anchors)
        feat_maps = self._collect_head_inputs(x)     # hook 抓到的 3 层 feat
        preds  = [torch.cat([cv2[i](f), cv3[i](f), cv4[i](f)], 1)
                  for i, f in enumerate(feat_maps)]  # [B,no,h,w]
        protos = self.head.proto(feat_maps[0])       # [B,32,H/4,W/4]
        return RawOut(preds=preds, protos=protos)
```

> **实现陷阱**：Ultralytics 的 Segment head 在 `training/export` 分支返回不同结构（`(y, (mc,p))` vs `(y,p)`），不同小版本还会变。**用 forward hook 直接抓 `cv2/cv3/cv4/proto` 的输入输出**最稳，然后 `tests/test_decode_matches_official.py` 断言"我们的 raw 解码 + NMS == 官方 `predict()` 输出"，一旦 ultralytics 升级就靠这个测试兜底。

### 3.2 `models/decode.py` —— 解码 + 代理索引匹配

```python
def make_anchors(feats, strides): ...        # 复用 ultralytics utils.ops.make_anchors
def dist2bbox(distance, anchors, xywh=True): ...  # DFL 解码（复用官方）

@dataclass
class Decoded:
    boxes:  Tensor   # [N,4] xyxy，输入像素坐标（NMS 前候选）
    scores: Tensor   # [N, nc] sigmoid 后
    coeffs: Tensor   # [N, 32]
    levels: Tensor   # [N] 该候选来自哪一层
    yx:     Tensor   # [N,2] anchor 网格坐标

def decode_head(preds, protos, strides, reg_max, nc, nm) -> Decoded:
    # 每层 split 通道 → dfl(cv2) → dist2bbox；sigmoid(cv3)；cv4 直接当 coeffs
    ...

def match_reference_to_candidates(decoded_clean, ref_boxes, ref_classes, iou_thr=0.6):
    """返回每个参照实例 i → 干净候选索引 idx_i（NMS 前）。
       这是优化期唯一允许使用的"代理实例"。"""
    # 策略：先按框 IoU 找候选，再在同 anchor 邻域(±1 格)内取该类分数最高者
    ...
```

**为什么需要代理索引**：优化期不能跑 NMS，但参照实例的"身份"必须跨迭代稳定。做法是：干净推理时把每个参照实例**锚定到某个 pre-NMS 候选的 (level, y, x)**；攻击迭代中，在 adv 前向里读**同一锚点邻域**内该类分数最高的候选，视为该实例的代理。评测期一律走完整 `predict()`。

> 必须单独报告 **"代理成功但完整流程失败"比例**（roadmap 列为高概率风险）。

### 3.3 `models/masks.py` —— 掩码生成（与官方一致）

```python
def instance_masks(coeffs, protos, boxes_xyxy, input_hw, retina=False):
    """mask[n] = sigmoid(coeffs[n] @ protos.flatten) → crop_mask(box) → 上采样
       光栅化网格与 Ultralytics v8SegmentationLoss.single_mask_loss 严格对齐。"""
    h, w = protos.shape[-2:]
    logits = torch.einsum("nk,khw->nhw", coeffs, protos.reshape(coeffs.shape[1], -1))
    masks  = logits.sigmoid()
    masks  = crop_mask(masks, boxes_xyxy * scale_to_proto)   # 官方 crop_mask
    if retina:
        masks = F.interpolate(masks[None], input_hw, mode="bilinear")[0]
    return masks                                             # 可微

def soft_iou(a, b, eps=1e-6):  return (a*b).sum() / ((a+b-a*b).sum()+eps)
def hard_iou(a, b):            return soft_iou((a>0.5).float(), (b>0.5).float())
```

**关键**：用 `sigmoid` 输出的**软 IoU** 做梯度；报指标时用 **0.5 二值化**后的**硬 IoU**。两者口径必须分开并在论文里写清，否则复现会被质疑。

### 3.4 `attack/constraints.py` —— 三个保持约束

对参照实例 `i`（干净值带 `clean` 上标，扰动值带 `δ`）：

```python
# C1 置信度/类别保持（sigmoid 多标签 → 直接约束目标类分数）
C_cls[i]  = kappa_c * conf_clean[i] - conf_delta[i]          # ≤0；kappa_c=0.8
#   类别保持用 margin proxy:  max_other_logit - target_logit + m ≤ 0
# C2 框保持（可微 IoU，报告用硬 IoU）
C_box[i]  = tau_b - soft_iou_xyxy(box_clean[i], box_delta[i])  # ≤0；tau_b=0.9
# C3 掩码退化目标（作为约束而非 loss）
C_mask[i] = (1 - tau_m) - (1 - soft_iou(mask_clean[i], mask_delta[i]))  # ≤0；tau_m=0.5
#           ⇒ soft_iou(mask) ≤ tau_m
```

**区域可控版（阶段 3）**：区域掩码 `R_i`（边界带 / 上中下 1/3 / 关键点部位）：

```python
C_mask_region[i] = (1-tau_m) - (1 - soft_iou(mask_delta[i]*R_i, mask_clean[i]*R_i))
C_mask_out[i]    = soft_iou(mask_delta[i]*(1-R_i), mask_clean[i]*(1-R_i)) - tau_out  # 区域外保持
```

### 3.5 `attack/objectives.py` —— 五种方法统一插件（阶段 2）

所有方法共用同一个 δ 框架，**只替换目标组合方式**，差异才能归因到"保持条件怎么表达"：

| 方法 | `Objective.compose(ctx)` | 对应 |
| --- | --- | --- |
| mask_only | `f = -soft_iou(mask_delta, mask_clean)`（不做保持） | Zhang 2022 |
| joint | `f = -soft_iou(mask) - λ_c*conf - λ_b*box_iou` | 破坏上限参照 |
| fixed_weight | `f = -soft_iou(mask) + Σ w_j * relu(violation_j)`，w 网格搜索 | 最简选择性 |
| dynamic_weight | 迭代中按违反量自适应升权（Guo 2024 复刻） | **必须打过的强基线** |
| constrained（ours） | 见 3.6，约束走 AL，不进 f | 本项目方法 |

```python
class Objective(Protocol):
    def value(self, ctx: AttackContext) -> Tensor: ...   # 可微标量
    def extra_step(self, ctx): ...                        # 动态权重/乘子更新钩子
```

### 3.6 `attack/lagrangian.py` —— 增广拉格朗日（本项目算法）

对 `g_j(δ) ≤ 0`（即上面的 C_mask/C_cls/C_box），带乘子 `λ_j ≥ 0`、罚参数 `ρ`：

```
L_A(δ,λ,ρ) = f(δ) + (1/2ρ) Σ_j [ max(0, λ_j + ρ g_j(δ))² − λ_j² ]

内层 (K 步梯度):
  δ ← δ − η ∇_δ L_A
  δ ← clamp(δ, −ε, ε)                 # 投影到 L∞ 球
  δ ← clamp(x+δ, 0, 1) − x            # 保证仍是合法图像
外层:
  λ_j ← max(0, λ_j + ρ g_j)           # 乘子上升
  ρ   ← β · ρ   (β=1.1~2.0)           # 罚递增
```

这里 `f(δ) = -Σ_i (1 - soft_iou(mask_i^δ, mask_i^clean))`（最大化掩码退化），保持条件全部用约束表达。

**最小扰动的做法**：固定阈值 τ，对 ε 做**二分 / 阶梯搜索**（如 ε ∈ {8/255, 4/255, 2/255, 1/255}），找出"仍可行（三约束全满足）"的最小 ε。论文主结果是"达成选择性退化所需最小 L∞"。

**proximal splitting 对应关系**（借 Rony 2023 的建模思想）：L∞ 投影就是约束集 `{||δ||∞≤ε}∩[0,1]` 的 prox 算子；如果之后要上更一般的约束（如结构化稀疏），把投影换成对应的 prox 即可，框架不变。

```python
class AugmentedLagrangian:
    def optimize(self, wrapper, x, ref, cfg):
        delta = init_delta(x, mode=cfg.init, eps=cfg.eps)  # zeros / 灰噪 / 参照实例区域
        lam = {k: zeros_like_constraints for k in ("cls","box","mask")}
        rho = cfg.rho0
        for outer in range(cfg.outer):
            for k in range(cfg.inner):
                ctx = build_ctx(wrapper(x + delta), ref)
                loss = obj_value(ctx) + self._penalty(ctx, lam, rho)
                loss.backward()
                delta = project(delta, x, cfg.eps, opt_step)
            lam = update_multipliers(lam, ctx, rho)     # λ ← max(0, λ+ρg)
            rho *= cfg.beta
            if self._feasible(ctx) and self._degraded(ctx): break   # 提前停
        return x + delta
```

### 3.7 `attack/pgd.py` —— 基线

`mask_only / joint / fixed_weight` 直接用 PGD（带 momentum 可选，`MI-FGSM` 作为强 baseline 备选）；`dynamic_weight` 复刻 Guo 2024 的按违反量动态调权。**每方法的调参搜索次数必须对等并报告**（roadmap 明确要求）。

---

## 4. 评测协议（阶段 1，最不能出错的部分）

### 4.1 冻结参照实例集 `eval/reference_set.py`

```python
def build_reference_set(model, dataset, cfg):
    """在干净输入上完整推理；只保留：
       - 与人工标注匹配（框 IoU≥0.5）、类别正确
       - 掩码质量达标（与标注 mask IoU ≥ 0.75）
       输出带版本号（模型哈希 + 数据集哈希 + 阈值）的 JSON，之后只读。"""
```

**参照集一经冻结不改**，所有成功率分母都是它。

### 4.2 五类互斥归类 `eval/match.py`

扰动后完整推理 → 与参照实例按**框 IoU 一对一匹配**（贪心或匈牙利）→ 每个参照实例**归入且仅归入**一类：

```
1. 选择性退化      : 类别不变 ∧ conf≥κ_c·c_clean ∧ boxIoU≥τ_b ∧ maskIoU≤τ_m   ← 命中
2. 检测保持但掩码未退化 : 前三项成立 ∧ maskIoU>τ_m
3. 漏检
4. 类别改变
5. 框偏移超限
```

阈值**两档报告**（严格/宽松），避免"阈值挑结果"。掩码分辨率、上采样方式、`retina_masks`、NMS 阈值全部写进配置并在论文报告。

### 4.3 指标 `eval/metrics.py`

```python
SDR  = #选择性退化 / #参照实例          # 主指标
DPR  = #(类别+conf+box 全保持) / #参照实例   # Detection Preservation Rate
maskIoU, boxIoU, BoundaryIoU (R4 定义)   # 均值
mAP50_box / mAP50_mask                    # COCO API（pycocotools）
RSS (region selectivity) = ΔIoU_in / ΔIoU_out   # 阶段 3
```

### 4.4 自然退化对照（阶段 1 收尾）

高斯噪声 / 运动模糊 / 亮度变化 **同一套分类规则**统计 → 回答"自然条件下是否也会出现选择性退化"。这个问题本身有价值，成本很低，别跳过。

---

## 5. 机理分析（阶段 2 亮点，`analysis/`）

1. **组件替换（核心）**：掩码由 `coeffs @ protos` 得到，两个因子分别来自干净/扰动输入，做四象限：

```
mask_A = sigmoid(coeffs_adv @ protos_clean)   # 退化只经系数？
mask_B = sigmoid(coeffs_clean @ protos_adv)   # 退化只经原型？
→ 与 clean mask 的 IoU 对比，判定"退化由原型还是系数承载"
```

2. **退化形态统计**：面积比、内部空洞数（`scipy.ndimage.binary_fill_holes` 差）、连通域数（`ndimage.label`）、质心上下位移。
3. **框裁剪影响**：验证"框保持时掩码只能在框内变化"在数据上的体现。
4. **模型尺寸**：n vs s 对比（可选跨尺寸迁移）。

> 若约束方法打不过动态加权，**如实报告**，贡献移到协议 + 机理分析。

---

## 6. 阶段 3：区域 / 边界可控退化

| 退化目标 | 区域 `R_i` 定义 | 验证问题 |
| --- | --- | --- |
| 边界带 | Boundary IoU 定义：`dilate(m,w) & ~erode(m,w)` | 只坏轮廓、保主体 |
| 几何局部 | 框内上/中/下 1/3 | 位置可否指定 |
| 语义部位 | COCO 关键点 → 腿部/躯干 | 部位可否指定 |
| 退化形态 | 收缩 / 空洞 / 碎裂（不同目标 mask 的构造） | 形态可否指定 |

指标：区域选择性 `RSS`、DPR、SDR（分母不变）、区域面积占比分档（50%/25%/10%）、幅度-成功率曲线。

**机理问题**：原型是输入的 1/4（640→160），局部控制精度受它限制 → 用**不同输入尺寸**测"最小可控区域"，直接检验这个上界。

---

## 7. 阶段 4：迁移与泛化（`scripts/run_transfer.py`）

迁移矩阵（源→目标，同一套分类规则）：

```
n→s（同代，仅容量差）
一代 YOLO-Seg → 另一代（结构近、训练不同）
YOLO-Seg → YOLACT（同属原型-系数范式）
YOLO-Seg → Mask R-CNN（掩码生成方式完全不同 → 选择性很可能消失）
```

**预处理稳定性**（数字域，不做物理）：JPEG q∈{95,75,50}、缩放 ∈{0.5,0.75}、高斯模糊 σ∈{0.5,1.0} → 选择性还剩多少。

跨类别：车辆等类别复测阶段 2 最佳方法。

---

## 8. 实现顺序（建议按此推进，每步有验证）

| 步 | 交付 | 验证 |
| --- | --- | --- |
| 1 | `wrapper.py` + `test_decode_matches_official.py` | raw 解码+NMS == 官方 predict（差距 <1e-3） |
| 2 | `masks.py` + `test_mask_equiv_ultralytics.py` | 我们的 mask == 官方 loss 内部 mask |
| 3 | `reference_set.py` + 干净指标复现 | COCO 官方数值差距 <1 个点，参照集冻结 |
| 4 | `pgd.py` + `objectives.py`（mask_only/joint/fixed） | 能跑出选择性退化，单图耗时/显存记录 |
| 5 | `lagrangian.py` | 二约束 AL 玩具问题收敛到可行 |
| 6 | `match.py` + `metrics.py` + `run_eval.py` | 五类归类自洽（和=参照集）；代理失败比单独统计 |
| 7 | `dynamic_weight` 基线 | 复刻 Guo 2024 关键行为 |
| 8 | `analysis/component_swap.py` + morphology | 出四象限结论 |
| 9 | 阶段 3 区域约束 + `region_selectivity.py` | 指定区域退化显著高于区域外 |
| 10 | `run_transfer.py` | 出完整迁移矩阵 |

---

## 9. 已知陷阱 / 必须做的对照

1. **NMS 不可微** → 优化用 pre-NMS 代理索引；评测一律完整推理；单独报"代理成功/流程失败"比。
2. **sigmoid vs softmax**：v8 是逐类 sigmoid，`conf = max_class_score`。保持检测=保持该分数，不是保持 softmax 概率。
3. **软 IoU（梯度）vs 硬 IoU（指标）** 必须分开；mask 二值化阈值（0.5）、上采样方式固定并报告。
4. **原型 1/4 分辨率** + `retina_masks` 对 Boundary IoU 敏感 → 写进协议固定。
5. **框裁剪**决定"框保持时掩码只能框内变化"，是机理分析的一条，也是攻击的天然约束。
6. **调参预算对等**：每方法超参搜索次数一致并报告，否则"约束方法更优"会被质疑。
7. **ultralytics 升级漂移** → 锁包版本 + 权重 sha256 + 回归测试兜底。
8. **随机种子**：训练/评测/攻击种子全部固定；`test_protocol_determinism.py` 保证同输入同输出。
9. **多实例图像**：默认所有参照实例同时作为保持对象，只对指定实例退化，其他实例漏检计入副作用。
10. **复现口径**：clean 指标必须先复现官方值（<1 个点）再进后续阶段。

---

## 10. 阶段 0 待补（不写代码也要做）

- 搜 Zhang 2022（YOLACT）与 Guo 2024 的全部被引；知网/万方中文检索；SOLO/CondInst/Mask2Former 的鲁棒性工作；**Tokiwagi IEICE 2026 全文**确认是否涉及"检测保持"。
- 期望产出：一张"最接近工作对比表"（模型/任务/是否保持检测/是否区域可控），直接进开题报告。

---

## 附：与资料包的对应关系

- 本文第 3–4 节 = `02_technical_roadmap.md` 阶段 1–2 的可执行落地
- 第 6 节 = 阶段 3；第 7 节 = 阶段 4
- 第 5 节 = roadmap "机理分析"（RQ2 亮点）
- 第 9 节"已知陷阱" = roadmap 第 9 节风险表的工程化展开
- `03_feasibility_and_novelty.md` 的四点原创性 → 分别由 `eval/reference_set.py`（评测协议）、`objectives.py` 的 dynamic_weight 基线、`component_swap.py`（机理）、阶段 3/4 承载
