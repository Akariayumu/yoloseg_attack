# 来源与复核说明

检索冻结日期：2026-09-22。

## 来源优先级

1. CVF、PMLR、USENIX、arXiv 论文页面或 PDF；
2. 论文中声明的作者官方 GitHub；
3. GitHub API 元数据和本地 `--depth 1` 检出；
4. README 只用于环境、资产与运行状态，不替代论文结论。

未使用博客、媒体转述或 Papers with Code 排名支撑核心结论。

## 本地审计位置

外部仓库位于 `repos/papers/`，被主仓库 `.gitignore` 排除：

- `physical_attacks_embodied_nav`
- `iap`
- `adversarial_camou`
- `fashion_adv`
- `napguard`
- `apde`
- `pbcat`
- `patchguard`

`repositories.csv` 的 commit、依赖、许可证和测试判断来自这些本地快照。GitHub stars、forks 和更新时间只用于辅助判断维护活跃度；它们会变化，且不作为科学权威性证据。

## 解释限制

- “官方代码”只表示论文或项目页指向该仓库，不表示代码完整、可复现或许可可用。
- “未见许可证”表示审计时仓库根目录/API 未发现明确许可证，不是法律意见。
- 自然性、物理可实现性和防御稳健性都严格限定在原论文的任务与实验设置内。
- 报告中的实现建议是对多项证据的工程推断，已与论文事实分开表述。
