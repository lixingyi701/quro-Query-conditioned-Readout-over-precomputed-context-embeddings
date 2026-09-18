# docs/ — QuRO 自己的设计、实验与进展

与 [`../article/`](../article/) 分开：那里是**读别人论文的笔记**，这里是**我们自己写的东西**。

| 文件 | 内容 |
|---|---|
| [`RESIDUAL_RESULTS.md`](RESIDUAL_RESULTS.md) | **最新优先阅读**：残差臂四次运行的完整结果——恒等精确成立、训练后无收益、p-control 证明续训有害 |
| [`TRAINING_DATA_REDESIGN.md`](TRAINING_DATA_REDESIGN.md) | **下一步设计**（未实现）：数据盘点与 2×2 判决方案，先换数据不换模块 |
| [`RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md`](RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md) | 本轮残差方向的交接与执行顺序（步骤 A–E 均已执行完毕） |
| [`PISCO_RESIDUAL_EXPERIMENT.md`](PISCO_RESIDUAL_EXPERIMENT.md) | 已实现的 R 臂、从实际 P checkpoint 起步、恒等验证与训练命令 |
| [`QURO_EXPERIMENTAL_DESIGN.md`](QURO_EXPERIMENTAL_DESIGN.md) | 实验设计总纲：研究问题、基线、判据、摊薄论证 |
| [`QURO_V0.1_IMPLEMENTATION_PLAN.md`](QURO_V0.1_IMPLEMENTATION_PLAN.md) | v0.1 实施方案 + §10.5–§10.8 实施过程中的全部实测发现 |
| [`QURO_V0.2_RESULTS_AND_ANALYSIS.md`](QURO_V0.2_RESULTS_AND_ANALYSIS.md) | **v0.2 实验结果与分析**（主文档）：ξ_off 扫描、D1 实验、定位讨论 |
| [`QURO_RELATED_WORK.md`](QURO_RELATED_WORK.md) | 相关工作梳理与切割 |
| [`PERCEIVER_IO_INNOVATION_ANALYSIS.md`](PERCEIVER_IO_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（Perceiver IO 路线） |
| [`ENCODER_SCALING_INNOVATION_ANALYSIS.md`](ENCODER_SCALING_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（放大 encoder 路线） |

原始实验数据在 [`../results/`](../results/)。

## 当前状态（2026-09-18）

- **已观察到**：单 seed HotpotQA dev、固定 query 表示、B=8，C1 43.10 EM > S 37.50；P 为 54.50。结果支持继续研发，不证明全部机制或部署价值。
- **残差臂已跑完**：恒等起点逐题 2000/2000 精确复现 P；训练后 54.35（p=0.79，打平）；算力匹配的 joint vs p-control 为 −0.10（p=0.94）。**超过 P 的目标未达成。**
- **本轮最重要的发现**：p-control 表明在当前 30000 条上继续训练本身有害（−1.60，p=0.044）。瓶颈在数据配方，不在模块。
- **下一步变量是数据**：HotpotQA 尚有 60447 条未用，46.5% 的文档已在缓存中，增量编码约 19 分钟。设计见 `TRAINING_DATA_REDESIGN.md`，**尚未实现**。
- **当前暂停**：问题压缩 D4/D5、删 system prompt、KD 扫参和忠实性解释、R 的超参调整；KL 修复仍未做，启用 KD 前必须先修 `src/distill.py` 的尾部数值问题。
- **长期约束保留**：本轮冻结离线压缩器，复用已有缓存。
- 历史报告保留供追溯，不将过时摘要或尚未成立的机制论断当成最新执行要求。具体任务见顶部最新交接。
