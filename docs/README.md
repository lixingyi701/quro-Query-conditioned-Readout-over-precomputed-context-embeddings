# docs/ — QuRO 自己的设计、实验与进展

与 [`../article/`](../article/) 分开：那里是**读别人论文的笔记**，这里是**我们自己写的东西**。

| 文件 | 内容 |
|---|---|
| [`RESIDUAL_RESULTS.md`](RESIDUAL_RESULTS.md) | **最新优先阅读**：残差臂三轮完整结果——恒等精确成立；数据耗尽是第一轮零收益的真因；模块边际三 seed 方向一致但未确立，且在 TriviaQA 迁移上不成立 |
| [`TRAINING_DATA_REDESIGN.md`](TRAINING_DATA_REDESIGN.md) | 数据盘点与 2×2 判决方案（已执行，H1 成立） |
| [`RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md`](RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md) | 本轮残差方向的交接与执行顺序（步骤 A–E 均已执行完毕） |
| [`PISCO_RESIDUAL_EXPERIMENT.md`](PISCO_RESIDUAL_EXPERIMENT.md) | 已实现的 R 臂、从实际 P checkpoint 起步、恒等验证与训练命令 |
| [`QURO_EXPERIMENTAL_DESIGN.md`](QURO_EXPERIMENTAL_DESIGN.md) | 实验设计总纲：研究问题、基线、判据、摊薄论证 |
| [`QURO_V0.1_IMPLEMENTATION_PLAN.md`](QURO_V0.1_IMPLEMENTATION_PLAN.md) | v0.1 实施方案 + §10.5–§10.8 实施过程中的全部实测发现 |
| [`QURO_V0.2_RESULTS_AND_ANALYSIS.md`](QURO_V0.2_RESULTS_AND_ANALYSIS.md) | **v0.2 实验结果与分析**（主文档）：ξ_off 扫描、D1 实验、定位讨论 |
| [`QURO_RELATED_WORK.md`](QURO_RELATED_WORK.md) | 相关工作梳理与切割 |
| [`PERCEIVER_IO_INNOVATION_ANALYSIS.md`](PERCEIVER_IO_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（Perceiver IO 路线） |
| [`ENCODER_SCALING_INNOVATION_ANALYSIS.md`](ENCODER_SCALING_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（放大 encoder 路线） |

原始实验数据在 [`../results/`](../results/)。

## 当前状态（2026-09-20）

- **已观察到**：单 seed HotpotQA dev、固定 query 表示、B=8，C1 43.10 EM > S 37.50；P 为 54.50。结果支持继续研发，不证明全部机制或部署价值。
- **残差臂三轮已跑完**（58 runs / 48 paired tests 入库）。恒等起点在两份缓存下均逐题 2000/2000 精确复现 P。
- **数据是决定性变量**：无模块的 p-control 在 P 训练过的 30000 条上 −1.60（p=0.044），在留出的 60447 条上 +1.75（p=0.030），摆动 +3.35（p=0.00015）。第一轮"继续训练有害"的结论**只在已拟合的数据上成立**。
- **HotpotQA 上的提升以跨域能力为代价**：同一批权重零样本迁移到 TriviaQA，六个臂全部低于源 P 的 71.95（68.35~70.40，全部 p ≤ 0.02）。掉的是参数记忆（错配地板 59.50→53.45~56.00），但主指标是跌的，证据值升高不作为收益。
- **模块边际方向一致但未确立，且不迁移**：3000 步 HotpotQA 三 seed 为 +1.10 / +0.10 / +1.00（均值 +0.73±0.55，无一单独显著）；9000 步为 −0.05（p=1.00）；TriviaQA 为 −0.75 / +0.15 / +0.20（均值 −0.13）。
- **不要用 57.35 代表方法**：那是 seed 42，而 seed 42 对 joint 和 p-control 同时最优，三 seed 均值为 56.08；且它正是迁移时唯一反号的那个。
- **下一步**：补 seed 的性价比已下降（三个独立方向都指向零）。零机时的下一步是按 `hop_type` 拆 bridge/comparison 复查边际是否只集中在多跳。在此之前不调 R 的超参——效应量小于 seed 噪声。
- **当前暂停**：问题压缩 D4/D5、删 system prompt、KD 扫参和忠实性解释、R 的超参调整；KL 修复仍未做，启用 KD 前必须先修 `src/distill.py` 的尾部数值问题。
- **长期约束保留**：本轮冻结离线压缩器，复用已有缓存。
- 历史报告保留供追溯，不将过时摘要或尚未成立的机制论断当成最新执行要求。具体任务见顶部最新交接。
