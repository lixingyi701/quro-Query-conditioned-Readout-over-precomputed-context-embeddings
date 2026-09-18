# docs/ — QuRO 自己的设计、实验与进展

与 [`../article/`](../article/) 分开：那里是**读别人论文的笔记**，这里是**我们自己写的东西**。

| 文件 | 内容 |
|---|---|
| [`RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md`](RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md) | **最新优先阅读**：用户确定的残差方向、讨论结论、下一步执行顺序 |
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
- **用户确定的主线**：先从 P 的全部原始 latent 开始，用 query 条件化残差争取质量提点，再谈预算缩减和消融。
- **代码已实现**：R 臂零初始化时保留 P 输入，加载实际训练过的 P decoder；CPU 契约验证通过，真实 7B/Hotpot 实验待服务器运行。
- **当前暂停**：问题压缩 D4/D5、删 system prompt、KD 扫参和忠实性解释；KL 修复不是本轮 CE-only 的前置依赖。
- **长期约束保留**：本轮冻结离线压缩器，复用已有缓存。
- 历史报告保留供追溯，不将过时摘要或尚未成立的机制论断当成最新执行要求。具体任务见顶部最新交接。
