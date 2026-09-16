# docs/ — QuRO 自己的设计、实验与进展

与 [`../article/`](../article/) 分开：那里是**读别人论文的笔记**，这里是**我们自己写的东西**。

| 文件 | 内容 |
|---|---|
| [`QURO_EXPERIMENTAL_DESIGN.md`](QURO_EXPERIMENTAL_DESIGN.md) | 实验设计总纲：研究问题、基线、判据、摊薄论证 |
| [`QURO_V0.1_IMPLEMENTATION_PLAN.md`](QURO_V0.1_IMPLEMENTATION_PLAN.md) | v0.1 实施方案 + §10.5–§10.8 实施过程中的全部实测发现 |
| [`QURO_V0.2_RESULTS_AND_ANALYSIS.md`](QURO_V0.2_RESULTS_AND_ANALYSIS.md) | **v0.2 实验结果与分析**（主文档）：ξ_off 扫描、D1 实验、定位讨论 |
| [`QURO_RELATED_WORK.md`](QURO_RELATED_WORK.md) | 相关工作梳理与切割 |
| [`PERCEIVER_IO_INNOVATION_ANALYSIS.md`](PERCEIVER_IO_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（Perceiver IO 路线） |
| [`ENCODER_SCALING_INNOVATION_ANALYSIS.md`](ENCODER_SCALING_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（放大 encoder 路线） |

原始实验数据在 [`../results/`](../results/)。

## 当前状态（2026-09-16）

- **机制已证实**：D1 下 query 条件读出比 query 无关版高 20.85 个 EM 点（p=1.8e-96）
- **但在标准设定下冗余**：D0 下 decoder 拿着问题明文，自己就完成证据匹配
- **定位待重建**：见 `QURO_V0.2_RESULTS_AND_ANALYSIS.md` §8
- **主表一格未填**：同压缩率基线、多数据集、效率曲线全部待做
- **长期约束**：v0.3 之前不微调压缩器（理由见 `QURO_V0.2_RESULTS_AND_ANALYSIS.md` §9）
- **新颖性**：与 RRK(2604.26483) 的切割成立——它输出标量做重排，我们输出喂给生成器的 soft token；RRK 笔记 §9.2 自承未把 query 条件读出作为通用生成问题隔离研究
