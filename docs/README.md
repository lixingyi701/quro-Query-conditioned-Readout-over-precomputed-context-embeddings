# docs/ — QuRO 自己的设计、实验与进展

与 [`../article/`](../article/) 分开：那里是**读别人论文的笔记**，这里是**我们自己写的东西**。

| 文件 | 内容 |
|---|---|
| [`QURO_EXPERIMENTS_AND_MULTIDATASET_PLAN_20261008.md`](QURO_EXPERIMENTS_AND_MULTIDATASET_PLAN_20261008.md) | **截至 2026-10-08 的实验总账与多数据集计划**：完整历史结果、SQ 的 +0.62/+0.37 F1 正向增量、最新 QER 结果，以及五数据集评估、统一训练、基线与复用成本协议 |
| [`QURO_PROJECTOR_RESEARCH_UPDATE_20261006.md`](QURO_PROJECTOR_RESEARCH_UPDATE_20261006.md) | **截至 2026-10-06 的投影器分支完整总结**：场景与动机、创新边界、具体实现、各轮实验、标准 SQ/S0 两种子消融、当前问题与后续缺口 |
| [`QURO_RESEARCH_UPDATE_20261002.md`](QURO_RESEARCH_UPDATE_20261002.md) | **截至 2026-10-02 的综合实验总结**：六个分支、方案解释、正负结果、失败定位与结论边界；包含小预算读出、R/RQ、Direct-State 和 W |
| [`QURO_EXPERIMENTAL_DESIGN.md`](QURO_EXPERIMENTAL_DESIGN.md) | 实验设计总纲：研究问题、基线、判据、摊薄论证 |
| [`QURO_V0.1_IMPLEMENTATION_PLAN.md`](QURO_V0.1_IMPLEMENTATION_PLAN.md) | v0.1 实施方案 + §10.5–§10.8 实施过程中的全部实测发现 |
| [`QURO_V0.2_RESULTS_AND_ANALYSIS.md`](QURO_V0.2_RESULTS_AND_ANALYSIS.md) | **v0.2 实验结果与分析**（主文档）：ξ_off 扫描、D1 实验、定位讨论 |
| [`QURO_RELATED_WORK.md`](QURO_RELATED_WORK.md) | 相关工作梳理与切割 |
| [`PERCEIVER_IO_INNOVATION_ANALYSIS.md`](PERCEIVER_IO_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（Perceiver IO 路线） |
| [`ENCODER_SCALING_INNOVATION_ANALYSIS.md`](ENCODER_SCALING_INNOVATION_ANALYSIS.md) | 早期方案可行性分析（放大 encoder 路线） |

原始实验数据在 [`../results/`](../results/)。

## 当前主线（2026-10-08）

详见 [实验总账与多数据集评估计划](QURO_EXPERIMENTS_AND_MULTIDATASET_PLAN_20261008.md)。后续实验优先级以这份报告为准；下方保留此前的阶段记录。

- **保留 SQ 的小幅正向 query 增量**：首轮 SQ−S0m 为 +0.62 F1；标准 SQ−S0 两种子平均为 +0.37 F1，95% 配对 CI [−0.01, +0.74]。两轮比较分别报告，不合并成精确的 +0.50。
- **暂停新增 QER 扩展**：seed42、500 步、dev2000 上 A/B/C F1 为 63.70/63.50/63.50；这轮内容监督未带来额外 QA 收益。
- **扩展原 SQ 的实验覆盖**：HotpotQA、NQ、TriviaQA、PopQA、WebQuestions；先评已有 SQ/S0 权重，再做统一混合训练与三种子主实验，并补基线、统一指标和真实复用成本。

## 投影器分支进展（截至 2026-10-06）

详见 [投影器分支完整总结](QURO_PROJECTOR_RESEARCH_UPDATE_20261006.md)。该报告基于 `feat/pisco-joint-query-projector@102653d`，不表示分支代码已合并到 main。

- **共同适配收益明显，query 增量尚未确立**：标准 SQ−S0 两种子平均 test F1 +0.37，题目级配对 95% CI [−0.01, +0.74]；comparison 存在局部信号，bridge 接近零。
- **支持监督与 γ 扩展未建立额外 QA 收益**：分类改善、错配问题敏感性与正常 QA 增量需要分别判断。
- **下一步优先补机制与复用证据**：复核当前 SQ/S0 的读取缺口，检验同一缓存多问题，并测量端到端成本。

## 综合进展（截至 2026-10-02）

详见 [综合实验总结](QURO_RESEARCH_UPDATE_20261002.md)。该报告汇总 feature 分支记录，不表示这些分支的实现已合并到 main。

- **诊断进展快于方法进展**：raw–memory 能力缺口已确认，大部分缺口定位到含答案段；尚未区分信息损失与读取失败。
- **局部正结果存在**：低预算 HotpotQA 上可学习读出优于余弦规则，域内 decoder 适配有效；不能把这些结果直接扩大为新增模块稳定优于强基线。
- **R/RQ、缩放、Direct-State 和 W 尚未建立稳定增量**：应使用匹配训练对照，保留不同预算、数据集与样本量的边界。
- **监督审计已实现，效果尚未知**：发布版重置与答案段教师目标属于后续候选，普通 SKD 是基线。

## 历史状态（2026-09-16，后续修正见综合报告）

以下保留当时的阶段记录；对照定义及机制解释以综合报告中的更正为准。

- **机制已证实**：D1 下 query 条件读出比 query 无关版高 20.85 个 EM 点（p=1.8e-96）
- **但在标准设定下冗余**：D0 下 decoder 拿着问题明文，自己就完成证据匹配
- **定位待重建**：见 `QURO_V0.2_RESULTS_AND_ANALYSIS.md` §8
- **主表一格未填**：同压缩率基线、多数据集、效率曲线全部待做
- **长期约束**：v0.3 之前不微调压缩器（理由见 `QURO_V0.2_RESULTS_AND_ANALYSIS.md` §9）
- **新颖性**：与 RRK(2604.26483) 的切割成立——它输出标量做重排，我们输出喂给生成器的 soft token；RRK 笔记 §9.2 自承未把 query 条件读出作为通用生成问题隔离研究
