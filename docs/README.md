# docs/ — QuRO 自己的设计、实验与进展

与 [`../article/`](../article/) 分开：那里是**读别人论文的笔记**，这里是**我们自己写的东西**。

| 文件 | 内容 |
|---|---|
| [`PUBLIC_QA_TRAINING.md`](PUBLIC_QA_TRAINING.md) | 复用 COCOM/PISCO 的 453k 公开混合问题池：固定版本下载、评测排除、检索接入、缺失清单、长度审计和服务器训练命令 |
| [`PUBLIC_QA_TRAINING_RESULTS.md`](PUBLIC_QA_TRAINING_RESULTS.md) | **公开混合集训练结果**：90k 问题 + SPLADE-v3/DeBERTa-v3 检索（kilt-128 top-5），SQX/SQ/S0X/S0 单 seed；HotpotQA test 上混训低于领域内训练，SQ−S0 +2.15 F1，SQX−SQ −1.68 F1 |
| [`CROSS_DOCUMENT_PROJECTOR_VALIDATION.md`](CROSS_DOCUMENT_PROJECTOR_VALIDATION.md) | SQX 复核：旧预测配对 CI、独立数据顺序、两 seed 分阶段验证和 query×跨文档消融，为逐跳读取/关系交互提供依据 |
| [`CROSS_DOCUMENT_PROJECTOR_RESULTS.md`](CROSS_DOCUMENT_PROJECTOR_RESULTS.md) | **SQX 跨文档验证结果**：阶段 A/B/C 完整数字；顺序对齐后 bridge 均值 +0.35 F1pp（CI 跨零，未过门槛）；S0X−S0 接近零，交互项 bridge +0.36；探索性结论 |
| [`PROJECTOR_INPUT_ATTRIBUTION.md`](PROJECTOR_INPUT_ATTRIBUTION.md) | eval-only 投影器输入归因：原始/范数/方向/完整四臂、真实 decoder 残差 hook、答案内容与 EOS 分离、配对分析 |
| [`PROJECTOR_INPUT_ATTRIBUTION_RESULTS.md`](PROJECTOR_INPUT_ATTRIBUTION_RESULTS.md) | **四臂归因实验结果**：约 47 F1pp 收益几乎完全来自方向，范数无贡献；SQ−S0 +0.37 F1 集中在 comparison 题，bridge 题无增益 |
| [`QUERY_PROJECTOR_ABLATION.md`](QUERY_PROJECTOR_ABLATION.md) | 标准 S0 消融：移除 query 分支、同尺寸文档 MLP、初始化对齐及独立训练协议 |
| [`QUERY_PROJECTOR_ABLATION_RESULTS.md`](QUERY_PROJECTOR_ABLATION_RESULTS.md) | **SQ−S0 主消融结果**：去掉 query 分支只少约 0.4 F1（两 seed，全量 test），集中在 comparison 题，bridge 题为零 |
| [`QUERY_MODULATED_FUSION_RESULTS.md`](QUERY_MODULATED_FUSION_RESULTS.md) | **FiLM 调制对照结果**：FiLM−AddG +0.04 F1，零收益；γ 冻结对照 G0 表明共有的 0.9 点来自样本顺序；单 seed 噪声约 0.8 F1 |
| [`SUPPORT_OUTPUT_SUPERVISION_RESULTS.md`](SUPPORT_OUTPUT_SUPERVISION_RESULTS.md) | **输出端支持监督结果**：QA 零收益（−0.01 F1），最终 E 上的头更不用问题；离线 top-k@6 覆盖 87% gold |
| [`QUERY_MODULATED_FUSION.md`](QUERY_MODULATED_FUSION.md) | 问题条件文档调制：等容量 AddG/FiLM、恒等起点、日志与配对确认协议 |
| [`SUPPORT_OUTPUT_SUPERVISION.md`](SUPPORT_OUTPUT_SUPERVISION.md) | 离线 Recall/both@4/@6 与最终 E 分类监督：梯度、匹配训练和判据 |
| [`SUPPORT_DOCUMENT_SUPERVISION.md`](SUPPORT_DOCUMENT_SUPERVISION.md) | SQ 支持文档辅助监督：实现、可见性注释、匹配继续训练与判据 |
| [`SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md`](SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md) | **支持文档监督首轮结果**：QA 零收益（+0.07 F1），辅助头学会分文档但不用问题 |
| [`SHARED_QUERY_PROJECTOR_RESULTS.md`](SHARED_QUERY_PROJECTOR_RESULTS.md) | 共享文档投影器首轮结果：格式收益与约 1 个 F1 点的问题条件化空间 |
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
