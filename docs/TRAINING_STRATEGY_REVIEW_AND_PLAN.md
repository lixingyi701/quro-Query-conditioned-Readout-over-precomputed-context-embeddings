# 训练方案复审：阶段成果、原因分析与下一步计划

> 日期：2026-09-17  
> 审查代码基线：`08f3ae76603734ce74125950443f3d648319d3ef`  
> 依据：ARM_MATRIX_RESULTS、HANDOFF、results/arm_matrix.json，以及训练与读出实现。  
> 本文为本轮讨论形成的建议，尚未实现或运行。未访问服务器原始 checkpoint 和训练日志，数字引用仓库记录。  
> 阅读对象：后续代码实施与实验运行负责人。先读本文，再决定新增训练实验；不要把假说当成已经定位的瓶颈。

## 1. 核心判断

**现在应把改训练方法提到主要任务。** 理由是：对照定义已经修正，多跳任务已有正收益，预算扫描显示单纯增大 B 的收益递减，而训练仍基本沿用初始 CE 配方。

此前先修实现、补对照、扫预算是合理的；现在继续把训练放到最后不合理。但尚不能断言训练是唯一瓶颈，或预算已经被完全排除。

项目目标是同预算、合理训练资源下的可信质量或效率收益，不要求某个模块必须有效。建议主线：

**稳定 query 表示 → 验证优化配方 → 全缓存教师的答案分布蒸馏 → 独立验证支持文档覆盖监督。**

## 2. 目前取得的成果

HotpotQA dev，D0，qdrop=0，固定单预算，seed=42：

| B | C1 EM | S EM | C1−S |
|---:|---:|---:|---:|
| 8 | 43.35 | 37.20 | +6.15，显著 |
| 16 | 45.85 | 43.10 | +2.75，显著 |
| 32 | 46.80 | 47.25 | −0.45，不显著 |
| P：全部 80 latent | 54.50 | — | — |

来源：[结果文档](ARM_MATRIX_RESULTS.md)、[汇总数据](../results/arm_matrix.json)。

当前最有价值的发现：

> 在已测单 seed 条件下，低预算多跳问答中，可学习读出比余弦 top-B 提供更高质量。

它比“系统跑通”或“D1 能传递问题信息”更接近论文贡献。

尚未解决：
- B=32 没有证据说明 C1 超过简单 S。
- 相比 P 仍差 7.70 个 EM 点。
- D0 需要真正 A0 对照，区分一般可学习压缩与 query 条件化的增量。
- 没有多 seed 和完整在线时延，不能声称稳定性或实际加速已成立。
- B=8 共享预算的 42.80 与固定预算的 43.35 不混用。

## 3. 对已有原因分析的修正

### 3.1 “预算不是瓶颈，已经排除”应收窄

B=8→32，C1 仍提升 3.45 EM。正确结论是：

> 当前架构、训练时长与优化配置下，单纯增加 B 无法充分追回差距。

这不证明容量充足。更大的 B 可能需要更好的槽位分工、训练时长、初始化或表示对齐。暂缓扩大 B 扫描合理，但不作信息论排除。

### 3.2 不能限定“只剩合成与训练信号不足两个原因”

还可能存在：query 表示漂移、readout/decoder 学习速度失衡、先验强度不合适、槽位冗余、latent 混合后的可读性下降、关键证据截断、欠拟合或过拟合。原因可能共同作用。

### 3.3 分支干预不决定唯一因果解释

输出为 E=s·alpha·Z+Delta。

- Delta 通过 cross-attention 读取 memory；即使 delta_only 有效，也不意味着注意力不是因果路径。
- pool_only 下降可能来自完整模型依赖 Delta 校准后的分布，不证明 pooling 无效。
- 同 checkpoint 干预测“当前依赖”；移除分支重训测“可补偿能力”。
- 不把干预结果作为是否允许测试覆盖监督的硬门槛。
- pool_only 是加权混合，不是严格的离散选择。

### 3.4 “证据值”不是精确信息容量

clean EM − mismatch-doc EM 是干预敏感性指标。错配文档可能主动误导，且各臂 decoder 适配不同。

不要据此声称“精确丢失 30% 证据”“23 分是表示容量上限”。同时保留无文档与错配文档控制。

### 3.5 先验的预算效应仍有混淆

B=8、qdrop=1 与 B=32、qdrop=0 同时改变两个变量，不能把差异完全归因于 B。需同数据、同 qdrop、同优化条件比较。

删除余弦先验也不会自动省掉 7B query 编码：C0 的 query cross-attention 仍需要 query hidden states。C1 通常复用同一次编码的 pooled vector。A0 更快不能直接估算“余弦先验的边际成本”。

### 3.6 CE 的梯度不是简单的“稀疏信号”

CE 对 logits 已有稠密梯度。KL 的价值是教师分布提供的额外信息，不是把无梯度变成有梯度。

冻结 decoder 参数也不阻断输入 embedding 的梯度。若只训 readout，decoder 前向仍须保留对输入的 autograd，不能整个包进 no_grad。

## 4. 当前训练配方的具体不足

代码基线中：
- 主目标为答案 CE；
- 前约 300 步施加逐渐衰减的残差惩罚；
- readout 与 decoder LoRA 同时更新；
- 共用一个 AdamW 学习率，默认 1e-4；
- 3000 步后保存最后 checkpoint，再评测；
- KL 与支持文档监督仅在计划中，尚未实现。

按默认 batch=8、grad_accum=2、30k 训练样本估计，3000 步约 1.6 次数据遍历。需以运行配置确认。这个数字既不证明欠训练，也不足以证明收敛。

优先补充训练/验证曲线、最佳 checkpoint 和模块梯度/更新统计，避免只凭最终一次 EM 判断容量。

## 5. 第一阶段：稳定表示与优化基线

### 5.1 固定第一轮实验条件

K=10，HotpotQA 标准 distractor 数据，D0，qdrop=0，固定 B=16，冻结离线 PISCO 缓存。

选择 B=16 是因为已有对 S 的增量、质量高于 B=8；不是宣称 B=16 最优。方案确定后回测 B=8/32。

### 5.2 W3：显式 query 表示策略

- fixed_adapter：冻结 backbone，共享底座但 query 使用独立冻结 adapter；decoder 使用可训练 adapter。
- shared_current：保留当前行为作为历史对照，不预设 fixed_adapter 一定涨点。
- 同一比较组固定 query 策略，不在引入 KL 时悄悄同时换策略。

验收：
1. decoder 更新前后，fixed query 输出在容差内一致。
2. query 参数不变，decoder 可训参数确有更新。
3. adapter 切换后恢复 requires_grad、optimizer 参数身份与 train/eval 状态。
4. checkpoint 恢复包含冻结 query adapter 的可追溯状态。

### 5.3 分组优化与收敛检查

候选起点（不是验证过的最优值）：
- readout LR=1e-4；
- decoder LoRA LR=1e-5；
- 其他训练预算尽量保持一致。

每约 500 步评测固定开发子集，记录 train/validation NLL、EM/F1，保存 best 与 last。最终在完整 dev 上核验候选，测试集不参与选择。

若延长训练，CE 对照也要获得相同额外步数。记录累计更新、样本量与训练成本，不能把更多计算归因于新损失。

短暂冻结 decoder 的 readout-only warmup 可以另作实验，先不与所有改动同时叠加。冻结 decoder 时仍保留对 soft token 的梯度。

## 6. 第二阶段：P 的答案分布蒸馏

### 6.1 教师与学生

教师 T：已训练 P checkpoint，读取同一份文档缓存的全部 latent，固定参数，eval。  
学生 S：同样问题与缓存，经 readout 变成 B 个 soft token，再生成答案。

教师条件：
p_T(y_t | q, Z, y_<t)

学生条件：
p_S(y_t | q, R_theta(q,Z), y_<t)

目的：使短读出尽量保留全部缓存输入时的预测行为。它主要针对缓存到短输出的损失，不要求重新训练离线压缩器。

### 6.2 损失

```text
L = L_CE + lambda_KD * T^2 * mean_answer_tokens KL(p_teacher^T || p_student^T)
```

保留 gold CE。P 会犯错，不能用教师完全替代标签，也不能保证追回全部 7.70 分。

可先固定 temperature=2，小范围试 lambda_KD={0.1,0.5,1.0}。先观察 loss 和梯度数量级，再决定是否调整；不做大规模盲扫。

### 6.3 实现契约

1. 教师必须固定，不能随学生 decoder LoRA 更新；优先离线生成训练集教师分布，避免状态串扰。
2. 第一版统一在 gold answer 上 teacher forcing，教师/学生使用完全相同的答案前缀。
3. 仅在有效答案位置计算 KL，包括协议约定的 EOS；不计算 prompt/padding。
4. 两者 prompt 长度不同，按答案相对位置对齐，正确处理 causal shift。
5. tokenizer/vocabulary/目标截断规则一致；缓存记录教师 hash、tokenizer 版本、target ids、temperature 与有效长度。
6. KL 的 log_softmax/概率运算用 FP32，并按有效答案 token 数归一化。
7. 开始大跑前，以小批全词表在线教师计算对照离线缓存，检查目标一致性。
8. 教师分布缓存只来自训练数据；开发/测试不得作为蒸馏训练样本。

### 6.4 top-k 缓存不能随意重新归一化

只存 top-k logits，再对这 k 项 softmax，不等于原始教师分布。

建议保存全词表归一化后的 top-k 概率与剩余尾部质量。学生取相同 token 的概率，剩余词表聚合成 tail：

```text
KL_approx =
  sum_{v in topk} pT(v) log[pT(v)/pS(v)]
  + pT(tail) log[pT(tail)/pS(tail)]
```

这是聚合近似，不是完整 KL；温度固定后再生成缓存。注意尾部数值稳定性。实际缓存大小按真实答案长度、索引 dtype 与概率 dtype 测算，不套用统一“16 token/184MB”估值。

若教师错误明显造成冲突，可后续探索训练集上的可靠性加权，但保留未经筛选的基线，并报告额外筛选策略；不把它混进第一版 KD。

## 7. 第三阶段：支持文档覆盖监督

HotpotQA 提供 supporting_facts/gold_ranks，可以提供直接的训练侧证据信号。

将多头注意力聚合为文档级质量：
```text
a[b,d] = sum_{j in document d} mean_heads alpha[h,b,j]
L_cov = -mean_{d in gold_documents} log(eps + max_b a[b,d])
```

目标：每篇支持文档至少被一个 slot 明显关注，不强迫每个 slot 都平均关注两篇 gold。

这只是候选代理目标：
- max 会使梯度集中到当前最佳槽，可根据实测再考虑平滑聚合，不预设其最优。
- 不保证槽位多样性或信息保真；attention 好看不等于答案更准。
- 先单独比较 CE+coverage，再与 KL 组合。
- 文档级标签不能决定哪个 latent 包含哪句话，不对 gold 文档内所有 latent 强制均匀监督。
- supporting_facts 只用于训练，不进入推理输入。
- 检查支持句是否被离线编码长度截断；被裁掉的证据不能通过监督恢复。缺失样本的辅助 loss 应有明确处理。
- 确认返回的 attention 没有 detach，辅助 loss 确实更新目标参数。
- 最终报告 QA 与支持文档覆盖，以及必要的文档删除干预，不能仅报告训练 loss。

如果引入该额外标注，应增加同监督的轻量选择器基线，区分标注收益与复杂结构收益。

## 8. 最小实验矩阵

第一轮固定 B=16、D0、相同初始化来源与数据协议：

| 运行 | 训练配方 | 回答的问题 |
|---|---|---|
| R0 | fixed query + 原 CE 配方 | W3 改动后的新基线 |
| R1 | R0 + 分组学习率 | 是否存在优化失衡 |
| R2 | R1 + KL | 完整缓存教师是否有效 |
| R3 | R1 + coverage | 直接证据信号是否有效 |
| R4 | R1 + KL + coverage | 两类监督是否互补 |

R4 等 R2/R3 至少一项有效再做。所有运行匹配学生更新预算与 checkpoint 选择规则；教师预计算成本另报。

必须补的对照：
- S+KL：S 的 decoder 也能从教师获益。最终比 C+KL 与 S+KL，不只比 C+KL 与旧 S。
- D0 A0：检查训练改善的是一般可学习压缩还是条件化。
- 若 coverage 有效，补同标注的简单监督式选择器。
- 关键获胜配方做多 seed；paired test 不替代 seed 方差。

分支干预和 gold 覆盖诊断可同步进行，不作为全部训练改动的阻塞关卡。

## 9. 如何按结果决策

| 观察 | 优先解释/下一步 |
|---|---|
| R1 明显提升 | 优化配方是重要因素，先保留简单改动 |
| R2 提升而 R3 无效 | 答案分布/可读性监督更有帮助，避免硬加覆盖 |
| R3 提升 | 选择训练信号有价值，查两篇支持文档覆盖与 QA 是否同步改善 |
| R2/R3 单独有效，R4 无效 | 可能目标冲突或权重失衡，不强行保留组合 |
| C 与 S 在同 KD 下增幅相同 | 蒸馏主要是通用训练收益，重新核算 readout 的净贡献 |
| 增加步数仅降低 train loss，dev 退化 | 优先处理泛化，不继续延长训练 |
| 所有训练调整均无稳定增量 | 再根据诊断考虑表示/架构，而不是无限叠 loss |

不将单一指标的小幅波动当作收益。主指标、开发协议、训练预算事先固定；关键结果用多 seed 和配对置信区间确认。

## 10. 下一步代码与运行清单

| 优先级 | 代码位置 | 工作 |
|---|---|---|
| P0 | config.py、src/model.py、generator 相关代码 | fixed_adapter/shared_current，冻结状态与 checkpoint 验收 |
| P0 | src/train.py | 参数分组、间隔验证、best/last checkpoint、模块梯度统计 |
| P1 | teacher 缓存生成脚本 | 固定 P、答案相对位置、top-k+tail、metadata 与数值对照 |
| P1 | src/model.py、训练配置 | 返回答案 logits，CE+KD，mask/shift/FP32 归一化 |
| P1 | src/readout.py、数据 collator | gold 文档 mask、可微 attention、coverage loss |
| P1 | 运行脚本与结果收集 | R0–R4、S+KD、D0 A0、训练资源和协议记录 |
| P1 | 独立 benchmark | 单次 query 编码/读出，完整在线计时 |
| P2 | 最佳配方 | B=8/32、多 seed、保留测试集 |

本轮先不实施 RL、大规模对比学习、扩大 K 或端到端微调离线压缩器。不是永久排除，而是当前成本与归因复杂度更高。

## 11. 对论文贡献的定位

候选贡献仍是：低预算条件下，在可复用压缩文档表示上实现更有效的在线读出。

KL、CE、普通注意力监督本身不是新颖性。它们可以让方法得到充分训练，但论文需要展示：
- 同预算下对强简单基线的稳定增量；
- 获益的任务条件与失败边界；
- 完整在线质量—成本权衡；
- 同训练配方比较后的净贡献。

现在有继续推进的依据，尚不能保证 CCF C 录用。下一阶段应回答“充分训练后，这个低预算收益能否稳定扩大并具有实际价值”。

## 12. 来源

- [ARM_MATRIX_RESULTS](ARM_MATRIX_RESULTS.md)
- [HANDOFF](HANDOFF.md)
- [训练实现](../src/train.py)
- [模型实现](../src/model.py)
- [readout 实现](../src/readout.py)
- [配置](../config.py)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

知识蒸馏的一般思想参考上述原文；本文的阶段、损失组合和超参数是针对当前项目的待验证建议，不是引用论文已证明的结论。
