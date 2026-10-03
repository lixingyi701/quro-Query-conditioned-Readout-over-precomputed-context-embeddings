# QuRO 向学长汇报：诊断逐渐明确，解决方案尚未建立

日期：2026-10-02。

> 本文是截至 2026-10-02 的实验总结快照，于 2026-10-03 上传 main。分支头、提交数量及“最新”结论均指该总结时点；后续实验请另行记录。

根据 GitHub 当前六个现存分支、各分支全部可达提交记录、关键实验文档、部分原始汇总 JSON 与最新监督接口整理。共 116 条去重提交；这不是 116 次实验。未重跑 GPU，未访问服务器原始日志。已删除分支、未推送实验不在此次可核查范围。

## 1. 汇报的核心结论

**目前诊断进展明显快于方法进展：已经确认压缩证据相对原文的任务表现缺口，并将大部分缺口定位到含答案段；但尚未区分压缩信息损失与读取失败，也没有建立一个稳定、可归因于新增模块的解决方案。**

“完全没效果”和“已经找到根因”都不准确。前者忽略低预算读出与域内 LoRA 的有效结果，后者忽略从输入干预定位到机制解释之间尚缺的证据。

English summary: Diagnostic progress is stronger than methodological progress. The raw–memory gap is established and largely localized to the answer-bearing paragraph. However, information loss and readout failure remain unresolved, and no stable benefit attributable to the new modules has been established.

## 2. 六个分支各自负责什么

| 分支 | 当前头提交 | 可达提交数 | 实际研究内容 | 当前结论 |
|---|---|---:|---|---|
| `feat/pisco-identity-residual` | [f9c79356](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/f9c79356afe749a645574dc145640fb5ca9ddf33) | 66 | 全量 latent 主体 + 恒等初始化 query 残差 | 弱正向域内边际，未稳定建立，未零样本迁移 |
| `feat/pisco-query-writeback` | [0f03522b](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/0f03522b6e1db4a75cb911bd5abc80959e7c25b6) | 68 | query-as-Q 读取后写回 latent | 对匹配对照的增量约为零 |
| `feat/published-reader-reset` | [34828e98](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/34828e98a7a587e331b230d33ef6bc1096b2a366) | 108 | 恢复发布版起点，审计任务输出序列监督 | 已实现审计入口；尚无真实模型有效性结果 |
| `feat/reader-state-workspace` | [2cdf0295](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/2cdf029554fe523e29df0f31825f25561a28d8d0) | 107 | Direct-CE、Direct-State、W-CE 与退化诊断 | State 没有 QA 净收益，W 未形成有效能力 |
| `feat/selecom-infeasibility` | [7f477ebf](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/7f477ebfe7f7152627435be0bd8bd5eec76a2d94) | 103 | SeleCom 机制复现、几何/缩放、D0/D2、raw/memory 混合诊断 | 不支持简单 attention dominance 解释；缺口定位到含答案段 |
| `main` | [580af023](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/580af023c3d5bf8fbab592a7587916f3b5b5dd13) | 65 | 早期 QuRO、余弦先验、预算扫描、KD、query 压缩 | 小预算下有局部收益，未补齐强 P baseline 差距 |

这些分支大量共享历史，不能将其理解为六项独立研究成功或失败。main 的头提交是 9 月 23 日，最新研究记录位于 feature 分支；仅看 main 会遗漏后续否定和修订。

## 3. 原始目标与当前研究焦点的偏移

原目标是“可复用文档缓存 + query 条件轻量读出”：离线压缩与在线预算分离，同一文档应服务多个不同 query，减少重复计算。当前多数结果围绕单次 QA、固定预算和读取行为诊断。

因此，**离线可缓存是系统性质，尚不是本项目独立建立的创新效果**。PISCO/COCOM 已有离线压缩基础；本项目仍需证明在线读出的额外价值。已有材料尚未给出完整的固定文档多 query 主实验、端到端延迟与复用摊薄曲线。不能仅凭 soft token 数减少，宣称已获得相同比例的系统加速。

来源：[原始实验设计](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/580af023c3d5bf8fbab592a7587916f3b5b5dd13/docs/QURO_EXPERIMENTAL_DESIGN.md)。

## 4. 实验轨迹：做了什么，最终留下了什么

### 4.0 先看懂名字：这些方案分别改了哪里？

以下都以冻结的 PISCO 文档缓存为基础。记原文为 $D$，问题为 $q$，离线缓存为 $Z_D$，答案为 $a$。一次召回 $K$ 篇文档、每篇有 $m$ 个 latent，共有 $M=K m$ 个向量。HotpotQA 的常见设置是 $K=10,m=8$，即最多 80 个有效向量。

**latent / soft token** 是连续向量，不是解压后可直接阅读的词句。**readout（读出）** 是将缓存加工成 decoder 能消费的证据输入。**Cross-Attention（CA，交叉注意力）** 中，Q 决定“谁在读取”，K/V 来自“被读取的内容”；这里的 attention Q 不总等于自然语言问题 q。

| 名称 | 一句话含义 | 证据送入 decoder 的方式 | 主要改变 |
|---|---|---|---|
| 全量 P baseline | 把全部缓存直接交给 decoder | $[Z_D][q]\rightarrow a$ | 原始参照 |
| 小预算读出 C | 根据问题，把全部缓存再压成少量证据向量 | $[E_B(q,Z_D)][q]\rightarrow a$，例如 80→8 | 在线证据数量与内容 |
| 全量残差 R | 全部缓存保留，每个向量加上问题相关的修正 | $[Z_D+\Delta_R(q,Z_D)][q]\rightarrow a$ | 在线证据向量的内容 |
| Query 写回 RQ | 问题先从缓存读取，再把读到的特征分配回全部 latent | $[Z_D+\Delta_{RQ}(q,Z_D)][q]\rightarrow a$ | 修正量的计算方式 |
| Direct-State | 原输入不变，增加“像 raw 教师一样形成中间状态”的监督 | 仍是 $[Z_D][q]\rightarrow a$ | 训练目标 |
| W-CE | 缓存放在序列外，由 decoder 内的工作位置分层读取 | $[q][W]\rightarrow a$，外部 $Z_D$ 经 CA 写入 W | 证据接口与计算位置 |

方括号只表示输入顺序，省略 system/chat 模板。表中的“全量”指**送入全部已缓存向量**，不表示缓存无损保留了原文所有信息。

**这些在线操作都复用预计算的离线缓存，不为每个问题重新压缩原文。** R/RQ 每个问题生成的 E 是临时在线结果；它们不覆盖磁盘上的 $Z_D$。W 也不修改外部 Z，更新的是 W 的状态。历史不同轮次曾更换或扩充缓存，不能据此认为所有实验都使用了字节相同的缓存文件。

符号补充：$H_q$ 是问题编码器输出的 token 表示，h 是向量维度；CE 是答案交叉熵，LoRA 是本项目用于适配 decoder 的可训练低秩参数；P₁ 是域内训练后的 decoder，具体 checkpoint 依相应实验记录。读出基线臂 S 表示余弦 top-B，后文小写 s 表示 prompt 末位置，两者无关。

English guide: Small-budget readout compresses the cache again; R and RQ refine all cached vectors; Direct-State changes supervision; W changes where and how the decoder receives evidence.

### 4.1 小预算二次读出：有局部正结果，但未超过强基线

**它具体在做什么？** 最早的目标是“原文已离线压缩一次，在线再按问题压一次”。例如 10 篇文档共 80 个 latent，decoder 不再读全部 80 个，而只读 B=8 个输出向量。B 是证据 soft-token 预算，不是答案长度，也不是召回文档数。

C 使用 B 个可学习的输出槽位。以 xattn 配置为例，槽位先读取问题表示，再读取文档缓存；槽位之间还做 self-attention，允许不同槽位协调所承载的内容。这里的槽位是 readout 模块内部的工作向量，不是后来插进 decoder 的 W。

省略投影、FFN、head 和多轮细节，可写成：

$
U_B=\operatorname{QueryCondition}(\text{learned slots}_B,H_q),
\qquad
\alpha=\operatorname{AttentionWeights}(U_B,Z_D),
$
$
E_B=s_{\mathrm{scale}}\alpha Z_D+\Delta_B
\in\mathbb R^{B\times h}.
$

其中 $\alpha$ 的形状是 $B\times M$：每个输出槽位对全部缓存向量给出一组权重。池化旁路 $\alpha Z_D$ 直接组合原 latent，$s_{\mathrm{scale}}$ 是可学习尺度，$\Delta_B$ 是学习出来的补充；余弦先验会在启用时偏置注意力分数。最终输入是 $[E_B][q]$，正常 D0 下问题明文仍在 decoder 中。

**“读出”不等于从 80 个中硬挑 8 个。** 每个输出可以是多个 latent 的加权组合，再加学习修正。非参数基线 S 才是按问题相似度硬选 top-B、原样取出对应向量。A/C 的区别则是有没有相应 query 条件路径，必须按修正后的 A0/A1/C0/C1 定义比较。

例如同一份传记，问出生地时希望 8 个输出更多承载出生相关信息，问毕业学校时希望输出内容改变。这是预期功能，不能仅看注意力变了就宣布证据选对了。

**它也曾使用“残差”结构，但与后来的 R 不同。** 它的旁路是 B 个池化输出，不是原始 M 个 latent；即使 $\Delta_B=0$，80→8 的二次压缩已经发生，零步不等于全量 P。后来的 R 才有严格的 $E=Z_D$ 恒等起点。


实现参考：[早期读出代码](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/580af023c3d5bf8fbab592a7587916f3b5b5dd13/src/readout.py)。

**实验结果：**

早期 TriviaQA 上 C 与 A/S 接近。D1 删除 decoder 的问题明文后，读出可以携带 query 信息；这是能力诊断，不能直接证明 D0 正常使用场景下有额外收益。

旧 A 对照实际上仍开着 query 余弦先验。提交 328b9cca 修正为 A0/A1/C0/C1 四格。后续 S 的 query 表示路径又被发现受 train-mode dropout 影响，修复见 f70d7284、d5768100。汇报应使用修正后的匹配结果：

- HotpotQA，D0，B=8，fixed_adapter：C1 EM 43.10，S 37.50，差 +5.60pp。
- C1 加 KD λ=.5：EM 44.10，仍低于 P 的 54.50；KD 的 +1.00pp EM 未显著，substring +1.85pp 为弱正向结果。
- 预算从 B=8 扩到 B=32，旧记录中 C1 EM 43.35→46.80，仍距 P 7.70pp。S 的旧预算曲线含已知绕行 bug，不应作为修正后新结果引用。
- 80 个 soft tokens 降到 8 个，不等于总 prefill 降十倍；该主表中总 prefill 约 156→75，仅约 2.08 倍。

结论：**可学习读出相对简单规则有局部价值，但二次压缩造成的质量代价尚未解决。** 所有这些比较受数据集、预算和单 seed 限定。

来源：[臂矩阵](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/580af023c3d5bf8fbab592a7587916f3b5b5dd13/docs/ARM_MATRIX_RESULTS.md)；[训练配方及修正结果](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/580af023c3d5bf8fbab592a7587916f3b5b5dd13/docs/TRAINING_RECIPE_RESULTS.md)。

### 4.2 全量残差 R：最佳成绩不能直接归因于模块

**“学习 query 条件残差”就是保留原始缓存，加一份随问题变化的修正：**

$
E=Z_D+\Delta_R(Z_D,q),\qquad E,Z_D\in\mathbb R^{M\times h}.
$

R 的修正分支先让**每个 latent 读取问题**（latent 作 attention Q，问题表示作 K/V），再让这些分支状态做 latent self-attention，结合其他文档的信息，最后投影出同样数量、同样维度的 $\Delta_R$。原始 $Z_D$ 通过恒等旁路直接保留。

没有人工给定的“正确残差”。它通过最终答案 CE 学习：冻结 decoder 时，梯度仍可经过 decoder 传回修正模块；joint 则同时训练模块和 decoder LoRA。p-control 是不加模块、仅给 decoder 相同额外训练预算的对照。

输出投影零初始化，因此零步 $\Delta_R=0,E=Z_D$。这解决“新增模块一上来就损坏原接口”的问题，**不保证之后学到的修正有用，也不强制修正量始终很小**。R 保留全部 latent，本轮没有进一步缩短证据序列，也不据此声称加速。


**实验结果：**

为了减少接口扰动，保留全部 80 个 latent，只学习 query 条件残差。恒等初始化与 P 在 2000 题上逐题一致，避免随机接口损伤。

- 原 30k 数据继续训练：joint 52.80，p-control 52.90，模块增量 −0.10pp。
- 换成原 P 未训练的 60447 道题，seed42：joint 57.35，p-control 56.25，模块增量 +1.10pp，而不是相对旧 P 的 +2.85pp。
- 三 seed 增量为 +1.10/+0.10/+1.00，均值 +0.73pp，方向为正但未统计建立。
- 同 seed 9000 步：joint 56.05，p-control 56.10，边际 −0.05pp。
- TriviaQA 零样本迁移，三 seed 平均模块边际 −0.13pp；Hotpot 域内增益未迁移。

结论：**存在弱正向线索，尚无稳定方法增量。** 换新训练数据对继续训练的作用很大，不应把新数据与额外 LoRA 训练的收益全部计给残差模块；迁移失败也不等于证明模块在所有多跳任务无效。

来源：[残差三轮结果](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/f9c79356afe749a645574dc145640fb5ca9ddf33/docs/RESIDUAL_RESULTS.md)；提交 2428f8a6、8c6cd5c2、f9c79356。

### 4.3 Query 写回 RQ：改变读取方向没有产生稳定收益

**RQ 仍然输出全部 M 个 latent 的残差修正，但把“谁先读取谁”换了。**

- R：每个 latent 先读取问题，再在 latent 分支内交互。
- RQ：**问题的每个 token 先读取全部 latent，再把读到的特征写回 latent 位置。**

记问题的 token 表示为 $H_q\in\mathbb R^{T_q\times h_q}$。省略多头拆分、LayerNorm、dropout 和投影细节，一个 head 的读写为：

$
A=\operatorname{softmax}_{M}
\left(\frac{Q(H_q)K(Z_D)^\top}{\sqrt{d_h}}\right),
\quad A\in\mathbb R^{T_q\times M},
$
$
R_q=A\,V(Z_D),\qquad
C_{\mathrm{write}}=A^\top R_q,
$
$
E=Z_D+\Delta_{RQ},\qquad
\Delta_{RQ}=W_{\mathrm{out}}\operatorname{GELU}(C_{\mathrm{write}})+b_{\mathrm{out}}.
$

可以分成两个动作理解：

1. **读取**：A 的第 t 行表示第 t 个问题 token 从哪些 latent 读取；$R_{q,t}$ 是它读到的文档特征。
2. **写回**：用同一张注意力图反向分配这些特征。第 j 个 latent 收到的特征为 $\sum_t A_{tj}R_{q,t}$，再经可学习投影成为修正量。

“写回”不是把问题文字写进文档，也不是把 $A^\top$ 当作注意力的逆运算。转置后没有再次归一化，因此不同 latent 接收的强度可以不同。最终仍把全部 E 和问题明文送进 decoder。

当时希望它让问题先聚合相关证据，再把这种聚合结果用于调整证据输入。这仍是设计假设。RQ 没有 R 的 latent self-attention，因此两者差别不只有 attention 方向。

**RQ 的输出投影同样零初始化，所以零步 E=Z_D。磁盘缓存保持不变，修改只存在于本次在线前向。**


**实验结果：**

query 先读 latent，再将读出修正写回原始 latent：

- HotpotQA：RQ joint 56.05，匹配 p-control 56.25，增量 −0.20pp，p=.80。
- TriviaQA：RQ 相对同一 p-control +0.10pp，p=.92。

结论：**当前配置的边际近零，尚无继续据此扩大训练的正证据。** R 与 RQ 还同时存在跨文档结构差异，不能把差异纯归因于 Q/K 方向。

来源：[RQ 设计与结果](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/0f03522b6e1db4a75cb911bd5abc80959e7c25b6/docs/QUERY_WRITEBACK_EXPERIMENT.md)；提交 0f03522b。

### 4.4 SeleCom 复现与 latent 几何：排除了简单故事，没有找到修复

复现发现压缩输入行为依赖 prompt/interface，但未支持“memory 全局抢走 instruction attention”解释；真实 K=10 QA 条件下，冲突指令现象也没有表现出稳定的压缩特有边际。

latent 与文本 token 的尺度、更新统计不同，随后尝试缩放、去均值和主方向编辑：

- 同预算域内 LoRA：原尺度 P₁ substring .5990。
- scale=.10：.5845；scale=.05：.5880，均未超过 P₁。
- 固定 decoder 下部分几何编辑损害重建，说明接口兼容性受损；去均值可逆，不能据此证明信息被删除。
- 相对更新小，不等于 memory 不参与计算；逐层余弦连乘不是首尾余弦；只保留部分奇异值作分母不能报告总能量占比。

结论：**理解了表示与消费接口的特性，但“把 latent 改得更像文本”没有形成有效解法。** 9 月 23 日警示文档应优先于更早报告中的强机制断言。

来源：[复现结果](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/FULL_COMPRESSION_INFEASIBILITY_RESULTS.md)；[缩放与几何报告](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/LATENT_CONTEXTUALISATION.md)；[纠正机制过度推断](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md)；output_scale_stage_c.json。

### 4.5 D0/D2 与混合输入诊断：最明确的失败定位

同样本、同 decoder、逐段相同 128-token 截断：

| decoder，K=10，dev2000 | memory substring | raw substring | 配对差距 |
|---|---:|---:|---:|
| 域内 P₁ | 60.0% | 67.8% | 报告配对差 +7.75pp，CI [5.95,9.60] |
| 发布 PISCO | 48.65% | 60.7% | 约 +12pp |

P₁ 取值是摘要四舍五入；不能用 67.8−60.0 的显示差否定报告中更精确的配对差。

K=2 两篇 gold 文档时，P₁ 的 raw−memory 差距仍约 +7.2pp：干扰存在影响，但不是缺口的全部原因。

1193 道“恰有一段 gold 含答案”的 bridge 子集：

- 只将含答案段换 raw：P₁ +5.6pp，恢复约 71% 缺口；发布版约恢复 100%。
- 只将桥接段换 raw：P₁ +0.6pp，CI 含零。
- 答案靠后时差距更大，但这是事后分层相关，尚不是位置效应的因果结论。

D2 的 query→memory 可见路径确实存在，但功能干预在发布版与 P₁ 上不一致、作用小。**拓扑允许、状态改变、答案改善是三个不同层次**，不能从前两个自动推出第三个。

结论：**当前最可靠的定位是“含答案段的压缩表示或读取存在任务能力缺口”，而不是已经证明跨文档组合失败，也不是已经证明答案从 Z 消失。** 混合 raw 是 oracle 诊断，不是已完成的部署解法。

来源：[P0–P3 结果](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/READER_CAUSAL_ORDER_RESULTS.md)；提交 7f477ebf；p1_gold_mixed_p1.json。

### 4.6 Direct-State 与 W：能力诊断没有转化成新方法

这两个名字对应两条不同路线：**Direct-State 保留原证据接口、增加监督；W-CE 改证据接口、只用答案监督。** 不是“先 Direct-State 再把它放进 W”的同一条流水线。

#### 4.6.1 Direct-CE：先建立继续训练的对照

学生沿用原 D0 输入，全部 Z 直接进入 decoder：

$
[Z_D][q][s]\rightarrow a,\qquad
\mathcal L_{\mathrm{CE}}=-\sum_t\log p_\theta(a_t\mid Z_D,q,a_{<t}).
$

这里 $s$ 是**prompt 最后一个有效位置**，例如模板结尾的位置，不是新增的特殊 token；也不是前文余弦 top-B 基线“臂 S”。它位于答案之前，在 causal decoder 中可以读取前面的证据和问题。

首轮从已经域内适配的 P₁ 权重开始，压缩器与缓存冻结，训练 decoder LoRA，目标是 gold 答案与 EOS。Direct-CE 用来回答：**不加任何新监督或接口，仅继续训练会怎样？**

#### 4.6.2 Direct-State：让 memory 学生的中间状态接近 raw 教师

教师是冻结的 P₁ 副本，输入对应原文；学生仍输入原压缩缓存：

$
\text{教师：}[D_{\mathrm{raw}}][q][s_T]
\qquad
\text{学生：}[Z_D][q][s_S].
$

raw 同样逐段截到 128 token。提取状态时**不输入答案**，也不要求教师和学生的 s 有相同绝对序号；取的都是各自最后一个有效 prompt 位置。

在第 8、16、24 个 decoder block 输出处，让学生该位置的完整 4096 维向量与教师方向接近：

$
\mathcal L=
\mathcal L_{\mathrm{CE}}+
0.1\cdot\frac13\sum_{\ell\in\{8,16,24\}}
\left[1-\cos\left(h^\ell_{\mathrm{student},s_S},
h^\ell_{\mathrm{teacher},s_T}\right)\right].
$

教师状态可以离线存成每题 $[3,4096]$ 的目标。学生只训练原 decoder LoRA，部署时不用 raw 教师，也不增加 R/RQ 那样的前置编辑模块。

**设计意图**是：raw 教师已经在最后 prompt 位置处理过证据与问题，希望学生学到更有利于回答的状态。**实际监督**却只是三个完整向量的余弦相似度；它没有指定答案事实在哪个维度，也不是答案序列 logits 的 KL 蒸馏、教师生成答案 SKD 或文档重建。

因此“更像教师”与“更会答题”之间仍有待验证的关系。当前结果没有证明这个目标改善 QA；同起点 Direct-CE 用来分离状态项的作用与继续训练本身的退化。

#### 4.6.3 W-CE：让 decoder 内的工作位置分层读取外部缓存

W 是 workspace（工作区），首轮实现用 **16 个可学习初始向量**。与 R/RQ 的“先算出 E 再送进 decoder”不同，W 已经在 decoder 序列里，外部 Z 在部分层向它提供信息：

$
[q][W][s]\rightarrow a,\qquad Z_D\text{ 位于序列外}.
$

因果 self-attention 中，W 可以读取前面的问题和更早的 W；后续 s/答案位置可以读取 W。前置问题位置不能读取后面的 W，因此这里的“Q/W 交互”不是双向注意力。

在第 8、16、24 个 block 输出后，仅更新 W 位置：

$
W^\ell\leftarrow W^\ell+
g_\ell\operatorname{CA}_\ell
\bigl(\operatorname{LN}(W^\ell),\operatorname{LN}(Z_D)\bigr).
$

CA 中 W 作 attention Q，外部 Z 作 K/V。写入后 W 进入下一层，通过正常 self-attention 影响后面的回答位置。可理解为：**同一组工作位置先读取问题，再在不同深度读取文档，携带读到的结果继续计算。**

缓存 Z 始终冻结，变化的是 W。首轮训练 W 初始向量、三层 CA 和 decoder LoRA，损失只有答案 CE，没有 State 项。不同层的多次读取发生在 prefill；实现并非每生成一个答案 token 都重新做整套 W 写入。

**最关键的起点区别：** CA 输出矩阵零初始化，零步注入量为零，而原 Z 已不在序列内。于是零步 W 模型没有文档内容通路，它不等于 P₁；模型需要学出新的有效接口。R/RQ 则在零步仍完整保留原来的 Z 通路。

W 几乎停留在闭卷水平、错配 Z 后变化很小，提示当前接口未形成有益的证据依赖。不能仅凭“梯度非零”或“训练 CE 下降”就认为它学会了读文档。

#### 4.6.4 汇报时怎样区分这几种尝试？

| 路线 | 用一句话讲 | 实验真正要验证的事 |
|---|---|---|
| 小预算 C | 让外部读出模块把 80 个缓存向量浓缩成 8 个 | 更少证据输入能否保住回答质量 |
| 全量 R | 全部向量保留，让每个向量根据问题获得修正 | 保住原接口后，修正是否有额外价值 |
| 写回 RQ | 问题先读证据，再把读到的特征分配回全部向量 | 这种残差计算方式是否优于强对照 |
| Direct-State | 接口不动，用 raw 教师中间状态监督学生 | 完整状态的相似性是否带来 QA 收益 |
| W-CE | 原证据退出输入序列，由工作位置在中间层读取 | 新接口是否能学会有效利用冻结 Z |

**这些是方案的计算含义，不是已经确认的有效机制。** 下面再看实测结果，就能区分“模块做了什么”“当时为什么设计它”和“它最终是否有效”。


#### 4.6.5 首轮结果与解释

Direct-State 保留 D0/Z 输入，对齐答案前最后一个 prompt 位置的三层完整 hidden states。W 用 16 个工作位置代替原 80 个 Z 输入，在中间层以 CA 读取外部 Z。

dev 前 500 条，首轮结果：

| 条件 | substring | 说明 |
|---|---:|---|
| P₁ / Direct step0 | .640 | 原始工作接口 |
| Direct-CE step3000 | .600 | 继续训练下降 |
| Direct-State step3000 | .598 | best 仍是 step0；相对 CE 无净改善 |
| W step0 | .278 | 新接口零步没有 Z 贡献 |
| W 最终 | .274 | 未形成有效验证能力 |

首轮 Direct-State 3000 步用时约 2.7 小时，峰值显存 14.6GB（仓库记录转述服务器摘要，并非本次实测）。

10 月 2 日最新审查进一步记录：降低 LR 减轻早期退化但未形成持续正收益；State 可以使几何更接近教师而 QA 无益；W 错配 Z 几乎不变，证据特异性利用很弱。这些是文档记录，仓库未提供本次可直接复核的完整服务器日志。

结论：**整向量相似度不是任务信息监督；W 的随机新接口破坏了原有读取起点，只有短答案 CE 未训练出有效读取。** 负结果针对当前配置，不证明所有功能蒸馏或 workspace 都无效。

来源：[首轮恢复报告](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/2cdf029554fe523e29df0f31825f25561a28d8d0/docs/READER_RECOVERY_RUNBOOK.md)；[最新设计审查](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/34828e98a7a587e331b230d33ef6bc1096b2a366/docs/READER_RESET_REVIEW.md)；提交 452c0c55、830a437a、34828e98。

## 5. 为什么“有点挨个试”是有依据的判断

并非每次实验都毫无动机：各轮都有明确候选、对照及一定工程验证。但候选之间没有持续建立一条充分验证的机制链。

典型跳跃有三处：

1. **C≈A → decoder 已做匹配 → 换数据/预算/读出。** C≈A 本身不足以唯一确定冗余原因，历史 A 定义还曾有问题。
2. **memory 更新小、尺度大 → 上下文化不足 → 缩放/改序。** 内部统计不证明实际答案失败由它引起；缩放也同时影响读取接口。
3. **含答案段 raw 可以恢复 → 对齐 raw 状态或用 W 去读 Z。** raw 输入恢复说明问题位置，不证明 Z 中仍有可恢复信息，也不保证最后 prompt 整向量监督会学到该信息。

**缺失的是：失败表现 → 与答案相关的可验证机制 → 针对机制的动作 → 相同预算强对照上的泛化收益。**

另有三项放大因素：

- 相当一部分训练预算用于修正对照、query 表示路径、缓存映射或接口问题；这是工程和实验可信度成本，不能算成模型创新收益。
- 基线身份随阶段变化：发布版、域内 P/P₁、继续训练 p-control。超过旧 checkpoint 并不等于超过匹配强基线；不同 P/P₁ 实验也不能因为名称相似混用。
- 同一个 dev 被反复观察，小子集选 best 和微小涨幅容易形成选择偏差；题目层面的配对 CI 不代替训练 seed 不确定性。

合理评价是：**目前项目不是没有认识积累，而是认识积累尚未约束出一个可验证、稳定有效的解决方案。**

## 6. 目前已经知道和仍然不知道的事

| 层次 | 已知 | 未知 |
|---|---|---|
| 缓存 | 冻结离线 Z 可供 QA 使用，正确/错误证据有行为差异 | 能否覆盖同文档不同问题所需全部关键事实 |
| query readout | 可以携带 query 信息，小预算 Hotpot 上优于余弦规则 | 如何在保留质量和总成本的同时稳定超过强方法 |
| 表示/接口 | memory 与文本统计不同；接口适配作用明显 | 哪些统计差异真正限制答案能力 |
| 失败位置 | 大部分 raw−memory 缺口定位到含答案段 | 该段的答案信息是丢失，还是读取器无法恢复 |
| 训练 | 域内 LoRA 有效；从已适配起点继续训练可能退化 | 什么目标能带来未见题上的、证据相关的净收益 |
| 创新 | 可缓存 + 在线预算解耦的研究目标仍成立 | 新方法增量及端到端多 query 复用收益尚未建立 |

## 7. 最新分支应如何汇报

提交 34828e98 的实际内容是：

- 恢复发布版 PISCO 为主要训练起点，域内 P₁ 留作强参考。
- 保留原 D0 与全量 Z，不添加默认新结构。
- 从 memory / 全 raw / 只含答案段 raw / 只桥接段 raw 四种教师视图自由生成，审计序列监督。
- 导出普通 gold、普通 raw SKD、共同样本上的 raw/答案段教师目标。
- 拒绝 token 完全相同的目标组重复训练。

**这是一项监督目标审计与对照完善，不是已有有效的新方法。** 普通 SKD 已知，答案段局部教师是尚待验证的候选；截至仓库最新提交，无真实模型新效果结果。它也不是已经实现的“改变事实、答案随之改变”的成对反事实训练。

不能将它汇报为“现在已经找到解决办法”或“只差跑一下就会成功”。

## 8. 适合向学长请教的决策问题

1. 是否继续坚持冻结原 PISCO 缓存，仅改读取器；还是允许依据含答案段定位修改压缩器？这里需要判断可恢复性与资源边界。
2. 是否把研究问题收敛为“可复用压缩表示的事实覆盖与读取”，避免同时追小预算、多跳、指令遵循和多查询复用？
3. 什么证据足以让诊断进入方法设计，而不是再以一个结构试跑替代研究判断？

本次汇报不必承诺下一种架构，也不宜承诺整个路线无效。关键是诚实说明：**已经缩小问题范围，尚未完成机制与解决方案闭环。**

## 9. 可直接使用的口头汇报稿（约三分钟）

学长，我现在这个项目进展不太顺利。最初想做的是：文档先离线压成可重复使用的缓存，在线再根据问题做轻量读出，减少每次问答的成本。系统已经跑起来，也出现过一些局部正结果，比如在 HotpotQA 的小预算下，可学习读出超过了余弦选择。但是它仍明显落后于把全部压缩向量直接送进 decoder，所以还不能说解决了问题。

之后我尝试了保留原始 latent 的残差修正、query 写回、尺度调整，以及独立 workspace 和状态蒸馏。残差方案最好的单次成绩超过旧 baseline，但同预算继续训练的 baseline 也涨了；模块本身的边际没有稳定建立，换数据集也没有复现。最近的状态蒸馏没有超过训练起点，workspace 则没有形成有效的证据利用能力。

诊断方面比方法设计更有进展。现在用同权重、同文档和同截断的对照确认，原文比 memory 更好；去掉干扰文档之后缺口还在。更关键的是，只把含答案段恢复成原文，就能补回大部分差距，说明主要问题与含答案段的压缩表示或读取有关。但我还没区分清楚，是答案信息在压缩时丢了，还是信息仍在、当前 decoder 读不出来。

我觉得目前确实有一点挨个试的感觉：几次从内部统计的异常直接推到结构修改，但中间缺少证明“这个机制影响答案，而且这个修改能修复它”的证据。现在对模型和失败位置的理解更清楚了，但还没有形成一个稳定超过强对照、能归因给新方法的解决方案。我这次主要想请教的，就是怎么把这个失败定位进一步转成有依据的方法设计，以及是否需要允许修改压缩器。

## 10. 关键提交索引

| 时间/提交 | 记录内容 | 对汇报的意义 |
|---|---|---|
| 54e56fe6 | 接入真实冻结 PISCO/COCOM，重建 v0.1 | 早期随机原型与真实实验需区分 |
| 328b9cca | A 非 query 无关、残差开关不正确，修正四格 | 部分旧主张须撤回或更正 |
| d5768100 | 修正 S 路径及训练策略结果 | 使用匹配修正版优势 |
| fae85449 | KD 对可学习读出弱正向 | 有效线索但未补齐 P |
| 2428f8a6 | R 首轮零收益 | 恒等起点不保证优化后增益 |
| 8c6cd5c2 | 新数据改变继续训练效果 | 数据收益与模块收益分开 |
| f9c79356 | R 边际不迁移 | 最好成绩不能代表稳定改善 |
| 0f03522b | RQ 两数据集边际近零 | 改读写方向未解决问题 |
| b0009fa1 | 缩放三项门槛失败 | 内部统计目标与任务收益未对应 |
| 75a931b3 | 撤回多项强机制解释 | 以修订文档覆盖早期口头机制故事 |
| 7f477ebf | P0–P3 定位到含答案段 | 最明确的可汇报诊断 |
| 830a437a | 记录 Direct-State 最终退化 | 当前训练未超过起点 |
| 34828e98 | 发布版重置与序列监督审计 | 已准备新审计，效果仍未知 |

证据优先级：原始可核查汇总与匹配比较 > 最新修订报告 > 历史报告的机制推断 > 提交标题。不同配置、指标及样本量不可串成一条提升曲线。
