# QuRO 投影器分支研究总结：场景、动机、方法、实验与当前问题

**整理日期：2026-10-06**  
**主要分支：`feat/pisco-joint-query-projector`**  
**核对快照：`102653d18f48d74603c3cf95e25c6af28a2c081a`**

> 本文以投影器分支为主线，背景参考 [2026-10-02 研究总结](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/main/docs/QURO_RESEARCH_UPDATE_20261002.md)。实验数字来自仓库已经提交的结果记录，结构与训练接口经过当前代码核对；本次整理没有重跑真实 7B 模型，也没有重新访问服务器原始预测和训练日志。文中将“已经观察到的结果”“对结果的解释”和“尚待验证的假设”分别表述。
>
> 本文的分数均以百分制展示；差值是**百分点（pp）**。例如 F1 从 65.21 到 65.52，差值约为 +0.31 F1 点。配对差和置信区间由未四舍五入的逐题分数计算，可能与表内显示值相减略有不同。这里的 test 是项目内部 5405 条测试集，**不是 HotpotQA 官方隐藏测试集**。

## 1. 核心结论与研究状态

目前最准确的结论是：**在冻结发布版 PISCO、复用离线压缩缓存的设置下，残差投影器能明显改变回答方式并改善当前 QA 指标；但显式注入 query 的额外收益仍然很小，尚未建立稳定、可推广的优势。**

最新标准消融已经把投影器的问题条件分支真正移除，分别独立训练 SQ 和 S0。全量内部 test 上，seed 42、43 的平均差为：

| 指标 | SQ − S0，两种子平均 | 配对 bootstrap 95% CI |
|---|---:|---|
| EM | +0.29 | [−0.14, +0.69] |
| F1 | **+0.37** | **[−0.01, +0.74]** |
| substring | +0.34 | [−0.08, +0.77] |

两个 seed 的 test F1 差值方向一致，分别为 +0.31、+0.43；但 dev 上分别为 −0.06、+0.73。全体收益未得到稳健确认。test 的 comparison 子集存在较好的探索性信号，平均 +1.24 F1；bridge 子集仅 +0.14，接近零。

这轮工作并非“query 完全没有进入模型”。只替换投影器的问题、保留 decoder 的正确问题时，SQ 的预测和分数会变化；γ 干预也显示问题能改变条件向量的方向。但**对问题敏感、给错问题受到伤害、正确问题优于独立训练的无问题模型，是三个不同的命题**。目前前两个有证据，第三个只有较弱的小幅正向信号。

已有扩展的结果也相当一致：

- hidden 支持文档监督：分类指标改善，QA 的匹配差仅 +0.07 F1。
- 最终输出 E 上的支持监督：辅助梯度直接进入生成接口，QA 的匹配差仍为 −0.01 F1。
- FiLM 相对等容量加法调制：+0.04 F1，没有建立乘性融合优势。
- 冻结 γ 为零的同顺序 G0 对照：AddG 和 FiLM 仅比 G0 高 +0.08、+0.12 F1，均未确立。
- 标准 S0 消融：投影器的大部分指标提升不需要问题条件分支。

因此，当前成果更接近**一个实现完整、对照逐步完善、瓶颈已经收窄的研究原型**。距离一篇以“query-conditioned projector 带来稳定改善”为主张的完整方法论文，核心缺口仍是方法有效性与作用机制，而不是缺少另一种融合运算。

**English executive summary.** QuRO studies query-conditioned adaptation of reusable, precomputed soft document memories. The current projector is a shared per-document residual MLP preceded by memory-to-question cross-attention, with the published PISCO encoder, decoder and adapters frozen. It preserves all memory slots and learns from answer cross-entropy. The latest independently trained, two-seed SQ versus document-only S0 comparison yields only +0.37 F1 on the internal 5,405-example test set, with a question-level paired 95% confidence interval of [−0.01, +0.74]. Support-document supervision and multiplicative fusion have not established additional QA gains. Query swaps show functional sensitivity but do not establish an equally large beneficial contribution. The architectural idea remains plausible; stable query-specific gains, evidence-use diagnostics, fair baselines and end-to-end cost measurements remain open.

## 2. 我们在解决什么场景的问题

### 2.1 场景：文档可以复用，问题不断变化

典型 RAG 系统会为每个问题检索一组文档，然后将原文和问题一起交给生成模型。相同文档可能在多个问题中重复出现，原文也会被重复处理。软压缩希望把文档预先编码成少量连续向量，让生成模型在较短上下文中使用文档信息。

我们关心的具体场景是：**文档先离线压缩并缓存；新问题到来时，只基于问题与现有缓存做在线适配，再生成答案。** 在线阶段不重新读取完整原文，不重新运行文档压缩器。

令文档集合为 \(D=(D_1,\ldots,D_K)\)，问题为 \(q\)，缓存为：

\[
Z_i=\operatorname{Enc}_{\theta_e}(D_i),\qquad
Z_D=(Z_1,\ldots,Z_K).
\]

原始压缩读取路径为：

\[
\hat a\sim\operatorname{Dec}_{\theta_d}(Z_D,q).
\]

投影器分支把它改为：

\[
E_i=\operatorname{Proj}_{\phi}(Z_i,q),\qquad
\hat a\sim\operatorname{Dec}_{\theta_d}(E_1,\ldots,E_K,q).
\]

在这条路径里，\(Z_i\) 仍是与问题无关的可复用缓存，\(E_i\) 是每个问题在线生成的适配结果。**可缓存的是 Z，而不是通常随问题变化的 E。**

### 2.2 当前实验实例

| 项目 | 当前设置 |
|---|---|
| 基础模型 | 发布版 PISCO-Mistral，底座 Mistral-7B-Instruct-v0.2 |
| 数据 | HotpotQA 的当前项目数据划分 |
| 候选文档数 | 每题 K=10 |
| 编码器可见原文 | 每篇最多 128 个正文 token，另有编码模板 token |
| 每篇缓存 | m=8 个 memory，每个维度 d=4096 |
| 全题 memory | 10×8=80 个 soft tokens |
| 缓存 | `/data02/quro/cache/hotpot-pisco-r16` |
| 投影器输出 | 每篇仍为 8 个 memory，全题仍为 80 个 |
| decoder 输入 | D0：memory 在前，正确问题文本在后 |

这里的“full memory”是**保留全部已有缓存槽**，不代表保留完整文档、更不代表信息无损。128 token 以外的正文没有进入这一版缓存；128 token 内的信息也可能在 128→8 的压缩过程中被损失或难以读取。

### 2.3 原始目标与本轮目标的区别

QuRO 原始方向还包含在线小预算读出：从全部缓存中生成更少的 soft tokens，以降低 decoder 的输入成本。投影器分支先固定输出预算，研究另一个更基础的问题：

> 在不改缓存、不增加 decoder memory 数、不更新原模型的前提下，显式利用问题适配压缩表示，能否提高读取质量？

因此，本轮是**质量与接口适配实验**，不是已经实现的二次压缩或加速方案。每篇 8→8、全题 80→80；任何时延优势都需要另外测量。

## 3. 我的研究动机，以及为什么转向投影器

### 3.1 动机一：让文档缓存与问题相关的处理分离

可以把当前动机归纳为：

> 我希望文档只压缩一次，之后针对不同问题，用较轻的在线模块把同一份缓存调整成更适合当前问题的表示，而不是每个问题都重新编码文档。

如果压缩器在编码原文时就把问题作为输入，那么表示可能更有针对性，但缓存通常也会随问题变化。我们希望保留问题无关的基础表示，把条件化处理放到缓存之后。

这个动机成立与否，最终要看同一份 Z 面对多个不同问题时的效果与成本。目前实验使用缓存，但还没有完成“同文档、多个问题”的专门训练和复用收益评估。

### 3.2 动机二：压缩表示与生成模型之间可能存在可学习的读取障碍

之前的实验已经说明：把相同长度的原文和 memory 交给相同 reader，任务表现存在明显差距。这个差距可能来自两类原因：

1. 压缩时没有保留回答所需的关键事实。
2. 事实以某种形式保留在 Z 中，但当前 decoder 没有把它稳定读取出来。

仅凭 raw–memory 差距，无法把两者区分开。投影器对应第二种可能：如果 Z 中仍有可恢复的任务信息，一个读取接口上的变换可能帮助冻结 decoder 使用它。

“表示尺度、密度和分布不同”提供了适配动机，但它们本身不是已证明的失败根因。过去简单的缩放、去均值等操作没有解决问题，不能据此推导“只要做归一化或增大 query 幅度就会有效”。

### 3.3 动机三：把新增模块的贡献与原模型续训分开

之前残差模块与 decoder LoRA 联合训练时，总分有时上涨，但同预算的直接读取续训对照也会上涨。最佳分数不能直接归因于新模块。

本轮选择冻结原 encoder、decoder、两组发布版 LoRA、词嵌入和 LM head，只训练投影器。这样可以更清楚地回答：

- 新接口有没有学习到有用的变换？
- 改善是否真的需要 query？
- 是否只是域内答案格式适配？

冻结能改善归因清晰度，也限制了可调整空间。不能因为这条约束下效果弱，就断言所有 query-conditioned 软压缩都无效；也不能为了得到更高分，悄悄把 decoder 更新收益算到投影器名下。

### 3.4 动机四：先用简单模型检验假设

这次从两层 MLP 开始，随后改为每篇文档共享的残差 MLP，保留原始 memory 直通路径。设计目标是先检验“缓存之后的问题条件适配”本身，而不是立即引入多层新 reader 或复杂中间层注入。

非线性也有明确作用。若把文档和问题拼接后只通过一个线性层，输出只能分解成文档项加问题项。通过 GELU，再映射到输出空间，条件信息可以改变文档特征的响应方式；但具有这种表达能力并不意味着训练已经学到了所需的证据选择。

## 4. 与旧总结的关系：哪些背景仍然重要

本节只保留会影响投影器判断的历史证据，旧分支的完整结果仍以 2026-10-02 总结为准。

### 4.1 小预算读出有局部正结果，但不等于本轮投影器有效

修正后的旧实验中，HotpotQA、D0、B=8、fixed_adapter 设置下，可学习读出 C1 的 EM 为 43.10，余弦规则 S 为 37.50，差 +5.60。加 KD 后 C1 达到 44.10，仍低于直接读取 P 的 54.50。

这说明可学习读出有局部价值，也说明二次压缩有明显质量代价。它不能作为当前 SQ 的直接增益证据：预算、结构和原模型训练条件不同。

旧实验还提醒我们：80 个 memory 减到 8 个，不等于总 prefill 减少十倍。该表的总 prefill 约从 156 降至 75，只有约 2.08 倍。投影器分支目前甚至没有减少 memory 数。

### 4.2 raw–memory 缺口是动机证据，不能直接当作当前失败诊断

旧分支在同样本、同 decoder、逐段相同 128-token 截断下观察到：

| reader 与输入 | memory substring | raw substring | raw − memory |
|---|---:|---:|---|
| 域内适配 P₁，K=10，dev2000 | 约 60.0 | 约 67.8 | 配对 +7.75，CI [5.95, 9.60] |
| 发布版 PISCO，K=10，dev2000 | 48.65 | 60.7 | 约 +12 |

P₁ 在 K=2、只保留两篇 gold 文档时仍有约 +7.2 的差距。因此，干扰文档不是缺口的全部解释。

在 1193 道“恰有一段 gold 含答案”的 bridge 题上，只把含答案段替换成 raw，P₁ 恢复约 +5.6 substring 点、约 71% 的缺口；只替换桥接段只有约 +0.6，区间跨零。最可靠的定位是**含答案段的压缩表示或其读取存在问题**，尚不能说“答案事实已从 Z 中消失”，也不能直接说“根因就是跨文档推理失败”。

上述 reader 与当前训练后的 SQ/S0 不同。新投影器可能改变读取行为，所以历史定位应作为待复核假设，不能直接迁移为当前模型的机制结论。

### 4.3 已试过的路线不能重新包装成未经探索的修复

旧分支还试过全量残差、query 读缓存再写回、输入顺序变化、尺度调整、状态蒸馏和独立 workspace 等。它们没有建立稳定的新模块收益。

尤其是 attention 方向：当前 SQ 是 memory 读取问题；旧 RQ 已试过问题读取缓存再写回。不能仅凭当前 query 效果小，就把“交换 Q/K/V 方向”认定为必然有效的新办法。

## 5. 创新点：现在可以怎样准确表述

### 5.1 最值得检验的贡献候选

当前最清楚的架构主张是：

> **在可离线复用的软压缩 memory 与冻结生成模型之间，显式引入问题条件的残差投影器，按当前问题在线调整缓存的读取接口。**

它包含几个可以分别检验的要素：

| 要素 | 当前实现 | 尚需证明的价值 |
|---|---|---|
| 缓存之后的显式问题条件化 | q 直接进入投影器，不只进入 decoder | 超过同尺寸文档 MLP 的稳定增益 |
| 文档缓存仍与 q 无关 | 在线不重新编码原文 | 同一 Z 面对多个 q 的实际复用收益 |
| 两端冻结 | 只更新投影器和启用的辅助模块 | 相对合理适配基线的质量与成本优势 |
| 每篇共享投影 | 同一权重处理不同文档，K 不进入权重形状 | 不同 K、数据集、文档池的有效泛化 |
| 残差接口 | E=Z+Δ，零初始化起点保持原读取 | 残差适配对证据使用的可解释改善 |

这些构成研究设计，不自动构成已经验证的论文贡献。

### 5.2 “首个把 query 注入投影器”需要限定和文献核查

我们希望强调 query 的注入位置：它不是在离线压缩阶段进入原文编码器，而是在缓存之后、生成之前进入投影器。但不能仅凭目前查阅的几篇论文就写成“首个 query-conditioned 软压缩方法”。

主要相关工作的区别如下：

| 工作 | 与我们相关的设计 | 当前方案的具体区别 |
|---|---|---|
| [PISCO](https://arxiv.org/html/2501.16075v1) | 文档压成 memory，decoder 根据 memory 与问题回答；训练涉及编码和解码适配 | 当前在既有缓存与冻结 decoder 之间额外加入显式问题条件变换 |
| [SeleCom](https://arxiv.org/html/2602.15856v1) | selector 读取问题与文档，再由 projector 映射到生成模型空间 | 当前在线输入是预先得到的 Z 与 q，问题条件化放在缓存之后的投影阶段 |
| [xRAG](https://arxiv.org/html/2405.13792v1) | 复用检索表示，以两层 MLP 桥接生成模型，仅桥接模块可训练 | 当前桥接的是多槽软 memory，并显式用 q 条件化残差变换 |

因此，**两层 MLP、冻结两端、只训练投影器、复用离线表示，都不能单独宣称首次提出**。更窄的候选新意是这些约束下的“缓存后显式 query-conditioned residual adaptation”。是否已有工作采用同一设计，仍需系统检索确认。

此外，项目自身的旧残差路线已包含问题条件修正。本次投影器是在已有思路上收敛出更明确的冻结约束、共享文档结构与控制实验；不能把项目内部已经尝试过的 query 残差重新描述成完全不同的技术类别。

### 5.3 建议的当前名称与边界

当前方法可以称为：**Query-conditioned residual projector over reusable soft memories**，中文为“面向可复用软记忆的问题条件残差投影器”。

“软压缩技术”描述的是它所在系统；当前新模块本身并没有减少 token 数。论文若以压缩效率为主题，必须补充预算或系统成本证据。若以读取质量为主题，必须建立 query 相对 S0 的有效增量。

## 6. 具体实现：从联合展平版到共享文档版

### 6.1 旧联合展平版 JQ / J0m

第一版 `JointQueryProjector` 把全部文档 memory 和固定上限的问题特征展平：

\[
\begin{aligned}
u &= \operatorname{GELU}\left(W_z\operatorname{vec}(\operatorname{LN}Z_D)
+W_q\operatorname{vec}(\operatorname{LN}Q)+a_z\right),\\
E &= Z_D+\operatorname{reshape}(W_o u+a_o).
\end{aligned}
\]

默认参数：K=10、m=8、d=4096、问题上限 T=64、隐藏宽度 r=128。输入文档向量为 327680 维，问题向量为 262144 维，输出仍是 80×4096。总参数为 **117,768,320，约 117.77M**。

Wz 与 Wq 分开实现，与拼接后线性映射数学等价；独立初始化避免巨大的文档输入宽度压低问题分支的初始方差。默认问题表示为冻结 word embeddings，按位置展平，所以并非无序词袋；也可用冻结 generator 的 contextual states。

这个设计有三个明确限制：

1. 文档位置和问题位置分别对应固定权重，依赖最大 K、T 与文档排列。
2. 参数随最大 K 和 T 增大，改变这些上限通常需要新投影器。
3. 全题残差由一个 r 维隐藏向量决定，Δ 落在一个全局低维仿射输出空间。

第三点约束的是**残差修正**，不是把全部 Z 强行压缩成 128 维；Z 的直通路径仍在。结构限制是事实，但尚不是已测出的 QA 失败原因。

JQ/J0m 代码仍保留为独立对照。当前核对的结果记录没有提供其真实 QA 主表，因此本文不为它填写成绩，也不把它与 SQ 的变化解释为已验证的性能提升。

### 6.2 当前共享文档版 SQ

`SharedDocumentProjector` 对每篇文档使用相同权重。核心路径是：

\[
\begin{aligned}
H_q &= \operatorname{FrozenDecoder}(q)_{\text{last-layer, all valid tokens}},\\
C_i &= \operatorname{MultiHeadAttention}\left(
Q=W_Q\operatorname{LN}(Z_i),
K=W_K\operatorname{LN}(H_q),
V=W_V\operatorname{LN}(H_q)\right),\\
h_i &= W_z\operatorname{vec}(\operatorname{LN}(Z_i))+a_z,\\
b_i &= W_c\operatorname{vec}(\operatorname{LN}(C_i)),\\
u_i &= \operatorname{GELU}(h_i+b_i),\\
\Delta_i &= \operatorname{reshape}(W_o u_i+a_o),\\
E_i &= Z_i+\Delta_i.
\end{aligned}
\]

这里 h 包含文档分支 bias，b 的映射没有 bias。attention 的投影宽度为 256，8 个 head，每个 head 为 32 维。

| 张量 / 模块 | 默认形状或映射 |
|---|---|
| 每篇 Z_i | 8×4096 |
| 问题 H_q | T_q×4096，T_q 为实际有效长度 |
| attention 条件 C_i | 8×256 |
| 文档分支 memory_proj | 32768→512，含 bias |
| 条件分支 context_proj | 2048→512，不含 bias |
| 隐藏状态 u_i | 512 |
| 输出 out_proj | 512→32768，含 bias |
| 每篇 E_i | 8×4096 |

Q 来自文档 memory，K/V 来自问题，所以 C_i 是**文档相关的问题读出**。它并不是将整段原文重新编码，也不是直接从多个文档中抽取答案 token。文档内容仍主要经 h 与 Z 的直通路径进入输出。

每篇有自己的条件系数 u_i，但共享同一输出基底 W_o。K 篇文档可以拥有 K×r 个隐藏系数；并不是每篇训练一套不同的输出矩阵。

### 6.3 为什么当前结构支持可变 K 和问题长度

SQ 只展平单篇的 m 个 memory 和 m 个条件向量；K 与 T_q 不作为全连接权重的固定轴。实际 batch 中的问题按最大实际长度 padding，默认 `max_query_len=256` 是特征安全上限，不是强制补到 256。

同一共享投影器 checkpoint 在接口上可处理 K=2、K=10 等文档数；改变问题上限不改变权重形状。不过，**可运行不等于已经验证质量泛化**。当前结果没有完成 K=2 主评估。

文档之间不加排名位置编码，投影器对文档置换等变；篇内 8 个 memory 槽仍有各自的位置权重。decoder 自身有顺序与因果位置，所以不能据此声称最终答案对文档顺序不变。

### 6.4 归一化、残差和 mask

- memory、query、attention 输出在残差分支中使用无仿射参数的 LayerNorm。
- 原始 Z 的直通路径不归一化，最终 E 不强制对齐固定范数。
- output 权重与 bias 全零初始化，第 0 步有效 E=Z。
- 无效文档和 query padding 经 mask 隔离，文档 padding 即使含 NaN 也不应污染有效输出。
- LayerNorm 使输入 RMS 大致稳定，但 epsilon、近零方差输入等意味着它不是严格固定范数；attention 也不保证整体输出范数恒定。

零输出初始化时，第一步 CE 主要更新输出层；输出层非零后，梯度才进一步进入前面的文档与条件分支。这是预期的训练路径，不是条件分支未接通的证据。

### 6.5 SQX、SL、SW 与参数量

| 代号 | 差异 | 默认可训练参数 | 是否额外运行问题 LM |
|---|---|---:|---|
| SQ | 全 token contextual query + memory→query attention | 37,782,016 | 是 |
| SQX | SQ 的 u_i 上加一层跨文档 self-attention | 38,832,640 | 是 |
| SL | 只取最后有效问题 token，经 value 投影广播到各槽 | 35,684,864 | 是 |
| SW | 冻结 word embeddings + 无参数正弦位置特征 + attention | 37,782,016 | 否 |
| S0m | 固定条件向量，保留 SQ attention 和 MLP | 37,782,016 | 否 |
| S0 | 删除问题条件模块，只保留共享文档残差 MLP | 33,587,712 | 否 |

SQX 在 u 上做文档 attention 后残差相加，再由 W_o 输出；这一层不加文档位置编码。它提供跨文档交互能力，但目前没有建立稳定收益。

SL 的最后 token hidden 不是已证明最优的问题摘要；SW 加位置特征是为了避免无位置 attention 退化为词袋读出。它们的接口相近，不代表表示分布相同。SL 参数稍少，因此不是严格容量匹配对照。

## 7. 冻结、训练、评估与系统成本

### 7.1 哪些参数更新，哪些仍被使用

| 部分 | 使用方式 | 更新状态 |
|---|---|---|
| 文档压缩 encoder | 已离线生成缓存，当前训练不在线运行 | 冻结 |
| encoder_adapter | 已包含在缓存生成模型中 | 冻结 |
| decoder 与 decoder_adapter | 正常生成，adapter 保持激活 | 冻结 |
| 词嵌入、LM head | 正常使用 | 冻结 |
| contextual query 编码 | 冻结 decoder 的 question-only 前向 | no_grad |
| 缓存 Z 与问题 H_q | 作为投影器固定输入 | detach |
| 投影器与启用的辅助模块 | 处理 Z/q，输出 E | 按实验配置更新 |

冻结模型权重不等于对答案前向使用 no_grad。**答案 CE 必须通过冻结 decoder 对 E 求梯度，才能更新投影器。** 当前实现保留这一计算图；只在固定问题特征提取时关闭梯度。

LM 保持 eval，关闭 dropout；原问题文本仍完整进入 decoder。投影器特征截断不应被误写为 decoder 也截掉了问题。

### 7.2 基础训练目标

\[
\mathcal L_{QA}=-\sum_t\log p_{\theta_d}(a_t\mid E_D,q,a_{<t}).
\]

使用 gold 答案 teacher forcing，包含 EOS；prompt 与 padding 的 label 为 −100。本轮基础投影器不使用 KD、状态对齐、预算 dropout、query dropout 或残差惩罚，也不采用数据里的 teacher_output 作为目标。

| 项目 | 基础独立训练 | 辅助监督 / γ 续训 |
|---|---|---|
| 起点 | 发布版 PISCO + 新投影器 | 首轮 SQ 的 last 权重 |
| 训练步数 | 3000 | 1000 |
| 学习率 | 5e-5 | 2e-5 |
| 优化器 | AdamW，weight decay 0.01 | AdamW |
| 调度 | 5% warmup，线性衰减 | 5% warmup，线性衰减 |
| 梯度裁剪 | 1 | 1 |
| batch / 累积 | 2 / 8，有效每步 16 条 | 2 / 8 |
| 训练集 | 30000 题 | 相同题目顺序的可见性注释版本 |
| 样本呈现量 | 约 48000，约 1.6 遍训练集 | 约 16000 |
| 在线验证 | 每 250 步，dev 前 500 | 每 100 步，dev 前 500 |
| checkpoint 选择记录 | EM | F1 |
| 本文主结果 | last | last |

续训是 weights-only warm start，optimizer 与 scheduler 重置，不能与从头训练或者保留优化器状态的 resume 混为一谈。

### 7.3 评估指标与 checkpoint 规则

EM 检查归一化答案完全匹配，F1 为 token 级匹配，substring 检查预测中是否包含归一化 gold 别名。substring 较宽松，不能代替正式 EM/F1；但可帮助观察长回答与短回答的格式差异。

主表使用贪心解码，最多 32 个新 token。当前 dev 评估 2000 条；首轮 test 常先评前 2000 条，后来对部分臂扩到 5405 条。

训练末尾自动评估使用 last。报告 best 必须显式加载 best，并对所有臂使用同一个预先确定的 dev 选择规则。

最新记录中 SQ/S0 的 best 步数不同：SQ42=3000、S042=1250、SQ43=2750、S043=1250。**不同臂选中不同步数，本身不构成不能比较 best 的理由。** 公平规则可以为不同模型选出不同 checkpoint。当前 last 应保留为既有主分析，best 可另做统一规则下的补充；不能看 test 后挑选更有利的 checkpoint。

### 7.4 系统成本尚未测清

SQ 在线要额外运行一次 7B decoder 的问题前向，再运行约 37.78M 参数的投影器。S0、S0m、SW 不需要这次 contextual query 前向。投影器参数少，不等于整个在线过程一定便宜。

当前各臂 decoder 的 memory 数相同，SQ 没有通过减少输入 token 获得加速。首轮训练中 GPU 被其他任务共享，SQ 约 8462 秒、S0m 约 5296 秒，不能据此判断方法开销。

完整成本需要在相同 GPU、batch、精度与输入长度下分别测：query 编码、投影、decoder prefill、答案生成、峰值显存，以及同文档多 query 时摊销的离线成本。

## 8. 首轮共享投影器实验：一到两个点的信号如何变化

来源：`SHARED_QUERY_PROJECTOR_RESULTS.md`，代码起点 `1610106`，结果提交 `45ff79e`。各训练臂 seed=42，从发布版独立训练 3000 步，CE-only，无辅助头，主表为 last。

### 8.1 dev 与 test 前 2000 条

| 臂 | dev EM | dev F1 | dev sub | test EM | test F1 | test sub |
|---|---:|---:|---:|---:|---:|---:|
| base：发布版直接读取，不训练 | 1.75 | 16.03 | 48.50 | 0.90 | 16.41 | 49.90 |
| S0m | 50.15 | 63.74 | 54.85 | 50.95 | 65.01 | 55.30 |
| SQ | 50.05 | 63.63 | 54.85 | 51.60 | 66.15 | 56.30 |
| SQX | 50.60 | 63.91 | 55.25 | 52.00 | 66.71 | 56.35 |
| SL | 50.20 | 63.63 | 54.45 | 51.15 | 65.66 | 55.85 |
| SW | 50.35 | 63.93 | 55.10 | 51.30 | 66.00 | 55.95 |

| 相对 S0m 的 F1 差 | dev2000，95% CI | test 前 2000，95% CI |
|---|---|---|
| SQ − S0m | −0.11 [−0.94, +0.71] | +1.14 [+0.26, +1.97] |
| SQX − S0m | +0.17 [−0.86, +1.19] | +1.69 [+0.62, +2.81] |
| SL − S0m | −0.11 [−1.05, +0.86] | +0.65 [−0.44, +1.70] |
| SW − S0m | +0.19 [−0.64, +1.01] | +0.98 [+0.10, +1.85] |

之前认为“SQ 比固定随机条件对照高 1–2 点”，对应的是这轮 test 前 2000 条，尤其 SQ 的 +1.14 与 SQX 的 +1.69。这个记忆有真实结果依据，但它不是全量结果，也没有在 dev 复现。

### 8.2 扩到全量 test 后，信号明显缩小

| 范围 | S0m F1 | SQ F1 | SQX F1 | SQ − S0m | SQX − S0m |
|---|---:|---:|---:|---|---|
| 全部 5405 | 65.37 | 65.99 | 66.21 | +0.62 [+0.11, +1.14] | +0.85 [+0.18, +1.48] |
| 前 2000，此前已看过 | 65.01 | 66.15 | 66.71 | +1.14 [+0.26, +2.02] | +1.69 [+0.65, +2.73] |
| 后 3405，此轮新增 | 65.57 | 65.89 | 65.93 | +0.32 [−0.35, +0.97] | +0.35 [−0.49, +1.14] |
| bridge，4296 | 63.99 | 64.31 | 65.05 | +0.32 [−0.24, +0.89] | +1.06 [+0.33, +1.80] |
| comparison，1109 | 70.70 | 72.47 | 70.74 | +1.77 [+0.49, +3.14] | +0.04 [−1.40, +1.58] |

全量 EM 差：SQ +0.48 [−0.09, +1.05]，SQX +0.57 [−0.17, +1.33]。

全量 F1 区间虽然不跨零，但这轮已经看过前 2000 条，并依据这些结果选择部分臂做全量评估；后 3405 条只有约 +0.3，dev 仍无明确增益。它应视为弱信号，不能单凭这张表写成稳定收益。

SQX 的 bridge 与 SQ 的 comparison 模式都值得记录，但它们是探索性子组结果，dev 未可靠复现。要归因跨文档 attention，还需要匹配的 SQX−SQ 或与 S0X 等对照，不能把 SQX−S0m 的全部差异只算到新增文档 attention 上。

### 8.3 控制条件说明模型使用了什么

| 条件 | 投影器问题 | decoder 问题 | 文档 |
|---|---|---|---|
| normal | 正确 | 正确 | 正确 |
| mismatch-q | 换成另一题 | 正确 | 正确 |
| mismatch-q-both | 换成另一题 | 同时换 | 正确 |
| mismatch-doc | 正确 | 正确 | 换成另一题的文档 |

| 全量 test F1 | normal | mismatch-q | mismatch-q-both | mismatch-doc |
|---|---:|---:|---:|---:|
| S0m | 65.37 | 65.37 | 1.52 | 27.72 |
| SQ | 65.99 | 64.91 | 1.51 | 27.84 |
| SQX | 66.21 | 65.17 | 1.44 | 28.08 |

SQ 与 SQX 的投影器错配问题掉分分别为 +1.07 [0.59, 1.54]、+1.04 [0.56, 1.54]。S0m 全部预测不变。

这些结果证明有问题相关的功能变化，也证明正确文档对分数重要。但换错文档仍有约 28 F1，提示短答适配后 decoder 可能依赖问题和参数知识回答一部分样本；不能把所有正确答案都算成从缓存中读取到的事实。

### 8.4 base 的巨大涨幅主要包含回答格式适配

base 在首轮 test 前 2000 条只有约 16 F1，却有 49.9 substring；72% 的输出写满 32-token 上限，平均长 28.5 token。S0m 平均输出约 4.1 token。

长解释即使包含答案，也会使 EM/F1 很低。训练后短答显著改善这两个指标。S0m 不看实际问题却获得同样量级的提升，所以不能把 base→SQ 约 49 个 F1 点解释为 query 条件化的效果。

但也不宜断言“全部都是格式，没有任何答案能力变化”：substring 净增 5.4 点；S0m 相对 base 新答对 357 题、丢失 249 题，新答对的 357 题中有 282 题的 base 输出被截断。格式、截断与内容变化在现有审计中交织，尚未完成严格分解。

公平主比较应是 SQ 对 S0/S0m，而不是只对未经当前答案监督适配的 base。base 可用于展示工程起点，但不是 query 创新点的收益基线。

## 9. 支持文档辅助监督：分类学会了，QA 没有改善

### 9.1 为什么加这个监督

单纯答案 CE 可能让投影器优先学会短答与通用接口适配，而不必显式判断问题需要哪些证据。因此尝试加入支持文档标签，推动每篇文档的隐藏表示区分 gold 与 distractor。

hidden 头为：

\[
\ell_i=w_s^\top\operatorname{LN}(u_i)+a_s,
\qquad \mathcal L=\mathcal L_{QA}+\lambda\mathcal L_{support}.
\]

头仅 513 个参数。每题两篇 gold 为正，其余为负；分别对正负取平均再等权汇总，避免负例数量主导。λ=0.1，前 100 步线性升权。

生成仍使用全部 80 个 memory。辅助头不在推理时删文档、门控 E 或修改预算。

### 9.2 可见性处理

如果 gold 文档没有任何一条支持句完整进入编码器实际看到的前 128 token，该正例被屏蔽，不参与辅助损失，也不被改成负例。

| 集合 | 题目数 | 可见正例 / 正例总数 | 两篇 gold 都可见 | 无可见正例的题 |
|---|---:|---|---|---:|
| train | 30000 | 55429 / 60000，92.4% | 25613，85.4% | 184 |
| dev | 2000 | 3658 / 4000，91.5% | 1675，83.8% | 17 |
| test | 5405 | 9898 / 10810，91.6% | 4543，84.1% | 50 |

train 中 29730 题同时有可用正例与负例，实际参与辅助损失。可见性注释避免给模型不可实现的标签约束，但不能恢复从未进入缓存的证据。

### 9.3 匹配续训设计与结果

两个臂都从首轮 SQ last 加载权重、重置优化器与调度，续训 1000 步，lr=2e-5，seed=42，并创建相同的随机头。区别只有辅助损失权重：Head 为 0，Doc 为 0.1。

| dev2000，last | EM | F1 | substring |
|---|---:|---:|---:|
| SQ+Head，CE-only 续训 | 50.45 | 64.22 | 55.10 |
| SQ+Doc，CE+支持 BCE | 50.70 | 64.30 | 55.35 |
| Doc − Head | +0.25 [−0.05, +0.55] | **+0.07 [−0.16, +0.29]** | +0.25 [−0.05, +0.55] |

bridge F1 差 +0.09 [−0.14, +0.32]，comparison 约 0，双 gold 可见子集 +0.13 [−0.10, +0.39]。没有建立 QA 收益。

| 文档识别指标 | SQ+Head | SQ+Doc | SQ+Doc，错配问题 |
|---|---:|---:|---:|
| Recall@2 | 20.0% | 55.2% | 55.4% |
| both@2 | 2.2% | 24.2% | 24.3% |

换问题后 logit 相关系数为 0.925，1719/2000 题的 top-2 集合不变，即 86%。投影器错配问题的 QA F1 掉分，Head 为 1.11 [0.25, 1.96]，Doc 为 1.16 [0.36, 1.97]，两者之差只有 +0.05 [−0.31, +0.40]。

**支持分类确实学到了可预测信号，但没有表现出更强的问题相关证据识别，也没有转化成生成收益。** “它学会了文档先验”是一种合理解释；文档风格、实体类型、干扰池构造等可能提供捷径，但尚未通过成对数据或文档特征控制证明具体原因。

## 10. 最终输出 E 上的支持监督：排除一个接口解释

### 10.1 为什么改变监督位置

hidden 头直接作用于 u，未直接以最终 E 为分类输入。曾有一个解释：分类改善可能停留在辅助隐藏空间，没有让 decoder 接口变好。

第二轮把头改成：

\[
\ell_i=w_s^\top\operatorname{LN}\left(\frac1m\sum_j E_{ij}\right)+a_s.
\]

头有 4097 个参数，辅助梯度明确通过 E 回传到 W_o。两臂都从**未受 Doc 监督的首轮 SQ last** 开始，仍为 1000 步、lr=2e-5、seed=42。不是在上一轮 Doc checkpoint 上继续叠加监督。

### 10.2 结果

| dev2000，last | EM | F1 | substring |
|---|---:|---:|---:|
| SQ+HeadE，λ=0 | 49.65 | 63.55 | 54.40 |
| SQ+DocE，λ=0.1 | 49.65 | 63.54 | 54.45 |
| DocE − HeadE | 0.00 | **−0.01 [−0.35, +0.31]** | +0.05 |

bridge 差 −0.09 [−0.44, +0.28]，comparison +0.32 [−0.42, +1.32]，双 gold 可见子集 −0.02 [−0.37, +0.32]。没有建立 QA 收益或显著损害。

| DocE 排序指标 | 正确问题 | 错配问题 |
|---|---:|---:|
| Recall@2 / both@2 | 53.4% / 22.3% | 53.2% / 22.0% |
| Recall@4 / both@4 | 73.8% / 52.5% | 73.8% / 52.3% |
| Recall@6 / both@6 | 85.7% / 72.6% | 85.7% / 72.4% |

logit 相关系数升到 0.992，95.6% 的 top-2 集合不变。错配问题的 QA F1 掉分，HeadE 为 0.73 [−0.05, 1.58]，DocE 为 0.64 [−0.17, 1.52]，差 −0.09 [−0.54, +0.34]。

因此，支持监督的阴性结果不能再主要归因于“辅助梯度没有接到生成出口”。它确实接到了出口，当前标签、目标与表示组合仍未带来生成改善。

输出头比 hidden 头更少受问题影响，与“分类可以直接读取原始 Z 的文档特征”相容；但没有直接测定或隔离 mean(Z) 与 mean(Δ) 的分类贡献，不能把这个解释写成已证实的机制。

### 10.3 离线 top-k 的含义与局限

对上一轮 hidden 头的已有预测做离线统计：

| 头 | Recall@2 / @4 / @6 | both@2 / @4 / @6 |
|---|---|---|
| 训练后的 SQ+Doc | 55.2 / 75.6 / 87.4% | 24.2 / 55.1 / 76.0% |
| 随机 SQ+Head | 20.0 / 35.5 / 51.2% | 2.2 / 9.4 / 21.3% |

Recall@6=87.4% 指 gold 文档的平均覆盖比例，**不是 87.4% 的题同时保留两篇 gold**；同时保留两篇的比例是 both@6=76.0%。

K=6 对应约 48 个 memory，但这里只统计了排序后的证据覆盖，没有真正把 80 个 memory 降到 48 后运行 QA，也没有测时延。不能据此宣称已经实现高质量文档选择或压缩加速。

## 11. 加法与 FiLM：改变融合形式仍未建立收益

### 11.1 设计与等容量对照

在相同 h、b 路径上增加问题条件 γ：

\[
\begin{aligned}
\gamma_i&=\tanh\left(W_\gamma\operatorname{LN}(b_i)+a_\gamma\right),\\
u_i^{AddG}&=\operatorname{GELU}(h_i+b_i+\gamma_i),\\
u_i^{FiLM}&=\operatorname{GELU}(h_i+b_i+\gamma_i\odot h_i).
\end{aligned}
\]

Wγ 与 bias 全零初始化，r=512 时新增 262,656 参数。γ 不只是一个全局问题向量，因为 b 已依赖文档 memory 与问题的交互；更准确地说，它是按文档自适应的问题条件特征增益。

FiLM 改变隐藏特征的增益 1+γ，原始 Z 的直通系数仍是 1。AddG 保留完全相同的 γ 模块，作为容量匹配对照。

两个臂从首轮 SQ last 只加载权重，续训 1000 步、lr=2e-5、seed=42。实际这轮 **λ=0，只有答案 CE**；保留相同的随机 output 支持头，但它没有接受辅助训练。方案脚本的历史默认 λ=0.1，不能把脚本默认值误写成这轮实际设置。

### 11.2 正常 QA 与问题错配

| dev2000，last | EM | F1 | substring |
|---|---:|---:|---:|
| AddG | 50.75 | 64.43 | 55.65 |
| FiLM | 50.80 | 64.47 | 55.70 |
| FiLM − AddG | +0.05 [−0.25, +0.35] | **+0.04 [−0.18, +0.26]** | +0.05 [−0.25, +0.35] |

bridge 差 +0.05 [−0.16, +0.27]，comparison 为 0.00 [−0.79, +0.79]。正常 QA 没有达到该轮预定确认条件，因此没有扩到 test，也没有换 checkpoint 追求正结果。

错配投影器问题造成 F1 掉分：AddG 1.16 [0.31, 2.07]，FiLM 1.30 [0.45, 2.24]。两者差 +0.14 [−0.26, +0.54]，没有建立 FiLM 更强的问题依赖。

### 11.3 调制确实更新，不能再只用“信号太小”解释

| 量 | AddG | FiLM |
|---|---:|---:|
| RMS(h) | 4.22 | 4.16 |
| RMS(b) | 1.35 | 1.36 |
| RMS(γ) | 0.065 | 0.099 |
| 实际注入 RMS | 0.065 | 0.365 |
| 实际注入 / RMS(b) | 0.048 | 0.269 |

FiLM 的实际注入量约为 AddG 的 5.6 倍，QA 仍相同。这个结果说明本轮 γ 并非一直保持零或完全没有学习；“增加乘性修正量”没有自动产生收益。

它并不排除所有尺度因素，也不能推广成“任何更强的 query 交互都不会有效”。可成立的结论仅是：**这一版按文档 γ 调制，在这一训练起点和目标下没有建立新增 QA 优势。**

## 12. 冻结 γ 对照与干预：修正一次看起来像收益的现象

### 12.1 为什么需要 G0

AddG 与 FiLM 相对历史 HeadE 都高约 0.9 F1，看起来可能是共同新增 γ 路径的收益。但新建 `gamma_proj` 的默认初始化会消费随机数，即使随后把权重置零，也会改变后续训练样本顺序。

起点输出相同并不足够。HeadE 第一条训练 loss 为 0.1239，AddG/FiLM 为 0.1904，说明这不是完整匹配的续训对照。

G0 保留 γ 模块与相同初始化随机数消耗，但每步在梯度裁剪前将 γ 梯度设为 None，AdamW 不更新这些参数，γ 权重保持零。它与 AddG/FiLM 的首步 loss 都为 0.1904。

### 12.2 G0 结果

| dev2000，last | EM | F1 | substring |
|---|---:|---:|---:|
| G0 | 50.65 | 64.35 | 55.55 |
| AddG | 50.75 | 64.43 | 55.65 |
| FiLM | 50.80 | 64.47 | 55.70 |

| 主比较 | F1 差，95% CI |
|---|---|
| AddG − G0 | +0.08 [−0.12, +0.29] |
| FiLM − G0 | +0.12 [−0.18, +0.43] |
| HeadE − G0 | −0.81 [−1.65, +0.00] |

同顺序、γ 不训练的 G0 也得到约 64.35 F1。**这使“共同约 0.9 点来自 γ”的解释失去支持；结果与训练顺序变化造成差异相容。** 不能继续用 AddG/FiLM 对历史 HeadE 的差值作为结构收益。

但这一次约 0.8 点的差异不是所有单 seed 实验的通用“噪声下限”。它说明训练过程变化可以产生相近量级的分差；要估计训练方差，仍需多次独立重复，而不是把一次观察变成固定阈值。

### 12.3 γ 置零、γ 换问题与完整问题错配

γ-only 干预固定正确的 h、b、文档与 decoder 问题，只把 γ 置零，或让 γ 来自另一题的问题。完整 mismatch-q 则同时改变 b 和 γ。

| 臂 | 干预 | F1 掉分：正常 − 干预 | γ 相对变化 | γ 余弦 | E 变化 / Δ |
|---|---|---|---:|---:|---:|
| AddG | γ 置零 | +0.20 [0.01, 0.43] | 1.00 | — | 0.008 |
| AddG | γ 换问题 | +0.35 [0.05, 0.68] | 1.28 | 0.15 | 0.009 |
| AddG | 完整 mismatch-q | +1.16 [0.28, 1.99] | 1.28 | 0.15 | 0.211 |
| FiLM | γ 置零 | −0.01 [−0.50, +0.48] | 1.00 | — | 0.059 |
| FiLM | γ 换问题 | +0.48 [0.06, 0.90] | 1.13 | 0.32 | 0.045 |
| FiLM | 完整 mismatch-q | +1.30 [0.41, 2.16] | 1.13 | 0.32 | 0.209 |

相对变化为 \(\|\gamma'-\gamma\|/\|\gamma\|\)，E 变化 / Δ 为 \(\|E'-E\|/\|\Delta\|\)。同一干预点估计的 CI 在不同脚本/记录中可能因 bootstrap 实现与抽样而略有变化，本表采用 γ 干预专表。

这些结果有三个含义：

1. γ 的方向明显随问题变化。换问题前后 RMS 相近，不能解释成“γ 与问题无关”；余弦只有 0.15–0.32。
2. γ-only 对生成输入的改变较小，AddG 不到 Δ 的 1%，FiLM 约 5–6%；完整问题错配约为 Δ 的 21%。
3. FiLM γ 置零几乎不掉分，换成错误 γ 却掉分，再次说明有害条件扰动不等于正确条件提供了同等收益。AddG 置零的 +0.20 是单 checkpoint 的微弱功能证据，也不替代独立训练的模块对照。

不同干预的掉分不能直接相减来分配 b 与 γ 的贡献，因为整体是非线性模型，且这些是单 seed、多项探索性比较。

## 13. 标准无 query 消融 S0：实现方式与最新主结果

### 13.1 为什么不能只把训练后的问题置零

我们希望回答的是：**正常训练的无 query 投影器，比正常训练的 query 投影器差多少？**

如果只在一个已经训练好的 SQ 中将 query 置零，得到的是这个 checkpoint 对已学条件路径的依赖或受扰动程度；它不是另一个模型在没有该模块时能达到的水平。

S0m 是固定输入、保留条件模块的容量对照。它的输入不是每步重采样的随机噪声，而是初始化时生成、之后固定的 4 个向量。它可能有不同的归纳偏置，不能先验认定一定带来负效果；它也不等同于直接删除分支。

### 13.2 当前选择：真正删除分支，使用公共开关

实现通过：

```text
--projector_query_mode none
```

SQ 与 S0 共享的文档 MLP 维度不变：

\[
\begin{array}{ll}
\mathrm{SQ}:&u_i=\operatorname{GELU}(h_i+b_i),\\
\mathrm{S0}:&u_i=\operatorname{GELU}(h_i),\\
\text{共同输出}:&E_i=Z_i+\operatorname{reshape}(W_o u_i+a_o).
\end{array}
\]

两个输入分支已经各自映射到同一个 512 维空间，移除 b 不需要改变 h 或输出的维度。S0 仍是 32768→512→32768，每篇输出 8×4096。

S0 不保留 Q/K/V、`context_proj`、query/context LayerNorm 或固定条件向量，也不会运行问题特征编码。前向代码用 `zeros_like(h_doc)` 统一加法接口，但**对应条件模块已经从模型中移除**，没有这些参数、优化器状态和 checkpoint 权重。

| 参数部分 | 数量 |
|---|---:|
| 文档输入与输出 MLP，S0 保留 | 33,587,712 |
| SQ 的 Q/K/V 投影 | 3,145,728 |
| SQ 的 context_proj | 1,048,576 |
| SQ 总计 | 37,782,016 |
| 删除条件分支减少 | 4,194,304 |

参数减少是标准模块消融的自然结果，应如实报告。不能因为某种实现保留了未更新模块，就声称有效训练容量相同。

为了使公共权重与训练随机数状态对齐，S0 构造期间先按 SQ 的顺序初始化条件模块，再将它们移除。临时构造只为消耗相同随机数，不参与训练或推理。相同 seed、相同设置下，共享参数初始化与后续 RNG 状态一致。

`needs_query=False` 确保额外问题编码不运行。decoder 仍接收正确问题，所以 S0 不是无问题问答或闭卷消融。

### 13.3 加载保护与工程验证

- SQ、S0、S0m 的 query mode 与 checkpoint layout 必须一致，不能静默混载；weights-only warm start 也受保护。
- `result.json` 记录 arm、mode、是否需要问题编码以及参数统计。
- S0 条件梯度记为 null，不把不存在的参数伪装成正在学习。
- `--projector_fusion none` 只关闭 γ，不关闭 query；标准 S0 要显式设置 query mode none。
- S0 不允许 AddG/FiLM，避免 γ bias 在没有 query 时形成另一条静态增益路径。
- CPU 契约测试覆盖共同初始化、零步输出、冻结与梯度路径、mask、checkpoint、optimizer/resume 等；这些测试只能确认实现，不代表真实 QA 改善。

### 13.4 最新两 seed 实验的协议

seed 42、43 各一对 SQ/S0，均从发布版 PISCO **独立初始化并训练** 3000 步；CE-only，无跨文档 attention、支持头或 γ；lr=5e-5，batch2×累积8，相同训练数据、验证规则与 memory 预算。

同 seed 下零步 dev500 输出相同，EM=2.6、F1=16.7；首步 loss 相同，seed42 为 2.7689，seed43 为 3.0077。共同初始化与随机数控制得到核验。主比较使用 last。

### 13.5 全量 test 主结果

| 臂 | seed | EM | F1 | substring |
|---|---:|---:|---:|---:|
| SQ | 42 | 51.19 | 65.52 | 55.97 |
| S0 | 42 | 50.90 | 65.21 | 55.54 |
| SQ | 43 | 51.38 | 65.62 | 56.06 |
| S0 | 43 | 51.10 | 65.20 | 55.80 |

| SQ − S0 | EM，95% CI | F1，95% CI | substring，95% CI |
|---|---|---|---|
| seed42 | +0.30 [−0.24, +0.83] | +0.31 [−0.17, +0.77] | +0.43 [−0.11, +0.94] |
| seed43 | +0.28 [−0.33, +0.87] | +0.43 [−0.10, +0.96] | +0.26 [−0.33, +0.87] |
| 两 seed 平均 | +0.29 [−0.14, +0.69] | **+0.37 [−0.01, +0.74]** | +0.34 [−0.08, +0.77] |

平均区间的做法是先对每道题取两 seed 的差值平均，再 bootstrap 题目，2000 次，抽样 seed=0。这个区间反映题目抽样不确定性，**不包含完整的训练 seed 方差**。两个训练 seed 不足以估计这种方差。

| test 子集 F1 差 | 两 seed 平均 | seed42 | seed43 |
|---|---|---|---|
| bridge，4296 | +0.14 [−0.25, +0.53] | −0.05 [−0.55, +0.46] | +0.33 [−0.24, +0.89] |
| comparison，1109 | +1.24 [+0.23, +2.30] | +1.66 [+0.36, +3.04] | +0.81 [−0.52, +2.20] |

comparison 是当前较强的局部信号，但它来自反复查看的子集分析，且 dev 没有稳定复现。它支持下一步有针对性的验证，不足以立即改变主结论。

### 13.6 dev 主结果

| 臂 | seed | EM | F1 | substring |
|---|---:|---:|---:|---:|
| SQ | 42 | 49.85 | 63.63 | 54.95 |
| S0 | 42 | 49.95 | 63.69 | 54.50 |
| SQ | 43 | 50.30 | 63.88 | 54.90 |
| S0 | 43 | 49.65 | 63.15 | 54.35 |

| dev SQ − S0 | EM，95% CI | F1，95% CI | substring，95% CI |
|---|---|---|---|
| seed42 | −0.10 [−1.15, +0.85] | −0.06 [−0.90, +0.73] | +0.45 [−0.50, +1.40] |
| seed43 | +0.65 [−0.30, +1.60] | +0.73 [−0.08, +1.53] | +0.55 [−0.35, +1.45] |
| 两 seed 平均 | +0.28 [−0.47, +0.97] | +0.33 [−0.29, +0.96] | +0.50 [−0.22, +1.20] |

dev bridge 平均 +0.25 [−0.36, +0.85]；comparison 平均 +0.68 [−1.26, +2.66]，seed42 为 −0.23，seed43 为 +1.58。方向与幅度仍不稳定。

### 13.7 错配问题、随机条件和重跑差异

| SQ 正常 − 投影器错配问题，F1 | seed42 | seed43 |
|---|---|---|
| dev2000 | +0.18 [−0.60, +0.98] | +1.51 [0.62, 2.41] |
| test5405 | +0.45 [−0.02, +0.92] | +1.16 [0.62, 1.70] |

S0 错配投影器问题时，dev2000 和 test5405 的预测全部不变。SQ 在 test 中分别有 4829/5405、4756/5405 个预测不变，即约 10.7%、12.0% 的输出发生变化；并非每次变化都使答案变差。

S0 seed42 相对旧 S0m 的 test F1 差为 −0.15 [−0.62, +0.33]。这没有支持“固定随机条件造成明显负效果”的判断，但也不是正式的统计等效证明。标准 S0 的加入让主论证无需依赖随机条件对照。

旧 SQ seed42 为 65.99，新 SQ seed42 为 65.52，同配置、同 seed 仍相差 0.47 F1 [0.07, 0.90]，528 条预测不同。GPU 非确定性与中间重构改变计算顺序是候选解释，目前没有定位证明具体来源。

因此，最新主表使用同一轮代码下新训练的 SQ/S0；不能把旧 SQ=65.99 与新 S0=65.21 拼成更大的“标准消融收益”。

## 14. 综合证据：现在能说什么，不能说什么

### 14.1 按问题汇总

| 研究问题 | 当前证据 | 当前判断 |
|---|---|---|
| 新模块是否能改变冻结 reader 的行为？ | base 与训练后投影器的长度、EM/F1 明显变化 | 是，接口适配已发生 |
| 大涨幅是否需要真实 query？ | S0/S0m 获得同样量级的改善 | 不需要；不能将大涨幅归因 query |
| 投影器是否对 query 敏感？ | SQ mismatch-q 改变预测；γ 方向随问题变化 | 是，但程度随 seed 与路径变化 |
| 真实 query 是否稳定优于无 query 投影器？ | SQ−S0 两 seed 平均 +0.37 F1，区间跨零 | 尚未确立 |
| 支持监督是否促进问题相关证据使用？ | 分类改善，错配 q 后排序基本保持，QA 差接近零 | 当前监督未建立此作用 |
| 乘性融合是否优于加法？ | FiLM−AddG +0.04，FiLM−G0 +0.12 | 本轮没有建立优势 |
| 缓存是否丢失了所有关键答案事实？ | raw 更好；answer 段替换有效 | 不能据此判断事实已不可恢复 |
| 当前能否声称加速或多 query 复用收益？ | 保持全部 80 memory，缺少分项时延和复用测试 | 不能 |

### 14.2 必须分开的三个差值

令 \(S(\cdot)\) 表示 QA 分数：

\[
\begin{aligned}
\text{接口适配差}&=S(\mathrm{S0})-S(\mathrm{base}),\\
\text{query 的训练增量}&=S(\mathrm{SQ})-S(\mathrm{S0}),\\
\text{错配条件的伤害}&=S(\mathrm{SQ},q)-S(\mathrm{SQ},q').
\end{aligned}
\]

前者包含额外训练、格式与表示变换的共同作用；第二项才是标准独立训练下的 query 增量；第三项是在固定模型上的条件干预。三者没有一般性的相等关系。

正确 q 能改变输出，并不保证 q 改变输出的方式对主任务有益。γ 置零与 γ 换问题的差异已经给出了具体例子。不能用干预掉分代替模块消融，也不能因消融参数减少就放弃正常消融。

### 14.3 需要修正的历史表述

| 容易沿用的说法 | 本文采用的更准确表述 |
|---|---|
| “我们已经高了 1–2 点” | 首轮 test 前 2000 条有该信号；全量与标准消融更小 |
| “换 query 掉 1 点，所以 query 贡献 1 点” | 错配条件伤害约该量级；独立训练增量约 +0.37 |
| “随机条件会害模型，不能作对照” | 它是固定条件容量对照；是否有害要测，主消融另用 S0 |
| “base→SQ 的 49 点全是方法创新” | 大部分共同收益无须 query，包含显著短答/截断适配 |
| “分类头没接到最终出口才没效果” | DocE 已直接监督 E，QA 仍不变 |
| “γ RMS 不变说明没有读问题” | 幅度近似不变，方向显著变化 |
| “FiLM 没学起来，所以实验没测到” | γ 有更新、实际注入非零，当前 QA 仍无结构优势 |
| “0.8 F1 是所有实验的噪声下限” | 一次顺序变化观察不能给出通用噪声阈值 |
| “best 步数不同，不能公平比较” | 同一 dev 规则可以公平选择不同步数；禁止混比或 test 择优 |
| “bridge 没涨，所以一定缺跨文档 attention” | bridge 阴性只定位题型；历史 answer 段诊断与 SQX 结果不支持直接跳到该根因 |

## 15. 当前遇到的问题，以及各解释的证据强度

### 15.1 主问题：query 没有提供稳定的额外任务收益

这是目前最直接的问题，不依赖机制猜测。标准消融已经实现并运行两 seed，分差仍接近训练与样本波动的量级。后续不能只展示 query attention 的可视化、梯度非零或错配掉分来补足主结果。

### 15.2 可能的任务冗余：decoder 已看到问题和全部 memory

在 D0 中，decoder 已接收正确问题与完整 80 个缓存槽，自己也能根据问题读取文档。投影器可能学到的是通用接口调整，而 query 在这一层的额外作用较小。

这是合理假设，但没有通过相同信息条件、预算变化或专门的证据读取任务得到因果验证。不能由当前小增益直接断言“提前注入 query 永远冗余”。

### 15.3 训练目标可能允许绕开问题相关证据处理

短答案 CE 能奖励格式正确、输出短、记忆中的常识答案以及文档接口适配。它没有直接要求同一文档在不同问题下形成不同的证据读出。

现有支持标签也可能由文档本身预测一部分。hidden 与 output 分类头在错配问题后仍保持排序，是这种解释的行为证据；但具体数据捷径尚未查明。

所以问题可能不只是“query 融合得不够强”，也可能是**当前监督对利用 query 的必要性不足**。两轮支持监督和 γ 对照削弱了继续单纯修改融合公式的理由。

### 15.4 缓存可见性与可恢复性仍未解决

每篇 128-token 截断至少产生两种不同问题：正文窗口之外的事实不在输入中，窗口之内的事实可能在压缩或读取时失败。

投影器不能访问未进入缓存的原文；也没有保证能恢复压缩时已经丢失的事实。另一方面，raw 替换有效只说明原文接口提供了更强的任务能力，不能证明当前 Z 完全没有相关信息。

必须把窗口可见性、压缩事实覆盖、生成读取能力分开，否则实验阴性容易被归因到错误层次。

### 15.5 bridge 没有收益，但尚未定位到跨文档组合

SQ 单篇投影只依赖本篇与 q，跨文档组合留给 decoder；这是能力边界。SQX 提供了跨文档交互，但首轮总体与新增 test 子集没有建立稳定优势。

旧 mixed-raw 诊断曾把大部分缺口定位到含答案段，而不是只替换桥接段。因此，当前 bridge 无收益可能涉及局部事实读取、桥接实体确定、跨文档组合或多种因素；应先测当前 SQ/S0，不能仅因题型叫“多跳”就决定需要更深 attention。

### 15.6 表达能力与优化过程还没有被诊断清楚

共享 MLP 隐藏宽度为 512，残差落在共享输出基底的空间中；这可能限制某些修正，也可能已经足够。由于 Z 直通保留，不能把它理解为把全文信息压到 512 维。

当前参数量与 QA 平台期不足以诊断欠拟合，更大的投影器也没有自动优先级。同 seed 重跑差异、样本顺序混杂和小 dev500 波动说明，先解决归因与可重复性比单纯扩容量更有价值。

### 15.7 当前证据范围较窄

结果主要限于一个冻结发布版 reader、HotpotQA、K=10、每篇 8 memory、固定原文窗口与有限训练步数。最新主消融仅两个 seed；辅助扩展多数为一个 seed 的 1000 步续训。

没有完成新的跨数据集验证、query 多次复用评估、不同压缩率/预算的质量成本曲线，也没有形成针对当前 SQ/S0 的 raw–memory 诊断闭环。过去其他分支的 TriviaQA 或 raw 结果不能充当这些空缺的替代品。

## 16. 还需要补什么：按决策价值排列

### 16.1 先补清楚现有结果，而不是立即新增结构

1. **统一 checkpoint 补充比较。** 保留 last 主表，按相同 dev EM 规则评估各臂 best，明确写成补充分析。不同 best 步数允许比较；结果不能反过来改写原主判据。
2. **记录可重复性所需信息。** 固定代码 SHA、数据 ID 顺序、模型/缓存版本、训练样本顺序、精度与 GPU 环境；将 sampler RNG 与模块初始化 RNG 分离，减少新增模块消费随机数带来的混杂。
3. **如要确认 +0.4 的量级，补一对 seed44。** 它有助于稳定性估计，但只是更精确地描述当前小效应，不能自动解决方法收益不足。

现有 dev/test 已反复用于方法决策，新增确认实验应尽量保留未参与选择的数据或新任务。不能把反复查看同一 test 后的最好结果称为独立确认。

### 16.2 对当前 SQ/S0 复核失败位置

优先级较高的是在相同题目、相同冻结 reader、相同原文截断下，对当前模型复核以下诊断：全部 memory、全部 raw、仅含答案段 raw、仅桥接段 raw。

这是 oracle 诊断，目的是判断当前缺口主要来自局部答案读取还是文档组合，并观察 SQ 相对 S0 是否改变缺口。替换时需要明确绕过/保留哪条投影路径，避免原文与 memory 接口混杂；不应将 gold 文档标签用于部署时选择。

当前投影器分支并没有直接提供旧分支完整的 mixed-raw CLI，复用旧脚本需要适配当前模型接口。这是建议补的实验，不是本报告已经运行的新结果。

### 16.3 让“同一缓存面对不同问题”成为真正的研究测试

构造或选择同一文档/文档池对应多个问题、且问题需要不同证据的成对数据。关键要求是固定 Z 与候选池，只改变 q 和目标证据，避免标签仅依靠文档风格就能预测。

评估 SQ/S0 的正常 QA、对应关系错配和证据变化。若需要设计新辅助目标，可考虑要求 E 支持恢复问题相关的可见证据文本，并在辅助路径限制 decoder 直接利用 q；这样可能增加 query 经过投影器传递信息的必要性。

这只是候选实验设计，尚未实现、尚未证明有效，也不能声称具有新颖性。正常主 QA 的 D0 路径应保持可比，辅助任务成功仍须转化为主 QA 收益。数据拆分要避免同文档多问题跨集合造成不当泄漏。

### 16.4 完整方法论文还缺的证据

| 缺口 | 最小需要回答的问题 | 当前状态 |
|---|---|---|
| 核心有效性 | query 相对合理无 query 投影器是否稳定有效？ | 有标准消融，增量未确立 |
| 公平基线 | 是否超过同训练预算的格式/接口适配？ | S0 已有，其他适配与合理 base 审计仍需补齐 |
| 作用机制 | 改善来自证据读取，而不是错配伤害或输出格式？ | 功能干预已有，机制闭环未完成 |
| 泛化 | 是否在新的数据、问题分布、K 或压缩率下成立？ | 未完成当前主方法验证 |
| 系统价值 | 质量改善是否值得额外 query 编码成本？ | 未测端到端成本 |
| 缓存复用 | 同一文档多 query 的质量与摊销成本如何？ | 目标成立，专门证据缺失 |
| 新颖性 | 与已有 query 压缩和表示桥接工作的边界是什么？ | 有候选差异，尚无完整首创核查 |
| 可复现性 | 代码、数据、环境、选择规则能否支撑复现？ | 实现较完整，训练不确定性仍待整理 |

CCF-C 是会议分类，不是固定分数门槛。不能用“补齐消融就能发”评价当前状态。若按 query 增益型方法论文推进，最关键的缺口仍是有效增量；若按事实读取诊断与缓存适配的研究问题推进，也需要更清晰的可复现发现与外部验证。

### 16.5 继续与停止的条件

如果新的同缓存多问题测试能显示 SQ 有稳定增量，且增量与正确证据使用相关，再设计针对性训练或预算方案。若当前缓存连受控可见证据任务都无法恢复，应考虑重新设计缓存覆盖或压缩训练。

如果强格式适配基线下 query 仍长期接近零，也没有成本优势，应调整主张，承认本设置中主要有效的是无问题接口适配。不能依靠更复杂的模块、不断更换子集或反复选择 checkpoint 来补成论文结论。

## 17. 可直接用于向学长汇报的版本

我现在把问题收敛到了“文档已经离线压成可复用的 soft memory 后，能不能根据当前问题，通过一个投影器改善冻结 decoder 的读取”。文档缓存不随问题变化，query 在缓存之后进入新模块；encoder、decoder 和原来的 LoRA 都冻结，只训练投影器。

实现上，当前是每篇共用的残差 MLP。每篇有 8 个 4096 维 memory，先用这些 memory 对冻结 decoder 的问题 token hidden 做 attention，再把文档分支和条件分支映射到同一个 512 维空间，通过 GELU 输出残差。最后仍是 Z 加残差，8 个槽进、8 个槽出，没有二次压缩。输出层零初始化，所以起点与原 PISCO 的有效输入一致。

实验最需要区分的是共同适配收益和 query 增量。发布版直接读这份缓存时，答案经常很长，32 token 截断明显，所以 EM/F1 很低。训练投影器后分数涨很多，但无 query 的文档 MLP 也能做到，因此不能把这部分算成 query 创新。

首轮 test 前 2000 条确实看到过 SQ 比固定条件对照高 1.14 F1，跨文档版高 1.69；扩到全量后只剩 0.62 和 0.85，在新增加的 3405 条上约 0.3，dev 也没有稳定正结果。现在我补了真正删除问题分支的 S0，两 seed 独立训练，全量 test 平均只高 0.37 F1，区间接近并跨过零。comparison 有局部正向信号，bridge 几乎没有收益。

为了让模块更依赖问题，我试过支持文档监督，分类 recall 明显涨，但换错问题后排序基本不变，QA 没涨；把监督直接接到最终 E 也没改善。后来做乘性 FiLM 和等容量加法，对照差接近零。进一步加冻结 γ 的同顺序对照，发现原来相对历史 baseline 约 0.9 点的上涨不需要 γ，训练顺序变化就能解释同量级差异。

所以现在我能证明问题进入了投影器、能影响输出，但还不能证明它提供了足够稳定的额外任务价值。下一步我更想确认当前 SQ/S0 的 raw–memory 缺口究竟还落在哪一段，以及在同一缓存面对多个问题时，条件化是否有必要。根据这个诊断再决定改监督、改缓存，还是承认当前有效部分主要是接口适配，而不是继续盲改融合结构。

## 18. 代码、实验来源与复核入口

### 18.1 当前实现文件

下列链接固定到本报告核对的分支快照，避免以后更新影响对本文的复核。

| 文件 | 主要内容 |
|---|---|
| [src/projector.py](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/src/projector.py) | Joint/Shared projector、条件分支、文档 attention、支持头、γ 干预 |
| [src/model.py](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/src/model.py) | 冻结接入、问题编码、输入层替换、CE/辅助损失、checkpoint 保护 |
| [src/train.py](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/src/train.py) | 训练、验证、控制评估、参数统计、G0 更新约束 |
| [config.py](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/config.py) | preset、结构开关、兼容性校验与臂命名 |
| [src/metrics.py](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/src/metrics.py) | EM/F1/substring 定义 |

### 18.2 设计与结果记录

| 轮次 | 设计 | 结果 |
|---|---|---|
| 联合展平版 | [JOINT_QUERY_PROJECTOR.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/JOINT_QUERY_PROJECTOR.md) | 本次来源未给出真实 QA 主表 |
| 共享投影器 | [SHARED_QUERY_PROJECTOR.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/SHARED_QUERY_PROJECTOR.md) | [SHARED_QUERY_PROJECTOR_RESULTS.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/SHARED_QUERY_PROJECTOR_RESULTS.md) |
| hidden 支持监督 | [SUPPORT_DOCUMENT_SUPERVISION.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/SUPPORT_DOCUMENT_SUPERVISION.md) | [SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md) |
| output 支持监督 | [SUPPORT_OUTPUT_SUPERVISION.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/SUPPORT_OUTPUT_SUPERVISION.md) | [SUPPORT_OUTPUT_SUPERVISION_RESULTS.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/SUPPORT_OUTPUT_SUPERVISION_RESULTS.md) |
| γ / FiLM / G0 | [QUERY_MODULATED_FUSION.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/QUERY_MODULATED_FUSION.md) | [QUERY_MODULATED_FUSION_RESULTS.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/QUERY_MODULATED_FUSION_RESULTS.md) |
| 标准 SQ/S0 消融 | [QUERY_PROJECTOR_ABLATION.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/QUERY_PROJECTOR_ABLATION.md) | [QUERY_PROJECTOR_ABLATION_RESULTS.md](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/102653d18f48d74603c3cf95e25c6af28a2c081a/docs/QUERY_PROJECTOR_ABLATION_RESULTS.md) |

### 18.3 关键提交与服务器结果位置

| 内容 | 代码 / 结果提交 | 已报告服务器 run 根目录 |
|---|---|---|
| 共享结构与首轮结果 | `1610106` / `45ff79e` | `/data02/quro/runs/shared_projector_v1/` |
| hidden 支持监督 | `2375fb9` / `223301c` | `/data02/quro/runs/support_doc_v1/` |
| output 支持监督 | `49f1c16` / `fbafd7e` | `/data02/quro/runs/support_output_v1/` |
| AddG / FiLM | `58ab32c` / `6835784` | `/data02/quro/runs/query_fusion_v1/` |
| G0 与 γ 干预 | `d3d066e` / `2cca3bf` | 同上，`gamma_frozen/` 与 `intervention/` |
| 标准 S0 与两 seed 结果 | `e4fbff8` / `102653d` | `/data02/quro/runs/query_ablation/` |

服务器路径是来源索引，未在本次整理中访问。方案文档里的示例输出目录与实际 run 目录可能不同，应以结果文档和实际 `config.json` 为准。

### 18.4 已有分析脚本

| 脚本 | 作用 |
|---|---|
| `scripts/analyze_shared_projector.py` | 首轮各臂、全量/新增子集、query 控制的逐题配对统计 |
| `scripts/analyze_support_supervision.py` | 支持监督匹配差与相关控制 |
| `scripts/analyze_support_topk.py` | Recall/both 与排序集合变化；不执行筛选后 QA |
| `scripts/annotate_support_visibility.py` | 当前编码窗口内的支持句可见性 |
| `scripts/analyze_projector_fusion.py` | AddG 与 FiLM 匹配比较 |
| `scripts/analyze_gamma_intervention.py` | γ 置零/替换与生成输入变化 |

标准两 seed S0 结果文档报告的分析输出位于服务器 `query_ablation/analysis.txt`；当前分支脚本列表中没有一个单独命名的 SQ/S0 两 seed 汇总脚本，不能把其他脚本说成已经自动复现该专表。

### 18.5 标准消融的最小复现命令

以下是协议示例，不表示本文启动了新训练。模型、缓存、训练/验证文件必须为同一版本，两臂各用独立输出目录；seed43 时共同修改 seed 与目录。

```bash
# SQ：从发布版独立训练
python -m src.train --preset pisco_shared_projector \
  --projector_query_mode conditioned --projector_fusion none \
  --support_loss_weight 0 --seed 42 --steps 3000 --lr 5e-5 \
  --batch_size 2 --grad_accum 8 --select_metric em \
  --eval_every 250 --eval_every_samples 500 --eval_max_samples 2000 \
  --query_control --doc_control \
  --out_dir /data02/quro/runs/reproduce_projector/sq_s42

# S0：相同文档 MLP，真正移除条件分支，独立训练
python -m src.train --preset pisco_shared_projector \
  --projector_query_mode none --projector_fusion none \
  --support_loss_weight 0 --seed 42 --steps 3000 --lr 5e-5 \
  --batch_size 2 --grad_accum 8 --select_metric em \
  --eval_every 250 --eval_every_samples 500 --eval_max_samples 2000 \
  --query_control --doc_control \
  --out_dir /data02/quro/runs/reproduce_projector/s0_s42

# 示例：S0 last 的内部全量 test
python -m src.train --preset pisco_shared_projector \
  --projector_query_mode none --eval_only \
  --resume_from /data02/quro/runs/reproduce_projector/s0_s42/checkpoint_last.pt \
  --eval_files test=/data02/quro/data/hotpot/test.jsonl \
  --eval_max_samples 5405 --query_control --doc_control \
  --out_dir /data02/quro/runs/reproduce_projector/s0_s42_fulltest
```

SQ 全量评估使用对应 mode、checkpoint 与输出目录，其余相同。当前 `eval_max_samples=0` 会取零条，而不是无限条；更换数据时显式使用实际样本数。best 补充评估只换成各自 best checkpoint，并保持同一选择规则与评估集合。

### 18.6 文献与历史总结

- [QuRO 2026-10-02 总结：诊断逐渐明确，解决方案尚未建立](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/main/docs/QURO_RESEARCH_UPDATE_20261002.md)
- [PISCO: Pretty Simple Compression for Retrieval-Augmented Generation](https://arxiv.org/abs/2501.16075)
- [Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective（SeleCom）](https://arxiv.org/abs/2602.15856)
- [xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token](https://arxiv.org/abs/2405.13792)

文献比较用于界定研究问题，不是完整新颖性审查。本文没有依据这些少量来源确认“首个”主张，也没有把相关工作的论文结果与我们内部数据划分直接比较。
