# RAG 中的软上下文压缩(Soft Context Compression)综述

> 面向"把检索文档压成 soft token 再喂给生成端 LLM"这一技术路线的系统梳理。
> 主线:ICAE → COCOM → PISCO → SeleCom;旁线补齐 xRAG / 500xCompressor / mean-pooling / DEX-Comp / LCLM。
> 整理时间:2026-09。文中数字均来自公开论文/官方页面,未取到的行已明确标注。

---

## 1. 问题形式化

给定 query $q$ 与检索得到的文档集 $D=\{d_1,\dots,d_k\}$:

- **压缩器(compressor / encoder / selector)** $f_\theta: d \mapsto E \in \mathbb{R}^{\ell \times h}$,其中 $h$ 是生成器的隐层维度,$\ell \ll |d|$。
- **压缩率** $\xi = |d| / \ell$。
- **生成器(decoder / generator)** $g_\phi(a \mid \text{prompt} \oplus E_1 \oplus \cdots \oplus E_k \oplus q)$。

$E$ 常被称作 soft token / memory slot / context embedding / gist。关键点:**它与词表 embedding 同维,但不落在词表 embedding 的流形上**——这既是它能承载超越单词信息的原因,也是它容易与生成器的原生分布失配的原因(SeleCom 所谓的"infeasibility"正是指这一点)。

与之对立的是 **hard compression**(LLMLingua/LLMLingua-2、RECOMP-extractive 等):输出仍是自然语言 token,可读、即插即用、不需要改生成器,但压缩率上限低、且离散删词天然有损。软压缩的卖点是压缩率可以高一到两个数量级,代价是必须动生成器(至少是 LoRA)、且不可读。

---

## 2. 设计空间:六个近似正交的轴

读这个领域的论文时,把任何一篇往下面六个格子里放,脉络就清楚了。

| 轴 | 取值 | 代表工作 |
|---|---|---|
| **① 压缩载体** | memory token 的输出隐状态 | ICAE、COCOM、PISCO |
| | memory token 的 **KV cache** | 500xCompressor |
| | 检索器 dense embedding + MLP 投影 | xRAG |
| | 隐状态 **mean-pooling** | Simple Context Compression |
| **② 压缩器骨干** | LLM + LoRA(与生成器同源) | ICAE、COCOM、PISCO |
| | 小型 BERT | COCOM-light |
| | 独立 decoder-only 小模型 | SeleCom |
| | 独立小 encoder,联合预训练 | LCLM(0.6B enc / 4B dec) |
| **③ 生成器是否可训** | 冻结 | ICAE、500xCompressor、xRAG |
| | 一并训练(LoRA 或全参) | COCOM、PISCO、SeleCom、DEX-Comp |
| **④ 是否 query-conditioned** | query-agnostic(可离线预压全库) | ICAE、COCOM、PISCO、xRAG、DEX-Comp |
| | query-conditioned(在线编码) | SeleCom、ATACompressor、LooComp、QGC |
| **⑤ 训练信号** | AE 重构 + LM 续写 预训练 → SFT | ICAE、COCOM |
| | 纯序列级知识蒸馏(SKD) | PISCO |
| | 蒸馏 warm-start + 硬样本 RL | DEX-Comp |
| | 大规模合成 QA + 课程学习 | SeleCom |
| **⑥ $\ell$ 的确定方式** | 固定常数 | PISCO、xRAG |
| | 按 $\xi$ 随文档长度线性伸缩 | COCOM |
| | 单模型支持多压缩率 | Simple Context Compression |

**④ 是整个领域目前最本质的张力**,后文 §6.1 展开。

---

## 3. 四篇核心工作精读

### 3.1 ICAE(2023.07,Microsoft,[arXiv:2307.06945](https://arxiv.org/abs/2307.06945))

**定位:证明这条路走得通。**

- 架构:encoder = 目标 LLM + LoRA(额外参数 **< 1%**);decoder = **冻结的目标 LLM 本身**。
- 做法:在长文本后追加 $k$ 个可学习的 memory token,取它们的输出隐状态作为 memory slots,直接拼进冻结 decoder 的输入。
- 预训练双目标:
  - **AE**:特殊 token `[AE]` 触发,要求 decoder 从 memory slots 重构原文;
  - **LM**:特殊 token `[LM]` 触发,要求 decoder 做续写。
  两者共同保证 memory slots 既"存得住"又"用得上"。
- 微调:少量 instruct 数据,让 memory slots 能与各种 prompt 交互。
- 结果:基于 Llama 实现 **4×** 压缩,推理延迟与显存均有收益。作者把它类比认知科学里的工作记忆。

**局限(被后续论文反复引用):**
1. 压缩后的下游任务表现与未压缩上下文仍有可见差距;
2. **高压缩率下崩塌**;
3. AE 目标要求"全量重构",这在 RAG 场景下既做不到也不必要(SeleCom 的核心批判)。

> ICAE 与 PCC 一起把 "compression token" 这一范式确立为默认架构——这个默认值到 2025–2026 才开始被系统性质疑。

---

### 3.2 COCOM(2024.07,Naver Labs Europe,[arXiv:2407.09252](https://arxiv.org/abs/2407.09252))

**定位:把通用上下文压缩专门化为 RAG 上下文压缩。**

相对 ICAE 的三个关键改动:

1. **解码器也训练。** 此前工作只调压缩模块、冻结生成端;COCOM 明确 tune 所有组件。这是它能在高压缩率下不崩的主因。
2. **压缩器与生成器共享同一个 LLM**(COCOM-light 例外,用 `bert-base-uncased` 当压缩器,成本更低)。
3. **面向多文档。** 早期方法基本是单段落设定,COCOM 直接处理 top-k 个上下文。

- Context embedding 数量 $k=|E|$ 由压缩率 $\xi$ 与文档 token 数 $n$ 共同决定,即随文档长度自适应。
- 预训练任务:从 context embeddings 做 auto-encoding + language modeling;之后在 QA 数据上微调。
- 实验设置:检索用 SPLADE-v3,DeBERTa-v3 做 top-50 重排,取 top-5 文档;数据涵盖 NQ / MS MARCO / HotpotQA / WikiQA;指标 EM / Match。
- 压缩率 $\xi \in \{4, 16, 128\}$。

**结果:**
- $\xi=4$ 时在 ASQA 上与未压缩基线**无显著差异**;
- 相对未压缩上界,$\xi=4$ 平均掉 ~4 分,$\xi=128$ 掉 ~10 分;相对 no-context 下界高出最多 17 分;
- 端到端最高 **5.69×** 加速;
- 在 NQ 等数据集上效率与精度双双超过 ICAE 和 xRAG。

**工程上真正的杀手锏:query-agnostic。** 文档编码与 query 无关 ⇒ 整个语料库可以**离线预压缩并落盘**,在线只付生成端的钱。公开权重如 `naver/cocom-v1-128-mistral-7b`(用 5 个上下文训练,推理也应给 5 个)。

---

### 3.3 PISCO(2025.01,Naver Labs Europe,[Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.800/) / [arXiv:2501.16075](https://arxiv.org/abs/2501.16075))

**定位:把前人视作必需品的"AE/LM 预训练"整个删掉,效果反而更好。**

- **核心方法:序列级知识蒸馏(SKD)。** teacher = 看到**完整未压缩文档**的 LLM;student = compressor–decoder 架构,只看压缩后的 embedding。损失就是 student 输出与 teacher 输出序列之间的交叉熵。
- **因此不需要标注 QA 数据**,只需要从文档自动生成开放式问题即可训练。这一点大幅降低了复现门槛。
- 每篇文档压成固定的 $\ell$ 个 embedding。注意 PISCO 设定下文档很短(约 128 token),所以 "rate 16" ⇒ 8 个 embedding,"rate 128" ⇒ **1 个 embedding**。读它的压缩率数字时必须带上这个前提。
- 训练时检索文档数 $k=5$。

**结果:**
- **16× 压缩,精度损失仅 0–3%**,覆盖多种 RAG QA 任务;
- 比同期压缩模型高约 **8 个点**;
- 单张 A100 上 **48 小时**(ACL 版本写的是 24 小时,应为版本修订)即可微调一个 7–10B 模型,最高支持 128× 压缩。

**局限:**
1. **性能上界被 teacher 锁死**——学生的目标就是模仿未压缩模型,原则上永远追不上、更谈不上超过。这正是 DEX-Comp 攻击的靶心。
2. 仍是 query-agnostic 的全量压缩,无法针对具体问题过滤无关信息(SeleCom 的批评)。

---

### 3.4 SeleCom(2026.01,复旦 Zian Jia 等,WWW '26,[arXiv:2602.15856](https://arxiv.org/abs/2602.15856))

**定位:质疑"full-compression"这个范式本身,把 encoder 从"压缩器"重新定义为"query 条件下的信息选择器"。**

**两条批判(论文的立论核心):**
- **(I) 不可行性 infeasibility** —— 自编码式的全量压缩目标与 LLM 下游的生成行为相冲突;压缩器被迫保留一切,而生成器只需要其中一小部分。
- **(II) 不必要性 non-necessity** —— 全量压缩把注意力摊到大量无关 token 上,**稀释了任务相关信息的密度**,压缩表示被噪声信号饱和,反过来带偏下游生成。这也解释了"为什么很多软压缩方法打不过不压缩的 RAG"这个长期尴尬现象。

**架构三件套:**
1. **Selector**:**decoder-only 自回归骨干**(而非常见的 encoder),同时输入 query 与 document,把 query 条件下的必要信息自回归地压进 embedding;用 decoder-only 是为了借用其世界知识。
2. **Projector**:把 selector 的输出映射到生成器的语义空间。
3. **Generator**:LLM,基于这些 embedding 完成 RAG。

**两阶段训练:** 第一阶段用**大规模、多样、难度分级的合成 document-oriented QA 数据 + 课程学习**,训练 selector 做精准无噪的 query 条件抽取;第二阶段做下游 RAG 适配。

**评测:** NQ / TriviaQA / WebQuestions / PopQA / HotpotQA / FactKG 六个任务,指标 EM / F1 / LLM-judge;效率用 **TIL(总推理延迟)、GFLOPs、TTFT** 三件套。所有 baseline 用公开 checkpoint + 推荐配置复跑。

**Table 1 部分数据(NQ,Mistral-7B-Instruct,top-1,EM / F1 / LLM-judge):**

| 方法 | EM | F1 | LLM |
|---|---|---|---|
| LLM w/o RAG | 4.13 | 14.30 | 38.21 |
| LLM with RAG | 7.98 | 18.19 | 40.95 |
| LLM with RAG\*(在 SeleCom 同款数据上微调) | 41.12 | 49.86 | 50.88 |
| LLMLingua-2 (4×,hard) | 5.87 | 15.13 | 35.29 |
| ICAE (4×) | 14.43 | 25.99 | 37.83 |
| xRAG (164×) | 3.02 | 12.45 | 40.83 |
| COCOM (128×) | 31.86 | 40.15 | 42.44 |
| PISCO / SeleCom | *检索时被截断,需查原表* |

> ⚠️ **SeleCom 与 PISCO 自身的行没能从检索片段中取到**,请以原文 Table 1 为准。这两行恰恰是最关键的对比,不要引用我上面这张表下结论。

**总体结论:** SeleCom 稳定优于所有压缩方法,并与不压缩基线可比;top-5 检索下进一步提升;计算与延迟降低 **33.8%–84.6%**,且随 top-k 增大优势扩大(baseline 的开销陡增,它只是轻微退化)。

**代价(论文没有回避不了的地方):** query-conditioned ⇒ **无法离线预压全库**,每来一个 query 都要重新编码召回的文档。它靠"selector 很轻量"把这笔账做平,但这与 COCOM/PISCO 的离线缓存优势是**结构性冲突**的。

---

## 4. 需要放进坐标系的旁线工作

| 工作 | 关键贡献 | 对你有用的点 |
|---|---|---|
| **Gist tokens** | 改 attention mask 缓存 gist 激活,最高 26× | 载体可以是"注意力可见性"而非新增模块 |
| **AutoCompressor** | 递归压缩出 summary vectors,~15× | 长文档分段递归的原型 |
| **xRAG**(NeurIPS 2024,[arXiv:2405.13792](https://arxiv.org/abs/2405.13792)) | 把**检索器的 dense embedding 当作一种模态**,用轻量 MLP projector 投进 LLM 输入空间,单 token 极限压缩;plug-and-play | 极端压缩的上界探针;但长文档信息损失巨大(SeleCom 表里 164× 下 EM 仅 3.02) |
| **500xCompressor**([arXiv:2408.03094](https://arxiv.org/abs/2408.03094)) | 类 ICAE,但传给 decoder 的是压缩 token 的 **KV 值**而非输出 embedding。KV 保留细节更多,**高压缩率下优势尤其明显**;1/4/16 个 token 压 96–480 token,6×–480×,保留原能力的 62.26%–72.89% | **载体维度是被严重低估的设计轴** |
| **Simple Context Compression**(Cornell,[arXiv:2510.20797](https://arxiv.org/abs/2510.20797)) | ① **mean-pooling 一致优于 compression-token 架构**;② **multi-ratio 训练**:一个压缩器服务多个压缩率,代价很小;③ 开源 **BenchPress** 标准评测套件 | 直接动摇了 ICAE 以来的默认架构;multi-ratio 是自适应压缩的现成基建 |
| **DEX-Comp**([arXiv:2609.05152](https://arxiv.org/abs/2609.05152)) | 两阶段:**纯蒸馏**(只用未压缩 RAG **答对**的样本 warm start)+ **hard exploration**(只在未压缩 RAG **答错**的 query 上做 RL)。16× 压缩、4×–24× 加速,**号称第一个 query-independent 软压缩方法在多种检索深度下超过未压缩 RAG**;提出 Resilience Rate / Boost Rate 两个诊断指标 | 直接破解 PISCO 的 teacher 天花板 |
| **Token Overflow**(EACL 2026 SRW,[arXiv:2602.12235](https://arxiv.org/abs/2602.12235)) | 定义 **token overflow**:压缩表示已不足以回答该 query。发现 query-agnostic 饱和度统计能区分压缩/未压缩表示,但**检测 overflow 能力有限**;query+context 联合探针在 HotpotQA/SQuADv2/TriviaQA 上平均 **0.72 AUC-ROC** ⇒ **overflow 本质是 query 相关的,不是压缩表示的内禀属性** | 给"自适应压缩率"提供了理论与工具支撑 |
| **LCLM**([arXiv:2606.09659](https://arxiv.org/abs/2606.09659)) | 大规模架构搜索得出:**scaling decoder 比 scaling encoder 重要得多**。0.6B enc / 4B dec,各自 350B token 持续预训练,1:4/1:8/1:16;RULER@4k 8.8×、LongBench@64k 5.2× 加速 | 说明这条路可以 scale;但依赖重训练 + 改推理栈 |
| **CompressionAttack** | 首个把 prompt 压缩当作攻击面的框架(HardCom 离散扰动 / SoftCom 隐空间扰动) | 安全性是几乎空白的子方向 |

---

## 5. 演进主线:四次转折

1. **2023 · ICAE —— "可行性证明"**
   冻结的 LLM 确实能读懂由 LoRA encoder 产出的 memory slot,只要用 AE + LM 双目标训。范式确立:compression token + 冻结 decoder。

2. **2024 · COCOM —— "专门化为 RAG"**
   共享 backbone、**训练解码器**、原生支持多文档、query-agnostic 从而可离线预压全库。压缩率从 4× 推到 128×。

3. **2025 · PISCO —— "做减法"**
   AE/LM 预训练不是必需的,**纯序列级蒸馏就够,而且更好**。训练成本降到单卡 A100 两天,16× 下损失 0–3%。

4. **2026 · 四路分叉**
   - **A. 突破 teacher 上限**(保住 query-agnostic 的缓存优势):DEX-Comp 的"蒸馏 warm-start + 硬样本 RL"。
   - **B. 放弃可缓存性换信息密度**:SeleCom 的 query-conditioned 选择器;同路的还有 ATACompressor、LooComp、QGC。
   - **C. 简化架构**:mean-pooling 打败 compression token;multi-ratio 单模型多压缩率。
   - **D. 直接 scale**:LCLM 的架构搜索 + 大规模持续预训练。

---

## 6. 核心矛盾与开放问题

### 6.1 可缓存性 ↔ 信息密度(本领域第一性张力)

| | query-agnostic | query-conditioned |
|---|---|---|
| 代表 | COCOM、PISCO、DEX-Comp | SeleCom、ATACompressor |
| 优势 | 全库离线预压,在线**零编码成本**;向量可复用 | 信息密度高,天然过滤无关内容 |
| 劣势 | 被迫全量压缩 ⇒ 稀释 + overflow | 每 query 重编码;向量不可复用;存储无意义 |

这两者目前是**非此即彼**的。真正有价值的空白是**混合**:比如离线做粗粒度压缩(保住缓存),在线用极轻量的 query 条件模块做**再选择 / 重加权 / 局部展开**——这条路上 ArcAligner(自适应递归对齐)算是早期尝试,但远没被做透。

### 6.2 蒸馏范式的天花板

PISCO 式 SKD 的最优解就是"和 teacher 一样",不可能更好。DEX-Comp 用"只在 teacher 失败的样本上做 RL"绕开了这一点,并声称已超过未压缩 RAG。开放问题:过程级奖励、多 teacher 集成、自一致性信号能否进一步推高上界。

### 6.3 容量不匹配与 overflow

固定 $\ell$ 面对的是长度和信息量都高度不均的文档。Token Overflow 论文给出的关键结论是 **overflow 是 query 相关的**——同一份压缩表示对 A 问题够用、对 B 问题就已经溢出。

> **这里有一个很成型的组合机会**:先用轻量探针预测 overflow 风险,再**动态分配 token 预算**(而 multi-ratio 训练已经证明"一个压缩器服务多个压缩率"代价很小)。检测器 + 多率压缩器 = 自适应压缩率,两块积木都是现成的,但据我检索还没有人把它们拼起来。

### 6.4 架构默认值可能是错的

三条独立证据都指向同一个方向——ICAE 以来的"compression token + 输出隐状态 + 冻结大 decoder"未必最优:
- **载体**:KV 值 > 输出 embedding(500xCompressor),高压缩率下差距更大;
- **聚合方式**:mean-pooling > compression token(Simple Context Compression);
- **参数分配**:scale decoder > scale encoder(LCLM)。

这三条彼此正交,**据我所知还没有工作把它们组合消融过**。

### 6.5 评测不可比

- backbone(Llama / Mistral / Qwen)、检索器(SPLADE / Contriever / BGE)、top-k、指标(EM / F1 / LLM-judge)各家各不同;
- **压缩率的定义本身就不统一**:COCOM 的 $\xi$ 是真实比率,PISCO 的 rate 依赖"文档约 128 token"这一前提,xRAG 的 164× 是按平均长度折算的;
- 效率指标也不统一(wall-clock / GFLOPs / TTFT / TIL)。

BenchPress 是目前唯一的标准化尝试;SeleCom 坚持"用公开 checkpoint 在推荐设置下复跑所有 baseline"也是好实践。

### 6.6 忠实性与可解释性(几乎空白)

DEX-Comp 自己承认软压缩带来"检索信息如何被使用"的不透明性,并把 faithfulness / citation 评测列为 future work。压缩后无法做引用溯源,这对生产系统是硬伤。

### 6.7 安全性(几乎空白)

CompressionAttack 是唯一系统性工作。soft token 不可读 ⇒ 传统的 prompt 注入检测手段全部失效,这是一个明显的攻击面。

---

## 7. 结合你的方向可切入的几个点

按"空白程度 × 可做性"排序:

1. **选择性压缩 + 自适应 token 预算**(§6.3)。query-conditioned selector 决定"选什么",overflow 探针决定"给多少个 token"。SeleCom 只做了前者且 $\ell$ 固定,Token Overflow 只做了检测没接下游,multi-ratio 提供了训练基建。三块拼图现成,交叉点是空的。

2. **把 RL 用到 query-conditioned 选择器上**(§6.2 × §6.1)。SeleCom 用的是 SFT + 课程学习,没用 RL;DEX-Comp 用了 RL 但是 query-agnostic。**"query-conditioned + 硬样本 RL"这个格子是空的**,而且两边的动机是叠加的(选择器本就该学"什么该丢",RL 的稀疏信号正好告诉它丢错了什么)。

3. **两级缓存架构**(§6.1)。离线 query-agnostic 粗压 + 在线轻量 query-conditioned 精选,把 SeleCom 的信息密度优势和 COCOM/PISCO 的缓存优势合起来。这个 story 对工业界非常有说服力,难点在于两级之间的接口设计与联合训练。

4. **载体消融**(§6.4)。selective 压缩 + KV-cache 载体的组合没人做过;顺带把 mean-pooling vs compression token 在 RAG(而非通用长文本)设定下重新验证一遍。这类工作偏实证,但结论扎实、容易被引。

5. **统一复现与评测**。在 BenchPress 上把 ICAE / COCOM / PISCO / SeleCom 在同 backbone、同检索器、同 top-k、同压缩率定义下跑齐。工作量大但风险低,而且做完之后上面 1–4 项的实验骨架就都有了。

---

## 8. 参考文献

**核心四篇**
- ICAE — [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945),arXiv:2307.06945([MSR 页面](https://www.microsoft.com/en-us/research/publication/in-context-autoencoder-for-context-compression-in-a-large-language-model/))
- COCOM — [Context Embeddings for Efficient Answer Generation in RAG](https://arxiv.org/abs/2407.09252),arXiv:2407.09252([权重](https://huggingface.co/naver/cocom-v1-128-mistral-7b))
- PISCO — [PISCO: Pretty Simple Compression for Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-acl.800/),Findings of ACL 2025 / arXiv:2501.16075([Naver Labs](https://europe.naverlabs.com/research/publications/pisco/))
- SeleCom — [Rethinking Soft Compression in RAG: A Query-Conditioned Selector Perspective](https://arxiv.org/abs/2602.15856),arXiv:2602.15856,WWW '26

**旁线**
- xRAG — [Extreme Context Compression for RAG with One Token](https://arxiv.org/pdf/2405.13792),NeurIPS 2024
- 500xCompressor — [Generalized Prompt Compression for LLMs](https://arxiv.org/html/2408.03094v1),arXiv:2408.03094
- Simple Context Compression — [Mean-Pooling and Multi-Ratio Training](https://arxiv.org/abs/2510.20797),arXiv:2510.20797([BenchPress](https://github.com/lil-lab/benchpress))
- DEX-Comp — [Compression Beyond the Uncompressed](https://arxiv.org/abs/2609.05152),arXiv:2609.05152
- Token Overflow — [Detecting Overflow in Compressed Token Representations for RAG](https://aclanthology.org/2026.eacl-srw.59/),EACL 2026 SRW / arXiv:2602.12235
- LCLM — [End-to-End Context Compression at Scale](https://arxiv.org/abs/2606.09659),arXiv:2606.09659([代码](https://github.com/LeonLixyz/LCLM))
- PCC — [Pretraining Context Compressor for LLMs with Embedding-Based Memory](https://aclanthology.org/2025.acl-long.1394.pdf),ACL 2025

**综述与相关方向**
- [Prompt Compression for Large Language Models: A Survey](https://arxiv.org/html/2410.12388v2),NAACL 2025([taxonomy repo](https://github.com/ZongqianLi/Prompt-Compression-Survey))
- [Contextual Compression in RAG for LLMs: A Survey](https://arxiv.org/pdf/2409.13385),arXiv:2409.13385
- [Beyond Hard and Soft: Hybrid Context Compression](https://arxiv.org/html/2505.15774v1),arXiv:2505.15774
- ATACompressor([arXiv:2602.03226](https://arxiv.org/pdf/2602.03226))、LooComp([arXiv:2603.09222](https://arxiv.org/pdf/2603.09222))、ArcAligner([arXiv:2601.05038](https://arxiv.org/pdf/2601.05038))、COMI([arXiv:2602.01719](https://arxiv.org/pdf/2602.01719))

---

## 9. 本文的可信度说明

- 各方法的**架构描述、训练目标、定性结论**均来自论文/官方页面,可信。
- **具体数字**中,以下未能核到原表,引用前请自行核对:
  - SeleCom 与 PISCO 在 SeleCom Table 1 中自身的行(检索片段被截断);
  - PISCO 消融表的具体数值(SKD vs. gold-label、有/无预训练,在 rate 16 与 128 下);
  - COCOM 的逐压缩率 GFLOPs 表。
- PISCO 训练时长在 arXiv 版(48h)与 ACL 版(24h)之间不一致,应为版本修订。
- 本文写作时沙箱环境无法直接抓取 arXiv 全文,内容基于检索到的论文摘要、HTML 片段与官方页面综合而成;§6–§7 中"据我检索还没有人做过"一类判断是基于本轮检索范围的**弱断言**,投稿前需做正式的相关工作排查。
