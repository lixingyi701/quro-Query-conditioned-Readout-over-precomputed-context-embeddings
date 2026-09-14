# 专题补充阅读——QCFuse 和 DisComp

> **专题目标**：补充考察“离线可复用表示 + 在线 query-aware 操作”的混合压缩路线，并明确 QCFuse、DisComp 与 QuRO 的关系、可借鉴实验和当前实现优先级。  
> **当前项目决定（2026-09-14）**：QCFuse 与 QuRO 的系统思想高度相关，但 QuRO 现阶段暂不研究 KV cache，因此只做方法定位和实验设计储备，不进入 v0.0 实现；DisComp 属于硬压缩，方法同构性较弱，只作为可选比较对象和混合范式旁证。

---

## 1. 一页结论

| 工作 | 压缩/缓存对象 | Query 参与位置 | 与 QuRO 的相关度 | 当前处理方式 |
|---|---|---|---|---|
| **QCFuse** | Position-independent KV cache、chunk anchors | 在线选择需要重算的原 token 位置 | 高：同样采用 offline reuse + online query-aware processing | 必须理解并主动切割；暂不研究或复现 KV-cache 系统 |
| **DisComp** | 文本摘要与剪枝后的离散 token | 在线句子相关性排序与剪枝 | 中低：共享混合范式，但不属于 soft compression | 定向阅读；必要时作为 hard-compression 补充基线 |

二者共同提供的最重要结论是：

$$
\boxed{
\text{Offline reusable representation}
+
\text{Online lightweight query-specific operation}
}
$$

正在成为一种合理的系统折中。但它们都没有覆盖 QuRO 的完整方法：

$$
\mathbf Z_d=f_{\mathrm{off}}(d),\qquad
\mathbf E_q=r_{\mathrm{online}}
\left(q,[\mathbf Z_{d_1};\ldots;\mathbf Z_{d_K}]\right),
$$

其中在线阶段只读取预计算 soft embeddings，**不重新访问或计算 source tokens**。

---

# Part I：QCFuse

## 2. 论文信息与问题定义

> **论文**：*QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving*  
> **arXiv**：[2606.05875](https://arxiv.org/abs/2606.05875)  
> **代码**：[uYanJX/QCFuse](https://github.com/uYanJX/QCFuse)  
> **注意**：本笔记讨论的是 RAG serving / KV-cache fusion 方向的 QCFuse，不是其他同名工作。

QCFuse 研究的是 RAG 中的重复 prefill：不同 query 可能重复召回相同文档块，如果每次都重新对这些 token 做完整 prefill，会产生大量冗余计算。

Position-Independent Caching（PIC）可以提前为每个 chunk 计算 KV cache，并在不同 query、不同拼接位置复用。但每个 chunk 的 cache 是独立生成的，缺少：

- 当前 query 与文档之间的条件化关系；
- 多个召回 chunk 之间的相互作用；
- 当前请求中的完整位置和前缀上下文。

因此，cache fusion 通常需要选择部分原始 token 重新计算 K/V，以恢复请求特定的上下文化表示。QCFuse 的核心问题是：

> 怎样使用 query-aware 的选择信号找到真正需要重算的 token，同时避免 selector 本身读取完整 context 和全部层 KV，从而成为新的在线延迟瓶颈？

---

## 3. QCFuse 方法

### 3.1 两阶段结构

QCFuse 将处理分为离线和在线两个阶段。

#### Phase I：Cache Preparation

对可复用文档 chunk 预计算并保存：

1. position-independent chunk KV cache；
2. 每个 chunk 的少量 anchor token/cache；
3. 通过离线 profiling 得到的 critical layer 集合。

#### Phase II：Query-Aware Fusion

当前 query 到来后：

1. query 只读取压缩的 chunk anchors，而不是完整文档；
2. 根据 query-conditioned signal 对原始 context token 位置评分；
3. 只加载少数 critical layers 的相关 K 状态完成定位；
4. 得到需要重新计算的位置集合 $\mathcal P$；
5. 对 $p\in\mathcal P$ 的 K/V 做稀疏重算并写回 fused cache；
6. 将稀疏重算与后续层 KV 加载重叠执行。

定义 recomputation ratio：

$$
\rho=\frac{|\mathcal P|}{N},
$$

其中 $N$ 是当前请求的 context token 总数。$\rho$ 越大，越接近 full prefill 的质量，但 TTFT 和计算成本也越高。

### 3.2 它解决的核心两难

QCFuse 将现有策略分成三类：

| 策略 | Query 信号 | 优点 | 问题 |
|---|---|---|---|
| Query-agnostic selection | 无 | 快、易于流水化 | 容易把预算分给与当前请求无关的位置 |
| Lightweight query-aware selection | 部分层或最终层 | 延迟较低 | 相关性信号可能不够可靠 |
| Full-view query-aware selection | 完整 context、多层 KV | 定位质量高 | 必须提前加载大量 KV，阻塞 layer-wise pipeline |

QCFuse 的选择是：保留 query-aware selection，但让 query 只访问压缩 evidence view。

这和 QuRO 的核心直觉非常接近：

> Query specificity 不应以重新处理完整原文为代价，而应该建立在可复用的压缩中间表示上。

---

## 4. QCFuse 实验

### 4.1 模型和系统

论文在统一的 SGLang BF16 serving stack 中实现，测试：

- Mistral-v0.3-7B；
- Llama-3.1-8B；
- Qwen3-8B；
- Qwen3-14B。

系统实验使用两张 NVIDIA H20，并实现 Triton 稀疏 KV recomputation kernel。因此其主要结果同时受 selector、缓存加载、硬件带宽和 kernel 实现影响，不应只理解成模型结构对比。

### 4.2 数据集

论文使用两类任务：

| Benchmark | 子任务 | 主要能力 |
|---|---|---|
| LongBench | MuSiQue、2WikiMQA、HotpotQA | 多文档、多跳证据融合 |
| RULER | multi-query retrieval、multi-value retrieval、variable tracking | 精确检索、多个 key/value、状态追踪 |

主要设置将 context 切成 512-token chunks，每个请求使用约 20 个 chunks。它适合检验 context 增长后 query-aware selector 是否仍能定位分散证据。

### 4.3 基线

QCFuse 的对手主要属于 KV-cache reuse/fusion，而不是 soft compression：

- Full Prefill；
- Direct PIC Reuse；
- CacheBlend；
- EPIC；
- FusionRAG；
- ProphetKV。

它们覆盖的核心轴是：是否使用 query、选择时需要加载多少 KV、是否兼容 layer-wise pipeline、重算比例是多少。

### 4.4 主要结果

论文报告 QCFuse 在 matched-quality operating point 上：

- 相对 full prefill，平均 TTFT 加速约 $1.7\times$；
- 相对 ProphetKV，平均 TTFT 加速约 $1.5\times$；
- 在部分任务上，以部分 token 重算恢复到 full-prefill-level quality。

最值得 QuRO 借鉴的不是绝对数字，而是评价方式：

1. 画 quality--TTFT frontier，而不是只报告准确率或 FLOPs；
2. 在 matched quality 下比较系统成本；
3. 改变 context/chunk 数测试扩展性；
4. 测试带宽敏感性；
5. 测试并发负载下的吞吐；
6. 对 query-aware signal 和压缩 evidence view 分别做消融。

---

## 5. QCFuse 与 QuRO 的关系

### 5.1 相同点

两者都观察到：

- 文档或 chunk 相对稳定；
- query 在请求之间变化；
- 昂贵的文档侧计算应该离线完成并复用；
- 在线阶段仍然需要 query-aware 操作恢复请求特异性；
- full-view query-aware processing 可能抵消缓存带来的延迟收益。

从系统抽象看，二者都属于：

$$
\text{offline precomputation}
+
\text{online query-conditioned adaptation}.
$$

### 5.2 根本区别

| 维度 | QCFuse | QuRO |
|---|---|---|
| 离线载体 | 各层 KV cache + anchors | 每篇文档的 $m$ 个 soft latents |
| 在线 query 操作 | 选择原 token 位置进行 KV 重算 | $B$ 个 output slots 读取 $Km$ 个 cached latents |
| 是否保留原 token 级 cache | 是 | 不要求 |
| 是否回访 source-token positions | 是 | 否 |
| 在线输出 | 融合后的完整/部分重算 KV cache | $B$ 个 generator-conditionable soft tokens |
| 主要研究问题 | RAG serving、cache fusion、TTFT | Soft compression、readout、生成质量、重复查询摊销 |

最重要的切割句是：

> **QCFuse uses a compressed view to decide which original token positions should be recomputed, whereas QuRO performs the entire online path in compressed embedding space and never revisits the source tokens.**

### 5.3 它是否覆盖 QuRO 的创新

没有。QCFuse 支持“离线/在线时间分解”这一大方向，但没有研究：

- 从预计算 soft-compressed embeddings 中生成新的 soft embeddings；
- $B$ 个 output slots 对 $K$ 篇文档的全局联合读出；
- 压缩器 latent 是否足以替代原文；
- generator 如何消费 readout 输出；
- 软压缩倍率、最终输出预算 $B$ 与答案质量的关系。

但它会削弱一种过宽的 QuRO 创新表述：我们不能声称首次提出“offline cache + online query-aware operation”。创新必须落在 **compressed embedding readout for generation without source revisit** 上。

---

## 6. QCFuse：当前研究决定

### 6.1 现在保留什么

- related work 中主动讨论并精确切割；
- 借鉴 offline/online 系统叙事；
- 借鉴 quality--TTFT、matched-quality、throughput 和 scalability 评价；
- 考虑使用 MuSiQue、2WikiMQA、HotpotQA 做多文档联合读出；
- 考虑使用 RULER 的精确检索任务作为 representation sufficiency 压力测试。

### 6.2 现在暂缓什么

- 不研究 KV-cache compression/fusion；
- 不实现 PIC；
- 不实现 token-position selective recomputation；
- 不实现 SGLang/Triton cache-fusion pipeline；
- 不复现 CacheBlend、EPIC、FusionRAG、ProphetKV；
- 不将 QCFuse 放入 v0.0 模型主表的同构基线。

暂缓原因不是 QCFuse 不重要，而是当前 QuRO 的最小研究闭环应先回答：

$$
\boxed{
\text{能否从 cached soft latents 中进行有效的 query-conditioned generation readout？}
}
$$

在这个命题得到验证之前扩展到 KV-cache 系统，会混入新的表示空间、运行时和工程变量，削弱主实验的可解释性。

### 6.3 何时重新启动 KV-cache 方向

满足以下条件后再重新评估：

1. QuRO 的 xattn readout 已在真实 PISCO/COCOM cache 上跑通；
2. query-conditioned 相对 agnostic control 有稳定增益；
3. 重复查询摊销曲线和交叉点 $Q^*$ 已验证；
4. 需要进一步降低 TTFT，且瓶颈被定位到 generator prefill/KV 而不是 compressor/readout；
5. 投稿方向转向 serving/system，要求与 KV-cache fusion 进行更直接比较。

---

# Part II：DisComp

## 7. 论文信息与核心方法

> **论文**：*DisComp: A Two-Stage Prompt Optimization Framework Combining Task-Agnostic and Task-Aware Compression*  
> **会议**：Findings of NAACL 2025  
> **论文页面**：[ACL Anthology](https://aclanthology.org/2025.findings-naacl.58/)

DisComp 是文本空间的两阶段硬压缩：

$$
D
\xrightarrow{\text{task-agnostic summarization}}
\widetilde D
\xrightarrow[\text{query relevance}]{\text{sentence pruning}}
\widetilde D_q.
$$

### 7.1 Stage 1：Task-agnostic summarization

- 使用 GPT-3.5-Turbo 生成教师摘要；
- 以 T5-large 为 student；
- 使用 cross-entropy loss 和 keyword matching loss 蒸馏摘要能力；
- 目标是产生较短、通用且保留关键词的文本表示。

这部分可以在 query 到来前完成，具有一定可复用性。

### 7.2 Stage 2：Task-aware sentence pruning

- 将 query 和摘要句子编码为 contextual embeddings；
- 计算句子与 query 的相关性；
- 删除低分句子；
- 将剩余离散文本输入下游 LLM。

Stage 2 的 query conditioning 发生在已经缩短的摘要上，因此比直接使用大模型重读完整原文便宜。

---

## 8. DisComp 实验与数据

### 8.1 数据集

| 数据集 | 设置 | 对 QuRO 的价值 |
|---|---|---|
| LongBench | 单文档 QA、多文档 QA、摘要、few-shot、synthetic、code | 可扩展评测，但任务过杂，需选择子集 |
| ZeroSCROLLS | 多种长文本理解任务 | 可作泛化补充，不是 RAG 主训练集 |
| NaturalQuestions | 每个问题配 20 篇文档，只有一篇含答案 | 很适合测试多文档噪声和全局联合读出 |

NaturalQuestions 还将正确文档放在第 1、5、10、15、20 个位置，测试 lost-in-the-middle 和文档顺序敏感性。这个设置可以迁移到 QuRO：

- 保持同一 query 和证据文档；
- 改变证据文档在 $K$ 篇输入中的位置；
- 检查 source/order embedding 是否有效；
- 检查 $B$ 个 output slots 能否忽略大量无关文档。

### 8.2 基线

DisComp 主要比较：

- Selective-Context；
- LLMLingua；
- LLMLingua-2；
- LongLLMLingua；
- BM25、Gzip、SBERT、OpenAI embedding 等句子检索方法。

这些都是文本选择、文本摘要或离散 token 压缩方法，不是 PISCO、COCOM、ICAE 一类 soft embedding compressor。

### 8.3 结果应该怎样看

论文报告 DisComp 在 LongBench、ZeroSCROLLS 和 NaturalQuestions 上优于多种 task-agnostic/task-aware 文本压缩方法，并在部分对比中比 LongLLMLingua 等在线压缩器明显更快。

但对 QuRO 而言，这些结果只能说明：

> Task-agnostic reduction 与 query-aware pruning 可以互补。

它不能直接说明：

- soft latents 能否保存未来 query 所需的信息；
- query-conditioned embedding readout 是否有效；
- generator 是否能消费新的 readout embeddings；
- 重复查询时 soft cache 与文本摘要哪一个更优。

---

## 9. DisComp 与 QuRO 的关系

| 维度 | DisComp | QuRO |
|---|---|---|
| 方法类别 | Hard/text compression | Soft embedding compression + readout |
| 离线表示 | 可读文本摘要 | 连续 latent memory |
| 在线选择粒度 | 句子 | Latent/soft-token |
| 生成器接口 | 普通文本 token | `inputs_embeds` / generator-compatible vectors |
| 是否需要生成器适配 | 通常不需要 | 需要 generator LoRA 或适配 |
| 主要优势 | 通用、可解释、可用于闭源 LLM | 高压缩率、端到端可训练、表示级融合 |
| 与 QuRO 的同构程度 | 低 | — |

DisComp 的价值主要有三点：

1. 为“query-agnostic first stage + query-aware second stage”提供文本空间先例；
2. 提供 LongBench/NaturalQuestions 的多文档和位置敏感评测方案；
3. 可作为 hard compression 组中的补充比较对象。

它不应成为 QuRO 的主要对手。主表更应该优先包括：

- Uncompressed RAG；
- PISCO / COCOM；
- Query-agnostic readout control；
- SeleCom（query-conditioned 质量上限）；
- ArcAligner（生成路径上的近邻）；
- LLMLingua-2（标准 hard-compression 代表）。

如果资源和篇幅充足，再加入 DisComp；如果只能选择一个 hard baseline，优先使用更标准、复现链路更清晰的 LLMLingua-2。

---

## 10. DisComp：当前研究决定

### 保留

- related work 中作为 hybrid hard-compression 先例；
- NaturalQuestions 20 文档与证据位置实验；
- task-agnostic/task-aware 两阶段消融思路；
- 有余力时作为补充 hard baseline。

### 暂不投入

- 不复现 GPT-3.5 摘要数据生成；
- 不训练其 T5-large summarizer；
- 不把 DisComp 纳入 soft-compression 主方法比较；
- 不把它升级为新的必读主论文。

---

## 11. 两篇论文对 QuRO 的共同启发

QCFuse 与 DisComp 分别在 KV cache 和文本空间说明，纯 query-independent 与纯 query-conditioned 两条路线之间存在可组合空间：

$$
\text{Reusable coarse representation}
\rightarrow
\text{Query-conditioned refinement/selection}.
$$

QuRO 的差异是把该原则落实到 soft embedding generation：

$$
D
\xrightarrow[\text{offline once}]{\text{PISCO/COCOM}}
\mathbf Z_D
\xrightarrow[\text{online per query}]{\text{multi-slot readout}}
\mathbf E_q
\xrightarrow{\text{generator}}
y.
$$

我们当前要证明的不是“混合路线存在”，而是三个更具体的命题：

1. **Sufficiency**：宽松压缩后的 $\mathbf Z_D$ 足以支持未来不同 query；
2. **Specificity**：query-conditioned readout 显著优于 query-agnostic readout；
3. **Amortization**：重复查询下，离线压缩成本能够快速被摊薄，并优于每次重读原文的 query-conditioned compressor。

---

## 12. 阅读与实现优先级

### 当前顺序

$$
\text{ArcAligner 完整精读}
>
\text{QCFuse 定向理解}
>
\text{DisComp 定向阅读}.
$$

### QCFuse 需要读到的范围

- Introduction：query awareness 与 pipeline 的两难；
- §3.1--3.2：offline/online workflow 与 selection bottleneck；
- §3.3--3.4：anchor query probing 和 critical-layer localization 的概念；
- §4：quality--TTFT、context scaling、bandwidth、throughput、ablation；
- 不深入实现 Triton kernel 和 KV-cache 工程细节。

### DisComp 需要读到的范围

- 两阶段方法；
- NaturalQuestions 20-document 设置；
- task-agnostic/task-aware 消融；
- latency 比较的口径；
- 不复现摘要器训练和数据生成。

---

## 13. 最终结论

### QCFuse

与 QuRO 的系统思想高度相关，是必须引用和正面切割的邻近工作；但其核心对象是 KV-cache fusion 和原 token 稀疏重算，不是 soft embedding readout。当前只保留方法定位、数据集和系统评价思路，**暂不启动 KV-cache 研究或实现**。

### DisComp

共享“先通用压缩、再 query-aware 选择”的混合思想，但属于硬压缩，和 QuRO 的直接对比性有限。它可以作为补充 hard baseline，主要价值是 NaturalQuestions 多文档位置实验和两阶段消融设计。

### 对 QuRO 的定位影响

不能将创新宽泛地表述为首次组合 offline cache 与 online query awareness。应聚焦为：

$$
\boxed{
\text{Query-conditioned multi-document readout over reusable soft-compressed memories,}
\\
\text{without revisiting source tokens.}
}
$$

