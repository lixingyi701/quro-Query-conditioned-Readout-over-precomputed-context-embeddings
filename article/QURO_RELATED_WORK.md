# QuRO — Related Work 初稿

> 英文正文可直接改写入稿;文末附中文写作说明(§7)解释论证结构与雷区。
> 引用键为占位符,请替换为你的 `.bib` 键名。
> 起草日期:2026-09-08

---

## 2 Related Work

### 2.1 Context Compression for RAG

Retrieval-augmented generation grounds language models in external evidence, but the retrieved passages inflate the input and make the prefill stage the dominant serving cost. Context compression addresses this by shortening the evidence before it reaches the generator, and existing work falls into two families. **Hard compression** operates in token space, pruning tokens or sentences \citep{llmlingua, llmlingua2} or rewriting passages into shorter text \citep{recomp}. It preserves a text-only interface and remains compatible with closed-source models, but it must make its decisions at query time, and under aggressive budgets a single mistake can delete answer-critical evidence. **Soft compression** instead maps passages into a short sequence of continuous vectors that the generator consumes in place of tokens. It achieves compression rates an order of magnitude higher, at the cost of requiring access to — and adaptation of — the generator. Our work is situated entirely within the soft family.

### 2.2 The Compression-Token Lineage

Soft compression for LLMs was established by ICAE \citep{icae}, which appends learnable memory tokens to a context, encodes them with a LoRA-adapted copy of the target LLM, and feeds the resulting hidden states to the *frozen* LLM as memory slots. Training combines an auto-encoding objective (reconstruct the context) with a language-modeling objective (continue it). ICAE demonstrated feasibility at roughly 4× compression, but degrades sharply at higher rates. 500xCompressor \citep{500x} retains this design while changing the *carrier*, passing the key-value states of the compression tokens rather than their output embeddings, which preserves more detail precisely in the high-rate regime.

COCOM \citep{cocom} specialized the paradigm to RAG along three axes: it tunes the decoder rather than freezing it, shares a single backbone between compressor and generator, and handles multiple retrieved documents natively. It reports parity with uncompressed RAG at $\xi=4$ and a moderate degradation at $\xi=128$. PISCO \citep{pisco} then removed the auto-encoding/language-modeling pretraining stage altogether, showing that pure sequence-level knowledge distillation from an uncompressed teacher — trained on synthetic document-grounded questions, with no labeled QA data — both simplifies the recipe and improves accuracy, retaining 16× compression at a 0–3% accuracy cost. DEX-Comp \citep{dexcomp} identified the ceiling implied by that recipe: a distilled student can at best imitate its teacher. It warm-starts on the teacher's *correct* answers and then applies reinforcement learning exclusively to queries the uncompressed system fails, reporting the first query-independent soft compressor to surpass uncompressed RAG.

A parallel thread questions the architecture itself. \citet{simplecomp} show that mean-pooling hidden states consistently outperforms the ubiquitous compression-token design, and that a single compressor can be trained to serve multiple compression ratios at modest cost; \citet{densityaware} independently reproduce the mean-pooling finding. **QuRO is agnostic to this choice: our offline stage reuses any published compressor as a frozen module, and we ablate across PISCO, COCOM, and a mean-pooling variant in §7.7.**

### 2.3 The Cacheability–Specificity Dilemma

We organize prior soft compressors along the axis that motivates this work: whether compression is conditioned on the query.

**Query-agnostic** methods \citep{cocom, pisco, xrag, dexcomp} encode each document independently of any query, so the entire corpus can be compressed offline once and the resulting vectors reused across all future queries — the online cost of encoding is zero. The price is that the encoder must preserve *everything*, since it cannot know what will be asked.

**Query-conditioned** methods take the opposite trade. SeleCom \citep{selecom} argues this most sharply, recasting the encoder from a compressor into a query-conditioned *selector*, on two grounds: auto-encoder-style full compression is **infeasible**, in that the reconstruction objective conflicts with the generator's downstream behavior, and **unnecessary**, in that forcing all content into the representation dilutes the density of task-relevant information. Related work pursues the same intuition through leave-one-out scoring \citep{loocomp}, task-aware selective encoding \citep{atacompressor}, and query-guided token retention \citep{quito}. All of them, however, forfeit precomputation: every retrieved document must be re-encoded for every query, and the stored representation — if any — cannot be reused.

**This is a genuine dilemma rather than a gap in engineering, and neither branch escapes it.** QuRO's contribution is to observe that the two properties are separable in *time* rather than mutually exclusive: query-independent encoding and query-conditioned selection can occupy different stages of a single pipeline, with the expensive stage amortized offline and the query-dependent stage made cheap by construction.

### 2.4 Two-Stage and Cascaded Designs

Several recent systems place computation on both sides of the offline/online boundary, and we position QuRO against them explicitly.

RRK \citep{rrk} is the closest precedent for our mechanism: it compresses a collection offline with a *frozen* PISCO compressor and trains a decoder to score query–document pairs directly over those precomputed embeddings, reaching 3–18× speedups over smaller rerankers. RRK establishes that query-conditioned computation over frozen compressed representations is viable — but it targets **reranking**, producing a scalar relevance score. QuRO carries the mechanism to the **generation** path, where the readout must emit a representation the generator can condition on, not a score.

QCFuse \citep{qcfuse} implements the same cascade in **KV-cache space**: precomputed chunk caches are reused and a query-aware selector marks tokens for recomputation, using per-chunk anchors and critical-layer profiling. It articulates the dilemma we address — fast query-agnostic or final-layer selectors miss relevant evidence, whereas full-view query-aware selectors stall the layer-wise pipeline. QCFuse remains a serving-systems contribution operating over original token positions; QuRO operates in the compressed embedding space and never revisits the source tokens.

ArcAligner \citep{arcaligner} is the closest work on the generation path. It inserts a lightweight module into the language model's layers to *align* precomputed context slots, with a learned gate controlling recursive alignment depth. Crucially, ArcAligner improves how the generator **utilizes** compressed slots; it does not use the query to **select** among them. QuRO's readout is query-conditioned by construction, and we treat ArcAligner as our principal baseline for isolating the value of that conditioning.

Finally, \citet{beyondrag} precompute a *task-aware* reusable cache, occupying an intermediate point on our axis: reusable across queries within a task, but not conditioned on any individual query.

### 2.5 Latent-Bottleneck Architectures

QuRO's readout is an instance of the Perceiver family \citep{perceiver, perceiverio}, whose defining property is that a fixed-size latent array decouples computational cost from input length. We stress a distinction that clarifies our novelty relative to prior text-compression work in this family.

Existing applications adopt the Perceiver **encoder**: learned latents serve as queries and cross-attend over the raw context as keys and values. LCIRC \citep{lcirc} stacks Perceiver blocks into a recurrent long-form compressor; IC-Former \citep{icformer} condenses contextual embeddings into learnable digest tokens independently of the target LLM; MemCom \citep{memcom} applies layer-wise memory-to-source cross-attention; and STILL \citep{still} compacts the KV cache of a frozen LLM with learned latent queries trained by KL distillation. Because their latents are learned and query-independent, all of them inherit the limitation that compression cannot see the task objective.

QuRO instead adopts the Perceiver **IO decoder**: the *query itself* forms the output query array, cross-attending over already-compressed latents as keys and values. This inverts what is conditioned on what, and it maps the two halves of Perceiver IO onto the two halves of RAG serving — the encoder's bottleneck is paid once per document offline, while the decoder's readout is paid once per query at cost $O(B \cdot km)$, independent of the source document length. It also inherits a property we exploit in §7.5: in Perceiver IO the number of output queries determines the output length, so the generator-side token budget becomes a per-query decision.

We note the standard critique of this family. \citet{treecrossattn} observe that a Perceiver IO decoder can only retrieve what the latents already hold, so an aggressively small latent set discards information irrecoverably. **We take this objection seriously and answer it structurally rather than rhetorically: QuRO deliberately runs its offline stage at a *generous* ratio — cheap, because it is amortized over all queries — and defers aggressive reduction to the query-conditioned readout. §6 verifies empirically, using a query-conditioned overflow probe \citep{overflow}, that the offline stage has not yet discarded answer-critical evidence at the operating ratio.**

### 2.6 Adaptive Compression Budgets

A final thread allocates compression capacity non-uniformly. DAST \citep{dast} and COMI \citep{comi} distribute a fixed sample-level budget across segments by informativeness, while \citet{densityaware} vary the sample-level ratio itself, predicting it from intrinsic information density and quantizing it into discrete buckets; they report that a fully continuous ratio underperforms a static one, since models handle input-dependent continuous structural hyperparameters poorly. Separately, \citet{overflow} define *token overflow* — the regime in which a compressed representation no longer suffices to answer a given query — and show that it is detectable only with query conditioning, query-agnostic saturation statistics being insufficient.

**These two observations compose. All existing adaptive-budget methods allocate by document-side density and remain query-agnostic; yet overflow, the phenomenon that justifies a larger budget, is itself query-dependent. QuRO's readout makes query-conditioned budgeting natural, and we evaluate it in §7.5 under the discrete-bucket regime that \citet{densityaware} found necessary.**

---

## 3 引用清单(建议 `.bib` 条目)

| 键 | 论文 | 出处 |
|---|---|---|
| `icae` | In-context Autoencoder for Context Compression | arXiv:2307.06945 |
| `500x` | 500xCompressor | arXiv:2408.03094 |
| `cocom` | Context Embeddings for Efficient Answer Generation in RAG | arXiv:2407.09252 |
| `pisco` | PISCO: Pretty Simple Compression for RAG | Findings of ACL 2025 |
| `dexcomp` | Compression Beyond the Uncompressed | arXiv:2609.05152 |
| `selecom` | Rethinking Soft Compression in RAG | WWW '26 / arXiv:2602.15856 |
| `xrag` | xRAG: Extreme Context Compression with One Token | NeurIPS 2024 |
| `rrk` | Efficient Listwise Reranking with Compressed Document Representations | arXiv:2604.26483 |
| `qcfuse` | QCFuse: Query-Aware Cache Fusion via Compressed View | arXiv:2606.05875 |
| `arcaligner` | ArcAligner | arXiv:2601.05038 |
| `beyondrag` | Beyond RAG: Task-Aware KV Cache Compression | arXiv:2503.04973 |
| `simplecomp` | Simple Context Compression: Mean-Pooling and Multi-Ratio Training | arXiv:2510.20797 |
| `densityaware` | Density-aware Soft Context Compression | arXiv:2603.25926 |
| `dast` | DAST: Dynamic Allocation of Soft Tokens | arXiv:2502.11493 |
| `comi` | COMI: Coarse-to-fine Context Compression | arXiv:2602.01719 |
| `overflow` | Detecting Overflow in Compressed Token Representations | EACL 2026 SRW / arXiv:2602.12235 |
| `loocomp` | LooComp | arXiv:2603.09222 |
| `atacompressor` | ATACompressor | arXiv:2602.03226 |
| `quito` | QUITO: Query-Guided Context Compression | arXiv:2408.00274 |
| `perceiver` | Perceiver: General Perception with Iterative Attention | ICML 2021 |
| `perceiverio` | Perceiver IO | ICLR 2022 |
| `lcirc` | LCIRC | arXiv:2502.06139 |
| `icformer` | In-Context Former | arXiv:2406.13618 |
| `memcom` | MemCom(逐层 memory slot) | 待补全出处 |
| `still` | STILL / neural KV cache compaction | Baseten research,待确认是否有正式论文 |
| `treecrossattn` | Tree Cross Attention | ICLR 2024 / arXiv:2309.17388 |
| `llmlingua`,`llmlingua2`,`recomp` | 硬压缩基线 | 标准引用 |

---

## 4 中文写作说明

### 4.1 论证结构:为什么这样组织

这版 related work 不是按时间也不是按方法分类,而是**按一条论证线铺**,每一节都在为 QuRO 的存在理由添一块砖:

```
2.1  确立赛道(软压缩)
2.2  确立谱系(compression-token 一脉的演进)  → 顺带把"架构选择未定"埋下,为 2.6 铺垫
2.3  ⭐ 提出"可缓存性 vs query 特异性"困境    → 这是你的组织性框架,全文的支点
2.4  两阶段级联的邻居                         → 承认前人,精确切割
2.5  Perceiver 谱系 + encoder/decoder 之分     → 你的机制novelty落点
2.6  自适应预算                               → 你的附加贡献落点
```

**2.3 是全文支点。** 别把它写成普通的分类,要写成"这是一个真实的两难,不是工程没做到位"。困境立得越硬,QuRO"在时间上拆开两者"这一招就越显得是解法而非拼凑。

### 4.2 三个必须精确切割的邻居

这三家挨得最近,切割写不好就会被认为是增量工作。切割的措辞已写进正文,复述其逻辑:

| 邻居 | 相同 | **不同(你的护城河)** |
|---|---|---|
| **RRK** | 冻结压缩器 + 在压缩表示上做 query 条件计算 | 它输出**标量分数**(reranking);你输出**生成器可条件化的表示**。这是任务与输出空间的根本差异 |
| **QCFuse** | 离线预计算 + 在线 query 感知选择 | 它在 **KV cache 空间**、且最终要**回到原始 token 位置重算**;你全程在压缩 embedding 空间,不回访源 token |
| **ArcAligner** | 离线压缩 + 在线轻量模块 | 它做**对齐/利用**(query 无关);你做 **query 条件选择**。这是最容易被混淆的一家,措辞要最狠 |

> **策略建议:把 RRK 写成盟友而非对手。** "IR 侧已证明这个机制可行,我们把它带到生成侧"比"我们和 RRK 不一样"有力得多——既借了它的可信度,又天然划清了范围。

### 4.3 2.5 的写法是全文最关键的段落

必须让读者一眼看懂:**Perceiver 的 encoder 半已经很拥挤(LCIRC / IC-Former / MemCom / STILL),decoder 半基本空着。**

如果这一层区分没写清楚,审稿人会说"Perceiver 做压缩早有人做了"。**"latent 当 Q 去读原文" vs "query 当 Q 去读 latent"这个反转必须用一句话说死**,正文里那句 "This inverts what is conditioned on what" 就是干这个的,建议保留并加粗或斜体。

### 4.4 Tree Cross Attention 必须主动引用

它是针对 Perceiver IO 最锋利的批评(固定 latent 集合会不可逆地丢信息),**审稿人大概率会拿它打你。主动引、正面答,比被挖出来强得多。**

答法在正文已给出且必须是**结构性**的而非修辞性的:离线压得松(便宜,因为摊薄) + 激进缩减推迟到 query 已知之后 + **§6 用 overflow 探针实证**。注意末句刻意写了 "answer it structurally rather than rhetorically" ——这句是写给审稿人看的态度声明,建议保留。

### 4.5 待办与雷区

- `memcom`、`still` 两条**出处未核实**(STILL 目前只找到 Baseten 的 research 博客,不确定有无正式论文)。若查不到正式引用,要么删,要么以脚注形式提及。**不要用不确定的引用**。
- 2.2 提到的 COCOM/PISCO 具体数字(4× 打平、$\xi{=}128$ 退化、16× 损失 0–3%)**引用前请核对原文**。
- 正文中所有"据我们所知为首次"的暗示都已刻意避开,只写了 "closest precedent"、"carries the mechanism to"。**投稿前做完正式相关工作排查再决定要不要加 novelty claim** ——现在这版即使有人先做了,也不至于变成硬伤。
- 若目标是 ACL/EMNLP,2.1–2.6 全文约 900 词,通常需压到 600–700 词:优先合并 2.1 进 intro,2.6 缩成 3 句。**2.3、2.4、2.5 一个字都别砍**,那是三块承重墙。
- 若目标是 WWW/SIGIR(和 SeleCom、RRK 同场),**把 2.4 的 RRK 段和 §5 的摊薄论证往前提**,IR 社区对 serving 成本和可缓存性的敏感度远高于 NLP 社区。
