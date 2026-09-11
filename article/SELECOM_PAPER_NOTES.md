# SeleCom 论文阅读报告：从 Query-conditioned Selector 到可复用的 Query-conditioned Readout

> 论文：*Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective*  
> 作者：Yunhao Liu, Zian Jia, Xinyu Gao, Kanjun Xu, Yun Xiong  
> 会议：WWW 2026  
> arXiv：2602.15856  
> 官方代码：[yhliu7458/SeleCom](https://github.com/yhliu7458/SeleCom)  
> 训练数据：[Ryan7458/SeleCom_Training](https://huggingface.co/datasets/Ryan7458/SeleCom_Training)  
> 阅读目的：明确 SeleCom 与 QuRO 的创新边界，并确定其公开数据在 QuRO 训练和评测中的复用方式。

---

## 1. 核心结论

SeleCom 已经系统提出并验证了 **query-conditioned soft compression**：压缩器不再试图保留整篇文档，而是读取 query 与原文，只选择回答当前 query 所必需的信息。因此，QuRO 不能再把“引入 query”“选择 query 相关内容”或“用 QA 目标替代文档重构”作为主要创新。

QuRO 相对 SeleCom 真正可辩护的贡献是：

> **将 query-independent document representation 与 query-conditioned information selection 在时间上解耦：文档只在离线阶段编码一次并缓存；查询到来后，轻量 readout 在不重新访问原文的条件下，从可复用表示中选择并生成固定预算的软 token。**

形式化地说，SeleCom 直接学习：

$$
\mathbf E_{q,d}=f_{\mathrm{sel}}(q,d),
$$

每个新 query 都要重新处理原文 $d$。QuRO 将该映射分解为：

$$
\mathbf Z_d=f_{\mathrm{off}}(d),\qquad
\mathbf E_{q,d}=r_{\mathrm{online}}(q,\mathbf Z_d),
$$

其中 $\mathbf Z_d$ 可离线计算、落盘并跨 query 复用。我们的核心问题不是“能否做 query-conditioned selection”，而是：

> **Query conditioning 是否必须发生在原文编码阶段，还是可以推迟到查询时，在预计算的宽松文档表示上完成？**

---

## 2. SeleCom 的问题定义

### 2.1 对 full-compression paradigm 的批评

传统软压缩方法学习：

$$
\mathbf Z=f_{\mathrm{enc}}(d),\qquad |\mathbf Z|\ll |d|,
$$

并通过重构等目标，试图让少量向量保存文档的全部语义。SeleCom 将这一范式归纳为两个问题。

1. **Infeasibility（不可兼容性）**：问题并不是数学意义上绝对无法压缩，而是高倍率全文重构压力会与 LLM 的指令遵循、问题理解和下游生成行为冲突。论文观察到，全文压缩得到的 embedding 会吸引过强的生成器注意力，使生成器忽略 query 或额外指令。
2. **Non-necessity（非必要性）**：对于给定 query，原文中真正有用的通常只是局部证据。强迫 encoder 保存所有内容，会把有限表示容量浪费在无关信息上，降低答案相关信息的密度。

因此，SeleCom 将 encoder 的角色由 full compressor 改写为 query-conditioned selector，只保留回答当前 query 所需的信息。

### 2.2 SeleCom 的核心假设

SeleCom 的核心假设可以写成：

$$
I(d;q)\ll I(d),
$$

即面向特定 query 所需的文档信息量远小于文档总信息量。既然高倍率压缩无法无损保存一切，就应优先保存当前 query 相关的证据，而不是平均保存整篇文档。

这一判断对 QuRO 同样重要，但它只证明了 **query-conditioned selection 的必要性**，没有解决该选择过程能否建立在 **可复用、预计算的文档表示** 上。

---

## 3. SeleCom 方法

### 3.1 整体结构

SeleCom 由三部分组成：

1. **Selector**：同时读取 query 与文档，提取 query 相关信息；
2. **Projector**：将 selector 输出映射到 generator 的 token embedding 空间；
3. **Generator**：将压缩 embedding 与 query 一起用于最终回答生成。

对于每篇文档 $d_i$，SeleCom 构造：

$$
[\text{instruction};q;d_i;
\underbrace{\texttt{<ENCODE>},\ldots,\texttt{<ENCODE>}}_{p}],
$$

并以 Qwen3-Embedding-0.6B 作为 decoder-only selector。最后 $p$ 个 `<ENCODE>` 位置的隐藏状态构成 selector 输出：

$$
\mathbf H^{i}_{1:p}
=
\operatorname{Select}([\mathbf T_i;\langle En\rangle_p])
[\langle En\rangle_p].
$$

随后将这些隐藏状态拼接，经一层 MLP 投影，再切分成 $n$ 个生成器侧软 token：

$$
\mathbf E_i
=
\operatorname{Split}_{n}
\left(
\operatorname{Proj}
\left(
\operatorname{Concat}(\mathbf H^{i}_{1:p})
\right)
\right).
$$

论文默认 $p=8,n=2$，即每篇文档最终被表示为两个 generator soft tokens。

论文将 selector 描述为 autoregressive，准确地说，它使用的是因果 decoder-only backbone；`<ENCODE>` token 已作为输入的一部分，并不是像答案生成一样逐个采样。因此 QuRO 可以强调浅层并行 cross-attention 不需要对原文执行完整 selector，但不应不严谨地把 SeleCom 描述成“逐 token 生成八个 embedding”。

### 3.2 Stage 1：训练 Selector 与 Projector

Stage 1 使用合成的 $(q_i,d_i,a_i)$ 三元组。Selector 和 projector 可训练，generator 冻结，通过答案的 next-token prediction loss 更新：

$$
\mathcal L
=
-\sum_t
\log P_{θ}
\left(
a_{i,t}\mid \mathbf E_i,q_i,a_{i,<t}
\right).
$$

该阶段不要求重构文档，而是直接让压缩表示服务于 QA。这既提高了任务相关性，也意味着“QA-oriented compression objective”本身已经不能作为 QuRO 的独立创新。

### 3.3 Stage 2：训练 Generator

Stage 2 冻结 selector 和 projector，在多个公开 QA 数据集上使用 LoRA 训练 generator，使其学会利用插入 token embedding 空间的压缩向量。官方实现使用 LoRA rank 64。

论文报告的主要设置为：

| 项目 | Stage 1 | Stage 2 |
|---|---:|---:|
| 训练轮数 | 1 epoch | 3 epochs |
| 每卡 batch size | 10 | 3 |
| 学习率 | $5\times10^{-5}$ | $10^{-4}$ |
| 可训练模块 | Selector、projector、特殊 token embedding | Generator LoRA |
| 硬件 | 8× RTX 4090 48GB | 8× RTX 4090 48GB |
| 报告用时 | 约 40 h | 约 12 h |

---

## 4. 数据构造与公开数据

### 4.1 Stage 1：14M 合成文档 QA

SeleCom 从约 3300 万篇 Wikipedia 文档开始，依次执行：

1. 基于长度过滤过长和过短文档；
2. 用 LLM-as-a-judge 排除代码、表格、HTML 等非自然文本；
3. 用 LLM-as-a-scorer 对信息密度打 1–10 分，剔除低于 6 分的文档；
4. 生成简单事实问题与需要推理的困难问题；
5. 验证答案是否被文档支持；
6. 将问题难度分成 1–5，剔除过易的 1 和过难的 5，保留 2、3、4；
7. 使用难度递增的 curriculum learning 训练 selector。

所有 LLM 角色均使用 Qwen3-30B-A3B-Instruct-2507。公开 Stage 1 数据统计如下：

| 属性 | 数值 |
|---|---:|
| 样本数 | 14,004,010 |
| 下载大小 | 约 7.28 GB |
| 解码后大小 | 约 11.33 GB |
| Parquet shards | 23 |
| 字段 | `question`, `answer`, `document`, `difficulty` |

论文还对 1000 个合成样本进行了人工验证，报告 document quality、factuality、supportability 和 difficulty label 等指标，用于支持数据质量。

### 4.2 Stage 2：公开 QA 训练数据

公开 Stage 2 数据统计如下：

| 属性 | 数值 |
|---|---:|
| 样本数 | 868,152 |
| 下载大小 | 约 1.96 GB |
| 解码后大小 | 约 3.34 GB |
| Parquet shards | 7 |
| 字段 | `question`, `documents`, `answer` |

其中 `documents` 是字符串列表，因此特别适合训练 QuRO 的多文档联合 readout。

### 4.3 下游评测

SeleCom 在六个知识密集任务上评测：

- Natural Questions；
- TriviaQA；
- WebQuestions；
- PopQA；
- HotpotQA；
- FactKG。

主要指标为 EM、F1 和 Qwen3-30B-A3B-Instruct LLM-as-a-judge；FactKG 只报告 Accuracy。官方仓库另外发布了对应的 Eval_QDA 文件。当前说明中，HotpotQA 文件主要面向 top-1 设置；单文档实验使用 `documents[0]`。

---

## 5. 实验结果与论文价值

SeleCom 的重要结果不是某个单独数字，而是它展示了三个现象：

1. 在极高压缩率下，query-conditioned selection 明显优于 query-agnostic full compression；
2. 在多个任务上，SeleCom 能接近甚至超过相同训练数据适配的未压缩 RAG；
3. top-5 设置下，在 HotpotQA 等多跳任务中，过滤无关信息可能比向 generator 填入全部原文更有效。

论文默认约 82× 压缩率，在 NQ、TriviaQA、WebQuestions、PopQA、HotpotQA、FactKG 上普遍优于 ICAE、xRAG、COCOM、PISCO 等软压缩基线。相对未压缩 RAG，论文报告计算量和延迟降低 33.8%–84.6%。

这些结果为 QuRO 提供了一个非常强的前提：**query conditioning 确实有价值**。但 SeleCom 每次 query 都必须重新处理原始文档，因此它同时构成 QuRO 的质量上限与主要效率对手。

---

## 6. SeleCom 与 QuRO 的创新边界

| 维度 | SeleCom | QuRO |
|---|---|---|
| Query 是否参与 | 是 | 是 |
| Query 参与位置 | 原文 selector 阶段 | 缓存 latent 的在线 readout 阶段 |
| 每次 query 是否读取原文 | 是 | 否 |
| 文档表示能否跨 query 复用 | 否 | 是 |
| 离线阶段 | 无可复用压缩结果 | 预计算并存储 $\mathbf Z_d$ |
| 在线输入 | $q+d$ | $q+\mathbf Z_d$ |
| 多文档机制 | 每篇文档分别产生固定数量 token | 可在 $km$ 个 latent 上进行全局联合读出 |
| 最终预算 | 每篇文档固定 $n$ 个 | 全局预算 $B$，可扩展为 query-dependent |
| 系统目标 | 单次 query 的选择质量和压缩效率 | 多 query 负载下的质量、延迟、存储与复用权衡 |

### 6.1 不能再声明的创新

以下表述已被 SeleCom 覆盖，QuRO 不应再使用：

- 首次提出 query-conditioned soft compression；
- 使用 query 选择文档中的相关内容；
- 将 encoder 从 compressor 改造成 selector；
- 通过选择性压缩缓解信息稀释；
- 使用 QA objective 代替 reconstruction objective。

### 6.2 QuRO 的主创新：时间分解

现有方法面对一个 **cacheability–specificity dilemma**：

- query-agnostic compressor 可离线缓存，但不知道未来会问什么，必须尽量保存所有信息；
- SeleCom 等 query-conditioned selector 能针对问题保留证据，但每次查询都要重新读取原文，无法复用。

QuRO 的贡献不是在两端选择其一，而是把两种属性放在不同时间阶段：

1. 离线阶段做 query-independent、相对宽松的文档表示；
2. 在线阶段在 query 已知后进行轻量、激进的信息缩减；
3. 在线 readout 不重新访问 source tokens。

推荐的论文表述为：

> Existing soft compressors face a cacheability-specificity dilemma: query-agnostic representations are reusable but must preserve information for unknown future queries, whereas query-conditioned selectors such as SeleCom identify relevant evidence only by re-encoding the source document for every query. QuRO factorizes these two roles across time, combining a reusable query-independent document memory with a lightweight query-conditioned readout that never revisits the source tokens.

### 6.3 全局多文档读出

对于 $k$ 篇文档，每篇离线保存 $m$ 个 latent：

$$
\mathbf Z=[\mathbf Z_{d_1};\ldots;\mathbf Z_{d_k}]
\in\mathbb R^{km\times h}.
$$

QuRO 使用 query 构造 output query array，并一次性读出：

$$
\mathbf E
=
\operatorname{Readout}(q,\mathbf Z)
\in\mathbb R^{B\times h}.
$$

SeleCom 默认给每篇文档固定分配 $n$ 个 token；QuRO 的 $B$ 是所有召回文档共享的全局预算。相关文档可获得更大注意力，无关文档可接近零贡献，多跳问题则可以联合融合多篇文档。这一差异应主要在 HotpotQA 及带噪 top-$k$ 检索中验证。

---

## 7. 最终预算 $B$ 与 Overflow

### 7.1 $B$ 的定义

$B$ 是 readout 最终产生并送入 generator 的 soft-token 数，而不是 batch size、文档数或离线 latent 数。若召回 $k=5$ 篇文档，每篇缓存 $m=32$ 个 latent，则 readout 可以访问 $km=160$ 个 latent；若 $B=16$，则：

$$
160\text{ cached latents}
\xrightarrow{\text{query-conditioned readout}}
16\text{ generator soft tokens}.
$$

对应三类需要严格区分的量：

| 符号 | 含义 |
|---|---|
| $m$ | 每篇文档离线保存的 latent 数 |
| $km$ | 当前 query 可访问的缓存 latent 总量 |
| $B$ | generator 最终接收的 soft-token 数 |

生成器侧有效压缩率应统一定义为：

$$
\xi_{\mathrm{eff}}
=
\frac{\sum_{i=1}^{k}|d_i|}{B}.
$$

主实验必须按 $B$ 或 $\xi_{\mathrm{eff}}$ 对齐，不能用离线压缩率代替最终预算。与 SeleCom top-5、$n=2$ 比较时，一个直接公平的设置是 QuRO 也使用 $B=10$。

### 7.2 两类 overflow

必须区分：

1. **Offline-representation overflow**：答案证据已经在 $d\rightarrow\mathbf Z_d$ 阶段丢失。增加 $B$ 无法恢复；只能降低离线压缩率、增加 $m$ 或改善离线 compressor。
2. **Readout-budget overflow**：证据仍存在于 $\mathbf Z$ 中，但 $B$ 太小，无法充分选出或表达。增加 $B$ 可能恢复答案；自适应预算主要处理这一类问题。

因此，“overflow 风险在离散档位间变化”更准确的说法是：

> 根据当前 $(q,\mathbf Z)$ 在不同输出预算下发生 readout-budget overflow 的预测风险，从若干离散预算档位中选择 $B$。

例如：

$$
B\in\{4,8,16,32\}.
$$

训练时可分别运行各档预算，定义能够正确回答问题的最小档位：

$$
B^*
=
\min\{B:\operatorname{Correct}(q,\mathbf Z,B)=1\},
$$

再训练轻量 router 预测 $p(B^*\mid q,\mathbf Z)$。离散档位有利于 batching、稳定训练和公平比较。

动态预算属于扩展贡献。主实验首先固定 $B$，证明 query-conditioned readout 有效；随后再比较固定预算、query-difficulty router 和 overflow-risk router，避免把主要变量混在一起。

---

## 8. 核心亮点：重复查询下的缓存复用与成本摊薄

### 8.1 为什么这是 QuRO 的主贡献

SeleCom 对每个 query 都计算：

$$
(q,d)\rightarrow f_{\mathrm{selector}}(q,d).
$$

同一篇文档被询问 $Q$ 次，selector 就要处理 $Q$ 次原文。QuRO 只在离线阶段计算一次：

$$
d\rightarrow\mathbf Z_d,
$$

之后每个 query 只执行：

$$
(q,\mathbf Z_d)\rightarrow r_{\mathrm{readout}}(q,\mathbf Z_d).
$$

二者累计成本为：

$$
C_{\mathrm{SeleCom}}(Q)
=Q\,C_{\mathrm{selector}},
$$

$$
C_{\mathrm{QuRO}}(Q)
=C_{\mathrm{offline}}
+Q\,C_{\mathrm{readout}}.
$$

交叉点为：

$$
Q^*
=
\frac{C_{\mathrm{offline}}}
{C_{\mathrm{selector}}-C_{\mathrm{readout}}}.
$$

当 $Q>Q^*$ 时，QuRO 的累计计算成本低于 SeleCom。QuRO 因而不是只优化“单次请求延迟”，而是在优化共享知识库中的长期 RAG serving cost。

在完成实测前，不能声称 $Q^*$ 一定为个位数；论文必须报告真实测量得到的交叉点。

### 8.2 该实验应作为主图

控制实验横轴设置为：

$$
Q\in\{1,2,4,8,16,32,64\},
$$

表示每篇文档平均被多少个 query 使用。纵轴至少报告：

- 累计 GPU 时间；
- 累计 GFLOPs；
- amortized cost per query；
- TTFT；
- TIL；
- QuRO 的额外缓存存储量。

比较方法至少包括：

- 未压缩 RAG；
- SeleCom；
- query-agnostic direct compression；
- QuRO。

公平性要求：

1. 固定相同 query、检索结果和生成器；
2. 锁定 generator 侧最终 token 数 $B$；
3. 分别报告 compression subsystem cost 与 end-to-end cost；
4. QuRO 的冷启动结果计入 $C_{\mathrm{offline}}$；
5. 热缓存结果直接复用 $\mathbf Z_d$；
6. 同时报出索引构建/更新成本与存储开销。

除“同一文档重复 $Q$ 次”的可控实验外，还应模拟更真实的 Zipf 访问分布：少量热门文档被高频访问，大量长尾文档只访问一两次。报告 cache-hit rate 与 amortized cost/query 的关系，可以把 QuRO 从单纯的压缩模块提升为面向真实 RAG serving 的系统方案。

### 8.3 预期论文结论的正确写法

在完成实验后，应使用类似以下可验证表述：

> QuRO incurs a one-time offline encoding cost but replaces SeleCom's per-query source-document encoding with a lightweight readout over cached memories. The measured crossover point $Q^*$ shows how many document reuses are required for this upfront cost to amortize.

如果质量与 SeleCom 接近且 $Q^*$ 较小，即使 QuRO 未在每个数据集上超过 SeleCom，论文故事仍然成立：它用可控的存储成本换取了显著下降的长期在线计算成本。

---

## 9. SeleCom 数据如何用于 QuRO

### 9.1 Phase 0：构建离线表示

对 Stage 1 文档计算：

$$
\mathbf Z_d=f_{\mathrm{off}}(d).
$$

主实验应优先冻结已有 query-agnostic compressor，预计算并缓存 $\mathbf Z_d$。训练 readout 时直接加载缓存，不应在每个 epoch 重新编码原文；否则虽然数学路径相似，却无法验证预计算表示上的训练和服务优势。

### 9.2 Phase 1：使用 14M 数据训练 readout

将每条 $(q,d,a)$ 转换为 $(q,\mathbf Z_d,a)$：

$$
\mathbf E=r_{\theta}(q,\mathbf Z_d),\qquad
a\sim g(\mathbf E,q).
$$

推荐首先冻结 offline compressor 和 generator，只训练：

- query encoder；
- cross-attention readout；
- projector；
- output/budget query embeddings。

监督仍采用答案 next-token loss。这样与 SeleCom 使用完全相同的数据和目标，唯一区别是信息路径：

$$
\text{SeleCom}: q+d\rightarrow\mathbf E,
$$

$$
\text{QuRO}: d\rightarrow\mathbf Z_d,\quad
q+\mathbf Z_d\rightarrow\mathbf E.
$$

这构成最直接、最公平的对照。

### 9.3 Phase 2：使用 868K 数据适配 generator

Stage 2 的 `documents` 列表适合构造：

$$
\mathbf Z=[\mathbf Z_{d_1};\ldots;\mathbf Z_{d_k}],
$$

并训练多文档全局 readout。至少报告两种训练设置：

| 设置 | Offline compressor | Readout | Generator |
|---|---|---|---|
| 对齐 SeleCom | 冻结 | 冻结 | LoRA |
| QuRO 完整版 | 冻结 | 继续训练 | LoRA |

第一种隔离 generator adaptation 的作用，第二种测量联合优化上限。

### 9.4 需要警惕的问题

1. **Stage 1 不足以自动保证表示可跨 query 复用。** 如果每篇文档只对应一个问题，而 offline encoder 与 readout 一起接受 QA loss，query-independent encoder 可能只学会保存该训练问题所需的信息。主实验应冻结已有 compressor；若从头训练 Perceiver encoder，应为同一文档聚合多个 query，或加入 reconstruction/teacher-distillation 辅助目标。
2. **Curriculum learning 需要自行核验。** 公布的数据包含 `difficulty`，但当前公开 `SelectTrainDataset` 只是普通加载，collator 也未使用该字段；Hugging Face Trainer 默认随机采样。不能假设官方脚本自动实现 curriculum。可显式采用 difficulty 2 → 2+3 → 2+3+4 的阶段训练，并做随机混合对照。
3. **防止数据泄漏。** Stage 2 来自公开 QA 训练集，必须保存 benchmark provenance，检查与 Eval_QDA 的 exact/near duplicate，并区分 in-domain adaptation 与 held-out transfer。

---

## 10. 必做对照实验

在锁定相同 generator soft-token budget $B$ 的条件下，建议主表包含：

| 方法 | 是否重新读原文 | Query-conditioned | 可缓存 | 输出 |
|---|---:|---:|---:|---:|
| SeleCom | 是 | 是 | 否 | $k\times n$ tokens |
| Query-agnostic direct | 否 | 否 | 是 | $B$ tokens |
| Similarity top-$B$ | 否 | 非参数 | 是 | $B$ slots |
| QuRO | 否 | 可学习 | 是 | $B$ tokens |
| QuRO oracle | 否 | Gold evidence | 是 | $B$ tokens |

必须回答：

1. 同样 $B$ 下，QuRO 是否优于 query-agnostic direct compression？
2. QuRO 是否优于简单相似度 top-$B$？
3. QuRO 与 SeleCom 的质量差距是多少？
4. Offline representation 在多大 $\xi_{\mathrm{off}}$ 下尚未发生严重 overflow？
5. 每篇文档平均被查询多少次后，QuRO 的累计成本低于 SeleCom，即实测 $Q^*$ 是多少？
6. 全局 $B$ 是否优于 SeleCom 式 per-document fixed allocation，尤其在带噪 top-$k$ 和多跳任务中？

其中第 5 项应是主实验和摘要级结论，而不是附录中的普通效率消融。

---

## 11. 对 QuRO 贡献结构的最终建议

贡献优先级建议调整为：

1. **Temporal factorization**：将 query-independent encoding 与 query-conditioned selection 分配到离线和在线两个时间阶段；
2. **Reusable query-conditioned generation path**：在不重新访问原文的情况下，从预计算表示中产生 generator-consumable soft tokens；
3. **Repeated-query amortization**：系统刻画缓存复用、存储开销、在线成本与实测交叉点 $Q^*$；
4. **Global multi-document readout**：使用全局 $B$ 在多个召回文档之间竞争和融合信息；
5. **Adaptive budget（扩展）**：根据 query difficulty 或 readout-overflow risk 从离散档位中选择 $B$。

前 3 项共同构成论文的主故事；第 4 项提供多文档方法优势；第 5 项应在主方法成立后作为增强模块，不应在早期抢占核心叙事。

最终定位可以概括为：

> SeleCom proves that high-ratio soft compression should be query-conditioned. QuRO asks a different systems and modeling question: can query-conditioned selection be delayed until query time while the expensive document representation is computed only once? By reading from reusable precomputed memories instead of re-encoding source documents, QuRO targets the quality of selective compression and the amortized efficiency required by repeated-query RAG workloads simultaneously.

---

## 12. 当前判断

SeleCom 没有把 QuRO 的方向做掉，但它迫使我们放弃宽泛的“query-conditioned compression”新颖性声明。QuRO 能否成立，取决于以下两个可证伪前提：

1. 宽松的离线表示在目标压缩率下尚未系统性丢失 query 所需证据；
2. 缓存 latent 上的轻量 readout 能接近 SeleCom 的选择质量，同时在文档重复使用后显著降低累计成本。

因此最优先的 go/no-go 实验不是动态 $B$，而是：

1. 扫描 $\xi_{\mathrm{off}}$ 与 offline overflow；
2. 在 NQ/HotpotQA 上比较 query-agnostic、similarity top-$B$、QuRO 与 SeleCom；
3. 测量冷启动、热缓存和 $Q^*$ 摊薄曲线。

只要 QuRO 能在可接受的质量差距内获得明确、可复现的重复查询收益，其“可缓存性与 query 特异性的时间解耦”就构成一个完整且有现实意义的论文贡献。
