# ArcAligner 论文阅读报告：被复杂化的 Context-Slot LoRA，以及对 QuRO 实验归因的警示

> **论文**：Jianbo Li, Yi Jiang, Sendong Zhao, Bairui Hu, Haochun Wang, Bing Qin, *ArcAligner: Adaptive Recursive Aligner for Compressed Context Embeddings in RAG*  
> **arXiv**：[2601.05038](https://arxiv.org/abs/2601.05038)  
> **论文标注代码地址**：[liunian-Jay/ArcAligner](https://github.com/liunian-Jay/ArcAligner)  
> **代码状态（阅读时）**：论文声称代码公开，但截至本次阅读没有可用的公开实现，因此本文只分析论文给出的公式、伪代码、训练设置和实验，不将作者代码或 checkpoint 视为可复用资源。  
> **阅读目的**：本文是固定阅读顺序 RRK → Perceiver IO → SeleCom → Tree Cross Attention → ArcAligner 的第五篇。重点不是复述全部 RAG 实验，而是回答：ArcAligner 相比 COCOM 究竟新增了什么、它是否真的进行了 query-conditioned readout、其复杂结构是否值得加入 QuRO，以及怎样避免把 decoder 微调收益误认为 readout 收益。

---

## 1. 核心结论

ArcAligner 并不是新的压缩器，也不是 QuRO 所需要的 query-conditioned readout。它假设文档已经被编码成一组 context slots，然后尝试让冻结 LLM 更好地读取这些非自然语言 embedding。

把论文的复杂表述还原后，核心只有两项：

1. **Masked / context-slot-only LoRA**：在 decoder 内加入 LoRA，但只保留 context-slot 位置上的 LoRA residual，普通 prompt、query 和生成 token 仍沿冻结模型路径传播；
2. **Gated recursive refinement**：在选定 Transformer 层内重复应用同一个 block，并用 slot-wise gate 决定每个 context slot 是否接受额外更新。

因此，最简洁的概括是：

$$
\boxed{
\text{ArcAligner}
=
\text{context-slot mask 下的 decoder LoRA}
+
\text{同层递归 gate}
}
$$

它真正讨论的是：

> 已经得到压缩 embedding 后，decoder 内部应怎样适配这些 embedding？

而 QuRO 的核心问题是：

> 给定 query，怎样从可离线缓存的完整压缩表示 $\mathbf Z_D$ 中读出当前任务所需的信息？

二者位于相邻阶段，但不是同一个问题。ArcAligner 对 QuRO 最重要的价值不是直接复用其结构，而是提醒我们警惕一个严重的实验混淆：**如果 readout 和 decoder LoRA 同时变化，最终提升可能只是 decoder 被训练得更会读取 soft embeddings，而不一定来自 query-conditioned readout。**

当前决定是：

- 不将 ArcAligner 加入 QuRO v0.0 主方法；
- 不在当前阶段复现其 masked LoRA、递归 block 和 STE gate；
- 生成器需要适配时，优先使用结构简单、代码公开、已有强证据支持的 COCOM 式标准 decoder LoRA；
- 将主要变量和实验预算留给 QuRO 的离线缓存、query-conditioned readout、多文档联合读取和重复查询摊薄。

---

## 2. ArcAligner 解决的不是“压缩”，而是“压缩后的可用性”

### 2.1 输入表示

给定 query $q$，检索器得到 passage。ArcAligner 先按句子切分文档，再使用冻结的 SFR-Embedding-Mistral 将每个句子编码为一个向量：

$$
\mathbf E\in\mathbb R^{m\times d_r},
$$

其中 $m$ 基本对应句子数。随后用轻量 projector 映射到生成器 hidden size：

$$
\widetilde{\mathbf E}
=W_\psi(\mathbf E)
\in\mathbb R^{m\times d}.
$$

因此它的“压缩”主要来自：

$$
\text{一句话的多个 token}
\longrightarrow
\text{一个 sentence embedding slot}.
$$

这并不是 learned Perceiver compressor，也不是 SeleCom 式 query-conditioned selector。论文主要创新发生在 $\widetilde{\mathbf E}$ 进入 decoder 以后。

### 2.2 Context-slot interface

论文在输入序列中预留 $m$ 个 context-slot 位置：

$$
r=\{r_1,\ldots,r_m\}.
$$

这些位置不用普通词表 embedding，而由投影后的句向量替换：

$$
H_i^{(0,0)}=
\begin{cases}
\widetilde E_j,&i=r_j,\\
\operatorname{Emb}(u_i),&\text{otherwise}.
\end{cases}
$$

附录给出的 prompt 形式是：

```text
Refer to the background document:
[B] [B] ... [B]

Question: [Q]
```

每个 `[B]` 只是占位符，其输入 embedding 会被相应的 sentence embedding 替换。这与软提示或 `inputs_embeds` 接口相似。

---

## 3. Selective LoRA：复杂公式的本质是一个 mask

论文将第 $\ell$ 个冻结 Transformer block 记为 $F_\theta^{(\ell)}$，LoRA 产生的增量记为 $\Delta F_\phi^{(\ell)}$。定义 context-slot mask：

$$
M_r\in\{0,1\}^{n\times1},
\qquad
(M_r)_i=\mathbb I[i\in r].
$$

ArcAligner layer 写成：

$$
\mathcal A^{(\ell)}(H)
=F_\theta^{(\ell)}(H)
+M_r\odot\Delta F_\phi^{(\ell)}(H).
$$

拆开看就是：

$$
H_i'=
\begin{cases}
F_\theta(H)_i+\Delta F_\phi(H)_i,&i\in r,\\
F_\theta(H)_i,&i\notin r.
\end{cases}
$$

所以它不是一种全新的“深层语义对齐器”，而是把 LoRA residual 的输出范围缩小到 context slots：

```python
base_output = frozen_block(hidden_states)
lora_delta = lora_branch(hidden_states)
output = base_output + context_slot_mask * lora_delta
```

论文将其解释为把 context embeddings 视作一种特殊模态，让 LoRA 只负责将这种模态对齐到 LLM hidden space。这个动机合理，但从机制上看，新增操作主要就是 mask。

### 3.1 它仍然属于 decoder 微调

必须准确描述：ArcAligner 并没有保持 decoder 完全不变。

它采用：

$$
\text{frozen base decoder weights}
+
\text{trainable decoder LoRA}.
$$

因此，“只训练外部模块，不训练 decoder”是错误的。LoRA 本身就是 parameter-efficient decoder adaptation。ArcAligner 的区别只在于：LoRA residual 不作用于所有 token，而只作用于 context slots。

### 3.2 论文没有充分说明的实现细节

论文公式在整个 Transformer block 输出层面定义 $\Delta F_\phi(H)$，但没有明确：

- LoRA 究竟插入 attention 的 $q/k/v/o$ 哪些投影；
- 是否也插入 MLP 的 gate/up/down projection；
- mask 是施加在每个 LoRA linear 的输出上，还是施加在整个 block 的 LoRA-induced residual 上；
- 递归时是否复用全部 KV 计算；
- 怎样在 batch 内对不同 slot 的二值 gate 做真正的稀疏 dispatch。

在没有代码的情况下，这些选择会导致明显不同的实现与计算量，降低了论文的可复现性。

---

## 4. Adaptive Recursive Alignment

### 4.1 每层先进行一次强制更新

所有 token 在每一层至少正常传播一次：

$$
H^{(\ell+1,0)}
=\mathcal A^{(\ell)}\left(H^{(\ell,0)}\right).
$$

### 4.2 Gate 为每个 slot 决定是否继续

在带 gate 的层中，第 $t$ 次递归前，对每个 context slot 预测一个标量：

$$
g^{(\ell,t)}
=\sigma\left(
\operatorname{MLP}^{(\ell)}
\left(H_r^{(\ell+1,t-1)}\right)
\right),
$$

其中：

$$
g^{(\ell,t)}\in[0,1]^{m\times1}.
$$

随后再次应用同一个第 $\ell$ 层：

$$
\widetilde H^{(\ell+1,t)}
=\mathcal A^{(\ell)}
\left(H^{(\ell+1,t-1)}\right).
$$

context slots 根据 gate 接受或拒绝候选更新：

$$
H_r^{(\ell+1,t)}
=H_r^{(\ell+1,t-1)}
+g^{(\ell,t)}\odot
\left(
\widetilde H_r^{(\ell+1,t)}
-H_r^{(\ell+1,t-1)}
\right).
$$

普通 token 在额外循环中保持第一次传播的结果：

$$
H_{\bar r}^{(\ell+1,t)}
=H_{\bar r}^{(\ell+1,0)}.
$$

直观上，ArcAligner 允许某个 context slot 在同一个 Transformer block 中接受一至三次迭代变换，而不是所有 token 一起加深网络。

### 4.3 STE 二值门控

训练时使用：

$$
g_{\mathrm{hard}}^{(\ell,t)}
=\mathbb I[g^{(\ell,t)}\ge0.5],
$$

$$
g_{\mathrm{STE}}^{(\ell,t)}
=g^{(\ell,t)}+
\operatorname{stopgrad}
\left(g_{\mathrm{hard}}^{(\ell,t)}-g^{(\ell,t)}\right).
$$

前向使用严格的 0/1，反向近似使用 sigmoid gate 的梯度；推理时直接按 0.5 阈值二值化。

论文设置最大循环次数 $T=3$，并在实验配置中将所有 Transformer 层设为 loop layers。

### 4.4 “条件计算”并没有被充分证明

按照论文算法，每次递归都先计算：

$$
\widetilde H=\mathcal A^{(\ell)}(H),
$$

然后才用 gate 决定是否接受候选表示。如果实际实现也这样执行，那么即使某个 gate 为零，主要的 Transformer 计算已经发生。

因此：

$$
\text{conditional update}
\neq
\text{conditional computation}.
$$

要获得真正的计算节省，需要额外实现 active-slot packing、动态 batch dispatch 或专用 slot-only recurrent block。论文既没有提供代码，也没有报告 TTFT、GFLOPs、吞吐量或端到端 speedup，所以不能从现有证据确认 gate 真正降低了计算成本。

此外，训练目标中没有显式的计算惩罚，例如：

$$
\mathcal L
=\mathcal L_{\mathrm{task}}
+\lambda\sum_{\ell,t,j}g_j^{(\ell,t)}.
$$

因此 gate 学到的是哪些额外更新有利于降低 NLL，而不是严格预算约束下的最优计算分配。它也不是 QuRO 所讨论的、由 query 难度或 overflow 风险决定的离散输出预算 $B$。

---

## 5. 三阶段训练

### 5.1 Stage I：重建式 alignment pretraining

作者冻结基础 LLM 和 sentence encoder，训练 projector 与 selective LoRA，使模型从 compressed slots 重建或复述原文：

$$
\mathcal L_{\mathrm{rec}}
=-\sum_{i=1}^{|y|}
\log p(y_i\mid H^{(0,0)},y_{<i}).
$$

数据来自约 20 万条 Wikipedia passages，训练一轮。论文采用多种 paraphrase 指令，而不只是机械复述，这一点有助于减少模型对原始表面词序的依赖。

### 5.2 Stage II：关闭 gate 的 RAG finetuning

从 Stage I 初始化后，在 HotpotQA 上训练约 90K 样本：

- 继续更新 projector 和 selective LoRA；
- gate 关闭；
- 每层只进行一次 mandatory refinement；
- 目标为 ground-truth answer NLL。

先稳定训练 decoder 读取 compressed slots，再引入离散 gate，这个课程式训练顺序是合理的。

### 5.3 Stage III：gate-aware finetuning

最后启用 gate，并联合训练：

$$
\{W_\psi,\Phi_{\mathrm{LoRA}},\Theta_{\mathrm{gate}}\}.
$$

论文主要配置为：

| 项目 | 设置 |
|---|---:|
| Backbone | Mistral-7B-Instruct-v0.2 |
| Sentence encoder | SFR-Embedding-Mistral（冻结） |
| LoRA rank | 128 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Max loops | 3 |
| Loop layers | all |
| Stage I learning rate | $2\times10^{-4}$ |
| Stage II/III learning rate | $2\times10^{-5}$ |
| Stage I samples | 约 200K |
| Stage II/III samples | 约 90K HotpotQA |

值得注意的是，论文只使用 NLL，没有额外的 gate 稀疏正则、预算约束或知识蒸馏目标。

---

## 6. 与 COCOM 的关系：ArcAligner 没有首次提出 decoder alignment

这是本次阅读中最重要的修正。

COCOM 已经明确指出：context embeddings 与自然 token embeddings 分布不同，仅训练 compressor/projector、冻结 decoder 会限制效果，因此需要微调 decoder。COCOM 使用 parameter-efficient LoRA，并把 LoRA 加到所有线性层。

COCOM 全版本中：

$$
\phi_{\mathrm{comp}}=\theta_{\mathrm{LLM}},
$$

即 compressor 与 generator 使用同一个 Mistral，在不同前向中承担两个角色：

$$
D+\langle CTX\rangle
\xrightarrow{\text{Mistral+LoRA}}
\mathbf Z_D,
$$

$$
[\mathbf Z_D;q]
\xrightarrow{\text{same Mistral+LoRA}}
y.
$$

所以 COCOM 已经联合优化压缩端和生成端，并通过 decoder-tuning 消融证明了生成器适配的重要性。ArcAligner 不能把“让 decoder 学会读取压缩 embedding”作为新的基本贡献。

两者的准确区别是：

| 维度 | COCOM | ArcAligner |
|---|---|---|
| Compressor | Mistral 本身；COCOM-light 使用 BERT | 冻结 SFR-Embedding-Mistral |
| Compressor 是否训练 | 是 | 否 |
| Decoder base weights | 冻结 | 冻结 |
| Decoder adaptation | 所有线性层的标准 LoRA | 只保留 context-slot 位置的 LoRA residual |
| LoRA 影响的 token | context、prompt、query、生成 token | context slots |
| 额外递归 | 无 | 同层最多三次 |
| Gate | 无 | slot-wise binary gate |
| Query-conditioned compression/readout | 否 | 否 |
| 文档表示离线缓存 | 支持 | 支持 |

因此，ArcAligner 更准确的定位是：

$$
\boxed{
\text{COCOM 式 decoder adaptation 的局部化、递归化版本}
}
$$

而不是一种从零开始的新 generator-alignment 范式。

---

## 7. ArcAligner 不是 query-conditioned readout

论文把 context slots 放在 question 前面。对于 causal decoder：

$$
H_{[B]}
\not\leftarrow
H_{[Q]}.
$$

前面的 context slots 无法注意到后面的 query。因此 gate：

$$
g_j^{(\ell,t)}
=f\left(H_{r_j}^{(\ell,t)}\right)
$$

虽然在某个 QA 前向过程中计算，但它并没有显式读取 query。它至多根据 slot 本身的表示判断“是否难以对齐”，而不能判断“这个 slot 对当前 query 是否相关”。

所以更准确的名称是：

$$
\text{context-conditioned difficulty gate},
$$

而不是：

$$
\text{query-conditioned relevance gate}.
$$

这与 QuRO 有本质区别：

| 方法 | 在线阶段决定什么 | 是否显式依赖 query |
|---|---|---:|
| Perceiver IO decoder | output queries 从 latents 读出什么 | 是 |
| SeleCom | 从原文生成哪些 query-relevant soft tokens | 是 |
| Tree Cross Attention | 沿哪条树路径读取 | 是 |
| ArcAligner | 哪些 compressed slots 接受额外层内变换 | 按论文 prompt，否 |
| QuRO | 从可缓存 $\mathbf Z_D$ 中生成哪些 task-conditioned soft tokens | 是 |

ArcAligner 的 query 仍然可以通过普通 causal attention 读取前面的 context slots并生成答案，但这是 decoder 的常规条件生成，不等于 context representation 本身经过 query-conditioned readout。

---

## 8. 实验证据为什么不够充分

### 8.1 缺少最关键的公平消融

论文需要证明的是：

$$
\text{context-only LoRA + recursion + gate}
>
\text{ordinary decoder LoRA}.
$$

但它没有提供一个干净的：

$$
\text{same frozen encoder}
+
\text{same projector}
+
\text{standard full-token decoder LoRA}
$$

基线。

现有消融主要包括：

- `w/o Recursion`；
- `w/o Gate (Max Loop)`；
- `w/o LoRA & Recursion`。

理想的归因表应当是：

| 变体 | LoRA 作用范围 | Recursion | Gate |
|---|---|---:|---:|
| Projector only | 无 | 无 | 无 |
| Standard decoder LoRA | 所有 token | 无 | 无 |
| Context-only LoRA | context slots | 无 | 无 |
| Fixed recursion | context slots | 固定循环 | 无 |
| ArcAligner | context slots | 动态循环 | 有 |

缺少标准 decoder LoRA 后，无法判断提升究竟来自：

1. decoder 终于被微调了；
2. LoRA 被限制到 context slots；
3. 同一个 block 被重复调用；
4. gate 确实学到了有效的动态路径。

COCOM 已经证明第 1 项本身能够带来显著收益，因此这个缺失尤其关键。

### 8.2 Recursion 的增益并不大

完整 ArcAligner 相比 `w/o Recursion` 的准确率提升大致为：

| 数据集 | w/o Recursion Acc | ArcAligner Acc | 差值 |
|---|---:|---:|---:|
| HotpotQA | 32.60 | 33.40 | +0.80 |
| NaturalQA | 36.90 | 37.04 | +0.14 |
| TriviaQA* | 62.00 | 62.80 | +0.80 |

提升存在，但不足以说明增加递归执行、二值 gate、STE 和动态调度的工程复杂度一定值得。

### 8.3 Baseline 的压缩率不统一

主表中各方法的压缩率分别包括：

- xRAG：$128\times$；
- COCOM：$4\times$、$16\times$；
- LLMLingua-2：$3\times$；
- ArcAligner：$24\times$。

这些数字不能视为相同预算下的严格对比。论文也没有统一报告生成器实际接收的 context-token 数、存储量和在线计算量。

### 8.4 检索设置过于受限

作者先检索 top-20，再用 RankZephyr rerank，但最终只把 top-1 passage 交给模型。因此论文没有真正验证：

- top-$K$ 多文档证据融合；
- 不同文档之间的 slot budget 分配；
- 多跳问题所需的跨文档联合读取；
- 同一文档被多个 query 重复使用时的摊薄收益。

这与 QuRO 重点研究的 $K\times m$ 多文档 latent 联合读出并不等价。

### 8.5 缺少实际效率指标

论文没有充分报告：

- TTFT；
- 总推理延迟；
- GFLOPs；
- 峰值显存；
- 递归 block 的实际调用次数与 wall-clock speedup；
- gate 为零时是否真的跳过计算。

因此，现有实验更能说明“某些额外变换可以提高 QA 指标”，但不足以支持强的动态计算效率结论。

---

## 9. 对 QuRO 的直接影响：必须隔离 readout 与 decoder adaptation

QuRO 同样面临 compressed embeddings 能否被 decoder 理解的问题。若我们同时：

- 新增 query-conditioned readout；
- 新增 generator LoRA；
- 改变 prompt/query 的注入位置；
- 改变输出 soft-token 数量；

最终准确率提高后，就无法判断主要贡献来自哪一项。

因此应明确区分两种能力：

### 9.1 Readout 能力

$$
\mathbf E_q
=r_\omega
\left(
\text{prompt},q,
[\mathbf Z_{d_1};\ldots;\mathbf Z_{d_K}]
\right).
$$

它负责：

- 理解任务 prompt；
- 理解 query；
- 在多文档 cached latents 中选择和融合证据；
- 输出固定或离散自适应预算 $B$ 的 generator-conditionable soft tokens。

### 9.2 Decoder 适配能力

$$
y\sim g_\phi(\cdot\mid\mathbf E_q,\text{generation instruction}).
$$

它负责将 $\mathbf E_q$ 转化为正确输出。若使用 LoRA：

$$
g_{\phi+\Delta\phi_{\mathrm{LoRA}}},
$$

则 decoder 自身也在学习这种新 soft-token 接口。

两者可以联合训练，但实验上必须分别测量贡献。

---

## 10. Prompt 与 query 放入 readout 的当前设计

当前更合理的 QuRO 设计是让 readout 同时看到 task prompt 与 query，而不是只给一个很短的 query embedding：

$$
\mathbf Q_{1:B}
=f_{\mathrm{query}}
\left(
\text{prompt},q,
\mathbf P_{1:B}
\right),
$$

$$
\mathbf E_q
=\operatorname{CrossAttn}
\left(
\mathbf Q_{1:B},
\mathbf Z,
\mathbf Z
\right).
$$

其中：

$$
\mathbf Z=[\mathbf Z_{d_1};\ldots;\mathbf Z_{d_K}]
\in\mathbb R^{Km\times h}.
$$

这样 readout 不只是做关键词相关性匹配，而是可以根据完整任务指令决定：

- 需要事实检索、比较、摘要还是多跳组合；
- 哪些文档和 slots 与任务有关；
- 应该输出哪些类型的 generator-side representation；
- 当前 query 是否需要更大的离散预算 $B$。

如果训练时同时加入 decoder LoRA，那么：

$$
\mathcal L_{\mathrm{SKD/NLL}}
\rightarrow
\text{decoder LoRA}
\rightarrow
\mathbf E_q
\rightarrow
\text{readout},
$$

readout 与 decoder 会共同学习 soft-token 通信协议。此时已经能够实现 COCOM 所强调的“让 decoder 学会读取压缩表示”，没有必要再增加 ArcAligner 的 masked residual、递归 block 和 STE gate。

若冻结 decoder 后效果已经足够，则结果更能直接证明 readout 的表示质量；若冻结 decoder 不足，再加入标准 COCOM 式 LoRA 是更简单、可复现且变量更少的方案。

---

## 11. QuRO 应采用的归因实验

最小但足够清晰的设计是：

| Offline representation | Readout | Generator adaptation | 目的 |
|---|---|---|---|
| $\mathbf Z_D$ | 无 | Frozen | 直接读取缓存表示的下界 |
| $\mathbf Z_D$ | 无 | Standard LoRA | 测 decoder adaptation 本身的收益 |
| $\mathbf Z_D$ | Query-agnostic readout | Standard LoRA | 控制新增参数量与二次压缩 |
| $\mathbf Z_D$ | Prompt+query-conditioned readout | Frozen | 最严格地验证 readout 表示质量 |
| $\mathbf Z_D$ | Prompt+query-conditioned readout | Standard LoRA | 完整 QuRO 系统 |
| 原始文档 | 无 | 同规模 LoRA | 未压缩、同训练预算强上界 |

其中最关键的差值分别是：

### Query conditioning 的贡献

$$
\Delta_{\mathrm{query}}
=
\operatorname{Score}
(\text{prompt+query readout})
-
\operatorname{Score}
(\text{query-agnostic readout}).
$$

### Decoder LoRA 的贡献

$$
\Delta_{\mathrm{LoRA}}
=
\operatorname{Score}
(\text{readout+LoRA})
-
\operatorname{Score}
(\text{readout+frozen decoder}).
$$

### 压缩/任务训练混淆

$$
\operatorname{Score}
(\text{compressed+LoRA})
\quad\text{必须与}\quad
\operatorname{Score}
(\text{full context+same LoRA/data})
$$

比较。否则压缩模型的收益可能只是来自额外 QA 微调。

此外继续保留 mismatch-query 对照：如果替换 query 后 readout 输出和答案几乎不变，说明所谓 query-conditioned readout 可能退化成 query-agnostic 二次压缩。

---

## 12. 为什么当前不实现 ArcAligner

### 12.1 没有可用代码和 checkpoint

论文虽然提供 GitHub 地址并声称公开代码，但本次阅读时没有可用实现。复现需要自行决定 LoRA 插入位置、mask 粒度、递归 KV 处理和动态 slot dispatch，结果很难保证与论文一致。

### 12.2 引入的变量过多

若加入 ArcAligner，需要同时选择：

- LoRA 作用层；
- LoRA rank；
- context-only mask 的位置；
- gated layers；
- 最大循环深度 $T$；
- gate MLP 结构；
- STE 阈值；
- 是否添加 compute penalty；
- 是否实现真正的稀疏执行。

这些变量会与 QuRO 自身的 readout 深度、output query 结构、预算 $B$、离线压缩率、generator LoRA 和蒸馏目标交叉，严重扩大实验矩阵。

### 12.3 当前证据不足以证明复杂度值得

论文中 recursion 相对无 recursion 的提升较小；缺少标准 decoder LoRA 基线；真实效率没有被充分报告。因此，在 QuRO 的当前阶段，为这套结构支付实现与消融成本不划算。

### 12.4 更简单的替代已经存在

如果 frozen generator 无法读取 QuRO 输出，直接采用 COCOM 式 standard decoder LoRA 即可：

$$
\boxed{
\text{prompt+query-conditioned readout}
+
\text{standard generator LoRA}
}
$$

它具备以下优势：

- 代码和 checkpoint 生态更成熟；
- 容易接入现有 PEFT 训练；
- 不需要修改 Transformer block 的 forward；
- 不引入递归和离散 gate；
- 更容易公平比较 frozen decoder 与 adapted decoder；
- 将论文的主要方法变量保留在 readout，而不是 generator 内部。

---

## 13. 对论文创新性的最终评价

ArcAligner 提出了一个合理但被包装得较复杂的局部改造：

$$
\text{global decoder LoRA}
\longrightarrow
\text{context-slot-only LoRA}
\longrightarrow
\text{gated repeated block application}.
$$

它的优点是：

- 明确关注 compressed embeddings 的 decoder-side usability；
- 尝试减少 LoRA 对普通语言 token 路径的直接扰动；
- 使用三阶段训练降低离散 gate 的优化难度；
- 提醒后续工作不能把“信息被压进去了”和“decoder 能用它”视为同一件事。

但其局限也很明显：

- decoder alignment 已由 COCOM 等工作明确提出；
- selective LoRA 的机制本质是 mask；
- gate 不显式读取 query；
- 条件更新不一定转化为条件计算；
- 缺少普通 decoder LoRA 这一关键公平基线；
- recursion 增益较小；
- 各 baseline 压缩率不统一；
- top-1 retrieval 不能证明多文档联合能力；
- 缺少真实效率评测；
- 代码和 checkpoint 不可用。

因此，对 QuRO 的最终判定是：

| 用途 | 决定 |
|---|---|
| 当前主架构组件 | 不采用 |
| v0.0 实现 | 不加入 |
| 必跑同构 baseline | 不自行复现；优先使用可复现的 COCOM/PISCO |
| Related Work | 保留，作为 decoder-side localized alignment 方法 |
| 消融启发 | 保留 frozen decoder vs standard LoRA |
| 未来扩展 | 仅在标准 LoRA 明确造成通用能力退化时，再考虑 context-only adaptation |

---

## 14. 对 QuRO 主线的最终贡献

ArcAligner 没有改变 QuRO 的主架构，但帮助我们进一步收紧了研究问题：

$$
\boxed{
\mathbf Z_D=f_{\mathrm{off}}(D)
\quad\text{离线计算并缓存}
}
$$

$$
\boxed{
\mathbf E_q=
r_{\mathrm{online}}
(\text{prompt},q,\mathbf Z_D)
\quad\text{在线进行 query-conditioned readout}
}
$$

$$
\boxed{
y\sim g
(\mathbf E_q),
\quad
g\text{ 可冻结或使用标准 LoRA，但其贡献必须单独报告}
}
$$

一句话概括二者差异：

> **ArcAligner decides how deeply a compressed slot is adapted inside the generator; QuRO decides what task-relevant information is read from reusable compressed memories before generation.**

QuRO 当前应坚持：离线 compression 负责保留宽松、可复用的完整语义；在线 readout 同时读取 prompt、query 与多文档 cached latents；decoder 适配只作为独立可控变量。这样才能避免把 generator LoRA 的收益误写成 query-conditioned readout 的收益，并保持方法结构、实验归因和论文叙事足够清晰。

