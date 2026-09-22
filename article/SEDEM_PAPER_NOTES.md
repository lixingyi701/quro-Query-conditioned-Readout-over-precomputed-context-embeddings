# SeDeM 论文阅读报告：从压缩记忆到选择性解压与中间层注入

> 论文：*SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering*  
> 作者：Maryam Haghifam, Jason Cong, Yizhou Sun  
> 单位：University of California, Los Angeles  
> arXiv：[2608.00311](https://arxiv.org/abs/2608.00311)，本文按 2026-09-10 的 v2 分析  
> 会议：EMNLP 2026 Main Conference  
> 阅读目的：明确 SeDeM 相对 ICAE、SeleCom 与当前 QuRO 实现的贡献边界，并判断其对“可缓存全量压缩 + query-conditioned readout”路线的影响。

---

## 1. 核心结论

SeDeM 不只是“ICAE 加一个 selector”。它改变了压缩表示被生成器使用的方式：

> **压缩 memory 只负责紧凑存储；query 到来后先选择相关 memory block，再将其展开为 decoder 中间层可消费的 hidden states。**

其基本链路为：

$$
D
\rightarrow M_D
\xrightarrow[]{Q,\ \mathrm{TopK}}
M_{D,Q}^{\mathrm{selected}}
\rightarrow \widehat H_{D,Q}
\rightarrow \text{decoder intermediate layer}
\rightarrow A.
$$

它解决的是 ICAE 类方法中的角色冲突：少量 memory token 既要保存文档，又要直接充当生成器 soft prefix。SeDeM 将“存储表示”和“消费表示”解耦，用 decompressor 把选中的紧凑 memory 恢复到更接近正常 decoder activation 的表示空间。

但必须同时看到：

1. 主实验只有 **4× 存储压缩**；
2. 每个被选中的 32-slot block 会重新展开成 128 个 decoder positions；
3. query 主要决定“选哪个 block”，并不细粒度地改变 block 的解压内容；
4. selector 使用 block-level evidence supervision；
5. 论文的核心模块需要自行进行两阶段训练，不能像当前 QuRO 一样直接把冻结 PISCO latent 接到一个 readout 上。

因此，SeDeM 更准确的定位是：

> **基于软 memory 的 retrieve-and-reconstruct 系统，而不是极短 soft-token budget 下的 dense query-conditioned readout。**

---

## 2. 一个必须澄清的训练事实

### 2.1 不是完整重训 encoder 和 decoder 主干

不能笼统写成“SeDeM 把 encoder 和 decoder 全部重新训练”。论文明确说明：

- encoder backbone 全程冻结；
- decoder backbone 全程冻结；
- Stage 1 自行训练 compressor 的输入投影和 decompressor；
- Stage 2 继续训练 compressor/decompressor，并训练 selector 与 decoder LoRA。

准确表述应为：

> **SeDeM 使用 Llama-3.2-1B/3B Base 构建自己的 encoder-compressor-selector-decompressor-decoder 链路；基础 encoder/decoder 权重冻结，但压缩器、解压器、选择器和 decoder LoRA 均由作者重新训练。它不是直接复用 ICAE、PISCO 或 COCOM 的压缩 checkpoint。**

### 2.2 各模块的训练状态

| 模块 | Stage 1 | Stage 2 | 说明 |
|---|---|---|---|
| Llama encoder backbone | 冻结 | 冻结 | 提取中间层逐 token hidden states |
| Compressor $W_{\mathrm{in}}$ | 训练 | 继续训练 | mean pooling 本身无参数，线性投影可训练 |
| Decompressor | 训练 | 继续训练 | 两层 MLP，将一个 memory slot 展开为 $C$ 个 hidden states |
| Top-K selector | 不启用 | 训练 | 多头投影 + ColBERT-style late interaction |
| Decoder backbone | 冻结 | 冻结 | 不做 full fine-tuning |
| Decoder LoRA | 不启用 | 训练 | rank 64，插入每层的 Q/K/V/O projection |

这与当前 QuRO 的控制思路不同。QuRO 固定 PISCO/COCOM compressor，主要训练在线 readout 和可选 decoder LoRA；SeDeM 则需要先学会自己的 hidden-state compression/decompression，再做下游 QA 适配。

---

## 3. 相对 ICAE 的关键变化

ICAE 的基本链路是：

$$
D\rightarrow Z_D\rightarrow \text{Generator},
$$

即让压缩 memory slots 同时承担：

1. 保存文档信息；
2. 直接作为生成器输入。

SeDeM 改为：

$$
D\rightarrow M_D
\xrightarrow[]{Q}\mathrm{Select}(M_D)
\rightarrow\mathrm{Decompress}
\rightarrow \widehat H^{(\ell_{\mathrm{inj}})}.
$$

| 环节 | ICAE | SeDeM |
|---|---|---|
| 压缩表示来源 | 特殊 memory tokens 经过 LLM self-attention | 中间层逐 token hidden states |
| 压缩方式 | LLM 内部全局聚合 | 相邻 hidden states 的局部 mean pooling + projection |
| query 是否参与离线压缩 | 否 | 否 |
| query 的在线作用 | 与 memory 一起交给生成器 | 给 memory blocks 排序并选择 Top-K |
| generator 消费形式 | 直接读取紧凑 memory tokens | 读取解压后的中间层 hidden states |
| 注入位置 | 输入层 soft prefix | decoder 中间层 |
| 是否可缓存 | 可以 | 可以 |
| 是否显式保留局部地址 | 弱 | 强：每个 slot 对应连续 token window |

最实质的创新不是 selector，而是：

> **compact storage representation → decoder-compatible reconstructed representation。**

---

## 4. 方法细节

### 4.1 中间层 hidden-state compressor

文档首先被切成长度为 $T$ 的 segments。冻结 LLM encoder 只运行到抽取层 $\ell_{\mathrm{ext}}$：

$$
H_s^{(\ell_{\mathrm{ext}})}
=
\mathrm{LLMEnc}_{\ell_{\mathrm{ext}}}(X_s)
\in\mathbb R^{T\times d_{\mathrm{enc}}}.
$$

论文主设置为：

$$
T=128,\qquad C=4,\qquad N_M=T/C=32.
$$

每连续 $C=4$ 个 hidden states 做 mean pooling：

$$
\bar h_{s,j}
=
\frac{1}{C}\sum_{k=1}^{C}
H_s[(j-1)C+k].
$$

之后用共享投影映射到 decoder hidden dimension：

$$
m_{s,j}=\bar h_{s,j}W_{\mathrm{in}}.
$$

一个 128-token segment 因而存成 32 个 memory slots，即 4× storage compression。

这一设计故意保留局部性：第 $j$ 个 memory slot 始终来自一个明确的连续 token window。论文的 learnable-query compressor 消融表明，全局 attention 虽然更灵活，却会把信息分散到不同 slots，使 Top-K block selection 更困难。

### 4.2 Query-conditioned Top-K selector

所有 segment memories 形成全局 memory bank。主设置下，一个 block 对应一个 128-token segment。

query 使用同一个 encoder 并运行到相同的 $\ell_{\mathrm{ext}}$。selector 使用 $R=4$ 个 learned scoring heads，将 query tokens 和 memory slots 投影到共享空间：

$$
\alpha_{r,q,n,b}
=
\cos\left(W_Q^{(r)}h_q, W_K^{(r)}m_{n,b}\right).
$$

随后采用 ColBERT-style MaxSim：

$$
s_n
=
\frac{1}{R}
\sum_r\sum_q\max_b\alpha_{r,q,n,b}.
$$

最后选择得分最高的 $K$ 个 blocks。

需要注意：这不是一个比 QuRO cross-attention 更复杂的 query readout。它本质是 learned projection、cosine MaxSim 和 hard Top-K。SeDeM 的系统复杂度主要来自 compressor/decompressor、训练目标和中间层注入，而不是 selector 本身。

### 4.3 Selective decompressor

每个被选 memory vector 经过共享两层 MLP：

$$
a_{s,j}
=
\mathrm{GELU}(\mathrm{LN}(m_{s,j})W_1+b_1),
$$

$$
z_{s,j}=a_{s,j}W_2+b_2,
\qquad z_{s,j}\in\mathbb R^{Cd_{\mathrm{dec}}}.
$$

然后 reshape 为：

$$
\widehat H_{s,j}\in\mathbb R^{C\times d_{\mathrm{dec}}}.
$$

因此：

$$
32\text{ memory slots}
\rightarrow
128\text{ reconstructed hidden states}.
$$

选择 $K$ 个 blocks 后，decoder 上层实际处理约 $128K$ 个重建位置，而不是 $32K$ 个压缩 slots。

### 4.4 Decoder 中间层注入

decoder 先只处理 query-side prefix 到注入层 $\ell_{\mathrm{inj}}$，然后把重建状态插入：

$$
[\widehat H_{\mathcal I(Q)}; H_Q^{(\ell_{\mathrm{inj}})}].
$$

之后仅运行 $\ell_{\mathrm{inj}}+1$ 到最后一层。

这避免了让人工 soft tokens 从输入层穿过完整 decoder，也使重建目标可以直接对齐到 decoder 内部 activation distribution。但它需要修改 LLM forward、attention mask、position handling 和中间层数据流，工程侵入性明显高于 QuRO 的标准 `inputs_embeds` 接口。

---

## 5. 两阶段训练

### 5.1 Stage 1：重建预训练

Stage 1 在 300M SlimPajama tokens 上训练 5 epochs，只更新 compressor projection 和 decompressor。目标为：

$$
\mathcal L^{(1)}
=
\mathcal L_{\mathrm{ctx}}
+\lambda_{\mathrm{distill}}\mathcal L_{\mathrm{distill}}
+\lambda_{\mathrm{rec}}^{(1)}\mathcal L_{\mathrm{rec}}.
$$

其中：

- $\mathcal L_{\mathrm{ctx}}$：重建状态条件下的 next-token loss；
- $\mathcal L_{\mathrm{distill}}$：raw-context frozen decoder 到 reconstructed-state decoder 的 KL distillation；
- $\mathcal L_{\mathrm{rec}}$：逐位置和局部 pooled hidden states 的 cosine alignment。

这一步是 SeDeM 相对当前 QuRO 增加的主要训练成本。我们的 PISCO latent 已有训练好的 generator interface；SeDeM 必须自己学会如何把局部 memory 恢复成 decoder-compatible states。

### 5.2 Stage 2：有监督选择与 QA 训练

Stage 2 初始化 Stage 1 的 compressor/decompressor，随机初始化 selector，并加入 decoder LoRA：

$$
\mathcal L^{(2)}
=
\mathcal L_{\mathrm{LM}}
+\lambda_{\mathrm{ret}}\mathcal L_{\mathrm{ret}}
+\lambda_{\mathrm{rec}}^{(2)}\mathcal L_{\mathrm{rec}}.
$$

selector 的监督来自 block-level evidence labels，$\mathcal L_{\mathrm{ret}}$ 由 InfoNCE 和 pairwise margin loss 组成。

训练最多 3 epochs/dataset。作者使用：

- compressor 和 decoder LoRA 学习率：$10^{-4}$；
- selector projections 学习率：$5\times10^{-4}$；
- LoRA rank：64；
- LoRA scaling：128；
- decoder 每层 Q/K/V/O projections 均插入 LoRA。

---

## 6. 与 SeleCom 的区别

SeleCom 与 SeDeM 都反对不加区分的全文压缩，但 query 进入系统的时间不同。

| 维度 | SeleCom | SeDeM |
|---|---|---|
| 文档压缩是否依赖 query | 是 | 否 |
| query 的作用 | 参与原文信息选择和软压缩 | 只对已经缓存的 blocks 排序 |
| 新 query 是否重读原文 | 是 | 否 |
| 输出给 generator 的形式 | 少量 query-specific soft embeddings | Top-K blocks 解压后的长 hidden-state sequence |
| query 条件化粒度 | 细粒度内容选择 | 主要是 block-level routing |
| 多 query 缓存复用 | 弱 | 架构上支持 |

SeleCom 学习：

$$
E_{Q,D}=f(Q,D).
$$

SeDeM 学习：

$$
M_D=f_{\mathrm{off}}(D),
\qquad
\widehat H_{Q,D}
=
g(\mathrm{TopK}(Q,M_D)).
$$

SeDeM 更适合重复查询，但 query 条件化较粗；SeleCom 每次重新读取原文，代价更大，但可直接形成 query-specific embedding。

---

## 7. 与当前 QuRO 实现的对比

当前仓库中至少有三类相关 readout：

1. **QuRO-C/xattn**：将全部 PISCO latents 压缩成固定 $B$ 个 query-specific soft tokens；
2. **QuRO-R**：latent-as-Q 读取 query，再做 latent self-attention，输出 $E=Z+\Delta(Q,Z)$；
3. **QuRO-RQ**：query-as-Q 读取文档，再通过 $A^\top$ 写回原 latents，输出 identity residual。

| 方法 | 离线表示 | 在线 query 操作 | token 数变化 | generator 接口 |
|---|---|---|---:|---|
| QuRO-C | PISCO global latents | output slots cross-attend query 与全部 latents | $Km\rightarrow B$ | input `inputs_embeds` |
| QuRO-R | PISCO global latents | latent 读 query + latent self-attention | $Km\rightarrow Km$ | input `inputs_embeds` |
| QuRO-RQ | PISCO global latents | query 读 latent，再 $A^\top$ 写回 | $Km\rightarrow Km$ | input `inputs_embeds` |
| SeDeM | 局部 pooled intermediate states | ColBERT-style Top-K blocks | $32K\rightarrow128K$ | decoder 中间层注入 |

可以将四者概括为：

- QuRO-C：**read and compress**；
- QuRO-R/RQ：**read and rewrite**；
- SeDeM：**retrieve and reconstruct**。

### 7.1 SeDeM 在 QuRO 单个 readout 之外额外实现了什么

若在当前仓库中复现 SeDeM，至少还要增加：

1. 中间层逐 token hidden-state extractor；
2. 固定长度 segmentation；
3. 局部 pooling 和 block-aware cache protocol；
4. memory slot 到原文窗口的映射；
5. query 同层编码；
6. block-level late-interaction selector；
7. 离散 Top-K；
8. $d\rightarrow Cd$ decompressor；
9. decoder 中间层 injection hook；
10. attention mask 和 position handling；
11. hidden-state reconstruction target/cache；
12. Stage 1 reconstruction + KD trainer；
13. Stage 2 evidence-ranking loss；
14. encoder/decoder 跨模型投影。

因此 SeDeM 不是替换 `src/readout.py` 中一个模块，而是重新实现从 memory writer 到 decoder internal interface 的完整系统。

### 7.2 不能直接把 PISCO latent 接上 SeDeM decompressor

PISCO latent 是全局混合、generator-ready 的 soft prompt memory；SeDeM memory slot 则绑定连续局部窗口。SeDeM 的 decompressor 默认存在：

$$
m_{s,j}\leftrightarrow C\text{ 个相邻 token states}.
$$

PISCO latent 没有这一一对应关系。因此若要融合两者，需要重新定义解压目标，而不是简单添加一个 $d\rightarrow Cd$ MLP。

---

## 8. 实验结果

### 8.1 主结果

SeDeM 在 Llama-3.2-1B 和 3B 设置下均超过主表中的 ICAE、HMT、500xCompressor、Activation Beacon 与 LongLLMLingua。

3B setting 的 F1：

| 方法 | 2Wiki | MuSiQue | QASPER | HotpotQA-Dist. |
|---|---:|---:|---:|---:|
| Full-context fine-tuned | 62.50 | 32.54 | 23.44 | 36.73 |
| ICAE | 38.77 | 14.00 | 20.21 | 37.99 |
| 500xCompressor | 52.14 | 6.98 | 13.73 | 39.15 |
| Activation Beacon | 25.31 | 19.36 | 23.16 | 45.09 |
| **SeDeM** | **67.25** | 21.85 | **26.74** | **58.30** |

SeDeM 在 2Wiki、QASPER 和 HotpotQA 上超过 full-context fine-tuning，但在 MuSiQue 上明显落后，说明 Top-K selected evidence 对某些多跳问题覆盖不足。

### 8.2 Decompression 消融

| 方法 | QASPER F1 | HotpotQA F1 |
|---|---:|---:|
| Direct memory conditioning，linear compressor | 18.06 | 29.69 |
| Direct memory conditioning，MLP compressor | 18.30 | 30.13 |
| **SeDeM decompression + intermediate injection** | **26.74** | **49.39** |

更强的 memory writer 不能补偿取消 decompression 带来的损失。这是全文最有力的证据：提升确实与 decoder-compatible reconstruction 有关，而不仅是模块参数更多。

### 8.3 与 ComprExIT 的匹配比较

v2 新增了最近邻 ComprExIT 对照。结果不是 SeDeM 全面胜出：

- learned-$K=2$ SeDeM 在 HotpotQA 和 2Wiki 上分别比最佳 ComprExIT 低 4.57 和 6.05 F1；
- full-bank SeDeM 在两个数据集上高于 ComprExIT；
- SeDeM 的 TTFT 更低，表现为明显的质量-效率工作点差异。

这说明 selection 是效率来源，但 tight Top-K 同时带来质量损失；full-bank 更能体现 compressor-decompressor 本身的质量。

### 8.4 Selection 与 compression 的控制实验

v2 对 v1 的主要补强是加入控制实验：

- 相同 gold-selected input 下，SeDeM 64.95 F1，ICAE 60.03；
- matched raw-text RAG 在 HotpotQA $K=2$ 时达到 66.31 F1，高于 SeDeM 58.30；
- answer-string distant supervision 达到 51.78 F1，接近 gold-supervised learned-$K=2$ 的 50.93；
- boundary shift 只造成约 0.81–2.14 F1 下降。

因此应谨慎表述：

> SeDeM 证明了 compression-decompression pathway 优于直接 memory conditioning；但它没有证明自己在答案质量上优于读取 Top-K 原文的 RAG。它主张的是可复用存储和在线效率之间的折中。

### 8.5 长度扩展

RULER qa_2 上：

| 方法 | 4K | 8K | 16K |
|---|---:|---:|---:|
| Full context | 49.4 | 48.8 | 49.4 |
| SeDeM | 31.6 | 27.2 | 22.6 |
| ICAE | 11.2 | 13.0 | 7.6 |

SeDeM 始终高于 ICAE，但自身质量随长度明显下降，不能声称 length-invariant。

---

## 9. 效率与显卡资源

### 9.1 论文明确报告的硬件

训练和 baseline reproduction 的共享环境为：

- PyTorch 2.5.1；
- ROCm 6.2；
- AMD MI300X、MI325X、MI250X、MI210 GPU 集群；
- bf16 mixed precision；
- FlashAttention-2 或 SDPA。

论文致谢中说明 AMD 提供了部分实验计算资源。

### 9.2 显卡数量：必须如实标注

**论文没有报告 SeDeM Stage 1 或 Stage 2 每次训练使用多少张 AMD GPU，也没有报告总 GPU-hours。** 型号列表不能被解释为“使用了 4 张卡”，因为 MI300X/MI325X/MI250X/MI210 是四种硬件型号，不是设备数量。

论文唯一明确给出数量的硬件设置是效率测试：

| 实验 | GPU 数量 | GPU 型号 | 其他设置 |
|---|---:|---|---|
| SeDeM vs ICAE TTFT/吞吐 | **1 张** | NVIDIA A100-SXM4-40GB | 独占，batch size 1，bf16 |
| SeDeM vs ComprExIT TTFT | **1 张** | NVIDIA A100-SXM4-40GB | 独占，batch size 1，bf16，context length 1536 |
| Stage 1/Stage 2 训练 | **未披露** | AMD MI300X/MI325X/MI250X/MI210 | 无法从论文推断卡数 |

因此，复现成本只能确认：Stage 1 使用 300M tokens、训练 5 epochs，Stage 2 每个数据集最多 3 epochs；不能确认需要几张卡或训练多长时间。

### 9.3 延迟结果

相对 ICAE，SeDeM 报告：

- 1B：平均 TTFT 降低 1.74×，decode throughput 提升 1.08×；
- 3B：平均 TTFT 降低 2.46×，decode throughput 提升 1.10×。

需要注意，TTFT 采用 fully-online protocol，包含 segmentation、encoder extraction、compression、selection、decompression 和 decoder prefill；论文没有实际评估文档 memory 预计算后的多 query 摊销收益。

---

## 10. 对论文贡献强度的判断

### 10.1 真正成立的贡献

1. 将 compact storage 与 decoder conditioning 解耦；
2. 提出 selective decompression + intermediate-layer injection；
3. 用局部 block structure 保持 memory 的可选择性；
4. 通过两阶段 reconstruction/distillation 训练 decoder-compatible states；
5. 支持 1B encoder → 3B decoder 的 cross-model memory use；
6. v2 通过 identical-input、full-bank 和 raw-text RAG 控制，部分分离了 selection 与 compression 的贡献。

### 10.2 较弱或并非新颖的部分

- mean pooling；
- 两层 MLP expansion；
- ColBERT-style MaxSim；
- hard Top-K；
- evidence-supervised retrieval；
- decoder LoRA；
- 两阶段训练和 KD。

这些组件单独都不新，论文的价值在于围绕“存储/消费解耦”形成完整系统。

### 10.3 仍然存在的问题

1. 没有与 SeleCom、PISCO、COCOM 做主表比较；
2. 主表使用 evidence supervision，而多项 baseline 没有同等监督；
3. 4× storage compression 较温和；
4. decoder-side 长度是约 $128K$，在线收益主要来自 Top-K selection，而非让生成器直接消费极少 soft tokens；
5. RULER 只到 16K，且质量随长度下降；
6. 只研究 QA/retrieval-oriented tasks，尚未覆盖 summarization；
7. 宣称 memory 可跨 query 复用，但没有真正构造“一文多问”实验，也没有报告摊销交叉点；
8. 训练 GPU 数量、GPU-hours 和 wall-clock time 未披露。

---

## 11. 对 QuRO 的直接影响

SeDeM 已经占据了以下宽泛主张：

> query-independent reusable memory + query-conditioned online use。

因此 QuRO 不能再声称首次提出“缓存文档表示，再根据 query 读取”。QuRO 应将贡献收紧到：

1. 对全量 PISCO/COCOM latents 做 dense query-conditioned transformation，而不是 Top-K block routing；
2. 不要求 memory slot 与原文局部窗口一一对应；
3. 不依赖 gold evidence labels；
4. 输出保持极短 budget，例如 8 个 soft tokens，而不是解压为 $128K$ 个 positions；
5. 专门评估同一文档对应多个 query 的 cache amortization、query diversity 和 information coverage；
6. 比较 `read and compress`、`read and rewrite` 与 `retrieve and reconstruct` 三条路线。

建议将 SeDeM 加入 QuRO 的核心近邻与实验对照：

| 对照 | 要回答的问题 |
|---|---|
| PISCO direct | query-conditioned transformation 是否有增益 |
| QuRO-C | 极短 dense readout 是否能维持质量 |
| QuRO-R/RQ | 保留全部 PISCO latents 的 query-conditioned refinement 是否有效 |
| SeDeM-style Top-K | 硬 block routing 是否已经足够 |
| SeDeM-style select-and-expand | 增加 decoder positions 能换回多少质量 |
| Raw-text RAG | soft memory 的效率收益是否值得质量损失 |

---

## 12. 最终评价

SeDeM 的系统工程量明显大于 QuRO 的一个 cross-attention readout，但它并没有提出一个更复杂的 query-conditioned attention。它真正完成的是：

> **重新训练局部 hidden-state compressor/decompressor，建立可寻址 memory bank，用 query 做 hard routing，并把选中的 compressed blocks 恢复成 decoder 中间层 activation。**

因此两者的本质区别是：

$$
\boxed{\text{QuRO：稠密 query-conditioned read/rewrite，保持短预算}}
$$

对比

$$
\boxed{\text{SeDeM：硬选择 relevant blocks，再恢复较长 hidden sequence}}
$$

SeDeM 证明了“压缩表示不必直接充当生成器输入”，也证明了 decoder-compatible decompression 值得研究；但它没有解决极短 soft-token budget 下如何从全量缓存表示中融合分散证据，也没有真正研究同一文档多 query 的缓存收益。这两点仍然是 QuRO 可以重点推进的空间。
