# Cache-to-Cache 论文阅读报告：跨模型 KV 融合、隐藏状态与软压缩，以及对 QuRO 的启发

> **论文**：Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang. *Cache-to-Cache: Direct Semantic Communication Between Large Language Models*.  
> **版本**：ICLR 2026；arXiv:2510.03215v2，2026-03-02。  
> **原文**：[arXiv](https://arxiv.org/abs/2510.03215v2) · [PDF](https://arxiv.org/pdf/2510.03215v2)  
> **作者标注的代码地址**：[thu-nics/C2C](https://github.com/thu-nics/C2C)。本报告依据论文，不宣称已审查或复现作者代码。  
> **阅读日期**：2026-09-19。  
> **阅读目的**：理解跨模型 KV cache 通信的实现和实证；重点厘清最后一层隐藏状态、压缩 soft tokens 与多层 KV cache 的联系和区别，并提炼对 QuRO 接收端适配的启发。  
> **范围**：QuRO 部分整理本次讨论中的方法抽象与待验证假设，不构成对仓库最新实现或实验结果的审计。

## 1. 核心结论

C2C 让两个模型分别理解输入，再把 Sharer 的 KV cache 经学习变换后融合进 Receiver 的 KV cache，由 Receiver 生成答案。它省掉了 Sharer 逐 token 生成中间分析文字的过程。

对 QuRO 最重要的结论有五条：

1. **C2C 主要研究连续表示通信和融合，不以缩短上下文表示序列为目标。** 默认融合保持 Receiver 的 cache 长度，没有完成我们关心的文档 token 数 $n\to m$ 压缩。
2. **外来表示有用，不等于直接替换接收端表示就有效。** Table 8 的纯投影替换明显失败；结合双方表示并保留 Receiver 原 cache 后，效果大幅改善。
3. **完整 KV cache 不是最后一层隐藏状态的另一种格式。** 它包含各 attention 层各自的 K/V，来自不同深度的隐藏状态。
4. **我们的 soft tokens 最终也会形成生成器的 KV cache。** 区别是输入端注入后由生成器计算 KV，还是直接在各层 cache 接口注入。
5. **残差是值得借鉴的设计原则，不能单独作为 QuRO 的核心创新。** QuRO 需要证明紧凑、可复用的文档表示上，query 条件化读出带来额外收益。

C2C 没有提供等存储预算、等训练预算下与我们这类软压缩读出的直接对照，因此不能据此断言 KV 通信优于 soft tokens。

## 2. 方法：从文本交接改为 cache 融合

### 2.1 两种通信路径

传统 Text-to-Text（T2T）：

1. Sharer 读取问题，生成分析文字。
2. Receiver 读取原问题和分析文字。
3. Receiver 生成回答。

Cache-to-Cache（C2C）：

1. Sharer 和 Receiver 分别对输入执行 prefill，产生自己的 KV cache。
2. 学习模块对齐、投影和融合两者的 cache。
3. Receiver 使用融合后的上下文 cache 生成回答；后续生成 token 的 cache 由 Receiver 自身产生。

**Receiver 也处理原输入，并不是仅凭 Sharer 的外来 cache 作答。** 论文 Figure 1 的一般场景区分共享 context 与后续 query；主问答实验中，两个模型对任务输入形成表示。不能把这一设置自动等同于 QuRO 的 query 无关文档缓存。

### 2.2 Token 与层对齐

不同模型的 tokenizer、层数和 KV 维度可能不同。正文 §3.3.3、附录 A.1.1–A.1.2 给出：

- **模板部分**：通过 padding 对齐不同 chat template 的长度。
- **消息部分**：将 Receiver token 解码成字符串，再用 Sharer tokenizer 编码；一对多时，默认选择字符串覆盖最长的候选 token。
- **层对齐**：采用 terminal alignment，从输出侧逐层匹配，即最后一层对最后一层，再向浅层对齐。
- **替代方案**：归一化层深度后配对；论文称末端对齐实现简单且实验略优。

Token 对齐属于启发式近似，不能视为任意 tokenizer 间的无损语义映射。

### 2.3 残差融合与门控

将正文 Eq. (3) 中包含在 fuser 内的门控显式写出，可概括为：

$$
\widetilde C_R^{(\ell)}
=
C_R^{(\ell)}
+
g_\ell F_\ell\left(C_R^{(\ell)},C_S^{(G(\ell))}\right).
$$

其中 $G(\ell)$ 是层映射，$F_\ell$ 生成修正，$g_\ell$ 控制该层是否接收注入。

默认 fuser 包含：

1. 拼接 Receiver 和 Sharer 表示，进行投影与特征融合。
2. 输入相关的 head 调制，对投影信息动态加权。
3. 每层可学习门：训练中使用温度退火的 Gumbel-sigmoid，推理中进行二值选择。
4. 保留 Receiver 原始 cache，将融合结果残差加回。

需要区分：**输入相关的动态 head 权重**与**可学习的逐层门**不是同一个模块，也不应笼统称为 query-conditioned readout。

### 2.4 训练

正文 §3.3.4：

- 冻结 Sharer 和 Receiver 参数；
- 只训练 C2C 融合模块；
- 用融合 cache 条件下的 Receiver 回答进行 next-token prediction 监督。

$$
\mathcal L_{\mathrm{C2C}}
=
-\sum_t\log p_R(y_t\mid y_{<t},\widetilde C_R(X)).
$$

冻结 Receiver 参数不等于阻断其计算图：答案损失仍需通过 Receiver 的计算传播到融合模块。

主实验使用 OpenHermes2.5 前 500k 条样本。部分 scaling、behavior 分析使用 MMLU 训练；LongBench 实验使用该基准的不同训练/测试集合。不同表格不能默认为同一训练设置。

### 2.5 三层 MLP 与 C2C-C：避免混淆

论文有三个不同对象：

| 对象 | 作用 |
|---|---|
| Cache transformation oracle | 用三层 MLP 与 MSE 对齐源、目标最后层 KV，观察表示空间可转换性 |
| 默认 C2C | 双方 cache 拼接、投影、融合、动态权重与残差门控 |
| C2C-C | 先用三层 MLP 将 Sharer cache 映射到 Receiver 维度，再进行双方融合 |

不能把默认方法简化成“一个三层 MLP 将所有 KV 转过去”。

## 3. 两个 oracle：为什么作者认为这条路线可行？

### 3.1 Cache enrichment oracle（§3.2.1，Table 1）

设示例为 $E$，问题为 $X$：

- Direct：只处理 $X$，使用 $C(X)$。
- Few-shot：处理 $E\oplus X$，使用全部 cache。
- Oracle：处理 $E\oplus X$，但丢弃示例位置，仅保留问题位置的 cache。

$$
C^*(X)=C_{[|E|:|E|+|X|]}(E\oplus X).
$$

| 设置 | 解码时 cache 长度 | 准确率 |
|---|---:|---:|
| Direct | $|X|$ | 58.42 |
| Few-shot | $|E|+|X|$ | 63.39 |
| Oracle | $|X|$ | 62.34 |

**支持的结论**：即便解码时不保留示例 token 的 cache，示例对问题表示的影响仍可改善回答。

**边界**：示例依然参与了 prefill；该实验不证明零成本获得额外信息，也不证明任意短表示足以无损承载长文档。

作者还发现层间差异明显，增强某些层可能损害效果，由此引入层选择门。

### 3.2 Cache transformation oracle（§3.2.2，附录 A.3.2）

以 Qwen3-4B 与 Qwen3-0.6B 为例，训练映射器后，t-SNE 中源表示落入目标表示分布区域。

这是可转换性的探索证据；**二维可视化重叠并不能单独证明语义完整保留、因果等价或回答能力迁移**。后续任务准确率才提供更直接的行为证据。

同理，effective rank 增加是表示变化的诊断，不等价于严格的信息量增长，更不能直接证明保留了答案所需信息。

## 4. 实验结果与解释边界

### 4.1 主结果：小模型间也能获得融合收益（Table 4）

Receiver 固定为 Qwen3-0.6B，以下为 Sharer = Qwen2.5-0.5B 的结果，准确率单位为 %：

| 测试集 | Receiver | T2T | C2C |
|---|---:|---:|---:|
| MMLU-Redux | 35.53 | 41.03 | 42.92 |
| OpenBookQA | 39.20 | 44.00 | 52.60 |
| ARC-C | 41.04 | 49.48 | 54.52 |
| C-Eval | 32.04 | 35.88 | 41.77 |
| 四项平均（按表中数值计算） | 36.95 | 42.60 | 47.95 |

该配置相对 Receiver 提升约 **11.00 个百分点**，相对 T2T 提升约 **5.36 个百分点**。

Table 4 的三类 Sharer 对应的平均准确率差值：

| Sharer | C2C − Receiver | C2C − T2T |
|---|---:|---:|
| Qwen2.5-0.5B | +11.00 pp | +5.36 pp |
| Llama3.2-1B | +9.64 pp | +4.15 pp |
| Qwen3-4B-Base | +11.88 pp | +3.06 pp |

Base 模型直接按指令回答表现很差时，C2C 仍可能利用其内部表示。这表明“能否按指定格式输出答案”与“表示能否为另一模型提供帮助”是两个问题。

### 4.2 提速来源：省掉 Sharer 中间解码（Table 3）

MMLU-Redux 上的时间拆分：

| 方式 | 总时间 |
|---|---:|
| Receiver-only | 308 ms |
| Sharer-only | 346 ms |
| T2T | 1596 ms |
| C2C | 445 ms |

T2T 的 Sharer 生成约 80 个 token，其 decode 耗时 1312 ms；C2C 用约 90 ms 的融合代替该步骤。

**不能将相对 T2T 的加速，写成相对单模型推理的加速。** 本例 C2C 比 Receiver-only 慢。Table 3 与 Table 4 的报告数字也应各自引用，不混拼成同一测量。

正文设置为单张 A100、batch size 1；这些结果不直接覆盖跨机器网络传输完整 KV cache 的场景。Base Sharer 的指令遵循与较长中间输出也会影响 T2T 耗时，因此最大加速比不宜作为普适结论。

### 4.3 长文本（Table 5）

Qwen3-0.6B Receiver + Qwen2.5-0.5B Sharer：

| 长度区间 | Receiver | Sharer | T2T | C2C |
|---|---:|---:|---:|---:|
| 0–4k | 30.52 | 24.94 | 33.46 | 37.31 |
| 4–8k | 26.03 | 23.18 | 29.70 | 34.01 |
| 8k+ | 25.99 | 16.44 | 25.64 | 30.72 |

这些是 LongBench 汇总得分，不应统一解释为选择题准确率。该实验在 LongBench 不同数据集合上训练与测试，并非主通用训练配置的直接零样本长文本结果。

### 4.4 融合与残差消融：对 QuRO 最有价值的证据（Table 8）

| 方法 | 四项平均准确率 |
|---|---:|
| Project：投影 Sharer cache 并替换 Receiver cache | 20.70 |
| +Fuse：融合双方 cache，并残差加回 Receiver | 44.88 |
| +Gate：增加逐层门控 | 47.95 |

读法：

- 纯投影替换不足以可靠传递可用信息。
- 保留 Receiver 表示并融合外来信息，是该配置有效工作的关键。
- 层选择进一步改善结果。

**归因限制**：Project → +Fuse 同时改变了融合输入和残差保留，不能将 +24.18 pp 全部归因于残差；也不能据此证明任何残差 readout 都有效。

### 4.5 训练容量与更强变体

Table 6 的 C2C 配置包含 **478M 可训练参数**；Receiver 全量微调对照为 **596M**，Identical 配置为 **529M**。冻结主模型不等于只训练很小的适配器。

该表中异构 C2C 优于单模型微调和同模型融合，支持互补表示的作用，但不能推出所有收益都与新增容量或训练无关。

附录 Table 9 的 Qwen3-4B → Qwen3-0.6B 设置中，MMLU-Redux 为：

| 方法 | 得分 |
|---|---:|
| Receiver | 35.53 |
| Sharer | 71.38 |
| T2T | 42.95 |
| C2C | 45.92 |
| C2C-C | 62.78 |

增强 fuser 有明显潜力，但该表最大回答长度为 8、最大通信长度为 256；不能与正文 64-token 回答上限的实验不加区分地合并。

### 4.6 论文尚未证明的内容

- 任意模型对可以无训练直接交换 cache。
- KV 通信在等字节预算下优于压缩 soft tokens。
- 多模型、多轮工具调用系统中都有同样的效果和速度。
- 更高 effective rank 意味着更高任务信息量。
- 不传明文就获得可靠隐私保护。
- 更弱 Sharer 总能帮助更强 Receiver；论文明确讨论了噪声注入导致的退化。

## 5. 最后一层隐藏状态与 KV cache：数学关系

### 5.1 KV 来自各层输入表示，而非统一来自最后层输出

以典型 pre-norm、使用 RoPE 的 Transformer 为例：

$$
U^{(\ell)}=\operatorname{Norm}_\ell(H^{(\ell-1)}),
$$

$$
K^{(\ell)}
=
\operatorname{RoPE}\left(U^{(\ell)}W_K^{(\ell)}\right),
\qquad
V^{(\ell)}=U^{(\ell)}W_V^{(\ell)}.
$$

随后该层完成 attention、残差和 MLP 等运算，得到 $H^{(\ell)}$。式中省略 head reshape 等细节，具体模型可能还有 Q/K norm 等步骤。

因此：

$$
\text{最后层输出表示}=H^{(L)}
$$

与

$$
\text{完整 KV cache}
=
\{K^{(\ell)},V^{(\ell)}\}_{\ell=1}^{L}
$$

并不等价。即便最后一层的 KV，通常也由进入该层的表示计算，而不是由该层最终输出计算。

### 5.2 二者各自的用途

| 比较项 | 最后一层隐藏状态 | 完整 KV cache |
|---|---|---|
| 深度位置 | 最后阶段的表示 | 多个 attention 层 |
| 内容组织 | 上下文化特征 | 各层历史位置的 K/V |
| 原生用途 | 继续预测或作为外部特征 | 后续 token 在各层读取前文 |
| 对模型结构依赖 | 维度与语义空间 | 另涉及层、head、位置编码、mask |
| 能否一般性互换 | 不能保证 | 不能保证 |

对于标准因果 Transformer，在位置、mask 等配置正确的前提下，后续 token 可使用历史 KV 继续计算，无需重新运行历史 token。KV 因而适合作为续写状态。

但不能断言最后层输出无损包含所有中间层信息，也不能断言多层 KV 在任何下游任务上都更有用。**载体保留的信息、接收端可利用的信息、最终任务效果，需要分别检验。**

### 5.3 提取隐藏状态不自动构成压缩

若输入有 $n$ 个 token，提取最后层仍得到 $n\times d$ 个数，序列并未缩短。

软压缩成立的关键是将文档转为少量承载信息的向量：

$$
Z_D\in\mathbb R^{m\times d_z},\qquad m\ll n.
$$

这些向量可来自压缩器特定槽位的最后层输出，但不能只凭“取最后层”判断压缩率。

## 6. 我们的 soft tokens 最终也会形成生成器 KV

按本次讨论中的方法抽象：

$$
D\xrightarrow{E}Z_D
\xrightarrow{R_\theta(\cdot,q)}\widetilde Z_{D,q}
\xrightarrow{\text{generator inputs\_embeds}}
\{K_G^{(\ell)},V_G^{(\ell)}\}_{\ell=1}^{L_G}.
$$

我们的接口在输入端；生成器自己把 soft tokens 逐层变换为适合自身使用的 KV。C2C 则在 Receiver 已形成 cache 后，直接学习修改各层 KV。

| 维度 | QuRO 讨论路线 | C2C |
|---|---|---|
| 载体 | 紧凑文档 latent / soft tokens | 多层逐 token KV |
| 注入位置 | 生成器输入端 | 各层 attention cache |
| 序列压缩 | $n\to m$ | 默认保持 Receiver cache 长度 |
| 关键目标 | 紧凑表示上按 query 读出、复用 | 跨模型语义融合、省去中间解码 |
| 适配方式 | 读出后经过生成器层级计算 | 直接学习各层 cache 修正 |
| 主要风险 | 信息损失、读出破坏可用表示 | 对齐误差、多层状态不兼容 |

### 6.1 存储成本

在统一数值精度下：

$$
\operatorname{Size}(Z_D)\propto m d_z,
$$

$$
\operatorname{Size}(\mathrm{KV})
\propto 2Lnh_{\mathrm{kv}}d_h.
$$

后者包含 K/V 两份、多层和保留的 token。GQA 可减少 KV heads；量化可降低字节数。上述公式比较载体本身，**不意味着我们的生成器运行时不需要 KV**。

因此，应分别报告离线缓存大小、在线峰值显存、传输字节数和最终生成时的 cache 大小。

### 6.2 “可缓存”不是 QuRO 独占的属性

KV 同样可用于前缀缓存与复用。QuRO 更具体的研究问题是：

> 如何让 query 无关、紧凑的文档表示，在多次不同查询中保持可用，并通过查询相关读出改善回答？

此外，$Z_D$ 可复用不意味着 $\widetilde Z_{D,q}$ 及其生成器 KV 能跨 query 原样复用。若读出修改了文档 soft tokens，生成器通常需要为新的表示重新 prefill。端到端成本分析必须包含这一步。

## 7. 对 QuRO 的启发与建议

以下属于本次讨论提出的研究建议，不是 C2C 的实验结论，也不是仓库最新结果的重新判定。

### 7.1 将“信息保留”和“接收端可用性”分开

压缩表示可能包含答案，但 readout 的变换可能使它偏离生成器熟悉的分布。C2C 的替换失败提示我们：应保留已验证可用的输入通路，再学习任务相关修正。

等长残差形式可以写为：

$$
\widetilde Z_{D,q}
=
Z_D+\alpha\Delta_\theta(Z_D,q).
$$

若 latent 维度已经与 baseline 生成器输入接口一致，就不必为了形式额外引入 projector；若不一致，则必须明确接口映射。

修正项需与 $Z_D$ 同形状。若未来将输出长度改为 $p\ne m$，不能直接逐元素相加，需重新定义保留原载体的路径。

### 7.2 初始化与梯度

可使用较小残差强度，使初始行为接近 baseline。若门严格为零，修正分支参数在最初可能收不到梯度；若门与修正输出同时为零，还可能导致两侧均无法启动。

应选择梯度可流动的初始化，并检查：

- 初始化时输出是否接近 baseline；
- 残差幅度相对原 latent 的比例；
- 门与修正分支是否得到有效梯度。

不应将 C2C 的 Gumbel 门控作为必须照搬的结构。

### 7.3 最有解释力的对照

| 配置 | 目标 |
|---|---|
| 原始 $Z_D$ + query | 建立当前设置下可比较的 baseline |
| $Z_D+\Delta(Z_D)$ + query | 测量通用表示适配收益 |
| $Z_D+\Delta(Z_D,q)$ + query | 测量 query 条件化的额外价值 |

尽量匹配模块容量，固定压缩器、压缩率、数据划分和生成器训练设置。若 decoder LoRA 参与训练，各配置应有可比较的训练、早停与验证选模流程，避免将训练策略收益全部归因于 readout。

对现有或旧 PISCO baseline 的超越可以如实报告；若要进一步声称“query 条件化导致提升”，仍需要上述机制对照。这是两种不同强度的主张。

### 7.4 重复查询与证据层级

后续按优先级补充：

1. 检验 residual query-conditioned readout 在已有 baseline 上的稳定收益；已有结果以当前实验记录为准。
2. 用 query 无关适配器对照，判断条件化是否必要。
3. 增加第二数据集和合理压缩率档位，检验结论是否只适用于单一设置。
4. 对同一文档配置多个不同问题，测量文档表示复用。
5. 将离线编码、在线 readout、生成器 prefill、decode 分开计时，报告随查询次数变化的总成本。

不能仅用单次 QA 准确率提升证明重复查询优势；也不能仅用缓存存在证明端到端加速。

### 7.5 论文定位

可以形成的研究叙述是：

> 在 query 无关的紧凑文档表示上，学习保留原始信息通路的 query-conditioned 修正，使生成器更有效地利用缓存表示，并验证这种读出在质量、压缩预算与重复查询成本之间的收益。

这仍需对应实验证据支持。残差、门控、连续表示通信和“可以缓存”本身均不足以单独构成创新。

当前不必因为 C2C 的结果就改成多层 KV 注入。可以将“从紧凑 latent 直接生成少量前缀 KV”记录为后续方向，但那会引入新的层级映射、位置处理、内存预算和训练问题，不能视为当前输入端接口的免费替换。

## 8. 引用定位与后续阅读

| 主题 | 原文位置 |
|---|---|
| Cache enrichment 与 conversion oracle | §3.2；Tables 1–2；附录 A.3.1–A.3.2 |
| 残差融合与训练目标 | §3.3；Figure 5；Eqs. (3)–(4) |
| Token / layer 对齐 | §3.3.3；附录 A.1.1–A.1.2 |
| 主实验与耗时 | §4.1–4.2；Tables 3–4 |
| 长文本与新增容量对照 | Tables 5–6 |
| 融合与门控消融 | Table 8 |
| C2C-C 强弱模型通信 | 附录 A.1.3；Table 9 |
| 局限与后续扩展 | §5 |

与本仓库已有报告联读：

- [RRK](RRK_PAPER_NOTES.md)：压缩表示的质量和消费效果。
- [Perceiver IO](PERCEIVER_IO_PAPER_NOTES.md)：输入、latent 与输出 query 的解耦。
- [SeleCom](SELECOM_PAPER_NOTES.md)：query 相关压缩与重复查询场景的区别。
- [ArcAligner](ARC_ALIGNER_PAPER_NOTES.md)：生成器适配收益与 readout 收益的归因。

**最终判断**：C2C 是 QuRO 在“连续表示如何被接收模型使用”这一问题上的重要相关工作。它支持认真研究融合、保留原通路与接收端适配；它尚未替我们证明紧凑文档表示上的 query-conditioned readout 有效，更未证明应该放弃 soft-token 路线。
