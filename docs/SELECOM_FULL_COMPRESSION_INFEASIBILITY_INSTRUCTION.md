# SeleCom「全量压缩不可行性」复现实验：Claude Code 执行指示

> 状态：P0，立即优先于继续增加 QuRO/readout 新模块  
> 执行者：Claude Code（服务器端）  
> 目标模型：仓库现有 PISCO，直接作为 full-compression baseline  
> 本文只规定实验、产物和决策门槛；Phase 1 不训练模型、不实现新解决模块。

## 1. 为什么现在做这件事

当前工作应先暂停“先设计模块、再寻找解释”的路线，回到 SeleCom 对 full compression 的核心批评：**both infeasible and unnecessary**。本轮优先检验其中的 **infeasible**。

这里的“不可行”不是指数学上无法压缩，而是指：全量重建目标可能把压缩向量训练成对 decoder 具有过强吸引力的记忆，使 decoder 在同时看到 query/instruction 与全量压缩记忆时，仍然优先读取、复述文档，忽略当前指令。SeleCom Figure 2 的关键现象是：

- reconstruction 指令下，全压缩模型可以复述文档；
- 冲突指令（例如“忽略文档，只输出指定随机字符串”）下，全压缩模型仍倾向输出文档内容；
- 非压缩模型能转向 instruction；
- 注意力图中，全压缩模型在两类指令下都持续聚焦压缩文档位置。

我们要在 PISCO 上回答的不是“能不能画出相似颜色的图”，而是：

> **query/instruction 与 PISCO 全量压缩向量一同进入 decoder 后，decoder 到底把注意力、预 softmax 分数和有效生成决策分配给了谁？这种分配是否造成了可复现的 instruction/query suppression？**

最终论文叙事应严格按以下顺序成立：

1. 行为实证：出现什么失败；
2. 机制诊断：失败与哪一种 decoder 内部现象对应；
3. 因果验证：改变该现象是否改变失败；
4. 定向方案：只针对已确认机制增加模块；
5. 对照改进：方案确实修复目标失败，且不破坏原能力。

## 2. 本轮硬性边界

在完成本文的 Phase 1–3 之前：

- 暂停继续扩展 R/RQ/readout 结构；
- 不重新训练 PISCO compressor、adapter 或 decoder；
- 不把 QuRO readout 接进主复现实验；
- 不用少数“好看样例”代替总体统计；
- 不预设 SeleCom 的解释在 PISCO 上必然成立；
- 不把 attention weight 单独当作因果解释；
- 不改写、覆盖任何历史结果目录。

Phase 1 使用现有 PISCO direct/full-memory 路径，即当前仓库的 `P` 类 baseline。模型、tokenizer、prompt builder、adapter、cache 和生成参数必须来自仓库已有配置；不要另写一个表面相似但输入拼接方式不同的 PISCO demo。

## 3. 研究问题与预注册假设

### RQ1：行为层面是否复现“不可行”

当 PISCO 压缩记忆与冲突 instruction 同时输入 decoder 时，模型是否比 raw-text/non-compression 条件更容易忽略 instruction，转而输出文档相关内容？

### RQ2：是否存在 memory attention dominance

相同 target token、相同 decoder 和相同 instruction 下，PISCO memory positions 是否获得异常高的：

- attention 总质量；
- 按 source token 数量归一化后的 attention density；
- pre-softmax QK score；
- key/value 或 residual-stream 范数贡献。

### RQ3：注意力现象是否解释行为失败

样本级 memory/instruction dominance 是否与 instruction-following failure 显著相关？对 memory 通道做最小、可逆的推理时干预后，行为是否同步改变？

### 假设

- **H1（行为）**：PISCO full compression 在 conflict instruction 下比 raw text 更常泄漏/复述文档，精确输出随机目标串的比例更低。
- **H2（分配）**：PISCO memory 对 instruction 的 attention mass、density 或 QK-logit 比值，在 conflict 条件下仍显著偏高。
- **H3（持续性）**：上述偏高不是单个 head/单层偶发现象，而会在中后层或多个生成步持续出现。
- **H4（机制候选，不是既定事实）**：memory hidden state / K 范数、方向性或 logit 尺度可能放大 memory attention。必须测量，不能直接沿用 SeleCom 的因果措辞。
- **H5（替代解释）**：若只有 attention mass 偏高、density 和 pre-softmax score 不偏高，则现象可能主要来自 memory token 数量，而不是压缩向量的特殊吸引力。

## 4. 先设置“资格门槛”

PISCO 未必与 SeleCom 的 full-reconstruction compressor 具有完全相同训练目标。因此，在讨论“过度关注压缩记忆”前，必须先证明 PISCO memory 在当前任务里确实携带并影响了文档信息。

每个正式样本至少做以下验证：

1. **Correct-memory**：正确文档的 PISCO memory；
2. **Mismatched-memory**：来自另一篇文档、长度相同的 PISCO memory；
3. **No-memory**：不提供文档记忆；
4. 可选 **Zero-memory**：保持槽位和位置不变，但将 memory embedding 置零，只作为接口/位置诊断。

只有 correct-memory 相比 mismatched/no-memory 在 reconstruction、文档问答或文档辨识上有清晰增益，才允许把该样本纳入“full-compression infeasibility”机制分析。否则该样本只能说明 PISCO 没有有效传递文档信息，不能说明它压制了 instruction。

## 5. 实验分为两级，避免多文档混淆

### Level A：SeleCom Figure 2 的最小复现

先使用单篇、短文档：

- 文档长度建议 32–96 个 tokenizer tokens，并记录 PISCO 实际接收的 tokens；
- 选择包含可唯一辨识的人名、数字或事实的文本；
- 使用当前 PISCO 单文档压缩产出的全部 latent；
- 先做 20–50 个可人工检查的样本；
- 从中固定 3–5 个样本用于论文式可视化，但结论来自完整集合。

每篇文档构造两类指令：

1. **Reconstruction**：要求逐字复述给定背景；
2. **Conflict instruction**：要求忽略文档，只输出一个每例随机生成、与文档无重叠的 nonce 字符串。

nonce 不要全数据集固定为同一个字符串。建议每例 8–16 个可稳定分词的字符/短 token，并提前验证 tokenizer 切分；报告 exact match 与 token-level match。

如果 PISCO 本身无法完成逐字重建，不得据此结束实验。改用同时具备以下两个属性的 **document-grounded task**：

- correct-memory 明显优于 mismatched/no-memory；
- conflict instruction 的目标与文档答案明确冲突。

这时将结果命名为“SeleCom infeasibility mechanism replication on PISCO”，而不是“verbatim reconstruction exact replication”。

### Level B：回到当前 HotpotQA 设置

在最小复现脚本稳定后，使用仓库固定的 HotpotQA dev 数据和已有 PISCO cache 扩展：

- 第一轮用固定随机种子的 200–500 例做诊断；
- 如成本允许，再跑完整 2k dev；
- 不触碰现有 test 集；
- 保留当前 `K` 篇文档、每篇 latent 数和拼接顺序，记录总 memory token 数；
- 同时报告单文档与多文档结果，不能把 `K×m` 增长造成的长度效应误写成压缩机制。

HotpotQA 至少包含：

- 原始 QA query + correct memory；
- 同一 query + mismatched memory；
- query swap（保留 memory，替换另一问题）；
- conflict instruction + correct memory；
- raw-text/non-compression 对照（如果同一 decoder 接口支持）。

## 6. 核心实验矩阵

主比较是 2×2：

| 文档表示 | Reconstruction / grounded task | Conflict instruction |
| --- | --- | --- |
| PISCO full-compression memory | 测文档利用与重建倾向 | 测 instruction suppression |
| Raw document tokens | 同任务的非压缩参照 | SeleCom 式 instruction-following 参照 |

必须额外保留：

| 控制 | 用途 |
| --- | --- |
| Instruction only / no memory | 验证 PISCO decoder 本身能否输出 nonce |
| Mismatched memory | 验证行为是否真正依赖文档记忆 |
| Zero memory（可选） | 分离“有占位/位置”与“有内容” |
| Same memory + instruction 位置前后互换（次要） | 检查位置/recency 混淆，不作为主结果 |

公平性要求：

- 优先使用同一 PISCO decoder、同一 adapter 和同一生成配置比较 memory 与 raw tokens；
- 若 PISCO decoder 的 raw-text 接口不成立，可补充 base/instruction decoder 的 raw-text 结果，但必须标注“不同 decoder，仅为参考”，不能作严格因果对照；
- prompt 中除 document representation 外的系统词、query、instruction、分隔符和目标格式保持一致；
- greedy decoding，固定 `max_new_tokens`、EOS 规则、dtype、seed 和 batch 策略；
- 所有模型与 adapter 设为 eval，关闭 dropout。

## 7. 明确 decoder 中的 token/position 分组

不要根据字符串长度猜索引。必须从仓库实际的 `inputs_embeds` / `input_ids` 组装逻辑生成 position manifest，至少标出：

- `prefix/system`；
- `document_delimiter`；
- `memory`（PISCO soft tokens）或 `raw_document`；
- `query`；
- `instruction`；
- `answer_prefix`；
- teacher-forced 或已生成的 `output_history`。

每条样本保存以下审计信息：

- 各组 `[start, end)`；
- 组内 token/latent 数；
- 原始 prompt 文本与 tokenized 版本；
- 文档 ID、query ID、cache key；
- 实际 PISCO latent shape；
- 是否发生 truncation；
- decoder 前最终序列总长度。

给 position manifest 写单元测试：组间不重叠、覆盖所有输入位置、长度与实际 tensor 一致；memory 起止位置必须通过一个可人工核对样例验证。

## 8. Attention 采集与热力图定义

### 8.1 两种 forward 分开做

1. **Free generation**：用于行为结果；
2. **Teacher-forced forward**：用于主 attention 对照。

主热力图采用 teacher forcing，因为它能让 compressed 与 raw-text 条件读取完全相同的 target tokens，避免两个模型已经生成不同输出后，attention 差异只是 output-history 不同。

同时保存少量 free-generation 的逐步 attention，作为案例展示，不作为唯一统计依据。

### 8.2 运行设置

- 诊断 forward 使用 `output_attentions=True`、`use_cache=False`；
- 若 FlashAttention/某些 SDPA backend 不返回可靠 attention weights，诊断运行切换到 eager/math backend；
- 先在 5–10 例上确认 eager 与正常推理 backend 的 logits/输出在可接受误差内；
- 不为全数据保存完整 `[layers, heads, T, T]`，应在线聚合 group statistics；
- 只为固定可视化样本保存完整 attention tensor。

### 8.3 必须同时报告 mass 与 density

设第 `l` 层、第 `h` 个 head、目标位置 `t` 对 source group `g` 的 attention 为：

```text
Mass(l,h,t,g) = Σ_{j∈g} Attn(l,h,t,j)
Density(l,h,t,g) = Mass(l,h,t,g) / |g|
```

两者都必须报告：

- `Mass` 回答 decoder 的总注意力预算分给了哪个组；
- `Density` 控制组长度，回答平均每个 memory/raw token 的吸引力；
- 只画 Mass 会天然偏向有更多 token 的 PISCO multi-document memory；
- 只画 Density 又会忽略一个长 memory 区域真实占走的大量总预算。

定义并预先固定核心比值（加小 `epsilon`）：

```text
memory_instruction_mass_ratio = Mass(memory) / (Mass(instruction) + epsilon)
memory_instruction_density_ratio = Density(memory) / (Density(instruction) + epsilon)
```

### 8.4 三类主图

1. **Figure-2-style source-group heatmap**  
   x 轴为 source groups，y 轴为 target/output token；compressed/raw × reconstruction/conflict 共四幅图。固定同一颜色范围。

2. **Layer × source-group heatmap**  
   对 target positions 与 heads 做预注册聚合，显示 memory dominance 出现在早/中/晚哪一层。另画每层置信区间曲线。

3. **Generation-step × source-group heatmap**  
   显示模型是从第一个生成 token 就忽略 instruction，还是随输出历史逐渐被 memory 拉回。

论文式总图可按 early/middle/late layer thirds 汇总，但原始逐层统计必须保留。不要只挑某一层或某几个 head。

### 8.5 Head 级稳健性

除均值外至少报告：

- head-level median / IQR；
- memory-dominant heads 的比例；
- bootstrap 95% CI（以样本为重采样单位）；
- 结果是否仅由极少数 head 驱动。

## 9. 不止看 softmax 后的 attention

SeleCom 给出的潜在解释包括压缩向量方向性、范数极化与 reconstruction objective 缺少 instruction-following 约束。我们需要把这些当成待检验机制。

至少收集：

1. memory、instruction、query hidden states 的 L2 norm（embedding 输入及各层进入 attention 前）；
2. 各层 projected K/V norm，按 group 汇总；
3. target Q 对 memory、instruction、query 的 pre-softmax QK logits：均值、最大值、分位数；
4. attention entropy 与 top-attended source position；
5. memory 与普通 token embedding / query hidden state 的 cosine statistics；
6. 如实现成本可控，记录 attention output 按 source group 的 value-weighted contribution norm。

关键判别：

- Mass 高、Density 不高、QK logits 不高：优先解释为 token 数/长度效应；
- Density 和 QK logits 都高，且与 norm 同步：支持尺度/范数机制；
- logits 高但 norm 不高：检查方向对齐或特定 heads；
- attention 并不异常但行为失败：不能继续宣称 SeleCom 式 attention capture，应检查表示可读性、prompt、adapter 或输出先验。

## 10. 行为指标

### Conflict instruction

- nonce exact match；
- nonce token-level accuracy；
- instruction-following rate；
- document leakage：输出与文档的 token overlap / longest copied span；
- 输出是否命中文档标题、实体或 grounded answer；
- 目标 nonce 的 teacher-forced NLL / first-token log-prob。

### Reconstruction / grounded task

- 短文档 reconstruction：Exact Match、ROUGE-L、token F1；
- QA：现有 EM、substring、F1，保持仓库口径；
- correct-memory 相对 mismatched/no-memory 的 paired delta。

### 机制关联

逐样本关联：

- dominance ratios 与 instruction failure；
- QK-logit gap 与 leakage；
- memory norm 与 document-copy span；
- 单文档 latent 数、Hotpot 总 latent 数与失败率。

报告 paired bootstrap CI；样本数允许时，可做控制输入长度后的简单回归。不要只报告总体均值。

## 11. Phase 3：最小因果干预（仍不是新方案）

只有 Phase 1–2 观察到稳定行为失败和候选内部机制后，才做推理时、可逆的小干预验证。目的只是检验因果方向，不是提交最终模块。

按观察结果选择最小集合：

- memory attention logit 加固定 bias / temperature sweep；
- 对 memory K 做 norm matching，保持方向不变；
- memory attention mask（极端负控制，不作为可用方案）；
- 保持 memory 内容不变，仅改变 instruction 的位置或重复一次；
- 对 latent 顺序打乱（仅检验顺序依赖，不能等同信息不变）。

必须同时检查两端：

1. conflict instruction 是否恢复；
2. reconstruction/QA 的文档利用能力是否下降。

如果压低 memory attention 只让模型更听话、却完全丢失文档能力，这只证明存在 trade-off，不等于找到解决方案。

## 12. 结果后的决策树

| 观察结果 | 结论 | 后续方案方向 |
| --- | --- | --- |
| 行为失败 + memory Density/QK dominance + 干预可缓解 | SeleCom 式机制在 PISCO 上成立 | 设计 query/instruction-conditioned memory calibration、gate 或 readout；目标是动态平衡而非盲目删记忆 |
| 行为失败 + 只有 Mass 高 | 主要是长度/预算竞争 | 优先研究选择、top-k、分块预算或 length-aware normalization |
| 行为失败 + attention/logit 不异常 | 不能用 attention capture 解释 | 检查表示解码、prompt/interface、adapter、decoder 中间层注入或训练目标 |
| attention dominance + 行为不失败 | dominance 未证明有害 | 不围绕该热力图增加模块；继续找真正性能瓶颈 |
| PISCO 未通过 correct-vs-mismatch 资格门槛 | memory 没有被有效利用 | 先修复/确认 PISCO baseline 与输入接口，禁止讨论 infeasibility |
| PISCO 未复现 SeleCom 现象 | SeleCom 批评不可直接迁移 | 将论证转向 “unnecessary”、query specificity 或当前 P baseline 的实际错误类型 |

任何最终新模块必须能用一句“观测—动作”对应关系描述，例如：

> 观测到中后层 memory QK logits 在 conflict query 下仍系统性压过 instruction，因此模块根据 query/instruction 对 memory 通道做条件化校准，并保持离线 PISCO cache 不变。

若无法写出这种对应关系，就不进入方案实现。

## 13. Claude Code 的实现任务拆分

### Task 0：核对实现与 SeleCom 口径

- 在代码中定位 PISCO `P` baseline 的真实 prompt / `inputs_embeds` 拼接路径；
- 定位 latent cache shape、每文档 latent 数、adapter 与 decoder 加载方式；
- 核对 SeleCom 正文/附录/公开代码中 Figure 2 的 attention 聚合层、heads、target rows；
- 如果找不到精确定义，在结果报告中明确写“conceptual reproduction”，并采用本文预注册定义，不得反推一个更好看的聚合口径。

### Task 1：行为复现

建议新增：

```text
scripts/diagnose_full_compression_infeasibility.py
```

支持 Level A 与 Level B；输出逐例 prompt、生成、指标、group boundaries 和 manifest。

### Task 2：attention / norm / logit 采集

在同一入口增加诊断模式，或新增清晰模块。不要复制一套与正式 PISCO 不同的模型加载代码。优先复用仓库现有 loader、prompt builder 和 cache reader。

### Task 3：绘图

建议新增：

```text
scripts/plot_full_compression_infeasibility.py
```

绘图脚本只读取保存后的统计，不重新跑模型；固定图形配置、颜色范围和排序，支持从 manifest 完整重建。

### Task 4：测试

至少新增：

```text
tests/test_attention_grouping.py
```

覆盖 position manifest、group mass 求和、density、padding/batch mask、teacher-forced target slicing，以及单步手算 attention 对照。

### Task 5：结果报告

生成：

```text
docs/FULL_COMPRESSION_INFEASIBILITY_RESULTS.md
```

报告必须先给行为结论，再给 attention/logit/norm 证据，最后按第 12 节选择一个分支；不要在结果未知前写“验证了我们的假设”。

## 14. 输出目录与可复现性

建议每次运行使用新目录：

```text
results/full_compression_infeasibility/<timestamp_or_run_id>/
  manifest.json
  config.yaml
  examples.jsonl
  behavioral_metrics.json
  grouped_attention_stats.npz
  norm_logit_stats.npz
  full_attention_examples/        # 仅固定少量样本
  figures/
    figure2_conceptual_replication.png
    layer_group_heatmap.png
    generation_step_heatmap.png
    norm_logit_diagnostics.png
```

`manifest.json` 至少记录：

- git commit、是否 dirty；
- 模型、PISCO/adapter checkpoint 与 cache 路径/标识；
- tokenizer 和 prompt template；
- 数据 split、样本 ID、抽样 seed；
- GPU 型号/数量、CUDA、PyTorch、Transformers 版本；
- dtype、attention backend、batch size；
- generation 参数；
- 每文档和每样本的 latent 数；
- 运行命令与 wall-clock time。

不要覆盖已有 run；失败 run 保留 manifest 和错误摘要。

## 15. 验收门槛

### Phase 1 可验收

- PISCO correct-memory 通过资格门槛；
- Figure 2 的四个核心条件均有 free-generation 行为结果；
- teacher-forced source-group heatmap 可从 manifest 重画；
- mass 与 density 同时报告；
- raw/full-compression 的 prompt 非文档部分一致并有审计记录；
- 至少一个完整样本可人工核对所有 position groups。

### Phase 2 可验收

- 逐层、逐步、逐 head 统计完成；
- pre-softmax logits、norms 与 attention 同时存在；
- Hotpot 单/多文档长度效应被分开讨论；
- 有 paired CI 和样本级机制—行为关联；
- 结论能落入第 12 节某一分支，包括“未复现”。

### 进入解决方案设计的门槛

必须同时满足：

1. 失败行为可稳定复现；
2. PISCO memory 被证明是有效文档载体；
3. 某一内部机制跨样本稳定；
4. 最小干预改变该机制时，行为按预期改变；
5. 已量化对原 reconstruction/QA 能力的代价。

未满足时，继续诊断，不增加新模块。

## 16. 最终要回答的五句话

结果报告开头必须用数据回答：

1. PISCO 全量压缩记忆是否真的被 decoder 使用？
2. 加入 conflict query/instruction 后，模型是服从指令还是被文档记忆拉回？
3. 这种行为首先出现在哪些层、heads 和生成步？
4. 主因更像 token 数量、QK/logit 尺度、向量范数、位置效应，还是根本不是 attention capture？
5. 因此下一模块具体要改变哪个量，并保留什么能力？

这五句没有明确答案之前，不进入下一轮架构设计。
