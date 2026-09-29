# QuRO：压缩证据读取与 D0/D2 因果顺序的测试—训练执行方案

> 2026-09-30 · 工作分支 `feat/selecom-infeasibility`  
> 性质：待验证的实验协议；不是机制结论，也不是新架构实现清单。  
> 本文整合 [Reader Trajectory 原案](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/4babc66933b663532a21cdb661498b3370a25952) 与 [D0/D2 原案](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/commit/7e9eaa137d1018f2780396a7e4691eb55f66e6fa)，并更正两案的训练门槛。历史提交保留原文供追溯。

## 0. 当前决策

主问题：**同一份离线压缩文档表示 `Z=C(D)` 面向不同问题时，QA 差距究竟源于证据丢失、证据读取/组合失败，还是输入顺序改变了有用的在线计算路径？**

先确认公平 raw–memory 缺口和逐跳证据，再以 D0/D2 检查条件化计算写入的位置，最后用任务功能干预决定是否训练。D0/D2 sensitivity 是拓扑诊断，**不再是训练准入证**。仅在有答案相关的因果效应且失败定位明确时，做最小 matched training。暂不实施 workspace、external cross-attention 或 recurrent readout。

| 路径 | 当前状态 | 可以证明什么 | 不能证明什么 |
|---|---|---|---|
| 公平 raw–memory + K=2 gold/逐跳 | 待统一评测和标注 | 当前任务缺口及事实读取、组合、干扰的候选位置 | 单跳失败即证明压缩表示没有该信息 |
| D0/D2 counterfactual sensitivity | 代码已有；需 GPU 执行 | 条件化状态写在哪侧、是否越过数值噪声 | 状态变化有助于回答 |
| D0/D2 task-functional patch / path blocking | 尚未实现 | 特定状态路径在固定模型中对答案的因果贡献 | 训练后必定可达到被 patch 的状态 |
| matched LoRA、reader-state distillation | 均未启动 | 特定训练策略在指定预算下是否改善 QA | 该输入顺序或架构的普遍优劣 |

先前 [Stage C 结果](../results/output_scale_stage_c.json) 中 P₁ substring=0.5990，output_scale=0.1/0.05 分别为 0.5845/0.5880；域内 decoder LoRA 显著有效，缩放没有收益。结论限于已测配置。memory 的相对残差更新小，不等于不参与计算；详见 [纠偏文档](LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md)。

## 1. 两个顺序的精确定义与共同约束

文档离线压缩、缓存：`D → Z=C(D)`，不接收 query。在线使用 `(Z,Q) → G_θ → A`。固定 PISCO cache、decoder、memory 数量、retrieved docs、截断口径、生成和评分参数。

- **D0 / memory-first**：`[prefix, M₁…Mₙ, Q₁…Qₜ, suffix, A]`。memory 位置看不到后面的 query；query、suffix、answer 位置能逐层读取前面的 memory。query 位置已经是动态 reader。
- **D2 / question-first**：`[prefix, Q₁…Qₜ, M₁…Mₙ, suffix, A]`。memory 位置能读取前面的 query；query 位置看不到后面的 memory。answer 仍能读取两者。
- 两侧位置还可读取各自更早的位置，不能把 D2 误称为 memory 与 query 的双向注意力；两种输入顺序都允许在线 query-conditioned computation，只是写入状态的位置不同。
- D2 **改变两条路径**：打开 Q→M，同时关闭 M→Q。即使适配后 D2 QA 较高，仍需固定 D2 布局内的功能干预，才可归因于 Q→M。
- 两者都保留可复用的离线 Z。D2 的 decoder memory K/V 已依赖当前 query，通常不能作为跨 query 的相同前缀 K/V 复用；D0 的完全相同 memory 前缀有这种可能。未来效率表分别计入离线缓存、在线 prefill 与生成。

PISCO 论文主 prompt 为 Background→Question。COCOM 论文图示不能直接代表其公开 checkpoint：公开 `modeling_cocom.py` 的 `generate_from_text` 实际构造 memory→question。此处 D2 是独立的 question-first 研究接口，不称为“COCOM checkpoint exact reproduction”。参考 [PISCO](https://aclanthology.org/2025.findings-acl.800.pdf)、[COCOM](https://arxiv.org/pdf/2407.09252v3) 和 [COCOM 公开实现](https://huggingface.co/naver/cocom-v1-128-mistral-7b/blob/main/modeling_cocom.py)。

## 2. P0：先确认任务缺口（暂不训练）

**目的**：排除“比较了不同 checkpoint、样本、harness、解码格式”导致的假差距。

对同一批有 ID 的 HotpotQA dev 题，锁定检索文档、文档文本与缓存版本、截断、query 文本、生成上限、评分脚本。分别在发布版 PISCO decoder 与当前 P₁ best decoder 下，用 raw documents 与 compressed memory 评测。每个 decoder 内 raw/memory 用**相同权重**，报告 EM、F1、substring、teacher-forced answer NLL、输出长度、bridge/comparison、逐例 paired CI。原有 P₁ 是 memory 适配的 decoder，故其 raw 条件仅为同权重表征对照，不是“raw 自己适配后的上界”。只有要声称各自适配系统上界时才另训 matched raw QA LoRA。

预先指定主要任务指标与最小实用差距，并在同样本上检查 CI。若差距不存在或很小：停止以“修复压缩损伤”为主线，转向多查询复用/效率。若仍稳定存在：进入 P1，不把差距直接归因于 reader。

## 3. P1：K=2 gold 与逐跳证据诊断

**目的**：区分单篇事实读取失败、桥接实体取得失败、跨文档连接失败与 K=10 干扰。先构建人工核验的 bridge 样本；子问题与 oracle 提示不泄漏最终答案，并保留原文档 ID/顺序。对于相同样本和 decoder，raw/memory 各跑：

1. K=2 gold 原多跳问题；
2. 第一跳子问题（bridge entity）；
3. 已给正确 bridge entity 的第二跳子问题；
4. 原问题 + oracle bridge entity；
5. K=10 原设置；
6. 需要定位时做 raw₁+memory₂ 与 memory₁+raw₂，并交换文档顺序作检查。

| 观察 | 优先行动 |
|---|---|
| 单跳 memory 就明显弱 | 优先查压缩保真、解码适配、证据覆盖；单跳失败**不能**单独证明 Z 中无信息 |
| 两个单跳可答、组合题失败 | 读出/组合成为更强候选，再做 P2/P3 |
| oracle bridge 恢复 | 定位第一跳获取或跨文档连接 |
| K=2 接近 raw、K=10 差距扩大 | 优先研究干扰与选择 |
| 某篇改 raw 即恢复 | 定位到该篇的读取/压缩或其与另一篇的组合 |

分析时检查模型参数知识与文档依赖（可用实体/事实的对照），避免把不使用文档也能答对误认为成功读取 Z。重建文本只作辅助证据，不以桥接实体出现与否判定 Z 有无该事实。

## 4. P2：D0/D2 拓扑诊断（已有代码，先修复并运行）

**目的**：确定问题相关的变化写入 query 侧还是 memory 侧。使用当前 `scripts/diagnose_causal_order.py`，固定 `M_A,Q_A`，分别将 `Q_A→Q_B` 和 `M_A→M_B`，在两个布局采集真实 decoder block input/output。保存：

- `Δ_M^l, Δ_Q^l = mean_i ||h_{out}^l-h_{in}^l|| / ||h_{in}^l||`；
- 主指标 `S_{M←Q}^l = mean_i[1-cos(h_{M_i}^l(M_A,Q_A), h_{M_i}^l(M_A,Q_B))]`；
- 对称检查 `S_{Q←M}^l` 与 relative-L2；
- 原输入 answer NLL；可选生成 QA，**不进入拓扑 gate**。

D0 的 `S_{M←Q}` 与 D2 的 `S_{Q←M}` 在严格 causal mask 下应接近数值底，主要用于验证采集、位置与方向。D2 的 `S_{M←Q}>0` 是条件化迹象，不等于有用的证据处理。原脚本从不同样本 A/B 交换 query/documents；**“同一文档多个真实问题”功能测试尚未由该脚本实现**，应在 P3 单独准备。

有效性要求：同一对样本在两种布局的 memory slot 和 query token 位置各自完全一致；无 NaN/Inf；至少 32 个 exact-position pairs。位置不匹配或非有限值即 `RERUN_INVALID`，不解释机制。smoke 的 8 pairs 只查加载、hook、位置和输出，不能用于正式判定。

现有阈值保持为预注册的**拓扑信号筛查**：≥32 pairs、D2–D0 按 pair 跨层均值的 bootstrap 95% CI 下界 >0、D2 均值 ≥1e-4 且相对 D0 ≥3 倍、至少 25% 的 block CI 下界 >0。通过记 `GO_FUNCTION_TEST`：下一步测任务作用。未通过记 `HOLD_DIAGNOSTIC`：本 checkpoint 尚无稳健信号，核查样本与控制；由于 D2 尚未适配，不把它解释为“训练后 D2 必然无效”。

运行：

~~~bash
python tests/test_causal_order.py
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_causal_order.py --preset pisco_hotpot --pairs 8
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_causal_order.py --preset pisco_hotpot --pairs 64
~~~

正式 run 先不加 `--generate`；完成机制检查后可另跑 `--generate` 补 QA。输出目录由脚本生成 `manifest.json`、`pairs.jsonl`、`summary.json`，读取 `summary.json/topology_gate`。代码更名后旧的 `training_gate` 字段不再适用，服务器上的下游汇总脚本如引用旧字段，应一并更新。

## 5. P3：任务功能干预（新的必要实现，当前尚无代码）

先在同一 D2 布局、同一文档 Z 上准备回答依赖不同证据的 `Q_a,Q_b`，核验不同问题所需事实/答案与 token 位置。**主干预**：以 `(Z,Q_a)` 为接收运行，在指定 block 的 memory 位置回填 `(Z,Q_b)` 的对应 states，保留接收运行的 query、suffix、其他位置和后续网络。比较原始运行、同状态回填（应近似无影响）、不同问题回填、错误文档/无关状态回填；对可实现的版本再阻断 D2 中 Q→M 的读取路径。按 early/middle/late 预先选少数层，在 dev 定层，在未用于选择层的样本上复核。

这检验的是“Q 诱导的 memory 变化对 `Q_a` 的回答有功能作用”，而不是 cosine 大小。报告逐例 answer NLL、生成 EM/F1/substring，且检查错误问题的答案是否错误地迁移。patch 只取**答案生成前的 prompt states**，不从含 gold answer 的 donor 轨迹偷带答案。单次 patch 会改变模型状态分布，因此配合 identity、错误问题与路径阻断解释；不能从一次掉分直接宣称机制。

D0 并行做同布局 correct-memory→mismatch-memory 的 reader patch 作为信息流 positive control，并把 query 后的模板尾部、首个答案预测位置纳入候选位置。若要研究真实压缩缺口，再在校验 position control 后探索 raw→memory reader patch。raw donor 可能包含 Z 已丢失的事实；即使 rescue，也只证明 donor state 可帮助答案，不能证明学生能从 `Z,Q` 生成它。raw/memory 长度不同，仅将 query 的绝对 RoPE index 对齐不保证相对距离等价。

Memory-write blocking（分别阻断 memory attention、MLP、二者残差写入，同时保留其供后续读取的 K/V）是**判断能否采用“静态 memory”结构假设的分支测试**。阻断掉分不否定 reader 方向；阻断无害也不证明增强 memory 更新无用。该干预目前也未实现。

功能信号须在预定样本及配对统计上改善 answer NLL，并在生成指标/逐跳样本中显示有意义的同向效果；层位不能事后从大量扫描中挑单一显著峰。预先登记最小实用效果，例如对有稳定 raw–memory gap 的失败子集要求 patch 恢复 ≥20% 缺口，且 paired CI 下界 >0；具体阈值在正式执行前固定，不随结果移动。只改 hidden similarity 而 QA/NLL 没变化，则不进入相应的训练分支。

## 6. 测试→训练的决策树

| 证据状态 | 决策 |
|---|---|
| P0 缺口不稳定 | 停止“修复性能缺口”的训练叙事 |
| P1 单跳也弱 | 优先做压缩保真/解码适配定位；不得仅凭 raw patch rescue 训练 trajectory |
| P2 sensitivity 稳定，但 P3 无任务作用 | 不启动完整 D2 训练；记录为拓扑现象 |
| D2 同文档不同问题的 memory 状态对 QA/NLL 有稳定、可复核的功能贡献 | 进入 T-D2 matched adaptation |
| D0 reader patch 稳定 rescue，逐跳证据大体可用；raw patch 并非唯一证据 | 进入 T-D0 最小 reader-state distillation |
| 两种作用都存在 | 先做变量更少的 D0/D2 顺序适配，检查剩余缺口，再考虑状态蒸馏 |
| patch/阻断位置或结果不稳 | 扩充数据或修正干预；不凭偶然层设计模块 |

这里的 gate 是**训练预算决策**，不是 Transformer 的普适定理。任何负结果限定于当前 PISCO cache、Mistral/LoRA、数据及预算。

### T-D2：matched D0/D2 LoRA continuation

两组都从同一发布 PISCO decoder LoRA 出发，固定 cache、训练行、seed、steps、batch、grad accumulation、LR/decoder LR、scheduler/warmup、eval cadence、best checkpoint 选择规则；唯一变量为 `--decoder_input_mode D0` 或 `D2`。直接复制当前 P₁/P-direct 的已确认 recipe，不在本文发明新超参；训练期间不改压缩率、readout、KD、output_scale。两组都是从 D0 预训练接口继续，故这是**相同初始条件与预算下的适配对照**，不自动等于两个顺序各自的最优上限。

训练后做：

| train / eval | D0 | D2 |
|---|---:|---:|
| D0 continued | A | B |
| D2 adapted | C | D |

`A≫B, D≈A` 支持 zero-shot D2 的 interface mismatch 解释；`A≫D` 仅说明当前预算下 D2 未追平；`D>A` 必须加 seeds、配对 CI、bridge/comparison 和 K=2 分层。**无论 QA 结果如何**，对 D2 best checkpoint 重跑 P2 与 P3：检查 Q→M 信号和答案作用是否保留，避免把适配后的补偿路径误称为原定机制。

### T-D0：当前 P 架构的最小 reader-state 训练

仅在 P3 指定的有功能作用层及位置监督状态；从 `Z,Q` 产生 student states，raw-context teacher 需先证明有任务能力，且不得把跨表示 patch 的 rescue 等同于信息存在。训练比较相同预算的 `CE-only`、`CE+答案层蒸馏`、`CE+reader-state loss`（若预算允许再做组合），以 answer CE 与生成 QA 为最终标准。压缩与 raw 的最优中间轨迹不必逐层相同：state distance 下降而 QA 不升，判为本损失失败；QA 提升后再检查 bridge 与多查询复用，仍不自动启动 workspace。

## 7. 交付、统计与停止纪律

每个正式 run 记录 git SHA、模型/adapter/cache 身份、检索与数据清单、tokenizer 和 Transformers 版本、prompt 序列、精确 slot/query/suffix 位置、seed、样本 ID、生成/评分参数。机制诊断用实际 decoder block 前后 hook，final RMSNorm 单独处理；逐例配对 bootstrap 只覆盖给定 checkpoint 的样本不确定性，不覆盖训练 seed 方差。

先在 dev 定样本规则、层和阈值，保留独立样本复核及最后的锁定 test。比较效率时分别报告离线编码与摊薄、多查询在线 prefill、KV/显存、answer latency，以及相同答案质量下的预算。任一新模块都必须在启动前写明“哪个功能测试要求它存在”；不能用 norm/cosine/attention heatmap 的变化代替答案能力。

**实施边界**：本次代码修复仅使已有 P2 probe 的拼写错误、非有限值检查及 gate 语义与本文一致。P0、P1、P3 的新评测与 patch 实现、训练运行仍需服务器按上述顺序完成；没有 GPU 运行结果前，不对 reader bottleneck 或 D2 优越性作结论。
