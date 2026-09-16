# warning_and_target：下一步代码设计与实验运行指导

> 日期：2026-09-16  
> 审查基线：`13877aeb8e10a868e6ab594b54236f01727fd290`  
> 范围：根据 docs/、src/、scripts/ 与 results/ 的静态核查及已上传结果制定下一步工作。未访问服务器原始 checkpoint/config/predictions，未复跑训练。  
> 状态：本文是实施要求，不代表下列修复、基准测试或实验已经完成。

## 0. 目标与执行原则

**项目目标是获得可信、可复现的质量或效率收益，不预设 xattn、Perceiver、多槽位或其他具体模块必须有效。** 如果简单规则已经足够，允许简化方法、调整贡献定位；若更复杂的模块在明确条件下有效，再保留它。模块的解释应服从实验结果。

目前已有可用的高压缩工作点，但尚未证明可学习 readout 对简单基线有稳定增量。下一步首先修正对照定义、query encoder 状态与成本计量，然后做小规模有效性验证，避免扩大无法解释的实验。

本文对 HANDOFF 与历史结果解释作以下限定：

- 暂缓“机制已经证实，只剩定位问题”“C≈A 证明 decoder 替代 query 条件化”等结论。
- 历史结果保留，不删除或重命名成已经修正后的实验；先按真实配置重新标注。
- 暂不微调离线压缩器，目的是控制变量与减少重建缓存成本，并非因为微调必然破坏可缓存性。
- 本轮无需同时实现所有可选改进。优先级为：正确对照 → 稳定表示/完整计费 → 收益验证 → 扩展论文实验。
- 允许调整研究路线，但必须保留负结果，不能事后更换指标或测试子集来制造收益。

## 1. 当前证据与尚不能推出的结论

| 已报告实验 | 结果 | 当前可用解释 |
|---|---|---|
| m=8 PISCO，D0，B=8 | C EM 70.90，S 71.70 | 尚未证明 C 优于余弦 top-B |
| m=8 PISCO，P vs C | 平均 43.8→8 soft tokens；EM 70.75→70.90 | 有值得研究的质量—长度工作点；不是已证明非劣效或实际加速 |
| m=32 切块 PISCO | C EM 68.55，A 69.00 | 两个当前实现接近；A 是否真正 query 无关待核实 |
| m=32 COCOM | C EM 66.10，A 68.95 | C 的 EM 显著退化，不能因 substring 不显著就抹去 |
| D1，qdrop=1.0 | C EM 26.75，A 5.90 | 问题信息通路存在明显差异；不直接证明证据选择机制 |

数据见 [xi_off_sweep.json](../results/xi_off_sweep.json)、[d1_query_dropout.json](../results/d1_query_dropout.json)。

注意：results/ 的两个文件是汇总记录，不是足以重建全部归因的原始运行包。

## 2. P0：必须先检查的代码与实验契约

### W1. agnostic 臂可能仍通过余弦先验读取 query

当前默认 `cosine_prior=True`。`--output_query_mode agnostic` 只关闭 output slots 的显式 query 输入；`QuROModel.readout_cached()` 仍会因 `uses_cosine_prior` 调用生成器获取 query vector，并通过 `cosine_bias()` 改变注意力。

`scripts/run_gonogo.sh` 中 A 臂没有传 `--no_cosine_prior`。因此按当前脚本运行的 A 应定义为“仅余弦条件化”，不能称为完全 query 无关。历史运行可能用了额外参数，必须检查原始配置后确认。

**操作要求：**

1. 找出历史 A/C 各臂的 config.json、代码 commit、启动参数及 checkpoint 配置。
2. 对照 `cosine_prior`、`output_query_mode`、`adaptive_budget`、query encoder 与 LoRA 设置。
3. 固定同一份 latent、document_mask 和 B，只替换 query，测试真正 A0 的输出是否不变。
4. 该不变性只约束 readout 对固定输入文档的行为；上游检索本身仍可依赖 query。

重新定义：

| 臂 | 余弦 query 先验 | output slots 的真实 query 输入 | 目的 |
|---|---|---|---|
| A0 | 关闭 | 关闭 | 真正 query 无关读出 |
| A1 | 开启 | 关闭 | 仅余弦条件化 |
| C0 | 关闭 | 开启 | 仅学习式条件化 |
| C1 | 开启 | 开启 | 当前完整方法 |
| S | 余弦 top-B | 无可学习 slots | 简单非参数读出 |
| P | 无 | 无 | 全部缓存 latent 直喂参考 |

现有 CLI 的 A0 对应 `--output_query_mode agnostic --no_cosine_prior`，且固定 B、关闭 adaptive budget。

**参数量说明：** 当前 agnostic 不创建 xattn query block，因此与 C 并非严格同参数量。若做机制归因，保留相同 query block，输入固定的 query 无关占位序列及固定 mask；其长度不能随真实 query 长度变化。若只做系统比较，可以保留简化 A0，但必须报告实际参数量与成本，不能声称“唯一差别是看不看 query”。

### W2. no_residual_readout 并不是去掉 Delta

当前：
`E = base_scale * (alpha @ Z) + Delta`。

`--no_residual_readout` 实际返回 `Delta`，去掉的是 raw latent pooling 旁路，并改变 out_proj 的初始化策略。因此不能用这个开关执行 HANDOFF 所述“去掉 Delta”的实验。

建议新增清晰输出模式（以下为待实现接口）：

| output mode | 输出 |
|---|---|
| full | s·alpha·Z + Delta |
| pool_only | s·alpha·Z |
| delta_only | Delta |

- 先用同一 checkpoint 做推理分支干预，测当前依赖；再重训必要变体，测移除后的可补偿能力。
- 对重训变体明确记录初始化，避免将初始化损伤误判为分支必要性。
- 保留旧参数兼容时必须写清其等价模式。
- Delta 有效并不要求废弃 readout 名称；只需避免将输出描述为纯 latent 选择，或将 alpha 当作整个模型的完整因果解释。

### W3. query encoder 与 decoder LoRA 共享：no_grad 不等于权重冻结

`GeneratorQueryEncoder` 持有与 decoder 同一个 lm 对象。query 前向使用 no_grad/eval，但 decoder LoRA 在答案损失下更新，因此后续 query 表示也可能变化。

这不是自动成立的训练 bug，而是**与“固定 query encoder”描述不一致的设计选择**。它可能影响：
- query—缓存 latent 的几何关系和先验；
- 各臂的可比性；
- 训练中缓存 query embedding 是否有效；
- 精度增益到底来自 readout 还是 decoder/query 表示共同适配。

区分三个状态，不要共用一个 freeze 布尔值表达：

| 状态 | 准确定义 |
|---|---|
| 无 query 路径反传 | query 前向 no_grad，但共享权重可能被其他损失更新 |
| 固定 query 表示函数 | query 使用的 backbone、adapter 和其他影响输出的参数均不更新 |
| 固定离线文档编码器 | 文档缓存对应的 encoder 版本固定；与上述两者不是同一件事 |

**下一步推荐：增加显式 query 表示策略。**

1. `fixed_adapter`：共享冻结 backbone，query 使用单独的、冻结的初始 adapter 副本；decoder 使用可训练 adapter。无需预设复制整个 7B。
2. `shared_current`：保留当前行为，明确标为共享当前 decoder adapter，作为对照。
3. 如果考虑关闭 adapter 后编码 query，这是另一种表示方案，不能默认与原空间等价。

实现注意事项：
- adapter 切换可能改变 requires_grad；必须验证切换前后参数可训集合与 optimizer 参数身份不变。
- 完成 query 前向后恢复 decoder adapter 和 train/eval 状态；不能静默改动下一次生成/反传。
- 固定 query adapter 必须随 checkpoint 保存或通过不可变 revision+hash 精确恢复，不能依赖可变下载路径。
- 共享模型的临时 adapter 切换不适合无锁并发请求；服务并发应采用隔离执行或独立实例，并计入内存成本。

**验收：**
- 对固定 query，训练若干 decoder 步后，fixed_adapter 的表示在规定数值容差内不变。
- query 冻结参数 hash 不变，decoder adapter 在训练后确有变化。
- checkpoint 保存/恢复后，相同 query 的表示可复现。
- 各臂使用同一 query 表示策略；S 的 top-B 规则无可训参数，不等于整个 S 系统无可训参数。

### W4. mismatch-query 要只干预目标路径

当前 `QuRODataset.query_shift` 同时改变 query_ids、query_gen_ids 与 decoder prompt 的 query。D0 的准确率下降不能单独归因于 readout。

增加两个独立字段：
- `readout_query`：进入 query encoder、先验、output query 的问题；
- `decoder_query`：进入生成器 prompt 的真实问题。

主诊断：固定文档和答案，decoder_query 保持正确，只替换 readout_query。错配问题优先来自同一文档集合内的其他问题，并记录是否碰巧共享答案。

### W5. 多预算不是“任意 B”，混合预算批次有额外风险

当前 output slots 受 max_budget 限制，配置会过滤超过 max_budget 的 buckets。扫描到 16 必须显式设置 `--budget 16`，不能只改 buckets。

当前混合预算处理按 batch 最大 B 生成全部 slots，然后遮掉较小预算的尾部；slot self-attention 没有预算掩码。因此一个 B=4 样本可能受同批 B=16 样本影响，其前四个输出不一定等于单独 B=4 的读出。

要求：
- 第一阶段固定预算或按预算分组执行，消除该变量。
- 上自适应预算前，实现预算分组或正确的逐样本 slot mask。
- 测试同一样本单独执行与混合预算批次执行的输出一致性。
- budget dropout 训练的是多预算能力，不自动保证输出嵌套、跨预算切片等价或未训练预算的性能。

## 3. P0：重新建立完整成本计量

### 3.1 单次在线成本

统一报告：
`T_online = T_cache_load + T_query_encode + T_readout + T_prefill + T_decode`。

若从检索前开始计时，则另含 T_retrieve；必须声明边界。GPU 阶段可能重叠，最终以端到端墙钟时间为准，分段耗时用于解释。

**不能漏记 7B query 编码。** 当前 QuRO 并非仅运行一个几十 M 的在线 readout。若固定 query encoder 后在训练时缓存问题表示，属于训练优化；服务新 query 时仍需要编码，除非单独声明精确 query 缓存命中率。

### 3.2 评测循环不等于 serving benchmark

当前 evaluate() 在 generate_answer() 后再次调用 readout_cached() 以收集统计。直接计整个 evaluate 的耗时会重复计算 query/readout。

待实现：
- serving 路径每条 query 只编码和读出一次，并返回供统计复用的中间计量。
- 独立 benchmark，排除文件输出、指标计算和额外诊断前向。
- GPU warmup、同步计时；TTFT、总延迟、吞吐按实际请求协议测量。
- 同一硬件、dtype、batch/concurrency、输出上限与解码策略；报告实际生成长度。
- 同时报 batch=1 时延与批量吞吐；报告热/冷缓存、峰值显存与 CPU/I/O 成本。
- 主表测真实流水线；诊断时可复用统一 query embedding 隔离模块成本，但不能把这当作服务总成本。

### 3.3 摊薄比较必须包含简单基线

`Total(Q) = Build + sum_q Online(q)`，并单列文档更新重压缩成本及存储量。

至少比较 C1、S、A0、P；再与 SeleCom 等需要在线处理文档的方法比较。

- C1/S/A0 共用同一文档缓存时，离线成本基本相同。C1 若质量无增量且每 query 更贵，文档复用不会自动逆转劣势。
- 与其他方法的交叉点仅在在线成本差为正、质量约束满足时有意义；不能预设“个位数查询必回本”。
- 区分全库预构建和按需构建；报告总文档数、实际命中文档数、文档命中频次分布、更新频率及缓存状态。
- 减小在线 B 不会缩小已经持久化的 m×h 文档缓存。

### 3.4 三种压缩指标分别报告

| 指标 | 定义/用途 |
|---|---|
| evidence token ratio | 实际送入离线编码器的原文 token 数 / 输出 soft token 数 |
| decoder 输入长度 | system、分隔符、问题、soft token 等全部计入 |
| storage bytes | 文档缓存真实字节数，不用 token 压缩率代替 |

已有 43.8→8 的证据 token 缩减，对应 prefill 约 111→71，不能称为同比例加速。原文截断前后 token 数、答案证据截断情况也应记录。

## 4. P1：最小收益验证矩阵

### 4.1 第一轮：先固定变量

默认保留冻结 PISCO 文档缓存、D0、相同数据、相同 decoder 初始化与训练预算。选定 fixed_adapter 策略，先跑 B=8 的 A0/A1/C0/C1/S/P。

两类问题分开：
- **系统收益：** 各臂按同样资源约束合理训练 decoder LoRA，比较整套系统质量与时延。
- **机制诊断：** 固定 decoder/query 表示或用同 checkpoint 干预，减少共同适配的混淆。

不要把前者描述为 readout 单模块的纯因果增益。记录 S/P 是否也训练 decoder；“0 参数”只能描述其读出规则。

### 4.2 第二轮：预算与任务条件

确认运行契约后，主要比较 A0、A1、C1、S：
- B ∈ {1,2,4,8,16}；训练 max_budget=16，记录各 bucket 更新次数。
- 若共享预算训练在某点明显不佳，补一个单预算训练对照，区分共享训练折衷与结构瓶颈。
- 保留 TriviaQA 作为历史对齐。
- 加 NQ 与 HotpotQA 小规模诊断；通过后再做正式全量评测。
- 构造同一文档集合多个问题的诊断，问题分别依赖不同事实。
- 加同主题干扰及标准检索 top-k；当前 gold+随机干扰与真实检索评测分表报告。

不预设 C−A 随 B 单调增大。极低预算可能让所有方法失败，最大收益可能出现在中间预算。

### 4.3 核心读出增量之外的可接受方向

若 C1≈S：
- 检查 A0 是否已经足够；有依据时删掉昂贵 query 编码或复杂模块。
- 允许把简单选择/池化作为主方法候选，研究其适用边界与质量—成本曲线。
- 继续保留复杂模块的前提是稳定质量增益、更好鲁棒性或可度量的其他价值。
- 不能只赢过高成本外部基线却略去表现相同、成本更低的 S。

## 5. 实验解释与基线警告

1. **D1 只作诊断。** 看得到问题与完全看不到问题的系统信息不对称；C>A0 不能自动证明证据选择。原 D1 还受 A 定义问题影响。D1 绝对精度大幅低于 D0，不能仅凭缩短 prompt 当作主贡献。
2. **不显著不等于等价。** 报差值与配对置信区间；欲主张质量保持，实验前约定可接受退化范围。EM 显著退化而 substring 不显著时分别报告，不合并成“无差异”。
3. **错配文档不等于严格闭卷地板。** 不相关证据可能伤害生成。保留无文档与错配文档两种控制；二者均为诊断，不是信息论上下界。
4. **逐样本优化 soft prompt 不是信息容量证明。** 用 gold 优化成功只表明找到了可诱导答案的向量；失败可能是优化限制。降低优先级，不作为 go/no-go 或“必须加 B”的依据。
5. **COCOM-128 是参考点，不是自动匹配的对照。** 122× 与 88× 不相同，checkpoint/训练/decoder 也不同。应报告实际 token 预算与各自合理工作点，不能单凭胜负归因两段式结构。
6. **共享缓存是继承能力。** PISCO/COCOM 与 S 都可复用缓存；“一份缓存、多 B、摊薄曲线”不是 QuRO 独占。贡献必须落在可检验的增量。
7. **冻结不等于通用。** 冻结 PISCO 仍可能与特定模型表示空间绑定。训练后 query 无关的压缩器仍可缓存；跨模型/readout 可迁移性要单独验证。
8. **文献区别不代替实验。** RRK 面向重排，最佳配置会微调压缩器；ArcAligner 的内部对齐与外部预算缩减不同，但胜过它不能独立归因为 query 条件化。
9. **开发集与测试集分开。** 不断检查的 TriviaQA 2000 条应视作开发诊断依据；锁定新的独立测试协议。按文档/问题检查去重，必要时报告已见/未见文档子集；语料共享本身不是自动泄漏。
10. **先约定主要指标。** 按数据集采用适用的 EM/F1，substring 辅助解释；多 seed 与配对统计各解决不同不确定性，不能互相代替。

## 6. 代码改动清单与验收

| 优先级 | 文件/模块 | 改动与验收 |
|---|---|---|
| P0 | config.py、src/model.py、src/generator.py | 显式 query 表示策略；冻结 adapter 状态、恢复与 hash 验收 |
| P0 | src/model.py、src/readout.py、src/perceiver.py | A0/A1/C0/C1 定义；真实 A0 对 query 不变；记录真实参数量 |
| P0 | src/readout.py | full/pool_only/delta_only；分支输出契约与初始化记录 |
| P0 | src/data.py、src/model.py | readout_query 与 decoder_query 分离；单路径干预 |
| P0 | src/train.py、独立 benchmark 脚本 | 一次 query 编码/读出；完整在线计费，避免重复统计前向 |
| P1 | scripts/run_gonogo.sh | 按新臂定义启动；显式保存完整配置、commit、标签，旧运行不覆盖 |
| P1 | src/model.py、预算相关代码 | 先预算分组；验证单样本/混合预算输出一致 |
| P1 | 数据准备与结果汇总 | 固定文档多 query；检索与证据覆盖；置信区间与多 seed |
| P2 | docs/HANDOFF.md、结果分析 | 在确认历史配置后标注失效解释与新结论 |

只新增针对上述风险的必要测试，不以“shape 正确、无 NaN”替代语义契约。

## 7. 运行归档要求

每个正式运行至少保存：
- run_id、代码 commit（若工作区有改动，另存 diff）、完整 config 与实际启动参数；
- 数据 split/版本/hash、缓存 manifest/hash、压缩器及 query/decoder adapter 版本；
- seed、硬件、dtype、训练样本/更新次数、各 B 采样次数；
- readout/query encoder/decoder 的实际可训参数量与共享关系；
- 每题 id、预测、gold、指标、B、输入长度及对照类型；
- 质量聚合、配对比较、计时协议与性能统计。

服务器大文件沿用 /data02/quro/。GitHub 保存轻量配置、汇总与可追溯索引；逐题数据按许可和体积选择归档位置。现有 results/ 汇总不能替代上述记录。

## 8. 推荐执行顺序与决策门槛

时间仅为任务排序参考，不是未经测量的运行耗时承诺。

| 阶段 | 任务 | 决策 |
|---|---|---|
| 1 | 审计历史 config；完成 W1–W4 与最小计费改造 | 旧结论哪些成立，哪些需重跑 |
| 2 | 固定 query 表示下 B=8 最小臂矩阵 | 余弦、学习式条件化、共同适配各值多少 |
| 3 | 多预算 + 固定文档多问题 + NQ/HotpotQA 诊断 | 在哪些真实任务条件下有收益 |
| 4 | 完整在线计时与复用曲线，必须包含 S/A0/P | 质量收益是否值得额外计算 |
| 5 | 优胜方案多 seed、独立测试、必要外部基线 | 是否扩大成论文 |

**继续条件：** 在事先规定的质量/成本约束下，至少一条路线获得稳定、可解释、可复现的增量。质量提高但成本增加，应报告权衡；质量近似而成本下降，应有非劣效依据和实际计时。

**转向条件：** 复杂 C 与 S 接近且更贵，优先简化或重定位，不继续盲目堆结构。

**暂不扩实验条件：** 控制修正后无稳定质量收益，完整计费也无成本优势。先解决研究问题，不用更多消融数量替代贡献。

## 9. 论文目标

候选主张：**在可复用文档表示上，以预算受控的在线读出改善重复查询场景的质量—成本权衡。**

CCF C 投稿潜力取决于证据，而非模块名或代码量。建议最终形成：互补 QA 数据集、可信简单基线、多预算曲线、复用成本、关键消融、多 seed 和可追溯运行记录。不要求每个设计都有效，但正文必须如实说明有效的是哪一部分、适用条件是什么。

## 10. 核查入口

- [HANDOFF](HANDOFF.md)
- [当前结果与分析](QURO_V0.2_RESULTS_AND_ANALYSIS.md)
- [实验设计](QURO_EXPERIMENTAL_DESIGN.md)
- [模型与 query 编码](../src/model.py)
- [读出与输出分支](../src/readout.py)
- [query slots 实现](../src/perceiver.py)
- [训练与评测](../src/train.py)
- [启动脚本](../scripts/run_gonogo.sh)
- [RRK 原文](https://arxiv.org/html/2604.26483v1)
- [SeleCom 原文](https://arxiv.org/html/2602.15856v1)
- [ArcAligner 原文](https://arxiv.org/html/2601.05038v1)
