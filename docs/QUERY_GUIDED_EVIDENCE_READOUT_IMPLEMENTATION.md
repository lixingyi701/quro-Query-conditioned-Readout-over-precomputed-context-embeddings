# Query-guided evidence readout：评审修订、实现与服务器运行协议

日期：2026-10-07。实现基于 projector 分支 `ea23e93`。
原提案：[main / 927557f](https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/927557f/docs/QURO_QUERY_GUIDED_EVIDENCE_READOUT_PROPOSAL.md)。

**结论：接受远程评审的三个实质建议，先完成低成本数据审计和短程架构 pilot，再启动三臂。研究主线仍是增强 query 对最终文档表示的影响，为 QuRO 方法论文提供可解释的增益。**
此处给出可运行的实现；未使用服务器数据、S0/SQ 实际 checkpoint 或 GPU，尚无真实覆盖率、pilot 或 QA 增益结论。

## 1. 对远程评审的判断

| 建议 | 判断与修订 | 对正式实验的影响 |
|---|---|---|
| 冻结 S0 可能限制新读出 | 合理。S0 是 QA 优化结果，不保证适合条件读出；比较冻结和联合优化 f0 | pilot 选择后，A/B/C 全部采用同一种冻结策略 |
| 先核实固定文档池的多问覆盖 | 合理且应立即做。统计有序池、无序池、文档复用、不同问题对应不同目标的比例 | 原始池变化不足时，采用训练集自然问题配对，并重新审计 |
| pilot 加仅第一步 attention | 合理。三臂不能分离第二步贡献 | 增加 first_only；可选 slotwise 保留参数并禁止跨槽读取 |

需要限定三个判断：

1. “2k-sample dev pilot”指在 **train 上短程训练，dev 前 2000 个固定问题上评估**。它需要 GPU 训练；仅数据审计和配对准备不需要 GPU。250 optimizer steps 是起始诊断预算，不是充分训练的保证。
2. 同池多问中位数为 1 是捷径风险信号，不能单凭它推断 query 监督不可能有效。应同时看多问池样本比例、证据目标是否随问题改变及可见事实覆盖。文档复用不等于同池多问。
3. 冻结与联合训练对比改变可训练参数量；first_only 改变计算结构和参数量。这些首先指导实现选择，不单独构成纯机制因果证明。slotwise 具有相同参数数目，但单键 softmax 的 Q/K 无有效梯度，有效容量仍不相同。

评审把 HeadE 称为 embedding 对齐不够准确：现有 HeadE 是最终 E 上的文档分类，Direct-State 才是另一种状态对齐路径。这不改变对内容生成监督的建议。

## 2. 最终结构与原提案的具体化

冻结发布版 PISCO compressor 与 QA reader；原始缓存每文档 `8×4096`，10 篇仍输出 80 槽，不新增在线文档编码。S0 初始化来自对应 seed 的 QA-only、document-only checkpoint。

对每篇文档独立执行：

\[
E_0=Z+f_0(Z),\quad U=W_z\mathrm{LN}(E_0),\quad Q=W_q\mathrm{LN}(H_q)
\]
\[
C=\mathrm{Attn}_q(U,Q,Q),\quad S=U+C
\]
\[
R=S+\mathrm{Attn}_d(\mathrm{LN}(S),U,U),\quad
E=E_0+W_{out}\mathrm{FFN}(\mathrm{LN}(R)).
\]

这是对提案 attention 方程的工程具体化：保留第一步和第二步的瓶颈残差，避免第二步完全覆盖已获得的问题条件。它并未增加跨文档读写或输出槽。`r=256`、8 heads、FFN `256→512→256`，attention dropout 为零。每个 attention 有独立投影参数。

第二步 Value 来自 **同一篇文档的 U**；问题通过 S 影响读取权重，同时经残差保留。相比旧 SQ 的问题特征注入，新增的是同篇内容读取与重组路径。最终输出仍为 native E0 加 delta；不在返回给 reader 的 E 上施加 LayerNorm。

在 `d=4096, m=8, base_hidden=512, r=256, heads=8` 下，冻结 S0 方案新增可训练参数 3,956,992；S0 基座另含 33,587,712 参数。联合方案训练两者。参数数目由模块直接计数，不是运行效率实测。

只将最终 `W_out.weight/bias` 置零，其余新层正常初始化，所以零步 E 严格等于加载的 E0。第一步更新时上游梯度为零是这个初始化的正常结果；Wout 更新后，两个 attention 应有梯度。

padding 文档先从计算批次中移除，再分别进行 attention。这样 padding 的 Q/K/V 均不参与，且没有全 mask softmax。问题 padding 在投影前清零并设置 key padding mask；输出 padding 槽清零。

模式：

| 配置 | 结构 | 用途 |
|---|---|---|
| `evidence_stage=full` | 两阶段，R=S+同篇 attention | 新方法 |
| `first_only` | R=S，跳过第二阶段；第二阶段参数不优化 | 快速结构对照 |
| `slotwise` | 第二阶段每槽只能读自身，禁止跨槽；保留参数 | 补充内容重组对照 |
| `projector_query_mode=agnostic_matched` | 用固定 4 个 query 向量代替真实问题 | C 臂；参数及共有层初始化匹配 |
| `evidence_base_trainable` | 联合优化 f0；reader 仍冻结 | 冻结策略 pilot／必要时正式方案 |

C 不读取真实问题的内容和长度，但 QA reader 仍得到正常问题。固定 4-token 条件的长度与真实问题不同，因此这是问题条件路径的控制，不是在线成本完全匹配的控制。

## 3. 内容监督与数据边界

QA 和 evidence loss **复用一次 readout 得到的同一份最终 E**。辅助 reader 接收全部有效 memory，固定 D0 问题字段为：

```text
Return the supporting evidence.
```

辅助提示没有真实问题、答案、支持文档 ID 或 gold rank，也不筛选 gold memory。目标为缓存 encoder 实际输入前缀内完整出现的 supporting sentences，按照文档／句子顺序组合。辅助生成只用于训练；推理直接 QA，不生成证据。

`L = L_QA + 0.1 L_evidence`，两个 CE 各按其有效 token 求平均。辅助损失在有目标的行上求平均；无有效目标的行只做 QA。默认目标上限 128 tokens（包含 EOS），过长目标整行跳过，**不截断到半句或丢掉后一个 hop**。可在 train 上查看长度分布后统一设置 64/128；A/B/C 的数据文件和预算保持一致。

可见性按发布版输入 `<ENC><bos>document<eos>` 右截断到 131 tokens 的 offset 判断。逐句核实，不沿用文档级“至少一句可见”标志来代表全部支持句可见。部分支持句不可见时，只监督可见句，并记录 partial；这不保证完整回答证据已存于 Z。

准备过程读取原文，但发生在离线标注阶段。训练阶段只读取缓存，证据文本只作为 teacher-forcing 的输出目标；前一目标 token 仅进入 causal reader，不进入投影器。目标前缀本身可能帮助预测后续句子，所以“没有明文问题”是信息路径约束，不能宣称从数学上迫使模型完全使用 memory。最终判断依赖 B 对 A/C 的正常 QA 比较。

目标保存 source/cache/target digest。修改问题、支持句、文档池或缓存后旧注释失效。checkpoint 包含冻结 S0 的全部权重，避免重载时退回随机基座。

## 4. 文件与接口

| 文件 | 功能 |
|---|---|
| `src/evidence_projector.py` | S0 基座＋两阶段文档内 attention、first_only／slotwise |
| `src/evidence.py` | 逐句可见性、目标来源校验、固定辅助指令 |
| `src/model.py` | 严格 S0 初始化、同 E 双任务、梯度路径、完整 checkpoint |
| `src/data.py` | evidence 目标单独组装，错配／超长目标屏蔽 |
| `config.py` / `src/train.py` | 新 preset、参数、full-budget 评估、覆盖日志、样本顺序 hash |
| `scripts/audit_evidence_pools.py` | CPU 数据覆盖率审计 |
| `scripts/build_evidence_pool_pairs.py` | 可选自然问题固定池配对，无新标签／编码 |
| `scripts/prepare_evidence_targets.py` | 离线目标准备，无 LM forward |
| `scripts/run_evidence_experiments.py` | pilot／三臂命令生成；显式 execute 才运行 |
| `scripts/analyze_evidence_comparison.py` | 同 ID／gold 配对 EM/F1，核实训练来源与顺序 |
| `tests/test_evidence_projector.py` | 新路径的 CPU 工程检查 |

## 5. 服务器执行顺序

以下路径均替换为服务器实际位置。运行前用同一环境安装项目依赖。S0 必须是标准移除 query 分支的 S0，不能用 S0m、SQ、HeadE 或 decoder 继续训练 checkpoint 替代。

### 5.1 CPU 审计原始训练集

```bash
python scripts/audit_evidence_pools.py \
  --train_file /data02/quro/data/hotpot/train.jsonl \
  --output /data02/quro/runs/qer/audit_original.json
```

重点看 `ordered_pool`、`unordered_pool` 的 `row_fraction_multi_question` 和 `multi_question_pools_with_distinct_targets`。原始 support 文本变化还不是可见目标变化，准备目标后再审计一次。若只有零星自然多问池，先走 5.2；不把单问池直接当作充分捷径防控证据。

### 5.2 需要时使用现有真实问题配对

```bash
python scripts/build_evidence_pool_pairs.py \
  --train_file /data02/quro/data/hotpot/train.jsonl \
  --max_pairs 2000 --seed 42 --include_original \
  --output /data02/quro/data/hotpot/qer_train_paired.jsonl
```

脚本挑选共享支持文档、问题和支持内容不同的两道真实 train 题；构造共同的 10-doc pool，保留双方全部 gold 文档，从双方原始文档中补足干扰项，同一顺序供两问使用。重映射 gold/support ranks，删除旧注释。所有文档已有缓存，不产生新原文或新答案。输出记录 original_id，训练入口会检查它与 dev ID 的重叠。

共享支持文档上的问题不一定有不同证据句，脚本检查双方整体 supporting sentence 签名不同；仍须在重新准备可见目标后确认目标差异未因截断消失。配对数量不足会明确报错或报告实际数量，不补造标签。只有 train 输入，禁止混入 dev/test。

`--include_original` 的默认建议保持原始分布并追加少量配对，但 4000/90000 的比例本身未必能提供足够约束：根据实际 audit 决定固定的混合比例，记录在实验协议中。可去掉该参数输出仅配对的开发数据；不能把只在配对子集上训练的结果冒充原始主实验。

若使用增强数据，A/B/C 均用相同混合文件。要把 B 与 S0/SQ 的差值解释为结构而非数据收益，还需要 S0/SQ 在相同增强数据和训练预算下继续训练的参考臂；历史 checkpoint 只能作为整体系统参考，不能隔离增强收益。

### 5.3 准备可见目标并重新审计

```bash
python scripts/prepare_evidence_targets.py \
  --train_file /data02/quro/data/hotpot/qer_train_paired.jsonl \
  --corpus /data02/quro/data/hotpot/corpus.jsonl \
  --cache_dir /data02/quro/cache/hotpot-pisco-r16 \
  --pisco_path /data02/quro/models/pisco-mistral \
  --output /data02/quro/data/hotpot/qer_train_visible.jsonl

python scripts/audit_evidence_pools.py \
  --train_file /data02/quro/data/hotpot/qer_train_visible.jsonl \
  --output /data02/quro/runs/qer/audit_visible.json
```

不增强时将输入改为原始 train。准备脚本需要原始 corpus 和 tokenizer，不需要 GPU。核实 `.report.json` 的完整／部分可见比例；正式训练还会输出 `evidence_coverage.json`，报告 token cap 后有效目标、超长和 partial 行。

### 5.4 两 seed 短程 pilot

```bash
python scripts/run_evidence_experiments.py --phase pilot --slotwise \
  --s0 42=/path/to/S0_seed42/checkpoint_last.pt \
  --s0 43=/path/to/S0_seed43/checkpoint_last.pt \
  --train_file /data02/quro/data/hotpot/qer_train_visible.jsonl \
  --dev_file /data02/quro/data/hotpot/dev.jsonl \
  --cache_dir /data02/quro/cache/hotpot-pisco-r16 \
  --pisco_path /data02/quro/models/pisco-mistral \
  --out_dir /data02/quro/runs/qer
```

先打印／保存命令；加 `--execute` 运行。默认每 seed 三个必需臂，`--slotwise` 加一个补充臂：

| pilot 臂 | S0 | 阶段 | loss |
|---|---|---|---|
| A-frozen | 冻结 | full | QA |
| A-joint | 联合优化 | full | QA |
| A-first | 冻结（默认） | first_only | QA |
| A-slotwise | 冻结（默认） | slotwise | QA |

每臂同一 seed 从同一 S0 fresh start，默认 250 optimizer steps、batch 2×grad_accum 8；验证 step0/125/250，固定 dev2000、预设 F1 选择指标。未使用 dev 进行训练。比较同 seed step0 的 QA 相等；比较同 step 的样本顺序 hash 相等。不要用只有 loss 曲线的 pilot 判断冻结策略。

如果联合训练在两个 seed 的正常 dev QA 上方向一致地更好且训练稳定，正式三臂统一联合训练；若方向混合或差异微小，延长同预算 pilot，不能当作冻结假设已成立。若选 joint，`--base joint` 重跑相同 full／first_only／slotwise 比较，使第二步判断也基于选定基座策略，使用新的 out_dir。

短程 pilot 用来避免明显失败和选结构，不用于证明微小最终收益，不以它的最好点充当全量结论。first_only 不弱于 full 时，应优先保留更简单版本或继续核实优化，而不是先写“第二步有效”。

### 5.5 B 辅助路径的真实 reader 预检

正式三臂前，可单 seed 用正式 B 命令加 `--steps 10`，新 out_dir；只检查有限 loss、目标覆盖、梯度和显存。CPU toy 检查不能验证发布版 reader 能否执行辅助指令。查看少量 train 样本的目标／预测可帮助发现纯输出格式错误；这是工程预检，不做额外大规模诊断研究。

双任务增加一次 reader 训练 forward，显存和耗时需要实测。出现 OOM 时把 **所有正式臂** 的 microbatch 同步改小，并用 grad_accum 保持有效 batch；记录变更。不得仅给 B/C 改样本顺序或训练步数后称为相同预算。

### 5.6 正式三臂

将上面的 `--phase pilot --slotwise` 改为 `--phase formal --base frozen`（或 joint），使用新的实验目录，加 `--execute`。

| 臂 | 真实问题进入 readout | evidence loss | 核心对比 |
|---|---|---|---|
| A | 是 | 0 | 结构＋QA |
| B | 是 | 0.1 | B−A：内容监督增益 |
| C | 固定向量 | 0.1 | B−C：辅助目标下的问题条件增益 |

默认 full、3000 steps、两个 seed、同数据／批次／scheduler／S0 初始来源。**正式臂重新从 S0 初始化，不继续各自 pilot 最优点**。B−C 不是 query×auxiliary 的交互项；交互主张需要 query 无关且无 auxiliary 的第四臂。

三臂在 dev 完成结构和超参数选择后，再用固定 last checkpoint 或事先约定的 dev-selected checkpoint 统一评估全量 test。runner 只传 dev，不在 pilot／调参中自动打开 test。最后测试例：

```bash
python src/train.py --preset pisco_evidence_projector \
  --train_file /data02/quro/data/hotpot/qer_train_visible.jsonl \
  --eval_files test=/data02/quro/data/hotpot/test.jsonl \
  --cache_dir /data02/quro/cache/hotpot-pisco-r16 \
  --generator_path /data02/quro/models/pisco-mistral \
  --resume_from /data02/quro/runs/qer/formal/seed42/B/checkpoint_last.pt \
  --evidence_loss_weight 0.1 --eval_only --eval_max_samples 999999 \
  --out_dir /data02/quro/runs/qer/test_seed42_B
```

联合基座时加 `--evidence_base_trainable`；C 加 `--projector_query_mode agnostic_matched`；A 使用 weight 0。架构字段必须与保存时匹配。重载包含冻结基座，无需再次传 S0 checkpoint。不要同时传 s0_checkpoint 和 resume_from。

### 5.7 同题配对比较

```bash
python scripts/analyze_evidence_comparison.py \
  --baseline /data02/quro/runs/qer/formal/seed42/A \
  --candidate /data02/quro/runs/qer/formal/seed42/B \
  --output /data02/quro/runs/qer/seed42_B_minus_A.json
```

再运行 B 对 C。脚本核实同题 ID／gold，拒绝静默取交集；报告 EM/F1、百分点差和 paired bootstrap CI，检查 S0 hash、train 文件 hash、cache digest、样本顺序 hash。跨 seed 分别比较，报告两 seed 而不是把训练 seed 当独立问题扩充 n。

test 评估目录保存训练时的 provenance；指定 `--split test` 比较。若对历史 S0/SQ 使用 `--allow_unmatched_training`，报告保留不匹配字段，不把它解释成隔离结构因素的对照。

当前训练 resume 沿用旧入口，未恢复 DataLoader 游标／RNG 状态；其样本序列不能宣称与未中断训练逐步一致。用于比较的 pilot 和正式臂应 fresh start；中断后重新跑或独立记录复现偏差。

若采用 dev-selected best checkpoint，不同臂可能选在不同 step，因此 checkpoint 中记录的样本顺序前缀 hash 不同。主要归因优先比较统一 last；best 比较另按相同预设选择规则报告，并显式解释前缀差异，不能把它误称为同一步配对。

## 6. 可写入论文的边界与交付状态

新模块有合理结构区别和更密集的内容监督；是否构成有价值的 QuRO 改进取决于实际 B−A、B−C 和正常 QA 增益。暂不宣称事实恢复、不可恢复性、严格证据忠实性，亦不把已知 +0.37 F1 作为这版方法效果。

工程检查覆盖恒等起点、padding／query mask、文档独立性、固定问题控制、辅助梯度、冻结 reader、冻结 S0 保存重载、目标失配屏蔽和训练入口。2026-10-07 CPU 回归运行 `python -m unittest discover -s tests -q`，131 项通过；环境为 Python 3.12、torch 2.5.1+cpu、transformers 4.44.2、peft 0.12.0。其中包括小型随机 bf16 Mistral＋两个 PEFT adapters 的 QA/evidence 反传冻结检查，**没有使用发布版权重**。`compileall` 与 `git diff --check` 通过。真实数据审计、发布版 B 预检、两 seed pilot、正式训练、效率与 test 均待服务器执行。
