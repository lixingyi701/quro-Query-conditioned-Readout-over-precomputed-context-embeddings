# 交接状态（截至 2026-09-17）

> **2026-09-17 训练方案复审：** 下一步实施前请读 [TRAINING_STRATEGY_REVIEW_AND_PLAN.md](TRAINING_STRATEGY_REVIEW_AND_PLAN.md)。它收窄了本文“预算已排除、只剩两个原因、分支干预可决定损失位置”的表述，并提出固定 query 表示、分组优化、CE/KL/证据覆盖的受控实验矩阵。下文历史分析保留；该新文档是待验证建议，不是已运行结果。

新会话从这里开始。本文合并了原先的 HANDOFF 与 `warning_and_target.md`——后者的实施要求已有 W1/W2/W4/W5 完成，留着两份只会让人读到作废的结论。**结果不在本文。**臂定义与预算曲线见 [`ARM_MATRIX_RESULTS.md`](ARM_MATRIX_RESULTS.md)；
训练配方、蒸馏与输入压缩见 [`TRAINING_RECIPE_RESULTS.md`](TRAINING_RECIPE_RESULTS.md)。

---

## 1. 三十秒版本

QuRO = **离线 query 无关压缩（复用冻结 PISCO/COCOM）+ 在线 query 条件读出 → B 个 soft token → 冻结 Mistral + LoRA**。

截至目前，三件事已经定了：

1. **可学习读出确实赢 0 参数的余弦规则**——但只在低预算区间。HotpotQA B=8：**+5.60**（p=1.2e-08，同为 `fixed_adapter`）；B=16：+2.75；**B=32：−0.45（n.s.，S 反超）**。
   > 曾公布的 +6.15 虚高约一分：S 的余弦先验此前在 train 模式下计算（dropout 生效）。见 [`TRAINING_RECIPE_RESULTS.md`](TRAINING_RECIPE_RESULTS.md) §2。**`bs8S_S` 及所有 A1 数字已作废。**
2. **PISCO 原方法 P 仍领先 7.70 分**（46.80 vs 54.50），而 prefill 只省 1.53×。这是当前最大的未解决问题。
3. **KD 的收益只经过 readout。**λ=0.5 对 C1 是 +1.00 EM（n.s.）/ +1.85 substring（p=0.022），对 S 是**精确的零**（p=1.00）——所以它不是通用训练收益。
4. **system prompt 可以整个删掉**（−0.05 EM，p=1.00），**问题压缩掉 10 分但证据利用反而上升**（地板 22.55→5.5，证据值 20.55→27.35）。
5. **单纯加预算追不回差距。**B 翻 4 倍只多榨出 2.4 分证据值。注意措辞：这不等于"容量已够"——更大的 B 可能需要更好的槽位分工或训练（见复审 §3.1）。分支干预已排除"输出是自由分支合成的"这一解释：`pool_only ≈ full`。

数字与检验见上述两份结果文档，原始数据见 [`results/arm_matrix.json`](../results/arm_matrix.json)（37 次运行 + 24 组配对检验）。

---

## 2. 执行原则

**项目目标是获得可信、可复现的质量或效率收益，不预设 xattn、Perceiver、多槽位或其他具体模块必须有效。** 如果简单规则已经足够，允许简化方法、调整贡献定位；若更复杂的模块在明确条件下有效，再保留它。模块的解释应服从实验结果。

- 历史结果保留，不删除、不重命名成修正后的实验；先按真实配置重新标注。
- 允许调整研究路线，但必须保留负结果，**不能事后更换指标或测试子集来制造收益**。
- **v0.3 之前不微调压缩器。**目的是控制变量与减少重建缓存成本，不是因为微调必然破坏可缓存性。

### 报结论的纪律（都是踩过的坑）

- **检验先于结论**——n=500 时 C 领先 1.6 分，n=2000 时 99 胜 101 负。
- **EM / substring / F1 三个一起报**，方向不一致就分别报告，不合并成"无差异"。
- **任何精度数字旁边必须有无证据下界**（错配文档或闭卷）。没有地板的精度数字无法解读：TriviaQA 的 D0 里有 59 分来自闭卷。
- **不显著 ≠ 等价。**报差值与配对置信区间；欲主张质量保持，实验前约定可接受退化范围。
- **跨输出格式的基线不能用 EM**（差的是啰嗦程度不是正确率）。
- **多 seed 与配对统计各解决不同不确定性**，不能互相代替。目前全部结果是单 seed。
- **开发集与测试集分开。**TriviaQA 那 2000 条已是开发数据；HotpotQA 的 dev/test 已切开，test（5405 条）未动过。

---

## 3. 代码契约：已修的坑

这几条原本是 `warning_and_target` 的 P0，现已全部实现并有契约测试覆盖（`python tests/test_shapes.py`，113 项）。**留在这里是因为它们描述了当前代码的语义，不是历史记录。**

### W1 ✅ 臂的四格定义

query 有**两条独立通路**进 readout：余弦先验（`cosine_bias()` 给第一层注意力加偏置）和学习式条件化（query 进 output slots）。`--output_query_mode agnostic` 只关后者。

历史上每个 A 臂都带 `cosine_prior: True`，所以全是 A1，真正的 A0 从未跑过。

| 臂 | 余弦先验 | query 进 output slots |
|---|---|---|
| A0 | 关 | 关 |
| A1 | 开 | 关 |
| C0 | 关 | 开 |
| C1 | 开 | 开 |
| S | 余弦 top-B，无可学习槽位 | |
| P | 全部 latent 直喂 | |

定义在 `config.ARMS`，用 `--arm` 启动。`arm_label()` 把**实际实现的臂**写进 `result.json`，标错的启动会在结果文件里现形。

**参数量**：plain `agnostic` 不创建 query cross-attention block，因此与 C **不是**同参数量。做机制归因时用 `--agnostic_param_matched`（`agnostic_matched`：保留同一个 block，喂固定的可学习占位序列，长度固定所以不泄漏真实 query 长度）。

### W2 ✅ 输出分支与初始化解耦

旧的 `--no_residual_readout` 同时改了输出分支**和** `out_proj` 的初始化，所以它的结果两边都归因不了。现已拆开：

| `--readout_output_mode` | 输出 |
|---|---|
| `full` | `s·α·Z + Δ` |
| `pool_only` | `s·α·Z`（纯选择——"readout"这个名字靠它撑着） |
| `delta_only` | `Δ`（自由分支——输出是合成的） |

`--out_proj_init {zeros,default}` 独立控制初始化。`readout_cached(output_mode=...)` 支持**同一 checkpoint 上的推理期干预**：它测的是"训好的模型当前依赖什么"，重训某个分支测的是"移除后能补偿多少"，两者回答不同问题、不能互相替代。

### W4 ✅ 两条 query 路分离

`query_shift` 原本同时改 readout 的问题和 decoder prompt 的问题，所以错配下降无法归因。现在拆成 `readout_query_shift` / `decoder_query_shift`，评测变体：

- `mismatch-q`：**只**换 readout 的问题，decoder 仍拿对的 → 下降可归因于读出选错证据
- `mismatch-q-both`：旧的联动版，保留以对齐历史
- `mismatch-doc`：换证据，留问题和答案 → **证据地板**

逐题 dump 记录两个问题，以及换进来的问题是否碰巧共享答案（`mismatch_shares_answer`）。

### W5 ⚠ 部分：固定预算已支持，预算分组仍未做

混合预算下 slot 按批内最大 B 生成、小预算事后遮掉，而 **slot self-attention 没有预算掩码**，所以同批的 B=8 行产出的不是它单独运行时的结果。

`FIXED_BUDGET=1 BUDGET=N bash scripts/run_arms.sh ...` 给出单预算训练+评测（关 dropout），B 扫描必须用它。**上自适应预算前，仍需实现预算分组或逐样本 slot mask**，并测单样本执行与混合批执行的输出一致性。

`--budget N` 必须显式传：只改 `budget_buckets` 会被 `max_budget` 过滤掉。

### W3 ⚠ query 表示策略（已实现，但只在单 adapter 主路径上验证）

`GeneratorQueryEncoder` 与 decoder 共用同一个 `lm`，`no_grad` 挡得住梯度、挡不住漂移——decoder 的 LoRA 一更新，query 编码出来的东西就变了。**此前每一次运行都带着这个混淆。**

`query_encoder.representation` 现在把两种状态分开命名，不再和 `freeze` 挤在一个布尔值里：

| 取值 | 含义 |
|---|---|
| `shared_current` | query 走当前的 decoder adapter。历史行为，**仍是默认**，老运行照常复现 |
| `fixed_adapter` | 构造时复制一份已发布 adapter 并冻结，query 前向临时切过去 |

CLI：`--query_representation {shared_current,fixed_adapter}`。

**哪个更好是实验问题，不预设。**`fixed_adapter` 去掉的是一个混淆，不是保证涨点。

契约测试（CPU stub）量化了这个混淆：decoder 训 5 步后，`fixed_adapter` 的 query 表示变化 **0.00e+00**，`shared_current` 变化 **7.11e-01**。

**已知边界**：只支持**恰好一个激活 adapter**——`peft.PeftModel.set_adapter` 收单个名字而 transformers 的收列表，多 adapter 集合无法可移植地恢复，所以构造时直接拒绝而不是只恢复第一个。

实现上四处必须注意：

- **`set_adapter` 会把目标 adapter 的 `requires_grad` 设成 True**（PEFT 文档明说）。切换前后都做 `requires_grad` 快照恢复，否则 optimizer 拿到的可训集合会变——而它持有的是 Parameter 对象，症状是 **decoder 静默不训练**，不报错。
- **`peft.PeftModel` 与 transformers 的 `PeftAdapterMixin` 接口不一致**：`add_adapter` 参数顺序相反，`active_adapters` 一个是属性一个是方法。PISCO 的 decoder 是后者。代码按实际签名分派。
- **所有需要生成器空间 query 向量的臂必须走 `QuROModel.query_vector()`。**A1/S 的 `needs_query=False`，不走 encoder 的 forward；旧代码在那里回退到裸 LM 上的 `pool_query_in_generator_space`，**绕过 adapter 切换和 eval 保护**。真实 PISCO 上实测：C1 用 `quro_query_adapter`/eval，而 A1、S 用 `decoder_adapter`/train（LoRA dropout 生效），运行记录却都写 `fixed_adapter`。**这两个臂正是 C1 的对照**，所以这不是标签错误而是对比被污染。
- **冻结副本随 checkpoint 保存并校验。**它不可训练所以被 `save()` 的过滤漏掉，且**无法从模型路径重建**——`generator_lora_init="random"` 会在复制之前重置 decoder adapter，本地目录也不是不可变版本。现在权重进 checkpoint，hash 不符报错，表示策略两个方向不符也报错（`fixed_adapter` 的 checkpoint 装进 `shared_current` 的模型是另一个系统）。

### 训练配方（2026-09-17 新增）

| 开关 | 作用 |
|---|---|
| `--decoder_lr` | readout 从零开始、decoder LoRA 从 PISCO 热启动，此前共用一个学习率。按**模块身份**分组并断言划分，同一张量落进两组会被 AdamW 走两次且不报错 |
| `--eval_every` / `--eval_every_samples` | 训练中在固定小 dev 片上验证。此前只在最后一步打分，所以"1500 步见顶后过拟合"和"根本没到"无法区分 |
| `--select_metric` | 决定 `checkpoint_best.pt` 的指标，**运行前固定**，不能看完曲线再挑。最终评测仍打 `checkpoint_last`，所以 best 不会悄悄变成头条数字 |
| `--warm_start` | 与 `--resume_from` 连用：**只加载权重**，重建 optimizer 与调度。续训和热启动是两回事——续训必须恢复 optimizer/调度/best（否则下一次验证会覆盖更好的 checkpoint），热启动必须不恢复（否则新传的学习率会被存档里的覆盖）。旧 checkpoint 只有一个参数组，分组 LR 下续训会报错，现在给的是可操作的提示而不是 torch traceback |

### 其他已修

- **`cosine_bias` 的 NaN**：有效 latent 少于 budget 时 `topk` 取到 −inf，整行 attention 变 −inf，softmax 出 NaN，反传毒化全部权重。B=32 上 C1 曾因此全程训废。**前向输出始终有限、只有梯度是 NaN**——"形状对、无 NaN"的检查抓不到它。
- **`--preset` 的硬编码选项表**，与 `config.PRESETS` 各写一份，新 preset 被静默挡在外面。现从 `PRESETS` 推导。
- **`gold_ranks`**：`check_attention_targeting.py` 原本假设 gold 是连续块（`gold_rank` + `n_gold`），HotpotQA 的两篇 gold 只有 21% 相邻。现用 `gold_ranks` 显式记位置，旧格式仍兼容。

---

## 4. 代码契约：未修的坑

剩下的不是代码 bug，是**尚未建立的度量**。

### 成本计量仍未建立

`T_online = T_cache_load + T_query_encode + T_readout + T_prefill + T_decode`，边界要声明。

- **不能漏记 7B 的 query 编码。**QuRO 在线跑的不是只有几十 M 的 readout。实测 A0（两条 query 路全关、跳过编码）的训练步速比其余臂快 36%，就是这笔钱。
- **evaluate() 不是 serving benchmark。**它在 `generate_answer()` 后又调了一次 `readout_cached()` 收统计，直接计时会重复计算 query/readout。需要独立 benchmark：每条 query 只编码读出一次、GPU warmup、同步计时、报 batch=1 时延与批量吞吐、报热/冷缓存与峰值显存。
- **三种压缩指标分别报**：evidence token ratio、decoder 输入长度（system/分隔符/问题/soft token 全计）、storage bytes。**不能用 token 压缩率代替存储字节数。**
- **摊薄比较必须含简单基线**：`Total(Q) = Build + Σ Online(q)`，至少比 C1、S、A0、P。C1/S/A0 共用同一缓存时离线成本基本相同，**C1 若质量无增量且每 query 更贵，文档复用不会自动逆转劣势**。

---

## 5. 实验解释的警告

1. **D1 只作诊断。**看得到问题与完全看不到问题的系统信息不对称，C > A0 不能自动证明证据选择。D1 绝对精度远低于 D0，**不能仅凭缩短 prompt 当作主贡献**。即便 D1 追平 D0，省的只是 prompt 里的问题明文，而在线仍要跑 7B 的 query 编码——那笔钱没省掉，只是换了地方付。
2. **A0 在 D1 落在地板是结构决定的，不是发现。**D1 删了问题明文，A0 又关掉两条 query 路，系统里没有任何地方存在问题信息。它的作用是管道检查。
3. **错配文档 ≠ 严格闭卷地板。**不相关证据可能伤害生成。保留无文档与错配文档两种控制，二者都是诊断，不是信息论上下界。
4. **逐样本优化 soft prompt 不是容量证明。**用 gold 优化成功只说明存在可诱导答案的向量；失败可能只是优化限制。不作 go/no-go 依据。
5. **P 不是同工作点的对照。**它吐 K×m 个 token（HotpotQA 上 80），C 吐 B 个。正确表述是"在 10× 更少 token 下的质量"，不是"赢了 PISCO"。
6. **共享缓存是继承能力。**PISCO/COCOM 与 S 都能复用缓存，"一份缓存、多 B、摊薄曲线"不是 QuRO 独占。贡献必须落在可检验的增量。
7. **冻结 ≠ 通用。**冻结 PISCO 仍可能与特定模型表示空间绑定；跨模型/readout 的可迁移性要单独验证。
8. **文献区别不代替实验。**RRK 面向重排且最佳配置会微调压缩器；ArcAligner 的内部对齐与外部预算缩减不同。胜过它们不能独立归因为 query 条件化。
9. **不要改标准数据集的干扰设定。**曾计划给 HotpotQA 补 BM25 干扰段做 K 扫描，已放弃：往标准 distractor 设定里塞干扰会让数字与已发表结果不可比，且加干扰同时让任务对所有方法变难，P 掉下来无法归因到它的 token 数。数据在 `/data02/quro/data/hotpot/{train,dev}_k{20,50}.jsonl`，不用。

---

## 6. 下一步

完整版见 [`ARM_MATRIX_RESULTS.md` §10](ARM_MATRIX_RESULTS.md)。摘要：

按 [`TRAINING_STRATEGY_REVIEW_AND_PLAN.md` §8](TRAINING_STRATEGY_REVIEW_AND_PLAN.md) 的矩阵推进，**工作点定在 B=8**——那是可学习读出相对余弦规则优势最大的点（+6.15，p=5.5e-10），也是每 token 证据效率最高的点（2.56）。B=16/32 更接近 P，但那里 C1≈S，方法没有可主张的增量。

| # | 实验 | 状态 |
|---|---|---|
| — | 输出分支干预 | ✅ 已跑。`pool_only ≈ full`（B=8 −0.65 / B=32 +0.45），**注意力池化撑起全部效果，Δ 可忽略**。所以注意力是因果路径，针对它的监督不会打空 |
| — | W3 query 表示策略 | ✅ 已实现 |
| **R0** | `fixed_adapter` + 原 CE 配方，B=8 | 去掉 query 漂移混淆后的新基线 |
| **R1** | R0 + 分组学习率 | 是否存在优化失衡 |
| **R2** | R1 + KL 蒸馏 P | 全缓存教师是否有效。建议离线预存 teacher 分布，**注意 top-k 不能直接重归一化**（见复审 §6.4） |
| **R3** | R1 + gold 覆盖监督 | `supporting_sentences` 已在数据里，25× 于答案的监督密度 |
| R4 | R1 + KL + coverage | R2/R3 至少一项有效再做 |

**必须配套的对照**：S+KL（教师对 S 的 decoder 一样有用，最终要比 C+KL 与 S+KL）、D0 下的 A0、若 coverage 有效则补同标注的简单选择器基线。

**尚未做的诊断**：均匀注意力对照（把 α 换成均匀，才是真正的"平均池化"基线）、gold 覆盖诊断、多 seed。

**对比学习排在 KL 之后**：它要先定义正负样本，而实验 2、3 的靶子明确得多。

**暂不改 readout 架构**：三个差异极大的 readout 曾落在 0.5 分以内，改架构是在优化实测为平坦的维度。

---

## 7. 运行归档要求

每个正式运行至少保存：run_id、代码 commit（工作区有改动则另存 diff）、完整 config 与实际启动参数；数据 split/版本/hash、缓存 manifest/hash、压缩器与 query/decoder adapter 版本；seed、硬件、dtype、训练样本/更新次数、各 B 采样次数；readout/query encoder/decoder 的实际可训参数量与共享关系；每题 id、预测、gold、指标、B、输入长度及对照类型；质量聚合、配对比较、计时协议。

`scripts/run_arms.sh` 已自动存 `commit.txt` 与 `worktree.diff`。大文件留 `/data02/quro/`，GitHub 只放轻量配置、汇总与可追溯索引。

---

## 8. 现成资产

### 缓存（`/data02/quro/cache/`，均为 memmap）

| 目录 | m | 文档数 | 大小 | 说明 |
|---|---:|---:|---:|---|
| `gonogo-pisco-r16` | 8 | 263,890 | 17 GB | PISCO 原生，TriviaQA/gonogo |
| `hotpot-pisco-r16` | 8 | 260,236 | 17 GB | **HotpotQA，当前主力**，已过分片级往返校验 |
| `chunk4-part0` | 32 | 263,890 | 69 GB | PISCO 切块（128→4×32），ξ_off=3.8 |
| `cocom4-part0` | 32 | 263,890 | 69 GB | COCOM rate 4，需配 `--generator_path` + `--generator_n_mem 32` |

> `*-part1..3` 是构建时的分片来源，已合并进 `part0`，不要单独使用。

### 数据（`/data02/quro/data/`）

- `gonogo/{corpus,train,dev}.jsonl` — 98k 训练 / 2k dev，含 4 篇随机干扰
- `trivia/queries.jsonl` — 2000 条评测，**已是开发数据**
- `hotpot/{corpus,train,dev,test}.jsonl` — 260k 段语料；30k 训练 / 2k dev / **5405 test（未动过）**
- `hotpot/{train,dev}_k{20,50}.jsonl` — K 扫描数据，已放弃，见 §5.9

### checkpoint（`/data02/quro/runs/*/checkpoint_last.pt`）

主力：`hp2d0_{C1,S,P}`（HotpotQA 主表）、`bs{8,16,32}_C1`、`bs{8,16,32}S_S`、`bs32_C0`（B 扫描）、`w1_{A0,A1,C0,C1}` 与 `hp1_*`（四格分解）。

### 磁盘

`/data02` 449G 可用。**`/home` 是共享盘、会被别人占满，大文件一律写 `/data02`。**

---

## 9. 常用命令

```bash
# 臂矩阵（自动存 commit/diff，并核对实际实现的臂）
PRESET=pisco_hotpot EVAL_FILES="dev=/data02/quro/data/hotpot/dev.jsonl" \
QDROP=0.0 GPUS="0,1,2,3" PREFIX=run bash scripts/run_arms.sh A0 C0 C1 S P

# 固定单预算（B 扫描必须用）
FIXED_BUDGET=1 BUDGET=32 GPUS="0" PREFIX=bs32 bash scripts/run_arms.sh C1

# 单臂手工启动
CUDA_VISIBLE_DEVICES=0 python -m src.train --preset pisco_hotpot \
  --arm C1 --steps 3000 --num_workers 4 --budget 8 \
  --eval_input_modes D0 --doc_control --eval_max_samples 2000 \
  --eval_files dev=/data02/quro/data/hotpot/dev.jsonl \
  --tag NAME --out_dir /data02/quro/runs/NAME

# 汇总所有运行 + 配对检验 -> results/arm_matrix.json
python scripts/collect_arm_matrix.py

# 契约测试（纯 CPU，无下载）
python tests/test_shapes.py                                  # 66 passed

# 诊断
python scripts/check_query_sensitivity.py --checkpoint ...   # readout 是否响应 query
python scripts/check_attention_targeting.py --checkpoint ... # 注意力有没有落在 gold
python scripts/check_readout_stats.py --checkpoint ...       # 输出分布、有效支撑、槽间冗余
python scripts/evidence_subset.py --control ...              # 只在真正依赖证据的题上重算
```

**`--num_workers 4` 必须带**：不带的话缓存读取阻塞训练线程，m=32 下 2.88 s/step vs 0.73 s/step。

---

## 10. 踩过的坑（别再踩）

| 坑 | 症状 | 出处 |
|---|---|---|
| A 臂没关余弦先验 | "query 无关对照"其实看得见 query，结论归因错误 | §3 W1 |
| `cosine_bias` 遇短行 | 梯度 NaN 而**前向输出正常**，B>16 时全程训废 | §3 |
| 混合预算批次 | B 小的行被同批 B 大的行污染，slot self-attn 无预算掩码 | §3 W5 |
| `--preset` 选项表写两份 | 新 preset 被静默挡掉，进程秒退 | §3 |
| gold 连续块假设 | HotpotQA 两篇 gold 只有 21% 相邻，诊断静默算错 | §3 |
| 缓存合并索引错位 | 一半文档返回别人的 latent；形状对、无 NaN、读回校验全过 | 已修 + 分片级往返校验 |
| COCOM mem token 前置 | 因果掩码下所有文档压出同一常数向量 | `probe_mem_side()` 实测判定 |
| `num_workers=0` | GPU 掉到 0%，慢 4 倍 | |
| 残差惩罚在初始化点梯度 inf | 全部梯度 NaN | 改用 mean square，不取 sqrt |
| LM 逃过 train/eval 切换 | LoRA dropout 在评测时仍生效 | |
| RG prompt 整体截断 | 问题被静默切掉，跑出低于闭卷的 0.8% | |
| HF 下载慢 | 10 MB/min → 装 `hf_transfer` 后 1913 MB/min；`HF_ENDPOINT=https://hf-mirror.com` 可用 | |

**共同教训**：这些坑里有一半是"形状对、无 NaN、不报错，但语义是错的"。只加针对具体风险的语义契约测试，不以 shape 检查充数。

---

## 11. 目录约定

```
article/   别人论文的阅读笔记
docs/      我们自己的设计、方案、实验、进展   ← 本文在这里
results/   原始实验数据（json）
src/ scripts/ tests/   代码
/data02/quro/          模型、缓存、运行输出（不进 git）
```

### docs 索引

- [`ARM_MATRIX_RESULTS.md`](ARM_MATRIX_RESULTS.md) — 臂定义、四格分解、B 扫描
- [`TRAINING_RECIPE_RESULTS.md`](TRAINING_RECIPE_RESULTS.md) — **最新**：训练配方、KL 蒸馏、问题压缩（37 次运行 + 24 组配对检验）
- [`QURO_EXPERIMENTAL_DESIGN.md`](QURO_EXPERIMENTAL_DESIGN.md) — 实验设计
- [`QURO_V0.2_RESULTS_AND_ANALYSIS.md`](QURO_V0.2_RESULTS_AND_ANALYSIS.md) — v0.2 分析，**其 A 臂结论已被 §3 W1 推翻**
- [`QURO_RELATED_WORK.md`](QURO_RELATED_WORK.md) — 相关工作与切割
