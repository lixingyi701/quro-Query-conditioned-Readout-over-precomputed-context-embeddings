# 投影器输入归因：范数、方向、答案内容与结束行为

这组实验回答的是：在同一个训练好的投影器上，最终压缩向量的长度和方向分别与哪些 decoder 行为有关。它是 **eval-only 的机制干预**，不替代独立训练的 SQ/S0 标准模块消融，也不直接证明 query 的增量贡献。

实现入口：

- [`../scripts/eval_projector_inputs.py`](../scripts/eval_projector_inputs.py)：恢复 checkpoint 配置，生成四臂预测与诊断。
- [`../scripts/analyze_projector_inputs.py`](../scripts/analyze_projector_inputs.py)：逐题配对差值、bootstrap 区间与交互项。
- [`../src/projector_diagnostics.py`](../src/projector_diagnostics.py)：输入干预、几何统计、内容/EOS 评分和 decoder hook。
- [`../tests/test_projector_diagnostics.py`](../tests/test_projector_diagnostics.py)：CPU 契约测试及小型 Mistral 检查。

当前实现已通过 CPU 工程验证，**尚未产生真实 HotpotQA 机制实验结果**。不得把 toy 输出写入论文效果表。

## 1. 动机与可识别的贡献

我们发现缓存 memory 与普通 word embedding 的范数可能不同，投影器又显著改善了生成表现，因此需要区分三个解释：

1. 投影器改变 memory 的尺度，改善它与 decoder 残差流的相互作用。
2. 投影器改变向量方向，改善文档信息的可读性或任务适配。
3. 投影器主要改变答案长度、结束概率及输出格式，F1/EM 因而提高。

这三个解释可以同时成立。范数接近 word embedding 只能作为描述性观察，不能直接作为“对齐成功”的证据；压缩 memory 也没有必须与离散词向量同分布的理论要求。当前两 seed SQ−S0 的 query 增量仍小，参见 [`QUERY_PROJECTOR_ABLATION_RESULTS.md`](QUERY_PROJECTOR_ABLATION_RESULTS.md)。应先定位投影器共有的收益，再讨论 query 的额外作用。

## 2. 四臂输入：固定 decoder、位置、长度和问题

对每个有效 memory token，记原始缓存为 \(z\)，训练后完整输出为 \(e=z+\delta\)。四臂定义为：

| mode | decoder 接收的 memory | 保留的因素 |
|---|---|---|
| `original` | \(z\) | 原始范数、原始方向 |
| `scale_only` | \(\frac{\|e\|_2}{\|z\|_2}z\) | 学到的范数、原始方向 |
| `direction_only` | \(\frac{\|z\|_2}{\|e\|_2}e\) | 原始范数、学到的方向 |
| `full` | \(e\) | 完整投影输出 |

每批只调用一次 `readout_cached`。四臂复用同一个 E、document mask、memory slot、D0 prompt、问题明文、gold answer 和生成上限；没有训练步骤、优化器或 checkpoint 写回。支持 `shared_projector` 的 SQ/S0/已有融合 checkpoint，也支持同样保持槽位的 `joint_projector`。不支持改变 memory 长度的 readout。

`original` 使用原始缓存，但仍沿用该 checkpoint 的 reader 和 prompt；它不是另一个已训练模型。`scale_only`/`direction_only` 是推理时混合输入，可能偏离训练分布。其差值可说明当前 checkpoint 对这类干预的响应，不能直接当作独立训练“只有范数/只有方向”模型的能力。

公式以 float32 实现，写入 decoder 时仍按现有 `assemble_inputs` 转为 word embedding 的 dtype。`geometry` 同时记录写入前的 Z/E 几何和四臂实际写入 dtype 后的 RMS，因此 BF16 舍入不被隐藏。

padding 在任何计算前置零，不进入统计。若 \(\|z\|\le\epsilon\)，`scale_only` 回退到 z；若 \(\|e\|\le\epsilon\)，`direction_only` 回退到 e。默认 \(\epsilon=10^{-8}\)。输出记录两类近零 token 数，未定义的角度/比率不纳入分布。若真实实验出现这类 token，需检查并说明，不能把该部分当作严格的长度/方向分解。

## 3. 几何与 decoder 层内诊断

`result.json.geometry` 汇总以下逐 token 标量的数量、均值、p05/p50/p95 和 pooled RMS：

- 原始 Z、完整 E、delta 的 RMS。
- \(\|\delta\|/\|z\|\)、cos(Z,E)。
- delta 沿 Z 的有符号径向分量、径向 RMS、切向 RMS、径向能量占比。
- 四臂转换为 decoder dtype 后的输入 RMS。
- decoder 实际 prompt 中问题部分的 **word embedding** RMS。

问题参考使用完整 D0 prompt 与“相同 memory 数、空问题”prompt 的最长公共 token 前缀/后缀，取插入问题导致变化的 token 区间。它包含分词边界随问题插入而改变的 token；不是独立分词后的截断问题，也不是 query encoder 的 contextual hidden states。具体区间与 prompt token IDs 保存在逐题输出，可审计边界。

`predictions.jsonl` 还保存每题几何均值，便于将改变幅度与效果差值做配对分析。聚合时所有有效 token 等权，不先对 batch RMS 求平均；每题均值与全体 token 加权统计是不同口径。

加 `--trace_decoder` 可采集 gold teacher-forced 前向中的各层：

| stage | 实际读取位置 |
|---|---|
| `block_input` | attention 前的残差流 |
| `attention_update` | attention 输出、加回残差之前 |
| `after_attention_residual` | attention 加回后的残差、MLP 前归一化之前 |
| `mlp_update` | MLP 输出、加回残差之前 |
| `block_output` | 两次残差更新后的块输出 |
| `final_norm_input` / `final_norm_output` | decoder 最终 norm 的输入/输出，分开记录 |

各 stage 分别统计 memory、question、全部 prompt text（包含 question）和 answer 区域。输出 `sum_sq`、有效元素数及 pooled RMS；attention/MLP 更新另除以各自更新前残差的 RMS。answer 区域包含 gold 内容和 terminal EOS。这些 hook 只作用于完整 gold 的评分前向，不混入问题编码、候选答案评分或 autoregressive decode。

支持现有 Mistral/Llama pre-norm block 布局及 toy reader；不支持的结构明确报错。不开此开关可降低诊断成本。不保存全维激活或 attention 矩阵。

pre-norm 中，正比例缩放近似被第一层 RMSNorm 抵消：

\[
\operatorname{RMSNorm}(cz)
=\gamma\odot\frac{cz}{\sqrt{c^2\operatorname{mean}(z^2)+\epsilon}}
\approx\operatorname{RMSNorm}(z),\quad c>0.
\]

但残差支路仍保留 cz 的幅度，更新与残差的相对大小会改变，后续层并非整体尺度不变。因此“norm 不同”不足以推导第一层 attention 必然读取失败，也不能推导缩放完全无效。hook 用于检验实际残差动力学，不能把 `hidden_states[-1]-hidden_states[-2]` 当成最后一块的更新，因为最后一项可能已经过最终 norm。

## 4. 答案内容与 EOS 分开测量

生成保留既有 greedy decoder 调用。指标仍是 alias-max 的 EM/F1/substring，同时保存真实新生成 token IDs、首次 EOS 前 token 数、是否出现 EOS、是否在未出现 EOS 时触及生成上限。EOS=PAD 的 reader 也按首次 EOS 判断，不把后续 batch padding 算入答案。

teacher forcing 使用数据适配后的**第一个完整 gold alias**，按训练惯例添加前导空格，并追加一个 EOS；不受 `max_answer_len` 截断。逐题记录该 target 是否会超出训练长度上限。评分按 causal shift：位置 t 的 logit 预测 t+1 的 gold token，prompt/padding 均不计分。

| 字段 | 含义 |
|---|---|
| `content_logprob_sum` / `content_logprob_mean` | 仅答案内容 token 的 log 概率，排除 EOS |
| `content_tokens` | 完整 gold 内容 token 数 |
| `eos_after_gold_logprob` / `eos_after_gold_probability` | 完整 gold 后下一 token 为 EOS 的得分 |
| `eos_during_gold_mean_probability` | 在各 gold 内容 token 前预测 EOS 的平均概率 |
| `token_weighted_content_logprob` | 全体答案内容 token 加权的聚合得分 |

`qa` 默认内容指标按题平均，另给 token 加权口径。`eos_during_gold_mean_probability` 使用 gold prefix，描述 teacher-forced 条件下的结束倾向，不等于生成时的提前停止率。

这组指标降低了生成长度/格式对内容评价的干扰，但 gold likelihood 上升仍可能包含任务习惯或答案先验的适配，不能单独证明跨文档推理或语义恢复。

可选 `--candidates_file` 提供固定的人工审核错误答案，每个评估 ID 对应一个 `wrong_answer`：

```json
{"id": "question-id", "wrong_answer": "a reviewed incorrect answer"}
```

实际文件为 JSONL，ID 集合必须与此次评估子集完全一致。脚本拒绝重复 ID、空答案和与任一 gold alias 归一化后相同的答案；是否语义上确实错误仍由候选构建者审核。四臂使用同一候选，不从其他样本自动拼出所谓“错误答案”。

`content_candidate_margin` = 正确 gold 的每 token 平均 log 概率 − 错误答案的每 token 平均 log 概率，两者均排除 EOS，并记录各自 token 数。它是固定候选下的诊断 margin，不是校准后的正确概率；候选长度和选取方式仍可能影响结果。

## 5. 运行协议与命令

先用固定 dev 前缀定位机制，再对预先约定的完整 split 和两个 seed 确认。SQ/S0 各自读取已有的 last checkpoint，不重新训练，也不在看到结果后改选 best。保持同一 split、ID 顺序、cache、reader、batch size 和 generation cap。两个 seed 分别报告；这里的逐题 bootstrap 不估计训练 seed 方差。

在仓库根目录运行，下列路径由实际机器设置：

```bash
CKPT_ROOT=/path/to/runs/query_ablation
EVAL_FILE=/path/to/hotpot/dev.jsonl
CACHE_DIR=/path/to/the/original/latent-cache
OUT_ROOT=/path/to/runs/projector_input_attribution

python scripts/eval_projector_inputs.py \
  --checkpoint "$CKPT_ROOT/sq_s42/checkpoint_last.pt" \
  --eval_file "$EVAL_FILE" \
  --cache_dir "$CACHE_DIR" \
  --out_dir "$OUT_ROOT/sq_s42_dev2000" \
  --device cuda:0 --batch_size 2 --max_samples 2000 \
  --trace_decoder

python scripts/analyze_projector_inputs.py \
  --run "$OUT_ROOT/sq_s42_dev2000" --bootstrap 2000 --seed 0
```

`--cache_dir` 和 `--generator_path` 仅用于迁移**同一份**资源；不能换 reader/cache 后仍称为同一实验。默认使用 checkpoint 的生成上限。省略 `--max_samples` 评估全量；数值必须大于零。输出目录必须为空，不会覆盖已有结果。四臂都包含生成与 gold scoring；trace 可关闭，候选评分会增加一次每题/每臂前向。

对 `s0_s42`、`sq_s43`、`s0_s43` 重复以上命令，分别使用新目录。完整 test 确认时换 split、删除 `--max_samples`，保留其余设置。

若需要排查 32 token 上限的影响，另建输出目录，四臂统一增加 `--max_new_tokens 128`。脚本记录 `generation_cap_overridden`；该诊断与默认上限结果分别报告。不要只给某一臂提高上限。

CPU 工程验证：

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_projector_diagnostics.py -v
```

toy backend 只供工程测试，脚本将其标为 `toy_smoke_only`。其 frozen reader 依照 checkpoint 的 seed 和原始词表文件重新构建，不能作为真实模型 checkpoint 的论文评测。

## 6. 输出与归因判据

| 文件 | 内容 |
|---|---|
| `predictions.jsonl` | 每题四臂预测、QA、内容/EOS、生成行为、几何、槽位与问题 token 边界；每批 flush |
| `result.json` | 完成标志、checkpoint/eval/cache manifest 哈希、配置、运行参数、几何分布、聚合 QA 与可选层内诊断 |
| `paired_analysis.json` | 整体逐题对照、95% percentile bootstrap 区间，以及数据已有 type/hop_type/level 的探索性分层 |

只有完整结束后才写 `result.json`，分析脚本拒绝未完成 run、缺臂、重复 ID 及不一致的计数/QA 汇总。保存原 checkpoint 配置与迁移后的 effective 配置、实际 generation cap、设备、torch 版本和代码 commit，便于复核。reader/cache 文件内容本身的完整哈希仍需沿用既有资源管理；manifest 哈希不能替代 latent 文件内容校验。

分析报告六个 contrast：full−original、scale−original、direction−original、full−scale、full−direction，以及交互项：

\[
I=F(\mathrm{full})-F(\mathrm{scale\_only})-F(\mathrm{direction\_only})+F(\mathrm{original}).
\]

F 可为 F1、内容 log 概率或其他记录指标。不能把 full 总收益机械分成两项可加贡献；若交互明显，说明两种因素的组合响应不同于单独干预响应。JSON 中 QA 与差值均使用 0–1 fraction，终端 F1 表换算为百分点；log 概率使用自然对数。

| 观察 | 支持的解释 | 仍需保留的边界 |
|---|---|---|
| scale_only 接近 full，且内容得分/margin 改善 | 尺度因素足以恢复当前 checkpoint 的大部分读取收益 | 不等于必须匹配 word embedding 范数，也不是独立训练尺度模型的能力 |
| direction_only 接近 full，内容得分/margin 改善 | 方向变化对当前 decoder 的内容预测有作用 | 方向还包含格式/任务习惯信息，不能直接称为语义恢复 |
| F1/EM 改善而内容得分/margin变化小，EOS/长度改变大 | 结束与输出行为适配是收益的候选解释 | 需要与 substring、真实预测及增加生成上限的诊断一致 |
| full 超过两种单因素输入，交互明显 | 范数与方向组合响应重要 | 混合输入偏离训练分布也可能造成差距 |
| SQ/S0 都有相似几何及行为改善 | 改善主要属于共有文档投影路径的候选解释 | query 增量仍以独立训练 SQ−S0 和匹配证据为准 |

跨文档桥接能力需要后续独立诊断，如固定检索文档的答案段/桥接段原文替换与证据可见性分层。本次不将方向收益、teacher-forced likelihood 或问题替换敏感性自动归结为多跳推理提升。
