# 固定 Z：Direct-CE / Direct-State / W-CE 执行说明

> 后续结果与停止决定见 [发布版起点复审](READER_RESET_REVIEW.md)。本轮三臂及低LR复验均未建立State/W增益，不默认重跑。

2026-10-01；基础提交 `7f477eb`；实现分支 `feat/reader-state-workspace`。

2026-10-02 追加：[首轮退化后的诊断与短程修复](READER_RECOVERY_RUNBOOK.md)。本文件保留第一轮协议；不要直接按第 6 节重跑长训练。新增目标选项仅用于另行记录的短程对照，默认行为仍与首轮一致。

这是下一阶段的训练执行方案，取代旧因果顺序方案中的训练准入规则。保留 P0–P3 作为已完成的历史诊断；**不再安排位置效应实验，不要求补齐 P3 才能训练，不恢复旧 embedding readout**。

## 1. 本轮要回答的问题

固定 PISCO 压缩器与文档缓存 Z，检验更好的监督或独立在线 workspace 能否改善压缩证据的利用。P1 混合实验优先定位到答案段，但尚未区分事实读取与跨文档组合；本轮不预设其机制，也不声称 Z 已保留所有答案信息。

| arm | decoder 输入 | 额外目标 | 训练参数 |
|---|---|---|---|
| `direct-ce` | 原 D0 `[Z][Q][S][A]` | 无 | P₁ decoder LoRA |
| `direct-state` | 同上 | 冻结 P₁ raw 的 S 中间状态 | 与 Direct-CE 完全相同的 LoRA |
| `w-ce` | `[Q][W][S][A]`，Z 在序列外 | 无，只有 gold CE | 相同 LoRA＋W＋cross-attention |

三组共用 P₁ 权重起点、数据、训练步数、有效 batch、评测频率和 checkpoint 选择规则。所有组明确使用 gold answer，忽略数据中的 `teacher_output`。P₁ checkpoint 只加载权重，不继承旧优化器。`--resume` 才恢复本轮优化器、调度器、步数、最佳分数、随机状态及数据顺序。

首轮教师固定为 P₁ raw，结论限于域内改善。State 有效后再做发布版 raw 教师替换和答案 KD 对照。本实现首轮不启用答案 KL、W-State、动态预算或压缩器训练。

## 2. 具体结构与损失

### Direct-CE

复用 `PiscoPromptBuilder(D0)` 与 `assemble_inputs`。全部有效 Z slots 原样写入，不选择、不缩放；题目与模板保持旧 D0。只有答案 token 与 EOS 参与 CE。训练脚本中的 `B` 不再控制 memory 数量，数量由实际文档数×每篇 8 slots 决定。

### Direct-State

教师以 `[raw documents][Q][S]` 前向，**不输入答案**。raw 按每篇 128 token 截断，与当前缓存口径一致。S 是各自 prompt 的最后一个有效 token，不是新 token；按每行 prompt 长度定位，不用教师/学生的相同绝对下标。

层位为第 8、16、24 个 decoder block 输出（CLI 使用一基编号，内部 7、15、23）；不使用最终 RMSNorm 输出冒充 block 输出。每题存 `[3,H]` FP16 向量，loss 的余弦计算为 FP32：

\[
L=L_{CE}+\lambda\frac{1}{3}\sum_{l\in\{8,16,24\}}
\left[1-\cos(h^l_{student,S},\operatorname{stopgrad}(h^l_{raw,S}))\right],\quad\lambda=0.1.
\]

首轮无投影头，不逐 token 重建原文，不要求所有层轨迹一致。训练中记录 CE/state/总 loss，但按 QA 选 checkpoint。`--state_weight 0` 明确禁用教师缓存和状态 hook，用于与 Direct-CE 等价检查。

缓存为 `states.bin` FP16 memmap、`index.json`、`manifest.json`。每题键包含 ID、问题和有序文档 IDs，不包含答案。manifest 绑定 P₁ SHA256、训练文件 SHA256、latent manifest SHA256、tokenizer、模板、层位、截断和 raw corpus SHA256。完整性标记最后写入；不完整/错版本/缺样本会报错，不会静默跳过。缓存读取验证文件 checksum。现有 latent cache 本身无逐字节内容校验，本代码绑定其 manifest 与路径；运行后不得原地改写 latent 文件或底座模型权重。

### W-CE

16 个共享、可学习的 FP32 初始向量（std=0.02），每题重新展开。复用已有 MEM token ID 作为 prompt 占位，嵌入全部替换为 W，无需改 tokenizer 或扩大词表；**这些占位不再写入 Z**。实际文本为原 system/chat prefix＋`Question:{Q}`＋W＋chat suffix。W 的后面必须有真实模板尾部，过长或特殊 token 冲突报错。

第 8/16/24 个 block 输出后，仅在 W 位置执行：

\[
W^l\leftarrow W^l+g_l\,\mathrm{CA}_l(\mathrm{LN}_W(W^l),\mathrm{LN}_Z(Z)).
\]

CA 内部维度 512、8 heads、每层独立 Q/K/V/O 投影与两侧 LayerNorm；新模块 FP32，残差写回 decoder dtype。Z detach，padding slots 屏蔽。O 零初始化、g=0.1：第一步 O 可以得到梯度，Q/K/V 的非零任务梯度在 O 更新后出现。W 从答案 CE 经 decoder 反传学习，单独设学习率。

W 的 self-attention 仍是因果关系：W 可读 Q 和更早 W，Q 看不到 W；后面的 S/A 可以读 W。第一次 CA 前 W 已经过前面 block。CA 不施加到 Q、S、A 上。固定 Z 不被在线修改。

新 CA 在 block 输出处写入，下一个 block 才能通过 self-attention 使用更新，因此禁止把最后一个 block 作为 W 写入层。生成只在 prefill 计算 W；单 token cached decode 不重新写 W。每层对应的 KV 缓存与完整前向一致。当前 CA 无额外文档 rank/slot embedding，直接把 Z 当作外部向量集合，这是首轮简化假设，不隐含保留显式段落边界。

W-CE 虽从 P₁ decoder 初始化，但改变了接口，零步不等于 P₁。新增约 25M CA 参数（H=4096、3层、d=512；以 manifest 实测为准），所以跨结构是整个方案的比较，不是参数匹配的纯机制证明。

## 3. 文件与运行环境

- `src/reader_experiment.py`：三组输入、状态采集、CA、状态 loss、教师缓存。
- `src/reader_runtime.py`：P₁ 加载检查、确定性数据顺序、评测、断点。
- `scripts/run_reader_experiment.py`：`cache / train / eval` 三个子命令。
- `scripts/compare_reader_experiments.py`：逐题配对 bootstrap，整体和题型分层。
- `tests/test_reader_experiment.py`：无需下载权重的真实小型 Mistral＋PEFT 检查。

使用服务器已有的 PISCO 环境；不要为此升级正式环境的 torch/CUDA。单进程单卡，通过 gradient accumulation 保持有效 batch。**不要用 torchrun**；可用不同 GPU 独立运行三个 arm。

本地验证环境：Python 3.12、torch 2.14.1+cpu、transformers 4.46.3、peft 0.13.2；小型随机 Mistral，没有下载正式权重。服务器版本写入每次 manifest。正式 P₁ 权重、数据和 CUDA 显存/性能需在服务器上验证；本地测试不代表正式实验结果。

```bash
git fetch origin
git switch feat/reader-state-workspace
conda activate selecom
python -m pytest -q tests/test_reader_experiment.py
python scripts/run_reader_experiment.py train --help
```

若缺 pytest，仅安装测试依赖即可；项目训练依赖沿用 `requirements.txt`。所有下面命令在仓库根目录执行。建议在 tmux 会话中运行。

```bash
P1_CKPT=/data02/quro/runs/oscale_P1/checkpoint_last.pt
TRAIN_DATA=/data02/quro/data/hotpot/train.jsonl
DEV_DATA=/data02/quro/data/hotpot/dev.jsonl
CORPUS_DATA=/data02/quro/data/hotpot/corpus.jsonl
READER_ROOT=/data02/quro/runs/reader_state_workspace_v1
mkdir -p "$READER_ROOT"
```

cache/model 路径默认从 P₁ config 读取；如果迁移，三个 arm 和教师构建都显式传入相同 `--cache_dir` / `--generator_path`。路径及内容身份被绑定，不应训练中途改变。

## 4. 先检查 Direct 零步复现

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  --arm direct-ce --init_checkpoint "$P1_CKPT" \
  --train_file "$TRAIN_DATA" --eval_file "$DEV_DATA" \
  --steps 0 --eval_samples 2000 --eval_batch_size 8 \
  --out_dir "$READER_ROOT/zero_step"
```

预期在相同数据、底座、解码设置下接近 P0 复现值 substring=0.600 / EM=0.552。读 `validation.jsonl`。零步结果明显偏离应先查输入/checkpoint/backend；不直接开始长训练。`steps=0` 也写 best/last，可复用 `eval` 对照。历史 runner 的部分 backend 数值差异可能改变极少数 greedy 输出，需逐例核对而不是随意放宽阈值。

## 5. 教师状态缓存与短程检查

先用前 32 个训练样本生成独立 smoke 缓存：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py cache \
  --init_checkpoint "$P1_CKPT" --train_file "$TRAIN_DATA" \
  --corpus "$CORPUS_DATA" --layers 8,16,24 --batch_size 2 \
  --limit_train 32 --out_dir "$READER_ROOT/teacher_smoke"
```

随后三个 arm 均运行 2 步；它只查工程，不比较 QA：

```bash
SMOKE_ARGS=(--init_checkpoint "$P1_CKPT" --train_file "$TRAIN_DATA"
  --eval_file "$DEV_DATA" --limit_train 32 --eval_samples 16
  --steps 2 --batch_size 1 --grad_accum 2 --eval_batch_size 2
  --eval_every 1 --save_every 1 --log_every 1 --grad_checkpointing)

CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  "${SMOKE_ARGS[@]}" --arm direct-ce --out_dir "$READER_ROOT/smoke_direct_ce"
CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  "${SMOKE_ARGS[@]}" --arm direct-state --state_weight 0.1 \
  --teacher_cache "$READER_ROOT/teacher_smoke" --out_dir "$READER_ROOT/smoke_direct_state"
CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  "${SMOKE_ARGS[@]}" --arm w-ce --workspace_tokens 16 --cross_dim 512 --cross_heads 8 \
  --out_dir "$READER_ROOT/smoke_w_ce"
```

要求 loss/梯度有限、decoder_grad_norm 非零、W 的 workspace_grad_norm 非零，teacher cache 完整，checkpoint 能用下一节 eval 重新加载。O 零初始化导致第一个 backward 的 Q/K/V 梯度为零是预期；后续仍为零才需要排查。为降低显存，默认训练 microbatch=2、accum=8（名义有效 batch=16，跨 epoch 的末批可能更小；三个 arm 数据顺序一致）。

`--grad_checkpointing` 仅使用 non-reentrant 重算。Hook context 在 backward 后才退出，重算期间使用同一批 Z。其他方式手动启用 reentrant checkpointing 不受支持。

## 6. 正式教师缓存与三组训练

短程检查通过后，完整缓存训练集。不要使用 dev/test 构建训练目标。脚本不会在线更新教师，也不会因教师答错而删训练样本。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py cache \
  --init_checkpoint "$P1_CKPT" --train_file "$TRAIN_DATA" \
  --corpus "$CORPUS_DATA" --layers 8,16,24 --batch_size 2 \
  --out_dir "$READER_ROOT/teacher_train"

TRAIN_ARGS=(--init_checkpoint "$P1_CKPT" --train_file "$TRAIN_DATA"
  --eval_file "$DEV_DATA" --steps 3000 --seed 42
  --batch_size 2 --grad_accum 8 --decoder_lr 1e-4
  --warmup_ratio 0.05 --weight_decay 0.01 --grad_clip 1.0
  --grad_checkpointing --layers 8,16,24
  --eval_every 250 --eval_samples 500 --eval_batch_size 8
  --save_every 250 --select_metric substring --max_new_tokens 32)

CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  "${TRAIN_ARGS[@]}" --arm direct-ce --out_dir "$READER_ROOT/direct_ce_s42"

CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  "${TRAIN_ARGS[@]}" --arm direct-state --state_weight 0.1 \
  --teacher_cache "$READER_ROOT/teacher_train" --out_dir "$READER_ROOT/direct_state_s42"

CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train \
  "${TRAIN_ARGS[@]}" --arm w-ce --workspace_tokens 16 \
  --cross_dim 512 --cross_heads 8 --gate_init 0.1 --workspace_lr 1e-4 \
  --out_dir "$READER_ROOT/w_ce_s42"
```

这些是第一轮固定配置。decoder LR 沿用 P₁ recipe 的 1e-4；W 新参数首轮也用 1e-4。不要观察到某组落后后只给该组额外步数却仍称“相同预算”。W-CE、teacher cache 的额外成本需要报告。state loss 权重的调参如需扩展，先登记候选与验证预算，不能按 state loss 最低选模型。

best 包括零步：继续训练如果变差，best 可以是 step=0。`checkpoint_best.pt`、`best_checkpoint.json` 和 `best_predictions.json` 对应选择集；last 是最后一步。checkpoint 含可训练参数（不复制 7B 底座）、优化器、调度器和 RNG，所以评测仍需要原底座和 P₁ 文件。

中断恢复：**原命令全部参数保持一致，只追加**：

```bash
--resume "$READER_ROOT/direct_state_s42/checkpoint_last.pt"
```

不可把 steps 从 3000 改成 6000 并声称无缝恢复：cosine schedule 已不同，脚本会拒绝轨迹参数变化。默认每 250 步保存一次；失败最多重跑该区间。脚本拒绝覆盖已有非空目录。恢复时继续追加日志；若人为恢复更早断点，同一步可能出现多条日志，按最后一次记录及最终 checkpoint 分析。

## 7. 对 best 做全量 K=10、K=2 评测与配对比较

```bash
for ARM_NAME in direct_ce direct_state w_ce; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py eval \
    --checkpoint "$READER_ROOT/${ARM_NAME}_s42/checkpoint_best.pt" \
    --eval_file "$DEV_DATA" --eval_batch_size 8 \
    --out_dir "$READER_ROOT/${ARM_NAME}_s42/eval_dev_k10"
  CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py eval \
    --checkpoint "$READER_ROOT/${ARM_NAME}_s42/checkpoint_best.pt" \
    --eval_file "$DEV_DATA" --eval_batch_size 8 --gold_only \
    --out_dir "$READER_ROOT/${ARM_NAME}_s42/eval_dev_k2"
done

python scripts/compare_reader_experiments.py \
  --reference "$READER_ROOT/direct_ce_s42/eval_dev_k10" \
  --candidate "$READER_ROOT/direct_state_s42/eval_dev_k10" "$READER_ROOT/w_ce_s42/eval_dev_k10" \
  --output "$READER_ROOT/paired_dev_k10.json"
```

K=2 仍问原问题，只保留 `gold_ranks` 的两篇段落，并按原检索顺序排列。比较脚本严格要求相同评测文件、K 条件和题目集合，输出 candidate−reference（QA 越大越好，NLL 越小越好）。CI 只覆盖当前 checkpoint 的样本差异，不覆盖训练 seed 方差。

主比较是 Direct-State/W-CE 相对**匹配预算的 Direct-CE**，不能把相对发布版 PISCO 的全部提升记为新方法贡献。substring 为预定主指标，EM/F1/NLL 与 bridge/comparison 一起报告。3000 步以外的 seed 43/44 复核只在有希望的配置上进行，并给 Direct-CE 相同复核预算。

dev 2000 条已经参与多轮研究，当前结果仍是探索性。锁定配置后再显式指定独立评测集；本脚本不会自动读取 preset 中的 test。训练检查 train/eval query-ID 重叠；同文档跨问题泄漏需要按数据划分另行核验，不能把 ID 不重叠当成全面无泄漏证明。

## 8. 结果如何决定下一步

| 结果 | 决策 |
|---|---|
| state 更像但 QA 不升 | 本状态监督未成功，不凭 cosine 宣称机制 |
| Direct-State 提升 | 补 Direct-KD、发布版 raw 教师替换，验证独立价值 |
| W-CE 提升 | 补单次/多次读取、W 数量和计算预算对照；再考虑 W-State |
| W 零步低，训练后追平 | 说明接口可适配，还不是收益 |
| 都没提升 | 记录当前预算下负结果，不自动恢复位置实验或强行解释信息丢失 |

训练日志提供总梯度、decoder/W 分组梯度；completion 提供峰值 allocated 显存和耗时。eval_seconds 包含 NLL 与生成，**不能当成纯答案 latency**；正式效率声明另做相同 batch/设备的独立计时。教师构建时间另列；W 参数量见 manifest。新路径没有理论速度保证。

## 9. 验证边界

提交前本地实际结果：`tests/test_reader_experiment.py` **12 passed**；原 `tests/test_shapes.py` **123 passed, 0 failed**；原 `tests/test_causal_order.py` **11 passed, 0 failed**。CLI help 和本说明 8 段 bash 的语法检查通过。集成测试替换了大模型加载为随机小型 Mistral＋PEFT，cache/train/eval/resume 使用本次正式代码入口；不代表真实 P₁ 数据测试。

本次应完成：Direct 与旧输入的逐 tensor 等价；state_weight=0 梯度等价；左右 padding 下 S 对齐和答案不泄漏；state loss 梯度；W 两步梯度与 padding mask；非重入 checkpoint 梯度一致；cached decode 与完整重算一致；参数保存加载；教师缓存身份错误拒绝；三个 arm 的 cache→train→eval→resume 集成检查。

尚需服务器完成：实际 P₁ 零步 2000 条复现、BF16/CUDA backend smoke、正式训练与独立 QA 评测。没有 GPU 实测结果前，不称“实现已验证有效”或“已超过 baseline”。
