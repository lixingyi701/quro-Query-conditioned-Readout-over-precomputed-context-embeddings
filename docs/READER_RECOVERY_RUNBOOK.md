# Reader 首轮退化：诊断与短程修复

2026-10-02。分支 `feat/reader-state-workspace`。本文件是首轮三臂训练后的追加探索方案，不改写原预登记或替换原 best 结果。

## 1. 当前问题与边界

依据用户提供的服务器摘要，尚未读取服务器原始日志：Direct-CE 在 dev 前 500 条上 substring 从 .640 到 .600，NLL 从 .604 到 .793；W 从 .278 到 .274；Direct-State 在约 2240 步时无改善。可以开始诊断，不必等待最后几百步。正在运行的任务可自然结束，不需要杀进程。完整最终数字另行记录。

| 问题 | 目前支持 | 尚不支持 |
|---|---|---|
| Direct-CE 训练 loss 很低、dev 退步 | 当前继续训练方案损害泛化 | 数据已彻底“用尽”；一定只由过拟合导致 |
| Direct-State 无 QA 增益 | 当前配置没有提供可见收益 | 仅凭不同 batch 的 loss 就断言无法对齐；仅凭 loss 量级断言梯度被压制 |
| W 验证表现几乎不变 | 当前新接口未形成有效的验证集能力 | Z 必须在序列内才能使用；所有 workspace 方法都无效 |

代码事实：W 的 CA 输出矩阵零初始化，所以 step=0 完全不依赖 Z。非零 workspace 总梯度和低训练 CE 均不能单独证明文档利用。初轮旧接口 LoRA 与全新接口同时训练且只用答案 CE，是需要排查的训练设计。

不重做位置效应实验；不恢复旧 embedding editor；暂不重建教师缓存或三臂重跑 90k 数据。

## 2. 更新与已有实验收尾

在新的 shell/tmux 中更新。已运行 Python 进程无需重启；保留原始输出目录。

```bash
git fetch origin
git switch feat/reader-state-workspace
git pull --ff-only
conda activate selecom
OLD=/data02/quro/runs/reader_state_workspace_v1
REC=/data02/quro/runs/reader_recovery_v1
P1_CKPT=/data02/quro/runs/oscale_P1/checkpoint_last.pt
TRAIN=/data02/quro/data/hotpot/train.jsonl
DEV=/data02/quro/data/hotpot/dev.jsonl
mkdir -p "$REC/snapshots"
```

原 best 选择结果仍是主记录。全量 last K=10/K=2 是补充分析，不用其结果反向修改 best 规则。可以复用原 runbook 第 7 节，仅将 checkpoint 换为 last、输出目录另取名。Direct 两个 step-zero best 若完全相同，可复用其评测。W best 是新接口零步，不等于 P₁。

以下诊断使用快照，避免运行中的 `checkpoint_last.pt` 被后续保存替换。只复制一次；不要把同一快照路径覆盖为新 step。`last` 可能是最近保存步而非日志当前步，诊断 JSON 记录实际 step。

```bash
cp -n "$OLD/w_ce_s42/checkpoint_last.pt" "$REC/snapshots/w_last.pt"
cp -n "$OLD/direct_state_s42/checkpoint_best.pt" "$REC/snapshots/state_best.pt"
cp -n "$OLD/direct_state_s42/checkpoint_last.pt" "$REC/snapshots/state_available.pt"
```

此时 best 若仍是 step=0，便是 P₁ 的状态对照。若 best 已变，使用之前保留的 step-zero reader checkpoint，不要把原 P₁ checkpoint 直接传给诊断工具（格式不同）。

## 3. W 是否依赖证据：现在就可以做

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/diagnose_reader_experiment.py evidence \
  --checkpoint "$REC/snapshots/w_last.pt" --eval_file "$DEV" \
  --limit 500 --batch_size 8 --out_dir "$REC/w_evidence"
```

固定同一批问题，运行三种条件：正确 Z、其他问题的 Z、将所有 CA gate 置零。第三项保留训练后的 W/LoRA，只在推理时切断外部 Z 输出，随后恢复 gate，不改 checkpoint。不是修改 prompt 位置。

错配规则：按文档数分组，使用确定性的循环置换，要求每对 donor/recipient 文档 ID 完全不相交；不交换问题或答案，不随 batch size 改变 donor。无法组成对照就报错，可扩大样本量。`donors.json` 保存完整映射；文档不相交不保证语义或答案完全无关，解释时仍需注意。

`summary.json` 给出各条件 QA/NLL、all/bridge/comparison 的配对 bootstrap，以及输出字符串改变率。差值方向为干预减正确：NLL 上升、QA 下降支持对应证据有用；变化很小提示当前路径作用弱；显著改善提示路径可能有害。CI 是给定 donor 置换与 checkpoint 的样本不确定性，不是训练 seed 方差。

不要只看“输出变了”：变化可以无益。也不要把不显著当作严格等效。正确 Z 与关闭 CA 几乎一样时，下一轮优先处理证据通路的训练；存在有益依赖但泛化差时，优先研究训练泛化，而非宣布它没读到东西。

## 4. 固定样本状态对齐与梯度

复用完整 train 教师缓存；不需要 raw 教师再跑一次，不把 dev 放进训练缓存。默认用原训练文件前 128 条固定样本，不声称是留出泛化结果。

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/diagnose_reader_experiment.py state \
  --checkpoints "$REC/snapshots/state_best.pt" "$REC/snapshots/state_available.pt" \
  --teacher_cache "$OLD/teacher_train" --limit 128 --batch_size 1 \
  --grad_batches 8 --out_dir "$REC/state_fixed"
```

每个 checkpoint 都用相同问题、同一教师状态、eval 模式关闭 dropout：

- prompt-only 前向提取 S，报告每题/每层 `1-cos` 与 relative-L2（分母为教师状态范数）。不输入答案 token。
- 前 8 个 batch 额外用 gold CE 的 teacher-forcing 前向，在同一次前向上分别求 CE 与 state 对 decoder LoRA 的梯度，不做更新。
- 报告 `||g_CE||`、`||g_state||`、`lambda*||g_state||/||g_CE||` 及两梯度夹角余弦。负余弦表示当前固定 batch 上方向冲突；不是在所有训练样本上成立的结论。
- 这里的梯度是 eval 模式下的确定性诊断，不等同于训练日志中的 dropout 梯度。Direct-CE checkpoint 若加入对照，其 state 梯度是反事实计算，不表示该组曾使用 state loss。

先逐层/逐样本比较，不再把训练日志不同 batch 的 state loss 首尾当作对齐曲线。若状态距离确实下降但 QA 无益，优先质疑监督对象；若未下降，结合梯度比值与冲突判断优化问题。小比值支持“局部状态梯度弱”，仍不自动决定把 lambda 放大多少。

如显存不足，维持 batch=1，先设 `--grad_batches 0` 做纯前向；梯度部分需要单独的空输出目录再运行。新脚本有实际 backward 诊断，但不进行任何 optimizer step。

## 5. Direct 短程训练：先拆开学习率与目标变化

新增 `--train_target gold|p1`，默认仍是 gold，不改变第一轮命令行为。`p1` 读取 P₁ checkpoint 中 `prefer_teacher_output`：原设置偏好教师且该行有教师答案时用该答案，否则使用 gold。它是继承目标选择规则，不是完整恢复 P₁ 的优化器、调度器或所有训练设置。

训练 manifest 记录实际 gold/teacher 条数、实际 target 与第一条 gold 不同的条数。若 teacher 条数为 0，或实际答案全部一致，目标切换就不是有效解释，不重复跑等价目标组。验证/评测仍强制 gold，所以各组 NLL 可比。缓存中间状态不依赖所选 CE 答案，原教师缓存不必因目标选项而重建。

三个固定短程配置，均从原 P₁ 权重重新启动，300 步、相同 batch/seed、每 50 步验证。它们用于诊断，不能直接与第一轮的 step=300 曲线作严格学习率比较，因为总步数改变了 cosine schedule。

| 配置 | 目标 | 峰值 LR | 对照含义 |
|---|---|---|---|
| G-high | gold | 1e-4 | 在相同短程 schedule 下的高 LR 对照 |
| G-low | gold | 1e-5 | 与 G-high 隔离 LR 改变 |
| P-low | 继承 P₁ 偏好 | 1e-5 | 与 G-low 隔离目标选择改变 |

三组不是完整 2×2 因子实验，不估计所有交互。不要读到某组落后就临时延长步数；若后续扩展预算，重新登记同预算对照。

```bash
SHORT=(--init_checkpoint "$P1_CKPT" --train_file "$TRAIN" --eval_file "$DEV"
  --arm direct-ce --steps 300 --seed 42 --batch_size 2 --grad_accum 8
  --warmup_ratio 0.05 --weight_decay 0.01 --grad_clip 1.0 --grad_checkpointing
  --layers 8,16,24 --eval_every 50 --eval_samples 500 --eval_batch_size 8
  --save_every 50 --select_metric substring --max_new_tokens 32)

CUDA_VISIBLE_DEVICES=2 python scripts/run_reader_experiment.py train \
  "${SHORT[@]}" --train_target gold --decoder_lr 1e-4 --out_dir "$REC/gold_high"
CUDA_VISIBLE_DEVICES=2 python scripts/run_reader_experiment.py train \
  "${SHORT[@]}" --train_target gold --decoder_lr 1e-5 --out_dir "$REC/gold_low"
CUDA_VISIBLE_DEVICES=2 python scripts/run_reader_experiment.py train \
  "${SHORT[@]}" --train_target p1 --decoder_lr 1e-5 --out_dir "$REC/p1_low"
```

上面按顺序运行，也可在独立 tmux/GPU 并行。按首轮约 3.6 秒/步估算，每组训练计算约 18 分钟，验证与 I/O 另计；不是服务器实测承诺。不会自动提交新 GPU 作业。

先检查各 manifest 的实际 target 统计。若 low LR 减轻退化，下一轮用较小 LR；若同 LR 下继承目标更稳定，后续 Direct-CE/State 匹配该目标；若两者都不解决退化，才优先扩大未见问题数据。用 QA 和 gold NLL 一起判断，不按训练 CE 或状态 loss 最低选模型。不把单 seed、500 条上的微小差异宣布为方法胜利。

## 6. 下一步按诊断分支决定

- Direct 配置稳定后再加 state，与匹配目标/LR/预算的 Direct-CE 比较。状态梯度弱、冲突强、或已对齐却无益，是三个不同问题，不能统一用提高 lambda 解决。
- W 若没有有益证据依赖，下一次候选是先冻结 decoder 学习读取接口，再低 LR 联合训练；这尚未被验证，本提交没有擅自替换 W 架构或启动该训练。先用本轮诊断决定是否值得实现，避免同时引入更多变量。
- `hotpot_full/train.jsonl` 的 90k 不是天然未见数据。先与 P₁ 30k 按问题 ID/内容做差集，核对 dev 重叠、缓存规格与截断一致性，再让 Direct-CE 试新数据；不把缓存目录里的 `r16` 字样直接当作已核对兼容。
- 保留原 best、补充 last、诊断输出与新短程实验各自身份。旧 checkpoint 可直接 eval/诊断，缺少 `train_target` 的旧 checkpoint 在 resume 时按 gold 解释；resume 不能改成 p1 或换 LR 来冒充同一轨迹。

## 7. 本地验证范围

tiny Mistral+PEFT CPU 测试覆盖三臂 cache/train/eval/resume、旧参数 checkpoint 恢复、训练目标继承而 eval 保持 gold、错配文档置换、禁用 CA 后恢复 gate、固定状态诊断和梯度计算。实际 P₁/GPU/服务器数据上的效果尚需执行上面命令，代码测试不代表方法已修复成功。
