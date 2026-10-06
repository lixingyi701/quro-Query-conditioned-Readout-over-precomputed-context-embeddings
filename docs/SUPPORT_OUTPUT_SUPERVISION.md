# 支持监督后续：离线 top-k 诊断与输出端监督（2026-10-06）

对应两个独立问题：已有分类器能否在较小预算下保留证据；支持损失直接经过
输出矩阵后能否改善 QA。top-k 只做诊断，本轮训练和生成仍使用全部 80 个 memory。
首轮结果见 [SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md](SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md)，
标签与可见性规则见 [SUPPORT_DOCUMENT_SUPERVISION.md](SUPPORT_DOCUMENT_SUPERVISION.md)。

## 任务一：从已有预测补齐证据覆盖率

入口：[`scripts/analyze_support_topk.py`](../scripts/analyze_support_topk.py)。
只依赖 Python 标准库，读取 predictions JSON 数组，不加载模型、原文或缓存。

```bash
task_support_run=/data02/quro/runs/support_doc_v1/sq_ce_doc
python scripts/analyze_support_topk.py \
  --predictions "$task_support_run/predictions_dev_D0_Bfull.json" \
  --mismatch_predictions "$task_support_run/predictions_dev_mismatch-q_D0_Bfull.json" \
  --topk 2,4,6 --memories_per_document 8 \
  --output_json "$task_support_run/support_topk_dev.json"
```

输出正确/错误 readout query 下的 Recall/both@2/@4/@6，全部已标注 gold 都可见的
子集，以及换 query 后入选集合不变的比例。按**集合**比较，名次互换不算改变。
配对差为错误 query 减正确 query，bootstrap 默认 2000 次、seed 0。CLI 用百分数，
JSON 指标和 CI 用 0–1 比例。hypothetical memory budget 仅是假设保留的向量数。

Recall@k = 入选 gold 数 / 已标注 gold 数；both@k 只对恰有两篇 gold 的题定义。
使用原始 `labels` 和 `label_mask`，不用训练 `loss_mask`：不可见正例仍是评估
正例。无标签/无 gold 的题跳过并计数。k 大于有效文档数时保留全部有效文档，
报告实际平均预算，不能把小 K 的高召回解释成 K=10 的筛选能力。

重复 id、配对 id/标签/文档顺序不一致，以及有效位置的非有限 logits 会报错。
旧预测没有 `doc_ids` 仍可分析，但须确保传入原实验的 query-only mismatch；
`document_order_verified_questions` 记录有双方 doc_ids 可核验的题数。
`mismatch_shares_answer_questions` 提示共享答案的错问题不是独立强反例。
输出不能覆盖输入预测。空子集指标是 null，不伪造为零。

这些数字不等于筛选后的 QA 或实际加速。分类主要靠文档先验依然可能有筛选
用途，是否改善 QA/时延要另做生成评估。此任务不要求重新训练。

## 任务二：分类头接最终 E

新增配置/CLI：`--support_head_input hidden|output`，默认 hidden 保留旧头。

\[
\begin{aligned}
u_i&=\operatorname{GELU}(W_z\operatorname{vec}(\operatorname{LN}Z_i)
                       +W_c\operatorname{vec}(\operatorname{LN}C_i)+b),\\
E_i&=Z_i+\operatorname{reshape}(W_o u_i+b_o),\\
p_i&=\operatorname{LN}\left(\frac1m\sum_{j=1}^{m}E_{i,j}\right),\\
s_i&=a^\top p_i+b_s,\qquad
\mathcal L=\mathcal L_{QA}+\lambda\mathcal L_{doc}(s,y).
\end{aligned}
\]

output 头读取**同一组送给 decoder 的 E**，不 detach。池化和无仿射 LN 只用于
分类分支，生成向量保持原尺度和顺序。encoder、decoder、两组 LoRA 全冻结，
不加门控/硬选择，不改每篇 8→8 或 D0；生成时不计算分类头，不需要训练标签。

| 头的位置 | 分类参数 | 辅助损失直接更新 Wo | 生成使用分数 |
|---|---:|---|---|
| hidden：LN(u) | 512+1=513 | 否 | 否 |
| output：LN(mean(E)) | 4096+1=4097 | 是 | 否 |

旧头更新 u 已经会改变 E，且答案 CE 一直更新 Wo，不能把首轮阴性结果归结为
“辅助梯度与生成完全隔断”。本轮检验直接监督最终生成输入是否有训练价值。
写 e=vec(E)、g=∂L_doc/∂e，则 ∂L_doc/∂Wo=g uᵀ、∂L_doc/∂u=Woᵀg。
从零初始化且 Wo=0 时，第一步后者为零；辅助损失先更新 Wo，随后才能到达
query 分支。主实验使用已有 SQ 的非零输出矩阵，测试核验两种梯度状态。

均值池化只约束八个槽位的平均特征，不保证每个槽位承载支持事实。E 仍含 Z，
分类仍可能利用文档先验；共享 E 也不保证冻结 decoder 使用分类特征。梯度
非零只验证接口，不能提前证明 QA 或 query 条件化收益。

## 一对匹配训练命令

两臂都建立同样的 4097 参数头，从同一个**尚未加 Doc 监督的 SQ last**开始。
只加载权重，重置 optimizer/scheduler；初始化 seed、数据顺序和训练预算一致。
不要从上一轮 SQ+Doc 开始，它已受到旧辅助目标更新。可见性注释直接复用。
先做短 smoke 核对第 0 步 QA 与 SQ 一致，再在同等独占资源下运行。

```bash
task_sq_checkpoint=/data02/quro/runs/shared_projector_v1/shared_contextual/checkpoint_last.pt
task_output_args=(
  --preset pisco_shared_projector
  --support_head --support_head_input output
  --resume_from "$task_sq_checkpoint" --warm_start
  --steps 1000 --lr 2e-5 --seed 42 --batch_size 2 --grad_accum 8
  --select_metric f1 --eval_every 100 --eval_every_samples 500
  --eval_max_samples 2000
  --train_file /data02/quro/data/hotpot-support-visible/train.jsonl
  --eval_files dev=/data02/quro/data/hotpot-support-visible/dev.jsonl
  --support_visibility_policy visible --query_control --doc_control
)
python -m src.train "${task_output_args[@]}" --support_loss_weight 0 \
  --out_dir /data02/quro/runs/support_output_v1/sq_ce
python -m src.train "${task_output_args[@]}" --support_loss_weight 0.1 \
  --support_warmup_steps 100 \
  --out_dir /data02/quro/runs/support_output_v1/sq_ce_doc
```

标识为 **SQ+HeadE / SQ+DocE**。λ=0 的头保持随机，其文档指标不是训练好的
query-free selector。主比较是两个新臂的 QA 配对差。旧 hidden 实验仅作历史
参照：头维度不同，不是严格只改变监督位置的容量对照。

复用已有 QA 配对脚本，并对新输出运行同一 top-k 脚本：

```bash
python scripts/analyze_support_supervision.py \
  --run_dir /data02/quro/runs/support_output_v1 \
  --projector_dir /data02/quro/runs/shared_projector_v1 \
  --dev /data02/quro/data/hotpot/dev.jsonl
python scripts/analyze_support_topk.py \
  --predictions /data02/quro/runs/support_output_v1/sq_ce_doc/predictions_dev_D0_Bfull.json \
  --mismatch_predictions /data02/quro/runs/support_output_v1/sq_ce_doc/predictions_dev_mismatch-q_D0_Bfull.json \
  --output_json /data02/quro/runs/support_output_v1/sq_ce_doc/support_topk_dev.json
```

主分析比较 last。训练自动评估 last；best 如需评估，两臂用相同选择规则，
显式 eval-only 加 `--support_head --support_head_input output` 和对应路径，不加
warm_start。普通 resume/eval 必须匹配头的位置；旧无头 SQ 仅在显式 warm start
时允许加新头。旧 hidden 头可按默认模式加载，但不能静默换成 output；新头
权重缺失会报错。

config/result/checkpoint 均记录 support_head_input；新评估保存 doc_ids 和
@4/@6/全 gold 可见子集指标。日志增加 output_projection_grad_norm，它是
裁剪后的**联合**梯度，包含答案 CE，不能当作单独辅助梯度。单独路径由测试验证。

## 判据与后续选择

- 主判据：SQ+DocE 的 QA 配对差优于 SQ+HeadE，CI 支持正收益且主要子集没有
  明显退化。单 seed 仅视为初步证据，不提前声称稳定收益。
- QA 改善仍需错 query 对照；错 query 掉分更多本身不算成功。
- 分类改善、QA 不变：支持损失到达 Wo 仍不足以改善生成。再讨论显式任务
  耦合，不立即叠加门控，也不能继续归因于旧出口分离。
- top-k 覆盖率好：支持评估筛选后 QA/时延，不证明投影重写改善。进入筛选或
  门控路线前，做同 checkpoint 的全部文档与 oracle gold-only 诊断。

oracle 筛选不是这次输出监督训练的前置门槛。不同时改 query 表示、预算或冻结
策略。本地没有 7B checkpoint、服务器预测或 GPU；不提供真实 @4/@6 数字，
不宣称新训练已有收益。

## 核验

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*projector.py' -v
python -S -m unittest discover -s tests -p test_support_topk.py -v
OMP_NUM_THREADS=1 python tests/test_shapes.py
python -m compileall -q config.py src scripts/analyze_support_topk.py
git diff --check
```

核验覆盖最终 E/尺度不变、独立辅助梯度和零初始化门槛、冻结 reader/LoRA、
标签不进 prompt、两臂主 trainer、旧 checkpoint/头位置不匹配/缺权重，以及
集合比较、不可见正例、padding、空子集、错位、重复 id 和无第三方依赖的 CLI。

本次本地核验：投影器系列 **65 项通过**，纯标准库 top-k **8 项通过**，已有
shape/integration **113 项通过**，无跳过；编译和 diff 检查通过。BF16 Mistral
与两组 PEFT LoRA 测试使用随机小模型，验证接口与冻结约束，不代表发布版 QA 效果。
