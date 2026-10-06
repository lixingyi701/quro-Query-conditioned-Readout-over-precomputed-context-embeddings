# 标准无 query 投影器消融：S0

`--projector_query_mode none` 在共享投影器上移除 query 条件分支，仅训练文档
残差 MLP。它与已有 `agnostic_matched`（S0m，固定随机条件向量）是两个不同
对照。本页说明实现与未来实验协议，**不包含新的真实模型 QA 结果**。

## 结构与参数

SQ 的 `h_i` 已包含 `memory_proj` 的 bias；`b_i` 是无 bias 的 `context_proj`
输出。两个分支分别映射到同一个 r 维空间，再相加，不需要改变文档 MLP 的宽度。

\[
\begin{aligned}
\mathrm{SQ}:&\quad u_i=\mathrm{GELU}(h_i+b_i),\\
\mathrm{S0}:&\quad u_i=\mathrm{GELU}(h_i),\\
&\quad E_i=Z_i+\mathrm{reshape}(W_o u_i+b_o).
\end{aligned}
\]

| 模式 | 标记 | 实际条件输入 | 默认可训练参数（无支持头、无跨文档 attention） |
|---|---|---|---:|
| `conditioned` | SQ | 真实问题 | 37,782,016 |
| `none` | S0 | 无条件分支 | 33,587,712 |
| `agnostic_matched` | S0m | 固定 4 个随机向量 | 37,782,016 |

默认 m=8、d=4096、r=512，SQ 与 S0 的文档 MLP 均为
`32768 -> 512 -> 32768`，输出仍是本篇 `8 x 4096` 个数。每篇共用同一套权重。
S0 不保留 `to_query/to_key/to_value/context_proj`、query/context LayerNorm
或固定条件向量；这些模块没有参数、优化器状态或 checkpoint 权重。

S0 的 `needs_query=False`，不会运行额外的问题编码前向。decoder 仍接收正确
问题明文、全部有效 memory 与相同 D0 prompt。缓存、冻结约束、答案 CE 和
每篇 m->m 的预算不变；不是闭卷消融，也不是关闭整个投影器。

可选跨文档 attention 与支持头仍能使用：对应 S0X、S0+Head/Doc、S0+HeadE/DocE。
做主比较时，两臂的这些开关和损失权重必须相同。首轮主消融建议关闭它们，
仅用 CE。query mode `none` 不允许 AddG/FiLM 或冻结 gamma：同时去掉 query
及其调制路径，避免 gamma bias 学成静态增益。

注意：`--projector_fusion none` 只关闭 gamma，**不会**关闭 query。

## 初始化、加载与统计

- 保留旧 SQ 的初始化顺序：构造期间消费相同的随机初始化抽样，然后在 S0 中
  移除条件模块。公共文档 MLP、可选文档 attention/支持头及后续全局 RNG 状态
  与同 seed、同设置的 SQ 逐位一致。临时初始化不参与训练或推理。
- 输出矩阵和 bias 仍零初始化，SQ/S0 第 0 步均为 E=Z；QA loss 与 decoder
  输入一致。第一步输出层获得梯度；其更新后，文档输入层也可从 CE 学习。
- `result.json` 记录 S0/SQ 标记、query mode、是否实际调用问题编码，以及真实
  可训练参数数目。S0 的条件 RMS 为零；支持日志中的条件梯度为 null，而非
  假装这些参数仍在训练。
- 普通 resume、评估和 weights-only warm start 都要求相同 query mode。
  S0 与 SQ/S0m checkpoint 不能互载；原 SQ/S0m checkpoint 无新增 layout 字段
  时仍可加载。S0 缺少公共 MLP 权重或混入条件权重会报错。
- 主消融从同一个发布版 PISCO **分别初始化并训练** SQ/S0，不从训练后的 SQ
  关闭 query 再续训。后者是另一项依赖性/恢复实验，会继承原有条件化训练历史。

## 运行协议

下面两条命令固定 seed=42、3000 步、CE-only，其余设置使用同一
`pisco_shared_projector` 预设。预设读取发布版 PISCO、相同 HotpotQA 训练数据和
相同缓存；实际路径以服务器配置为准，如需覆盖，两臂使用相同的
`--generator_path/--cache_dir/--train_file/--eval_files`。每臂使用独立进程和输出目录。

```bash
# SQ：真实问题条件，独立训练，不指定 --resume_from
python -m src.train --preset pisco_shared_projector \
  --projector_query_mode conditioned --projector_fusion none \
  --support_loss_weight 0 --seed 42 --steps 3000 --lr 5e-5 \
  --batch_size 2 --grad_accum 8 --select_metric em \
  --eval_every 250 --eval_every_samples 500 --eval_max_samples 2000 \
  --query_control --doc_control --tag query_ablation_sq_s42 \
  --out_dir /data02/quro/runs/query_ablation_s42/sq

# S0：关闭独立条件分支，保留同尺寸文档 MLP
python -m src.train --preset pisco_shared_projector \
  --projector_query_mode none --projector_fusion none \
  --support_loss_weight 0 --seed 42 --steps 3000 --lr 5e-5 \
  --batch_size 2 --grad_accum 8 --select_metric em \
  --eval_every 250 --eval_every_samples 500 --eval_max_samples 2000 \
  --query_control --doc_control --tag query_ablation_s0_s42 \
  --out_dir /data02/quro/runs/query_ablation_s42/s0
```

之后补 seed 时两臂一起改 seed 和输出目录。数据顺序、训练条数、学习率调度、
验证频率、memory 预算和 checkpoint 选择规则必须对齐。当前 trainer 的结束
评估使用 **last**；如果使用 best，两臂都应另外加载各自 best 并报告选择规则，
不要将一臂的 best 与另一臂的 last 混比。

```bash
# S0 全量 test 评估：仍须显式选择相同 query mode
python -m src.train --preset pisco_shared_projector \
  --projector_query_mode none --eval_only \
  --resume_from /data02/quro/runs/query_ablation_s42/s0/checkpoint_last.pt \
  --eval_files test=/data02/quro/data/hotpot/test.jsonl \
  --eval_max_samples 5405 --query_control --doc_control \
  --out_dir /data02/quro/runs/query_ablation_s42/s0_full_eval

# 本地契约测试，不下载真实模型
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_projector_no_query.py' -v
```

这里的 5405 是当前内部 test 文件的条数；更换数据时按实际条数设置上限，
SQ 使用同一文件和上限。不要传 0：当前入口会截取零条。

主要报告同一题目集合上的 SQ-S0 正常 QA（EM/F1/substring）及配对 CI，辅以
各 seed 的差值。S0 换投影器问题时输出应不变；SQ 的错问题掉分是功能干预，
不能替代 SQ-S0。参数减少是标准模块消融的一部分，应如实报告；S0m 可作为
额外的容量/条件输入对照，不需要把它称作直接删除分支的消融。

本次只实现开关与接口验证，没有启动服务器上的发布版 7B 训练。
