# RQ：从 P baseline 起步的 query-as-Q 读出与写回

本变体从 `feat/pisco-identity-residual` 的 `f9c7935` 派生，独立分支为
`feat/pisco-query-writeback`。它是待验证的修正器变体，不代表已取得任务收益。

## 1. 与原方法分开命名

| 方法 | arm | readout.kind | 核心路径 | 运行脚本 |
|---|---|---|---|---|
| 全量 P baseline | P | pisco_direct | Z 原样输入 | 两个脚本均有 p-control |
| 已有恒等残差 | R | pisco_residual | latent 读 query，再 latent self-attention | run_pisco_residual.sh |
| 本次新变体 | RQ | pisco_query_writeback | query 读 latent，再用 A 的转置写回 | run_query_writeback.sh |

R 的网络、默认选择和原运行脚本不变。RQ 新增独立类
`src/query_writeback.py:PiscoQueryWritebackReadout`，不复用旧 learned output slots。
配置、result.json 的 arm/kind、运行目录均区分两者。
`model.load()` 拒绝 RQ 与其他 readout checkpoint 交叉加载；从 P 初始化必须走
`--baseline_run` 的 decoder/query 权重移植路径，不用宽松 load 偷换 readout。

## 2. 实际计算

Z 为全部 M 个缓存 latent，Hq 为固定 query encoder 的 Tq 个 token 表示。
先分别 LayerNorm，再线性生成 Q、K、V，按 head 拆分。每个 head 执行：

$$
A=\operatorname{softmax}_{M}(Q(H_q)K(Z)^\top/\sqrt{d_h}),\quad
R_q=AV(Z),\quad C=A^\top R_q.
$$

各 head 的 C 拼接后：

$$
\Delta Z=W_{out}\operatorname{Dropout}(\operatorname{GELU}(C))+b_{out},
\qquad E=Z+\Delta Z.
$$

- Wout、bout 全零初始化，其余参数正常初始化。不叠加零初始化门。
- 原始 Z 不做 norm、缩放、投影、重排或截断。初始 E=Z，与 R 采用同一恒等检查。
- **A 按文档 latent 轴归一化，A 的转置不再次归一化**，保留 latent 接收信息的强弱。
- attention 形状为 `(batch, head, query_token, latent)`，不是旧 QuRO 的 output-slot 轴。
  `--dump_attn` 仍保存每批张量列表；result.json 新增 attention_axes，分析时不要误用旧 slot 图。
- 没有 latent self-attention、余弦先验、额外 query 压缩或 query skip。
- 当前只实现一次 read/write；`num_blocks != 1` 会报错，不静默忽略配置。
- 对 attention 分数、读出和写回使用 FP32（显式关闭对应运算的 autocast）；输出桥可跟随 AMP。
- padding 在投影前清理，文档 padding 不参与 softmax，query padding 对应 A 行严格为零，
  padded latent 的修正为零。全部文档或全部 query 无效的样本明确报错。
- dropout 作用在写回特征上，不对 A 随机删边，读写使用同一张 A。

这保留了初始模型功能，不表示 attention 已经预训练，也不保证训练后一定提升。
写回强度可随有效 query 长度变化，这是未额外归一化的明确设计选择，后续需观察长度泛化。
所有原 latent 仍传给生成器，不能声称进一步压缩或端到端加速。

## 3. 服务器运行

切换到 `feat/pisco-query-writeback`，在仓库根目录、已配置好 PISCO 环境中运行。
默认 baseline 为 `/data02/quro/runs/hp2d0_P`，可用 BASELINE_RUN 覆盖。
该目录必须含实际 P 的 `config.json` 和 `checkpoint_last.pt`。

```bash
# 初始模型完整 dev 评测：应复现同一配置下的 P
CUDA_VISIBLE_DEVICES=0 bash scripts/run_query_writeback.sh init

# 冻结已训练 P decoder，只训练新模块
CUDA_VISIBLE_DEVICES=0 bash scripts/run_query_writeback.sh train

# 联合训练及匹配的 P 继续训练对照
CUDA_VISIBLE_DEVICES=0 bash scripts/run_query_writeback.sh joint
CUDA_VISIBLE_DEVICES=1 bash scripts/run_query_writeback.sh p-control
```

新运行默认目录为 `query_writeback_{init,train,joint,p-control}_s42`，旧 R 的
`residual_*` 不变；目录存在时脚本拒绝覆盖。SEED、TAG、STEPS、LR、D_READOUT、
BASELINE_RUN、QURO_RUNS_DIR 均可覆盖。

沿用第二轮的留出训练集时，R、RQ 和 P 对照必须采用同样的数据设置，例如：

```bash
TRAIN_FILE=/data02/quro/data/hotpot_full/train_heldout.jsonl \
CACHE_DIR=/data02/quro/cache/hotpotfull-pisco-r16 \
TAG=query_writeback_ext_joint_s42 \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_query_writeback.sh joint
```

脚本与 R 一样继承 P 的数据和模板，默认 d_readout=256、1 个阶段、D0、明文问题、
固定 query adapter、CE、B 标签 80、不做 budget dropout。P/R/RQ 实际都保留全部有效 latent。
输出包含 baseline_identity.json，失败会停止；两条样本的检查不替代完整 dev 初始化评测。

间隔验证保存 best；训练末尾默认完整评测 last，与旧脚本一致。若比较 best，
请为各方法使用同一验证选模规则，并显式从各自 best checkpoint 评测：

```bash
python -m src.train \
  --config_json /data02/quro/runs/query_writeback_joint_s42/config.json \
  --resume_from /data02/quro/runs/query_writeback_joint_s42/checkpoint_best.pt \
  --out_dir /data02/quro/runs/query_writeback_joint_s42_best_eval \
  --eval_only --eval_input_modes D0 --eval_budgets 80
```

RQ 不使用 output_query_mode 的 learned-slot 配置；结果中该项为 null，日志显示
query_as_q_writeback。判断方法请以 arm/kind 为准。已有汇总脚本的历史 REGISTRY
未添加尚未运行的实验，不将新方法写进旧结果表。

## 4. 比较与限制

首轮在相同 P checkpoint、数据、seed、训练长度、decoder 策略和选模规则下比较
P / R / RQ。同宽度不意味着参数量相同，请保留程序的 parameter_report；R 与 RQ
同时改变了方向和文档间交互结构，不能把全部差异单独归因为 Q/K 对调。

先测冻结 decoder 的 RQ，再视情况比较 joint 与匹配 p-control。这个实现没有替代
已有实验结论，也没有修改缓存、压缩器、原 R checkpoint 或旧实验结果。

## 5. 本次验证

```bash
OMP_NUM_THREADS=1 python tests/test_query_writeback.py
OMP_NUM_THREADS=1 python tests/test_refinement.py
OMP_NUM_THREADS=1 python tests/test_shapes.py
bash -n scripts/run_query_writeback.sh
```

2026-09-20 CPU 验证：RQ 8 项、R 3 项、原 shape/行为检查 102 项通过。
包含手工非均匀单 query 读写算例、Q/K 方向、NaN padding 隔离、长度 padding 不变性、
latent 重排等变性、梯度解锁、CPU BF16 autocast、训练过的 P 权重移植、冻结 decoder
与 checkpoint 恢复、R/RQ 误加载拒绝。

另通过 `src.train` 的 toy 两步训练、自动初始恒等检查、保存及重新加载评测，
并检查结果标识和 attention 轴。未运行真实 7B GPU 训练；尚无 HotpotQA/TriviaQA 新分数。
