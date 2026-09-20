# RQ：从 P baseline 起步的 query-as-Q 读出与写回

本变体从 `feat/pisco-identity-residual` 的 `f9c7935` 派生，独立分支为
`feat/pisco-query-writeback`。第 1–5 节是 2026-09-20 首轮运行**之前**写的方案，原样保留；
**首轮结果见第 6 节：两个数据集、一个 seed，模块边际都是零。**方案里"待验证、不代表已取得
任务收益"的限定，现在有了数据支撑，不再是预防性措辞。

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

（2026-09-20 复跑：RQ 8 项、R 3 项通过，`test_shapes.py` 现为 113 项全过——该文件在
写作本节后又有增补，102 是当时的计数，不是回归。）

---

## 6. 首轮结果（2026-09-20）

> 数据：[`results/arm_matrix.json`](../results/arm_matrix.json)（62 次运行、55 组配对检验）
> 生成：`python scripts/collect_arm_matrix.py`
> 运行目录：`/data02/quro/runs/query_writeback_ext_{init,train,joint}_s42` 及 `..._joint_s42_trivia`

### 6.0 二十秒版本

1. **恒等起点精确成立。**零初始化的 RQ 在完整 dev 上打出 54.50，与源 P **逐题 2000/2000 完全相同**（delta=0.00，0 胜 0 负）。后面的负数不是实现缺陷。
2. **HotpotQA 上模块边际 −0.20（p=0.80）。**joint 56.05 vs 匹配的 p-control 56.25。
3. **TriviaQA 零样本上 +0.10（p=0.92）。**70.50 vs 同一对照的 70.40。
4. **两个数据集上都是零，符号还相反**——这正是噪声的形状，不是"一个数据集上有效"。
5. **冻结版低于自己的起点**（54.25 vs 54.50，−0.25，p=0.66）：模块单独学不出可用信号。R 的冻结版当时是 +0.50。
6. **RQ 没有复现 R 那个本就未确立的正边际。**同 seed 同数据下 RQ 比 R 低 1.30（p=0.030）。
7. **但 RQ 与 R 的高低随数据集翻号**：HotpotQA 上 R 高 1.30，TriviaQA 上 RQ 高 0.85（p=0.11）。两个都没有稳定信号时本该如此，不能反过来当作 RQ 的战果。

### 6.1 运行配置

与 R 的第二轮（held-out）**逐项匹配**：同一个 `hp2d0_P/checkpoint_last.pt`、
`train_heldout.jsonl` 60447 条、`hotpotfull-pisco-r16` 缓存、3000 步、lr 1e-4、
seed 42、B=80 不截断、D0、明文问题、固定 query adapter、纯 CE。

```bash
TRAIN_FILE=/data02/quro/data/hotpot_full/train_heldout.jsonl \
CACHE_DIR=/data02/quro/cache/hotpotfull-pisco-r16 \
TAG=query_writeback_ext_init_s42 \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_query_writeback.sh init     # 同理 train / joint

TAGS="query_writeback_ext_joint_s42" GPUS=0 bash scripts/run_trivia_transfer.sh
```

**p-control 没有重跑**，直接用 R 的 `residual_ext_p-control_s42`。那个臂根本没有模块
（`pisco_direct`），对 R 和 RQ 是同一个对照；重跑只会引入一次无谓的 seed 抽样。
代价是 RQ 只有 seed 42 一个点，不能像 R 那样报三 seed。

可训参数：RQ readout 4.21M vs R 4.09M，同 `d_readout=256`，差 3%。
**同宽度不等于同参数量**，但这个差距不足以解释 1.30 的落差。

### 6.2 HotpotQA dev 2000，D0，B=80，last checkpoint

| 臂 | 模块 | decoder LoRA | EM | F1 | sub |
|---|---|---|---:|---:|---:|
| 源 P（`hp2d0_P`） | 无 | — | 54.50 | 0.682 | 59.80 |
| RQ `init` | 4.21M，Δ=0 | 冻结 | **54.50** | 0.682 | 59.80 |
| RQ `train` | 4.21M | 冻结 | 54.25 | 0.680 | 59.35 |
| R `train` | 4.09M | 冻结 | 55.00 | 0.684 | 59.95 |
| **`p-control`** | **无** | 41.94M | **56.25** | 0.692 | 60.70 |
| R `joint` | 4.09M | 41.94M | 57.35 | 0.701 | 61.95 |
| **RQ `joint`** | **4.21M** | 41.94M | **56.05** | 0.697 | 60.85 |

配对 McNemar（同 2000 题逐题）：

| 比较 | Δ EM | 胜/负 | p |
|---|---:|---:|---:|
| RQ `init` vs 源 P | **+0.00** | **0/0** | 1.000 |
| RQ `train` vs RQ `init` | −0.25 | 40/45 | 0.665 |
| **RQ `joint` vs `p-control`** | **−0.20** | 69/73 | **0.801** |
| RQ `joint` vs R `joint` | −1.30 | 54/80 | 0.030 |

第一行是实现检查，**0 胜 0 负**意味着两个模型在 2000 题上给出了完全一致的判对判错，
不是"分数碰巧相等"。第三行是唯一能把收益归因给 RQ 模块的比较，它是负的，且远不显著。

### 6.3 TriviaQA 2000 题零样本迁移

同一批权重，不在 TriviaQA 上训练，`gonogo-pisco-r16` 缓存，`--doc_control` 必开。

| 臂 | 正确文档 EM | 错配文档地板 | 证据值 |
|---|---:|---:|---:|
| 源 P（B=8） | 71.95 | 59.50 | +12.45 |
| `p-control` s42 | 70.40 | 54.45 | +15.95 |
| R `joint` s42 | 69.65 | 53.45 | +16.20 |
| **RQ `joint` s42** | **70.50** | **53.95** | **+16.55** |

| 比较 | Δ EM | 胜/负 | p |
|---|---:|---:|---:|
| **RQ `joint` vs `p-control`** | **+0.10** | 54/52 | **0.923** |
| RQ `joint` vs R `joint` | +0.85 | 59/42 | 0.111 |
| RQ `joint` vs 源 P | −1.45 | 73/102 | 0.034 |

**RQ 的证据值是三个 HotpotQA 训练臂里最高的（+16.55），这不作为战果。**理由与
[`RESIDUAL_RESULTS.md`](RESIDUAL_RESULTS.md) 第 6.2 节相同：证据值升高是**所有**在
HotpotQA 上继续训练的臂的共同副作用，没有模块的 `p-control` 自己就有 +15.95；
它来自参数记忆下降（地板 59.50 → 53.95）而非证据利用变强，而主指标 EM 是跌的
（−1.45，p=0.034）。这正是 D4/D5 被拒时的论证形状，不拿来给 RQ 翻案。

### 6.4 结论与不做什么

**结论：RQ 首轮无收益。**两个数据集上模块边际分别是 −0.20 和 +0.10，都落在 2000 题的
1 SE ≈ 1.1 之内，方向相反。现有证据强度**弱于 R**——R 至少在 HotpotQA 上三 seed 同号。

**不建议补 seed 43/44。**R 的 +0.73 ± 0.55 本身就未确立，RQ 在 R 表现最好的那个 seed 上
是负的；按同样的效应量外推，加 seed 更可能只是把 0 测得更准。若仍要补，joint 与
p-control 必须成对、同数据同步数，不能拿 RQ 的新 seed 去比 R 现成的 p-control。

**不能写的话**：不能说"RQ 优于 R"（TriviaQA 上的 +0.85 p=0.11，且 HotpotQA 上反向）；
不能说"Q/K 对调有效或无效"——R 与 RQ 同时改变了方向**和**文档间交互结构（R 有 latent
self-attention，RQ 没有），1.30 的差不能单独归因给 Q/K 对调；不能用证据值代替 EM 下结论。

本轮没有修改缓存、压缩器、R 的任何 checkpoint 或旧实验结果；`p-control` 是复用的既有运行。
