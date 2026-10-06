# 问题条件的文档特征调制：等容量加法 vs FiLM（2026-10-06）

本轮检验：在相同 PISCO 缓存、冻结 reader、输入预算和训练数据下，问题条件的
乘性调制能否比等容量加法更好地整理送给 decoder 的证据。保持当前 HotpotQA
配置，不构造混合文档池，不改变 query 表示，不进行文档删除。

最新 [DocE 结果](SUPPORT_OUTPUT_SUPERVISION_RESULTS.md) 已完成：DocE−HeadE 为
−0.01 F1 [−0.35,+0.31]，没有检出 QA 收益或显著损害。它们检验支持监督的输出
接口；这里的一对新臂检验融合方式，两个问题不应混写为同一个结论。

## 公式与代码

沿用 `SharedDocumentProjector` 的冻结独立问题前向、memory→query cross-attention、
context LayerNorm，以及每篇文档共享的 m→m MLP。代码中的文档分支带 bias，
问题 context 分支不带 bias：

\[
\begin{aligned}
h_i &= W_z\operatorname{vec}(\operatorname{LN}Z_i)+a_z,\\
b_i &= W_c\operatorname{vec}(\operatorname{LN}C_i),\\
\gamma_i &= \tanh(W_\gamma\operatorname{LN}(b_i)+a_\gamma),\\
u_i^{\rm AddG} &= \operatorname{GELU}(h_i+b_i+\gamma_i),\\
u_i^{\rm FiLM} &= \operatorname{GELU}(h_i+b_i+\gamma_i\odot h_i),\\
E_i &= Z_i+\operatorname{reshape}(W_o u_i+a_o).
\end{aligned}
\]

`fusion_norm` 无仿射参数，`gamma_proj` 是 r→r Linear，weight/bias 全零。
两臂新增参数相同，r=512 时是 512²+512=262,656。γ 逐特征取值于 [−1,1]，
FiLM 中的文档增益是 1+γ；它调制投影器隐特征，原始 Z 的残差系数仍为 1。

γ 的条件不是单独的问题向量：C 已由该文档的槽读取问题产生。因此准确名称是
**问题条件、按文档自适应的特征增益**。γ 有 bias，允许学习共同增益；结构
提供条件化通路，不保证训练后必然使用问题。

| 配置 | 预激活 | 模块 | 臂标签（λ=0.1，output 头） |
|---|---|---|---|
| `--projector_fusion none`（默认） | h+b | 原版，无 gamma_proj | SQ+DocE |
| `--projector_fusion additive` | h+b+γ | 新增 γ，加法对照 | SQ+DocE+AddG |
| `--projector_fusion film` | h+b+γ⊙h | 同一个 γ 模块，乘性调制 | SQ+DocE+FiLM |

默认 `none` 保留旧参数键和旧行为。两个新臂的主比较只改变融合运算；旧 DocE
是历史参照，不能替代等容量 AddG。两种注入的幅度本来不同：AddG 是 γ，FiLM
是 γ⊙h。预先测量并报告尺度，不同时加入幅度匹配或独立学习率。

## 初始化、梯度和加载

γ 在支持头之后创建，保留现有 memory/query/输出矩阵和支持头的随机初始化
顺序。从同一尚未加 Doc 监督的 SQ last 做 weights-only warm start；γ=0，
第 0 步 E、QA loss 和生成行为与 SQ 一致，两臂支持头也按同一 seed 初始化。

设 δ 为答案损失对融合后 GELU 预激活的梯度，在 γ=0 时：

\[
\frac{\partial L_{QA}}{\partial W_\gamma}=
\begin{cases}
\sum_i\delta_i\operatorname{LN}(b_i)^\top,&\text{AddG},\\
\sum_i(\delta_i\odot h_i)\operatorname{LN}(b_i)^\top,&\text{FiLM}.
\end{cases}
\]

SQ 的 Wo 已非零，条件非退化时第一步就可更新 γ。若从原生 Wo=0 开始，第一步
QA 梯度不会到达 γ；不能把 warm start 的结论推广到所有初始化。

- 仅 `--warm_start` 会同时启用新增支持头和新增融合模块的加载许可，且不加载
  optimizer/scheduler。旧 checkpoint 缺少 fusion 元数据时按 `none` 处理。
- 从 `none` 显式 warm start 到 AddG/FiLM 时，新 γ 重置为零；其余原有投影权重
  必须完整。即使模型对象之前用过，也不保留残留 γ 权重。
- 普通 resume / `--eval_only` 要求融合方式一致且 γ 的 weight/bias 完整；显式
  warm start 也不允许将已训练 AddG checkpoint 静默换成 FiLM。
- 融合方式写入 config.json、checkpoint layout、result.json 和臂标签。
- Encoder、decoder、两套原 LoRA 全冻结。训练目标仍为答案 CE + λ 支持监督，
  分类头仍读取同一 E；生成不需要支持标签，也不运行分类头。

## 诊断日志

每个 train 日志步骤及验证/最终评估输出：

| 字段 | 含义 |
|---|---|
| `fusion_h_rms` / `fusion_b_rms` | 文档／问题 context 两个 MLP 分支的尺度 |
| `fusion_gamma_rms` | γ 的幅度 |
| `fusion_product_rms` | γ⊙h 的幅度，两臂都计算，AddG 中它不是实际注入 |
| `fusion_update_rms` | 实际注入：AddG 的 γ 或 FiLM 的 γ⊙h |
| `fusion_product_over_b_rms` | RMS(γ⊙h)/RMS(b) |
| `fusion_update_over_b_rms` | 实际注入幅度/RMS(b) |
| `gamma_projection_grad_norm` | γ Linear 裁剪后的联合梯度 L2 范数（训练日志） |

平方和与有效元素数跨 microbatch/评估 batch 相加后再开方，不平均 batch RMS；
排除文档 padding，不持有诊断计算图。b 恰为零时比例写 JSON null，非有限统计
报错。train RMS 对应 optimizer.step **之前**的前向，梯度来自 QA+λDoc；不能
把其非零直接解释为 QA 单独贡献。代码测试另行验证纯答案 CE 的梯度。

`validation.step=0` 记录 SQ 起点的 h/b 和全零 γ。1000 步是否足够应结合实际
注入、梯度、训练曲线判断；γ≈0.01 不是通用的失败或“没测出来”阈值。小扰动
也可能经 Wo 和 reader 放大。错 query 掉分更大本身不是成功判据。

## 服务器启动：一对匹配的 1000 步续训

脚本为 `scripts/run_projector_fusion.sh`，默认与 DocE 一致：lr=2e-5、seed=42、
batch=2、grad_accum=8、λ=0.1、支持损失前 100 步线性加权；验证 dev500，最终
评估 dev2000，checkpoint_last 为主报告。每篇 8 个槽、K=10 时仍为 80 memory。
正常问题评估先于错问题控制，`--query_control --doc_control` 沿用 DocE 的配置。

先进入服务器仓库并拉取本分支；确认 GPU2/3 空闲后运行：

```bash
git switch feat/pisco-joint-query-projector
git pull --ff-only
bash scripts/run_projector_fusion.sh additive --dry-run
bash scripts/run_projector_fusion.sh film --dry-run

mkdir -p /data02/quro/runs/query_fusion_v1/logs
tmux new-session -d -s fusion_addg 'env CUDA_VISIBLE_DEVICES=2 bash scripts/run_projector_fusion.sh additive > /data02/quro/runs/query_fusion_v1/logs/additive.log 2>&1'
tmux new-session -d -s fusion_film 'env CUDA_VISIBLE_DEVICES=3 bash scripts/run_projector_fusion.sh film > /data02/quro/runs/query_fusion_v1/logs/film.log 2>&1'
```

两臂可以与 DocE 并行，前提是资源独占。脚本不替用户选择 GPU，不覆盖已有
train_log/checkpoint_last；已有 tmux 名称也不会被替换。默认输出目录是
`/data02/quro/runs/query_fusion_v1/{additive,film}`。

可统一设置 `SQ_CHECKPOINT`、`FUSION_RUN_ROOT`、`SUPPORT_DATA_DIR`，模型/缓存路径
沿用 preset 和现有环境。本提交按用户贴出的配对方案保留 λ=0.1 为启动默认；
最新结果报告另建议下一轮 CE-only（λ=0）。若采用该建议，在**两臂都未启动**时
统一设 `SUPPORT_LOSS_WEIGHT=0`（仍保留同一支持头），臂标签相应成为
SQ+HeadE+AddG / SQ+HeadE+FiLM；两臂始终保持相同设置。两种训练目标是不同的
预先约定，不能看某一臂结果后仅修改另一臂，也没有自动择优逻辑。

第 0 步核对两臂 F1 与 SQ 起点一致、γ_RMS=0、h/b_RMS 一致。第一步日志的 QA
loss 应一致，γ 梯度应有限且非零。不要从旧 SQ+Doc 或某一融合臂接着训练另一臂。

## 结果分析与确认判据

```bash
python scripts/analyze_projector_fusion.py \
  --additive_run /data02/quro/runs/query_fusion_v1/additive \
  --film_run /data02/quro/runs/query_fusion_v1/film \
  --split dev --output_json /data02/quro/runs/query_fusion_v1/fusion_dev.json
```

分析器仅用标准库：正常问题下 EM/F1/substring 的 FiLM−AddG 配对差和 bootstrap
95% CI（2000 次，seed=0）；两臂 normal−mismatch-q 的 F1 掉分与掉分差；最终
评估的调制统计。JSON 为 0–1，终端 QA 表为百分点。不同 ID 集、重复 ID、
问题/答案/文档顺序不一致、错问题控制不匹配会拒绝，不静默取交集。它不会
自动启动额外训练或覆盖输入 JSON。

预先约定顺序与结论：

1. 主指标是正常问题下 FiLM−AddG 的 F1。dev2000 为正则进入相同 last checkpoint
   的项目全量 test5405 确认，不依据 test 换 checkpoint、λ 或融合公式。
2. 只有 test 的配对 CI 也支持正收益，才算本设置下的初步结构收益；单 seed
   不宣称稳定。报告 EM/substring、bridge/comparison 等子集，不能只选有利项。
3. 错 query 和调制统计用于解释。更强错问题掉分、非零 γ 或非零梯度均不能
   单独证明增益来自有效的 query 条件化；若正确 QA 不升，不算成功。
4. 阴性结果限定于这个初始化、数据和训练预算，不直接否定所有乘性调制。若
   实际注入很弱，先据日志诊断；本轮不自动叠加新损失、独立 LR 或 null key。

若已按 [支持监督文档](SUPPORT_DOCUMENT_SUPERVISION.md) 准备注释，test.jsonl
应在同一 `hotpot-support-visible` 目录中。确认其确有 5405 条；缺少注释时使用
原注释脚本处理 test，不重新编码缓存。以下为顺序评估命令，不带 warm start：

```bash
task_fusion_root=/data02/quro/runs/query_fusion_v1
for task_fusion_mode in additive film; do
  CUDA_VISIBLE_DEVICES=2 python -m src.train \
    --preset pisco_shared_projector --projector_fusion "$task_fusion_mode" \
    --support_head --support_head_input output --support_loss_weight 0.1 --support_warmup_steps 100 \
    --eval_only --resume_from "$task_fusion_root/$task_fusion_mode/checkpoint_last.pt" \
    --train_file /data02/quro/data/hotpot-support-visible/train.jsonl \
    --eval_files test=/data02/quro/data/hotpot-support-visible/test.jsonl \
    --eval_max_samples 5405 --support_visibility_policy visible --query_control \
    --out_dir "$task_fusion_root/${task_fusion_mode}_test5405"
done
python scripts/analyze_projector_fusion.py \
  --additive_run "$task_fusion_root/additive_test5405" \
  --film_run "$task_fusion_root/film_test5405" --split test \
  --output_json "$task_fusion_root/fusion_test5405.json"
```

若训练统一使用 λ=0，确认命令也统一使用 0。全量预测必须有 5405 个相同 ID。
项目 test 是此前固定的数据划分，不能描述为官方隐藏测试集。较早评估过的
test 前缀需在最终报告中说明，已有的子集分析不能替代全量配对确认。

现有头按分数 top-6 **生成**的任务另行接入；`analyze_support_topk.py` 仍只计算
覆盖率。本轮 γ 融合没有暗中实现文档过滤或宣称已验证实际加速。

## 本地核验与边界

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*projector.py' -v
OMP_NUM_THREADS=1 python tests/test_projector_fusion.py -v
python -S tests/test_projector_fusion_analysis.py -v
OMP_NUM_THREADS=1 python tests/test_shapes.py
python -m compileall -q config.py src scripts tests
bash -n scripts/run_projector_fusion.sh
git diff --check
```

新测试包括非零 SQ 输出与 QA loss 的精确恒等起点、支持头 RNG 对齐、两臂等
容量、纯答案 CE 更新 γ、BF16 小型 Mistral 和两套 PEFT adapter 冻结、完整
checkpoint roundtrip/resume/旧版 warm start/损坏文件拒绝、padding/可变 K 和
query 长度/文档置换、RMS 汇总、真实 trainer 两步 smoke、配对分析和无依赖 CLI。
这些是接口核验，不是发布版 7B 的 QA 实验；本地没有启动服务器训练。

## 后续：γ 冻结对照与无需重训的 γ 干预（2026-10-06）

首轮结果（[QUERY_MODULATED_FUSION_RESULTS.md](QUERY_MODULATED_FUSION_RESULTS.md)）中，
AddG 和 FiLM 相对 HeadE 都约 +0.9 F1，但数据顺序不同，不能归因于 γ。本节只做两件事，
不加新结构。

**1. γ 冻结对照（G0）。** `--projector_gamma_frozen` 保留 γ 模块（同样的随机数消耗，
所以训练样本顺序与 AddG/FiLM 完全一致），但每步在梯度裁剪前把 γ 的梯度置为
None：裁剪忽略它，AdamW 跳过它，权重与优化器状态始终为零；若权重偏离零会直接
报错。其余设置与 AddG 相同（λ=0、1000 步、从 SQ last 开始），臂名 `SQ+HeadE+G0`。
γ≡0 时 additive 与 film 的前向完全相同，所以只需一个对照臂。

注意：AddG/FiLM 的梯度总范数包含 γ 的梯度，G0 不包含，所以三臂在裁剪系数上有
细微差别；这是"不训练 γ"本身的一部分，不另做处理。

- AddG−G0、FiLM−G0 都为正：收益来自新增的条件非线性路径，而非乘性本身。
- 都不为正：共有的约 0.9 点来自数据顺序等噪声，停止这一版 γ 扩展。

**2. γ 干预（eval-only，`--gamma_control`）。** 对已训练的 AddG/FiLM last checkpoint：

| 变体 | γ 的来源 | h、b、decoder 的问题 |
|---|---|---|
| normal | 正确问题 | 正确问题 |
| `gamma-zero` | γ=0 | 正确问题 |
| `gamma-swap` | 下一条样本的问题（与 mismatch-q 同一错位） | 正确问题 |
| `mismatch-q`（已有） | 错问题 | h 不变；b、γ 都来自错问题；decoder 正确 |

每个干预变体都同时用原问题跑一遍未干预的读出，记录配对变化（按有效文档汇总平方和，
不平均各 batch 的比值）：`change_gamma_rel = ‖γ'−γ‖/‖γ‖`、`change_gamma_cosine`
（逐文档 cos 的平均）、`change_e_over_delta_ref = ‖E'−E‖/‖Δ‖`、`change_e_over_memory
= ‖E'−E‖/‖Z‖`。参照 γ 为零时比值记为 null。

```bash
R=/data02/quro/runs/query_fusion_v1
python scripts/analyze_gamma_intervention.py \
  --control G0=$R/gamma_frozen --arm AddG=$R/additive --arm FiLM=$R/film \
  --reference HeadE=/data02/quro/runs/support_output_v1/sq_ce \
  --intervention AddG=$R/intervention/additive --intervention FiLM=$R/intervention/film \
  --dev /data02/quro/data/hotpot/dev.jsonl --output_json $R/gamma_control_dev.json
```

主判据是 AddG−G0 与 FiLM−G0 的 QA 配对差；干预结果用来定位 γ 路径在已训练模型里
的作用（normal−gamma-zero 是 γ 的功能贡献，normal−gamma-swap 是 γ 对问题的依赖），
不能替代 QA 收益。eval-only 的 normal 结果应与训练时的最终评估逐条一致，作为确定性检查。
