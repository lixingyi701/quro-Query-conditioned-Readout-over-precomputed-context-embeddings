# SQX 跨文档交互：复核与分阶段验证

日期：2026-10-08。基础：投影器分支 `69001c3`。

状态：已有跨文档模块；本轮补充训练顺序控制、分析脚本与运行协议。尚未运行真实权重的新增训练；旧预测文件位于服务器，本文没有重算出新的 CI。

## 1. 为什么值得验证，但不能先认定成功

10 月 3 日单 seed 完整 test 的结果：

| 题型 | SQ F1 | SQX F1 | SQX−SQ |
|---|---:|---:|---:|
| bridge，4296 题 | 64.31 | 65.05 | +0.74 |
| comparison，1109 题 | 72.47 | 70.74 | −1.73 |
| 全部，5405 题 | 65.99 | 66.21 | +0.22 |

来源：[首轮报告](SHARED_QUERY_PROJECTOR_RESULTS.md)。以上是单次训练的点估计；报告中的 `[+0.33,+1.80]` 是 SQX−S0m 的 bridge 区间，不是 SQX−SQ 的区间。

+0.74 F1pp 是温和的可用信号。bridge 占 79.48%，若 comparison 不变，它可贡献约 +0.59 整体 F1pp；当轮 comparison 的下降抵消了约 0.35 点，只剩整体 +0.22。不能把平均 F1 的变化换算为“多答对多少道题”。

SQX 比 SQ 多 1,050,624 参数（约 +2.78%），同样输出 K×8 个 memory，不增加 decoder 的证据 token，但投影器多了一次 attention，实际时间/显存仍需测量。没有一个 F1 点数能单独决定是否可发表；稳定性、同预算对照和机制证据才使它有研究价值。

## 2. 旧对照存在的训练顺序风险

旧 `build_loaders()` 用全局 PyTorch RNG。SQX 初始化额外 `MultiheadAttention` 会消耗随机数，即使训练 seed 一样，也可能改变后续 shuffle 顺序。此前 FiLM/G0 已发现约 0.9 F1 的差异与样本顺序有关；因此旧 SQX 的 +0.74 不能纯归因于跨文档交互。

新增 `--data_order_seed` 使用独立的 DataLoader generator，并记录实际每个 microbatch 的 ID 序列 SHA-256、训练文件 SHA-256、缓存 manifest SHA-256、训练预算与完成步数。记录保存在日志、checkpoint、最终 `result.json`，eval-only 重载保留原训练记录。缓存 manifest hash 是索引/元数据标识，不是 latent 文件全部字节的 hash。

默认 `data_order_seed=None` 保留历史行为。新验证显式传与模型 seed 相同的数据 seed。本协议只接受 fresh start；恢复训练尚未恢复 sampler/worker 状态，因此拒绝该选项下的训练 resume。checkpoint 评估仍允许。

**历史 S0/SQ 如果没有这项记录，不能自动充当严格匹配的新基线。** 它们继续作为历史参考；不得给旧 checkpoint 补写一个新 hash 来声称原训练顺序相同。

## 3. 验证的问题与结构

| 臂 | Query | 跨文档 attention | 参数约 M |
|---|---|---|---:|
| S0 | 无条件分支 | 无 | 33.59 |
| SQ | 真实问题 | 无 | 37.78 |
| S0X | 无条件分支 | 有 | 34.64 |
| SQX | 真实问题 | 有 | 38.83 |

跨文档 attention 已在 `SharedDocumentProjector` 中：每篇八个 memory 和 query 条件形成一个 512 维 U_i；对 K 个 U_i 做一层八头 self-attention，以残差形式加入后送到共享 Wo，输出每篇八个 memory 残差。它不是对全部 80 个 memory token 直接做 attention，不是显式逐跳读取。支持可变 K、文档 mask，无文档排名位置编码；投影器置换等变不代表最终 causal decoder 的答案顺序不变。

所有新臂从发布版 PISCO 分别初始化；冻结原 compressor、decoder/LoRA；K=10、m=8、D0、答案 CE、无 SupportDoc、无 FiLM、无 KD。统一 3000 步，lr=5e-5，线性调度，5% warmup，batch=2，累积=8；保留历史每 250 步 dev500 验证与 EM 选择规则，但主分析统一 **last** checkpoint。dev500 的峰值不充当结果。

## 4. 分阶段执行，避免一次投入八个训练

### A. 先重算旧预测：无需训练

```bash
python scripts/analyze_cross_document_validation.py \
  --run 42:SQ=/data02/quro/runs/shared_projector_v1/fulltest_SQ \
  --run 42:SQX=/data02/quro/runs/shared_projector_v1/fulltest_SQX \
  --data_file /data02/quro/data/hotpot/test.jsonl --split test --legacy \
  --output /data02/quro/runs/crossdoc_validation_v1/legacy_sqx_minus_sq.json
```

按服务器实际路径修改。重算 SQX−SQ 的 bridge/comparison/整体 F1、EM、substring 与配对 CI。`--legacy` 明确标记历史探索性比较；即使 CI 下界为正，也没有排除训练顺序/seed 的混杂。

### B. 主验证：SQ/SQX × seed42/43，四个训练

可通过环境变量覆盖所有路径：

```bash
export CROSSDOC_RUN_ROOT=/data02/quro/runs/crossdoc_validation_v1
export CROSSDOC_DATA_DIR=/data02/quro/data/hotpot
export CROSSDOC_CACHE_DIR=/data02/quro/cache/hotpot-pisco-r16
export CROSSDOC_PISCO_PATH=/data02/quro/models/pisco-mistral

# 首先查看命令；一个进程对应一个臂，不自动并发抢卡。
bash scripts/run_cross_document_validation.sh train SQX 42 --dry-run

# 在各自 tmux/GPU 中分别执行，seed43 同样执行两臂。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cross_document_validation.sh train SQ 42
CUDA_VISIBLE_DEVICES=1 bash scripts/run_cross_document_validation.sh train SQX 42
```

训练结束在各自 run 目录保存 dev2000 预测。分析：

```bash
python scripts/analyze_cross_document_validation.py \
  --run 42:SQ=$CROSSDOC_RUN_ROOT/seed42/SQ \
  --run 42:SQX=$CROSSDOC_RUN_ROOT/seed42/SQX \
  --run 43:SQ=$CROSSDOC_RUN_ROOT/seed43/SQ \
  --run 43:SQX=$CROSSDOC_RUN_ROOT/seed43/SQX \
  --data_file $CROSSDOC_DATA_DIR/dev.jsonl --split dev \
  --output $CROSSDOC_RUN_ROOT/sqx_minus_sq_dev.json
```

同题配对、严格检查 ID/gold/query/架构和训练记录；拒绝静默取交集。分别报告每个 seed；跨 seed 均值先对每题平均再 bootstrap，不把两个 seed 当两倍独立样本。CI 仅描述题目采样不确定性，两个训练 seed 仍不足以准确估计训练随机性。

### C. 有信号再补 S0/S0X × seed42/43

相同 launcher 的 arm 改为 `S0`、`S0X`。分析时加入四个 `--run`，自动增加：

- S0X−S0：不依赖 query 的跨文档收益。
- SQ−S0、SQX−S0X：无/有交互时的问题条件收益。
- (SQX−SQ)−(S0X−S0)：query 与跨文档结构的交互差值，按同题四臂差计算 CI。

标准 S0 去掉条件分支，所以参数不同；这是标准模块消融，不能声称等有效容量。正交互是本配置中的联合效应证据，不单独证明模型完成了两跳推理。

### D. 固定 checkpoint 的完整评估与可选功能干预

```bash
# 完整 test；四个候选 run 分别执行，未通过 dev 候选判据前不自动跑 test。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cross_document_validation.sh eval SQX 42

# 可选：仅为机制补充，在独立目录评估 projector 错问题，decoder 仍看正确问题。
CROSSDOC_SPLIT=dev CROSSDOC_QUERY_CONTROL=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_cross_document_validation.sh eval SQX 42
```

默认 checkpoint 是本臂 `checkpoint_last.pt`，可用 `CROSSDOC_CHECKPOINT` 指向真正匹配的历史 checkpoint。新输出目录必须不存在，不覆盖旧结果。完整评估不截取 2000 题；分析改 `--split test`，路径指向各臂 `eval_test`。错问题分析应对所有 supplied runs 统一生成 controls，分析路径指向 `eval_dev_query_control`；无 query 臂应保持输出不变。

## 5. 在新结果出现前固定预算判据

阶段 B 的主指标：正常 QA 的 bridge F1，比较 SQX−SQ。EM、substring、comparison F1、整体 F1 为完整伴随报告；不得只展示 bridge 的涨幅。

作为本轮是否追加训练的实用门槛：两个 seed 的 bridge 差值均为正，平均至少 +0.5 F1pp，且平均整体 F1 不下降，才进入阶段 C 与固定 last 的完整评估。+0.5 是围绕旧 +0.74 设置的资源投入门槛，不是统计显著性或投稿门槛。同时公布全部差值与 CI，不因为过线就声称成功。

若 bridge 正、comparison 负并导致整体退化：先记录题型取舍，不直接改成全局默认，也不基于 test 再挑 gate/层数。若方向不一致或平均接近零：结束本轮普通文档级 attention 验证，保留结果；不继续遍历 attention 深度与超参。

更强的“稳定收益”表述需要：受控两 seed 方向一致、完整结果保留 bridge 收益、配对区间支持正向差异，并在另一数据集复现。当前 test 已被历史实验反复查看，是复核集合；真正新的外部证据来自未用于设计的多跳数据集，例如 MuSiQue，不能再称这 5405 题为未见 holdout。

## 6. 给后续方向 1/2 提供什么依据

| 本轮结果 | 支持的下一步 | 不能推出 |
|---|---|---|
| SQX−SQ 正；S0X−S0 也相近 | 一般跨文档融合有帮助，保留这一基线 | query 驱动交互已经成立 |
| SQX−SQ 正且交互差值正 | 优先验证第一跳结果更新第二跳条件 | attention 自发学会了正确桥接实体 |
| bridge 正、comparison 明显负 | 有条件的文档对交互/问题路由值得考虑 | 全局加入 attention 更好 |
| 普通 SQX 没有稳定收益 | 下一步需要更具体的逐跳/事实诊断 | 所有跨文档方法无效，或 Z 不含答案 |

方向 1：中间桥接实体/状态进入下一跳 query，参照 [DecompRC](https://aclanthology.org/P19-1613/) 与 [MDR](https://arxiv.org/abs/2009.12756)。先做正确第二跳子问题的 oracle 测试，再决定是否训练；与旧 RQ 单次读写区分。

方向 2：实体/文档对关系约束的交互，参照 [HGN](https://aclanthology.org/2020.emnlp-main.710/)。普通 SQX 是它必须超过的直接基线；若新增实体/标题元数据，应单独说明缓存与标注成本。

**本轮先回答“普通跨文档融合是否改善现有表示”，不一次混入逐跳更新、关系标签和新缓存。**

## 7. 实现文件

- `scripts/run_cross_document_validation.sh`：单臂启动、dry-run、fresh 输出、显式跨文档开关。
- `scripts/analyze_cross_document_validation.py`：旧两臂与新四臂、分题型/seed、配对 CI、交互差值及训练记录校验。
- `--data_order_seed`：独立训练 sampler，与实际 microbatch 顺序记录、checkpoint 保存/重载。
- `tests/test_cross_document_validation.py`：分析对齐/交互、顺序混杂拒绝、四臂小模型训练及重载。

本地工程检查不等于真实 PISCO 实验结果；服务器跑完后应新增结果文档，包含训练记录、完整题型表与实际效率。

2026-10-08 本地验证：全量 127 项 CPU 测试通过；新增 8 项检查通过，覆盖四臂实际训练顺序一致、checkpoint→eval-only 保留训练记录，以及真实小模型输出直接进入严格配对分析。八种 train/eval 启动命令的 dry-run、Python 编译、bash 语法与 `git diff --check` 通过。环境为 Python 3.12、torch 2.5.1 CPU、transformers 4.44.2、peft 0.12.0；未加载真实发布版权重或服务器预测。
