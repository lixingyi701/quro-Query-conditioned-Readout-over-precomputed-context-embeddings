# 开放域 RAG：执行记录与结果

协议：[`OPEN_RAG_SCALE_EXECUTION.md`](OPEN_RAG_SCALE_EXECUTION.md)（a4c5605）。工作根 `/data02/quro/data/open_rag_scale_20261009/`。

## 1. 阶段 A：已有 checkpoint 的共同证据评测

HotpotQA 项目 dev2000，经同一 SPLADE-v3 → DeBERTa-v3 排序重新检索；K5 取同一排序前五。所有方法共享逐题证据。

| 模型 | checkpoint |
|---|---|
| PISCO | 发布版，`pisco_direct`，K×8 memories |
| 混训 SQ / S0 | `public_qa_90k/seed42/{SQ,S0}/checkpoint_best.pt`（step 6500 / 5000） |
| 旧专训 SQ | `query_ablation/sq_s42/checkpoint_last.pt`（HotpotQA 原生 K10 训练 3000 步） |

### 绝对分（%）

| 模型 | K5 EM | K5 F1 | K5 substring | K10 EM | K10 F1 | K10 substring |
|---|---:|---:|---:|---:|---:|---:|
| PISCO | 0.65 | 10.12 | **42.70** | 1.00 | 10.54 | 42.70 |
| 混训 SQ | **33.85** | **44.93** | 35.75 | 33.50 | 44.70 | 35.35 |
| 混训 S0 | 33.35 | 43.92 | 35.30 | 32.80 | 43.92 | 34.70 |
| 旧专训 SQ | 31.20 | 42.30 | 34.60 | 30.95 | 41.55 | 33.65 |

PISCO 输出长句，EM/F1 不可比，只看 substring。

### 配对差（pp，95% 题目 bootstrap CI，单 seed）

| 对比 | K5 F1 | K5 substring | K10 F1 | bridge F1（K5） | comparison F1（K5） |
|---|---|---|---|---|---|
| 混训 SQ − S0 | **+1.02** [+0.01, +1.99] | +0.45 [−0.60, +1.50] | +0.77 [−0.32, +1.87] | +1.12 | +0.56 |
| 混训 SQ − 旧专训 SQ | **+2.63** [+1.24, +4.12] | +1.15 [−0.25, +2.65] | **+3.14** [+1.65, +4.70] | +3.34 | −0.41 |
| 混训 SQ − PISCO（substring） | — | **−6.95** [−8.80, −5.25] | −7.35（substring） | −6.10（substring） | −10.58（substring） |

| 同模型 K10 − K5 | EM | F1 | substring |
|---|---|---|---|
| PISCO | +0.35 [+0.10, +0.65] | +0.41 [+0.03, +0.80] | +0.00 [−1.35, +1.20] |
| 混训 SQ | −0.35 | −0.24 [−1.33, +0.92] | −0.40 |
| 混训 S0 | −0.55 | +0.01 [−1.14, +1.06] | −0.60 |
| 旧专训 SQ | −0.25 | −0.75 [−1.98, +0.55] | −0.95 |

### 解读

1. **在共同检索证据下，混训 SQ 高于旧专训 SQ**（F1 +2.63，主要在 bridge）。此前"混训比专训低 13 F1"是在原生 K10（含 gold 支持段落）上测的，两者证据协议不同。
2. **query 分支在 K5 下有小幅正效应**：SQ−S0 +1.02，CI 下界刚过零；K10 下 +0.77，CI 跨零。
3. **投影器模型的 substring 仍比 PISCO 低约 7 分**，comparison 题差约 10.6 分。对应 D2 表"SQ 仍低于 PISCO 的 match"一行。
4. **检索 K10 无收益**：PISCO 的 substring 不变，未达到 +1pp 且 CI 下界 > 0 的门槛；投影器 K10−K5 均约为零。按 D2 不开 K10 训练分支。

## 2. 阶段 B：全量问题与 K5 缓存

- 导出：全量 447,689 条训练题（规范化去重 427,543）/ 4,541 dev；新 90k 和 dev 与旧 `public_qa_90k_questions` 逐字节相同。
- 检索：复用旧 94,541 题检索；补检索 350,885 题，拆成四份四卡并行（单卡搜索每 shard 约 7 分钟，四卡总计约 3.5 小时）。
- 缓存：`/data02/quro/cache/open_rag_full_r16`，1,409,070 段，92 GB；round-trip 与 check_rag_data 全部通过。
- 长度 audit：所有来源 over_cap=0。

### 工程问题

- **问题 join 歧义**：公共池里有大量跨来源的同文问题，旧检索中同一规范化问题对应多行，`missing` 报 ambiguous。修复：候选的有序证据完全一致时安全复用，否则仍拒绝；`missing` 把歧义题列入重检索并按 ID 对接（歧义题从 275 降到 9）。
- **audit tokenizer**：`pisco-mistral` 目录含自定义代码，`AutoTokenizer` 拒绝加载；改用同底座 `/home/lxy/selecom/baselineModel/Mistral-7B-Instruct-v0.2`。
- **磁盘**：删除 6 个旧缓存中已打包的冗余 shard（232 GB）。删除前逐个验证 memmap 布局、字节数、索引排列，并逐字节比对 939 个抽样文档。

## 3. 阶段 C：同预算四 run（运行中）

2026-10-10 08:05 启动，`/data02/quro/runs/open_rag_scale_s42/{full,90k}_{SQ,S0}_s42`，tmux `orag_C`。

- T = 27,981 步，有效 batch 16；名义遍数 full 1.00、90k 4.97；快照步 9000 / 16875 / 27981。
- 来源配比两池一致（hotpotqa 19.6/19.5%，nq_open 19.4/19.4%，squad 19.4/19.5%，triviaqa 13.6/13.7% …）。
- 核验：同池 SQ/S0 的第 1 步 `train_order_digest`、loss（full 4.3352、90k 3.9849）和 RMS(h) 相同；四 run 的第 0 步 dev 相同（F1 12.48，substring 50.3）。
- 第 0 步 dev 4541 条约 520 秒。每 1000 步验证一次，共约 28 次，验证总耗时约 4 小时。

## 文件

| 内容 | 路径 |
|---|---|
| 阶段 A 预测与 result | `/data02/quro/runs/open_rag_protocol_k{5,10}/{PISCO,SQ,S0,old_SQ}/` |
| 阶段 A 配对分析 | `/data02/quro/runs/open_rag_analysis/`，副本在 `results/open_rag_stageA/` |
| 阶段 C 执行计划 | `results/open_rag_stageA/stageC_execution_plan.json` |
