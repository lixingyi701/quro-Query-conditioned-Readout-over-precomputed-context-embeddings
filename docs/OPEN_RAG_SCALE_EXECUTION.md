# 开放域 RAG：证据对齐、数据规模与训练预算执行流程

日期：2026-10-09。分支：`feat/pisco-joint-query-projector`。
接续结果：[`PUBLIC_QA_TRAINING_RESULTS.md`](PUBLIC_QA_TRAINING_RESULTS.md)，截至 `f84b64c`。

## 1. 本轮决定与执行顺序

**先完成已有 checkpoint 在共同检索证据上的评测；随后固定检索 K5，比较相同训练预算下增加问题覆盖与重复旧问题。默认不改成 K10 训练，也不直接把全量数据训练三遍。**

1. 对同一批 HotpotQA 开发题构造检索 K10，K5 取同一排序前五篇。评测发布版 PISCO、混训 SQ/S0 和旧专训 SQ。K5 是主协议；检索 K10 是诊断。原生 K10 是另一协议，历史结果单列。
2. 使用完整合格公共问题池，准备固定检索 K5 和缓存；生成其嵌套 90k 子集，两者共享开发集。
3. 四卡分别跑 `full-SQ`、`full-S0`、`90k-SQ`、`90k-S0`。**四个 run 的优化步数、有效 batch、初始化、LR 日程和开发集相同。**预算约为全量一次遍历。
4. 用开发结果决定后续方向；固定 checkpoint 后，评测多个任务。若数据扩容有效，优先补第二个 seed，再考虑全量第二/第三遍。只有检索 K10 显示明确优势时，才开独立 K10 训练分支。

阶段 1 是低成本确认协议，不是要求混训 SQ 先赢过旧专训才允许阶段 2。数据/缓存校验通过后，阶段 2 可继续准备；结果可能支持或否定扩数据收益。

本轮使用 SQ 作为主模型，S0 保留为必要对照。已有跨协议 HotpotQA 结果中，SQ−S0 为 **+2.15 F1pp**，95% 题目 bootstrap CI `[+1.37,+2.91]`；query 分支的正贡献已有证据。SQX−SQ 为 −1.68 F1pp，当前不再扩大跨文档模块的投入。

## 2. 为什么这样调整 K、题目数和 epoch

### 2.1 固定 K5，而非根据原生 K10 结果直接改 K

现有 90k 混训使用 Wikipedia-KILT 检索 K5，13–15 F1 的退化是在 **HotpotQA 原生 distractor K10** 上观察的。它同时改变了证据来源、篇数、支持事实覆盖、排序和输入预算，不能据此认定检索 K5 不够。

本轮区分以下输入：

| 输入 | 来源及排序 | m8 时的 soft-token 数 | 用途 |
|---|---|---:|---|
| 检索 K5 | SPLADE Top 50 → 同一 DeBERTa 排序前五 | 40 | 主训练与主评测 |
| 检索 K10 | 上述同一排序前十 | 80 | 开发集上的 K 诊断 |
| 原生 K10 | HotpotQA 自带支持段落与干扰段落 | 80 | 历史阅读/迁移实验，独立报告 |

**检索 K10 不等于原生 K10。**不插入 gold 支持段落，不使用支持事实重排检索结果。共享投影器支持可变 K，因此已有 K5 checkpoint 可以直接做检索 K10 诊断，但它仍是训练外文档预算。

### 2.2 训练曲线降低了盲目加 epoch 的优先级，没有证明题目多样性是唯一瓶颈

90k × 有效 batch16 × 9000 步 ≈ 144k 次样本呈现，即约 1.6 遍。已有曲线显示后段 loss 平均只再降约 0.09，混合 dev F1 在 5000–6500 步附近达到峰值。另一方面，LR 在第 9000 步已经衰减到零，且没有逐来源开发曲线。

因此这些记录只支持“当前 90k、当前日程没有持续的后期验证增益”，不能证明 HotpotQA 单独已收敛、三 epoch 一定无益，或全量问题一定有效。

### 2.3 用同算力比较新问题和重复问题

令全量训练条目数为 N、batch 为 b=2、累积为 a=8。脚本按实际 `drop_last=True` loader 计算：

```text
T = ceil(floor(N / b) / a)
有效 batch = b × a = 16
```

若 N 仍为 447,689，T=27,981，名义呈现 447,696 个样本：

| run | 问题池 | 更新数 | 名义遍数 | 判断用途 |
|---|---:|---:|---:|---|
| full-SQ | N≈447k | T≈27,981 | ≈1.00 | 新问题覆盖、主模型 |
| full-S0 | 同一全量 | 同一个 T | ≈1.00 | 全量上的 query 对照 |
| 90k-SQ | 全量的嵌套 90k | 同一个 T | ≈4.97 | 相同预算重复旧题 |
| 90k-S0 | 同一 90k | 同一个 T | ≈4.97 | 小池上的 query 对照 |

这不是因为我们预计重复五遍会大涨，而是为了避免“全量跑更多步赢了，所以一定是新题有效”的归因错误。它比较的是**同一更新预算下的问题池选择**；自然抽样的来源比例仍有小幅差异，manifest 会逐来源记录。

原始公共池有跨来源重复问题；约 447k 指训练条目数，不自动等于互不重复的问题数。计划同时记录规范化后去重的问题数。`drop_last` 和最后一个梯度累积可能略跨越下一轮，因此一次遍历是近似值；实际呈现量以 checkpoint 的 `data_order_provenance.examples` 为准。

不把旧 9000 步 run 作为严格的同算力对照：它的 LR 总日程、选点指标和开发切片不同。新增四 run 从发布版 PISCO 和恒等初始化投影器开始，不从旧 HotpotQA/混训投影器 warm-start。

### 2.4 前人训练多久：已核实原文

| 工作 | QA 训练 epoch | 有效 batch | 文档数 | 可对齐与不同之处 |
|---|---:|---:|---:|---|
| COCOM，附录表 11 | 2 | 64 | 5 | 混合 QA 微调；训练 LoRA |
| PISCO，附录表 5、§4.1 | 1 | 128 | 5 | 约 453k 问题，检索文档生成教师 silver label；训练适配器及 memory-token embeddings |
| 本轮 | 全量≈1遍，与90k≈5遍同预算对照 | 16 | 5 | 冻结发布版 compressor/decoder/LoRA，只训投影器，使用 gold CE |

来源：[COCOM 原文](https://arxiv.org/html/2407.09252v3)、[PISCO 原文](https://aclanthology.org/2025.findings-acl.800.pdf)。COCOM 论文表 12 统计 493,473 条，而所用公开 `multi_qa` 版本为 453,023 条；不声称两者逐条完全相同。epoch 和更新数也不能脱离 batch、可训练模块、监督目标直接比较。

## 3. 固定协议与服务器准备

- 问题池固定 `dmrau/multi_qa` revision `b0b01e2a0f6e251e9cdd191f5018bec7b170a554`，保留作者来源比例，不额外过采样 HotpotQA。
- 固定现有 kilt-128、SPLADE-v3 和 DeBERTa-v3 本地 snapshot，检索 Top 50，再取有序 Top 5/10；同一问题所有方法共享证据。
- 固定发布版 PISCO r16、每文档 m8、h4096、fp16、压缩器输入上限128；不能改缓存截断规则来隐式增加证据。
- SQ/S0：LR 5e-5，linear，warmup 5%，AdamW weight decay0.01，grad clip1；batch2、累积8；不加 KD、支持监督、FiLM 或跨文档 attention。
- 新 run 每1000步验证固定混合 dev 前1000条，按 **F1** 选 best；保存逐来源 n/EM/F1/substring。稀少来源的小 n 指标只作观察。
- 保存第9000步、第 `ceil(3*floor(n90k/2)/8)` 步和最后一步；90k=90000时中间点为16875。快照也触发相同开发验证。四 run 的 LR 总日程始终同为 T。
- 对照规模效果优先比较 **同一步快照或 last**；各自 best 的比较回答最终训练配方效果，含选点差异，两者分开报告。
- 所有任务采用同一个混合 checkpoint，不为每个测试任务挑一个 checkpoint。

以下在服务器仓库中执行，长作业置于 `tmux`。路径来自已上传结果记录；NQ/PopQA/WebQ/ASQA 的具体文件名需要服务器执行者通过现有目录核对，下面用环境变量传入。

```bash
cd /home/lxy/quro
git pull --ff-only
export RAG_ROOT=/data02/quro
export RAG_STAGE=$RAG_ROOT/data/open_rag_scale_20261009
export PISCO_MODEL=$RAG_ROOT/models/pisco-mistral
export OLD_QUESTIONS=$RAG_ROOT/data/public_qa_90k_questions
export OLD_RETRIEVAL=$RAG_ROOT/data/public_qa_90k_retrieval
export SHARED_INDEX=$OLD_RETRIEVAL/enc
export HOTPOT_DEV=$RAG_ROOT/data/hotpot/dev.jsonl
export HOTPOT_TEST=$RAG_ROOT/data/hotpot_full/test.jsonl
mkdir -p "$RAG_STAGE"
rg --files "$RAG_ROOT/data/eval_sets" "$RAG_ROOT/data/trivia"
# 核对并设置 NQ_EVAL、TRIVIA_EVAL、POPQA_EVAL、WEBQ_EVAL 的真实 JSONL 路径。
# ASQA_EVAL 若尚未取得，则先不评测 ASQA；不是用训练题替代评测题。
```

保持服务器已能工作的 transformers4.57.6/tokenizers0.22.2/sentence-transformers5.1.2 环境，不为本轮盲目升级依赖。数据准备/计划/分析脚本不加载模型；检索需要 torch/numpy/scipy/pyarrow 和现有检索模型。

## 4. 阶段 A：已有 checkpoint 的共同证据评测

先只用 **HotpotQA 项目 dev2000** 做协议诊断。核心多任务主评测为 HotpotQA、NQ、TriviaQA、PopQA；WebQ 可作为未参与此次投影器混训的迁移集。ASQA-short 取得评测问题、补齐排除后再加入。

### A1. 转出问题并重新检索

```bash
python scripts/prepare_open_rag.py eval-queries \
  --inputs hotpot_dev="$HOTPOT_DEV" \
  --out_dir "$RAG_STAGE/hotpot_dev_questions"

WORK_DIR="$RAG_STAGE/hotpot_dev_retrieval" INDEX_DIR="$SHARED_INDEX" KEEP=10 GPUS=0,1,2,3 \
  bash scripts/run_open_rag_retrieval.sh \
  --queries_jsonl "$RAG_STAGE/hotpot_dev_questions/eval.queries.jsonl" \
  --allow_legacy_index

python scripts/prepare_open_rag.py attach-eval \
  --queries_dir "$RAG_STAGE/hotpot_dev_questions" \
  --retrieval_jsonl "$RAG_STAGE/hotpot_dev_retrieval/retrieval.jsonl" \
  --ks 5,10 --out_dir "$RAG_STAGE/hotpot_dev_ready"

CORPUS="$RAG_STAGE/hotpot_dev_ready/corpus.jsonl" \
  CACHE="$RAG_ROOT/cache/open_rag_hotpot_dev_r16" CHECKPOINT="$PISCO_MODEL" \
  GPUS=0,1,2,3 bash scripts/build_hotpot_cache.sh
```

`--allow_legacy_index` 是显式复用此前已运行的43分片索引，不是重新索引 Wikipedia。第一次使用前核对模型 snapshot、语料和编码上限256与旧记录一致；脚本登记元数据，并检查编码行数与 parquet 行数一致。之后变更问题文件、keep/world/模型等参数会拒绝复用不匹配的输出。

正式扩全量前，服务器应先对32条问题跑一次新检索搜索/重排 smoke，确认当前 CUDA 的 sparse CSR fp16 支持、输出文档 ID、K10 完整性和实际显存。CPU 已验证搜索数学正确性，不能替代此项服务器检查。可将上面 eval.queries.jsonl 的前32行写到新的 smoke 问题文件，使用新的 WORK_DIR；不要替换正式问题文件。

### A2. 先查看执行计划，再运行 K5 与检索 K10

```bash
export MIXED_SQ=$RAG_ROOT/runs/public_qa_90k/seed42/SQ/checkpoint_best.pt
export MIXED_S0=$RAG_ROOT/runs/public_qa_90k/seed42/S0/checkpoint_best.pt
# OLD_SQ：旧 HotpotQA 专训 SQ 的真实 checkpoint，须核对 config 中 query_mode=conditioned、cross_document=false。
: "${OLD_SQ:?set the existing HotpotQA-only SQ checkpoint path}"

python scripts/run_open_rag.py eval --k 5 \
  --eval_files hotpot_dev="$RAG_STAGE/hotpot_dev_ready/k5/hotpot_dev.jsonl" \
  --cache_dir "$RAG_ROOT/cache/open_rag_hotpot_dev_r16" --generator_path "$PISCO_MODEL" \
  --checkpoints SQ="$MIXED_SQ" S0="$MIXED_S0" old_SQ="$OLD_SQ" \
  --out_dir "$RAG_ROOT/runs/open_rag_protocol_k5" --gpus 0,1,2,3
# 上面不加载模型；检查 execution_plan.json。原样追加 --execute 开始评测。

python scripts/run_open_rag.py eval --k 10 \
  --eval_files hotpot_dev="$RAG_STAGE/hotpot_dev_ready/k10/hotpot_dev.jsonl" \
  --cache_dir "$RAG_ROOT/cache/open_rag_hotpot_dev_r16" --generator_path "$PISCO_MODEL" \
  --checkpoints SQ="$MIXED_SQ" S0="$MIXED_S0" old_SQ="$OLD_SQ" \
  --out_dir "$RAG_ROOT/runs/open_rag_protocol_k10" --gpus 0,1,2,3 --execute
```

发布版 PISCO 自动加入，使用 `pisco_direct`、全部 K×8 memories，即 K5 的40或K10的80。不复用曾误设 B8 的 `pisco_hotpot` 基线。eval-only 中 train_file 只是现有 loader API 所需的占位数据，不执行训练。

```bash
# 同一证据条件：混训 SQ − PISCO。
python scripts/analyze_open_rag.py \
  --left "$RAG_ROOT/runs/open_rag_protocol_k5/SQ/predictions_hotpot_dev_D0_Bfull.json" \
  --right "$RAG_ROOT/runs/open_rag_protocol_k5/PISCO/predictions_hotpot_dev_D0_B40.json" \
  --out "$RAG_ROOT/runs/open_rag_analysis/k5_sq_minus_pisco.json"

# 同一 checkpoint 改变证据预算：检索 K10 − 检索 K5。
python scripts/analyze_open_rag.py \
  --left "$RAG_ROOT/runs/open_rag_protocol_k10/PISCO/predictions_hotpot_dev_D0_B80.json" \
  --right "$RAG_ROOT/runs/open_rag_protocol_k5/PISCO/predictions_hotpot_dev_D0_B40.json" \
  --allow_evidence_change --out "$RAG_ROOT/runs/open_rag_analysis/pisco_k10_minus_k5.json"
```

同样生成 SQ−S0、混训SQ−旧专训SQ，以及 SQ 的 K10−K5 对比。脚本严格匹配全部 ID、问题和 gold 别名，拒绝只取交集；默认检查证据 ID/顺序相同。EM、F1、substring一起报告，包含 bridge/comparison 分组 n 和配对 CI。substring 对长输出更宽容，但也受输出长度与误命中影响；不能称其完全不受答案风格影响。

## 5. 阶段 B：全量问题与固定 K5 缓存

### B1. 使用相同排除清单导出全量和嵌套90k

```bash
: "${NQ_EVAL:?set NQ evaluation JSONL}"
: "${TRIVIA_EVAL:?set TriviaQA evaluation JSONL}"
: "${POPQA_EVAL:?set PopQA evaluation JSONL}"
: "${WEBQ_EVAL:?set WebQuestions evaluation JSONL}"
EXCLUDES=(--exclude_jsonl "$HOTPOT_DEV" --exclude_jsonl "$HOTPOT_TEST"
          --exclude_jsonl "$NQ_EVAL" --exclude_jsonl "$TRIVIA_EVAL"
          --exclude_jsonl "$POPQA_EVAL" --exclude_jsonl "$WEBQ_EVAL")
if [[ -n "${ASQA_EVAL:-}" ]]; then EXCLUDES+=(--exclude_jsonl "$ASQA_EVAL"); fi

python scripts/prepare_public_qa.py export --local_only --hf_cache_dir "$RAG_ROOT/hf-cache" \
  --seed 42 --dev_fraction 0.01 --out_dir "$RAG_STAGE/full_questions" "${EXCLUDES[@]}"
python scripts/prepare_public_qa.py export --local_only --hf_cache_dir "$RAG_ROOT/hf-cache" \
  --seed 42 --dev_fraction 0.01 --limit 90000 \
  --out_dir "$RAG_STAGE/90k_questions" "${EXCLUDES[@]}"
```

增加 ASQA 排除等条件后，N 和90k成员可能改变，以新 manifest 为准，不硬写447,689。按规范化问题分组划分避免跨来源同题泄漏；两个导出共享开发集。若本地 HF 缓存路径不同，改用固定版本的 `--input_parquet`；不改变数据 revision。

### B2. 只补检索新增问题，复用旧94,541条检索

```bash
python scripts/prepare_open_rag.py missing \
  --queries_dir "$RAG_STAGE/full_questions" \
  --retrieval_jsonl "$OLD_RETRIEVAL/retrieval.jsonl" --max_docs 5 \
  --out_dir "$RAG_STAGE/full_pending"

WORK_DIR="$RAG_STAGE/full_extra_retrieval" INDEX_DIR="$SHARED_INDEX" KEEP=5 GPUS=0,1,2,3 \
  bash scripts/run_open_rag_retrieval.sh --queries_dir "$RAG_STAGE/full_pending" --allow_legacy_index

python scripts/prepare_public_qa.py attach --queries_dir "$RAG_STAGE/full_questions" \
  --retrieval_jsonl "$OLD_RETRIEVAL/retrieval.jsonl" \
  --retrieval_jsonl "$RAG_STAGE/full_extra_retrieval/retrieval.jsonl" \
  --max_docs 5 --out_dir "$RAG_STAGE/full_ready"

python scripts/prepare_open_rag.py subset --full_ready "$RAG_STAGE/full_ready" \
  --small_queries "$RAG_STAGE/90k_questions" --out_dir "$RAG_STAGE/90k_ready"

python scripts/prepare_public_qa.py audit --queries_dir "$RAG_STAGE/full_questions" \
  --tokenizer_path "$PISCO_MODEL" --max_query_len 256 --max_answer_len 128
```

若 pending 总数为0，跳过其检索并在 attach 时只传旧检索文件。若 PISCO 发布目录不带完整 tokenizer，将 audit 的 tokenizer_path 指向同一 reader 使用的本地 Mistral-v0.2 tokenizer。attach 不得接受部分覆盖，不填随机文档，不替换 gold。

新 search 将所有问题向量保留为 **CPU CSR**，GPU只加载 `query_block=128` 的稠密向量和对应分数矩阵。以前 `torch.cat` 所有 query 向量的实现，扩全量时仅这些 fp16 向量就约25.7GiB，拼接峰值更高；新路径消除了这一随问题总数增长的 GPU 占用。稀疏文档索引、分数矩阵与模型仍占显存，GPU smoke 应记录实际峰值。

### B3. 新建共享缓存，检查全部覆盖

```bash
CORPUS="$RAG_STAGE/full_ready/corpus.jsonl" CACHE="$RAG_ROOT/cache/open_rag_full_r16" \
  CHECKPOINT="$PISCO_MODEL" GPUS=0,1,2,3 bash scripts/build_hotpot_cache.sh
python scripts/check_rag_data.py "$RAG_STAGE/full_ready/train.jsonl" "$RAG_STAGE/full_ready/dev.jsonl" \
  "$RAG_STAGE/90k_ready/train.jsonl" --cache_dir "$RAG_ROOT/cache/open_rag_full_r16" --max_docs 5
```

默认新建完整共享缓存，四 run 全部读取这份缓存，保留旧90k缓存。这里复用了文档**检索索引和映射**，没有实现旧 packed latent cache 的增量合并；旧段落会在新缓存中重压缩一次。采用已有分片打包与字节往返校验，压缩 worker 失败立即阻止合并。

每文档 m8×4096×fp16 =65,536字节；缓存原始 latent 大小为 `unique_docs×65,536`，打包时还需临时空间及保留分片。按 attach manifest 的实际 unique_docs 估算磁盘/RAM，不把“约一百万段、1.5小时”当作已验证保证。

## 6. 阶段 C：四个同预算 run

```bash
python scripts/run_open_rag.py train \
  --full_ready "$RAG_STAGE/full_ready" --small_ready "$RAG_STAGE/90k_ready" \
  --cache_dir "$RAG_ROOT/cache/open_rag_full_r16" --generator_path "$PISCO_MODEL" \
  --seed 42 --data_order_seed 42 --gpus 0,1,2,3 \
  --out_dir "$RAG_ROOT/runs/open_rag_scale_s42"
# 先检查 execution_plan.json 中 N、unique_questions、来源配比、T、nominal_passes、输入 SHA256。
# 原样追加 --execute 启动。运行前需完整完成阶段 B；不要只拿到问题文件便启动。
```

启动器检查 m/h/rate/截断、compressor/reader 路径、全部缓存覆盖、90k嵌套关系、相同开发文件和 train/dev 问题隔离；最多每个所列 GPU 一个进程，检查退出码，分开保存日志，不覆盖已有 run。

训练结束后核对：同一问题池的 SQ/S0 `order_sha256`、实际 examples、completed_steps 一致；四 run 的预算及 LR 日程一致。full 与90k之间训练题顺序当然不同，不要求 digest 相等。

```bash
# 训练结尾自动生成同一完整混合dev的last预测；直接做同预算数据规模比较。
python scripts/analyze_open_rag.py \
  --left "$RAG_ROOT/runs/open_rag_scale_s42/full_SQ_s42/predictions_dev_D0_Bfull.json" \
  --right "$RAG_ROOT/runs/open_rag_scale_s42/90k_SQ_s42/predictions_dev_D0_Bfull.json" \
  --out "$RAG_ROOT/runs/open_rag_analysis/full_minus_90k_last_mixed_dev.json"
python scripts/analyze_open_rag.py \
  --left "$RAG_ROOT/runs/open_rag_scale_s42/full_SQ_s42/predictions_dev_D0_Bfull.json" \
  --right "$RAG_ROOT/runs/open_rag_scale_s42/full_S0_s42/predictions_dev_D0_Bfull.json" \
  --out "$RAG_ROOT/runs/open_rag_analysis/full_sq_minus_s0_last_mixed_dev.json"
```

新的快照是可评测权重，**不是支持恢复 sampler 的续训快照**。现有 trainer 会拒绝 `data_order_seed + 非eval resume`。不得给旧9000步 last加大steps就声称完成三epoch；旧LR已到零，sampler状态也没恢复。若后续有依据扩大总预算，当前实现应从同一发布版/恒等初始化新跑完整日程，并使用新输出目录，报告额外计算成本。

成本粗估仅供排程：原90k SQ的9000步约5.24小时、S0约3.42小时，线性外推到27,981步约16.3/10.6小时每run；四卡同时运行约一晚到一天。缓存更大、并发I/O和验证设置可能改变速度，以服务器实测吞吐修正。不要提前宣称训练已完成。

## 7. 阶段 D：多任务统一评测与下一步判据

### D1. 固定评测数据和一个最终 checkpoint

用阶段 A 同样的 `eval-queries → 检索KEEP=5 → attach-eval --ks 5 → 建缓存` 准备：

```bash
python scripts/prepare_open_rag.py eval-queries \
  --inputs hotpot_test="$HOTPOT_TEST" nq="$NQ_EVAL" trivia="$TRIVIA_EVAL" \
           popqa="$POPQA_EVAL" webq="$WEBQ_EVAL" \
  --out_dir "$RAG_STAGE/benchmark_questions"

WORK_DIR="$RAG_STAGE/benchmark_retrieval" INDEX_DIR="$SHARED_INDEX" KEEP=5 GPUS=0,1,2,3 \
  bash scripts/run_open_rag_retrieval.sh \
  --queries_jsonl "$RAG_STAGE/benchmark_questions/eval.queries.jsonl"
python scripts/prepare_open_rag.py attach-eval \
  --queries_dir "$RAG_STAGE/benchmark_questions" \
  --retrieval_jsonl "$RAG_STAGE/benchmark_retrieval/retrieval.jsonl" \
  --ks 5 --out_dir "$RAG_STAGE/benchmark_ready"
CORPUS="$RAG_STAGE/benchmark_ready/corpus.jsonl" CACHE="$RAG_ROOT/cache/open_rag_benchmark_r16" \
  CHECKPOINT="$PISCO_MODEL" GPUS=0,1,2,3 bash scripts/build_hotpot_cache.sh

python scripts/run_open_rag.py eval --k 5 \
  --eval_files hotpot_test="$RAG_STAGE/benchmark_ready/k5/hotpot_test.jsonl" \
    nq="$RAG_STAGE/benchmark_ready/k5/nq.jsonl" trivia="$RAG_STAGE/benchmark_ready/k5/trivia.jsonl" \
    popqa="$RAG_STAGE/benchmark_ready/k5/popqa.jsonl" webq="$RAG_STAGE/benchmark_ready/k5/webq.jsonl" \
  --cache_dir "$RAG_ROOT/cache/open_rag_benchmark_r16" --generator_path "$PISCO_MODEL" \
  --checkpoints full_SQ="$RAG_ROOT/runs/open_rag_scale_s42/full_SQ_s42/checkpoint_best.pt" \
    full_S0="$RAG_ROOT/runs/open_rag_scale_s42/full_S0_s42/checkpoint_best.pt" \
    90k_SQ="$RAG_ROOT/runs/open_rag_scale_s42/90k_SQ_s42/checkpoint_best.pt" \
    90k_S0="$RAG_ROOT/runs/open_rag_scale_s42/90k_S0_s42/checkpoint_best.pt" \
  --out_dir "$RAG_ROOT/runs/open_rag_scale_best_benchmark" --execute
```

正式任务评测只在开发选点结束后进行。HotpotQA这里的test5405是官方validation的项目留出部分，历史实验已查看过结果；不冒充官方隐藏test或从未查看的盲测。新的参数/epoch/K决策继续使用项目dev和混合dev。

`run_open_rag.py eval --eval_files` 可同时传多个 `name=路径`。分别评测full/90k的同一步快照或last，再评测各自按固定mixed-dev规则选的best；全部任务共享同一个相应checkpoint。发布版PISCO每个证据设置只需评测一次，避免重复计算；后续同证据的快照/last评测可加 `--skip_pisco`，不能跨问题集或K复用基线分数。

例如阶段 A 已建好的HotpotQA开发证据可以先用于同预算比较：

```bash
python scripts/run_open_rag.py eval --k 5 \
  --eval_files hotpot_dev="$RAG_STAGE/hotpot_dev_ready/k5/hotpot_dev.jsonl" \
  --cache_dir "$RAG_ROOT/cache/open_rag_hotpot_dev_r16" --generator_path "$PISCO_MODEL" \
  --checkpoints SQ="$RAG_ROOT/runs/open_rag_scale_s42/full_SQ_s42/checkpoint_last.pt" \
                S0="$RAG_ROOT/runs/open_rag_scale_s42/full_S0_s42/checkpoint_last.pt" \
  --out_dir "$RAG_ROOT/runs/open_rag_full_last_dev" --execute
```

对90k同预算run使用另外一个输出目录，随后用 `analyze_open_rag.py` 比较full−90k、SQ−S0、SQ−PISCO。不要用测试指标选择 best。

### D2. 如何决定后续投入

下面0.5/1.0 F1或match百分点是本轮排程的实际收益门槛，不是普遍统计定理。CI反映题目抽样，第二seed检验训练随机性；不可互相代替。

| 观察 | 下一步 | 可得结论与限制 |
|---|---|---|
| 同预算full−90k在mixed dev改善≥0.5 F1pp，配对CI下界>0，Hotpot检索K5不明显退化 | 优先对full-SQ/S0补seed43，再固定checkpoint做多任务主表 | 支持同更新预算下扩大问题池的收益；不能保证继续扩量单调有效 |
| 90k重复≈5遍追上或超过full≈1遍 | 保留重复训练的有效checkpoint，检查学习曲线再选预算 | 当前预算更看重重复学习；不能再声称“问题种类不够”已证实 |
| full和90k同预算都无持续改善，SQ仍低于PISCO的match | 停止默认加到full三遍；检查实际可见证据和gold/教师输出差异 | 当前扩容/重复不是已验证解法；不据此否定query相对S0的贡献 |
| full后段连续多个验证点仍有稳定增长，且Hotpot来源不反向退化 | 另立full两遍预算；仍上升才考虑三遍 | 新日程从同初始化开始；多epoch改变LR日程，报告为训练配方比较 |
| 检索K10使PISCO在Hotpot dev的match提高≥1pp且配对CI下界>0，SQ也呈一致内容收益 | 再做同问题池、同更新数的检索K5/K10 SQ/S0训练对照 | 支持探索更多检索文档；80对40 soft tokens有额外成本，主K5表仍单列 |
| K10只有EM/F1上升而match不升，或PISCO/SQ方向不一致 | 先分析输出长度、证据与读出，不直接开启K10训练 | 分数变化不能简单归因检索覆盖不足 |

mixed-dev决策用新的固定开发切片，并记录每来源n。若“改善≥0.5”依据全dev，应对相同checkpoint生成全dev预测后进行配对分析，不能把两条带噪声的曲线点相减当作配对CI。是否“不明显退化”按Hotpot开发F1下降超过1pp且CI上界<0识别；否则报告不确定，而非自动判定安全。

若只剩两张卡，可在阶段C命令中加 `--cohorts full --gpus 0,1` 先执行full-SQ/S0；但完成90k同预算run前，只能说“全量+更多更新的综合变化”，不能宣称已隔离独立问题覆盖收益。之后可用 `--cohorts 90k` 和新输出目录补对照。

有依据补第二seed时，复用已固定的文件和缓存，将阶段C命令改为 `--cohorts full --seed 43 --data_order_seed 42 --out_dir "$RAG_ROOT/runs/open_rag_scale_s43"`。这次只跑full-SQ/S0；数据划分/顺序保持seed42，独立改变模型初始化seed，不再为复核重复整个数据构造或90k控制。

## 8. 需要回传的材料与当前实现状态

服务器回传：

1. 全量/90k问题与attach manifest、执行计划、模型及语料版本、GPU smoke与缓存校验记录。
2. 四run的config、逐步train_log、best_checkpoint、每来源开发曲线、快照/last/best的step及实际examples。
3. 每任务每方法预测与result、三指标、答案长度分布、Hotpot bridge/comparison分组与配对CI；原生K10结果另表。
4. wall time、GPU峰值显存、缓存唯一段落数、磁盘大小；seed43与seed42单独列出后再汇总。

代码入口：

| 文件 | 作用 |
|---|---|
| `scripts/prepare_open_rag.py` | 评测问题规范化、移除原生证据、K5/K10前缀attach、缺检索清单、嵌套90kready文件 |
| `scripts/run_open_rag_retrieval.sh` | 共享索引搜索与重排，检查worker退出码与合并覆盖 |
| `scripts/build_splade_retrieval.py` | CPU CSR问题向量、GPU分块精确TopK、输入指纹与索引/排序检查 |
| `scripts/run_open_rag.py` | 同证据评测、相同更新数的四run训练、CPU计划与GPU调度 |
| `scripts/analyze_open_rag.py` | 严格同题/同证据比较，配对CI和逐来源/逐hop分析 |
| `src/train.py`、`src/metrics.py` | 同预算快照、名义遍数、逐来源验证、预测中保留证据IDs |

本次实现和CPU验证不包含真实服务器GPU检索/训练结果。新增测试覆盖证据前缀/labels、开发集一致性、同预算启动、PISCO全部memory、stale输出拒绝、失败worker阻止打包、分块检索与全量点积一致，以及真实CPU训练快照可重新评测。服务器仍须完成CUDA smoke和上述正式运行。

本地验证：`python -m unittest discover -s tests` 共153项，147通过、6项因可选依赖跳过；`python tests/test_shapes.py` 102项通过。新增开放域执行测试13项通过，并通过Python编译、shell语法和 `git diff --check`。CPU使用隔离测试依赖，未升级服务器环境。

## English execution summary

Keep retrieved K5 as the primary protocol. First evaluate existing checkpoints on identical retrieved evidence; K10 uses the prefix-compatible ranking as a development diagnostic. Then compare a full-pool pass with approximately five repetitions of a nested90k pool under the same optimizer updates and LR schedule, using SQ/S0 in each cohort. Further epochs, K10 training, and seed replication follow development evidence rather than the old native-context test gap.
