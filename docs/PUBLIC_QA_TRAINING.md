# 复用 COCOM/PISCO 公开混合训练集

更新：2026-10-08。适用分支：`feat/pisco-joint-query-projector`。

## 1. 决定与发布范围

优先复用作者的 **`dmrau/multi_qa`**，不再把之前建议的 NQ 25k、TriviaQA
25k、HotpotQA 30k、ASQA-short 3k、FactKG 7k 配额作为默认主实验。
后者可以保留为后续任务扩展；当前先使用公开混合来源，减少自行重建数据的差异。

公开源：[multi_qa](https://huggingface.co/datasets/dmrau/multi_qa)。本实现固定版本
`b0b01e2a0f6e251e9cdd191f5018bec7b170a554`；唯一训练 parquet 的 SHA256 为
`ca0cd7ce892b1c6e764bbde906dc62b3127d2ef76156f594083a400fd736fe72`。

[PISCO 论文 §4.1](https://aclanthology.org/2025.findings-acl.800.pdf) 明确链接这一
453k 问题池，说明从 Wikipedia-KILT 检索 top-5 文档，再由教师生成 silver label。
因此公开 **问题池** 和 **最终 PISCO 蒸馏训练包** 是不同资产：

| 资产 | 核实状态 | 我们的处理 |
|---|---|---|
| 混合训练问题与参考答案 | 已公开，453,023 行，56,580,316 字节下载量 | 直接下载固定版本 |
| 分块文档库 | [dmrau/kilt-128](https://huggingface.co/datasets/dmrau/kilt-128) 已公开，约 21M 行 | 需要重建作者检索流程时复用；不必先压缩整个库 |
| 每个训练问题的完整 top-5 文档映射 | 此次没有找到完整公开下载包 | 优先接入服务器已有结果；缺失问题输出补齐清单 |
| 各教师的完整训练 silver answers | 在核实的问答发布和 BERGEN 分支中没有找到完整公开包 | 当前投影器仍用已验证的 gold CE；不声称复制了 PISCO 蒸馏标签 |
| 适配后的 PISCO 权重 | 当前实验已有 | 冻结同一权重，缓存和 reader 均复用 |

本次直接检查了 `multi_qa` 数据文件，而非仅根据论文估算。原始字段只有
`id`、`content`、`label`，没有文档、检索分数或教师输出。作者的
[`pisco-eval` 分支](https://github.com/naver/bergen/tree/pisco-eval) 中
`KILTMULTIQA` 也把 `response_files` 作为另外输入，逐 ID 替换标签。
该分支当前代码写的是 `dmrau/combined_qa`；可访问的相近发布
`dmrau/combined_qa.hf` 仍是问题/原始标签，不是蒸馏包。本实现采用论文明确链接的
`multi_qa`，不跟随这种后续名称变化。

我们可以写“复用 COCOM/PISCO 发布的混合问答训练来源”；在没有取得同一检索和
silver label 之前，不能写“完整复用 PISCO 原始蒸馏训练集”。这不影响使用该公开来源
做投影器训练和 S0/SQ 控制实验。

## 2. 实测组成与答案风格

下表计数来自固定版本 parquet，按作者 ID 前缀统计，未经抽样。

| 来源 | 原始问题数 | 当前标签风格 |
|---|---:|---|
| NQ-open | 87,925 | 实体、日期、数值等短答案 |
| TriviaQA | 61,797 | 短事实答案，通常附多个别名 |
| HotpotQA | 88,839 | 多跳问题，最终答案仍较短，含 yes/no |
| ASQA | 4,353 | 此混合源是短参考标签视图；不等于原生长回答任务 |
| MS MARCO | 59,699 | 短句、定义、解释，较事实短答案更长 |
| AdversarialQA | 29,966 | 抽取式短语 |
| WikiQA | 813 | 答案句子 |
| SciQ | 11,679 | 科学问题的短答案文本 |
| FreebaseQA | 20,356 | 知识库事实短答案 |
| SQuAD | 87,596 | 抽取式答案片段 |
| **合计** | **453,023** | 短答案为主，包含句子级输出 |

保留所有原始答案标签和标签顺序，训练仍取第一个非空公开标签，评测保留全部标签。
不把 TriviaQA 的多个别名拼成一条答案，也不把 ASQA 的短标签伪装成长回答。
作者数据卡列出的过滤涉及它使用的分词器：问题长度上限 128、标签上限 64；
这些数值不保证对我们 reader 的 Mistral tokenizer 同样成立，需要实际审计。

本地验证：去除 MS MARCO 中没有非空标签的 45 行和 AdversarialQA 中规范化后为空的
1 个问题，剩 452,977 行。
固定 `seed=42`、`dev_fraction=0.01`，按规范化问题分组划分，得到训练 448,432、
开发 4,545 行；同一问题即使来自不同来源也不会进入两个分区。
**这个本地验证还没有传入服务器的评测文件**，正式导出必须传入所有调参/评测题
进行排除，届时行数会相应变化。训练集内部开发集不是任何基准的官方 test。

## 3. 本次训练与测试范围

- 主训练来源：公开 10 来源混合集。先以同一公开池的固定随机子集，例如 90k，
  验证更大规模混合训练，再按算力扩大到全部合格样本。90k 是计算预算选择，
  使用均匀无放回抽样，保留公开配比的期望，不是重新指定 10 个来源的配额。
- 主测试：NQ、TriviaQA、HotpotQA、ASQA-short、PopQA。
- 补充迁移测试：WebQuestions、FactKG；这三者中的 PopQA/WebQuestions/FactKG
  不进入本次公开问题池训练。表述为“未参加此次投影器训练”，不推断冻结模型的全部训练历史。
- ASQA-long 和有监督 FactKG 混训分别作为后续扩展，不为了训练而下载其测试题。
- 采用一个最终混合训练 checkpoint 评测所有任务；旧小规模 checkpoint 的迁移
  不是执行混训的前置条件。

固定数据 seed42 的本地 90k 抽样验证覆盖全部 10 个来源：NQ 17,453、TriviaQA
12,371、HotpotQA 17,514、ASQA 849、MS MARCO 11,841、AdversarialQA 5,970、
WikiQA 167、SciQ 2,267、FreebaseQA 4,079、SQuAD 17,489。
这是尚未排除服务器评测题的管线验证计数，正式 manifest 才是论文应引用的计数。

标准多数据集 RAG 采用固定检索 top-5、同一文档排序、同一 PISCO m8/d4096 缓存。
原有 Hotpot distractor K10 bridge/comparison 测试继续作为独立诊断协议；不把它的
原生支持段落、SeleCom 的支持文档筛选或 top-1 文件当成作者 top-5 检索结果。

S0、SQ、S0X、SQX 使用完全相同的训练/开发文件和缓存，并固定
`data_order_seed`。投影器训练继续冻结 PISCO compressor、reader 和原有 LoRA。
公开来源的复用控制数据来源；各臂共享实际文件/排序/目标/训练预算控制我们的模块对照。
已有 PISCO 固定基线与仅训练投影器的比较，不能据此描述为重新训练了同预算 PISCO。

## 4. 脚本接口

`scripts/prepare_public_qa.py` 不导入 torch，不加载模型；JSONL 操作只需标准库。
下载/读取 parquet 另用 `requirements-data.txt`，可在 CPU 上完成。

### 4.1 下载并导出作者问题池

服务器的默认大文件根目录为 `/data02/quro`，可以通过 `QURO_ROOT` 覆盖。
下例路径是准备命令的目标与示例；本次没有连接服务器核实其现有文件。

```bash
python -m pip install -r requirements-data.txt

python scripts/prepare_public_qa.py export \
  --hf_cache_dir /data02/quro/hf-cache \
  --out_dir /data02/quro/data/public_qa_90k_questions \
  --limit 90000 --seed 42 --dev_fraction 0.01 \
  --exclude_jsonl /absolute/path/to/nq_eval.jsonl \
  --exclude_jsonl /absolute/path/to/triviaqa_eval.jsonl \
  --exclude_jsonl /absolute/path/to/hotpot_dev.jsonl \
  --exclude_jsonl /absolute/path/to/hotpot_test.jsonl \
  --exclude_jsonl /absolute/path/to/asqa_eval.jsonl \
  --exclude_jsonl /absolute/path/to/popqa_eval.jsonl \
  --exclude_jsonl /absolute/path/to/webquestions_eval.jsonl \
  --exclude_jsonl /absolute/path/to/factkg_eval.jsonl
```

应替换上述所有实际评测路径，包括此前用来调参的开发题。排除规则同时检查
源 ID 和规范化问题，以覆盖 native ID、KILT ID、服务器自建 ID 的差别。
尚未准备齐评测题时可以先不传排除文件做下载/检查，manifest 会记录未做评测排除；
正式训练文件应重新导出到新目录。

输出：`train.queries.jsonl`、`dev.queries.jsonl`、`manifest.json`。
它们只包含问题和答案，必须完成 retrieval attach 才能训练。

其他用法：

```bash
# 复用已经下载的 parquet，不再联网。
python scripts/prepare_public_qa.py export \
  --input_parquet /absolute/path/to/train-00000-of-00001.parquet \
  --out_dir /data02/quro/data/public_qa_full_questions

# 只查看现有三类来源的 CPU 小规模流程；仍使用公开混合源，而非另下原始数据。
python scripts/prepare_public_qa.py export \
  --sources nq_open,triviaqa,hotpotqa --limit 30000 \
  --out_dir /data02/quro/data/public_qa_three_sources_questions
```

不指定 `--limit` 采用全部合格训练样本。`--local_only` 只允许使用已有 HF 下载缓存。
`--input_jsonl` 接受作者的 `id/content/label` 格式；输出目录必须为空，避免覆盖已有
可复现的数据版本。切分、抽样和顺序只由数据 seed 决定，与模型初始化 seed 独立。

### 4.2 接入已有检索文件

检索 JSONL 可以有原生源 ID，或问题文本。支持以下两类文件：

```json
{"id":"nq_open123","query":"...","documents":[{"doc_id":"p1","text":"..."},{"doc_id":"p2","text":"..."}]}
{"id":"server-id","question":"...","retrieved_doc_ids":["p1","p2","p3","p4","p5"]}
```

`documents` 是完整有序列表；上面第一行省略了后三篇。没有 passage ID 的原文文档
使用与 `src.data` 相同的内容 hash。ID-only 文件需要已存在的 corpus 或缓存 manifest。
源 ID 优先匹配，未匹配到时仅接受唯一的规范化问题匹配；ID 与问题冲突、多个候选、
同一文档 ID 对应不同原文会报错。

```bash
python scripts/prepare_public_qa.py attach \
  --queries_dir /data02/quro/data/public_qa_90k_questions \
  --retrieval_jsonl /absolute/path/to/nq_train_retrieved.jsonl \
  --retrieval_jsonl /absolute/path/to/triviaqa_train_retrieved.jsonl \
  --retrieval_jsonl /absolute/path/to/remaining_sources_retrieved.jsonl \
  --corpus_jsonl /absolute/path/to/existing_corpus.jsonl \
  --cache_manifest /absolute/path/to/existing_cache/manifest.json \
  --max_docs 5 \
  --out_dir /data02/quro/data/public_qa_90k_ready
```

原文内嵌且没有现有缓存时，去掉 `--corpus_jsonl`、`--cache_manifest` 即可。
`--retrieval_jsonl`、`--corpus_jsonl` 可以重复。脚本取现有排序前 5 篇，不重新排序，
不填随机段落，也不按 gold supporting facts 偷换检索。

完整覆盖后输出 `train.jsonl`、`dev.jsonl`、文档去重的 `corpus.jsonl` 和 manifest。
检索文件中的旧答案/教师答案不会替换公开 gold 标签。

如果服务器只有 NQ/TriviaQA 检索文件，attach 返回非零并写出：

- `missing_retrieval.jsonl`：带 query、source、源 ID 的待检索问题，可以直接作为补齐输入；
- `missing_documents.jsonl`：找到了 passage ID、但 corpus/缓存都未覆盖的文档；
- manifest 中的逐来源缺失计数、已匹配行数、需压缩的唯一文档数。

不生成部分覆盖的 `train.jsonl`，因此不会无意中把“10 来源混训”变成“NQ/TriviaQA 混训”。
补齐后对同一个问题池在新的空目录重跑 attach。

### 4.3 需要重建检索时

公开问题池不包含文档，下载了它仍不能直接进入在线训练。缺失问题应对固定文档库
执行 SPLADE-v3 检索 + DeBERTa-v3 重排，保留排序前 5 篇。可复用
[BERGEN](https://github.com/naver/bergen) 的检索实现；其输出必须导出为上一节的 JSONL
契约。当前脚本负责接入结果，不声称已经跑完 21M 文档库的检索/建索引。

正式重建需记录 corpus 版本、retriever/reranker checkpoint、top-k、文档文本拼接与
分块规则。BERGEN 当前 `kilt_multi_qa` 配置默认是 KILT100w，processor 也存在
上述 `combined_qa` 名称差异；不能只运行当前默认配置就声称精确复现论文中的
`multi_qa` + `kilt-128`。作者没有发布的 top-5 映射需要自行生成并固定供所有臂复用。

如果沿用我们已有检索流程而不是重建作者的流程，也可以训练，但应写明“复用
公开问题池 + 本项目固定检索”，避免把检索条件差异归因给投影器。

### 4.4 审计真实目标长度

```bash
python scripts/prepare_public_qa.py audit \
  --queries_dir /data02/quro/data/public_qa_90k_questions \
  --tokenizer_path /absolute/path/to/local/Mistral-7B-Instruct-v0.2 \
  --max_answer_len 128 --max_query_len 256
```

逐来源输出 query/首标签的 p50、p95、最大 token 数和超限数量。答案用
`" " + answer.strip()` 编码，与训练 dataset 在加 EOS 前一致。有超限返回非零；
应提高训练上限到覆盖目标，再对全部臂使用同一上限，不默默截断较长答案。
审计只读取本地 reader tokenizer，不下载或加载 reader 权重。

实际用公开 Mistral-v0.2 tokenizer（版本
`63a8b081895390a26e140280378bc85ec8bce07a`）检查了本地固定 90k 子集及其 4,545 行
开发集：训练中 191 个首标签超过 64 tokens，开发中 6 个；训练首标签最大 90 tokens，
问题最大 148 tokens。训练答案 p50/p95：NQ 4/10、TriviaQA 4/9、HotpotQA 4/10、
ASQA-short 5/11、MS MARCO 17/52、WikiQA 32/61。使用 128 target / 256 query
覆盖这批数据，因此下面的新混训命令将 target cap 设为 128。
随后对全部 448,432 行训练、4,545 行开发完成了同一 tokenizer 审计：训练问题最大
151、首标签最大 94 tokens；开发问题最大 130、首标签最大 75 tokens。
128 target / 256 query 均没有超限。
正式排除、换样本规模或更换实际 tokenizer 后仍应运行 audit。

### 4.5 构建/扩充缓存与开始训练

```bash
python scripts/build_latent_cache.py \
  --documents /data02/quro/data/public_qa_90k_ready/corpus.jsonl \
  --out_dir /data02/quro/cache/public_qa_pisco_r16 \
  --adapter src.compressors.pisco:build --device cuda --resume

python scripts/check_rag_data.py \
  /data02/quro/data/public_qa_90k_ready/train.jsonl \
  /data02/quro/data/public_qa_90k_ready/dev.jsonl \
  --cache_dir /data02/quro/cache/public_qa_pisco_r16 --max_docs 5

python -m src.train --preset pisco_shared_projector \
  --train_file /data02/quro/data/public_qa_90k_ready/train.jsonl \
  --eval_files dev=/data02/quro/data/public_qa_90k_ready/dev.jsonl \
  --cache_dir /data02/quro/cache/public_qa_pisco_r16 \
  --max_docs 5 --max_query_len 256 --max_answer_len 128 \
  --gen_max_new_tokens 128 --steps 9000 --batch_size 2 --grad_accum 8 \
  --seed 42 --data_order_seed 42 --projector_query_mode conditioned \
  --out_dir /data02/quro/runs/public_qa_90k/SQ_seed42
```

`max_answer_len=128` 应在正式数据长度审计通过后使用；需要更大上限就同步修改命令。
90k 训练行、有效 batch16、9,000 optimizer steps，对应约 144k 次样本呈现，即 1.6 个
遍历量。全量约 448k 行不能用相同步数就描述为相同 epoch；应按实际行数重新计算。

如果 attach 使用了已有缓存 manifest，实际训练也必须覆盖其中的全部文档。
最方便的是对同一个兼容缓存目录用 `--resume` 追加新文档，再跑完整覆盖检查。
如果创建新缓存目录，应先复制/合并相同 compressor、m、h、dtype 的旧缓存，
或者提供全部 ID 的原文重新构建；不能只缓存新增原文而忘记已有 ID-only 文档。
当没有任何新增文档时，直接使用旧完整缓存，无需运行构建命令。

S0 使用 `--projector_query_mode none`，其他设置保持同一份配置。SQX/S0X 如需训练
增加 `--projector_cross_document`。最终统一模型的七数据集评测用相同生成上限；
FactKG 使用 Accuracy，WebQuestions 保留原生实体集合，ASQA-short 和原生 ASQA-long
分别命名。新增参数默认不改变旧实验的 48 target/32 generation 上限。

## 5. 已完成与待执行

已完成：固定公开版本的核实与真实 parquet 下载；全量导出与 90k 抽样验证；真实
Mistral tokenizer 长度审计；13 项准备/训练接口测试和全套 140 项 CPU 回归测试，
覆盖跨来源评测排除、切分/抽样复现、原始标签保留、检索 rank/ID 保留、文档去重、
不完整覆盖拒绝、缓存覆盖和冲突拒绝；训练 CLI 新增目标与生成长度设置。

服务器待执行：核实已有数据/检索/缓存清单，取得七数据集评测问题；正式排除并导出，
补缺失检索，审计本地 tokenizer 长度，追加缓存，启动匹配训练和统一评测。
本次未连接服务器，未运行真实 7B 训练或宣称获得新的 QA 分数。
