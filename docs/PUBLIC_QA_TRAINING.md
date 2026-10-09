# 复用 COCOM/PISCO 公开混合训练集

更新：2026-10-08。适用分支：`feat/pisco-joint-query-projector`。

2026-10-09 接续执行见 [`OPEN_RAG_SCALE_EXECUTION.md`](OPEN_RAG_SCALE_EXECUTION.md)：保留开放域检索K5，补齐共同证据评测，再以相同更新预算比较全量与嵌套90k问题池。本文件保留早期构造说明；整批90k检索/缓存/训练已完成，实际状态以 [`PUBLIC_QA_TRAINING_RESULTS.md`](PUBLIC_QA_TRAINING_RESULTS.md) 为准。

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

### 1.1 标准答案是否还要另外拼接？当前实现到哪一步？

**公开问答已经自带标准参考答案，不需要再另外收集、拼接标准答案。**
`content` 是问题，`label` 是该问题的参考答案列表。`export` 已经把它们转换为
`query` 与 `answers`，同时保留源 ID；这一步已用真实公开全量 parquet 跑通。

需要另外接入的是 **这个问题对应的检索文档**，使每个训练样本具有：

```text
问题 query + 有序检索文档 retrieved_doc_ids + 标准答案 answers
```

这里的“+”表示在同一个数据样本里关联字段。答案作为训练监督，文档和问题作为
模型回答所需的输入；推理时不提供标准答案。训练时的答案 token teacher forcing
属于正常 CE 训练，不代表推理提示词里也能看到答案。

| 环节 | 当前实现 | 是否已完成真实全量数据准备 |
|---|---|---|
| 读取公开问题与原始标准答案 | `export` 已实现：`content → query`、`label → answers` | 是，已经下载并转换公开问题池；正式评测排除仍需服务器实际文件 |
| 为每道问题接入检索文档 | `attach` 已实现：按源 ID 或唯一规范化问题匹配检索记录，加入有序 `retrieved_doc_ids` | 代码和测试已完成；尚未在服务器核实、补齐并接入整批真实检索文件 |
| 构造可训练文件与缓存 | 完整匹配后输出 `train.jsonl/dev.jsonl/corpus.jsonl`；复用现有 cache 构建与覆盖检查 | 尚未完成服务器上的整批检索、缓存和训练 |
| 生成 PISCO 教师答案 | 本脚本不生成，也不导入替换 gold | 当前方案使用公开原始答案，不要求先生成教师答案 |

因此，**问答转换和文档匹配代码都已经实现；“整批真实训练数据已在服务器构造好”
这件事还没有完成。** 目前缺少或待核实的是检索结果/文档缓存，而非标准答案。

完整文件中的 `answers` 保留评分参考；当前训练 loader 取 `answers[0]` 作为 gold CE
目标。检索文件即使另有 `answers` 或 `teacher_output`，也不会覆盖公开问题池的
标准参考答案。多个别名不会被字符串拼接成一条训练答案。

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

## 6. 备选：最初建议的五来源 90k 配方与完整构造流程

本节留存公开训练集复用方案之前提出的“我推荐的训练、测试安排”。它是我们设计的
**任务配额混合集**，不是 COCOM/PISCO 论文的原始配比，也不是 §3 中从公开 10 来源池
均匀抽取的 90k。默认主实验仍按前文复用公开池；需要突出多跳和跨任务输出覆盖时，
可以采用本节备选，给它独立的数据版本和 manifest。

### 6.1 最初的 90k 建议配方

以下配额均指 **划分开发集、排除调参/评测问题后，最终写入训练文件的样本数**。
开发和测试样本不占用 90k 训练配额。

| 数据集与任务视图 | 建议训练数 | 训练答案风格与作用 | 测试安排 |
|---|---:|---|---|
| NQ-open | 25,000 | 实体、日期、数值等短事实答案；保持主要单跳能力 | 官方公开评测划分，EM/token F1/Match |
| TriviaQA | 25,000 | 短事实答案；保留别名评分，训练用明确的一个 gold 文本 | 官方公开评测划分，EM/token F1/Match |
| HotpotQA | 30,000 | 最终答案仍较短，包含 bridge/comparison 及 yes/no；加大多跳训练占比 | 标准检索 QA 指标；另报 bridge/comparison，原有 K10 诊断单列 |
| ASQA-short | 约 3,000 | 复用全量压缩基线的短参考答案视图，覆盖歧义问题 | 明确标注 ASQA-short；不当作原生长回答结果 |
| FactKG | 7,000 | 声明验证，输出 `True`/`False`；约 3,500 正例 + 3,500 负例 | 官方评测划分，Accuracy；可按原生推理类型分组 |
| PopQA | 0 | 此次投影器训练不使用 | 固定最终 checkpoint 直接测试，EM/token F1/Match |
| WebQuestions | 0 | 此次投影器训练不使用 | 固定最终 checkpoint 直接测试，保留完整实体答案集合 |
| **训练合计** | **约 90,000** | 四类 QA 来源 + 一类二元验证来源 | **同一个最终 checkpoint 测试上述七个数据集** |

若 ASQA 完成排除和划分后不足 3,000 条，使用全部合格 ASQA 训练题，把不足部分
补到 NQ/TriviaQA，例如尽量均分，并记录真实配额；不重复复制 ASQA 题凑数。
FactKG 在其官方训练部分内分层抽样，尽量保持两类均衡，具体标签数写入 manifest。

这个备选的目的，是让同一投影器学习短事实 QA、多跳 QA、歧义问题短回答和声明验证，
再用 PopQA/WebQuestions 检查没有参加本次投影器训练的数据集上的复用。
FactKG 在这个备选中参加训练，因而应报告为跨任务联合训练效果；在前文公开池方案中，
FactKG 则属于此次训练之外的迁移测试。两套实验的训练范围必须分别标清。

### 6.2 第一步：确定来源、官方划分与独立评测文件

1. NQ、TriviaQA、HotpotQA、ASQA-short 可以优先从 `dmrau/multi_qa` 的对应
   ID 前缀提取，复用作者已经处理的问题与参考答案；也可使用各自官方训练来源，
   但应记录具体版本、配置、划分和预处理。
2. FactKG 另取官方训练部分，转换为同一 JSONL 契约。当前公开混合问题池没有
   FactKG，不能仅靠下载 `multi_qa` 得到这 7k。
3. 固定七数据集的正式评测文件，以及已经用于模型选择的开发题；PopQA 和
   WebQuestions 不从其评测问题里抽取训练题。
4. 对训练候选同时检查源 ID 和规范化问题，排除全部调参/评测重合。NQ 与 ASQA
   有来源联系，需要做跨来源的同问题检查，不只逐个数据集内部检查。
5. 在合格候选池中按问题分组划出独立开发部分，之后才抽取表中的训练配额。
   同一问题在不同数据集有记录时也放入同一个分区。

若选择已有的 Hotpot K10 distractor 训练资产，应将其标成独立协议；标准多数据集
主实验使用固定检索 top-5，不能将两种证据条件混写为同一配置。

### 6.3 第二步：统一问题、训练目标与评分参考

统一数据样本至少保留 `id/source_id/source/source_split/query/answers`；文档接入后
加入有序 `retrieved_doc_ids`。将训练目标和评测参考的语义明确下来：

- NQ/TriviaQA/HotpotQA：训练取一个明确的标准答案文本，评分保留完整参考答案。
  别名代表同一事实的不同写法，不拼成一长串目标。
- ASQA-short：使用短参考标签视图，保留来自该视图的评分参考；当前 loader 以
  首标签为 CE 目标。原生 ASQA 的整段消歧解释不属于这套短回答目标。
- FactKG：把原生真值标签转换为字符串 `True` 或 `False`。问题可以采用固定
  指令 `Verify the following claims with "True" or "False": {claim}`，训练和测试
  使用同一约定，监督目标只包含真值文本。
- WebQuestions：原生多个答案可能是需要共同覆盖的不同实体，不能全部当作
  同一实体的别名，或只保留第一项后声称完整集合回答。测试数据保留所有原生实体。
  若另报某基线的单答案兼容视图，应独立命名。

当前训练接口读取 `answers[0]`，并不会自动使用一个额外命名为 `training_target`
的字段。复用公开标签时直接保留其顺序；从官方原始来源转换时，若选定了 canonical
训练文本，应把它放在 `answers[0]` 并保留其他评分参考，或者先明确修改 loader 契约。
保留答案集合、长回答参考、标签格式等辅助元数据，便于后续扩展，不改变当前 CE 目标。

### 6.4 第三步：给每道题关联固定检索文档

1. 用相同检索器、重排器和固定文档库，对每道训练/开发问题取得有序 top-5 文档，
   可优先复用服务器已有结果。问题池和标准答案已具备时，需要补的是这个映射。
2. 参考 COCOM/PISCO 的 open-domain 检索协议：SPLADE-v3 检索、DeBERTa-v3
   重排、固定 top-5。若采用本项目已有检索设置，如实记录实际版本，不宣称是作者
   原始 top-5 映射。
3. FactKG 需要明确实际证据来源：复用已准备的文本检索文档时保持相同来源和顺序；
   若使用原生 KG 证据并转换为文本，则在训练、测试与基线中统一该格式并独立说明。
   它是声明验证任务，不能不作说明就当作普通短事实问答。
4. 保留每题真实文档列表和排名，使用实际压缩器能够读取的内容；原文对照也应
   使用相同可见文档片段。固定分块和正文 token 预算，避免两臂证据不同。
5. 检索不完整时输出缺失清单，补齐后再形成最终训练文件；不靠随机段落或 gold
   支持文档替换维持“每题 5 篇”的外观。

Hotpot 原有 K10 的答案段/桥接段替换、逐跳读取诊断继续单独报告，不和 top-5
标准检索结果混合计算。

### 6.5 第四步：抽样、缓存与最终训练文件

在已确定 split、目标格式和检索协议的候选池中，以固定数据 seed 无放回抽取各来源
配额，合并并固定训练顺序。两类 FactKG 分层抽样；ASQA 不足时按 §6.1 补额。
所有模型臂复用同一份最终数据文件，不能分别重新抽样。

对关联文档全局去重，离线压缩一次；相同文档可供多个问题反复使用。复用已有缓存，
只追加未覆盖文档。保留 PISCO m8/d4096 规格和 frozen reader/LoRA，在线训练按
`retrieved_doc_ids` 查缓存，不重新压缩原文。

输出应包括：

- `train.jsonl`：约 90k 条，包含问题、检索文档 ID 和标准答案；
- `dev.jsonl`：独立开发题，沿用同一预处理和检索约定；
- 七数据集独立评测文件；
- 全局 `corpus.jsonl`、完整覆盖的缓存 manifest；
- 混合数据 manifest：源版本、ID、划分、真实配额、标签分布、重复/排除计数、
  检索设置、文件 hash、长度分布、数据 seed。

### 6.6 第五步：训练目标、长度与控制设置

沿用现有 **gold CE，仅训练投影器**。冻结 PISCO compressor、reader 及原有 LoRA。
SQ/S0 等对照使用相同训练数据、缓存、训练步数、有效 batch、学习率、调度、
生成配置与数据顺序 seed，模型初始化 seed 与数据 seed 分开记录。

原始建议曾用短目标上限 64、统一 greedy 生成上限 128；随后 §4.4 的真实 Mistral
审计发现公开短标签也可能超过 64。因此实际执行这套备选时，优先以 **target128 /
query256 / generation128** 为起点，对最终混合数据重新审计；不直接沿用旧 48/32，
也不将超长目标静默截断。上述审计实测针对公开池，尚未覆盖另加的 FactKG 转换文件。

约 90k 训练条目、有效 batch16、9,000 optimizer steps，对应约 144k 次样本呈现，
约 1.6 个遍历量。步数表示参数更新次数，不等于独立问题数；正式训练量以实际行数
和梯度累积配置计算。先完成混合训练，再固定最终 checkpoint 测试七数据集，
无需先证明旧单来源 checkpoint 可迁移才能开始。

### 6.7 测试指标与长回答扩展

| 任务视图 | 主要报告内容 |
|---|---|
| NQ、TriviaQA、HotpotQA、PopQA | EM、token F1、Match；Hotpot 补充 bridge/comparison |
| ASQA-short | 明确短参考协议，与相同协议下的全量压缩基线比较 |
| WebQuestions | 完整实体集合的覆盖与集合 F1；基线简化视图另列 |
| FactKG | Accuracy，必要时按推理类型分组 |
| 后续 ASQA-long | 原生完整长回答目标和任务指令，ROUGE-L、Disambig-F1、DR，另设足够的训练/生成长度 |

ASQA-long 应作为独立扩展：读取完整回答，保留歧义解释与多答案关系，明确修改
目标加载及评测流程。不能只把当前 ASQA-short 文件的生成上限加大就声称完成长回答任务。

### 6.8 这套备选的代码状态

`prepare_public_qa.py export` 已实现公开池下载、按来源过滤、问题分组切分、统一
无放回抽样；`attach` 已实现检索文档关联；`audit` 已实现长度检查。
这些公共步骤可复用。

**当前脚本没有实现本节五来源的逐来源固定配额采样，也没有实现 FactKG 官方训练
来源下载/转换与 3.5k/3.5k 平衡抽样。** `--limit 90000` 只是在所选公开来源中统一
抽样，不会自动产生 25k/25k/30k/3k/7k 配方。本次将原方案完整留存为备选，
没有把它写成已经完成的训练数据构造或实验结果；实际启用时需补齐配额与 FactKG
转换，再复用现有文档关联、缓存和训练接口。
