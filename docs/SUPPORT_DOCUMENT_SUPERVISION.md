# 支持文档辅助监督：实现与首轮运行方案

首轮结果见 [SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md](SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md)。

后续两项任务见 [SUPPORT_OUTPUT_SUPERVISION.md](SUPPORT_OUTPUT_SUPERVISION.md)：
离线 @4/@6 诊断与 `--support_head_input output` 输出端监督。本页描述默认 hidden 头。

## 目的与范围

在已训练的 SQ 上，检验支持文档标签能否让 query–memory 交互产生更有用的
投影表示。当前报告显示主要收益来自通用短答适配，query 的稳定额外收益尚未
确立。本轮保留 SQ、冻结发布版 PISCO 和原 encoder/decoder LoRA、原 D0 模板、
每篇 8→8、K=10 共 80 个 memory；仅增加训练期分类头及辅助损失。

不新增门控、不硬选 top-2、不缩放整个 memory、不改变 query 表示、不启用
跨文档 attention。支持文档监督不是新的基础技术：HotpotQA 提供支持事实，
HGN 等工作已有联合段落/支持事实/答案监督。这里研究的是它在可复用冻结
PISCO 缓存上的训练价值，不预设它必然改善 QA 或完成多跳推理。

参考：[HotpotQA](https://aclanthology.org/D18-1259/)、
[HGN §3.4](https://arxiv.org/pdf/1911.03631)。

## 结构和梯度

```text
C_i = query cross-attention(Z_i, H_q)
u_i = GELU(Wz vec(LN Z_i) + Wc vec(LN C_i) + b)
E_i = Z_i + reshape(Wo u_i + bo)                 # 原生成路径
support_logit_i = Linear(LN(u_i))                # 新辅助头
```

`u_i` 为 512 维，无仿射 LN + 单个 Linear(512,1) 只增加 **513 个参数**。
新增模块在原参数初始化之后创建，未开启时保持原初始化行为。
辅助头不 detach `u_i`；损失更新 memory MLP、context MLP 和 query attention，
不直接更新 Wo，也不更新原模型。Wo 继续由答案 CE 更新。

训练用 `return_support=True` 获取 logits；`generate_answer` 不计算辅助头，
输入的 E 与原来完全相同。评估为文档诊断另行计算 logits，不能把这部分诊断
运行时间当成部署时延。推理不需要 gold 标签、可见性注释或原文。

共享表示接受辅助梯度不保证答案受益：头可能学会分类，decoder 实际使用的
信息仍未改善。本轮首先检验这个监督假设，结果出来后再决定是否需要门控。

## 标签、可见性与损失

`gold_ranks` 中所有文档为正例：桥接段和答案段都包括，不以答案字符串匹配
构造标签。其他文档是“未标注为支持”的负例，不代表逻辑上完全无关。
每篇独立 sigmoid，采用多标签 BCE，允许两篇同时为正。

每题对正负类分别取均值，再等权平均；最后对有效题取均值：

\[
\ell_b=-\frac12\left(\frac1{|P_b|}\sum_{i\in P_b}\log\sigma(s_i)
+\frac1{|N_b|}\sum_{i\in N_b}\log\sigma(-s_i)\right),\qquad
\mathcal L=\mathcal L_{QA}+\lambda\mathcal L_{doc}.
\]

缺标签、padding、文档错配均屏蔽；不可见正例屏蔽而不改成负例。没有可用
正例或负例的题只计算 QA，文档损失为可微的零。题目被裁到 K=2 时同样处理，
不能把裁掉的 gold 文档当负例。全训练集没有有效文档监督时启动直接报错。

默认 `visible` 策略：正例至少有一条完整支持句位于实际 encoder 输入前缀内
才参与辅助损失。`original` 是显式原始文档身份监督对照，会保留不可见正例；
不作为首轮默认。两种策略都保留原始标签用于评估。

离线脚本只加载 tokenizer，遵循本仓库/发布版 decoder-as-encoder 的实际规则：
`<ENC><bos>document<eos>`、不自动加 special tokens、右截断到 **128+3**。
这比单独数 document 的前 128 token 更准确；EOS 被截掉时可见正文长度不一定
恰为 128。仅支持发布版 PISCO 128/16、m=8，不支持独立 BERT compressor。

注释绑定缓存 manifest 的 SHA256、问题/支持事实、原文档顺序及原始标签。
改缓存 manifest（包括重新 pack）、改题目/支持事实或文档顺序后应重新注释；
训练中检测到陈旧注释会报错。旧 manifest 未记录 doc_max_length 时，脚本根据
验证过的 checkpoint 配置使用 128，并保留对原 manifest 的绑定。

可见性只证明支持句进入 encoder，不证明事实保存在压缩向量中。原文只用于
离线注释，不进入投影器或 QA decoder。脚本缓存文档的截断边界，避免同一
文档在不同问题下反复 tokenization；不重新生成 latent。

## 先准备注释并检查覆盖率

```bash
python scripts/annotate_support_visibility.py \
  --input_files train=/data02/quro/data/hotpot/train.jsonl,dev=/data02/quro/data/hotpot/dev.jsonl,test=/data02/quro/data/hotpot/test.jsonl \
  --corpus /data02/quro/data/hotpot/corpus.jsonl \
  --cache_dir /data02/quro/cache/hotpot-pisco-r16 \
  --pisco_path /data02/quro/models/pisco-mistral \
  --out_dir /data02/quro/data/hotpot-support-visible
```

先读 `visibility_report.json`：可见正例数量、两篇都可见的题数、没有可见正例
的题数、未知可见性的正例数。可监督覆盖率很低时，应先分析截断而非直接把
辅助权重调大。输入文件不会被覆盖，文档 ID 和缓存保持不变。

## 两臂从同一 SQ checkpoint 继续训练

两臂都创建同一辅助头，保证初始化随机数消耗和参数量匹配；CE-only 的头没有
损失、不会训练，其文档指标没有可解释的意义。真正的主比较是 QA。

共同起点使用已报告的 SQ last；新阶段重置 optimizer/scheduler，避免继承原
3000 步结束时接近零的学习率。1000 步、lr=2e-5 是保守的小试设置，两臂相同；
不是已经测出的最优超参数。辅助目标 λ=0.1、前 100 步线性升权。

```bash
task_sq_checkpoint=/data02/quro/runs/shared_projector_v1/shared_contextual/checkpoint_last.pt
task_stage_args=(
  --preset pisco_shared_projector --support_head
  --resume_from "$task_sq_checkpoint" --warm_start
  --steps 1000 --lr 2e-5 --seed 42 --batch_size 2 --grad_accum 8
  --select_metric f1 --eval_every 100 --eval_every_samples 500
  --eval_max_samples 2000
  --train_file /data02/quro/data/hotpot-support-visible/train.jsonl
  --eval_files dev=/data02/quro/data/hotpot-support-visible/dev.jsonl
  --support_visibility_policy visible --query_control --doc_control
)

python -m src.train "${task_stage_args[@]}" --support_loss_weight 0 \
  --out_dir /data02/quro/runs/support_doc_v1/sq_ce

python -m src.train "${task_stage_args[@]}" --support_loss_weight 0.1 \
  --support_warmup_steps 100 \
  --out_dir /data02/quro/runs/support_doc_v1/sq_ce_doc
```

不要只比较新方法与旧 checkpoint：CE-only continuation 排除额外训练量的收益。
两臂分别使用 SQ+Head、SQ+Doc 标识，配置/结果记录 head、λ、warmup、可见性策略。
最好使用同等独占 GPU；时延不可由共享 GPU 的训练耗时推断。

只有 `--warm_start` 允许旧 SQ checkpoint 缺失新增头：原投影器权重仍须完整，
结构/query 模式仍须一致。普通 resume/eval 不允许静默补头；新格式 checkpoint
若丢失已经训练过的头，也必须拒绝。正常 resume 恢复 optimizer/scheduler。

区间验证按 dev QA F1 选 best，包括 stage step 0。**训练结束自动评估的是 last**。
如需评估 best，应显式加载，且两臂使用同一事先确定的选择规则：

```bash
python -m src.train --preset pisco_shared_projector --support_head \
  --support_loss_weight 0.1 --eval_only \
  --resume_from /data02/quro/runs/support_doc_v1/sq_ce_doc/checkpoint_best.pt \
  --eval_files dev=/data02/quro/data/hotpot-support-visible/dev.jsonl \
  --query_control --doc_control --eval_max_samples 2000 \
  --out_dir /data02/quro/runs/support_doc_v1/sq_ce_doc_best_dev
```

CE-only best 用其路径和 `--support_loss_weight 0`。小试默认只看 dev，不再次
用已有 test 选结构/λ。已分析过的 5405 条可用于复测，最终正面结论还需要新的
留出评估及训练 seed 稳定性。

## 记录与判据

- `train_log.jsonl`：QA loss、support loss、每个 microbatch 的平均有效题数量、当前辅助权重、辅助头
  和 context projection 的梯度范数。梯度范数是裁剪后联合梯度，不是两损失
  分别的梯度；不能据此宣称已测出梯度冲突。
- validation/result：QA EM/F1/substring；原始支持文档 Recall@2；恰有两篇 gold
  时的 both@2；按 gold 数取 top-k 的集合命中率；所有正例可见子集的文档 recall
  和 QA F1。top-2 只用于评估，decoder 仍输入全部 memory。
- predictions：每篇 logits、原标签、标签 mask、损失 mask 和可见性，方便
  按同一问题做配对分析。文档错配不计算支持指标；query 错配仍以原问题 gold
  衡量识别退化，但不用于辅助训练。缺 gold 的新问题仍可正常生成。

首轮成功条件是 CE+Doc 的 QA 超过匹配 CE continuation，且没有明显退化；
文档识别改善、换 query 掉分更多，单独都不算 QA 成功。两臂不能独立证明收益
来自 query：若结果为正，再补有/无真实 query 的匹配监督对照。

文档识别上升但 QA 不变时，只能说分类任务学会了，不证明 memory 更可用；
此时再讨论是否把相关性接到残差门控上。QA 下降则先检查权重与标签覆盖，
而不是立即叠加新结构。句级/latent 槽位监督暂不做，因为没有句子到槽位的
可靠对应。

## 本地核验

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*projector.py' -v
OMP_NUM_THREADS=1 python tests/test_shapes.py
python -m compileall -q config.py src scripts/annotate_support_visibility.py
git diff --check
```

测试涵盖平衡 BCE、不可见正例/缺标签/NaN padding、双正例、原始输出不变、
辅助梯度进入 query attention、冻结原模型、旧权重 warm-start、新头完整性、
标签不进入 prompt、生成不计算头、真实 fast tokenizer 截断边界、离线注释 CLI、
BF16 小型 Mistral+两组 PEFT LoRA，以及主 trainer 的保存/评估接入。

本地测试验证实现，不是发布版 7B 的 QA 实验。真实训练、覆盖率和性能需由
服务器运行上述命令测量，不能提前声称支持监督已有收益。

本次核验：投影器系列 **56 项测试通过**（其中 16 项为本扩展测试），已有
shape/integration 检查 **113 项通过**；编译和 diff 空白检查通过。真实 BF16
小型 Mistral 与 PEFT 测试使用随机小模型，不需要下载发布版权重。
