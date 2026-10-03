# 冻结发布版 PISCO 的联合 query 条件投影

**本页保留旧展平对照 JQ/J0m。当前推荐主臂是
[共享文档投影器](SHARED_QUERY_PROJECTOR.md)，预设 `pisco_shared_projector`。**

首轮问题：相同文档缓存、相同发布版 decoder、相同 memory 输入预算，
仅训练 query 条件投影器，能否改善 QA？默认以 HotpotQA 多跳设置验证。

## 两个设计选择

**全部文档联合投影。** 每篇单独处理时，文档 i 的输出只能依赖该文档和
query；跨文档融合留给 decoder。联合处理时，文档 i 的输出也能依赖文档 j，
更直接地检验跨文档适配。联合展平并不是实现多跳的必要条件，attention 也能
提供跨文档交互；本轮选择它是为了落实简单的两层 MLP 假设。

默认 K=10、m=8，输出仍为 80 个 memory。短样本补齐到 K，mask 后只有真实
文档的 m 个输出进入 decoder，和直接读取缓存的预算一致。文档按缓存读取顺序
排列，展平的权重具有槽位/检索排名依赖，因此此版本不宣称排列不变或任意 K
泛化。改变最大 K 需要新投影器；一个 K=10 模型可以处理 K<=10 的样本。

**默认有序 word embedding。** 使用冻结发布版 decoder 的原始词嵌入，query
token 按顺序展平；各位置对应不同的投影权重，因此不会像均值池化那样完全
丢失词序。投影器仍需从 QA 监督学习组合语义。该表示不需要额外 LM 前向。

`--query_encoder_kind generator` 切换到冻结 decoder 的逐 token contextual
hidden states，投影器输入长度、维度和参数量一致。它可能让问题语义更易解析，
也会增加一次问题前向的时延；没有证据预先认定哪一种 QA 更好。query 前向只
看到问题，不包含缓存、答案或 teacher 输出。在 decoder 全冻时默认
`shared_current` 已是固定函数，无需复制 LoRA。

默认 T=64，query 特征超长时截取前 T token；decoder 的原始完整问题仍使用
原模板。结果记录 `query_feature_truncation_fraction`。如截断率影响实验，训练
前固定更大的 `--max_query_len`，不能用不同 T 的投影器混载 checkpoint。

## 结构与参数

令 Z 为所有文档 memory，Q 为有序问题特征：

```text
u = GELU(Wz vec(LN(Z)) + Wq vec(LN(Q)) + b1)
delta = reshape(Wo u + b2, K*m, d)
E = Z + delta
```

第一层的 Wz/Wq 实现与 `[vec(Z); vec(Q)] -> Linear` 数学等价，独立初始化
可以避免巨大的 memory fan-in 让 query 分支初始方差过小。GELU 提供 query 与
memory 的非线性交互；单层拼接 Linear 只能加上一个与文档无关的 query 项。

LayerNorm 无可训练仿射参数，只作用于残差分支输入，按每个 token 的 hidden
维归一化。Z 的直通路径不归一化，最终 E 不做硬范数匹配。填充在归一化前后
均被屏蔽。输出末层权重、bias 零初始化，step 0 的有效 E 恰好为原始 Z；第一
步先更新末层，随后 CE 梯度进入第一层，这是零初始化的预期行为。

默认 d=4096、K=10、m=8、T=64、隐藏层 r=128：投影器 117,768,320 参数，
约 117.77M。定长展平很宽，但低维隐藏层避免了 `(K*m*d)^2` 的直接映射。
参数随 K 和 T 线性增长。r 约束的是残差变换，原始缓存由直通路径保留，
此轮不是将所有文档内容强制压缩成 128 维后独立重建。

## 冻结和训练

- 原始 encoder、decoder、两组发布版 LoRA、词嵌入、LM head 全部冻结。
  保留 decoder_adapter 激活，LM 保持 eval 关闭 dropout；只优化投影器参数。
- 训练时文档来自已有缓存，不加载或运行在线文档压缩器。冻结 decoder 的
  **答案前向必须保留 autograd**，否则 CE 无法回传到 E。
- 只用正确答案的 teacher-forcing CE，包含 EOS；prompt 和 padding label
  为 -100。忽略数据中的 `teacher_output`，不加 KD、状态对齐、预算 dropout
  或残差惩罚。保留原 PISCO D0 的 memory-before-question 顺序和模板。
- 默认 AdamW，lr=5e-5，warmup 5%，线性衰减，梯度裁剪 1。batch=2，累积
  8 次。每 250 步在固定 dev 子集上选择 EM，step 0 也可成为最佳 checkpoint。
  最后的自动评估仍使用 last；报告 best 要显式加载 `checkpoint_best.pt`。

数据与训练组织参考 [SeleCom](https://arxiv.org/html/2602.15856v1) 的答案监督/
课程学习，以及 [COCOM-light](https://arxiv.org/html/2407.09252v3) 的 QA 数据
混合和表示投影。它们不直接证明「两端全冻，仅训这里的联合投影器」会成功：
SeleCom 的第一阶段同时训练 selector，COCOM-light 的主要训练也涉及 decoder。

复用仓库已有的 canonical corpus/query 格式和 cache-building 工具即可。先用
SeleCom 的简单、证据可见 QA 做可选 warm-up，再通过 `--warm_start` 在
HotpotQA/检索 QA 上训练同一个投影器；两个阶段均不解冻原模型。做多跳主实验
应保持 K=10 并包含真实多证据问题及干扰文档，单文档 warm-up 本身不能验证
跨文档推理。COCOM 的 question/answer 数据需要先配套检索文档，不能直接
当成已有缓存的 `(query, doc_ids, answer)`。

PISCO 现有编码入口把每篇原文限制为 128 token。训练前检查关键证据是否确实
进入这个窗口，或先确定分段方案并重建所有对照共用的缓存；不能只用未截断
原文判断监督是否有效。dev/test 不按正确答案挑选窗口。数据拆分应尽量按
文档/实体分组，同一缓存文档配多个不同 query 有助于检验条件适配。

## 运行

从仓库根目录运行。`PISCO_MISTRAL` 必须指向发布版 checkpoint，缓存必须
由同一版本生成，且 manifest 中 m=8。不要替换成自行续训的 reader。

```bash
# 默认联合 word-embedding projector
python -m src.train --preset pisco_joint_projector \
  --tag joint_word --out_dir /data02/quro/runs/joint_word \
  --query_control --doc_control

# 相同结构和参数量，逐 token contextual query 表示
python -m src.train --preset pisco_joint_projector \
  --query_encoder_kind generator \
  --tag joint_contextual --out_dir /data02/quro/runs/joint_contextual \
  --query_control --doc_control

# 参数量相同、固定 query 占位输入的独立训练对照
python -m src.train --preset pisco_joint_projector \
  --projector_query_mode agnostic_matched \
  --tag joint_agnostic --out_dir /data02/quro/runs/joint_agnostic \
  --query_control --doc_control

# 发布版直接读取同一缓存的基线
python -m src.train --preset pisco_joint_projector \
  --readout pisco_direct --eval_only --eval_every 0 \
  --tag published_direct --out_dir /data02/quro/runs/published_direct

# 显式评估最佳 checkpoint；query 类型和投影参数必须与训练一致
python -m src.train --preset pisco_joint_projector --eval_only \
  --resume_from /data02/quro/runs/joint_word/checkpoint_best.pt \
  --out_dir /data02/quro/runs/joint_word_best --query_control --doc_control
```

覆盖路径用 `--generator_path`、`--cache_dir`、`--train_file`、`--eval_files`。
若改变 K，必须同时设置 `--max_docs K --budget K*8 --budget_buckets K*8`，其中
命令行应写计算后的整数。模型还会核对 checkpoint 的 m 与缓存规格。

`--query_control` 的 `mismatch-q` 只替换 projector 的 query，decoder 仍保留
正确问题；`mismatch-q-both` 同时替换两条路径，不能将它解释为单独的 projector
依赖。`JQ` 与 `J0m` 是两个独立训练实验；J0m 使用固定非零 query 输入，
具有完全相同的可训练参数量，但不编码实际问题。

## 验证与边界

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_joint_projector.py -v
OMP_NUM_THREADS=1 python tests/test_shapes.py
python -m compileall -q config.py src tests/test_joint_projector.py
git diff --check
```

CPU 检查覆盖：零初始化与直接缓存读取的有效 embedding、输入、logits、CE
相同；仅 projector 更新；答案/EOS 掩码；query 词序依赖；跨文档依赖；短批次
和缺失文档填充；masked NaN 隔离；固定 query 对照；contextual 选项；投影器
checkpoint 回载与冻结 reader 覆盖拦截。安装 transformers/peft 后，还测试
随机微型 Mistral 上 encoder_adapter/decoder_adapter 两组 LoRA 均冻结。

这些测试验证代码路径，不代表发布版 7B 模型上的 EM/F1、GPU 内存或线上时延
结果。真实 checkpoint 必须继续做与原生 PISCO 的零步输出对比，之后再做
同缓存 QA 实验。投影器无法恢复缓存已丢失的事实；联合输入提供交互能力，
并不保证两层 MLP 已学会多跳推理。
