# 共享文档投影器：当前推荐主臂

在冻结发布版 PISCO 与相同缓存/输入预算的约束下，主臂改为：每篇 m 个 memory
共用一套 MLP，先以 memory 查询逐 token 的 contextual question states，再生成
本篇 m 个残差。原来的全局展平版保留为独立对照，不改变其配置或 checkpoint。

首轮结果见 [SHARED_QUERY_PROJECTOR_RESULTS.md](SHARED_QUERY_PROJECTOR_RESULTS.md)。

## 判断与结构

全局展平版的输出残差处于一个全局 r 维仿射子空间，且文档排名对应不同权重。
这是真实的归纳偏置/表达限制，但不是已测出的 QA 失败原因。共享版各篇使用
**相同输出基底、独立条件系数**；K 篇残差整体最多有 K*r 个自由系数，而不是
每篇学一套不同基底。共享版可以直接复用同一 checkpoint 处理 K=2、K=10。

对每篇 Z_i 的 m 个 memory：

```text
H = frozen_decoder(question_only).hidden_states[-1]
C_i = MultiHeadCrossAttention(Q=LN(Z_i), K=LN(H), V=LN(H), valid_question_mask)
U_i = GELU(Wz vec(LN(Z_i)) + Wc vec(LN(C_i)) + b1)
Delta_i = reshape(Wo U_i + b2, m, d)
E_i = Z_i + Delta_i
```

query cross-attention **位于 MLP 之前**，替代问题的定长展平输入。C_i 始终是
m×d_a，不依赖 T_q。只展平本篇 m 个 memory 和 m 个条件向量；m 个训练过的
memory 槽位仍具有篇内顺序，文档之间不设排名权重/位置编码。

可选 `--projector_cross_document` 在 U_i 上加一层文档 self-attention，然后由
同一个 Wo 输出残差。这层 attention 带文档 mask，不加文档/排名位置编码，
因此投影器对文档置换等变：换输入文档顺序，只会相应交换输出块。decoder
本身仍有顺序/因果位置，不能据此声称最终答案对文档顺序不变。此开关用于
单独检验多跳是否需要投影层跨文档融合，默认关闭，不预设收益。

## Query 选择和尺度

默认使用冻结 decoder 的问题独立前向，保留所有有效问题 token 的 hidden
states。输入仅是问题 token，不包含文档、答案或生成后缀。原问题仍进入 D0
decoder prompt；新模块负责用问题调整缓存，并不替代问题文本。

`--projector_conditioning last` 只取最后有效问题 token 的 hidden state h_q，
经过同一形式的 value projection/LN，再广播到本篇 m 个 memory，交给相同
MLP。它省掉 query attention，而不省掉 contextual question 的 LM 前向。
此臂参数稍少，不能将其当作严格的参数量匹配对照；h_q 是可用条件向量，
并没有证据保证它是最佳语义摘要。

`--query_encoder_kind word_embedding` 使用原始冻结词嵌入，不额外运行问题 LM。
只用于 cross-attention 模式；添加无参数 sinusoidal token 位置特征，并再次
归一化。没有这一步时，word embedding + 无位置 attention 会成为词袋读出。
与 contextual states 具有相同 hidden 维度/注入接口，不等于同分布或同范数。

问题在 batch 内按实际最大长度 padding，由有效 token mask 隔离；投影器不
补到固定 T、不含按问题位置独立的全连接权重。默认 `max_query_len=256` 是
数据入口的安全上限，通常不会给短问题多做计算。可以根据真实训练长度 p99
调整，比如 64；`query_feature_truncation_fraction` 记录实际截断比例。
改变这个上限不会改变共享投影器权重形状，可以回载同一 checkpoint。

memory、问题特征以及 attention 输出均在残差分支中使用无仿射 LayerNorm。
进 MLP 的 query 特征 RMS 因而基本不随问题长度增长，**不是严格固定范数**：
epsilon 和零/近零方差输入仍会影响输出范数。softmax 使每个 head 的读出成为
其 value 的凸组合，也不保证多头拼接后的整体范数固定。最终 E 不做硬范数
匹配，保留原始 Z 的幅度与直通路径。

## 预算、冻结与训练

- 每篇 m→m；有效 K=2 时输出 2*m，K=10 时输出 10*m。没有 memory 筛选、
  缩减、补齐到最大 K 或预算 dropout。结果使用 `B=full` 和实际 token 计数。
- 原 encoder/decoder、两组 LoRA、词嵌入、LM head 全冻结；decoder_adapter
  保持激活，eval 关闭 dropout。冻结 decoder 的答案前向保留 autograd。
- Wo/b2 零初始化，起点有效 E=Z。contextual query 前向和原始缓存都 detach，
  只有新投影模块（含开启的 attention）接收 gold answer CE 梯度。
- E 替换 D0 输入层的 memory embedding，memory 在前、完整问题在后；
  不改问题 token、不新增逐层 KV prefix、不换成中间层注入。
- AdamW、lr=5e-5、5% warmup、线性衰减与 gold CE 延续前轮设置；step 0
  是可选的最佳 checkpoint。训练数据与课程组织沿用上一轮 runbook。

默认 d=4096、m=8、MLP r=512、attention d_a=256、heads=8：

| 臂 | 结构 | 可训练参数 |
|---|---|---:|
| SQ | 共享文档 MLP + token query attention | 37,782,016 |
| SQX | SQ + 文档间 attention | 38,832,640 |
| SL | last-token h_q + 共享文档 MLP | 35,684,864 |
| S0m | SQ，query 换固定占位输入 | 与 SQ 完全相同 |

S0m 不读取实际 query 或其长度，固定 4 个占位向量走同样 attention/MLP；X
开关在主臂和匹配对照中应保持一致。比较全局展平版 JQ 与共享版 SQ 时，要
固定问题表示、截断上限、数据和训练设置，并报告容量差异。不能把默认 JQ
word/r128 与默认 SQ contextual/r512 的分数差单独归因于共享权重。

## 运行

```bash
# 当前主臂
python -m src.train --preset pisco_shared_projector \
  --tag shared_contextual --out_dir /data02/quro/runs/shared_contextual \
  --query_control --doc_control

# 多跳跨文档交互独立消融
python -m src.train --preset pisco_shared_projector --projector_cross_document \
  --out_dir /data02/quro/runs/shared_crossdoc --query_control --doc_control

# 单向量 h_q 对照，仍为冻结 decoder 的问题独立前向
python -m src.train --preset pisco_shared_projector --projector_conditioning last \
  --out_dir /data02/quro/runs/shared_last --query_control --doc_control

# 有位置 word embedding 的低成本对照
python -m src.train --preset pisco_shared_projector --query_encoder_kind word_embedding \
  --out_dir /data02/quro/runs/shared_word --query_control --doc_control

# 参数量匹配的固定 query 对照
python -m src.train --preset pisco_shared_projector --projector_query_mode agnostic_matched \
  --out_dir /data02/quro/runs/shared_agnostic --query_control --doc_control

# 同一 checkpoint 改为 K<=2：不需要重新训练，不需要设置 budget
python -m src.train --preset pisco_shared_projector --eval_only --max_docs 2 \
  --resume_from /data02/quro/runs/shared_contextual/checkpoint_best.pt \
  --out_dir /data02/quro/runs/shared_k2

# 测试共享版、旧展平版和冻结训练接入
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*projector.py' -v
OMP_NUM_THREADS=1 python tests/test_shapes.py
```

checkpoint/缓存路径通过原有 `--generator_path`、`--cache_dir` 覆盖，必须来自
同一发布版 PISCO。共享版禁止 budget/sweep 标志；`--max_docs` 是数据截取上限，
不属于投影器权重形状。last、cross-document、word/contextual 等结构/表示
变化必须使用对应 checkpoint，代码会拦截混载。

代码测试不能证明真实 QA 改善；仍需真实缓存与发布版 7B reader 的 EM/F1、
时延实验。历史 raw-memory bridge gap 或 Q→M 弱信号支持研究动机，不能单独
证明当前失败原因或某个 query 摘要必然不足。
