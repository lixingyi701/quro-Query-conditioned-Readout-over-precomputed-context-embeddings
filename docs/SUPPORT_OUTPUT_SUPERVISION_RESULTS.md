# 输出端支持监督与离线 top-k 结果（2026-10-06）

**一句话结论：** 把支持损失直接接到送给 decoder 的最终 E 上，QA 仍然是零收益：
SQ+DocE − SQ+HeadE = −0.01 F1 [−0.35, +0.31]。头学会了分文档（Recall@2 0.53），
但比上一轮的 hidden 头更不依赖问题：给投影器换问题后，logit 相关 0.992，95.6%
的题前 2 篇文档集合不变。离线 top-k（任务一）显示，上一轮训练过的 hidden 头在
k=6（约 48 个 memory）时覆盖 87% 的 gold，换问题后各项差都在 ±0.6pp 内。

设计和判据见 [SUPPORT_OUTPUT_SUPERVISION.md](SUPPORT_OUTPUT_SUPERVISION.md)，
首轮结果见 [SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md](SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md)。
代码：`feat/pisco-joint-query-projector@49f1c16`。运行输出在
`/data02/quro/runs/support_output_v1/`（启动脚本与日志在 `logs/`，QA 分析在
`analysis.txt`，DocE top-k 在 `topk_doc.txt` 与 `sq_ce_doc/support_topk_dev.json`）。

## 1. 任务一：已有 hidden 头的离线 top-k（不重新训练）

输入是上一轮 `support_doc_v1` 两臂的 dev 预测（2000 条，last），用
`scripts/analyze_support_topk.py --topk 2,4,6 --memories_per_document 8`，结果写在
各臂目录的 `support_topk_dev.json`。只是证据覆盖率，不是筛选后的 QA 或时延。

| 头 | Recall@2/@4/@6（%） | both@2/@4/@6（%） | 换问题后集合不变（%） |
|---|---|---|---|
| SQ+Doc（训练过，λ=0.1） | 55.2 / 75.6 / 87.4 | 24.2 / 55.1 / 76.0 | 86.0 / 77.9 / 77.1 |
| SQ+Head（随机头，λ=0） | 20.0 / 35.5 / 51.2 | 2.2 / 9.4 / 21.3 | 51.3 / 39.8 / 35.5 |

- 两篇 gold 都可见的 1675 题上，训练过的头 Recall@6 88.9%、both@6 78.7%。
- 训练过的头，错问题减正确问题的配对差全部在 ±0.6pp 内，CI 都跨零：排序不依赖问题。
- 随机头换问题后 Recall 反而掉 1–2pp（CI 不跨零），只说明随机投影里有问题成分，
  不是训练出来的选择能力。

## 2. 任务二：两个臂

**共同设置**：从上一轮 SQ last（`shared_projector_v1/shared_contextual/checkpoint_last.pt`，
尚未受过 Doc 监督）只加载权重，optimizer/scheduler 重置；1000 步，lr 2e-5，
batch 2 × 累积 8，seed 42，单 seed；可见性注释数据 `hotpot-support-visible`，
`visible` 策略。两臂都建立同样的 output 头（`--support_head_input output`，
`a^T LN(mean_j E_ij) + b`，4097 参数），初始化相同。

| 代号 | 目录 / tmux / GPU | λ | 训练目标 |
|---|---|---|---|
| **SQ+HeadE** | `sq_ce` / `so_sq_ce` / GPU0 | 0 | 只有答案 CE，头保持随机 |
| **SQ+DocE** | `sq_ce_doc` / `so_sq_ce_doc` / GPU1 | 0.1，前 100 步线性升权 | CE + 0.1 × 平衡 BCE |

**起点核对**：第 0 步 dev500 两臂都是 EM 0.514 / F1 0.643，与 SQ last 一致；第 1 步
qa_loss 都是 0.1239，数据顺序一致；DocE 的 support_head_grad_norm 从 0.002 增到
0.077，辅助梯度确实到达。

## 3. QA（dev 2000，last checkpoint）

| 臂 | EM | F1 | substring |
|---|---:|---:|---:|
| SQ+HeadE | 49.65 | 63.55 | 54.40 |
| SQ+DocE | 49.65 | 63.54 | 54.45 |
| SQ（不续训，参照） | 50.05 | 63.63 | 54.85 |
| S0m（不看问题，参照） | 50.15 | 63.74 | 54.85 |

| 配对差（bootstrap 95% CI） | F1 |
|---|---|
| **DocE − HeadE（主比较）** | **−0.01 [−0.35, +0.31]**（EM +0.00，substring +0.05） |
| bridge（n=1622） | −0.09 [−0.44, +0.28] |
| comparison（n=378） | +0.32 [−0.42, +1.32] |
| 两篇 gold 都可见（n=1675） | −0.02 [−0.37, +0.32] |
| HeadE − SQ / DocE − SQ | −0.08 / −0.09（均不显著） |

dev500 验证曲线两臂都在 63.5–65.6 之间波动，best 都在第 400 步（HeadE 65.6、
DocE 65.1），差别在噪声内，主分析只用 last。

## 4. 问题依赖

**换问题掉分**（只给投影器和头换问题，decoder 仍看正确问题）：HeadE 0.73
[−0.05, +1.58]，DocE 0.64 [−0.17, +1.52]，差 −0.09 [−0.54, +0.34]。输出监督没有
让投影器更依赖问题。

**DocE 的头**（dev 2000）：

| 指标 | 正确问题 | 错问题 |
|---|---:|---:|
| Recall@2 / both@2 | 53.4 / 22.3 | 53.2 / 22.0 |
| Recall@4 / both@4 | 73.8 / 52.5 | 73.8 / 52.3 |
| Recall@6 / both@6 | 85.7 / 72.6 | 85.7 / 72.4 |

换问题后 logit 相关 0.992（上一轮 hidden 头 0.93），每题 max|Δlogit| 中位数 0.17
（logit 标准差 1.18），top-2 集合不变 95.6%（hidden 头 86%）。配对差全部在 ±0.4pp
内，CI 跨零。检索排名前 2 的 Recall@2 是 19.6%。

## 5. 结论

1. **按判据属于"分类改善、QA 不变"。** 支持损失直接更新 Wo 仍不足以改善生成，
   首轮的阴性结果不能再归因于辅助梯度与生成出口分离。
2. **输出端的头比 hidden 头更不用问题。** 这与"E = Z + Δ 的槽均值以原始 Z 为主，
   分类可以直接读 Z"的假设方向一致，但没有做 ‖mean Δ_i‖/‖mean Z_i‖ 等通路级测量，
   只是相关观察。
3. **现有监督不要求使用问题。** 每个样本的 10 篇文档都属于同一个问题，标签可以
   主要由文档本身预测；两轮的头都没有学到问题依赖。
4. **下一步**：支持监督连续两轮对 QA 零收益，下一轮转向直接作用于生成路径的问题
   乘性调制（FiLM 对照，λ=0）。混合池（同池换问题换标签）
   的评估与训练留作后续机制测试。
