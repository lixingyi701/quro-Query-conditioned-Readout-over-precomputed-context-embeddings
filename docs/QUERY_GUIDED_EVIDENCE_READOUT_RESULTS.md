# Query-Guided Evidence Readout：pilot 与 A/B/C 实验结果

日期：2026-10-08  
分支：`feat/query-guided-evidence-readout`  
实现基础：`363053a`，bug fix：`954338f`

## 0. 代码 bug 修复

**问题**：`scripts/prepare_evidence_targets.py` 在调用 `annotate_evidence` 之前未对 `query` 字段做 `.strip()`，但 `src/data.py:adapt_row()` 在加载数据时会对 query 执行 `.strip()`。两处的 `source_digest` 计算结果不同，导致训练入口在 `require=True` 校验时对含尾随空白字符的 query 行抛出 `stale/misaligned evidence target` 异常。

**修复**（`954338f`）：在 `prepare_evidence_targets.py:46` 调用 `annotate_evidence` 之前，先对 query 执行 `.strip()`，与 `adapt_row` 对齐。重新生成的 `qer_train_visible.jsonl` 经全量验证（34000 行，模拟 `adapt_row` 后逐行校验）零失败。

## 1. 数据准备与审计结果

### 1.1 原始训练集审计

```
ordered_pool:   multi_q_pools=0       row_fraction=0.0000   distinct_targets=0
unordered_pool: multi_q_pools=42      row_fraction=0.0029   distinct_targets=26
document_reuse: multi_q_pools=47731   row_fraction=0.4263   distinct_targets=46892
```

结论：有序池无任何多问样本；无序池只有 55 行（0.29%）有不同目标。文档复用（42%）是单文档出现在不同池，不等于同池多问。需要走自然问题配对增强。

### 1.2 配对增强

`build_evidence_pool_pairs.py --max_pairs 2000 --seed 42 --include_original`：从 30000 条原始训练数据中找出共享支持文档、supporting sentence 签名不同的自然问题对，构造 2000 对，输出 34000 行（含原始 30000 行）。

### 1.3 可见目标准备

`prepare_evidence_targets.py` 在修复 query-strip bug 后重新运行：

| 统计 | 数值 |
|---|---|
| 总行数 | 34000 |
| 有可见目标行 | 33782（99.4%） |
| 全部支持句可见行 | 26554（78.1%） |
| 支持句总数 | 81139 |
| 可见支持句数 | 72879（89.8%） |

### 1.4 增强后重新审计

```
ordered_pool:   multi_q_pools=2000    row_fraction=0.1176   distinct_targets=1977
unordered_pool: multi_q_pools=2035    row_fraction=0.1203   distinct_targets=2002
```

2000 个配对池中 1977 个（98.9%）有不同目标，捷径防控条件满足。

## 2. CPU 工程验证

`OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_evidence_projector.py -v`：12 项测试全部通过（5.16 秒）。覆盖：
- 恒等起点（step 0 E 严格等于 S0 输出）
- padding/query mask，文档独立性
- 固定问题控制（agnostic_matched）
- 辅助梯度路径，冻结 reader，冻结 S0 保存重载
- 目标失配屏蔽，训练入口

## 3. B 辅助路径工程预检（10 步）

step 0 dev EM=50.6 / F1=64.4，step 1 evidence_loss=0.81 / qa_loss=0.145，两个 loss 均有梯度。辅助 reader 前向路径正常。

`[evidence] active=29804/34000 overlong=3978 partial=6723`（有效目标 87.7%）。

## 4. Pilot 结果（8 臂，QA-only，seed42 × seed43，250 步）

### dev500 F1 @ step 0/125/250

| 臂 | seed42 step0 | s125 | s250 | seed43 step0 | s125 | s250 |
|---|---|---|---|---|---|---|
| A-frozen | 64.43 | 64.67 | 64.24 | 63.37 | 63.48 | 63.59 |
| A-joint | 64.43 | 64.68 | 63.31 | 63.37 | 63.64 | 62.72 |
| A-first | 64.43 | 64.34 | 64.19 | 63.37 | 63.56 | 63.31 |
| A-slotwise | 64.43 | 64.43 | 64.50 | 63.37 | 63.77 | 63.78 |

step 0 在同 seed 四臂中完全相同，初始化一致性确认。

### dev2000 full eval（last checkpoint，step250）

| 臂 | seed42 EM | F1 | seed43 EM | F1 |
|---|---|---|---|---|
| A-frozen | 49.95 | 63.70 | 49.80 | 63.30 |
| A-joint | 49.70 | 63.40 | 48.60 | 62.60 |
| A-first | 49.90 | 63.70 | 49.70 | 63.30 |
| A-slotwise | 50.15 | 63.80 | 49.90 | 63.40 |
| S0（参考） | 49.95 | 63.69 | 49.65 | 63.15 |

**pilot 结论：**
- A-joint 两 seed 均退化（seed42 −0.3 F1，seed43 −0.7 F1）；冻结 S0 策略确立，joint 放弃。
- A-frozen / A-first / A-slotwise 与 S0 持平（±0.15 F1）；250 步 QA-only 无增益，也无退化。
- 第二步跨槽 attention（full）与 slotwise 相比无额外优势。

## 5. A/B/C 正式验证结果（seed42，500 步，并行）

### 设计

| 臂 | query 进入 readout | evidence loss | 核心对比 |
|---|---|---|---|
| A | 真实 query，conditioned | 0 | QA-only 结构基线 |
| B | 真实 query，conditioned | 0.1 | B−A：内容监督是否带来 QA 增益 |
| C | 固定向量，agnostic_matched | 0.1 | B−C：辅助目标下 query 条件化是否必要 |

三臂同一 S0（seed42）fresh start，冻结基座，同数据/批次顺序（step 1 train_order_digest 三臂相同）。

### dev500 F1 @ step 0/125/250/375/500

| 臂 | step0 | step125 | step250 | step375 | step500 |
|---|---|---|---|---|---|
| A | 64.43 | 64.17 | 64.74 | 64.19 | 64.32 |
| B | 64.43 | 64.39 | 65.06 | 63.47 | 63.59 |
| C | 64.43 | 64.51 | 65.13 | 63.40 | 63.59 |

step 0 三臂完全相同，初始化一致性确认。

### dev2000 full eval（last checkpoint，step500）

| 臂 | EM | F1 | substr |
|---|---|---|---|
| S0（参考） | 49.95 | 63.69 | 54.50 |
| A（QA-only） | 49.75 | 63.70 | 54.50 |
| B（evidence，true query） | 49.70 | 63.50 | 54.45 |
| C（evidence，fixed query） | 49.75 | 63.50 | 54.55 |

**B−A = −0.20 F1，B−C = 0.00 F1。**

### 结论

内容监督（B）没有带来正常 QA 增益，相对 A 小幅下降。B 和 C 在 step 500 数字完全相同，说明真实 query 和固定 query 的条件差异在 QA 上没有可识别的贡献。

evidence_loss 确实下降（step 500 时约 0.52），模型学会了生成证据文本，但这个能力没有迁移到 QA——与此前 SupportDoc/HeadE/FiLM 的历史完全一致。

**500 步有限预算下，当前方案没有显示出值得投入全量训练的信号。**

## 6. 边界

- 这些结论适用于 500 步有限预算；不能排除更长训练下出现不同趋势。
- B 和 C 在 step250 时都有约 0.7 F1 的短暂正峰（B=65.06，C=65.13），随后回落；step250 best checkpoint 的分数（B F1=63.5%，C F1=63.5%）与 last checkpoint 一致，排除了 step250 峰值是真实收益的可能。
- 当前实验只使用 seed42；seed43 的 A/B/C 对比未运行。
