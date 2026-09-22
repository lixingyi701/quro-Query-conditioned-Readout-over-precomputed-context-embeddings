# SeleCom「全量压缩不可行性」在 PISCO 上的复现结果

> 对应指示：`docs/SELECOM_FULL_COMPRESSION_INFEASIBILITY_INSTRUCTION.md`
> 状态：Phase 1–2 完成，Phase 3 最小干预完成
> 代码：`src/infeasibility.py`、`scripts/diagnose_full_compression_infeasibility.py`、
> `scripts/plot_full_compression_infeasibility.py`、`tests/test_attention_grouping.py`
> 产物根目录：`$QURO_ROOT/results/full_compression_infeasibility/`
> 本轮不训练任何模型，不实现任何解决模块。

## 0. 先用数据回答指示 §16 的五句话

1. **PISCO 全量压缩记忆是否真的被 decoder 使用？**
   是，而且差距很大。正确 memory 的重建 ROUGE-L 比错配 memory 高
   **+0.489 [+0.439, +0.540]**，比无 memory 高 **+0.511 [+0.461, +0.563]**（n=40 配对
   bootstrap）。资格门槛通过，后续机制讨论成立。

2. **加入 conflict instruction 后，模型是服从指令还是被文档记忆拉回？**
   被拉回，**而且这确实是压缩特有的**。同一 decoder、同一条指令、同一个 nonce，只换文档
   表示：raw text 65% 遵循，PISCO memory **22.5%**，差
   **−0.425 [−0.575, −0.275]**。不给任何背景时是 100%。
   SeleCom Figure 2(a) 的**行为现象在 PISCO 上复现了**。

3. **这种行为首先出现在哪些层、heads 和生成步？**
   **它不出现在注意力里**，所以这个问题没有答案，而"没有答案"本身是结果。conflict 条件下
   memory 在每一层、每个生成步都**低于** instruction：density 比值 0.278，mass 比值
   0.122，value 贡献比值 0.087，pre-softmax QK logit 低 0.93 nat。只有 15.7% 的 head
   出现 memory density > instruction density。样本级上，这些量**预测不了哪一条会失败**
   （density_ratio ~ 遵循率 rho = −0.023, p = 0.89）。

4. **主因更像 token 数量、QK/logit 尺度、向量范数、位置效应，还是根本不是 attention capture？**
   **根本不是 attention capture。** SeleCom 归因链的前提为真、推论为假：
   memory 位置的残差流范数确实被极化到 **104**，是文本 token 的 **740×**，且 32 层几乎不变；
   但 Mistral 是 pre-norm 架构，**RMSNorm 在 Q/K 投影前把这个范数除掉了**——memory 的
   key 范数在第 0 层是 **7.59**，反而**小于** instruction 的 **12.84**。范数极化存在，
   放大不存在。
   同时也不是长度效应、不是位置效应、不是文档内容：
   置零槽位（同样 9 个位置）行为接近 raw（62.5%），错配 latent 的压制与正确 latent
   **完全相同**（25.0% vs 22.5%，CI 含 0）。

5. **因此下一模块具体要改变哪个量，并保留什么能力？**
   见 §7。简短版：**现在还不能提出模块**，因为指示 §15 的"进入解决方案设计的门槛"第 3、4
   条未满足——没有任何一个内部机制跨样本稳定地预测失败。已确立的是一个**否定性的方向约束**：
   任何以"降低 memory 注意力 / 校准 memory 通道"为动作的模块，都在改一个被证明与失败无关的量。

---

## 1. 做了什么

在仓库现有的 **PISCO `P` baseline**（`config.ARMS["P"] = pisco_direct`）上，不训练、不改
结构，只改变 decoder 看到的文档表示与指令，测量行为与 decoder 内部量。

| run | level | prompt 脚手架 | 规模 | 用途 |
| --- | --- | --- | --- | --- |
| `levelA-pisco-prompt` | A | PISCO system prompt + `Question:` | 40 doc × 9 条件 | 主复现（仓库实际服务配置） |
| `levelA-selecom-prompt` | A | 无 system prompt，无 `Question:` | 40 doc × 9 条件 | 脚手架对照（SeleCom 原文写法） |
| `levelB-pisco-prompt` | B | PISCO | HotpotQA dev 200 行，K=10 | 扩展到真实任务 |
| `levelB-k1-length-control` | B | PISCO | HotpotQA dev 200 行，K=1 | 长度效应对照（8 vs 80 latent） |
| `levelA-intervene-*` | A | 两种 | 40 doc × 18 条件 | Phase 3 最小可逆干预 |

Level A 文档从 **HotpotQA corpus 的缓存文档**中抽（32–96 tokenizer tokens），所以用的
latent 就是真实系统在服务的那批，而不是为演示重新压缩的玩具文档。

### 1.1 Task 0 审计：这是仓库的 PISCO prompt，不是一个长得像的 demo

每个 run 的 `manifest.json` 都记录了：

```
"D0_token_identical": true,     # 我的渲染与 PiscoPromptBuilder(mode="D0") 逐 token 相同
"D0_slots_identical": true,     # <MEM*> 槽位索引完全一致
"RG_token_identical": true      # 原文条件与 PiscoPromptBuilder(mode="RG") 逐 token 相同
```

`tests/test_attention_grouping.py` 把这三条钉成断言。模型、tokenizer、adapter
（`decoder_adapter`，`lora_init=frozen`）、cache reader 全部走 `src/model.py::build_model`，
没有第二套加载代码。

### 1.2 数值有效性

| 检查 | Level A | Level B (K=10) |
| --- | --- | --- |
| 打补丁的 eager attention vs 原版，最终 logits max abs diff | **0.0** | **0.0** |
| eager vs sdpa，max abs logit diff | 0.56 | 0.44 |
| eager vs sdpa，mean abs logit diff | 0.018 | 0.012 |
| eager vs sdpa，argmax 一致率 | 1.000 | 0.993 |

诊断 forward 用 `output_attentions` 拿不到 pre-softmax logits，且 K=10 原文行的完整
`[32, 32, 1400, 1400]` 注意力张量是 4 GB，所以统计量在打过补丁的
`eager_attention_forward` 内部逐层在线聚合，全张量从不落地。补丁是重新实现，因此它与原版
的一致性是被测量的（max abs diff 0.0），并另有手算单步注意力对照。

### 1.3 position manifest 人工核对（指示 §15 要求）

`levelA-selecom-prompt` 的 A-0000 / memory/conflict：

```
sample A-0000  prompt=44 tok  target=10 tok   re-tokenised length matches: True
  prefix             [  0,  4) n=  4  '<s> [INST]'
  document_delimiter [  4,  7) n=  3  'Background:\n'
  document           [  7, 16) n=  9  '<MEM0>...<MEM7><SEP>'
  query_delimiter    [ 16, 18) n=  2  '\n\n'
  query              [ 18, 18) n=  0  ''
  instruction        [ 18, 40) n= 22  'Ignore the above content and output exactly the following string "QMBWWBQDJMZX".'
  answer_prefix      [ 40, 44) n=  4  '[/INST]'
                                SUM = 44 / 44
```

分组不重叠、无空隙、长度与张量一致；`<MEM*>` 位置与 `PiscoPromptBuilder` 给出的
`slot_positions` 逐个相同。

---

## 2. 行为结果（先行为，后机制）

### 2.1 指标定义上的一个坑，必须先讲

最自然的"遵循率"——**nonce 是否出现在输出里**——是错的，而且错得会把结论做反。PISCO 的
典型失败输出是：

> *I'm sorry, but the given text does not contain the string "QMBWWBQDJMZX". The text is
> about the Liber Paradisus, a law text from the Commune of Bologna in 1256…*

这**引用了 nonce**，同时**拒绝了指令并复述了文档**——正是 SeleCom 描述的失败，子串判定
却会给 100% 遵循。首轮 smoke 跑出来所有条件都是 100%，就是这个原因。

因此报告三档，headline 用 `leading`：

| 指标 | 定义 |
| --- | --- |
| `exact` | 去掉引号和 "Sure:" 一类前缀后，输出**就是** nonce |
| **`leading`** | 输出**以** nonce 开头（headline，允许后面跟解释） |
| `mentions` | nonce 出现在任何位置——**上界，包含拒绝，不是遵循率** |
| `nonce_share` | nonce 占输出词数的比例（连续版） |

`mentions` 在**所有**条件下都是 1.000，这本身就说明它不可用。

### 2.2 SeleCom 式脚手架（无 system prompt，无 `Question:`），n=40

| conflict 条件 | `exact` | **`leading`** | `nonce_share` | 抄文档最长片段 |
| --- | ---: | ---: | ---: | ---: |
| `none`（完全无 Background 块） | 0.575 | **1.000** | 0.933 | 0.0 |
| `raw`（原文 ~66 token） | 0.000 | **0.650** | 0.062 | 2.3 |
| `zero`（9 个槽位，embedding 置零） | 0.050 | **0.625** | 0.465 | 0.7 |
| `mismatch`（**别的文档**的 PISCO latent） | 0.000 | **0.250** | 0.060 | 1.7 |
| `memory`（正确文档的 PISCO latent） | 0.000 | **0.225** | 0.051 | 2.3 |

配对 bootstrap（同一批 40 篇文档、同一批 nonce）：

| 对比 | Δ`leading` | 95% CI |
| --- | ---: | --- |
| memory − raw | **−0.425** | [−0.575, −0.275] |
| memory − zero | **−0.400** | [−0.551, −0.250] |
| memory − none | −0.775 | [−0.900, −0.650] |
| **mismatch − memory** | **+0.025** | **[−0.075, +0.125]** |

三条读数：

- **行为现象复现。** memory 相对 raw 掉 42.5 个点，CI 不含 0。SeleCom 的
  Figure 2(a) 在 PISCO 上是真的。
- **不是槽位、不是位置、不是序列长度。** `zero` 有同样 9 个槽位、同样的位置、同样的
  prompt，行为却与 `raw` 无异（62.5% vs 65%）。
- **与文档内容无关。** 装错文档的 latent 压制得一模一样（CI 含 0）。起作用的是
  "槽位里装着 PISCO latent 分布中的向量"这件事本身，不是它压的是哪篇文档。

### 2.3 PISCO 自己的脚手架下，这个效应被脚手架淹没

| conflict 条件 | `leading`（PISCO prompt） | `leading`（SeleCom prompt） |
| --- | ---: | ---: |
| `none` | 0.375 | 1.000 |
| `raw` | 0.050 | 0.650 |
| `zero` | 0.150 | 0.625 |
| `mismatch` | 0.000 | 0.250 |
| `memory` | 0.025 | 0.225 |

memory − raw = **−0.025 [−0.075, +0.000]**。

PISCO 的 system prompt 本身就是一条**站着的命令**：

> *You are a helpful assistant. Your task is to extract relevant information from provided
> documents and to answer to questions as briefly as possible.*

再加上把指令放进 `Question:` 槽，等于告诉一个 RAG 微调过的 decoder"这是一个关于背景的
问题"。两者合起来把所有条件压到 ≤5%，连 `none` 都只有 37.5%。

**这是一个必须写进结论的范围限定**：在 `P` baseline 的**实际服务配置**里，conflict
instruction 的失败**不是压缩特有的问题**——脚手架已经先把指令遵循压没了。压缩的边际
贡献只有在去掉脚手架后才测得出来。

### 2.4 资格门槛（指示 §4）

| 重建任务，n=40 | ROUGE-L | token F1 | 抄文档最长片段 | teacher-forced NLL |
| --- | ---: | ---: | ---: | ---: |
| `raw`（上界） | 0.916 | 0.929 | 27.4 | 0.097 |
| **`memory`** | **0.616** | **0.706** | **13.3** | **0.480** |
| `mismatch` | 0.126 | 0.158 | 1.7 | 2.071 |
| `none` | 0.104 | 0.124 | 2.0 | 1.977 |

memory − mismatch ROUGE-L **+0.489 [+0.439, +0.540]**，memory − none
**+0.511 [+0.461, +0.563]**，NLL 方向一致（−1.59 / −1.50）。
**门槛通过**：PISCO memory 确实是有效的文档载体，可以进入机制分析。

---

## 3. 机制结果：SeleCom 的解释在 PISCO 上不成立

SeleCom §3.2.2 / Appendix A.1 给的是一条三步因果链：

> (1) 全量重建目标把压缩向量 Z 推向 LLM token embedding 的"正半空间"，方向相似 → 注意力
> 分数为正；(2) 方向对齐后，训练压力把 Z 的范数 R_Z 极化到极端值，**放大注意力分数**；
> (3) 重建目标不含指令遵循约束，其它 token 分数保持低 → 注意力在各层塌缩到 Z 上，
> LLM "只看见"压缩向量。

逐步核对。

### 3.1 第 (2) 步的前提为真：范数极化确实存在

| 量 | 值 |
| --- | ---: |
| 普通 token 的 embedding 范数（Mistral，均值） | 0.172 |
| PISCO 训练的 `<MEM*>` / `<SEP>` embedding 范数 | 1.27 |
| **写进槽位的 PISCO latent 范数** | **≈ 104** |

memory 位置的残差流范数在**第 0 层就是 104**，是同层文本 token（0.14）的 **740×**，
并且**在 32 层里几乎不变**（layer 16 仍是 101.9，layer 31 是 105.3），而文本 token
从 0.14 长到 28。memory 位置在残差流里是**巨大且静止的离群点**。

### 3.2 第 (2) 步的推论为假：pre-norm 架构把这个范数除掉了

Mistral 在 Q/K 投影**之前**过 RMSNorm。残差范数因此**不进入注意力 logit**。实测：

| conflict 条件 | ‖h‖ L0 | ‖h‖ L16 | ‖k‖ L0 | ‖k‖ L8 | ‖k‖ L16 | ‖k‖ L31 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| memory · document | **104.05** | 101.93 | **7.59** | 14.88 | 17.78 | 19.73 |
| memory · instruction | 0.14 | 5.10 | **12.84** | 19.19 | 19.43 | 15.73 |
| raw · document | 0.13 | 5.41 | 13.05 | 19.08 | 19.18 | 16.11 |
| raw · instruction | 0.14 | 5.10 | 12.84 | 19.47 | 19.75 | 16.00 |

**740× 的残差范数差，到 key 上变成 7.59 vs 12.84——memory 的 key 反而更小。**
SeleCom 的"范数极化放大注意力分数"隐含假设注意力分数随 ‖Z‖ 增长，这在 pre-norm
transformer 上不成立。链条在第 (2) 步断掉。

### 3.3 第 (1) 步部分为真，但不足以翻盘

pre-softmax QK logit（全层全头均值）：

| | document | instruction | gap |
| --- | ---: | ---: | ---: |
| memory | −3.939 | −3.011 | **−0.928** |
| raw | −4.885 | −2.967 | **−1.918** |

memory 相对 raw 确实有 **+0.990 [+0.896, +1.084]** nat 的方向对齐优势——这是 SeleCom
说的"正半空间"效应的真实残留。**但绝对值仍在 instruction 之下**：decoder 给指令的
QK 分数始终高于给 memory 的。

### 3.4 第 (3) 步为假：注意力没有塌缩到 Z 上

conflict 条件下（n=40，全层全头全目标步平均）：

| conflict 条件 | mass 比值 | density 比值 | value 贡献比值 | memory-主导 head 比例 |
| --- | ---: | ---: | ---: | ---: |
| memory | 0.122 | **0.278** | **0.087** | 0.157 |
| raw | 0.332 | 0.108 | 0.185 | 0.017 |
| zero | 0.353 | 0.800 | 0.124 | 0.374 |

（比值均为 document ÷ instruction；`v_contrib` = ‖Σ_{j∈g} A_tj·v_j‖，即该组真正注入
attention 输出的量。）

- memory 的**每 token 吸引力**确实是 raw 的 **2.6×**（+0.169 [+0.153, +0.186]），
  memory-主导 head 比例是 raw 的 **9×**——SeleCom 指的那个方向上有一致的梯度。
- **但没有一个量越过 1。** memory 拿到的注意力总量是 instruction 的 0.122，每 token
  是 0.278，**真正注入残差流的量只有 instruction 的 0.087**。
- 所以"模型只看见压缩向量、忽略指令 token"是**事实错误**：decoder 给指令的注意力比给
  memory 的多一个量级，然后仍然不照做。

顺带排除掉一个替代解释（指示 §H5）：memory 的 mass 比 raw **低**（0.122 vs 0.332），
因为它只有 9 个 token 而 raw 有 66 个。所以这里**不存在**"只是 token 数量多"的长度
效应——方向正好相反。

### 3.5 注意力分配跟着任务走，不跟着表示走

同一批 PISCO memory，只换指令文字（`figures/layer_group_heatmap.png`）：

| 任务 | memory density 峰值 | instruction density 峰值 | 谁赢 |
| --- | ---: | ---: | --- |
| reconstruct（"逐字复述背景"） | 0.021（中层） | 0.004 | **document** |
| conflict（"忽略背景，输出 XXX"） | 0.004 | 0.015（中层） | **instruction** |

表示不变、分配翻转 → 注意力份额是**任务的函数**，不是压缩表示的固定属性。这是对
"across layers 塌缩到 Z"最直接的否证。

### 3.6 样本级关联为零（指示 §10）

在 `memory/conflict` 这 40 条**内部**（Spearman）：

| | rho | p |
| --- | ---: | ---: |
| density_ratio ~ 是否遵循 | −0.023 | 0.886 |
| mass_ratio ~ 是否遵循 | +0.054 | 0.739 |
| qk_gap ~ 是否遵循 | −0.054 | 0.739 |
| dominant_head_fraction ~ 是否遵循 | +0.117 | 0.473 |
| density_ratio ~ 抄文档最长片段 | +0.024 | 0.884 |

`raw/conflict` 同样全为零。唯一强相关的是 `nll ~ leading`（rho = −0.708），而那是
同义反复（nonce 的 teacher-forced NLL 当然预测 nonce 会不会被输出）。

**如果注意力优势是失败的原因，注意力越偏向 memory 的样本应该越容易失败。实测毫无关系。**

### 3.7 `zero` 对照不是一个干净的"空槽位"对照

必须说明：零向量的 RMSNorm 是退化的（0/√ε → 0），于是 key ≈ 0、QK logit ≈ 0，而正常
token 的 logit 在 −3 ~ −5，所以**置零槽位天然是 attention sink**。这也解释了为什么它的
注意力统计在两种脚手架下差异巨大（PISCO prompt 下 density 比值 9.78，SeleCom prompt 下
0.80）。

因此 `zero` 的**行为**结果（62.5% / 15.0%，两种脚手架下都远高于 memory）可以用，
它的**注意力**统计不可用作解释。

---

## 4. Level B：HotpotQA

<!-- PENDING: levelB-pisco-prompt (K=10) and levelB-k1-length-control (K=1) -->

---

## 5. Phase 3：最小可逆干预

<!-- PENDING: levelA-intervene-selecom / levelA-intervene-pisco -->

---

## 6. 结论落在指示 §12 决策树的哪一格

<!-- PENDING -->

---

## 7. 下一步

<!-- PENDING -->

---

## 8. 复现

```bash
# Phase 1–2（四个 run，各占一张卡）
bash scripts/run_full_compression_infeasibility.sh 40 200
CUDA_VISIBLE_DEVICES=3 python -u scripts/diagnose_full_compression_infeasibility.py \
    --level B --rows 200 --max_docs 1 --run_id levelB-k1-length-control

# Phase 3
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_full_compression_infeasibility.py \
    --level A-intervene --rows 40 --prompt_style selecom_literal --system_prompt none \
    --run_id levelA-intervene-selecom

# 出图（只读已保存的统计，不重新跑模型）
python scripts/plot_full_compression_infeasibility.py <run_dir>

# 加了新统计量之后，从 examples.jsonl 重算汇总，不必重跑 GPU
python scripts/diagnose_full_compression_infeasibility.py --summarize_only <run_dir>

# 契约测试
python tests/test_attention_grouping.py
```

每个 run 目录包含 `manifest.json`（git commit、环境、模型/cache 标识、prompt 模板、
抽样 seed、每样本 latent 数、生成参数、wall clock）、`config.yaml`、`examples.jsonl`
（逐例 prompt、生成、指标、group spans）、`grouped_attention_stats.npz`、
`norm_logit_stats.npz`、`full_attention_examples/`、`figures/`。已有 run 从不覆盖。

## 9. 与 SeleCom 原文口径的关系

SeleCom 正文与附录**没有说明** Figure 2 的注意力在哪些层、哪些 head、哪些 target row 上
聚合。因此本报告是 **conceptual reproduction**：采用指示 §8 的预注册定义
（mass 与 density 同报、按 example 做 bootstrap、target 行为 teacher-forced 目标 token），
而不是反推一个更好看的聚合口径。这一点也写进了每个 `manifest.json` 的
`selecom_reference` 字段。

两条 prompt 直接取自 SeleCom §3.2.1 原文：

- Reconstruction：`Verbatim the background again (including title and content).`
- Conflict：`Ignore the above content and output exactly the following string "…".`

nonce 改为**每例随机生成**（12 个大写字母，经 tokenizer round-trip 校验），而不是全集固定
一个串——固定串会让"模型输出它"混入对该串的记忆，也会让 exact-match 率变成单次抽样的性质。
