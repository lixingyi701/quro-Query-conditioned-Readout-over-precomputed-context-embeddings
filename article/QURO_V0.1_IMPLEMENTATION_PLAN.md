# QuRO v0.1 实施方案：接入冻结 PISCO/COCOM，重做 query-aware readout 与训练链路

> 起草日期：2026-09-15
> 对应设计文档：[`QURO_EXPERIMENTAL_DESIGN.md`](QURO_EXPERIMENTAL_DESIGN.md) §10 的第 1、2 步（go/no-go 关卡）
> 状态：已批准，实施中

---

## 0. Context

`/home/lxy/quro` 此前是 v0.0.0：一套自洽的「离线 latent 缓存 + 在线 query 条件读出」骨架，18 个形状/梯度契约测试有 17 个通过。但它**从未接过真实压缩器**——`src/offline.py` 只定义了一个 `encode_texts -> (B,m,h)` 的空契约，README 自己也写明「A finalized built-in PISCO/COCOM adapter」是刻意未做的。所以当时跑出来的任何数字都来自随机初始化的 prototype encoder，没有实验意义。

本轮目标：把 v0.0 的契约接到真实组件上，并真正跑出 go/no-go 结论——**在锁定生成器侧预算 B 的前提下，query 条件读出（C）相对 query 无关二次压缩（A）的差值到底有多大**。这个差值不够大，整篇论文没有故事。

### 0.1 已核实的环境事实

| 项 | 状态 |
|---|---|
| GPU | 4× A800 80GB，空闲 |
| torch 2.6.0+cu124 / transformers 4.57.6 / peft 0.18.1 / flash_attn 2.7.4 | 已装 |
| Mistral-7B-Instruct-v0.2 | 本地 `/home/lxy/selecom/baselineModel/` |
| Qwen3-Embedding-0.6B | 本地同上 |
| SeleCom stage1 (14M 条, 12GB jsonl) / stage2 (868K 条, 3.4GB jsonl) | 本地 `/home/lxy/selecom/data/` |
| TriviaQA eval（7993 行，带 documents）/ NQ dev parquet | 本地同上 |
| `naver/pisco-mistral` | 可下载，**仅 0.69GB**（只存 adapter + 首尾层），基座指向 Mistral-7B-Instruct-v0.2 |
| `naver/cocom-v1-{4,16,128}-mistral-7b` | 可下载，各 14.57GB，同一基座 |
| **`/home` 已 100% 满（剩 3.4G）** | 大文件一律写 `/data02`（767G 可用） |

### 0.2 已确认的关键参数

- PISCO-mistral：`compr_rate=16`, `doc_max_length=128` → **每篇文档 m=8 个 latent，h=4096**（Mistral 隐藏维）。
- COCOM v1 rate 4/16/128 → m = 32/8/1，**同一个 Mistral 基座**，天然构成 §7.4 的 ξ_off 扫描且无混淆变量。
- PISCO 的 latent 直接活在 Mistral 的表示空间里 → `projector` 理论上可以是 identity，且 QuRO 的 generator 就是 PISCO 的 decoder。
- SeleCom 数据实测文档长度：stage1 均值 114 词 / p90 157 词；stage2 均值 116 / p90 167。PISCO 在 128 token 处截断，**会截掉约 20–30% 的尾部**，ξ_off 的分母必须按实际喂进去的 token 数算。
- stage2 文档数分布（前 2000 行抽样）：1 篇 36%，2 篇 21%，**10 篇 43%** → 多文档全局 readout 有足量训练数据。

### 0.3 本轮已拍板的选择

1. 生成端 = Mistral-7B-Instruct-v0.2 + LoRA，**从 PISCO 的 `decoder_adapter` 热启动**（随机初始化 LoRA 作对照组）。
2. 第一轮只做 **go/no-go 小实验**（~80k stage1 + ~20k stage2，缓存 ~16GB）。
3. Q 侧 = 冻结 Qwen3-Embedding-0.6B 的**逐 token 隐状态**（保留 xattn 模式）。
4. 所有权重/缓存/输出写 `/data02`，代码留在 `/home/lxy/quro`。

---

## 1. 接入 PISCO / COCOM

v0.0 的 `src/offline.py` 契约刚好够用，不需要改协议，只需要补一个真实实现。

**新增 `src/compressors/pisco.py`**：
- 把 `modelling_pisco.py` / `modeling_cocom.py` vendored 到 `third_party/`（HF 的 `trust_remote_code` 在离线环境不稳，且要记录实验用的确切版本）。
- 唯一补丁：`config.decoder_model_name` 改指本地 Mistral 路径，避免重新下载 14GB 基座。补丁只改下载下来的 `config.json`，vendored 代码保持原样。
- 暴露工厂 `build(checkpoint, device)`，返回满足 `encode_texts(list[str]) -> (B, m, 4096)` 的对象，内部直接调 PISCO 的 `compress_documents()`（`modelling_pisco.py:1027`）。
- 同一模块再暴露 `build_generator(checkpoint, device)`，返回加载好的 `COCOM` 对象供生成端复用（见 §5）。

**`scripts/build_latent_cache.py` 基本不动**，只需 `--adapter src.compressors.pisco:build`。已有的 `LatentCacheWriter` 分片、manifest、NaN 检查、source_token_count 记账全部复用。

**建缓存时补记的字段**（`cache.py` 的 manifest 加 key，向后兼容）：`doc_max_length`、`compr_rate`、真实喂进压缩器的 token 数——否则 ξ_off 算不准。

**缓存规模核算**：250k 篇 × 8 × 4096 × 2B(fp16) = **16.4GB**。这是本轮规模的硬约束来源：stage1 全量 14M 篇要 900GB，不可行。

---

## 2. 现有 query-aware 部分的审查

### 2.1 结构溯源

沿用的是 **Perceiver IO 的 decoder 半**，代码结构忠实：`OutputQueryBuilder`(`src/perceiver.py:253`) 造 output query array → `PerceiverDecoder`(`:328`) 做单次 cross-attention(Q=output query, K/V=latent)。五种 Q 构造模式（agnostic / add / film / concat / xattn）对应设计文档 §7.1 的消融表，这部分**设计合理，予以保留**。

### 2.2 参数量实测

| 配置 | output_query | readout | projector | cache_proj | 合计 |
|---|---:|---:|---:|---:|---:|
| v0.0 的 qwen3emb 预设 (d=768) | 4.54M | 4.14M | 3.54M | 0.79M | **14.09M** |
| 直接在 PISCO 空间跑 (d=4096) | 92.35M | 117.49M | 0 | 0 | **211.03M** ❌ |
| **瓶颈维 d_r=1024（采用）** | 7.36M | 7.35M | 4.20M | 4.20M | **24.20M** ✅ |

设计文档 §2.1 定的目标是 **< 50M**。直接用 PISCO 的 4096 维会到 211M，效率论证会被削弱。**采用 4096 → 1024 瓶颈**：readout 在 1024 维做，输出再升回 4096。加上 Mistral LoRA r=16 (q/k/v/o, 32 层) 的 13.63M，全系统可训参数 ≈ **38M**。

### 2.3 必须修的六个问题

1. **readout 只有一个 cross-attn block，B 个 slot 之间没有自注意力**（`perceiver.py:336-348`）。B 个输出位彼此不知道对方选了什么 → 冗余坍缩（多个 slot 读同一条证据）。Perceiver IO 原版 decoder 确实是单次 cross-attn，但它的 output query 带结构化位置编码，我们的 slot 只有可学习先验。改成 `[cross-attn → slot self-attn → FF] × L`，`L ∈ {1,2,4}` 可配，正好是 §7.2 的消融轴。

2. **预算切片是隐式假设**（`perceiver.py:301` `base_slots = self.slots[:p]`）。B=4 时用 slot 0–3，等于假定 slot 有嵌套（matryoshka）结构，但训练时从没强制过。→ 训练时按 bucket **随机采样 B**（budget dropout），一套权重服务所有 B。否则主表每个 B 都要重训，§7.4 的 ξ_off × B 扫描直接爆炸。

3. **输出尺度未校准**（`perceiver.py:341`）。`out_norm` 是 LayerNorm，输出 scale≈1；而 PISCO 的 `compressed_embs` 是 Mistral **最后一层隐状态**（幅度大得多，且有 outlier 维），decoder LoRA 是按那个分布训的。→ 改成**残差读出**：`E = Σ_j α_j·Z_j + Δ`，α 来自 cross-attention 权重，天然继承 Z 的尺度；并把 Δ 分支初始化为近零，使**第 0 步的 QuRO ≈ PISCO 的一个注意力池化版本**。这同时解决设计文档 §8.1 点名的「Perceiver 收敛慢」。

4. **prompt 拼装绕开了 PISCO 模板**（`src/model.py:315-347`）。见 §7，必须复用。

5. **`document_source` 用检索排名做 embedding**（`model.py:240` `torch.arange(k)`）。排名是相关性的强先验，readout 可能学会「只看 rank 0」而不是学 query 条件选择。→ 保留但做成可关（`--without_document_source` 已有），另加一组「打乱 rank」对照。同时补一个文档内的 **slot-index embedding**（m 维，成本近零），当前 m 个 latent 之间完全无序。

6. **`tests/test_shapes.py:156` 的 adaptive-budget 用例报 `IndexError`**（toy 词表越界）。17/18 通过，这一个要修。

---

## 3. 训练数据

用 SeleCom 公开数据，理由在 `SELECOM_PAPER_NOTES.md:462-473`：**同样的数据、同样的 loss、同样的 query 编码器底座，唯一区别是信息路径**——这是对 SeleCom 最干净的对照。

| 阶段 | 来源 | 本轮取量 | 用途 |
|---|---|---:|---|
| Phase 1 | `data/stage1/stage1_train_data.jsonl`（单文档，字段 `question/answer/document/difficulty`） | 80k 行 | 训 readout，学「从单篇 Z 里按 query 取证据」 |
| Phase 2 | `data/stage2/stage2_train_data.jsonl` 中 `len(documents)==10` 的行 | 20k 行 | 训**全局多文档 readout** + generator LoRA |
| Eval | `data/trivia_qa/trivia_qa_eval.jsonl`（7993 行带 documents）+ 从 stage2 留出 2k 行 | — | EM/F1 |

**新增 `scripts/prepare_selecom_data.py`**，产出两个文件：
- `corpus.jsonl`：`{"doc_id", "text", "source_token_count"}`，全局去重（本轮 ~250k 篇）；
- `queries.jsonl`：QuRO 的 cache-first 格式 `{"id","query","retrieved_doc_ids","answers","difficulty"}`，`src/data.py:58` 的 `adapt_row` 已支持 `retrieved_doc_ids`，无需改协议。

**三个必须处理的坑**：
- **截断记账**：PISCO 硬编码 128 token 截断（`modelling_pisco.py:1030`）。`source_token_count` 记录**真实喂进去的 token 数**（=min(len,128)），另存 `original_token_count`，两者都进 manifest。
- **数据泄漏**：stage2 来自公开 QA 训练集，必须对 eval 集做 exact/near-dup 检查后再报数（`SELECOM_PAPER_NOTES.md:496`）。放进 `scripts/check_leakage.py`。
- **difficulty 字段**：实测分布 EASY 21% / MEDIUM 59% / HARD 20%。本轮**不做 curriculum**（SeleCom 官方脚本其实也没做，同上注 495），但字段透传，留给后续消融。

---

## 4. 损失函数与训练链路

### Phase 1（go/no-go，本轮重点）

```
L = CE(a | E, q)                            # 主损失，与 SeleCom Stage 1 完全同一目标
  + λ_warm · L_align                        # 仅前 ~200 步，λ 线性衰减到 0
```

- **主损失**：答案的 next-token CE。`src/model.py:359` 的 `qa_loss` 已经是这个，保留。
- **`L_align`（新增，解决冷启动）**：在 `B = k·m` 的满预算下，让 readout 输出还原缓存 latent，即 `‖E − Z‖²`。配合 §2.3 第 3 点的残差初始化，让训练从「QuRO ≡ PISCO」出发，之后学到的每一分都能归因到 query 条件读出。直接回应设计文档 §8.1。
- **随机预算（budget dropout）**：每个 batch 从 `budget_buckets` 采一个 B。
- **明文 query dropout**：见 §7.3，这是本轮新增的训练策略而非仅评测消融。
- 冻结：offline compressor（不加载）、query encoder；可训：output query builder、readout stack、projector、slot/source embedding。Generator 在 Phase 1 **冻结**（用 PISCO 的 decoder_adapter 原样），以隔离 readout 的贡献。

### Phase 2

解冻 generator LoRA 继续训（stage2 多文档数据）。按 `SELECOM_PAPER_NOTES.md:485-489` 报两种设置：readout 冻结 + LoRA 训（隔离 decoder 适配作用）、两者都训（联合上限）。

### Phase 3（本轮不做，留接口）

自适应预算的 budget CE。需要先跑「逐档评测取能答对的最小档」生成标签（`SELECOM_PAPER_NOTES.md:327-335`）。`QueryBudgetSelector`(`model.py:79`) 与 `budget_loss_weight` 已就位，缺的是标签生成脚本。

### 明确不做

- **文档重构损失**：压缩器冻结，重构无从谈起；且 PISCO 已证明 QA 目标优于重构。
- **额外的 teacher 蒸馏**：SeleCom stage1 的 answer 本身就是 Qwen3-30B 生成的，已经是合成 teacher。`teacher_output` 字段（`data.py:83`）保留，`scripts/make_teacher_outputs.py` 留空接口，Phase 2 再评估。

---

## 5. 生成端

**Mistral-7B-Instruct-v0.2 + LoRA(r=16, q/k/v/o)，从 `pisco-mistral` 的 `decoder_adapter` 热启动。**

做法是**直接把加载好的 PISCO `COCOM` 对象当作 generator**，而不是自己 `AutoModelForCausalLM.from_pretrained`：
- 它的 `.decoder` 就是 Mistral-7B-Instruct-v0.2 + 已训好的 `decoder_adapter`，且 embedding 层已按 `<MEM0..7>/<AE>/<ENC>/<SEP>` resize 并从 `decoder_first_last_layers.pth` 恢复；
- 它已经会读这个空间里的 soft token → QuRO 第 0 步就 ≈ PISCO，而不是从随机开始；
- PISCO 基线 = 同一个对象，把 `compressed_embs` 直接喂进去 → **同骨干、同 prompt、同 LoRA 初值，唯一差异是信息路径**，主表的公平性无懈可击。

改 `src/generator.py`，加 `build_pisco_generator(cfg)` 分支，与现有 `toy` / HF 分支并列。`GeneratorConfig` 加 `kind: "toy" | "hf" | "pisco"`。

**对照组**（`--generator_lora_init` 开关）：`pisco`（热启动，默认）/ `random`（随机 LoRA）/ `frozen`（完全不训 decoder）。

---

## 6. 小规模演示实验

### Level 0：链路 smoke（~15 分钟，1 卡）

`scripts/run_smoke.sh`：2000 行 stage1 → 建 2000 篇缓存 → 训 100 步 → 评 200 条。只验证「不崩、loss 下降、缓存维度对得上、生成的是人话」。

### Level 1：go/no-go 对照（~6–8 小时，1 卡）

`scripts/run_gonogo.sh`：80k stage1 + 20k stage2(k=10)，缓存 ~16GB，训 ~3000 步。**锁定 B=8**，跑四个 arm：

| arm | output query | 说明 |
|---|---|---|
| **A** | `agnostic` | query 无关的二次压缩（设计文档 §7.1 变体 A） |
| **C** | `xattn` | QuRO 本体 |
| **S** | similarity top-B | 非参数：按 `⟨e_q, z⟩` 取 top-8 个槽，不训 readout |
| **P** | PISCO 原生 | B = k·m，预算不匹配，作参考行 |

外加 **mismatch-query 对照**（`train.py:89` 的 `--query_control` 已实现）：把 query 换成隔壁样本的，C 应该明显掉点。

**go 判据（跑之前写死，避免事后找补）**：
1. `C − A` 在 EM 上 ≥ 3 个点，且 `C > S`；
2. C 在 mismatch-query 下掉点 ≥ 5 个点（证明真用了 query，而不是多了一层参数）；
3. C 在 B=8 时不显著劣于 P 在 B=k·m 时（说明 8 个 query 条件 token ≈ k·m 个 query 无关 token）。

三条都不过 → 回设计文档 §9 的预案，不要往主实验铺算力。

---

## 7. decoder 输入的构成：从消融升格为设计主张

### 7.1 默认模板

严格复刻 PISCO 的模板（`modelling_pisco.py:1036`），只把 mem 槽位数从 `k·m` 改成 `B`：

```
[system] You are a helpful assistant. Your task is to extract relevant information
         from provided documents and to answer to questions as briefly as possible.
[user]   Background:
         <MEM…×B><SEP>          ← 前向时替换成 QuRO 读出的 B 个 soft token
         Question: {query 明文}
[assistant]
```

即 **system prompt + 证据 embedding + query 明文**，证据在前、问题在后。不复用这个模板就丢掉热启动，而且和 PISCO 基线不同 prompt，主表直接不可比。

`src/model.py:315` 的 `_assemble` 顺序是对的但没套 chat 模板，改为新增 `src/prompt.py` 的 `PiscoPromptBuilder`：按 B 生成槽位（B > 8 时循环复用 8 个 mem token id）、套 chat 模板、返回槽位下标供 `inputs_embeds` 替换。

### 7.2 「decoder 只收 soft embedding」这件事的定位

先把两件容易混淆的事分开：

- **「decoder LoRA 学会读 soft token」——不新。** COCOM 的核心消融结论就是「训练 decoder 是关键的」，SeleCom Stage 2 整段就是 generator LoRA 学读 soft token，ICAE 同理。`SELECOM_PAPER_NOTES.md:231-239` 已把这类表述列入「不能再声明的创新」。

- **「decoder 输入端只剩 soft embedding，明文 query 被 readout 吸收」——这是另一件事，值得主张。**

关键在于**匹配动作发生在哪里**。前人的 decoder 输入永远是 `[明文 system prompt] + [文档 soft token] + [明文 query]` 的混合序列，它的 LoRA 同时扛三件活：解析真实词表 embedding、读 soft token、**把 query 和证据做对齐匹配**。第三件事是被迫的——PISCO/COCOM 的压缩器 query 无关，「文档哪部分回答了这个问题」只能留给 decoder 做。

QuRO 之后，匹配已经在 readout 里完成，decoder 只剩「把一个已选好的表示说成话」。**这个结构性转移只有 query 条件 + 可缓存的组合才做得到**：SeleCom 的 soft token 虽也是 query 条件的，但它没有这个动机，也没做。

### 7.3 训练策略：query 明文 dropout

**D1 不只是评测模式，更应该是训练模式。** 若 prompt 里没有明文 query，query 信息的**唯一通路就是 E**——loss 想降下去就必须逼 readout 真做 query 条件选择，否则无路可走。

实现为 `--query_text_dropout p`：训练时以概率 p 丢掉明文 query。这比只在评测时测 D1 强得多，因为它把「E 必须自含问题」变成训练约束而非事后检验。

对应地要记下一个**方法论副作用**：D0 保留明文 query，意味着 agnostic 基线（arm A）也能靠 decoder 自己去对齐证据，**会人为缩小 A-vs-C 的差值**。所以 §6 的 go/no-go 判据要在 **D0 和 D1 下各测一遍**，D1 下的 A-vs-C 才是 query 条件效应的上界估计。

### 7.4 四档 decoder 输入（`--decoder_input_mode`）

| 档 | decoder 输入 | 检验什么 |
|---|---|---|
| **D0** | system + Background:[E] + Question: q 明文 | 默认，主表用这个 |
| **D1** | system + Background:[E]，**删掉 query 明文** | E 是否已自含「问的是什么」 |
| **D2** | system + Question: q + Background:[E] | 顺序敏感性 |
| **D3** | 仅 [E]，无 system 无 prompt | decoder 能否纯从 embedding 读语义 |

### 7.5 让 D1 成为贡献而非脚注的三个可测后果

单说「去掉明文 query 也不掉点」只是个消融。要成为贡献，必须测出**它买到了什么**：

1. **prefill 变短**：省掉 query 的明文 token，在线成本再降一截。直接计入 §5 的效率表。
2. **decoder 适配负担变轻**——**最硬的证据**。扫 LoRA rank `r ∈ {2,4,8,16,32}`，看 QuRO 是否在更小 rank 上就饱和，而 PISCO 需要更大 rank。若 QuRO 在 r=4 追平 PISCO 的 r=16，说明活确实被搬走了，而不是换了个地方做。
3. **归因干净**：去掉明文 query 后还能答对，选择就一定发生在 readout，不可能是 decoder 代劳。

**诚实的风险**：若 D1 仅仅是「不掉点」而没买到 1/2 里的任何东西，它就退回成一个分析性消融，不应写成贡献。这一条在跑出数之前不预设结论。

---

## 8. 实施顺序与文件清单

| # | 内容 | 主要文件 |
|---|---|---|
| 1 | vendored PISCO/COCOM + 本地基座补丁 + adapter 工厂 | `third_party/modelling_pisco.py`、`src/compressors/pisco.py`、`src/paths.py` |
| 2 | 数据准备：SeleCom → corpus/queries，含去重与截断记账 | `scripts/prepare_selecom_data.py`、`scripts/check_leakage.py` |
| 3 | 建缓存（manifest 增记 compr_rate / doc_max_length / 真实 token 数） | `scripts/build_latent_cache.py`、`src/cache.py` |
| 4 | readout 重做：瓶颈维、slot self-attn 堆叠、残差读出与近零初始化、slot-index embedding | `src/perceiver.py`、`src/readout.py` |
| 5 | PISCO prompt 构造、B 槽位注入、四档 decoder 输入、query 明文 dropout | `src/prompt.py`、`src/model.py` |
| 6 | generator 接 PISCO，三种 LoRA 初始化开关 | `src/generator.py`、`config.py` |
| 7 | 训练链路：budget dropout、`L_align` warmup、Phase1/2 冻结策略 | `src/train.py`、`config.py` |
| 8 | 基线 arm：similarity top-B、PISCO 原生直喂 | `src/baselines.py` |
| 9 | 脚本与预设 | `scripts/run_smoke.sh`、`scripts/run_gonogo.sh`、`config.py` 新增 `pisco_demo` 预设 |
| 10 | 测试 | `tests/test_shapes.py`、`tests/test_pisco_contract.py` |

路径约定（环境变量，默认写进 `src/paths.py`）：

```
QURO_ROOT=/data02/quro
  ├── models/      pisco-mistral, cocom-v1-{4,16,128}-mistral-7b
  ├── data/        corpus.jsonl, queries_*.jsonl
  ├── cache/       pisco-r16/, cocom-r4/ ...
  └── runs/        gonogo_{A,C,S,P}_{D0,D1}/
```

---

## 9. 验证方式

1. **单元契约**：`python tests/test_shapes.py` 全部通过（含当前失败的 adaptive-budget）；`tests/test_pisco_contract.py` 验证 `encode_texts` 返回 `(B,8,4096)`、prompt 槽位数 == B、残差初始化下 `‖E − AttnPool(Z)‖` 接近 0。
2. **离线/在线一致性**：对同一篇文档，`compress_documents()` 直出的 latent 与从缓存读回的 latent 逐元素误差在 fp16 精度内。
3. **端到端回归**：把 QuRO 退化成 PISCO（`output_query_mode=agnostic`、`B=k·m`、readout 恒等、decoder 原样），用 PISCO 官方 `generate_from_text()` 在同一批样本上比对生成结果，**应当基本一致**。这是整条链路最强的正确性检查。
4. **Level 0 smoke**：`bash scripts/run_smoke.sh`，loss 单调下降、生成非乱码。
5. **Level 1 go/no-go**：`bash scripts/run_gonogo.sh`，产出 `runs/*/result.json`，按 §6 三条判据给出 go / no-go 结论，并把结论写回 `QURO_EXPERIMENTAL_DESIGN.md` §6。

---

## 10.5 实施过程中的实测发现（2026-09-15）

以下都是跑出来的数，不是推断。它们改变了本轮的评测集选择与数据构造。

### 10.5.1 PISCO 的压缩不跨 batch 确定

| 条件 | 结果 |
|---|---|
| 同一 batch 跑两次 | 逐位相同 |
| batch=16 vs batch=64（同一篇文档） | max\|Δ\| = 0.66，mean\|Δ\| = 0.024，cosine ≥ 0.9997 |
| batch=1 vs batch=16 | max\|Δ\| = 0.31 |

bf16 下不同 batch 形状走不同 kernel、累加顺序不同。两个后果：

- **PISCO 基线必须读同一份缓存**，不能在线重算，否则混进一个不受控的干扰变量（每份缓存都带 batch 组成的指纹）。
- 7B bf16 + 贪心解码是混沌系统，**单条样本的系统间对比是噪声**，只看聚合指标。

`scripts/check_pisco_equivalence.py` 实测 81.2% 逐字相同、substring 精度完全相等（21.88% vs 21.88%），这已是可达上限，判为通过。

### 10.5.2 截断损失 32.7%

```
mean fed tokens/doc 120.9  (untruncated 179.6,  truncation loss 32.7%)
xi_off = fed_tokens / m = 15.12x
```

PISCO model card 明写「documents of size up to 128 tokens」「cropped to about 128 tokens」，不是隐瞒；是 **SeleCom 的文档（180 token）超出了 PISCO 的设计点（~128）**，两边 passage 切分口径不同。ξ_off 必须按 fed tokens 算 15.1×，按原文长度算会虚报成 22.5×。缓存 manifest 两个数都记。

顺带一个可写进论文的观察：`n_mem_tokens = doc_max_length // compr_rate` 与实际输入长度无关，**喂多长的文档直接改变真实 ξ_off**（128 token → 16×，256 token → 32×）。前人报 rate 时不同时报 passage 长度，这个数没有意义，印证设计文档 §3 规则 4。

### 10.5.3 ⭐ SeleCom 数据的评测地板（改变了主评测集选择）

| 数据集 | yes/no 占比 | gold 中位词数 | 常量预测地板 EM |
|---|---:|---:|---:|
| SeleCom gonogo train | 42.3% | 6 | 6.60% |
| SeleCom gonogo dev | 43.2% | 6 | 6.75% |
| **TriviaQA eval** | **0.0%** | **2** | **0.35%** |

SeleCom stage1/2 是 LLM 合成问答，四成是是非题，gold 还是完整句子（"Yes, it officially ended plural marriage in Utah."）。常量输出 `"Yes"` 就能拿 EM 6.6%。

**决定**：TriviaQA 作为**主评测集**，SeleCom dev 只作 in-domain 参考。`metrics.constant_baseline()` 把这个地板计算进每一行结果（`constant_baseline_em` / `em_above_constant`），summarize 的 go 判据加一条 Gate 0：**C 必须先高于常量地板**，否则后面所有比较都没有意义。

### 10.5.4 ⭐ 80% 的训练数据没有"选择"可做（改变了数据构造）

stage1 是单文档：K=1、m=8，所以 `K·m = 8` 个 latent，而 B=8 —— **8 选 8，压缩比 1:1，readout 只是线性重组**，query 条件无处发挥。只有 stage2 的 K=10（80 → 8，10:1）才是真选择问题。而 stage1 占 80%。

对应的实测症状（smoke，150 步，qdrop=0.5）：

| | 未训练 | 150 步后 |
|---|---:|---:|
| query 敏感度 `‖E(q)−E(q′)‖/‖E(q)‖` | 0.030 | **0.019 ↓** |
| 注意力熵 / log(K·m) | 99.85% | **83.2% ↓** |

注意力变尖锐但 query 依赖性反而下降 —— 学到的是**与 query 无关的显著性模式**，即 arm C 自发塌缩成 arm A。

**受控验证**（400 步，同样的数据行，唯一差异是有无干扰文档）：

| 400 步 | query 敏感度（中位） | 注意力 TVD | 端到端 EM vs 常量地板 |
|---|---:|---:|---:|
| 无干扰（K=1，8 latent → B=8，1:1） | **0.007** | 0.006 | 14.06% vs 14.06% = **+0.00%** |
| 有干扰（K=5，40 latent → B=8，5:1） | **0.534** | 0.442 | 9.38% vs 4.69% = **+4.69%** |

76 倍差距，步数已控制。**这给出一个可证伪的预测：query 条件读出的收益应随 K·m/B 单调增长**，而 K·m/B = 1 时必然为零。主实验应当直接扫这条曲线——它比单点的 A-vs-C 差值更有说服力，因为它预言了失败点的位置。

**修正**：`prepare_selecom_data.py --distractors 4`，给单文档行掺 4 篇同语料随机文档，gold 放在**随机 rank**（不是 rank 0，否则 readout 学会"永远读第一篇"，而 `add_document_source` 的 rank embedding 会助长这一点）。`mean_docs_per_query` 2.8 → 6.0，全部训练数据变成 5:1 及以上的真实选择问题。干扰文档取自同一语料，**corpus 不变，缓存不必重建**。这同时正好对齐设计文档 §6.3 想测的「带噪 top-k」场景。评测集同样处理（`corpus_from_queries.py --distractors`），并在结果中显式标注这是 noisy top-k 设定而非原版 TriviaQA RAG。

### 10.5.5 「未训练也有效果」是误读

150 步时 D0 的 EM=9.38%，**低于**常量地板 12.73%（smoke dev）。之前把它当作"链路在学"的证据是错的。真实原因分三层：

1. 指标地板；
2. 系统里未训练的只有 readout —— decoder 是 PISCO 已训好的 adapter，D0 下还有明文 query，Mistral-7B 本身有大量参数化知识，generator LoRA 的 41.94M 也在训；
3. 初始化时 readout 按设计就是恒等操作（注意力熵 = 均匀的 99.85%），`E ≈ mean(Z)` 复制 B 份，是主题级平均而非证据选择。

### 10.5.6 两个已修的实现 bug

- **残差惩罚在初始化点梯度为 inf**：`delta.pow(2).mean().sqrt()` 配零初始化的 `out_proj`，`delta` 恰为 0，`sqrt` 在 0 处导数无穷 → 全部梯度 NaN。零初始化与 RMS 形式互相冲突。改用均方（不开根号），并加了「初始化点全部梯度有限」的断言测试。
- **LM 逃过了 train/eval 模式切换**：为了不进 `state_dict` 把 LM 存在 list 里，导致 `nn.Module` 的模式切换管不到它；PISCO 的 adapter 用 `lora_dropout=0.1`，评测时 dropout 仍在生效，贪心解码不确定。`QuROModel.train()` 现在显式驱动。

---

## 10. 本轮明确不做

- ξ_off 扫描（需另外下载 3 个 14.5GB 的 COCOM checkpoint，等 go/no-go 过了再做）。
- Token Overflow 探针（设计文档 §6.1）。
- 自适应预算的标签生成与训练。
- TTFT / GFLOPs / 摊薄曲线 q*。
- LLM-judge 指标、HotpotQA/PopQA/FactKG/ASQA 等其余数据集。
- §7.5 的 LoRA rank 扫描（go/no-go 通过后立刻做，它是 D1 主张成立与否的关键证据）。
