# 交接状态（截至 2026-09-16）

新会话从这里开始。先读本文，再按需展开。

> **⚠ 本文第 1、2、4 节已被 [`warning_and_target.md`](warning_and_target.md) 推翻或限定，先读那篇。**
> 关键一条已经过实证审计（2026-09-16，遍历 `/data02/quro/runs/*/config.json`）：
> **磁盘上每一个 A 臂都带 `cosine_prior: True`**（`d1_A_full`、`m32chunk_A`、
> `m32cocom_A`、`v2_A_agnostic`；更早的 `gonogo_A_D0` 时期配置里还没有这个字段）。
> 余弦先验通过 `cosine_bias()` 给第一层注意力加偏置，是 query 进入 readout 的
> **第二条独立通路**，`--output_query_mode agnostic` 关不掉它。
> 所以那些 A 全是 **A1（余弦条件化）**，从来没有跑过真正 query 无关的 A0，
> 下面第 1 节"比 query 无关版高 20.85 个 EM 点"这句话没有对照组支撑。
> 新契约测试给出的量级：随机初始化下换 query，A1 输出变化 max|diff|=5.50，
> 而不带先验的 C0 只有 0.203——被当作对照组的那一臂反而更依赖 query。
>
> 新臂定义见 `config.ARMS`，用 `scripts/run_arms.sh` 启动；`scripts/run_gonogo.sh`
> 保持原样以便复现历史运行，但已在脚本头部标注其 A 臂实为 A1。

---

## 1. 三十秒版本

QuRO = **离线 query 无关压缩（复用冻结 PISCO/COCOM）+ 在线 query 条件读出 → B 个 soft token → 冻结 Mistral + LoRA**。

**已证实**：readout 确实能做 query 条件选择——D1 下（decoder 拿不到问题明文）比 query 无关版高 **20.85 个 EM 点，p=1.8e-96**。

**未证实**：标准设定（D0，问题明文在 prompt 里）下这个能力是**冗余的**——7B decoder 自己就完成了证据匹配。C ≈ A ≈ S，两个独立的栈一致。

**所以当前是定位问题，不是可行性问题。**

详见 [`QURO_V0.2_RESULTS_AND_ANALYSIS.md`](QURO_V0.2_RESULTS_AND_ANALYSIS.md)。

---

## 2. 主要数字（TriviaQA 全量 2000 条，B=8，D0）

| arm | 含义 | soft token | ξ_eff | substring | EM |
|---|---|---:|---:|---:|---:|
| P | PISCO 原样全喂，不二次压缩 | 43.8 | 15.99× | 77.20% | 70.75% |
| S | 余弦 top-B，**0 可训参数** | 8 | 87.65× | 76.10% | **71.70%** |
| C | ours（query 条件读出） | 8 | 87.65× | 76.00% | 70.90% |

**注意 P 与 C 不在同一压缩率上**（16× vs 88×）。正确表述：QuRO 在 88× 下 EM 与 PISCO 在 16× 下持平。SeleCom 默认 82×，与我们同工作点。

D1 评测（问题明文删除，soft token 是唯一通道）：

| | EM | substring | 地板（错配文档） |
|---|---:|---:|---:|
| C | **26.75%** | 29.35% | 0.35% |
| A（不看 query） | 5.90% | 12.65% | 0.15% |

---

## 3. 长期约束（不要违反，理由已论证）

1. **v0.3 之前不微调压缩器**（§9）。会把通用缓存绑死到某个 readout，削弱本方法唯一的结构性优势。
2. **不要现在改 readout 架构**。三个差异极大的 readout 落在 0.5 分以内，改架构是在优化实测为平坦的维度。
3. **报结论的纪律**（§4，都是踩过的坑）：
   - 检验先于结论——n=500 时 C 领先 1.6 分，n=2000 时 99 胜 101 负
   - EM / substring / F1 三个一起报，方向不一致就报「无可靠差异」
   - 任何精度数字旁边必须有**无证据下界**（错配文档或闭卷）
   - 跨输出格式的基线不能用 EM（差的是啰嗦程度不是正确率）

---

## 4. 下一步（按诊断力排序）

> 本节排序已作废，按 `warning_and_target.md` §8 的阶段 1→5 执行：先修对照定义、
> query 表示与成本计量，再做收益验证。下表保留仅为记录当时的判断。
> 第 1 项的前提是错的：`--no_residual_readout` 去掉的是**池化旁路**、留下 Δ，
> 且同时改掉了 `out_proj` 的零初始化，做不了它声称的"Δ 消融"。
> 现已拆成 `--readout_output_mode {full,pool_only,delta_only}` 与
> `--out_proj_init {zeros,default}` 两个正交开关，且 `readout_cached(output_mode=...)`
> 支持同一 checkpoint 上的推理期分支干预。

| # | 实验 | 成本 | 决定什么 |
|---|---|---|---|
| 1 | ~~**Δ 消融**（`--no_residual_readout`）~~ 见上方更正 | 一轮训练 ~40min | ~~D1 那 20.85 分来自 α（注意力）还是 Δ（自由分支）？~~ |
| 2 | **容量上界**（逐样本优化 soft prompt，不训练任何模块） | ~1h | 8×4096 装不装得下？分离「容量」与「可计算性」 |
| 3 | **COCOM-128 基线**（同 ~82× 的一段式压缩） | 下载 14.5G + 2GB 缓存 | 「两段式结构」是不是真贡献（§8.2 的可证伪假说） |
| 4 | **B 扫描** `budget_buckets=[1,2,4,8,16]` | 一轮训练给五个点 | 预言：**C−A 随 B 减小而增大**（§7.5）。不增大则「冗余论」被推翻 |
| 5 | **换数据集**：NQ（本地有原始格式，需转换）、**HotpotQA 多跳** | 转换纯 CPU | TriviaQA 证据空间仅 12.2 分，且单事实问答对多槽位架构最不利 |
| 6 | **摊薄曲线**（TTFT / GFLOPs / q\*） | — | 目前唯一别人结构上做不出来的图，也是论文真正的卖点 |

**新颖性**：与 RRK(2604.26483) 的切割成立——它输出标量做重排，我们输出喂给生成器的 soft token；RRK 笔记 §9.2 自承未把 query 条件读出作为通用生成问题隔离研究。RRK 的实现仍值得借鉴（尤其「架构可分离 ≠ 训练独立」这一概念）。

---

## 5. 现成资产

### 缓存（`/data02/quro/cache/`，均为 memmap，可直接用）

| 目录 | m | 文档数 | 大小 | 说明 |
|---|---:|---:|---:|---|
| `gonogo-pisco-r16` | 8 | 263,890 | 17 GB | **PISCO 原生，最快，默认用这个** |
| `chunk4-part0` | 32 | 263,890 | 69 GB | PISCO 切块（128→4×32），ξ_off=3.8 |
| `cocom4-part0` | 32 | 263,890 | 69 GB | COCOM rate 4，**需配 `--generator_path` + `--generator_n_mem 32`** |

> `*-part1` 只是构建时的分片来源，已合并进 `part0`，不要单独使用。

### 数据（`/data02/quro/data/`）

- `gonogo/{corpus,train,dev}.jsonl` — 98k 训练 / 2k dev，含 4 篇随机干扰
- `trivia/queries.jsonl` — 2000 条评测，含干扰，**主评测集**
- `all_corpus.jsonl` — 上面两者的语料合并，建缓存用

### checkpoint（`/data02/quro/runs/*/checkpoint_last.pt`）

`m32chunk_C`、`m32cocom_C`、`d1_C_full`、`d1_A_full`

### 磁盘

`/data02` 449G 可用。**`/home` 是共享盘、会被别人占满**，大文件一律写 `/data02`。

---

## 6. 常用命令

```bash
# 训练 + 评测（一臂）
CUDA_VISIBLE_DEVICES=0 python -m src.train --preset pisco_gonogo \
  --steps 3000 --num_workers 4 --budget 8 --budget_buckets 4,8 \
  --eval_budgets 8 --eval_input_modes D0,D1 --doc_control --eval_max_samples 2000 \
  --eval_files trivia=/data02/quro/data/trivia/queries.jsonl \
  --tag NAME --out_dir /data02/quro/runs/NAME

# arm 开关
--prior_mode rank                      # C（ours，默认）
--output_query_mode agnostic           # A（不看 query 的对照）
--readout similarity_topb              # S（0 参数余弦）
--readout pisco_direct                 # P（不二次压缩）
--query_text_dropout 1.0               # 训练时永远不给问题明文

# 契约测试（纯 CPU，无下载）
python tests/test_shapes.py            # 31 passed

# 诊断脚本
python scripts/check_query_sensitivity.py --checkpoint ...   # readout 是否响应 query
python scripts/check_attention_targeting.py --checkpoint ... # 注意力有没有落在 gold 文档
python scripts/check_readout_stats.py --checkpoint ...       # 输出分布、有效支撑、槽间冗余
python scripts/evidence_subset.py --control ...              # 只在真正依赖证据的题上重算
```

**`--num_workers 4` 必须带**：不带的话缓存读取阻塞训练线程，m=32 下 2.88 s/step vs 0.73 s/step。

---

## 7. 踩过的坑（别再踩）

| 坑 | 症状 | 出处 |
|---|---|---|
| 缓存合并索引错位 | 一半文档返回别人的 latent；形状对、无 NaN、读回校验全过 | §4.5，已修 + 加了分片级往返校验 |
| COCOM mem token 前置 | 因果掩码下所有文档压出同一常数向量 | §4.6，`probe_mem_side()` 实测判定 |
| `num_workers=0` | GPU 掉到 0%，慢 4 倍 | §10.7.4 |
| 残差惩罚在初始化点梯度 inf | 全部梯度 NaN | §10.5.7 |
| LM 逃过 train/eval 模式切换 | LoRA dropout 在评测时仍生效 | §10.5.7 |
| RG prompt 整体截断 | 问题被静默切掉，跑出低于闭卷的 0.8% | §4 |
| HF 下载慢 | 10 MB/min → 装 `hf_transfer` 后 1913 MB/min | §10.7.4 |

---

## 8. 目录约定

```
article/   别人论文的阅读笔记
docs/      我们自己的设计、方案、实验、进展   ← 本文在这里
results/   原始实验数据（json）
src/ scripts/ tests/   代码
/data02/quro/          模型、缓存、运行输出（不进 git）
```
