# QuRO 实验设计方案

> **QuRO** = Query-conditioned Readout over precomputed context embeddings(占位名,可替换)
> 一句话:**离线做 query 无关的宽松全量压缩(Perceiver IO 的 encoder 半),在线用 query 作为 output query array 做一次轻量 cross-attention 读出(decoder 半),把激进的压缩推迟到 query 已知之后。**
> 起草日期:2026-09-08

---

## 0. 方法形式化(仅为让实验设计可读)

**离线阶段(每篇文档一次,全库摊薄):**

$$\mathbf{Z}_d = f_{\text{off}}(d) \in \mathbb{R}^{m \times h}, \quad \xi_{\text{off}} = |d| / m$$

$f_{\text{off}}$ 为**冻结**的已有压缩器(PISCO / COCOM checkpoint)。$\mathbf{Z}_d$ 落盘。

**在线阶段(每 query 一次):**

给定 query $q$ 与召回的 $k$ 篇文档,拼接得 latent 集合 $\mathbf{Z} = [\mathbf{Z}_{d_1}; \cdots; \mathbf{Z}_{d_k}] \in \mathbb{R}^{km \times h}$。

构造 output query array:

$$\mathbf{Q} = \big[\, \mathbf{e}_q \oplus \mathbf{P} \,\big] \in \mathbb{R}^{B \times h}$$

其中 $\mathbf{e}_q$ 为 query 的编码,$\mathbf{P}$ 为 $B$ 个可学习的 budget query。读出:

$$\mathbf{E} = \text{CrossAttn}(\mathbf{Q}, \mathbf{Z}, \mathbf{Z}) \in \mathbb{R}^{B \times h}$$

**代价 $O(B \cdot km \cdot h)$,与原文长度 $\sum|d_i|$ 无关**——这是 Perceiver IO 的核心性质,也是本方法效率论证的全部来源。

生成:$a \sim g_\phi(\cdot \mid \text{prompt} \oplus \mathbf{E} \oplus q)$。

**三个必须区分的压缩率:**

| 符号 | 含义 | 谁付这个成本 |
|---|---|---|
| $\xi_{\text{off}} = \lvert d\rvert / m$ | 离线压缩率 | 存储;每文档一次 |
| $B$ | 送进生成器的 token 数 | 生成器 prefill;每 query 一次 |
| $\xi_{\text{eff}} = \big(\sum_i \lvert d_i\rvert\big) / B$ | **生成器侧有效压缩率** | **对比基线时必须锁这个** |

---

## 1. 研究问题

| RQ | 问题 | 对应实验 |
|---|---|---|
| **RQ1** | 在锁定 $\xi_{\text{eff}}$ 的前提下,"宽松离线压缩 + query 读出"能否优于"激进离线压缩 + 直接喂"? | §4 主实验 |
| **RQ2** | stage 1 在多大的 $\xi_{\text{off}}$ 下**尚未发生 overflow**?即"迟延压缩"的理论前提是否成立? | §6 分水岭实验 ⭐ |
| **RQ3** | 相比 query-conditioned 但不可缓存的 SeleCom,在**摊薄成本**上的交叉点 $q^*$ 在哪? | §5 效率 |
| **RQ4** | readout 的收益来自 query 条件选择,还是仅仅来自"多了一层可训参数"? | §7 消融(核心) |
| **RQ5** | 读出预算 $B$ 由 query 决定(而非由文档密度决定)能否带来额外收益? | §7.5 |
| **RQ6** | 同一个 readout 能否跨压缩器复用(PISCO / COCOM / 自训)? | §7.7 泛化 |

**RQ2 是整个方案的分水岭。** 若 stage 1 在 4× 就已大面积 overflow,则"信息已被过早丢弃",本方法的立论崩塌,需走 §9 预案。**建议先做 RQ2 的小规模探针实验再投入主实验算力。**

---

## 2. 实验设置

### 2.1 骨干与压缩器

| 组件 | 选择 | 理由 |
|---|---|---|
| 生成器 | **Mistral-7B-Instruct** 主 + **Llama-3.1-8B-Instruct** 副 | Mistral 对齐 SeleCom Table 1,数字可直接横向比 |
| 离线压缩器 $f_{\text{off}}$ | **冻结** PISCO / COCOM 公开 checkpoint | 复用 Naver 已发布权重与离线索引;方法变成可插拔模块,审稿人无法质疑"换压缩器所以赢" |
| readout | 1–2 层 cross-attention + MLP(Perceiver IO decoder) | 参数量目标 < 50M |
| 生成器适配 | LoRA | COCOM 明确结论:**训练解码器对效果是关键的**,这部分不能省 |

> ⚠️ 待确认:Naver 的 PISCO checkpoint 具体 repo 名(COCOM 已确认为 `naver/cocom-v1-128-mistral-7b`)。RRK 论文的代码在 `github.com/naver/bergen`,应能查到 PISCO 权重名。

### 2.2 检索

- 语料:Wikipedia(KILT dump),对齐主流设定
- 检索器:**SPLADE-v3** + **DeBERTa-v3** top-50 重排 —— 完全对齐 COCOM/PISCO,消除检索差异这个混淆变量
- $k \in \{1, 5, 10, 20, 30\}$ —— $k{=}1,5$ 对齐 SeleCom,$k$ 到 30 对齐 DEX-Comp

### 2.3 数据集

| 数据集 | 类型 | 为什么要它 |
|---|---|---|
| NQ | 单跳开放域 | 对齐 SeleCom/COCOM,主表锚点 |
| TriviaQA | 单跳 | 同上 |
| WebQuestions | 单跳 | 同上 |
| **HotpotQA** | **多跳** | **跨文档融合是本方法的主战场** |
| **PopQA** | **长尾** | ArcAligner 在长尾上强,必须正面对比 |
| FactKG | 事实核查 | 对齐 SeleCom 第六个任务 |
| ASQA | 长答案生成 | COCOM 在 $\xi{=}4$ 时与未压缩无显著差异的数据集,检验长输出下是否仍成立 |

指标:**EM / F1 / LLM-judge**(对齐 SeleCom);ASQA 另报 ROUGE-L + STR-EM。

> 若时间允许,追加在 **BenchPress**(`github.com/lil-lab/benchpress`)上跑一轮——目前唯一的标准化上下文压缩评测套件,能显著提升可信度。

### 2.4 训练

**Stage A(主):PISCO 式序列级知识蒸馏**
- teacher = 看完整未压缩文档的同骨干 LLM
- student = 冻结压缩器 + readout + 生成器 LoRA
- 损失 = teacher 输出序列上的 CE
- 数据 = 从文档合成的开放式问题(**无需标注 QA**,继承 PISCO 的低门槛优势)
- 训练时 $k{=}5$(对齐 PISCO)

**Stage B(可选,冲高):DEX-Comp 式硬样本 RL**
- 仅在**未压缩 RAG 答错**的 query 上做 RL
- 动机叠加:readout 本就该学"丢什么",而稀疏的正确性信号正好告诉它丢错了什么
- 这一格(query-conditioned + 硬样本 RL)据检索为空;若 Stage A 已达标,Stage B 可作为独立贡献点

**算力参照**:PISCO 全量微调 7–10B 在单卡 A100 上 48h。QuRO 只训 readout(小)+ LoRA,压缩器冻结且 latent 可预先算好缓存,**单卡 A100 预计 24h 内可完成一轮 Stage A**。

---

## 3. ⭐ 压缩率对齐协议(公平性的关键)

**这是最容易被审稿人攻击、也最容易自欺欺人的地方。写死协议再跑实验。**

**规则 1:主表按 $\xi_{\text{eff}}$ 对齐,不按 $\xi_{\text{off}}$ 对齐。**

基线 PISCO(rate 16,文档 ~128 token)在 $k{=}5$ 时生成器看到 $5 \times 8 = 40$ 个 token。那么 QuRO 必须设 $B = 40$。**生成器侧看到的 token 数完全相同**,差异只来自"这 40 个 token 是怎么来的"。

**规则 2:存储成本必须单独、显著地报出来。**

QuRO 的 $\xi_{\text{off}}$ 更松 ⇒ 每文档存的向量更多 ⇒ 存储比 PISCO 大。报:

$$\text{Storage} = N_{\text{docs}} \times m \times h \times \text{bytes} \quad (\text{GB / 百万文档})$$

并给**存储–精度权衡曲线**($\xi_{\text{off}}$ 扫描)。**主动把这个劣势摆到台面上,比被审稿人挖出来强得多**,而且它正好衬托 §5 的摊薄论证。

**规则 3:所有基线用公开 checkpoint 在推荐设置下复跑**(SeleCom 的做法),不抄原文数字。若某基线无公开权重,标注为"引自原文,设定不完全可比"。

**规则 4:压缩率定义随论文而变,必须换算后再比。** COCOM 的 $\xi$ 是真实比率;PISCO 的 rate 隐含"文档 ~128 token"前提;xRAG 的 164× 是按平均长度折算。主表统一换算成"生成器侧 token 数"这一绝对量,原始 rate 放脚注。

---

## 4. 主实验

**表 1 结构**(每个数据集 × $k \in \{1,5\}$,报 EM / F1 / LLM-judge):

| 组 | 方法 | 生成器侧 token 数 | 可离线缓存 |
|---|---|---|---|
| 下界 | LLM w/o RAG | 0 | — |
| 上界 | 未压缩 RAG | $\sum\lvert d_i\rvert$ | 否 |
| **强上界** | **未压缩 RAG\*(同数据微调)** | $\sum\lvert d_i\rvert$ | 否 |
| Hard | LLMLingua-2 | 匹配 | 否 |
| Soft / query-agnostic | ICAE / xRAG / COCOM / PISCO / DEX-Comp | 匹配 | **是** |
| Soft / query-conditioned | SeleCom(+ ATACompressor、LooComp 若有权重) | 匹配 | 否 |
| **在线精化** | **ArcAligner** | 匹配 | 是 |
| 非参数消融 | 按 $\langle \mathbf{e}_q, \mathbf{z}\rangle$ 取 top-$B$ 个槽 | 匹配 | 是 |
| | **QuRO(ours)** | 匹配 | **是** |

**必须打赢的两个对手,性质不同:**
- **ArcAligner** —— 同样是"离线压缩 + 在线轻量模块",但它做的是**对齐/利用**而非 query 条件**选择**。赢它 = 证明"query 条件"这件事本身有价值。**这是方法论层面最关键的一场。**
- **SeleCom** —— query-conditioned 的质量上限。若质量打平即可(它不可缓存),效率论证会替你赢下全场;若质量被它显著压制,说明"迟延压缩"确有信息损失,需回看 RQ2。

**非参数 top-$B$ 选槽这一行非常重要**:它是"不训练的 query 条件选择"。若 QuRO 赢不过它,说明 cross-attention 没学到东西,只是在做相似度检索。

---

## 5. 效率与摊薄实验(本方法的主场)

### 5.1 常规效率(对齐 SeleCom 三件套)
**TTFT**、**TIL(总推理延迟)**、**GFLOPs**,在 $k \in \{1,5,10,20,30\}$ 上扫。

预期形状:QuRO 的在线代价是 $O(B \cdot km)$,随 $k$ 线性且系数极小;SeleCom 需对每篇召回文档跑一遍 selector,随 $k$ 线性但系数是一个完整 decoder-only 前向。**$k$ 越大差距越大。**

### 5.2 ⭐ 摊薄曲线(核心图,别人做不出来的图)

横轴:每篇文档被命中的平均 query 数 $\bar{q}$
纵轴:总计算成本(离线 + 在线 × $\bar{q}$)

$$C_{\text{QuRO}}(\bar q) = C_{\text{offline}} + \bar q \cdot C_{\text{readout}}, \qquad C_{\text{SeleCom}}(\bar q) = \bar q \cdot C_{\text{selector}}$$

求出**交叉点 $q^*$**。因为 $C_{\text{readout}} \lll C_{\text{selector}}$,预期 $q^*$ 很小(个位数),即**只要每篇文档被查过几次,QuRO 就全面胜出**。真实 RAG 系统里 $\bar q$ 远大于此。

同图叠加存储成本轴(双 y 轴或附表),完整呈现"用存储换在线计算"这笔账。

### 5.3 索引更新成本
语料变动时的增量重压缩成本(DEX-Comp 提过这一点)。QuRO 与所有 query-agnostic 方法同级,SeleCom 无此成本但也无此收益——如实说明。

---

## 6. ⭐ 分水岭实验:stage 1 到底丢没丢东西

**这一节决定论文能不能立住,建议最先做。**

### 6.1 Overflow 探针(直接检验 RQ2)

复现 Token Overflow(EACL 2026 SRW)的探针:在 query + context 表示上训轻量分类器,判断"压缩表示是否已不足以回答该 query"(原文在 HotpotQA/SQuADv2/TriviaQA 上平均 **0.72 AUC-ROC**;并已证明**纯 query-agnostic 的饱和度统计量测不出 overflow**——所以探针必须是 query 条件的)。

**扫 $\xi_{\text{off}} \in \{1, 2, 4, 8, 16, 32, 128\}$,画 overflow 率曲线,找拐点。**

- 若拐点在 16–32 之间 ⇒ 选 $\xi_{\text{off}} = 4$~$8$ 有充分余量,**立论成立**,且这条曲线本身就是论文里最有说服力的一张图。
- 若 4× 就已大面积 overflow ⇒ 走 §9 预案。

### 6.2 Oracle 分解(把失败归因讲清楚)

设三个条件:

| 条件 | 给 readout 的输入 | 测什么 |
|---|---|---|
| **Oracle-选择** | 直接指定 gold 证据所在的槽 | readout 的选择能力天花板 |
| **Oracle-容量** | $B$ 放大到 $km$(不做缩减) | stage 1 的信息容量天花板 |
| 实际 | 正常 | — |

- 实际 ≪ Oracle-选择 ⇒ **选错了**,该改 readout / 加 RL(Stage B)
- Oracle-选择 ≈ 实际 但 ≪ Oracle-容量 ⇒ **$B$ 不够**,该做自适应预算
- Oracle-容量 也上不去 ⇒ **stage 1 真的丢了**,回 §9

这套分解能让"为什么没赢"变成可回答的问题,审稿人非常吃这一套。

### 6.3 注意力可视化
readout 的 cross-attention 权重落在哪些槽上 vs gold 证据所在槽。**这是软压缩领域罕见的可解释性证据**——DEX-Comp 明确把"软压缩的不透明性"列为 future work,你可以顺手把这块补上,作为附加贡献。

---

## 7. 消融

### 7.1 output query array 的构成(**最核心的消融**,直接回答 RQ4)

| 变体 | $\mathbf{Q}$ | 说明 |
|---|---|---|
| A | 仅可学习 $\mathbf{P}$ | **退化为 query-agnostic 二次压缩** —— 若 A 就已达到完整版效果,则 query 条件毫无价值,论文核心主张被证伪 |
| B | 仅 $\mathbf{e}_q$ | 输出长度被 query 长度绑死 |
| C | $\mathbf{e}_q \oplus \mathbf{P}$(完整) | 默认 |
| D | C + 文档级位置/来源编码 | 检验跨文档区分度 |

**A vs C 的差值就是"query 条件"这个卖点的实际价值。这个数必须足够大,否则整篇文章没有故事。**

### 7.2 readout 深度
1 / 2 / 4 层 cross-attention。Perceiver IO 的卖点是浅而廉价;若必须堆到 4 层才有效,效率论证会被削弱,需重新权衡。

### 7.3 挂载位置
(i) 生成器输入前(默认,接口干净、可组合) vs (ii) 插入生成器中间层。
若试 (ii),用 QCFuse 的先验:**证据定位峰值在中间层而非末层**,优先试中层。

### 7.4 $\xi_{\text{off}}$ 扫描
$\{1, 4, 8, 16, 32, 128\}$ × 固定 $\xi_{\text{eff}}$。**这条曲线是全文的主论点曲线**:预期在 4–8 附近取得精度/存储的最佳权衡,与 §6.1 的 overflow 拐点相互印证。

### 7.5 自适应预算 $B$(RQ5)
| 变体 | $B$ 的决定方式 |
|---|---|
| 固定 | 常数 |
| 密度驱动 | 按文档信息密度(= Density-aware 2603.25926 的做法) |
| **query 驱动** | 按 query 难度 / overflow 探针风险打分 ⇒ **量化到离散档** |

⚠️ Density-aware 那篇的关键教训:**完全动态的连续比率反而不如静态**——模型处理不了输入相关的连续结构超参。**务必量化到离散档位。** 另注意其收益峰值在平均比率 4–16,与你的 $\xi_{\text{off}}$ 目标区间吻合。

"query 驱动的预算分配"是 Density-aware / DAST / COMI 都没做的轴(它们全是密度驱动、query 无关),**若成立,这是可独立成段的贡献**。

### 7.6 冻结 vs 微调 stage 1
默认冻结(可插拔性 + 复用索引)。做一组解冻对照,量化"可插拔"的代价。注意 COCOM 的反向结论(训解码器很关键)只针对解码器,不与冻结压缩器矛盾——RRK 已证明冻结 PISCO 压缩器 + 微调 decoder 可行。

### 7.7 跨压缩器泛化(RQ6)
同一份 readout 权重接:PISCO / COCOM / 自训 mean-pooling 压缩器。
若能零样本或轻量适配后迁移,**"通用 readout 插件"这个定位的说服力会强很多**。

> 附带:若需自训 stage 1,直接用 **mean-pooling** 而非 compression token —— 已有**两篇独立工作**(Cornell 2510.20797、Density-aware 2603.25926)确认 mean-pooling 显著更优。别再默认 ICAE 那套架构。

---

## 8. 已知工程坑(Perceiver 一脉的共同问题)

1. **收敛慢** —— cross-attention 进 latent 缺乏直接监督/对齐信号。缓解:先用重构类辅助损失预热 readout,再上 SKD 主损失。
2. **放大上游 encoder 收益有限** —— 与 LCLM "scale decoder 比 scale encoder 重要"是同一现象的两次独立观察。**别把预算砸在压缩器上**,砸在生成器适配和 readout 训练上。
3. **因果掩码与位置编码** —— 若 readout 用自回归实现需加因果掩码,通常配 RoPE。QuRO 的 readout 是并行的(非自回归),这一点上比 SeleCom 的自回归 selector 天然更快,**值得在论文里明说**。
4. **latent 的文档归属** —— $km$ 个槽拼在一起后,readout 需要能区分来源。7.1 的变体 D 就是为此准备的。

---

## 9. 风险与预案

| 风险 | 触发条件 | 预案 |
|---|---|---|
| **stage 1 已 overflow** | §6.1 显示 4× 就大面积溢出 | ① $\xi_{\text{off}}$ 降到 2× 甚至 1×(退化为"KV/隐状态缓存 + 读出",接近 QCFuse 但在 embedding 域);② 离线存**多视角** latent(不同 $\xi_{\text{off}}$ 或不同语义子空间),在线按 query 路由 |
| **质量打不过 SeleCom** | 主表落后 | 论文重心移到 §5 摊薄 + 存储论证:**"在真实 serving 负载下,可缓存性是比单点质量更重要的指标"**。配合 $q^*$ 交叉点图,这依然是完整的故事 |
| **A vs C 差值太小** | §7.1 显示 query 条件无价值 | 严重。说明 readout 只是在做二次压缩。检查 $\mathbf{e}_q$ 的注入方式、readout 容量、以及训练信号是否真的需要 query 区分(SKD 的 teacher 是 query 相关的,理论上应该有信号) |
| **被 Naver 抢发** | RRK 的生成侧版本出现 | 抢时间。RRK 已把 reranking 侧做完,同一批人做生成侧是自然延伸。**建议优先跑通 §6 + §4 的 NQ/HotpotQA 两个数据集,先挂 arXiv 占位** |

---

## 10. 建议执行顺序

```
第 1 步(1–2 周)  §6.1 overflow 探针 + ξ_off 扫描   ← 立论验证,不过关就别往下走
第 2 步(1 周)    §7.1 的 A vs C 小规模对照         ← 卖点验证
第 3 步(2–3 周)  Stage A 训练 + §4 主实验(NQ/HotpotQA 先行)
第 4 步(1 周)    §5 效率与摊薄曲线                 ← 论文的招牌图
第 5 步(2 周)    补齐数据集 + §7 完整消融 + §6.2 oracle 分解
第 6 步(可选)    Stage B 硬样本 RL
```

**第 1、2 步是 go/no-go 关卡,总共约 3 周,算力消耗很小。强烈建议严格按此顺序,不要先铺主实验。**

---

## 11. 可信度说明

- 各基线的架构与定性结论来自论文/官方页面,可信。
- 以下数字**引用前需核对原文**:SeleCom Table 1 中 SeleCom 与 PISCO 自身的行;PISCO 消融表具体数值;COCOM 逐压缩率 GFLOPs 表;PISCO 公开 checkpoint 的 repo 名。
- "据检索为空"类判断(query-conditioned + 硬样本 RL;Perceiver IO decoder 半用于 RAG 读出;query 驱动的预算分配)均为**基于有限次检索的弱断言**,投稿前必须做正式的相关工作排查。
- 本方案写作时环境无法抓取 arXiv 全文,内容基于检索到的摘要、HTML 片段与官方页面综合而成。
