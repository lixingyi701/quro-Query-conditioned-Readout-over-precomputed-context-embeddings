# Reader Trajectory：从“静态压缩记忆”现象到测试—训练路线

> 日期：2026-09-30  
> 分支：`feat/selecom-infeasibility`  
> 上游文档：[FULL_COMPRESSION_INFEASIBILITY_RESULTS.md](FULL_COMPRESSION_INFEASIBILITY_RESULTS.md)、[LATENT_CONTEXTUALISATION.md](LATENT_CONTEXTUALISATION.md)  
> 纠偏基准：[LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md](LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md)  
> 性质：**实验路线文档，不是机制结论。** 本文只把已有现象组织成可证伪的测试问题；任何新训练模块都必须由测试结果触发。

---

## 0. 当前决策：先测试 reader trajectory，不直接设计新模型

最近一轮讨论从下面的观察出发：

- PISCO memory 位置在 decoder 中的相对逐层更新明显小于普通文本/query 位置；
- query/context token 在 decoder 中仍然显著演化；
- 但此前把“memory 变化小”直接解释成“memory 是无效只读 K/V”“因此跨文档组合失败”，已经被纠偏文档明确判为推断过界；
- `output_scale` Stage C 进一步说明：**让 memory 位置更容易变化，并没有自动换来 QA 增益。**

因此当前不再采用下面这条未经验证的推理：

```
memory 更新小
=> memory 上下文化不足
=> 需要让 memory 像普通 token 一样变化
=> 设计 workspace / cross-attention / recurrent readout
```

改成严格的“测试—训练”顺序：

```
观察
=> 功能性干预
=> 定位失败发生在哪里
=> 只有因果证据支持时，才进入最小训练实验
=> 训练结果再决定是否需要新结构
```

**当前最重要的问题不是“新模块怎么设计”，而是：压缩表示相对 raw context 的性能差距，是否真的来自 decoder 内部 reader state 的错误演化。**

---

## 1. 思路演进：为什么研究对象从 memory state 转向 reader trajectory

### 1.1 第一阶段：复现 SeleCom 对 full compression 的批评

最初目标是检查全量软压缩进入 decoder 后是否出现 SeleCom 所描述的异常：

- memory 是否过度吸引注意力；
- 是否压制后续 instruction；
- memory 与 raw 的行为差异是否来自 attention dominance。

实验最终没有支持“attention capture 是主因”的简单机制解释。尤其在受控干预下，attention dominance 增强并没有稳定导致行为变坏。

**保留的结论：** full compression 的确存在行为/性能差异，但不能把它直接归因为全局 attention dominance。

### 1.2 第二阶段：发现 memory residual state 相对稳定

随后测得 memory 位置的相对逐层更新远小于普通文本位置。旧报告一度把它解释成：

> memory 几乎是只读 K/V；它自身不参与计算；raw token 能逐层组合而 memory 不能。

这里的“测量”值得保留，但“机制解释”过强。

纠偏文档已经指出：

- 小相对更新不等于没有功能；
- layer-specific K/V projection 仍然会变化；
- 后续 query/answer 位置可以跨层读取 memory；
- memory 自身是否大幅演化，与系统是否能完成跨文档推理不是同一件事。

### 1.3 第三阶段：`output_scale` 对“强制 memory 上下文化”给出负结果

Stage C 将 memory 写入 residual stream 的尺度降低，希望让其更容易被 decoder block 改写。

结果：

- QA 没有增益；
- 训练后 memory 的相对更新甚至重新下降；
- 目标域 decoder LoRA 本身却带来明显收益。

因此目前没有证据支持：

> “memory 更新不够大”就是性能差距的原因。

这一步非常重要：它阻止我们继续把“让 memory 更像普通 token”当成默认目标。

### 1.4 当前重新表述：static memory / dynamic reader 只是待验证假设

在标准 decoder-only causal 序列中：

```
[ memory Z ][ query Q ][ answer A ]
```

memory 在 query 前面，因此 memory 本身不可能被后来的 query 条件化；而 query token 可以看到前面的 memory：

```math
Q^{(l+1)} = f_l(Q^{(l)}, Z^{(l)}).
```

所以一个更自然的观察是：

> **query token 本身已经是 decoder 内部的 reader state。**

它逐层变化，并且每一层都能重新读取前面的 compressed memory。

这意味着：单纯提出“再放几个 workspace token，让它们从 `W^0` 变成 `W^1`”并不是新机制——现有 decoder 本来就在做类似的逐层 reader computation。

因此目前不把 workspace、external memory cross-attention、recurrent readout 视为下一步既定方案。它们只有在测试证明“现有 reader trajectory 是失败位置”之后才值得讨论。

---

# 2. 测试阶段的总目标

测试阶段要回答四个递进问题：

1. **差距是否真实存在？**  
   在公平训练和统一评测口径下，raw 与 memory 还有多少差距？

2. **差距是信息缺失，还是信息存在但使用失败？**  
   尤其是 HotpotQA bridge 题：单跳事实是否已经丢失？

3. **memory 自身逐层更新是否具有功能作用？**  
   “更新小”必须通过阻断实验转化为因果证据。

4. **真正影响答案的是否是 query/reader trajectory？**  
   需要 activation patching，而不是只看 norm/cosine。

只有第 4 个问题得到正面证据，才允许进入“修 reader trajectory”的训练实验。

---

# 3. Test 0：先补齐公平 raw–memory gap

## 3.1 目的

当前历史数字存在 decoder 是否训练、样本数和 harness 不完全一致的问题。

因此在谈“压缩导致 reader trajectory 错误”之前，先确认：

```math
Delta_{mathrm{raw-mem}}
=
mathrm{Score}_{raw}
-
mathrm{Score}_{memory}
```

在**同样本、同 checkpoint、同生成参数、同评分口径**下是否仍然稳定存在。

## 3.2 条件

至少比较：

| decoder | raw | memory | 用途 |
|---|---|---|---|
| 发布 PISCO decoder | ✓ | ✓ | 冻结系统表示差异 |
| 当前目标域训练后的 P₁ decoder | ✓ | ✓ | 同一训练 decoder 下的表示差异 |

如果之后需要声称“各自适配后的系统上界”，再增加 raw QA LoRA；但不要先训练新的 raw 系统扩大变量。

## 3.3 指标

同时报告：

- EM；
- F1；
- substring；
- teacher-forced answer NLL；
- 输出长度；
- bridge / comparison 分层；
- 逐例配对结果。

## 3.4 决策意义

- **若训练后的 P₁ 下 raw–memory 差距已经很小：** 不再把“修复压缩性能差距”当主任务，转向效率/多查询复用。
- **若差距稳定存在：** 才进入 Test 1–4，定位它发生在哪里。

---

# 4. Test 1：K=2 gold + 逐跳诊断——先判断是不是信息已经丢了

这是进入任何 reader 机制分析前的最高优先级测试。

## 4.1 目的

bridge 题表现差，可以同时由两类原因解释：

### H-info：压缩阶段已经丢信息

例如桥接实体或第二跳事实没有保留下来。

### H-compose：单篇事实仍在，但 decoder 没把两篇证据连接起来

这两种解释会导向完全不同的方案，因此必须先分开。

## 4.2 测试矩阵

对同一批人工可核验 bridge 样本，同时跑 raw 与 memory：

1. 原始 K=2 gold 多跳问题；
2. 第一跳子问题：找 bridge entity；
3. 第二跳子问题：直接提供正确 bridge entity，再问最终事实；
4. 原始问题 + oracle bridge entity；
5. K=10 原始设置；
6. 可选混合输入：doc1 raw + doc2 memory，doc1 memory + doc2 raw。

## 4.3 解释

| 结果 | 优先解释 |
|---|---|
| memory 单跳已经明显差 | 优先查压缩保真/解码能力，不进入 reader-workspace 训练 |
| 两个单跳都能答，但原多跳失败 | composition / reader computation 成为更强候选 |
| 给 bridge entity 后明显恢复 | 第一跳获取或跨文档连接值得继续定位 |
| K=2 接近 raw、K=10 差距扩大 | 干扰/选择问题优先 |
| 某一篇换 raw 即恢复 | 定位到具体跳的表示/读取瓶颈 |

**关键原则：** 重建文本中“桥接实体出现/不出现”只能作辅助证据，不能单独证明信息存在或消失。

---

# 5. Test 2：Memory-write blocking——“更新小”是否真的无关紧要

## 5.1 目的

目前已知：

```math
rac{|h_M^{l+1}-h_M^l|}{|h_M^l|}
```

较小。

但这只是测量。

要判断 memory 是否主要作为稳定的信息源，需要直接阻断其 residual 写路径。

## 5.2 干预

保持非-memory 位置正常更新，memory 仍然向后续 token 提供每层 K/V。

分别测试：

1. normal；
2. block memory attention residual write；
3. block memory MLP residual write；
4. block both。

概念上：

```math
h_{M}^{l+1}=h_M^l
```

但每一层仍允许：

```math
K_M^l=W_K^l h_M^l,qquad
V_M^l=W_V^l h_M^l
```

被后续 query/answer 读取。

## 5.3 指标

- QA EM/F1/substring；
- answer NLL；
- bridge/comparison 分层；
- 每层 memory/query state 统计，用于确认干预实际生效。

## 5.4 目的不是证明“static memory 正确”

解释必须保持保守：

- **blocking 明显掉分：** memory 的小更新具有功能价值；“static memory”假设被削弱。
- **blocking 基本不掉分：** 当前系统对 memory residual evolution 依赖较弱；这支持继续研究 reader side，但**不证明增强 memory 更新一定无用，也不证明 external memory 架构会成功**。

建议在运行前固定一个非劣界限（例如 substring 绝对变化 1 点以内），并用逐例 bootstrap CI 判断，而不是看完结果再定义“差不多”。

---

# 6. Test 3：先测 reader trajectory，而不是继续测 memory 几何

## 6.1 目的

当前 P baseline 本来已经实现：

```math
Q^{(0)}
ightarrow
Q^{(1)}(Z)
ightarrow
Q^{(2)}(Z)
ightarrowcdots
ightarrow
Q^{(L)}(Z).
```

因此真正的问题应改写为：

> raw context 与 compressed memory 是否把**相同 query**驱动到了不同的内部计算轨迹？

## 6.2 测量对象

对相同问题，收集：

```math
h_{Q,mathrm{raw}}^{(l)},qquad
h_{Q,mathrm{mem}}^{(l)}.
```

同时保留：

```math
h_{A,mathrm{raw}}^{(l)},qquad
h_{A,mathrm{mem}}^{(l)}
```

作为补充。

不要再把单纯的 hidden-state norm 或相邻层 cosine 当作功能结论。它们只描述轨迹外观。

更有意义的辅助量包括：

- 同一 query token 的 raw/memory 表示差异随层变化；
- correct-memory vs mismatch-memory 对 query state 的影响量；
- bridge / comparison 的轨迹差异；
- 最终 answer NLL 与这些量的样本级关联。

## 6.3 位置编码混淆必须控制

raw prefix 与 memory prefix 长度不同，因此 query 的 RoPE position 通常不同。

直接比较或 patch：

```math
h_{Q,mathrm{raw}}^{(l)}
ightarrow
h_{Q,mathrm{mem}}^{(l)}
```

可能把“上下文表示差异”和“绝对位置差异”混在一起。

因此 raw–memory trajectory 测试必须至少有一种 position control：

- 给 raw / memory 的 query span 使用相同的显式 position anchor；
- 先验证这种 position anchoring 本身不会显著改变各自 QA；
- 若 anchoring 改变行为过大，则 raw→memory patch 只能作为探索性结果，不能当机制证据。

在此之前，可先做**同布局、同位置**的 correct-memory vs mismatch-memory patch，作为更干净的内部验证。

---

# 7. Test 4：Activation patching——当前最关键的因果测试

## 7.1 核心问题

不是问：

> query state 看起来是否变化很大？

而是问：

> **把“正确条件下的 reader state”放进失败条件，能不能救回答案？**

只有后者能支持 reader trajectory 是因果瓶颈。

## 7.2 第一阶段：同布局 patch（优先）

先利用长度和位置完全一致的条件：

```
correct memory + Q
mismatch memory + Q
```

在层 `l` 将 mismatch run 的 query span 替换为 correct-memory run 的 query hidden states：

```math
h_{Q,mathrm{mismatch}}^{(l)}
leftarrow
h_{Q,mathrm{correct}}^{(l)}.
```

扫描若干层，例如 early / middle / late，而不是一开始每层全扫。

### 目的

证明：

> memory 中的有效证据是否通过 query/reader state 向后传播，并且这种 state 对答案有因果作用。

如果连这个同布局 patch 都不能改变答案，直接做 raw→memory trajectory distillation 的依据就很弱。

## 7.3 第二阶段：raw → memory patch（带 position control）

只有完成 §6.3 的 position control 后，再测试：

```math
h_{Q,mathrm{mem}}^{(l)}
leftarrow
h_{Q,mathrm{raw}}^{(l)}.
```

主要观察：

- answer NLL 是否下降；
- 原本 memory 错、raw 对的样本中，有多少被 rescue；
- rescue 是否集中在 bridge；
- 哪些层的 patch 稳定有效。

## 7.4 必须做反向和错误 patch 对照

至少包括：

- raw/correct → memory/mismatch：目标方向；
- memory/mismatch → raw/correct：反向干预；
- 来自其它样本/错误文档的 query state patch：排除“任意替换 hidden state 都像 regularization 一样有用”。

## 7.5 允许进入训练的核心信号

最重要的不是 hidden similarity，而是：

> **某个稳定层区间的 reader-state patch 能在逐例配对下显著恢复答案能力。**

如果 patch 只改变表示距离但不改变 QA/NLL，则不能据此训练 trajectory alignment。

---

# 8. 测试 → 训练的明确门槛

任何“workspace / recurrent readout / trajectory distillation”训练前，至少检查以下门槛。

| Gate | 要求 | 未通过时 |
|---|---|---|
| G0 公平差距 | matched raw–memory gap 稳定存在 | 不再以性能缺口为主故事 |
| G1 信息可用 | 至少主要失败样本的单跳事实仍可从 memory 读取 | 回压缩器/保真度，不训练 reader |
| G2 memory 写路径 | blocking memory write 不造成明显任务损失 | 若明显掉分，static-memory 假设被削弱 |
| G3 reader 因果性 | query-state patch 能显著改善 NLL/QA | 若无 rescue，不训练 trajectory |
| G4 层定位稳定 | patch 效应在样本/bootstrap/bridge 子集上有稳定层区间 | 若不稳定，不据此设计层级模块 |

建议预注册一个“非平凡 rescue”效果门槛，例如：

```math
mathrm{RescueFraction}
=
rac{mathrm{Score}_{patch}-mathrm{Score}_{memory}}
{mathrm{Score}_{raw}-mathrm{Score}_{memory}}
```

要求 paired CI > 0，并在实验开始前固定一个最小实用值（例如 20%），避免看结果后调标准。

**只有 G0–G4 大体成立，才进入下面的训练阶段。**

---

# 9. 训练阶段：先做最小训练，不直接加 workspace

即使 Test 4 支持 reader trajectory 是瓶颈，也不代表必须立即添加新结构。

第一训练实验应该最小化变量。

## 9.1 Training 1：现有 P 架构上的 reader-state distillation

Teacher：

```math
D_{mathrm{raw}}+Q
ightarrow
h_{Q,T}^{(l)}
```

Student：

```math
Z_D+Q
ightarrow
h_{Q,S}^{(l)}
```

只在 Test 4 已经证明有因果作用的层集合 `S` 上加入状态损失：

```math
mathcal L
=
mathcal L_{mathrm{answer}}
+
lambda
sum_{lin S}
d!left(
operatorname{Norm}(h_{Q,S}^{(l)}),
operatorname{Norm}(h_{Q,T}^{(l)})
ight).
```

这里先不引入 workspace、不改 cache、不增加额外 readout。

### 为什么先做这个

因为 Test 4 如果已经证明：

> “把正确 reader state 放进来可以 rescue”，

最小动作就是训练现有 decoder 自己产生更接近该 state 的轨迹。

如果连这个训练都没有任务收益，直接加新 architecture 的因果依据会明显变弱。

## 9.2 必须保留的训练对照

至少：

- CE-only P₁；
- CE + reader-state distillation；
- 相同训练预算和 decoder LoRA；
- 可选 wrong/mismatch teacher state，检查收益是否来自正确 trajectory，而不是泛化正则化。

## 9.3 训练结果如何解释

### 情况 A：state 更接近 teacher，QA 也提升

支持：

> reader trajectory 是可训练、且与任务相关的瓶颈。

此时才值得继续判断是否需要更强的 reader capacity。

### 情况 B：state 更接近 teacher，但 QA 不提升

削弱：

> “trajectory mismatch 本身导致答案失败”。

不要因为 loss 成功下降就宣布机制成立。

### 情况 C：state 无法明显靠近 teacher

可能是现有 decoder capacity/优化限制，但仍不能自动推出“workspace 必须有效”。

只有结合 Test 4 的强 patch rescue，才有理由进入结构扩展。

---

# 10. Workspace / external-memory 只作为二阶段候选，不是当前方案

如果满足：

1. memory 单跳信息可用；
2. memory-write blocking 基本无害；
3. raw/correct reader-state patch 有稳定且明显的 rescue；
4. 现有架构上的 trajectory distillation 无法充分实现该状态；

才考虑更强结构，例如：

```
[Q][W][A]
```

并让少量 `W` 在特定 decoder 层显式读取外部静态 `Z`。

但需要牢记：

- “`W^0 -> W^1 -> ...`”本身不是创新，现有 query token 本来就在逐层演化；
- 当前 P 的 self-attention 已经允许 query 每层读取前面的 memory；
- 显式 cross-attention 只有在测试证明现有读取路径能力不足时才有依据；
- `4096 -> 1024 -> 4096` bottleneck 是否有害，也需要独立验证，不能因为结构看起来更 native 就假设会更好。

因此本文**不把 QuRO-W 当作下一步实现任务**。

---

# 11. 推荐执行顺序

```
P0  Fair raw–memory gap
    |
    v
P1  K=2 gold + hop decomposition
    |
    +-- 单跳就失败 ------------> 回压缩/保真度方向，停止 reader 训练
    |
    v
P2  Memory-write blocking
    |
    +-- blocking 明显伤害 ------> static-memory 假设削弱
    |
    v
P3  Reader trajectory measurement + position control
    |
    v
P4  Activation patching
    |
    +-- patch 无 rescue ---------> 停止 trajectory/workspace 方向
    |
    v
T1  Existing P + trajectory distillation
    |
    +-- QA 无增益 --------------> 不直接升级复杂结构
    |
    v
T2  仅在强因果证据仍存在时，再评估 workspace / external-memory capacity
```

---

# 12. 当前最重要的研究纪律

过去几轮实验已经说明：**一个内部现象看起来很合理，不等于它就是性能瓶颈。**

本轮必须坚持：

> **测试的目的不是为预想的新模型寻找证据，而是决定这个新模型是否还有必要被训练。**

具体要求：

1. 测量与机制结论分开；
2. 观察性相关与因果干预分开；
3. 每个训练模块前先写明“哪个测试结果要求它存在”；
4. 负结果直接终止对应路线，不通过改口继续保留模块；
5. 不以 norm、cosine、attention heatmap 的“更像文本”作为成功标准；
6. 最终成功标准仍然是任务能力、配对因果恢复和完整效率收益。

当前的核心假设只能写成：

> **H-reader（待验证）：PISCO compressed memory 可能保留了相当一部分任务证据，但相对 raw context，它没有把 decoder 后续的 query/reader state 驱动到同样有效的计算轨迹。**

本文的全部测试就是为了决定：**H-reader 是否值得进入训练阶段。**
