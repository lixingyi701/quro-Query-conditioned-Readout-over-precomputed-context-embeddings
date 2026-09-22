# 缓存 latent 在 decoder 里到底被怎么处理：contextualisation 与可分离性

> 上游：`docs/FULL_COMPRESSION_INFEASIBILITY_RESULTS.md`
> 状态：第 0 步完成；可分离性扫描进行中
> 代码：`src/infeasibility.py`、`scripts/diagnose_full_compression_infeasibility.py`
> 产物：`$QURO_ROOT/results/full_compression_infeasibility/step0-contextualisation/`、
> `subspace-selecom/`、`subspace-pisco/`

## 0. 换靶子

上一轮证明 instruction suppression **不是** QuRO 的瓶颈：在 `P` baseline 的实际服务配置下
（K=10 + PISCO prompt + 真实问题），memory − raw = +0.015 [−0.045, +0.075]。

要攻的是另一个数字：

| HotpotQA dev, K=10, n=200 | substring |
| --- | ---: |
| `raw`（1126 token 原文） | 0.625 |
| **PISCO memory（80 latent）** | **0.520** |
| `none` | 0.255 |

**10.5 个点的压缩代价。** 历史上三条读出臂（A1/C1、R、RQ）都在"选哪些证据"这个方向上
拿到零边际，所以本轮换一个提法：**问题可能不在选什么，而在缓存 latent 被送进 decoder 的
形态**。

---

## 1. 第 0 步：memory 位置是被 decoder"加工"，还是只被"读取"？

### 1.1 为什么残差范数本身回答不了

上一轮测到 memory 位置以残差范数 104 进入、以 105 离开，而文本 token 从 0.14 长到 365。
但**范数不等于"没被加工"**：pre-norm 架构里

```
h_{l+1} = h_l + f_l(RMSNorm(h_l))
```

更新量 `f_l(·)` 是从**归一化后**的向量算出来的，它的幅度由这一层决定，与 `‖h_l‖` 无关。
同样一个幅度 O(1) 的更新，加在范数 0.14 的 token 上等于重写，加在范数 104 的槽位上几乎
没有发生。所以要直接测两个量：

- **相对更新** `‖h_{l+1} − h_l‖ / ‖h_l‖`
- **逐层旋转** `cos(h_{l+1}, h_l)`

### 1.2 结果（n=40，Level A，SeleCom 脚手架）

| layer | memory·document | memory·instruction | raw·document | raw·instruction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | **0.0355** | 1.8377 | 1.8912 | 1.8254 |
| 1 | **0.0350** | 1.1275 | 1.0868 | 1.1432 |
| 4 | **0.0557** | 0.6804 | 0.6877 | 0.6781 |
| 8 | **0.0743** | 0.5729 | 0.5688 | 0.5851 |
| 16 | **0.0645** | 0.4719 | 0.4378 | 0.4781 |
| 24 | **0.0642** | 0.2765 | 0.3104 | 0.2823 |
| 30 | **0.1310** | 0.4401 | 0.4063 | 0.4442 |
| **0–30 均值** | **0.0709** | **0.5499** | **0.5456** | **0.5551** |

（layer 31 是喂给 LM head 的最后一块，所有条件的更新都在那里暴涨一个量级，因此主干
单独统计。）

**同一条序列里，文本 token 的残差每层被更新 8× 于 memory 槽位（前半段 11×）。**
注意三个对照数字几乎相同（0.550 / 0.546 / 0.555）——**被冻住的不是"document 这一组"，
而是"memory 这种位置"**。

累积旋转（逐层 cos 的连乘，全栈下界）：

| | ∏ cos(h_{l+1}, h_l) |
| --- | ---: |
| **memory · document** | **0.6731** |
| memory · instruction | 0.0135 |
| raw · document | 0.0172 |
| raw · instruction | 0.0128 |

**文本位置走完 32 层后，残差与它的输入 embedding 已近乎正交（cos ≈ 0.014），被彻底重写；
memory 位置仍与写进去的 latent 保持 0.67 对齐。**

reconstruct 指令下同一批数字完全一致（document 0.0709，同样 8×），所以这不是任务效应，
是结构性的。

### 1.3 这句话的含义

> **PISCO latent 以一个使它对各层免疫的幅度插入残差流。它是 decoder 可以
> attend 的只读 K/V 源，但它本身不参与计算。**

这一条把上一轮几个反常一次解释掉：

- **为什么注意力没被"抢走"**：它们从来就只是静态 K/V 源，不是在竞争计算资源的 token。
- **为什么 decoder LoRA 的指令遵循能力不迁移到 memory 通道**（raw +0.65 / memory −0.125）：
  LoRA 改的是残差变换 `f_l`，而在 memory 位置上 `f_l` 的输出相对残差小一个数量级。
- **为什么 α 缩放对两端都无效**：缩放不改变"被读"的方向（attention 走 RMSNorm，尺度无关），
  只改变"被加工"的程度，而后者本来就接近零。

### 1.4 对 R / RQ 零结果的一个候选解释（假设，未验证）

R（残差臂）与 RQ（query-as-Q 写回）都把学到的修正 **写进 memory 槽位**。这确实会改变
**被读到的内容**（attention 经过 RMSNorm，对方向敏感），所以它们不是"改了个没用的量"。

但它们都**无法让 memory 位置参与计算**。如果 memory 相对 raw 的 10.5 点缺口有相当一部分
来自"raw token 经过 32 层组合、后续位置读到的是组合后的表示，而 latent 读到的永远是
压缩当时的那个向量"，那么**任何对静态 K/V 条目的修改都补不回来**——这与 R / RQ 的零边际
一致。

**这是候选解释，不是已验证的因果。** 能检验它的是 §2 的方案 #1：如果把读出写在会被加工的
尺度上确实拿到边际，这条解释成立；如果仍然零，说明缺口在别处。

---

## 2. 由此而来的方案 #1：scale-calibrated readout

**观测 → 动作**（指示 §12 要求的那句话）：

> 观测到 memory 位置以残差范数 104 进入、每层相对更新仅 0.071（文本 token 0.55，8×），
> 全栈累积旋转 0.67（文本 0.014），即压缩向量插入后不再被 decoder 逐层上下文化；
> 因此让 readout 在 **token embedding 尺度**输出并与 decoder LoRA 联合训练，使 memory
> 位置像 token 一样被加工，**离线 PISCO cache 完全不变**。

- **与已被证伪的 α 缩放的区别**：α 扫描是**推理时**缩放一个按范数 104 训练过的 decoder，
  所以 α=0.02 时 ROUGE-L 掉到 0.255 是必然。这里是**训练时**改输出值域，decoder 一起学着
  在新尺度上读。
- **与 QuRO 主张的关系**：仍然是"离线全量压缩 + 在线读出"，只改读出的值域。cacheability
  不受影响。
- **事先定死的 go/no-go**：
  1. hotpot dev substring 相对 P **≥ +1.5 点**；
  2. 训完重测本文 §1 的相对更新，memory 位置须升到文本 token 的同量级（≥ 0.3/层）。
  两条都满足才算成立。只满足 2 不满足 1 → 上下文化不是缺口所在，这条解释作废。
- **风险**：中。起点不再是恒等（R/RQ 的恒等起点换来了干净但为零的结果），可能一开始更差。

---

## 3. 方案 #2：压制成分是否可与文档信息分离

### 3.1 已有的分解证据

上一轮的 `mean` 干预给出 `Z_i = μ_row + Δ_i` 的分解读数：

| 干预 | 遵循率 | 重建 ROUGE-L |
| --- | ---: | ---: |
| 原样 | 0.225 | 0.616 |
| **全槽位取均值（留 μ_row，丢 Δ_i）** | **0.225（压制全留）** | 0.303 |
| 范数匹配随机方向（μ 和 Δ 都丢） | 0.625 | 0.118 |

**μ 承载压制，Δ 承载文档信息。** 于是"能不能只移走 μ"是一个有是非答案的问题，不是一个
设计选择。

### 3.2 latent 的语料级几何（8192 篇文档、65536 个 latent）

| 量 | 值 |
| --- | ---: |
| 单个 latent 的典型范数 | ≈ 104 |
| **语料均值 μ_global 的范数** | **56.41** |
| 最大奇异方向占二阶矩能量 | **47.5%** |
| 前 8 个方向占二阶矩能量 | 72.5% |

**每个 PISCO latent 有约 54% 的幅度是一个固定的、与文档无关的偏移**；整个分布是围绕单一
主方向的一个窄锥。这正是 SeleCom 理论里"正半空间"的定量版本——而且它说明
`8 × 4096` 的 cache 里，每篇文档独有的信息远少于名义维度。

### 3.3 扫描设计与判据

| 条件 | 去掉什么 | 留下什么 |
| --- | --- | --- |
| `global_mean` | 全部文档信息 | 只有 μ_global（互补检验） |
| `decenter` | μ_global | Δ（范数同时缩到 ≈87） |
| `decenter_renorm` | μ_global | Δ，**范数复原到 ‖Z‖**（把尺度混淆排除） |
| `deproject:k`，k ∈ {1,2,4,8,16} | 前 k 个主方向 | 其余，范数复原 |
| `mean`（参照） | Δ | μ_row |

每个条件**同时**测 conflict 与 reconstruct。判据：

- **可分离**：某个 k 上遵循率明显回升（→ raw 的 0.65 一侧）而 ROUGE-L 基本保持
  → 创新点成立：**"软压缩表示里导致指令压制的成分是一个低秩、query-无关的偏移，
  可在不损失文档信息的前提下移除"**。
- **不可分离**：遵循率和 ROUGE-L 一起掉（像 `norm_matched_random` 那样）
  → 同样是结果：**SeleCom 的 infeasible 在一个它自己没说的意义上成立**——压制与文档信息
  在表示层不可分。
- **`global_mean` 若单独就能压制**（遵循率 ≈ 0.225，而它不含任何文档信息）
  → 压制成分连 row-specific 都不是，是一个全语料共享的常向量。

两种结果都能写，这是这个实验相对 R / RQ 的关键区别：R / RQ 只有正结果才有意义，
所以零收益等于白跑。

### 3.4 结果

<!-- PENDING: subspace-selecom / subspace-pisco -->

---

## 4. 复现

```bash
# 第 0 步：逐层相对更新与旋转（记录在 norm_logit_stats.npz 里）
python -u scripts/diagnose_full_compression_infeasibility.py \
    --level A --rows 40 --prompt_style selecom_literal --system_prompt none \
    --run_id step0-contextualisation

# 方案 #2：可分离性扫描（两种脚手架各一次）
python -u scripts/diagnose_full_compression_infeasibility.py \
    --level A-subspace --rows 40 --prompt_style selecom_literal --system_prompt none \
    --run_id subspace-selecom
python -u scripts/diagnose_full_compression_infeasibility.py \
    --level A-subspace --rows 40 --run_id subspace-pisco
```

`latent_statistics.pt`（μ_global 与前 32 个主方向）随 run 一起保存，
`manifest.json` 的 `latent_statistics` 字段记录估计口径（文档数、seed、能量占比）。
