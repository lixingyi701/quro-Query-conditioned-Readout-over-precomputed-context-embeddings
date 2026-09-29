# D0 / D2 Causal-Order：测试→训练实验方案

> 日期：2026-09-29  
> 分支：feat/selecom-infeasibility  
> 目的：把 D0（memory-first）与 D2（question-first）从普通 prompt ablation 提升为一个有明确停止条件的 causal-order 机制实验。  
> 核心原则：**第一阶段不以 zero-shot QA 高低决定是否训练 D2。** 发布版 PISCO decoder LoRA 本来按 D0 接口训练，直接换成 D2 同时引入 causal-order change 与 interface mismatch；因此 QA 只能作为辅助观测。

## 1. 要回答的问题

当前 PISCO 风格 decoder 的 D0 顺序近似为：

\[
[\text{prefix}, M_1,\ldots,M_N,Q_1,\ldots,Q_T,A].
\]

在 causal self-attention 下：

\[
h_{M_i}^{(l)}=f(M_{\le i},\text{prefix}),
\]

所以 memory state 不可能读取后面的 query。query state 则可以读取前面的 memory：

\[
h_{Q_j}^{(l)}=f(M,Q_{\le j},\text{prefix}).
\]

仓库已有 D2：

\[
[\text{prefix},Q_1,\ldots,Q_T,M_1,\ldots,M_N,A].
\]

此时方向反转：

\[
h_{M_i}^{(l)}=f(Q,M_{\le i},\text{prefix}),
\]

因此 memory hidden state 在结构上允许变成 query-conditioned；反过来 query hidden state 看不到后面的 memory。

第一阶段的目标不是证明 D2 更强，而是验证：

> **仅改变 causal order，query-specific computation 是否从 query 侧转移到 memory 侧？**

如果这个现象不存在，则没有充分理由花一轮训练预算适配 D2；如果现象稳定存在，才进入 matched LoRA adaptation。

---

## 2. 为什么只看 ΔM_l / ΔQ_l 不够

保留原有逐层相对更新：

\[
\Delta_M^{(l)}
=
\mathbb E_i
\frac{\|h_{M_i,\mathrm{out}}^{(l)}-h_{M_i,\mathrm{in}}^{(l)}\|}
{\|h_{M_i,\mathrm{in}}^{(l)}\|},
\]

\[
\Delta_Q^{(l)}
=
\mathbb E_j
\frac{\|h_{Q_j,\mathrm{out}}^{(l)}-h_{Q_j,\mathrm{in}}^{(l)}\|}
{\|h_{Q_j,\mathrm{in}}^{(l)}\|}.
\]

它们回答的是“哪些位置在被写入”，不能回答“这些变化是不是由 query / memory 导致”。

例如 D2 中 ΔM 变大，可能来自：
- query 真正改变了 memory；
- position/RoPE 改变；
- prompt boundary 改变；
- decoder 进入不同分布。

因此第一阶段必须加入 **counterfactual sensitivity**。

---

## 3. 两组对称干预

对一对 prompt-layout 完全匹配的样本 A、B，记：
- A 的 memory 为 M_A，query 为 Q_A；
- B 的 memory 为 M_B，query 为 Q_B。

对 D0、D2 分别跑三次：

1. original：M_A + Q_A
2. query swap：M_A + Q_B
3. memory swap：M_B + Q_A

### 3.1 主指标：memory 对 query 的敏感性

固定 M_A，只换 query：

\[
S_{M\leftarrow Q}^{(l)}
=
\mathbb E_i
\left[
1-\cos
\left(
h_{M_i}^{(l)}(M_A,Q_A),
h_{M_i}^{(l)}(M_A,Q_B)
\right)
\right].
\]

因 causal mask：

- D0：query 位于 memory 后，应接近数值误差底；
- D2：memory 位于 query 后，允许显著大于 0。

这是 **test→train gate 的唯一主指标族**。

同时保存 symmetric relative-L2：

\[
R_{M\leftarrow Q}^{(l)}
=
\mathbb E_i
\frac{\|h_i(Q_A)-h_i(Q_B)\|}
{\frac12(\|h_i(Q_A)\|+\|h_i(Q_B)\|)}.
\]

但 gate 默认使用 cosine sensitivity，避免同时挑多个显著性口径。

### 3.2 对称验证：query 对 memory 的敏感性

固定 Q_A，只换 memory：

\[
S_{Q\leftarrow M}^{(l)}
=
\mathbb E_j
\left[
1-\cos
\left(
h_{Q_j}^{(l)}(M_A,Q_A),
h_{Q_j}^{(l)}(M_B,Q_A)
\right)
\right].
\]

理论上：

- D0：query 可以读取 memory，因此可以 > 0；
- D2：query 位于 memory 前，应接近数值误差底。

这不是训练 gate，但它是非常重要的方向性 sanity check。理想结果是：

\[
\text{D0}: S_{M\leftarrow Q}\approx0,\quad S_{Q\leftarrow M}>0
\]

\[
\text{D2}: S_{M\leftarrow Q}>0,\quad S_{Q\leftarrow M}\approx0.
\]

这比单独观察 ΔM / ΔQ 更能说明 causal topology 的确改变了 interaction 被写到哪里。

---

## 4. 必须消除的位置混杂

不能只按“query token 数相同”配对。SentencePiece/chat template 的边界合并可能让两个表面长度相同的 query 在实际 prompt 中落在不同 token position。

当前实现要求 A/B 在以下内容上 **完全相同**：

- D0 memory slot positions；
- D0 query token positions；
- D2 memory slot positions；
- D2 query token positions；
- retrieved document 数量。

也就是说，counterfactual query / memory 的替换不会改变被比较位置的 RoPE index。

如果无法找到足够 pair，正确操作是降低 --pairs 或重新构造配对集；**禁止为了凑样本放松 position match。**

---

## 5. hidden state 采集口径

不要继续直接把 output.hidden_states 的最后一步当作 decoder block update。

新代码在每一个真实 Mistral decoder block 上分别注册：
- forward-pre hook：记录 block input；
- forward hook：记录 block output。

只保存 memory/query positions。

因此：

\[
\Delta^{(l)}
=
\frac{\|h_{\mathrm{block-out}}^{(l)}-h_{\mathrm{block-in}}^{(l)}\|}
{\|h_{\mathrm{block-in}}^{(l)}\|}.
\]

这样不会把 final RMSNorm 混成“最后一个 decoder block 的变化”。

---

## 6. 第一阶段同时记录 QA，但 QA 不进入 gate

每个 original run 仍记录：
- teacher-forced answer NLL；
- 可选 --generate 后记录 EM / F1 / substring。

原因是这些数据后面分析 interface mismatch 有用。

但：

\[
QA_{D2}^{zero-shot}<QA_{D0}
\]

**不能作为 HOLD 判据。**

因为 published PISCO decoder LoRA 已按 D0 训练。zero-shot D2 同时测了：

\[
\text{causal-order change}
+
\text{LoRA/interface mismatch}.
\]

反过来，zero-shot D2 偶然更高也不能直接当成 Q→M 架构优越的证据。

---

## 7. 测试→训练 gate（预注册）

默认阈值已经固化进 src/causal_order.py 和诊断脚本。它们是工程决策阈值，不是 Transformer 的普适自然常数；**正式运行后不要根据结果再移动阈值。**

### Gate 0：实验有效性

必须满足：
- position control 全部通过；
- 无 NaN/Inf；
- 至少 32 个 exact-position counterfactual pairs。

任何一项失败：

\[
\boxed{\text{RERUN / INVALID}}
\]

而不是解释机制。

### Gate 1：总体效应稳健

对每个 pair 先跨层平均：

\[
\bar S^{(n)}=\frac1L\sum_l S^{(n,l)}.
\]

对：

\[
\bar S_{M\leftarrow Q,D2}
-
\bar S_{M\leftarrow Q,D0}
\]

做 paired bootstrap。

要求 95% CI 下界：

\[
CI_{low}>0.
\]

### Gate 2：效应不是数值噪声

默认同时要求：

\[
\operatorname{mean}(S_{M\leftarrow Q,D2})\ge10^{-4}
\]

且：

\[
\frac{\operatorname{mean}(S_{D2})}
{\max(\operatorname{mean}(S_{D0}),10^{-12})}
\ge3.
\]

绝对阈值防止“D0 几乎为零，所以任何浮点差异都有巨大倍数”。

### Gate 3：效应不能只来自一个偶然层

逐层做 paired bootstrap，要求至少 25% decoder blocks 的：

\[
CI_{low}^{(l)}>0.
\]

### 最终判据

四个 gate 全部满足：

\[
\boxed{\text{GO\_TRAIN}}
\]

否则：

\[
\boxed{\text{HOLD\_NO\_TRAIN}}
\]

HOLD 的含义是：

> 目前没有足够强的机制证据值得投入 matched D2 training。

它不等价于：
- Q→M 普遍无效；
- soft compression 不能 query-condition；
- D0 永远优于 D2。

如果样本少或效应临界，先扩 pair / 排查 position control，而不是直接训练。

---

## 8. 已实现代码

### src/causal_order.py

提供：
- real decoder block hooks；
- ΔM_l / ΔQ_l；
- cosine / relative-L2 sensitivity；
- paired bootstrap；
- 预注册 GO_TRAIN / HOLD_NO_TRAIN gate。

### scripts/diagnose_causal_order.py

直接复用：
- pisco_hotpot preset；
- 现有 PISCO cache；
- PiscoPromptBuilder 的 D0/D2；
- assemble_inputs；
- 发布 PISCO decoder adapter。

probe 时设置 generator_lora_init=frozen：发布/current decoder adapter 仍生效，但不会发生任何训练。

输出：

~~~text
/data02/quro/runs/causal_order_d0_d2/<timestamp>/
  manifest.json
  pairs.jsonl
  summary.json
~~~

summary.json 首先看：

~~~text
training_gate
primary_memory_query_sensitivity
complementary_query_memory_sensitivity
memory_update_D2_minus_D0
query_update_D0_minus_D2
~~~

### tests/test_causal_order.py

CPU-only contract test，不下载模型，覆盖：
- metric shape；
- query swap -> memory sensitivity；
- memory swap -> query sensitivity；
- identical states 接近 numerical floor；
- strong synthetic D2 effect -> GO_TRAIN；
- near-floor -> HOLD；
- pair 数不足 -> HOLD；
- QA 明确不参与 gate。

---

## 9. 运行顺序

### 9.1 先跑 CPU contract

~~~bash
python tests/test_causal_order.py
~~~

### 9.2 GPU smoke

8 pair 只检查：
- 模型能加载；
- prompt position 能匹配；
- hook 跑在正确 block；
- JSON 能落盘。

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_causal_order.py \
  --preset pisco_hotpot \
  --pairs 8
~~~

注意：8 pair 永远不应通过默认 min_pairs=32 gate。

### 9.3 正式 test run

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_causal_order.py \
  --preset pisco_hotpot \
  --pairs 64
~~~

第一次正式判断先不要 --generate，避免把生成耗时混入机制 probe。

### 9.4 QA 辅助版

机制 gate 完成后可补：

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_causal_order.py \
  --preset pisco_hotpot \
  --pairs 64 \
  --generate
~~~

该 run 的 QA 仍不改变 training_gate。

---

## 10. 只有 GO_TRAIN 后才进入 matched LoRA adaptation

不是“重新训练一个 D2 模型 vs 发布 PISCO D0”。正确对照是从 **同一个发布 PISCO decoder LoRA** 出发做两个 matched continuation：

### T0：D0 continued control

\[
\theta_{PISCO}
\rightarrow
\theta_{D0}.
\]

### T2：D2 adapted

\[
\theta_{PISCO}
\rightarrow
\theta_{D2}.
\]

两边必须锁死：
- train rows；
- seed；
- steps；
- batch / grad accumulation；
- LR、decoder_lr、scheduler、warmup；
- full PISCO cache；
- arm=P；
- generator LoRA init；
- eval cadence / selection metric。

唯一实验变量：

~~~text
--decoder_input_mode D0
vs
--decoder_input_mode D2
~~~

**steps / lr / decoder_lr 不在本文发明新值。** 应直接复制当前 P1 / P-direct 已确认的训练 recipe。

命令骨架：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --preset pisco_hotpot \
  --arm P \
  --decoder_input_mode D0 \
  --generator_lora_init pisco \
  --num_workers 4 \
  --eval_input_modes D0,D2 \
  --tag causal_order_D0_control \
  --out_dir /data02/quro/runs/causal_order_D0_control \
  <复制当前P1的steps/lr/decoder_lr等参数>
~~~

~~~bash
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --preset pisco_hotpot \
  --arm P \
  --decoder_input_mode D2 \
  --generator_lora_init pisco \
  --num_workers 4 \
  --eval_input_modes D0,D2 \
  --tag causal_order_D2_adapted \
  --out_dir /data02/quro/runs/causal_order_D2_adapted \
  <完全相同的P1训练参数>
~~~

---

## 11. 训练后必须看 2×2，不只看两个 diagonal

| train / eval | D0 eval | D2 eval |
|---|---:|---:|
| D0-continued | A | B |
| D2-adapted | C | D |

解释规则：

### A >> B，但 D ≈ A

最符合“zero-shot D2 主要是 interface mismatch”。

结论：
- 不应拿 B 证明 Q→M 无效；
- D2 经适配可以工作；
- 下一步重新跑 D2 checkpoint 的机制 probe，看 query-conditioned memory trajectory 是否仍存在。

### A >> B，且 A >> D

在当前 PISCO latent + Mistral + 数据 / 训练预算下，memory-first 更容易被利用。

允许写：
> 在当前受控设定中，matched D2 adaptation 仍未恢复到 D0。

不允许写：
> question-first / query-conditioned memory 普遍无效。

### A ≈ D，但内部机制方向相反

这是非常有价值的结果：

\[
D0: S_{M\leftarrow Q}\approx0,\quad S_{Q\leftarrow M}>0
\]

\[
D2: S_{M\leftarrow Q}>0,\quad S_{Q\leftarrow M}\approx0.
\]

说明相似 QA 能力可以由不同 causal computation topology 实现。接下来才值得研究哪一种 state 更适合 multi-hop bridge / iterative readout。

### D > A

先不写架构优越结论。补：
- seeds；
- paired significance；
- bridge/comparison 分层；
- 训练后机制 probe。

---

## 12. 训练后必须重跑机制 probe

对 D2 best checkpoint：

~~~bash
python scripts/diagnose_causal_order.py \
  --preset pisco_hotpot \
  --pairs 64 \
  --checkpoint /data02/quro/runs/causal_order_D2_adapted/checkpoint_best.pt
~~~

要回答：

1. D2 的 S_M<-Q 是否仍明显 > D0；
2. LoRA 是否放大 / 压低这个 trajectory；
3. QA 恢复是否与机制变化同方向；
4. 是否出现“QA 恢复，但 query-conditioned memory signal 消失”的退化补偿。

只有这样，第二阶段才能区分：
- topology 被真正利用；
- decoder 只是找到另一条补偿路径。

---

## 13. 停止条件

1. 测试 gate 未通过：不启动 D2 LoRA training。
2. zero-shot D2 QA 下降：不能跳过 gate，也不能据此判死 D2。
3. position manifest 不一致：该 pair 作废。
4. 不用 final RMSNorm 混入的 output.hidden_states 最后一步支持 block-update 机制结论。
5. matched training 期间不同时改变 readout、KD、output scale、compression rate。
6. D2 训练后必须重跑 causal-order probe，不能只看 EM/F1。
7. 任意负结果都限定在当前 PISCO latent / decoder / data / budget。

**一句话执行逻辑：**

> 先证明 Q→M 确实把 query-specific computation 写进 memory；只有这个现象在严格 position control 下稳定存在，才花训练预算让 decoder 学会利用它；训练后再用 2×2 cross-order evaluation 区分 topology 效应与 LoRA/interface mismatch。
