# QuRO：P0 / P1 / P2 / P3（受控版）执行结果

> 2026-10-01 · 分支 `feat/selecom-infeasibility` · 对应 [执行方案](READER_CAUSAL_ORDER_EXECUTION_PLAN.md)  
> 数据：HotpotQA dev 2000 条，K=10 检索，文档截断 128 token（与压缩器一致），greedy，max_new_tokens=32。  
> Decoder：发布版 PISCO adapter（frozen）与 P₁（`oscale_P1/checkpoint_last.pt`，D0 域内 LoRA）。  
> 所有正式 run 的阈值均在运行前登记，见 [results/reader_causal_order/PREREG.md](../results/reader_causal_order/PREREG.md)；汇总 JSON 在同目录。

## 一句话结论

raw–memory 缺口真实存在（P₁ 下 +7.75 substring）。它**不是** K=10 干扰造成的，约 71% 定位在**含答案的那一段**，并随答案在段内的位置单调增大。这指向压缩保真或从 memory 中抽取靠后内容，而不是跨文档组合。D2 的 Q→M 路径存在（P2），但其答案作用**在两个 decoder 上不一致**（P3），按 §6 不启动 T-D2 训练。

## P2：D0/D2 拓扑诊断

`scripts/diagnose_causal_order.py --pairs 64`，发布版 PISCO。判定 `GO_FUNCTION_TEST`。

- 按 causal mask 应为 0 的两个方向，relative-L2 精确为 0：采集与位置有效。
- D2 的 S_{M←Q} = 1.17e-4（CI 1.11–1.26e-4），relative-L2 1.3%，逐层单调上升（最后一层 3.2e-4 / 2.4%）。作为对照，D0 的 S_{Q←M} = 0.16 / 51%。
- **门槛缺陷**：「D2 ≥ 3×D0」以 D0 为分母，而 D0 是 −1e-6 的浮点底，倍数没有意义。GO 实际上只靠 1e-4 绝对下限（均值 1.17e-4；28% 的样本对低于它）。复用前应改为与噪声底比较。
- **为什么小**（16 条，eager attention）：D2 中 memory 把约 7% 的注意力（浅层约 13%）给了 question，但 memory 残差范数约 119 且跨层恒定，文本位置为 0.1→28。每层 attention 写入只占 memory 范数的 0.1–1%。RMSNorm 抹去尺度后，读者只看到约 1e-4 的方向变化。主因是尺度，而不是可见性。

## P0：公平 raw–memory 缺口

`scripts/eval_raw_vs_memory.py`。同权重、同样本、同 harness，只换证据位置的内容。复现检查：P₁ D0 = 0.600（Stage C 0.599），PISCO D0 = 0.4865（0.485）。

| decoder | AG | D0 memory | RG raw | RG − D0 substring |
|---|---:|---:|---:|---:|
| P₁ | 0.274 | 0.600 | 0.678 | **+7.75 [5.95, 9.60]** |
| PISCO | 0.245 | 0.487 | 0.607 | +12.0 [10.1, 14.0] |
| bare Mistral | 0.229 | — | 0.619 | — |

P₁ 下 bridge 为 +9.3 [7.2, 11.3]，comparison 为 +1.1 [−2.7, 4.8]。以 AG 为基线，memory 拿到 raw 证据价值的约 81%。按预登记（≥2 分且 CI 下界 >0）判为**稳定缺口** → 进入 P1。P₁ 下的 raw 只是同权重对照，不是 raw 适配后的上限。

## P1（第一部分）：K=2 gold 与混合 raw/memory

`scripts/eval_gold_mixed.py`。纯条件与 P0 的 D0/RG prompt 逐 token 一致（逐行断言）。

| P₁，substring | all | bridge | comparison |
|---|---:|---:|---:|
| K=2 缺口 RR − MM | +7.2 [5.6, 8.9] | +7.4 [5.5, 9.2] | +6.35 [2.9, 10.1] |
| 去掉干扰：memory K2 − K10 | +5.8 | +7.1 | — |
| 去掉干扰：raw K2 − K10 | +5.25 | +5.2 | — |
| 交换顺序（MM / RR） | +0.05 / −0.5 | — | — |

角色定位（1193 道 bridge 题，恰有一段 gold 含答案；K=2 缺口为 +8.0）：

| 只把哪段变 raw | P₁ | PISCO |
|---|---:|---:|
| 答案段 | **+5.6 [3.3, 7.9]**（约 71%） | +11.2（约 100%） |
| 桥接段 | +0.6 [−1.0, 2.2] | −0.9 |
| 两段都 raw，相对只有答案段 raw | +2.35 [1.1, 3.7] | +0.7（不显著） |

按答案在 128-token 答案段内的位置（P₁，事后分析，未预登记）：<32 token：+1.6 [−2.0, 5.1]；32–79：+11.7 [8.6, 15.0]；≥80：+15.4 [8.4, 22.4]。PISCO 同向（+9.9 / +11.4 / +21.0）。

解读与限制：
- 缺口在没有干扰时依然存在，而且桥接实体从 memory 中基本取得到。按 §3 的决策表，这一行对应「单跳 memory 明显弱」，应优先查压缩保真与解码适配，而不是读出/组合。
- 单跳读不出**不等于** Z 中没有该信息。位置效应也可能与题型或答案类型混杂，需要做因果检验（见下一步）。

## P3（受控版）：D2 Q→M 路径的功能干预

`scripts/patch_causal_order.py`，200 对精确位置配对，双向共 400 个接收样本。

- 阻断范围是整个 memory 区间（slot 加 SEP）对「query 起点到 memory 起点」的读取。query 之后的模板和 SEP 会读 query，只屏蔽 query token 会泄漏。运行中已验证：阻断后，换问题时 memory 状态逐位相同。
- 有效性：`id_all` 逐位无影响。阳性对照 `xdoc_all` 的 ΔNLL 为 +1.1（PISCO）和 +1.7（P₁），substring 分别 −16 和 −31.5。

| 干预 vs none | PISCO（主）ΔNLL / Δsub | 判定 | P₁ ΔNLL / Δsub | 判定 |
|---|---|---|---|---|
| block_QM | +0.015 [−0.014, 0.045] / +1.75（n.s.） | NO_DETECTABLE_FUNCTION | +0.027 [0.015, 0.039] / −1.5 [−3.25, 0.00] | NLL_ONLY |
| xq_all | −0.073 [−0.099, −0.046] / −1.75（n.s.） | HARMFUL_ZERO_SHOT（仅 NLL） | +0.021 [0.009, 0.035] / −1.75 [−3.5, −0.25] | ANSWER_FUNCTIONAL（勉强） |

单层的次要分析（xq@12 在 dev 一半上最大）没有在 holdout 上复核。两个 decoder 都没有出现错误答案迁移。

解读：P₁ 上这条路径有微弱的正作用，约为换文档效应的 1.5%。主 decoder PISCO 上没有可检测的作用，NLL 方向甚至相反（PISCO 的长答案输出使短答案 NLL 成为较差的尺度）。D2 零样本使用时接口不匹配的损失（P₁ 约 16 分）远大于这条路径的贡献。按 §6「patch/阻断结果不稳」→ 不启动 T-D2。另外，xq 用的是另一道题的问题，**不是**计划要求的「同文档、依赖不同证据」的问题，所以这只是 P3 的受控子集。

## 下一步建议

> 以下为本次结果产生时的历史建议。用户后续决定：**不进行位置效应实验**，不继续扩大 D0/D2 诊断，转入 [Direct-CE / Direct-State / W-CE](READER_STATE_WORKSPACE_RUNBOOK.md)。下面的位置检验与完整 P3 不作为当前待办或训练准入条件。

1. **位置效应的因果检验（主线）**：把答案段的句子重排，让答案句移到段首，**重新压缩**后比较 memory 与 raw。若缺口随之消失，则可确认是压缩对段内靠后内容的保真不足。可以配合 K=1 答案段探针，以及在 Z 上做答案信息的线性/重建探针。
2. **P1 第二部分**：逐跳子问题需要 LLM 起草加人工抽检。它同时可以给出真正「同文档、不同证据」的问题对，用于完整的 P3。
3. P2 门槛的倍数判据改为与噪声底比较。
