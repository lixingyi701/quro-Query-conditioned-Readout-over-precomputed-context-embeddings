# QuRO 实验总账与多数据集评估计划

整理日期：2026-10-08。已更新八个远程分支的引用，读取各分支已提交的实验报告、历史修订和相关实现。最新 QER 结果提交为 `d85eb0c`；原共享投影器分支为 `69001c3`；main 为 `65895d3`。

本文汇总仓库记录的主要实验系列，不把历史版本、不同指标或不同 reader 的数字连接成一条提升曲线。真实 GPU 实验由服务器执行；本次没有重跑训练，也没有取得未提交的服务器逐题预测或原始日志。表中 EM、F1、substring 均转换为百分制，差值单位为百分点（pp）。旧报告的显示精度与逐题配对计算不同，差值优先沿用原分析结果。

**当前决策：保留已有共享 query 投影器 SQ 作为多数据集扩展的主方法；记录并暂停新增 QER 内容监督方案；扩展评估覆盖、匹配基线和复用成本。**

## 1. Query 分支的结论需要准确保留

目前已经观察到小幅正向增量，不能写成“在投影器中加入问题没有效果”。两轮实验分别为：

| 比较 | 评估范围 | ΔF1 | 95% 配对 CI | 条件 |
|---|---|---:|---|---|
| SQ − S0m | 内部 test 5405 | **+0.62** | [+0.11, +1.14] | 首轮、单 seed42；S0m 使用固定随机条件向量 |
| SQX − S0m | 内部 test 5405 | +0.85 | [+0.18, +1.48] | 增加跨文档 attention，单 seed42 |
| SQ − S0 | 内部 test 5405 | **+0.37** | [−0.01, +0.74] | 标准移除 query 分支、两 seed 平均 |
| SQ − S0 | dev2000 | +0.33 | [−0.29, +0.96] | 同一标准消融 |
| SQ − S0 | test comparison，1109 | **+1.24** | [+0.23, +2.30] | 两 seed 平均，探索性题型分析 |
| SQ − S0 | test bridge，4296 | +0.14 | [−0.25, +0.53] | 两 seed 平均 |

因此，“约半个 F1 点的小幅正向增量”是对已有观察的概括。论文主表必须分别写 +0.62 和 +0.37，不能把不同轮次合并为精确的 +0.50，更不能提前写成所有数据集均稳定提高 0.5。两 seed 的标准 test 增量方向一致；总体区间仍较宽，正向观察与泛化确定性应分别陈述。[E01][E02][E18]

推荐表述：

> 在冻结发布版 PISCO、复用离线缓存的条件下，问题条件投影器在 HotpotQA 内部测试集上观察到约 0.4–0.6 F1 点的额外收益；标准两种子消融为 +0.37 F1，comparison 子集为 +1.24 F1。后续多数据集实验将检验这一增量的适用范围及在线成本。

QER、支持监督或 FiLM 未建立额外收益，不能覆盖掉这组 SQ/S0 正向结果。Bridge 的小幅平均增量也应保留数值，使用“尚未得到可靠确认”，而不是“无效”或“真实上限”。

## 2. 方法身份和比较对象

| 名称 | 计算或训练内容 | 是否减少 decoder memory | 与当前 SQ 的关系 |
|---|---|---|---|
| 发布版 PISCO | 冻结发布权重，直接读取离线 Z | 否 | 原系统参照 |
| 历史 P / P₁ | 在项目任务上适配过 decoder LoRA 的直接读取模型 | 否 | 强域内参考；不同报告里的 P/P₁ 不保证同一 checkpoint |
| 早期 C / C1 | Query 条件二次读出，通常 80→8 | 是 | 旧小预算方案 |
| S | 余弦 top-B，原样选缓存槽 | 是 | 旧非参数基线 |
| R | 全量 Z 加条件残差；可同时训练 decoder LoRA | 否 | 旧残差路线 |
| RQ | Query 读取 Z 后转置分配、写回全量 Z | 否 | 旧写回路线 |
| S0m | 与 SQ 保持条件模块，输入固定随机向量 | 否 | 首轮参数匹配的无真实问题对照 |
| S0 | 文档 MLP，彻底移除 query 模块 | 否 | 当前主要 query 消融 |
| SQ | 共享逐文档残差 MLP，memory 读取问题 token 后形成条件项 | 否 | **当前主方法候选** |
| SQX | SQ 加跨文档 hidden attention | 否 | 单 seed 扩展，不默认加入主方法 |
| QER A/B/C | 从训练好的 S0 出发，增加两阶段 attention；B/C 加证据生成目标 | 否 | 最新候选，不能把效果算到原 SQ 名下 |

当前 SQ 保留每篇 8 个槽，10 篇共 80 个槽；缓存与 query 无关，在线输出 E 随 query 改变。SQ 新模块没有进一步降低槽数，效率主张应来自实测的缓存复用成本和质量成本比较。S0 不需要单独编码问题，SQ 需要一次冻结 decoder 的 question-only 前向，这部分必须计入在线成本。[E01][E02]

## 3. 最早阶段：原型、数据和接口检查

| 实验 | 已记录结果 | 使用方式 |
|---|---|---|
| v0.0 随机 prototype encoder | 尚未接入真实压缩器 | 工程历史，不作论文效果 |
| PISCO 跨 batch 编码 | batch16 vs64：最大绝对差0.66，平均差0.024，cos≥0.9997 | 数值复现记录，不作方法增益 |
| 单文档400步 vs 加干扰400步 | query 敏感度中位数0.007→0.534；无干扰 EM 与常量地板均14.06；加干扰 EM9.38、地板4.69 | 早期数据设计诊断 |
| 早期 TriviaQA 证据子集 | 全集 EM：P64.80、S63.40、C62.20、A60.00；210题子集：P35.71、S28.10、C18.10、A17.14 | 历史探索；不同子集和后续版本不得混用 |

早期 A 对照仍包含 query 余弦先验，后来修正为 A0/A1/C0/C1；缓存合并曾发生文档映射错位；S 路径和 train/eval dropout 也有修复。涉及已知错误的历史对照保留为研究记录，主表采用修正版。[E03][E04][E05]

## 4. 小预算读出：TriviaQA 和离线压缩率扫描

### 4.1 TriviaQA，原生 m=8，B=8，2000题

| 模型 | memory 数 | EM | F1 | substring |
|---|---:|---:|---:|---:|
| P 直接读取 | 约43.8 | 70.75 | 77.0 | 77.20 |
| S 余弦 top-B | 8 | 71.70 | 76.5 | 76.10 |
| C 二次读出 | 8 | 70.90 | 76.2 | 76.00 |

C 相对 S 为 −0.80 EM、−0.10 substring；旧小预算方案在该配置没有超过余弦规则。此处并非当前 SQ 的跨数据集成绩。[E04]

### 4.2 更宽缓存 m=32，输出仍为8槽

| 压缩栈 | 模型 | EM | F1 | substring |
|---|---|---:|---:|---:|
| 切块 PISCO | P | 71.20 | 77.8 | 78.90 |
| 切块 PISCO | A | 69.00 | 74.5 | 73.90 |
| 切块 PISCO | C | 68.55 | 74.0 | 73.70 |
| 切块 PISCO | S | 69.15 | 74.2 | 73.40 |
| COCOM 实验栈 | P | 67.10 | 76.2 | 77.05 |
| COCOM 实验栈 | A | 68.95 | 74.8 | 73.80 |
| COCOM 实验栈 | C | 66.10 | 73.8 | 73.50 |
| COCOM 实验栈 | S | 66.05 | 73.8 | 73.85 |

切块栈 C−S substring +0.30，COCOM 栈 −0.35；扩大缓存未让当时的二次读出获得明显优势。COCOM 实验曾调整公开实现的 memory 位置，不能把该成绩当作未经修改的官方 checkpoint 基线。[E03][E04]

## 5. D1 和四格消融：query 信息能进入读出

D1 删除 decoder 中的问题明文；A0 真正没有 query 路径，A1 只含余弦条件，C0 只含学习式条件，C1 同时含两者。

| 数据集 / D1 EM | A0 | A1 | C0 | C1 |
|---|---:|---:|---:|---:|
| TriviaQA | 0.40 | 5.90 | 27.75 | 26.75 |
| HotpotQA | 10.45 | 12.70 | 30.55 | 31.15 |

学习式条件单独增量 C0−A0：TriviaQA +27.35 EM，HotpotQA +20.10 EM。历史常用 C1−A1 为 TriviaQA +20.85 EM；它测的是在余弦路径之上添加学习式条件，而不是相对完全无 query 的 A0。

TriviaQA D1 的 C1 F1=32.7，A1 F1=11.4，差21.3 F1点。正常 D0 保留问题明文时，这种大增量不能直接复现。D1 是信息路径分析，不能并入正常 QA 主表充当同样大小的收益。[E04][E05]

## 6. HotpotQA 小预算、预算扫描和 KD

### 6.1 修正后的匹配比较，D0、B=8、dev2000

| 配置 | 模型 | EM | substring | 备注 |
|---|---|---:|---:|---|
| fixed_adapter | C1 | 43.10 | 46.15 | 单一 LR |
| fixed_adapter | S | 37.50 | 该表未列 | 表中主比较采用 EM，避免跨报告混用 sub |
| shared_current，修复后 | S | 38.65 | 未在该表列出 | 与对应 C1 的 EM 差+4.70 |
| 固定 query adapter，decoder LR降低 | C1 | 41.40 | 未在该表列出 | 相对43.10为−1.70 EM |
| 同数据，无 KD | C1 | 43.10 | 46.15 | KD 主对照 |
| KD λ=0.5 | C1 | **44.10** | **48.00** | 相对无KD：+1.00 EM、+1.85 sub |
| KD λ=1.0 | C1 | 42.50 | 46.35 | 更高权重未改善 |
| KD λ=0.5 | S | 37.50 | 40.35 | EM不变 |
| 全量直接读取 | P | 54.50 | 59.80 | 80槽、reader已适配 |

fixed_adapter 下 C1−S = **+5.60 EM**。这是旧小预算方案的真实正结果；仍低于全量 P。KD 的 +1.00 EM 在该分析中 p=.23，substring +1.85 的 p=.022，分别保留指标结论。[E06]

### 6.2 固定 B 扫描

| C1 输出槽 B | EM | substring | 总 prefill 约值 |
|---|---:|---:|---:|
| 8 | 43.35 | 47.20 | 75 |
| 16 | 45.85 | 49.95 | 84 |
| 32 | 46.80 | 51.15 | 102 |
| P，全80槽 | 54.50 | 59.80 | 156 |

旧 S 预算曲线包括已经确认的绕行 bug，不能直接作为修正版主对照。C1 自身曲线作为历史记录保留。80→8 是 memory 数减少十倍，表中的总 prefill 只减少约2.08倍，未测端到端加速。[E05][E06]

### 6.3 问题和模板压缩

| decoder 输入 | EM | prefill约值 |
|---|---:|---:|
| D0，问题明文 | 43.10 | 75 |
| D0 + KD | 44.10 | 75 |
| D4，问题压缩、保留脚手架 | 32.90 | 59 |
| D4 + KD | 33.25 | 59 |
| D5，问题和模板进一步压缩 | 32.85 | 17 |
| D5 + KD | 32.45 | 17 |

D4 相对 D0 −10.20 EM；D5 相对 D4 −0.05 EM。这支持该版本中进一步删除脚手架的边际很小，不能外推为所有 prompt、模型和数据集均可删除 system 指令。[E06]

## 7. 全量 R：域内小幅正向、迁移未复现

### 7.1 已适配 P 上再次使用旧30k，dev2000

| 模型 | EM | F1 | substring |
|---|---:|---:|---:|
| 起点 P | 54.50 | 68.22 | 59.80 |
| R，冻结 decoder | 54.35 | 67.89 | 59.90 |
| R，joint | 52.80 | 66.80 | 58.50 |
| P继续训练，p-control | 52.90 | 66.73 | 58.15 |

Joint−p-control 为 −0.10 EM；此配置下追加模块没有净增量。[E07]

### 7.2 换至 P 未训练的60447条

| seed | R joint EM | p-control EM | 模块增量 |
|---|---:|---:|---:|
| 42，3000步 | 57.35 | 56.25 | +1.10 |
| 43，3000步 | 55.40 | 55.30 | +0.10 |
| 44，3000步 | 55.50 | 54.50 | +1.00 |
| 三seed平均 | 56.08 | 55.35 | **+0.73** |
| 42，9000步 | 56.05 | 56.10 | −0.05 |

seed42、3000步的 R F1=70.12，p-control=69.21；9000步分别70.06和69.61。总体 EM 正向观察不能从单次最优57.35归因全部+2.85给新模块；匹配增量为+1.10。增加至9000步没有保留这个 EM 增量。[E07]

### 7.3 TriviaQA 零额外训练迁移，2000题

| 模型 | EM | F1 |
|---|---:|---:|
| 源 P | 71.95 | 77.29 |
| R joint42 | 69.65 | 74.86 |
| p-control42 | 70.40 | 75.83 |
| R joint43 | 68.50 | 74.38 |
| p-control43 | 68.35 | 74.25 |
| R joint44 | 69.15 | 75.13 |
| p-control44 | 68.95 | 74.64 |

三个 seed 的模块 EM 增量为 −0.75、+0.15、+0.20，平均−0.13。该结果属于旧 R，**不是当前冻结两端 SQ 的迁移结果**。[E07]

## 8. Query 写回 RQ

| HotpotQA dev2000 | EM | F1 | substring |
|---|---:|---:|---:|
| 源 P / RQ恒等起点 | 54.50 | 约68.2 | 59.80 |
| RQ，冻结 decoder | 54.25 | 约68.0 | 59.35 |
| p-control | 56.25 | 约69.2 | 60.70 |
| RQ，joint | 56.05 | 约69.7 | 60.85 |

HotpotQA RQ−p-control 为−0.20 EM；TriviaQA2000题上 RQ70.50、p-control70.40，为+0.10 EM。两次边际均很小。R和RQ还存在 self-attention 路径差别，不能把二者差异纯归因于交换 attention 方向。[E08]

## 9. 几何、缩放和指令诊断

| 实验 | 数值结果 | 当前支持的结论 |
|---|---|---|
| SeleCom式冲突指令，40文档 | leading：raw65.0、memory22.5、错文档memory25.0 | 该 prompt 下 memory 的行为与 raw 有差别 |
| 重建，40文档 | ROUGE-L：raw91.6、memory61.6、错文档12.6 | 发布版memory包含可用文档信息 |
| 真实HotpotQA K10，200题 | 冲突指令leading：memory22.5、raw21.0 | 单文档效应不直接迁移至真实K10条件 |
| 域内decoder适配，dev2000 | substring48.50→59.90，F1 16.03→67.75 | 域内适配能改善当前任务输出 |
| α=.10 +匹配训练 | substring58.45，相对P₁59.90为−1.45 | 该缩放方案没有超过P₁ |
| α=.05 +匹配训练 | substring58.80，相对P₁为−1.10 | 同上 |
| 去语料均值，40文档 | leading37.5、重建ROUGE-L27.4；原memory22.5/61.6 | 指令行为与重建出现权衡 |
| 去前16个主方向，40文档 | leading52.5、ROUGE-L16.9 | 当前线性编辑破坏原reader可读性 |

后续修订撤回了“相对更新小=不参与计算”“逐层余弦连乘=首尾余弦”“编辑失败=信息消失/线性不可分”等强解释。这里保留实测现象，不采用已撤回的机制定论。[E09][E10][E11]

## 10. D0/D2、raw替换与失败位置

| 同reader、同段落截断，dev2000 | memory substring | raw substring | raw−memory |
|---|---:|---:|---|
| 域内 P₁ | 约60.0 | 约67.8 | +7.75，[5.95,9.60] |
| 发布版 PISCO | 约48.65 | 约60.7 | 约+12，[10.1,14.0] |

K2只取gold时，P₁仍有+7.2 substring差距，[5.6,8.9]。1193道“恰有一篇gold含答案”的 bridge 题中，只替换含答案段为raw：P₁+5.6，[3.3,7.9]，约恢复71%缺口；只替换桥接段+0.6，[−1.0,2.2]。这是 oracle 输入诊断，不是推理时可自动使用的解法。[E12]

D2 的 memory 位置可以看到前面的问题，记录了方向变化；功能干预在发布版和域内 P₁ 间并不一致。不能用拓扑允许或 hidden 改变代替最终 QA 增益。

## 11. Direct-State 和 W

| dev500，第一轮 | substring | F1 | 备注 |
|---|---:|---:|---|
| P₁ / Direct起点 | 64.0 | 69.5 | 同起点 |
| Direct-CE，3000步 | 60.0 | 报告未列完整数值 | 继续训练对照 |
| Direct-State，3000步 | 59.8 | 66.5 | 整向量三层cosine监督 |
| W，step0 | 27.8 | 未列 | 随机workspace没有文档通路 |
| W，最终 | 27.4 | 未列 | 未形成明显有效证据利用 |

低LR复验的仓库修订记录：300步对照中1e-5比1e-4稳定；低LR best 的 State−CE substring为−0.4pp，CI[−1.2,+0.4]；step500 State63.4、CE63.2。完整低LR曲线未提交，不能从记忆补成原始表格。W错配Z变化小，说明当前实现的证据特异性利用很弱。[E13][E14]

发布版重置与答案段教师序列审计的代码已经实现，但该分支没有提交真实模型的新 QA 成绩，状态为“效果待测”。[E14]

## 12. 共享投影器 SQ：目前的主方法结果

### 12.1 首轮，dev2000 / test前2000

| 模型 | dev EM | dev F1 | dev sub | test2000 EM | test2000 F1 | test2000 sub |
|---|---:|---:|---:|---:|---:|---:|
| 发布版PISCO | 1.75 | 16.03 | 48.50 | 0.90 | 16.41 | 49.90 |
| S0m | 50.15 | 63.74 | 54.85 | 50.95 | 65.01 | 55.30 |
| SQ | 50.05 | 63.63 | 54.85 | 51.60 | 66.15 | 56.30 |
| SQX | 50.60 | 63.91 | 55.25 | 52.00 | 66.71 | 56.35 |
| SL，最后一个query状态 | 50.20 | 63.63 | 54.45 | 51.15 | 65.66 | 55.85 |
| SW，词嵌入+位置 | 50.35 | 63.93 | 55.10 | 51.30 | 66.00 | 55.95 |

所有可训练臂均从发布版独立训练3000步，只训练投影器；encoder、decoder及发布适配器冻结。首轮各臂单seed42，最终memory仍80槽，生成上限32tokens。[E01]

### 12.2 首轮全量内部test5405

| 范围 | S0m F1 | SQ F1 | SQX F1 | SQ−S0m | SQX−S0m |
|---|---:|---:|---:|---:|---:|
| 全5405 | 65.37 | 65.99 | 66.21 | **+0.62** | +0.85 |
| 前2000 | 65.01 | 66.15 | 66.71 | +1.14 | +1.69 |
| 后3405 | 65.57 | 65.89 | 65.93 | +0.32 | +0.35 |
| bridge4296 | 63.99 | 64.31 | 65.05 | +0.32 | +1.06 |
| comparison1109 | 70.70 | 72.47 | 70.74 | +1.77 | +0.04 |

这轮后3405题仍有正向点估计；其区间较宽，不宜用“未显著”替代实际+0.32/+0.35数值。[E01]

### 12.3 标准 SQ/S0，独立训练，两seed

| 范围 | 模型 | seed | EM | F1 | substring |
|---|---|---:|---:|---:|---:|
| dev2000 | SQ | 42 | 49.85 | 63.63 | 54.95 |
| dev2000 | S0 | 42 | 49.95 | 63.69 | 54.50 |
| dev2000 | SQ | 43 | 50.30 | 63.88 | 54.90 |
| dev2000 | S0 | 43 | 49.65 | 63.15 | 54.35 |
| 内部test5405 | SQ | 42 | 51.19 | 65.52 | 55.97 |
| 内部test5405 | S0 | 42 | 50.90 | 65.21 | 55.54 |
| 内部test5405 | SQ | 43 | 51.38 | 65.62 | 56.06 |
| 内部test5405 | S0 | 43 | 51.10 | 65.20 | 55.80 |

配对主分析：test ΔF1 seed42+0.31、seed43+0.43，平均+0.37；ΔEM平均+0.29；Δsubstring平均+0.34。表中分数四舍五入后相减可能分别显示+0.42、+0.29等，优先沿用未四舍五入的配对统计。[E02]

Query替换时 decoder 仍得到正确问题：test正常−错问题 F1，seed42+0.45，[−0.02,+0.92]；seed43+1.16，[+0.62,+1.70]。S0预测保持不变。这支持 SQ 有问题依赖，但不能将错问题掉分全部等同于正确问题的净收益。

### 12.4 发布版到投影器的整体提升

首轮 dev substring48.50→SQ54.85，+6.35；test前2000 substring49.90→SQ56.30，+6.40。同期F1约16→64–66。

其中输出格式有明显变化：发布版平均生成约28.5tokens，72%达到32token上限；S0m平均4.1tokens。不能把约47–49个F1点全部讲成事实读取提升，也不能据此否定系统最终指标的实际改善。正式多数据集主表同时报告EM/F1和PISCO风格的Match/substring，并记录生成长度、EOS和截断比例。[E01]

## 13. 支持监督、输出监督和 FiLM

| 扩展 | dev2000主方法F1 | 匹配对照F1 | 匹配ΔF1 / CI |
|---|---:|---:|---|
| hidden支持监督SQ+Doc | 64.30 | SQ+Head64.22 | +0.07，[−0.16,+0.29] |
| 输出E支持监督SQ+DocE | 63.54 | SQ+HeadE63.55 | −0.01，[−0.35,+0.31] |
| FiLM vs AddG | 64.47 | 64.43 | +0.04，[−0.18,+0.26] |
| AddG vs同顺序G0 | 64.43 | 64.35 | +0.08，[−0.12,+0.29] |
| FiLM vs同顺序G0 | 64.47 | 64.35 | +0.12，[−0.18,+0.43] |

hidden头 Recall@2从20.0%升到55.2%，但换错误问题后为55.4%；前2篇集合86%不变。输出E头 Recall@2正常53.4%、错问题53.2%。这些辅助分类指标改善没有建立相应的正常QA增量。[E15][E16]

AddG的γ置零掉0.20 F1，[+0.01,+0.43]；FiLM的γ置零掉−0.01，[−0.50,+0.48]。这类固定checkpoint干预和独立训练增量的含义不同。主方法保留原SQ，暂不加入这几项扩展。[E17]

## 14. 投影器输入方向和范数归因

dev2000输入干预：original=Z；scale_only使用E范数与Z方向；direction_only使用Z范数与E方向；full=E。

| 模型 | original F1 | scale_only F1 | direction_only F1 | full F1 |
|---|---:|---:|---:|---:|
| SQ42 | 16.0 | 16.0 | 63.6 | 63.7 |
| S042 | 16.0 | 16.1 | 63.8 | 63.8 |
| SQ43 | 16.0 | 16.0 | 63.9 | 63.7 |
| S043 | 16.0 | 16.0 | 63.6 | 63.1 |

full−original约+47.09～+47.73；scale−original四组CI均含0；full−direction约−0.43～+0.11。S043中full−direction为−0.43，[−0.82,−0.08]，范数路径可有小幅负向响应。

当前checkpoint的整体改善主要由方向变化保留，不能写成“decoder整体严格尺度不变”或“向量方向改善已经证明事实恢复”。这一归因并未抹去SQ相对S0的小幅增量。[E18]

## 15. 最新 QER：pilot和内容监督三臂

### 15.1 数据与工程

原始训练集30000题，有序doc pool无多问池；无序pool的多问行占比0.29%，不同目标pool26个。新增2000对自然问题后共34000行；有序多问行占11.76%，2000对中1977对有不同目标。

可见目标：33782/34000；所有支持句可见：26554/34000；可见支持句72879/81139。128-token目标上限后，实际有效目标29804/34000（87.7%），过长3978，partial6723。

`954338f`修正 query strip与目标digest不一致，重新准备后34000行校验零失败。辅助前向10步预检完成，报告有限loss和梯度。以上都是此次真实服务器结果，覆盖率不再是未完成事项。[E19]

### 15.2 QA-only pilot，250步，dev2000

| 臂 | seed42 EM | seed42 F1 | seed43 EM | seed43 F1 |
|---|---:|---:|---:|---:|
| S0参考 | 49.95 | 63.69 | 49.65 | 63.15 |
| A-frozen，full | 49.95 | 63.70 | 49.80 | 63.30 |
| A-joint | 49.70 | 63.40 | 48.60 | 62.60 |
| A-first | 49.90 | 63.70 | 49.70 | 63.30 |
| A-slotwise | 50.15 | 63.80 | 49.90 | 63.40 |

full与first在显示精度下持平；full比slotwise两个seed均低0.10 F1。Joint相对frozen分别−0.30、−0.70。这支持在当前配置保留冻结策略，未证明跨槽重组具有收益，也未证明所有联合优化都会损坏S0。

原dev500 pilot中，step250−step0：frozen −0.19/+0.22，joint −1.12/−0.65，first −0.24/−0.06，slotwise +0.07/+0.41。dev500与dev2000分别记录，不混为同一个估计。[E19]

### 15.3 A/B/C，seed42，500步，dev2000

| 模型 | 真实query进入新readout | evidence权重 | EM | F1 | substring |
|---|---|---:|---:|---:|---:|
| S0参考 | 无新模块 | 0 | 49.95 | 63.69 | 54.50 |
| A | 是 | 0 | 49.75 | 63.70 | 54.50 |
| B | 是 | 0.1 | 49.70 | 63.50 | 54.45 |
| C | 固定条件向量 | 0.1 | 49.75 | 63.50 | 54.55 |

**B−A=−0.20 F1，B−C=0.00 F1；新增内容监督没有带来额外QA收益。** C的QA decoder仍收到真实问题，控制的是新readout中的问题路径。

dev500在step250：A64.74、B65.06、C65.13；step500：A64.32、B63.59、C63.59。最新报告另称B/C的best在dev2000均63.5，但未附完整best逐题输出，本次不自行补算置信区间。evidence teacher-forcing loss约0.81→0.52，说明辅助目标可以优化；单凭该loss不等于已经验证了自由生成证据的质量。

阶段决策：记录这组阴性结果，暂停QER完整3000步，保留原SQ主线。它否定的是这一轮新扩展在给定预算下的预期收益，不是否定原query投影器。[E19]

## 16. 数字与证据状态

1. 当前“test5405”由HotpotQA公开validation划出的内部留出集组成，不能标成官方隐藏test。它已多轮被观察；新数据集需预先固定训练/调参/测试用途。
2. 原报告在SQ43 test F1上显示65.62−65.20=0.42，但配对统计写+0.43，这是四舍五入精度差，不自行覆盖配对结果。
3. 输入归因报告的full绝对分与标准SQ/S0主表并非处处一致；分别保留，正式主表统一重评估后使用同一份输出，不能算成新增训练收益。
4. QER提交标题使用“joint破坏基座”；证据支持的是当前配置joint较差，不能证明唯一机制。
5. 最新方向归因报告中的“bridge无效、真实上限、query归因已确立”超出当前总体和子组证据。本文保留正向数值与区间，不采用这些过强表述。
6. 原数据审计中0.29%是无序pool多问行比例；另提55行不同目标，不应把55/30000也写成0.29%。两个统计应分开。

## 17. 多数据集扩展：从当前SQ开始

SeleCom论文覆盖NQ、TriviaQA、WebQuestions、PopQA、HotpotQA及FactKG；PISCO主实验覆盖NQ、TriviaQA、HotpotQA、ASQA、PopQA。两者指标体系也不同：SeleCom QA报告EM/F1及LLM judge，PISCO核心Match为归一化答案包含关系。

这提供多数据集和多指标评估的具体参考。本项目第一轮推荐五个短答案QA数据集，FactKG另属事实核验、ASQA另涉及长答案质量，暂不与短答案主表混合。下述安排是新实验计划，所有新结果均为TBD。[L01][L02]

| 数据集 | 本轮角色 | 数据/证据来源 | 初始动作 | 需明确的协议 |
|---|---|---|---|---|
| HotpotQA | 多跳、多证据 | 现有10段distractor文件与缓存 | 重用SQ/S0，补统一指标 | 当前test5405是内部划分；K10与top-1分开 |
| Natural Questions / NQ-open | 真实用户事实问答 | 官方短答案评估版本或SeleCom公开QDA | 建缓存，先评现有权重 | 确认NQ-open版本、split及别名 |
| TriviaQA | 事实型、丰富别名 | 服务器已有评估文档或公开QDA | 优先评当前SQ/S0 | 旧R结果不能代替新SQ；保留所有gold aliases |
| PopQA | 低频实体/事实 | 官方评估版本或公开QDA | 建缓存，跨数据集评估 | 保留完整答案集合；不按模型表现筛样本 |
| WebQuestions | 问题表达与答案集合 | 官方评估版本或公开QDA | 建缓存，跨数据集评估 | 明确数据版本与多答案评分口径 |

SeleCom官方`data/eval/README.md`列出了六个QDA JSONL文件，链接指向`Ryan7458/Eval_QDA`；本次已核对README，尚未下载并检查该发布数据的实际行数和样本内容。官方README明确指出其HotpotQA文件用于top-k=1，每题只有一个文档。**不能把它当成当前HotpotQA十段distractor，也不能在单文档文件上写“top-5多跳评估”。** 官方Mistral版selector和generator目录也已在其模型发布页面核实。[L03][L04]

开放域QA优先复用公开固定文档池，避免先花预算重建整套检索索引。若公开文件不可取得，使用已经准备好的官方数据与统一retrieval产物，并记录检索模型、语料版本、chunk和top-K。同一数据集内所有方法读取完全相同的文档列表。

## 18. 两阶段扩大实验，控制训练数量

### 阶段一：现有模型的跨数据集评估

固定当前SQ42、SQ43、S042、S043，不训练新投影器。在NQ、TriviaQA、PopQA、WebQuestions评估相同四个checkpoint，共16个模型×数据集评估任务；Hotpot已有结果按统一协议复核。另评发布版PISCO和原文RAG作为系统参照。

这一阶段回答“仅在Hotpot训练的投影器迁移到其他任务后表现如何”。它是投影器的零额外训练迁移，不是“所有底座从未见过这些数据”的严格零样本声明。四个新数据集全部报告，保留正负结果。

### 阶段二：统一混合训练，形成正式主表

推荐固定一个30k短答案混合训练集：Hotpot、NQ、TriviaQA各10k，均来自各自train，去除与调参/评估题的ID和规范化问题重叠；同时记录答案别名和文档来源。PopQA/WebQuestions不进入这轮投影器训练，作为投影器层面的跨任务评估。

| 设置 | 推荐值 |
|---|---|
| 主模型 | 原共享SQ，不加入QER、支持头或FiLM |
| 主要消融 | S0文档MLP；S0m固定条件作为补充等参数对照 |
| 起点 | **发布版PISCO重新初始化SQ/S0，不从Hotpot last续训** |
| encoder / decoder / LoRA | 均冻结；仅更新投影器 |
| 每篇缓存 | 发布版PISCO128正文token、m8、d4096 |
| 主训练预算 | 3000步、有效batch16，每臂48k样本呈现 |
| 初始优化配方 | 沿用已工作SQ配方：AdamW、5e-5、5%warmup、线性衰减 |
| seed | 42、43、44；SQ/S0各三seed，共六次训练 |
| checkpoint | 预先固定last作主要归因；best按同一调参规则补充 |
| 数据集覆盖 | 一个统一checkpoint分别评五个QA数据集 |

这份30k混合配方是拟议设置，不是已完成结果。3000步对应48k样本呈现，约1.6遍30k。若训练集扩大到90k并希望保持相同遍历率，应将两臂同步增至约9000步；训练数据扩大和数据集评估覆盖扩大是两个独立选择，不默认全部同时加倍。

同seed的共享参数初始化与训练样本顺序需相同。SQ新增模块消耗随机数时，应通过独立DataLoader RNG或明确匹配初始化序列避免改变批次顺序；记录全程样本顺序摘要，step0相等或step1 loss相等不能代替完整顺序核验。

阶段一与阶段二分表。阶段二的混合训练不能仍写成“Hotpot-only迁移”，也不能把数据扩大收益全部计给query模块。

## 19. 主表与基线矩阵

| 方法 | 要回答的问题 | 对照/实现要求 |
|---|---|---|
| 原文RAG，同Mistral底座 | 质量与在线处理成本参照 | 官方原文接口；记录adapter、prompt和可见证据，不能默默套压缩模板 |
| 发布版PISCO，原生接口 | 原系统水平 | 使用正式发布checkpoint、固定缓存和native prompt |
| PISCO，统一短答指令 | 简单prompt适配能解释多少涨幅 | 只改问题指令，不训练；与native结果并列 |
| S0 | 通用文档投影收益 | 与SQ同数据、预算、seed和共享参数起点 |
| **SQ** | query条件投影的额外作用 | 主要比较SQ−S0；保持原始方案 |
| S0m | query信号还是参数容量 | 固定条件、相同模块；建议在Hotpot/NQ补充 |
| SeleCom-Mistral | query相关处理放在压缩前后的系统区别 | 使用公开Mistral selector+generator；重新运行共同数据，不拷论文数字 |

至少先完成PISCO/S0/SQ/原文RAG的五数据集矩阵；SeleCom优先在NQ与Hotpot跑通，再扩其余数据集。COCOM/xRAG可用于补充相关基线，但不同公开checkpoint、单文档限制和memory预算应注明，不用修改版COCOM充当官方系统。

**同可见证据要落到真实encoder输入上。**当前缓存把`<ENC><bos>doc<eos>`右截到131tokens，对照应保存实际可见的正文前缀：原文RAG和共同证据版SeleCom只读取同一前缀。若另跑SeleCom/原文的完整文档版本，作为明确标记的完整证据系统结果；不同可见内容不用于隔离readout贡献。

不同压缩方法可使用不同原生memory数，但主表必须写出槽数和实际压缩率，并给质量成本图；不能把PISCO每篇8槽与SeleCom每篇2槽称作相同memory预算。当前SQ/S0之间则严格保持同槽数。

| 方法 | Hotpot F1/EM/Match | NQ | TriviaQA | PopQA | WebQuestions |
|---|---|---|---|---|---|
| PISCO native | TBD | TBD | TBD | TBD | TBD |
| PISCO short-answer prompt | TBD | TBD | TBD | TBD | TBD |
| S0，混合训练，3seed | TBD | TBD | TBD | TBD | TBD |
| SQ，混合训练，3seed | TBD | TBD | TBD | TBD | TBD |
| SQ−S0 | TBD | TBD | TBD | TBD | TBD |
| 原文RAG，共同可见证据 | TBD | TBD | TBD | TBD | TBD |
| SeleCom，共同可见证据 | TBD | TBD | TBD | TBD | TBD |

新主表全部为TBD；旧Hotpot分数不直接搬入改变训练集、文档池、指令或生成预算后的新表。

## 20. 指标、统计与输出格式

每个QA数据集同时报告EM、token F1、归一化Match/substring。不同数据集的别名和多答案集合采用各自公开评估口径；若额外使用统一alias-max指标，应明确标签，避免把WebQuestions多答案集合问题悄悄改成任选一个答案。

PISCO论文的Match与现有`src/metrics.py`的substring思想一致，精确归一化实现仍需在新协议中固定。所有方法共用同一评分入口。保留平均答案token数、EOS率、达到生成上限比例，评估内容正确与输出格式两个层面。[L02]

历史主表上限32tokens继续保留为旧协议。新评估推荐统一贪心解码、上限128tokens，在调参集上先锁定所有方法的prompt和答案预算；与历史32tokens的结果分表。达到上限的回答比例可解释长度影响，不在看过test之后专门为某臂改上限。

按seed列出SQ−S0，再报告三seed均值和标准差；同题配对bootstrap报告题目抽样区间。训练seed波动与题目抽样不确定性分别报告，不能把同一批问题×三个seed当成三倍独立样本。主要比较预先固定为每个数据集的SQ−S0 F1，同时完整保留EM/Match。

全量公开可评估split优先于前500题。若公开split没有单独test，应从train建立调参集合，固定公开dev只作最终评估；不同发布文件已有dev/test时保留原定义。旧Hotpot/Trivia反复使用过的集合标为历史开发证据，不能重新宣称完全未触碰。

## 21. 复用效率与成本实验

新增投影器不减少80槽，SQ也不预设比S0或缓存PISCO更快。优势候选是保持cache可复用，同时提高正常QA质量；相对每问题读取原文的selector，是否更省成本由测量决定。

| 测量 | 需要计入 |
|---|---|
| 离线成本 | 文档去重、PISCO编码、缓存写入、存储大小 |
| 在线TTFT/总延迟 | 读取缓存、question-only编码、投影、decoder prefill与生成 |
| GPU显存 | 同硬件、精度和批量下的峰值 |
| 多问题复用 | 固定相同文档池，1/2/4/8/16个真实问题；每个Z只编码一次 |
| 累计/平均成本 | 冷启动和已有缓存两种情况分别报告 |

自然同文档问题池优先使用真实数据。QER准备的配对训练题不能冒充未见评估样本；新增复用评估从留出数据形成，记录文档池和问题数。

令共享文档池的离线编码成本为C_off，SQ每问完整在线成本为C_SQ，selector每问完整成本为C_sel，则复用r问的成本为C_off+r·C_SQ与r·C_sel。只有实测C_sel>C_SQ时才有正的摊薄阈值r*=C_off/(C_sel−C_SQ)。cache PISCO/S0同样共享C_off，因此SQ相对它们的在线增量必须单独展示。

## 22. 执行优先级与最小交付

1. **保存现有结果，锁定SQ/S0身份。** QER停留在已完成记录；不再把它加入多数据集主方法。
2. **先评当前SQ/S0的TriviaQA与NQ。** 已有权重直接使用；再扩PopQA/WebQuestions，四个新数据集全部报告。
3. **完成统一混合数据和六次SQ/S0主训练。** 采用同一固定配方，而不是每个数据集另设计一个模块或反复找最佳条件。
4. **补PISCO native、短答prompt、原文RAG和SeleCom。** 统一文档池/可见证据/评分；保留原生系统与共同证据比较的区别。
5. **完成复用延迟与存储曲线。** 至少Hotpot与NQ两种工作负载，避免只用soft-token数量代替效率。
6. **形成论文材料。** 五数据集主表、SQ−S0消融、质量成本表、真实多问题复用图；阴性扩展放入附录或开发记录。

论文中心表述可围绕：**可复用的文档软缓存、缓存之后的显式query条件残差投影、冻结两端的轻量适配，以及跨数据集质量与多问题成本。** 不需要靠扩大模块复杂度体现创新，也不把当前约半个点的query增量抹去。实验规模扩大后，根据实际正负结果决定主张范围。

## English summary

The latest QER extension yields no additional QA gain: on dev2000, A/B/C score 63.70/63.50/63.50 F1 after 500 steps. This does not erase the original shared projector's positive query increment: +0.62 F1 against the fixed-condition S0m in the first test run, and +0.37 F1 against independently trained document-only S0 across two seeds. Keep SQ as the main method and expand evaluation to HotpotQA, NQ, TriviaQA, PopQA, and WebQuestions. First evaluate the existing checkpoints across datasets, then train matched SQ/S0 models on one fixed mixed training set. Report EM, F1, Match, generation length, and measured cache-reuse costs under explicit evidence and retrieval protocols.

## Sources

Source references are appended below. Experimental values come from repository reports; the new multi-dataset tables are plans, not observed results.


- [E01] SHARED_QUERY_PROJECTOR_RESULTS.md（`69001c3`）。
- [E02] QUERY_PROJECTOR_ABLATION_RESULTS.md（`69001c3`）。
- [E03] QURO_V0.1_IMPLEMENTATION_PLAN.md（`65895d3`）。
- [E04] QURO_V0.2_RESULTS_AND_ANALYSIS.md（`65895d3`）。
- [E05] ARM_MATRIX_RESULTS.md（`65895d3`）。
- [E06] TRAINING_RECIPE_RESULTS.md（`65895d3`）。
- [E07] RESIDUAL_RESULTS.md（`f9c7935`）。
- [E08] QUERY_WRITEBACK_EXPERIMENT.md（`0f03522`）。
- [E09] FULL_COMPRESSION_INFEASIBILITY_RESULTS.md（`7f477eb`）。
- [E10] LATENT_CONTEXTUALISATION.md（`7f477eb`）。
- [E11] LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md（`7f477eb`）。
- [E12] READER_CAUSAL_ORDER_RESULTS.md（`7f477eb`）。
- [E13] READER_RECOVERY_RUNBOOK.md（`34828e9`）。
- [E14] READER_RESET_REVIEW.md（`34828e9`）。
- [E15] SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md（`69001c3`）。
- [E16] SUPPORT_OUTPUT_SUPERVISION_RESULTS.md（`69001c3`）。
- [E17] QUERY_MODULATED_FUSION_RESULTS.md（`69001c3`）。
- [E18] PROJECTOR_INPUT_ATTRIBUTION_RESULTS.md（`69001c3`）。
- [E19] QUERY_GUIDED_EVIDENCE_READOUT_RESULTS.md（`d85eb0c`）。
- [L01] SeleCom 原论文。
- [L02] PISCO 原论文。
- [L03] SeleCom 官方评估数据说明。
- [L04] SeleCom 公开模型目录。

[E01]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/69001c3efab20aba7229e3920133cc4e97e6ab9e/docs/SHARED_QUERY_PROJECTOR_RESULTS.md
[E02]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/69001c3efab20aba7229e3920133cc4e97e6ab9e/docs/QUERY_PROJECTOR_ABLATION_RESULTS.md
[E03]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/65895d3170fa6b23c9fc64e3732318d987201e55/docs/QURO_V0.1_IMPLEMENTATION_PLAN.md
[E04]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/65895d3170fa6b23c9fc64e3732318d987201e55/docs/QURO_V0.2_RESULTS_AND_ANALYSIS.md
[E05]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/65895d3170fa6b23c9fc64e3732318d987201e55/docs/ARM_MATRIX_RESULTS.md
[E06]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/65895d3170fa6b23c9fc64e3732318d987201e55/docs/TRAINING_RECIPE_RESULTS.md
[E07]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/f9c79356afe749a645574dc145640fb5ca9ddf33/docs/RESIDUAL_RESULTS.md
[E08]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/0f03522b6e1db4a75cb911bd5abc80959e7c25b6/docs/QUERY_WRITEBACK_EXPERIMENT.md
[E09]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/FULL_COMPRESSION_INFEASIBILITY_RESULTS.md
[E10]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/LATENT_CONTEXTUALISATION.md
[E11]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/LATENT_CONTEXTUALISATION_WARNING_AND_NEXT_STEPS.md
[E12]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/7f477ebfe7f7152627435be0bd8bd5eec76a2d94/docs/READER_CAUSAL_ORDER_RESULTS.md
[E13]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/34828e98a7a587e331b230d33ef6bc1096b2a366/docs/READER_RECOVERY_RUNBOOK.md
[E14]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/34828e98a7a587e331b230d33ef6bc1096b2a366/docs/READER_RESET_REVIEW.md
[E15]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/69001c3efab20aba7229e3920133cc4e97e6ab9e/docs/SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md
[E16]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/69001c3efab20aba7229e3920133cc4e97e6ab9e/docs/SUPPORT_OUTPUT_SUPERVISION_RESULTS.md
[E17]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/69001c3efab20aba7229e3920133cc4e97e6ab9e/docs/QUERY_MODULATED_FUSION_RESULTS.md
[E18]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/69001c3efab20aba7229e3920133cc4e97e6ab9e/docs/PROJECTOR_INPUT_ATTRIBUTION_RESULTS.md
[E19]: https://github.com/lixingyi701/quro-Query-conditioned-Readout-over-precomputed-context-embeddings/blob/d85eb0cc46ad7b3ff1b671adbf2d9fd9fb9170a6/docs/QUERY_GUIDED_EVIDENCE_READOUT_RESULTS.md
[L01]: https://arxiv.org/html/2602.15856v1
[L02]: https://aclanthology.org/2025.findings-acl.800/
[L03]: https://github.com/yhliu7458/SeleCom/blob/main/data/eval/README.md
[L04]: https://huggingface.co/Ryan7458/Selecom/tree/main
