# 残差臂 R：从已训练 P 出发的恒等初始化修正

> 日期：2026-09-18
> 数据：[`results/arm_matrix.json`](../results/arm_matrix.json)（41 次运行、29 组配对检验）
> 生成：`python scripts/collect_arm_matrix.py`
> 范围：HotpotQA dev 2000 条，D0，B=80（全部缓存 latent，不截断），3000 步，seed 42。**全部单 seed。**
> 实现：分支 `feat/pisco-identity-residual`，方案见 [`PISCO_RESIDUAL_EXPERIMENT.md`](PISCO_RESIDUAL_EXPERIMENT.md)、[`RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md`](RESIDUAL_DIRECTION_HANDOFF_2026_09_18.md)

本轮要回答的是交接文档定下的首要问题：**从实际 P baseline 出发，学一个修正，能不能超过 P。**
答案是不能。以下是四次运行的完整记录。

---

## 0. 三十秒版本

1. **恒等起点精确成立。**`Delta=0` 时 R 与 P 在 dev 2000 条上**逐题 2000/2000 完全相同**，EM/F1/substring 三项全等。实现无误。
2. **训练后没有收益。**R（冻结 decoder）54.35 vs P 54.50，**p=0.79，与 P 无法区分**。全程只改变 59/2000 道题的预测，赢 28 负 31。
3. **联合适配下残差模块贡献为零。**算力匹配的 joint vs p-control：−0.10 EM，p=0.94。这是本轮唯一公平的对照，结果是彻底的零。
4. **最有信息量的是 p-control：继续训练本身有害。**什么模块都不加，只把 P 的 decoder LoRA 在同一批数据上再训 3000 步，EM 掉 1.60（p=0.044）。joint 掉 1.70（p=0.027）。
5. **所以瓶颈不在 R，在数据配方。**P 已在这 30000 条上训到收敛，继续训是在拟合噪声。R 在冻结设定下能打平，是因为冻结保护了它。
6. **「冻结限制太大、要解冻给优化空间」这条假设被否掉了**——解冻带来的是退化，不是空间。

---

## 1. 四次运行

| 运行 | 阶段 | readout | decoder LoRA | 可训参数 |
|---|---|---|---|---:|
| `residual_init_s42` | 步骤 B | 4.09M `pisco_residual` | 冻结 | 0（`--eval_only`） |
| `residual_train_s42` | 步骤 C | 4.09M `pisco_residual` | **冻结** | 4.09M |
| `residual_joint_s42` | 步骤 D | 4.09M `pisco_residual` | 41.94M | 46.03M |
| `residual_p-control_s42` | 步骤 D 对照 | 0（`pisco_direct`） | 41.94M | 41.94M |

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pisco_residual.sh init
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pisco_residual.sh train
CUDA_VISIBLE_DEVICES=1 bash scripts/run_pisco_residual.sh joint
CUDA_VISIBLE_DEVICES=2 bash scripts/run_pisco_residual.sh p-control
```

四者都从 `/data02/quro/runs/hp2d0_P/checkpoint_last.pt` 加载同一个已训练 decoder，共用源运行的数据、缓存、tokenizer 和生成配置，query adapter hash 全部为 `17fc60e8afb48424`。

**`joint` 与 `p-control` 只差那 4.09M 残差模块**：同样 41.94M 的 decoder LoRA 预算、同样 3000 步、同样 lr 1e-4、同样 seed。这是本轮设计里唯一能把收益归因给 R 的那一组。

---

## 2. 恒等复现（步骤 B）

```
[baseline identity] {'identity': True, 'ce': 0.003759278915822506, 'tokens_per_row': [80, 80]}
[eval] dev|D0|B=80: EM=54.50% F1=0.682 sub=59.80% prefill=156 tok
```

| | EM | F1 | substring |
|---|---:|---:|---:|
| `hp2d0_P`（源） | 54.50 | 0.6822 | 59.80 |
| `residual_init_s42` | 54.50 | 0.6822 | 59.80 |

逐题比对：**2000/2000 预测字符串完全一致**，配对检验 net 0（+0/−0）。

小批恒等检查（2 条样本上 latent、mask、logits、CE 逐项相同）与完整 dev 复现两者都做了，后者才是真正的证据。

---

## 3. 主结果

| 运行 | 臂 | EM | F1 | substring | vs P（EM） | 配对检验 |
|---|---|---:|---:|---:|---:|---|
| `hp2d0_P` | P | **54.50** | 0.6822 | 59.80 | — | — |
| `residual_init_s42` | R (Δ=0) | 54.50 | 0.6822 | 59.80 | +0.00 | 2000/2000 相同 |
| `residual_train_s42` | R | 54.35 | 0.6789 | 59.90 | −0.15 | +28/−31，**p=0.79** |
| `residual_joint_s42` | R | 52.80 | 0.6680 | 58.50 | −1.70 | +95/−129，p=0.027 |
| `residual_p-control_s42` | P | 52.90 | 0.6673 | 58.15 | −1.60 | +103/−135，p=0.044 |

**关键对照：**

| 对比 | ΔEM | net | p |
|---|---:|---:|---:|
| `joint` vs `p-control`（算力匹配） | **−0.10** | +83/−85 | **0.94** |

检验为精确二项 McNemar，由 `scripts/collect_arm_matrix.py` 生成。

---

## 4. 训练曲线

500 条 dev 子集，仅用于选 best；完整结果评的是 last。

| step | `train`（冻结） | `joint` | `p-control` |
|---:|---:|---:|---:|
| 0 | 56.40 | 56.40 | 56.40 |
| 500 | 54.60 | 51.00 | 50.20 |
| 1000 | 55.20 | 53.00 | 52.40 |
| 1500 | 54.80 | 53.20 | 53.60 |
| 2000 | 55.40 | 54.00 | 54.80 |
| 2500 | 55.40 | 53.40 | 54.40 |
| 3000 | **55.80** | 53.60 | 54.20 |
| best | step 3000（=last） | step 2000 | step 2000 |

两条解冻的曲线是同一个形状：**先塌 5~6 分，再往回爬，到 3000 步仍未回到起点。**这不是"收敛得更好"，是扰动后的不完全恢复。

`joint − p-control` 逐点为 +0.8 / +0.6 / −0.4 / −0.8 / −1.0 / −0.6，**符号在中途翻转**，与全量评测的 −0.10、p=0.94 一致：噪声。

---

## 5. 陷阱：子集不能当全量读

**dev 前 500 条比全量容易 1.9 分。** P 在同一子集上是 **56.40**，全量是 54.50。

`src/data.py:118` 是 `rows = rows[:limit]`，`[val]` 取的是确定的前 500 条前缀，所以这个换算是可复核的：把 init 那份 2000 条逐题预测取前 500 条重算即得 56.40。

后果很实际：`train` 在 step 3000 的子集 EM 是 **55.8**。若拿它对 54.50 比，会得出"已经超过 baseline +1.3 分"的结论；对着同子集的 56.40 比，真相是 **−0.6**；而全量真值是 **−0.15，不显著**。

**任何 `[val]` 的数都不能直接和 `[eval]` 的数比较。**

另：`joint` 与 `p-control` 的 best（step 2000）≠ last（step 3000），上表报告的 52.80 / 52.90 均为 last，按交接文档要求。`train` 的 best 恰在 step 3000，best == last。

---

## 6. 诊断：模块确实训起来了

失败模式要先排除"梯度没到"。检查 `residual_train_s42/checkpoint_last.pt`：

| 张量 | 初始 | 训练后 |
|---|---|---|
| `readout.out_proj.weight` | 严格 0 | L2 = **2.578**，absmax 0.0172 |
| `readout.out_proj.bias` | 严格 0 | L2 = 0.201 |
| 全部 43 个张量（4.09M） | — | **43/43 非零** |

输出投影从零初始化离开了，冻结的 decoder LoRA 448 个张量原样持久化。所以这不是实现缺陷，是**学到的修正确实存在，但没有用**：它只动了 59/2000 道题，动了的部分输赢各半。

---

## 7. 这一轮排除了什么

- **恒等残差参数化本身不足以带来收益。**保留 P 的每一个原始 latent、零初始化输出、CE 端到端学修正——这套做法在这个配方下的结果是精确的打平。
- **解冻 decoder 不是出路。**p-control 证明额外训练预算在这批数据上是负的，joint 拿到同样预算后也没能反超。
- **不能声称"平齐即成功"。**交接文档的目标是超过 P，没有达到。R 与 P 无法区分不是胜利，只是没有退化。

**没有测过的不要写：**本轮未跑错配文档控制、未跑多 seed、未测在线时延、未做预算缩减。KD 路径未启用，`src/distill.py` 中 FP32 `1-1e-9` 舍入至 1 的尾部问题**仍未修复**。

---

## 8. 下一步该换的变量

不是 R 的超参（宽度、blocks、lr）。p-control 把问题定位在模块之外：在一个本身向下的优化面上调模块容量调不出东西。

该动的是数据与目标那一层。可用资源的实际盘点见 [`TRAINING_DATA_REDESIGN.md`](TRAINING_DATA_REDESIGN.md)。

---

## 附：产物

```
/data02/quro/runs/residual_init_s42/
/data02/quro/runs/residual_train_s42/
/data02/quro/runs/residual_joint_s42/
/data02/quro/runs/residual_p-control_s42/
```

每个目录含 `config.json`、`commit.txt`、`worktree.diff`、`console.log`、`train_log.jsonl`、逐题 `predictions_dev_D0_B80.json`、`result.json`；三个训练运行另含 `checkpoint_last.pt` / `checkpoint_best.pt`（冻结的 P decoder 与 query adapter 一并保存，重载不会退回公开权重）。`init` 与 `train` 另含 `baseline_identity.json`。
