# 从已训练 P 基线出发的残差实验

当前阶段只争取正常输入下的质量提升。保留全部缓存 latent、原始顺序、D0 明文问题和 PISCO 模板；不二次压缩、不使用 KD、不压缩问题、不添加覆盖监督。此版本不声称压缩或加速。

## 模块

新增 R 臂（`pisco_residual`）：

```
Z: (batch, K*m, h)                      原始 P latent，恒等旁路
Q: (batch, query_len, hq)               冻结 query 编码器
U = Linear(LayerNorm(Z))               默认宽度 256
V = Linear(LayerNorm(Q))
H = QueryCrossAttention(U, V)          每个原始 latent 读取问题
H = LatentSelfAttention(H, mask)       修正分支读取其他文档的 latent
Delta = W_out LayerNorm(H) + b_out
E = Z + Delta
```

两个 attention block 均含 pre-LN、FFN 与内部残差。`W_out` 与 `b_out` 为零初始化；恒等旁路没有 norm、缩放或投影。无需再把 gate 置零：初始输出严格为 Z，第一步输出投影有梯度，后续梯度进入内部条件化网络。

修正由答案 CE 学习，没有人工给定的“正确残差”。这借鉴残差参数化的优化思路，并不证明旧模型的问题来自深度退化，也不保证训练后一定优于 P。

R 和 P 都传回全部有效 latent；`budget` 不截断它们。结果中的 `budget_semantics=all_cached_latents` 和实际输入长度优先于 B 标签。Hotpot K=10、m=8 时最多 80 个有效 latent。

## 必须从实际 P checkpoint 起步

`--baseline_run` 读取指定 P 运行的配置和 `checkpoint_last.pt`，加载训练过的 decoder 权重，不加载旧 optimizer/step。默认固定 query 表示、冻结 decoder，只训练修正模块，CE 的梯度仍经过 decoder 回传到输入。默认不添加旧 readout 的残差惩罚。

若源 checkpoint 保存过固定 query adapter，复用该 snapshot；否则从刚加载的 P decoder 复制。新的 checkpoint 同时保存冻结 P decoder 和 query adapter，避免重载后退回公开权重。

每次 R 初始化自动用两条训练样本验证：latent、mask、答案位置 logits 和 CE 与同一 decoder 下的 P 逐项相同，否则停止。结果保存为 `baseline_identity.json`。这是小批实现检查，不是完整 Hotpot EM 复现。

## 在服务器执行

先激活已有 PyTorch/PISCO 环境。默认源是 `/data02/quro/runs/hp2d0_P`，可用 `BASELINE_RUN` 覆盖；每次使用新的 TAG，脚本拒绝覆盖历史结果。保持源模型权重文件与缓存不变。

```bash
# 1. 完整 dev 上评测零修正模型，核对原 P 的 54.50 EM
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pisco_residual.sh init

# 2. 冻结这个已训练 P 的 decoder，训练新增分支
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pisco_residual.sh train
```

第一项继承源运行的评测数据；若源配置、权重、tokenizer、样本或生成设置不同，不能期待重现 54.50。先核对预测及配置，不把差异归因于残差。

第二项的完整评测默认仍是 last checkpoint，间隔 500 条 dev 子集只用于保存 best。不要把子集最优值作为主结果。训练会在非有限 loss/梯度时停止，防止静默污染参数。

如冻结 decoder 的优化受限，允许再做联合适配，但必须给 P 相同的额外训练预算：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pisco_residual.sh joint
CUDA_VISIBLE_DEVICES=1 bash scripts/run_pisco_residual.sh p-control
```

两者均从同一 P checkpoint 开始，共用源数据/训练 batch 配置、步数和学习率；模块构建后重置 seed，避免初始化耗用随机数改变样本顺序。联合适配若只超过旧 P、却不超过同样继续训练的 P，不能将提升归因于 R。

重新评测训练后的 R（恢复已保存的完整配置和冻结 decoder）：

```bash
python -m src.train \
  --config_json /data02/quro/runs/residual_train_s42/config.json \
  --resume_from /data02/quro/runs/residual_train_s42/checkpoint_last.pt \
  --out_dir /data02/quro/runs/residual_train_s42_recheck \
  --eval_only --eval_input_modes D0 --eval_budgets 80
```

当前环境只运行 CPU 契约与 toy 流程；真实 7B/Hotpot 训练需要服务器上的权重、缓存和 GPU。先实现初始平齐，再检验训练后的提升；有正结果后再讨论预算缩减和消融。

## KL 修复的位置

此 CE 实验不进入 KD 路径，因此 KL 尾部数值修复不是前置依赖。现有 `src/distill.py` 的风险仍在，不应直接恢复 KD。将来启用前再修复尾部计算并验证缓存、loss 和梯度有限性。

## CPU 验证

```bash
OMP_NUM_THREADS=1 python tests/test_refinement.py
OMP_NUM_THREADS=1 python tests/test_shapes.py
```

覆盖零初始化恒等、真实 latent 数/顺序、padding 隔离、修正分支解锁梯度、非零修正后的 query 敏感性、已训练 P 权重移植、冻结 decoder 不更新，以及 R checkpoint 保存恢复的 logits 一致性。
