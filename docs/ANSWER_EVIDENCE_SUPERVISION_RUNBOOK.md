# 含答案段监督：先审计目标，再决定是否训练

研究依据与限制以 [修订复审](READER_RESET_REVIEW.md) 为准。
这不是“换个结构再试一遍”的执行表。旧四臂300步/1pp门槛已撤回，不默认运行 direct-read/direct-mlp。

## 1. 输入与起点

使用已验证的发布版 PISCO 本地快照和同源128→8缓存。`--init_source published` 不载入P₁；不要传 `--init_checkpoint`。外部backbone若不在发布快照目录，另行固定版本并记录。

先使用 `prepare_reader_reset.py` 对项目旧30k、full训练池、全部已用dev/最终留出query做 ID 或归一问题去重，并从未见训练池抽tune：

```bash
# 以下为已有服务器目录布局示例。先按实际路径设置；不会下载或重建缓存。
RESET=/data02/quro/runs/answer_evidence_v1
GEN=/data02/quro/models/pisco-mistral
CACHE=/data02/quro/cache/hotpotfull-pisco-r16
SEEN=/data02/quro/data/hotpot/train.jsonl
POOL=/data02/quro/data/hotpot_full/train.jsonl
DEV=/data02/quro/data/hotpot/dev.jsonl
TEST=/data02/quro/data/hotpot/test.jsonl
# corpus是doc_id/text的JSONL，不是query训练文件；可传多份。
CORPUS=/data02/quro/data/hotpot_full/corpus.jsonl

python scripts/prepare_reader_reset.py \
  --seen "$SEEN" --pool "$POOL" --exclude "$DEV" "$TEST" \
  --cache_manifest "$CACHE/manifest.json" --tune_size 1000 \
  --out_dir "$RESET/data"
```

有其他dev/test文件需全部追加；这里只读query作排除，不评测最终留出集。
语义近重复和发布版上游训练重叠不在此审计的保证范围。共享文档会记录，不自动删除。
缓存若未覆盖full中的新题，先处理这一实际缺失，不能偷偷换回旧dev做tune。

数据准备只创建split，不表示要跑发布版/P₁×新旧数据四格实验。当前主训练文件为清理后的 `old_train.jsonl`，从发布版首次做项目域适配。

## 2. 默认唯一动作：不更新参数的目标审计

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/build_answer_evidence_targets.py \
  --mode audit --generator_path "$GEN" --cache_dir "$CACHE" \
  --train_file "$RESET/data/old_train.jsonl" \
  --exclude_files "$RESET/data/tune.jsonl" "$DEV" "$TEST" \
  --corpus "$CORPUS" --max_docs 10 --batch_size 4 \
  --max_new_tokens 64 --limit 512 --seed 42 \
  --out_dir "$RESET/teacher_audit"
```

随机抽训练题，而非前512道dev。教师固定发布版decoder，M/R所有题、A/B角色明确bridge题；保持文档顺序和每段128tokens。
默认训练目标长度上限沿用48，不静默截断教师答案：无EOS或超过48tokens的输出不成为SKD目标。若真实输出大量超限，先说明覆盖率问题；需要改长度时必须将教师/学生/gold控制一起改，并重新生成manifest，不能只改一个参数冒充同一实验。

输出：

- `teacher_audit.jsonl`：问题、gold、角色、四视图自由输出与评分、gold逐token NLL、EOS及目标token序列。
- `summary.json`：M/R/A/B与 R−M、A−M、B−M、A−B、A−R 的配对结果，包含内容NLL；每项都保留样本数。A/B相关比较只用共同角色子集。
- `manifest.json`：输入/权重/缓存指纹、模型版本、完成标志。`mode=audit` 不能被训练消费。

查看：`raw_eligible`、`matched_eligible`、`raw_targets_token_different_from_gold`、`matched_targets_token_different`。
例：raw生成答对率高但 `raw_targets_token_different_from_gold=0`，意味着这条序列目标与gold等价，不能声称已引入更强监督。
非零也不等于有价值：查看A修复M、R/A相反判断、输出更长但EM变差的题，检查是否有无依据解释。无需重跑位置实验。
512题不作为训练效果门槛，也不是Z可恢复性的证明。脚本只报告，不自动决定扩展。

## 3. 目标质量支持时，才导出完整训练目标

没有自动把audit升级为export；export成本包含全训练集的四视图前向/生成，应基于审计实测耗时估算。本地未给出GPU耗时保证。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/build_answer_evidence_targets.py \
  --mode export --generator_path "$GEN" --cache_dir "$CACHE" \
  --train_file "$RESET/data/old_train.jsonl" \
  --exclude_files "$RESET/data/tune.jsonl" "$DEV" "$TEST" \
  --corpus "$CORPUS" --max_docs 10 --batch_size 4 \
  --max_new_tokens 64 --seed 42 --out_dir "$RESET/targets"
```

export禁止 `--limit`；所有输出保留相同题序、gold和检索文档。每个teacher不通过筛查就回退gold；R/A matched使用同一集合。R-all保留所有合格raw目标，作为普通SKD的强基线。
若中断，manifest保持未完成，不可训练；当前无断点续导出，需要新输出目录重启。这一限制应在全量任务前考虑，不应让未完成文件进入训练。

## 4. 真实训练接入与工程smoke

下面只演示**2步工程验证**，不是研究结果。两个从发布版独立初始化的任务仅目标文件不同；没有新结构或state loss。小smoke数据选完整文件以避免抽到全为gold的前缀；发生等价拒绝时检查审计结果，不删除校验。

```bash
COMMON=(--init_source published --generator_path "$GEN" --cache_dir "$CACHE"
  --arm direct-ce --state_weight 0 --eval_file "$RESET/data/tune.jsonl"
  --seed 42 --batch_size 2 --grad_accum 8 --max_docs 10
  --decoder_lr 1e-5 --warmup_ratio 0.05 --weight_decay 0.01 --grad_clip 1
  --steps 2 --eval_every 2 --save_every 2 --eval_samples 32 --eval_batch_size 4
  --max_new_tokens 64 --grad_checkpointing)

CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train "${COMMON[@]}" \
  --train_file "$RESET/targets/gold.jsonl" --train_target gold \
  --out_dir "$RESET/smoke_gold"
CUDA_VISIBLE_DEVICES=0 python scripts/run_reader_experiment.py train "${COMMON[@]}" \
  --train_file "$RESET/targets/raw_skd_all.jsonl" --train_target teacher \
  --target_manifest "$RESET/targets/manifest.json" --out_dir "$RESET/smoke_raw_skd"
```

`manifest.json` 应显示实际teacher/gold条数与目标差异；评测无论训练目标为何都使用gold。
模型、tokenizer、cache、模板、截断、eval排除文件与export不匹配即报错。
此smoke的LR用于工程验证，不是已选定的发布版最佳LR。不把2步/300步结果当方法筛选，不从短调度直接resume到更长调度。

正式训练的配方应在新tune上为普通gold/SKD基线确定，并记录步数、样本呈现、监督token量和计算时间；再锁定日程比较。没有审计产物和发布版学习曲线前，本runbook不假定再次3000步的收益，亦不自动发起长训。
只有R/A监督不同且质量可信时才比较 `raw_skd_matched` 与 `answer_skd_matched`；不能用A-matched对一个被限制在别的样本集合的R作因果比较。
现有 `run_reader_experiment.py eval` 和 `compare_reader_experiments.py` 可在同一eval集合做逐题比较；训练文件不同不妨碍相同评测条件的比较。配方冻结之前保持最终测试封存。

## 5. 本地验证与服务器尚缺信息

```bash
python -m pytest -q tests/test_reader_experiment.py tests/test_reader_reset.py tests/test_answer_evidence.py
```

tiny Mistral+PEFT测试覆盖真实前向/梯度和导出→训练→评测链路；其中导出资格测试使用受控生成文本，单独的真实生成测试核对M视图与旧D0一致。它们不是PISCO质量实验。
尚缺真实发布版权重/数据上的审计产物、有效标签差异、显存与吞吐、匹配训练后的泛化结果。缺失项能由服务器输出补充；“Z中究竟保留了多少可用信息”则不能由一次阴性训练直接解答。
