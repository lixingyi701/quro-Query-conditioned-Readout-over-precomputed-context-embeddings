#!/bin/bash
# After retrieval.jsonl exists: attach -> PISCO cache (4 GPU slices + verified pack)
# -> coverage check -> SQX/SQ/S0X/S0 seed42 on public_qa_90k (one GPU each).
# Arguments are written out in the tmux command; a multi-line string variable
# split at its newlines and broke the first launch (`--seed: command not found`).
set -euo pipefail
cd /home/lxy/quro
R=/data02/quro/data/public_qa_90k_retrieval
Q=/data02/quro/data/public_qa_90k_questions
READY=/data02/quro/data/public_qa_90k_ready
CACHE=/data02/quro/cache/public_qa_pisco_r16
RUNS=/data02/quro/runs/public_qa_90k
SEED=${SEED:-42}
log() { echo "[$(date +%T)] $*" | tee -a $R/run_train.log; }

until grep -q "rerank done" $R/run_pipeline.log 2>/dev/null; do sleep 60; done
log "retrieval ready: $(wc -l < $R/retrieval.jsonl) rows"

if [ ! -f $READY/train.jsonl ]; then
  python scripts/prepare_public_qa.py attach --queries_dir $Q --retrieval_jsonl $R/retrieval.jsonl \
    --max_docs 5 --out_dir $READY 2>&1 | tail -40 | tee -a $R/run_train.log
fi
[ -f $READY/train.jsonl ] || { log "attach failed"; exit 1; }
log "attach done: train $(wc -l < $READY/train.jsonl), dev $(wc -l < $READY/dev.jsonl), corpus $(wc -l < $READY/corpus.jsonl)"

if [ ! -e $CACHE ]; then
  log "cache build start"
  CORPUS=$READY/corpus.jsonl CACHE=$CACHE GPUS=0,1,2,3 bash scripts/build_hotpot_cache.sh > $R/cache_build.log 2>&1
  log "cache build done"
fi
python scripts/check_rag_data.py $READY/train.jsonl $READY/dev.jsonl --cache_dir $CACHE --max_docs 5 \
  2>&1 | tail -20 | tee -a $R/run_train.log

gpu=0
for arm in SQX SQ S0X S0; do
  case $arm in
    SQX) qmode="conditioned"; xdoc="--projector_cross_document" ;;
    SQ)  qmode="conditioned"; xdoc="" ;;
    S0X) qmode="none";        xdoc="--projector_cross_document" ;;
    S0)  qmode="none";        xdoc="" ;;
  esac
  out=$RUNS/seed${SEED}/$arm
  mkdir -p $out
  tmux new-session -d -s pqa_${arm}_s${SEED} \
    "cd /home/lxy/quro && CUDA_VISIBLE_DEVICES=$gpu python -m src.train \
      --preset pisco_shared_projector \
      --train_file $READY/train.jsonl \
      --eval_files dev=$READY/dev.jsonl \
      --cache_dir $CACHE \
      --generator_path /data02/quro/models/pisco-mistral \
      --max_docs 5 --max_query_len 256 --max_answer_len 128 --gen_max_new_tokens 128 \
      --steps 9000 --lr 5e-5 --batch_size 2 --grad_accum 8 \
      --eval_every 500 --eval_every_samples 500 --eval_max_samples 2000 \
      --select_metric em --projector_fusion none --support_loss_weight 0 \
      --seed $SEED --data_order_seed $SEED --num_workers 4 \
      --projector_query_mode $qmode $xdoc \
      --tag pqa90k_${arm}_s${SEED} --out_dir $out \
      2>&1 | tee $RUNS/${arm}_s${SEED}.log"
  log "launched $arm seed${SEED} on GPU $gpu -> $out"
  gpu=$((gpu+1))
done
