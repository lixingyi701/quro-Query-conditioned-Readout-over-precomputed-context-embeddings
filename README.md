# QuRO v0.1

**Query-conditioned Readout over precomputed context embeddings**

Existing soft compressors face a cacheability–specificity dilemma. Query-agnostic
compressors (COCOM, PISCO) are reusable but must preserve information for
questions they have not seen. Query-conditioned selectors (SeleCom) keep exactly
the right evidence but re-read the source document on every query. QuRO
factorises the two roles across *time*: a reusable query-independent document
memory built once offline, and a lightweight query-conditioned readout that never
revisits source tokens.

```text
offline, once per document
    d  --frozen PISCO/COCOM-->  Z_d in R^(m x h)  -->  sharded disk cache

online, once per query
    retrieved IDs      -->  Z in R^(K*m x h)          (free: a disk read)
    query tokens       -->  B output queries
    CrossAttn(Q, Z, Z) -->  B soft tokens  -->  frozen Mistral + LoRA  -->  answer
```

Online attention cost is `O(B * K * m)` and does not depend on the original
document lengths. That is the whole basis of the efficiency argument.

See [`article/QURO_EXPERIMENTAL_DESIGN.md`](article/QURO_EXPERIMENTAL_DESIGN.md)
for the experiment plan and [`article/QURO_V0.1_IMPLEMENTATION_PLAN.md`](article/QURO_V0.1_IMPLEMENTATION_PLAN.md)
for what this version implements and why.

## What v0.1 adds over v0.0

v0.0 implemented the architectural contract against a randomly initialised
prototype encoder, so none of its numbers meant anything. v0.1 connects real
components:

- **Real frozen compressors.** `naver/pisco-mistral` (rate 16 → `m=8`, `h=4096`)
  and `naver/cocom-v1-{4,16,128}-mistral-7b`, all on the same Mistral base, so
  sweeping the offline rate introduces no confound.
- **The generator is PISCO's own decoder.** Mistral-7B-Instruct-v0.2 plus the
  published `decoder_adapter`, with PISCO's prompt template and `B` memory slots
  instead of `K*m`. A PISCO baseline is then literally the same object with a
  different readout.
- **Residual readout.** `E = s * AttnPool(alpha, Z) + Delta` with `Delta`
  zero-initialised, so step 0 *is* attention-pooled PISCO and every point gained
  afterwards is attributable to query conditioning. It also matches the scale of
  the cached latents (measured std ≈ 1.9), which a LayerNorm'd output would not.
- **Slot self-attention** in each readout block, so the `B` output slots can
  avoid all reading the same evidence.
- **Budget dropout.** `B` is sampled per step, which makes the nested slot
  structure that `slots[:B]` assumes actually true, so one checkpoint serves
  every budget.
- **Decoder input modes D0–D3** and `--query_text_dropout`, for the question
  "now that the soft tokens are query-conditioned, does the decoder still need the
  question in plain text?"

## Setup

`/home` on this machine is a full shared disk, so weights, caches and run outputs
live under `QURO_ROOT`. Paths are resolved in [`src/paths.py`](src/paths.py) and
every one is overridable by environment variable.

```
QURO_ROOT=/data02/quro
  ├── models/   pisco-mistral, cocom-v1-{4,16,128}-mistral-7b
  ├── data/     corpus.jsonl, train.jsonl, dev.jsonl
  ├── cache/    gonogo-pisco-r16/ ...
  └── runs/     gonogo_{A,C,S,P}_*/
```

```bash
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download as d; \
  d('naver/pisco-mistral', local_dir='/data02/quro/models/pisco-mistral')"
```

`pisco-mistral` is only 0.69 GB: it ships adapters plus the resized first/last
layers and loads Mistral separately. `src/compressors/pisco.py:ensure_local_base`
rewrites `decoder_model_name` in the downloaded config to point at the local
Mistral copy, so nothing is re-downloaded and loading works offline.

## Running

```bash
bash scripts/run_smoke.sh          # ~15 min: contract tests, cache, regression, short train
bash scripts/run_gonogo.sh 1       # arms A / C / S / P, one per GPU
bash scripts/run_gonogo.sh 2       # the query-text-dropout arms
python scripts/summarize.py        # table + go/no-go verdict
```

Data preparation and cache building, if you want them separately:

```bash
python scripts/prepare_selecom_data.py --out_dir /data02/quro/data/gonogo \
  --stage1_rows 80000 --stage2_rows 20000
python scripts/build_latent_cache.py \
  --documents /data02/quro/data/gonogo/corpus.jsonl \
  --out_dir /data02/quro/cache/gonogo-pisco-r16 \
  --adapter src.compressors.pisco:build --resume
python scripts/check_rag_data.py /data02/quro/data/gonogo/*.jsonl \
  --cache_dir /data02/quro/cache/gonogo-pisco-r16
```

Throughput on one A800: ~60 documents/s to compress, so the 261k-document
go/no-go corpus takes about 75 minutes and 17 GB at fp16.

## Correctness checks

```bash
python tests/test_shapes.py                  # 31 CPU contract tests, no downloads
python scripts/check_pisco_equivalence.py    # QuRO degenerated to PISCO vs PISCO itself
```

The equivalence check is the strongest single test: with `readout=pisco_direct`
and no training, QuRO must reproduce PISCO's answers, which validates the cache
round-trip, prompt template, slot indexing and embedding injection at once.

Expect agreement around 70–90%, not 100%. PISCO's compression is deterministic
for a fixed batch but not across batch sizes — measured here, one document
compressed in a batch of 16 versus 64 differs by up to 0.66 absolute (mean 0.024)
at cosine similarity ≥ 0.9997, and greedy decoding over a 7B bf16 model turns that
into the occasional synonym swap. Two consequences: the PISCO baseline must read
the *same cache* rather than recompress, and single-example comparisons between
systems are noise.

## Compression accounting

Three distinct quantities, none of which substitutes for another:

- `xi_off = fed tokens / m` — offline; paid in storage, once per document.
- `B` — soft tokens the generator sees; paid per query at prefill.
- `xi_eff = sum(fed tokens over retrieved docs) / B` — the only one that makes two
  systems comparable, because it is measured where the cost is paid.

`fed tokens` means tokens the compressor actually consumed. PISCO truncates every
document at 128 tokens (its model card recommends passages cropped to ~128), and
SeleCom's documents average 180 Mistral tokens, so **about a third of each
document never reaches the compressor**. The cache records both the fed and the
untruncated count so this stays visible; computing `xi_off` against untruncated
length would inflate 15.1× into 22.5×.

Main-table comparisons lock `B`. Storage cost is reported separately, never
netted off.

## Not claimed in v0.1

- The `xi_off` sweep over COCOM rate 4/16/128 (checkpoints not yet downloaded).
- Token-overflow probing, oracle selection/capacity decomposition.
- Adaptive budget training (the selector and loss exist; the labels do not).
- TTFT / GFLOPs / the amortisation crossover `q*`.
- LLM-judge scoring, and every dataset beyond the SeleCom splits and TriviaQA.

These are the next experiments, not hidden features.
