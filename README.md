# QuRO v0.0

**Query-conditioned Readout over precomputed context embeddings**

QuRO separates context compression into a reusable offline representation and a
small query-time readout:

The implementation follows the repository's
[experimental design](article/QURO_EXPERIMENTAL_DESIGN.md); the accompanying
[related-work analysis](article/QURO_RELATED_WORK.md) explains the distinction
from PISCO, COCOM, SeleCom, and other compression baselines.

```text
offline, once per document
document d_i -> frozen PISCO / COCOM -> Z_i (m, h) -> sharded disk cache

online, once per query
retrieved IDs -> Z (K, m, h) -> flatten (K*m, h)
natural-language query -> B structured output queries
CrossAttention(Q=query-conditioned slots, K/V=Z) -> B soft tokens -> generator
```

This directory is version **0.0.0**: it implements and tests the architectural
contract. It does not contain paper results or claim that QuRO already beats
SeleCom, PISCO, COCOM, or ArcAligner.

## What v0.0 implements

- Independent per-document latent representation `(m, h)`.
- Sharded `doc_id -> latent` disk cache with a manifest and source-token counts.
- Multi-document online interface `(batch, K, m, h) -> (batch, K*m, d)`.
- Learned retrieval-rank/document-source embeddings.
- Perceiver IO readout with query-constructor families:
  - `agnostic`: learned slots only (Ablation A).
  - `add` / `film`: pooled-query conditioning.
  - `concat`: projected `[slot; pooled query]`, the explicit e_q plus P variant.
  - `xattn`: semantic slots first cross-attend to query tokens.
- Fixed B and query-conditioned discrete budget buckets.
- Optional identity/linear/MLP bridge to the generator embedding space.
- Frozen generator backbone with optional generator LoRA.
- Sequence-level distillation through a `teacher_output` field.
- Mismatch-query causal control.
- EM, token F1, substring accuracy, source-token count, readout-token count,
  and effective generator-side compression `xi_eff`.

The old single-document Perceiver encoder is retained only as a smoke-test and
cache-construction prototype. Cache-backed experiments are the QuRO path.

## Stable interfaces

### Latent cache

Each cached document has exactly one tensor:

```python
Z_d.shape == (m, h)
```

The cache manifest records compressor/checkpoint identity, m, h, dtype,
document location, and source-token count. Unknown retrieved IDs raise an error;
evidence is never silently discarded.

```python
from src.cache import LatentCache

cache = LatentCache("/path/to/cache")
latents, document_mask, source_counts = cache.get_many([
    ["doc-17", "doc-42"],
    ["doc-9"],
])
# latents:       (2, 2, m, h)
# document_mask: (2, 2)
```

### Online readout

```python
result = model.readout_cached(
    doc_latents=latents,
    document_mask=document_mask,
    query_ids=query_ids,
    query_mask=query_mask,
    budget=[8, 16],
)

soft_tokens = result["soft_tokens"]
soft_token_mask = result["soft_token_mask"]
```

The readout cost is `O(B*K*m)` and is independent of original document length.

## Preparing a PISCO/COCOM cache

QuRO does not copy or guess private APIs from research repositories. An adapter
factory must return either a callable or an object exposing:

```python
encode_texts(list_of_strings) -> Tensor[batch, m, h]
```

The returned model is forced into evaluation mode and all module parameters are
frozen.

```bash
python scripts/build_latent_cache.py \
  --documents data/corpus.jsonl \
  --adapter my_pisco_adapter:build \
  --checkpoint /models/pisco-checkpoint \
  --out_dir cache/pisco \
  --dtype float16
```

`data/corpus.jsonl`:

```json
{"doc_id": "wiki:123", "text": "Document text ...", "source_token_count": 128}
```

If PISCO/COCOM latents were already exported:

```bash
python scripts/import_latent_cache.py \
  --input exported_latents.pt \
  --compressor naver/pisco-checkpoint \
  --out_dir cache/pisco
```

Accepted exports are `{doc_id: tensor(m,h)}` or a dictionary containing
`doc_ids`, `latents`, and optional `source_token_counts`.

## Query/training data

Cache-first JSONL:

```json
{
  "id": "nq-001",
  "query": "Who designed the building?",
  "retrieved_doc_ids": ["wiki:123", "wiki:891"],
  "answers": ["Jane Doe"],
  "teacher_output": "Jane Doe designed the building.",
  "budget": 8
}
```

- `teacher_output` is preferred over `answers` and produces sequence-level
  teacher-sequence CE.
- `answers` are retained for evaluation.
- `budget` is optional for fixed-budget runs. Adaptive-budget training requires
  a discrete bucket label on every training row.
- Every retrieved ID in every split must exist in the selected cache.

## Training

```bash
export CACHE_DIR=/data/cache/pisco
export RAG_TRAIN_FILE=/data/quro/train.jsonl
export RAG_DEV_FILE=/data/quro/dev.jsonl
export ENCODER_PATH=/models/Qwen3-Embedding-0.6B
export GENERATOR_PATH=/models/Qwen2.5-1.5B-Instruct

bash scripts/run_qwen3emb.sh
```

Main trainable components in cache mode are the query constructor, source
embeddings, readout, optional projector, supervised budget controller, and
generator LoRA. The query encoder and offline compressor are frozen; the
offline compressor is not loaded during online training.

Core A-vs-C/D ablation:

```bash
CACHE_DIR=... RAG_TRAIN_FILE=... RAG_DEV_FILE=... \
  BS_LIST="4 8 16 32" bash scripts/run_ablation.sh
```

## Local contract test

```bash
pip install -r requirements.txt
python tests/test_shapes.py
```

The test covers independent multi-document encoding, K*m online memory
construction and masking, query sensitivity, Ablation A, per-example discrete
budgets, sharded cache round-trip, teacher-output selection, and gradient flow.

## Compression accounting

Results record three distinct quantities:

- Offline compression: `xi_off = source tokens / m`.
- Online generator budget: `B`.
- Effective generator-side compression:
  `xi_eff = sum(source tokens over retrieved documents) / B`.

Baseline comparisons must lock B/xi_eff; storage cost must be reported
separately.

## Deliberately not claimed in v0.0

- A finalized built-in PISCO/COCOM adapter: their current research APIs need to
  be pinned and validated on the target server first.
- A trained overflow-risk probe or learned labels for adaptive budgets.
- Oracle-selection/oracle-capacity decomposition.
- Uncompressed RAG, non-parametric top-B, SeleCom, and ArcAligner result tables.
- TTFT/GFLOPs/amortization crossover measurements.
- Evidence-token attribution inside opaque external latents.

These are the next go/no-go experiments, not hidden completed features.
