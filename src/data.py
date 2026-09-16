"""Cache-first data pipeline: queries in, cached latents out.

Canonical row (produced by ``scripts/prepare_selecom_data.py``)::

    {"id": ..., "query": ..., "retrieved_doc_ids": [...], "answers": [...],
     "teacher_output": ..., "budget": ...}

Documents are addressed only by ID.  Raw text is accepted for convenience --
IDs are then derived by content hash, the same way the corpus was built -- but the
online path never tokenises it: if training re-encoded documents each epoch, the
claim that QuRO trains and serves from precomputed representations would be
untested.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .cache import LatentCache


def doc_id_for(text: str) -> str:
    """Content-addressed document ID, shared by corpus building and training."""
    return "d:" + hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:20]


def encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _document_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        title = str(value.get("title", "")).strip()
        text = str(value.get("text", value.get("content", value.get("document", ""))))
        return f"Title: {title}\nContent: {text}" if title else text
    return str(value)


def _answers(row: Dict[str, Any]) -> List[str]:
    value = row.get("answers", row.get("answer", []))
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(x) for x in values if str(x).strip()]


def adapt_row(row: Dict[str, Any], cfg, idx: int) -> Dict[str, Any]:
    out = dict(row)
    row_id = str(row.get("id", row.get("q_id", idx)))
    query = str(row.get("query", row.get("question", ""))).strip()
    if not query:
        raise ValueError(f"row {row_id} has no query/question")

    doc_ids = row.get("retrieved_doc_ids", row.get("doc_ids"))
    if doc_ids is not None:
        doc_ids = [str(x) for x in doc_ids]
    else:
        raw_docs = row.get("documents")
        if raw_docs is None and "document" in row:
            raw_docs = [row["document"]]
        doc_ids = [doc_id_for(_document_text(x)) for x in (raw_docs or [])]
    if cfg.max_docs:
        doc_ids = doc_ids[: cfg.max_docs]
    if not doc_ids:
        raise ValueError(f"row {row_id} has no retrieved documents")

    answers = _answers(row)
    teacher = row.get("teacher_output", row.get("teacher_answer"))
    use_teacher = teacher is not None and cfg.prefer_teacher_output
    target = str(teacher) if use_teacher else (answers[0] if answers else "")
    if not target:
        raise ValueError(f"row {row_id} has neither teacher_output nor answer")

    out.update({
        "id": row_id, "query": query, "retrieved_doc_ids": doc_ids,
        "answers": answers, "answer": answers[0] if answers else target,
        "target": target, "target_source": "teacher" if use_teacher else "gold",
    })
    return out


class QuRODataset(Dataset):
    """Query rows, plus the mismatch controls.

    The question reaches the system by two independent routes -- the readout (query
    encoder, cosine prior, output slots) and the decoder prompt -- and they are
    shifted separately:

    ``readout_query_shift``
        Swaps only the question the *readout* sees.  With ``decoder_query_shift=0``
        the decoder still gets the right question, so any drop is attributable to
        the readout selecting the wrong evidence.  This is the main diagnostic.
    ``decoder_query_shift``
        Swaps only the question in the prompt.  Bounds how much of a mismatch drop
        is simply the decoder being asked something else.
    ``query_shift``
        Legacy: shifts both at once.  Kept so historical runs can be reproduced,
        but a drop under it is *not* attributable to the readout
        (docs/warning_and_target.md W4).
    """

    def __init__(self, path, tokenizer, data_cfg, query_tokenizer=None,
                 query_shift=0, document_shift=0, limit=None, corpus=None,
                 readout_query_shift=None, decoder_query_shift=None):
        rows = read_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        self.rows = [adapt_row(row, data_cfg, i) for i, row in enumerate(rows)]
        self.tok = tokenizer
        self.query_tok = query_tokenizer if query_tokenizer is not None else tokenizer
        self.cfg = data_cfg
        self.query_shift = int(query_shift)
        # An explicit per-path shift wins; otherwise both inherit the legacy value.
        self.readout_query_shift = int(query_shift if readout_query_shift is None
                                       else readout_query_shift)
        self.decoder_query_shift = int(query_shift if decoder_query_shift is None
                                       else decoder_query_shift)
        self.document_shift = int(document_shift)
        # Only the uncompressed RG baseline reads raw text; the compressed path
        # must never see it, or the cacheability claim would be untested.
        self.corpus = corpus or {}
        self.eos = getattr(tokenizer, "eos_token_id", None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # The mismatch control swaps in a neighbour's query while keeping this
        # row's documents and answer: a query-conditioned readout must degrade.
        # The two routes are shifted independently so a drop can be attributed.
        n = len(self.rows)
        readout_row = self.rows[(index + self.readout_query_shift) % n]
        readout_query = readout_row["query"]
        decoder_query = self.rows[(index + self.decoder_query_shift) % n]["query"]
        # A mismatch control is only clean when the swapped-in question does not
        # happen to have the same answer; carried through so the per-item dump can
        # report the collision rate rather than leaving it assumed to be zero.
        readout_answers = readout_row.get("answers") or [readout_row["answer"]]
        # The document control keeps the query and the gold answer but swaps in a
        # neighbour's evidence.  Without it there is no way to tell an answer read
        # out of the cached latents from one recalled from the decoder's
        # parametric memory -- and on TriviaQA a 7B model answers a large share of
        # the questions closed-book.
        document_row = self.rows[(index + self.document_shift) % len(self.rows)]

        target_ids = encode_text(self.tok, " " + row["target"].strip())[: self.cfg.max_answer_len]
        if self.eos is not None:
            target_ids.append(self.eos)
        return {
            "id": row["id"],
            # "query" is what the decoder prompt renders; query_ids/query_gen_ids
            # are what the readout consumes.  They are only the same string when
            # the two shifts agree.
            "query": decoder_query,
            "readout_query": readout_query,
            "readout_query_answers": readout_answers,
            "retrieved_doc_ids": document_row["retrieved_doc_ids"],
            "document_texts": [self.corpus[d] for d in document_row["retrieved_doc_ids"]
                               if d in self.corpus],
            "query_ids": encode_text(self.query_tok, readout_query)[: self.cfg.max_query_len],
            "query_gen_ids": encode_text(self.tok, readout_query)[: self.cfg.max_query_len],
            "target_ids": target_ids,
            "budget": row.get("budget"),
            "raw": row,
        }


def _pad_2d(sequences, pad_id):
    width = max(1, max((len(x) for x in sequences), default=1))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        if seq:
            ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = True
        else:
            mask[i, 0] = True
    return ids, mask


class QuROCollator:
    """Resolve document IDs through the latent cache and pad the query tensors."""

    def __init__(self, cache: LatentCache, pad_id: int, query_pad_id: Optional[int] = None,
                 max_docs: Optional[int] = None, require_budget_labels: bool = False):
        if cache is None:
            raise ValueError("QuRO is cache-first: a LatentCache is required")
        self.cache = cache
        self.pad_id = pad_id
        self.query_pad_id = pad_id if query_pad_id is None else query_pad_id
        self.max_docs = max_docs
        self.require_budget_labels = bool(require_budget_labels)

    def __call__(self, batch):
        query_ids, query_mask = _pad_2d([x["query_ids"] for x in batch], self.query_pad_id)
        gen_ids, gen_mask = _pad_2d([x["query_gen_ids"] for x in batch], self.pad_id)
        doc_ids = [x["retrieved_doc_ids"][: self.max_docs] for x in batch]
        latents, document_mask, counts = self.cache.get_many(doc_ids, max_docs=self.max_docs)

        out = {
            "ids": [x["id"] for x in batch],
            # "queries" feeds the decoder prompt only; the readout's question is
            # query_ids/query_gen_ids, and readout_queries records it for the
            # per-item dump so a mismatch run is auditable after the fact.
            "queries": [x["query"] for x in batch],
            "readout_queries": [x.get("readout_query", x["query"]) for x in batch],
            "readout_query_answers": [x.get("readout_query_answers", []) for x in batch],
            "retrieved_doc_ids": doc_ids,
            "document_texts": [x["document_texts"][: self.max_docs] for x in batch],
            "query_ids": query_ids, "query_mask": query_mask,
            "query_gen_ids": gen_ids, "query_gen_mask": gen_mask,
            "target_ids": [x["target_ids"] for x in batch],
            "cached_latents": latents, "document_mask": document_mask,
            "source_token_counts": counts,
            "raw": [x["raw"] for x in batch],
        }
        budgets = [x.get("budget") for x in batch]
        if any(v is not None for v in budgets) and not all(v is not None for v in budgets):
            raise ValueError("budget labels must be present for every row in a batch or none")
        if self.require_budget_labels and not all(v is not None for v in budgets):
            raise ValueError("adaptive-budget training requires a budget label on every row")
        if all(v is not None for v in budgets):
            out["budget"] = torch.tensor(budgets, dtype=torch.long)
        return out


def move_to_device(batch: Dict[str, Any], device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_toy_tokenizer(paths_: Sequence[str], data_cfg, save_to: Optional[str] = None):
    """Vocabulary must cover every split, or held-out answers become UNK."""
    from .toy import ToyTokenizer
    texts: List[str] = []
    for path in dict.fromkeys(paths_):
        if not path or not os.path.exists(path):
            continue
        for i, raw in enumerate(read_jsonl(path)):
            row = adapt_row(raw, data_cfg, i)
            texts += [row["query"], row["target"]] + row["answers"]
    tokenizer = ToyTokenizer.build_from_texts(texts or ["placeholder"])
    if save_to:
        tokenizer.save(save_to)
    return tokenizer


def load_corpus(paths_: Sequence[str]) -> Dict[str, str]:
    """doc_id -> text, for the uncompressed RG baseline only."""
    corpus: Dict[str, str] = {}
    for path in paths_:
        if not path:
            continue
        for row in read_jsonl(path):
            corpus[str(row["doc_id"])] = str(row.get("text", row.get("document", "")))
    return corpus
