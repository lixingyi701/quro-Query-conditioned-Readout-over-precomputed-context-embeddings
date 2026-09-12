"""Cache-first data pipeline for QuRO v0.0.

Canonical query rows:
  {"id": ..., "query": ..., "retrieved_doc_ids": [...],
   "answers": [...], "teacher_output": ...}

The online collator resolves document IDs through LatentCache. Raw document text
is accepted only for prototype encoding and cache construction.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .cache import LatentCache


def encode_text(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "itos"):
        return tokenizer.encode(text, add_special_tokens=False)
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
        return (f"Title: {title}\nContent: {text}" if title else text)
    return str(value)


def _document_id(value: Any, row_id: str, rank: int) -> str:
    if isinstance(value, dict):
        for key in ("doc_id", "id", "passage_id", "_id"):
            if value.get(key) is not None:
                return str(value[key])
    return f"{row_id}:doc:{rank}"


def _answers(row: Dict[str, Any]) -> List[str]:
    value = row.get("answers", row.get("answer", []))
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(x) for x in values if str(x).strip()]


def adapt_row(row: Dict[str, Any], cfg, idx: int) -> Dict[str, Any]:
    """Normalize synthetic, DPR-like, and cache-first rows."""
    out = dict(row)
    row_id = str(row.get("id", row.get("q_id", idx)))
    query = str(row.get("query", row.get("question", "")))
    if not query:
        raise ValueError(f"row {row_id} has no query/question")

    docs_raw = row.get("documents")
    if docs_raw is None and "document" in row:
        docs_raw = [row["document"]]
    docs_raw = list(docs_raw or [])[: max(1, cfg.max_docs)]
    texts = [_document_text(x) for x in docs_raw]

    explicit_ids = row.get("retrieved_doc_ids", row.get("doc_ids"))
    if explicit_ids is not None:
        doc_ids = [str(x) for x in explicit_ids][: max(1, cfg.max_docs)]
    else:
        doc_ids = [_document_id(value, row_id, rank) for rank, value in enumerate(docs_raw)]
    if not doc_ids:
        raise ValueError(f"row {row_id} has no retrieved documents")
    if len(texts) not in (0, len(doc_ids)):
        raise ValueError(f"row {row_id}: document text count and doc_id count differ")

    answers = _answers(row)
    teacher_output = row.get("teacher_output", row.get("teacher_answer"))
    target = (str(teacher_output) if teacher_output is not None and cfg.prefer_teacher_output
              else (answers[0] if answers else ""))
    if not target:
        raise ValueError(f"row {row_id} has neither teacher_output nor answer")

    out.update({
        "id": row_id,
        "query": query,
        "documents": texts,
        "retrieved_doc_ids": doc_ids,
        "answers": answers,
        "answer": answers[0] if answers else target,
        "target": target,
        "target_source": "teacher" if teacher_output is not None and cfg.prefer_teacher_output else "gold",
    })
    return out


class RAGCompressionDataset(Dataset):
    """Query rows; query_shift creates the mismatch-query causal control."""

    def __init__(self, path, tokenizer, data_cfg, query_shift=0,
                 enc_tokenizer=None, limit=None):
        rows = read_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        self.rows = [adapt_row(row, data_cfg, i) for i, row in enumerate(rows)]
        self.tok = tokenizer
        self.enc_tok = enc_tokenizer if enc_tokenizer is not None else tokenizer
        self.cfg = data_cfg
        self.query_shift = int(query_shift)
        self.eos = getattr(tokenizer, "eos_token_id", None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        query_row = self.rows[(index + self.query_shift) % len(self.rows)]
        query_ids = encode_text(self.enc_tok, query_row["query"])[: self.cfg.max_query_len]
        prompt_ids = encode_text(
            self.tok, self.cfg.qa_prompt_template.format(query=row["query"]))
        target_ids = encode_text(self.tok, " " + row["target"].strip())[: self.cfg.max_answer_len]
        if self.eos is not None:
            target_ids.append(self.eos)
        documents = [
            encode_text(self.enc_tok, text)[: self.cfg.max_doc_len]
            for text in row["documents"]
        ]
        return {
            "id": row["id"],
            "retrieved_doc_ids": row["retrieved_doc_ids"],
            "document_input_ids": documents,
            "query_ids": query_ids,
            "prompt_ids": prompt_ids,
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
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = True
        else:
            mask[i, 0] = True
    return ids, mask


def _pad_documents(batch_documents, pad_id):
    batch_size = len(batch_documents)
    k = max(1, max((len(x) for x in batch_documents), default=0))
    length = max(1, max((len(doc) for docs in batch_documents for doc in docs), default=0))
    ids = torch.full((batch_size, k, length), pad_id, dtype=torch.long)
    token_mask = torch.zeros((batch_size, k, length), dtype=torch.bool)
    document_mask = torch.zeros((batch_size, k), dtype=torch.bool)
    for i, docs in enumerate(batch_documents):
        for j, doc in enumerate(docs):
            document_mask[i, j] = True
            if doc:
                ids[i, j, :len(doc)] = torch.tensor(doc)
                token_mask[i, j, :len(doc)] = True
            else:
                token_mask[i, j, 0] = True
    return ids, token_mask, document_mask


class QuROCollator:
    """Resolve cache IDs or construct independent prototype document tensors."""

    def __init__(self, pad_id, enc_pad_id=None, cache: Optional[LatentCache] = None,
                 max_docs=None, require_budget_labels=False):
        self.pad_id = pad_id
        self.enc_pad_id = pad_id if enc_pad_id is None else enc_pad_id
        self.cache = cache
        self.max_docs = max_docs
        self.require_budget_labels = bool(require_budget_labels)

    def __call__(self, batch):
        query_ids, query_mask = _pad_2d(
            [item["query_ids"] for item in batch], self.enc_pad_id)
        doc_ids = [item["retrieved_doc_ids"][:self.max_docs] for item in batch]
        out = {
            "ids": [item["id"] for item in batch],
            "retrieved_doc_ids": doc_ids,
            "query_ids": query_ids,
            "query_mask": query_mask,
            "prompt_ids": [item["prompt_ids"] for item in batch],
            "target_ids": [item["target_ids"] for item in batch],
            "raw": [item["raw"] for item in batch],
        }
        budgets = [item.get("budget") for item in batch]
        if any(value is not None for value in budgets) and not all(
                value is not None for value in budgets):
            raise ValueError("budget labels must be present for every item in a batch or none")
        if self.require_budget_labels and not all(value is not None for value in budgets):
            raise ValueError("adaptive-budget training requires a budget label for every row")
        if all(value is not None for value in budgets):
            out["budget"] = torch.tensor(budgets, dtype=torch.long)
        if self.cache is not None:
            latents, document_mask, counts = self.cache.get_many(
                doc_ids, max_docs=self.max_docs)
            out.update(cached_latents=latents, document_mask=document_mask,
                       source_token_counts=counts)
        else:
            docs = [item["document_input_ids"][:self.max_docs] for item in batch]
            if any(len(x) == 0 for x in docs):
                raise ValueError("prototype mode requires document text; use cache for ID-only rows")
            ids, token_mask, document_mask = _pad_documents(docs, self.enc_pad_id)
            out.update(document_input_ids=ids, document_token_mask=token_mask,
                       document_mask=document_mask)
        return out


Collator = QuROCollator


def move_to_device(batch: Dict[str, Any], device):
    out = dict(batch)
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
    return out


def build_toy_tokenizer(paths: Sequence[str], data_cfg, save_to: Optional[str] = None):
    from .toy import ToyTokenizer
    texts = [data_cfg.qa_prompt_template.format(query="")]
    for path in dict.fromkeys(paths):
        if not path or not os.path.exists(path):
            continue
        for i, raw in enumerate(read_jsonl(path)):
            row = adapt_row(raw, data_cfg, i)
            texts += row["documents"] + [row["query"], row["target"]] + row["answers"]
    tokenizer = ToyTokenizer.build_from_texts(texts)
    if save_to:
        tokenizer.save(save_to)
    return tokenizer


def encode_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    if hasattr(tokenizer, "itos"):
        from .toy import _TOKEN_RE
        matches = list(_TOKEN_RE.finditer(text))
        ids = [tokenizer.stoi.get(x.group(0), tokenizer.stoi["<unk>"]) for x in matches]
        return ids, [(x.start(), x.end()) for x in matches]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoded["input_ids"], [tuple(x) for x in encoded["offset_mapping"]]
