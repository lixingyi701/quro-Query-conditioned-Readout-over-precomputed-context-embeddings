"""Convert the SeleCom release into QuRO's cache-first corpus + query files.

SeleCom publishes two training sets (``SELECOM_PAPER_NOTES.md`` §4):

* stage 1 -- 14M synthetic ``(question, answer, document, difficulty)`` rows,
  one document per query;
* stage 2 -- 868K ``(question, documents, answer)`` rows, several documents per
  query, which is what the global multi-document readout needs.

QuRO trains on the same data with the same objective as SeleCom so that the only
difference between the two systems is the information path, not the supervision
(``SELECOM_PAPER_NOTES.md:462-473``).  This script splits that data into:

* ``corpus.jsonl``  -- ``{"doc_id", "text"}``, globally deduplicated; the offline
  compressor runs over this exactly once;
* ``train/dev.jsonl`` -- ``{"id", "query", "retrieved_doc_ids", "answers", ...}``,
  already in the format ``src/data.py:adapt_row`` expects.

Documents are identified by a content hash, so the same passage reached through
stage 1 and stage 2 is compressed and cached once.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import paths


def doc_id_for(text: str) -> str:
    return "d:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def stream_jsonl(path: str, limit: int | None = None):
    with open(path, encoding="utf-8") as f:
        for line in itertools.islice(f, limit):
            line = line.strip()
            if line:
                yield json.loads(line)


YES_NO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def as_answers(value) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(x) for x in values if str(x).strip()]


def is_yes_no(answers: list[str]) -> bool:
    """SeleCom's synthetic set is ~42% yes/no with verbose gold answers.

    Two problems follow.  A constant ``"Yes"`` scores 6.6% EM on that mix, which
    is enough to make an untrained system look alive; and a yes/no question needs
    almost no evidence *selection*, so those rows give the readout little reason
    to condition on the query.  Dropping them is available as a flag rather than
    a default, because keeping them is what makes the comparison with SeleCom
    like-for-like.
    """
    return bool(answers) and bool(YES_NO.match(answers[0]))


class CorpusBuilder:
    """Assign stable IDs to documents and remember each one only once."""

    def __init__(self):
        self.texts: dict[str, str] = {}

    def add(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("empty document")
        key = doc_id_for(text)
        self.texts.setdefault(key, text)
        return key

    def write(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for doc_id, text in self.texts.items():
                f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")
        return len(self.texts)


def collect_stage1(path: str, limit: int, corpus: CorpusBuilder,
                   drop_yes_no: bool = False) -> list[dict]:
    rows = []
    for i, row in enumerate(stream_jsonl(path, limit)):
        document, question = row.get("document", ""), row.get("question", "")
        answers = as_answers(row.get("answer"))
        if not (document.strip() and question.strip() and answers):
            continue
        if drop_yes_no and is_yes_no(answers):
            continue
        rows.append({
            "id": f"s1-{i}",
            "query": question,
            "retrieved_doc_ids": [corpus.add(document)],
            "answers": answers,
            "difficulty": row.get("difficulty"),
            "source": "selecom-stage1",
        })
    return rows


def collect_stage2(path: str, limit: int, min_docs: int, scan_limit: int,
                   corpus: CorpusBuilder, drop_yes_no: bool = False) -> list[dict]:
    """Keep only genuinely multi-document rows; they are what trains global readout."""
    rows = []
    for i, row in enumerate(stream_jsonl(path, scan_limit)):
        documents = [str(x) for x in row.get("documents", []) if str(x).strip()]
        question, answers = row.get("question", ""), as_answers(row.get("answer"))
        if len(documents) < min_docs or not question.strip() or not answers:
            continue
        if drop_yes_no and is_yes_no(answers):
            continue
        rows.append({
            "id": f"s2-{i}",
            "query": question,
            "retrieved_doc_ids": [corpus.add(d) for d in documents],
            "answers": answers,
            "source": "selecom-stage2",
        })
        if len(rows) >= limit:
            break
    return rows


def add_distractors(rows: list[dict], corpus: CorpusBuilder, n_distractors: int,
                    rng: random.Random) -> None:
    """Pad single-document rows with random passages, in place.

    Without this, 80% of the training set gives the readout nothing to select:
    a stage-1 row has K=1, so K*m = 8 cached latents and a budget of B=8 is a
    1:1 recombination.  Query conditioning cannot pay off where there is no
    competition for the budget, and the readout is free to collapse into a
    query-agnostic second compression.

    Distractors come from the same corpus, so the latent cache does not change.
    The gold document is placed at a random rank rather than first: retrieval
    rank is a strong relevance prior, and leaving gold at rank 0 would let the
    readout learn "always read the first document" instead of learning to select.
    """
    if n_distractors < 1:
        return
    pool = list(corpus.texts)
    for row in rows:
        gold = row["retrieved_doc_ids"]
        if len(gold) > 1:
            continue
        picked, seen = [], set(gold)
        while len(picked) < n_distractors and len(seen) < len(pool):
            candidate = rng.choice(pool)
            if candidate not in seen:
                seen.add(candidate)
                picked.append(candidate)
        position = rng.randrange(len(picked) + 1)
        row["retrieved_doc_ids"] = picked[:position] + gold + picked[position:]
        row["gold_rank"] = position          # kept for oracle-selection analysis
        row["n_distractors"] = len(picked)


def write_rows(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(paths.DATA_DIR, "gonogo"))
    ap.add_argument("--stage1_rows", type=int, default=80_000)
    ap.add_argument("--stage2_rows", type=int, default=20_000)
    ap.add_argument("--stage2_min_docs", type=int, default=10,
                    help="stage 2 mixes 1/2/10-document rows; keep the multi-document ones")
    ap.add_argument("--stage2_scan", type=int, default=200_000,
                    help="rows to scan before giving up on reaching --stage2_rows")
    ap.add_argument("--dev_fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1_file", default=paths.SELECOM_STAGE1)
    ap.add_argument("--stage2_file", default=paths.SELECOM_STAGE2)
    ap.add_argument("--distractors", type=int, default=4,
                    help="random passages added to single-document rows so the "
                         "readout has something to select from (0 disables)")
    ap.add_argument("--drop_yes_no", action="store_true",
                    help="drop yes/no rows (~42%% of SeleCom); they barely exercise selection")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    corpus = CorpusBuilder()

    print(f"[1/3] reading stage 1 ({args.stage1_rows} rows)")
    stage1 = collect_stage1(paths.require(args.stage1_file, "stage1 jsonl"),
                            args.stage1_rows, corpus, args.drop_yes_no)
    print(f"      kept {len(stage1)} rows")

    print(f"[2/3] reading stage 2 (target {args.stage2_rows} rows with >= {args.stage2_min_docs} docs)")
    stage2 = collect_stage2(paths.require(args.stage2_file, "stage2 jsonl"),
                            args.stage2_rows, args.stage2_min_docs, args.stage2_scan, corpus,
                            args.drop_yes_no)
    print(f"      kept {len(stage2)} rows")

    rng = random.Random(args.seed)
    add_distractors(stage1, corpus, args.distractors, rng)
    add_distractors(stage2, corpus, args.distractors, rng)

    train, dev = [], []
    for group in (stage1, stage2):
        shuffled = list(group)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * args.dev_fraction)
        dev += shuffled[:cut]
        train += shuffled[cut:]
    rng.shuffle(train)
    rng.shuffle(dev)

    print("[3/3] writing")
    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    num_documents = corpus.write(corpus_path)
    write_rows(os.path.join(args.out_dir, "train.jsonl"), train)
    write_rows(os.path.join(args.out_dir, "dev.jsonl"), dev)

    stats = {
        "num_documents": num_documents,
        "num_train": len(train),
        "num_dev": len(dev),
        "stage1_rows": len(stage1),
        "stage2_rows": len(stage2),
        "stage2_min_docs": args.stage2_min_docs,
        "drop_yes_no": args.drop_yes_no,
        "distractors": args.distractors,
        "yes_no_fraction": round(
            sum(is_yes_no(r["answers"]) for r in train + dev) / max(1, len(train) + len(dev)), 4),
        "mean_docs_per_query": round(
            sum(len(r["retrieved_doc_ids"]) for r in train + dev) / max(1, len(train) + len(dev)), 3),
        "estimated_cache_gb_at_m8_fp16": round(num_documents * 8 * 4096 * 2 / 1e9, 2),
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
