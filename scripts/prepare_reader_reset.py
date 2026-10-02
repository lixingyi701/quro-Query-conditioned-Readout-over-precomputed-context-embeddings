"""Audit existing/full data and create deterministic, non-overlapping reset splits.

Standard-library only. Query overlap is ID OR normalized question equality.
Shared document IDs are reported, not removed: document reuse is a target use case.
No claims about near-duplicates or the published model's upstream training data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def question(row):
    return " ".join(unicodedata.normalize("NFKC", str(row.get("query", row.get("question", "")))).casefold().split())


def rowid(row):
    return str(row.get("id", row.get("q_id", "")))


def docs(row):
    return [str(d) for d in row.get("retrieved_doc_ids", row.get("doc_ids", []))]


def read(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        if not rowid(row) or not question(row) or not docs(row):
            raise ValueError(f"{path}: explicit nonempty query ID, question and cached doc IDs required")
    return rows


def keys(rows):
    return {rowid(r) for r in rows}, {question(r) for r in rows}


def clean(rows, blocked, label, rejected):
    ids, qs = map(set, blocked)
    kept = []
    for row in rows:
        reasons = []
        if rowid(row) in ids:
            reasons.append("id_overlap_or_duplicate")
        if question(row) in qs:
            reasons.append("question_overlap_or_duplicate")
        if reasons:
            rejected.append({"source": label, "id": rowid(row), "reasons": reasons})
        else:
            kept.append(row)
        # Also block alternate IDs/questions on rejected rows (conservative).
        ids.add(rowid(row)); qs.add(question(row))
    return kept


def prepare(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite nonempty output")
    if args.tune_size < 1:
        raise ValueError("tune_size must be positive")
    seen, pool = read(args.seen), read(args.pool)
    excluded = [r for path in args.exclude for r in read(path)]
    rejected = []
    old = clean(seen, keys(excluded), "seen", rejected)
    fresh = clean(pool, keys(seen + excluded), "pool", rejected)
    # Sort before seeded shuffle so input file order cannot choose the holdout.
    rng = random.Random(args.seed)
    fresh.sort(key=lambda r: (rowid(r), question(r)))
    rng.shuffle(fresh)
    if len(fresh) <= args.tune_size:
        raise ValueError("not enough genuinely new queries for tune plus training")
    tune, fresh = fresh[:args.tune_size], fresh[args.tune_size:]
    # Match old/new training size AND coarse question-type composition.
    old.sort(key=lambda r: (rowid(r), question(r)))
    rng.shuffle(old)
    need = Counter(str(r.get("hop_type", "unknown")) for r in old)
    available = Counter(str(r.get("hop_type", "unknown")) for r in fresh)
    if not old or any(available[k] < n for k, n in need.items()):
        raise ValueError(f"cannot match full cleaned old set by hop_type: need={dict(need)}, available={dict(available)}")
    matched = []
    for row in fresh:
        kind = str(row.get("hop_type", "unknown"))
        if need[kind] > 0:
            matched.append(row); need[kind] -= 1
    outputs = {"old_train.jsonl": old, "new_train_matched.jsonl": matched, "tune.jsonl": tune}
    manifest = json.loads(Path(args.cache_manifest).read_text())
    expected = {"latent_size": 8, "hidden_size": 4096, "doc_max_length": 128, "compr_rate": 16}
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"cache {name}: expected {value}, got {manifest.get(name)}")
    cache_docs = manifest.get("documents", {})
    needed = {d for rows in outputs.values() for row in rows for d in docs(row)}
    missing = sorted(needed - set(cache_docs))
    if missing:
        raise ValueError(f"cache missing {len(missing)} docs, examples: {missing[:5]}")
    names = list(outputs)
    overlaps = {}
    for i, name in enumerate(names):
        ids, qs = keys(outputs[name])
        d = {x for r in outputs[name] for x in docs(r)}
        for other in names[i+1:]:
            oi, oq = keys(outputs[other])
            assert not (ids & oi or qs & oq)
            overlaps[f"{name} / {other}"] = len(d & {x for r in outputs[other] for x in docs(r)})
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {"seed": args.seed, "inputs_sha256": {str(p): digest(p) for p in [args.seen, args.pool, *args.exclude]},
              "cache_manifest_sha256": digest(args.cache_manifest),
              "cache_metadata": {k: v for k, v in manifest.items() if k not in {"documents", "shards"}},
              "outputs": {n: {"rows": len(r), "sha256": digest(out/n),
                           "hop_types": dict(Counter(str(x.get("hop_type", "unknown")) for x in r))}
                          for n, r in outputs.items()},
              "shared_document_counts": overlaps, "excluded_rows": rejected,
              "note": "Exact ID/normalized-question audit only. Tune is new relative to supplied P1 train; "
                      "it is not an independent final test. Cached vectors and external backbone revision "
                      "still require release-equivalence verification; published-model data overlap unknown."}
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({n: len(r) for n, r in outputs.items()}))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seen", required=True, help="exact 30k file used by P1")
    p.add_argument("--pool", required=True, help="full train pool, not validation/test")
    p.add_argument("--exclude", nargs="+", required=True, help="ALL dev and final-test query files")
    p.add_argument("--cache_manifest", required=True)
    p.add_argument("--tune_size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--out_dir", required=True)
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
