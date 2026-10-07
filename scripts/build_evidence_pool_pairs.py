"""Optional TRAIN-only augmentation using existing real questions and cached docs.

Pairs sharing a support document receive one identical 10-document pool. Answers
and support text are copied, not generated. Re-run evidence preparation afterward.
All comparison arms must use the same augmented training rows.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def build_pairs(rows, seed=42, max_pairs=2000, pool_size=10):
    if max_pairs < 1 or pool_size < 2:
        raise ValueError('positive pair cap and pool size >= 2 required')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('source training IDs must be unique')
    rng = random.Random(seed)
    by_support, candidates = defaultdict(list), []
    for i, row in enumerate(rows):
        ranks = row.get('gold_ranks', [])
        docs = row['retrieved_doc_ids']
        if (not row.get('supporting_sentences') or not ranks
                or any(type(r) is not int or not 0 <= r < len(docs) for r in ranks)):
            continue
        for rank in set(ranks):
            by_support[docs[rank]].append(i)
    for doc_id in sorted(by_support):
        indices = sorted(set(by_support[doc_id]))
        rng.shuffle(indices)
        # Bounded neighbour search avoids quadratic work on frequent documents.
        for j, i in enumerate(indices):
            candidates.extend((i, other) for other in indices[j+1:j+21])
    rng.shuffle(candidates)
    used, seen_pools, augmented = set(), set(), []
    for ia, ib in candidates:
        if ia in used or ib in used:
            continue
        a, b = rows[ia], rows[ib]
        if a['query'].strip() == b['query'].strip():
            continue
        signature = lambda r: tuple(sorted((r['retrieved_doc_ids'][f['doc_rank']], f['text'].strip())
                                           for f in r['supporting_sentences']))
        if signature(a) == signature(b):
            continue
        gold = set(a['retrieved_doc_ids'][r] for r in a['gold_ranks']) | set(
            b['retrieved_doc_ids'][r] for r in b['gold_ranks'])
        if len(gold) > pool_size:
            continue
        distractors = sorted((set(a['retrieved_doc_ids']) | set(b['retrieved_doc_ids']))-gold)
        rng.shuffle(distractors)
        pool = sorted(gold)+distractors[:pool_size-len(gold)]
        if len(pool) != pool_size or tuple(sorted(pool)) in seen_pools:
            continue
        rng.shuffle(pool)
        seen_pools.add(tuple(sorted(pool)))
        group = 'pair-'+hashlib.sha256(json.dumps([a['id'], b['id'], pool]).encode()).hexdigest()[:16]
        for row in (a, b):
            old = row['retrieved_doc_ids']
            # Reordering invalidates annotations and all rank-derived fields.
            new = {k: v for k, v in row.items() if k not in {
                'support_annotation', 'evidence_annotation', 'gold_ranks', 'n_gold',
                'n_distractors', 'budget', 'supporting_sentences', 'retrieved_doc_ids'}}
            ranks = sorted(pool.index(old[r]) for r in row['gold_ranks'])
            facts = [{**f, 'doc_rank': pool.index(old[f['doc_rank']])}
                     for f in row['supporting_sentences']]
            new.update(id=group+'-'+row['id'], original_id=row['id'], augmentation_group=group,
                       retrieved_doc_ids=list(pool), gold_ranks=ranks, n_gold=len(ranks),
                       n_distractors=pool_size-len(ranks), supporting_sentences=facts)
            augmented.append(new)
        used.update((ia, ib))
        if len(augmented)//2 >= max_pairs:
            break
    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_pairs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_original', action='store_true', help='append pairs to the original training rows')
    args = parser.parse_args()
    source, output = Path(args.train_file), Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError('augmentation must not overwrite source rows')
    data = source.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    pairs = build_pairs(rows, seed=args.seed, max_pairs=args.max_pairs)
    if not pairs:
        raise ValueError('no eligible natural pairs; do not invent labels or silently continue')
    result = rows+pairs if args.include_original else pairs
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in result))
    report = dict(source_sha256=hashlib.sha256(data).hexdigest(), seed=args.seed,
                  pairs=len(pairs)//2, rows=len(result), include_original=args.include_original,
                  protocol='train-only natural questions; same augmented rows for all arms')
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
