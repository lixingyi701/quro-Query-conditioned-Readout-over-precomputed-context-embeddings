"""Paired last-checkpoint QA comparison, with training-protocol checks."""
import argparse
import json
from pathlib import Path
import random


def compare(baseline, candidate, split='dev', resamples=2000, allow_unmatched=False):
    results = [json.loads((Path(p)/'result.json').read_text()) for p in (baseline, candidate)]
    matches = {}
    for key, extract in (
        ('s0_checkpoint', lambda r: (r.get('s0_provenance') or {}).get('sha256')),
        ('train_file', lambda r: (r.get('data_provenance') or {}).get('train_sha256')),
        ('cache', lambda r: (r.get('data_provenance') or {}).get('cache_digest')),
        ('sample_order', lambda r: r.get('train_order_digest')),
    ):
        a, b = map(extract, results)
        matches[key] = a is not None and a == b
    if not all(matches.values()) and not allow_unmatched:
        raise ValueError('training protocol is missing or mismatched: '+str(matches))
    predictions = []
    for path in (baseline, candidate):
        rows = json.loads((Path(path)/f'predictions_{split}_D0_Bfull.json').read_text())
        items = {row['id']: row for row in rows}
        if len(items) != len(rows):
            raise ValueError('duplicate prediction IDs')
        predictions.append(items)
    a, b = predictions
    if not a or set(a) != set(b):
        raise ValueError('prediction ID sets differ; never intersect silently')
    ids = sorted(a)
    if any(a[i]['golds'] != b[i]['golds'] for i in ids):
        raise ValueError('prediction gold labels differ')
    if resamples < 100:
        raise ValueError('use at least 100 bootstrap resamples')
    output = dict(baseline=str(baseline), candidate=str(candidate), split=split,
                  n=len(ids), matched_training=matches, metrics={})
    for metric in ('f1', 'em'):
        differences = [100*(b[i][metric]-a[i][metric]) for i in ids]
        rng, n = random.Random(0), len(ids)
        boot = sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n
                      for _ in range(resamples))
        output['metrics'][metric] = dict(
            baseline=100*sum(a[i][metric] for i in ids)/n,
            candidate=100*sum(b[i][metric] for i in ids)/n,
            delta_pp=sum(differences)/n,
            paired_bootstrap_ci95_pp=[boot[int(.025*resamples)], boot[int(.975*resamples)-1]])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--split', default='dev')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--allow_unmatched_training', action='store_true',
                        help='explicit external S0/SQ reference; not an isolated treatment comparison')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.split, args.resamples, args.allow_unmatched_training)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
