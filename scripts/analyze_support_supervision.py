"""Tables for the first support-supervision wave (docs/SUPPORT_DOCUMENT_SUPERVISION_RESULTS.md).

Compares SQ+Head (CE-only continuation) with SQ+Doc (CE + support BCE) on dev,
plus the original SQ / S0m checkpoints of the shared-projector wave. Every
arm-vs-arm number is a paired difference over the same 2000 questions with a
2000-resample bootstrap 95% CI (seed 0).
"""

import argparse, json, random, statistics


def load(path):
    return {x["id"]: x for x in json.load(open(path))}


def mean(P, ids, k):
    return 100 * sum(P[i][k] for i in ids) / len(ids)


def boot(d, n=2000):
    rnd, N = random.Random(0), len(d)
    ms = sorted(sum(d[rnd.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n) - 1]


def fmt(d):
    lo, hi = boot(d)
    return f"{sum(d) / len(d):+.2f} [{lo:+.2f},{hi:+.2f}]"


def paired(A, B, ids, k):
    return fmt([100 * (A[i][k] - B[i][k]) for i in ids])


def top2(logits):
    return set(sorted(range(len(logits)), key=lambda k: (-logits[k], k))[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/data02/quro/runs/support_doc_v1")
    ap.add_argument("--projector_dir", default="/data02/quro/runs/shared_projector_v1")
    ap.add_argument("--dev", default="/data02/quro/data/hotpot/dev.jsonl")
    a = ap.parse_args()
    dev = [json.loads(l) for l in open(a.dev)]
    meta = {r["id"]: r for r in dev}
    arms = {"Head": "sq_ce", "Doc": "sq_ce_doc"}
    P = {n: load(f"{a.run_dir}/{d}/predictions_dev_D0_Bfull.json") for n, d in arms.items()}
    MQ = {n: load(f"{a.run_dir}/{d}/predictions_dev_mismatch-q_D0_Bfull.json") for n, d in arms.items()}
    P["SQ"] = load(f"{a.projector_dir}/shared_contextual/predictions_dev_D0_Bfull.json")
    P["S0m"] = load(f"{a.projector_dir}/shared_agnostic/predictions_dev_D0_Bfull.json")
    ids = list(P["Head"])

    print("### dev 2000, last checkpoint")
    for n in P:
        print(f"  {n:4s}", "  ".join(f"{k} {mean(P[n], ids, k):.2f}" for k in ("em", "f1", "substring")))
    print("  Doc-Head", "  ".join(f"{k} {paired(P['Doc'], P['Head'], ids, k)}"
                                  for k in ("em", "f1", "substring")))
    for x, y in (("Head", "SQ"), ("Doc", "SQ"), ("Head", "S0m"), ("Doc", "S0m")):
        print(f"  {x}-{y} f1 {paired(P[x], P[y], ids, 'f1')}")
    subsets = {h: [i for i in ids if meta[i]["hop_type"] == h] for h in ("bridge", "comparison")}
    subsets["all gold visible"] = [i for i in ids if all(
        v is True for y, v in zip(P["Doc"][i]["support"]["labels"], P["Doc"][i]["support"]["visible"]) if y)]
    for name, s in subsets.items():
        print(f"  {name} (n={len(s)}) Doc-Head f1 {paired(P['Doc'], P['Head'], s, 'f1')}")

    print("\n### query swap: F1 drop (normal - mismatch-q)")
    drops = {n: [100 * (P[n][i]["f1"] - MQ[n][i]["f1"]) for i in ids] for n in arms}
    for n in arms:
        print(f"  {n:4s} {fmt(drops[n])}")
    print(f"  Doc-Head {fmt([d - h for d, h in zip(drops['Doc'], drops['Head'])])}")

    print("\n### support metrics (result.json, dev 2000)")
    for n, d in arms.items():
        m = json.load(open(f"{a.run_dir}/{d}/result.json"))["metrics"]
        for split in ("dev|D0|B=full", "dev/mismatch-q|D0|B=full"):
            print(f"  {n:4s} {split:26s}", {k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in m[split].items() if k.startswith("support_")})
    rank = sum(len(set(r["gold_ranks"]) & {0, 1}) / len(r["gold_ranks"]) for r in dev) / len(dev)
    print(f"  retrieval-rank top-2 Recall@2: {rank:.4f}")

    print("\n### does the trained head use the query? (SQ+Doc, normal vs mismatch-q)")
    N, M = P["Doc"], MQ["Doc"]
    xs = [v for i in ids for v in N[i]["support"]["logits"]]
    ys = [v for i in ids for v in M[i]["support"]["logits"]]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    r = (sum((p - mx) * (q - my) for p, q in zip(xs, ys))
         / (sum((p - mx) ** 2 for p in xs) ** .5 * sum((q - my) ** 2 for q in ys) ** .5))
    shift = statistics.median(max(abs(p - q) for p, q in zip(N[i]["support"]["logits"],
                                                            M[i]["support"]["logits"])) for i in ids)
    same = sum(top2(N[i]["support"]["logits"]) == top2(M[i]["support"]["logits"]) for i in ids)
    print(f"  logit corr {r:.4f}  logit std {statistics.pstdev(xs):.3f}  "
          f"median per-question max|dlogit| {shift:.4f}  top-2 unchanged {same}/{len(ids)}")

    print("\n### validation curve (dev 500): step:F1/Recall@2")
    for n, d in arms.items():
        v = [json.loads(l)["validation"] for l in open(f"{a.run_dir}/{d}/train_log.jsonl")
             if l.startswith('{"validation"')]
        print(f"  {n:4s}", " ".join(f"{x['step']}:{100 * x['f1']:.1f}/{x.get('support_recall_at_2', 0):.2f}"
                                    for x in v))
    log = [json.loads(l) for l in open(f"{a.run_dir}/sq_ce_doc/train_log.jsonl") if l.startswith('{"step"')]
    print("  Doc support_loss:", " ".join(f"{x['step']}:{x['support_loss']:.3f}" for x in log[::5]))


if __name__ == "__main__":
    main()
