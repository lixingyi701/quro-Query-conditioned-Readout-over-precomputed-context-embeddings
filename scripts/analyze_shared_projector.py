"""Tables for the first shared-projector wave (docs/SHARED_QUERY_PROJECTOR_RESULTS.md).

Reads predictions written by src.train under RUN_DIR and prints:
  1. dev/test (first 2000 rows) for the five trained arms and the published reader;
  2. full test (5405 rows) for S0m/SQ/SQX, split into rows already seen in (1) and new rows;
  3. the query-swap control on full test.
Every arm-vs-arm number is a paired difference over the same questions, with a
2000-resample bootstrap 95% CI (seed 0).
"""

import argparse, json, os, random

ARMS = [("published_direct", "base", "B80"), ("shared_agnostic", "S0m", "Bfull"),
        ("shared_contextual", "SQ", "Bfull"), ("shared_crossdoc", "SQX", "Bfull"),
        ("shared_last", "SL", "Bfull"), ("shared_word", "SW", "Bfull")]
FULL = [("fulltest_S0m", "S0m"), ("fulltest_SQ", "SQ"), ("fulltest_SQX", "SQX")]
CONTROLS = ("mismatch-q", "mismatch-q-both", "mismatch-doc")


def load(run_dir, arm, split, ctrl, b):
    p = f"{run_dir}/{arm}/predictions_{split}{'_' + ctrl if ctrl else ''}_D0_{b}.json"
    return {x["id"]: x for x in json.load(open(p))} if os.path.exists(p) else None


def mean(P, ids, k):
    return 100 * sum(P[i][k] for i in ids) / len(ids)


def boot(d, n=2000):
    rnd, N = random.Random(0), len(d)
    ms = sorted(sum(d[rnd.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n) - 1]


def paired(A, B, ids, k):
    d = [100 * (A[i][k] - B[i][k]) for i in ids]
    lo, hi = boot(d)
    return f"{sum(d) / len(d):+6.2f} [{lo:+.2f},{hi:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/data02/quro/runs/shared_projector_v1")
    ap.add_argument("--data_dir", default="/data02/quro/data/hotpot")
    a = ap.parse_args()
    meta, order = {}, {}
    for s in ("dev", "test"):
        rows = [json.loads(l) for l in open(f"{a.data_dir}/{s}.jsonl")]
        order[s] = [r["id"] for r in rows]
        meta.update({r["id"]: r for r in rows})

    for split in ("dev", "test"):
        print(f"\n### {split}, first 2000 rows, last checkpoint")
        print("arm     EM     F1    sub | F1 under: mismatch-q  -q-both  -doc | F1 bridge  comparison")
        preds = {}
        for arm, tag, b in ARMS:
            P = load(a.run_dir, arm, split, None, b)
            if P is None:
                continue
            preds[tag] = P
            ids = list(P)
            ctrl = [load(a.run_dir, arm, split, c, b) for c in CONTROLS]
            ctrl = [mean(C, ids, "f1") if C else float("nan") for C in ctrl]
            hop = [mean(P, [i for i in ids if meta[i]["hop_type"] == h], "f1")
                   for h in ("bridge", "comparison")]
            print(f"{tag:5s} {mean(P, ids, 'em'):6.2f} {mean(P, ids, 'f1'):6.2f} "
                  f"{mean(P, ids, 'substring'):6.2f} | {ctrl[0]:10.2f} {ctrl[1]:8.2f} "
                  f"{ctrl[2]:5.2f} | {hop[0]:9.2f} {hop[1]:11.2f}")
        for tag in ("SQ", "SQX", "SL", "SW"):
            ids = sorted(set(preds[tag]) & set(preds["S0m"]))
            print(f"  {tag}-S0m", "  ".join(f"{k} {paired(preds[tag], preds['S0m'], ids, k)}"
                                            for k in ("em", "f1", "substring")))

    full = {tag: {c: load(a.run_dir, arm, "test", c, "Bfull") for c in ("",) + CONTROLS}
            for arm, tag in FULL}
    if all(full[t][""] for t in full):
        subsets = {"all 5405": order["test"], "first 2000": order["test"][:2000],
                   "new 3405": order["test"][2000:]}
        for h in ("bridge", "comparison"):
            subsets[h] = [i for i in order["test"] if meta[i]["hop_type"] == h]
        for name, ids in subsets.items():
            print(f"\n### full test: {name} (n={len(ids)})")
            for t in full:
                P = full[t][""]
                print(f"  {t:4s} EM {mean(P, ids, 'em'):6.2f}  F1 {mean(P, ids, 'f1'):6.2f}  "
                      f"sub {mean(P, ids, 'substring'):6.2f}")
            for t in ("SQ", "SQX"):
                print(f"  {t}-S0m", "  ".join(
                    f"{k} {paired(full[t][''], full['S0m'][''], ids, k)}"
                    for k in ("em", "f1", "substring")))
        ids = order["test"]
        print("\n### full test controls: F1 under each condition")
        for t in full:
            print(f"  {t:4s}", "  ".join(f"{c or 'normal'} {mean(full[t][c], ids, 'f1'):.2f}"
                                         for c in full[t] if full[t][c]))
        print("  query-swap drop (normal - mismatch-q):")
        for t in full:
            print(f"    {t:4s}", "  ".join(
                f"{k} {paired(full[t][''], full[t]['mismatch-q'], ids, k)}"
                for k in ("em", "f1", "substring")))
        print("  swapped-query arm vs S0m (mismatch-q - S0m normal):")
        for t in ("SQ", "SQX"):
            print(f"    {t:4s}", "  ".join(
                f"{k} {paired(full[t]['mismatch-q'], full['S0m'][''], ids, k)}"
                for k in ("em", "f1")))


if __name__ == "__main__":
    main()
