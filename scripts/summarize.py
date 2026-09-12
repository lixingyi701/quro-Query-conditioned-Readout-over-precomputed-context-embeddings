"""Summarize QuRO v0.0 result.json files."""

import json
import os
import sys


def load(directory):
    path = os.path.join(directory, "result.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    value["_dir"] = directory
    return value


def main(directories):
    runs = [value for value in map(load, directories) if value]
    if not runs:
        raise SystemExit("no result.json found")
    names = []
    for run in runs:
        for name in run["metrics"]:
            if name not in names:
                names.append(name)
    print("| tag | query mode | m | budgets | cache | " +
          " | ".join(names) + " |")
    print("|---|---:|---:|---:|---|" + "---:|" * len(names))
    for run in runs:
        cells = []
        for name in names:
            metric = run["metrics"].get(name)
            cells.append("--" if metric is None else
                         f"{metric['em']:.1%}/{metric['f1']:.3f} (xi={metric.get('xi_eff')})")
        print("| " + " | ".join([
            str(run.get("tag") or os.path.basename(run["_dir"])),
            str(run["output_query_mode"]),
            str(run["offline_m"]),
            ",".join(map(str, run["budget_buckets"])),
            str(run["cache_compressor"]),
            *cells,
        ]) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
