"""
Cross-run aggregation of independent PTKGE campaigns.

Each independent run lives in `results/runN/` with its own `results_table.json`
(produced by `PTKGE_RUN_ID=runN python collect_results.py`). This script pools
the per-run `ensemble_mean_rmse` (and single-seed headline) of every dataset
across all discovered runs and reports mean +/- std -- the final reported
number with run-to-run variance.

Usage:
    python aggregate_runs.py                 # auto-discover results/run*/
    python aggregate_runs.py --runs run1 run2 run3
"""
import os
import sys
import re
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


_RUN_RE = re.compile(r"^run(\d+)$")


def discover_runs(explicit=None):
    if explicit:
        run_ids = explicit
    else:
        run_ids = []
        for name in sorted(os.listdir(C.RESULTS_DIR)):
            full = os.path.join(C.RESULTS_DIR, name)
            if os.path.isdir(full) and _RUN_RE.match(name):
                run_ids.append(name)
    runs = []
    for rid in run_ids:
        tbl = os.path.join(C.RESULTS_DIR, rid, "results_table.json")
        if not os.path.exists(tbl):
            print(f"[aggregate] WARNING: {rid}/results_table.json missing -- skipped",
                  file=sys.stderr)
            continue
        with open(tbl) as f:
            runs.append((rid, json.load(f)))
    if not runs:
        print("[aggregate] no run subfolders with results_table.json found",
              file=sys.stderr)
        sys.exit(1)
    runs.sort(key=lambda x: int(_RUN_RE.match(x[0]).group(1)))
    return runs


def aggregate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="explicit run ids (default: auto-discover results/run*/)")
    args = ap.parse_args()

    runs = discover_runs(args.runs)
    n_runs = len(runs)
    print(f"[aggregate] {n_runs} independent run(s): "
          f"{', '.join(r[0] for r in runs)}", flush=True)

    # collect per-dataset stats across runs
    datasets = C.ALL_DATASETS
    pooled = {ds: dict(ens_mean=[], ens_median=[], single=[]) for ds in datasets}
    for rid, tbl in runs:
        by_ds = {r["dataset"]: r for r in tbl if r.get("dataset")}
        for ds in datasets:
            r = by_ds.get(ds)
            if not r or r.get("status") == "missing":
                continue
            pooled[ds]["ens_mean"].append(float(r["ensemble_mean_rmse"]))
            pooled[ds]["ens_median"].append(float(r["ensemble_median_rmse"]))
            pooled[ds]["single"].append(float(r["single_seed_mean"]))

    rows = []
    for ds in datasets:
        ens = pooled[ds]["ens_mean"]
        med = pooled[ds]["ens_median"]
        sng = pooled[ds]["single"]
        if not ens:
            rows.append(dict(dataset=ds, status="missing")); continue
        rows.append(dict(
            dataset=ds,
            kind="copolymer" if ds in ("ea", "ip") else "single",
            n_runs=len(ens),
            per_run_ens_mean=[round(x, 6) for x in ens],
            per_run_ens_median=[round(x, 6) for x in med],
            per_run_single=[round(x, 6) for x in sng],
            ens_mean_across_runs_mean=float(np.mean(ens)),
            ens_mean_across_runs_std=float(np.std(ens)),
            ens_median_across_runs_mean=float(np.mean(med)),
            ens_median_across_runs_std=float(np.std(med)),
            single_across_runs_mean=float(np.mean(sng)),
            single_across_runs_std=float(np.std(sng)),
        ))

    out_json = os.path.join(C.RESULTS_DIR, "aggregated_runs.json")
    with open(out_json, "w") as f:
        json.dump(dict(n_runs=n_runs, runs=[r[0] for r in runs], rows=rows), f, indent=2)

    # ---- Markdown ----
    lines = [
        "# PTKGE optimization — cross-run aggregated results", "",
        f"Mean +/- std of each run's per-dataset ensemble RMSE over {n_runs} independent "
        "run(s) (identical seeds and config; variance is hardware/cuDNN non-determinism).",
        f"Runs: {', '.join(r[0] for r in runs)}.",
        "RMSE in original units.", "",
        "| dataset | kind | n_runs | per-run ens_mean | ens_mean (mean +/- std) | "
        "ens_median (mean +/- std) | single (mean +/- std) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("status") == "missing":
            lines.append(f"| {r['dataset']} | - | 0 | - | MISSING | - | - |"); continue
        per_run = ", ".join(f"{x:.4f}" for x in r["per_run_ens_mean"])
        lines.append(
            f"| {r['dataset']} | {r['kind']} | {r['n_runs']} | {per_run} | "
            f"{r['ens_mean_across_runs_mean']:.4f} +/- {r['ens_mean_across_runs_std']:.4f} | "
            f"{r['ens_median_across_runs_mean']:.4f} +/- {r['ens_median_across_runs_std']:.4f} | "
            f"{r['single_across_runs_mean']:.4f} +/- {r['single_across_runs_std']:.4f} |")
    out_md = os.path.join(C.RESULTS_DIR, "AGGREGATED_RUNS.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_md} and aggregated_runs.json")


if __name__ == "__main__":
    aggregate()
