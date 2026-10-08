"""
Aggregate the per-dataset ensemble summaries across multiple independent runs
(run1, run2, run3, ...) into a single mean +/- std table.

For each dataset, the headline ensemble_mean_rmse of every run is collected
(from results/<RUN_TAG>/ensemble_<ds>.json), and the final reported RMSE is
mean +/- std over the N independent runs.

Usage:
    python aggregate_runs.py                       # auto-detects run1/run2/.../runN
    python aggregate_runs.py --runs run1 run2 run3
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def _load_ensemble(run_tag, dataset):
    f = os.path.join(C._OPT_DIR, "results", run_tag, f"ensemble_{dataset}.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def aggregate(runs):
    rows = []
    for ds in C.ALL_DATASETS:
        per_run = []           # one dict per run with ens_mean / ens_median / single
        for r in runs:
            s = _load_ensemble(r, ds)
            if s is None:
                print(f"[aggregate] WARN: {r}/ensemble_{ds}.json missing")
                continue
            per_run.append(dict(
                run=r,
                ens_mean=s["ensemble_mean_rmse_splits_mean"],
                ens_median=s["ensemble_median_rmse_splits_mean"],
                single=s["single_seed_all_mean"],
                n_splits=s["n_splits"],
                protocol=s["protocol"],
            ))
        if not per_run:
            rows.append(dict(dataset=ds, status="missing", runs=per_run))
            continue

        ens_mean_vals = np.array([p["ens_mean"] for p in per_run])
        ens_median_vals = np.array([p["ens_median"] for p in per_run])
        single_vals = np.array([p["single"] for p in per_run])

        rows.append(dict(
            dataset=ds, kind="copolymer" if ds in ("ea", "ip") else "single",
            protocol=per_run[0]["protocol"], n_splits=per_run[0]["n_splits"],
            n_runs=len(per_run),
            ens_mean_mean=float(ens_mean_vals.mean()),
            ens_mean_std=float(ens_mean_vals.std(ddof=0)),
            ens_median_mean=float(ens_median_vals.mean()),
            ens_median_std=float(ens_median_vals.std(ddof=0)),
            single_mean=float(single_vals.mean()),
            single_std=float(single_vals.std(ddof=0)),
            per_run=per_run,
            per_run_ens_mean=[float(v) for v in ens_mean_vals],
        ))

    # ---------- write JSON ----------
    out_json = os.path.join(C._OPT_DIR, "results", "aggregate_runs.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    # ---------- write markdown ----------
    lines = [
        f"# CPA-GAT (ORIG_BB, atomfeat35) — aggregation over {len(runs)} independent runs",
        "",
        f"Runs: {', '.join(runs)}",
        "Final RMSE = mean +/- std over the independent runs of each run's ens_mean RMSE.",
        "(Each run's ens_mean RMSE is itself the mean over splits of the per-split 3-init ensemble.)",
        "RMSE in original units.",
        "",
        "| dataset | kind | protocol | n_runs | ens_mean RMSE | ens_median RMSE | single RMSE | per-run ens_mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("status") == "missing":
            lines.append(f"| {r['dataset']} | - | - | - | MISSING | - | - | - |")
            continue
        per_run_str = ", ".join(f"{v:.4f}" for v in r["per_run_ens_mean"])
        lines.append(
            f"| {r['dataset']} | {r['kind']} | {r['protocol']} | {r['n_runs']} | "
            f"{r['ens_mean_mean']:.4f} +/- {r['ens_mean_std']:.4f} | "
            f"{r['ens_median_mean']:.4f} +/- {r['ens_median_std']:.4f} | "
            f"{r['single_mean']:.4f} +/- {r['single_std']:.4f} | {per_run_str} |")

    lines.append("")
    lines.append("## Per-run detail")
    for r in rows:
        if r.get("status") == "missing":
            continue
        lines.append(f"\n**{r['dataset']}**")
        for p in r["per_run"]:
            lines.append(
                f"- {p['run']}: ens_mean {p['ens_mean']:.4f} | "
                f"ens_median {p['ens_median']:.4f} | single {p['single']:.4f}")

    out_md = os.path.join(C._OPT_DIR, "results", "AGGREGATE.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_md} and aggregate_runs.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run tags to aggregate (default: auto-detect run1..runN)")
    args = ap.parse_args()
    if args.runs is None:
        results_root = os.path.join(C._OPT_DIR, "results")
        runs = sorted(d for d in os.listdir(results_root)
                      if d.startswith("run") and
                      os.path.isdir(os.path.join(results_root, d)))
        if not runs:
            print("No run subdirectories found under results/.")
            sys.exit(1)
    else:
        runs = args.runs
    aggregate(runs)
