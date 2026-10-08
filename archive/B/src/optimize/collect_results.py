"""
Collect all per-dataset ensemble summaries into a single results table
(RESULTS.md + results_table.json) for the CPA-GAT campaign.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def collect():
    rows = []
    for ds in C.ALL_DATASETS:
        f = os.path.join(C.RESULTS_DIR, f"ensemble_{ds}.json")
        if not os.path.exists(f):
            rows.append(dict(dataset=ds, status="missing"))
            continue
        with open(f) as fh:
            s = json.load(fh)
        rows.append(dict(
            dataset=ds,
            kind="copolymer" if ds in ("ea", "ip") else "single",
            protocol=s["protocol"],
            n_splits=s["n_splits"],
            single_seed_mean=s["single_seed_all_mean"],
            single_seed_std=s["single_seed_all_std"],
            single_seed_aggregation=s.get("single_seed_aggregation"),
            ensemble_mean_rmse=s["ensemble_mean_rmse_splits_mean"],
            ensemble_mean_std=s["ensemble_mean_rmse_splits_std"],
            ensemble_median_rmse=s["ensemble_median_rmse_splits_mean"],
            ensemble_median_std=s["ensemble_median_rmse_splits_std"],
            per_split=[dict(sp=p["split_tag"], n=p["n_init"],
                            single_mean=p["single_seed_mean"],
                            init_std=p["single_seed_std"],
                            swa_mean=p.get("swa_rmse_mean"),
                            ens_mean=p["ensemble_mean_rmse"],
                            ens_median=p["ensemble_median_rmse"])
                       for p in s["per_split"]],
        ))

    with open(os.path.join(C.RESULTS_DIR, "results_table.json"), "w") as f:
        json.dump(rows, f, indent=2)

    lines = ["# CPA-GAT (ORIG_BB) optimization — final results", "",
             "RMSE in original units.",
             "`ens_mean` / `ens_median` = per-split init-seed ensemble, then mean +/- std over splits (tg/tm) or the single official split (ea/ip).",
             "`single` headline: tg/tm = mean +/- std over splits of the per-split init-mean; ea/ip = mean +/- std over init seeds.",
             "Per-split `init_std` = std over the 3 init seeds within that split.", "",
             "| dataset | kind | protocol | n_splits | single RMSE | ens mean RMSE | ens median RMSE |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "missing":
            lines.append(f"| {r['dataset']} | - | - | - | MISSING | - | - |"); continue
        lines.append(
            f"| {r['dataset']} | {r['kind']} | {r['protocol']} | {r['n_splits']} | "
            f"{r['single_seed_mean']:.4f} +/- {r['single_seed_std']:.4f} | "
            f"{r['ensemble_mean_rmse']:.4f} +/- {r['ensemble_mean_std']:.4f} | "
            f"{r['ensemble_median_rmse']:.4f} +/- {r['ensemble_median_std']:.4f} |")
    lines.append("")
    lines.append("## Per-split detail")
    for r in rows:
        if r.get("status") == "missing":
            continue
        lines.append(f"\n**{r['dataset']}**")
        for p in r["per_split"]:
            swa = f" swa_mean {p['swa_mean']:.4f}" if p.get("swa_mean") is not None else ""
            lines.append(f"- split {p['sp']} (n_init={p['n']}): "
                         f"single {p['single_mean']:.4f} (init_std {p['init_std']:.4f}) | "
                         f"ens_mean {p['ens_mean']:.4f} | ens_median {p['ens_median']:.4f}{swa}")

    with open(os.path.join(C.RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {C.RESULTS_DIR}/RESULTS.md and results_table.json")


if __name__ == "__main__":
    collect()
