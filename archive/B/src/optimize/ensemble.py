"""
Per-dataset ensembling of the init-seed runs (same protocol as the PTKGE folder).

Within one split the test set is identical across init seeds, so predictions can
be averaged (mean / median) in original units. For tg/tm the ensemble is computed
inside each of the 2 splits, then the 2 per-split ensemble RMSEs are aggregated
as mean +/- std; for ea/ip the single official split is used. Each run also
carries a `swa_test_rmse` reference (SWA-averaged model); the ensemble itself
averages the val-selected best-model predictions.
"""
import os
import sys
import glob
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def _rmse_mae_r2(preds, targets):
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    assert preds.shape == targets.shape, f"preds {preds.shape} != targets {targets.shape}"
    err = preds - targets
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                mae=float(np.mean(np.abs(err))),
                r2=float(1 - np.sum(err ** 2) / max(np.sum((targets - targets.mean()) ** 2), 1e-12)))


def _load_runs(dataset):
    runs = []
    for f in glob.glob(os.path.join(C.RESULTS_DIR, f"{dataset}_sp*_init*.json")):
        with open(f) as fh:
            runs.append(json.load(fh))
    runs.sort(key=lambda r: (r["split_tag"], r["init_seed"]))
    return runs


def ensemble_dataset(dataset, write=True):
    runs = _load_runs(dataset)
    if not runs:
        print(f"[ensemble {dataset}] no run files found", flush=True)
        return None

    by_split = {}
    for r in runs:
        by_split.setdefault(r["split_tag"], []).append(r)

    split_summaries = []
    for sp_tag, group in sorted(by_split.items()):
        group.sort(key=lambda r: r["init_seed"])
        targets = np.asarray(group[0]["test_targets"], dtype=np.float64).reshape(-1)
        preds_mat, single, swa = [], [], []
        for r in group:
            p = np.asarray(r["test_preds"], dtype=np.float64).reshape(-1)
            preds_mat.append(p)
            single.append(_rmse_mae_r2(p, targets)["rmse"])
            if r.get("swa_test_rmse") is not None:
                swa.append(r["swa_test_rmse"])
        preds_mat = np.vstack(preds_mat)
        assert preds_mat.ndim == 2 and preds_mat.shape == (len(group), len(targets)), \
            f"preds_mat {preds_mat.shape} expected ({len(group)}, {len(targets)})"
        mean_m = _rmse_mae_r2(preds_mat.mean(axis=0), targets)
        median_m = _rmse_mae_r2(np.median(preds_mat, axis=0), targets)
        split_summaries.append(dict(
            split_tag=sp_tag, n_init=len(group),
            init_seeds=[int(r["init_seed"]) for r in group],
            single_seed_rmse=[float(x) for x in single],
            single_seed_mean=float(np.mean(single)), single_seed_std=float(np.std(single)),
            swa_rmse_mean=float(np.mean(swa)) if swa else None,
            ensemble_mean_rmse=mean_m["rmse"], ensemble_mean_mae=mean_m["mae"],
            ensemble_mean_r2=mean_m["r2"],
            ensemble_median_rmse=median_m["rmse"], ensemble_median_mae=median_m["mae"],
            ensemble_median_r2=median_m["r2"],
            test_size=int(len(targets)),
        ))
        print(f"[ensemble {dataset} split={sp_tag}] "
              f"single {np.mean(single):.4f}+/-{np.std(single):.4f}  "
              f"ens_mean {mean_m['rmse']:.4f}  ens_median {median_m['rmse']:.4f}  "
              f"(n_init={len(group)}, n_test={len(targets)})", flush=True)

    ens_mean_across = [s["ensemble_mean_rmse"] for s in split_summaries]
    ens_median_across = [s["ensemble_median_rmse"] for s in split_summaries]
    per_split_single_means = [s["single_seed_mean"] for s in split_summaries]
    if len(split_summaries) > 1:
        # tg/tm: single-model headline = mean+-std over splits of the per-split init-mean
        # (same two-level structure as the ensemble, so the two +/- are comparable).
        single_head_mean = float(np.mean(per_split_single_means))
        single_head_std = float(np.std(per_split_single_means))
        single_agg = "mean/std over splits of per-split init-mean"
    else:
        # ea/ip: one split -> single headline = that split's mean+-std over init seeds
        single_head_mean = float(split_summaries[0]["single_seed_mean"])
        single_head_std = float(split_summaries[0]["single_seed_std"])
        single_agg = "mean/std over init seeds"

    summary = dict(
        dataset=dataset, n_splits=len(split_summaries),
        protocol="multi-split x multi-init" if dataset in C.MULTI_SPLIT_DATASETS
                 else "given-split x multi-init",
        per_split=split_summaries,
        ensemble_mean_rmse_splits_mean=float(np.mean(ens_mean_across)),
        ensemble_mean_rmse_splits_std=float(np.std(ens_mean_across)),
        ensemble_median_rmse_splits_mean=float(np.mean(ens_median_across)),
        ensemble_median_rmse_splits_std=float(np.std(ens_median_across)),
        single_seed_all_mean=single_head_mean,
        single_seed_all_std=single_head_std,
        single_seed_aggregation=single_agg,
    )
    print(f"[ensemble {dataset}] FINAL "
          f"ens_mean={np.mean(ens_mean_across):.4f}+/-{np.std(ens_mean_across):.4f} "
          f"(over {len(ens_mean_across)} split(s)); "
          f"single={single_head_mean:.4f}+/-{single_head_std:.4f} [{single_agg}]", flush=True)

    if write:
        with open(os.path.join(C.RESULTS_DIR, f"ensemble_{dataset}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "tg"
    ensemble_dataset(ds)
