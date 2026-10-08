"""
Run all standard baseline experiments: 2 datasets x 4 models x 2 seeds x 3 runs = 48 runs.
Each run: train on train set, select best epoch by val RMSE, eval once on test.
Per (dataset, model, seed): aggregate 3 runs -> test RMSE mean +/- std.
Per (dataset, model): aggregate 6 runs (2 seeds x 3 runs) -> grand mean +/- std.
Results are appended to results.json after each run.
"""
import os
import json
import time
import argparse
import numpy as np
import torch

from data_utils import load_dataset
from models import build_model
from train import train_graph, train_ecfp

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")

DATASETS = ['Tg', 'Tm']
MODELS = ['ecfp_mlp', 'gcn', 'gin', 'gat']
SEEDS = [42, 71]
N_RUNS = 3         # 3 model-init runs per (dataset, model, split-seed)
EPOCHS = 300
PATIENCE = 50


def device_setup():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_record(rec):
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, 'r') as f:
            db = json.load(f)
    else:
        db = []
    db.append(rec)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(db, f, indent=2)


def already_done(dataset, model, seed, run_id):
    """Check if a single run already in results.json (for resume)."""
    if not os.path.exists(RESULTS_JSON):
        return None
    with open(RESULTS_JSON, 'r') as f:
        db = json.load(f)
    for r in db:
        if (r['dataset'] == dataset and r['model'] == model
                and r['split_seed'] == seed and r['run_id'] == run_id):
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--n_runs', type=int, default=N_RUNS)
    args = ap.parse_args()

    device = device_setup()
    print(f"[device] {device}")
    if device == 'cuda':
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # preload datasets for both seeds
    data_cache = {}
    for ds in args.datasets:
        for sd in args.seeds:
            key = (ds, sd)
            if key not in data_cache:
                data_cache[key] = load_dataset(ds, sd)

    t0_all = time.time()
    for ds in args.datasets:
        for sd in args.seeds:
            data = data_cache[(ds, sd)]
            scaler = data['scaler']
            for mdl in args.models:
                for run_id in range(1, args.n_runs + 1):
                    prev = already_done(ds, mdl, sd, run_id)
                    if prev is not None:
                        print(f"[skip] {ds} {mdl} seed={sd} run={run_id} -> "
                              f"rmse={prev['test_rmse']:.4f}")
                        continue
                    init_seed = 1000 * sd + run_id  # deterministic, distinct per run
                    print(f"\n[RUN] {ds} | {mdl} | split_seed={sd} | "
                          f"init_seed={init_seed} | run {run_id}/{args.n_runs}")
                    t0 = time.time()
                    torch.manual_seed(init_seed); np.random.seed(init_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(init_seed)
                        torch.cuda.empty_cache()
                    m = build_model(mdl, node_dim=data['node_dim'],
                                    ecfp_bits=data['ecfp_bits'], device=device)
                    if mdl == 'ecfp_mlp':
                        res = train_ecfp(
                            m,
                            data['train_ecfp'], data['train_y_norm'],
                            data['val_ecfp'], data['val_y_norm'],
                            data['test_ecfp'], data['test_y_norm'],
                            scaler, device,
                            epochs=EPOCHS, patience=PATIENCE, seed=init_seed,
                        )
                    else:
                        res = train_graph(
                            m, data['train_graphs'], data['val_graphs'],
                            data['test_graphs'], scaler, device,
                            epochs=EPOCHS, patience=PATIENCE, seed=init_seed,
                        )
                    res.update({
                        'dataset': ds, 'model': mdl,
                        'split_seed': sd, 'init_seed': init_seed,
                        'run_id': run_id, 'elapsed_s': round(time.time() - t0, 1),
                    })
                    print(f"   -> test_rmse={res['test_rmse']:.4f}  "
                          f"test_mae={res['test_mae']:.4f}  "
                          f"best_epoch={res['best_epoch']}  "
                          f"({res['elapsed_s']}s)")
                    save_record(res)

    print(f"\n[done] all runs finished in {time.time() - t0_all:.1f}s")


if __name__ == '__main__':
    main()
