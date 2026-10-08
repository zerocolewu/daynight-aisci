"""
Run the full CPA-GAT (ORIG_BB) evaluation matrix.

Per dataset:
    tg, tm  -> 2 splits {42,71} x 3 init seeds {42,71,2026} = 6 runs each
    ea, ip  -> official split x 3 init seeds = 3 runs (copolymer, connected A-B graph)
A run whose result JSON already exists is skipped (idempotent resume; --force re-runs).
After all runs of a dataset finish, the per-split (3-init) ensemble is computed and the
per-split ensemble RMSEs are aggregated as mean +/- std over the 2 splits (tg/tm).

Usage:
    python run.py --datasets all
    python run.py --datasets tg tm
    python run.py --datasets tg --quick
    python run.py --datasets all --force         # re-run even existing results
The GPU is selected by the caller via CUDA_VISIBLE_DEVICES (see run_all.sh).
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import train as T
import ensemble as E


def split_seeds_for(dataset, splits, quick):
    if dataset in C.MULTI_SPLIT_DATASETS:
        return [71] if quick else list(splits)
    return [None]


def init_seeds_for(inits, quick):
    return [42] if quick else list(inits)


def make_tag(dataset, sp, init, quick):
    base = f"{dataset}_sp{'given' if sp is None else sp}_init{init}"
    # quick smoke runs are prefixed so they never collide with real results and
    # are never picked up by the ensemble glob ({ds}_sp*_init*.json).
    return f"quick_{base}" if quick else base


def already_done(path):
    # train_one writes the JSON only after test evaluation, so existence implies a
    # complete run; the field checks guard against a corrupt / half-written file.
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            r = json.load(f)
        return "test_rmse" in r and bool(r.get("test_preds"))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--splits", nargs="+", type=int, default=C.SPLIT_SEEDS)
    ap.add_argument("--inits", nargs="+", type=int, default=C.INIT_SEEDS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the result JSON already exists")
    ap.add_argument("--skip-ensemble", action="store_true")
    args = ap.parse_args()

    datasets = C.ALL_DATASETS if args.datasets == ["all"] else args.datasets
    for d in datasets:
        if d not in C.ALL_DATASETS:
            print(f"[skip] unknown dataset {d}"); continue
        for sp in split_seeds_for(d, args.splits, args.quick):
            for init in init_seeds_for(args.inits, args.quick):
                tag = make_tag(d, sp, init, args.quick)
                out = os.path.join(C.RESULTS_DIR, f"{tag}.json")
                if not args.force and already_done(out):
                    print(f"[skip] {tag} (result exists; --force to re-run)", flush=True)
                    continue
                print(f"\n===== RUN {tag} =====", flush=True)
                try:
                    T.train_one(d, sp, init, quick=args.quick)
                except Exception as e:
                    import traceback
                    print(f"[ERROR] {tag}: {e}\n{traceback.format_exc()}", flush=True)
        if not args.skip_ensemble:
            print(f"\n===== ENSEMBLE {d} =====", flush=True)
            E.ensemble_dataset(d)


if __name__ == "__main__":
    main()
