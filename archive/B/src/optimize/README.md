# CPA-GAT optimization (35-dim atom features)

Self-contained optimization of the CPA-GAT (Cardinality-Preserving Attention GAT)
polymer property-prediction model over four benchmark properties (Tg, Tm, EA, IP).
The model uses the project's best single configuration, ORIG_BB (simple original
CPA-GAT + `is_backbone` node feature + SWA), with the learning rate tuned to
1e-4. The chemistry part of the node features is replaced with a richer 35-dim
atom encoding. See `plan.md` for the full experiment plan.

The whole matrix is run three independent times (`run1`, `run2`, `run3`) with
the identical recipe to measure pure run-to-run variance; the final reported
RMSE per dataset is mean ± std over the three runs' ensemble-mean RMSE.

## Files
| file | role |
|---|---|
| `config.py` | paths (auto local/server), dataset metadata, protocol, ORIG_BB recipe, `RUN_TAG`-aware results dir |
| `data_utils.py` | 3-RU oligomer graphs, 35-dim atom features, RWPE + PSPE + is_backbone, connected copolymer, splits, molecule-count logging |
| `model.py` | `CPAGATv2Conv` (frozen CPA core) + `CPAGATv2` |
| `train.py` | per-run training engine (ORIG_BB, lr 1e-4, decoupled split/init, SWA) |
| `run.py` | run the full dataset × split × init matrix + per-dataset ensemble |
| `ensemble.py` | per-split init-seed mean/median ensembling, cross-split aggregation |
| `collect_results.py` | assemble a single run's RESULTS.md + results_table.json |
| `aggregate_runs.py` | aggregate the per-run ensembles across run1/run2/run3 into mean ± std |
| `run_all.sh` | single-run two-GPU launcher |
| `run_variance.sh` | three-independent-run launcher (run2, run3, then aggregate) |

## How to run
From this folder:

```bash
# quick sanity (3 epochs, 1 split, 1 init)
python run.py --datasets tg --quick

# one independent run on two GPUs (tg,ea on GPU0; tm,ip on GPU1)
bash run_all.sh

# three independent runs (run1/run2/run3) + final aggregation
bash run_variance.sh
# tail -f logs/run2_gpu0.log logs/run2_gpu1.log
```

`RUN_TAG` env var routes a run's outputs to `results/<RUN_TAG>/` so multiple
independent runs coexist without overwriting each other. `run_variance.sh`
sets `RUN_TAG=run2` then `RUN_TAG=run3` (run1 is the same matrix without a tag,
or equivalently `RUN_TAG=run1`).

The project root auto-detects between the local workstation and the GPU server.
To force a root, set `AISCIENTIST_ROOT`.

## Outputs
- `results/run1/`, `results/run2/`, `results/run3/` — each holds that run's
  per-run JSONs, per-dataset `ensemble_<ds>.json`, and `RESULTS.md`.
- `results/AGGREGATE.md` + `results/aggregate_runs.json` — the final mean ± std
  table across the three independent runs (ens_mean / ens_median / single), in
  original units.
