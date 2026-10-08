# PTKGE optimization

Self-contained optimization of the PTKGE polymer property-prediction model over
the four benchmark properties (Tg, Tm, EA, IP). The model is the verified
`ptkge_scaled` configuration (GIN + periodic-table physics embeddings + ECFP4,
initialized from PI1M contrastive pretraining). See `plan.md` for the full
experiment plan; the headline change is a **connected A-B copolymer graph** for
EA/IP (replacing the disjoint-union representation that left the two monomers
disconnected).

## Files
| file | role |
|---|---|
| `config.py` | paths (auto local/server), dataset metadata, protocol, recipe |
| `data_utils.py` | featurization, connected copolymer graphs, splits, molecule-count logging |
| `model.py` | `PTKGEModel` (frozen) + contrastive loss + ECFP4 utils |
| `embeddings.py` | periodic-table physics element embeddings |
| `model_copolymer.py` | `PTKGECopolymer` — extras head over the same encoder |
| `pretrain.py` | regenerate the PI1M contrastive checkpoint (optional; checkpoint provided) |
| `train.py` | per-run training engine (decoupled split/init, val-selected test eval) |
| `run.py` | run the full dataset × split × init matrix + per-dataset ensemble |
| `ensemble.py` | per-split init-seed mean/median ensembling, cross-split aggregation |
| `collect_results.py` | assemble the final RESULTS.md + results_table.json |
| `run_all.sh` | two-GPU launcher |
| `checkpoints/pretrained_physics.pt` | pretrained encoder (reused) |

## How to run
From this folder:

```bash
# quick sanity (3 epochs, 1 split, 1 init) on one dataset
python run.py --datasets tg --quick

# full campaign on two GPUs (tg,ea on GPU0; tm,ip on GPU1)
bash run_all.sh
# tail -f logs/gpu0.log logs/gpu1.log

# or run a subset on a single GPU
CUDA_VISIBLE_DEVICES=0 python run.py --datasets tg tm

# assemble the final table after everything finishes
python collect_results.py
```

The project root auto-detects between the local workstation and the GPU server.
To force a root, set `AISCIENTIST_ROOT`.

## Outputs
`results/RESULTS.md` holds the final RMSE table (single-seed mean ± std, and the
per-split init-seed ensemble mean/median, all in original units). Per-run JSON and
the ensemble JSONs are alongside it.
