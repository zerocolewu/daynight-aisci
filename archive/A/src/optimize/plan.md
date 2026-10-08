# PTKGE optimization plan

## Goal
Lower polymer property-prediction RMSE across the four benchmark properties
(Tg, Tm, EA, IP) with the PTKGE model — a GIN encoder over atomic numbers,
augmented with physics-informed periodic-table element embeddings and an ECFP4
branch, initialized from contrastive pretraining on PI1M. The model and the
pretraining recipe are fixed (the verified `ptkge_scaled` configuration); the
work in this folder is a clean, protocol-aligned evaluation plus one targeted
fix to the copolymer pipeline.

## What is frozen
- **Architecture**: `PTKGEModel` — GIN (5 layers, hidden 256) with periodic-table
  physics element-identity embedding (8 properties), ECFP4 (2048-bit) MLP branch,
  concatenation fusion, MLP predictor. Atomic-number-only node features.
- **Pretraining**: InfoNCE contrastive on PI1M (ECFP4-Tanimoto positives), physics
  strategy, AdamW lr 5e-5, cosine, 30 epochs. The resulting checkpoint
  `checkpoints/pretrained_physics.pt` is reused verbatim (see `pretrain.py` to
  regenerate from scratch).

## The one modeling change: connected copolymer graphs for EA/IP
The previous EA/IP pipeline represented a copolymer as the **disjoint union** of
the two monomer graphs — A and B were never bonded, so the GNN could not pass
messages across the monomer interface and could not model the composition effect
that defines copolymer properties. This is the root cause of the poor EA/IP RMSE.

Here, each copolymer is built as a single **connected A-B molecule**: an explicit
bond is drawn between monomer A's exit connection point and monomer B's entry
point, the outer stars are capped with hydrogen, and the inner stars are removed
(`build_connected_copolymer_mol`). The atomic-number GIN now propagates across
the interface. Composition scalars `[rA, rB, type-one-hot]` are concatenated to
the fused embedding (`PTKGECopolymer`). ECFP4 is computed on the connected
molecule. These two datasets are trained **from scratch** — PI1M is single-
monomer, so the contrastive prior does not transfer, and the corpora are large
(~30k train) enough to train directly.

## Evaluation protocol (decoupled split / init)
- **Tg, Tm** (no official split): 2 partitions from `split_seed in {42, 71}`,
  each a deterministic 72/8/20 split. Inside each split, 3 training seeds
  `init_seed in {42, 71, 2026}`. -> 6 runs each.
- **EA, IP** (official split): the given `ea_split.pkl` /
  `ip_split.pkl` defines one partition; 3 training seeds. -> 3 runs each.
- Targets are normalized with the per-split train mean/std; the model predicts in
  normalized space, metrics are reported in original units. Early stopping on
  validation RMSE (original units); the best checkpoint is evaluated once on test.
- **Ensembling**: inside each split, the 3 init-seed predictions are averaged
  (mean and median) in original units. For Tg/Tm the two per-split ensemble
  RMSEs are aggregated as mean ± std; for EA/IP the single ensemble is
  reported. Single-seed mean ± std is reported alongside. No best-seed or
  best-split selection.

## Run matrix
| dataset | splits | init seeds | runs | pretrained? |
|---|---|---|---|---|
| Tg  | 2 | 3 | 6 | yes |
| Tm  | 2 | 3 | 6 | yes |
| EA  | 1 (official) | 3 | 3 | no (copolymer, scratch) |
| IP  | 1 (official) | 3 | 3 | no (copolymer, scratch) |

## Outputs
- `results/{dataset}_sp{split}_init{seed}.json` — per run: metrics, counts,
  original-unit `test_preds` / `test_targets`, config.
- `results/ensemble_{dataset}.json` — per-dataset ensemble + single-seed stats.
- `results/RESULTS.md` + `results/results_table.json` — final table.
