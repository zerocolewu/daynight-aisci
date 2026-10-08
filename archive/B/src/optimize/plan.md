# CPA-GAT optimization plan (35-dim atom features)

## Goal
Optimize the CPA-GAT polymer property model (Cardinality-Preserving Attention GAT)
across four benchmark properties (Tg, Tm, EA, IP). The model core and the
project's best configuration (ORIG_BB) are frozen; the chemistry part of the
node features is replaced with a richer 35-dim atom encoding, and the whole
matrix is run three independent times to quantify run-to-run variance. The
learning rate is tuned to 1e-4.

## What is frozen
- **CPA core** (`CPAGATv2Conv`): `o = o_std + g * scatter_sum(v)` where
  `o_std` is standard GATv2 attention, `s = scatter_sum(v over neighbors+self)`,
  and `g = sigmoid(W_g h + b_g)`. This cardinality-preserving sum channel is the
  mechanism that makes the model size/extent-aware and is never altered.
- **ORIG_BB configuration** (the project's empirically best single model):
  the *simple* original CPA-GAT — mean pooling, simple predictor, **no** edge
  features, **no** residuals (LayerNorm on) — plus the `is_backbone` node feature
  and SWA. The earlier "v2" bells-and-whistles (edge/residual/multi-readout) were
  ablated away: they did not beat ORIG_BB.

## Node representation
A head-to-tail 3-repeat-unit oligomer graph per polymer. Node features (70-dim):

| block | dims | notes |
|---|---|---|
| atom chemistry | 35 | 13 atomic-num one-hot + 6 degree + 1 aromatic + 1 in-ring + 1 formal charge + 6 H-count + 5 hybridization + 1 mass + 1 chirality |
| backbone positional encoding (dce, mi) | 2 | double-BFS longest-path endpoints |
| random-walk positional encoding | 32 | RWPE |
| `is_backbone` flag | 1 | double-BFS longest path; single most effective feature addition |

The 35-dim atom block is the change introduced in this folder (the original
recipe used an 8-dim chemistry block). The total node dimension is therefore
35 + 2 + 32 + 1 = 70.

## Copolymer graphs (EA/IP)
The two monomers are joined into a single connected A-B molecule (explicit bond
between A's exit and B's entry), with the composition vector `u = [rA, rB,
type-one-hot]` consumed by the head — identical handling to the single-polymer
graphs, so the CPA mechanism and message passing operate across the A-B interface.

## The changes vs the original ORIG_BB run
- Chemistry node block: 8-dim → 35-dim (total node dim 43 → 70).
- Learning rate `1e-3 -> 1e-4`.
Everything else (AdamW, weight decay 1e-4, warmup-then-cosine, SWA from 50% of
epochs, MSE, early stop patience 50, 300 epochs, mean pool, simple predictor) is
preserved.

## Evaluation protocol (decoupled split / init)
- **Tg, Tm**: 2 partitions from `split_seed in {42, 71}` (deterministic
  72/8/20), each with 3 training seeds `init_seed in {42, 71, 2026}` -> 6 runs.
- **EA, IP**: the official `ea_split.pkl` / `ip_split.pkl` partition, 3 training
  seeds -> 3 runs.
- Targets normalized per split (train mean/std); model predicts normalized,
  metrics in original units. Best-by-val-RMSE checkpoint evaluated once on test;
  the SWA model is reported as an extra reference.
- **Per-run ensembling**: within one independent run, average the 3 init-seed
  predictions per split (mean/median, original units). For Tg/Tm aggregate the
  2 per-split ensembles as mean ± std; for EA/IP report the single ensemble.
  Single-seed mean ± std reported alongside. No best-seed/best-split selection.

## Three independent runs (variance measurement)
The entire matrix above is executed three times (`run1`, `run2`, `run3`) with
the identical recipe and identical seeds — the only source of variation across
runs is GPU non-determinism (scatter/atomic-add ops). For each dataset the
final headline RMSE is mean ± std over the three runs' `ens_mean` RMSE.

## Run matrix (per independent run)
| dataset | splits | init seeds | runs | u_dim |
|---|---|---|---|---|
| Tg  | 2 | 3 | 6 | 0 |
| Tm  | 2 | 3 | 6 | 0 |
| EA  | 1 (official) | 3 | 3 | 6 |
| IP  | 1 (official) | 3 | 3 | 6 |

## Outputs
- `results/run{1,2,3}/{dataset}_sp{split}_init{seed}.json` — per run: metrics
  (incl. SWA), counts, original-unit `test_preds` / `test_targets`, config,
  param count.
- `results/run{1,2,3}/ensemble_{dataset}.json` — per-dataset ensemble +
  single-seed stats, for that independent run.
- `results/run{1,2,3}/RESULTS.md` (+ `results_table.json`) — that run's table.
- `results/AGGREGATE.md` (+ `results/aggregate_runs.json`) — final mean ± std
  table across the three independent runs.
