# Archived Research Runs

Directory names match Runs A–E in the manuscript.

| Run | Previous archive name | Backbone | Workflow | Terminal model |
| --- | --- | --- | --- | --- |
| A | ablation2archive | GLM-5 | Full day–night | PTKGE |
| B | ablation6archive | DeepSeek-v4-Pro | Full day–night | PCPA-GAT |
| C | ablation4archive | GLM-5 | Day-only | PolyPerio-DenseGNN |
| D | ablation3archive | GLM-5 | Full day–night | TCP |
| E | ablation5archive | GLM-5.1 | Full day–night | Chemprop |

Each `rmse_by_day.json` retains the original run labels, values, definitions,
and source references. `mem.json`, available daily memory snapshots, and the
small memory embedding files preserve the recorded research state.

Within experiment `src/` directories, the release includes code, logs,
documentation, configurations, and small textual result files. Dataset copies,
model weights, pretraining/feature caches, binary prediction caches, and plots
are omitted. Missing binary data or weights must be obtained or regenerated
before rerunning the corresponding experiments.

## Historical paths and packaging

Original logs and code can contain the previous archive names and original
server paths. Use the table above when adapting a working copy. Runs B and E
are actual directories in this repository, not links to an author's disk.

The three Python file links in
`B/src/5_day5_momucon_finetune/` have been materialized from their same-run
targets in `B/src/5_day5_momucon_ntxent/`. Links to omitted data/cache
directories in Run E are not distributed.

Large text logs are losslessly gzip-compressed, after credential redaction
where needed. `ARCHIVE_INDEX.json` at the repository root records the
published files and their checksums. Historical draft summaries remain named
as such; they are not substituted for final experiment results.
