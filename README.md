# Poly-Circa Code and Research Records

This repository contains the day–night research controller and the code,
logs, memory records, and textual results associated with Runs A–E of
Poly-Circa. The standardized post-run benchmark is included alongside the
autonomous research projects.

## Repository contents

| Path | Contents |
| --- | --- |
| `daynight_v3.py` | Current full day–night controller, including its prompts |
| `tool/` | Memory, retrieval, literature-processing, and experiment-execution tools |
| `mem.json` | The supplied root-level memory state; retain an untouched copy before starting a new run |
| `archive/A/` | GLM-5 full-cycle run; terminal model PTKGE |
| `archive/B/` | DeepSeek-v4-Pro full-cycle run; terminal model PCPA-GAT |
| `archive/C/` | GLM-5 day-only run; terminal model PolyPerio-DenseGNN |
| `archive/D/` | GLM-5 full-cycle run; terminal model TCP |
| `archive/E/` | GLM-5.1 full-cycle run; terminal model Chemprop |
| `archive/A/src/optimize/` | Standardized PTKGE evaluation |
| `archive/B/src/optimize/` | Standardized PCPA-GAT evaluation |
| `archive/standard-baseline/` | Standardized ECFP+MLP, GCN, GIN, and GAT baselines |
| `ARCHIVE_INDEX.json` | Published-file paths, sizes, SHA-256 checksums, and packaging transformations |

Each run retains its available memory snapshots, daily RMSE summary, research
summaries, and experiment code. The archived projects include controls,
unsuccessful experiments, diagnostic investigations, and subsequent iterations.
Their individual reports describe the associated protocols. The standardized
post-run evaluation is separate from these discovery-stage experiments.

Manuscript figures and the separately prepared figure source-data tables are
not duplicated here; they are maintained with the journal submission materials.
Downloaded third-party papers, dataset copies, trained weights, pretraining
and feature caches, and other generated binary experiment artifacts are not
included. Small memory embeddings outside experiment `src/` directories are
retained with the corresponding memory records.

## Reading the archive

Text logs larger than 10 MiB are stored as `.log.gz` or `.txt.gz` files.
For example, read a compressed log without changing it:

```sh
gzip -dc archive/A/full.log.gz | less
```

The archive preserves experimental results and historical paths. Earlier
directory names can be resolved using `archive/README.md`. Some scripts still
refer to the original workstation or server paths; adapt those paths in a
separate working copy before execution. Credentials encountered during release
preparation are redacted only in the publication copy.

`mem.json` and `rmse_by_day.json` are experimental records, not executable
instructions. Historical plans and summaries should be interpreted together
with the corresponding experiment code and recorded results.

Historical intermediates are retained as recorded, including partial result
files and incomplete scripts. The release does not repair those
artifacts or replace unsuccessful results with later iterations.

## Running code

Use a separate working directory when starting an agent: the controller writes
logs and updates memory. The root controller is the current full-cycle entry
point, not a frozen entry point for every archived run or for the day-only
condition. Run-specific controller snapshots are included where present in the
archive.

Set your own credentials using `.env.example` as a guide. The controller and
tools use Agno, an OpenAI-compatible model endpoint, `python-dotenv`, and the
configured experiment-execution CLI. The model identifiers and provider URLs
are defined in the source. Model experiments additionally use project-specific
scientific Python dependencies, including PyTorch, PyTorch Geometric, and RDKit.

To retrain models, first obtain the relevant datasets, restore the expected
directory layout and precise splits, and provide or regenerate any required
pretrained weights. See the individual experiment reports and the READMEs in
`archive/A/src/optimize/` and `archive/B/src/optimize/`. This code-and-records
release does not include all training inputs or a single environment lockfile
covering every historical experiment.
