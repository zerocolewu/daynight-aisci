"""
Configuration for the PTKGE optimization campaign.

Paths auto-detect between the local macOS workstation and the Linux GPU server,
so the same folder runs unchanged in both places. Dataset metadata, the
evaluation protocol, and the ptkge_scaled model / training recipe live here.
"""
import os

# --------------------------------------------------------------------------- #
# Project root: env override (AISCIENTIST_ROOT) > walk up from this folder to
# the ancestor whose direct child is an 'archive' directory. No hard-coded host
# paths, so the same folder runs unchanged on the workstation and the GPU server.
# --------------------------------------------------------------------------- #
_OPT_DIR = os.path.dirname(os.path.abspath(__file__))            # .../ablationXarchive/src/optimize


def _detect_root():
    cur = os.environ.get("AISCIENTIST_ROOT", "")
    if cur:
        return cur
    cur = _OPT_DIR
    for _ in range(12):                                   # plenty of levels up
        if os.path.isdir(os.path.join(cur, "archive")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:                                 # reached filesystem root
            break
        cur = parent
    return cur


ROOT = _detect_root()

DATASET_DIR = os.path.join(ROOT, "archive", "A", "src", "alldataset")

# --------------------------------------------------------------------------- #
# This folder's own sub-directories
# --------------------------------------------------------------------------- #
CACHE_DIR = os.path.join(_OPT_DIR, "cache")
RESULTS_DIR = os.path.join(_OPT_DIR, "results")
CHECKPOINT_DIR = os.path.join(_OPT_DIR, "checkpoints")
LOG_DIR = os.path.join(_OPT_DIR, "logs")
for _d in (CACHE_DIR, RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Independent-run isolation. Set PTKGE_RUN_ID (e.g. "run1", "run2", "run3") to
# scope per-run outputs to a subfolder of RESULTS_DIR. Cache and checkpoints
# are shared across runs (they depend only on the dataset, not on training
# randomness), so only the result JSONs / ensembles / per-run summaries land
# inside RUN_DIR. Empty/unset -> write directly to RESULTS_DIR (legacy mode).
# --------------------------------------------------------------------------- #
RUN_ID = os.environ.get("PTKGE_RUN_ID", "").strip()
RUN_DIR = os.path.join(RESULTS_DIR, RUN_ID) if RUN_ID else RESULTS_DIR
os.makedirs(RUN_DIR, exist_ok=True)

# Pretrained PTKGE encoder (InfoNCE contrastive on PI1M, physics embeddings).
# This checkpoint is the pretrained result reused verbatim from the base run.
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "pretrained_physics.pt")

# --------------------------------------------------------------------------- #
# Evaluation protocol
# --------------------------------------------------------------------------- #
SPLIT_SEEDS = [42, 71]                                 # tg/tm: 2 splits x 3 init seeds = 6 runs each.
INIT_SEEDS = [42, 71, 2026]                           # training randomness (3 init seeds per split)
SPLIT_RATIO = (0.72, 0.08, 0.20)      # train / val / test

# Datasets without an official split -> 2 splits x 3 init seeds = 6 runs each.
MULTI_SPLIT_DATASETS = ["tg", "tm"]
# Datasets that ship an official split (ea_split.pkl / ip_split.pkl)
# -> 1 split x 3 init seeds = 3 runs each.
GIVEN_SPLIT_DATASETS = ["ea", "ip"]
SPLIT_PKL = {"ea": "ea_split.pkl", "ip": "ip_split.pkl"}

ALL_DATASETS = ["tg", "tm", "ea", "ip"]

# Featurize-cache version tag (baked into the cache filename). Bump this whenever
# the featurizer / node schema / copolymer builder changes, so a stale cache is
# never silently reused.
CACHE_VERSION = "v1"

# --------------------------------------------------------------------------- #
# Dataset metadata: csv name, smiles column(s), target column, graph kind
# --------------------------------------------------------------------------- #
DATASETS = {
    "tg":  dict(csv="Tg.csv",  smi="Smiles",                       y="Tg",    kind="single"),
    "tm":  dict(csv="Tm.csv",  smi="Smiles",                       y="Tm",    kind="single"),
    "ea":  dict(csv="ea.csv",  smi=("smiles_A", "smiles_B"),       y="value", kind="copolymer"),
    "ip":  dict(csv="ip.csv",  smi=("smiles_A", "smiles_B"),       y="value", kind="copolymer"),
}

# --------------------------------------------------------------------------- #
# ptkge_scaled recipe (model + finetune). Pretrain config is in pretrain.py.
# --------------------------------------------------------------------------- #
MODEL_DEFAULTS = dict(
    num_layers=5, hidden_dim=256, embedding_strategy="physics",
    embedding_dim=8, fp_dim=2048, dropout=0.1,
)

# Single-polymer fine-tuning: differential LR, early stop on val RMSE (no scheduler).
FINETUNE_SINGLE = dict(epochs=300, patience=30, batch_size=32,
                       encoder_lr=5e-5, head_lr=1e-4, weight_decay=1e-5)

# Copolymer (EA/IP) trained from scratch on the connected A-B graph.
FINETUNE_COPOLYMER = dict(epochs=300, patience=30, batch_size=64,
                          lr=1e-3, weight_decay=1e-5)
