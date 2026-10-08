"""
Configuration for the CPA-GAT optimization campaign.

Paths auto-detect between the local macOS workstation and the Linux GPU server.
Dataset metadata, the evaluation protocol, and the ORIG_BB model / training
recipe (the project's best single configuration) live here.
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

DATASET_DIR = os.path.join(ROOT, "archive", "ablation6archive", "src", "alldataset")

# --------------------------------------------------------------------------- #
# This folder's own sub-directories
# --------------------------------------------------------------------------- #
CACHE_DIR = os.path.join(_OPT_DIR, "cache")
# RUN_TAG env var routes results to results/<RUN_TAG>/ so multiple independent
# runs (run1, run2, run3) coexist without overwriting each other.
_RUN_TAG = os.environ.get("RUN_TAG", "").strip()
RESULTS_DIR = os.path.join(_OPT_DIR, "results", _RUN_TAG) if _RUN_TAG \
              else os.path.join(_OPT_DIR, "results")
LOG_DIR = os.path.join(_OPT_DIR, "logs")
for _d in (CACHE_DIR, RESULTS_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Evaluation protocol
# --------------------------------------------------------------------------- #
SPLIT_SEEDS = [42, 71]          # tg/tm: 2 splits x 3 init seeds = 6 runs each
INIT_SEEDS = [42, 71, 2026]                            # training randomness (3 init seeds per split)
SPLIT_RATIO = (0.72, 0.08, 0.20)

MULTI_SPLIT_DATASETS = ["tg", "tm"]          # 2 splits x 3 init = 6 runs
GIVEN_SPLIT_DATASETS = ["ea", "ip"]           # official split x 3 init = 3 runs
SPLIT_PKL = {"ea": "ea_split.pkl", "ip": "ip_split.pkl"}

ALL_DATASETS = ["tg", "tm", "ea", "ip"]

# --------------------------------------------------------------------------- #
# Dataset metadata (column names match the ablation6archive corpus).
# --------------------------------------------------------------------------- #
DATASETS = {
    "tg":  dict(csv="Tg.csv",  smi="Smiles",                 y="Tg",    kind="single"),
    "tm":  dict(csv="Tm.csv",  smi="Smiles",                 y="Tm",    kind="single"),
    "ea":  dict(csv="ea.csv",  smi=("smiles_A", "smiles_B"), y="value", kind="copolymer"),
    "ip":  dict(csv="ip.csv",  smi=("smiles_A", "smiles_B"), y="value", kind="copolymer"),
}

# --------------------------------------------------------------------------- #
# ORIG_BB recipe — the project's best single configuration:
#   simple original CPA-GAT (mean pool, simple predictor, no edge / residual)
#   + is_backbone node feature + SWA. Only change vs the original run: lr 1e-3 -> 1e-4.
# --------------------------------------------------------------------------- #
N_MER = 3          # repeat units in the head-to-tail oligomer graph
RWPE_K = 32        # random-walk positional encoding dimension
# Rich atom features (35) = 13 atomic-num + 6 degree + 1 aromatic + 1 in-ring
# + 1 formal charge + 6 H-count + 5 hybridization + 1 mass + 1 chirality.
NODE_DIM = 35 + 2 + RWPE_K + 1   # rich(35) + PSPE(2) + RWPE(32) + is_backbone(1) = 70

# Featurize-cache version tag (baked into the cache filename). Bump whenever the
# featurizer / node schema / multimer / copolymer builder changes, so a stale
# cache is never silently reused.
CACHE_VERSION = f"v2_m{N_MER}_rw{RWPE_K}_d{NODE_DIM}"

ORIG_BB = dict(
    epochs=300, lr=1e-4, weight_decay=1e-4, warmup=5, patience=50, batch_size=64,
    hidden_dim=256, num_layers=4, heads=4, dropout=0.0,
    use_edge=False, use_residual=False, use_norm=True, use_input_bn=False,
    pool="mean", predictor_type="simple", loss="mse",
    use_swa=True, swa_start_frac=0.5,
    use_backbone=True,
)
