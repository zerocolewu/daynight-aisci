"""
Configuration for PolyBG-CPA experiment.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer/data/tg_cache/tg_graphs_3ru_42dim.pkl"
TG_CSV = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DIAGNOSTICS_DIR = os.path.join(BASE_DIR, "diagnostics")

# Split
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
TEST_RATIO = 0.20

# Atom encoder (CPA-GAT)
ATOM_HIDDEN_DIM = 256
ATOM_NUM_LAYERS = 4
ATOM_HEADS = 4
ATOM_IN_DIM = 42  # 8 rich + 2 PSPE + 32 RWPE

# Bond encoder (CPA-GAT)
BOND_HIDDEN_DIM = 128
BOND_NUM_LAYERS = 3
BOND_HEADS = 4
BOND_IN_DIM = 14

# Predictor
PREDICTOR_DROPOUT = 0.1
PREDICTOR_HIDDEN = 192

# Training
BATCH_SIZE = 64
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-6
GRAD_CLIP = 1.0
SEEDS = [71, 42, 123]  # primary + 2 extra for variance
