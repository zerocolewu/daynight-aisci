#!/usr/bin/env python3
"""Master orchestration script for SelfGNN Polymer experiment.

Runs the full pipeline:
  0. Data preparation (PI1M + Tg)
  1. Pretraining (Split+Standardize, 200 epochs)
  2. Finetuning (3 seeds: 71, 42, 123)
  3. Ablations:
     A) No-pretrain baseline
     B) Split-only pretraining → finetune
     C) Standardize-only pretraining → finetune
     H) No-BN pretraining (50 epochs)
     G) Frozen encoder
  4. Report compilation
"""

import os
import sys
import json
import subprocess
import time

BASE_DIR = '/home/home_public/aisci/AIscientist/src/4_4_day4_selfgnn_polymer'
PI1M_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
TG_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

# Check environment
print("=" * 70)
print("SELF-GNN POLYMER EXPERIMENT")
print("=" * 70)
print(f"Working dir: {BASE_DIR}")
print(f"PI1M dataset: {PI1M_PATH}")
print(f"Tg dataset: {TG_PATH}")

# Verify imports
try:
    from data_utils import PI1MDataset, TgDataset, psmiles_to_graph
    from models import StudentModel, TeacherModel, FinetuneModel
    print("All imports OK")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)


def run_cmd(cmd, desc, timeout=None):
    """Run a command and print its output in real-time."""
    print(f"\n{'─'*60}")
    print(f"▶ {desc}")
    print(f"  CMD: {cmd}")
    print(f"{'─'*60}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=BASE_DIR,
    )
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (code={proc.returncode})"
    print(f"\n  Status: {status} | Elapsed: {elapsed/60:.1f} min")
    return proc.returncode == 0


def run_py(script_args, desc, timeout=None):
    """Run a Python script."""
    cmd = f"python {script_args}"
    return run_cmd(cmd, desc, timeout)


# ─── Step 0: Verify data preparation ────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 0: DATA PREPARATION")
print("=" * 70)

# Check if PI1M dataset already processed
pi1m_processed = os.path.exists('./data_pretrain/processed/pi1m_200000.pt')
tg_processed = os.path.exists('./data_tg/processed/tg_3ru.pt')

if not pi1m_processed:
    print("Processing PI1M dataset (200K samples)...")
    os.makedirs('./data_pretrain', exist_ok=True)
    run_py(f'-c "from data_utils import PI1MDataset; '
           f'PI1MDataset(csv_path=\\'{PI1M_PATH}\\', n_samples=200000, root=\\'./data_pretrain\\')"',
           "PI1M data preparation", timeout=3600)
else:
    print("PI1M dataset already processed ✓")

if not tg_processed:
    print("Processing Tg dataset (3-RU)...")
    os.makedirs('./data_tg', exist_ok=True)
    run_py(f'-c "from data_utils import TgDataset; '
           f'TgDataset(csv_path=\\'{TG_PATH}\\', n_ru=3, add_rwpe=True, rwpe_dim=32, root=\\'./data_tg\\')"',
           "Tg data preparation", timeout=3600)
else:
    print("Tg dataset already processed ✓")


# ─── Step 1: Pretraining (Full Split+Standardize) ───────────────────────────
print("\n" + "=" * 70)
print("STEP 1: PRETRAINING (Full Split+Standardize)")
print("=" * 70)

if not os.path.exists('./pretrain_output/pretrain_best.pt'):
    run_py(
        'pretrain.py --augmentation combined --epochs 200 --batch_size 64 '
        '--pi1m_path /home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        "Pretraining: Split+Standardize, 200 epochs",
        timeout=86400,
    )
else:
    print("Pretraining already complete ✓")

# Step 1b: Split-only pretraining (ablation B)
print("\n>>> Ablation B: Split-only pretraining")
if not os.path.exists('./pretrain_output_split/pretrain_best.pt'):
    run_py(
        'pretrain.py --augmentation split --epochs 200 --batch_size 64 '
        '--save_dir ./pretrain_output_split '
        '--data_root ./data_pretrain_split '
        '--pi1m_path /home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        "Pretraining: Split-only, 200 epochs",
        timeout=86400,
    )
else:
    print("Split-only pretraining already complete ✓")

# Step 1c: Standardize-only pretraining (ablation C)
print("\n>>> Ablation C: Standardize-only pretraining")
if not os.path.exists('./pretrain_output_std/pretrain_best.pt'):
    run_py(
        'pretrain.py --augmentation standardize --epochs 200 --batch_size 64 '
        '--save_dir ./pretrain_output_std '
        '--data_root ./data_pretrain_std '
        '--pi1m_path /home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        "Pretraining: Standardize-only, 200 epochs",
        timeout=86400,
    )
else:
    print("Standardize-only pretraining already complete ✓")

# Step 1h: No-BN pretraining (ablation H)
print("\n>>> Ablation H: No-BN pretraining")
if not os.path.exists('./pretrain_output_nobn/pretrain_best.pt'):
    run_py(
        'pretrain.py --augmentation combined --epochs 50 --batch_size 64 '
        '--save_dir ./pretrain_output_nobn --no_batchnorm '
        '--data_root ./data_pretrain_nobn '
        '--pi1m_path /home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        "Pretraining: No-BN, 50 epochs",
        timeout=86400,
    )
else:
    print("No-BN pretraining already complete ✓")


# ─── Step 2: Finetuning (PRIMARY: Full Split+Standardize pretraining) ────────
print("\n" + "=" * 70)
print("STEP 2: FINETUNING (Primary: Full Split+Standardize)")
print("=" * 70)

run_py(
    'finetune.py '
    '--pretrain_path ./pretrain_output/pretrain_best.pt '
    '--save_dir ./finetune_output '
    '--data_root ./data_tg_ft '
    '--seeds 71 42 123 '
    '--epochs 500 --batch_size 64 ',
    "Finetuning: Full pretrain, 3 seeds",
    timeout=86400,
)


# ─── Step 3: Ablations ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: ABLATIONS")
print("=" * 70)

# A) No-pretrain baseline
print("\n>>> Ablation A: No-pretrain baseline")
run_py(
    'finetune.py '
    '--no_pretrain '
    '--save_dir ./finetune_output_nopretrain '
    '--data_root ./data_tg_nopretrain '
    '--seeds 71 42 123 '
    '--epochs 500 --batch_size 64 ',
    "Finetuning: No pretrain, 3 seeds",
    timeout=86400,
)

# B) Split-only pretraining → finetune
print("\n>>> Ablation B: Split-only pretraining → finetune")
run_py(
    'finetune.py '
    '--pretrain_path ./pretrain_output_split/pretrain_best.pt '
    '--save_dir ./finetune_output_split '
    '--data_root ./data_tg_split '
    '--seeds 71 42 123 '
    '--epochs 500 --batch_size 64 ',
    "Finetuning: Split-only pretrain, 3 seeds",
    timeout=86400,
)

# C) Standardize-only pretraining → finetune
print("\n>>> Ablation C: Standardize-only pretraining → finetune")
run_py(
    'finetune.py '
    '--pretrain_path ./pretrain_output_std/pretrain_best.pt '
    '--save_dir ./finetune_output_std '
    '--data_root ./data_tg_std '
    '--seeds 71 42 123 '
    '--epochs 500 --batch_size 64 ',
    "Finetuning: Standardize-only pretrain, 3 seeds",
    timeout=86400,
)

# G) Frozen encoder
print("\n>>> Ablation G: Frozen encoder")
run_py(
    'finetune.py '
    '--pretrain_path ./pretrain_output/pretrain_best.pt '
    '--save_dir ./finetune_output_frozen '
    '--data_root ./data_tg_frozen '
    '--freeze_encoder '
    '--seeds 71 42 123 '
    '--epochs 500 --batch_size 64 ',
    "Finetuning: Frozen encoder, 3 seeds",
    timeout=86400,
)


# ─── Step 4: Compile Report ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: COMPILING REPORT")
print("=" * 70)

def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

report = {}

# Collect all results
experiments = {
    'D_Full_Split+Standardize': './finetune_output/finetune_results.json',
    'A_NoPretrain': './finetune_output_nopretrain/finetune_results.json',
    'B_SplitOnly': './finetune_output_split/finetune_results.json',
    'C_StandardizeOnly': './finetune_output_std/finetune_results.json',
    'G_FrozenEncoder': './finetune_output_frozen/finetune_results.json',
}

for name, path in experiments.items():
    res = load_results(path)
    if res:
        seeds = res.keys()
        rmses = [float(res[s]['test_rmse']) for s in seeds]
        report[name] = {
            'seeds': list(seeds),
            'test_rmse': rmses,
            'mean': np.mean(rmses),
            'std': np.std(rmses),
        }
        print(f"  {name}: {np.mean(rmses):.2f} ± {np.std(rmses):.2f} K")

# Print summary table
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")

if report.get('A_NoPretrain'):
    baseline_mean = report['A_NoPretrain']['mean']
    print(f"\n{'Experiment':<35} {'RMSE (K)':<20} {'Δ':<10}")
    print(f"{'-'*65}")
    for name, r in report.items():
        delta = r['mean'] - baseline_mean
        sign = "+" if delta > 0 else ""
        print(f"  {name:<33} {r['mean']:.2f} ± {r['std']:.2f}    {sign}{delta:.2f}")
else:
    for name, r in report.items():
        print(f"  {name}: {r['mean']:.2f} ± {r['std']:.2f} K")

# Print pretraining diagnostic summary
print(f"\n{'='*70}")
print("PRETRAINING DIAGNOSTICS")
print(f"{'='*70}")
for pretrain_name, hist_path in [
    ('Full', './pretrain_output/pretrain_history.json'),
    ('Split-only', './pretrain_output_split/pretrain_history.json'),
    ('Std-only', './pretrain_output_std/pretrain_history.json'),
    ('No-BN', './pretrain_output_nobn/pretrain_history.json'),
]:
    h = load_results(hist_path)
    if h and 'diag' in h and h['diag']:
        first = h['diag'][0]
        last = h['diag'][-1]
        collapse = "YES" if last.get('collapse_warning') else "NO"
        print(f"\n  {pretrain_name}:")
        print(f"    First diag (ep {first['epoch']}): cos_mean={first['cos_mean']:.4f}, "
              f"cos_std={first['cos_std']:.4f}, collapse={first.get('collapse_warning','?')}")
        print(f"    Last diag (ep {last['epoch']}): cos_mean={last['cos_mean']:.4f}, "
              f"cos_std={last['cos_std']:.4f}, collapse={collapse}")
        # Loss curve
        if 'loss' in h:
            print(f"    Final loss: {h['loss'][-1]:.4f} (best: {min(h['loss']):.4f})")

print(f"\n{'='*70}")
print("EXPERIMENT COMPLETE")
print(f"{'='*70}")

# Save report
with open(os.path.join(BASE_DIR, 'experiment_report.json'), 'w') as f:
    # Convert numpy types
    clean_report = {}
    for k, v in report.items():
        clean_report[k] = {
            'seeds': v['seeds'],
            'test_rmse': [float(x) for x in v['test_rmse']],
            'mean': float(v['mean']),
            'std': float(v['std']),
        }
    json.dump(clean_report, f, indent=2)
print(f"\nReport saved to experiment_report.json")
