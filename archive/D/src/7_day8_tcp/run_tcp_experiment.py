"""
Master script to run complete TCP experiment.

Executes all 6 phases in sequence:
1. Pseudo-label Generation
2. Distribution-aware Sampling
3. TCP Pre-training
4. Fine-tuning on Tg
5. Ablation Studies
6. Analysis
"""
import os
import sys
import time
import torch
from datetime import datetime

print("=" * 100)
print(" " * 30 + "TCP EXPERIMENT MASTER")
print("=" * 100)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check environment
print("\n" + "=" * 100)
print("ENVIRONMENT CHECK")
print("=" * 100)

import sys
python_path = sys.executable
print(f"\nPython: {python_path}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check datasets
import os
dataset_dir = '/home/home_public/aisci/AIscientist/src/alldataset'
pi1m_path = os.path.join(dataset_dir, 'PI1M.csv')
tg_path = os.path.join(dataset_dir, 'Tg.csv')

print(f"\nDatasets:")
print(f"  PI1M: {pi1m_path} - {'✓' if os.path.exists(pi1m_path) else '✗'}")
print(f"  Tg:   {tg_path} - {'✓' if os.path.exists(tg_path) else '✗'}")

if not os.path.exists(pi1m_path) or not os.path.exists(tg_path):
    print("\n✗ ERROR: Datasets not found!")
    sys.exit(1)

print(f"\n{'='*100}\n")

# Create output directory
output_dir = '/home/home_public/aisci/AIscientist/src/7_day8_tcp'
os.makedirs(output_dir, exist_ok=True)

# Track execution times
phase_times = {}
phase_status = {}

def run_phase(phase_num, phase_name, module_name, function_name, args=None, kwargs=None):
    """Run a single phase and track status."""
    print("\n" + "=" * 100)
    print(f"PHASE {phase_num}: {phase_name}")
    print("=" * 100 + "\n")

    start_time = time.time()

    try:
        # Import module
        module = __import__(module_name)

        # Run function
        func = getattr(module, function_name)
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        phase_times[phase_num] = elapsed
        phase_status[phase_num] = 'SUCCESS'

        print(f"\n{'='*100}")
        print(f"PHASE {phase_num} COMPLETE - Elapsed: {elapsed/60:.1f} min")
        print(f"{'='*100}\n")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        phase_times[phase_num] = elapsed
        phase_status[phase_num] = 'FAILED'

        print(f"\n{'='*100}")
        print(f"✗ PHASE {phase_num} FAILED - Elapsed: {elapsed/60:.1f} min")
        print(f"Error: {str(e)}")
        print(f"{'='*100}\n")

        import traceback
        traceback.print_exc()

        return None


# ============================================================================
# PHASE 1: Pseudo-label Generation
# ============================================================================
result = run_phase(
    1, "PSEUDO-LABEL GENERATION",
    'phase1_pseudo_labels', 'generate_pseudo_labels',
    kwargs={
        'csv_path': '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
        'output_path': '/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt'
    }
)

if result is None:
    print("\n✗ Phase 1 failed. Stopping execution.")
    sys.exit(1)


# ============================================================================
# PHASE 2: Distribution-aware Sampling
# ============================================================================
result = run_phase(
    2, "DISTRIBUTION-AWARE SAMPLING",
    'phase2_sampling', 'distribution_aware_sampling'
)

if result is None:
    print("\n✗ Phase 2 failed. Stopping execution.")
    sys.exit(1)


# ============================================================================
# PHASE 3: TCP Pre-training
# ============================================================================
result = run_phase(
    3, "TCP PRE-TRAINING",
    'phase3_pretraining', 'tcp_pretraining'
)

if result is None:
    print("\n✗ Phase 3 failed. Stopping execution.")
    sys.exit(1)


# ============================================================================
# PHASE 4: Fine-tuning on Tg
# ============================================================================
result = run_phase(
    4, "FINE-TUNING ON Tg PREDICTION",
    'phase4_finetuning', 'phase4_finetuning'
)

if result is None:
    print("\n✗ Phase 4 failed. Stopping execution.")
    sys.exit(1)


# ============================================================================
# PHASE 5: Ablation Studies
# ============================================================================
result = run_phase(
    5, "ABLATION STUDIES",
    'phase5_ablation', 'phase5_ablation_studies'
)

if result is None:
    print("\n✗ Phase 5 failed. Continuing to Phase 6...")


# ============================================================================
# PHASE 6: Analysis
# ============================================================================
result = run_phase(
    6, "ANALYSIS AND COMPARISON",
    'phase6_analysis', 'phase6_analysis'
)

if result is None:
    print("\n✗ Phase 6 failed.")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print(" " * 35 + "EXPERIMENT COMPLETE")
print("=" * 100)

total_time = sum(phase_times.values())
print(f"\nTotal execution time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")

print(f"\nPhase Summary:")
print(f"{'Phase':<30} {'Status':<15} {'Time (min)':<15}")
print("-" * 60)

phase_names = {
    1: "Pseudo-label Generation",
    2: "Distribution Sampling",
    3: "TCP Pre-training",
    4: "Fine-tuning",
    5: "Ablation Studies",
    6: "Analysis"
}

for phase_num in sorted(phase_times.keys()):
    name = phase_names[phase_num]
    status = phase_status[phase_num]
    time_min = phase_times[phase_num] / 60
    print(f"{name:<30} {status:<15} {time_min:<15.1f}")

print("\n" + "-" * 60)

# List output files
print(f"\nOutput Files:")
output_files = [
    'PI1M_pseudo_labels.pt',
    'selected_indices.pt',
    'TCP_pretrained_model.pt',
    'Tg_finetuned_results.txt',
    'ablation_results.txt',
    'thermodynamic_consistency.txt',
    'comparison_with_baseline.txt',
]

for filename in output_files:
    filepath = os.path.join(output_dir, filename)
    exists = "✓" if os.path.exists(filepath) else "✗"
    size = ""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        size = f"({size_mb:.1f} MB)"
    print(f"  {exists} {filename:<40} {size}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*100}\n")
