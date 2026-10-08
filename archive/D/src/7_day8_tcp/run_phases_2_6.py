"""
Simplified runner for TCP experiment phases 2-6.
Phase 1 (pseudo-label generation) is already complete.
"""
import os
import sys
import time
import torch
from datetime import datetime

print("=" * 100)
print(" " * 20 + "TCP EXPERIMENT - PHASES 2-6 (Phase 1 already complete)")
print("=" * 100)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check Phase 1 output
phase1_output = '/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt'
if not os.path.exists(phase1_output):
    print(f"\n✗ ERROR: Phase 1 output not found: {phase1_output}")
    sys.exit(1)

print(f"\n✓ Phase 1 output found: {os.path.getsize(phase1_output) / (1024*1024):.1f} MB")

# Track execution
phase_times = {}
phase_status = {}

def run_phase(phase_num, phase_name, module_name, function_name):
    """Run a single phase."""
    print("\n" + "=" * 100)
    print(f"PHASE {phase_num}: {phase_name}")
    print("=" * 100 + "\n")

    start_time = time.time()

    try:
        module = __import__(module_name)
        func = getattr(module, function_name)
        result = func()

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
# PHASE 2: Distribution-aware Sampling
# ============================================================================
print("\n" + "=" * 100)
print("NOTE: Phase 2 uncertainty estimation may take 30-60 minutes")
print("=" * 100)

result = run_phase(
    2, "QUICK DISTRIBUTION-AWARE SAMPLING",
    'phase2_sampling_simplified', 'quick_sampling'
)

if result is None:
    print("\n✗ Phase 2 failed. Trying to continue...")
else:
    # Save selected indices
    torch.save(result, '/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt')
    print(f"✓ Saved selected indices")


# ============================================================================
# PHASE 3: TCP Pre-training
# ============================================================================
result = run_phase(
    3, "TCP PRE-TRAINING",
    'phase3_pretraining', 'tcp_pretraining'
)

if result is None:
    print("\n✗ Phase 3 failed. Trying to continue...")


# ============================================================================
# PHASE 4: Fine-tuning on Tg
# ============================================================================
result = run_phase(
    4, "FINE-TUNING ON Tg PREDICTION",
    'phase4_finetuning', 'phase4_finetuning'
)

if result is None:
    print("\n✗ Phase 4 failed. Trying to continue...")


# ============================================================================
# PHASE 5: Ablation Studies
# ============================================================================
result = run_phase(
    5, "ABLATION STUDIES",
    'phase5_ablation', 'phase5_ablation_studies'
)

if result is None:
    print("\n✗ Phase 5 failed. Trying to continue...")


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

# Check output files
print(f"\nOutput Files:")
output_dir = '/home/home_public/aisci/AIscientist/src/7_day8_tcp'
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
