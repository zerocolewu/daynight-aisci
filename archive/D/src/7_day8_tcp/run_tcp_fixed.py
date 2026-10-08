"""Run TCP experiment phases 3-6 with fixes."""
import sys
import os
import time

# Add path
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')

print('='*80)
print('TCP EXPERIMENT - PHASES 3-6')
print('='*80)
print()

start_time = time.time()

# Phase 3: TCP Pre-training
print('='*80)
print('PHASE 3: TCP PRE-TRAINING')
print('='*80)
try:
    from phase3_pretraining import tcp_pretraining
    phase3_start = time.time()
    tcp_pretraining()
    phase3_elapsed = (time.time() - phase3_start)/60
    print('\n' + '='*80)
    print(f'PHASE 3 COMPLETE - Elapsed: {phase3_elapsed:.1f} min')
    print('='*80 + '\n')
except Exception as e:
    print('\n' + '='*80)
    print(f'PHASE 3 FAILED: {e}')
    print('='*80 + '\n')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Phase 4: Fine-tuning
print('='*80)
print('PHASE 4: FINE-TUNING ON Tg')
print('='*80)
try:
    from phase4_finetuning import phase4_finetuning
    phase4_start = time.time()
    phase4_finetuning()
    phase4_elapsed = (time.time() - phase4_start)/60
    print('\n' + '='*80)
    print(f'PHASE 4 COMPLETE - Elapsed: {phase4_elapsed:.1f} min')
    print('='*80 + '\n')
except Exception as e:
    print('\n' + '='*80)
    print(f'PHASE 4 FAILED: {e}')
    print('='*80 + '\n')
    import traceback
    traceback.print_exc()

# Phase 5: Ablation
print('='*80)
print('PHASE 5: ABLATION STUDIES')
print('='*80)
try:
    from phase5_ablation import phase5_ablation
    phase5_start = time.time()
    phase5_ablation()
    phase5_elapsed = (time.time() - phase5_start)/60
    print('\n' + '='*80)
    print(f'PHASE 5 COMPLETE - Elapsed: {phase5_elapsed:.1f} min')
    print('='*80 + '\n')
except Exception as e:
    print('\n' + '='*80)
    print(f'PHASE 5 FAILED: {e}')
    print('='*80 + '\n')
    import traceback
    traceback.print_exc()

# Phase 6: Analysis
print('='*80)
print('PHASE 6: ANALYSIS AND COMPARISON')
print('='*80)
try:
    from phase6_analysis import phase6_analysis
    phase6_start = time.time()
    phase6_analysis()
    phase6_elapsed = (time.time() - phase6_start)/60
    print('\n' + '='*80)
    print(f'PHASE 6 COMPLETE - Elapsed: {phase6_elapsed:.1f} min')
    print('='*80 + '\n')
except Exception as e:
    print('\n' + '='*80)
    print(f'PHASE 6 FAILED: {e}')
    print('='*80 + '\n')
    import traceback
    traceback.print_exc()

print('\n' + '='*80)
total_elapsed = (time.time() - start_time)/60
print(f'ALL PHASES COMPLETE - Total time: {total_elapsed:.1f} min')
print('='*80)
