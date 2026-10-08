# LEMP Training - Monitoring Information

**Training Started:** 2026-03-19 11:10 CST
**Status:** Running (10+ hours)
**Process ID:** 3577974

## Progress Tracking

### Completed Configurations:
1. ✅ Full LEMP (5 folds)
2. ✅ No Long-Range/Baseline (5 folds)
3. ✅ No Fingerprint (5 folds)
4. ✅ No GIN (5 folds)
5. 🔄 Cutoff 5Å (in progress - 2/5 folds as of 22:02)

### Remaining:
6. Cutoff 10Å (5 folds)
7. Cutoff 15Å (5 folds)

## Expected Completion
- Estimated: ~23:30-00:30 CST (11:30 PM - 12:30 AM)
- Total folds: 35 (5 folds × 7 configurations)
- Progress: ~27/35 folds (77%)

## Monitoring Scripts Active
- Status log: `/tmp/lemp_final_status.log`
- Multiple background monitors checking every 3-5 minutes
- Results will appear in:
  - `ablation_results.json`
  - `analysis.json`

## Results to Check
When training completes, check:
1. `/home/home_public/aisci/AIscientist/src/4_04_lemp/ablation_results.json`
2. `/home/home_public/aisci/AIscientist/src/4_04_lemp/analysis.json`

## Key Metrics to Evaluate
- Full LEMP RMSE (target: < 34.0)
- Long-range contribution (hypothesis: > 1.5 RMSE improvement)
- Fingerprint contribution (should be additive/positive)
- GIN contribution
