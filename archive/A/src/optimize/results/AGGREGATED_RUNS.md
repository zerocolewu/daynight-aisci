# PTKGE optimization — cross-run aggregated results

Mean +/- std of each run's per-dataset ensemble RMSE over 3 independent run(s) (identical seeds and config; variance is hardware/cuDNN non-determinism).
Runs: run1, run2, run3.
RMSE in original units.

| dataset | kind | n_runs | per-run ens_mean | ens_mean (mean +/- std) | ens_median (mean +/- std) | single (mean +/- std) |
|---|---|---|---|---|---|---|
| tg | single | 3 | 31.4222, 31.4289, 31.3897 | 31.4136 +/- 0.0171 | 31.5683 +/- 0.0124 | 32.2624 +/- 0.0575 |
| tm | single | 3 | 49.6565, 49.2558, 49.9859 | 49.6328 +/- 0.2985 | 49.8573 +/- 0.1496 | 51.1314 +/- 0.3994 |
| ea | copolymer | 3 | 0.0442, 0.0443, 0.0438 | 0.0441 +/- 0.0002 | 0.0446 +/- 0.0002 | 0.0466 +/- 0.0003 |
| ip | copolymer | 3 | 0.0340, 0.0343, 0.0336 | 0.0340 +/- 0.0003 | 0.0343 +/- 0.0003 | 0.0364 +/- 0.0006 |
