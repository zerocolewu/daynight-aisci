# PTKGE optimization — final results

RMSE in original units.
`ens_mean` / `ens_median` = per-split init-seed ensemble, then mean +/- std over splits (tg/tm) or the single official split (ea/ip).
`single` headline: tg/tm = mean +/- std over splits of the per-split init-mean; ea/ip = mean +/- std over init seeds.
Per-split `init_std` = std over the 3 init seeds within that split.

| dataset | kind | protocol | n_splits | single RMSE | ens mean RMSE | ens median RMSE |
|---|---|---|---|---|---|---|
| tg | single | multi-split x multi-init | 2 | 32.2019 +/- 1.3900 | 31.3897 +/- 1.3211 | 31.5714 +/- 1.2930 |
| tm | single | multi-split x multi-init | 2 | 51.6522 +/- 0.0870 | 49.9859 +/- 0.2223 | 50.0403 +/- 0.1387 |
| ea | copolymer | given-split x multi-init | 1 | 0.0463 +/- 0.0001 | 0.0438 +/- 0.0000 | 0.0444 +/- 0.0000 |
| ip | copolymer | given-split x multi-init | 1 | 0.0355 +/- 0.0005 | 0.0336 +/- 0.0000 | 0.0339 +/- 0.0000 |

## Per-split detail

**tg**
- split 42 (n_init=3): single 33.5920 (init_std 0.0664) | ens_mean 32.7108 | ens_median 32.8644
- split 71 (n_init=3): single 30.8119 (init_std 0.1731) | ens_mean 30.0686 | ens_median 30.2784

**tm**
- split 42 (n_init=3): single 51.7392 (init_std 1.3081) | ens_mean 49.7636 | ens_median 49.9016
- split 71 (n_init=3): single 51.5652 (init_std 0.4349) | ens_mean 50.2082 | ens_median 50.1790

**ea**
- split given (n_init=3): single 0.0463 (init_std 0.0001) | ens_mean 0.0438 | ens_median 0.0444

**ip**
- split given (n_init=3): single 0.0355 (init_std 0.0005) | ens_mean 0.0336 | ens_median 0.0339