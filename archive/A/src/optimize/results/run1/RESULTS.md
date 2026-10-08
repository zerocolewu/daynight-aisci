# PTKGE optimization — final results

RMSE in original units.
`ens_mean` / `ens_median` = per-split init-seed ensemble, then mean +/- std over splits (tg/tm) or the single official split (ea/ip).
`single` headline: tg/tm = mean +/- std over splits of the per-split init-mean; ea/ip = mean +/- std over init seeds.
Per-split `init_std` = std over the 3 init seeds within that split.

| dataset | kind | protocol | n_splits | single RMSE | ens mean RMSE | ens median RMSE |
|---|---|---|---|---|---|---|
| tg | single | multi-split x multi-init | 2 | 32.2456 +/- 1.5030 | 31.4222 +/- 1.4286 | 31.5519 +/- 1.3523 |
| tm | single | multi-split x multi-init | 2 | 51.0602 +/- 0.5482 | 49.6565 +/- 0.2453 | 49.8577 +/- 0.4654 |
| ea | copolymer | given-split x multi-init | 1 | 0.0467 +/- 0.0002 | 0.0442 +/- 0.0000 | 0.0445 +/- 0.0000 |
| ip | copolymer | given-split x multi-init | 1 | 0.0369 +/- 0.0011 | 0.0340 +/- 0.0000 | 0.0343 +/- 0.0000 |

## Per-split detail

**tg**
- split 42 (n_init=3): single 33.7486 (init_std 0.6286) | ens_mean 32.8508 | ens_median 32.9042
- split 71 (n_init=3): single 30.7426 (init_std 0.1932) | ens_mean 29.9935 | ens_median 30.1996

**tm**
- split 42 (n_init=3): single 51.6083 (init_std 0.7307) | ens_mean 49.9018 | ens_median 50.3231
- split 71 (n_init=3): single 50.5120 (init_std 1.1980) | ens_mean 49.4113 | ens_median 49.3922

**ea**
- split given (n_init=3): single 0.0467 (init_std 0.0002) | ens_mean 0.0442 | ens_median 0.0445

**ip**
- split given (n_init=3): single 0.0369 (init_std 0.0011) | ens_mean 0.0340 | ens_median 0.0343