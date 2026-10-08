# PTKGE optimization — final results

RMSE in original units.
`ens_mean` / `ens_median` = per-split init-seed ensemble, then mean +/- std over splits (tg/tm) or the single official split (ea/ip).
`single` headline: tg/tm = mean +/- std over splits of the per-split init-mean; ea/ip = mean +/- std over init seeds.
Per-split `init_std` = std over the 3 init seeds within that split.

| dataset | kind | protocol | n_splits | single RMSE | ens mean RMSE | ens median RMSE |
|---|---|---|---|---|---|---|
| tg | single | multi-split x multi-init | 2 | 32.3397 +/- 1.3329 | 31.4289 +/- 1.2584 | 31.5818 +/- 1.2013 |
| tm | single | multi-split x multi-init | 2 | 50.6817 +/- 0.5392 | 49.2558 +/- 0.1856 | 49.6739 +/- 0.2701 |
| ea | copolymer | given-split x multi-init | 1 | 0.0470 +/- 0.0006 | 0.0443 +/- 0.0000 | 0.0448 +/- 0.0000 |
| ip | copolymer | given-split x multi-init | 1 | 0.0367 +/- 0.0003 | 0.0343 +/- 0.0000 | 0.0347 +/- 0.0000 |

## Per-split detail

**tg**
- split 42 (n_init=3): single 33.6726 (init_std 0.9867) | ens_mean 32.6873 | ens_median 32.7831
- split 71 (n_init=3): single 31.0068 (init_std 0.1495) | ens_mean 30.1705 | ens_median 30.3804

**tm**
- split 42 (n_init=3): single 51.2210 (init_std 0.7735) | ens_mean 49.4414 | ens_median 49.9440
- split 71 (n_init=3): single 50.1425 (init_std 0.8984) | ens_mean 49.0702 | ens_median 49.4037

**ea**
- split given (n_init=3): single 0.0470 (init_std 0.0006) | ens_mean 0.0443 | ens_median 0.0448

**ip**
- split given (n_init=3): single 0.0367 (init_std 0.0003) | ens_mean 0.0343 | ens_median 0.0347