# CPA-GAT (ORIG_BB) optimization — final results

RMSE in original units.
`ens_mean` / `ens_median` = per-split init-seed ensemble, then mean +/- std over splits (tg/tm) or the single official split (ea/ip).
`single` headline: tg/tm = mean +/- std over splits of the per-split init-mean; ea/ip = mean +/- std over init seeds.
Per-split `init_std` = std over the 3 init seeds within that split.

| dataset | kind | protocol | n_splits | single RMSE | ens mean RMSE | ens median RMSE |
|---|---|---|---|---|---|---|
| tg | single | multi-split x multi-init | 2 | 32.8163 +/- 0.2020 | 31.7381 +/- 0.1810 | 31.9737 +/- 0.3841 |
| tm | single | multi-split x multi-init | 2 | 51.0142 +/- 0.4925 | 49.6283 +/- 0.6655 | 49.6973 +/- 0.6610 |
| ea | copolymer | given-split x multi-init | 1 | 0.0408 +/- 0.0003 | 0.0399 +/- 0.0000 | 0.0400 +/- 0.0000 |
| ip | copolymer | given-split x multi-init | 1 | 0.0303 +/- 0.0000 | 0.0294 +/- 0.0000 | 0.0295 +/- 0.0000 |

## Per-split detail

**tg**
- split 42 (n_init=3): single 33.0183 (init_std 0.8638) | ens_mean 31.9191 | ens_median 32.3579 swa_mean 33.0817
- split 71 (n_init=3): single 32.6143 (init_std 0.3670) | ens_mean 31.5570 | ens_median 31.5896 swa_mean 32.1113

**tm**
- split 42 (n_init=3): single 50.5217 (init_std 0.5720) | ens_mean 48.9628 | ens_median 49.0363 swa_mean 50.0668
- split 71 (n_init=3): single 51.5067 (init_std 0.9721) | ens_mean 50.2937 | ens_median 50.3583 swa_mean 50.9535

**ea**
- split given (n_init=3): single 0.0408 (init_std 0.0003) | ens_mean 0.0399 | ens_median 0.0400 swa_mean 0.0407

**ip**
- split given (n_init=3): single 0.0303 (init_std 0.0000) | ens_mean 0.0294 | ens_median 0.0295 swa_mean 0.0302