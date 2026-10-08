# LEMP Training Experiment Status

## Experiment Configuration
- Dataset: Tg prediction for 7170 polymer repeat units
- Model: LEMP (Long-range Entanglement-aware Message Passing)
- Validation: 5-fold cross-validation
- Max epochs: 200 with early stopping (patience=20)

## Ablation Studies
1. **Full LEMP**: All components enabled (long-range + fingerprint + GIN)
2. **No Long-Range**: Baseline without long-range interactions
3. **No Fingerprint**: Without ECFP4 fingerprint
4. **No GIN**: Without local GIN graph network
5. **Cutoff 5Å**: Long-range cutoff = 5Å
6. **Cutoff 10Å**: Long-range cutoff = 10Å
7. **Cutoff 15Å**: Long-range cutoff = 15Å

## Fixes Applied During Setup
1. Fixed collate function to properly batch fingerprints
2. Changed node dimension from 44 to 25 (matching actual features)
3. Converted 3D positions from numpy to torch tensors
4. Fixed structure factor encoder to output 6*K dimensions (192 features)

## Targets
- RMSE target: <34.0
- Long-range contribution target: >1.5 RMSE improvement

## Current Status
Training started at 10:53. Data loading in progress...
