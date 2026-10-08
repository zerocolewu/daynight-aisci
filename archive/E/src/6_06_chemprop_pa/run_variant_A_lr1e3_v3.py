import sys, os, pickle, numpy as np, torch
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_06_chemprop_pa/')

from finetune import finetune

with open('results/tg_graphs.pkl', 'rb') as f:
    tg_data = pickle.load(f)

print(f"Loaded {len(tg_data)} Tg graphs for Variant A (LR=1e-3)")

# First, run with just 1 seed to test
results_a_test = finetune(
    pretrain_checkpoint_path=None,
    tg_data=tg_data,
    use_knode=True,
    hidden_dim=1200,
    depth=7,
    dropout=0.0,
    lr=1e-3,
    seeds=(71,),  # Just 1 seed for testing
    split_seed=71,
    epochs=300,
    patience=30,
    batch_size=16,
    device='cuda:1',
    variant_name='A_baseline_lr1e-3_test',
)

# Save intermediate results
with open('results/variant_A_lr1e-3_seed1.pkl', 'wb') as f:
    pickle.dump(results_a_test, f)

print(f"\nSeed 1 Test RMSE: {results_a_test['seeds'][71]['test_rmse']:.2f} K")

# Now run all 3 seeds
results_a_3 = finetune(
    pretrain_checkpoint_path=None,
    tg_data=tg_data,
    use_knode=True,
    hidden_dim=1200,
    depth=7,
    dropout=0.0,
    lr=1e-3,
    seeds=(71, 42, 123),
    split_seed=71,
    epochs=300,
    patience=30,
    batch_size=16,
    device='cuda:1',
    variant_name='A_baseline_lr1e-3',
)

with open('results/variant_A_lr1e-3.pkl', 'wb') as f:
    pickle.dump(results_a_3, f)

print(f"\nVariant A (LR=1e-3) Test RMSE: {results_a_3['summary']['mean_test_rmse']:.2f} ± {results_a_3['summary']['std_test_rmse']:.2f}")
