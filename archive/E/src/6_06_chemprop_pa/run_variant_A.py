import sys, os, pickle, numpy as np, torch
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_06_chemprop_pa/')

from finetune import finetune

# Load Tg data
with open('results/tg_graphs.pkl', 'rb') as f:
    tg_data = pickle.load(f)

print(f'Loaded {len(tg_data)} Tg graphs for Variant A (LR=1e-4)')

# Run Variant A: No pre-training, LR=1e-4 with smaller batch size
results_a_1 = finetune(
    pretrain_checkpoint_path=None,
    tg_data=tg_data,
    use_knode=True,
    hidden_dim=1200,
    depth=7,
    dropout=0.0,
    lr=1e-4,
    seeds=(71, 42, 123),
    split_seed=71,
    epochs=300,
    patience=30,
    batch_size=16,  # Reduced to 16 for stability
    device='cuda',
    variant_name='A_baseline_lr1e-4',
)

# Save results
with open('results/variant_A_lr1e_4_final.pkl', 'wb') as f:
    pickle.dump(results_a_1, f)

print(f'\nVariant A (LR=1e-4) Test RMSE: {results_a_1["summary"]["mean_test_rmse"]:.2f} ± {results_a_1["summary"]["std_test_rmse"]:.2f}')
print('Training completed successfully!')
