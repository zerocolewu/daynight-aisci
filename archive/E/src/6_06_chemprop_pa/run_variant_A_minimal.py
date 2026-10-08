import sys, os, pickle, numpy as np, torch
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_06_chemprop_pa/')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

from finetune import finetune

with open('results/tg_graphs.pkl', 'rb') as f:
    tg_data = pickle.load(f)

print(f"Loaded {len(tg_data)} Tg graphs for Variant A (LR=1e-3) - MINIMAL CONFIG")

# Minimal configuration to avoid OOM
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
    batch_size=4,  # Very small batch size
    device='cuda:0',
    variant_name='A_baseline_lr1e-3_minimal',
)

with open('results/variant_A_lr1e-3.pkl', 'wb') as f:
    pickle.dump(results_a_3, f)

print(f"\nVariant A (LR=1e-3) Test RMSE: {results_a_3['summary']['mean_test_rmse']:.2f} ± {results_a_3['summary']['std_test_rmse']:.2f}")
