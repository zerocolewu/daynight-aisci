#!/bin/bash
# Run v5b ablation studies with seed=42

set -e

echo "====================================="
echo "v5b Ablation Study: No Fourier-KAN"
echo "====================================="
python -c "
import sys
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_polykanssm')
exec(open('train_v5b.py').read().replace('from model_v5b import', 'from model_v5b_no_kan import').replace('PolyKANSSM_v5b', 'PolyKANSSM_v5b_no_kan').replace('\"num_seeds\": 5', '\"num_seeds\": 1').replace('\"seeds\": [42, 123, 456, 789, 2024]', '\"seeds\": [42]').replace('results_v5b.json', 'results_v5b_ablation_no_kan.json').replace('checkpoints_v5b', 'checkpoints_v5b_ablation_no_kan').replace('diagnostics_v5b', 'diagnostics_v5b_ablation_no_kan'))
"

echo ""
echo "====================================="
echo "v5b Ablation Study: No JK"
echo "====================================="
python -c "
import sys
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_polykanssm')
exec(open('train_v5b.py').read().replace('from model_v5b import', 'from model_v5b_no_jk import').replace('PolyKANSSM_v5b', 'PolyKANSSM_v5b_no_jk').replace('\"num_seeds\": 5', '\"num_seeds\": 1').replace('\"seeds\": [42, 123, 456, 789, 2024]', '\"seeds\": [42]').replace('results_v5b.json', 'results_v5b_ablation_no_jk.json').replace('checkpoints_v5b', 'checkpoints_v5b_ablation_no_jk').replace('diagnostics_v5b', 'diagnostics_v5b_ablation_no_jk'))
"

echo ""
echo "====================================="
echo "v5b Ablation Study: With Virtual Node"
echo "====================================="
python -c "
import sys
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_polykanssm')
exec(open('train_v5b.py').read().replace('from model_v5b import', 'from model_v5b_with_vn import').replace('PolyKANSSM_v5b', 'PolyKANSSM_v5b_with_vn').replace('\"num_seeds\": 5', '\"num_seeds\": 1').replace('\"seeds\": [42, 123, 456, 789, 2024]', '\"seeds\": [42]').replace('results_v5b.json', 'results_v5b_ablation_with_vn.json').replace('checkpoints_v5b', 'checkpoints_v5b_ablation_with_vn').replace('diagnostics_v5b', 'diagnostics_v5b_ablation_with_vn'))
"

echo ""
echo "====================================="
echo "All Ablation Studies Completed!"
echo "====================================="
