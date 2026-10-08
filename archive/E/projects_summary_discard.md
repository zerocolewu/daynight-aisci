# Ablation5 Archive Projects Summary

```yaml
projects:
  - name: "1_01_data_centric_foundation"
    description: "数据中心基础实验 - 评估不同结构表示(mon/dimer/trimer/cyclized/augmented)对Tg预测的影响"
    model_architecture:
      type: "GIN (Graph Isomorphism Network) + EdgeAwareGINConv"
      layers: 3
      hidden_dim: 256
      node_features: 30
      edge_features: 7
      components:
        - "EdgeAwareGINConv: 融合边特征的GIN卷积层"
        - "Node Embedding: 30D→256D"
        - "GIN layers: 可学习epsilon参数"
        - "Mean Pooling: 图级表示"
        - "Prediction Head: 256→256→1 (MLP)"
      total_params: "~445,700"
    training:
      dataset: "Tg数据集 (7,170 p-SMILES)"
      split: "72/8/20"
      batch_size: 32
      epochs: 500
      optimizer: "Adam (lr=0.001)"
      early_stopping: 30
      seed: 71
    results:
      monomer_baseline:
        RMSE: 52.71
        val_RMSE: 41.37
      dimer:
        RMSE: 51.29
        val_RMSE: 39.04
      trimer:
        RMSE: 50.19
        val_RMSE: 39.65
        note: "最佳"
      cyclized_trimer:
        RMSE: 53.28
        val_RMSE: 39.62
        note: "严重过拟合 (train=26.53)"
      augmented_trimer:
        RMSE: 50.52
        val_RMSE: 40.07
      conclusion: "多聚体表示持续改进性能(trimer最佳)，环化导致过拟合，特征增强收益有限"

  - name: "1_2025_07_11_ka_gps"
    description: "KA-GPS - Kolmogorov-Arnold增强GPS图Transformer"
    model_architecture:
      type: "KAN-enhanced GPS (General Powerful Scalable Graph Transformer)"
      hidden_dim: 128
      layers: 4
      attention_heads: 4
      components:
        - "GPS-KAN层: 局部MPNN + 全局Transformer"
        - "Fourier KAN层: 可学习傅里叶系数"
        - "GINEConv: 局部消息传递"
        - "Multi-head Attention: 全局交互"
        - "RWSE: 随机游走结构编码 (6D)"
        - "Multi-pooling: mean+sum+max"
        - "KAN Prediction Head"
      total_params: "~1.3M (KA-GPS), ~1.2M (MLP-GPS)"
    training:
      dataset: "Tg数据集 (p-SMILES + RWSE编码)"
      split: "72/8/20"
      batch_size: 32
      epochs: 500
      optimizer: "Adam (lr=1e-3)"
      scheduler: "ReduceLROnPlateau (factor=0.5)"
      weight_decay: 1e-5
      dropout: 0.1
      early_stopping: 30
      seed: 71
    results:
      MLP_GPS:
        RMSE: 50.8
      KA_GPS:
        RMSE: 51.2
      GAT_baseline:
        RMSE: 50.5
      PerioGT_SOTA:
        RMSE: 30.8
      conclusion: "KAN增强未带来显著改进，复杂架构未能超越简单基线"

  - name: "1_day1_arch_improve"
    description: "Day1架构改进 - GIN-GAT混合与Virtual Node架构评估"
    model_architecture:
      type: "GIN-GAT Hybrid / GIN+Virtual Node"
      hidden_dim: 256
      layers: 3
      components:
        - "EdgeAwareGINConv: 基础GIN"
        - "EdgeAwareGINAttentionConv: GIN-GAT混合 (注意力加权聚合)"
        - "Virtual Node: 256D虚拟节点连接所有节点"
        - "Attention: W_attn * [h_v || h_u || W_edge * e_uv]"
        - "Multi-pooling: 虚拟节点排除池化"
      total_params: "~478K-481K"
    training:
      dataset: "Trimer/Cyclized trimer图"
      split: "72/8/20"
      batch_size: 32
      epochs: 500
      optimizer: "Adam (lr=0.001)"
      early_stopping: 30
      seed: 71
    results:
      gin_gat_trimer:
        RMSE: 51.66
        overfitting_gap: 23.67
      gin_gat_cyclized:
        RMSE: 54.01
        overfitting_gap: 18.15
      gin_vn_trimer:
        RMSE: 53.78
        overfitting_gap: 22.88
      gin_vn_cyclized:
        RMSE: 51.14
        overfitting_gap: 19.18
        note: "最佳"
      conclusion: "GIN-GAT注意力选择性低(接近均匀聚合)，Virtual Node有助于降低过拟合但未显著提升性能"

  - name: "2_polymer_jepa_pretrain"
    description: "JEPA预训练 - 原始版本(含关键bug)"
    model_architecture:
      type: "JEPA (Joint Embedding Predictive Architecture)"
      encoder: "5-layer GIN, hidden_dim=256"
      dropout: 0.5
      pooling: "global_add_pool (后发现有误)"
      predictor: "2-layer MLP (context+pos → hidden → target)"
    training:
      dataset: "PI1M (预训练) + Tg (微调)"
      split: "72/8/20"
      batch_size: 256 (预训练), 64 (微调)
    results:
      baseline_no_pretrain:
        RMSE: 54.89
      jepa_pretrained:
        RMSE: 54.79
      conclusion: "存在关键bug(池化函数不匹配/dropout过高)，预训练效果不明显"

  - name: "2_2_polymer_jepa_pretrain"
    description: "JEPA预训练v2 - 修复bug + 对比学习变体"
    model_architecture:
      type: "JEPA + Contrastive Learning"
      encoder: "5-layer GIN, hidden_dim=256"
      dropout: 0.1
      pooling: "global_mean_pool (修复)"
      predictor: "2-layer MLP"
      projection_head: "VICReg投影头 (variance+covariance)"
      positional_encoding: "3D repeat unit位置编码"
    training:
      dataset: "PI1M (300K trimers预训练) + Tg数据集"
      split: "72/8/20"
      batch_size: 256 (预训练), 64 (微调)
      optimizer: "Adam (lr=1e-3预训练, 5e-4微调)"
      epochs: 30 (预训练), 500 (微调)
      early_stopping: 30
      VICReg:
        lambda_var: 50
        lambda_cov: 50
    results:
      baseline_no_pretrain:
        RMSE: 54.89
      jepa_pretrained_fixed:
        RMSE: 54.79
      contrastive_pretrained:
        RMSE: 50.77
        note: "最佳 (Strategy B, 端到端微调)"
      monomer_trimer_similarity:
        JEPA: 0.995
        Contrastive: 0.88
      conclusion: "修复关键bug后对比预训练有效(RMSE-4.12)，但未达目标<50.19；对比学习优于JEPA"

  - name: "2_gps_spse_trimer"
    description: "GPS-SPSE - 混合MPNN+Transformer + 最短路径结构编码"
    model_architecture:
      type: "GPS (General Powerful Scalable) + SPSE"
      blocks: 5
      hidden_dim: 256
      attention_heads: 4
      components:
        - "GINEConv: 局部MPNN (edge features)"
        - "Transformer: 全局注意力 + SPSE偏置"
        - "SPSE: 最短路径正弦编码 (dim=32)"
        - "Progressive Dropout: [0.10, 0.15, 0.20, 0.25, 0.30]"
        - "Auxiliary Denoising: 10%节点特征掩码 (coeff=0.1)"
      total_params: "892K (MPNN 75.3%, Transformer 18.9%)"
    training:
      dataset: "Tg数据集 (trimer图 + SPSE距离预计算)"
      split: "72/8/20"
      batch_size: 64
      optimizer: "Adam (lr=5e-4, weight_decay=1e-5)"
      scheduler: "CosineAnnealingLR"
      gradient_clip: 1.0
      early_stopping: 30
      multi_seed: [71, 42, 123]
    results:
      best_seed_71:
        RMSE: 53.22
        train_RMSE: 38.78
      multi_seed_mean:
        RMSE: 54.41
        RMSE_std: 0.84
      overfitting_gap: -14.44
      conclusion: "SPSE结构编码有一定潜力，但模型可能存在欠拟合(负gap)"

  - name: "3_periogt_lite"
    description: "PerioGT-Lite - 轻量版PerioGT，精确匹配Day1基线架构"
    model_architecture:
      type: "GIN (精确复制Day1基线)"
      layers: 3
      hidden_dim: 256
      dropout: "无"
      pooling: "global_mean_pool"
      supervised_head: "256→128→1"
      projector: "256→128→64 + L2归一化"
    training:
      phases:
        - "1. 预处理PI1M和Tg数据"
        - "2. 对比预训练"
        - "3. 微调 (基线 vs 预训练)"
        - "4. 诊断分析"
      batch_size: 32
      optimizer: "Adam (lr=1e-3)"
    results:
      conclusion: "精确复现Day1架构，用于公平评估对比预训练效果"

  - name: "3_day3_pcr_v2"
    description: "PCR-v2 - 混合池化GIN (监督用sum_pool/对比用mean_pool)"
    model_architecture:
      type: "GIN + Hybrid Pooling"
      encoder: "5-layer EdgeAwareGINConv, hidden_dim=256"
      dropout: 0.5
      BatchNorm: "MLP内部 + 卷积层后"
      hybrid_pooling:
        supervised_head: "global_sum_pool → 回归"
        contrastive_projector: "global_mean_pool → L2归一化嵌入"
      projector: "256→128→64"
    training:
      dataset: "Tg数据集"
      experiments: ["baseline", "alpha=0.01", "alpha=0.05", "alpha=0.1"]
      optimizer: "Adam (lr=1e-3)"
      batch_size: 64
      early_stopping: 50
      seed: 71
    results:
      baseline_sum_pool:
        RMSE: 77.50
        note: "严重退化"
      conclusion: "混合池化方案未能修复性能退化问题，global_mean_pool持续降低性能"

  - name: "4_04_pgft"
    description: "PCFT - Prompt条件化微调，预训练表示作为条件输入"
    model_architecture:
      type: "Dual-Encoder GIN"
      components:
        - "Encoder P (frozen): GIN on monomer → h_pre (256D)"
        - "Encoder T (trainable): GIN on trimer → h_task (256D)"
        - "Prediction Head: MLP(h_task || h_pre) → Tg (512→256→128→1)"
      gin_layers: 3
      hidden_dim: 256
      dropout: 0.5
    training:
      dataset: "Tg数据集 (7170), 72/8/20"
      optimizer: "Adam (lr=1e-3, weight_decay=1e-5)"
      batch_size: 32
      epochs: 500
      early_stopping: 30
      multi_seed: [42, 71, 123]
      pretrain_checkpoint: "Day3 PA预训练权重"
    results:
      baseline:
        RMSE: 52.46
        RMSE_std: 0.20
      random_init_pcft:
        RMSE: 53.34
        RMSE_std: 0.36
      pa_pretrained_pcft:
        RMSE: 52.72
        RMSE_std: 0.76
      standard_ft:
        RMSE: 53.53
        RMSE_std: 0.37
      both_frozen:
        RMSE: 54.02
        RMSE_std: 1.43
      conclusion: "PCFT未能超越基线(diff=+0.26)，Prompt条件化策略未改善表示利用"

  - name: "4_4_blochspec"
    description: "BlochSpec v1 - Bloch光谱特征增强GIN (使用global_add_pool)"
    model_architecture:
      type: "GIN + Bloch光谱特征"
      layers: 3
      hidden_dim: 256
      pooling: "global_add_pool"
      spectral_features: 12
      variants:
        - "Baseline GIN (trimer, add_pool)"
        - "BlochSpec (GIN + 12光谱特征)"
        - "GIN + Random Features"
        - "GIN + Unweighted Laplacian"
    training:
      dataset: "Tg数据集 (trimer图)"
      optimizer: "Adam + CosineAnnealingLR"
      batch_size: 32
      multi_seed: [42, 71, 123]
    results:
      baseline_gin_trimer:
        RMSE: 66.15
        RMSE_std: 0.27
      blochspec:
        RMSE: 61.52
        RMSE_std: 5.99
        improvement: "-7.0%"
      pure_spectral_mlp:
        RMSE: 76.71
        RMSE_std: 1.26
      gin_random_features:
        RMSE: 58.68
        RMSE_std: 1.41
      gin_unweighted_lap:
        RMSE: 58.45
        RMSE_std: 1.09
        note: "最佳"
      conclusion: "初步结果看似正面，但随机特征同样有效，暗示池化方法存在问题"

  - name: "4_4_4_blochspec"
    description: "BlochSpec v2 - 修复池化为global_mean_pool的消融实验"
    model_architecture:
      type: "GIN + 光谱特征 (修复版)"
      layers: 3
      hidden_dim: 256
      pooling: "global_mean_pool (关键修复)"
      spectral_features: 12
      variants:
        - "Baseline GIN (mean_pool)"
        - "BlochSpec (GIN + 光谱)"
        - "WiderMLP (512→256, 参数控制)"
        - "ReducedGNN (hidden=244, 维度控制)"
        - "Pure Spectral MLP"
    training:
      dataset: "Tg数据集"
      optimizer: "Adam (lr=1e-3) + CosineAnnealingLR"
      batch_size: 32
      epochs: 500
      early_stopping: 30
      multi_seed: [42, 71, 123]
    results:
      baseline_mean_pool:
        RMSE: 51.71
        RMSE_std: 0.33
      blochspec:
        RMSE: 53.25
        RMSE_std: 0.51
        note: "+1.54 (+3.0%)"
      wider_mlp:
        RMSE: 51.37
        RMSE_std: 1.76
        note: "最佳"
      reduced_gnn:
        RMSE: 53.49
        RMSE_std: 0.10
      pure_spectral_mlp:
        RMSE: 76.71
        RMSE_std: 1.26
      conclusion: "修复池化后Bloch特征不再有益，与随机特征效果类似，光谱特征对Tg预测无实质帮助"

  - name: "5_6_chemprop_polymer"
    description: "Chemprop-Polymer - 有向键级消息传递神经网络"
    model_architecture:
      type: "Directed Bond MPNN (Chemprop-style)"
      layers: 3
      hidden_dim: 300
      key_features:
        - "每条无向边→2条有向边，独立消息"
        - "更新步骤包含反向边消息"
        - "周期边作为常规有向边处理"
        - "Prediction: 2-layer MLP (300→300→1)"
    training:
      dataset: "Tg数据集 (不同图构建方式)"
      optimizer: "Adam + gradient clipping"
      batch_size: 32
      epochs: 200
      early_stopping: 30
      ablations:
        - "A: 单体图"
        - "B: 三聚体图"
        - "C: 完整聚合物+周期边"
        - "D: 最小周期特征"
        - "E: 随机周期边"
        - "F: hidden_dim=256"
    results:
      A_monomer:
        RMSE: 45.76
        RMSE_std: 0.28
      B_trimer:
        RMSE: 43.92
        RMSE_std: 0.45
      C_polymer_full:
        RMSE: 43.46
        RMSE_std: 0.89
        note: "最佳"
      D_polymer_minimal:
        RMSE: 43.71
        RMSE_std: 0.69
      E_polymer_random:
        RMSE: 43.58
        RMSE_std: 1.20
      F_polymer_256:
        RMSE: 44.48
        RMSE_std: 1.23
      conclusion: "完整聚合物图优于三聚体(RMSE -1.46)，Chemprop有向消息传递架构效果显著"

  - name: "5_day5_polymer_bipartite_v2"
    description: "PolymerBipartite v2 - 边界/内部原子分离处理架构"
    model_architecture:
      type: "Bipartite GIN (Boundary/Interior分离)"
      gin_layers: 5
      hidden_dim: 256
      components:
        - "RU (Repeat Unit) 聚合 + 可学习门控标量"
        - "Gate: 边界=0.28, 内部=0.06 (从-0.92/-2.72学习)"
        - "LayerNorm: RU间耦合后"
        - "Discriminability Loss (margin=0.85)"
    training:
      dataset: "Tg数据集 (含RU分配)"
      optimizer: "Adam + gradient clipping"
      batch_size: 32
      epochs: 200
      early_stopping: 30
      multi_seed: [42, 71, 123]
    results:
      polymer_bipartite_v2:
        RMSE: 53.81
        RMSE_std: 0.94
      gin_trimer_baseline:
        RMSE: 54.00
        RMSE_std: 1.96
      random_ru_control:
        RMSE: 54.88
      param_matched_baseline:
        RMSE: 53.66
        note: "最佳"
      conclusion: "边界/内部分离策略带来1.07 RMSE改进，但不如参数匹配基线"

  - name: "6_06_chemprop_hpo"
    description: "Chemprop超参优化 - 系统OFAT搜索"
    model_architecture:
      type: "Chemprop-style Directed Bond MPNN"
      layers: 7
      hidden_dim: 1200
      pooling: "global mean pooling"
      prediction: "2-layer MLP"
      dense_skip: "密集跳跃连接 (最佳配置)"
    training:
      dataset: "Tg数据集 (聚合物图)"
      optimizer: "Adam"
      batch_size: 64
      HPO_phases:
        - "Phase 1: OFAT (depth 3-7, hidden 300-1200, LR 1e-4~1e-3, dropout 0-0.3)"
        - "Phase 2: 组合最佳配置"
        - "Phase 3: 密集跳跃连接探索"
        - "Phase 4: 多种子验证"
      multi_seed: [42, 71, 123]
    results:
      phase1_best:
        RMSE: 35.72
        config: "depth=7, hidden=1200, LR=0.001, dropout=0.1"
      phase3_dense_best:
        RMSE: 35.48
        config: "dense_d5_h1200"
      phase4_multi_seed:
        RMSE: 40.00
        RMSE_std: 2.09
      improvement_vs_day5: "-7.98 RMSE"
      conclusion: "系统性HPO带来巨大提升，深度和宽度是关键因素，密集跳跃连接进一步改善"

  - name: "6_06_chemprop_pa"
    description: "Chemprop-Polymer-K - K-node全局上下文 + 对比预训练"
    model_architecture:
      type: "Chemprop-Polymer-K + Contrastive Pretraining"
      layers: 7
      hidden_dim: 1200
      components:
        - "Directed Bond MPNN (7层)"
        - "K-node: 虚拟原子连接所有真实原子 (528D特征)"
        - "Projector Head: 1200→128→64 (仅预训练)"
        - "Contrastive Loss: temperature τ=0.07"
      augmentations: "Trimer-shift (15% bond drop, 25% feature mask)"
    training:
      pretrain_dataset: "PI1M (300K样本)"
      finetune_dataset: "Tg (6,960图)"
      optimizer: "Adam (lr=1e-3 or 1e-4)"
      batch_size: 32 (预训练), 64 (微调)
      variants:
        - "A: 无预训练 (基线)"
        - "B: PI1M + K-node (主要)"
        - "C: 随机图 + K-node (控制)"
        - "D: PI1M无K-node"
    results:
      variant_A_no_pretrain:
        RMSE: "~40-45"
      pretrain_best_loss: "~2.5-3.5 (30 epochs)"
      conclusion: "对比预训练+K-node架构完整，预训练收敛良好；最终微调结果待完整日志确认"

  - name: "baseline"
    description: "GAT基线模型 - 三种聚合物性质预测"
    model_architecture:
      type: "GAT (Graph Attention Network)"
      layers: 3
      hidden_channels: 64
      attention_heads: 4
      dropout: 0.2
      components:
        - "GATConv layers (含边特征)"
        - "Mean Pooling"
        - "2-layer MLP prediction head"
        - "Layer Normalization"
    training:
      datasets:
        - "Egc (2,433样本)"
        - "Tg (4,971样本)"
        - "Tm (2,615样本)"
      split: "72/8/20"
      optimizer: "Adam + ReduceLROnPlateau"
      batch_size: 16
      epochs: 500
      early_stopping: 50
      seed: 42
    results:
      Egc:
        RMSE: 0.6326
        MAE: 0.4615
        R2: 0.8361
        note: "最佳"
      Tg:
        RMSE: 50.52
        MAE: 37.44
        R2: 0.7912
      Tm:
        RMSE: 79.30
        MAE: 60.42
        R2: 0.5026
        note: "最难预测"
      conclusion: "电子性质(Egc)最可预测，热性质(Tm)最具挑战；含硅/离子聚合物预测困难"

  - name: "alldataset"
    description: "数据集仓库 (无模型代码)"
    contents:
      - "Egc.csv: 电化学能隙数据"
      - "Tg.csv: 玻璃化转变温度数据"
      - "Tm.csv: 熔融温度数据"
      - "PI1M.csv: 大规模聚合物预训练数据"
      - "cyclize_p_smiles.py: p-SMILES环化处理"
      - "multimer.py: 多聚体生成"

summary:
  total_projects: 16
  code_projects: 15
  data_only_projects: 1
  best_rmse: 35.48
  best_model: "6_06_chemprop_hpo (dense_d5_h1200)"
  best_multi_seed_rmse: 40.00
  best_multi_seed_model: "6_06_chemprop_hpo (3-seed mean)"
  key_findings:
    - "Chemprop有向键消息传递架构大幅领先其他方法"
    - "系统性超参优化(HPO)比架构创新带来更大收益"
    - "全局mean_pool vs add_pool对结果影响巨大，多个项目因此需要修正"
    - "多聚体表示(trimer)优于单体，但环化导致过拟合"
    - "对比预训练(Contrastive)优于JEPA，但提升有限"
    - "Bloch光谱特征、KAN增强、Prompt条件化等高级技巧均未带来显著改进"
    - "模型深度(7层)和宽度(1200)是Chemprop性能的关键"
    - "简单架构+好的超参 > 复杂架构+默认超参"
    - "池化策略选择是影响结果的最关键实现细节之一"
```
