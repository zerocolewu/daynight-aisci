# Ablation6 Archive Projects Summary

```yaml
projects:
  - name: "baseline"
    description: "GAT基线模型 - 三属性预测(Egc/Tg/Tm)"
    model_architecture:
      type: "GAT (Graph Attention Network)"
      layers: 3
      hidden_channels: 64
      attention_heads: 4
      activation: "ELU (GAT层), ReLU (FC层)"
      dropout: 0.2
      prediction_head: "64→32→1 (LayerNorm + Dropout)"
      readout: "Global mean pooling"
    training:
      split: "72/8/20"
      batch_size: 16
      epochs: 500
      optimizer: "Adam (lr=1e-3, weight_decay=1e-5)"
      scheduler: "ReduceLROnPlateau (factor=0.5, patience=10)"
      early_stopping: 50
      seed: 42
    results:
      Egc:
        R2: 0.8361
        RMSE: 0.6326
        MAE: 0.4615
        samples: "2433 train / 676 test"
      Tg:
        R2: 0.7912
        RMSE: 50.52
        MAE: 37.44
        samples: "4971 train / 1382 test"
      Tm:
        R2: 0.5026
        RMSE: 79.30
        MAE: 60.42
        samples: "2615 train / 727 test"
      conclusion: "Egc预测最好，Tm最难；基线GAT表现有限"

  - name: "alldataset"
    description: "数据工具集 - pSMILES环化与多聚体生成"
    utilities:
      - "cyclize_p_smiles.py: pSMILES(*)→环状分子"
      - "multimer.py: 单体SMILES→n聚体"
    datasets:
      Egc: "3380样本"
      Tg: "7170样本"
      Tm: "3651样本"
      PI1M: "988795样本 (预训练)"

  - name: "1_day1_followup_grin_elene"
    description: "GRIN + ELENE实验 - 验证稀疏注意力和ELENE边编码"
    model_architecture:
      type: "GAT + 多种Readout/ELENE组合"
      base: "3层GATConv, 128 hidden, 4 heads, ELU, Dropout=0.1"
      conditions:
        A: "GAT + mean pooling (基线)"
        B: "GAT + max pooling"
        C: "GAT + max pooling + L1稀疏 (λ=1e-3, warmup=50)"
        D: "GAT + mean pooling + ELENE边编码 (度三元组)"
        E: "GAT + max pooling + L1 + ELENE"
      node_features: "47维 (rich)"
      edge_features: "10维 + 可选ELENE(3维)"
    training:
      dataset: "Tg (7170样本)"
      split: "72/8/20"
      batch_size: 32
      epochs: 300
      optimizer: "Adam (lr=1e-3)"
      early_stopping: 30
      seeds: "71, 72, 73"
      augmentation: "{1,3}-RU mixed"
    results:
      GAT_mean:
        RMSE: 44.57
        RMSE_std: 1.73
        MAE: 30.88
        R2: 0.8392
      GAT_max:
        RMSE: 46.74
        RMSE_std: 2.46
      GAT_max_L1:
        RMSE: 46.83
        RMSE_std: 1.82
        sparsity: "0.72%"
      GAT_mean_ELENE:
        RMSE: 45.70
        RMSE_std: 0.78
      GAT_max_L1_ELENE:
        RMSE: 46.97
        RMSE_std: 0.56
      conclusion: "所有假设均未得到支持；mean readout优于max；ELENE无提升；L1稀疏度极低(~0.7-2%)"

  - name: "1_day1_multimer_feature_gat"
    description: "多聚体×特征等级消融实验"
    model_architecture:
      type: "GAT (3层, 128 hidden, 4 heads, ELU)"
      readout: "Global mean pooling"
      feature_levels:
        Basic: "100维 (原子序号one-hot)"
        Standard: "43维 (原子类型/手性/度/电荷/芳香性/杂化/价电子)"
        Rich: "47维 (Standard + 环数) + 10维边特征"
      multimer_variants: ["1-RU", "2-RU", "3-RU", "{1,3}-mixed"]
    training:
      dataset: "Tg (7170样本)"
      split: "72/8/20"
      batch_size: 64
      epochs: 300
      optimizer: "Adam (lr=1e-3)"
      early_stopping: 30
      seeds: "71, 72, 73"
    results:
      best_config: "3-RU × Rich"
      best_RMSE: 41.61
      best_RMSE_std: 0.90
      best_R2: 0.8573
      best_MAE: 28.75
      basic_avg: "RMSE ≈ 69.0K"
      standard_avg: "RMSE ≈ 45.2K"
      rich_avg: "RMSE ≈ 42.8K"
      conclusion: "特征丰富度影响最大(Basic→Rich降低~27K)；3-RU最优但混合策略反而下降"

  - name: "2_day2_spse_maxgat"
    description: "SPSE边编码 + Max聚合消融"
    model_architecture:
      type: "GAT + SPSE/MaxGAT变体"
      base: "4层GATConv, 256 hidden, 4 heads, ELU, Dropout=0.1"
      node_features: "47维 (rich)"
      edge_features: "10维 + 可选SPSE(4维)"
      conditions:
        baseline: "GAT + rich + 3-RU + mean聚合"
        treatment_A: "+ SPSE边编码 (路径计数k=1,2,3,4)"
        treatment_B: "+ SPSE + Max聚合 (MaxGATConv)"
        treatment_Bv2: "+ SPSE + Max + 残差门控"
    training:
      dataset: "Tg (7170样本, 3-RU)"
      split: "72/8/20 (seed=71)"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=5e-4, weight_decay=1e-4)"
      scheduler: "ReduceLROnPlateau"
      early_stopping: 30
    results:
      baseline:
        RMSE: 36.85
        MAE: 24.66
        R2: 0.8881
      treatment_A_SPSE:
        RMSE: 37.99
        MAE: 25.00
        R2: 0.8811
      treatment_B_SPSE_Max:
        RMSE: 38.27
        MAE: 26.29
        R2: 0.8793
      treatment_Bv2_ResidualMax:
        RMSE: 38.96
        MAE: 26.80
        R2: 0.8750
      conclusion: "SPSE路径计数退化为常数(k=1全为1,k=2全为0)；Max聚合导致方差坍缩(仅保留0.4%方差)；基线最优"

  - name: "2_day_2_lspe_gat"
    description: "LSPE-GAT - 解耦结构/位置编码更新"
    model_architecture:
      type: "GAT + 多种位置编码策略"
      base: "4层GATConv, 256 hidden, 4 heads"
      conditions:
        static_rwpe: "静态RWPE拼接 (k=16/24/32)"
        lspe_gat: "解耦结构/位置更新 (GATv2Conv + tanh(MLP))"
        lspe_gat_edge: "LSPE + 边门控 (σ(Linear([h_i;h_j;p_i;p_j;e_ij])))"
        skip_rwpe: "逐层重新注入RWPE"
      readout: "LSPE: concat[mean_pool(h); mean_pool(p)]"
    training:
      dataset: "Tg (7170样本, 3-RU)"
      split: "72/8/20 (seed=71)"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=1e-3, weight_decay=1e-4)"
      scheduler: "CosineAnnealingLR"
      early_stopping: 50
    results:
      static_rwpe_k16:
        RMSE: 36.29
        MAE: 24.01
        R2: 0.8914
      static_rwpe_k24:
        RMSE: 37.13
        R2: 0.8873
      static_rwpe_k32:
        RMSE: 37.65
        R2: 0.8831
      lspe_gat:
        RMSE: 37.96
        R2: 0.8812
      lspe_gat_edge:
        RMSE: 37.85
        R2: 0.8819
      skip_rwpe_k16:
        RMSE: 37.34
        R2: 0.8850
      conclusion: "静态RWPE k=16最优(36.29K)；LSPE解耦更新无提升；边门控增加复杂度无收益"

  - name: "3_3_day3_pspe"
    description: "PSPE消融 - GATConv版本 (ELU, patience=50)"
    model_architecture:
      type: "GATPredictor (4层GATConv, concat模式)"
      hidden_dim: 256
      attention_heads: 4
      activation: "ELU"
      dropout: 0.1
      predictor: "256→128→1 (ReLU)"
      input_dim: "79(baseline) / 80(+PSPE单项) / 81(+PSPE组合)"
      features: "47维rich + 32维RWPE + 可选PSPE(DCE/MI)"
    training:
      dataset: "Tg (5162 train / 573 val / 1435 test)"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=1e-3, weight_decay=1e-4)"
      scheduler: "CosineAnnealingLR"
      early_stopping: 50
    results:
      baseline:
        RMSE: 36.98
        MAE: 24.51
        R2: 0.8873
      pspe_dce:
        RMSE: 36.19
        MAE: 22.99
        R2: 0.8921
        delta: "-0.79K"
      pspe_mi:
        RMSE: 35.99
        MAE: 23.91
        R2: 0.8933
        delta: "-0.99K"
      pspe_combined:
        RMSE: 35.44
        MAE: 22.92
        R2: 0.8965
        delta: "-1.54K"
      conclusion: "PSPE组合特征最优(35.44K)；DCE和MI均有效，组合效果最好"

  - name: "3_day3_graphmae_pretrain"
    description: "GraphMAE预训练 - 掩码节点特征重建"
    model_architecture:
      type: "GraphMAE → GAT微调"
      encoder: "4层GAT + 残差连接 + BatchNorm, 256 hidden, 4 heads"
      decoder: "MLP (256→128→13) 重建rich特征"
      predictor: "GAT encoder + 256→128→1"
      masking_ratio: 0.25
      input_features: "45维 (13 rich + 32 RWPE)"
    training:
      pretrain:
        dataset: "PI1M (50K图)"
        epochs: 50
        batch_size: 64
        lr: 1e-3
      finetune:
        dataset: "Tg (3-RU multimer)"
        epochs: 300
        batch_size: 32
        lr: 3e-4
        optimizer: "AdamW"
        early_stopping: 30
    results:
      pretrained:
        RMSE: 39.69
        MAE: 25.82
        R2: 0.8703
      baseline:
        RMSE: 40.27
        MAE: 27.83
        R2: 0.8665
      improvement: "+0.58K RMSE (+1.44%)"
      conclusion: "GraphMAE预训练有微小提升但基线偏离预期(~36K→40K)，效果有限"

  - name: "3_day3_pspe"
    description: "PSPE消融 - GATv2Conv版本 (ReLU, patience=30)"
    model_architecture:
      type: "GATv2Predictor (4层GATv2Conv, concat模式)"
      hidden_dim: 256
      attention_heads: 4
      activation: "ReLU"
      dropout: 0.1
      predictor: "256→128→64→1 (ReLU, 3层MLP)"
      input_dim: "79(baseline) / 80(+PSPE单项) / 81(+PSPE组合)"
      features: "47维rich + 32维RWPE + 可选PSPE(DCE/MI)"
    training:
      dataset: "Tg (5162 train / 573 val / 1435 test)"
      batch_size: 64
      epochs: 200
      optimizer: "AdamW (lr=1e-3, weight_decay=1e-5)"
      scheduler: "ReduceLROnPlateau"
      early_stopping: 30
    results:
      baseline:
        RMSE: 38.68
        MAE: 25.23
        R2: 0.8767
      pspe_dce:
        RMSE: 36.43
        MAE: 23.40
        R2: 0.8906
        delta: "-2.25K"
      pspe_mi:
        RMSE: 36.65
        MAE: 24.13
        R2: 0.8894
        delta: "-2.03K"
      pspe_combined:
        RMSE: 36.66
        MAE: 23.63
        R2: 0.8893
        delta: "-2.02K"
      conclusion: "DCE单独最优(36.43K)；GATv2基线更高(38.68 vs 36.98)，但PSPE提升更大"

  - name: "4_4_day4_selfgnn_polymer"
    description: "SelfGNN-Polymer - BYOL风格自监督预训练"
    model_architecture:
      type: "SelfGNN (BYOL momentum learning)"
      encoder: "4层GATConv, 256 hidden, 4 heads, BatchNorm, ELU"
      readout: "Set2Set (256→512)"
      predictor_head: "512→256→128→1"
      input_features: "42维 (8 rich + 2 PSPE + 32 RWPE)"
      edge_features: "6维"
      augmentations: "Split + Standardize"
    training:
      pretrain:
        dataset: "PI1M (200K样本)"
        epochs: 200
        batch_size: 64
        optimizer: "AdamW + CosineAnnealing"
      finetune:
        dataset: "Tg"
        epochs: 300
        early_stopping: 30
      seeds: "71, 42, 123"
    results:
      full_pretrain:
        RMSE: 39.99
        RMSE_std: 1.13
        MAE: 26.32
        R2: 0.877
        best_epoch: 53
      no_pretrain:
        RMSE: 40.87
        RMSE_std: 2.48
      split_only:
        RMSE: 41.24
        RMSE_std: 1.00
      frozen_encoder:
        RMSE: 40.40
        RMSE_std: 0.98
      conclusion: "BYOL预训练带来0.88K提升；Set2Set readout有效但整体RMSE偏高"

  - name: "4_d5mc"
    description: "MoMuCon模板 - NT-Xent对比预训练 (50K, 无结果)"
    model_architecture:
      type: "GATv2 + NT-Xent对比学习"
      encoder: "4层GATv2, LayerNorm + 残差连接"
      readout: "Global mean pool (256×4=1024)"
      projection: "Linear(256→128) + BN + ReLU + Linear(128→128), L2 norm"
      input_features: "41维 (9 rich + 32 RWPE)"
      edge_features: "3维"
    training:
      pretrain:
        dataset: "PI1M (50K对)"
        epochs: 100
        batch_size: 128
        loss: "NT-Xent (τ=0.1)"
        positive_pairs: "(1-RU, 3-RU)"
    results:
      status: "模板项目，results目录为空，无实际训练输出"

  - name: "4_day4_selfgnn_polymer"
    description: "SelfGNN变体 - GAT + BatchNorm (无量化结果)"
    model_architecture:
      type: "GAT + BatchNorm"
      encoder: "4层GAT, BatchNorm"
      readout: "Global mean pooling"
      input_features: "10维 (8 rich + 2 PSPE)"
      augmentations: "Split + Standardize"
    training:
      pretrain:
        dataset: "PI1M (200K)"
        epochs: 200
        batch_size: 64
        early_stopping: 30
    results:
      status: "日志存在但无test metrics；代码结构与4_4版本相同"

  - name: "4_day5_momucon"
    description: "MoMuCon - 单体/多聚体对比预训练 (100K, 无结果)"
    model_architecture:
      type: "GATv2 + NT-Xent"
      encoder: "4层GATv2, LayerNorm + 残差"
      readout: "Global mean pool → 256维"
      input_features: "41维 (9 rich + 32 RWPE)"
    training:
      pretrain:
        dataset: "PI1M (100K对)"
        epochs: 100
        batch_size: 64
        loss: "NT-Xent (τ=0.1)"
        positive_pairs: "(1-RU, 3-RU)"
    results:
      status: "模板项目，results目录为空"

  - name: "4_day5_momucon_ntxent"
    description: "MoMuCon-NT-Xent变体 (与4_day5_momucon相同架构, 无结果)"
    model_architecture:
      type: "GATv2 + NT-Xent (共享权重孪生)"
      encoder: "4层GATv2, LayerNorm + 残差"
      input_features: "42维 (8 rich + 2 PSPE + 32 RWPE)"
    training:
      pretrain:
        dataset: "PI1M (50K对)"
        epochs: 100
        loss: "NT-Xent (τ=0.1)"
    results:
      status: "模板项目，无训练输出"

  - name: "5_5_day5_momucon_ntxent"
    description: "MoMuCon-NT-Xent - 50K预训练 + Tg微调"
    model_architecture:
      type: "GAT + NT-Xent对比预训练"
      encoder: "4层GAT, 256 hidden, 4 heads"
      projection_heads: "单体/多聚体分别投影"
      input_features: "42维 (8 rich + 2 PSPE + 32 RWPE)"
    training:
      pretrain:
        dataset: "PI1M (50K聚合物)"
        epochs: 300
        batch_size: 64
        optimizer: "AdamW (lr=1e-4)"
      finetune:
        dataset: "Tg"
        early_stopping: 30
    results:
      pretrained_rmse: 38.75
      baseline_rmse: 35.44
      delta: "+3.31K (恶化)"
      conclusion: "对比预训练反而恶化性能，缺乏适当归纳偏置"

  - name: "5_day5_gps_lite"
    description: "GPS-Lite - GAT + 全局注意力层"
    model_architecture:
      type: "GPS-Lite (GAT + 全局注意力)"
      baseline: "4层GAT + RWPE + PSPE (无全局注意力)"
      variants:
        no_bias: "GAT + 全局注意力 (无SPD偏置)"
        spd: "GAT + 全局注意力 + SPD偏置"
      node_features: "47维 (rich + RWPE k=32 + PSPE)"
      edge_features: "10维"
      params: "~3.3M"
    training:
      dataset: "Tg (3-RU, 72/8/20)"
      batch_size: 64
    results:
      baseline:
        RMSE: 35.19
        MAE: 23.17
        R2: 0.898
      no_bias:
        RMSE: 43.43
        delta: "+8.24K (恶化)"
      spd:
        RMSE: 43.14
        delta: "+7.95K (恶化)"
      conclusion: "全局注意力严重恶化性能(Δ>+8K)；SPD偏置未解决问题"

  - name: "5_day5_momucon_finetune"
    description: "MoMuCon微调 - 加载5_day5_momucon_ntxent预训练权重"
    model_architecture:
      type: "GAT encoder (来自5_day5_momucon_ntxent) + fresh predictor"
      input_features: "42维"
    training:
      checkpoint: "checkpoints/best_pretrain.pt"
      early_stopping: 30
      best_epoch: 144
    results:
      RMSE: 38.75
      baseline_rmse: 35.44
      delta: "+3.31K (恶化)"
      conclusion: "与5_5版本一致，预训练对比学习对Tg预测无益"

  - name: "5_day5_momucon_ntxent"
    description: "MoMuCon-NT-Xent - 200K预训练 (与5_5架构相同，更大预训练集)"
    model_architecture:
      type: "GAT + NT-Xent (同5_5_day5_momucon_ntxent)"
      encoder: "4层GAT, 256 hidden, 4 heads"
      input_features: "42维"
    training:
      pretrain:
        dataset: "PI1M (200K聚合物)"
    results:
      status: "架构与5_5版本相同，预训练规模更大但无独立结果文件"

  - name: "6_day6_cpa_gat"
    description: "CPA-GAT - 聚合物感知通道注意力"
    model_architecture:
      type: "CPA-GAT (Channel-wise Polymer Attention)"
      baseline: "BaselineGAT: 4层GATv2Conv, 256H, 4 heads, ELU"
      cpa: "CPAGAT: 每层CPA通道 (sigmoid门控 + 非归一化求和)"
      cpa_last: "CPALastLayerGAT: 仅最后一层CPA"
      degree_baseline: "DegreeBaselineGAT: + 度特征"
      input_features: "42维 (8 rich + 2 PSPE + 32 RWPE)"
      predictor: "256→128→1 (ReLU, Dropout=0.1)"
      readout: "Global mean pooling"
    training:
      dataset: "Tg (3-RU, 72/8/20)"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=1e-3, weight_decay=1e-4)"
      scheduler: "CosineAnnealingLR (T_max=300)"
      early_stopping: 30
      seeds: "42, 123, 456"
    results:
      baseline_gat:
        RMSE: 36.38
        RMSE_std: 0.14
      cpa_gat:
        RMSE: 34.77
        RMSE_std: 0.21
        improvement: "-1.61K"
      cpa_last_layer:
        RMSE: 36.47
        RMSE_std: 0.45
      degree_baseline:
        RMSE: 37.48
        RMSE_std: 0.78
      conclusion: "CPA机制提供1.61K提升；仅最后一层CPA无效；度特征反而有害"

  - name: "6_day7_cpa_pse"
    description: "CPA-GAT + 周期空间编码(PSE)消融"
    model_architecture:
      type: "CPA-GAT + PSE变体"
      configs:
        A: "CPA-GAT baseline (42维, 无PSE)"
        B: "CPA-GAT + full PSE (45维)"
        C: "CPA + modular_position (43维)"
        D: "CPA + periodic_distance (43维)"
        E: "CPA + is_backbone (43维)"
        F: "去PSPE + PSE (43维)"
        H: "标准GAT + full PSE (45维, 无CPA)"
      pse_features:
        modular_position: "[0,1] RU内位置"
        periodic_distance: "[0,1] 到邻接RU等价原子最短路径"
        is_backbone: "{0,1} 骨架原子标记"
    training:
      dataset: "Tg (3-RU, 72/8/20)"
      batch_size: 64
      optimizer: "AdamW (lr=1e-3)"
      scheduler: "CosineAnnealingLR"
      early_stopping: 30
      seeds: "71, 42, 123, 888, 2024"
    results:
      config_A_baseline:
        RMSE: 37.45
        MAE: 24.68
        RMSE_std: 0.86
      config_B_full_PSE:
        RMSE: 35.52
        MAE: 22.62
        RMSE_std: 0.41
        delta: "-1.93K"
      conclusion: "PSE提供1.93K显著提升；modular_position贡献最大；is_backbone与PSPE冗余"

  - name: "7_day7_cpa_pspe"
    description: "CPA + PSPE消融 - 隔离CPA与PSPE各自贡献"
    model_architecture:
      type: "BaselineGAT / CPAGAT / CPAGATwithPSPE"
      configs:
        A: "BaselineGAT (40维, 无PSPE)"
        B: "CPAGAT (40维, 无PSPE)"
        C: "CPAGAT + PSPE (42维)"
      pspe_features: "DCE(距离链端) + MI(单体索引)"
    training:
      dataset: "Tg (3-RU, 72/8/20)"
      batch_size: 64
      optimizer: "AdamW (lr=1e-3)"
      scheduler: "CosineAnnealingLR"
      early_stopping: 30
      seeds: "71, 72, 73"
    results:
      config_A_baseline:
        RMSE: 37.45
        MAE: 24.68
        RMSE_std: 0.86
      config_B_cpa:
        RMSE: 35.52
        MAE: 22.62
        RMSE_std: 0.41
        delta_cpa: "-1.93K"
      config_C_cpa_pspe:
        RMSE: 35.86
        MAE: 23.60
        RMSE_std: 1.06
        delta_pspe: "+0.34K (微弱)"
      conclusion: "CPA单独贡献1.93K提升；PSPE在CPA之上仅边际收益(+0.34K)"

  - name: "7_day7_polybg_cpa"
    description: "PolyBG-CPA - 并行键图CPA架构"
    model_architecture:
      type: "PolyBG-CPA (并行原子图+键图编码器)"
      atom_encoder: "4层GAT, 256H (CPA-GAT)"
      bond_encoder: "3层GAT, 128H"
      bond_graph_features: "14维 (键类型/共轭/环/可旋转/键序)"
      fusion: "并行编码 + 拼接readout"
      experiments:
        exp1: "CPA-GAT Atom-Only (42维)"
        exp2: "PolyBG-CPA Parallel (Atom 4L256H + Bond 3L128H)"
        exp3: "PolyBG-noCPA (无CPA并行)"
        exp4: "Bond-CPA-GAT Only (仅键图)"
        exp5: "Enriched-Edge CPA (8维边特征)"
        exp6: "Hierarchical-Pre (层级融合)"
    training:
      dataset: "Tg (3-RU, 72/8/20)"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=1e-3, weight_decay=1e-4, grad_clip=1.0)"
      early_stopping: 30
      seeds: "71, 42, 123"
    results:
      exp1_atom_only:
        RMSE: 35.80
        RMSE_std: 0.44
        MAE: 23.12
        params: "835K"
      exp2_parallel:
        RMSE: 35.64
        RMSE_std: 0.27
        MAE: 23.16
        params: "1.0M"
      exp3_no_cpa:
        RMSE: 36.99
        MAE: 24.45
        params: "715K"
      exp4_bond_only:
        RMSE: 45.37
        MAE: 29.18
        params: "176K"
      exp5_enriched_edge:
        RMSE: 40.45
        MAE: 27.49
        params: "1.37M"
      exp6_hierarchical:
        RMSE: 110.25
      conclusion: "并行架构仅边际提升(0.16K)；CPA机制是关键；纯键图效果差；层级融合严重失败"

summary:
  total_projects: 23
  runnable_projects: 16
  template_projects: 7
  best_rmse: 34.77
  best_model: "6_day6_cpa_gat (CPA-GAT)"
  best_with_pse: 35.52
  best_pse_model: "6_day7_cpa_pse (CPA-GAT + PSE)"
  day_by_day_best:
    day1: "1_day1_multimer_feature_gat (41.61K, 3-RU × Rich)"
    day2: "2_day_2_lspe_gat (36.29K, static RWPE k=16)"
    day3: "3_3_day3_pspe (35.44K, PSPE combined)"
    day4: "4_4_day4_selfgnn_polymer (39.99K, BYOL pretrain)"
    day5: "5_day5_gps_lite baseline (35.19K, GAT+RWPE+PSPE)"
    day6: "6_day6_cpa_gat (34.77K, CPA-GAT)"
    day7: "6_day7_cpa_pse (35.52K, CPA-GAT+PSE)"
  key_findings:
    - "CPA(通道注意力)机制是最有效的架构改进(1.6-1.9K提升)"
    - "特征工程(rich features)比架构修改影响更大"
    - "3-RU多聚体优于1/2-RU和混合策略"
    - "PSPE(DCE+MI)提供稳定但有限的提升(0.8-2.3K)"
    - "对比学习/自监督预训练(BYOL, NT-Xent, GraphMAE)对Tg预测帮助有限甚至有害"
    - "全局注意力层严重恶化性能(Δ>+8K)"
    - "SPSE路径计数在聚合物图中退化为常数"
    - "Max聚合导致方差坍缩，不适合此任务"
    - "并行键图架构边际收益(0.16K)，复杂度不值得"
    - "PSE(modular_position)是除CPA外最有价值的特征工程"
```

```
day_traj
  day1
  learn:
  exp:



```

