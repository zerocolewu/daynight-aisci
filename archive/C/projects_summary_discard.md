# Ablation4 Archive Projects Summary

```yaml
projects:
  - name: "1_day2_ppfn"
    description: "PPFN - 聚合物周期性片段网络，分层GNN架构用于Tg预测"
    model_architecture:
      type: "PPFN (Polymer Periodic Fragment Network)"
      hidden_dim: 256
      total_params: "~535万"
      components:
        - "原子级GNN: 4层GINEConv，处理三聚体图"
        - "片段级GNN: 2层TransformerConv，处理BRICS片段图"
        - "原子到片段聚合: 多头注意力机制"
        - "长程双模块: Transformer风格交叉注意力(2层)"
        - "预测头: 3个全局池化拼接 + MLP"
      atom_features: "27维 (18种原子类型 + 度 + 5种杂化 + 芳香性 + 形式电荷 + 环)"
      edge_features: "6维 (4种键类型 + 共轭 + 环)"
      fragment_features: "31维 (原子特征均值 + 片段大小 + 芳香性 + OH/NH标志)"
      activation: "ReLU"
      regularization: "Dropout(0.1-0.2) + LayerNorm + BatchNorm + 权重衰减(1e-4)"
    training:
      dataset: "7170聚合物样本 (Tg.csv)"
      split: "72/8/20"
      batch_size: 64
      epochs: 500
      optimizer: "Adam (lr=1e-3, weight_decay=1e-4)"
      scheduler: "CosineAnnealingWarmRestarts (T0=50, T_mult=2)"
      loss: "Huber Loss (β=10.0)"
      early_stopping: 50
      graph_construction: "三聚体 + 周期性边 + BRICS片段分割"
    results:
      test:
        RMSE: 37.36
        MAE: 24.21
        R2: 0.8917
      ablation_periodic_edge:
        with_periodic: "RMSE=37.76, R²=0.8893"
        without_periodic: "RMSE=38.73, R²=0.8836"
        contribution: "RMSE改善0.97 (2.51%)"
      conclusion: "分层架构有效结合原子级和片段级信息；周期性边贡献2.51%提升；BRICS片段分割提供可解释性"

  - name: "2_day2_rit_polymer"
    description: "RIT - 重复不变转换器，GINE+Transformer混合架构"
    model_architecture:
      type: "RIT (Repetition-Invariant Transformer)"
      hidden_dim: 256
      components:
        - "GINN编码器: 5层GINE，共享权重处理3个重复单元"
        - "重复不变聚合: 最大池化跨RU聚合"
        - "Transformer编码器: 4层, 8头注意力"
        - "注意力池化: 多头自注意力全局特征提取"
        - "多任务头: Tg回归 + 范围分类"
        - "偏差校正网络: 基于范围分类校正预测"
      node_features: "30维"
      edge_features: "7维"
      global_features: "16维"
      attention_heads: 8
      activation: "ReLU, GELU"
      regularization: "Dropout(0.1) + 权重衰减(1e-5) + 梯度裁剪(max_norm=5.0)"
    training:
      dataset: "7170聚合物样本"
      split: "72/8/20 (分层采样)"
      batch_size: 32
      epochs: 300
      optimizer: "AdamW (lr=1e-3, weight_decay=1e-5)"
      scheduler: "ReduceLROnPlateau"
      loss: "MSE + 交叉熵 (权重0.1)"
      early_stopping: 30
      ensemble: "5个模型集成 (seed=42-46)"
    results:
      target_rmse: 30.8
      best_ensemble_rmse: 34.14
      conclusion: "重复不变性通过多RU并行处理实现；多任务学习同时预测Tg值和范围；集成学习提升稳定性；未达SOTA目标30.8"

  - name: "3_03_iapt"
    description: "IAPT - 强化感知聚合物Transformer (基础版，目录为空，功能由v2继承)"
    model_architecture:
      type: "IAPT (Intensive-Aware Polymer Transformer)"
      note: "此目录为空，模型架构和结果继承自4_iapt_v2_intensive_aware"
    training:
      method: "与v2版本相同的基础实验框架"
    results:
      conclusion: "基础版本为v2的消融研究提供了对照基线，具体指标见4_iapt_v2"

  - name: "4_iapt_v2_intensive_aware"
    description: "IAPT v2 - 强化感知聚合物Transformer升级版，含线图GNN和3D等变特征"
    model_architecture:
      type: "IAPT v2 (Intensive-Aware Polymer Transformer v2)"
      hidden_dim: 256
      gnn_layers: 6
      line_graph_layers: 2
      line_graph_hidden: 128
      total_params: "873K-1M"
      components:
        - "GINEConv层: 6层图神经网络"
        - "线图GNN: 2层，捕获高阶连接模式"
        - "3D等变特征: 3维 (到质心距离 + 方向性)"
        - "虚拟节点: 全局信息聚合"
        - "多尺度池化: mean + max + sum + attention"
      atom_features: "27维 (原子类型 + 度 + 杂化 + 芳香性 + 电荷 + 环)"
      edge_features: "6维 (键类型 + 共轭 + 环)"
      global_features: "14维 (分子量 + 可旋转键数 + 环数等)"
      activation: "GELU, ReLU"
      regularization: "Dropout(0.15) + LayerNorm + BatchNorm + 权重衰减(1e-4) + 梯度裁剪(1.0)"
    training:
      dataset: "7170聚合物样本"
      split: "72/8/20"
      batch_size: 32
      epochs: 500
      optimizer: "AdamW (lr=5e-4)"
      scheduler: "CosineAnnealingWarmRestarts"
      loss: "Huber Loss"
      early_stopping: 40
      seeds: "[42, 123, 456, 789, 2024]"
      graph_construction: "三聚体 + 3D构象(ETKDGv3) + 周期边界"
    results:
      single_model_best:
        variant: "IAPT_v2_att (注意力池化)"
        RMSE: 35.19
        MAE: 23.65
        R2: 0.9039
      ensemble_5seeds:
        RMSE: 35.08
        MAE: 22.93
        R2: 0.9047
      ablation:
        mean_pooling: "RMSE=37.89, R²=0.8886"
        max_pooling: "RMSE=35.35, R²=0.9030"
        attention_pooling: "RMSE=35.19, R²=0.9039 (最佳)"
        sum_pooling: "RMSE=35.99, R²=0.8995"
        no_equivariant_features: "RMSE=36.76, R²=0.8951"
        no_line_graph: "RMSE=36.43, R²=0.8970"
      conclusion: "Attention池化最优；3D等变特征贡献1.57 RMSE；线图GNN贡献1.24 RMSE；集成达RMSE=35.08，仍未达目标30.8"

  - name: "5_day5_pretraining"
    description: "GraphMAE预训练 + 微调，大规模自监督学习"
    model_architecture:
      type: "GraphMAE (Graph Masked Autoencoder) + FineTuner"
      encoder:
        layers: 5
        hidden_dim: 256
        type: "GINEConv"
        features: "残差连接 + LayerNorm + ReLU + Dropout(0.15)"
      multi_scale_encoder:
        type: "JK-net风格多尺度聚合"
        layers_fused: "第3,4,5层中间表示"
        output_dim: 768
      decoder:
        layers: 3
        type: "GINEConv"
        features: "节点特征重建 + 边特征重建"
      masking:
        ratio: 0.65
        features: "学习掩码token + 重掩码机制 + 边掩码"
      fine_tuner:
        poolings: "多头注意力(4头) + mean + max"
        prediction_head: "3层MLP, Dropout(0.3)"
      node_features: 27
      contrastive_loss: "InfoNCE (τ=0.1)"
    training:
      pretrain:
        dataset: "PI1M (30万样本)"
        batch_size: 256
        epochs: 60
        optimizer: "AdamW (lr=1e-3)"
        scheduler: "Warmup(5轮) + 余弦退火"
        loss_weights: "λ_node=1.0, λ_edge=0.5, λ_cl=0.1"
        augmentation: "原子掩码20%, 边丢弃10%"
      finetune:
        dataset: "7170聚合物样本 (72/8/20)"
        batch_size: 64
        epochs: 500
        optimizer: "AdamW (头部lr=5e-4, 编码器渐进解冻lr→1e-5)"
        early_stopping: 100
        progressive_unfreezing: "前30轮冻结编码器"
    results:
      test:
        RMSE: 35.60
        MAE: 22.92
        R2: 0.9016
      best_val_rmse: 34.72
      train_rmse: 22.68
      conclusion: "GraphMAE预训练有效，R²超0.9；30万样本预训练提供良好初始表征；多尺度特征聚合有效；但训练-验证差距大(12RMSE)，存在过拟合"

  - name: "5_day5_pretraining_v2"
    description: "GraphMAE预训练v2 - 改进微调策略"
    model_architecture:
      type: "GraphMAE v2 + FineTuner"
      encoder:
        layers: 5
        hidden_dim: 256
        type: "GINEConv"
      decoder:
        layers: 3
        features: "节点+边特征重建"
      masking:
        ratio: 0.65
        features: "重掩码 + 边掩码"
      fine_tuner:
        poolings: "多头注意力(4头) + mean + max"
        prediction_head: "256→256→128→1"
        dropout: 0.3
    training:
      pretrain:
        dataset: "PI1M (30万样本)"
        batch_size: 256
        epochs: 60
        optimizer: "AdamW (lr=1e-3)"
        scheduler: "Warmup(5轮) + 余弦退火"
        loss_weights: "λ_node=1.0, λ_edge=0.5, λ_cl=0.1"
      finetune:
        dataset: "7170聚合物样本 (72/8/20)"
        batch_size: 32
        epochs: 500
        optimizer: "AdamW (头部lr=5e-4, 编码器解冻lr=1e-5)"
        scheduler: "ReduceLROnPlateau (factor=0.5, patience=20)"
        early_stopping: 80
        progressive_unfreezing: "前50轮冻结编码器"
    results:
      test:
        RMSE: 36.14
        MAE: 24.73
        R2: 0.899
      val_rmse: 35.99
      comparison_with_v1:
        v1_rmse: 35.60
        v2_rmse: 36.14
        note: "v2略差于v1，ReduceLROnPlateau替代CosineAnnealing未带来改善"
      conclusion: "v2微调策略调整未带来提升；渐进式解冻50轮vs v1的30轮效果相近；预训练框架已接近瓶颈"

  - name: "6_day6_polykanssm"
    description: "PolyKAN-SSM v5 - 结合傅里叶KAN和状态空间模型的GNN"
    model_architecture:
      type: "PolyKAN-SSM v5"
      hidden_dim: 256
      gnn_layers: 4
      total_params: "~221万"
      components:
        - "GINEConv: 4层图神经网络 (每层2×hidden MLP)"
        - "傅里叶KAN: 线性层+小幅度傅里叶扰动激活"
        - "跳跃连接(JumpingKnowledge): 拼接所有层输出"
        - "虚拟节点: 全局信息，轻量级更新"
        - "注意力池化: 对JK输出注意力加权"
        - "SSM风格连接: 可学习衰减系数(λ)控制信息流"
      node_features: "21维"
      edge_features: "23维 (键类型 + 共价/非共价 + 高斯距离编码)"
      fourier_frequencies: 4
      spectral_radius: 0.99
      activation: "SiLU (Swish)"
      regularization: "Dropout(0.25-0.3) + 权重衰减(0.002-0.005) + 梯度裁剪(1.0)"
    training:
      dataset: "7170聚合物样本"
      split: "72/8/20"
      batch_size: 32
      epochs: 500
      optimizer: "AdamW (lr=5e-4)"
      scheduler: "ReduceLROnPlateau (patience=15, factor=0.5)"
      early_stopping: 30
      seeds: "[42, 123, 456, 789, 2024]"
      graph_construction: "共价键 + 非共价作用(截止5Å) + 3D构象(ETKDG)"
    results:
      test_5seed_avg:
        RMSE: 39.21
        RMSE_std: 1.39
        MAE: 26.04
        MAE_std: 0.92
        R2: 0.877
        R2_std: 0.004
      best_single_seed:
        seed: 456
        RMSE: 37.55
        R2: 0.885
      train_val_gap: 16.41
      ablation:
        no_fourier_kan: "RMSE=39.81"
        no_jumping_knowledge: "RMSE=39.83"
        no_virtual_node: "RMSE=38.48 (最佳)"
      conclusion: "模型容量过大导致严重过拟合(训练-验证差距16.41)；傅里叶KAN贡献有限；虚拟节点移除反而改善，提示冗余设计；未达SOTA目标"

  - name: "7_day_7_polyperio"
    description: "PolyPerio-DenseGNN - 周期性感知密集连接图神经网络"
    model_architecture:
      type: "PolyPerio-DenseGNN"
      variants:
        baseline: "单体图，无周期边，无角编码，顺序连接"
        periodic: "三聚体 + 周期边"
        angular: "三聚体 + 周期边 + 角编码"
        full: "三聚体 + 周期边 + 角编码 + 密集连接"
      components:
        - "节点嵌入层"
        - "边条件MPNN层"
        - "密集连接块 (full变体)"
        - "注意力池化"
        - "预测头: MLP"
      node_features: "68维 (原子序数+度+电荷+氢+杂化+芳香+环)"
      edge_features: "41维 (基础8维 + 1维周期标志 + 32维RBF角编码)"
      attention_heads: 4
      activation: "SiLU"
      regularization: "Dropout(0.1) + LayerNorm"
    training:
      dataset: "7170聚合物样本 (Tg.csv)"
      split: "72/8/20"
      batch_size: 64
      epochs: 500
      optimizer: "Adam (lr=1e-3, weight_decay=1e-5)"
      scheduler: "ReduceLROnPlateau (factor=0.5, patience=15)"
      loss: "Huber Loss (δ=1.0)"
      early_stopping: 30
      seeds: "[42, 123, 456, 789, 2024]"
    results:
      baseline: "RMSE=42.39±3.22"
      periodic: "RMSE=38.24±1.65"
      angular: "RMSE=36.98±1.70, MAE=24.82±1.11, R²=0.888±0.008"
      full: "RMSE=38.81±1.71"
      params:
        baseline_periodic: 509474
        angular: 512546
        full: 641954
      conclusion: "周期图表示贡献最大(4.15 RMSE, -9.8%)；角编码额外改善1.26 RMSE；密集连接损害性能(+1.83)；最佳为angular变体RMSE=36.98"

  - name: "alldataset"
    description: "PPFN在多数据集上的验证实验"
    model_architecture:
      type: "PPFN (Polymer Periodic Fragment Network)"
      hidden_dim: "256 (基础) / 512 (宽模型)"
      components:
        - "原子级GNN: 4-8层GINEConv"
        - "片段级GNN: 2层TransformerConv"
        - "原子到片段聚合: 注意力机制"
        - "长程双模块: 交叉注意力"
      attention_heads: 4
      activation: "ReLU"
      regularization: "Dropout(0.1-0.15) + 权重衰减(1e-4)"
    training:
      dataset: "Tg数据集 (1435测试样本)"
      split: "72/8/20"
      batch_size: "64 (基础) / 32 (宽模型)"
      epochs: "300-500"
      optimizer: "Adam (lr=5e-4~1e-3)"
      scheduler: "CosineAnnealingWarmRestarts (T0=30-50)"
      loss: "Huber Loss (β=10.0)"
      early_stopping: "30-50"
      ensemble: 5
    results:
      base_periodic:
        RMSE: 42.76
        R2: 0.858
      deep_8layers:
        RMSE: 41.92
        R2: 0.864
      wide_512dim:
        RMSE: 39.21
        MAE: 26.21
        R2: 0.881
      ensemble_5models:
        RMSE: 36.74
        MAE: 24.04
        R2: 0.895
      ablation_periodic_edge:
        with: "RMSE=37.76"
        without: "RMSE=38.73"
        improvement: 0.97
      conclusion: "宽模型(512维)最佳；集成进一步提升2.5 RMSE；高Tg区域(>400K)误差显著增大(RMSE=72.0)"

  - name: "baseline"
    description: "GAT基线模型，多性质预测(Egc/Tg/Tm)"
    model_architecture:
      type: "GAT (Graph Attention Network)"
      hidden_channels: 64
      layers: 3
      attention_heads: 4
      dropout: 0.2
      components:
        - "GATConv: 输入→64×4"
        - "GATConv: 64×4→64 (多层)"
        - "GATConv: 64→64"
        - "预测头: Linear(64→32) → Linear(32→1)"
      node_features: "原子特征 (原子序数+度+形式电荷+杂化+芳香性+氢数)"
      edge_features: "键类型one-hot (单/双/三/芳香)"
      activation: "ELU (GAT层) + ReLU (预测头)"
      regularization: "LayerNorm + Dropout(0.2)"
    training:
      split: "72/8/20"
      batch_size: 16
      epochs: 500
      optimizer: "Adam (lr=1e-3, weight_decay=1e-5)"
      scheduler: "ReduceLROnPlateau (factor=0.5, patience=10)"
      loss: "MSE"
      early_stopping: 50
      seed: 42
    results:
      Egc:
        RMSE: 0.6326
        MAE: 0.4615
        R2: 0.8361
        samples: "训练2433/验证271/测试676"
        note: "最佳"
      Tg:
        RMSE: 50.52
        MAE: 37.44
        R2: 0.7912
        samples: "训练4971/验证553/测试1382"
      Tm:
        RMSE: 79.30
        MAE: 60.42
        R2: 0.5026
        samples: "训练2615/验证291/测试727"
        note: "最难预测"
      conclusion: "GAT基线在Egc上表现最佳(R²=0.84)；Tm预测最困难(R²=0.50)；Tg基线RMSE=50.52为后续改进提供参照"

summary:
  total_projects: 10
  best_single_model_rmse: 35.08
  best_single_model: "4_iapt_v2_intensive_aware (5种子集成, RMSE=35.08)"
  best_single_seed_rmse: 35.19
  best_single_seed_model: "4_iapt_v2_intensive_aware (IAPT_v2_att单种子)"
  target_rmse: 30.8
  target_achieved: false
  key_findings:
    - "三聚体+周期性边是聚合物Tg预测的关键表示策略，贡献约2-10% RMSE改善"
    - "3D等变特征和线图GNN对IAPT v2性能有显著贡献"
    - "GraphMAE大规模预训练(30万样本)有效但提升有限，R²可达0.9+"
    - "注意力池化优于mean/max/sum池化"
    - "集成学习稳定带来2-3 RMSE改善"
    - "傅里叶KAN和SSM连接等复杂组件在7K数据集上贡献有限甚至有害"
    - "密集连接在PolyPerio中反而损害性能，提示过参数化风险"
    - "高Tg区域(>400K)始终是预测难点，样本稀少"
    - "所有模型均未达SOTA目标RMSE<30.8，差距约4-5 RMSE"
```
