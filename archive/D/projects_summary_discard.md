# Ablation3 Archive Projects Summary

```yaml
projects:
  - name: "1_day2_reperioGT"
    description: "RE-PerioGT: 刚度增强周期图Transformer用于聚合物Tg预测"
    model_architecture:
      overview: "该模型是一个结合周期性图处理和刚度特征的图Transformer架构，用于预测聚合物的玻璃化转变温度(Tg)。整体流程从SMILES输入开始，经过原子/键特征编码、周期性处理、图Transformer层、刚度特征融合，最后通过对比学习增强表示学习。"

      components:
        - name: "Atom Encoder (原子编码器)"
          description: "将原始原子特征编码为高维嵌入向量。输入是91维的one-hot编码(118个元素+6个其他特征+未知)。通过三层线性层(Linear→ReLU→Linear→ReLU→Linear)将特征映射到256维。输出形状为[num_nodes, 256]。"

        - name: "Bond Encoder (键编码器)"
          description: "将键特征编码为高维嵌入。输入是9维one-hot编码(4种键类型+2共轭+2环+未知)。同样使用三层线性层映射到256维。输出形状为[num_edges, 256]。"

        - name: "Periodic Features Module (周期特征模块)"
          description: "处理聚合物的周期性边界。识别通配符(*)原子作为连接点，生成3D周期向量。通过Linear→ReLU→Linear→ReLU→Linear处理，然后与节点特征相加。"

        - name: "Periodic Graph Transformer Layer (周期图Transformer层)"
          description: "共4层，每层包含多头注意力(4头，每头64维)。键投影中加入边特征。对周期原子使用特殊的注意力衰减(30-50%)。包含LayerNorm、Dropout、残差连接。前馈网络使用GELU激活，中间层4倍扩展。"

        - name: "Rigidity Encoder (刚度编码器)"
          description: "将可旋转键比例(单个标量)编码为256维特征向量。使用Linear→ReLU→Linear→ReLU→Linear结构。"

        - name: "Prediction Head (预测头)"
          description: "接收图嵌入和刚度嵌入的融合特征。结构为Linear→ReLU→Dropout→Linear(256→128)→ReLU→Dropout→Linear(64→1)。输出Tg预测值。"

        - name: "Contrastive Learning Module (对比学习模块)"
          description: "包含投影头(Linear→ReLU→Linear，256→128)。使用InfoNCE损失，基于刚度构建正负样本对(正样本:|Δrigidity|<0.1，负样本:|Δrigidity|>0.3)，温度参数0.1。"

      data_flow: "SMILES → 原子/键特征提取 → 原子编码器(91→256) + 键编码器(9→256) → 周期特征处理 → 4层周期图Transformer → 全局池化 → 与刚度嵌入融合(±) → 预测头(256→128→64→1) → Tg预测"

    training:
      dataset: "7170聚合物样本，Tg范围[-139, 495]°C"
      split: "80/10/10"
      batch_size: 64
      epochs: 500
      optimizer: "Adam (lr=1e-4, weight_decay=1e-5)"
      scheduler: "Cosine annealing (to 1e-6)"
      loss: "MSE + 0.1 × InfoNCE"
      early_stopping: 50
      gradient_clip: 1.0

    results:
      full_model:
        RMSE: 35.55
        MAE: 24.95
        R2: 0.9047
        Pearson: 0.9513
      baseline_no_rigidity:
        RMSE: 36.41
        R2: 0.9000
      feature_only:
        RMSE: 35.56
        R2: 0.9046
      conclusion: "刚度特征带来0.86 RMSE改进，对比学习贡献有限(0.01 RMSE)但大幅增加训练时间"

  - name: "1_hept"
    description: "HEPT: 层次等变周期Transformer - 失败实验"
    model_architecture:
      overview: "HEPT是一个层次化图神经网络，结合等变消息传递、周期注意力和长程模块。但实验失败，性能不如GAT基线。"

      components:
        - name: "Equivariant Local Module (等变局部模块)"
          description: "基于MACE的E(3)等变消息传递。使用球谐函数实现旋转等变性。3层结构，64个径向基函数用于距离编码，截断距离5.0Å。输入(num_nodes, 31)→输出(num_nodes, 256)。"

        - name: "Periodic Attention Module (周期注意力模块)"
          description: "受PerioGT启发的多头注意力。8个注意力头，2层结构。创建虚拟周期图像(1x1x1超胞)，在图内进行自注意力。输入输出均为(num_nodes, 256)。"

        - name: "Long-Range Module (长程模块)"
          description: "通过注意力池化获取全局上下文。使用虚拟超级节点机制，门控机制融合局部和全局特征。输入(num_nodes, 256)→输出(num_nodes, 256)。"

        - name: "Hierarchical Pooling (层次池化)"
          description: "结合mean、sum(归一化)和max三种池化方式。使用可学习权重加权组合。输入节点特征，输出图级表示。"

        - name: "Output Head (输出头)"
          description: "3层MLP: 256→128→64→1。层间使用LayerNorm和Dropout(0.1)。"

      data_flow: "输入特征(31维原子特征) → 输入投影(31→256) → 等变消息传递(捕获局部结构) → 周期注意力(捕获长程相互作用) → 长程模块(添加全局上下文) → 层次池化 → 输出预测(256→1)"

    training:
      dataset: "7170聚合物样本"
      split: "80/10/10"
      batch_size: 32
      epochs: 300
      optimizer: "AdamW (lr=5e-4)"
      scheduler: "ReduceLROnPlateau"
      early_stopping: 25
      dropout: 0.15

    results:
      iteration1:
        RMSE: 49.58
        MAE: 35.93
        R2: 0.8041
      iteration2:
        RMSE: 53.84
        MAE: 39.56
        R2: 0.7689
      baseline_comparison:
        GAT: 50.5
        PerioGT: 30.8
      conclusion: "失败 - 性能比GAT基线还差，增强的分子特征未被模型使用，更大的模型反而更差"

  - name: "2_02_PerioGT_Ablation"
    description: "PerioGT消融研究 - Chemprop风格MPNN实现"
    model_architecture:
      overview: "实现论文'PerioGT'的周期图表示方法，使用Chemprop风格的有向键级消息传递。核心创新是周期边构建而非'周期注意力'。"

      components:
        - name: "Periodic Graph Construction (周期图构建)"
          description: "在末端原子(通配符)之间添加边来表示聚合物链的周期性。对于p-SMILES(如*CCO*)，添加双向边连接聚合物链末端。"

        - name: "Atom Feature Encoder (原子特征编码器)"
          description: "Chemprop风格原子特征: 原子序数(118维one-hot)、度数(7维)、形式电荷(7维)、氢原子数(5维)、杂化类型(5维)、芳香性(2维)、是否在环中(2维)。总计133维，通过线性层编码。"

        - name: "Bond Feature Encoder (键特征编码器)"
          description: "Chemprop风格键特征: 键类型(4种)、是否共轭(2维)、是否在环中(2维)。总计10维。"

        - name: "Bond Message Passing Layer (键消息传递层)"
          description: "Chemprop的核心创新 - 有向键级消息传递。消息计算: f(h_bond, h_reverse_bond, edge_features)。聚合: 对入射消息求和。更新: ReLU(W * (sum + atomic_features))。3层，隐藏维度300，无Dropout。"

        - name: "Atom Feature Integrator (原子特征整合器)"
          description: "将消息传递后的特征与原始原子特征拼接。输入维度300+133=433，输出维度300。结构: Concatenate→Linear→ReLU。"

        - name: "Global Pooling (全局池化)"
          description: "使用global_mean_pool聚合节点特征到图级表示。输出维度300。"

        - name: "Prediction FFN (预测前馈网络)"
          description: "2层全连接网络，隐藏维度300，ReLU激活，输出单个回归值。"

      data_flow: "SMILES → 原子特征(133维) + 键特征(10维) + 周期边构建 → 3层键消息传递 → 原子特征整合(433→300) → 全局池化 → FFN(300→300→1) → Tg预测"

    training:
      dataset: "7170聚合物样本，Tg范围[-139, 495]°C"
      split: "5折交叉验证，80/10/10"
      batch_size: 50
      epochs: 50
      optimizer: "Adam (lr=1e-3)"
      early_stopping: 20
      gradient_clip: 1.0

    results:
      test_rmse: 46.89
      test_rmse_std: 1.41
      test_mae: 33.96
      test_r2: 0.825
      paper_result: 37.35
      gap_to_paper: "+9.54"
      conclusion: "未达预期 - 缺少NOAM学习率调度导致约9.5 RMSE差距，论文实际使用标准MPNN而非'周期注意力'"

  - name: "2_day2_grin"
    description: "GRIN: 带归纳偏置的图表示用于节点特征选择"
    model_architecture:
      overview: "基于GIN的图神经网络，探索max聚合和L1正则化对聚合物Tg预测的影响。核心假设是max聚合能提供尺度不变性。"

      components:
        - name: "Input Layer (输入层)"
          description: "148维丰富的原子/键特征，包括原子序数(119维one-hot)、度数(11维)、杂化(6维)、形式电荷(5维)、氢原子数(5维)、芳香性标志、环成员。"

        - name: "Input Projection (输入投影)"
          description: "线性层将148维特征映射到256维隐藏维度。"

        - name: "GIN Convolution Layers (GIN卷积层)"
          description: "5层GIN卷积，每层包含MLP(两个线性层256→256→256)、BatchNorm、ReLU激活。两种变体: MaxGINConv使用max聚合(GRIN创新)，标准GIN使用sum聚合。"

        - name: "Readout Layer (读出层)"
          description: "三种池化选项: mean池化、max池化、或两者拼接。将节点特征聚合为图级表示。"

        - name: "Output Head (输出头)"
          description: "Sequential网络: 256→128→1。结构: Linear→LayerNorm→ReLU→Dropout(0.15)→Linear。"

      data_flow: "原子特征(148) → 输入投影(256) → 5层GIN(带BatchNorm+ReLU+Dropout) → 读出池化 → 输出头(256→128→1) → Tg预测"

      variants:
        - "Baseline: GIN + mean聚合，无L1正则化"
        - "Max Only: GIN + max聚合，无L1"
        - "Max + L1: GIN + max聚合 + L1稀疏正则化"
        - "Full GRIN: max聚合 + L1 + {1,3}-RU数据增强"

    training:
      dataset: "聚合物Tg数据集"
      split: "80/10/10"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=5e-4)"
      scheduler: "CosineAnnealingLR"
      early_stopping: 40
      gradient_clip: 1.0
      loss: "MSE + λ×L1 (λ=0.0005)"
      augmentation: "{1,3}-RU聚合物链增强(98.5%成功率)"

    results:
      baseline:
        RMSE: 32.31
        MAE: 20.85
        status: "达到目标<35.0"
      max_only:
        RMSE: 34.03
        MAE: 22.16
      max_L1:
        RMSE: 44.54
        MAE: 32.74
      full_GRIN:
        RMSE: 46.34
        MAE: 33.76
      conclusion: "GRIN框架对Tg预测无效，简单基线优于复杂变体，mean聚合优于max"

  - name: "3_day4_feature_baseline_tm_egc"
    description: "Tm和Egc预测的特征基线模型"
    model_architecture:
      overview: "GIN图神经网络与分子特征拼接，用于预测聚合物熔点(Tm)和电子能隙(Egc)。"

      components:
        - name: "Node Embedding Layer (节点嵌入层)"
          description: "线性层将可变维度的原子特征嵌入到256维。"

        - name: "GIN Convolution Layers (GIN卷积层)"
          description: "5层GINConv，每层结构: Linear→BatchNorm→ReLU→Linear→BatchNorm→ReLU。隐藏维度256。"

        - name: "Feature Processing Branch (特征处理分支)"
          description: "处理148维分子特征。结构: BatchNorm1d(148)→Linear(148, 256)。"

        - name: "Mean Pooling (均值池化)"
          description: "global_mean_pool聚合节点特征为图级表示。"

        - name: "Output Head (输出头)"
          description: "拼接图表示(256)和特征(256)得到512维。然后: Linear(512)→BN→ReLU→Dropout→Linear(128)→BN→ReLU→Dropout→Linear(1)。"

      data_flow: "图数据(x, edge_index) + 148维特征 → 节点嵌入(→256) → 5层GIN → 均值池化(256) || 特征处理(148→256) → 拼接(512) → 输出头(512→128→1)"

    training:
      datasets:
        Tm: "3636样本"
        Egc: "3374样本"
      split: "5折交叉验证"
      batch_size: 64
      epochs: 500
      optimizer: "AdamW (lr=1e-4)"
      early_stopping: 50
      loss: "MSE"
      preprocessing: "p-SMILES中*替换为[H]，StandardScaler归一化"

    results:
      Tm_v1:
        RMSE: "50.095 ± 3.140"
        R2: "0.664 ± 0.045"
        target: "<60.0"
      Tm_v2:
        RMSE: "86.597 ± 53.050"
        best_fold: 55.630
      Egc_v1:
        RMSE: "3.042 ± 3.109"
        best_fold: 0.436
      Egc_v2:
        RMSE: "1.192 ± 0.587"
      conclusion: "v1版本更稳定，数据集大小从v1到v2增加但稳定性下降"

  - name: "3_pfcl_pretraining"
    description: "PF-CL: 聚合物特征对比学习预训练"
    model_architecture:
      overview: "GIN编码器配合对比学习和特征预测的联合预训练框架，用于学习聚合物分子表示。"

      components:
        - name: "GINEncoder (GIN编码器)"
          description: "5层GIN，输入节点维度37、边维度10，隐藏维度256，Dropout 0.1。结构: 输入投影(Linear)→5层GINConv(带BatchNorm+ReLU+Dropout)→全局均值池化→输出投影。每层有残差连接(第一层除外)。"

        - name: "ContrastiveProjectionHead (对比投影头)"
          description: "用于对比学习的投影空间。结构: Linear(256→128)→ReLU→Dropout(0.1)→Linear(128→64)。"

        - name: "FeaturePredictionHead (特征预测头)"
          description: "预测分子特征作为辅助任务。结构: Linear(256→256)→ReLU→Dropout(0.1)→Linear(256→154)。"

        - name: "FineTuningHead (微调头)"
          description: "下游任务预测头。结构: Linear(256→128)→ReLU→Dropout(0.1)→Linear(128→64)→ReLU→Dropout(0.1)→Linear(64→1)。"

      data_flow: "图数据(节点37维，边10维) → GIN编码器(5层，隐藏256) → 全局均值池化 → 图嵌入(256) → 分支到: 对比投影(64维) + 特征预测(154维) | 微调时: 直接到微调头(1维输出)"

    training:
      pretraining:
        dataset: "PI1M (30,000样本)"
        epochs: 20
        batch_size: 128
        lr: 1e-3
        temperature: 0.07
        feature_loss_weight: 0.5
        optimizer: "AdamW"
        scheduler: "CosineAnnealingLR"
        augmentation: "shift/extend策略，概率0.5"
        loss: "InfoNCE + 0.5×MSE"
      finetuning:
        datasets: ["Tg", "Tm", "Egc"]
        folds: 5
        batch_size: 64
        epochs: 300
        early_stopping: 50

    results:
      Tg:
        RMSE: "39.23 ± 1.81"
        vs_baseline: "+21.4% WORSE"
      baseline_target: 32.31
      conclusion: "联合预训练表现不如基线，对比学习和特征预测目标冲突"

  - name: "4_day4_ensemble_uq"
    description: "多种子集成与不确定性量化"
    model_architecture:
      overview: "基于GIN的多种子集成方法，通过模型间 disagreement 提供不确定性估计，用于Tm和Egc预测。"

      components:
        - name: "GIN Backbone (GIN骨干网络)"
          description: "5层GIN，隐藏维度256。节点特征7维: 原子序数、度数、形式电荷、自由基电子、杂化、芳香性、原子质量。使用GINConv(双ReLU变换+BatchNorm)。"

        - name: "Feature Encoder (特征编码器)"
          description: "148维丰富的分子描述符: 基本分子性质(15维)、环信息(10维)、电子性质(8维)、拓扑指数(15维)、氢键(6维)、可旋转键(5维)、原子类型(9维)、杂化(4维)、键类型(4维)、度分布(6维)、环大小(6维)、官能团(20维)等。"

        - name: "Output Head (输出头)"
          description: "2层结构，隐藏维度128，ReLU激活，Dropout 0.2，最终输出单个回归值。"

      ensemble_strategy:
        seeds: [42, 123, 456, 789, 1024]
        models_per_fold: 5
        total_models: "25每属性"
        prediction: "5个种子的均值"
        uncertainty: "5个种子的标准差"

      data_flow: "p-SMILES → RDKit处理 → 148维特征 + 图构建(7维节点) → GIN处理(5层) → 全局均值池化 → 特征融合 → 输出头 → 预测值 → 5种子集成 → 均值预测±不确定性"

    training:
      datasets:
        Tm: "3636样本"
        Egc: "3374样本"
      split: "5折交叉验证，80/20"
      batch_size: 32
      epochs: 500
      optimizer: "AdamW (lr=1e-4)"
      early_stopping: 50
      loss: "MSE"

    results:
      Tm:
        mean_RMSE: "59.18 ± 1.12"
        best_RMSE: 57.64
        variance_reduction: "18.94%"
        uncertainty_correlation: "0.25-0.34"
        target: "<70 PASSED"
      Egc:
        mean_RMSE: "1.12 ± 0.50"
        best_RMSE: 0.73
        fold3_RMSE: 2.11
        target: "<0.9 FAILED"
      conclusion: "集成大幅减少Tm方差，Egc Fold 3持续过拟合，高不确定性样本错误高1.4-5.6倍"

  - name: "4_egc_conjugation"
    description: "Egc共轭特异性特征实验 - 解决Fold 3失败"
    model_architecture:
      overview: "GIN与共轭特异性特征融合，针对电子能隙(Egc)预测中的共轭系统进行优化。"

      components:
        - name: "Node Embedding Layer (节点嵌入层)"
          description: "9维原子特征(原子序数、度数、形式电荷、自由基电子、芳香性、总氢、杂化、总价、质量)→256维嵌入。结构: Linear→ReLU→Dropout(0.1)。"

        - name: "GIN Layers (GIN层)"
          description: "5层GIN，每层: Linear(256→256)→ReLU→Linear(256→256)→BatchNorm1d(256)。GINConv应用消息传递和邻域聚合。"

        - name: "Graph Pooling (图池化)"
          description: "global_mean_pool()聚合分子图节点特征，输出256维图嵌入。"

        - name: "Feature Fusion (特征融合)"
          description: "通用特征148维 + 共轭特征10维。共轭特征包括: aromatic_ring_count, aromatic_ring_fraction, conjugated_bond_fraction, conjugation_length, sp2_sp2_bond_fraction, pi_electron_density, pi_electron_count, aromatic_atom_fraction, avg_conjugated_bond_order, aromatic_heteroatom_count。分别嵌入后与图嵌入拼接。"

        - name: "Prediction Head (预测头)"
          description: "融合后512维(256图+256特征)。结构: Linear(512→256)→ReLU→Dropout(0.1)→Linear(256→128)→ReLU→Dropout(0.1)→Linear(128→1)。"

      data_flow: "SMILES → RDKit → 148通用特征 + 10共轭特征 + 图(9维节点) → 特征归一化 → GIN处理(5层) → 特征融合(分别嵌入后拼接) → 预测头(512→1) → Egc预测"

    training:
      dataset: "3380 Egc样本"
      split: "分层5折交叉验证(按single_bond_frac分5分位数)"
      batch_size: 32
      epochs: 500
      optimizer: "AdamW (lr=1e-4)"
      scheduler: "ReduceLROnPlateau"
      early_stopping: 50
      loss: "HuberLoss (delta=0.5)"
      gradient_clip: 1.0

    results:
      mean_RMSE: "0.5262 ± 0.0471"
      mean_MAE: "0.3371 ± 0.0161"
      mean_R2: "0.8854 ± 0.0232"
      fold_results:
        fold1: {RMSE: 0.4992, R2: 0.9028}
        fold2: {RMSE: 0.5132, R2: 0.8930}
        fold3: {RMSE: 0.6145, R2: 0.8411}
        fold4: {RMSE: 0.5271, R2: 0.8856}
        fold5: {RMSE: 0.4772, R2: 0.9044}
      baseline_comparison:
        baseline_RMSE: 1.19
        improvement: "55.8%"
        fold3_improvement: "72.7%"
      conclusion: "共轭特异性特征显著改善Egc预测，Fold 3从2.25降到0.6145"

  - name: "5_day5_fd_mtl"
    description: "FD-MTL: 特征分解多任务学习"
    model_architecture:
      overview: "基于GIN的多任务学习框架，通过特征分解分析共享和属性特异性特征，同时预测Tg/Tm/Egc三个聚合物属性。"

      components:
        - name: "Shared Encoder (共享编码器)"
          description: "GIN (Graph Isomorphism Network)，5层，隐藏维度256，mean聚合。包含节点嵌入层、5层GIN卷积、BatchNorm、ReLU激活。"

        - name: "Feature Processing (特征处理)"
          description: "148维分子描述符: 基本性质(15维)、环信息(10维)、电子性质(8维)、拓扑指数(15维)、氢键(6维)、可旋转键(5维)、EState指数(4维)、额外描述符(25维)、聚合物特异性(60维)等。"

        - name: "Feature Decomposition (特征分解)"
          description: "基于特征-属性相关性分解: 共享特征(51个，对≥2属性重要)、Tg特异性(24个)、Tm特异性(0个)、Egc特异性(5个)。相关性阈值0.3。"

        - name: "Task Heads (任务头)"
          description: "三个属性特异性预测头。每个头: 3层MLP，结构256→128→64→1。输入为[图嵌入, 共享特征, 属性特异性特征]拼接。激活ReLU，Dropout 0.2。"

      data_flow: "PyG Data(分子图+148维特征) → 节点嵌入(→256) → 5层GIN → mean池化 → 图嵌入(256) → 特征分解 → 拼接[图嵌入, 共享特征, 特异性特征] → 各属性预测头(256+features→128→64→1) → Tg/Tm/Egc预测"

    training:
      datasets:
        Tg: "7,170样本"
        Tm: "3,651样本"
        Egc: "3,380样本"
      split: "5折分层交叉验证"
      batch_size: 64
      epochs: 500
      optimizer: "AdamW (lr=1e-3)"
      early_stopping: 50
      loss: "MSE"
      feature_scaling: "StandardScaler"
      target_scaling: "StandardScaler"

    results:
      single_task_baseline:
        Tg: {RMSE: 188.37, std: 286.34}
        Tm: {RMSE: 89.69, std: 59.64}
        Egc: {RMSE: 1.0135, std: 0.3473}
      mtl_all_features:
        Tg: {RMSE: 39.50, std: 2.02}
        Tm: {RMSE: 59.10, std: 1.14}
        Egc: {RMSE: 0.5138, std: 0.0105}
      mtl_feature_decomposition:
        Tg: {RMSE: 40.10, std: 1.54}
        Tm: {RMSE: 58.92, std: 1.22}
        Egc: {RMSE: 0.5190, std: 0.0124}
      conclusion: "多任务学习显著优于单任务，特征分解提供轻微改进"

  - name: "6_day6_cfd_mtl"
    description: "CFD-MTL: 校正特征分解多任务学习 - 修正Day 5失败"
    model_architecture:
      overview: "修正版FD-MTL，使用经验证的强相关性特征而非通用RDKit描述符，同时预测Tg/Tm/Egc。"

      components:
        - name: "Encoder (编码器)"
          description: "GIN (Graph Isomorphism Network)，5层，隐藏维度256。每层结构: Linear(hidden_dim, hidden_dim)→BatchNorm1d→ReLU→Linear→BatchNorm1d→ReLU。使用全局均值池化。"

        - name: "Feature Decomposition (特征分解)"
          description: "148个验证的分子特征，按相关性分解: 共享(46个，对≥2属性重要)、Tg特异性(14个，额外刚度度量)、Tm特异性(0个)、Egc特异性(4个，共轭特异性度量)。关键特征: Tg-rotatable_bond_frac(r=-0.728), Egc-conjugated_bond_fraction(r=-0.7475)。"

        - name: "Task Heads (任务头)"
          description: "属性特异性预测头。输入[图嵌入, 共享特征, 属性特异性特征]拼接。结构: Linear(input, hidden)→BatchNorm1d→ReLU→Dropout(0.1)→Linear(hidden, hidden//2)→BatchNorm1d→ReLU→Dropout(0.1)→Linear(hidden//2, 1)。三个独立头分别输出Tg/Tm/Egc。"

      data_flow: "分子图 + 148维验证特征 → 节点嵌入 → 5层GIN(带BatchNorm) → 全局均值池化 → 图嵌入(256) → 特征分解(共享/特异性) → 拼接 → 各任务头 → Tg/Tm/Egc预测"

    training:
      datasets:
        Tg: "7145分子"
        Tm: "3636分子"
        Egc: "3374分子"
      split: "5折交叉验证"
      batch_size: 64
      epochs: 500
      optimizer: "AdamW (lr=1e-4)"
      early_stopping: 50
      loss: "MSE (各任务)"
      task_scheduling: "交替批次: tg→tm→egc"

    results:
      baseline:
        Tg_RMSE: 32.31
        Tm_RMSE: "59.18 ± 1.12"
        Egc_RMSE: "0.5262 ± 0.0471"
      single_task:
        Tg_RMSE: "98.27 ± 95.45"
        Tm_RMSE: "75.97 ± 15.20"
        Egc_RMSE: "1.0120 ± 0.0775"
      naive_mtl:
        Tg_RMSE: "44.67 ± 9.31"
        Tm_RMSE: "56.55 ± 2.25"
        Egc_RMSE: "0.5624 ± 0.0462"
      cfd_mtl:
        Tg_RMSE: "39.42 ± 1.44"
        Tm_RMSE: "56.16 ± 4.00"
        Egc_RMSE: "0.5265 ± 0.0334"
      conclusion: "CFD-MTL在Egc上匹配基线(0.5265 vs 0.5262)，Tm改进5.1%，特征分解减少Tg方差71%"

  - name: "7_day7_dcn_depth_scaling"
    description: "DCN-GIN: 密集连接网络深度缩放实验"
    model_architecture:
      overview: "DenseNet风格的GIN架构，通过密集连接实现更好的特征重用和梯度流动，支持更深的网络(5/10/15层)。"

      components:
        - name: "DenseGINConv Layer (密集GIN卷积层)"
          description: "输入原子特征(9维)+边特征(7维)。边编码器: Linear(7, hidden_dim)。MLP聚合器: Linear(256→256)→BatchNorm1d→ReLU→Linear(256→256)→BatchNorm1d→ReLU。GIN epsilon: 可学习标量。消息传递带边特征注入。输出: (1+ε)*x + propagate(edge_index, x, edge_embedding)。"

        - name: "TransitionLayer (过渡层)"
          description: "每4层后进行通道压缩，防止维度爆炸。结构: BatchNorm1d→ReLU→Linear压缩。压缩因子0.5(通道数减半)。"

        - name: "DenseBlock (密集块)"
          description: "4个DenseGINConv层组成的块，密集连接模式: 每层接收前面所有层输出的拼接。特征增长: 每层增加growth_rate(32)维。输出: 拼接所有前层特征→当前层处理。"

        - name: "Complete DCNGIN (完整DCN-GIN)"
          description: "输入: 原子特征(9维)+边特征(7维)+分子特征(148维)。原子编码器: Linear(9→256)。边编码器: Linear(7→256)。密集块: Block1(4层，256→384)→过渡(384→192)→Block2(4层，192→320)→过渡(320→160)。全局均值池化。特征融合: 分子特征投影(148→128)+图嵌入(160)拼接=288维。预测头: 288→128→64→1。"

        - name: "Standard GIN (标准GIN基线)"
          description: "顺序DenseGINConv层，无特征拼接或压缩，输出维度恒定256。"

      data_flow: "输入[原子(9维), 边(7维), 分子特征(148维)] → 原子/边编码(→256) → DenseBlock1(4层,256→384) → Transition(384→192) → DenseBlock2(4层,192→320) → Transition(320→160) → 全局均值池化(160) + 分子特征投影(148→128) → 拼接(288) → 预测头(288→128→64→1)"

    training:
      dataset: "7145聚合物Tg样本"
      split: "5折交叉验证，90%训练/10%验证"
      batch_size: 64
      epochs: 300
      optimizer: "Adam (lr=1e-3)"
      scheduler: "ReduceLROnPlateau (factor=0.5, patience=20)"
      early_stopping: 50
      dropout: 0.2
      hidden_dim: 256
      growth_rate: 32
      compression_factor: 0.5

    results:
      DCN_GIN_5:
        RMSE: "40.26 ± 1.42"
        MAE: "32.21 ± 1.14"
      DCN_GIN_10:
        RMSE: "38.59 ± 1.34"
        MAE: "30.87 ± 1.07"
      DCN_GIN_15:
        RMSE: "38.40 ± 1.18"
        MAE: "30.72 ± 0.94"
      Standard_GIN_10:
        RMSE: "44.16 ± 1.28"
      Standard_GIN_15:
        RMSE: "47.64 ± 3.83"
      baseline:
        Day2_5layer_GIN: 32.31
        CFD_MTL_3layer_GIN: 44.12
      conclusion: "密集连接有效缓解深层GIN的梯度消失，DCN-15比Standard-15好19.4%，深度10-15层最优"

  - name: "7_day8_tcp"
    description: "TCP: 热力学约束预训练失败分析与消融研究"
    model_architecture:
      overview: "标准GIN架构配合两种特征集(GRIN 148维 vs ReperioGT 153维)，用于诊断TCP预训练为何比Day 2基线差3K。"

      components:
        - name: "GIN Backbone (GIN骨干)"
          description: "5层GIN，隐藏维度256，Dropout 0.15，mean读出，使用BatchNorm，可训练epsilon。每层: nn.Linear(input, 256)→BatchNorm1d→ReLU→nn.Linear(256, 256)。"

        - name: "Prediction Head (预测头)"
          description: "结构: Linear(256, 256)→ReLU→Dropout(0.15)→Linear(256, 1)。"

        - name: "Feature Sets (特征集)"
          description: "GRIN特征: 148维原子特征+6维键特征=154维总计。ReperioGT特征: 153维原子特征+11维键特征=164维总计。"

      data_flow: "SMILES → 图构建 → 特征提取(148/153维) → GIN层(5层) → 全局池化(mean) → 预测头 → Tg预测"

    training:
      dataset: "Tg数据集"
      split: "5折交叉验证"
      batch_size: 64
      epochs: 300
      optimizer: "AdamW (lr=5e-4)"
      scheduler: "CosineAnnealingLR (T_max=300, eta_min=1e-6)"
      early_stopping: 40
      loss: "MSE"
      target_normalization: "StandardScaler"

    results:
      original_TCP:
        RMSE: "35.49 ± 1.60"
        scratch_baseline: "35.22 ± 1.55"
        negative_transfer: "-0.27 K"
      ablation1_GRIN:
        RMSE: "34.91 ± 1.82"
        gap_from_Day2: "+2.60 K"
      ablation2_ReperioGT:
        RMSE: "35.35 ± 1.48"
      conclusion: "特征集差异(0.44K)不显著，2.60-3.04K差距源于实现差异而非方法论问题，TCP负迁移(-0.27K)统计不显著"

  - name: "baseline"
    description: "GAT基线模型"
    model_architecture:
      overview: "标准图注意力网络(GAT)作为基线，用于Tg、Tm、Egc多属性预测对比。"

      components:
        - name: "GAT Layers (GAT层)"
          description: "3层图注意力网络，隐藏通道64，4个注意力头。"

      data_flow: "图数据 → 3层GAT(64通道,4头) → 全局池化 → 预测头 → 属性预测"

    training:
      split: "72/8/20"
      epochs: 500
      early_stopping: 50

    results:
      Egc:
        R2: 0.8361
        RMSE: 0.6326
        note: "最佳"
      Tg:
        R2: 0.7912
        RMSE: 50.52
      Tm:
        R2: 0.5026
        RMSE: 79.30
        note: "最难预测"

  - name: "alldataset"
    description: "完整数据集目录"
    datasets:
      - "Tg.csv: 7170聚合物玻璃化转变温度样本"
      - "Tm.csv: 熔点温度样本"
      - "Egc.csv: 电子能隙样本"

summary:
  total_projects: 14
  best_Tg_RMSE: 32.31
  best_Tg_model: "2_day2_grin (baseline)"
  best_Tm_RMSE: 56.16
  best_Tm_model: "6_day6_cfd_mtl"
  best_Egc_RMSE: 0.5262
  best_Egc_model: "4_egc_conjugation"
  key_findings:
    - "简单GIN+mean聚合基线在Tg上表现最佳(32.31)"
    - "特征工程至关重要: 共轭特征对Egc预测改进55.8%"
    - "密集连接(DCN)有效缓解深层GNN梯度消失"
    - "多任务学习对Tm有效，特征分解减少方差"
    - "对比学习预训练对Tg预测任务提升有限甚至负面"
    - "复杂架构(HEPT)不一定优于简单基线"
    - "集成方法大幅减少预测方差并提供可靠不确定性估计"
```
