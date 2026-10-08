# PolyPerio-DenseGNN (angular 变体) - ab4 (GLM-5, day-only)

> 一句话定位：ab4 / 基座 GLM-5 / **DAY-ONLY（night/internalize+associate 已消融）** / 最终(末天最佳)模型：PolyPerio-DenseGNN 的 angular 变体（4 变体消融最佳者），头部目录 `src/7_day_7_polyperio`（变体由 `build_model(variant='angular')` 配置，无独立子目录，模型代码 `models/polyperio_densegnn.py`）。

## 实验目的

这是 Poly-Circa AI-Scientist 的 ab4 autonomous run，基座 GLM-5，模式 **DAY-ONLY**——即把 night 阶段（internalize/associate 知识巩固）消融掉，只保留白天的 paper 阅读 + 假设 + 实验执行循环。7 天里每天自主读 2 篇论文并实现一个全新架构，在 Tg 数据集（7170 条 one-repeat-unit p-SMILES，72/8/20 划分）上试图击败 PerioGT 的 SOTA RMSE=30.8。**因为是 day-only，每天换架构，所以本 run 的判别关键不是"探索机制"本身，而是每天的实验是否因果衔接（继承前一天 finding）**。Day7（PolyPerio-DenseGNN）是最后一天，也是候选 day6/day7 中结果最好的，故为头条模型。Day7 的具体动机：验证三件事能否缩小与 SOTA 的差距——(1) 周期性聚合物图表示（三聚体 + 周期边, arXiv 2205.13757），(2) 角/方向编码（CTGNN 的 θ_x/θ_y/θ_z 方向余弦 RBF，超越仅距离 RBF），(3) 密集连接（DenseGNN 的 block 间 dense connectivity 作为隐式正则）。设计先验：Tg 是链刚性强相关的周期性材料属性，显式编码 repeat-unit 周期性与方向几何应有助于预测。

## 模型架构

头条模型 = PolyPerio-DenseGNN 的 angular 变体（4 变体消融中的最佳者）。输入到输出：

(1) **输入图构建**：p-SMILES 经 RDKit RWMol 拼接成 trimer（3 个 repeat unit），跨 unit 连接骨架原子、删除 wildcard(*) 原子、在首末原子间加一条 periodic edge 编码聚合物周期边界。

(2) **原子节点特征 144 维** one-hot = atomic_num(119 维：1-118 + wildcard 0) + degree(6) + formal_charge(5) + num_hs(5) + hybridization(5: SP/SP2/SP3/SP3D/SP3D2) + is_aromatic(2) + is_in_ring(2)。

(3) **边特征 41 维** = bond_type(4) + is_conjugated(2) + is_in_ring(2) + periodic_flag(1) + angular(32)；angular = 对方向余弦 cosθ_x/cosθ_y/cosθ_z 各做 8-center RBF（γ=5.0, centers linspace(-1,1,8)）+ edge-length 8-center RBF（centers linspace(0,8,8)）；因 RDKit 真 3D conformer 对 7K 分子太慢，改用拓扑伪 3D 坐标（到 3 个 landmark 原子的图距离）。

(4) **Node embedding**：Linear(144→96)+LayerNorm+SiLU；**Edge embedding**：Linear(41→96)+LayerNorm+SiLU。

(5) **骨干**：5 个 DenseMPNNBlock（angular 变体为顺序连接，非 dense）；每个 block = EdgeConditionedMPNN(add 聚合) + LayerNorm 残差；EdgeConditionedMPNN 内含 message_mlp[Linear(96×2+96→96)+LN+SiLU+Dropout+Linear(96→96)]、update_mlp[Linear(96+96→96)+LN+SiLU+Linear(96→96)]、edge_update_mlp[Linear(96+96×2→96)]。

(6) **Pooling**：AttentionPooling（gate_nn Linear(96→48)+SiLU+Linear(48→1), softmax 加权 global_add_pool）与 global_mean_pool 拼接 → 192 维。

(7) **预测头**：Linear(192→96)+LN+SiLU+Dropout → Linear(96→48)+SiLU+Dropout → Linear(48→1)，输出标量 Tg。

(8) **归一化/激活/Dropout**：全程 LayerNorm；激活 SiLU；Dropout=0.1。

(9) **损失**：HuberLoss(delta=1.0)，目标为原始 Tg（不做标准化）。

(10) **参数量 512,546**（hidden_dim=96）。注：full 变体额外开 dense_connectivity（每 block 前把历史特征 concat 再 Linear 投影回 96 维），参数增至 641,954；baseline/periodic 变体 edge_in=9（无 angular 的 32 维），参数 509,474。**重要**：angular 变体中 PolyPerioDenseGNN 的 `use_angular` 标志仅被存储、未在 forward 中实际使用，真正的 angular 信息是通过 edge_attr 的 32 维进入网络；即 angular 与 periodic 变体的模型结构相同，差别仅在输入边特征维度。

## 训练评估方式

**数据集**：Tg.csv 共 7170 条聚合物 p-SMILES；day7 经 RDKit 解析后 6859 条有效图（其余因 kekulize/sanitize 失败被丢弃），按 72/8/20 划分得 train=4938 / val=548 / test=1373。划分比例作用于有效样本集。

**种子协议**：5 个种子 [42, 123, 456, 789, 2024]，每个种子同时用于数据划分（random.Random(seed).shuffle）和权重初始化（set_seed: random/np/torch/cudnn deterministic）——即 **split 种子与 init 种子协变，未独立控制**（见不确定项）。

**预处理**：p-SMILES→trimer→periodic edge→angular RBF（拓扑伪坐标）；无特征标准化、无目标标准化。

**优化器**：Adam, lr=1e-3, weight_decay=1e-5。调度器：ReduceLROnPlateau(mode=min, factor=0.5, patience=15, min_lr=1e-6)，按 val RMSE 触发（log 实测 lr 序列 1e-3→5e-4→2.5e-4→1.25e-4→6.25e-5）。损失：HuberLoss(delta=1.0)。batch_size=64, max_epochs=500, early stopping patience=30（监控 val RMSE），梯度裁剪 max_norm=5.0。

**模型选择**：取 val RMSE 最低 epoch 的权重，在 test 上单次评估。**指标**：RMSE / MAE / R²（在 pooled test 预测上计算）。无 model-averaging ensemble（5 种子仅报告 mean±std）。

## 结果

**头条（angular 变体, 5 种子）**：Test RMSE = 36.98 ± 1.70, MAE = 24.82 ± 1.11, R² = 0.888 ± 0.008；best_val_rmse 37.90 ± 2.41。单种子明细：seed42=36.64, seed123=35.44（最佳单种子）, seed456=36.21, seed789=36.30, seed2024=40.29。**与 PerioGT SOTA(30.8)对比**：差 +6.18 RMSE，未达 SOTA。无 ensemble。

**Day7 四变体消融**（均 5 种子）：baseline（单体/无周期/无 angular）42.39±3.22 (R²0.853)；+periodic（三聚体+周期边）38.24±1.65 (R²0.880)；+angular（最佳）36.98±1.70 (R²0.888)；full（+dense）38.81±1.71 (R²0.876)——周期图贡献 −4.15 RMSE（−9.8%，最大单项增益），angular 额外 −1.26，dense 反而 +1.83（小数据过拟合）。

**全 run 视角**（均未达 SOTA 30.8）：day1 PPFN 最佳（3D conformer + 5 集成）RMSE 34.14 (R²0.9095)——**这是全 run 最低**；day4 IAPT v2 5 种子改进集成 RMSE 35.08 (MAE 22.93, R²0.9045)，为 **day4-7 五种子协议实验中最低**；day5 GraphMAE v1 35.60 / v2 36.14；day6 PolyKAN-SSM v5b 5 种子 38.17±1.46（最佳单种子 seed123=36.45）；day7 angular 36.98±1.70。

**其他性质**：Tm / Egc 仅出现在辅助 GAT baseline（单种子 seed=42）：Tg 50.52 (R²0.791), Tm 79.30 (R²0.503), Egc 0.6326 (R²0.836)；EA / IP 在本 run 中完全未涉及。所有 day 实验均只预测 Tg。

## 流程

**Day-by-day**（每行：目的 + 结果 + 状态）：
- **Day1**（05-06/07, PPFN 分层 GNN：GINE+GAT+原子-片段 cross-attention + 长程模块，3 轮迭代）：验证分层 + 周期边 + 3D conformer；最佳 v3（3D conformer + 5 集成）RMSE 34.14；完成，未达 SOTA，发现 3D 特征 +7.1%、周期边仅 +2.5%、trimer conformer 生成失败率 79%。
- **Day2**（05-07, RIT 重复不变 Transformer）：目的为 monomer conformer + 重复不变聚合 + Transformer 长程；但执行被持续性 API 速率限制（HTTP 429）阻断，仅留一个部分 checkpoint（best_model_seed0.pt）；状态 **killed/blocked**。
- **Day3**（05-07, IAPT 强化感知 Transformer）：设计了 intensive-property softmax attention 假设；同样被 429 阻断，**NOT EXECUTED，目录为空**；状态 **blocked**。
- **Day4**（05-07, IAPT v2 执行版）：GINE 骨干 + line graph + 等变特征 + 虚拟节点 + 多尺度池化；pooling 消融 mean 37.89/max 35.35/attention 35.19（单种子最佳）/sum 35.99, no-equivariant 36.76, no-line-graph 36.43；5 种子改进集成 35.08；完成，为 day4-7 五种子协议实验中最低（全 run 最低仍为 day1 PPFN 34.14），3D conformer 生成失败率约 57-59%（成功率 train 2235/5162=43.3%、val 244/573=42.6%、test 583/1435=40.6%）。
- **Day5**（05-07/08, GraphMAE+MoCL 预训练）：在 PI1M（实际 100K-300K 样本）上 GraphMAE 掩码自编码预训练后微调；v1 35.60 / v2 36.14；完成，过拟合（train-val gap ~12），预训练未缩小差距。
- **Day6**（05-08, PolyKAN-SSM）：Fourier-KAN + GNN-SSM + 非共价边(5Å) + JumpingKnowledge + 虚拟节点，迭代 v3→v4→v5→v5b；最佳 v5b 5 种子 38.17±1.46（单种子 seed123=36.45），过拟合（gap 10.3）；消融数字需区分版本——v5（主结果 39.21）消融 no-FourierKAN 39.81 / no-JK 39.83 / no-VN 38.48（"去 VN 反而最优"仅相对 v5 主结果 39.21 成立，38.48<39.21；对 v5b 主结果 38.17 不成立），v5b 自身消融 no-KAN 38.96 / no-JK 41.68（且无 v5b no-VN 消融文件）；完成，复杂组件在小数据上收益有限。
- **Day7**（05-08, PolyPerio-DenseGNN, 最终/最佳 day）：4 变体 × 5 种子消融，最佳 angular 36.98±1.70；验证周期图最关键、angular 次之、dense 有害；完成，未达 SOTA。

**多种子协议**：day4-7 主结果均 5 种子 [42,123,456,789,2024]；day1 集成用 [42,142,242,342,442]；day2/3 因 429 未完成。**消融贯穿全程**：day1 周期边 on/off，day4 池化/等变/线图，day6 KAN/JK/VN，day7 四变体。

**整体**：无任何模型达 SOTA 30.8；run 内最低为 day1 PPFN 集成 34.14，末天最佳为 day7 angular 36.98。**完成**：day1/4/5/6/7；**blocked/失败**：day2/3（API 429）。

## 备注

- 主要参考文件：`src/7_day_7_polyperio/{models/polyperio_densegnn.py, train.py, dataset.py, results/ablation_results.json, results/angular_seed{42,123,456,789,2024}_temp.json, logs/angular_full.log}`、`src/1_day2_ppfn/{README.md, results_v3_3d.json, results_v4.json}`、`src/4_iapt_v2_intensive_aware/{results.csv, results_v2.csv}`、`src/5_day5_pretraining{,_v2}/results.json`、`src/6_day6_polykanssm/{results_v5.json, results_v5b.json, results_v5b_ablation_no_kan.json, results_v5b_ablation_no_jk.json, ablation_results_v5.json, experiment_summary.txt}`、`src/baseline/README.md`、`mem.json`、`lite.log`。
- ✅ **本份已经过二次校验（本轮 verdict: has_issues），本轮修了**：day6 PolyKAN-SSM 消融数字归属（区分 v5/v5b 消融来源）、day4 conformer 失败率（83% → 实际 57-59%，原 83% 疑似误植 day1 数字）、全 run 最低标签（原标 day4 35.08，实际 day1 PPFN 34.14 更低）；并补全 missing_details：v5b 自身消融 no-KAN 38.96 / no-JK 41.68（无 v5b no-VN 文件）、day7 LR 序列末步 6.25e-5。其余关键数字（angular 36.98±1.70、四变体 42.39/38.24/36.98/38.81、day1 PPFN 34.14、day6 v5b 38.17±1.46 等）本轮未变动。
- 不确定项：
  - split 种子与 init 种子未独立控制：day7 用同一组 5 种子同时驱动数据划分与权重初始化，无法分离两者贡献；day1 集成种子为 [42,142,242,342,442] 与其余 day 不同。
  - hidden_dim=96 是由参数量反推（log 打印 512,546；train.py argparse 默认 hidden_dim=96，而模型 build_model 默认 128）；log 未直接回显 hidden_dim，但 512,546 与 hidden=96 + edge_in=41 的计算一致（angular 比 baseline 多 32×96=3072 参数）。
  - 自动生成的 `projects_summary_discard.md` 存在多处不准（如把 day7 节点特征标为 68 维，实际代码与 log 均为 144 维；把 day6 标为 39.21 而最新 v5b 实为 38.17）；本报告一律以代码/logs/results 原始文件为准。
  - Day2 RIT 状态含糊：experience（mem_080）记载被 429 完全阻断，但 results/ 下存在 best_model_seed0.pt，是否有部分跑完无法确定；discard 中 RIT 的 34.14 疑似继承自 day1。
  - Tg 单位未在 day7 显式声明；day5 results_v2 的 label_mean=141.95 提示为摄氏度，据此推断 Tg 为 °C，但不绝对。
  - Day7 评估集大小 1373 是对 6859 有效样本按 72/8/20 划分得到（7170 原始中约 311 条因 kekulize/sanitize 失败被丢）；不同 day 的图构建方式不同（day6 报告用了全部 7170），故各 day 的有效样本数不完全可比。
  - EA/IP 性质在本 run 完全未出现，Tm/Egc 仅来自辅助 GAT baseline（单种子），头条模型与所有 day 实验均只针对 Tg。
