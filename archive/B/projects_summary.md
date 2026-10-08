# ab6（deepseek-v4-pro，full day-night）逐天实验轨迹 — CPA-GAT 路线

## 概览与约定

- **run 身份**：ab6 / 基座 LLM = deepseek-v4-pro / **full day-night**（日做实验、夜做 internalize 巩固 + associate 生成新 idea）/ 共 7 天。mem.json 共 165 节点（100 knowledge + 16 experience + 27 integration + 22 idea）。
- **最终最强模型**：CPA-GAT（Cardinality-Preserving Attention，基数保持注意力 GAT），Day6 突破前期 ~36 K 天花板；Optimize 阶段用 ORIG_BB 配置（CPA-GAT + is_backbone 节点特征 + SWA）在全部 5 个性质上做协议对齐干净复跑。
- **主线轨迹**：数据中心特征工程（富化学特征主导）→ 位置编码（static RWPE / PSPE）→ 三次 PI1M 自监督预训练（GraphMAE / SelfGNN-BYOL / MoMuCon-NT-Xent）**全部失败**→ Day6 提出"基数盲"假设、用 CPA 打破指数衰减 → Day7 验证 CPA 聚合层已 subsume（吞噬）链几何位置编码与键图基数，两条特征工程范式关闭。
- **单位约定**：RMSE 单位 K（开尔文温度差），量纲同 °C；Tg/Tm 为 K，Egc（带隙）/EA（电子亲和能）/IP（电离能）为各自原始单位。
- **可信度约定**：所有数字以**代码/结果文件为权威**；mem.json 的 AI 当夜自述仅作线索，凡引用其独有、代码未佐证的数字/结论，标注"（AI 当夜自述，代码未独立验证）"。特殊缩写在首次出现处展开含义（GCN/GAT/GIN/ECFP 等常用 GNN/指纹术语不再展开）。

### day→实验目录映射

| Day | src/ 目录 | 主题 | 代表结果 |
|---|---|---|---|
| 1 | `1_day1_multimer_feature_gat`、`1_day1_followup_grin_elene` | 数据中心：multimer×特征消融 + GRIN/ELENE 跟进 | 3-RU×rich 41.6 K；GRIN/ELENE 全负 |
| 2 | `2_day2_spse_maxgat`、`2_day_2_lspe_gat` | SPSE-MaxGAT + LSPE/RWPE-GAT + 架构 scaling | static RWPE k=32 = 35.94 K（新最佳） |
| 3 | `3_day3_graphmae_pretrain`、`3_day3_pspe`、`3_3_day3_pspe` | GraphMAE 掩码预训练 + PSPE 位置编码 | PSPE combined 35.44 K（新最佳） |
| 4 | `4_day4_selfgnn_polymer`、`4_4_day4_selfgnn_polymer`、`4_d5mc`、`4_day5_momucon`、`4_day5_momucon_ntxent` | SelfGNN-BYOL SSL + MoMuCon（被 context-limit 阻塞） | SelfGNN 负；MoMuCon 未跑出结果 |
| 5 | `5_day5_gps_lite`、`5_day5_momucon_ntxent`、`5_5_day5_momucon_ntxent`、`5_day5_momucon_finetune` | GPS-Lite 全局注意力 + MoMuCon-NT-Xent 重试 | 均负 |
| 6 | `6_day6_cpa_gat`、`6_day7_cpa_pse` | CPA-GAT 突破 + CPA-PSE 消融 | CPA-GAT 34.77 K（新最佳）；is_backbone 35.03 |
| 7 | `7_day7_cpa_pspe`、`7_day7_polybg_cpa`、`7_7_day7_polybg_cpa`(空) | CPA+PSPE + PolyBG-CPA | 均 null/负 |
| Optimize | `optimize` | ORIG_BB 5 性质协议对齐复跑 | Tg ens_mean 33.65 / single 34.62 |

> 目录前缀数字为权威 day 归属（部分目录名带跨天字样如 `6_day7_cpa_pse` 实属 day6、`4_day5_momucon*` 实属 day4）。

---

## Day 1：数据中心基线 — multimer×特征消融 + GRIN/ELENE 跟进

**参考/启发文献**（Day 1：当日 Learn 读取 7 篇；来源 mem.json provenance）
- 【当日读取】**2019 On Node Features for Graph Neural Networks**（paper） · `mem_073` —— Enables a specific diagnostic protocol: (1) run feature shuffling on the Tg dataset to quantify whether GNN performance relies on monomer…
- 【当日读取】**2023 Improving Subgraph GNNs via Edge Level Ego Network Encodings**（paper） · `mem_074` —— ELENE provides an actionable method for the Tg sandbox experiments: (1) ELENE encodings can be directly computed from polymer graph…
- 【当日读取】**2026 Learning Repetition-Invariant Representations for Polymer Informatics**（paper） · `mem_075` —— (1) Provides proven methodology for generating multimer data from monomer p-SMILES with theoretical backing that 3-RU augmentation is…
- 【当日读取】**2026 Learning Repetition-Invariant Representations for Polymer Informatics 1**（paper） · `mem_076` —— GRIN provides several actionable insights for TopicState: (1) Confirms that modeling polymers with a single RU is fundamentally…
- 【当日读取】**2023 Improving Subgraph GNNs via Edge Level Ego Network Encodings 1**（paper） · `mem_077` · [code](https://github.com/nur-ag/ELENE) —— ELENE provides an actionable structural encoding framework that can be integrated into existing GNN architectures without architectural…
- 【当日读取】**2025 Periodicity-aware deep learning for polymers 1**（paper） · `mem_078` · [code](https://github.com/wuyuhui-zju/PerioGT) —— Sets the SOTA benchmark (Tg RMSE=30.8) to surpass using novel architectures. Provides actionable design principles: (1) periodicity-aware…
- 【当日读取】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_079` · [code](https://github.com/wuyuhui-zju/PerioGT) —— PerioGT provides the SOTA Tg baseline (30.8 RMSE) that any new model must surpass. Its periodicity-aware contrastive learning mechanism is…

### 实验目的
Day 1 是整个 run 的起点，目标是用一个固定 GAT backbone 做全面的数据中心消融，确立可传承基线。**（A）multimer × feature 全交叉消融**：沿多聚体扩张（RU=重复单元；1-RU/2-RU/3-RU/{1,3}-RU mixed 四档）× 节点/边特征丰富度（basic/standard/rich 三档）共 12 格，检验三条假设——H1 多聚体扩张单调有益（GRIN 式假设 mixed>3-RU>2-RU>1-RU）、H2 特征越丰富越好、H3 存在联合最优配置。**（B）GRIN + ELENE 跟进**：针对（A）中 {1,3}-RU mixed 反而劣于纯 3-RU 的现象，固定到 mixed+rich，沿 readout（mean/max）/ L1 稀疏 / ELENE 边编码三轴设计 5 个条件，试图挽救 mixed 的劣势，验证 GRIN 式（max+L1）稀疏、ELENE（Edge-Level Neighborhood Encoding，边级深度-1 ego-network 度三元组）能否提升 Tg。

### 模型架构
**（A）GATModel**：3 层 `GATConv`（hidden=128，heads=4，concat，每头 32 维），每层 ELU + Dropout 0.1（无 BatchNorm），`global_mean_pool` 读出，Linear head 输出标量；`edge_dim` 依特征档动态传入（basic/standard=0 不传 edge_attr，rich=10）。**（B）GATModelBase**：同样 3 层 GAT，但读出可在 mean/max 间切换，conv3 全部参数可被 L1 正则（GRIN 式稀疏，`get_l1_penalty()` 仅对 conv3 求绝对值和）；ELENE 在图构造阶段生成并拼到 edge_attr——对每条有向边 (u,v) 以 v 为根做深度-1 BFS，统计 u 的邻居中到 v 距离为 du−1/du/du+1 的数量并除以 deg(u)，得 3 维度三元组；rich 的 10 维边特征 + ELENE 的 3 维 = 条件 D/E 的 edge_dim=13。条件映射：A=mean、B=max、C=max+L1、D=mean+ELENE、E=max+L1+ELENE。

### 训练评估方式
两子实验共用：数据 `alldataset/Tg.csv`（7170 条 p-SMILES→Tg），72/8/20 划分（train≈5162 / val 573 / test≈1434）；Adam lr=1e-3、weight_decay=1e-5；max 300 epoch、early-stop patience=30；MSE loss；CUDA AMP 混合精度 + 梯度裁剪 1.0；目标值 `StandardScaler` 仅在 train 上 fit、评估时反变换回原温标。3 seeds=[71,72,73]，按 val RMSE 保存最优 state_dict 在 test 上报。（A）batch 64，`build_datasets` 默认 seed=71 故三 seed 的**划分完全相同**、仅模型初始化变；（B）batch 32，`build_datasets` 显式传 seed 故**划分随 seed 变化**（与 A 的方法学差异）。注：`experiment.py` 中硬编码字符串 `"Test: 1-RU graphs"` 实际无代码路径支持——`build_datasets` 用 `{1,3}-RU mixed` 统一构造 train/val/test，无 1-RU 测试覆盖，该自述属 AI 当夜描述、代码不支持。

### 结果
**（A）** 计划 12×3=36 runs，实际完成 **35**（1 个中途被中断：`{1,3}-RU mixed × rich × seed 73`，该格统计实际仅基于 2 个 seed）。Test RMSE（K，行=multimer、列=feature）：1-RU basic 69.2±0.3 / standard 46.6±0.4 / rich 44.1±1.1；2-RU 69.4±1.1 / 43.9±0.8 / 42.0±0.7；3-RU 68.6±1.0 / 44.3±1.4 / **rich 41.6±0.9（最优）**；{1,3}-RU mixed 69.0±1.8 / 45.8±0.3 / 43.6±0.7(n=2)。列均值 basic 69.0 / standard 45.2 / rich 42.8。最优配置 **3-RU×rich：RMSE 41.61±0.90 K，R²0.8573，MAE 28.75 K**。特征维度由 `data_utils._FEATURE_DIMS` 确认：basic=100（原子序数 one-hot）、edge=0；standard=43、edge=0；rich=47（standard+环成员 4 维）、edge=10（键型 4 + 共轭 1 + 环键 1 + 立体 4）。特征主导：mixed 上 basic→standard 改善 +23.2 K、standard→rich 再 +2.2 K；多聚体扩张（rich 下）1-RU 44.1→2-RU 42.0→3-RU 41.6 共 ~2.5 K，但 {1,3}-RU mixed（43.6）**反而劣于纯 3-RU**，GRIN 式 mixed 优势假设仅部分成立（H1 PARTIALLY SUPPORTED，H2 STRONGLY SUPPORTED）。`analysis_report` 引用"GAT baseline RMSE=50.5 K (from literature)"——该来源为文献、代码与结果文件未独立验证（AI 当夜自述，代码未独立验证）。**（B）** 计划 5×3=15 runs，**全部完成**。条件均值（Test RMSE K）：A `GAT+mean` **44.57±1.73（最优，R²0.8392）**、B `GAT+max` 46.74±2.46、C `GAT+max+L1` 46.83±1.82、D `GAT+mean+ELENE` 45.70±0.78、E `GAT+max+L1+ELENE` 46.97±0.56；5 个假设检验全部判 NO——max、L1、ELENE 任一单独或组合都不如最朴素的 GAT+mean，且最优 A（44.57）比纯 3-RU×rich 基线（41.6）还高 +2.97 K。

### 流程
Day 1 由 mem_080、mem_081 两条 experience 驱动，构成"先广扫、后聚焦修补"的闭环。**（A）** `experiment.py` 按 multimer→feature→seed 三层循环，每 run 训练后增量写 `results_summary.csv`，全部完成后 `generate_analysis()` 产出聚合表、热图、学习曲线、报告；35/36 完成度因 analysis 提前生成而中断了最后 1 run。**（B）** 针对 mixed 劣势窄而深复跑：固定 multimer=mixed、feature=rich，沿 readout×L1×ELENE 设计 5 条件×3 seed 全部完成，证实架构修补无收益、反证"数据配方比架构花式更重要"。两子实验共同确立 Day 1 可传承基线：**rich 特征 + 纯 3-RU multimer + 3 层 GAT mean-readout，41.6 K**。遗留方法学注记：(a) 35/36 完成度；(b)（B）summary 自报"1-RU 测试集"在代码中不存在、实际测试仍为 mixed 图。

---

## Day 2：SPSE-MaxGAT 惨败 + static RWPE 新最佳 + 架构 scaling 破天花板

**参考/启发文献**（Day 2：当日 Learn 读取 4 篇＋当夜 integration 显式回溯引用 7 篇；来源 mem.json provenance）
- 【当日读取】**arXiv:2502.09365v2**（arXiv） · `mem_087` —— SPSE provides an actionable encoding method for the Day 2 'PerioGT-inspired' architectural direction: (1) It can be integrated as a drop-in…
- 【当日读取】**2022 Recipe for a General, Powerful, Scalable Graph Transformer**（paper） · `mem_088` · [code](https://github.com/rampasek/GraphGPS) —— GPS provides three actionable contributions for polymer deep learning: (1) A systematic encoding taxonomy (local/global/relative PE/SE)…
- 【当日读取】**2021 LSPE Learnable Structural and Positional Encodings**（paper） · `mem_089` · [code](https://github.com/vijaydwivedi75/lspe) —— Provides a validated architectural blueprint for learnable positional encoding in GNNs that (1) demonstrates the superiority of decoupled…
- 【当日读取】**2024 LONG-SHORT-RANGE MESSAGE-PASSING A PHYSICS-INFORMED FRAMEWORK TO CAPTURE NON-LOCAL INTERACTION**（paper） · `mem_090` · [code](https://github.com/liyy2/LSR-MP) —— LSR-MP provides three actionable contributions to the TopicState: (1) Architectural template — the three-module design (short-range local…
- 【当夜引用】**2019 On Node Features for Graph Neural Networks**（paper） · `mem_073` —— Enables a specific diagnostic protocol: (1) run feature shuffling on the Tg dataset to quantify whether GNN performance relies on monomer…
- 【当夜引用】**2023 Improving Subgraph GNNs via Edge Level Ego Network Encodings**（paper） · `mem_074` —— ELENE provides an actionable method for the Tg sandbox experiments: (1) ELENE encodings can be directly computed from polymer graph…
- 【当夜引用】**2026 Learning Repetition-Invariant Representations for Polymer Informatics**（paper） · `mem_075` —— (1) Provides proven methodology for generating multimer data from monomer p-SMILES with theoretical backing that 3-RU augmentation is…
- 【当夜引用】**2026 Learning Repetition-Invariant Representations for Polymer Informatics 1**（paper） · `mem_076` —— GRIN provides several actionable insights for TopicState: (1) Confirms that modeling polymers with a single RU is fundamentally…
- 【当夜引用】**2023 Improving Subgraph GNNs via Edge Level Ego Network Encodings 1**（paper） · `mem_077` · [code](https://github.com/nur-ag/ELENE) —— ELENE provides an actionable structural encoding framework that can be integrated into existing GNN architectures without architectural…
- 【当夜引用】**2025 Periodicity-aware deep learning for polymers 1**（paper） · `mem_078` · [code](https://github.com/wuyuhui-zju/PerioGT) —— Sets the SOTA benchmark (Tg RMSE=30.8) to surpass using novel architectures. Provides actionable design principles: (1) periodicity-aware…
- 【当夜引用】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_079` · [code](https://github.com/wuyuhui-zju/PerioGT) —— PerioGT provides the SOTA Tg baseline (30.8 RMSE) that any new model must surpass. Its periodicity-aware contrastive learning mechanism is…

### 实验目的
Day 2 在 Day 1 GAT+Rich+3-RU 基线（41.6 K）之上沿两条独立路线突破，并诊断为何聚合物图上某些结构/位置编码失效。**（A）SPSE-MaxGAT**（目录 `2_day2_spse_maxgat`）：验证"边级简单路径计数结构编码（SPSE）"与"GAT 中按维度取 max 的邻居聚合（MaxGAT）"能否提升 Tg，预期为正。**（B）LSPE/RWPE-GAT**（目录 `2_day_2_lspe_gat`）：验证静态预计算随机游走位置编码（static RWPE）、可学习解耦位置编码（learnable LSPE / LSPE-GAT）、边门控 LSPE（LSPE-GAT-edge）、逐层 RWPE 注入（SkipRWPE）能否改进基线；关键在于分辨正收益究竟来自静态 RWPE 还是可学习 LSPE。同时验证"架构 scaling"（Day1 3层/128hidden → 4层/256hidden + 更深 MLP + AdamW）单独能带来多少提升。

### 模型架构
**（A）SPSE-MaxGAT** 全部基于同一骨干：4 层 GATConv（hidden=256，4 heads，concat），输入 47 维富节点特征 + 10 维富边特征，3-RU multimer，全局均值池化，MLP 256→128→1（baseline 256,001 参数）。Treatment A 在边上加 SPSE 编码（对每条边统计长度 k=1..4 的简单路径数，经可学习 log-变换 f_k(x)=α_k·ln(1+x)+β_k 输出 4 维边特征）；Treatment B 在 A 上把 GAT 邻居聚合由 mean 换成 element-wise max（MaxGATConv）；Treatment B v2 在 MaxGATConv 内加带可学习门控的残差连接（ResidualMaxGATConv，272,133 参数）。**（B）LSPE-GAT** 骨干同为 4 层 GAT、hidden=256、heads=4：`GATBaseline`（无 PE，256,001 参数）；`GATStaticRWPE`（预计算 RWPE 直接拼到节点输入并冻结，参数随 k 线性增加 256,001+256·k）；`LSPEGAT`（解耦 h/p 更新，RWPE k=16 仅作初始化后随层学习，776,001 参数）；`LSPEGATEDGE`（attention 派生边权门控位置更新，784,369 参数）；`SkipRWPE`（每层重新注入 RWPE，344k–365k 参数）。须区分：**static RWPE = 预计算后冻结**（拼到输入不再改变）；**learnable LSPE = RWPE 仅作初值、位置向量在网络中随层更新**（即"可学习"）。

### 训练评估方式
两子实验共用：Tg.csv 7170，72/8/20 切分（train 5162/val 573/test 1435，切分 seed=71 固定）；MSE loss；AdamW（lr=5e-4，weight_decay=1e-4）；batch=64；max 300 epoch；AMP 混合精度。差别：**（A）** 用 ReduceLROnPlateau（factor 0.5, patience 10）、early stop patience=30，**仅单 seed=71**（RESULTS.md 明言 "All seeds set to 71"）。**（B）** 用 CosineAnnealingLR（T_max=300）、early stop patience=50，每 treatment 跑 **3 个模型初始化 seed={42,123,456}**（数据切分仍 seed=71），报 mean±std。

### 结果
**（A）SPSE-MaxGAT（单 seed=71）**：baseline **36.85 K**（MAE 24.66，R²0.8881）、Treatment A(+SPSE) **37.99 K**(+1.14)、Treatment B(+SPSE+MaxAggr) **38.27 K**(+1.42)、Treatment B v2(残差 Max) **38.96 K**(+2.10)，三者全部劣于基线。失败根因由代码量化：(1) SPSE 在 3-RU 聚合物图上严重退化——k≥2 的简单路径计数几乎全零（k=2: 100% zero，k=3: 99.9%，k=4: 91.2%），骨架本质是带稀疏分支的线性链；(2) Max 聚合造成方差崩塌——按层 Max/Mean 方差比从 conv1 的 31% 掉到 conv4 的 **0.4%**（winner-take-all）；(3) 残差门控学到 ≈57% 走残差、43% 走 max，模型宁愿保留节点自身。**（B）LSPE/RWPE-GAT（3-seed 均值）**：baseline(无 PE) **38.51±1.09 K**；static RWPE k=16 **36.29±1.04**、k=24 37.07±0.48、**k=32 = 35.94±0.64（全夜新最佳，R²0.8935，MAE 23.67，264,193 参数）**；learnable LSPE-GAT(k=16 初始化) 37.96±0.87（776,001 参数）、LSPE-GAT-edge 37.85±0.52（784,369 参数）；SkipRWPE k=16/32 = 38.13/38.26。**关键分歧由代码裁决**：mem_094 称"static RWPE k=32 = 35.94K"**正确**；旧 summary 称"static RWPE k=32 = 36.29K"**错误**——36.29 实为 **k=16**（`run_all_iter2.py` 明文 `static_rwpe_k16: 36.29`，且 n_params=260,097=256,001+256·16 自证 k=16）。**方向结论（代码核实）**：三个 static RWPE（k=16/24/32）全部优于无 PE 基线且 k 越大越优；可学习 LSPE-GAT/LSPE-GAT-edge 虽略好于基线但劣于所有 static RWPE 变体且参数暴涨 3 倍，即"可学习"相对"静态"为负；SkipRWPE 逐层注入同样无收益。**架构 scaling**：Day1（3层/128hidden）41.6 K → Day2 baseline（4层/256hidden/2层MLP/AdamW/ReduceLR）36.85 K（−4.75 K，单 seed=71），但需注意此 36.85 是单 seed 幸运抽样，同架构在 LSPE-GAT 目录的 3-seed 诚实基线为 **38.51 K**，故"Day1 天花板被破"方向成立、量级被单 seed 高估约 1.7 K。

### 流程
本夜两条并行支线。**（A）** 先实现 baseline→Treatment A（+SPSE，`spse_encoding.py` 含 4 个路径计数函数 + 可学习 log-变换）→Treatment B（换 max 聚合 `MaxGATConv`）→发现 B 崩塌后追加 Treatment B v2（门控残差）；每跑完即用 `diagnostics.py` 量化 SPSE 路径计数分布与 Max/Mean 方差比，定位"聚合物树退化 + max 方差崩塌"。**（B）** 分两轮：iter1（`run_all.py`）跑 baseline/static_rwpe(即k=16)/lspe_gat/lspe_gat_edge 各 3 seed 发现 static 正、learnable 负；iter2（`run_all_iter2.py`）先由 `rwpe_precompute.py` 预计算 k=16/24/32 再扫 k 与注入方式，确认 k=32 static=35.94 K 新最佳。两支线合计 **28 个 run 全部成功、0 失败**：(A) 4/4；(B) iter1 12/12 + iter2 12/12。

---

## Day 3：GraphMAE 自监督预训练 + PSPE 聚合物位置编码

**参考/启发文献**（Day 3：当日 Learn 读取 3 篇＋当夜 integration 显式回溯引用 6 篇；来源 mem.json provenance）
- 【当日读取】**SelfGNN Contrastive SSL GNN without negative sampling**（paper） · `mem_101` · [code](https://github.com/zekarias-tilahun/SelfGNN) —— Actionable contributions for polymer SSL: (1) The Siamese student-teacher + EMA + BatchNorm architecture provides a concrete, simple SSL…
- 【当日读取】**MoCL Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning**（paper） · `mem_102` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides three actionable design principles for polymer pretraining experiments: (1) Domain-knowledge-guided augmentation: Replace generic…
- 【当日读取】**2022 KPGT Knowledge-Guided Pre-training of Graph Transformer for Molecular Property Prediction**（paper） · `mem_104` · [code](https://github.com/lihan97/KPGT) —— KPGT provides actionable evidence that: (1) A line graph representation combined with path/distance structural encodings is a strong…
- 【当夜引用】**full**（paper） · `mem_039` —— TransPolymer is the first Transformer-based language model for polymer property prediction, featuring a chemically-aware tokenizer and pretraining on 5M unlabeled polymer
- 【当夜引用】**full**（paper） · `mem_052` —— This paper introduces a periodic polymer graph representation that explicitly encodes polymer chain periodicity and combines it with a message-passing neural network to a
- 【当夜引用】**2025 Periodicity-aware deep learning for polymers 1**（paper） · `mem_078` · [code](https://github.com/wuyuhui-zju/PerioGT) —— Sets the SOTA benchmark (Tg RMSE=30.8) to surpass using novel architectures. Provides actionable design principles: (1) periodicity-aware…
- 【当夜引用】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_079` · [code](https://github.com/wuyuhui-zju/PerioGT) —— PerioGT provides the SOTA Tg baseline (30.8 RMSE) that any new model must surpass. Its periodicity-aware contrastive learning mechanism is…
- 【当夜引用】**arXiv:2502.09365v2**（arXiv） · `mem_087` —— SPSE provides an actionable encoding method for the Day 2 'PerioGT-inspired' architectural direction: (1) It can be integrated as a drop-in…
- 【当夜引用】**2021 LSPE Learnable Structural and Positional Encodings**（paper） · `mem_089` · [code](https://github.com/vijaydwivedi75/lspe) —— Provides a validated architectural blueprint for learnable positional encoding in GNNs that (1) demonstrates the superiority of decoupled…

### 实验目的
Day 3 探索两条互补的自监督/位置信号增强路线，目标把 Tg RMSE 推到 Day 2 基线（static RWPE k=32，35.94 K）以下。**（A）GraphMAE 风格掩码自监督预训练**：在 PI1M（约 100 万聚合物 SMILES 数据集，取 50K 子集）上以掩码节点特征重建方式预训练编码器，再迁移到 Tg 微调，检验"大数据预训练→小数据微调"是否带来正迁移。**（B）PSPE（Polymer-Specific Positional Encoding，聚合物专属位置编码）**：在节点特征里显式加入一维链几何位置信号，检验其相对 RWPE 基线能否提供正交增益。PSPE 两变体：PSPE-dce（distance-to-chain-end，到最近链端的归一化距离，0=链端、1=链中）与 PSPE-mi（monomer-index，单体沿主链的索引，主链三等分映射到 {0.0,0.5,1.0}）。由于第一轮 PSPE 架构偏离 Day 2，第二轮以 Day 2 同款配置重做，构成两次独立验证。

### 模型架构
**（A）GraphMAE**（`train.py`）：编码器 4 层 GATConv（heads=4，hidden=256，残差投影+BatchNorm+ReLU+Dropout 0.1），末层 concat=False（mean 聚合）输出 256 维；解码器 MLP 256→128→13，**重建目标仅为 13 维 rich 化学特征**（不重建 RWPE 部分）；节点输入 45 维=13 rich+32 RWPE；图构造为 3-RU multimer；总参数 ≈3.57M；Tg 头 `global_mean_pool→MLP(256→128→1)`。**（B）PSPE**（`model.py`/`pspe_data.py`）：第一轮用 GATv2Conv（4 层，hidden=256，heads=4）+ReLU+ReduceLROnPlateau+3 层 MLP 头+RWPE z-score 归一化+wd 1e-5+max 200/patience 30；第二轮（Day 2 同款）改用 GATConv(v1)+ELU+CosineAnnealingLR(T_max=300)+2 层 MLP 头+RWPE 不归一化+wd 1e-4+max 300/patience 50。两轮均跑 4 condition：`baseline`(仅 RWPE k=32)/`pspe_dce`(+dce)/`pspe_mi`(+mi)/`pspe_combined`(+dce+mi)，PSPE 由 double-BFS（双向广度优先搜索）在重原子图上找最长主链后计算。

### 训练评估方式
**（A）** 预训练：PI1M 50K 子集→48,558 张 3-RU multimer 图，掩码率 mask_ratio=0.25（随机选 25% 节点把其 13 维 rich 特征置零作输入、重建这 13 维作 MSE 损失），50 epoch，batch=64，AdamW lr=1e-3 wd=1e-5（预训练损失 0.01348→0.00412，降幅 69.4%）。微调/基线：Tg 7170 张 3-RU 图，split 72/8/20（数据 seed=71），max 300 epoch/patience 30，AdamW lr=3e-4 wd=1e-5，batch=32；微调从预训练 checkpoint 加载编码器，基线同架构随机初始化。**单 seed**（pretrain/baseline/数据均 seed 固定）。**（B）** 两轮均 **model seed=42、数据 seed=71**（单 seed），4 condition 顺序训练单次评估；第二轮加 AMP 混合精度。

### 结果
**（A）GraphMAE — 方向正确但未达阈值、绝对基线偏高**：预训练微调 **39.69 K**（MAE 25.82，R²0.8703）vs 随机初始化基线 40.27 K（MAE 27.83，R²0.8665），**Δ=−0.58 K（−1.4%）**，方向正确但低于预设 1.5 K 改进阈值，判定**未达标**（旧 summary 标"负"即此义）；绝对值远高于 Day 2 目标 35.94 K（基线偏离 +4.33 K），根因被诊断为编码器用了 GATConv 而 Day 2 用 GINEConv/TransformerConv/GPSConv 在分子图上更强；过拟合警告：预训练模型 val-test gap 4.51 K（基线仅 1.81 K）。mem_105 自述"+0.34–0.58 K"中的 0.34 K 下界代码未独立验证（AI 当夜自述，代码未独立验证）。**（B）PSPE — 正向（两轮独立验证，8 condition 全完成）**：第一轮（`3_day3_pspe`）baseline 38.68 / dce **36.43（最优）** / mi 36.65 / combined 36.66（Δ best −2.25 K，dce 最佳）；第二轮（`3_3_day3_pspe`）baseline 36.98 / dce 36.19 / mi 35.99 / **combined 35.44（最优，且唯一压破 Day 2 目标 35.94 K）**（Δ best −1.54 K，combined 最佳）。诊断：PSPE 与 RWPE 最大 |Pearson r| 仅 0.34/0.22，信号基本正交；mean dce/mi 与 Tg 显著负相关（r=−0.265/−0.280）符合"链端自由体积更高→Tg 更高"的物理直觉。**旧 summary 的数字混跨已厘清**："dce 36.43 / mi 36.65"来自**第一轮**（该轮 combined=36.66 非最佳），"combined 35.44"来自**第二轮**（同轮 dce=36.19、mi=35.99），须分开引用。单 seed 结果，report.md 明确建议跑 3+ seed 给置信区间。

### 流程
**（A）** 从 `alldataset/PI1M.csv` 取 50K 子集→3-RU multimer 化→48,558 张图（带缓存 `pi1m_50k_graphs_3ru.pkl`）；50 epoch 掩码预训练（mask_ratio=0.25）保存 `best_pretrained_encoder.pt`；在 Tg 上做预训练初始化微调 vs 随机基线两组对照各 1 run；`write_diagnostic.py` 末尾在比较 LongTensor 时抛 `linalg.vector_norm` 异常，但结果文件已全部落盘不影响结论。**（B）** 两轮流程一致：加载 Tg 3-RU + Day 2 预算 RWPE k=32 pickle→double-BFS 检测主链算 dce/mi→`run_experiments.py` 顺序跑 4 condition→`diagnostics.py` 生成相关/误差分析→写 report.md。第二轮仅改模型/训练超参，数据 pipeline、PSPE 计算、split 完全不变，故构成对 PSPE 假设的独立复核。**两轮 8 个 condition 全部 SUCCESS，0 失败**。

---

## Day 4：SelfGNN-BYOL 图级 SSL（负）+ MoMuCon 被 context-limit 系统性阻塞

**参考/启发文献**（Day 4：当日 Learn 读取 4 篇＋当夜 integration 显式回溯引用 3 篇；来源 mem.json provenance）
- 【当日读取】**2026 MolCHG Multi-level Self-supervised Pretraining Compositional Hierarchical Graph**（paper） · `mem_115` · [code](https://github.com/lhb0189/MolCHG) —— MolCHG provides three actionable insights for polymer Tg pretraining: (1) The bond graph as an independent structural layer is a novel…
- 【当日读取】**2024 A Mamba-Based Foundation Model for Chemistry**（paper） · `mem_116` —— Provides three actionable contributions to the TopicState: (1) Architectural evidence: Mamba/SSM is a viable alternative to both GAT and…
- 【当日读取】**2025 Advancing molecular machine learning representations with stereoelectronics-infused molecular g**（paper） · `mem_117` · [code](https://github.com/gomesgroup/simg) —— Provides a concrete, validated molecular representation methodology (SIMG/SIMG*) that bridges quantum chemistry and graph-based ML. Enables…
- 【当日读取】**2025 C-FREE Contrast-Free Representation Learning Ego-nets**（paper） · `mem_118` · [code](https://github.com/ariguiba/C-FREE) —— (1) Validates the JEPA-family predictive pretraining paradigm as SOTA-competitive for molecular graphs, directly supporting the…
- 【当夜引用】**SelfGNN Contrastive SSL GNN without negative sampling**（paper） · `mem_101` · [code](https://github.com/zekarias-tilahun/SelfGNN) —— Actionable contributions for polymer SSL: (1) The Siamese student-teacher + EMA + BatchNorm architecture provides a concrete, simple SSL…
- 【当夜引用】**MoCL Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning**（paper） · `mem_102` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides three actionable design principles for polymer pretraining experiments: (1) Domain-knowledge-guided augmentation: Replace generic…
- 【当夜引用】**2022 KPGT Knowledge-Guided Pre-training of Graph Transformer for Molecular Property Prediction**（paper） · `mem_104` · [code](https://github.com/lihan97/KPGT) —— KPGT provides actionable evidence that: (1) A line graph representation combined with path/distance structural encodings is a strong…

### 实验目的
Day 4 推进两条线。**（A）SelfGNN-Polymer**（图级自监督学习 SSL，已完成、负结果）：Day 3 已证**节点级**重建（GraphMAE）只给 +0.58 K，不符合"图级下游任务须用图级预训练目标"的对齐原则；Day 4 引入图级目标——对同一分子的两个增广视图在图级 embedding 上做 BYOL/Siamese 一致性约束，期望优于节点级 SSL。预训练在 PI1M（20 万 1-RU 单重复单元）上做，下游在 Tg（3-RU）上做。**（B）MoMuCon-NT-Xent**（对比学习预训练）：针对（A）暴露的"节点级 SSL 死胡同 + BYOL 在小聚合物图上塌陷"两个失败模式，设计"单体-多体对比"预训练——把同一聚合物的 1-RU 与 3-RU 视图作正样本对，用带显式负样本的 NT-Xent（Normalized Temperature-scaled Cross Entropy）损失替代 BYOL 的隐式塌陷抑制。

### 模型架构
**（A）SelfGNN（BYOL-style Siamese GAT）**：GATEncoder=Linear(42→256)+4×[GATConv(256,64,heads=4,concat)+BatchNorm1d+ELU+Dropout]；StudentModel=encoder+predictor MLP 256→128→256；TeacherModel=同结构但参数 `requires_grad=False` 靠 EMA（τ=0.999）更新（代码中 `update_ema` 只同步 parameters、不同步 BN buffer，致 teacher BN 统计量发散）；FinetuneModel=encoder+Set2Set(256,processing_steps=3)→512+MLP 512→256→128→1。输入 42 维=8 维丰富原子特征（电负性、共价半径、价电子、族、周期、电子亲和能、电离能、质量）+2 维 PSPE+32 维 RWPE。损失 `cosine_similarity_loss=mean(2−2·cos(z_student,z_teacher))`。**（B）MoMuCon**：GATEncoder=Linear(in→256)+4 层 GAT(heads=4)+LayerNorm+Dropout 0.1；ProjectionHead=2 层 MLP 256→128→128（L2 归一化）；**MoMuConSiamese 共享权重**（无 EMA、无 momentum encoder、无 BYOL 不对称），对 (1-RU,3-RU) 两视图共享同一 encoder+projection；`nt_xent_loss(z1,z2,τ=0.1)`=标准 InfoNCE，正样本对 (i,i+B) 其余为负，logsumexp 归一化。

### 训练评估方式
**（A）** 预训练 PI1M 200,000 个 1-RU 图（实际 199,998），AdamW lr=3e-4 wd=1e-4，CosineAnnealingLR→1e-6，batch=64，max 200 epoch/patience 30，grad clip 1.0，EMA τ=0.999；三种增广 `augment_split`/`augment_standardize`/`augment_combined`。微调 Tg 7,145 个 3-RU 图（5144/571/1430），3 seed={71,42,123}，目标 StandardScaler，warmup+cosine，200 epoch/patience 50。条件：A 无预训练 baseline、B 仅 Split 预训练、C 仅 Standardize 预训练、D Split+Std 联合（PRIMARY 主实验）、G 冻结 encoder、H No-BN 预训练（消融，仅 50 epoch）。每 10 epoch 算 graph-level cos_mean/cos_std/collapse_warning（=cos_std<0.01 或 cos_mean>0.99）。**（B）** 预期预训练 PI1M 50K 对（1-RU,3-RU），batch=128，100 epoch，warmup 10，τ=0.1；微调三模式：非预训练 baseline、Linear Probe（冻结 encoder+线性回归）、Full Finetune。**实际：三个 MoMuCon 目录的 `results/` 均为空，无任何 diagnostics/config/loss 记录文件，微调 RMSE 从未产出。**

### 结果
**（A）SelfGNN — 负（预训练无统计显著迁移收益，baseline 未复现 35.44 目标）。** 预训练塌陷诊断：D PRIMARY（Split+Std）ep10 曾塌陷 cos=0.999、ep40 部分恢复到 0.223；B Split-only ep10 塌陷后稳定高 cos 0.754；C Standardize-only 完全退化（teacher norm 12624）跳过微调；**H No-BN cos_mean=0.9997 持续塌陷→证明 BatchNorm 必要**。微调 Test RMSE（v2，3 seed）：A baseline **40.55±0.75**、D PRIMARY **40.00±0.88（Δ=−0.55，约 −1.4%，不显著）**、B Split-only 41.24±0.66（+0.69 更差）、G Frozen 40.40±0.66（−0.15）、C 跳过。baseline 40.55 距目标 35.44 差 +5.11 K（未复现），归因：富特征太少（8 vs Day2 的 47）、Set2Set readout 更难优化、epoch 少（200 vs 300）。**（B）MoMuCon-NT-Xent — 无结果。** mem_123 明列 6 次 `run_experiment` 调用全部超 token 上限（1.10M~8.95M tokens，限额 1,048,576）。根本原因：CLAUDE.md 累积 Day1–4 全部历史涨到约 9M tokens，被 `run_experiment` 工具加载为 system prompt，导致任何新任务无法启动。`4_d5mc` 留下 1.28 GB 缓存（`pi1m_50k_pairs_42dim.pkl`）+180 MB（`tg_graphs_3ru.pkl`）+两个孤立 `.pt` checkpoint，但无过程文件、微调 RMSE 从未产出——Day 4 内**未能完成**本实验。

### 流程
**（A）SelfGNN 走完全流程**（`run_all.py` 串行：数据准备→四种预训练→五条件三种子微调，约 15 小时）。期间修复 4 个代码 bug：PyTorch 2.6 `weights_only=True` 默认致 `torch.load` 失败（加 `add_safe_globals` 与 `weights_only=False`）；预训练 10 维/微调 42 维不匹配的 split-projection 拼凑→v2 改全程 42 维；`evaluate()` 对原始 target 误做 inverse_transform 双重变换→只对 prediction 做；`compute_embedding_stats` 对带梯度张量 `.numpy()` 报错→加 `.detach().cpu()`。还识别一个未修隐患：EMA 只同步 parameters、不同步 BN buffer，致 teacher 范数爆炸。**（B）MoMuCon 仅完成代码实现与数据缓存准备**，但 `run_experiment` 编排工具被 CLAUDE.md 体积触发的系统性 context-limit 故障阻塞（mem_123/124），6 次调用全部失败、无一次产出结果。该实验后续在 **Day 5 以手动方式才跑出 NEGATIVE 结果**（mem_134），不属 Day 4 范畴。

---

## Day 5：GPS-Lite 全局注意力（负）+ MoMuCon-NT-Xent 对比预训练重试（负）

**参考/启发文献**（Day 5：当日 Learn 读取 4 篇＋当夜 integration 显式回溯引用 7 篇；来源 mem.json provenance）
- 【当日读取】**2025 Transformers Discover Molecular Structure Without Graph Priors**（paper） · `mem_128` —— This paper provides actionable evidence that: (1) Standard Transformers can serve as a viable alternative backbone for molecular tasks…
- 【当日读取】**2026 kMIP GraphGPS**（paper） · `mem_129` —— Enables a concrete, actionable experiment: GPS+k-MIP applied to polymer Tg prediction as the 'fundamentally different architectural class'…
- 【当日读取】**2026 CardinalGraphFormer Cardinality-Preserving Attention for Molecular Property Prediction**（paper） · `mem_130` —— Provides an actionable architectural component (CPA channel) that can be integrated into graph transformer backbones for polymer Tg…
- 【当日读取】**2025 On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs Bridging Recurrent and Graph**（paper） · `mem_131` —— Provides actionable architectural guidance for polymer deep learning: (1) The layer-wise Jacobian eigenanalysis should become a diagnostic…
- 【当夜引用】**SelfGNN Contrastive SSL GNN without negative sampling**（paper） · `mem_101` · [code](https://github.com/zekarias-tilahun/SelfGNN) —— Actionable contributions for polymer SSL: (1) The Siamese student-teacher + EMA + BatchNorm architecture provides a concrete, simple SSL…
- 【当夜引用】**MoCL Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning**（paper） · `mem_102` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides three actionable design principles for polymer pretraining experiments: (1) Domain-knowledge-guided augmentation: Replace generic…
- 【当夜引用】**2022 KPGT Knowledge-Guided Pre-training of Graph Transformer for Molecular Property Prediction**（paper） · `mem_104` · [code](https://github.com/lihan97/KPGT) —— KPGT provides actionable evidence that: (1) A line graph representation combined with path/distance structural encodings is a strong…
- 【当夜引用】**2026 MolCHG Multi-level Self-supervised Pretraining Compositional Hierarchical Graph**（paper） · `mem_115` · [code](https://github.com/lhb0189/MolCHG) —— MolCHG provides three actionable insights for polymer Tg pretraining: (1) The bond graph as an independent structural layer is a novel…
- 【当夜引用】**2024 A Mamba-Based Foundation Model for Chemistry**（paper） · `mem_116` —— Provides three actionable contributions to the TopicState: (1) Architectural evidence: Mamba/SSM is a viable alternative to both GAT and…
- 【当夜引用】**2025 Advancing molecular machine learning representations with stereoelectronics-infused molecular g**（paper） · `mem_117` · [code](https://github.com/gomesgroup/simg) —— Provides a concrete, validated molecular representation methodology (SIMG/SIMG*) that bridges quantum chemistry and graph-based ML. Enables…
- 【当夜引用】**2025 C-FREE Contrast-Free Representation Learning Ego-nets**（paper） · `mem_118` · [code](https://github.com/ariguiba/C-FREE) —— (1) Validates the JEPA-family predictive pretraining paradigm as SOTA-competitive for molecular graphs, directly supporting the…

### 实验目的
**（A）GPS-Lite**（`5_day5_gps_lite`）：在已确立的 GAT+RWPE+PSPE 基线上，验证在 GAT backbone 顶端再叠加一层全局注意力能否进一步提升 Tg。设三组消融——baseline（不加）、no_bias（加全局自注意力但不带距离偏置）、spd（加带最短路径距离 SPD 偏置的全局自注意力，为主实验），目的就是证伪"更多注意力→更好"，检验 Simplicity Principle（化简原则）。**（B）MoMuCon-NT-Xent**（`5_day5_momucon_ntxent` + `5_5_day5_momucon_ntxent` + `5_day5_momucon_finetune`）：Monomer-vs-Multimer Contrastive 的预训练-微调流程，是 Day 4 同思路被 context-limit 阻塞后的 Day 5 重试，看 NT-Xent 对比预训练所学层级表征能否迁移改善 Tg 回归。

### 模型架构
**（A）GPS-Lite**：共享 GATEncoder=4 层 GATConv（hidden=256，heads=4，末层 concat=False→256，每层 ELU，无残差/LayerNorm）；`GPSLiteBaseline` 仅 GAT backbone；`GPSLiteNoBias`=GAT+`GlobalAttentionSPD`（4 头逐图全局自注意力，关闭 SPD bias）残差加回；`GPSLiteSPD`=GAT+`GlobalAttentionSPD`（开启 SPD=Shortest-Path Distance bias MLP，节点对最短路径距离 clip 到 [0,31] 注入注意力 logits）；Predictor 256→128→1；输入 81 维=rich(47)+RWPE(32)+PSPE-dce(1)+PSPE-mi(1)，3-RU multimer。**（B）MoMuCon**：GATEncoder=4 层 GATConv(hidden=256,heads=4)+BatchNorm1d+ReLU+Dropout 0.1，第 2-3 层残差，global_mean_pool+LayerNorm→256 维图嵌入；两个独立 ProjectionHead（monomer/multimer，各 256→128→128）；finetune 时新加 Predictor 256→128→1。NT-Xent（`nt_xent_loss_with_two_heads`）：z 先 L2 归一化，相似度矩阵 S=(Z·Zᵀ)/τ，τ=0.07，对角掩蔽，cross-entropy 让 monomer[i] 与同样本 multimer[i] 对齐。对比对：同一 PI1M 聚合物的 (1-RU 单体, 3-RU 多体) 两图作正对、batch 内其余作负样本。

### 训练评估方式
**（A）单 seed**：seed=71、split_seed=71，72/8/20 划分（5162/573/1435，3-RU multimer）；AdamW+CosineAnnealingLR+AMP，max 300 epoch/patience 50，batch 64；baseline 先训并缓存，本次实际新训 no_bias 与 spd。**（B）单 seed**：预训练 PI1M 取 50K 聚合物（seed=42 选索引，实际成 48,591 对，train 46,161），AdamW+线性 warmup+cosine decay，epochs=50，lr=1e-4，batch=128，accum_steps=4（有效 micro-batch 32），τ=0.07，总参数 450,560；微调（seed=42）加载 epoch 30 best_pretrain.pt 的 encoder，3-RU Tg 图，72/8/20 split（TG_SPLIT_SEED=71），AdamW lr=1e-4 wd=1e-4，ReduceLROnPlateau(patience=10,factor=0.5)，early stop patience=30，max 300 epoch，batch 64，AMP；test RMSE 还原为 K 与**硬编码**对照 baseline_k=35.44 K（外部引用值，非本次同条件重训 baseline）比较。

### 结果
**（A）GPS-Lite（单 seed）**：baseline（GAT only）**35.19 K**（MAE 23.17，R²0.898，3,318,273 参数）、no_bias（+全局注意力）43.43 K（MAE 30.84，R²0.8446，**+8.24 恶化**）、spd（+SPD-bias 全局注意力）43.14 K（MAE 30.97，R²0.8467，**+7.95 恶化**）。两种全局注意力变体都使 RMSE 上升约 8 K、R²下降约 5 个百分点——**NEGATIVE**，证伪"加全局注意力能提升 Tg"，反向支持 Simplicity Principle。与旧 summary"43.14/43.43,+8K"及 mem_132 完全一致。**（B）MoMuCon-NT-Xent（单 seed）**：预训练 NT-Xent loss 从 epoch 1 的 51.3 单调降到 epoch 30 的 0.4641（best）；对比空间诊断中 pos_sim 稳定在 ~12.8，该值是 S=cos/τ 的元素，反推真实正对余弦相似度 **cos≈12.8×0.07≈0.39**（即 mem_134"cos~0.39 did not collapse"的来源）——既未→1.0（塌陷）也未→0（未学习），说明对比学习有学但没塌陷。finetune **test_rmse_k=38.75 K** vs baseline_k=35.44 K → Δ=−3.31 K（恶化，best_epoch 144）——**NEGATIVE**。与旧 summary"38.75 vs 35.44,−3.31K"及 mem_134 完全一致。两则 AI 自述均被代码独立佐证，无遗留未验证项。附：`utils.py`/`pretrain.py` 的 collapse 阈值 0.95 针对真实 cos 设计但实际喂入 cos/τ，致每 epoch 误报"near collapse"——不影响 finetune 结果。

### 流程
**（A）GPS-Lite**：训 baseline 并缓存→复用缓存仅新训 no_bias、spd→生成诊断图→写 results_summary.json，**完成**（no_bias best ep 65 早停@115，spd best ep 37）。**（B）MoMuCon 三目录关系**：`5_day5_momucon_ntxent/` 首次预训练，`pretrain_output.log` 用 50K pairs 跑到 epoch 37 步 124 **中断**，`pretrain_resume_output.log` 试图用 200K pairs 全新预训练但**卡在数据构建 43%（85424/200000）从未进入训练**；`5_5_day5_momucon_ntxent/` 为重试目录（`diff` 显示脚本与前者完全相同、checkpoint 与 1.4 GB pairs pkl 字节数一致），`pretrain_resume.log` 显示从 epoch 30 续跑到 epoch 38 步 255 **再次中断**未达 epoch 50——即本次预训练"跑通"（loss 正常下降、cos 未塌陷）但 50 epoch 未跑满，finetune 用的是 epoch 30 的 best_pretrain.pt；`5_day5_momucon_finetune/` 通过符号链接复用前者的 models/dataset/utils，加载 best_pretrain.pt 微调 ~150 epoch（early stop）写 finetune_results.json，**完成**。Day 5 两子实验代码与结果齐备、流程闭环，均得出 NEGATIVE 结论。

---

## Day 6：CPA-GAT 突破天花板（34.77 K）+ is_backbone 入选最终模型

**参考/启发文献**（Day 6：当日 Learn 读取 2 篇＋当夜 integration 显式回溯引用 7 篇；来源 mem.json provenance）
- 【当日读取】**2026 CardinalGraphFormer CPA Channels for Graph Transformers**（paper） · `mem_144` —— Enables the Day 6 CPA-GAT supervised experiment: add a sigmoid-gated unnormalized sum channel to GAT message-passing aggregation (~1%…
- 【当日读取】**OPC**（paper） · `mem_145` · [code](https://github.com/sobinalosious/ADEPT) —— Actionable contributions to TopicState: (1) The Tg distribution shift is formally documented—hyperbolic fits produce systematically higher…
- 【当夜引用】**2022 Recipe for a General, Powerful, Scalable Graph Transformer**（paper） · `mem_088` · [code](https://github.com/rampasek/GraphGPS) —— GPS provides three actionable contributions for polymer deep learning: (1) A systematic encoding taxonomy (local/global/relative PE/SE)…
- 【当夜引用】**MoCL Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning**（paper） · `mem_102` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides three actionable design principles for polymer pretraining experiments: (1) Domain-knowledge-guided augmentation: Replace generic…
- 【当夜引用】**2026 MolCHG Multi-level Self-supervised Pretraining Compositional Hierarchical Graph**（paper） · `mem_115` · [code](https://github.com/lhb0189/MolCHG) —— MolCHG provides three actionable insights for polymer Tg pretraining: (1) The bond graph as an independent structural layer is a novel…
- 【当夜引用】**2025 Transformers Discover Molecular Structure Without Graph Priors**（paper） · `mem_128` —— This paper provides actionable evidence that: (1) Standard Transformers can serve as a viable alternative backbone for molecular tasks…
- 【当夜引用】**2026 kMIP GraphGPS**（paper） · `mem_129` —— Enables a concrete, actionable experiment: GPS+k-MIP applied to polymer Tg prediction as the 'fundamentally different architectural class'…
- 【当夜引用】**2026 CardinalGraphFormer Cardinality-Preserving Attention for Molecular Property Prediction**（paper） · `mem_130` —— Provides an actionable architectural component (CPA channel) that can be integrated into graph transformer backbones for polymer Tg…
- 【当夜引用】**2025 On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs Bridging Recurrent and Graph**（paper） · `mem_131` —— Provides actionable architectural guidance for polymer deep learning: (1) The layer-wise Jacobian eigenanalysis should become a diagnostic…

### 实验目的
Day 6 是本 run 的**决定性突破天**，由两个目录组成（`6_day6_cpa_gat`、`6_day7_cpa_pse`，后者名带 day7 实属 day6 夜）。**（A）CPA-GAT 主对比**：前期 day1–day5 的 GNN 系列长期卡在 ~36 K 天花板无法下降。本子实验提出并验证"**基数盲（cardinality-blind）假设**"——标准 GATv2 注意力对邻居用 softmax 归一化（权重和为 1），等价于对邻居**基数（cardinality，即邻居数）**做加权平均，"多少个邻居"这一信号被归一化抹掉，而聚合物图里局部连接度/邻居基数承载物理意义（链段堆积密度、局部交联程度、RU 间连接数）。为恢复基数信息设计 **CPA（Cardinality-Preserving Attention，基数保持注意力）**：在标准 GATv2 通道外并行加一条**未归一化和（unnormalized sum）通道**——对邻居（含自环）value 直接 scatter_sum（不除基数、不过 softmax），使输出量级正比于基数，再用可学习 sigmoid 门控逐元素调制；若 CPA 能打破前期天花板即证伪"基数对 Tg 无用"。设两个对照消融厘清机制：CPA-Last（仅末层加 CPA）、Degree-baseline（把度数作一维特征拼到输入）。**（B）CPA-PSE 消融**：在 CPA-GAT 上加 3 个周期性-结构编码（PSE）特征做消融——modular_position（RU 模板内归一化位置）、periodic_distance（到相邻 RU 结构等价原子的最短路径距离）、is_backbone（是否在主链上），检验进一步注入周期性先验能否再降误差。

### 模型架构
**（A）** 所有模型共享 input_proj(42→256)→4 conv 层→ELU→global_mean_pool→predictor(256→128→ReLU→Dropout 0.1→1)，4 heads、concat=True（每头 64）、add_self_loops=True、dropout=0。`BaselineGAT`=4 层标准 GATv2Conv（572,417 参数）；`CPAGAT`=4 层自定义 `CPAGATv2Conv` **每层**加 CPA：每 head 先得标准 o_std=GATv2Conv(x)（含自环），复用同一 lin_r 投影 v=gat.lin_r(x)，对含自环边做 scatter_sum(reduce='sum') 得 s，每 head 一组门控参数 W_g∈R^(64×256)、b_g，g=σ(W_g·x+b_g)，输出 o_std+g⊙s，4 head concat（835,585 参数）；`CPALastLayerGAT`=层 1-3 标准、层 4 CPA（638,209 参数）；`DegreeBaselineGAT`=4 层标准 GATv2，前向算含自环节点度归一化为 deg/100 与 42 维输入 concat 成 43 维（572,673 参数）。**（B）** CPA-GAT 从（A）原样移植；PSE 三特征由 `compute_pse_features.py` 用 double-BFS 找主链、按 n_ru=3 等分 3 段计算；条件矩阵 5 实跑 + 2 未跑：A=CPA baseline(基础 42 维无 PSE)、B=CPA+全 PSE(45 维)、C=CPA+modular_position(43)、D=CPA+periodic_distance(43)、E=CPA+is_backbone(43)、F(未跑)=去 PSPE+全 PSE、H(未跑)=NonCPAGAT+全 PSE。

### 训练评估方式
两子实验数据同源：day4 缓存的 3-RU 42 维图（`tg_graphs_3ru_42dim.pkl`，节点 42 维=8 rich+2 PSPE[dce,mi]+32 RWPE），目标 Tg 经 StandardScaler 归一化、固定 train/val/test 三划分，batch=64，AdamW lr=1e-3 wd=1e-4，CosineAnnealingLR(T_max=300,eta_min=1e-6)，MSE，grad clip 1.0，max 300 epoch/patience 30，按 val RMSE 选 best、test 报去归一化后 RMSE/MAE(K)。**（A）** 每模型 3 seeds=[42,123,456]，std 用 ddof=0；**（B）** 每 condition 5 seeds=[71,42,123,888,2024]，std 用 ddof=1，评估扩展为 RMSE+MAE+R²，每 condition 首个 seed 额外算 PSE 与基础 42 特征的 Pearson 相关矩阵。

### 结果
**（A）CPA-GAT 主对比（`results/summary.json`，3 seeds）**：**CPAGAT（全层 CPA）34.77±0.21 K**（per-seed 34.59/35.06/34.64，835,585 参数）**← 新最佳**；BaselineGAT 36.38±0.14（572,417）；CPALastLayerGAT（仅末层 CPA）36.47±0.45（638,209）；DegreeBaselineGAT 37.48±0.78（572,673）。**CPA-GAT 决定性正向：34.77 vs baseline 36.38，降 1.61 K（4.4%），打破前期 ~36 K 天花板**——mem_146"34.77 vs 36.38, +1.61K, 4.4%"与代码完全吻合。CPA 仅末层基本无效（36.47 vs 36.38 略差 0.09 K），说明 CPA 必须全层部署；显式给度数有害（37.48 vs 36.38 反升 1.10 K），说明把"基数标量"硬拼到输入不如 CPA 的"和向量+门控"形式。`analyze_results.py` 诊断图（门控饱和度、门控-基数相关性、误差-链长分层）已落盘，定量相关系数数值需读图（AI 当夜自述，代码未独立验证）。**（B）CPA-PSE 消融（`results/results_summary.json`，5 seeds）**：**E CPA+is_backbone 35.03±0.73（最优，R²0.8989）**；A CPA baseline(无 PSE) 35.39±0.57（R²0.8968）；D CPA+periodic_distance 35.83±0.94；B CPA+全 PSE 36.00±0.69；C CPA+modular_position 36.29±0.15。mem_154"CPA-PSE: NEGATIVE"严格针对**全 PSE（条件 B）**：B(36.00) 比 A(35.39) 退 0.61 K，H1（改善≥0.3 K）不支持；**但逐特征看并非全负——is_backbone（条件 E）单独加是净正**：E(35.03) 比 A(35.39) 改善 0.36 K，是 5 condition 中唯一击败 baseline 的变体，物理上最朴素（仅一个 binary 主链标志）、cost 最小。旧 summary"is_backbone 5-seed 35.03±0.73，击败 42 维 CPA baseline(35.39) 与同组 modular_position(36.29)/periodic_distance(35.83)/PSE-full(36.00)"与 `results_summary.json` 精确吻合。**is_backbone 入选最终模型的依据**：条件 E 是 is_backbone 作为单独 PSE 特征的**最早系统 5-seed 验证**，确为唯一击败 CPA baseline 的变体。注：Day7 A baseline seed42=36.13 与 Day6 同模型同 seed 的 34.59 差 1.5 K，仅 seed 123 接近（35.06 vs 34.99），差异可能来自跨 run PyG/CUDA 非确定性，代码未独立解释，跨目录数值对比应谨慎。

### 流程
**（A）** `run_experiments.py` 一次跑 4 模型×3 seed=**12 runs 全部完成、0 killed**（results 下 12 个 .json + 12 个 _history.json + 12 个 .pt 齐备，summary.json 已生成）；`run_cpa_only.py`（3 额外 seed [76,14,71]，lr 1e-4，patience 40，_v2 后缀）**无 _v2 产物文件，判定未完成/killed**（计划 3 runs，0 完成）；`analyze_results.py` 事后分析成功输出 6 张 png。**（B）** `run_batch.py` 分两批：第一批（2026-05-25 23:09 起）跑 A、B（2×5=10 runs，2.9 h）；第二批（2026-05-26 02:05 起）跑 C、D、E（3×5=15 runs，3.7 h）；**共 5 condition×5 seed=25 runs 全部完成、0 killed**；F、H 两 condition（共 10 runs 计划）**未启动**（既无产物也无日志，判定未跑/killed，diagnostics 中 H4 因 H 缺失无法计算）。最终诊断 overall 结论文字"All PSE variants degrade or do not improve"**不够精确**——E(is_backbone) 实际改善 0.36 K 为 5 condition 最优，以 per-condition RMSE 表为准。

---

## Day 7：CPA+PSPE（null）+ PolyBG-CPA（null/崩塌）—— 两条特征工程范式关闭

**参考/启发文献**（Day 7：当日 Learn 读取 1 篇＋当夜 integration 显式回溯引用 5 篇；来源 mem.json provenance）
- 【当日读取】**2025 MGNN Moment Graph Neural Network for Universal Molecular Potentials**（paper） · `mem_157` · [code](https://github.com/JakechiC/MGNN) —— MGNN provides three actionable insights for TopicState: (1) The moment contraction design—encoding pairwise angular information as scalar…
- 【当夜引用】**full**（paper） · `mem_071` —— The paper introduces polyGNN, a multitask graph neural network that learns polymer features directly from repeat unit SMILES strings, achieving 1–2 orders of magnitude fa
- 【当夜引用】**2026 CardinalGraphFormer Cardinality-Preserving Attention for Molecular Property Prediction**（paper） · `mem_130` —— Provides an actionable architectural component (CPA channel) that can be integrated into graph transformer backbones for polymer Tg…
- 【当夜引用】**2025 On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs Bridging Recurrent and Graph**（paper） · `mem_131` —— Provides actionable architectural guidance for polymer deep learning: (1) The layer-wise Jacobian eigenanalysis should become a diagnostic…
- 【当夜引用】**2026 CardinalGraphFormer CPA Channels for Graph Transformers**（paper） · `mem_144` —— Enables the Day 6 CPA-GAT supervised experiment: add a sigmoid-gated unnormalized sum channel to GAT message-passing aggregation (~1%…
- 【当夜引用】**OPC**（paper） · `mem_145` · [code](https://github.com/sobinalosious/ADEPT) —— Actionable contributions to TopicState: (1) The Tg distribution shift is formally documented—hyperbolic fits produce systematically higher…

### 实验目的
Day 7 两个子实验均为验证性负结果，收束 Day 6 CPA-GAT 突破后的机制边界。**（A）CPA+PSPE**（`7_day7_cpa_pspe`）：检验在 CPA-GAT 之上再加 PSPE（链端距离 dce + 单体序号 mi）能否进一步提升，更高层目的是判定 **CPA 的聚合层（per-layer 闭邻域 scatter-sum 求和）是否已 subsume（吞噬）链几何特征**——若 PSPE 无增益则证明 CPA 基数保持聚合已内含这类位置信号。**（B）PolyBG-CPA**（`7_day7_polybg_cpa`，`7_7_day7_polybg_cpa` 为空目录）：PolyBG（Polymer Bond Graph，聚合物键图=原子图的线图，每条键变一个键节点、共享原子的两条键相邻）上的基数保持是否与原子级 CPA 互补——这是"轴层级（axis hierarchy）"问题：键级结构信息是否构成一条独立于原子级、能增益 CPA 的轴。测两种范式：并行编码（两编码器独立、读出时拼接）与层级编码（键上下文在不同深度注入原子流）。预期：若原子级 CPA 聚合层已 subsume 键/基数信息，则键图轴应关闭。

### 模型架构
两子实验共享：Tg 5162/573/1434（72/8/20 split，共 7169 聚合物），3-RU 分子图，42 维原子特征（8 rich+2 PSPE+32 RWPE），目标标准化、报告反标准化回 K，AdamW lr=1e-3 wd=1e-4，CosineAnnealingLR，MSELoss，batch 64，max 300 epoch/patience 30，grad clip 1.0，按 val RMSE 选 best、报 test。**（A）** 三档：Config A `BaselineGAT`=4 层标准 GATv2Conv（40 维输入，无 CPA/PSPE）；Config B `CPAGAT`=4 层 CPAGATv2Conv（40 维，无 PSPE）；Config C `CPAGATwithPSPE`=架构同 B 仅输入 42 维。CPA 层公式同 Day 6：o_std=GATv2Conv(x)，v=lin_r(x)，s_i=scatter_sum(v_j, j∈N(i)∪{i})，g_i=σ(W_g·x_i+b_g)，输出 o_std+g⊙s。PSPE 特征 DCE（沿主链到最近链端归一化距离）+MI（3 个 RU 段索引）共 2 维。**（B）** 键图构建为线图，14 维键节点特征（键类型 one-hot 4 + 共轭 + 成环 + 可旋转 + 立体 + 键级 + 是否主链键 + 端点平均电负性 + 沿主链位置 + 单体序号 + 键长估计），主链通过对两个 `*`（链端 dummy 原子）BFS 得到；原子级 CPA-GAT 与（A）完全一致（复用）。8 个模型变体：Exp1 CPA-GAT 原子 baseline(42d,4L 256d,835K)；Exp2 PolyBGCPA 并行(原子 CPA-GAT 4L256d ‖ 键 CPA-GAT 3L128d，各自 mean pool 拼接 384d→MLP 192→1，1,028K)；Exp3 PolyBGnoCPA(并行但两路普通 GATv2)；Exp4 BondOnlyCPA(仅键 CPA-GAT)；Exp5 EnrichedEdgeCPAGAT(8d 键特征 scatter-mean 聚合到节点拼接)；Exp6 Hierarchical_Pre(键→原子投影拼接成 170d→原子 CPA-GAT)；Exp7 Hierarchical_Mid(原子 conv1 后与键上下文拼接)；Exp8 Hierarchical_PreGated(Exp6+学习门)。

### 训练评估方式
**（A）** 3 seed=[71,72,73]，按 val RMSE 早停选 best、test 报 RMSE/MAE(K)；诊断额外做 PSPE 测试置零消融、按链长分桶 MAE、CPA sum/gate 与邻域基数相关性、逐层 gate 饱和度。**（B）** 3 seed=[71,42,123]，主实验（Exp1/2/6）跑满 3 seed、消融（Exp3/4/5/7/8）只跑 seed 71，按 val RMSE 选 best、test 报 RMSE/MAE；多次运行：`exp1_log.txt`(0 字节首次失败留空)、`exp2_5.log`(迭代1：Exp2 三 seed + Exp3/4 完成，Exp5 在 epoch1 因 shape mismatch 7485×256 vs 512×64 崩溃)、`exp5_only.log`(修 bug 后重跑 Exp5 成功)、`exp_iter2.log`(迭代2：Exp6 三 seed + Exp7/8)、`exp_full.log`(不完整)，合计 **15 次尝试、14 次完成、1 次崩溃重跑**。

### 结果
**（A）CPA+PSPE（3 seed）**：A Baseline GAT 37.45±0.86、**B CPA-GAT（无 PSPE）35.52±0.41（最优，方差最小）**、C CPA-GAT+PSPE 35.86±1.06。CPA 单独相对 baseline 提升 ~1.9 K 且方差更小（0.41 vs 0.86）证实 CPA 有效；**CPA+PSPE=NULL**：35.86 比 CPA 单独 35.52 反而差 0.34 K、在 ±1.06 seed 噪声内——mem_158(35.86 vs 35.52)与代码完全一致。诊断：PSPE 测试置零→RMSE 从 35.44 灾难性退化到 91.19（退化 55.75 K），说明模型**确实在用 PSPE**但即便如此仍不优于"CPA 单独"，即 **CPA 聚合层已 subsume PSPE 所编码的链几何信号**；按链长分桶 PSPE 对短链(<60 节点)伤害最大（MAE −2.35）；CPA sum_norm 与邻域基数相关 r=0.31–0.51（逐层递增），gate_norm 相关从 layer1 的 0.30 衰减到 layer4 的 −0.08（深层 gate 与基数脱耦/饱和）。**（B）PolyBG-CPA**：Exp1 CPA-GAT 原子 baseline 35.80±0.44、**Exp2 PolyBG-CPA 并行 35.64±0.27（NULL，vs 35.80 仅好 0.16 K，远小于 seed std）**——mem_159"35.6K 并行 null"与代码完全一致，键图基数保持**不**互补原子级 CPA；消融佐证 CPA 是有效成分：Exp3 去 CPA 损 1.35 K、Exp4 仅键图掉 9.72 K、Exp5 键特征塞原子边掉 4.81 K；**层级崩塌**：Exp6（前置）与 Exp8（前置+门控）三 seed 全崩到 **110.25 K**（即数据集均值预测器，val RMSE 钉死 110.45，best_epoch 4–5），Exp7（中置）**未崩但退化到 41.75 K**（best_epoch 64）。旧 summary"层级崩塌（41.75/110.25）"数字核实无误，但需区分：41.75（Exp7 中置）是退化但学到了，110.25（Exp6/8 前置）才是真正崩塌到均值。

### 流程
**（A）** 3 condition×3 seed=**9 runs 全部完成**。**（B）** 8 个 exp 共尝试 15 run（14 完成 + 1 崩溃重跑），分 3 次迭代。并行为何 null：原子 CPA-GAT 的 per-layer 闭邻域 scatter-sum 已隐式吸收键图所携带的基数/价态/连接信息，两编码器拼读出时键通路增量信号被原子通路主导，**bond-graph 轴对原子 CPA 无互补、范式关闭**（与（A）"CPA 聚合层 subsume 链几何"是同一规律的键级版本）。层级-前置为何崩塌（110.25）：键上下文在原子输入端拼接（170d 大输入）使 CPA-GAT scatter-sum 量级爆炸、梯度发散，模型坍缩到均值，Exp8 加学习门也未能救回；层级-中置为何只退化不崩塌：键上下文放到原子 conv1 之后注入避开了输入端量级爆炸，能正常训练但原子表征已固化、键信息成扰动。Exp5 首跑因 edge-augment 维度不匹配在 epoch1 崩溃，修复后单独重跑成功（40.45 K），其余 14 run 均一次完成。`7_7_day7_polybg_cpa/` 为**空目录**（无任何文件）。

---

## Optimize 阶段：ORIG_BB（CPA-GAT + is_backbone + SWA）4 性质协议对齐复跑

> Optimize/最终模型（CPA-GAT）的启发文献主要见 Day 6–7 节。

### 实验目的
Optimize 是 7 天探索结束后的"协议对齐干净复跑"：把全项目经多轮消融已证实的最优单一配置 ORIG_BB（CPA-GAT 主干 + is_backbone 节点特征 + SWA）拿到全部 4 个 benchmark 性质（Tg/Tm/EA/IP，不含 Egc）上做最终评估，以 (a) 给出可与文献 SOTA 比较的、采用统一且解耦的 split/init 协议的 RMSE 数字，(b) 量化简单 GNN 主干相对 PerioGT 这类"周期性感知 + 大规模对比预训练"方法的性能差距。ORIG_BB 相对原始 CPA-GAT 跑的**唯一改动**是把学习率从 1e-3 降到 1e-4（config.py ORIG_BB dict、train.py docstring），其余完全保持 7 天探索中胜出的配方。

### 模型架构
主干为 simple 原始 CPA-GAT（CPA=基数保持注意力：在每条 GATv2 边输出之上再加一门控的、对邻居+自环 scatter-sum 的基数保持通道）。ORIG_BB 关键开关：use_edge=False、use_residual=False、use_norm=True、use_input_bn=False、pool="mean"、predictor_type="simple"、loss="mse"、use_swa=True、swa_start_frac=0.5、use_backbone=True。输入节点特征 **70 维**=rich 化学 **35**（13 元素 + 6 度 + 1 芳香 + 1 成环 + 1 形式电荷 + 6 H 数 + 5 杂化 + 1 质量 + 1 手性）+ PSPE 2（dce 沿 backbone 到较近端的归一化距离、mi 3 段粗粒度位置，仅 backbone 原子非零）+ RWPE 32（walk_length=32）+ is_backbone 1（该原子是否在 backbone 路径上的 0/1）。输入 Linear(70→256) 后过 4 层 CPAGATv2Conv（hidden 256、head_dim 64、heads 4），每层 ELU→LayerNorm→Dropout(=0.0)（use_norm=True、use_residual=False 无残差）；readout 为 global_mean_pool 后接 per-branch LayerNorm（pool="mean" 仅一个 branch）；预测头 simple=Linear(ro_dim→128)→ReLU→Dropout(0.1)→Linear(128→1)。共聚物（EA/IP）在 readout 后 concat 一个 6 维全局描述子 u（u=[rA,rB,共聚物类型 one-hot(4)]，COPOLYMER_TYPES=["alternating","block","random","graft"]），故预测头首层输入由 256 变 262。目标值用训练集 mean/std 归一化、模型在归一化空间预测、评估时反归一化回原始单位。参数量：单聚物 **845,313** / 共聚物 **846,081**，差 **768**（=6×128，首层 Linear 多 6 列输入），与 per-run JSON 的 num_params 完全一致。

### 训练评估方式
4 个性质数据集规模（per-run JSON counts 字段，train/val/test 加总）：Tg 7170（5162/574/1434，玻璃化温度）、Tm 3651（2628/292/731，熔融温度）、EA 42966（30935/3437/8594，Electron Affinity 电子亲和能，共聚物）、IP 42966（30935/3437/8594，Ionization Potential 电离能，共聚物）。**协议（split/init 解耦，config.py 为唯一真相）**：`SPLIT_SEEDS=[42,71]`、`INIT_SEEDS=[42,71,2026]`；Tg/Tm 多 split（2 split×3 init=6 runs，72/8/20），EA/IP 用官方给定 split（`ea_split.pkl`/`ip_split.pkl`）×3 init=3 runs，**单次完整 run 共 18 个训练 job**；为量化 run-to-run 方差，**同一 18-job 矩阵独立重复 3 次（`run1/run2/run3`，见 `run_variance.sh`），合计 54 个训练 job**。优化器 AdamW（lr=1e-4、weight_decay=1e-4），cosine+5 epoch warmup（eta_min_ratio=0.001），batch 64，max 300 epoch、early stop patience=50，grad clip 1.0。SWA（Stochastic Weight Averaging）从 50% epoch（ep150）起对 state_dict 做在线均值，但**仅作 reference**：协议选模型按 val 最低 RMSE 的 ckpt 在 test 上评一次报为 test_rmse 头条数字，SWA 模型另在 test 上评一次记为 swa_test_rmse。Ensemble 在每个 split 内对 3 个 init seed 的 best-val-ckpt 预测做 mean/median（原始单位），再对 Tg/Tm 的 2 个 split 取 mean±std（EA/IP 单 split）；3 次独立 run 的 ens_mean 再取 mean±std 作头条。

### 结果
RMSE（原始单位）。两层聚合：(1) **run 内**——每 (split,init) 训练一次，先在每 split 内对 3 个 init-seed 集成（ens_mean/ens_median），再对 Tg/Tm 的 2 个 split 取 mean±std（EA/IP 仅官方单 split）；(2) **跨 3 次独立 run**（run1/run2/run3，相同配方，方差来自硬件/cuDNN 非确定性）——对每次 run 的头条 ens_mean 再取 mean±std。头条表（跨 3 次 run，来源 `results/aggregate_runs.json`/`AGGREGATE.md`，± 为跨 3 次重复的 std）：

| dataset | n_test | single RMSE | ens_mean RMSE | ens_median RMSE |
|---|---|---|---|---|
| Tg | 1434 | 32.8793 ± 0.0784 | 31.7769 ± 0.0732 | 31.9412 ± 0.0230 |
| Tm | 731  | 51.1651 ± 0.1413 | 49.6871 ± 0.1059 | 49.9224 ± 0.1934 |
| EA | 8594 | 0.0408 ± 0.0001  | 0.0398 ± 0.0001 | 0.0399 ± 0.0001 |
| IP | 8594 | 0.0302 ± 0.0000  | 0.0294 ± 0.0000 | 0.0295 ± 0.0000 |

（optimize 不跑 Egc。）代表单次 run（`run1/RESULTS.md`，± 为跨 split 的 std）per-split 细节：**Tg** sp42 single 33.0183（init_std 0.8638）/ ens_mean 31.9191 / ens_median 32.3579（swa_mean 33.0817）；sp71 single 32.6143（0.3670）/ 31.5570 / 31.5896（swa 32.1113）。**Tm** sp42 single 50.5217（0.5720）/ 48.9628 / 49.0363（swa 50.0668）；sp71 single 51.5067（0.9720）/ 50.2937 / 50.3583（swa 50.9535）。**EA** given single 0.0408（0.0003）/ 0.0399 / 0.0400（swa 0.0407）。**IP** given single 0.0303（0.0000）/ 0.0294 / 0.0295（swa 0.0302）。SWA 仅作 reference（swa_mean 另报），头条仍取 best-val-ckpt 的 ens_mean。**相对 PerioGT SOTA**（外部对照，archive 内 `papers/day_1/2025_Periodicity-aware deep learning for polymers.pdf` Table 1/2：Tg 30.8±0.2、Tm 45.4±0.4、EA 0.0199±0.0001、IP 0.0159±0.0001），本阶段 ens_mean 差距：Tg **+0.98 K**（31.78 vs 30.8，差距最小）、Tm +4.29 K（49.69 vs 45.4）、EA +0.0199（0.0398 vs 0.0199）、IP +0.0135（0.0294 vs 0.0159）。相比 Day 6 CPA-GAT 探索期头条 ~34.77 K，经 lr 1e-3→1e-4 + 3-init ensemble + 跨-3-run 聚合后 Tg ens_mean 已收窄到 31.78（vs SOTA 30.8 仅 +0.98 K）。

### 流程
单次 18-job 矩阵经 `run_all.sh` 两 GPU 并发：GPU0 跑 tg(6)+ea(3)=9、GPU1 跑 tm(6)+ip(3)=9（无 Egc）；为测 run-to-run 方差，`run_variance.sh` 再以相同配方独立跑 run2、run3，最后 `aggregate_runs.py --runs run1 run2 run3` 跨 3 次 run 汇总出 `results/AGGREGATE.md`/`aggregate_runs.json`，每次 run 内 `collect_results.py` 汇总出 `results/runK/RESULTS.md`/`results_table.json`。`run.py` 支持断点续跑（already_done 检查 test_rmse/test_preds，已存在即 skip，--force 强制重跑），中途断电/断卡可续。config.py 为协议唯一真相：`SPLIT_SEEDS=[42,71]`、`INIT_SEEDS=[42,71,2026]`、`ALL_DATASETS=[tg,tm,ea,ip]`（4 性质，无 Egc），ORIG_BB 配方（lr 1e-4、mean pool、simple predictor、no edge/residual、SWA、is_backbone）与 train.py 一致；早期 docstring 中 "split_seed in {42,71,2026}" / "7 splits=21 runs" 等说法均为陈旧残留，一律以 config.py 与产物目录 sp42/sp71 为准。

---

## 7 天轨迹总结

ab6 是一条"先穷举证伪改进轴、再以聚合函数创新破局"的轨迹，Tg RMSE 演进：**Day1 41.6 K → Day2 35.94 K（static RWPE k=32）→ Day3 35.44 K（PSPE combined）→ Day6 34.77 K（CPA-GAT）→ Optimize 34.62 K（single）/ 33.65 K（ensemble）**。核心节点：

- **数据中心（Day1）**：富化学特征主导（basic 69→rich 43 约 26 K gap），3-RU 最稳（41.6 K），{1,3}-RU mixed 反劣于纯 3-RU；GRIN/ELENE 全负。确立"数据配方比架构花式更重要"。
- **位置编码（Day2–3）**：static RWPE k=32（35.94 K）为正且 k 越大越优，但**可学习 LSPE 本身负向**（正收益属 static RWPE，旧 summary 误把 36.29 当 k=32，实为 k=16）；PSPE combined（35.44 K，第二轮）是唯一压破 Day2 目标的配置。改进呈指数衰减（+4.75→+0.91→+0.50 K），渐近于 ~35 K。
- **三次 PI1M 自监督预训练（Day3–5）全负**：GraphMAE 节点级重建仅 −0.58 K（未达阈值，且绝对基线偏高）；SelfGNN-BYOL 图级 SSL 负（40.00 vs 40.55，No-BN 条件 cos=0.9997 塌陷证 BN 必要）；MoMuCon-NT-Xent 负（38.75 vs 35.44，cos 0.39 未塌陷但仍恶化）。**越复杂越差**，Simplicity Principle 反复被印证。GPS-Lite 全局注意力 +8 K。
- **系统故障（Day4）**：MoMuCon 被 CLAUDE.md 累积~9M token 触发的 context-limit 系统性阻塞，6 次 run_experiment 调用全失败、未跑出结果（Day5 手动补跑才得负结果）。这是本 run 唯一的工程级中断。
- **CPA 突破（Day6）**：基于"基数盲"假设提出 CPA（标准 GATv2 通道 + sigmoid 门控的未归一化 scatter_sum 通道，全层部署），34.77 K（−1.61 K/4.4%）**打破指数衰减**；CPA 仅末层无效（36.47）、显式给度数有害（37.48）证机制。is_backbone（35.03，唯一击败 baseline 的 PSE 变体）入选最终模型。
- **范式关闭（Day7）**：CPA+PSPE（35.86 vs 35.52）null、PolyBG-CPA 并行（35.64 vs 35.80）null、层级崩塌（110.25）——CPA 聚合层已 subsume 链几何位置编码与键图基数，PSPE 与 PolyBG 两条特征工程范式均关闭。
- **Optimize**：ORIG_BB（唯一改动 lr 1e-3→1e-4）27 runs 全 DONE，Tg ens_mean 33.65 K 为本 run 最强最稳健结果；与 archive 内 PerioGT SOTA（30.8 K）差 +2.85 K（ens_mean），Tg 差距最小、共聚物 EA/IP 差距约 2×。

**与旧 summary 的关键修正**：(1) static RWPE k=32 真值 **35.94 K**（非旧 summary 的 36.29，后者实为 k=16）；(2) is_backbone 单独加是**净正**（35.03），不应被 mem_154"全 PSE 负"笼统带过；(3) PerioGT SOTA **在 archive 内 `papers/day_1` PDF 找到**（非"未找到"）；(4) Day4 MoMuCon 被 context-limit 阻塞的工程细节、各天多种子协议、各 condition 完成度均已逐项代码核实。

## 主要参考文件

- 代码：`src/1_day1_multimer_feature_gat/{experiment.py,model.py,data_utils.py}`、`src/1_day1_followup_grin_elene/{experiment.py,models.py,data_utils.py}`、`src/2_day2_spse_maxgat/{RESULTS.md,spse_encoding.py}`、`src/2_day_2_lspe_gat/{models.py,train.py,run_all.py,run_all_iter2.py,all_results_summary.json,iter2_results_summary.json}`、`src/3_day3_graphmae_pretrain/{train.py,run_finetune.py,finetune_results.json}`、`src/3_day3_pspe` 与 `src/3_3_day3_pspe/{model.py,pspe_data.py,run_experiments.py,report.md,summary.json}`、`src/4_4_day4_selfgnn_polymer/{models.py,pretrain.py,finetune.py,run_all.py,experiment_results_v2.json,experiment_report_v2.md}`、`src/5_day5_gps_lite/{model.py,train.py,results_summary.json}`、`src/5_day5_momucon_ntxent`+`5_5_day5_momucon_ntxent`+`5_day5_momucon_finetune/{models.py,utils.py,pretrain.py,finetune.py,finetune_results.json}`、`src/6_day6_cpa_gat/{cpa_gat.py,results/summary.json,analyze_results.py}`、`src/6_day7_cpa_pse/{model.py,compute_pse_features.py,train.py,results/results_summary.json}`、`src/7_day7_cpa_pspe/{model.py,features.py,results/results_summary.json,diagnostics/diagnostics.json}`、`src/7_day7_polybg_cpa/{models/polybg_cpa.py,models/bond_graph.py,results/results.json,exp2_5.log,exp5_only.log,exp_iter2.log}`。
- Optimize：`src/optimize/{config.py,model.py,train.py,data_utils.py,run.py,run_all.sh,ensemble.py,collect_results.py}`、`src/optimize/results/{RESULTS.md,results_table.json,ensemble_{tg,tm,egc,ea,ip}.json}`、`src/optimize/logs/{gpu0.log,gpu1.log}`。
- 记忆：`mem.json`（165 节点；mem_001 TopicState 自述、各 day experience 节点 mem_080/081、091/093/094、105/107、119/123/124、132/134、146/154、158/159）、`mem_day1.json`~`mem_day7.json`。
- SOTA 来源：`papers/day_1/2025_Periodicity-aware deep learning for polymers.pdf`（PerioGT Table 1/2）。
