# ab2 项目实验逐日总结（projects_summary）

> **一句话定位**：ab2 = 基座 **GLM-5** / **full day-night**（白天做实验、夜晚 internalize/associate）自主推进 **7 天**的聚合物性质预测 run；最终头条模型为 **PTKGE**（周期表物理嵌入 + GIN + ECFP4 + PI1M InfoNCE 对比预训练），见文末 [Optimize 阶段](#optimize-阶段--ptkge-最终头条模型协议对齐--eaip-连通共聚物修复)。
> 旧的 `projects_summary.md` 已改名为 `projects_summary_discard2.md`（围绕最终模型组织、仅末尾带过逐日），本文改为**以天为主轴**重写。

## 概览与约定

- **资源**（来自 `log_analysis.md`）：7 天，非 run_exp 部分约 594 万 token、run_exp 部分约 8900 万 token，wall-clock ≈ **82 小时**。"ab2 / GLM-5 / full day-night" 的命名与归属来自任务设定，非从 run 内部元数据独立确认（`log_analysis.md` 只给 7 天/token/时长，未显式标注基库名）。
- **数据**（`src/alldataset/`）：单聚物 Tg.csv **7170**、Tm.csv 3651、Egc.csv 3380；共聚物 ea.csv / ip.csv 各 42966；预训练 PI1M.csv 988795（optimize 阶段全量使用）。
- **辅助目录**：`src/baseline/`（基线说明）、`src/alldataset/`（数据集），不计入"某一天"，仅在需要时作为参照。
- **来源权威性（全文统一）**：**代码 + 结果文件是唯一真相**，所有数字均从各 `src/<day>/` 的结果文件（.json/.txt/.md）摘录。`mem.json`（尤其 `mem_001` TopicState）只作为"动机 + AI 当夜裁决"的线索，且会幻觉——凡是来自 mem 而未在代码中独立验证的结论，行内一律标注"（AI 当夜自述，代码未独立验证）"。非常规缩写首次出现均展开释义。

### day → 实验目录映射

| 天 | 实验名（缩写展开） | 目录 | 关键结果（Tg RMSE，°C） | 当夜裁决 |
|---|---|---|---|---|
| Day 1 | Data-Centric Foundation | `src/1_05_data_centric/` | MLP+ECFP4 **38.78** ≫ GCN 45.69 | Data_Centric_Foundation **REJECTED** |
| Day 2 | FP-PGNN（指纹-周期图融合网络） | `src/2_day2_fp_pgnn/` | full **37.04**；指纹 +5.64、周期边 +0.29(ns) | 指纹主导 VALIDATED；周期边 NOT |
| Day 3 | PIPGA（周期不变图注意力） | `src/3_03_pippga/` | full **38.99**；指纹 +25.72、softmax +2.48 反劣 | Matformer 周期不变未能迁移 |
| Day 4 | LEMP（长程纠缠感知消息传递） | `src/4_04_lemp/` | Full 35.15 / 最佳 Cutoff5Å **35.02**；长程 **−0.08** | 长程假设 **REJECTED** |
| Day 5 | FGCL + SAPI（对比学习 + SMILES 增广） | `src/5_05_fgcl/` + `src/5_SAPI/` | FGCL full 41.65 vs supervised 41.59；SAPI +0.07 | 三机制均证伪 |
| Day 6 | PSCP（聚合物专用对比预训练） | `src/6_day6_pscp/` | baseline 37.76 → 集成 **35.42±0.07** | 表征学习范式转向 VALIDATED |
| Day 7 | PTKGE + iter2（周期表知识图谱嵌入） | `src/7_day7_ptkge/`(+iter2) | physics **34.47±1.83** 最佳（72/8/20 单划分 test **34.39**） | 物理先验 + 预训练有效 |
| Optimize | PTKGE 最终头条 | `src/optimize/` | Tg ens_mean **31.41** / 最佳单 run **30.59** | 协议对齐 + EA/IP 连通共聚物修复 |

> 注：各天 Tg RMSE 不可直接横向比较——划分方式（单划分 / 5-fold / 多 split）、种子、有效样本数（SMILES 清洗前后）、是否预训练均不同。每天数字在其自身协议内自洽。

---

## Day 1 — Data-Centric Foundation（数据中心化基线）

**目录** `src/1_05_data_centric/` ｜ **主脚本** `tg_prediction_full_pipeline.py`（1499 行）｜ **对应记忆** experience `mem_077`、integration `mem_078/079/080`、idea `mem_081–084`

**参考/启发文献**（Day 1：当日 Learn 读取 4 篇；来源 mem.json provenance）
- 【当日读取】**arXiv:2504.08859v2**（arXiv） · `mem_073` · [code](https://polyconf-icml25.github.io) —— PolyBench provides a high-quality 3D conformation dataset that enables the Self_Supervised_3D_Multimer_Pretraining hypothesis to be…
- 【当日读取】**arXiv:2512.08896v1**（arXiv） · `mem_074` · [code](https://github.com/sobinalosious/ADEPT) —— This paper enables several actionable decisions: (1) Prioritize fingerprint-based features + tree-based models as strong baselines for…
- 【当日读取】**2024 LONG-SHORT-RANGE MESSAGE-PASSING A PHYSICS-INFORMED FRAMEWORK TO CAPTURE NON-LOCAL INTERACTION**（paper） · `mem_075` · [code](https://github.com/liyy2/LSR-MP) —— Enables immediate testing of Long_Short_Range_Message_Passing hypothesis on polymer datasets (Tg, Tm, Egc). The bipartite atom-fragment…
- 【当日读取】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_076` · [code](https://github.com/wuyuhui-zju/PerioGT) —— Provides SOTA baseline numbers for Tg/Tm/Egc benchmarks. Introduces periodicity augmentation as a physics-inspired prior aligned with…

**实验目的。** 这是 7 天 trajectory 的冷启动，目标不是追高性能而是钉死"数据侧"和"模型侧"的下限：先用统计分析摸清 Tg 数据集的干净度与分布，再用两个最朴素的对照基线（指纹 MLP vs 图 GCN）确立"无任何聚合物归纳偏置时能到什么水平"，并初步探测 multimer（多重复单元）3D 表示是否可行。设计两个对照模型是要回答一个贯穿全 run 的核心问题——**子结构指纹与图拓扑，哪个对 Tg 更重要**：若 MLP 显著强于 GCN，说明指纹携带的子结构信息主导，后续应走"指纹+图融合"而非"纯图"路线。这一天的结论直接决定了 Day 2 FP-PGNN 的设计方向（idea `mem_081`）。

**模型架构。** Day 1 不是单模型而是一套 4 阶段 pipeline，含两个对照 baseline。（1）`MLPModel`：输入 ECFP4 指纹（`FingerprintExtractor` 用 `AllChem.GetMorganFingerprintAsBitVect`，radius=2、2048 bit）→ `Linear(2048→512)`+`BatchNorm1d`+`ReLU`+`Dropout(0.3)` → `Linear(512→256)`+BN+ReLU+Drop → `Linear(256→128)`+BN+ReLU+Drop → `Linear(128→1)`。（2）`SimpleGCN`：节点特征 5 维（atomic number、degree、formal charge、hybridization、aromatic）、边特征 4 维（bond types）→ `GCNConv(node→128)`→`GCNConv(128→128)`→`GCNConv(128→128)`，层间激活 → `global_mean_pool` → BN+Drop → `Linear(128→64)`+Drop → `Linear(64→1)`。（3）`MultimerGenerator`：把 repeat-unit SMILES 朴素串接成 2–5-mer，用 RDKit MMFF94（Merck Molecular Force Field 94）力场生成 3D conformer 并提取分子量 / 回转半径 / 端到端距离。（4）`DataAnalyzer`：统计 Tg 分布、离群、重复、SMILES 合法性。

**训练评估方式。** 数据 `alldataset/Tg.csv` 7170 条；划分 80/10/10 = 5736/717/717，用 `sklearn.train_test_split`，`random_state=42`，**单种子单划分（本日不做多种子、不做交叉验证、不做消融——这是定基线）**。预处理：ECFP4 为比特向量不做标准化；GCN 节点特征按上述 5 维手工构造；目标 Tg 直接用原始摄氏度（未标准化）。优化器两模型一致 `Adam(lr=1e-3, weight_decay=1e-5)`；损失 `MSELoss`；batch 32；max epochs 200、early stopping patience 20（监控验证集 loss）。评估指标 RMSE/MAE/R²，均在原始单位。全局随机种子 `RANDOM_SEED=42`（同时 seed numpy/torch/cuda）。

**结果。** MLP+ECFP4：**RMSE 38.78°C / MAE 26.80°C / R² 0.8781**，81 epoch 早停，best val loss 1410.85，模型 4.7 MB。GCN：RMSE 45.69°C / MAE 32.98°C / R² 0.8307，192 epoch，模型 175 KB。MLP 全面胜出（RMSE 低 6.9、训练快约 6×）。数据分析：7170 条 SMILES 100% 合法、100% 含 `*` 连接点、0 重复、0 统计离群；Tg 范围 −139~495°C，均值 141.87、中位 134、std 112.15，偏度 0.126（近似对称）但 Shapiro-Wilk 非正态。Multimer 3D 生成成功率低且随聚合度下降：2-mer 13%、3-mer 11%、4-mer 6%、5-mer 6%（n=100），说明朴素串接不足以构造可用 3D 链。诊断：误差在 Tg 区间大致均匀（误差-Tg 相关系数 −0.0314），仅极低（<−50°C）与极高（>400°C，n 极少）端误差大。

**流程。** 四阶段（数据分析→multimer 生成→baseline 训练→诊断）全部完成，wall-clock 约 34 分钟，日志无 killed/Traceback。本日为单种子单划分基线，未设多种子重复与消融。当夜 internalize 给出关键裁决（`mem_080`，AI 当夜自述，代码未独立验证）：`Data_Centric_Foundation` **被判定 REJECTED**——即"纯数据中心化+朴素基线"不足以达标；但同时从 MLP≫GCN 的事实提炼出后续主线"指纹+图双编码器融合"，该思路以 idea `mem_081`（Fingerprint-Periodic Graph Fusion）落地，直接演化为 Day 2 的 FP-PGNN。

---

## Day 2 — FP-PGNN（Fingerprint-Periodic Graph Neural Network，指纹-周期图融合网络）

**目录** `src/2_day2_fp_pgnn/` ｜ **主脚本** `model.py` / `dataset.py` / `train.py` / `analysis.py` ｜ **对应记忆** experience `mem_089`、integration `mem_091/092/093`、idea `mem_094–097`（由 Day 1 idea `mem_081` 催生）

**参考/启发文献**（Day 2：当日 Learn 读取 4 篇＋当夜 integration 显式回溯引用 4 篇；来源 mem.json provenance）
- 【当日读取】**2024 Geometry-enhanced pretraining on interatomic potentials**（paper） · `mem_085` · [code](https://github.com/cuitaoyong/GPIP) —— Actionable contributions: (1) Pretraining data generation strategy: Use classical MD with empirical force fields to generate abundant…
- 【当日读取】**arXiv:2209.11807**（arXiv） · `mem_086` —— Provides actionable techniques for polymer periodicity: (1) Formal definition of periodic invariance applicable to polymer repeat unit…
- 【当日读取】**arXiv:2106.04509**（arXiv） · `mem_087` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides actionable methodology for polymer pretraining: (1) Substructure substitution framework can be adapted for polymer repeat…
- 【当日读取】**2025 Advancing molecular machine learning representations with stereoelectronics-infused molecular g**（paper） · `mem_088` · [code](https://github.com/gomesgroup/simg) —— Actionable insights: (1) Explicit physics-informed graph augmentation (orbital nodes, interaction edges) provides both performance gains…
- 【当夜引用】**arXiv:2504.08859v2**（arXiv） · `mem_073` · [code](https://polyconf-icml25.github.io) —— PolyBench provides a high-quality 3D conformation dataset that enables the Self_Supervised_3D_Multimer_Pretraining hypothesis to be…
- 【当夜引用】**arXiv:2512.08896v1**（arXiv） · `mem_074` · [code](https://github.com/sobinalosious/ADEPT) —— This paper enables several actionable decisions: (1) Prioritize fingerprint-based features + tree-based models as strong baselines for…
- 【当夜引用】**2024 LONG-SHORT-RANGE MESSAGE-PASSING A PHYSICS-INFORMED FRAMEWORK TO CAPTURE NON-LOCAL INTERACTION**（paper） · `mem_075` · [code](https://github.com/liyy2/LSR-MP) —— Enables immediate testing of Long_Short_Range_Message_Passing hypothesis on polymer datasets (Tg, Tm, Egc). The bipartite atom-fragment…
- 【当夜引用】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_076` · [code](https://github.com/wuyuhui-zju/PerioGT) —— Provides SOTA baseline numbers for Tg/Tm/Egc benchmarks. Introduces periodicity augmentation as a physics-inspired prior aligned with…

**实验目的。** 承接 Day 1 对 Data_Centric_Foundation 的 REJECTED 裁决——Day 1 已证 MLP+ECFP4(38.78) 显著优于纯 GCN(45.69)，指纹捕获了消息传递漏掉的子结构（idea `mem_081`）。本天把该洞察正式化为 FP-PGNN，同时验证三个假设：H1 ECFP4 编码 GCN 缺失的子结构、H2 周期虚拟边编码聚合物拓扑、H3 指纹-图融合捕获互补信息；目标是把 RMSE 压到 <37.0 并缩小与 PerioGT SOTA(30.8) 的差距。

**模型架构。** 三路变体共用 p-SMILES→图构造（`dataset.py`）：解析 `[*:1]…[*:2]` 定位头/尾 dummy 原子，节点特征 28 维（19 类原子 one-hot[C,N,O,S,F,Cl,Br,I,P,Si,B,Na,K,Ca,Mg,Zn,Fe,Cu,Other] + 度[min(deg,6)/6] + 6 类杂化 one-hot + 芳香性 + 形式电荷 clip[-3,3]/3），边特征 5 维（4 类键 one-hot + 周期标记位）；周期边 = 头↔尾双向虚拟边（键类型置零、周期标记=1.0）；ECFP4 = Morgan r=2、2048 bit，不标准化。(1) **FP_PGNN（全模型，430,861 参数）**：`FingerprintEncoder` `Linear(2048→256)+ReLU+Linear(256→128)` → 128 维；`GraphStream` `Linear(28→128)+ReLU` 后接 3×`GATConv`(hidden128, heads4, 前两层 concat/末层不 concat, edge_dim5, add_self_loops)，每层残差+LayerNorm+Dropout(0.1)，`global_mean_pool`→128；融合层 `concat([graph_emb, fp_emb])`→256；预测头 `Linear(256→128)+ReLU+Drop0.1+Linear(128→64)+ReLU+Drop0.1+Linear(64→1)`。**代码关键发现**：`GraphStream` 内定义了逐层指纹调制分支（`Linear(128→4)+Sigmoid`、`FingerprintModulatedGATConv`）但 `forward` 中从未调用——`fingerprint_embedding` 作为参数传入却被丢弃，融合实际只是末段 `torch.cat` 纯拼接，并非设想的注意力调制（这与 `mem_089` 自述"简单拼接、非注意力调制"一致，且为代码独立验证）。(2) **BaselineGAT（117,633 参数）**：同 GraphStream 但无指纹、无周期边，pool 后接 `Linear(128→64)+ReLU+Drop+Linear(64→1)`。(3) **BaselineMLP（557,569 参数）**：仅 ECFP4，`Linear(2048→256)+ReLU+Drop+Linear(256→128)+ReLU+Drop+Linear(128→1)`（hidden=256）。

**训练评估方式。** Tg.csv 7170 样本（`analysis_results.json` n_samples=7170，全部合法）；**5-fold CV**（`KFold n_splits=5, shuffle=True, random_state=42`），单种子、无 repeated CV；ECFP 不标准化、Tg 原始 °C；Adam lr1e-3 wd1e-5 + `ReduceLROnPlateau`(factor0.5, patience10)；损失 **MSE**；batch 64；ep200 早停 patience20；指标 RMSE/R²/MAE（验证折 1/5）；5 消融配置 × 5 折 = **25 次训练**。

**结果。**（`results.json`，5 折 mean±std）**fp_pgnn_full 37.04±0.68 / R²0.891±0.005**（折 RMSE[36.39,36.81,38.13,37.49,36.37]，best_epoch[39,47,11,37,43]）；fp_pgnn_no_periodic 37.33±0.73 / 0.889；fp_pgnn_no_fingerprint 42.67±2.06 / 0.855；gat_baseline（纯 GAT）43.50±2.04 / 0.849；mlp_baseline（ECFP4）38.02±0.85 / 0.885。消融拆解（Δ=去该组件−全模型）：周期边 **+0.29**（小于折间 std0.68→不显著）、指纹 **+5.64**（权威值：原始 42.6725−37.0373=5.6352→+5.64；旧 summary 的 +5.63 系由已四舍五入的 42.67−37.04 反推所致，`mem_089` 的 +5.64 反而正确）、纯 GAT(双去) +6.47、MLP 相对全模型 +0.98。`analysis_results.json`：overall RMSE37.04 / R²0.891 / n7170，采样 500 条 MAE23.73，`periodic_edge_attention=null`（未抽取到注意力）。

**流程。** 25 折全完成、状态 SUCCESS，日志无 Traceback/无 killed 进程；wall-clock 未落盘（idea 预算 <2h@单卡 2080Ti）。一处实现缺陷：no_periodic / no_fingerprint 两个消融复用 `model_type='fp_pgnn'`，使每折 `checkpoint_fold{k}_fp_pgnn.pt` 被后跑的 no_fingerprint 覆盖，故仅余 15 个 checkpoint（fold{1-5}_{fp_pgnn,gat,mlp}），全模型权重未单独留存（代码已验证：`train.py` 各消融均传 `model_type='fp_pgnn'`）。当夜裁决（integration `mem_091/092/093`，**AI 当夜自述，代码未独立验证**；其数值结论已由 results.json 旁证）：H1 VALIDATED——指纹贡献主导（+5.64，约为周期边的 19×）；H2 NOT VALIDATED——头尾虚拟边未能编码周期性，周期性应入注意力而非图拓扑（援引 Matformer 周期不变性，破坏则 MAE 退化 53%）；H3 PARTIALLY VALIDATED——拼接融合有效但有限；`mem_093` 把 6.24 RMSE 与 SOTA 的差距分解为周期性编码 ~4-5 / 长程 ~1-2 / 融合 ~0-1。主目标 RMSE<37.0 差 0.04 未达（在噪声内），次目标"两组件各>1"仅指纹达标。由此催生 Day 3 四个 idea：`mem_094` PIPGA（Matformer 周期不变性改造）、`mem_095` FMPA（指纹调制周期注意力）、`mem_096` Tight-Binding 谱编码器、`mem_097` 转移积分分布编码。

---

## Day 3 — PIPGA（Periodic-Invariant Polymer Graph Attention，周期不变图注意力）

**目录** `src/3_03_pippga/` ｜ **主脚本** `run_experiments.py`（消融总入口，调 `src/model.py`+`src/train_ablation.py`）｜ **对应记忆** experience `mem_102`、integration `mem_104/105/106`、idea `mem_107–110`

**参考/启发文献**（Day 3：当日 Learn 读取 4 篇＋当夜 integration 显式回溯引用 4 篇；来源 mem.json provenance）
- 【当日读取】**2024 PointGAT A Quantum Chemical Property Prediction Model Integrating Graph Attention and 3D Geomet**（paper） · `mem_098` · [code](https://github.com/sevencheung2021/PointGAT) —— Actionable insights for polymer property prediction: (1) Multi-modal fusion architecture - PointGAT's approach of combining graph attention…
- 【当日读取】**arXiv:2303.04791v2**（arXiv） · `mem_099` —— Provides validated evidence that explicit long-range message passing yields consistent 10-16% improvements across architectures. The…
- 【当日读取】**2025 Learning non-local molecular interactions via equivariant local representations and charge equi**（paper） · `mem_100` · [code](https://github.com/tummfm/chemtrain) —— CELLI provides an actionable architecture for incorporating long-range interactions into polymer property prediction models. Key design…
- 【当日读取】**arXiv:2209.11807v1**（arXiv） · `mem_101` —— CRITICAL for Day 3 priorities: Provides exact implementation blueprint for periodic invariance. (1) Graph construction: Use radius-based…
- 【当夜引用】**2024 Geometry-enhanced pretraining on interatomic potentials**（paper） · `mem_085` · [code](https://github.com/cuitaoyong/GPIP) —— Actionable contributions: (1) Pretraining data generation strategy: Use classical MD with empirical force fields to generate abundant…
- 【当夜引用】**arXiv:2209.11807**（arXiv） · `mem_086` —— Provides actionable techniques for polymer periodicity: (1) Formal definition of periodic invariance applicable to polymer repeat unit…
- 【当夜引用】**arXiv:2106.04509**（arXiv） · `mem_087` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides actionable methodology for polymer pretraining: (1) Substructure substitution framework can be adapted for polymer repeat…
- 【当夜引用】**2025 Advancing molecular machine learning representations with stereoelectronics-infused molecular g**（paper） · `mem_088` · [code](https://github.com/gomesgroup/simg) —— Actionable insights: (1) Explicit physics-informed graph augmentation (orbital nodes, interaction edges) provides both performance gains…

**实验目的。** Day 2 的 FP-PGNN 已表明：简单周期边（连接 head/tail）仅贡献约 +0.29 RMSE，而 ECFP4 指纹贡献约 +5.64 RMSE，模型距 SOTA（PerioGT 30.8 RMSE，来自 mem 自述，代码未独立验证）仍有明显差距。Day 2 当夜生成的 idea `mem_094`（day=2）提出：Matformer 在 3D 晶体上的"周期不变性"原则（对单元胞边界平移不变）可直接迁移到 1D 聚合物重复单元——不同的 SMILES"切点"即不同的胞边界选择；并据此推断标准 softmax 注意力会错误地跨周期边界归一化，故必须改用 edge-wise（逐边）注意力。Day 3 的目的即用代码实现并证伪/证实该假设：仿 Matformer 的周期不变 + edge-wise attention，看能否把 RMSE 压到 <34.0 的目标，并通过五元消融分解指纹、周期多边、自环、注意力变体的各自贡献。

**模型架构。** 最终权威实现为 `src/model.py`（`results/full_model/results.json` 配置：`node_input_dim=23, node_dim=128, edge_dim=64, num_layers=4, num_heads=4, fp_dim=2048`）。输入数据由 `src/data.py` 构图：p-SMILES 中 `*` 替换为 `[At]` 解析，识别 head/tail 原子；对每个原子 BFS 求全图距离，`distance_threshold=3` 跳以内均为邻居；做 3-copy 扩展（left/center/right），中心 copy 内邻居记周期偏移 k=0，跨向右 copy 记 k=+1、向左记 k=−1，从而构造 multi-edge 图；同时为每个原子生成 4 维 self-loop 特征 `(d_head, d_tail, is_head, is_tail)`。每个样本另用 RDKit 算 2048 维 ECFP4（radius=2）指纹。前向流程（`PIPGA.forward`）：(1) `node_proj` 将 23 维原子特征线性投到 128；(2) 边特征由 `PeriodicOffsetEmbedding`（3 个可学习向量，对应 k∈{−1,0,+1}）查表得到 64 维；(3) 经 4 个 `PIPGALayer`：每层先 `EdgeWiseAttention` 再残差+LayerNorm，再 FFN(Linear 128→256→128 + ReLU) 残差+LayerNorm；若 `use_self_loop=True`，`SelfLoopEncoder`(MLP 4→64→64) 把 self-loop 特征编码后并入边特征。**周期不变的核心机制是 `EdgeWiseAttention`**：`W_q(128→128)` 出 query，`W_k([h_j‖e_ij], 192→128)` 出 key（key 拼接了边特征与目标节点），4 头（head_dim=32），逐边注意力系数 `α = sigmoid((q∘k)·Σ/√d)`，**显式不做邻域 softmax**，并对 `α` 施加 `LayerNorm(num_heads)` 稳定；消息 `v⊙α` 后 scatter-add 聚合。`FingerprintModulator` 把 2048 维 ECFP4 经 `proj(2048→128)` 与 `gate(128→1,sigmoid)` 生成逐图标量门，乘到节点特征上；(4) 读出为按图 segment-mean-pool 得 graph_emb[batch,128]；(5) 若用指纹则 concat `fp_proj(2048→128, ReLU)` 得 [batch,256]，否则 [batch,128]；(6) 预测头 `Linear(256→128)→ReLU→Linear(128→1)` 输出（归一化空间）。消融变体（`run_experiments.py` 配置矩阵）：**Full**=softmax:F, fp:T, self_loop:T, multi_edge:T；**No Fingerprint**=fp:F；**Softmax Attention**=softmax:T（改用 scatter-softmax 对目标节点邻域归一化、不加 attn LayerNorm）；**No Self-Loop**=self_loop:F；**Single Edge**=multi_edge:F（仅保留 k=0 边、偏置置零）。注意顶层 `model.py`/`train.py` 是更早的一版实现（`hidden_dim=256, num_layers=3, AdamW, 500 epochs`），与 `src/` 这套不是同一份代码。

**训练评估方式。** 数据 `Tg.csv`，7170 个聚合物重复单元，Tg mean=141.87、std=112.15、range=[−139.00, 495.00]（来自 `results.txt`）。划分：5-fold `KFold(n_splits=5, shuffle=True, random_state=42)`；每 fold 取 20% 作 test，剩余 80% 再切 ≈71%/9% 作 train/val（`n_val=int(len(train_val)*0.111)`）。预处理：对 Tg 目标按 train 折拟合 `StandardScaler` 并作用到 val/test，评估时反归一化回原始尺度算指标；ECFP4 由 RDKit 现算。优化器 **Adam**，`lr=1e-3, weight_decay=1e-5`；`ReduceLROnPlateau(mode=min, factor=0.5, patience=10)`；损失 **MSELoss**（归一化目标空间）；`batch_size=32`，`max_epochs=200`，`early_stopping_patience=20`（val RMSE 不降即停并保存 best checkpoint），`clip_grad_norm_=1.0`。指标：反归一化后的 RMSE / MAE / R²（Tg 原始单位）。仅 seed=42、单种子、5-fold CV，无多种子重复；五元消融各跑一遍 5-fold。

**结果。** 全部摘自 `results/<variant>/results.json`（与 `results/FINAL_REPORT.md` 一致，为权威值；`test_rmse_mean / std`）：**Full PIPGA 38.99 ± 2.10**（MAE 26.57 ± 2.16，R² 0.879 ± 0.013；逐 fold RMSE 42.29/37.15/39.76/39.39/36.34，best_epoch=1/86/4/26/10）；**No Fingerprint 64.71 ± 4.45**（MAE 49.28 ± 3.86，R² 0.665 ± 0.047，Δ=+25.72）；**Softmax Attention 41.47 ± 3.51**（MAE 29.12 ± 4.07，R² 0.863 ± 0.021，Δ=+2.48）；**No Self-Loop 39.93 ± 1.79**（MAE 27.94 ± 2.26，R² 0.873 ± 0.011，Δ=+0.95）；**Single Edge 39.09 ± 2.14**（MAE 26.85 ± 2.05，R² 0.878 ± 0.017，Δ=+0.11）。**目标 Full RMSE <34.0 未达，差 4.99**；三条消融门槛（edge-wise attention >1.5、self-loop >0.5、fingerprint >0.5）均"达到"，但实际由指纹主导。

**关于两个 Full RMSE（37.94 vs 38.99）的厘清（已查代码与结果文件）。** 两者来自不同次运行，**权威最终值是 38.99 ± 2.10**：它记录于 `results/full_model/results.json`（精确 38.9854）和 `results/FINAL_REPORT.md`，逐 fold 与 JSON 完全吻合，配置为 `node_dim=128/num_layers=4/max_epochs=200/patience=20`，由 `src/`+`run_experiments.py` 产生；experience `mem_102` 引用的也是这个数。**37.94 ± 3.50 是一版"中间态"消融跑的结果**，只残留在 `results/summary.txt` 里，其逐 fold 与各分量都与最终版对不上（最明显的是其 No-Fingerprint=40.77，而最终版为 64.71，差异巨大，说明该中间跑的指纹通路与最终版不同），已被 JSON/FINAL_REPORT 覆盖；`summary.txt` 内部还出现一处笔误 "37.89 ± 3.65"。此外目录顶层 `results.txt` 里的 **45.53 ± 4.98** 是**更早一次单模型跑**，来自原版顶层 `model.py`/`train.py`（`hidden_dim=256/num_layers=3/AdamW/epochs=500/patience=50`，`checkpoints/best_fold_0..4.pt` 属于它），与消融套件无关、非 5 元消融之一。

**流程。** 实际跑的消融组合即上述 5 变体 × 5 fold = 25 次训练，**全部完成**（5 个 `results/*/results.json` 齐全、`fold_results` 各 5 条；顶层 5 个 `checkpoints/best_fold_*.pt` 为更早那次单模型跑的产物）。**周期不变性测试未实际执行**：`src/periodic_invariance_test.py`（沿主链滑动切点生成等价 RU 视图、检验预测差均值 <0.5）脚本已写，但目录下找不到任何对应结果文件，`FINAL_REPORT.md` 亦将其标为 "INCOMPLETE / Not verified"，故"周期不变性"在数值上并未被独立验证。wall-clock 无法从日志恢复——`training_log.txt`（35MB）与 `training_output.log` 几乎全是 RDKit `MorganGenerator` 弃用警告；按 mem 时间戳估算，day3 idea `mem_094` 生成于 05:20、experience `mem_102` 写于 09:46，全天构思—实现—训练—复盘约 4.4 小时。当夜裁决（integration `mem_104/105/106`，**AI 当夜自述，代码未独立验证**）：`mem_104` 判定 Matformer 周期不变未能从 3D 晶格迁移到 1D 聚合物链（multi-edge 仅 +0.11、self-loop 仅 +0.95、而 ECFP4 +25.72），推翻了 `mem_091/093` 关于"周期性编码可回收 4–5 RMSE"的预期；`mem_105` 据此把"长程相互作用/缠结"升格为剩余 gap 的首要假设（援引 Ewald MP、CELLI、PointGAT 三篇，均属文献推断非本仓代码验证）；`mem_106` 宣布局部指纹主导（自述"ECFP4 贡献 66% 性能"），主张战略转向"指纹优先 + 轻量 GNN 增强"。这些因果性裁决是对消融增量的解读，代码只能验证增量本身、不能验证"Matformer 失配""66% 来自指纹"等归因。当晚并为 Day 4 催生 4 条 idea（`mem_107–110`，均 AI 构思、代码未验证）：LEMP（二部图长程缠结消息传递，仿 LSR-MP）、PAFT（把 PerioGT 的 PA 周期性增强改造为有监督对比微调）、SCTE（傅里叶链序列 + 拉普拉斯特征谱的谱拓扑编码）、CRIP（两阶段：先预测链刚性再映射到 Tg）。

---

## Day 4 — LEMP（Long-Range Entanglement-Aware Message Passing，长程纠缠感知消息传递）

**目录** `src/4_04_lemp/` ｜ **主脚本** `train_lemp.py`（驱动）、`lemp_model.py`（模型）、`dataset.py`（数据）｜ **对应记忆** experience `mem_115`、integration `mem_116/117/118`、idea `mem_119–122`

**参考/启发文献**（Day 4：当日 Learn 读取 4 篇＋当夜 integration 显式回溯引用 9 篇；来源 mem.json provenance）
- 【当日读取】**2025 A foundation machine learning potential with polarizable long-range interactions for materials**（paper） · `mem_111` · [code](https://github.com/reaxnet/reaxnet) —— ACTIONABLE: (1) The PQEq core-shell model provides a template for implementing explicit long-range modules in polymer GNNs - the spring…
- 【当日读取】**Periodic Polymer Graphs Learned Descriptors**（paper） · `mem_112` —— Provides validated baseline: polymer graph + MPNN achieves 34.91 RMSE on Tg, establishing that periodicity-aware representation alone is…
- 【当日读取】**Ewald Long-Range MP**（paper） · `mem_113` —— Actionable: Implement Ewald-style Fourier-space message passing for polymer Tg prediction. The structure factor embedding approach can…
- 【当日读取】**2025 SimPoly Simulation of Polymers with Machine Learning Force Fields Derived from First Principles**（paper） · `mem_114` —— Enables testing of long-range message passing architectures on real polymer MD simulations. Provides benchmark (PolyArena) for evaluating…
- 【当夜引用】**full**（paper） · `mem_052` —— This paper introduces a periodic polymer graph representation that explicitly encodes polymer chain periodicity and combines it with a message-passing neural network to a
- 【当夜引用】**2024 LONG-SHORT-RANGE MESSAGE-PASSING A PHYSICS-INFORMED FRAMEWORK TO CAPTURE NON-LOCAL INTERACTION**（paper） · `mem_075` · [code](https://github.com/liyy2/LSR-MP) —— Enables immediate testing of Long_Short_Range_Message_Passing hypothesis on polymer datasets (Tg, Tm, Egc). The bipartite atom-fragment…
- 【当夜引用】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_076` · [code](https://github.com/wuyuhui-zju/PerioGT) —— Provides SOTA baseline numbers for Tg/Tm/Egc benchmarks. Introduces periodicity augmentation as a physics-inspired prior aligned with…
- 【当夜引用】**arXiv:2209.11807**（arXiv） · `mem_086` —— Provides actionable techniques for polymer periodicity: (1) Formal definition of periodic invariance applicable to polymer repeat unit…
- 【当夜引用】**arXiv:2106.04509**（arXiv） · `mem_087` · [code](https://github.com/illidanlab/MoCL-DK) —— Provides actionable methodology for polymer pretraining: (1) Substructure substitution framework can be adapted for polymer repeat…
- 【当夜引用】**2024 PointGAT A Quantum Chemical Property Prediction Model Integrating Graph Attention and 3D Geomet**（paper） · `mem_098` · [code](https://github.com/sevencheung2021/PointGAT) —— Actionable insights for polymer property prediction: (1) Multi-modal fusion architecture - PointGAT's approach of combining graph attention…
- 【当夜引用】**arXiv:2303.04791v2**（arXiv） · `mem_099` —— Provides validated evidence that explicit long-range message passing yields consistent 10-16% improvements across architectures. The…
- 【当夜引用】**2025 Learning non-local molecular interactions via equivariant local representations and charge equi**（paper） · `mem_100` · [code](https://github.com/tummfm/chemtrain) —— CELLI provides an actionable architecture for incorporating long-range interactions into polymer property prediction models. Key design…
- 【当夜引用】**arXiv:2209.11807v1**（arXiv） · `mem_101` —— CRITICAL for Day 3 priorities: Provides exact implementation blueprint for periodic invariance. (1) Graph construction: Use radius-based…

**实验目的。** Day 3 的整合裁决 `mem_105` 把"长程相互作用"提升为解释与 SOTA 之间 8.19 RMSE 落差的首位假设——三篇文献（Ewald message passing、CELLI、PointGAT）均报告显式长程机制带来 10–40% 提升，加之 Day 3 的周期性编码失败（+0.11 RMSE），长程成了最可疑的缺口。`mem_107`（idea）据此提出 LEMP：用体素网格 + 结构因子 + Ewald 风格的长程相关来捕捉聚合物链缠结效应，补 ECFP4 指纹"看不到"的空间关联。本天目标有二：(i) 验证长程模块能否带来 >1.5 RMSE 改进（主假设），(ii) 验证指纹相对 GIN 是加性而非替代（次假设），并把整体 RMSE 压到 <34.0。

**模型架构。** 输入为一条 pSMILES→SMILES 解析后的聚合物重复单元分子图 + 2048-bit ECFP4 指纹 + RDKit ETKDG 生成的 3D 构象（含通配符时回退到 2D+z 的伪 3D）。LEMP 由三条分支经 concat 融合后接 MLP 回归头输出标量 Tg：

- **指纹分支（ECFPEncoder）**：`Linear(2048→256) → BatchNorm → ReLU → Dropout(0.1) → Linear(256→256)`，输出 256 维。
- **局部分支（LocalGIN）**：2 层 `GINConv`（`train_eps=True`），节点特征 25 维（10 原子类型 one-hot + 6 degree + 5 杂化 + aromatic/formal_charge/num_H/in_ring 共 4 标量；EXPERIMENT_STATUS 记录 setup 期把 `node_dim` 由默认 44 改为 25 以对齐数据），先 `Linear(25→128)` 编码，每层后接 BN+ReLU，`global_mean_pool` 池化得 128 维。
- **长程分支（LongRangeBranch）**：三子编码器各出 128 维，再 `Linear(384→128)+BN+ReLU+Dropout(0.1)` 融合成 128 维：
  1. *VoxelGridEncoder*：把原子落入 16×16×16、8 通道（按原子类型取模映射）的 3D voxel 网格，过 3 层 `Conv3d`（通道 8→32→64→128，kernel=3、padding=1，带 BN+ReLU+MaxPool）后 `Linear(128→128)`；
  2. *StructureFactorEncoder*：在 32 个 k 点上计算结构因子 S(k)（实/虚部 × xyz = 6 维/k 点 → 192 维），`Linear(192→128)+BN+ReLU+Linear(128→128)`；
  3. *EwaldLongRangeEncoder*：计算距离 >cutoff 的 64-bin 距离直方图作为长程相关，`Linear(64→128)+BN+ReLU+Linear(128→128)`——`cutoff` 仅作用于此子模块。
- **融合与预测**：开启的分支 concat（Full 模型 = 256+128+128 = 512 维）→ `Linear(→256)+BN+ReLU+Dropout(0.2) → Linear(→128)+ReLU+Dropout(0.1) → Linear(128→1)`。

消融变体靠构造函数开关 `use_long_range/use_fingerprint/use_gin` 关闭整条分支并相应缩减 fusion 输入维度（No Long-Range = 指纹+GIN；No Fingerprint = GIN+长程；No GIN = 指纹+长程）；cutoff 消融变体只把 Ewald 子模块的 `cutoff` 设为 5/10/15 Å，voxel 与结构因子不受影响。

**训练评估方式。** 数据：`src/alldataset/Tg.csv`，共 7170 条聚合物重复单元，全部成功处理、0 失败。划分与种子：`sklearn.KFold(n_splits=5, shuffle=True, random_state=42)`；每条配置跑 5 fold，7 条配置 × 5 fold = **35 runs**，无额外多种子。预处理：RDKit ETKDG 生成 3D 构象、ECFP4（radius=2, 2048 bit）、节点/边特征按 `dataset.py` 现算。优化：`Adam(lr=1e-3, weight_decay=1e-5)`，`ReduceLROnPlateau(factor=0.5, patience=10)`，损失 `MSELoss`，batch=32，max_epochs=200、早停 patience=20。指标：5 fold 拼合后统一计算 overall RMSE/MAE/R²（Tg 单位 K），同时报告 fold 级 avg±std RMSE。每 fold 存 `fold_{0..4}_best.pt`。变体覆盖：4 条组件消融（Full / No Long-Range / No Fingerprint / No GIN）+ 3 条 cutoff 消融（5/10/15 Å）。

**结果。** 全部数字来自 `ablation_results.json` / `analysis.json` / `FINAL_RESULTS_REPORT.md`（overall RMSE，K）：

| 配置 | avg RMSE | std | overall RMSE | MAE | R² |
|---|---|---|---|---|---|
| **Cutoff 5Å**（最佳） | 34.98 | 1.50 | **35.02** | 23.12 | 0.9023 |
| No Long-Range（baseline） | 35.04 | 1.35 | **35.06** | 23.13 | 0.9021 |
| **Full LEMP** | 35.11 | 1.47 | **35.15** | 23.25 | 0.9016 |
| Cutoff 15Å | 35.13 | 1.40 | **35.16** | 23.35 | 0.9015 |
| Cutoff 10Å | 35.16 | 1.38 | **35.19** | 23.58 | 0.9014 |
| No GIN | 36.42 | 1.09 | **36.43** | 24.56 | 0.8942 |
| No Fingerprint | 46.71 | 1.18 | **46.73** | 33.84 | 0.8262 |

组件贡献（`analysis.json`，定义为"移除后 RMSE 增量"）：**长程贡献 −0.08**（35.06 − 35.15，负，即加入长程反而劣化）；**指纹贡献 +11.58**（46.73 − 35.15，移除后崩塌）；**GIN 贡献 +1.29**（36.43 − 35.15，小而正）。Cutoff 扫描显示 5Å 最优、10/15 Å 略劣，距离依赖极弱。性能目标 <34.0 未达成：最佳 35.02 差 **1.02**、Full 35.15 差 **1.15**。

**流程。** 单进程 35 runs、wall-clock **12h54m**（11:10 启动、次日 00:04 完成，约 22 min/fold），CUDA 12.6 + PyTorch/PyG。`training_run5.txt`（最终成功 run）0 个 Traceback/Killed；`training_log.txt`/`training_final.txt` 中的少量 Traceback 来自 setup 期被修掉的早尝试（collate 批处理指纹、`node_dim` 44→25、3D 位置 numpy→torch、结构因子编码器输出 6×K=192 维四项修复，见 EXPERIMENT_STATUS），最终 7×5=35 runs 全部完成、产物落盘。**本天核心结论是"长程假设被证伪"**：主假设（长程 >1.5 RMSE 改进）以 −0.08 的负贡献失败，次假设（指纹加性）以 +11.58 强证伪性通过；AI 当夜在 `mem_115`（experience：长程假设 REJECTED）与 integration `mem_116`（"Tg 是局部化学问题而非空间相关问题"）、`mem_117`（指纹 dominance 三角验证：Day2 +5.64 / Day3 +25.72 / Day4 +11.58）、`mem_118`（PerioGT 机制谜题：周期性 +0.11、长程 −0.08 均无法解释 4.35 RMSE 落差）做出的归纳均为 AI 当夜自述、代码未独立验证；正是这些谜题催生了 Day 5 的 idea `mem_119`（SAPI，SMILES 增广）、`mem_120`（FGCL，指纹引导对比学习）、以及 `mem_121/122`（PEL / LFE，表征学习与可学习指纹嵌入方向）。

---

## Day 5 — FGCL + SAPI（双子实验：指纹引导对比学习 + SMILES 增广）

**目录** `src/5_05_fgcl/` + `src/5_SAPI/` ｜ **对应记忆** experience `mem_125`、integration `mem_127/128/129`、idea `mem_130–133`

**参考/启发文献**（Day 5：当日 Learn 读取 2 篇＋当夜 integration 显式回溯引用 4 篇；来源 mem.json provenance）
- 【当日读取】**2017 SMILES Enumeration as Data Augmentation**（paper） · `mem_123` · [code](https://github.com/Ebjerrum/SMILES-enumeration) —— This paper enables the following decisions/experiments for TopicState: (1) VALIDATES SMILES enumeration as a viable data augmentation…
- 【当日读取】**2021 MoCL Knowledge aware Contrastive Learning**（paper） · `mem_124` · [code](https://github.com/illidanlab/MoCL-DK) —— ACTIONABLE: Provides concrete methodology to test whether contrastive learning with domain knowledge explains PerioGT's performance. Key…
- 【当夜引用】**2025 A foundation machine learning potential with polarizable long-range interactions for materials**（paper） · `mem_111` · [code](https://github.com/reaxnet/reaxnet) —— ACTIONABLE: (1) The PQEq core-shell model provides a template for implementing explicit long-range modules in polymer GNNs - the spring…
- 【当夜引用】**Periodic Polymer Graphs Learned Descriptors**（paper） · `mem_112` —— Provides validated baseline: polymer graph + MPNN achieves 34.91 RMSE on Tg, establishing that periodicity-aware representation alone is…
- 【当夜引用】**Ewald Long-Range MP**（paper） · `mem_113` —— Actionable: Implement Ewald-style Fourier-space message passing for polymer Tg prediction. The structure factor embedding approach can…
- 【当夜引用】**2025 SimPoly Simulation of Polymers with Machine Learning Force Fields Derived from First Principles**（paper） · `mem_114` —— Enables testing of long-range message passing architectures on real polymer MD simulations. Provides benchmark (PolyArena) for evaluating…

本天双子实验承接 Day 3（PIPGA 周期性，+0.11）与 Day 4（LEMP 长程，−0.08）两条已否定路径，从"对比学习"与"数据增广"两个剩余表征层面继续追查 PerioGT 的 SOTA 优势来源；两实验共用同一份 `src/alldataset/Tg.csv`（7170 聚合物）、同一 5-fold KFold(shuffle, seed=42)，构成可横向对比的严格对照，且均得阴性结果，决定性地把假设空间推向自监督预训练（Day 6）。非常规缩写：FGCL=Fingerprint-Guided Contrastive Learning；SAPI=SMILES Augmentation for Polymer Invariance；MoCL=Molecular Contrastive Learning。

### (A) FGCL

**目录** `src/5_05_fgcl/` ｜ **主脚本** `train_fgcl.py`、`fgcl_model.py`（辅：`data_utils.py`、`generate_diagnostics.py`；报告 `TRAINING_REPORT.md`/`summary.txt`；结果 `results/{summary.csv,results.csv,detailed_results.json}`、`diagnostics/`）

**实验目的。** 测 MoCL 风格的"指纹引导对比学习"能否解释 PerioGT 的优势。承接 Day 4 后 integration `mem_129` 把"领域知识对比学习（MoCL，mem_124）"列为 RANK_2 待验假设，FGCL 即对其进行正面检验——验证 ECFP 相似度引导的对比预训练 + 正则是否给 Tg 回归带来增益。

**模型架构。** 双分支。① 监督主干：GIN 2 层图编码器（hidden 128，node_dim 44，global_mean_pool）→ 与 2048-bit ECFP4（Morgan r=2）拼接 → PredictionHead([graph 128 ⊕ ECFP 2048]→64→1)，约 280K 参数。② 对比分支：ProjectionHead(128→64→32)，InfoNCE 损失（temperature 0.1，L2 归一化后余弦相似度）。正负样本构造：代码 `info_nce_loss` 的正样本为 batch 对角线（每个分子对自身）、其余 batch 成员为负样本（对相似度行做 cross-entropy）；函数签名里声明的 ECFP-Tanimoto 阈值（pos≥0.7、neg≤0.3）**在损失体中并未实际使用**——"fingerprint-guided"正负选择只是名义存在，实质是投影空间的塌缩抑制正则。λ（对比损失权重）默认 0.5。

**训练评估方式。** 7170 样本，5-fold KFold(shuffle=True, random_state=42)；Adam lr=1e-3 + ReduceLROnPlateau；batch 32；监督损失 MSE；对比预训练 10 epochs + 微调 ≤100 epochs（early stopping patience=20）。消融四模式 + λ 扫描：`supervised_only`（纯监督）/`pretrain_only`（仅 10 ep 对比预训练）/`regularization_only`（微调期 λ=0.5 对比正则）/`full`（预训练 + 正则），另对 full 跑 λ∈{0.1, 0.5, 1.0, 2.0}。

**结果。**（5-fold test RMSE，源自 `results/summary.csv` 与 `diagnostics/summary_statistics.csv`，与 `TRAINING_REPORT.md` 互洽）：`full`(λ=0.5) **41.65 ± 1.80** ｜ `supervised_only` **41.59 ± 1.06** ｜ `regularization_only` 41.59 ± 1.32 ｜ `pretrain_only` 42.09 ± 2.33（最差且方差最大）｜ λ 扫描 0.1→41.76、0.5→41.85、1.0→41.59、2.0→41.60（基本持平）；各模式 R²≈0.86，best single-fold RMSE=38.74（full λ=2.0 fold1）。**结论：对比学习相对强监督基线无显著增益**——full 反而略逊于 supervised_only，pretrain_only 单独更差且更不稳；λ 不敏感。假设证伪：MoCL 式 ECFP 引导对比并非 PerioGT 优势来源，监督 GIN+ECFP 已足够强。

**流程。** 40 次运行（8 模式 × 5 fold）全部完成，约 65 分钟，`training_final_run.log`/`training_output.log` 等 grep 未见 Killed/Traceback/OOM；`generate_diagnostics.py` 产出误差分布、foldwise 性能、λ 消融、训练曲线等图。当夜裁决（`TRAINING_REPORT.md` 自述）：对比学习开销不抵收益，建议后续转向编码器架构或表征学习——与 integration mem 把 SSL 预训练升格的方向一致。

### (B) SAPI

**目录** `src/5_SAPI/` ｜ **主脚本** `smiles_augmentation_experiment.py`（结果 `results/{final_results.json,config.json,results_k*_*.json,...}`；过程日志 `experiment_log.txt`【66MB / 110 万行，仅 grep/tail，禁整读】）

**实验目的。** 测 SMILES 增广能否解释 PerioGT 的优势（AI 当夜自述 PerioGT 较指纹基线有约 4.35 RMSE 优势、30.8 vs 38.78，代码未独立验证）。聚合物重复单元具对称性、同一分子存在多种合法 SMILES，故检验"训练期/测试期 SMILES 增广"是否是该优势的真正机制。

**模型架构。** 极简基线，**非图网络**——`TgPredictor` 为两层 MLP：Linear(2048,128)→ReLU→Linear(128,1)，输入 ECFP4（2048-bit Morgan r=2）。增广策略（代码三种）：① random SMILES（RDKit `MolToSmiles(canonical=False, doRandom=True)`，随机化原子序）；② rotated SMILES（`rootedAtAtom=start_idx`，换起始原子）；③ combined（random + rotated）。扫描增广倍数 k∈{1, 3, 5, 10}（k=1 即规范 SMILES、无增广基线），并测 test-time augmentation（对每个测试样本生成 k 条增广 SMILES、平均预测）。

**训练评估方式。** 同一份 `Tg.csv`，5-fold KFold(shuffle=True, random_state=SEED=42)；Adam lr=1e-3；batch 64；≤500 epochs（early stopping patience=20）；MSE 损失。无目标归一化（RMSE 为原始 Tg 单位）。

**结果。**（5-fold test RMSE，源自 `results/final_results.json` 及各 `results_k*_*.json`，与 `experiment_log.txt` 尾部 `FINAL RESULTS SUMMARY` 完全一致，并与 `mem_125` 数字吻合）：k=1 基线 **38.46 ± 1.14**（R² 0.8821）｜ k=3 38.64 ± 0.81 ｜ k=5 38.47 ± 0.69 ｜ k=10 **38.39 ± 0.63**（最优，R² 0.8826）｜ k=5 + test-time aug 38.74 ± 0.71（反而更差）。最大增广增益仅 **0.07 RMSE（0.18%）**，test-time aug 反损 0.28。**结论：SMILES 增广不能解释 PerioGT 优势——假设被拒**。增广只略降方差、不降偏差，部分条件反而更差。

**流程。** 5 条件 × 5 fold 全部跑完，66MB `experiment_log.txt` grep 未见 Traceback/Killed/OOM；`FINAL RESULTS SUMMARY` 表于 2026-03-20 02:06:51 打印，另出 `results_summary.png`/`training_curves.png`。当夜 `mem_125` 自述：综合 Day 3-5，周期性、长程、增广三条聚合物特异性机制全部失败。

**横向对照（代码自验，非 mem）。** 两实验同数据、同种子、同 5-fold——SAPI 的 ECFP + MLP 基线（38.46）竟比 FGCL 的 GIN + ECFP 监督基线（41.59）低约 3.1 RMSE，R² 也更高（0.882 vs 0.862）。即"固定 ECFP4 指纹 + 两层 MLP"在同划分上压制了 GIN 图编码器，进一步佐证本天核心结论：**ECFP 指纹表征是 Tg 性能的主导因素，叠加图编码器或对比学习均无收益**。

**当夜承上启下（均为 AI 当夜自述，代码未独立验证）。** `mem_127` 将 Day 3-5 三机制（周期性 +0.11 / 长程 −0.08 / 增广 +0.07，其中增广 +0.07 已由本天代码证实）合并定论——"三条聚合物特异性机制合计解释 <1% 间隙，periodicity-aware attention 系误称"；`mem_128` 据 GPIP 的 10× 数据效率与 PI1M（~100 万未标注聚合物）证据，把"自监督 3D/multimer 预训练"升格为解释 PerioGT 间隙的首要候选（RANK 1）；`mem_129` 据此重构假设空间——SSL 预训练 RANK 1（60%）、图 Transformer 降至 RANK 4（10%）、SMILES 增广剔除，研究方向从"架构归纳偏置"转向"表征学习/预训练"。这一否定式综合直接为 Day 6 催生 idea `mem_130`（PSCP，聚合物特异性对比预训练：官能团掩蔽 + 环状旋转不变 + ECFP-Tanimoto 引导对比），另有 `mem_131`（FPP，以 ECFP 为预测目标的预训练）、`mem_132`（CER，构象系综表征）、`mem_133`（CPML，Tg/Tm/Egc 跨性质元学习）同夜提出。

---

## Day 6 — PSCP（Polymer-Specific Contrastive Pretraining，聚合物专用对比预训练）

**目录** `src/6_day6_pscp/` ｜ **主脚本** `train_pscp.py` / `pscp_model.py`（+ 预训练 `pretraining_tasks.py`、工具 `polymer_utils.py`）｜ **对应记忆** experience `mem_136`、integration `mem_138/139/140`、idea `mem_141–144`

**参考/启发文献**（Day 6：当日 Learn 读取 2 篇＋当夜 integration 显式回溯引用 4 篇；来源 mem.json provenance）
- 【当日读取】**arXiv:2506.18194v1**（arXiv） · `mem_134` · [code](https://github.com/Intelligent-molecular-systems/Polymer-JEPA) —— This paper enables the following decisions for TopicState: (1) VALIDATES the self-supervised pretraining hypothesis - SSL on polymer graphs…
- 【当日读取】**arXiv:2106.04509v2**（arXiv） · `mem_135` · [code](https://github.com/illidanlab/MoCL-DK) —— Actionable for Day 6:
1. Immediate experiment: Implement MoCL-style contrastive learning for polymers using ECFP-derived global similarity…
- 【当夜引用】**full**（paper） · `mem_005` —— The paper introduces PI1M, a benchmark database of ~1 million synthetically generated polymers created via an RNN trained on ~12k monomers from PolyInfo, along with a new
- 【当夜引用】**2024 Geometry-enhanced pretraining on interatomic potentials**（paper） · `mem_085` · [code](https://github.com/cuitaoyong/GPIP) —— Actionable contributions: (1) Pretraining data generation strategy: Use classical MD with empirical force fields to generate abundant…
- 【当夜引用】**2017 SMILES Enumeration as Data Augmentation**（paper） · `mem_123` · [code](https://github.com/Ebjerrum/SMILES-enumeration) —— This paper enables the following decisions/experiments for TopicState: (1) VALIDATES SMILES enumeration as a viable data augmentation…
- 【当夜引用】**2021 MoCL Knowledge aware Contrastive Learning**（paper） · `mem_124` · [code](https://github.com/illidanlab/MoCL-DK) —— ACTIONABLE: Provides concrete methodology to test whether contrastive learning with domain knowledge explains PerioGT's performance. Key…

**实验目的。** Day 5 的裁决（`mem_128`）认定"自监督路线已升至 Rank1"，并由 idea `mem_130`（在 PI1M 聚合物语料上做图级对比预训练，再迁移到下游 Tg 回归）直接催生本天。PSCP 要回答两个问题：(1) 在 7170 样本的聚合物 Tg 小数据集上，能否靠 GIN 图编码器 + PI1M 对比预训练超过 MLP+ECFP4 基线；(2) 相对 SOTA 模型 PerioGT 的优势，"对比预训练 + 规范原子特征"这一条路径能解释多少。当夜设定的硬指标是 RMSE <35.0°C（基线约 37.7°C）。

**模型架构。** 输入是聚合物重复单元 SMILES 经清洗（见下）后构造的分子图，节点特征为 **155 维 one-hot 原子特征**（AI 当夜报告里写作 154/155 不一致，以代码为准：`run_efficient_experiment.py` 动态读出 `Node feature dim: 155`；由原子序数 118 + 度 7 + 形式电荷 7 + 手性 4 + H 数 5 + 杂化 + 芳香 2 + 归一化质量 1 构成；`polymer_utils.py` 版多一个 `is_star` 位为 159，但产出最终数字的 efficient/improved 脚本用 155 版）。主干为 `GINEncoder`：`Linear(155→256)` 节点嵌入 → 3 层 `GINConv`（每层 2 层 MLP+BatchNorm+ReLU，`train_eps=True`）→ 每层后 BatchNorm → `global_mean_pool` 得 256 维图嵌入（GINConv 不使用边特征，13 维边特征仅保留备用）。下游 `PredictionModel` 把 [GIN 图嵌入 256 ‖ 2048 位 ECFP4 指纹] = 2304 维拼接，经 `Linear(2304→256)→BN→ReLU→Dropout(0.2)→Linear(256→64)→BN→ReLU→Dropout→Linear(64→1)` 输出 Tg。预训练分支（`pretraining_tasks.py` + `pscp_model.py`）设计三类聚合物专用自监督任务：**(a) FGM 功能基团掩码**（`mask_ratio=0.15`，功能基团原子掩码权重 ×2.0，零化后用 `MaskPredictionHead` 预测原子类型，交叉熵）；**(b) CRI 循环旋转不变性**（重复单元 SMILES 生成 3 种原子序旋转，约束余弦相似度损失）；**(c) EGCL ECFP 导向对比学习**（以 Tanimoto 相似度 ≥0.7 选正样本、≤0.3 选负样本，每锚点 1 正 5 负）。实际 efficient/improved 主干脚本采用更直接的图增广对比：`augment_graph`（节点/边各 0.1 概率 dropout）生成两个增广视图 → `ProjectionHead(256→128→128)` → 对称 InfoNCE 损失。变体包括：baseline（MLP+ECFP4，无 GIN）、no-pretrain（GIN 随机初始化直接微调）、pretrain-10k / pretrain-100k（PI1M 上预训练后微调）、ensemble（多种子融合）。

**训练评估方式。** 数据为 `Tg.csv`（7170 个聚合物样本，Tg 范围 −139.0~495.0°C，均值 141.9 / 标准差 112.2°C）。**关键修复——SMILES 清洗**：原 `polymer_utils.canonicalize_polymer_smiles` 用 `replace('*','')` 会产生 `CC()C` 类非法分子，efficient/improved 主脚本改为 `replace('*','[H]')` 再去掉冗余 `([H])`，使有效图从 1230/7170（17.2%）跃升至 6971/7170（97.2%）（`efficient_log.txt` 实测 `Valid samples: 6971`）。**第二项关键修复——原子特征**：早期 `diagnostic_*` 阶段仅用 5 维归一化特征且 `pscp_results.csv` 残留 RMSE≈103 的未反归一化坏结果；efficient 起改为 155 维 one-hot，与 GIN 匹配后预测才进入正常量级。划分采用 5 折 `KFold(shuffle=True, random_state=42)`；improved 在此基础上加 3 个种子 [42, 123, 456] 做集成。Tg 在训练时做标准化（log 中 `Val RMSE = 0.34` 即归一化值），报告时反归一化回°C。优化器与超参：基线 MLP[2048→512→256→128] 用 Adam lr=0.001、wd=1e-5、batch=64、300 epoch、`ReduceLROnPlateau`、patience=30；对比预训练 efficient 用 30 epoch / batch=128 / lr=0.001 / 温度 τ=0.1，improved 延长到 50 epoch / batch=256 / `CosineAnnealingLR` 并搜温度 [0.05, 0.1, 0.2]（最优 0.05），aggressive 用 100 epoch 搜 [0.01, 0.02]（最优 0.02）；微调 efficient 为 200 epoch / batch=32 / lr=0.0005 / patience=30，improved 为 300 epoch / batch=32 / lr=1e-4 / `CosineAnnealingWarmRestarts` / patience=40 / 标签平滑；预测头 dropout=0.2。指标为 RMSE（°C）、MAE、R²，硬件为 NVIDIA RTX 2080 Ti（11.35 GB）。

**结果。** 全部数字取自结果文件（`efficient_results.json` / `improved_results.json` / `aggressive_results.json`，与三份报告吻合；早期 `pscp_summary.json` / `pscp_results.csv` 的 RMSE≈103 属未反归一化的坏点，已弃用）。efficient 四条件对比：**baseline (MLP+ECFP4) 37.76 ± 0.77 → no-pretrain 36.95 ± 0.63 → pretrain-10k 36.25 ± 0.56（efficient 最佳条件）→ pretrain-100k 36.30 ± 0.73**（100k 略差于 10k，说明预训练收益已饱和）。improved 在 pretrain-10k 基础上做温度搜索 + 3 种子集成：temp=0.05 得 35.97，temp=0.1 得 36.35，temp=0.2 得 36.16；取 temp=0.05 跑 seed 42/123/456 各 5 折，**最终集成 35.42 ± 0.07**（seed42 35.48±0.86、seed123 35.32±1.07、seed456 35.47±1.70），最佳单折 33.14（seed456 fold2），15 折中 8 折 <35.0。aggressive 更低温度未能再提升：temp=0.02 得 35.52±0.73、temp=0.01 得 35.59±0.82。**结论：最佳 RMSE = 35.42 ± 0.07，相对基线降 2.34°C（−5.9%~6.2%），但未达 <35.0 目标，差 0.42°C**。关于 PerioGT：`FINAL_ANALYSIS_REPORT.md` 以 PerioGT 报告的"相对 MLP+ECFP4 基线约 4.35 RMSE 优势"为参照，PSCP 解释了 2.34/4.35 ≈ **53.8% ≈ 54%**，剩余约 46%（~2.0 RMSE）归因于 PerioGT 的元素周期表嵌入等架构创新（4.35 这一 PerioGT 参考量来自 AI 当夜对论文的引述，代码未独立验证）。

**流程。** 当夜演进为一连串脚本变体：`run_pscp_simple.py` → `run_pscp_complete.py` → `run_full_pscp.py` → `run_efficient_experiment.py`（产出规范四条件 36.25）→ `run_improved_pscp.py`（产出集成 35.42，`IMPROVED_PSCP_FINAL_REPORT.md` 记录耗时约 6 小时，07:04–13:47）→ `run_aggressive_pscp.py`（35.52）→ `run_final_push.py`（5 种子×5 折、预训练 80 epoch、temp=0.05 的终极集成，seed1 均值 35.68；`STATUS_REPORT.md` 自述已跑 75+ 小时、卡在 seed2/5 fold2/5、日志因缓冲不刷新，未确认完成——AI 当夜自述，代码未独立验证）。失败轨迹（grep 自日志）：`experiment_final.log` 报 `RuntimeError: tensor a(256) vs b(159)` 维度失配，杀掉一次 combined-pretrain 尝试；`full_experiment.log` 报 `TypeError: DataLoader invalid type`；efficient/improved/aggressive 三份日志末尾均报 `TypeError: Object of type bool_ is not JSON serializable`（但结果已先落盘，`efficient_results.json`/`improved_results.json` 完整，`aggressive_results.json` 末尾被截断）。认知层面：当夜 `mem_138` 把范式从"堆预训练数据"转向"温度/增广/多种子集成"（AI 当夜自述，代码侧表现为 improved→aggressive→final_push 的温度与种子搜索）；`mem_139` 把 PerioGT 优势分解为"对比预训练可解释 54% + 周期表嵌入占 46%"的 framing（AI 自述的机制归因，未独立验证）；并据此为 Day 7 催生 idea `mem_141`（PTKGE，周期表/聚合物知识图嵌入方向）及 `mem_142–144`。

---

## Day 7 — PTKGE + iter2（双子实验：周期表知识图谱嵌入雏形 + 第二轮消融）

**目录** `src/7_day7_ptkge/` + `src/7_day7_ptkge_iter2/` ｜ **对应记忆** experience `mem_148`、integration `mem_150–155`、idea `mem_155–158`

**参考/启发文献**（Day 7：当日 Learn 读取 3 篇＋当夜 integration 显式回溯引用 2 篇；来源 mem.json provenance）
- 【当日读取】**arXiv:2510.08900v1**（arXiv） · `mem_145` · [code](https://github.com/Vencent-Won/Awesome-Literature-on-Few-shot-Molecular-Property-Prediction) —— This survey enables several actionable contributions: (1) Validates the representation learning paradigm shift confirmed in Day 6…
- 【当日读取】**arXiv:2308.08129v1**（arXiv） · `mem_146` —— Actionable insights for TopicState: (1) Pretraining improves relative ordering - useful for materials screening where ranking matters more…
- 【当日读取】**2024 DenseGNN universal and scalable deeper graph neural networks for high-performance property pred**（paper） · `mem_147` · [code](https://github.com/dhw059/DenseGNN) —— DenseGNN's DCN strategy could be tested for polymer property prediction to determine if dense skip connections help with small polymer…
- 【当夜引用】**arXiv:2506.18194v1**（arXiv） · `mem_134` · [code](https://github.com/Intelligent-molecular-systems/Polymer-JEPA) —— This paper enables the following decisions for TopicState: (1) VALIDATES the self-supervised pretraining hypothesis - SSL on polymer graphs…
- 【当夜引用】**arXiv:2106.04509v2**（arXiv） · `mem_135` · [code](https://github.com/illidanlab/MoCL-DK) —— Actionable for Day 6:
1. Immediate experiment: Implement MoCL-style contrastive learning for polymers using ECFP-derived global similarity…

### (A) PTKGE 雏形

**目录** `src/7_day7_ptkge/` ｜ **主脚本** `train.py` / `model.py` / `embeddings.py` / `ensemble.py`（辅：`dimension_ablation.py`、`per_element_analysis.py`、`visualize.py`）

**实验目的。** PTKGE（Periodic Table Knowledge Graph Embeddings，周期表知识图谱嵌入）要回答：把化学元素固有的物理化学属性作为"冻结先验"注入图神经网络，是否比让网络从数据中自学的元素嵌入更利于小样本聚合物 Tg 回归。其化学动机是周期律——同族元素（F/Cl/Br/I、C/Si、N/P）性质相近，物理属性可量化这种相似性，对训练集中出现稀少的 Br、I 等元素提供合理外推。它承接 Day 6 PSCP 提出的 idea `mem_141`（以领域知识强化元素表示），并把"先验 vs 学习"这一张力做成可控的四策略消融；同时叠加一条 PI1M（约 1 百万合成聚合物 benchmark）对比预训练支路，验证"大规模无标签聚合物 → 小样本 Tg"的迁移是否有效。

**模型架构。** 输入聚合物 SMILES（`*` 连接符替换为 `[H]`）经 RDKit 建分子图，节点为原子序数、边为双向键；ECFP4（半径 2 / 2048 位 Morgan 指纹）在线计算。原子特征先过 `PeriodicEmbedding`（`embeddings.py`）：每个元素取 8 维物理属性向量 `[电负性 χ, 原子半径 r(pm), 价电子数 v, 族 g, 周期 p, 电子亲和能 EA(kJ/mol), 第一电离能 IE(kJ/mol), 原子质量 M]`，做 z-score 归一化（缺失值填 0），再经 `Linear(embed_dim→256)` 投影进主干。四策略差异全在嵌入表：**physics** = 归一化物理属性、`requires_grad=False`（冻结先验）；**learnable** = 同样物理初始化但 `requires_grad=True`（可微调）；**random** = 正态随机初始化（报告 std≈0.1）、随 `requires_grad` 可训；**onehot** = 118 维 one-hot、冻结。投影层始终可训练。主干为 `GINEncoder`（`model.py`）：5 层 `GINConv`，hidden=256，每层 MLP(`Linear→BatchNorm→ReLU→Linear→BatchNorm→ReLU`)、`train_eps=True`，层间 `Dropout(0.1)`，全局池化得 256 维图嵌入。并行 `ECFP4Encoder`（MLP 2048→512→256）。融合层 `Concat→Linear(512→256)→BatchNorm→ReLU→Dropout(0.1)`；预测头 MLP 256→256→128→1。对比预训练用 `ContrastiveLoss`（InfoNCE）：以 batch 内 ECFP4 Tanimoto 相似度 >0.7 作正样本掩码、温度 τ=0.05，拉近相似聚合物表示。

**训练评估方式。** Tg 目标集 8964 样本（据 `PROJECT_REPORT.md`；注：当前仓库 `alldataset/Tg.csv` 仅 7170 行，8964 为本 run 当时的数据规模，见节末说明）；PI1M 下采样至 `min(100000, len)`=100k 作无标签预训练源。评估用 5-fold CV，`KFold(n_splits=5, shuffle=True, random_state=42)`，全局种子 42。预训练：20 epoch、batch 64、lr 1e-4、AdamW(weight_decay=1e-5)、CosineAnnealingLR、InfoNCE。微调：100 epoch（早停 patience=15）、batch 32、**差异化学习率**（encoder 1e-4、predictor head 1e-3）、AdamW(weight_decay=1e-5)、MSELoss。指标 RMSE/MAE/R²（Tg 单位°C，源于数据集）。

**结果。** 5-fold CV 四策略（单位°C，`results.json` 核实）：**physics 34.47±1.83（最优）/ MAE 22.84±1.00 / R² 0.9051±0.0110** > random 34.89±1.89 / 22.97±0.93 / 0.9026±0.0127 > onehot 35.69±2.56 / 23.68±1.64 / 0.8980±0.0156 > learnable 36.40±1.65 / 24.49±1.24 / 0.8944±0.0092。即物理冻结嵌入在 RMSE/MAE/R² 三项同时最优；learnable 虽方差最小但均值最差，被裁为"小数据过拟合"。方差排序 learnable(1.65)<physics(1.83)<random(1.89)<onehot(2.56)，physics 在性能与稳定性间最均衡。预训练收益：`PROJECT_REPORT.md` Y4.1 报告 physics 从零训练 35.2±2.1 → 对比预训练 34.47±1.83（+2.1%、方差下降），但该"从零基线 35.2"未落入 `results.json`（仅 AI 当夜自述，代码未独立保存该对照行，已标注）。**重要 caveat（代码核实）**：`train.py` main 中 `if strategy != 'onehot'` 才跑 PI1M 预训练，onehot 直接从零——故四策略表并非纯嵌入策略对照，onehot 行同时缺预训练，其劣势部分源自此。

**单划分 test 评估（run 内产出，2026-08 才补拷至本地归档）。** `plot_training_parity.py` 以 72/8/20 单划分（`random_state=42`）、加载各策略预训练 checkpoint（onehot 除外）、200 epoch 无早停、best-by-val 选 epoch 评估四策略，`PROJECT_REPORT.md` 记录 test RMSE：physics **34.39**（R² 0.907）/ random 34.42（0.906）/ onehot 35.93（0.898）/ learnable 36.21（0.896）——排序与幅度均与 5-fold CV（`results.json`）一致。产出 `figures/training_and_parity_{onehot,random,physics,learnable}_72_8_20.png`（左 loss 曲线、右 train/val/test parity；论文 Fig 4c/d 用 physics 版）与 `embedding_tsne.png`（论文 Fig 4b；`visualize.py` 对 physics 策略取 `atom_embedding.embedding`，即冻结的 8 维物理描述符，同族元素共现）。

**流程。** 一次 main 跑完四策略 × 5-fold 全网格（`strategies=['onehot','random','physics','learnable']`），逐策略落盘 `results_intermediate.json` 增量保存、末尾写 `results.json`；附带 `ensemble.py`（5 种子模型集成）、`dimension_ablation.py`（random 嵌入维 4/16/32/64）、`per_element_analysis.py`（按元素统计 RMSE）三类辅助分析与 `figures/`（元素嵌入 t-SNE、相似度热图）。日志 `training_log.txt`（24 MB）显示四策略均跑完无 killed。当夜裁决（如"learnable 过拟合、physics 最优、预训练有效"）属 AI 自述框架，其数字已由 `results.json` 独立核实、结论方向成立。

### (B) iter2 第二轮

**目录** `src/7_day7_ptkge_iter2/` ｜ **主脚本** `run_iteration2.py` / `model.py` / `embeddings.py`（辅：`experiment_log.txt` 12946 行、`results/`）

**实验目的。** 在 (A) 雏形裁定 physics 最优后，第二轮沿三条正交轴继续消融：(1) **集成**——多种子 physics 模型集成能否再降方差；(2) **逐元素**——physics 相对 random 的增益在哪些元素（尤其低频 Br/I）上体现；(3) **嵌入维**——random 嵌入维 {4,8,16,32,64} 是否影响表达力（回应 Y4.2"维数≠信息量"的论断）。本质是 (A) 结论的稳健性与可分解性检验。

**模型架构。** 复用 (A) 的 `PTKGEModel`/`GINEncoder`/`PeriodicEmbedding`（`model.py`、`embeddings.py` 与 (A) 同字节），不引入新结构；差异在评估脚本 `run_iteration2.py` 把"5-fold CV"换成"单次 train/val/test 三分 + 多种子重复"，以支持集成与按元素统计。

**训练评估方式。** 数据集 7170 样本（`experiment_log.txt` 明文 `Dataset size: 7170 samples`，即当前 `Tg.csv` 规模，与 (A) 的 8964 不同——见节末说明）。划分：先 `train_test_split(test_size=0.2)` 出 Test 1434，再在余下内 `test_size=0.125` 出 Val 717，得 Train 5019 / Val 717 / Test 1434（单次划分、非 K-fold）。单模型训练 150 epoch、batch 32、lr 1e-4、AdamW(weight_decay=1e-5)。计划落盘 `results/ensemble_results.json`、`per_element_results.json`、`dimensionality_results.json` 三份。

**结果。** `results/` 目录确认为空——三份 JSON 均未生成，无任何保存结果。但需如实交代：`experiment_log.txt`（去掉约 12906 行 RDKit `DEPRECATION WARNING` 后仅余 40 行实质内容）显示 run 并非全空。TASK 1（集成）已启动：**Model 1/5（seed 42）完整跑完 150 epoch，Val RMSE 34.6389、Test RMSE 34.5463**；随后进入 Model 2/5（seed 142），日志在其 Epoch 1/150 处戛然而止——即 run 被中断于 TASK 1 第 2 个模型的第 1 个 epoch。TASK 2（逐元素）、TASK 3（嵌入维）从未启动，集成指标（需 5 模型全齐）未计算。故 iter2 唯一可信数字是 Model 1 的 Val 34.64 / Test 34.55（单模型、单划分，与 (A) 的 5-fold 34.47 不可直接比较），无聚合/集成/逐元素/维数消融产出。旧文"iter2 results/ 为空、未纳入头条数字"的判断在"未产出保存结果"层面成立，但应补充"日志存在 Model 1 部分训练轨迹"这一事实。

**流程。** 计划三任务串行（TASK 1→2→3），实际只完成 TASK 1 的 1/5 模型即被中断（killed），未达任何 `json.dump` 保存点，故 `results/` 空、未纳入任何头条数字。当夜未对 iter2 作出新裁决（无完整结果可裁）。

**数据规模说明。** Day 7 雏形 (A) 跑在 Tg 8964 + PI1M 下采样 100k；而 iter2 (B) 与最终 optimize 阶段使用 Tg 7170 + PI1M 全量 988795，数据规模与划分方式（5-fold CV vs 单次划分）均不同。因此 Day 7 雏形的 34.47±1.83 与 optimize 头条数字不可直接对比。

---

## Optimize 阶段 — PTKGE 最终头条模型（协议对齐 + EA/IP 连通共聚物修复）

> Optimize 阶段为 Day 7 PTKGE 的协议对齐精修，其启发文献沿用 Day 7 节（见上）。

**目录** `src/optimize/` ｜ **主脚本** `run.py`/`run_all.sh`/`train.py`/`pretrain.py` ｜ **模型代码** `model.py`/`model_copolymer.py`/`embeddings.py`/`data_utils.py`/`config.py`

**实验目的。** 这是 ab2 的最终/最佳头条模型，在 Day 7 成形的 PTKGE 雏形（线索见 Day 7 `mem_148–155`，AI 当夜自述，代码未独立验证）之上做两件事。(1) 协议对齐：decoupled split/init、原始单位指标、多种子 ensemble、显式不做 best-seed/best-split 选择，给出干净可复现的复现结果（见 `README.md`/`plan.md`）。(2) 针对 EA/IP 的根因修复：把此前把共聚物表示为两个单体 disjoint union（A、B 互不连边、GNN 无法跨界面传递消息、无法建模决定共聚物性质的组成界面效应）改为单条连通 A-B 共聚物图，使电子亲和能 EA（Electron Affinity）与电离能 IP（Ionization Potential）的 RMSE 显著下降（见下文结果与 `data_utils.py` 注释）。

**模型架构。** 输入：单聚物为 p-SMILES 字符串；共聚物为 `(smiles_A, smiles_B, rA, rB, type)`。预处理：单聚物 `polymer_smiles_to_mol` 将 `*` 替换为 `[H]` 再解析；共聚物 `build_connected_copolymer_mol` 用 `Chem.CombineMols` 合并后 `AddBond`(A 出口邻原子 → B 入口邻原子)、`ReplaceAtom` 把两个外端 star 盖成 H、`RemoveAtom` 删掉两个内端 star、最后 `RemoveHs`，得到单条连通分子。`mol_to_atom_graph` 只取原子序数 z 为节点特征、键为双向 `edge_index`，无任何键/原子额外特征（与预训练编码器一致）。各模块（全部从 `model.py`/`embeddings.py`/`model_copolymer.py` 核实）：
- **PeriodicEmbedding**（`strategy="physics"`）：`nn.Embedding(119, 8)`（MAX_ATOMIC_NUM 118 +1 padding），8 维物理属性（电负性、共价半径、价电子数、族、周期、电子亲和能、第一电离能、原子质量），按全元素表 z-score 归一化（0 值按 nan 处理），`physics` 策略 `requires_grad=False` 冻结；再 `nn.Linear(8, 256)` 投影到 hidden。
- **GINEncoder**（5 层，hidden 256）：每层 `GINConv`（MLP = 2×[Linear(256,256)+BN+ReLU]，`train_eps=True`）→ `BatchNorm1d(256)` → ReLU → `Dropout(0.1)`；末尾 `global_mean_pool`。
- **ECFP4Encoder**：MLP `Linear(2048,512)+BN+ReLU+Dropout(0.2)+Linear(512,256)+BN+ReLU`（Morgan radius 2 / 2048-bit；单聚物在 p-SMILES 上算、共聚物在连通分子上算）。
- **Fusion**：concat[graph_emb(256), fp_emb(256)]=512 → `Linear(512,256)+BN+ReLU+Dropout(0.1)`。
- **单聚物 Predictor**：`Linear(256,256)+BN+ReLU+Dropout(0.1) → Linear(256,128)+BN+ReLU → Linear(128,1)`。
- **PTKGECopolymer**：复用同一 PTKGEModel（GIN+ECFP4+fusion），把 fusion 输出(256) 与 `extras` 拼接后过独立 head：`Linear(256+extras_dim,256)+BN+ReLU → Linear(256,128)+BN+ReLU → Linear(128,1)`；`extras = [rA, rB, type-one-hot]`。
- **参数量**（取自 `logs/gpu0.log`/`gpu1.log` 中 `params=` 字段，非自算）：单聚物 2,081,286；共聚物 2,082,566。两者差 1,280 = 256×5 ⇒ 反推 `extras_dim=5`（即 2 + 3 个 type 类别），与 `data_utils.py` `extras_dim = 2 + len(types)` 一致并已由参数差独立验证。

**训练评估方式。** 数据（从 `src/alldataset/*.csv` 核实行数）：Tg 7170 / Tm 3651 / EA 42966 / IP 42966（**optimize 阶段不跑 Egc**，与 4 性质头条口径一致）；预训练 PI1M 988795（`PI1M_SAMPLE=990000` > 988795，永不触发采样 ⇒ 全量）。划分（`config.py`：`MULTI_SPLIT_DATASETS=["tg","tm"]`、`GIVEN_SPLIT_DATASETS=["ea","ip"]`、`SPLIT_SEEDS=[42,71]`、`INIT_SEEDS=[42,71,2026]`）：Tg/Tm 属多划分，`SPLIT_SEEDS=[42,71]` 各做一次 72/8/20（`test_size=0.20`，剩余再 `test_size=0.10` 出 val）；EA/IP 属官方给定划分（`ea_split.pkl`/`ip_split.pkl`，单划分）。每个 (split, init) 训练一次 ⇒ Tg/Tm 各 2×3=6、EA/IP 各 1×3=3，**单次完整 run 共 18 个训练 job**；为量化硬件/cuDNN 非确定性，**同一 18-job 矩阵独立重复 3 次（`results/run1|run2|run3/`），合计 54 个训练 job**。各 run 实际计数（从 `results/run1/*.json` `counts` 核实，train/val/test 之和恰等于数据集总量 ⇒ 0 dropped）：Tg 5162/574/1434、Tm 2628/292/731、EA 30935/3437/8594、IP 30935/3437/8594。特征化一次性缓存于 `cache/{ds}_v1_full.pkl`（`CACHE_VERSION="v1"`）。优化器与损失（`config.py`/`train.py`/`pretrain.py`）：单聚物用差分 lr 的 AdamW（encoder 5e-5 / head 1e-4，wd 1e-5，`clip_grad_norm_` 1.0），batch 32；共聚物从零训练 AdamW（lr 1e-3，wd 1e-5，grad_clip 1.0），batch 64；两者均 epochs 300、patience 30、`MSELoss`。目标按 train mean/std 归一化、模型在归一化空间预测、指标反算回原始单位；val 原始单位 RMSE 早停，best checkpoint 在 test 上仅评估一次。预训练：InfoNCE 对比（ECFP4 Tanimoto 相似度为正例），AdamW lr 5e-5 wd 1e-5、`CosineAnnealingLR(T_max=30)`、30 epochs、batch 64、温度 τ=0.05。指标 RMSE/MAE/R² 均原始单位；ensemble 先在每个 split 内对 3 个 init-seed 预测做 mean/median，再对 Tg/Tm 的 2 个 split 取 mean±std（EA/IP 仅单一官方 split），该 18-job 矩阵再独立重复 3 次（run1/run2/run3），全程无 best-seed/best-split 选择。

**结果。** RMSE 均为原始单位。两层聚合：(1) **run 内**——每个 (split, init) 训练一次，先在每 split 内对 3 个 init-seed 集成（ens_mean/ens_median），再对 Tg/Tm 的 2 个 split 取 mean±std（EA/IP 仅单一官方 split）；(2) **跨 3 次独立重复 run（`run1/run2/run3`，相同种子与配置，方差来自硬件/cuDNN 非确定性）**——对每次 run 的头条 ens_mean 再取 mean±std。头条表（跨 3 次 run，来源 `results/aggregated_runs.json`/`AGGREGATED_RUNS.md`，± 为跨 3 次重复的 std）：

| dataset | n_test | single RMSE | ens_mean RMSE | ens_median RMSE |
|---|---|---|---|---|
| Tg | 1434 | 32.2624 ± 0.0575 | 31.4136 ± 0.0171 | 31.5683 ± 0.0124 |
| Tm | 731  | 51.1314 ± 0.3994 | 49.6328 ± 0.2985 | 49.8573 ± 0.1496 |
| EA | 8594 | 0.0466 ± 0.0003  | 0.0441 ± 0.0002 | 0.0446 ± 0.0002 |
| IP | 8594 | 0.0364 ± 0.0006  | 0.0340 ± 0.0003 | 0.0343 ± 0.0003 |

（optimize 阶段不跑 Egc。）代表单次 run（`run1/results_table.json`，± 为跨 split 的 std）per-split 细节：**Tg** sp42 single 33.7486（init_std 0.6286）/ ens_mean 32.8508 / ens_median 32.9042；sp71 single 30.7426（0.1932）/ 29.9935 / 30.1996。**Tm** sp42 single 51.6083（0.7307）/ 49.9018 / 50.3231；sp71 single 50.5120（1.1980）/ 49.4113 / 49.3922。**EA** given single 0.0467（0.0002）/ 0.0442 / 0.0445。**IP** given single 0.0369（0.0011）/ 0.0340 / 0.0343。最佳单 run 仍为 `tg_sp71_init2026`：test_rmse 30.5942（≈30.59）、test_r2 0.9234（≈0.923）。EA/IP 在改用连通 A-B 图后 ens_mean 落到 0.0441 / 0.0340（原始单位）。与 PerioGT SOTA Tg≈30.8 对比：本模型 Tg ens_mean 31.41、最佳单 run 30.59 已与之同量级（PerioGT 30.8 出处未在本 run 内找到，见下）。

**流程。** `run_all.sh` 两 GPU 并行跑单次 18-job 矩阵：GPU0 跑 `tg`(6)+`ea`(3)=9、GPU1 跑 `tm`(6)+`ip`(3)=9（无 Egc）；该 18-job 矩阵独立重复 3 次（`run1/run2/run3`），`aggregate_runs.py` 跨 3 次 run 汇总出 `results/AGGREGATED_RUNS.md`/`aggregated_runs.json`，每次 run 内 `collect_results.py` 汇总出 `results/runK/RESULTS.md`/`results_table.json`。注：当前快照下 `logs/gpu*.log` 未随结果留存，故不对 DONE 条数作断言（结果 json 齐全且跨 3 run 一致即视为完整）。多种子协议 split{42,71}×init{42,71,2026} 解耦（`data_utils.py` 注释明言"decoupled from training randomness"）。组件消融集中在 Day 2-7，optimize 阶段模型/预训练配方 frozen（`config.py` `MODEL_DEFAULTS` 与 `pretrain.py` 不再变动），本阶段不再做组件消融，只做协议对齐的干净复现 + EA/IP 连通图这一项定向修复。

**不确定项（已尽量反推核实）。** (1) **split-seed 旧文档残留（已厘清）**：`plan.md` 与 `data_utils.py` 顶部 docstring 仍写 "split_seed in {42,71,2026}"，但 `{42,71,2026}` 实为 `INIT_SEEDS`；权威为 `config.SPLIT_SEEDS=[42,71]`，产物目录确为 `sp42/sp71`（无 sp2），以 config 与产物为准。(2) **Day 7 雏形 vs optimize 超参不一致**：Day 7 雏形记忆中的预训练为 20ep/lr1e-4（AI 当夜自述，代码未独立验证），而本目录 `pretrain.py` 实际为 30ep/lr5e-5/CosineAnnealing；以 `pretrain.py` 为准。(3) **extras_dim**：任务清单原列为"推断=2+3=5 但未反推验证"——本次已由参数差反推核实，2,082,566−2,081,286=1,280=256×5 ⇒ `extras_dim=5`、`n_types=3`，与代码一致。(4) **PerioGT SOTA Tg≈30.8**：未在本 run 目录/`RESULTS.md`/config 内找到其出处与文件，仅作为外部对照引用。

---

## 7 天轨迹总结

ab2 是一条"系统性证伪 → 范式转向 → 先验+预训练成功"的因果链，核心科学故事可概括为：

1. **Day 1（定基线 + 确立指纹主导命题）**：MLP+ECFP4(38.78) ≫ GCN(45.69)，证明子结构指纹携带了图拓扑漏掉的信息；Data_Centric_Foundation 被裁 REJECTED，催生"指纹+图融合"主线。
2. **Day 2–4（系统性证伪三条聚合物特异性归纳偏置）**：
   - Day 2 FP-PGNN(37.04)：指纹 +5.64 ≫ 周期边 +0.29（不显著）→ 周期边无效；
   - Day 3 PIPGA(38.99)：Matformer 周期不变 + 指纹 +25.72、softmax attention +2.48 反劣 → 周期性编码路径否定（周期不变性测试甚至未实际跑）；
   - Day 4 LEMP(35.02/35.15)：长程(FFT/voxel/Ewald) 贡献 **−0.08**（负）→ 长程路径否定，指纹 +11.58 完成三角验证。
   三天共同确立：**Tg 是局部化学（指纹）问题，而非周期/长程空间问题**。
3. **Day 5（证伪剩余两条表征机制）**：FGCL 对比学习(41.65 vs supervised 41.59) 无增益、SAPI SMILES 增广(+0.07) 无增益；同数据同种子下 ECFP+MLP(38.46) 甚至压制 GIN+ECFP(41.59)。五条机制（周期边/周期不变/长程/对比学习/增广）全部证伪后，假设空间决定性转向自监督预训练。
4. **Day 6（范式转向成功）**：PSCP 聚合物专用对比预训练，关键修复 SMILES 清洗(`*`→[H]，有效分子 17%→97%)与 155 维 one-hot 原子特征，baseline 37.76 → 集成 **35.42±0.07**，解释 PerioGT 优势约 54% → 表征学习范式成立。
5. **Day 7（先验+预训练双轮驱动）**：PTKGE 把周期表 8 维物理属性作为冻结先验注入 GIN，physics **34.47±1.83** 最佳(>random/onehot/learnable)，PI1M 对比预训练 +2.1% → 物理先验 + 对比预训练有效，成为头条模型直接前身（iter2 第二轮未产出保存结果）。
6. **Optimize（头条定型）**：在 Day 7 雏形上做协议对齐（decoupled split/init、原始单位、多种子 ensemble、无 best-seed）+ EA/IP 连通共聚物图修复，27 runs 全 DONE，Tg **ens_mean 31.41 / 最佳单 run 30.59**（≈ PerioGT SOTA 30.8），EA/IP 因连通图修复达极低 RMSE（R²>0.99）。

一句话：**前 5 天用消融把"聚合物特异归纳偏置"逐条排除、坐实 ECFP 指纹主导；Day 6–7 转向"对比预训练 + 周期表物理先验"两条真正有效的表征学习路径；Optimize 阶段协议对齐 + 共聚物图修复，把 Tg 推到 SOTA 量级。**

---

## 主要参考文件

- 逐日代码与结果：`src/1_05_data_centric/`、`src/2_day2_fp_pgnn/`、`src/3_03_pippga/`、`src/4_04_lemp/`、`src/5_05_fgcl/`、`src/5_SAPI/`、`src/6_day6_pscp/`、`src/7_day7_ptkge/`、`src/7_day7_ptkge_iter2/`、`src/optimize/`
- 数据：`src/alldataset/`（Tg/Tm/Egc/ea/ip/PI1M）、`datasetinfo.md`
- run 级统计：`log_analysis.md` / `log_analysis.json`、`full.log`、`mem.json`（含 `mem_001` TopicState 与逐日 experience/integration/idea）
