# ab4（GLM-5，DAY-ONLY）逐天实验轨迹 — 每天一个新架构（无夜巩固）

## 概览与约定

- **run 身份**：ab4 / 基座 LLM = GLM-5 / **DAY-ONLY 模式**——night 阶段（internalize 巩固 + associate 生成 idea）**已消融**，只保留白天的论文阅读 + 假设 + 实验执行循环 / 共 7 天 / **无独立 optimize 目录**（Day7 PolyPerio-DenseGNN 为 terminal/末天模型）。mem.json 共 107 节点（99 knowledge + 7 experience + **1 idea + 0 integration**）——**0 个 integration 节点**是 day-only 的关键结构特征（对照 ab2/3/6 full day-night 有 20–30 个 integration 节点），即夜巩固的 provenance 边完全缺失。
- **判别关键**：day-only 每天**也换架构**（探索机制本身不是判别量），真正的判别是"连续实验是否因果衔接、是否在前一天 finding 上累积"——本 run 实验间衔接较弱、Day2/3 还被 API 速率限制阻塞。
- **最终最强模型**：**全 run 真实最优是 Day1 PPFN 的 5-集成 34.14**（非 terminal）；terminal/末天最佳是 Day7 PolyPerio-DenseGNN angular 变体 36.98±1.70。
- **主线轨迹**：每天一个全新架构（PPFN→RIT→IAPT→IAPT v2→GraphMAE+MoCL→PolyKAN-SSM→PolyPerio-DenseGNN），Day2/3 被 API 429 阻塞（Day3 完全未执行），Day1 PPFN 3D conformer 5-集成 34.14 为全 run 最佳，Day4 IAPT v2 集成 35.08、Day5 预训练 35.60、Day6 PolyKAN-SSM 38.17、Day7 angular 36.98——**无任何模型达 SOTA 30.8**。
- **任务范围**：所有 day 实验均**只预测 Tg**；Tm/Egc 仅来自辅助 GAT baseline（单种子 seed=42），EA/IP 完全未涉及。
- **单位约定**：Tg RMSE 单位 K（开尔文温度差），量纲同 °C。
- **可信度约定**：所有数字以**代码/结果文件为权威**；mem.json 的 AI 当夜自述仅作线索，凡其独有、代码未佐证的标"（AI 当夜自述，代码未独立验证）"。特殊缩写在首次出现处展开。

### day→实验目录映射

| Day | src/ 目录 | 主题 | 代表结果 |
|---|---|---|---|
| 1 | `1_day2_ppfn` | PPFN 分层 GNN（3 轮迭代，3D conformer） | **v3 5-集成 34.14（全 run 最佳）** |
| 2 | `2_day2_rit_polymer` | RIT 重复不变 Transformer | 5-seed 集成 **43.14**（per-seed 44.79±0.39） |
| 3 | `3_03_iapt` | IAPT 强化感知 Transformer | **未执行（37×429），目录空**（Day4 本即 Day3 的 IAPT，作图复用 Day4 值） |
| 4 | `4_iapt_v2_intensive_aware` | IAPT v2（GINE+线图+等变+VN+多尺度池化） | improved 5-集成 35.08 |
| 5 | `5_day5_pretraining`、`5_day5_pretraining_v2` | GraphMAE+MoCL 预训练 | v1 35.60 / v2 36.14 |
| 6 | `6_day6_polykanssm` | PolyKAN-SSM（Fourier-KAN+SSM+非共价边） | v5b 38.17±1.46 |
| 7 | `7_day_7_polyperio` | PolyPerio-DenseGNN（4 变体） | angular 36.98±1.70（terminal） |

> 目录名常带跨天字样（如 `1_day2_ppfn` 前缀 1 归 day1），以前缀数字为权威 day 归属。

---

## Day 1：PPFN 分层 GNN — 3 轮迭代，3D conformer 5-集成 34.14（全 run 最佳）

**参考/启发文献**（Day 1：当日 Learn 读取 4 篇；来源 mem.json provenance）
- 【当日读取】**FragNet GNN Interpretability**（paper） · `mem_073` —— Actionable insights for polymer deep learning: (1) Hierarchical graph construction (bond→atom→fragment) could be adapted for polymer repeat…
- 【当日读取】**PeriodicGraphs Polymers**（paper） · `mem_074` —— Provides a validated physics-inspired representation (periodic graphs) that directly encodes polymer periodicity, achieving near-SOTA…
- 【当日读取】**2024 LONG-SHORT-RANGE MESSAGE-PASSING A PHYSICS-INFORMED FRAMEWORK TO CAPTURE NON-LOCAL INTERACTION**（paper） · `mem_075` · [code](https://github.com/liyy2/LSR-MP) —— LSR-MP provides an actionable framework for improving polymer property prediction models: (1) The BRICS fragmentation approach can be…
- 【当日读取】**2025 Periodicity-aware deep learning for polymers**（paper） · `mem_076` —— PerioGT establishes a strong baseline for Tg prediction (30.8 RMSE) that defines the performance target for subsequent research. It…

### 实验目的
Day 1 用自研分层图神经网络 **PPFN（Polymer Periodic Fragment Network，聚合物周期片段网络）** 预测 Tg，尝试击败 SOTA RMSE 30.8。一天内完成三轮主迭代：v1 复杂分层版→v2 简化周期边版→v3 加入 3D conformer 几何特征版，每轮以"Test RMSE<30.8"为成功判据。

### 模型架构
**(A) v1 复杂分层 PPFN**（`model.py`）：三层分层——(1) 原子层 4 层 GINEConv（带边特征的图同构网络）；(2) 片段层 2 层 TransformerConv（注意力类卷积，README 泛称 GAT，节点为 BRICS 断裂得到的化学片段）；(3) 长程二分模块 `LongRangeBipartiteModule`（2 层多头 cross-attention 4 头，在全原子与全片段二分图上捕捉非局部依赖）；预测头全局均值池化(768 维拼接)→MLP 768→512→256→1。节点 27 维（18 原子 one-hot+度+5 杂化+芳香+电荷+环）、边 6 维、片段 31 维；聚合物图由 p-SMILES 生成 trimer（3 RU）并加周期边（相邻 RU 边界 * 哑原子相连）。**实测参数 5,350,150**（README 写 3,641,349 为过时规划，弃用）。**(B) v2 简化 PPFN**（`model_v2.py`）：砍掉片段注意力和长程模块，仅 6 层 GINEConv+残差+LayerNorm+注意力全局池化，聚焦周期边，参数 978,953。**(C) v3 PPFN-v3-3D**（`model_v3_3d.py`）：v2 基础上引入 3D conformer 几何特征——RDKit ETKDG+MMFF 生成 3D 构象，抽取键长/3D 原子对距离(cutoff 内)/二面角/键角+12 个全局统计量（键长/角/二面角均值方差、体积、回转半径、密度、形状）；边特征 6→8 维（加 bond_length、3d_distance），节点 27 维不变+12 维全局；几何注意力（3D 距离调制消息传递）、门控注入全局特征、多尺度池化(mean+max+attention)；base_256(hidden 256,4 层)≈2.56M、wide_512(hidden 512,6 层)≈9.66M。

### 训练评估方式
Tg 7170，72/8/20（5162/573/1435），seed=42。v1 AdamW(lr 1e-3, wd 1e-5) batch 64 max 500 epoch；v2 lr 5e-4 wd 1e-4 6 层 batch 64 max 300 patience 30；v3_3d AdamW(lr 1e-3, wd 1e-5) ReduceLROnPlateau(factor 0.5 patience 15) **HuberLoss(delta 1.0)** batch 32 label StandardScaler 归一化 max 300 patience 30。**5-集成做法**：v3_3d 用 5 个 base_256 模型靠 RNG 自然推进产生不同初始化、训练后对测试预测取算术平均（v4/run_ensemble 则显式用 seeds 42/142/242/342/442）。

### 结果
核心指标（测试集，代码/JSON 证实）：**v1 复杂分层** Best val 35.61、Test **37.36**（MAE 24.21，R²0.8917，5.35M 参数）；**v2 简化（周期边消融）** 有周期边 Test **37.76**/无周期边 38.73→周期边贡献 **+2.5%**（RMSE 降 0.97，`ablation_results.json` pct_improvement=2.5064），参数 978,953（简化后参数少 5.5× 但 RMSE 与 v1 持平）；v2 wide_512 39.21、v2 5-集成 36.74；**v3_3D** base_256 单模型 Test **36.71**（val 33.20，R²0.8954）、wide_512 单模型 37.43、**5-集成 Test 34.14**（MAE 21.79，R²0.9095，`ensemble_rmse=34.1419`）——**全 Day/全 run 最佳**。3D 特征贡献：v2 集成 36.74→34.14 降 **7.1%**（单模型 39.21→36.71 降 6.4%）。距 SOTA 30.8 仍差 3.34，未达。**trimer conformer 失败率 ~79%（代码证实）**：成功率 train 21.0%(1086/5162)/val 21.3%(122/573)/test 20.6%(296/1435)，>100 原子分子跳过 conformer 生成用 2D 回退。

### 流程
时间线（日志 timestamp）：先实现 v1 复杂分层（01:34 起，154 epoch 后 Test 37.36 未达 SOTA）→反思发现复杂长程/片段注意力性价比低→v2 大刀阔斧简化到 978K 聚焦周期边（03:42–03:49 跑周期边/无周期边对照验证 +2.5% 小幅正贡献）→试改进版 virtual-node+Huber+cosine/plateau（cosine 44.39、plateau 41.42 均弃）+宽深配置（wide_512 39.21）+多 seed 集成（seeds 42/142/242/342/442，5 模型 36.74）→v3 引入 3D conformer（05:44 起，单模型耗时 2–5 s/聚合物）base_256 单 36.71/wide_512 37.43/**5-集成 34.14**。结论：3 轮迭代最佳 34.14 未达 SOTA 30.8，关键发现 (i) 3D 几何特征 +7.1% 最大增益但仍不够、(ii) 周期边仅 +2.5%、(iii) trimer conformer 生成 ~79% 失败是 3D 路线主要瓶颈。**纠正**：README 把片段层称 GAT，代码实为 TransformerConv；v1 参数 5.35M 非 README 的 3.64M。

---

## Day 2：RIT 重复不变 Transformer — 5-seed 集成 Test 43.14（per-seed 44.79±0.39）

**参考/启发文献**（Day 2：当日 Learn 读取 2 篇；来源 mem.json provenance）
- 【当日读取】**2024 A Euclidean transformer for fast and stable machine learned force fields**（paper） · `mem_078` —— Actionable insights for polymer deep learning: (1) For Tg prediction, 3D equivariant representations may improve extrapolation to novel…
- 【当日读取】**2026 Learning Repetition-Invariant Representations for Polymer Informatics**（paper） · `mem_079` —— Enables construction of multimer baseline models with guaranteed repetition invariance. Provides a principled approach (3 RUs minimum) for…

### 实验目的
设计 **RIT（Repetition-Invariant Transformer，重复不变 Transformer）**，联合三点改进：单体构象几何特征 + 重复不变聚合（对重复单元 RU 多份拷贝做 max 聚合）+ Transformer 长程建模，辅以 Tg 区间分类（<100/100–200/>200 K）多任务与 bias-correction 偏置修正，目标击败 SOTA 30.8（此前最佳 34.14）。

### 模型架构
`RITModel`（hidden 256，num_ru=3，num_classes=3）forward：(1) 输入投影节点 30→256、边 7→256；(2) 3 个可学习 RU 位置嵌入；(3) GINE 编码器 5 层 GINEConv（eps=0, train_eps, MLP=Linear-ReLU-Linear，**同一套权重对 3 个 RU 拷贝共享**、每 RU 只额外加自己位置嵌入，每层 LN+GELU+残差）；(4) **重复不变聚合**=在 RU 维度 torch.max（max pool 跨 3 RU 拷贝，对 RU 排列/重复次序不变）；(5) Transformer 长程模块 4 层 8 头 FFN=4×hidden pre-LN GELU（带可学习位置编码，>100 节点图线性插值扩展）；(6) 多头注意力池化(4 头)→图级向量，与 16 维全局构象统计投到 256 拼接成 512；(7) 多任务头 Tg 回归+区间分类(3 类)；(8) bias correction 把 combined+区间概率喂 bias_net 得偏置 tg_pred=tg_base+bias。数据侧：p-SMILES `*`→`[H]`→RDKit 单体 3D 构象（ETKDGv3 失败回退 ETKDG，MMFF 失败回退 UFF），抽重原子键长/键角/二面角+16 维全局统计（Rg/体积/MW/nrot/nrings/TPSA/logP/Fsp3 等），节点 30 维、边 7 维。**注意：数据层只建单体图，不构造真正 trimer；"多 RU"仅靠模型层 3 份位置嵌入实现。**

### 训练评估方式
按 Tg 区间（<100/100–200/>200 K）分层 72/8/20 = 5160/590/1420；AdamW(lr 1e-3, wd 1e-5)+ReduceLROnPlateau(factor 0.5 patience 10) max 300 epoch patience 30；损失 MSE+0.1×CE；grad clip 5.0；**5 集成 seeds 42–46**（seed0–seed4）每个存 best_model_seed{i}.pt，最终对测试预测取算术平均。

### 结果
**5-seed 集成已完成**（seeds 42–46，72/8/20 = 5160/590/1420，按 Tg 区间分层）。结果见 `results/results.json` + `RESULTS.md`（2026-07-31 补入本地 archive；此前本地缺失，旧文本误判为"blocked"）。**集成（5 模型预测算术平均）Test RMSE 43.14 / MAE 28.91 / R² 0.8541**（n_test=1420）。逐 seed Test RMSE：seed42 44.87 / seed43 44.58 / seed44 44.18 / seed45 45.37 / seed46 44.93 → **per-seed mean 44.79 ± 0.39（ddof=0）**；集成相对最佳单模型约降 1 K、R² 升 ~0.01。分区间 RMSE：Low(<100 K, n=559) 38.73、Mid(100–200 K, n=377) 41.31、High(>200 K, n=484) 48.98（高温区误差显著更大）。结论：RIT 显著劣于 Day1 PPFN 5-集成 34.14 与 SOTA 30.8——重复不变 max 聚合 + Transformer 长程 + 多任务/偏置修正未能提升，反而比 Day1 简单 GINE 基线差约 9 K。注：按 ab4"选高 + 逐种子口径"作图时，本天取 per-seed **44.79 ± 0.39**（高于集成 43.14）。

### 流程
main() 加载 Tg.csv→建 PolymerDataset（含单体构象生成最耗时）→按 Tg 区间分层划分→train_ensemble(num_seeds=5) 逐 seed→写 results.json（含 ensemble + individual_members）→出 rmse_comparison.png→与 30.8/34.14 对比。Day 2 在代码层面是完整的 RIT 设计（重复不变 max 聚合 + Transformer 长程 + 多任务 + 偏置修正），5 seed 全部跑完落盘；结论是相对 Day1 PPFN 退步，重复不变/长程/多任务组合本天无效。

---

## Day 3：IAPT 强化感知 Transformer — 完全未执行（37×429，目录空）

**参考/启发文献**（Day 3：当日 Learn 读取 2 篇；来源 mem.json provenance）
- 【当日读取】**2025 Learning non-local molecular interactions via equivariant local representations and charge equi**（paper） · `mem_081` —— CELLI provides an actionable contribution for polymer property prediction: (1) Architecture pattern: integrating physics-based priors…
- 【当日读取】**2024 Many-Body Function Corrected Neural Network with Atomic Attention (MBNN-att) for Molecular Prop**（paper） · `mem_082` —— Actionable insights for Tg prediction: (1) Atomic attention mechanism provides principled approach for intensive property…

### 实验目的
Day 3 唯一实验 **IAPT（Intensive-Aware Polymer Transformer，强化感知聚合物 Transformer）**，由 AI 设计但**未执行**。核心假设：Tg 是 **intensive property（强度性质，温度，与体系大小无关）**，而标准 sum/mean pooling 隐含 extensive property（广延性质，随系统尺寸线性增长）假设，会产生系统偏差（低温区过预测、高温区欠预测）；提出用 **softmax attention pooling** 提供尺寸不变聚合纠偏，叠加电荷平衡层捕获长程静电，目标击败 SOTA 30.8（Day1 最佳仅 34.14）。

### 模型架构
设计文档（mem_083）完整规定 7 模块，但**未在任何 .py 文件实现**（目录 `3_03_iapt/` 完全为空，无 model.py/train.py）：(1) 数据 3 RU（GRIN 论证 3 RU 为重复不变最小数量），仅为单体生成 3D 构象（Day1 trimer 构象失败率 79%），按 PerioGT 方式构造周期边；(2) 输入节点(原子序/度/电荷/杂化/芳香/环)+边(键型/共轭/环/周期标志)+几何(归一化键长/角/二面角)；(3) Encoder 5 层 GINEConv hidden 256；(4) 电荷平衡层（每原子类型学习电负性 χ 与硬度 η，q_i=(χ_i−χ_avg)/η_i，h_i'=h_i+α·q_i·W_q）；(5) **注意力池化** a_i=softmax(w^T·tanh(W_h·h_i))（softmax 归一化保证 intensive）；(6) RU 间 max 聚合 h_final=max(h_RU1,h_RU2,h_RU3) 提供重复不变性；(7) 预测头 MLP 256→128→64→1。计划消融：去注意力池化/去电荷平衡/去 max 聚合/去几何特征。

### 训练评估方式
**设计值（未执行）**：AdamW lr 1e-3 wd 1e-5，batch 32（或 64），MSELoss，early stop patience 30，max 500 epoch，ReduceLROnPlateau/CosineAnnealingLR；Tg 7170 样本 72/8/20 split，baseline GAT 50.5、SOTA PerioGT 30.8。**实际：0 seed、0 split、0 epoch、0 checkpoint**——全部未执行。

### 结果
**无任何实验结果**。`3_03_iapt/` 目录 0 文件 0 字节；无 log/checkpoint/results.csv/plot。orchestrator 在 Day3 时段（2026-05-07 15:33~16:12，约 30+ 分钟）连续发起 **7+ 次 run_experiment(project_name=03_iapt) 调用**，每次返回 `error_type: execution_failed, stdout: API Error: 429 (该模型当前访问量过大)`；Day3 区段（full.log 行 12600–13600）共出现 **37 次 429**，无一次成功。状态确认为 **blocked / NOT EXECUTED**。

### 流程
Day3 按 MetaMethod 流程：选 2 篇论文（MBNN-att 2024 intensive/extensive 多体注意力；CELLI 2024 长程电荷平衡层）→整合 Day1（单体构象、周期边）、Day2（GRIN max 聚合）insight 设计 IAPT→保存 mem_083（设计）+mem_084（journal）+mem_085（TopicState 新增 Intensive_Property_Hypothesis 与 Charge_Equilibration_Hypothesis 两条 active_hypotheses）→多次 run_experiment 调用**全部被 HTTP 429/API rate limit 阻断**→设计完整保留、待 Day4 重试。Day4 在 `4_iapt_v2_intensive_aware/` 实际落地了 IAPT-v2（属 Day4 范畴）。

---

## Day 4：IAPT v2 — intensive 洞察验证（mean 最差），5-集成 35.08

**参考/启发文献**（Day 4：当日 Learn 读取 4 篇；来源 mem.json provenance）
- 【当日读取】**arXiv:2008.08461**（arXiv） · `mem_086` · [code](https://github.com/bkmi/equivariant-benchmark) —— This paper provides actionable evidence for the IAPT architecture design (mem_083): equivariant layers are more parameter-efficient than…
- 【当日读取】**2025 Molecule Graph Networks with Many-Body Equivariant Interactions**（paper） · `mem_087` —— ENINet provides a principled approach to incorporate many-body geometric information that could improve Tg prediction: (1) The line graph…
- 【当日读取】**arXiv:2207.13779**（arXiv） · `mem_088` · [code](https://git.rwth-aachen.de/avt-svt/public/graph_neural_network_for_fuel_ignition_quality) —— This paper enables a concrete experiment: test different pooling functions (sum, mean, max, attention-based) for Tg prediction to determine…
- 【当日读取】**2025 SimPoly Simulation of Polymers with Machine Learning Force Fields Derived from First Principles**（paper） · `mem_089` —— Enables: (1) New benchmark dataset (PolyArena) could complement or replace current Tg dataset (7170 samples)—provides experimental…

### 实验目的
**IAPT v2（Intensive-Aware Polymer Transformer v2）**，目标击败 SOTA 30.8。核心假设 **intensive property 洞察**：Tg 作为 intensive property 对聚合物链长不敏感，模型应 max/attention 而非 mean 池化抑制尺寸相关信号。Day4 两阶段：先 `run_iapt_v2.py` 做 pooling/组件 ablation，再 `run_improved.py` 跑全组件改进版+5 种子集成。

### 模型架构
**(A) run_iapt_v2.py（ablation 版）**：p-SMILES→trimer(3 RU) 作图；原子 27+ 维（元素 one-hot 18+度+杂化 5+芳香+电荷+环）、键 6 维（键型 4+共轭+环，3D conformer 成功时+1 bond_length）；等变特征 compute_equivariant_features 给每原子 1 维（到分子几何中心欧氏距离/10）；GINE 骨干 5 层 GINEConv(hidden 256, eps=0, train_eps) 每层残差+LN+ReLU；池化 mean/max/sum/attention；周期边（端帽星号原子相连）。**代码事实**：当前存档 run_iapt_v2.py **未实现 line graph**（NUM_LINE_GRAPH_LAYERS=2 仅定义未引用），但 no_lg 行参数量比 mean 少 ~100K 暗示执行版本曾含 line graph、源码与产出不一致。**(B) run_improved.py（improved_ensemble，全组件真实出处）**：全局特征 14 维（13 RDKit 描述符 MolWt/NumRotatableBonds/NumRings/NumHeteroatoms/NumAromaticRings/NumAtoms/NumBonds/NumAliphaticRings/FractionCSP3/TPSA/MolLogP/NumHDonors/NumHAcceptors + 回转半径）；等变特征 3 维/原子（中心距离+方向一致性+邻居方向方差）；GINE 6 层+残差+LN+GELU；**虚拟节点**（vn_emb 每层加到所有节点、每层末 global_add_pool 更新）；**line graph（N_LG=2，build_lg 把每条边作节点、共享端点的边互连）**；Jumping knowledge（各 GINE 层输出按可学习 softmax 权重加权求和）；**多尺度池化** (global_mean+global_max+normalized_sum+attention)/4；拼接全局特征后 MLP。

### 训练评估方式
Tg 7170，72/8/20（5162/573/1435），np.random.seed(42) 随机 permutation（非 scaffold）。ablation：AdamW lr 1e-3 wd 1e-5，ReduceLROnPlateau(factor 0.5 patience 15)，MSELoss，batch 32，max 500 patience 30；improved：AdamW lr 5e-4 wd 1e-4，CosineAnnealingWarmRestarts(T_0=50,T_mult=2)，**HuberLoss(delta 1.0)**，batch 32，max 500 patience 40。目标 StandardScaler(train fit) 还原评估。**5 种子集成** seeds{42,123,456,789,2024} 对 5 模型 test 预测算术平均——**关键代码事实**：所有种子共用同一份 seed=42 切分 cache(processed_data_v2.pkl)，**种子差异仅来自模型初始化、非数据切分集成**。

### 结果
**Ablation（单种子 seed=42）**：IAPT_v2_mean **37.89**（R²0.8886，best_val 35.35）、IAPT_v2_max 35.35（R²0.9030，best_val **34.20**）、IAPT_v2_att(attention) **35.19（单种子最佳，R²0.9039）**、IAPT_v2_sum 35.99、IAPT_v2_no_eq(关等变) 36.76、IAPT_v2_no_lg(命名误导) 36.43（参数 773,387 vs mean 873,483）。**intensive 洞察验证成立**：mean 池化 37.89 是六变体中最差，max/attention 显著更优，与 mem_090"mean pooling worst"一致（代码独立验证）。**Improved（5 种子集成）**：RMSE **35.08**（MAE 22.93，R²0.9045，参数 1,954,381，best_val 36.22@ep69）；各种子 test seed42=39.23/seed123=38.05/**seed456=36.67（最佳单种子）**/seed789=37.68/seed2024=37.66；与 SOTA 30.8 差 4.28 未达，improved 集成(35.08)仅比 ablation 单种子最佳(35.19)好 0.11。**3D conformer 成功率**（ETKDGv3+MMFF/UFF）：train 43.3%(2235/5162)/val 42.6%(244/573)/test 40.6%(583/1435)，失败 57–59%（trimer 原子数超 100/150 或 ETKDG 失败，日志 UFFTYPER 频繁报 `Unrecognized atom type: *_`）。

### 流程
读 Tg.csv→np.random.seed(42) permutation 72/8/20→每 p-SMILES generate_multimer_smiles(3) 生成 trimer→RDKit 建图抽原子/键特征+周期边→ETKDGv3 生成 3D conformer→MMFF/UFF 最小化（成功则 bond_length 并入键特征、等变几何并入原子特征、回转半径并入全局特征）→(improved)build_lg 构建线图→ablation 跑 6 变体各 1 次→improved 跑 5 种子(共用 cache)预测平均得集成+独立训 seed=42 绘图模型→产 results.csv(ablation 6 行+improved_ensemble 1 行)、results_v2.csv(5 种子明细)、plots/。**纠正旧 summary**：(1) 虚拟节点+多尺度池化只在 run_improved.py 不在 ablation 行；(2) no_lg 行命名误导（源码已无 line graph 但参数证明执行版本曾有）；(3) 5 种子共用 seed=42 切分。

---

## Day 5：GraphMAE+MoCL 预训练 — v1 35.60 / v2 36.14（过拟合，未达 SOTA）

**参考/启发文献**（Day 5：当日 Learn 读取 4 篇；来源 mem.json provenance）
- 【当日读取】**2024 Geometry-enhanced pretraining on interatomic potentials**（paper） · `mem_092` · [code](https://github.com/cuitaoyong/GPIP) —— GPIP provides actionable contributions for the polymer Tg prediction task: (1) Pretraining strategy for PI1M: The three-task SSL framework…
- 【当日读取】**arXiv:2106.04509**（arXiv） · `mem_093` · [code](https://github.com/illidanlab/MoCL-DK) —— 1. Actionable for PI1M pre-training: MoCL's substructure substitution approach can be adapted for polymer SMILES—the bioisostere concept…
- 【当日读取】**arXiv:2205.10803**（arXiv） · `mem_094` · [code](https://github.com/THUDM/GraphMAE) —— Enables pre-training on PI1M dataset before fine-tuning on Tg prediction. Provides transferable techniques: (1) SCE loss for robust feature…
- 【当日读取】**2025 On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs Bridging Recurrent and Graph**（paper） · `mem_095` —— This paper enables three actionable improvements for polymer Tg prediction: (1) Diagnostic tool: analyze layer-wise Jacobian eigenvalues of…

### 实验目的
在 PI1M 聚合物语料（~100 万 p-SMILES）上做自监督预训练（GraphMAE 掩码自编码+MoCL 风格对比预训练），再在 Tg 下游微调，目标击败 SOTA 30.8。两版本：v1（iter3 代码，65% mask+edge+CL，此前 100K 实跑）；v2（更保守实现，GraphMAE 65% mask+edge+InfoNCE 对比，300K PI1M 预训练+渐进解冻微调）。

### 模型架构
**v1**（`main.py`/`model.py`）：GINEncoder 5 层 GINEConv(hidden 256, edge_dim 6, in_dim 27) 每层 LN+ReLU+残差，微调用 MultiScaleEncoder(JK-net 聚合各层)；GraphMAE 头 65% 节点掩码+边掩码+re-mask 解码（解码器 3 层 GINE），节点重建 SCELoss(power 5.0)、边重建 EdgeReconLoss；对比头 MoCL 风格 proj_head+ContrastiveLoss(InfoNCE temp 0.1)，两视图由 MoCLAUG(atom_mask 0.2, edge_drop 0.1) 与 ContrastiveAug(atom_mask 0.25, edge_drop 0.15, feature_noise 0.1) 生成；微调头 FineTuner MultiScaleEncoder+4 头 AttentionPooling+(mean‖max) 辅助+combine_proj+3 层 MLP dropout 0.3；损失权重 λ_node 1.0/λ_edge 0.5/λ_cl 0.1。**v2**：GINEncoder 直接用（无 MultiScale/JK），GraphMAE 一致，**对比损失用掩码视图 vs 无掩码原图(no_grad)（不调 MoCLAUG/ContrastiveAug，增强弱于 v1）**；微调头 FineTuner GINEncoder+4 头 AttentionPooling+(mean‖max)+MLP dropout 0.3，参数 1,370,373。

### 训练评估方式
预训练数据 PI1M.csv，df.sample(random_state=42)；微调 Tg.csv 随机 72/8/20（5162/573/1435）random_state=42 **非 scaffold**；全局 seed=42 **单 seed**；微调 AdamW(lr 1e-3, wd 1e-5) label z-score 归一化 MSE early stop patience 100，含"冻结编码器前 30 epoch→解冻"渐进策略。**v1 实跑**：100,000 PI1M(99298 valid, 702 失败) × 30 epoch batch 256 lr 1e-3+5 warmup+cosine，Loss→0.0003（无 Node/Edge/CL 分解→实际用更早无 CL 的 GraphMAE 代码），参数 718,823；微调 500 epoch 上限 early stop@180 best val@130。**v1 iter3 计划**：300,000×60 epoch mask 0.65 带 MoCL CL，run_log_iter3.txt 仅 13 行在"Processing 300000 SMILES"RDKit 阶段**中断未进训练**——故 results.json 的 35.60 来自 100K/30-epoch 实跑非当前 iter3 代码。**v2 实跑**：300,000 PI1M(297919 valid, 2081 失败) × 60 epoch batch 256，Loss 分解 Node≈0.0003/Edge≈0.0841/CL 2.23→0.81 best total 0.1231；微调 500 epoch patience 100 解冻@51 early stop@226 best val@146(val 35.99) label_mean 141.95 std 111.79；**evaluate() 阶段崩溃**：torch.load 读 checkpoint 触发 PyTorch 2.6 `_pickle.UnpicklingError`（weights_only=True 默认变更，numpy scalar 未在 safe globals），results.json 测试数字来自另一次成功评估。

### 结果
**v1**（results.json+run_log.txt）：Test RMSE **35.6044**（MAE 22.92，R²0.9016），train 22.68，val 34.72，**train-val gap 12.04**（过拟合明显，预训练未缩小差距），未达 SOTA。**v2**（results.json+run_log.txt）：Test RMSE **36.1399**（MAE 24.73，R²0.8986，val 35.99），label_mean 141.95 std 111.79，由 best-epoch 训练损失反推 train RMSE≈21.4，gap≈14–15（过拟合与 v1 相当或更重），较 v1 更差（36.14 vs 35.60）且评估流程因 torch 2.6 weights_only 默认变更中断。**订正**：(1) v1 的 35.60 来自 100K/30-epoch 实跑非当前 iter3 300K/60-epoch 代码（后者在 PI1M 加载阶段中断）；(2) v2 evaluate 阶段因 torch 2.6 weights_only 崩溃，results.json 来自另一次成功评估；(3) PI1M 实际 v1 100K(99298 valid)/v2 300K(297919 valid)；(4) 两版均单 seed=42 随机 72/8/20 非 scaffold。

### 流程
**v1**：main()→Phase1 pretrain(计划 300K/60 epoch)→Phase2 finetune(500 epoch)→evaluate()→画 training_curves.png+tsne(t-SNE random_state 42)；实跑路径为 100K/30-epoch 更早版本，iter3 300K 重跑在 PI1M RDKit 批处理阶段中断。**v2**：main()→pretrain(300K/60 epoch/GraphMAE+CL)完整跑完→finetune(500 epoch/渐进解冻@51)完整跑完(best val 35.99@146)→evaluate()在 torch.load 处崩溃（PyTorch 2.6 weights_only 默认改 True），results.json 测试数字来自另一次成功评估。

---

## Day 6：PolyKAN-SSM — v5b 38.17±1.46（Fourier-KAN/JK 正贡献，SSM 过复杂）

**参考/启发文献**（Day 6：当日 Learn 读取 3 篇；来源 mem.json provenance）
- 【当日读取】**2024 X2-GNN A Physical Message Passing Neural Network with Natural Generalization Ability to Large a**（paper） · `mem_098` —— Actionable contributions: (1) Physics-informed features: Implement S and H^core integral features as edge attributes in polymer GNNs—these…
- 【当日读取】**KA-GNN Kolmogorov Arnold GNN Molecular Property**（paper） · `mem_099` —— 1. Architecture Innovation: Fourier-KAN provides a principled alternative to MLPs with theoretical approximation guarantees—can be tested…
- 【当日读取】**Vanishing Gradients Over-smoothing Over-squashing GNN**（paper） · `mem_100` —— This paper enables: (1) Diagnostic Tool: Jacobian eigenanalysis can predict whether polymer GNN architectures will suffer from gradient…

### 实验目的
探索 **KAN（Kolmogorov-Arnold Network，用 Fourier 级数实现）+ SSM（State Space Model 状态空间模型，此处 GINE 卷积叠加 SSM 风格衰减残差 h_ssm=h·λ+h_msg，λ=exp(log_lambda) clamp≤0.999）+ 非共价边(5Å) + JumpingKnowledge(拼接所有层) + 虚拟节点** 组合能否逼近 Tg SOTA 30.8。当夜迭代 v3→v4→v5→v5b 共 4 主版本并在 v4/v5/v5b 上做消融，最终未能超越 SOTA（最佳 v5b 38.17±1.46）。

### 模型架构（逐版本）
所有版本建含**非共价边(cutoff 5.0Å)** 的分子图（共价键外对距离<5Å 原子对补一条非共价边 covalent_flag=0 含 16 维距离编码），前 100 图平均共价边 25.9、非共价边 62.4。节点 21 维、边 23 维。划分均 train 5162/val 573/test 1435（72/8/20）。**v3**：GINE 骨干+虚拟节点+SSM 衰减残差+Fourier-KAN 激活，hidden 256 num_layers 5 dropout 0.3 wd 5e-3 num_frequencies 8，2,491,915 参数。**v4**（推倒重写）：自定义 SSM-GNN(带多头注意力的 GNNSSMLayer)+FourierKANLayer+双池化，hidden 192 num_layers 4 num_heads 4 dropout 0.2 **移除虚拟节点**，1,054,290 参数。**v5**（综合 v3/v4）：GINE 骨干回归+Fourier-KAN 作**门控辅助分支**(h+sigmoid(kan_gate)·kan_out, gate 初值 0.1)+JumpingKnowledge(cat)+虚拟节点(轻量更新)+SSM 风格衰减残差+注意力加权和池化(add+mean)，hidden 256 num_layers 4 dropout 0.25 num_frequencies 4，2,209,159 参数。**v5b**：v5 基础上**去掉虚拟节点**、加强正则(dropout 0.25→0.3, wd 2e-3→5e-3)、新增 pred_l2_penalty=0.01、池化由加和改拼接(cat[graph_attn, graph_mean])，2,207,879 参数。

### 训练评估方式
5 种子主跑（v3/v4/v5/v5b 均 seeds{42,123,456,789,2024}）；Adam lr 5e-4（v4 为 1e-3）batch 32 max 500 epoch 早停 patience 30（v3 为 40）ReduceLROnPlateau(patience 15 factor 0.5)；目标标准化（mean 141.56 std 112.42）；按最优 val RMSE 选 checkpoint 报 test RMSE/MAE/R²。消融（v4/v5/v5b）均**单种子 seed=42**。

### 结果
（代码核验权威数字）：**v3 39.96±1.30**（gap 11.22 严重过拟合，最佳单种子 37.77 seed456）/ **v4 54.54±1.56**（gap 0.55 几乎无过拟合但性能崩塌，最佳 52.25 seed42）/ **v5 39.21±1.39**（gap 16.41 过拟合最严重，最佳 37.55 seed456）/ **v5b 38.17±1.46**（gap 10.34，最佳单种子 **36.45 seed123**）。v5b 各种子 test：seed42=38.74/seed123=36.45/seed456=37.32/seed789=40.70/seed2024=37.66。**v4 消融**（seed42）：no Fourier-KAN 50.13、no GNN-SSM(换 GINConv) **45.56（最佳消融）**→直接催生 v5 回归 GINE、no 非共价边 48.34→"less is more"。**v5 消融**（seed42）：no Fourier-KAN 39.81、no JK 39.83、**no VN 38.48（去 VN 反优于完整 v5 39.21）**→催生 v5b 去 VN。**v5b 自身消融**（seed42）：no Fourier-KAN 38.96（Fourier-KAN +0.79）、**no JK 41.68（JK +3.51，最关键组件）**；**with-VN 消融崩溃**（RuntimeError tensor size mismatch 1146 vs 32，无结果）→无 v5b no-VN 消融（v5b 本身已无 VN）。**纠正旧 summary**：(1) gap 10.34 属 **v5b 非 v5**（v5 gap 16.41）；(2) v5b **本身不含虚拟节点**（旧 summary 误把 VN 列为 v5b 组件，VN 只在 v3/v5）；(3) v4 gap 0.55 是 4 版本唯一不过拟合但绝对性能最差。

### 流程
v3(5 层含 VN/Fourier-KAN，过拟合 gap 11.2)→v4(推翻重写 SSM-attention，gap 0.55 但性能崩至 54.5；消融发现 GINConv>SSM、去 Fourier 更好、去非共价边更好)→v5(回归 GINE+门控 Fourier-KAN+JK+VN+轻量 SSM 残差，性能 39.21 但 gap 飙至 16.4；消融去 VN 反而更好 38.48)→v5b(去 VN+加强正则+pred_l2+池化改拼接，**当夜最佳 38.17±1.46，单 seed123=36.45，仍距 SOTA 30.8 差约 5.4–6K**)。v5b 消融证明 Fourier-KAN 与 JK 均为正贡献（尤其 JK 移除后退化至 41.68）；尝试给 v5b 加回 VN 因张量维度 bug 失败未产出。mem_101（38.17 vs 30.8，Fourier-KAN helped slightly [v5b 38.17<38.96]，GNN-SSM too complex [v4 GINConv>SSM]）证实。

---

## Day 7：PolyPerio-DenseGNN — angular 36.98±1.70（terminal 最优，非全 run 最佳）

**参考/启发文献**（Day 7：当日 Learn 读取 3 篇；来源 mem.json provenance）
- 【当日读取】**arXiv:2205.13757**（arXiv） · `mem_103` —— Actionable for Day 7: (1) Implement polymer graph representation exactly as described - trimer with periodic edge connecting terminals. (2)…
- 【当日读取】**arXiv:2405.11502**（arXiv） · `mem_104` —— Actionable insights for polymer property prediction: (1) Angular encoding provides ~17% improvement - validates incorporating directional…
- 【当日读取】**2024 DenseGNN universal and scalable deeper graph neural networks for high-performance property pred**（paper） · `mem_105` · [code](https://github.com/dhw059/DenseGNN) —— Actionable contributions: (1) DCN architecture can be integrated into polymer GNNs to enable deeper models without over-smoothing…

### 实验目的
Day7 是 ab4 terminal/末天模型。系统检验"周期性聚合物图表示 + 方向角编码 + 密集连接"三机制对 Tg 回归的独立贡献，通过 4 变体×5 种子消融定位各机制边际增益，追赶 PerioGT SOTA 30.8。四变体：**baseline**（单体基线，丢弃两个 * 原子，不三聚化/无周期边/无方向编码）、**+periodic**（trimer+周期边）、**+angular**（周期图+方向角编码，Day7 最佳）、**full**（+angular 基础上开 DenseGNN 式 dense connectivity）。**定位说明**：Day7 最佳 angular 36.98 是 terminal 最优，但**非全 run 全局最优**（全 run 最低仍为 Day1 PPFN 34.14），未超越 SOTA 30.8（差 +6.18）。

### 模型架构
通用骨架 `PolyPerioDenseGNN`=节点嵌入+边嵌入+5 个 DenseMPNNBlock+AttentionPooling 与 global_mean_pool 拼接+MLP 预测头，全程 LN+SiLU+Dropout 0.1，hidden 96。节点特征恒 144 维 one-hot=atomic_num(119 含 wildcard 0)+degree(6)+formal_charge(5)+num_hs(5)+hybridization(5)+is_aromatic(2)+is_in_ring(2)。差异在边特征与 block 连接：**baseline** build_monomer_simple 丢 * 原子，edge_in 9（bond_type 4+is_conjugated 2+is_in_ring 2+periodic_flag 1 恒 0），顺序连接，参数 509,474；**+periodic** build_trimer RWMol 拼 3 RU+周期边(atom 0↔atom n-1)，edge_in 9（周期边 bond_type 全 0、periodic_flag 1），顺序连接，参数 509,474；**+angular** trimer+周期边+32 维 angular 特征 concat，**edge_in 9+32=41**，angular 32 维=方向余弦 cosθ_x/θ_y/θ_z 各 8-center RBF(linspace(-1,1,8) γ 5.0)+边长 8-center RBF(linspace(0,8,8))，"3D 坐标"为**拓扑伪 3D 坐标**（compute_topological_coordinates 取 3 landmark 原子 atom0/atom n-1/atom n//2 用 GetDistanceMatrix 图距离作三维轴坐标，use_3d=False 未调 RDKit 构象），顺序连接（dense_connectivity=False），参数 512,546；**关键**：`self.use_angular` 仅存储、forward() 全程未引用→angular 信息唯一通道是经 edge_attr 32 维进入（dead code）。**full** trimer+periodic+angular+**dense connectivity 开启**（第 i block 前把之前所有节点特征拼接经 projections[i-1]=Linear(96·(i+1)→96) 投影回），4 投影层额外 ~129K 参数，参数 641,954。每 block EdgeConditionedMPNN(aggr='add')+LN 残差。

### 训练评估方式
Tg.csv，trimer 三变体 7170→RDKit 解析 6859 有效图（kekulize/sanitize 失败丢弃 311）72/8/20→4938/548/1373（baseline 单体 6959→5010/556/1393，各变体有效数据集大小不同→测试集实际不同，但差异≤1.4%）。**种子** SEEDS=[42,123,456,789,2024]，set_seed(seed) 单点同时控制数据划分(random.Random(seed) shuffle)与权重初始化→**split 与 init 未解耦**。Adam(lr 1e-3, wd 1e-5) ReduceLROnPlateau(factor 0.5 patience 15 min_lr 1e-6) **HuberLoss(delta 1.0)** batch 64 max 500 patience 30 grad clip 5.0；**目标直接回归原始 Tg 不标准化**；按 val RMSE 最低 epoch 存 state_dict test 评估；5 种子 mean±std（np.std 总体标准差，**无 ensemble**）。

### 结果
5 种子（mean±std，源自 ablation_results.json 并与 20 个 *_seed*_temp.json 交叉核对）：**baseline（单体）42.39±3.22**（R²0.8527）/ **+periodic 38.24±1.65**（R²0.8800）/ **+angular（最佳）36.98±1.70**（R²0.8879）/ **full（+dense）38.81±1.71**（R²0.8764）。**贡献分解**：周期图(baseline→+periodic) **−4.15（−9.8%，最大单项增益，且 std 3.22→1.65 显著稳定化）**、方向角(+periodic→+angular) −1.26、密集连接(+angular→full) **+1.83（反向恶化，full 多出 ~130K 参数在小数据上过拟合，best_epoch 普遍提前到 77–169）**。+angular 单种子明细：seed42=36.64/seed123=35.44（最佳单种子）/seed456=36.21/seed789=36.30/seed2024=40.29（最差），best_epoch 均 early stop。与 PerioGT SOTA 30.8 差 +6.18，未超越。mem_106（FAILED 36.98 vs 30.8，Periodic graph representation helps）证实。

### 流程
四变体由 train.py::main 顺序驱动（VARIANTS=['baseline','periodic','angular','full']）每变体 5 种子共 20 次训练；日志拆 logs/baseline_periodic.log（baseline+periodic 各 5 种子 edge_in 9）与 logs/angular_full.log（angular+full 各 5 种子 edge_in 41）；每次 run_experiment 完成保存中间 JSON（{variant}_seed{seed}_temp.json），变体内汇总写 results/ablation_results.json；analyze.py 读 JSON 出 ablation_comparison.png（含 SOTA 30.8 红虚线）/rmse_boxplot.png/improvement_over_baseline.png。

---

## 7 天轨迹总结

ab4 是 GLM-5 **DAY-ONLY**（夜巩固已消融）run，每天换一个全新架构，无任何模型达 SOTA 30.8。Tg RMSE 演进：**Day1 PPFN 34.14（5-集成，全 run 最佳）→ Day4 IAPT v2 35.08 → Day5 GraphMAE 35.60 → Day6 PolyKAN-SSM v5b 38.17 → Day7 PolyPerio-DenseGNN angular 36.98（terminal）**。核心节点：

- **Day1 PPFN**：3 轮迭代（v1 复杂分层→v2 简化周期边→v3 3D conformer），**v3 5-集成 34.14（R²0.9095）为全 run 最佳**；3D 几何特征 +7.1% 最大增益、周期边仅 +2.5%、trimer conformer 生成 ~79% 失败（>100 原子回退 2D）。
- **Day2 RIT**：重复不变 max 聚合+Transformer 长程+多任务+偏置修正设计完整，但**执行被阻塞**（数据预处理阶段中断），仅遗留 seed0 epoch100（val 38.61）部分 checkpoint；"API 429 阻断"是 AI 编排层自述（mem_080），训练日志无 429、代码未独立验证。
- **Day3 IAPT**：**完全未执行**——目录空、0 代码、Day3 区段 full.log 出现 **37 次 429**、7+ 次 run_experiment 全失败；设计（5 GINEConv+电荷平衡层+softmax 注意力池化+RU max 聚合）完整保留待 Day4 重试。
- **Day4 IAPT v2**：ablation 验证 **intensive 洞察**（mean 池化 37.89 最差、max/attention 显著更优），improved 全组件版（GINE+线图+等变+虚拟节点+多尺度池化）5-集成 35.08；3D conformer 成功率仅 43.3%/42.6%/40.6%。
- **Day5 GraphMAE+MoCL**：v1 35.60（实为 100K/30-epoch 实跑非 iter3 300K 代码）/ v2 36.14（300K 完整预训练但 evaluate 因 torch 2.6 weights_only 崩溃）；过拟合 gap ~12–15，预训练未缩小差距。
- **Day6 PolyKAN-SSM**：4 版本迭代 v3→v4→v5→v5b，v4 推倒重写 SSM 性能崩（消融发现 GINConv>SSM）催生 v5 回归 GINE，v5 消融去 VN 反优催生 **v5b 38.17±1.46（单 seed123=36.45）**；Fourier-KAN(+0.79)与 JK(+3.51 最关键)为正贡献，SSM/虚拟节点为负。
- **Day7 PolyPerio-DenseGNN**：4 变体×5 种子，**angular 36.98±1.70（terminal 最优）**；周期图 −4.15(−9.8% 最大且稳定化)、angular −1.26、dense +1.83（小数据过拟合反恶化）。

**day-only 结构特征**：mem.json **0 个 integration 节点**（对照 full day-night run 的 20–30 个），夜巩固的 provenance 边完全缺失——这是本 run 与 ab2/3/6 的核心结构差异，也是论文"夜巩固因果性"主张的最硬证据之一。判别关键不在"是否每天换架构"（day-only 也换），而在"连续实验是否因果衔接"——本 run Day2/3 被 API 阻塞、实验间衔接较弱。

## 主要参考文件

- 代码：`src/1_day2_ppfn/{model.py,model_v2.py,model_v3_3d.py,dataset.py,dataset_v3_3d.py,train_v3_3d.py,results.json,results_v3_3d.json,results_v4.json,conformer_analysis.json,IMPLEMENTATION_REPORT_V3_3D.md,training_log.txt}`、`src/2_day2_rit_polymer/{model.py,dataset.py,train.py,RESULTS.md,results/results.json,results/best_model_seed{0..4}.pt,results/rmse_comparison.png,training_output.log}`、`src/3_03_iapt/`(空目录)、`src/4_iapt_v2_intensive_aware/{run_iapt_v2.py,run_improved.py,results.csv,results_v2.csv,improved_log.txt,experiment_final.log}`、`src/5_day5_pretraining/`+`src/5_day5_pretraining_v2/{main.py,model.py,dataset.py,results.json,run_log.txt}`、`src/6_day6_polykanssm/{model_v3.py,model_v4.py,model_v5.py,model_v5b.py,results.json,results_v4.json,results_v5.json,results_v5b.json,ablation_results_v5.json,results_v5b_ablation_no_kan.json,results_v5b_ablation_no_jk.json,ablation_with_vn.log,experiment_summary.txt}`、`src/7_day_7_polyperio/{models/polyperio_densegnn.py,train.py,dataset.py,results/ablation_results.json,results/angular_seed{42,123,456,789,2024}_temp.json,logs/angular_full.log,logs/baseline_periodic.log}`。
- 记忆：`mem.json`（107 节点；mem_001 TopicState、各 day experience 节点 mem_077/080/083/090/097/101/106）、`lite.log`、`full.log`（Day3 区段 37×429）。
- baseline：`src/baseline/README.md`、`src/baseline/results/{Egc,Tg,Tm}_baseline_report.txt`。
