# TCP (Thermodynamic Consistency Pre-training) - ab3 (GLM-5, full day-night)

> 一句话定位：ab3 / 基座 GLM-5 / full day-night（ab2 过程的独立重复，仅随机性不同）/ 最终(terminal)模型：TCP（热力学一致性预训练，诚实负迁移结果），头部目录 `src/7_day8_tcp`。

## 实验目的

ab3 是 ab2（full day-night，基座 GLM-5）的独立重复，仅随机性不同。本 run 的 headline/terminal 模型是 `7_day8_tcp` 的 Thermodynamic Consistency Pre-training（TCP），它是整个 run 最后执行的一个 concept-leaping 实验，也是 4 个候选（`7_day7_dcn_depth_scaling` / `7_day8_tcp` / `6_day6_cfd_mtl` / `5_day5_fd_mtl`）中 Tg RMSE 最优者。其动机直接来自 Day 3 的失败教训：在 PI1M 上做 PF-CL 预训练产生 21.4% 负迁移，根因被归为 (a) 预训练分布未对齐下游、(b) 预训练任务冗余（预测推理时已计算的 148-dim 特征）、(c) 缺乏物理监督。TCP 的设计假设是：热力学定律（Tg<Tm、Tg/Tm≈0.65、Egc≈Tg/10+30）是对所有聚合物普适的约束，既非可计算特征（避免冗余）又与分布无关（避免分布失配），可作为"免费监督"用于 PI1M 预训练，从而实现正迁移并把 Tg 从 Day 2 的 32.31 K 推到 SOTA 30.8 K。设计上用 Bicerano 基团贡献法生成 PI1M 的 Tg/Tm/Egc 伪标签，用分布感知采样（MC-dropout 认知不确定性最低的 10 万样本）对齐分布，多性质预测头 + 热力学一致性损失做预训练，再 5-fold 微调 Tg。**需强调：TCP 最终给出的是诚实的负结果（负迁移），headline 选它是因为它同时满足"最终（chronologically terminal）"与"4 候选中 Tg 最佳（35.49）"两个判据；而 run 的真实最优 Tg 仍是不在候选清单内的 Day 2 GRIN baseline 32.31 K。**

## 模型架构

输入→输出全流程：polymer p-SMILES →（RDKit，sanitize 保留通配符 *，Compute2DCoords）→ mol_to_graph → PyG Data → GIN encoder → global_add_pool → 性质预测头 → Tg（微调阶段）或 Tg/Tm/Egc（预训练阶段）。（代码：`src/7_day8_tcp/phase3_pretraining.py` 的 TCPModel、`phase4_finetuning.py` 的 TCPFinetuneModel/BaselineModel；图构建复用 `src/1_day2_reperioGT/dataset.py` 的 mol_to_graph / get_atom_features / get_bond_features。）

**原子特征**（node feature x）：153 维 one-hot，内容 = 原子序数 119（1-118 + unknown）+ 度 8（0-6 + unknown）+ 形式电荷 8 + 总氢数 6 + 杂化 6（SP/SP2/SP3/SP3D/SP3D2 + unknown）+ 芳香性 3 + 成环 3；由 one_hot_encoding 逐项拼接。**边/键特征**（edge_attr）：11 维 one-hot = 键类型 5（单/双/三/芳香 + unknown）+ 共轭 3 + 成环 3。注意：edge_attr 被 mol_to_graph 计算并存入 Data，但编码器用的是 GINConv(nn.Linear(...))，该算子只接收 (x, edge_index)、不接收 edge_attr，因此 11 维键特征**实际未被 encoder 消费**（见备注不确定项）。通配符原子（*）原子序数=0，作为周期连接点；另外每分子计算 rigidity（可旋转键占比）与 periodic_mask/periodic_vectors 存为 Data 属性，但 TCPModel.forward 同样未使用它们。

**骨干**（encoder）：5 层 GIN，hidden_dim=256。每层 = GINConv(Linear(in,256)) [首层 in=153] → BatchNorm1d(256) → ReLU → Dropout(0.1)（仅 i>0 层加 dropout）；末层 global_add_pool 按图聚合得 256 维图表示。无 attention、无残差、无边特征注入。

**预测头**：预训练阶段 3 个并行 MLP 头（tg_head/tm_head/egc_head），各 256→128→1（Linear→ReLU→Dropout0.2→Linear）；微调阶段仅保留单个 tg_head（同结构），encoder 权重从预训练 checkpoint 拷贝（形状匹配的 key）。

**归一化/dropout**：仅 BatchNorm1d（每层后）+ Dropout0.1（encoder, i>0）+ Dropout0.2（头内）；无 target normalization（微调用原始 K 值 batch.y，见不确定项）。**损失**：预训练 L = L_pred + 0.1·L_consistency，其中 L_pred = MSE(Tg)+MSE(Tm)+MSE(Egc)；L_consistency = 10·ReLU(Tg−Tm).mean() + 5·ReLU(|Tg/Tm − 0.65| − 0.2).mean() + 0.5·ReLU(|Egc − (Tg/10+30)| − 20).mean()（分别惩罚 Tg≥Tm、比例偏离 0.65±0.2、Egc 偏离经验式）。微调损失 = MSE(Tg)。梯度裁剪 max_norm=1.0。

## 训练评估方式

**数据集与划分**：Tg 7170（src/alldataset/Tg.csv，Smiles/Tg 两列，解析成功率 100%）；Tm 3651；Egc 3380；PI1M 995,800 条未标注 p-SMILES（解析有效 988,775, 99.3%）。TCP 微调只在 Tg 上做 5-fold 交叉验证（sklearn KFold n_splits=5, shuffle=True, random_state=42——这是 split 种子=42）。预训练数据 = 从 PI1M 用分布感知采样选出的 10 万条（认知不确定性最低者，不确定性来自一个 3 层 GIN hidden=128 的 MC-Dropout 5-member ensemble），再 90/10 train/val 划分（train_test_split random_state=42）。伪标签由 Bicerano 式基团贡献法生成（Tg/Tm/Egc），100% 满足 Tg<Tm；伪标签 Tg 535.70±78.29 K（噪声大，见结果）。

**种子协议**：仅显式设 split 种子 42；未显式设 torch 初始化种子（依赖默认），即每 fold 单次随机初始化。故 reported mean±std 是跨 5 个 fold 的均值/标准差（fold-level），**非多种子重复**。Day 4 的 Tm/Egc ensemble UQ 才是真正的 5 seed × 5 fold 多种子协议（见流程）。

**预处理**：SMILES→分子图（见模型架构）；微调阶段不做 target StandardScaler（原始 K），这与 Day 2 GRIN baseline（用了 StandardScaler）不同，被怀疑是约 3 K gap 的成因之一（见不确定项）。

**优化器/调度**：预训练 Adam lr=1e-3, weight_decay=1e-5, ReduceLROnPlateau(mode=min, factor=0.5, patience=5)；batch=128；50 epochs, early stopping patience=10。微调 Adam lr=5e-4（无显式 wd）, ReduceLROnPlateau(factor=0.5, patience=10)；batch=64；max 300 epochs, early stopping patience=30（监控 val RMSE）。两阶段均 grad clip 1.0。

**指标**：每折验证集 RMSE(K)、MAE；主指标 Tg RMSE mean±std（5 fold）。无独立 test set（5-fold CV 的每折验证集即评估集）。硬件：2× NVIDIA RTX 2080Ti (11GB), CUDA 12.6, PyTorch 2.6.0+cu126, Python 3.10。

## 结果

**headline 模型（TCP, src/7_day8_tcp）Tg 微调 5-fold CV 结果**（split 种子 42, 1 init/fold）：
- TCP fine-tuned：Tg = 35.49 ± 1.60 K（folds 32.92 / 35.77 / 36.15 / 37.79 / 34.82）。
- Scratch baseline（同架构无预训练）：Tg = 35.22 ± 1.55 K（folds 33.14 / 34.96 / 35.50 / 37.89 / 34.58）。
- 绝对改进 = −0.28 K，相对改进 = −0.78% → **判定为 NEGATIVE TRANSFER（负迁移，在误差棒内）**。
- TCP 消融（run_ablation.py, 同 5-fold）：Ablation1 GRIN 148-dim 特征 34.91 ± 1.82 K；Ablation2 ReperioGT 153-dim 特征 35.35 ± 1.48 K；两者差 0.44 K（z=0.26，不显著）→ 特征维度非瓶颈。热力学一致性：预测 100% 满足 Tg<Tm。
- vs SOTA：TCP 35.49 比 PerioGT 2025 SOTA 30.8 高 4.69 K（15.2% 更差），比 Day 2 run-baseline 32.31 高 3.18 K（9.8% 更差）。
- Tm / Egc：TCP 仅微调 Tg，未在真实 Tm/Egc 测试集评估（无 TCP 的 Tm/Egc RMSE）。EA / IP：本 run 不涉及（数据集只有 Tg/Tm/Egc/PI1M），N/A。

**Run 全局最优**（供对比，非候选清单内）：Tg 最佳 = Day 2 GRIN baseline **32.31 K**（5 配置中 mean-aggregation 最优；单配置 test RMSE；该 run 结论：32.31 是无策展预训练数据的现实天花板）。Tm 最佳 = Day 4 multi-seed ensemble 59.18 ± 1.12 K（5 seed × 5 fold；vs 文献 baseline 79.3 GAT / SOTA 45.4 PerioGT）。Egc 最佳 = Day 4 共轭特征 + stratified CV 0.5262 ± 0.0471（vs baseline 0.63 GAT / SOTA 0.40 PerioGT）。CFD-MTL（Day 6）多任务：Tg 44.12 ± 1.07, Tm 59.93 ± 2.71（≈baseline 59.18）, Egc 0.552 ± 0.0406（≈baseline 0.5262）。DCN-GIN-15（Day 7）：Tg 38.40 ± 1.18。PF-CL（Day 3）：Tg 39.23 ± 1.81。FD-MTL（Day 5, 失败）：Tg 117.07。

**结论**：本 run 没有任何候选模型在 Tg 上超过 Day 2 baseline 32.31，更未触及 SOTA 30.8；TCP 作为 terminal 模型给出诚实的负迁移结果，同时第 7 次验证 Feature-First Principle。

## 流程

**多种子协议**：Day 4 的 Tm/Egc ensemble UQ 用 5 seed × 5 fold；DCN/CFD-MTL/TCP/PF-CL 均用 5-fold CV（1 init/fold，报 fold-level mean±std）。run 无 `src/optimize` 目录，最终模型即从 day 实验中判定为 TCP。

**Day-by-day**（共 7 个 day 边界；DCN 与 TCP 同在漫长的 Day 7 内执行，故"day8_tcp"实为 Day 7 尾部的 terminal 子实验）：
- **Day 1**（03-23 17:01→03-24 02:08）：读 PerioGT/MACE/LSR-MP。实验 HEPT（src/1_hept，等变+周期+长程混合）失败 Tg 49.58（2.88M params, v2 53.84），RE-PerioGT（src/1_day2_reperioGT）部分成功 baseline 36.41 / feature-only 35.56 / full-contrastive 35.55（刚性对比学习无增益）。教训：归纳偏置须匹配物理（周期性为聚合物主偏置）。状态：完成（科学失败）。
- **Day 2**（03-24 02:09→10:49）：读 PerioGT/GRIN/ADA-GNN。PerioGT 消融（src/2_02_PerioGT_Ablation）澄清两版 PerioGT（2022=37.35 无预训练 vs 2025=30.8 需 PI1M），GraphFormer 37.00 超过 2022。GRIN 测试（src/2_day2_grin）→ Tg 32.31（run 全局最优，GRIN max+L1 假设被拒，148-dim 特征贡献 57.6%）。状态：完成（目标达成）。
- **Day 3**（03-24 10:50→17:27）：读 GPIP/SIMG。PF-CL PI1M 预训练（src/3_pfcl_pretraining）失败 Tg 39.23 ± 1.81（负迁移，分布失配+任务冗余）。Tm/Egc baseline（src/3_day4_feature_baseline_tm_egc）Tm 86.60 ± 53.05（best fold 55.63）、Egc 1.19 ± 0.59（best 0.63），方差极大。状态：完成（预训练失败）。
- **Day 4**（03-24 17:27→03-25 02:46）：读 AutoGNNUQ 等 4 篇。multi-seed ensemble UQ（src/4_day4_ensemble_uq）→ Tm 59.18 ± 1.12（方差降 97.9%，成功）；Egc 1.12 ± 0.50（失败，Fold 3 数据难度）。共轭特征 + stratified CV（src/4_egc_conjugation）→ Egc 0.5262 ± 0.0471（成功，Fold 3 解决）。三性质 baseline 全部稳定。状态：完成（部分成功）。
- **Day 5**（03-25 02:46→06:05）：FD-MTL（src/5_day5_fd_mtl）灾难性失败 Tg 117.07 / Tm 118.22 / Egc 1.68——实现错误（用了 RDKit descriptors 而非已验证 148-dim 特征, r<0.05）。状态：完成（实现错误）。
- **Day 6**（03-25 06:05→08:49）：读 MBNN-att/PointGAT。CFD-MTL（src/6_day6_cfd_mtl，80 维已验证特征，3 层 GIN）部分成功：Tg 44.12 ± 1.07（比 baseline 差 36%）、Tm 59.93 ± 2.71（≈baseline）、Egc 0.552 ± 0.0406（≈baseline）；正迁移 Tg3%/Tm9%/Egc17%；否定"非对称迁移"假设。状态：完成（部分成功）。
- **Day 7**（03-25 08:50→03-26 18:05, ~33h）：读 DenseGNN。DCN-GIN 深度扩展（src/7_day7_dcn_depth_scaling）：DCN-5 40.26 ± 1.42 / DCN-10 38.59 ± 1.34 / DCN-15 38.40 ± 1.18；Standard-GIN-10 44.16 ± 1.28 / Standard-15 47.64 ± 3.83 → 稠密连接使深层可训但深度本身不足以追平 32.31。随后执行 terminal concept-leaping 实验 TCP（src/7_day8_tcp）：伪标签生成（Phase1, 100.0% 一致性）→ 分布感知采样（Phase2, 不确定性模型 31.93 K）→ 预训练（Phase3, 中途遇 tensor 维度不匹配 bug, 已修复）→ Tg 5-fold 微调（Phase4）→ 消融（Phase5）→ 分析（Phase6）；结果 Tg 35.49 ± 1.60，负迁移 −0.28 K，假设被拒。状态：完成（负结果，诚实报告）。

**消融总览**：GRIN 配置消融（Day 2）、RE-PerioGT baseline/feature/full（Day 1）、FD-MTL（错特征）vs CFD-MTL（正特征）（Day 5→6, Tg +60%）、DCN 深度 5/10/15 vs Standard（Day 7）、TCP vs scratch vs 特征集 148/153（Day 7）。

**完成 vs killed/失败**：全部 7 天均跑完，日志中无进程被 kill / OOM / 超时；"失败"均为科学失败（未达目标 RMSE）：HEPT、PF-CL 预训练、FD-MTL、DCN-depth-scaling、TCP（负迁移）。TCP Phase3 的 tensor 维度不匹配为可恢复 bug，当场修复后继续。EA/IP 全 run 未做。

## 备注

- 主要参考文件：`src/7_day8_tcp/{FINAL_REPORT.md, tcp_final_summary.md, results_summary.txt, Tg_finetuned_results.txt, comparison_with_baseline.txt, phase3_pretraining.py, phase4_finetuning.py, tcp_dataset.py}`、`src/1_day2_reperioGT/{dataset.py, final_results.json}`、`src/2_day2_grin/{FINAL_SUMMARY.md, QUICK_SUMMARY.txt}`、`src/7_day7_dcn_depth_scaling/{summary.txt, results.csv}`、`src/6_day6_cfd_mtl/FINAL_REPORT.md`、`src/5_day5_fd_mtl/SUMMARY.md`、`src/1_hept/{results.json, EXECUTIVE_SUMMARY.md}`、`lite.log`、`mem.json`（经 lite.log 内嵌 TopicState 读取）。
- ⚠️ **本份已经过二次校验（本轮 verdict: has_issues），本轮修了**：(1) Phase1 伪标签热力学一致性（流程 Day 7 / line 58）由 98.7%（误抄自 tcp_final_summary.md 的 stale 数据）改为实际的 100.0%（tcp_experiment_log.txt line 7088: `Tm > Tg: 988775/988775 (100.0%)`）；(2) Egc 共轭实验 Day 标签（结果/Run 全局最优, line 43）由"Day 5"改为"Day 4"（目录 `4_egc_conjugation`，与 day-by-day 节一致；数值 0.5262±0.0471 经 experiment_results.json 核实无误）。
- ⚠️ **本轮补充的解读告诫**（missing_details，未改数字、仅加注）：
  - **消融超参与头条 TCP finetune 不同**：ablation_results.json 显示两个 run_ablation.py 消融（Ablation1 GRIN 148-dim / Ablation2 ReperioGT 153-dim）均用 dropout=0.15、readout='mean'、patience=40、weight_decay=1e-5；而头条 TCP finetune（phase4_finetuning.py）用 dropout 0.1(encoder)/0.2(head)、global_add_pool、ES patience=30、无显式 weight_decay，且 Ablation1 采用 GRIN 148-dim atom / 6-dim bond 特征。因此 **"Ablation 2: TCP Setup (ReperioGT 153-dim)"（35.35）并非与头条 TCP（35.49）逐字节相同**——readout 与 dropout 亦变，故 0.44 K 的特征维度对比并非"其余全控、仅维度变"，读者解读时应注意此混杂因素。
  - **tcp_final_summary.md 为 stale/不同的一次伪标签生成**：该文档给出 Tg 308.45±52.87、Tm 452.12、Egc 60.87、一致性 98.7%，与实际执行的 tcp_experiment_log.txt（Tg 535.70±78.29、Tm 743.72、Egc 91.81、100.0% 一致性）不符。本总结已对 Tg 均值与一致性采用真实日志数据，但引用 tcp_final_summary.md 时应谨慎——该文件似描述另一次/中止的伪标签生成。
- 不确定项：
  - 初始化种子：TCP 微调仅显式设 split 种子 42（KFold random_state），未显式 torch.manual_seed 设 init 种子，故每 fold 为单次默认初始化；reported mean±std 是 fold 级而非多 init 种子级。无法从文件确认是否有未捕获的全局 seed 设置。
  - 键/边特征（11-dim edge_attr）被 mol_to_graph 计算并存入 Data，但 encoder 用 GINConv(nn.Linear) 不接收 edge_attr，实际未被消费；这是设计意图还是遗漏，文件未说明。
  - TCP 微调用原始 K 目标、无 StandardScaler，与 Day 2 GRIN baseline（用 StandardScaler）不同；FINAL_REPORT 把约 3 K gap 部分归于此，但未做对照实验确认，仅为推测。
  - headline 判定：TCP 被选为最终/最佳是因它同时是 chronological terminal 实验且在 4 候选中 Tg 最低（35.49）；但它是负迁移结果，且 run 真实最优 Tg（Day 2 GRIN 32.31）不在候选清单内。若判据改为"最成功的科学贡献"，CFD-MTL（正迁移、Tm/Egc 贴近 baseline）也可被视为候选最佳。
  - 日志只有 7 个 day 边界（Day 1-7），而"8 days 计划"/"day8_tcp"命名暗示第 8 天；实际 DCN 与 TCP 都在漫长的 Day 7（03-25 08:50→03-26 18:05）内执行，"day8"只是目录标签。无法确认是否存在被截断/未记录的独立 Day 8。
  - TCP 模型未在真实 Tm/Egc 测试集评估（仅 Tg 微调），故无 TCP 的 Tm/Egc RMSE；预训练阶段的 Tm/Egc 仅对 PI1M 伪标签。
  - EA/IP 本 run 完全不涉及（数据集只有 Tg/Tm/Egc/PI1M），无任何 EA/IP 数字。
  - Day 2 GRIN 的 32.31 K 为单配置（mean-aggregation baseline）的 test RMSE，非多种子 mean±std；其精确 split/init 种子协议未在本 run 文件中逐一核对（来自 2_day2_grin/FINAL_SUMMARY.md 文本，非原始 results JSON）。
  - CFD-MTL 报告表头把 59.18±1.12 标为"Literature Baseline"，实为本 run Day 4 ensemble 结果（Tm 真实文献 baseline=79.3 GAT, SOTA=45.4 PerioGT）；该表头标注具误导性，本总结已据 research_direction 修正。
